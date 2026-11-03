"""Common state machine for compute-optimal IRG hard-mask pruning.

The runner performs adaptive exact-mask experimentation over a candidate family
induced by the active response version.  The current implementation is a
production-safe orchestration kernel: it is provider-agnostic, content-
addressed, resumable through its ledgers, and functional on retained arrays or
callback-wrapped local runtimes.  It does not claim to replace backend-specific
Torch execution; those numerical kernels enter through MeasurementRequest
providers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np

from mechanistic_systems.measurement.request_providers import ProviderRegistry

from ..evidence_bce_precision import BCELossBoundaryDecisionRule
from mechanistic_systems.inference.sequential_bce import (
    BASELINE_EXCESS_CALIBRATED_BCE_V1,
    BaselineRiskCertificate,
    CalibrationImprovementProfile,
    RunningMoment,
    SequentialBCEDecisionReceipt,
    SequentialCalibratedBCEResolutionSpec,
    joint_v1_decision_interval,
    stratified_student_t_summary_from_moments,
)
from .acquisition import (
    AcquisitionForecast,
    AcquisitionSelectionReceipt,
    DecisionUncertaintySnapshot,
    ScoredRequest,
    UnifiedAcquisitionAdapter,
    UnifiedAcquisitionPlanner,
    select_request_batch,
)
from .boundary_search import AdaptiveBoundarySearch, BoundaryPoint
from .candidates import ActionCandidate, deduplicate_candidates
from .commit_authority import (
    CommitAuthorityMode,
    ExactHardMaskCommitAuthority,
    ExactHardMaskCommitDecision,
    _require_physical_glocal_commit_authority,
)
from .contracts import (
    ContextSelector,
    MeasurementProvenance,
    MeasurementRequest,
    ModelValidityDecision,
    content_hash,
)
from .cost_model import OnlineRequestCostModel
from .costs import CostEstimate, RealizedCost
from .frontier import CandidateStatus, OracleReceipt
from .geometric_partitions import (
    COVARIANCE_RANK_EXPANSION_CACHE_KEY,
    GEOMETRIC_PARTITION_FILTER_CACHE_KEY,
    CovarianceRankExpansionObligation,
    PartitionedCovarianceTrustRegion,
    UnresolvedCovarianceObligation,
    candidate_partition,
    derive_partition_covariance_geometry,
)
from .precision import OracleSequentialPrecisionPolicy
from .residuals import (
    ModelValidityController,
    PredictiveValidityReceipt,
    ResidualObservation,
    RobustResidualModel,
)
from .response_geometry import (
    RESPONSE_TRANSITION_CACHE_KEY,
    ResponseTransitionReceipt,
    response_transition_receipt_from_outcome,
)
from .state import PruningSessionState
from .trust_geometry import (
    GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY,
    GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY,
    GlobalTrustRankGrowthObligation,
)
from .versions.base import ResponseVersion


@dataclass(frozen=True)
class EpochResult:
    parent_state_hash: str
    candidate_ids: tuple[str, ...]
    evaluated_candidate_ids: tuple[str, ...]
    committed_candidate_id: str | None
    stop_reason: str
    boundary_diagnostics: Mapping[str, object]
    model_validity_action: str
    commit_authority_mode: CommitAuthorityMode = "exploratory"
    commit_authority_decision_hash: str | None = None
    candidate_pool_retired: bool = False


@dataclass(frozen=True)
class PruningRunResult:
    run_id: str
    initial_mask_hash: str
    final_mask_hash: str
    committed_candidate_ids: tuple[str, ...]
    epoch_results: tuple[EpochResult, ...]
    compute_summary: Mapping[str, object]
    frontier_ledger: Mapping[str, object]
    commit_authority_mode: CommitAuthorityMode = "exploratory"
    commit_authority_decision_hashes: tuple[str, ...] = ()


class CommonIRGRunner:
    _CANONICAL_VERSION_RESPONSE_KINDS = frozenset(
        {
            "jacobian_rows",
            "marginal_statistics",
            "response_moments",
            "response_factor",
            "metric_factor",
            "directional_curvature",
            "finite_dose_response",
        }
    )

    def __init__(
        self,
        *,
        version: ResponseVersion,
        providers: ProviderRegistry,
        oracle_context_count: int = 128,
        oracle_expected_cost: CostEstimate = CostEstimate(forward_passes=1.0),
        transition_context_count: int | None = None,
        transition_expected_cost: CostEstimate = CostEstimate(forward_passes=1.0),
        oracle_pack_size: int = 4,
        maximum_measurement_refreshes: int = 8,
        boundary_z_value: float = 2.0,
        boundary_resample_standard_error: float = 0.0,
        bce_precision_confidence_level: float = 0.95,
        boundary_metric_mode: str = "paired_bce_total_increment",
        baseline_risk_certificate: BaselineRiskCertificate | None = None,
        maximum_relative_bce_increase: float = 0.20,
        oracle_precision_policy: OracleSequentialPrecisionPolicy | None = None,
        cost_model: OnlineRequestCostModel | None = None,
        measurement_advisors: Sequence[
            Callable[
                [PruningSessionState, str, AdaptiveBoundarySearch | None],
                Sequence[MeasurementRequest],
            ]
        ] = (),
        outcome_consumers: (
            Mapping[
                str, Callable[[PruningSessionState, MeasurementRequest, object], None]
            ]
            | None
        ) = None,
        measurement_adapters: Sequence[UnifiedAcquisitionAdapter] = (),
        acquisition_memory_limit_bytes: float | None = None,
        minimum_acquisition_score: float = 0.0,
        acquisition_planner: UnifiedAcquisitionPlanner | None = None,
        model_validity_controller: ModelValidityController | None = None,
        require_typed_acquisition_forecasts: bool = False,
        commit_authority_mode: CommitAuthorityMode = "exploratory",
        commit_authority: ExactHardMaskCommitAuthority | None = None,
        runtime_transition_callback: Callable[..., None] | None = None,
    ) -> None:
        self.version = version
        self.providers = providers
        self.oracle_context_count = int(oracle_context_count)
        self.oracle_expected_cost = oracle_expected_cost
        self.transition_context_count = int(
            oracle_context_count
            if transition_context_count is None
            else transition_context_count
        )
        self.transition_expected_cost = transition_expected_cost
        self.oracle_pack_size = int(oracle_pack_size)
        self.maximum_measurement_refreshes = int(maximum_measurement_refreshes)
        self.runtime_transition_callback = runtime_transition_callback
        self.boundary_z_value = float(boundary_z_value)
        self.boundary_resample_standard_error = float(boundary_resample_standard_error)
        self.bce_precision_confidence_level = float(bce_precision_confidence_level)
        self.boundary_metric_mode = str(boundary_metric_mode)
        if self.boundary_metric_mode not in {
            "paired_bce_per_removed_route",
            "paired_bce_total_increment",
            "candidate_bce_relative_to_dense_baseline",
            "legacy_total_quality_delta",
        }:
            raise ValueError("unknown IRG boundary metric mode")
        if self.boundary_metric_mode in {
            "paired_bce_total_increment",
            "paired_bce_per_removed_route",
        }:
            BCELossBoundaryDecisionRule(
                confidence_level=self.bce_precision_confidence_level,
                maximum_bce_increment=0.0,
            ).validate()
        self.baseline_risk_certificate = baseline_risk_certificate
        self.maximum_relative_bce_increase = float(maximum_relative_bce_increase)
        if self.boundary_metric_mode == "candidate_bce_relative_to_dense_baseline":
            if self.baseline_risk_certificate is None:
                raise ValueError(
                    "candidate-only BCE adjudication requires a dense baseline certificate"
                )
            if (
                not math.isfinite(self.maximum_relative_bce_increase)
                or self.maximum_relative_bce_increase < 0.0
            ):
                raise ValueError(
                    "maximum relative BCE increase must be finite and nonnegative"
                )
            if self.oracle_pack_size != 1:
                raise ValueError(
                    "candidate-only BCE adjudication uses exactly one physical mask lane"
                )
            self.sequential_bce_spec = SequentialCalibratedBCEResolutionSpec(
                confidence_level=self.bce_precision_confidence_level,
                estimand_policy=BASELINE_EXCESS_CALIBRATED_BCE_V1,
                maximum_planned_looks=(
                    oracle_precision_policy.planned_looks
                    if oracle_precision_policy is not None
                    else 32
                ),
                precision_only_iid=bool(
                    oracle_precision_policy is not None
                    and oracle_precision_policy.precision_only_iid
                ),
            )
            if (
                float(self.baseline_risk_certificate.confidence_level) + 1e-15
                < self.sequential_bce_spec.reference_confidence_level
            ):
                raise ValueError(
                    "dense baseline certificate has insufficient confidence "
                    "for the declared combined decision"
                )
        else:
            self.sequential_bce_spec = None
        self.oracle_precision_policy = (
            oracle_precision_policy
            if oracle_precision_policy is not None
            else OracleSequentialPrecisionPolicy(
                confidence_level=(
                    self.sequential_bce_spec.proposal_confidence_level
                    if self.sequential_bce_spec is not None
                    else self.bce_precision_confidence_level
                ),
                initial_packet=self.oracle_context_count,
                minimum_packet=max(2, self.oracle_context_count // 4),
                maximum_packet=max(
                    self.oracle_context_count,
                    4 * self.oracle_context_count,
                ),
                maximum_contexts_per_candidate=max(
                    self.oracle_context_count,
                    32 * self.oracle_context_count,
                ),
                planned_looks=32,
            )
        )
        if self.boundary_metric_mode in {
            "paired_bce_total_increment",
            "paired_bce_per_removed_route",
            "candidate_bce_relative_to_dense_baseline",
        } and not math.isclose(
            float(self.oracle_precision_policy.confidence_level),
            (
                self.sequential_bce_spec.proposal_confidence_level
                if self.sequential_bce_spec is not None
                else self.bce_precision_confidence_level
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "oracle precision policy and BCE boundary use different "
                "confidence levels"
            )
        self.cost_model = cost_model or OnlineRequestCostModel()
        self.measurement_advisors = tuple(measurement_advisors)
        self.outcome_consumers = dict(outcome_consumers or {})
        self.measurement_adapters = tuple(measurement_adapters)
        adapter_by_kind: dict[str, UnifiedAcquisitionAdapter] = {}
        for adapter in self.measurement_adapters:
            for kind in adapter.supported_kinds:
                if kind in adapter_by_kind:
                    raise ValueError(
                        f"multiple unified acquisition adapters own kind {kind!r}"
                    )
                if kind in self.outcome_consumers:
                    raise ValueError(
                        f"kind {kind!r} is owned by both a typed adapter and "
                        "a legacy outcome consumer"
                    )
                adapter_by_kind[str(kind)] = adapter
        self._adapter_by_kind = adapter_by_kind
        self.acquisition_memory_limit_bytes = acquisition_memory_limit_bytes
        self.minimum_acquisition_score = float(minimum_acquisition_score)
        self.acquisition_planner = acquisition_planner or UnifiedAcquisitionPlanner()
        self.model_validity_controller = (
            model_validity_controller or ModelValidityController()
        )
        if commit_authority_mode not in {"exploratory", "glocal_v3_shadow"}:
            raise ValueError("unknown IRG commit-authority mode")
        if commit_authority_mode == "glocal_v3_shadow":
            if commit_authority is None:
                raise ValueError(
                    "shadow glocal-v3 execution requires an exact hard-mask "
                    "commit authority"
                )
            _require_physical_glocal_commit_authority(commit_authority)
        elif commit_authority is not None:
            raise ValueError(
                "an exact commit authority requires commit_authority_mode="
                "'glocal_v3_shadow'"
            )
        self.commit_authority_mode = commit_authority_mode
        self.commit_authority = commit_authority
        self.require_typed_acquisition_forecasts = bool(
            require_typed_acquisition_forecasts
            or commit_authority_mode == "glocal_v3_shadow"
        )
        if (
            self.oracle_context_count <= 0
            or self.transition_context_count <= 0
            or self.oracle_pack_size <= 0
            or self.maximum_measurement_refreshes <= 0
        ):
            raise ValueError("runner measurement limits must be positive")
        if (
            not math.isfinite(self.minimum_acquisition_score)
            or self.minimum_acquisition_score < 0.0
        ):
            raise ValueError("minimum_acquisition_score must be finite and nonnegative")
        self.residual_model = RobustResidualModel()
        self._residual_model_scope: tuple[str, str, str, str, str] | None = None

    @staticmethod
    def _version_state_hash(state: object) -> str:
        value = getattr(state, "state_hash", None)
        if callable(value):
            value = value()
        return str(value or content_hash(repr(state), prefix="opaque-version-state-"))

    @staticmethod
    def _bind_measurement_requirement(
        request: MeasurementRequest,
        *,
        requirement_class: str,
        obligation_id: str,
    ) -> MeasurementRequest:
        """Return a predictably re-identified request carrying a typed stage gate."""

        payload = dict(request.payload)
        payload.update(
            {
                "measurement_requirement_class": str(requirement_class),
                "measurement_obligation_id": str(obligation_id),
            }
        )
        return MeasurementRequest.create(
            kind=request.kind,
            state_hash=request.state_hash,
            version=request.version,
            context_selector=request.context_selector,
            candidate_ids=request.candidate_ids,
            expected_cost=request.expected_cost,
            expected_decision_value=request.expected_decision_value,
            reuse_keys=request.reuse_keys,
            provenance=request.provenance,
            payload=payload,
        )

    def _bind_residual_model_scope(self, session: PruningSessionState) -> None:
        """Keep allocation-only residual learning inside one frozen parent chart.

        A runner may execute several sessions.  Exact-minus-geometric residuals
        from another run, hard mask, calibration fit, or scientific version are
        not exchangeable and therefore must never silently guide this epoch.
        Repeated looks under the same frozen parent retain their observations.
        """

        response_state = getattr(
            session.version_state,
            "intrinsic_state",
            session.version_state,
        )
        scope = (
            str(session.run_id),
            str(session.identity.identity_hash),
            str(session.active_version),
            str(session.current_mask.mask_hash),
            str(getattr(response_state, "calibration_state_id", "")),
        )
        if scope != self._residual_model_scope:
            self.residual_model = RobustResidualModel()
            self._residual_model_scope = scope

    @staticmethod
    def _validity_action_priority(action: str) -> int:
        priorities = {
            "reuse": 0,
            "shrink": 1,
            "refresh": 2,
            "escalate": 3,
            "fallback": 4,
            "refuse": 5,
        }
        try:
            return priorities[str(action)]
        except KeyError as error:
            raise RuntimeError(f"unknown model-validity action {action!r}") from error

    @classmethod
    def _combine_model_validity(
        cls,
        version_decision: ModelValidityDecision,
        predictive_decision: ModelValidityDecision | None,
        *,
        predictive_receipt_hash: str | None,
    ) -> ModelValidityDecision:
        """Conservatively combine scientific-chart and exact-action diagnostics."""

        if predictive_decision is None:
            return version_decision
        if cls._validity_action_priority(predictive_decision.action) > (
            cls._validity_action_priority(version_decision.action)
        ):
            governing = predictive_decision
        else:
            governing = version_decision
        return ModelValidityDecision(
            action=governing.action,
            reason=(
                f"{version_decision.reason}; partition-local predictive check: "
                f"{predictive_decision.reason}"
            ),
            recommended_scale=min(
                float(version_decision.recommended_scale),
                float(predictive_decision.recommended_scale),
            ),
            predicted_rms=float(predictive_decision.predicted_rms),
            actual_rms=predictive_decision.actual_rms,
            remainder_rms=predictive_decision.remainder_rms,
            diagnostics={
                "combination_rule": "most_conservative_action_then_minimum_scale_v1",
                "version_action": version_decision.action,
                "version_diagnostics": dict(version_decision.diagnostics),
                "predictive_action": predictive_decision.action,
                "predictive_diagnostics": dict(predictive_decision.diagnostics),
                "predictive_validity_receipt_hash": predictive_receipt_hash,
            },
        )

    def _authorize_exact_commit(
        self,
        *,
        session: PruningSessionState,
        candidate: ActionCandidate,
    ) -> ExactHardMaskCommitDecision | None:
        """Execute and validate the selected transition's atomic shadow verdict."""

        if self.commit_authority_mode == "exploratory":
            return None
        authority = self.commit_authority
        if authority is None:  # Defensive: construction already refuses this.
            raise RuntimeError("shadow glocal-v3 commit authority is unavailable")
        if candidate.parent_state_hash != session.state_hash:
            raise ValueError(
                "selected candidate became stale after its predictable birth; "
                "regenerate candidates under the current state"
            )
        if candidate.parent_mask_hash != session.current_mask.mask_hash:
            raise ValueError(
                "selected candidate no longer descends from the current hard mask"
            )
        candidate.resulting_mask.removed_index_set_from(session.current_mask)
        decision = authority.authorize(session=session, candidate=candidate)
        if type(decision) is not ExactHardMaskCommitDecision:
            raise TypeError(
                "shadow glocal-v3 authority did not return an exact-mask "
                "decision receipt"
            )
        expected_mask_hash = (
            candidate.resulting_mask.mask_hash
            if decision.accepted
            else candidate.parent_mask_hash
        )
        if session.current_mask.mask_hash != expected_mask_hash:
            raise RuntimeError(
                "shadow glocal-v3 authority returned before atomically "
                "synchronizing the IRG mask"
            )
        if decision.accepted and (
            not session.committed_candidate_ids
            or session.committed_candidate_ids[-1] != candidate.candidate_id
        ):
            raise RuntimeError(
                "accepted shadow glocal-v3 authority did not commit the exact "
                "IRG candidate"
            )
        if decision.authority_id != authority.authority_id:
            raise ValueError(
                "commit-authority verdict identity differs from its provider"
            )
        decision.validate_transition(
            candidate_id=candidate.candidate_id,
            irg_parent_state_hash=candidate.parent_state_hash,
            parent_mask_hash=candidate.parent_mask_hash,
            candidate_mask_hash=candidate.resulting_mask.mask_hash,
        )
        return decision

    @classmethod
    def _invalidate_response_state_for_refresh(
        cls,
        state: object,
        *,
        current_mask_hash: str,
    ) -> object:
        """Make a frozen response state observably stale before a forced refresh.

        v1--v3 own ``measured_mask_hash`` directly; v4 owns it through its
        intrinsic state.  Invalidation is an atomic retirement of the old
        response chart, not an identity rewrite: no moment, factor, trust
        geometry, atlas, schedule, or diagnostic measured at the old mask may
        survive under the replacement identity.  The invalidation is
        content-addressed and fail-closed if a future version does not expose
        one of the explicitly supported state shapes.
        """

        if not is_dataclass(state):
            raise RuntimeError("model refresh requires a dataclass response state")
        field_names = {item.name for item in fields(state)}
        stale_hash = content_hash(
            {
                "current_mask_hash": str(current_mask_hash),
                "reason": "model_validity_forced_refresh",
            },
            prefix="irg-refresh-required-",
        )
        if "intrinsic_state" in field_names:
            intrinsic = getattr(state, "intrinsic_state")
            refreshed_intrinsic = cls._invalidate_response_state_for_refresh(
                intrinsic,
                current_mask_hash=current_mask_hash,
            )
            updates: dict[str, object] = {
                "intrinsic_state": refreshed_intrinsic,
            }
            # v4 higher-order terms and exact-transition receipts are charts,
            # not timeless observations.  Once the intrinsic chart is
            # invalidated they cannot remain attached to the replacement
            # intrinsic-state identity.
            route_count = len(getattr(state, "route_ids", ()))
            if "transition_receipts" in field_names:
                updates["transition_receipts"] = ()
            if "directional_curvature" in field_names:
                updates["directional_curvature"] = np.zeros(
                    route_count, dtype=np.float64
                )
            if "learned_route_residual" in field_names:
                updates["learned_route_residual"] = np.zeros(
                    route_count, dtype=np.float64
                )
            if "curvature_sample_count" in field_names:
                updates["curvature_sample_count"] = 0
            if "directional_second_response" in field_names:
                updates["directional_second_response"] = None
            if "residual_covariance_inflation" in field_names:
                updates["residual_covariance_inflation"] = None
            return replace(state, **updates)

        if "measured_mask_hash" not in field_names:
            raise RuntimeError(
                "model validity requested refresh, but the version state has no "
                "refreshable measured-mask identity"
            )

        route_count = len(getattr(state, "route_ids", ()))

        # v1: retain only the physical route axis, costs/protection, and the
        # calibration identity.  A fresh response acquisition reconstructs
        # every statistic from a sealed context prefix at the active mask.
        if "mean_response" in field_names and "diagonal_energy" in field_names:
            zeros = np.zeros(route_count, dtype=np.float64)
            updates = {
                "mean_response": zeros,
                "diagonal_energy": zeros,
                "mean_standard_error": zeros,
                "energy_standard_error": zeros,
                "sample_count": 0,
                "measured_mask_hash": stale_hash,
                "source_outcome_hashes": (),
            }
            for name in (
                "response_moments",
                "local_attribution_moments",
                "tripartite_pullback_moments",
                "pullback_phenotype_moments",
                "global_trust_geometry",
                "local_regime_summary",
            ):
                if name in field_names:
                    updates[name] = None
            return replace(state, **updates)

        # v2: return to its explicit unmeasured bootstrap representation.  In
        # particular, the atlas and interaction schedule are chart products;
        # covariance alone must never carry them across a physical transition.
        if "jacobian_factor" in field_names:
            updates = {
                "jacobian_factor": np.zeros((1, route_count), dtype=np.float64),
                "measured_mask_hash": stale_hash,
                "driver_residualized": False,
                "driver_hash": "none",
                "source_outcome_hashes": (),
                "response_moments": None,
                "response_factor": None,
                "response_factor_indices": (),
                "pullback_phenotype_moments": None,
                "global_trust_geometry": None,
                "functional_atlas": None,
                "interaction_schedule": None,
                "local_regime_summary": None,
                "bootstrap_unmeasured": True,
            }
            return replace(
                state,
                **{
                    name: value
                    for name, value in updates.items()
                    if name in field_names
                },
            )

        # v3: it predates v2's explicit bootstrap bit, so use the canonical
        # single-zero-row placeholder.  The stale mask identity forces fresh
        # moments before this placeholder can participate in candidate birth.
        if "route_factor" in field_names and "nuisance_factor" in field_names:
            nuisance_dimension = int(
                np.asarray(getattr(state, "nuisance_factor")).shape[1]
            )
            updates = {
                "route_factor": np.zeros((1, route_count), dtype=np.float64),
                "nuisance_factor": np.zeros((1, nuisance_dimension), dtype=np.float64),
                "measured_mask_hash": stale_hash,
                "source_outcome_hashes": (),
                "response_moments": None,
                "intrinsic_response_moments": None,
                "response_factor": None,
                "response_factor_indices": (),
                "nuisance_response_rows": None,
                "response_metric_whitener": None,
                "metric_route_rows": None,
                "metric_nuisance_rows": None,
                "pullback_phenotype_moments": None,
                "global_trust_geometry": None,
                "intrinsic_atlas_geometry": None,
                "functional_atlas": None,
                "interaction_schedule": None,
                "local_regime_summary": None,
                "variational_atlas_posterior": None,
            }
            return replace(
                state,
                **{
                    name: value
                    for name, value in updates.items()
                    if name in field_names
                },
            )

        raise RuntimeError(
            "model validity requested refresh for an unsupported response-state "
            "shape; refusing to relabel chart-bound evidence"
        )

    def _consume_pending_model_validity(
        self,
        session: PruningSessionState,
    ) -> str | None:
        """Consume the preceding epoch's trust action before proposing masks.

        A returned string is a fail-closed stop reason.  ``refresh`` invalidates
        the response state so ``ensure_minimum_response_state`` must obtain new
        physical rows.  ``shrink`` is applied after candidate generation.
        """

        action = str(session.decision.pending_model_validity_action)
        scale = float(session.decision.recommended_action_scale)
        if action == "refresh":
            if self.runtime_transition_callback is not None:
                # Runtime caches are subordinate to the scientific state.
                # Retire them before any refresh measurement or candidate can
                # be born; the callback is not permitted to mutate evidence.
                self.runtime_transition_callback(
                    action="refresh",
                    version=str(self.version.name),
                    mask_hash=str(session.current_mask.mask_hash),
                )
            session.version_state = self._invalidate_response_state_for_refresh(
                session.version_state,
                current_mask_hash=session.current_mask.mask_hash,
            )
            session.cache.pop(RESPONSE_TRANSITION_CACHE_KEY, None)
            session.decision.geometry_anchor_mask_hash = None
            session.decision.geometry_current_mask_hash = None
            session.decision.cumulative_geometry_removed_indices.clear()
            session.decision.geometry_transition_receipt_hashes.clear()
            session.decision.record_transition(
                "consume_model_refresh",
                "model-validity transition blocked proposals until response geometry was refreshed",
                prior_action=action,
                current_mask_hash=session.current_mask.mask_hash,
            )
            session.decision.pending_model_validity_action = "reuse"
            session.decision.recommended_action_scale = 1.0
            return None
        if action == "fallback":
            session.decision.requested_capability_transition = None
            return (
                "model validity requested a capability fallback, but the common "
                "runner has no implicit transition authority; register and execute "
                "a typed capability transition outside this runner"
            )
        if action == "escalate":
            return "model validity requires a higher scientific version before more candidates"
        if action == "refuse":
            return "model validity refused further candidate generation"
        if action not in {"reuse", "shrink"}:
            raise RuntimeError(f"unknown pending model-validity action {action!r}")
        if action == "shrink" and not (0.0 <= scale < 1.0):
            raise RuntimeError("shrink requires a recommended action scale in [0,1)")
        return None

    def _generate_candidates(
        self,
        session: PruningSessionState,
        budget_targets: Sequence[float],
    ) -> tuple[ActionCandidate, ...]:
        """Generate raw-score paths independently inside each geometric partition."""

        partitioning = session.identity.geometric_partitioning
        if partitioning is None:
            return tuple(self.version.generate_candidates(session, budget_targets))
        if (
            partitioning.route_axis_hash != session.current_mask.route_axis_hash
            or partitioning.route_count != session.current_mask.route_count
        ):
            raise RuntimeError(
                "geometric partitioning does not match the active hard-mask axis"
            )

        sentinel = object()
        prior = session.cache.get(GEOMETRIC_PARTITION_FILTER_CACHE_KEY, sentinel)
        generated: list[ActionCandidate] = []
        frontier_obligations: dict[str, CovarianceRankExpansionObligation] = {}
        session.cache.pop(COVARIANCE_RANK_EXPANSION_CACHE_KEY, None)
        geometry_anchor = (
            session.decision.geometry_anchor_mask_hash
            or str(getattr(session.version_state, "measured_mask_hash", ""))
            or session.current_mask.mask_hash
        )
        try:
            for partition in partitioning.partitions:
                session.cache[GEOMETRIC_PARTITION_FILTER_CACHE_KEY] = (
                    partition.route_indices
                )
                try:
                    derive_partition_covariance_geometry(
                        partitioning=partitioning,
                        partition=partition,
                        version_state=session.version_state,
                        current_mask_hash=geometry_anchor,
                    )
                except UnresolvedCovarianceObligation as error:
                    frontier_obligations[partition.partition_id] = replace(
                        error.obligation,
                        frontier_coverage_prerequisite=True,
                        reason=(
                            "partition lacks a certified safe-screen remainder "
                            "or complete decision-relevant response rank: "
                            + error.obligation.reason
                        ),
                    )
                    # With no certified remainder, an omitted factor direction
                    # may not be silently treated as a dominated action.
                    continue
                local_candidates = tuple(
                    self.version.generate_candidates(session, budget_targets)
                )
                for candidate in local_candidates:
                    actual = candidate_partition(
                        candidate,
                        parent_mask=session.current_mask,
                        partitioning=partitioning,
                    )
                    if actual.partition_id != partition.partition_id:
                        raise RuntimeError(
                            "version solver escaped its active geometric partition"
                        )
                    binding = {
                        **dict(candidate.generator_parameters),
                        "geometric_partition_id": partition.partition_id,
                        "geometric_partition_hash": partition.content_hash,
                        "geometric_partitioning_hash": partitioning.content_hash,
                        "raw_score_comparison_scope": "within_partition_only",
                        "unbound_candidate_id": candidate.candidate_id,
                    }
                    receipt = content_hash(
                        {
                            "candidate_id": candidate.candidate_id,
                            "partition_hash": partition.content_hash,
                            "partitioning_hash": partitioning.content_hash,
                            "route_indices": tuple(
                                int(value)
                                for value in candidate.resolve_removed_indices(
                                    session.current_mask
                                )
                            ),
                        },
                        prefix="irg-partition-legality-",
                    )
                    generated.append(
                        candidate.with_generator_receipt(
                            generator_parameters=binding,
                            legality_receipt=receipt,
                            tags=(
                                *candidate.tags,
                                "partition_local_score_path",
                            ),
                        )
                    )
        finally:
            if prior is sentinel:
                session.cache.pop(GEOMETRIC_PARTITION_FILTER_CACHE_KEY, None)
            else:
                session.cache[GEOMETRIC_PARTITION_FILTER_CACHE_KEY] = prior
        if frontier_obligations:
            session.cache[COVARIANCE_RANK_EXPANSION_CACHE_KEY] = dict(
                frontier_obligations
            )
            session.decision.record_transition(
                "register_covariance_frontier_obligations",
                (
                    "geometric partitions without a certified omitted-route "
                    "remainder were blocked pending whole-response rank growth"
                ),
                partition_ids=tuple(sorted(frontier_obligations)),
                obligation_hashes=tuple(
                    frontier_obligations[partition_id].content_hash
                    for partition_id in sorted(frontier_obligations)
                ),
                conservative_fallback="no candidate from unresolved partition",
                exact_hard_mask_commit_authority=True,
            )
        return deduplicate_candidates(generated)

    @staticmethod
    def _apply_pending_action_scale(
        session: PruningSessionState,
        candidates: Sequence[ActionCandidate],
    ) -> tuple[ActionCandidate, ...]:
        action = str(session.decision.pending_model_validity_action)
        if not candidates:
            return tuple(candidates)
        scale = (
            float(session.decision.recommended_action_scale)
            if action == "shrink"
            else 1.0
        )
        partitioning = session.identity.geometric_partitioning
        if partitioning is None:
            if action != "shrink":
                return tuple(candidates)
            # Compatibility behavior for unpartitioned exploratory sessions.
            largest = max(item.removed_count for item in candidates)
            maximum_width = int(math.floor(scale * largest))
            constrained = tuple(
                item for item in candidates if item.removed_count <= maximum_width
            )
            session.decision.record_transition(
                "consume_model_shrink",
                "legacy unpartitioned model-validity scale constrained total action width",
                recommended_scale=scale,
                original_candidate_count=len(candidates),
                retained_candidate_count=len(constrained),
                original_maximum_removed_routes=largest,
                permitted_maximum_removed_routes=maximum_width,
                compatibility_only=True,
            )
            session.decision.pending_model_validity_action = "reuse"
            session.decision.recommended_action_scale = 1.0
            return constrained

        geometries = {}
        assessments: list[Mapping[str, object]] = []
        constrained_list: list[ActionCandidate] = []
        unresolved = False
        nonexpandable_unresolved = False
        retained_obligations = session.cache.get(
            COVARIANCE_RANK_EXPANSION_CACHE_KEY,
            {},
        )
        expansion_obligations = {
            str(key): value for key, value in dict(retained_obligations).items()
        }
        if any(
            not isinstance(value, CovarianceRankExpansionObligation)
            for value in expansion_obligations.values()
        ):
            raise TypeError("covariance expansion cache contains an invalid obligation")
        trust = PartitionedCovarianceTrustRegion()
        geometry_anchor = (
            session.decision.geometry_anchor_mask_hash
            or str(getattr(session.version_state, "measured_mask_hash", ""))
            or session.current_mask.mask_hash
        )
        cumulative_removed = tuple(
            sorted(session.decision.cumulative_geometry_removed_indices)
        )
        for candidate in candidates:
            partition = candidate_partition(
                candidate,
                parent_mask=session.current_mask,
                partitioning=partitioning,
            )
            try:
                geometry = geometries.get(partition.partition_id)
                if geometry is None:
                    geometry = derive_partition_covariance_geometry(
                        partitioning=partitioning,
                        partition=partition,
                        version_state=session.version_state,
                        current_mask_hash=geometry_anchor,
                    )
                    geometries[partition.partition_id] = geometry
                assessment = trust.assess(
                    candidate,
                    parent_mask=session.current_mask,
                    partitioning=partitioning,
                    geometry=geometry,
                    action_scale=scale,
                    geometry_anchor_mask_hash=geometry_anchor,
                    cumulative_removed_indices=cumulative_removed,
                )
            except UnresolvedCovarianceObligation as error:
                unresolved = True
                obligation = replace(
                    error.obligation,
                    frontier_coverage_prerequisite=True,
                )
                expansion_obligations[obligation.partition_id] = obligation
                assessments.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "partition_id": partition.partition_id,
                        "action": "expand_covariance_rank",
                        "reason": str(error),
                        "obligation_hash": obligation.content_hash,
                        "missing_route_indices": (obligation.missing_route_indices),
                    }
                )
                continue
            except (RuntimeError, ValueError) as error:
                unresolved = True
                nonexpandable_unresolved = True
                assessments.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "partition_id": partition.partition_id,
                        "action": "refresh",
                        "reason": str(error),
                    }
                )
                continue
            assessments.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "partition_id": assessment.partition_id,
                    "action": assessment.action,
                    "reason": assessment.reason,
                    "covariance_leverage": assessment.covariance_leverage,
                    "covariance_limit": assessment.covariance_limit,
                    "layer_removed": dict(assessment.layer_removed),
                    "layer_removed_limits": dict(assessment.layer_removed_limits),
                    "layer_covariance_leverage": dict(
                        assessment.layer_covariance_leverage
                    ),
                    "layer_covariance_limits": dict(assessment.layer_covariance_limits),
                    "geometry_hash": assessment.diagnostics.get("geometry_hash"),
                }
            )
            if assessment.allowed:
                constrained_list.append(candidate)
        constrained = tuple(constrained_list)
        if expansion_obligations:
            session.cache[COVARIANCE_RANK_EXPANSION_CACHE_KEY] = dict(
                expansion_obligations
            )
        elif constrained:
            session.cache.pop(
                COVARIANCE_RANK_EXPANSION_CACHE_KEY,
                None,
            )
        session.decision.record_transition(
            "consume_partitioned_model_trust",
            "partition covariance and layer-local leverage constrained the complete-mask action set",
            recommended_scale=scale,
            original_candidate_count=len(candidates),
            retained_candidate_count=len(constrained),
            partitioning_hash=partitioning.content_hash,
            assessments=tuple(assessments),
            unresolved_geometry=unresolved,
            raw_score_comparison_scope="within_partition_only",
            exact_hard_mask_commit_authority=True,
        )
        session.decision.pending_model_validity_action = (
            "refresh"
            if candidates
            and not constrained
            and (nonexpandable_unresolved or not expansion_obligations)
            else "reuse"
        )
        session.decision.recommended_action_scale = 1.0
        return constrained

    def _covariance_expansion_request(
        self,
        session: PruningSessionState,
        obligation: CovarianceRankExpansionObligation,
        *,
        stage_gate: bool,
    ) -> MeasurementRequest:
        state = getattr(
            session.version_state,
            "intrinsic_state",
            session.version_state,
        )
        factor = getattr(state, "intrinsic_response_factor", None) or getattr(
            state, "response_factor", None
        )
        if factor is None:
            raise RuntimeError(
                "covariance rank expansion requires a current response factor"
            )
        contexts = tuple(str(value) for value in factor.context_ids)
        if not contexts:
            raise RuntimeError(
                "covariance rank expansion requires retained factor contexts"
            )
        route_ids = tuple(
            str(state.route_ids[index]) for index in obligation.required_route_indices
        )
        intrinsic = bool(getattr(self.version, "uses_intrinsic_quotient", False))
        retain = ("route_factor", "nuisance_factor") if intrinsic else ("route_factor",)
        metric_required = bool(
            intrinsic
            and session.identity.metric.geometry
            not in {"euclidean", "contrast_euclidean"}
        )
        return MeasurementRequest.create(
            kind="covariance_rank_expansion",
            state_hash=session.state_hash,
            version=self.version.name,
            context_selector=ContextSelector(
                selector_id="irg_covariance_rank_expansion_prefix_v1",
                requested_count=len(contexts),
                context_ids=contexts,
                random_seed=session.random_seed + session.iteration,
                parameters={
                    "sampling": "reuse_exact_response_factor_contexts",
                    "partition_id": obligation.partition_id,
                },
            ),
            expected_cost=CostEstimate(backward_passes=1.0),
            expected_decision_value=0.0,
            reuse_keys=(
                session.current_mask.mask_hash,
                obligation.source_factor_hash,
                obligation.content_hash,
            ),
            provenance=MeasurementProvenance(
                producer_id=(
                    "mechanistic_systems.circuits.irg." "covariance_rank_expansion"
                ),
                producer_version="1",
                code_hash=content_hash(
                    {
                        "runner": "CommonIRGRunner",
                        "operation": "covariance_rank_expansion",
                    }
                ),
            ),
            payload={
                "mask_hash": session.current_mask.mask_hash,
                "retain": retain,
                "route_indices": obligation.required_route_indices,
                "route_ids": route_ids,
                "response_dimension": (session.identity.metric.output_dimension),
                "screen_before_factor": False,
                "covariance_rank_expansion": True,
                "covariance_rank_reopening_opportunity": not stage_gate,
                "covariance_rank_obligation_hash": obligation.content_hash,
                "partition_id": obligation.partition_id,
                "missing_route_indices": obligation.missing_route_indices,
                "source_response_projection_residual_bound_per_unit_route_coefficient": (
                    obligation.response_projection_residual_bound_per_unit_route_coefficient
                ),
                "source_sketch_residual_spectral_bound": (
                    obligation.sketch_residual_spectral_bound
                ),
                "frontier_coverage_prerequisite": (
                    obligation.frontier_coverage_prerequisite
                ),
                "retain_task_metric_factor": metric_required,
                "require_task_metric_factor": metric_required,
                "measurement_requirement_class": (
                    "chart_validity_prerequisite" if stage_gate else "optional"
                ),
                **(
                    {
                        "measurement_obligation_id": (
                            "covariance-rank:"
                            f"{session.state_hash}:{obligation.content_hash}"
                        )
                    }
                    if stage_gate
                    else {}
                ),
            },
        )

    def _boundary_covariance_expansion_requests(
        self,
        session: PruningSessionState,
    ) -> tuple[MeasurementRequest, ...]:
        """Offer excluded-partition rank growth against conservative exact actions."""

        raw = session.cache.get(COVARIANCE_RANK_EXPANSION_CACHE_KEY)
        if not raw or not self.providers.supports("covariance_rank_expansion"):
            return ()
        obligations = {str(key): value for key, value in dict(raw).items()}
        if any(
            not isinstance(value, CovarianceRankExpansionObligation)
            for value in obligations.values()
        ):
            raise TypeError("covariance expansion cache contains an invalid obligation")
        return tuple(
            self._covariance_expansion_request(
                session,
                obligations[partition_id],
                stage_gate=bool(
                    obligations[partition_id].frontier_coverage_prerequisite
                ),
            )
            for partition_id in sorted(obligations)
        )

    @staticmethod
    def _has_required_covariance_frontier_obligation(
        session: PruningSessionState,
    ) -> bool:
        raw = session.cache.get(COVARIANCE_RANK_EXPANSION_CACHE_KEY)
        if not raw:
            return False
        obligations = tuple(dict(raw).values())
        if any(
            not isinstance(value, CovarianceRankExpansionObligation)
            for value in obligations
        ):
            raise TypeError("covariance expansion cache contains an invalid obligation")
        return any(value.frontier_coverage_prerequisite for value in obligations)

    def _consume_covariance_rank_expansion(
        self,
        *,
        session: PruningSessionState,
        request: MeasurementRequest,
        outcome: object,
    ) -> None:
        raw = session.cache.get(COVARIANCE_RANK_EXPANSION_CACHE_KEY)
        if not raw:
            raise RuntimeError(
                "covariance-rank outcome has no active omitted-route obligation"
            )
        obligations = {str(key): value for key, value in dict(raw).items()}
        partition_id = str(request.payload.get("partition_id", ""))
        obligation = obligations.get(partition_id)
        if not isinstance(obligation, CovarianceRankExpansionObligation):
            raise RuntimeError(
                "covariance-rank outcome belongs to no active partition obligation"
            )
        if (
            request.payload.get("covariance_rank_obligation_hash")
            != obligation.content_hash
        ):
            raise RuntimeError(
                "covariance-rank request differs from the retained obligation"
            )
        updated_version_state = self.version.update_response_state(
            session.version_state,
            outcome,
        )
        updated_state = getattr(
            updated_version_state,
            "intrinsic_state",
            updated_version_state,
        )
        updated_factor = getattr(updated_state, "intrinsic_response_factor", None)
        if callable(updated_factor):
            updated_factor = updated_factor()
        if updated_factor is None:
            updated_factor = getattr(updated_state, "response_factor", None)
        updated_factor_indices = tuple(
            int(value)
            for value in getattr(updated_state, "response_factor_indices", ())
        )
        if updated_factor is None:
            raise RuntimeError(
                "covariance rank expansion did not retain a response factor"
            )
        if updated_factor_indices != obligation.required_route_indices:
            raise RuntimeError(
                "covariance rank expansion did not deliver the exact frozen "
                "required route axis"
            )
        if not set(obligation.current_factor_indices).issubset(
            updated_factor_indices
        ) or set(obligation.missing_route_indices).difference(updated_factor_indices):
            raise RuntimeError(
                "covariance rank expansion failed to preserve the prior factor "
                "axis or discharge every omitted-route obligation"
            )
        updated_projection_residual = float(
            getattr(updated_factor, "projection_residual_bound", 0.0)
        )
        updated_factor_metadata = dict(getattr(updated_factor, "metadata", {}))
        updated_sketch_residual = max(
            float(
                updated_factor_metadata.get(
                    "sketch_residual_spectral_bound",
                    0.0,
                )
            ),
            float(
                updated_factor_metadata.get(
                    "omitted_route_residual_bound",
                    0.0,
                )
            ),
        )
        if (
            not math.isfinite(updated_projection_residual)
            or updated_projection_residual < 0.0
            or not math.isfinite(updated_sketch_residual)
            or updated_sketch_residual < 0.0
        ):
            raise RuntimeError(
                "covariance rank expansion returned invalid residual obligations"
            )
        measured_mask_hash = str(
            getattr(
                updated_state,
                "measured_mask_hash",
                "",
            )
        )
        if measured_mask_hash != session.current_mask.mask_hash:
            raise RuntimeError("covariance rank expansion did not bind the active mask")
        session.version_state = updated_version_state
        superseded_partition_ids = tuple(
            sorted(value for value in obligations if value != partition_id)
        )
        # One expanded response factor changes the source-state/factor hash for
        # every partition obligation.  Retaining siblings would make them
        # stale by construction.  The next candidate generation re-derives
        # any still-missing route obligations against the new factor.
        session.cache.pop(COVARIANCE_RANK_EXPANSION_CACHE_KEY, None)
        session.cache.pop(RESPONSE_TRANSITION_CACHE_KEY, None)
        session.decision.reset_geometry_transport(measured_mask_hash=measured_mask_hash)
        session.decision.record_transition(
            "covariance_rank_expansion",
            (
                "whole-response factor rank grew to discharge a "
                "decision-relevant omitted-route obligation"
            ),
            partition_id=partition_id,
            obligation_hash=obligation.content_hash,
            request_id=request.request_id,
            outcome_hash=getattr(outcome, "integrity_hash", "unknown"),
            required_route_indices=obligation.required_route_indices,
            delivered_route_indices=updated_factor_indices,
            updated_response_projection_residual_bound_per_unit_route_coefficient=(
                updated_projection_residual
            ),
            updated_sketch_residual_spectral_bound=updated_sketch_residual,
            superseded_partition_obligations=superseded_partition_ids,
            sibling_obligations_require_rederivation=bool(superseded_partition_ids),
            boundary_competition=bool(
                request.payload.get("covariance_rank_reopening_opportunity", False)
            ),
        )

    def _resolve_covariance_rank_obligation(
        self,
        session: PruningSessionState,
    ) -> str | None:
        raw = session.cache.get(COVARIANCE_RANK_EXPANSION_CACHE_KEY)
        if not raw:
            return None
        obligations = {str(key): value for key, value in dict(raw).items()}
        partition_id = min(obligations)
        obligation = obligations[partition_id]
        if not isinstance(obligation, CovarianceRankExpansionObligation):
            raise TypeError("covariance expansion cache contains an invalid obligation")
        partitioning = session.identity.geometric_partitioning
        state = getattr(
            session.version_state,
            "intrinsic_state",
            session.version_state,
        )
        response_factor = getattr(state, "intrinsic_response_factor", None)
        if callable(response_factor):
            response_factor = response_factor()
        if response_factor is None:
            response_factor = getattr(state, "response_factor", None)
        factor_hash = (
            ""
            if response_factor is None
            else str(
                getattr(response_factor, "content_hash", "")
                or content_hash(repr(response_factor), prefix="opaque-factor-")
            )
        )
        factor_indices = tuple(
            int(value) for value in getattr(state, "response_factor_indices", ())
        )
        measured_mask_hash = str(getattr(state, "measured_mask_hash", ""))
        stale_reasons: list[str] = []
        if (
            partitioning is None
            or obligation.partitioning_hash != partitioning.content_hash
            or obligation.partition_id
            not in {item.partition_id for item in partitioning.partitions}
        ):
            stale_reasons.append("partitioning")
        if (
            obligation.measured_mask_hash != session.current_mask.mask_hash
            or measured_mask_hash != session.current_mask.mask_hash
        ):
            stale_reasons.append("measured_mask")
        if obligation.source_state_hash != self._version_state_hash(
            session.version_state
        ):
            stale_reasons.append("version_state")
        if (
            factor_hash != obligation.source_factor_hash
            or factor_indices != obligation.current_factor_indices
        ):
            stale_reasons.append("response_factor")
        if stale_reasons:
            session.cache.pop(COVARIANCE_RANK_EXPANSION_CACHE_KEY, None)
            session.decision.record_transition(
                "discard_stale_covariance_rank_obligation",
                "covariance-rank obligation no longer belongs to the active chart",
                obligation_hash=obligation.content_hash,
                stale_coordinates=tuple(stale_reasons),
            )
            raise RuntimeError(
                "stale covariance-rank obligation differs on: "
                + ",".join(stale_reasons)
            )
        if not self.providers.supports("covariance_rank_expansion"):
            session.decision.pending_model_validity_action = "refuse"
            session.decision.requested_capability_transition = None
            session.decision.record_transition(
                "covariance_rank_expansion_unavailable",
                (
                    "decision-relevant omitted-route covariance could not be "
                    "expanded by the registered providers"
                ),
                partition_id=partition_id,
                obligation_hash=obligation.content_hash,
                requested_capability_transition=None,
            )
            return "blocked"
        request = self._covariance_expansion_request(
            session,
            obligation,
            stage_gate=True,
        )
        selected = self._select_request_batch(
            session,
            (request,),
            phase="response_state",
            maximum_requests=1,
        )
        if not selected:
            session.decision.record_transition(
                "covariance_rank_expansion_budget_stop",
                "required covariance-rank packet did not fit the hard caps",
                partition_id=partition_id,
                obligation_hash=obligation.content_hash,
            )
            return "blocked"
        outcome = self._execute_and_record(session, selected[0])
        self._consume_covariance_rank_expansion(
            session=session,
            request=selected[0],
            outcome=outcome,
        )
        return "expanded"

    @staticmethod
    def _active_global_trust_geometry(session: PruningSessionState):
        state = getattr(
            session.version_state,
            "intrinsic_state",
            session.version_state,
        )
        return getattr(state, "global_trust_geometry", None)

    def _resolve_global_trust_rank_obligation(
        self,
        session: PruningSessionState,
    ) -> str | None:
        raw = session.cache.get(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY)
        if not raw:
            return None
        obligations = tuple(dict(raw).values())
        if any(
            not isinstance(item, GlobalTrustRankGrowthObligation)
            for item in obligations
        ):
            raise TypeError(
                "global trust rank-growth cache contains an invalid obligation"
            )
        geometry = self._active_global_trust_geometry(session)
        if geometry is None:
            session.cache.pop(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, None)
            raise RuntimeError(
                "global trust rank-growth obligation lost its source geometry"
            )
        source_hashes = {item.source_geometry_hash for item in obligations}
        if source_hashes != {geometry.content_hash}:
            session.cache.pop(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, None)
            raise RuntimeError(
                "global trust rank-growth obligation is stale for this chart"
            )
        requested_rank = max(item.requested_rank for item in obligations)
        delivered_capacity = int(geometry.metadata.get("requested_rank", geometry.rank))
        if requested_rank <= delivered_capacity:
            session.cache.pop(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, None)
            return None
        if not self.providers.supports("response_moments"):
            session.decision.pending_model_validity_action = "refuse"
            session.decision.requested_capability_transition = None
            session.cache.pop(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, None)
            session.decision.record_transition(
                "global_trust_rank_growth_unavailable",
                "decision-relevant FD residual could not be tightened",
                current_rank=delivered_capacity,
                requested_rank=requested_rank,
                requested_capability_transition=None,
            )
            return "blocked"
        session.cache[GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY] = requested_rank
        session.cache.pop(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, None)
        session.decision.pending_model_validity_action = "refresh"
        session.decision.recommended_action_scale = 0.0
        session.decision.record_transition(
            "global_trust_rank_growth_scheduled",
            "candidate remained ambiguous only because of the certified FD residual",
            source_geometry_hash=geometry.content_hash,
            current_rank=delivered_capacity,
            requested_rank=requested_rank,
            action_hashes=tuple(sorted(item.action_hash for item in obligations)),
        )
        return "scheduled"

    def _execute_and_record(
        self, session: PruningSessionState, request: MeasurementRequest
    ):
        budget_estimate = request.expected_cost
        if self.commit_authority_mode == "glocal_v3_shadow":
            cost_envelope = getattr(
                self.commit_authority,
                "guidance_cost_envelope",
                None,
            )
            if callable(cost_envelope):
                budget_estimate = cost_envelope(request)
        if not session.compute_ledger.can_afford(budget_estimate):
            raise RuntimeError(
                "measurement request exceeds the remaining compute budget"
            )
        if self.commit_authority_mode == "glocal_v3_shadow":
            preflight = getattr(
                self.commit_authority,
                "preflight_guidance_measurement",
                None,
            )
            if not callable(preflight):
                raise RuntimeError(
                    "shadow commit authority cannot preflight guidance against "
                    "the campaign-global resource ledger"
                )
            preflight(request)
        outcome = None
        authority_executed = False
        if self.commit_authority_mode == "glocal_v3_shadow":
            execute_authority_measurement = getattr(
                self.commit_authority,
                "execute_guidance_measurement",
                None,
            )
            if callable(execute_authority_measurement):
                outcome = execute_authority_measurement(
                    session=session,
                    request=request,
                )
                authority_executed = outcome is not None
        if outcome is None:
            outcome = self.providers.execute(request)
        session.assert_request_state(outcome.state_hash)
        if self.commit_authority_mode == "glocal_v3_shadow" and not authority_executed:
            register = getattr(
                self.commit_authority,
                "register_guidance_measurement",
                None,
            )
            if not callable(register):
                raise RuntimeError(
                    "shadow commit authority cannot register realized guidance "
                    "resource usage"
                )
            register(request=request, outcome=outcome)
        session.compute_ledger.record(outcome.realized_cost)
        self.cost_model.observe(
            request,
            outcome.realized_cost,
            context_count=len(outcome.context_ids),
        )
        return outcome

    def _advisor_requests(
        self,
        session: PruningSessionState,
        *,
        phase: str,
        search: AdaptiveBoundarySearch | None = None,
    ) -> tuple[MeasurementRequest, ...]:
        requests: list[MeasurementRequest] = []
        for adapter in self.measurement_adapters:
            requests.extend(adapter.requests(session, str(phase), search))
        for advisor in self.measurement_advisors:
            requests.extend(advisor(session, str(phase), search))
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError(
                "measurement advisors emitted duplicate request identities"
            )
        for request in requests:
            if request.state_hash != session.state_hash:
                raise ValueError(
                    "measurement adapter emitted a request for a stale pruning state"
                )
            if request.version != session.active_version:
                raise ValueError(
                    "measurement adapter emitted a request for another response version"
                )
            authority_supports = False
            if self.commit_authority_mode == "glocal_v3_shadow":
                supports = getattr(
                    self.commit_authority,
                    "supports_guidance_measurement",
                    None,
                )
                authority_supports = bool(callable(supports) and supports(request))
            if not self.providers.supports(request.kind) and not authority_supports:
                raise RuntimeError(
                    f"measurement opportunity {request.kind!r} has no registered provider"
                )
        return tuple(requests)

    def _adapter_blocking_reasons(
        self,
        session: PruningSessionState,
        *,
        phase: str,
    ) -> tuple[str, ...]:
        """Return durable stage gates independently of emitted requests."""

        reasons = tuple(
            reason
            for adapter in self.measurement_adapters
            for reason in (adapter.blocking_reason(session, str(phase)),)
            if reason is not None
        )
        return tuple(dict.fromkeys(reasons))

    def _decision_uncertainty_snapshot(
        self,
        session: PruningSessionState,
        *,
        phase: str,
    ) -> DecisionUncertaintySnapshot:
        prediction_errors: list[float] = []
        boundary_gaps: list[float] = []
        released_costs: list[float] = []
        safety_widths: list[float] = []
        if session.frontier is not None:
            for candidate_id in session.frontier.candidate_ids:
                entry = session.frontier.entry(candidate_id)
                if entry.candidate.parent_state_hash != session.state_hash:
                    continue
                if entry.prediction is not None:
                    prediction_errors.append(float(entry.prediction.standard_error))
                    boundary_gaps.append(
                        abs(
                            float(entry.prediction.predicted_quality_loss)
                            - float(session.stopping.maximum_quality_loss)
                        )
                    )
                released_costs.append(
                    float(
                        entry.candidate.released_cost.coordinate(
                            session.identity.cost.primary_coordinate
                        )
                    )
                )
                if entry.standard_error is not None:
                    safety_widths.append(4.0 * float(entry.standard_error))
        oracle_sample_counts = []
        if session.frontier is not None:
            for candidate_id in session.frontier.candidate_ids:
                entry = session.frontier.entry(candidate_id)
                if entry.candidate.parent_state_hash != session.state_hash:
                    continue
                count = sum(receipt.count for receipt in entry.oracle_receipts)
                if count > 0:
                    oracle_sample_counts.append(count)
        response_variance = max(
            (value * value for value in prediction_errors),
            default=0.0,
        )
        response_state = getattr(
            session.version_state,
            "intrinsic_state",
            session.version_state,
        )
        state_sample_count = max(
            0,
            int(getattr(response_state, "sample_count", 0)),
        )
        if response_variance <= 0.0 and state_sample_count <= 0:
            response_variance = 1.0
        elif response_variance <= 0.0:
            response_variance = 1.0 / float(state_sample_count + 1)
        uncovered = sum(
            1 for value in session.decision.context_coverage.values() if int(value) <= 0
        )
        if phase == "response_state" and uncovered == 0:
            uncovered = 1
        return DecisionUncertaintySnapshot(
            state_hash=session.state_hash,
            frontier_boundary_gap=min(boundary_gaps, default=0.0),
            frontier_regret_scale=max(released_costs, default=1.0),
            safety_interval_width=max(safety_widths, default=1.0),
            calibration_variance=max(
                float(session.decision.calibration_uncertainty) ** 2,
                0.0,
            ),
            response_variance=response_variance,
            model_failure_probability=min(
                max(float(session.decision.model_residual_scale), 0.0),
                1.0,
            ),
            screen_uncertainty=min(
                1.0,
                float(uncovered)
                / float(max(uncovered, len(session.decision.context_coverage), 1)),
            ),
            uncovered_strata=uncovered,
            metadata={
                "phase": str(phase),
                "caller_expected_decision_value_is_authoritative": False,
                "exact_hard_mask_evidence_remains_commit_authority": True,
                "calibration_logits_sample_count": int(
                    session.decision.calibration_sample_count
                ),
                "calibration_refit_sample_count": int(
                    session.decision.calibration_sample_count
                ),
                "jacobian_rows_sample_count": state_sample_count,
                "marginal_statistics_sample_count": state_sample_count,
                "response_moments_sample_count": state_sample_count,
                "response_factor_sample_count": state_sample_count,
                "metric_factor_sample_count": state_sample_count,
                "covariance_rank_expansion_sample_count": state_sample_count,
                "hard_mask_oracle_sample_count": min(oracle_sample_counts, default=0),
                "model_validity_sample_count": len(
                    session.decision.predictive_validity_receipt_hashes
                ),
                "response_transition_sample_count": len(
                    session.decision.geometry_transition_receipt_hashes
                ),
            },
        )

    def _score_request(
        self,
        session: PruningSessionState,
        request: MeasurementRequest,
        *,
        phase: str,
        snapshot: DecisionUncertaintySnapshot | None = None,
    ) -> ScoredRequest:
        predicted_cost = self.cost_model.predict(
            request,
            context_count=max(1, request.context_selector.requested_count),
            packing_degree=max(1, len(request.candidate_ids)),
        )
        resolved_snapshot = (
            snapshot
            if snapshot is not None
            else self._decision_uncertainty_snapshot(session, phase=phase)
        )
        adapter = self._adapter_by_kind.get(str(request.kind))
        forecast = (
            None
            if adapter is None
            else adapter.forecast(
                session,
                request,
                resolved_snapshot,
                phase,
            )
        )
        if forecast is None and self.require_typed_acquisition_forecasts:
            forecast = self._glocal_builtin_modality_forecast(
                session,
                request,
                resolved_snapshot,
                phase=phase,
            )
            if forecast is None:
                raise RuntimeError(
                    "glocal-v3 shadow acquisition requires a typed "
                    "modality-owned forecast before provider execution; "
                    f"kind={request.kind!r}, phase={phase!r}, "
                    f"request_id={request.request_id!r}"
                )
        opportunity = self.acquisition_planner.opportunity(
            request,
            resolved_snapshot,
            phase=phase,
            forecast=forecast,
        )
        return self.acquisition_planner.score(
            opportunity,
            predicted_cost=predicted_cost.scalar(),
        )

    @staticmethod
    def _forecast_count_contraction(
        *,
        current_variance: float,
        current_count: int,
        added_count: int,
    ) -> float:
        """Predictable allocation fantasy with one explicit prior row.

        This contraction is never a claim-bearing interval.  It only prices
        the next physical packet after its modality and request identity have
        been validated.
        """

        variance = max(float(current_variance), 0.0)
        prior = max(int(current_count), 0) + 1
        added = max(int(added_count), 1)
        return variance * float(prior) / float(prior + added)

    def _glocal_builtin_modality_forecast(
        self,
        session: PruningSessionState,
        request: MeasurementRequest,
        snapshot: DecisionUncertaintySnapshot,
        *,
        phase: str,
    ) -> AcquisitionForecast | None:
        """Forecast canonical direct requests that have no effect adapter.

        Adapter-owned calibration and adaptive response-prefix requests must
        provide their own forecast.  This policy owns only direct requests
        emitted by the version/runner implementations whose producer,
        physical state, and modality-specific coordinates can be verified.
        """

        if (
            request.state_hash != session.state_hash
            or request.state_hash != snapshot.state_hash
            or request.version != session.active_version
        ):
            raise RuntimeError(
                "glocal modality forecast received a stale request/state binding"
            )
        kind = str(request.kind)
        added_contexts = max(1, int(request.context_selector.requested_count))
        request_binding = {
            "request_cache_key": request.cache_key,
            "request_payload_hash": content_hash(dict(request.payload)),
            "request_selector_hash": content_hash(request.context_selector),
            "phase": str(phase),
            "version": str(session.active_version),
            "mask_hash": session.current_mask.mask_hash,
        }

        if kind in self._CANONICAL_VERSION_RESPONSE_KINDS:
            if request.provenance.producer_id != "mechanistic_systems.circuits.irg":
                return None
            if kind == "directional_curvature":
                if (
                    request.context_selector.selector_id
                    != "irg_v4_screened_direction_selector_v2"
                    or not request.reuse_keys
                    or request.reuse_keys[0] != session.current_mask.mask_hash
                ):
                    raise RuntimeError(
                        "directional-curvature forecast lost its canonical "
                        "mask/selector binding"
                    )
            elif kind == "finite_dose_response":
                if (
                    request.context_selector.selector_id
                    != "irg_v4_finite_dose_contexts_v1"
                    or request.payload.get("mask_hash")
                    != session.current_mask.mask_hash
                ):
                    raise RuntimeError(
                        "finite-dose forecast lost its canonical mask/selector binding"
                    )
            elif (
                request.context_selector.selector_id
                != "irg_active_contrast_context_selector_v1"
                or request.payload.get("mask_hash") != session.current_mask.mask_hash
            ):
                raise RuntimeError(
                    "version-response forecast lost its canonical mask/selector "
                    "binding"
                )
            current_count = max(
                0,
                int(snapshot.metadata.get(f"{kind}_sample_count", 0)),
            )
            current_variance = max(float(snapshot.response_variance), 0.0)
            return AcquisitionForecast(
                request_id=request.request_id,
                state_hash=request.state_hash,
                forecast_model_id=(
                    "canonical_whole_vector_response_row_contraction_v1"
                ),
                uncertainty_source="frontier_response_geometry",
                current_variance=current_variance,
                fantasy_posterior_variance=self._forecast_count_contraction(
                    current_variance=current_variance,
                    current_count=current_count,
                    added_count=added_contexts,
                ),
                current_sample_count=current_count,
                fantasy_sample_count=current_count + added_contexts,
                rationale=(
                    "price the canonical whole-vector response packet on its "
                    "frozen physical context axis"
                ),
                metadata={
                    **request_binding,
                    "allocation_only": True,
                    "claim_bound_narrowing": False,
                    "modality": kind,
                    "count_axis": "whole_route_response_context_rows",
                    "route_indices": tuple(
                        int(value) for value in request.payload.get("route_indices", ())
                    ),
                    "screen_moment_hash": request.payload.get("screen_moment_hash"),
                },
            )

        if kind == "covariance_rank_expansion":
            if (
                request.provenance.producer_id
                != "mechanistic_systems.circuits.irg.covariance_rank_expansion"
                or request.context_selector.selector_id
                != "irg_covariance_rank_expansion_prefix_v1"
                or request.payload.get("mask_hash") != session.current_mask.mask_hash
            ):
                return None
            obligation_hash = str(
                request.payload.get("covariance_rank_obligation_hash", "")
            )
            missing = tuple(
                int(value) for value in request.payload.get("missing_route_indices", ())
            )
            retained = tuple(
                int(value) for value in request.payload.get("route_indices", ())
            )
            if (
                not obligation_hash
                or not missing
                or not set(missing).issubset(retained)
            ):
                raise RuntimeError(
                    "covariance-rank forecast lacks its frozen omitted-route "
                    "obligation"
                )
            current_directions = max(0, len(retained) - len(missing))
            added_directions = len(missing)
            current_variance = max(
                float(snapshot.response_variance),
                float(snapshot.screen_uncertainty) ** 2,
                1e-12,
            )
            posterior = self._forecast_count_contraction(
                current_variance=current_variance,
                current_count=current_directions,
                added_count=added_directions,
            )
            return AcquisitionForecast(
                request_id=request.request_id,
                state_hash=request.state_hash,
                forecast_model_id="omitted_route_covariance_rank_growth_v1",
                uncertainty_source="omitted_route_covariance_geometry",
                current_variance=current_variance,
                fantasy_posterior_variance=posterior,
                current_sample_count=current_directions,
                fantasy_sample_count=current_directions + added_directions,
                rationale=(
                    "price rank growth by the frozen omitted-route obligation "
                    "without treating the working covariance as safety evidence"
                ),
                metadata={
                    **request_binding,
                    "allocation_only": True,
                    "claim_bound_narrowing": False,
                    "count_axis": "represented_route_directions",
                    "obligation_hash": obligation_hash,
                    "partition_id": request.payload.get("partition_id"),
                    "missing_route_indices": missing,
                    "source_factor_hash": (
                        request.reuse_keys[1] if len(request.reuse_keys) > 1 else ""
                    ),
                },
            )

        if kind == "hard_mask_oracle":
            candidate_only = (
                self.boundary_metric_mode == "candidate_bce_relative_to_dense_baseline"
            )
            expected_selector = (
                "irg_candidate_absolute_bce_active_boundary_selector_v1"
                if candidate_only
                else "irg_paired_oracle_active_boundary_selector_v1"
            )
            if (
                request.provenance.producer_id
                != "mechanistic_systems.circuits.irg.runner"
                or request.context_selector.selector_id != expected_selector
                or not request.candidate_ids
                or request.payload.get(
                    (
                        "candidate_parent_mask_hash"
                        if candidate_only
                        else "parent_mask_hash"
                    )
                )
                != session.current_mask.mask_hash
                or bool(request.payload.get("paired_against_parent"))
                is (True if candidate_only else False)
            ):
                return None
            mask_hashes = tuple(
                str(value) for value in request.payload.get("candidate_mask_hashes", ())
            )
            if len(mask_hashes) != len(request.candidate_ids):
                raise RuntimeError(
                    "hard-mask oracle forecast candidate and mask axes differ"
                )
            current_count = max(
                0,
                int(snapshot.metadata.get("hard_mask_oracle_sample_count", 0)),
            )
            current_variance = (
                1.0
                if current_count == 0
                else min(
                    1.0,
                    max(
                        (0.5 * float(snapshot.safety_interval_width)) ** 2,
                        1e-12,
                    ),
                )
            )
            glocal_brier = self.commit_authority_mode == "glocal_v3_shadow"
            return AcquisitionForecast(
                request_id=request.request_id,
                state_hash=request.state_hash,
                forecast_model_id=(
                    "bounded_paired_brier_oracle_packet_v1"
                    if glocal_brier
                    else "paired_bce_boundary_precision_packet_v1"
                ),
                uncertainty_source=(
                    "bounded_exact_mask_contrast"
                    if glocal_brier
                    else "paired_bce_boundary_contrast"
                ),
                current_variance=current_variance,
                fantasy_posterior_variance=self._forecast_count_contraction(
                    current_variance=current_variance,
                    current_count=current_count,
                    added_count=added_contexts,
                ),
                current_sample_count=current_count,
                fantasy_sample_count=current_count + added_contexts,
                rationale=(
                    "price an exact paired hard-mask packet on the frozen "
                    "candidate axis; only the campaign evidence process may commit"
                    if glocal_brier
                    else "price the next exact paired BCE packet needed to "
                    "resolve the frozen complete-mask boundary"
                ),
                metadata={
                    **request_binding,
                    "allocation_only": True,
                    "claim_bound_narrowing": False,
                    "candidate_axis_hash": content_hash(tuple(request.candidate_ids)),
                    "candidate_mask_axis_hash": content_hash(mask_hashes),
                    "excluded_context_ids_hash": content_hash(
                        tuple(
                            request.context_selector.parameters.get(
                                "exclude_context_ids", ()
                            )
                        )
                    ),
                    **(
                        {"bounded_increment_range": (-1.0, 1.0)}
                        if glocal_brier
                        else {
                            "bce_tail_assumption": (
                                "paired finite-variance staged-Student-t"
                            ),
                            "claim_bearing": False,
                        }
                    ),
                },
            )

        if kind == "response_transition":
            constraint = session.identity.response_constraint
            if (
                request.provenance.producer_id
                != ("mechanistic_systems.circuits.irg.runner." "response_transition")
                or request.context_selector.selector_id
                != "irg_paired_response_transition_selector_v2"
                or len(request.candidate_ids) != 1
                or request.payload.get("parent_mask_hash")
                != session.current_mask.mask_hash
                or constraint is None
                or request.payload.get("response_constraint_hash")
                != constraint.content_hash
            ):
                return None
            actions = tuple(
                int(value) for value in request.payload.get("action_indices", ())
            )
            if not actions:
                raise RuntimeError(
                    "response-transition forecast requires a physical action axis"
                )
            current_count = max(
                0,
                int(snapshot.metadata.get("response_transition_sample_count", 0)),
            )
            failure_probability = min(
                max(float(snapshot.model_failure_probability), 0.0),
                1.0,
            )
            current_variance = failure_probability * (1.0 - failure_probability)
            return AcquisitionForecast(
                request_id=request.request_id,
                state_hash=request.state_hash,
                forecast_model_id="paired_response_transition_validity_packet_v1",
                uncertainty_source="local_model_validity",
                current_variance=current_variance,
                fantasy_posterior_variance=self._forecast_count_contraction(
                    current_variance=current_variance,
                    current_count=current_count,
                    added_count=added_contexts,
                ),
                current_sample_count=current_count,
                fantasy_sample_count=current_count + added_contexts,
                rationale=(
                    "price paired transition evidence for the selected physical "
                    "action and frozen response-safe constraint"
                ),
                metadata={
                    **request_binding,
                    "allocation_only": True,
                    "claim_bound_narrowing": False,
                    "action_axis_hash": content_hash(actions),
                    "candidate_id": request.candidate_ids[0],
                    "candidate_mask_hash": request.payload.get("candidate_mask_hash"),
                    "response_constraint_hash": constraint.content_hash,
                },
            )
        return None

    def _select_request_batch(
        self,
        session: PruningSessionState,
        requests: Sequence[MeasurementRequest],
        *,
        phase: str,
        maximum_requests: int = 1,
    ) -> tuple[MeasurementRequest, ...]:
        if not requests:
            return ()
        snapshot = self._decision_uncertainty_snapshot(session, phase=phase)
        scored_all = tuple(
            self._score_request(
                session,
                request,
                phase=phase,
                snapshot=snapshot,
            )
            for request in requests
        )
        resource_eligible: list[ScoredRequest] = []
        ineligible: dict[str, str] = {}
        for item in scored_all:
            request = item.request
            if self.commit_authority_mode == "glocal_v3_shadow":
                eligibility = getattr(
                    self.commit_authority,
                    "guidance_measurement_eligibility",
                    None,
                )
                if callable(eligibility):
                    verdict = eligibility(request)
                    if isinstance(verdict, tuple):
                        allowed, refusal_reason = verdict
                    else:
                        allowed, refusal_reason = bool(verdict), (
                            "shadow authority refused this guidance measurement"
                        )
                    if not bool(allowed):
                        ineligible[request.request_id] = str(refusal_reason)
                        continue
            predicted = CostEstimate(
                wall_seconds=float(item.predicted_cost),
                peak_memory_bytes=float(request.expected_cost.peak_memory_bytes),
            )
            if not session.compute_ledger.can_afford(predicted):
                ineligible[request.request_id] = (
                    "predicted wall cost exceeds remaining hard budget"
                )
                continue
            if (
                self.acquisition_memory_limit_bytes is not None
                and request.expected_cost.peak_memory_bytes
                > float(self.acquisition_memory_limit_bytes)
            ):
                ineligible[request.request_id] = (
                    "declared peak memory exceeds acquisition hard cap"
                )
                continue
            resource_eligible.append(item)
        required_groups: dict[str, list[ScoredRequest]] = {}
        for item in scored_all:
            if not item.stage_gate_required:
                continue
            if not item.obligation_id:
                raise RuntimeError(
                    "stage-gate measurement lost its frozen obligation ID"
                )
            required_groups.setdefault(item.obligation_id, []).append(item)
        if required_groups:
            eligible_ids = {item.request.request_id for item in resource_eligible}
            unavailable = tuple(
                obligation_id
                for obligation_id, alternatives in sorted(required_groups.items())
                if not any(
                    item.request.request_id in eligible_ids for item in alternatives
                )
            )
            if unavailable:
                for item in resource_eligible:
                    ineligible[item.request.request_id] = (
                        "selection stopped because prerequisite obligation(s) "
                        f"{unavailable!r} have no affordable alternative"
                    )
                scored: list[ScoredRequest] = []
            else:
                scored = [
                    item for item in resource_eligible if item.stage_gate_required
                ]
                for item in resource_eligible:
                    if not item.stage_gate_required:
                        ineligible[item.request.request_id] = (
                            "optional measurement deferred until all prerequisite "
                            "obligations are discharged"
                        )
        else:
            scored = []
            for item in resource_eligible:
                if item.score <= self.minimum_acquisition_score:
                    ineligible[item.request.request_id] = (
                        "predicted information value does not beat the frozen "
                        f"no-measurement outside option {self.minimum_acquisition_score}"
                    )
                    continue
                scored.append(item)
        selected = select_request_batch(
            tuple(scored),
            maximum_requests=maximum_requests,
            maximum_peak_memory_bytes=self.acquisition_memory_limit_bytes,
            require_kind_diversity=False,
        )
        receipt = AcquisitionSelectionReceipt(
            state_hash=session.state_hash,
            phase=str(phase),
            uncertainty_snapshot_hash=snapshot.content_hash,
            proposed_request_ids=tuple(item.request.request_id for item in scored_all),
            request_ids=tuple(item.request.request_id for item in scored),
            opportunity_hashes=tuple(item.opportunity_hash for item in scored),
            forecast_hashes=tuple(item.forecast_hash for item in scored),
            scores=tuple(float(item.score) for item in scored),
            predicted_costs=tuple(float(item.predicted_cost) for item in scored),
            selected_request_ids=tuple(request.request_id for request in selected),
            requirement_classes={
                item.request.request_id: item.requirement_class for item in scored_all
            },
            obligation_ids={
                item.request.request_id: item.obligation_id
                for item in scored_all
                if item.obligation_id is not None
            },
            outside_option_score=self.minimum_acquisition_score,
            ineligible_request_reasons=ineligible,
        )
        session.cache[f"irg_acquisition_selection:{receipt.content_hash}"] = receipt
        session.decision.acquisition_receipt_hashes.append(receipt.content_hash)
        session.decision.record_transition(
            "unified_acquisition_selection",
            "common planner selected measurements by predictable uncertainty reduction per learned cost",
            phase=phase,
            acquisition_receipt_hash=receipt.content_hash,
            request_ids=receipt.request_ids,
            selected_request_ids=receipt.selected_request_ids,
            forecast_hashes=receipt.forecast_hashes,
            ineligible_request_reasons=receipt.ineligible_request_reasons,
        )
        if not selected:
            resource_stop = any(
                "hard budget" in reason
                or "hard cap" in reason
                or "prerequisite obligation" in reason
                for reason in receipt.ineligible_request_reasons.values()
            )
            session.decision.record_transition(
                (
                    "acquisition_budget_stop"
                    if resource_stop
                    else "acquisition_value_stop"
                ),
                (
                    "no proposed measurement fits all hard resource or prerequisite caps"
                    if resource_stop
                    else "no optional measurement beats the frozen no-measurement outside option"
                ),
                phase=phase,
                acquisition_receipt_hash=receipt.content_hash,
            )
        return selected

    def _consume_auxiliary(
        self,
        session: PruningSessionState,
        request: MeasurementRequest,
        outcome: object,
    ) -> None:
        adapter = self._adapter_by_kind.get(str(request.kind))
        authority_consumed = False
        if self.commit_authority_mode == "glocal_v3_shadow":
            consume_authority_transition = getattr(
                self.commit_authority,
                "consume_certified_guidance_transition",
                None,
            )
            if callable(consume_authority_transition):
                authority_consumed = bool(
                    consume_authority_transition(
                        session=session,
                        request=request,
                        outcome=outcome,
                    )
                )
        if adapter is not None and not authority_consumed:
            adapter.consume(session, request, outcome)
            adapter_id = adapter.adapter_id
        else:
            adapter_id = (
                None
                if adapter is None
                else f"{adapter.adapter_id}:campaign_authority_consumed"
            )
        consumer = self.outcome_consumers.get(request.kind)
        if adapter is None and consumer is None and not authority_consumed:
            raise RuntimeError(
                f"measurement kind {request.kind!r} was selected by the common acquisition engine "
                "but has no registered outcome consumer"
            )
        if consumer is not None and not authority_consumed:
            consumer(session, request, outcome)
        session.decision.record_transition(
            "auxiliary_measurement_update",
            "common acquisition engine selected a non-version measurement",
            request_id=request.request_id,
            kind=request.kind,
            outcome_hash=getattr(outcome, "integrity_hash", "unknown"),
            acquisition_adapter_id=adapter_id,
        )

    def ensure_minimum_response_state(self, session: PruningSessionState) -> None:
        for _ in range(self.maximum_measurement_refreshes):
            version_requests = tuple(
                self._bind_measurement_requirement(
                    request,
                    requirement_class="candidate_birth_prerequisite",
                    obligation_id=(
                        "candidate-birth:"
                        f"{session.state_hash}:{request.kind}:{request.cache_key}"
                    ),
                )
                for request in self.version.required_measurements(session)
            )
            advisor_requests = self._advisor_requests(session, phase="response_state")
            # A registered stratified response adapter owns candidate-birth
            # sampling.  Do not let the version's compatibility request race
            # it with a separate fixed-size context selection.
            if "response_moments" in self._adapter_by_kind:
                version_requests = tuple(
                    request
                    for request in version_requests
                    if request.kind
                    not in {
                        "jacobian_rows",
                        "marginal_statistics",
                        "response_moments",
                        "response_factor",
                        "metric_factor",
                    }
                )
            pool = version_requests + advisor_requests
            if not pool:
                if session.cache.get(GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY) is not None:
                    raise RuntimeError(
                        "global trust rank-growth refresh exhausted its "
                        "measurement opportunities before delivering the "
                        "frozen requested rank"
                    )
                return
            selected = self._select_request_batch(
                session,
                pool,
                phase="response_state",
                maximum_requests=1,
            )
            if not selected:
                raise RuntimeError(
                    "no affordable response-state measurement was selected"
                )
            request = selected[0]
            outcome = self._execute_and_record(session, request)
            if request.request_id in {item.request_id for item in version_requests}:
                session.version_state = self.version.update_response_state(
                    session.version_state, outcome
                )
                measured_mask_hash = str(
                    getattr(session.version_state, "measured_mask_hash", "")
                )
                if measured_mask_hash != session.current_mask.mask_hash:
                    raise RuntimeError(
                        "fresh response measurement did not bind the active complete mask"
                    )
                requested_rank = session.cache.get(
                    GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY
                )
                if requested_rank is not None:
                    refreshed_geometry = self._active_global_trust_geometry(session)
                    if refreshed_geometry is None or int(
                        refreshed_geometry.metadata.get(
                            "requested_rank", refreshed_geometry.rank
                        )
                    ) < int(requested_rank):
                        session.decision.record_transition(
                            "global_trust_rank_growth_partial",
                            "same-mask rank growth is awaiting another "
                            "stratum of the frozen response design",
                            requested_rank=int(requested_rank),
                            delivered_rank=(
                                None
                                if refreshed_geometry is None
                                else int(
                                    refreshed_geometry.metadata.get(
                                        "requested_rank",
                                        refreshed_geometry.rank,
                                    )
                                )
                            ),
                        )
                    else:
                        session.cache.pop(GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY, None)
                        session.decision.record_transition(
                            "global_trust_rank_growth_delivered",
                            "same-mask response geometry delivered the requested FD rank",
                            requested_rank=int(requested_rank),
                            delivered_rank=int(
                                refreshed_geometry.metadata.get(
                                    "requested_rank", refreshed_geometry.rank
                                )
                            ),
                            geometry_hash=refreshed_geometry.content_hash,
                        )
                session.decision.reset_geometry_transport(
                    measured_mask_hash=measured_mask_hash
                )
                # A newly measured local geometry supersedes any exact
                # transition receipt calibrated at the preceding mask state.
                session.cache.pop(RESPONSE_TRANSITION_CACHE_KEY, None)
                transition = "measurement_update"
                reason = "active response state was incomplete or stale"
            else:
                self._consume_auxiliary(session, request, outcome)
                transition = "measurement_update"
                reason = (
                    "auxiliary measurement reduced active response-state uncertainty"
                )
                requested_rank = session.cache.get(
                    GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY
                )
                if requested_rank is not None:
                    refreshed_geometry = self._active_global_trust_geometry(session)
                    if refreshed_geometry is None or int(
                        refreshed_geometry.metadata.get(
                            "requested_rank", refreshed_geometry.rank
                        )
                    ) < int(requested_rank):
                        session.decision.record_transition(
                            "global_trust_rank_growth_partial",
                            "stratified rank growth is awaiting another "
                            "stratum of the frozen response design",
                            requested_rank=int(requested_rank),
                            delivered_rank=(
                                None
                                if refreshed_geometry is None
                                else int(
                                    refreshed_geometry.metadata.get(
                                        "requested_rank",
                                        refreshed_geometry.rank,
                                    )
                                )
                            ),
                        )
                    else:
                        session.cache.pop(GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY, None)
                        session.decision.record_transition(
                            "global_trust_rank_growth_delivered",
                            "stratified same-mask acquisition delivered the requested FD rank",
                            requested_rank=int(requested_rank),
                            delivered_rank=int(
                                refreshed_geometry.metadata.get(
                                    "requested_rank", refreshed_geometry.rank
                                )
                            ),
                            geometry_hash=refreshed_geometry.content_hash,
                        )
            session.decision.record_transition(
                transition,
                reason,
                request_id=request.request_id,
                outcome_hash=outcome.integrity_hash,
                kind=request.kind,
            )
        raise RuntimeError(
            "version state did not become ready within maximum_measurement_refreshes"
        )

    def _representatives(
        self,
        session: PruningSessionState,
        candidates: Sequence[ActionCandidate],
    ) -> tuple[BoundaryPoint, ...]:
        by_cost: dict[float, BoundaryPoint] = {}
        for candidate in candidates:
            prediction = session.frontier.entry(candidate.candidate_id).prediction
            if prediction is None:
                raise RuntimeError("candidate is missing its version prediction")
            scale = (
                float(candidate.removed_count)
                if self.boundary_metric_mode == "paired_bce_per_removed_route"
                else 1.0
            )
            if scale <= 0.0:
                raise RuntimeError("boundary candidates must remove at least one route")
            released = candidate.released_cost.coordinate(
                session.identity.cost.primary_coordinate
            )
            point = BoundaryPoint(
                candidate=candidate,
                released_cost=released,
                predicted_quality_loss=(
                    (
                        float(self.baseline_risk_certificate.estimate)
                        + prediction.predicted_quality_loss
                    )
                    if self.boundary_metric_mode
                    == "candidate_bce_relative_to_dense_baseline"
                    and self.baseline_risk_certificate is not None
                    else prediction.predicted_quality_loss / scale
                ),
                predicted_standard_error=prediction.standard_error / scale,
                expected_oracle_cost=1.0,
            )
            existing = by_cost.get(released)
            if existing is None or (
                point.predicted_quality_loss + 2.0 * point.predicted_standard_error,
                point.candidate.candidate_id,
            ) < (
                existing.predicted_quality_loss
                + 2.0 * existing.predicted_standard_error,
                existing.candidate.candidate_id,
            ):
                by_cost[released] = point
        return tuple(sorted(by_cost.values(), key=lambda item: item.released_cost))

    def _oracle_request(
        self,
        *,
        session: PruningSessionState,
        candidate_ids: Sequence[str],
        look: int,
        requested_context_count: int,
    ) -> MeasurementRequest:
        identifiers = tuple(str(item) for item in candidate_ids)
        if not identifiers:
            raise ValueError("oracle request requires at least one candidate")
        requested_count = int(requested_context_count)
        if requested_count <= 0:
            raise ValueError("oracle request requires a positive new-context packet")
        entries = tuple(
            session.frontier.entry(candidate_id) for candidate_id in identifiers
        )
        excluded = tuple(
            sorted(
                {
                    context_id
                    for entry in entries
                    for receipt in entry.oracle_receipts
                    for context_id in receipt.context_ids
                }
            )
        )
        candidate_only = (
            self.boundary_metric_mode == "candidate_bce_relative_to_dense_baseline"
        )
        selector = ContextSelector(
            selector_id=(
                "irg_candidate_absolute_bce_active_boundary_selector_v1"
                if candidate_only
                else "irg_paired_oracle_active_boundary_selector_v1"
            ),
            requested_count=requested_count,
            random_seed=session.random_seed + 104729 * (session.iteration + 1) + look,
            parameters={
                "exclude_context_ids": excluded,
                "common_random_numbers": not candidate_only,
                "frontier_boundary_only": True,
                "sample_size_policy_id": (
                    (
                        "candidate_absolute_bce_boundary_variance_gap_allocation_v1"
                        if candidate_only
                        else "paired_bce_boundary_variance_gap_allocation_v1"
                    )
                ),
            },
        )
        provenance = MeasurementProvenance(
            producer_id="mechanistic_systems.circuits.irg.runner",
            producer_version="1",
            code_hash=content_hash(
                {"runner": "CommonIRGRunner", "version": self.version.name}
            ),
        )
        return MeasurementRequest.create(
            kind="hard_mask_oracle",
            state_hash=session.state_hash,
            version=self.version.name,
            context_selector=selector,
            candidate_ids=identifiers,
            expected_cost=CostEstimate(
                wall_seconds=self.oracle_expected_cost.wall_seconds,
                device_seconds=self.oracle_expected_cost.device_seconds,
                forward_passes=max(1.0, self.oracle_expected_cost.forward_passes),
                backward_passes=self.oracle_expected_cost.backward_passes,
                calibration_fits=self.oracle_expected_cost.calibration_fits,
                transferred_bytes=self.oracle_expected_cost.transferred_bytes,
                peak_memory_bytes=self.oracle_expected_cost.peak_memory_bytes,
                uncertainty_fraction=self.oracle_expected_cost.uncertainty_fraction,
            ),
            expected_decision_value=float(len(identifiers)),
            reuse_keys=(
                *(entry.candidate.resulting_mask.mask_hash for entry in entries),
                session.identity.context.context_axis_hash,
                session.identity.metric.metric_id,
                session.identity.calibration.policy_id,
            ),
            provenance=provenance,
            payload={
                "candidate_mask_hashes": tuple(
                    entry.candidate.resulting_mask.mask_hash for entry in entries
                ),
                "parent_mask_hash": (
                    None if candidate_only else entries[0].candidate.parent_mask_hash
                ),
                "candidate_parent_mask_hash": (entries[0].candidate.parent_mask_hash),
                "paired_against_parent": not candidate_only,
                "oracle_reference_mode": (
                    "candidate_absolute_risk"
                    if candidate_only
                    else "paired_parent_candidate"
                ),
                "baseline_risk_certificate_hash": (
                    None
                    if self.baseline_risk_certificate is None
                    else self.baseline_risk_certificate.content_hash
                ),
                "sample_size_policy": dict(self.oracle_precision_policy.receipt),
                "requested_new_contexts": requested_count,
                "maximum_cumulative_contexts": (
                    int(self.oracle_precision_policy.maximum_contexts_per_candidate)
                    if (
                        self.oracle_precision_policy.maximum_contexts_per_candidate
                        is not None
                    )
                    else None
                ),
                "runtime_deadline_monotonic": (
                    None
                    if self.oracle_precision_policy.precision_only_iid
                    else session.cache.get("runtime_deadline_monotonic")
                ),
            },
        )

    def _next_oracle_context_count(
        self,
        *,
        search: AdaptiveBoundarySearch,
        candidate_ids: Sequence[str],
        threshold: float,
    ) -> int:
        if self.boundary_metric_mode not in {
            "paired_bce_total_increment",
            "paired_bce_per_removed_route",
            "candidate_bce_relative_to_dense_baseline",
        }:
            return int(self.oracle_context_count)
        observations = {
            candidate_id: tuple(search.observations[candidate_id].values)
            for candidate_id in candidate_ids
            if candidate_id in search.observations
            and search.observations[candidate_id].values
        }
        aggregate_estimates = {
            candidate_id: (
                search.observations[candidate_id].count,
                (
                    float(search.joint_decision_interval(candidate_id)[0])
                    if search.joint_decision_interval(candidate_id) is not None
                    else search.observations[candidate_id].mean
                ),
                (
                    math.sqrt(
                        search.observations[candidate_id].standard_error ** 2
                        + (
                            (1.0 + self.maximum_relative_bce_increase)
                            * float(self.baseline_risk_certificate.standard_error)
                        )
                        ** 2
                    )
                    if (
                        self.boundary_metric_mode
                        == "candidate_bce_relative_to_dense_baseline"
                        and self.baseline_risk_certificate is not None
                    )
                    else search.observations[candidate_id].standard_error
                ),
            )
            for candidate_id in candidate_ids
            if candidate_id in search.observations
        }
        if aggregate_estimates:
            return int(
                self.oracle_precision_policy.next_packet_from_estimates(
                    aggregate_estimates,
                    candidate_ids=candidate_ids,
                    threshold=float(threshold),
                )
            )
        return int(
            self.oracle_precision_policy.next_packet(
                observations,
                candidate_ids=candidate_ids,
                threshold=float(threshold),
            )
        )

    def _install_v1_joint_decision_interval(
        self,
        *,
        session: PruningSessionState,
        search: AdaptiveBoundarySearch,
        candidate_id: str,
        advance_look: bool,
    ) -> None:
        """Project all v1 uncertainty into its sole accept/reject contrast."""

        if (
            self.boundary_metric_mode != "candidate_bce_relative_to_dense_baseline"
            or self.baseline_risk_certificate is None
            or self.sequential_bce_spec is None
        ):
            return
        candidate = str(candidate_id)
        observation = search.observations.get(candidate)
        if observation is None:
            return
        if observation.moments_by_stratum:
            if any(
                moment.count < 2 for moment in observation.moments_by_stratum.values()
            ):
                return
            summary = stratified_student_t_summary_from_moments(
                observation.moments_by_stratum,
                stratum_weights=observation.stratum_weights,
                # Mean, variance, and Satterthwaite dof do not depend on this
                # placeholder alpha. The complete decision alpha is applied
                # once below by the joint envelope.
                alpha=0.5,
            )
            candidate_estimate = float(summary.mean)
            candidate_standard_error = float(summary.standard_error)
        else:
            values = np.asarray(observation.values, dtype=np.float64)
            if values.size < 2:
                return
            candidate_estimate = float(values.mean())
            candidate_standard_error = float(
                values.std(ddof=1) / math.sqrt(values.size)
            )
        profile_payload = session.cache.get(
            f"candidate_calibration_profile:{candidate}",
            {},
        )
        candidate_profile = (
            profile_payload
            if isinstance(profile_payload, CalibrationImprovementProfile)
            else CalibrationImprovementProfile(
                first_order_coefficient=float(
                    dict(profile_payload).get("first_order_coefficient", 0.0)
                ),
                second_order_coefficient=float(
                    dict(profile_payload).get("second_order_coefficient", 0.0)
                ),
                parameter_dimension=int(
                    dict(profile_payload).get("parameter_dimension", 2)
                ),
            )
        )
        look_key = f"joint_decision_look_index:{candidate}"
        prior_index = int(session.cache.get(look_key, -1))
        look_index = prior_index + 1 if advance_look else prior_index
        if look_index < 0:
            return
        session.cache[look_key] = look_index
        alpha = self.sequential_bce_spec.look_alpha(look_index)
        dense = self.baseline_risk_certificate
        envelope = joint_v1_decision_interval(
            candidate_estimate=candidate_estimate,
            candidate_standard_error=candidate_standard_error,
            dense_estimate=float(dense.estimate),
            dense_standard_error=float(dense.standard_error),
            dense_multiplier=1.0 + self.maximum_relative_bce_increase,
            candidate_calibration=candidate_profile,
            dense_calibration=dense.calibration_profile,
            alpha=alpha,
            look_index=look_index,
        )
        receipt = {
            **asdict(envelope),
            "content_hash": envelope.content_hash,
            "confidence_bearing_object": (
                "candidate_optimized_risk_minus_allowed_dense_optimized_risk"
            ),
            "component_error_stakes": {},
        }
        search.set_joint_decision_interval(
            candidate,
            estimate=envelope.estimate,
            lower=envelope.lower,
            upper=envelope.upper,
            receipt=receipt,
        )
        session.cache[f"joint_decision_envelope:{candidate}"] = receipt

    def _consume_oracle_outcome(
        self,
        *,
        session: PruningSessionState,
        search: AdaptiveBoundarySearch,
        request: MeasurementRequest,
        outcome,
        reasons: Mapping[str, str],
        resample: bool = False,
    ) -> tuple[str, ...]:
        values = np.asarray(outcome.values, dtype=np.float64)
        summary_only = bool(outcome.metadata.get("summary_only", False))
        if summary_only:
            if values.size:
                raise ValueError("summary-only oracle outcome contains raw rows")
            moments_raw = outcome.metadata.get("oracle_stratified_moments")
            if not isinstance(moments_raw, Mapping):
                raise ValueError("summary-only oracle outcome lacks moments")
            moments_by_candidate = {
                str(candidate_id): {
                    str(stratum): RunningMoment.from_mapping(moment)
                    for stratum, moment in dict(strata).items()
                }
                for candidate_id, strata in dict(moments_raw).items()
            }
            if set(moments_by_candidate) != set(request.candidate_ids):
                raise ValueError(
                    "summary-only oracle moments changed the candidate axis"
                )
        else:
            if values.ndim == 1:
                values = values[:, None]
            if values.shape != (
                len(outcome.context_ids),
                len(request.candidate_ids),
            ):
                raise ValueError(
                    "packed oracle outcome has the wrong context/candidate shape"
                )
            moments_by_candidate = {}
        consumed: list[str] = []
        oracle_strata_raw = outcome.metadata.get("oracle_strata")
        oracle_strata = (
            None
            if oracle_strata_raw is None
            else tuple(str(value) for value in oracle_strata_raw)
        )
        oracle_weights_raw = outcome.metadata.get("oracle_stratum_weights")
        oracle_stratum_weights = (
            None
            if oracle_weights_raw is None
            else {
                str(key): float(value)
                for key, value in dict(oracle_weights_raw).items()
            }
        )
        calibration_improvement_bounds_raw = outcome.metadata.get(
            "candidate_calibration_improvement_bounds",
            (),
        )
        calibration_improvement_bounds = tuple(
            float(value) for value in calibration_improvement_bounds_raw
        )
        if calibration_improvement_bounds and len(
            calibration_improvement_bounds
        ) != len(request.candidate_ids):
            raise ValueError("candidate calibration bounds changed the candidate axis")
        calibration_profiles_raw = outcome.metadata.get(
            "candidate_calibration_improvement_profiles",
            (),
        )
        calibration_profiles = tuple(dict(value) for value in calibration_profiles_raw)
        if calibration_profiles and len(calibration_profiles) != len(
            request.candidate_ids
        ):
            raise ValueError(
                "candidate calibration profiles changed the candidate axis"
            )
        if (
            not summary_only
            and oracle_strata is not None
            and len(oracle_strata) != values.shape[0]
        ):
            raise ValueError("oracle stratum labels do not align with outcome rows")
        replace_candidate_observations = bool(
            outcome.metadata.get("replace_candidate_observations", False)
        )
        if replace_candidate_observations:
            for candidate_id in request.candidate_ids:
                search.observations.pop(candidate_id, None)
        for column, candidate_id in enumerate(request.candidate_ids):
            candidate_values = (
                np.empty(0, dtype=np.float64) if summary_only else values[:, column]
            )
            candidate_moments = moments_by_candidate.get(candidate_id, {})
            receipt = OracleReceipt.create(
                candidate_id=candidate_id,
                state_hash=session.state_hash,
                context_ids=outcome.context_ids,
                paired_quality_deltas=candidate_values,
                stratified_moments={
                    key: value.to_dict() for key, value in candidate_moments.items()
                },
                calibration_state_id=outcome.calibration_state_id,
                realized_cost=outcome.realized_cost,
                measurement_request_id=request.request_id,
                metric_id=session.identity.metric.metric_id,
                metadata={
                    "boundary_reason": reasons.get(
                        candidate_id, "packed oracle evaluation"
                    ),
                    "resample": resample,
                    "outcome_integrity_hash": outcome.integrity_hash,
                    "packed_candidate_count": len(request.candidate_ids),
                    "boundary_metric_mode": self.boundary_metric_mode,
                    "oracle_value_semantics": (
                        "absolute_candidate_calibrated_bce"
                        if self.boundary_metric_mode
                        == "candidate_bce_relative_to_dense_baseline"
                        else "paired_candidate_minus_parent_calibrated_bce"
                    ),
                    "baseline_risk_certificate_hash": (
                        None
                        if self.baseline_risk_certificate is None
                        else self.baseline_risk_certificate.content_hash
                    ),
                    "oracle_allocation_receipt": outcome.metadata.get(
                        "oracle_allocation_receipt"
                    ),
                    "oracle_stratum_weights": oracle_stratum_weights or {},
                    "summary_only": summary_only,
                    "calibration_improvement_bound": (
                        None
                        if not calibration_improvement_bounds
                        else calibration_improvement_bounds[column]
                    ),
                    "calibration_improvement_profile": (
                        {} if not calibration_profiles else calibration_profiles[column]
                    ),
                    "joint_precision_allocation_receipts": tuple(
                        outcome.metadata.get(
                            "joint_precision_allocation_receipts",
                            (),
                        )
                    ),
                },
            )
            session.frontier.record_oracle(receipt)
            if self.boundary_metric_mode == "candidate_bce_relative_to_dense_baseline":
                session.cache[f"candidate_calibration_state:{candidate_id}"] = str(
                    outcome.calibration_state_id
                )
            candidate = session.frontier.entry(candidate_id).candidate
            scale = (
                float(candidate.removed_count)
                if self.boundary_metric_mode == "paired_bce_per_removed_route"
                else 1.0
            )
            if summary_only:
                if oracle_stratum_weights is None:
                    raise ValueError(
                        "summary-only oracle outcome lacks stratum weights"
                    )
                search.observe_moments(
                    candidate_id,
                    {
                        key: value.scaled(1.0 / scale)
                        for key, value in candidate_moments.items()
                    },
                    stratum_weights=oracle_stratum_weights,
                )
            else:
                boundary_values = candidate_values / scale
                search.observe(
                    candidate_id,
                    boundary_values,
                    strata=oracle_strata,
                    stratum_weights=oracle_stratum_weights,
                )
            if calibration_improvement_bounds:
                bound = calibration_improvement_bounds[column] / scale
                search.set_calibration_improvement_bound(
                    candidate_id,
                    bound,
                )
                session.cache[
                    f"candidate_calibration_improvement_bound:{candidate_id}"
                ] = float(bound)
            if calibration_profiles:
                session.cache[f"candidate_calibration_profile:{candidate_id}"] = (
                    CalibrationImprovementProfile(
                        first_order_coefficient=float(
                            calibration_profiles[column].get(
                                "first_order_coefficient",
                                0.0,
                            )
                        )
                        / scale,
                        second_order_coefficient=float(
                            calibration_profiles[column].get(
                                "second_order_coefficient",
                                0.0,
                            )
                        )
                        / scale,
                        parameter_dimension=int(
                            calibration_profiles[column].get(
                                "parameter_dimension",
                                2,
                            )
                        ),
                    )
                )
            self._install_v1_joint_decision_interval(
                session=session,
                search=search,
                candidate_id=candidate_id,
                advance_look=True,
            )
            consumed.append(candidate_id)
            prediction = session.frontier.entry(candidate_id).prediction
            assert prediction is not None
            self.residual_model.add(
                ResidualObservation(
                    candidate_id=candidate_id,
                    state_hash=session.state_hash,
                    predicted_quality_loss=prediction.predicted_quality_loss,
                    actual_quality_loss=float(
                        search.observations[candidate_id].mean * scale
                    ),
                    features={
                        "released_cost": session.frontier.entry(
                            candidate_id
                        ).candidate.released_cost.coordinate(
                            session.identity.cost.primary_coordinate
                        ),
                        "route_count": float(
                            session.frontier.entry(candidate_id).candidate.removed_count
                        ),
                    },
                )
            )
            session.decision.record_transition(
                "oracle_update",
                reasons.get(candidate_id, "packed oracle evaluation"),
                candidate_id=candidate_id,
                request_id=request.request_id,
                mean_quality_loss=float(search.observations[candidate_id].mean * scale),
                mean_boundary_sensitivity=float(search.observations[candidate_id].mean),
                boundary_metric_mode=self.boundary_metric_mode,
                packed_candidate_count=len(request.candidate_ids),
            )
        diagnostics = self.residual_model.diagnostics()
        residual_scale = float(diagnostics["residual_scale"])
        session.decision.model_residual_scale = (
            residual_scale if math.isfinite(residual_scale) else 1.0
        )
        return tuple(consumed)

    def _maybe_refine_dense_baseline(
        self,
        *,
        session: PruningSessionState,
        search: AdaptiveBoundarySearch,
        candidate_ids: Sequence[str],
    ) -> bool:
        """Refine the dense certificate whenever it dominates resolution cost."""

        if (
            self.boundary_metric_mode != "candidate_bce_relative_to_dense_baseline"
            or self.baseline_risk_certificate is None
        ):
            return False
        uncertain = tuple(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id in search.observations
            and self._index_is_uncertain(search, candidate_id)
        )
        if not uncertain:
            return False
        baseline = self.baseline_risk_certificate
        multiplier = 1.0 + self.maximum_relative_bce_increase
        baseline_radius = (
            0.5 * multiplier * (float(baseline.upper) - float(baseline.lower))
        )
        candidate_radius = max(
            (
                0.5
                * (search._bounds(candidate_id)[1] - search._bounds(candidate_id)[0])
                for candidate_id in uncertain
            ),
            default=math.inf,
        )
        if not math.isfinite(candidate_radius) or baseline_radius <= candidate_radius:
            return False
        provider = self.providers.provider("hard_mask_oracle")
        if callable(getattr(provider, "plan_joint_precision_action", None)):
            # The q-free allocator compares dense-reference sampling directly
            # with candidate sampling and both calibration roles on the next
            # loop.  Do not run the superseded radius heuristic as a second,
            # competing acquisition policy.
            return False
        certify = getattr(provider, "certify_baseline_risk", None)
        known_mask = getattr(provider, "known_mask", None)
        if not callable(certify) or not callable(known_mask):
            return False
        target = max(
            int(baseline.context_count)
            + int(self.oracle_precision_policy.minimum_packet),
            4 * int(baseline.context_count),
        )
        if self.oracle_precision_policy.maximum_contexts_per_candidate is not None:
            target = min(
                int(self.oracle_precision_policy.maximum_contexts_per_candidate),
                target,
            )
        if target <= int(baseline.context_count):
            return False
        if not session.compute_ledger.can_afford(self.oracle_expected_cost):
            return False
        before_usage = dict(getattr(provider, "resource_usage", {}) or {})
        started = __import__("time").perf_counter()
        refined = certify(
            known_mask(baseline.mask_hash),
            cumulative_context_count=target,
            confidence_level=(
                self.sequential_bce_spec.reference_confidence_level
                if self.sequential_bce_spec is not None
                else self.bce_precision_confidence_level
            ),
            maximum_planned_looks=(
                None if self.oracle_precision_policy.precision_only_iid else 2
            ),
            precision_only_iid=bool(self.oracle_precision_policy.precision_only_iid),
        )
        after_usage = dict(getattr(provider, "resource_usage", {}) or {})
        session.compute_ledger.record(
            RealizedCost(
                wall_seconds=__import__("time").perf_counter() - started,
                forward_passes=max(
                    0,
                    int(after_usage.get("forward_passes", 0))
                    - int(before_usage.get("forward_passes", 0)),
                ),
                calibration_fits=max(
                    0,
                    int(after_usage.get("calibration_fit_count", 0))
                    - int(before_usage.get("calibration_fit_count", 0)),
                ),
                packing_degree=1,
            )
        )
        if not isinstance(refined, BaselineRiskCertificate):
            raise TypeError("dense baseline refinement returned an invalid certificate")
        if (
            refined.mask_hash != baseline.mask_hash
            or refined.context_count <= baseline.context_count
        ):
            raise RuntimeError(
                "dense baseline refinement did not extend the frozen reference"
            )
        self.baseline_risk_certificate = refined
        session.cache["dense_baseline_refinement_count"] = (
            int(session.cache.get("dense_baseline_refinement_count", 0)) + 1
        )
        for observed_candidate_id in tuple(search.observations):
            self._install_v1_joint_decision_interval(
                session=session,
                search=search,
                candidate_id=observed_candidate_id,
                advance_look=True,
            )
        session.decision.record_transition(
            "dense_baseline_certificate_refined",
            "independent dense-reference uncertainty dominated the active "
            "candidate decision and was refined by its next predictable prefix",
            prior_certificate_hash=baseline.content_hash,
            refined_certificate_hash=refined.content_hash,
            prior_context_count=int(baseline.context_count),
            refined_context_count=int(refined.context_count),
        )
        return True

    @staticmethod
    def _index_is_uncertain(
        search: AdaptiveBoundarySearch,
        candidate_id: str,
    ) -> bool:
        index = search._index[str(candidate_id)]
        return index in set(search._uncertain_observed())

    def _candidate_only_decision_receipts(
        self,
        *,
        session: PruningSessionState,
        search: AdaptiveBoundarySearch,
    ) -> tuple[Mapping[str, object], ...]:
        if (
            self.boundary_metric_mode != "candidate_bce_relative_to_dense_baseline"
            or self.baseline_risk_certificate is None
            or self.sequential_bce_spec is None
        ):
            return ()
        output: list[Mapping[str, object]] = []
        for candidate_id in sorted(search.observations):
            observation = search.observations[candidate_id]
            lower, upper = search._bounds(candidate_id)
            joint_interval = search.joint_decision_interval(candidate_id)
            aggregate_estimate = (
                float(observation.mean)
                if joint_interval is None
                else float(joint_interval[0])
            )
            action = (
                "accept"
                if upper <= search.accept_quality_loss
                else ("reject" if lower > search.reject_quality_loss else "unresolved")
            )
            raw_stats: dict[str, Mapping[str, float | int]] = {}
            for stratum, moment in sorted(observation.moments_by_stratum.items()):
                raw_stats[stratum] = {
                    "count": moment.count,
                    "mean": moment.mean,
                    "variance": (
                        moment.sample_variance if moment.count > 1 else math.inf
                    ),
                }
            entry = session.frontier.entry(candidate_id)
            calibration_improvement_bound = float(
                session.cache.get(
                    f"candidate_calibration_improvement_bound:{candidate_id}",
                    0.0,
                )
            )
            allocation_receipts = tuple(
                dict(receipt.metadata["oracle_allocation_receipt"])
                for receipt in entry.oracle_receipts
                if isinstance(
                    receipt.metadata.get("oracle_allocation_receipt"),
                    Mapping,
                )
            )
            calibration_state_ids = tuple(
                dict.fromkeys(
                    str(receipt.calibration_state_id)
                    for receipt in entry.oracle_receipts
                )
            )
            if not calibration_state_ids:
                continue
            receipt = SequentialBCEDecisionReceipt(
                spec_hash=self.sequential_bce_spec.content_hash,
                proposal_id=str(candidate_id),
                estimand_policy=BASELINE_EXCESS_CALIBRATED_BCE_V1,
                aggregate_estimate=aggregate_estimate,
                lower=float(lower),
                upper=float(upper),
                accept_boundary=float(search.accept_quality_loss),
                reject_boundary=float(search.reject_quality_loss),
                action=action,
                raw_stratum_statistics=raw_stats,
                allocation_receipts=allocation_receipts,
                calibration_state_ids=calibration_state_ids,
                resource_usage=session.compute_ledger.summary(),
                baseline_risk_certificate_hash=(
                    self.baseline_risk_certificate.content_hash
                ),
                confidence_scope=self.sequential_bce_spec.confidence_scope,
                optimized_calibration_envelope={
                    "schema": "joint_v1_decision_envelope_v1",
                    "confidence_bearing_object": (
                        "candidate_optimized_risk_minus_allowed_dense_" "optimized_risk"
                    ),
                    "component_error_stakes": {},
                    "joint_decision_interval": (
                        {} if joint_interval is None else dict(joint_interval[3])
                    ),
                    "legacy_candidate_calibration_bound_diagnostic": (
                        calibration_improvement_bound
                    ),
                },
                joint_precision_allocation_receipts=tuple(
                    [
                        *(
                            dict(item)
                            for item in session.cache.get(
                                "joint_precision_allocation_receipts",
                                (),
                            )
                        ),
                        *(
                            dict(item)
                            for oracle_receipt in entry.oracle_receipts
                            for item in oracle_receipt.metadata.get(
                                "joint_precision_allocation_receipts",
                                (),
                            )
                        ),
                    ]
                ),
            )
            output.append(receipt.receipt)
        return tuple(output)

    def _predictive_validity_receipt(
        self,
        *,
        session: PruningSessionState,
        selected_candidate_id: str,
        parent_state_hash: str,
        parent_mask_hash: str,
    ) -> PredictiveValidityReceipt:
        """Bind exact action checks to one parent chart and comparability block."""

        selected_entry = session.frontier.entry(selected_candidate_id)
        if (
            selected_entry.candidate.parent_state_hash != parent_state_hash
            or selected_entry.candidate.parent_mask_hash != parent_mask_hash
        ):
            raise RuntimeError(
                "selected candidate does not belong to the predictive-validity parent"
            )
        partitioning = session.identity.geometric_partitioning
        if partitioning is None:
            partition_id = "unpartitioned_compatibility_block"

            def belongs(_candidate: ActionCandidate) -> bool:
                return True

        else:
            if session.current_mask.mask_hash != parent_mask_hash:
                raise RuntimeError(
                    "predictive-validity partition resolution requires the parent mask"
                )
            selected_partition = candidate_partition(
                selected_entry.candidate,
                parent_mask=session.current_mask,
                partitioning=partitioning,
            )
            partition_id = selected_partition.partition_id

            def belongs(candidate: ActionCandidate) -> bool:
                return (
                    str(
                        candidate.generator_parameters.get("geometric_partition_id", "")
                    )
                    == partition_id
                )

        candidates: list[str] = []
        predicted: list[float] = []
        actual: list[float] = []
        standard_errors: list[float] = []
        receipt_hashes: list[str] = []
        for candidate_id in session.frontier.candidate_ids:
            entry = session.frontier.entry(candidate_id)
            if (
                entry.candidate.parent_state_hash != parent_state_hash
                or entry.candidate.parent_mask_hash != parent_mask_hash
                or entry.prediction is None
                or not entry.oracle_receipts
                or not belongs(entry.candidate)
            ):
                continue
            mean = entry.mean_quality_loss
            standard_error = entry.standard_error
            if mean is None or standard_error is None:
                continue
            candidates.append(candidate_id)
            predicted.append(float(entry.prediction.predicted_quality_loss))
            actual.append(float(mean))
            standard_errors.append(float(standard_error))
            receipt_hashes.append(
                content_hash(
                    tuple(receipt.receipt_hash for receipt in entry.oracle_receipts),
                    prefix="irg-oracle-evidence-set-",
                )
            )
        if selected_candidate_id not in candidates:
            raise RuntimeError(
                "selected candidate lacks exact oracle evidence for model validity"
            )
        decision = self.model_validity_controller.assess(
            predicted,
            actual,
            actual_standard_errors=standard_errors,
        )
        receipt = PredictiveValidityReceipt(
            parent_state_hash=parent_state_hash,
            parent_mask_hash=parent_mask_hash,
            partition_id=partition_id,
            selected_candidate_id=selected_candidate_id,
            candidate_ids=tuple(candidates),
            predicted_changes=tuple(predicted),
            actual_changes=tuple(actual),
            actual_standard_errors=tuple(standard_errors),
            oracle_receipt_hashes=tuple(receipt_hashes),
            decision=decision,
        )
        session.cache[f"irg_predictive_validity:{receipt.content_hash}"] = receipt
        session.decision.predictive_validity_receipt_hashes.append(receipt.content_hash)
        conservative_scale = float(
            decision.diagnostics.get(
                "conservative_remainder_rms",
                decision.remainder_rms if decision.remainder_rms is not None else 1.0,
            )
        )
        session.decision.model_residual_scale = max(0.0, conservative_scale)
        session.decision.record_transition(
            "predictive_model_validity",
            decision.reason,
            predictive_validity_receipt_hash=receipt.content_hash,
            parent_state_hash=parent_state_hash,
            parent_mask_hash=parent_mask_hash,
            partition_id=partition_id,
            selected_candidate_id=selected_candidate_id,
            candidate_ids=receipt.candidate_ids,
            validity_action=decision.action,
            recommended_scale=decision.recommended_scale,
        )
        return receipt

    def _response_transition_request(
        self,
        *,
        session: PruningSessionState,
        candidate_id: str,
        look: int,
    ) -> MeasurementRequest:
        entry = session.frontier.entry(candidate_id)
        removed = tuple(
            int(value)
            for value in entry.candidate.resolve_removed_indices(session.current_mask)
        )
        if not removed:
            raise ValueError(
                "response-transition request requires a nonempty physical action"
            )
        constraint = session.identity.response_constraint
        if constraint is None:
            raise ValueError(
                "typed response transition requires a frozen response-safe constraint"
            )
        normalization_directions = tuple(
            tuple(float(value) for value in item.vector)
            for item in constraint.directions
        )
        normalization_tolerances = tuple(
            float(item.lower_tolerance) for item in constraint.directions
        )
        selector = ContextSelector(
            selector_id="irg_paired_response_transition_selector_v2",
            requested_count=self.transition_context_count,
            random_seed=session.random_seed
            + 130363 * (session.iteration + 1)
            + int(look),
            parameters={
                "common_random_numbers": True,
                "finite_hard_mask_transition": True,
                "selected_after_scalar_oracle_boundary_search": True,
            },
        )
        return MeasurementRequest.create(
            kind="response_transition",
            state_hash=session.state_hash,
            version=self.version.name,
            context_selector=selector,
            candidate_ids=(candidate_id,),
            expected_cost=self.transition_expected_cost,
            expected_decision_value=2.0,
            reuse_keys=(
                entry.candidate.resulting_mask.mask_hash,
                session.identity.context.context_axis_hash,
                session.identity.metric.metric_id,
                session.identity.calibration.policy_id,
            ),
            provenance=MeasurementProvenance(
                producer_id="mechanistic_systems.circuits.irg.runner.response_transition",
                producer_version="1",
                code_hash=content_hash(
                    {"runner": "CommonIRGRunner", "transition": "matrix_response_v1"}
                ),
            ),
            payload={
                "action_indices": removed,
                "coefficients": tuple(-1.0 for _ in removed),
                "candidate_mask_hash": entry.candidate.resulting_mask.mask_hash,
                "parent_mask_hash": entry.candidate.parent_mask_hash,
                # Matrix-response validity is normalized in the frozen
                # response-safe region, not in scalar oracle-loss units.
                "response_scale": 1.0,
                "normalization_directions": normalization_directions,
                "normalization_tolerances": normalization_tolerances,
                "normalization_energy_metric": constraint.energy_metric,
                "normalization_energy_tolerance": constraint.energy_tolerance,
                "response_constraint_hash": constraint.content_hash,
            },
        )

    def _acquire_selected_response_transition(
        self,
        *,
        session: PruningSessionState,
        candidate_id: str,
        look: int,
    ) -> tuple[ResponseTransitionReceipt, MeasurementRequest, object] | None:
        """Measure, but do not yet apply, a selected-mask response transition.

        The scalar oracle determines frontier feasibility.  This additional
        paired measurement is guidance-only.  Acquisition occurs while the
        physical parent mask is still current so providers see the exact
        parent-to-candidate action.  Applying it is deferred until after the
        authorized hard-mask commit, preventing a v4 response-state update
        from invalidating the candidate's predictable parent identity.
        """

        typed_response = (
            session.identity.response_spec is not None
            and session.identity.response_constraint is not None
            and session.identity.response_spec.output_dimension > 1
        )
        if not typed_response or not self.providers.supports("response_transition"):
            return None
        request = self._response_transition_request(
            session=session,
            candidate_id=candidate_id,
            look=look,
        )
        selected = self._select_request_batch(
            session,
            (request,),
            phase="model_validity",
            maximum_requests=1,
        )
        if not selected:
            return None
        request = selected[0]
        outcome = self._execute_and_record(session, request)
        receipt = response_transition_receipt_from_outcome(
            outcome,
            expected_response_dimension=session.identity.response_spec.output_dimension,
            receipt_state_hash=session.state_hash,
        )
        return receipt, request, outcome

    def _apply_committed_response_transition(
        self,
        *,
        session: PruningSessionState,
        candidate_id: str,
        receipt: ResponseTransitionReceipt,
        request: MeasurementRequest,
        outcome: object,
    ) -> None:
        """Apply guidance geometry only after the exact mask has committed."""

        if receipt.candidate_mask_hash != session.current_mask.mask_hash:
            raise RuntimeError(
                "response transition does not end at the committed hard mask"
            )
        session.cache[RESPONSE_TRANSITION_CACHE_KEY] = receipt
        session.decision.model_residual_scale = max(
            receipt.validity.normalized_mean_error,
            receipt.validity.normalized_second_moment_error,
        )
        if bool(
            getattr(
                self.version,
                "updates_response_state_after_exact_transition",
                False,
            )
        ):
            # A response implementation may explicitly own a conservative
            # residual-envelope update from the common exact-transition
            # receipt. Version ordinals do not confer this capability.
            session.version_state = self.version.update_response_state(
                session.version_state, outcome
            )
        session.decision.record_transition(
            "response_transition_update",
            f"selected {self.version.name} mask calibrated the shared matrix-valued response model",
            candidate_id=candidate_id,
            request_id=request.request_id,
            outcome_hash=outcome.integrity_hash,
            transition_receipt_hash=receipt.content_hash,
        )

    @staticmethod
    def _transport_geometry_after_commit(
        session: PruningSessionState,
        *,
        receipt: ResponseTransitionReceipt,
    ) -> None:
        """Advance one measured chart under an exact, accepted transition."""

        measured_mask_hash = str(
            getattr(session.version_state, "measured_mask_hash", "")
        )
        if not measured_mask_hash:
            raise RuntimeError("response state lacks a geometry-anchor mask identity")
        if receipt.candidate_mask_hash != session.current_mask.mask_hash:
            raise RuntimeError("transition receipt does not end at the committed mask")
        decision = session.decision
        if decision.geometry_anchor_mask_hash is None:
            decision.reset_geometry_transport(measured_mask_hash=measured_mask_hash)
        if decision.geometry_anchor_mask_hash != measured_mask_hash:
            raise RuntimeError(
                "transition receipt would mix two relinearization anchors"
            )
        indices = tuple(int(value) for value in receipt.action_indices)
        if not indices:
            raise RuntimeError("geometry transport requires a nonempty physical action")
        overlap = decision.cumulative_geometry_removed_indices.intersection(indices)
        if overlap:
            raise RuntimeError("geometry transport repeats an already removed route")
        decision.cumulative_geometry_removed_indices.update(indices)
        decision.geometry_current_mask_hash = session.current_mask.mask_hash
        decision.geometry_transition_receipt_hashes.append(receipt.content_hash)
        decision.record_transition(
            "geometry_transport",
            "exact response transition retained the partition-local chart without relinearization",
            geometry_anchor_mask_hash=decision.geometry_anchor_mask_hash,
            geometry_current_mask_hash=decision.geometry_current_mask_hash,
            cumulative_removed_indices=tuple(
                sorted(decision.cumulative_geometry_removed_indices)
            ),
            transition_receipt_hash=receipt.content_hash,
        )

    def run_epoch(
        self,
        session: PruningSessionState,
        *,
        budget_targets: Sequence[float] = (),
        commit: bool = True,
    ) -> EpochResult:
        if session.active_version != self.version.name:
            raise ValueError("session active version and runner version differ")
        if session.commit_authority_mode != self.commit_authority_mode:
            raise ValueError("session and runner commit-authority modes differ")
        if commit and (
            self.commit_authority_mode != "glocal_v3_shadow"
            or self.commit_authority is None
        ):
            raise ValueError(
                "commit=True requires a registered live exact-mask commit "
                "controller; exploratory evaluation must use commit=False"
            )
        blocked = self._consume_pending_model_validity(session)
        if blocked is not None:
            parent_state_hash = session.state_hash
            action = str(session.decision.pending_model_validity_action)
            session.decision.record_transition(
                "model_validity_block",
                blocked,
                pending_action=action,
                requested_capability_transition=(
                    session.decision.requested_capability_transition
                ),
            )
            return EpochResult(
                parent_state_hash=parent_state_hash,
                candidate_ids=(),
                evaluated_candidate_ids=(),
                committed_candidate_id=None,
                stop_reason=blocked,
                boundary_diagnostics={},
                model_validity_action=action,
                commit_authority_mode=self.commit_authority_mode,
            )
        self.ensure_minimum_response_state(session)
        parent_state_hash = session.state_hash
        parent_mask_hash = session.current_mask.mask_hash
        self._bind_residual_model_scope(session)
        candidates = list(
            self._apply_pending_action_scale(
                session,
                self._generate_candidates(session, budget_targets),
            )
        )
        if (
            candidates
            and self._has_required_covariance_frontier_obligation(session)
            and not self.providers.supports("covariance_rank_expansion")
        ):
            covariance_resolution = self._resolve_covariance_rank_obligation(session)
            if covariance_resolution not in {"fallback", "blocked"}:
                raise RuntimeError(
                    "required covariance frontier obligation had no provider "
                    "but did not fail closed"
                )
            return EpochResult(
                parent_state_hash=parent_state_hash,
                candidate_ids=(),
                evaluated_candidate_ids=(),
                committed_candidate_id=None,
                stop_reason=(
                    "covariance-rank expansion was unavailable; scientific "
                    "fallback was scheduled"
                    if covariance_resolution == "fallback"
                    else "required omitted-route frontier coverage is unresolved "
                    "and no covariance-rank provider is available"
                ),
                boundary_diagnostics={},
                model_validity_action=str(
                    session.decision.pending_model_validity_action
                ),
                commit_authority_mode=self.commit_authority_mode,
                candidate_pool_retired=covariance_resolution == "fallback",
            )
        if not candidates:
            trust_rank_resolution = self._resolve_global_trust_rank_obligation(session)
            if trust_rank_resolution is not None:
                return EpochResult(
                    parent_state_hash=parent_state_hash,
                    candidate_ids=(),
                    evaluated_candidate_ids=(),
                    committed_candidate_id=None,
                    stop_reason={
                        "scheduled": (
                            "decision-relevant global trust rank growth was "
                            "scheduled before candidates can be regenerated"
                        ),
                        "fallback": (
                            "global trust rank growth was unavailable; "
                            "scientific fallback was scheduled"
                        ),
                        "blocked": (
                            "global trust rank growth remains unresolved and "
                            "no legal fallback exists"
                        ),
                    }[trust_rank_resolution],
                    boundary_diagnostics={},
                    model_validity_action=str(
                        session.decision.pending_model_validity_action
                    ),
                    commit_authority_mode=self.commit_authority_mode,
                    candidate_pool_retired=True,
                )
            covariance_resolution = self._resolve_covariance_rank_obligation(session)
            if covariance_resolution is not None:
                return EpochResult(
                    parent_state_hash=parent_state_hash,
                    candidate_ids=(),
                    evaluated_candidate_ids=(),
                    committed_candidate_id=None,
                    stop_reason={
                        "expanded": (
                            "covariance-rank expansion changed the frozen "
                            "response state; candidates will be regenerated"
                        ),
                        "fallback": (
                            "covariance-rank expansion was unavailable; "
                            "scientific fallback was scheduled"
                        ),
                        "blocked": (
                            "covariance-rank expansion remains unresolved "
                            "under the hard resource/provider constraints"
                        ),
                    }[covariance_resolution],
                    boundary_diagnostics={},
                    model_validity_action=str(
                        session.decision.pending_model_validity_action
                    ),
                    commit_authority_mode=self.commit_authority_mode,
                    candidate_pool_retired=(
                        covariance_resolution in {"expanded", "fallback"}
                    ),
                )
            validity_action = str(session.decision.pending_model_validity_action)
            return EpochResult(
                parent_state_hash=parent_state_hash,
                candidate_ids=(),
                evaluated_candidate_ids=(),
                committed_candidate_id=None,
                stop_reason="active version generated no legal hard-mask action",
                boundary_diagnostics={},
                model_validity_action=validity_action,
                commit_authority_mode=self.commit_authority_mode,
            )
        for candidate in candidates:
            session.frontier.register(candidate)
            prediction = self.version.predict_action(candidate, session)
            session.frontier.record_prediction(prediction)
            session.decision.active_candidate_ids.add(candidate.candidate_id)
        points = self._representatives(session, candidates)
        bce_precision_mode = self.boundary_metric_mode in {
            "paired_bce_total_increment",
            "paired_bce_per_removed_route",
            "candidate_bce_relative_to_dense_baseline",
        }
        decision_rule = BCELossBoundaryDecisionRule(
            confidence_level=self.bce_precision_confidence_level,
            maximum_bce_increment=float(session.stopping.maximum_quality_loss),
        )
        if bce_precision_mode:
            decision_rule.validate()
        candidate_only_bce = (
            self.boundary_metric_mode == "candidate_bce_relative_to_dense_baseline"
        )
        accept_quality_loss = None
        reject_quality_loss = None
        if candidate_only_bce:
            assert self.baseline_risk_certificate is not None
            # Candidate and dense uncertainty are projected into the single
            # contrast R_candidate* - (1 + eta) R_dense*.  Both decisions are
            # therefore made against zero; the dense certificate is not a
            # separately confidence-inflated threshold.
            accept_quality_loss = 0.0
            reject_quality_loss = 0.0
        search = AdaptiveBoundarySearch(
            points,
            maximum_quality_loss=(
                (
                    0.5 * (float(accept_quality_loss) + float(reject_quality_loss))
                    if candidate_only_bce
                    and accept_quality_loss is not None
                    and reject_quality_loss is not None
                    else decision_rule.maximum_bce_increment
                )
                if bce_precision_mode
                else session.stopping.maximum_quality_loss
            ),
            practical_indifference=(
                0.0 if bce_precision_mode else session.stopping.practical_indifference
            ),
            cost_tolerance=session.stopping.cost_bracket_tolerance,
            z_value=self.boundary_z_value,
            resample_standard_error=self.boundary_resample_standard_error,
            confidence_level=(
                self.oracle_precision_policy.confidence_level
                if bce_precision_mode
                else None
            ),
            maximum_looks_per_candidate=(
                None
                if self.oracle_precision_policy.precision_only_iid
                else session.stopping.maximum_oracle_evaluations
            ),
            precision_only_iid=bool(self.oracle_precision_policy.precision_only_iid),
            resample_until_resolved=bce_precision_mode,
            accept_quality_loss=accept_quality_loss,
            reject_quality_loss=reject_quality_loss,
        )
        evaluated: list[str] = []
        stop_reason = "oracle evaluation limit reached"
        masks_evaluated = 0
        look = 0
        auxiliary_looks = 0
        candidate_pool_retired = False
        acquisition_prerequisite_blocked = False
        while (
            self.oracle_precision_policy.precision_only_iid
            or masks_evaluated < session.stopping.maximum_oracle_evaluations
        ):
            remaining = (
                self.oracle_pack_size
                if self.oracle_precision_policy.precision_only_iid
                else (session.stopping.maximum_oracle_evaluations - masks_evaluated)
            )
            proposals = search.propose_batch(min(self.oracle_pack_size, remaining))
            evaluable = tuple(
                proposal
                for proposal in proposals
                if proposal.action == "evaluate" and proposal.candidate_id is not None
            )
            if not evaluable:
                stop_reason = (
                    proposals[0].reason
                    if proposals
                    else "boundary search produced no experiment"
                )
                break
            candidate_ids = tuple(
                proposal.candidate_id
                for proposal in evaluable
                if proposal.candidate_id is not None
            )
            oracle_context_count = self._next_oracle_context_count(
                search=search,
                candidate_ids=candidate_ids,
                threshold=(
                    (
                        accept_quality_loss
                        if accept_quality_loss is not None
                        else decision_rule.maximum_bce_increment
                    )
                    if bce_precision_mode
                    else session.stopping.maximum_quality_loss
                ),
            )
            if oracle_context_count <= 0:
                stop_reason = (
                    "oracle context horizon exhausted while the BCE boundary "
                    "remained unresolved"
                )
                break
            if (
                candidate_only_bce
                and len(candidate_ids) == 1
                and candidate_ids[0] in search.observations
            ):
                provider = self.providers.provider("hard_mask_oracle")
                plan_joint = getattr(
                    provider,
                    "plan_joint_precision_action",
                    None,
                )
                execute_calibration = getattr(
                    provider,
                    "execute_joint_precision_calibration",
                    None,
                )
                known_mask = getattr(provider, "known_mask", None)
                if (
                    callable(plan_joint)
                    and callable(execute_calibration)
                    and callable(known_mask)
                ):
                    candidate_id = candidate_ids[0]
                    entry = session.frontier.entry(candidate_id)
                    observation = search.observations[candidate_id]
                    candidate_lower, candidate_upper = search._bounds(candidate_id)
                    candidate_calibration_bound = float(
                        session.cache.get(
                            "candidate_calibration_improvement_bound:"
                            f"{candidate_id}",
                            0.0,
                        )
                    )
                    oracle_interval_width = max(
                        0.0,
                        float(candidate_upper)
                        - float(candidate_lower)
                        - candidate_calibration_bound,
                    )
                    dense_mask = (
                        None
                        if self.baseline_risk_certificate is None
                        else known_mask(self.baseline_risk_certificate.mask_hash)
                    )
                    joint_receipt = session.cache.get(
                        f"joint_decision_envelope:{candidate_id}",
                        {},
                    )
                    forecast = plan_joint(
                        candidate_mask=entry.candidate.resulting_mask,
                        oracle_count=observation.count,
                        oracle_interval_width=oracle_interval_width,
                        requested_oracle_increment=oracle_context_count,
                        joint_radius_squared=(
                            None
                            if not isinstance(joint_receipt, Mapping)
                            else joint_receipt.get("radius_squared")
                        ),
                        dense_mask=dense_mask,
                        dense_reference_count=(
                            None
                            if self.baseline_risk_certificate is None
                            else self.baseline_risk_certificate.context_count
                        ),
                        dense_reference_interval_width=(
                            None
                            if self.baseline_risk_certificate is None
                            else max(
                                0.0,
                                float(self.baseline_risk_certificate.upper)
                                - float(self.baseline_risk_certificate.lower),
                            )
                        ),
                        requested_dense_reference_increment=(
                            None
                            if self.baseline_risk_certificate is None
                            else oracle_context_count
                        ),
                        dense_reference_multiplier=(
                            1.0 + self.maximum_relative_bce_increase
                        ),
                    )
                    if forecast.role != "oracle":
                        before_usage = dict(
                            getattr(provider, "resource_usage", {}) or {}
                        )
                        started = __import__("time").perf_counter()
                        if forecast.role == "dense_oracle":
                            assert dense_mask is not None
                            assert self.baseline_risk_certificate is not None
                            prior = self.baseline_risk_certificate
                            receipt = {
                                "schema": (
                                    "joint_calibration_oracle_allocation_receipt_v1"
                                ),
                                "selected_role": "dense_oracle",
                                "forecast": {
                                    **asdict(forecast),
                                    "content_hash": forecast.content_hash,
                                },
                            }
                            refined = provider.certify_baseline_risk(
                                dense_mask,
                                cumulative_context_count=forecast.target_count,
                                confidence_level=prior.confidence_level,
                                maximum_planned_looks=(
                                    None
                                    if self.oracle_precision_policy.precision_only_iid
                                    else 2
                                ),
                                precision_only_iid=bool(
                                    self.oracle_precision_policy.precision_only_iid
                                ),
                            )
                            self.baseline_risk_certificate = refined
                        else:
                            receipt = execute_calibration(
                                forecast=forecast,
                                candidate_mask=entry.candidate.resulting_mask,
                                dense_mask=dense_mask,
                            )
                        after_usage = dict(
                            getattr(provider, "resource_usage", {}) or {}
                        )
                        session.compute_ledger.record(
                            RealizedCost(
                                wall_seconds=(
                                    __import__("time").perf_counter() - started
                                ),
                                forward_passes=max(
                                    0,
                                    int(after_usage.get("forward_passes", 0))
                                    - int(before_usage.get("forward_passes", 0)),
                                ),
                                calibration_fits=max(
                                    0,
                                    int(
                                        after_usage.get(
                                            "calibration_fit_count",
                                            0,
                                        )
                                    )
                                    - int(
                                        before_usage.get(
                                            "calibration_fit_count",
                                            0,
                                        )
                                    ),
                                ),
                                packing_degree=1,
                            )
                        )
                        session.cache.setdefault(
                            "joint_precision_allocation_receipts",
                            [],
                        ).append(dict(receipt))
                        if forecast.role == "candidate_calibration":
                            refined_fit = provider.calibration_fits[
                                entry.candidate.resulting_mask.mask_hash
                            ]
                            search.set_calibration_improvement_bound(
                                candidate_id,
                                refined_fit.maximum_bce_excess_risk_bound,
                            )
                            session.cache[
                                "candidate_calibration_improvement_bound:"
                                f"{candidate_id}"
                            ] = float(refined_fit.maximum_bce_excess_risk_bound)
                            session.cache[
                                f"candidate_calibration_profile:{candidate_id}"
                            ] = CalibrationImprovementProfile(
                                first_order_coefficient=(
                                    refined_fit.first_order_excess_risk_coefficient
                                ),
                                second_order_coefficient=(
                                    refined_fit.second_order_excess_risk_coefficient
                                ),
                                parameter_dimension=(
                                    refined_fit.uncertainty_parameter_dimension
                                ),
                            )
                        elif forecast.role == "dense_calibration":
                            assert dense_mask is not None
                            assert self.baseline_risk_certificate is not None
                            refresh = getattr(
                                provider,
                                "refresh_baseline_calibration_certificate",
                                None,
                            )
                            if not callable(refresh):
                                raise RuntimeError(
                                    "dense calibration refinement cannot "
                                    "reapply its frozen oracle evidence"
                                )
                            refined = refresh(self.baseline_risk_certificate)
                            self.baseline_risk_certificate = refined
                        if forecast.role in {
                            "dense_calibration",
                            "dense_oracle",
                        }:
                            for observed_candidate_id in tuple(search.observations):
                                self._install_v1_joint_decision_interval(
                                    session=session,
                                    search=search,
                                    candidate_id=observed_candidate_id,
                                    advance_look=True,
                                )
                        elif forecast.role == "candidate_calibration":
                            self._install_v1_joint_decision_interval(
                                session=session,
                                search=search,
                                candidate_id=candidate_id,
                                advance_look=True,
                            )
                        session.decision.record_transition(
                            "joint_precision_acquisition",
                            "the selected statistical role promised the "
                            "largest decision-envelope contraction per "
                            "predicted GPU-second",
                            candidate_id=candidate_id,
                            selected_role=forecast.role,
                            allocation_receipt=dict(receipt),
                        )
                        continue
            oracle_request = self._oracle_request(
                session=session,
                candidate_ids=candidate_ids,
                look=look,
                requested_context_count=oracle_context_count,
            )
            advisor_requests = self._advisor_requests(
                session, phase="boundary", search=search
            )
            covariance_requests = self._boundary_covariance_expansion_requests(session)
            auxiliary_requests = (*advisor_requests, *covariance_requests)
            required_auxiliary_requests = tuple(
                item
                for item in auxiliary_requests
                if str(item.payload.get("measurement_requirement_class", ""))
                in {
                    "candidate_birth_prerequisite",
                    "chart_validity_prerequisite",
                }
            )
            adapter_blocking_reasons = self._adapter_blocking_reasons(
                session,
                phase="boundary",
            )
            selectable_requests = (
                (oracle_request, *required_auxiliary_requests)
                if auxiliary_looks >= self.maximum_measurement_refreshes
                else (
                    oracle_request,
                    *auxiliary_requests,
                )
            )
            if adapter_blocking_reasons and not required_auxiliary_requests:
                acquisition_prerequisite_blocked = True
                stop_reason = "no commit: " + "; ".join(adapter_blocking_reasons)
                break
            selected = self._select_request_batch(
                session,
                selectable_requests,
                phase="boundary",
                maximum_requests=1,
            )
            if not selected:
                blocking_requests = tuple(
                    item
                    for item in selectable_requests
                    if str(item.payload.get("measurement_requirement_class", ""))
                    in {
                        "candidate_birth_prerequisite",
                        "chart_validity_prerequisite",
                    }
                )
                acquisition_prerequisite_blocked = bool(
                    blocking_requests or adapter_blocking_reasons
                )
                stop_reason = (
                    "no commit: "
                    + (
                        "; ".join(adapter_blocking_reasons)
                        if adapter_blocking_reasons
                        else "a required response/chart measurement did not fit "
                        "the hard resource caps"
                    )
                    if acquisition_prerequisite_blocked
                    else "common acquisition engine found no affordable boundary measurement"
                )
                break
            request = selected[0]
            outcome = self._execute_and_record(session, request)
            if request.kind != "hard_mask_oracle":
                frozen_candidate_state_hash = session.state_hash
                if request.kind == "covariance_rank_expansion":
                    self._consume_covariance_rank_expansion(
                        session=session,
                        request=request,
                        outcome=outcome,
                    )
                else:
                    self._consume_auxiliary(session, request, outcome)
                auxiliary_looks += 1
                look += 1
                if session.state_hash != frozen_candidate_state_hash:
                    candidate_pool_retired = True
                    stop_reason = (
                        "auxiliary measurement changed the frozen parent state; "
                        "all candidates were retired before exact adjudication"
                    )
                    session.decision.record_transition(
                        "retire_candidate_pool",
                        stop_reason,
                        prior_state_hash=frozen_candidate_state_hash,
                        replacement_state_hash=session.state_hash,
                        request_id=request.request_id,
                        kind=request.kind,
                    )
                    break
                continue
            reason_by_id = {
                proposal.candidate_id: proposal.reason
                for proposal in evaluable
                if proposal.candidate_id is not None
            }
            resampled = any(proposal.resample for proposal in evaluable)
            consumed = self._consume_oracle_outcome(
                session=session,
                search=search,
                request=request,
                outcome=outcome,
                reasons=reason_by_id,
                resample=resampled,
            )
            evaluated.extend(consumed)
            masks_evaluated += len(consumed)
            look += 1
            self._maybe_refine_dense_baseline(
                session=session,
                search=search,
                candidate_ids=consumed,
            )

        terminal_adapter_blocks = self._adapter_blocking_reasons(
            session,
            phase="boundary",
        )
        if terminal_adapter_blocks:
            acquisition_prerequisite_blocked = True
            stop_reason = "no commit: " + "; ".join(terminal_adapter_blocks)

        committed: str | None = None
        transition_receipt: ResponseTransitionReceipt | None = None
        predictive_validity_receipt: PredictiveValidityReceipt | None = None
        authority_decision: ExactHardMaskCommitDecision | None = None
        try:
            selected = (
                None
                if candidate_pool_retired or acquisition_prerequisite_blocked
                else search.best_feasible_candidate_id()
            )
        except ValueError:
            stop_reason = "no confidently feasible complete mask was measured"
            selected = None
        if selected is not None:
            entry = session.frontier.entry(selected)
            released = entry.candidate.released_cost.coordinate(
                session.identity.cost.primary_coordinate
            )
            if released + 1e-12 < session.stopping.minimum_released_cost:
                stop_reason = (
                    "best feasible action does not release the required minimum cost"
                )
            elif commit:
                predictive_validity_receipt = self._predictive_validity_receipt(
                    session=session,
                    selected_candidate_id=selected,
                    parent_state_hash=parent_state_hash,
                    parent_mask_hash=parent_mask_hash,
                )
                transition_failure: Exception | None = None
                transition_bundle: (
                    tuple[ResponseTransitionReceipt, MeasurementRequest, object] | None
                ) = None
                try:
                    transition_bundle = self._acquire_selected_response_transition(
                        session=session,
                        candidate_id=selected,
                        look=look,
                    )
                except Exception as error:
                    if self.commit_authority_mode == "exploratory":
                        raise
                    transition_failure = error
                    session.decision.record_transition(
                        "response_transition_acquisition_failed",
                        "response-transition guidance was unavailable; an "
                        "accepted mask would force geometry refresh",
                        candidate_id=selected,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                if (
                    transition_bundle is not None
                    and self.commit_authority_mode == "glocal_v3_shadow"
                ):
                    _, transition_request, transition_outcome = transition_bundle
                    register_guidance = getattr(
                        self.commit_authority,
                        "register_guidance_observation",
                        None,
                    )
                    if not callable(register_guidance):
                        raise RuntimeError(
                            "shadow commit authority cannot register "
                            "response-transition guidance lineage"
                        )
                    register_guidance(
                        observation_id=(
                            "irg-response-transition:"
                            f"{transition_request.request_id}"
                        ),
                        context_ids=tuple(transition_outcome.context_ids),
                        sampling_design_hash=content_hash(
                            {
                                "role": "guidance_geometry",
                                "measurement_request_id": (
                                    transition_request.request_id
                                ),
                                "context_selector": (
                                    transition_request.context_selector
                                ),
                            }
                        ),
                    )
                authority_decision = self._authorize_exact_commit(
                    session=session,
                    candidate=entry.candidate,
                )
                if authority_decision is not None and not authority_decision.accepted:
                    if entry.status is not CandidateStatus.REFUSED:
                        raise RuntimeError(
                            "nonaccepted glocal authority returned without "
                            "atomically refusing the IRG candidate"
                        )
                    stop_reason = (
                        "no commit: shadow glocal-v3 authority returned "
                        f"{authority_decision.outcome}: {authority_decision.reason}"
                    )
                else:
                    if authority_decision is None:
                        session.commit_candidate(selected)
                    elif (
                        session.current_mask.mask_hash
                        != entry.candidate.resulting_mask.mask_hash
                        or not session.committed_candidate_ids
                        or session.committed_candidate_ids[-1] != selected
                    ):
                        raise RuntimeError(
                            "accepted glocal authority returned without the atomic "
                            "IRG commit"
                        )
                    committed = selected
                    carried_calibration_state_id = session.cache.get(
                        f"candidate_calibration_state:{selected}"
                    )
                    if carried_calibration_state_id is not None:
                        session.cache["current_calibration_state_id"] = str(
                            carried_calibration_state_id
                        )
                        session.decision.record_transition(
                            "carry_forward_candidate_calibration",
                            "accepted candidate calibration became the current "
                            "mask calibration lineage",
                            candidate_id=selected,
                            calibration_state_id=str(carried_calibration_state_id),
                        )
                    calibration_changed = bool(
                        authority_decision is not None
                        and authority_decision.parent_calibration_hash
                        != authority_decision.candidate_calibration_hash
                    )
                    if calibration_changed:
                        if transition_bundle is not None:
                            _, transition_request, transition_outcome = (
                                transition_bundle
                            )
                            session.decision.record_transition(
                                "retire_precommit_response_transition",
                                "the accepted mask installed a different exact "
                                "calibration fit, so response geometry measured "
                                "under the prior fit cannot be transported",
                                candidate_id=selected,
                                request_id=transition_request.request_id,
                                outcome_hash=transition_outcome.integrity_hash,
                                parent_calibration_hash=(
                                    authority_decision.parent_calibration_hash
                                ),
                                candidate_calibration_hash=(
                                    authority_decision.candidate_calibration_hash
                                ),
                            )
                            transition_bundle = None
                    if transition_bundle is not None:
                        measured_receipt, transition_request, transition_outcome = (
                            transition_bundle
                        )
                        try:
                            self._apply_committed_response_transition(
                                session=session,
                                candidate_id=selected,
                                receipt=measured_receipt,
                                request=transition_request,
                                outcome=transition_outcome,
                            )
                            transition_receipt = measured_receipt
                        except Exception as error:
                            transition_failure = error
                            transition_receipt = None
                            session.decision.record_transition(
                                "response_transition_application_failed",
                                "the exact mask committed, but response geometry "
                                "must refresh before another proposal",
                                candidate_id=selected,
                                error_type=type(error).__name__,
                                error_message=str(error),
                                transition_receipt_hash=(measured_receipt.content_hash),
                            )
                    if transition_receipt is not None:
                        suffix = " with paired response-transition calibration"
                    elif transition_failure is not None:
                        suffix = " with mandatory response-geometry refresh"
                    else:
                        suffix = ""
                    stop_reason = (
                        "committed best measured feasible complete mask"
                        f"{suffix}: {stop_reason}"
                    )
            else:
                session.decision.record_transition(
                    "provisional_selection",
                    "exploratory evaluation selected a measured feasible mask "
                    "without mutating canonical pruning state",
                    candidate_id=selected,
                    candidate_mask_hash=entry.candidate.resulting_mask.mask_hash,
                    parent_mask_hash=session.current_mask.mask_hash,
                )
                stop_reason = (
                    "provisional nonmutating selection of best measured feasible "
                    f"complete mask: {selected}"
                )
        validity = self._combine_model_validity(
            self.version.assess_model_validity(session),
            (
                predictive_validity_receipt.decision
                if committed is not None and predictive_validity_receipt is not None
                else None
            ),
            predictive_receipt_hash=(
                predictive_validity_receipt.content_hash
                if committed is not None and predictive_validity_receipt is not None
                else None
            ),
        )
        if (
            committed is not None
            and session.identity.geometric_partitioning is not None
            and validity.action in {"reuse", "shrink"}
        ):
            if transition_receipt is None:
                validity = ModelValidityDecision(
                    action="refresh",
                    reason=(
                        "partition-local geometry cannot cross a committed mask "
                        "without an exact response-transition receipt"
                    ),
                    recommended_scale=0.0,
                    diagnostics={
                        "prior_validity_action": validity.action,
                        "geometric_partitioning_hash": (
                            session.identity.geometric_partitioning.content_hash
                        ),
                    },
                )
            else:
                self._transport_geometry_after_commit(
                    session,
                    receipt=transition_receipt,
                )
        session.decision.pending_model_validity_action = validity.action
        session.decision.recommended_action_scale = float(validity.recommended_scale)
        if validity.action != "fallback":
            session.decision.requested_capability_transition = None
        session.decision.record_transition(
            "model_validity",
            validity.reason,
            validity_action=validity.action,
            recommended_scale=validity.recommended_scale,
        )
        boundary_diagnostics = dict(search.diagnostics())
        boundary_diagnostics["sequential_bce_decision_receipts"] = (
            self._candidate_only_decision_receipts(
                session=session,
                search=search,
            )
        )
        boundary_diagnostics["baseline_risk_certificate_hash"] = (
            None
            if self.baseline_risk_certificate is None
            else self.baseline_risk_certificate.content_hash
        )
        return EpochResult(
            parent_state_hash=parent_state_hash,
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            evaluated_candidate_ids=tuple(evaluated),
            committed_candidate_id=committed,
            stop_reason=stop_reason,
            boundary_diagnostics=boundary_diagnostics,
            model_validity_action=validity.action,
            commit_authority_mode=self.commit_authority_mode,
            commit_authority_decision_hash=(
                None if authority_decision is None else authority_decision.content_hash
            ),
            candidate_pool_retired=candidate_pool_retired,
        )

    def run(
        self,
        session: PruningSessionState,
        *,
        maximum_commits: int | None = 1,
        budget_targets: Sequence[float] = (),
    ) -> PruningRunResult:
        limit = None if maximum_commits is None else int(maximum_commits)
        if limit is not None and limit <= 0:
            raise ValueError("maximum_commits must be positive")
        initial_mask_hash = session.current_mask.mask_hash
        epochs: list[EpochResult] = []
        committed_count = 0
        while limit is None or committed_count < limit:
            result = self.run_epoch(session, budget_targets=budget_targets, commit=True)
            epochs.append(result)
            if result.committed_candidate_id is not None:
                committed_count += 1
                session.advance_iteration()
                continue
            if result.candidate_pool_retired:
                session.advance_iteration()
                continue
            if result.committed_candidate_id is None:
                break
        return PruningRunResult(
            run_id=session.run_id,
            initial_mask_hash=initial_mask_hash,
            final_mask_hash=session.current_mask.mask_hash,
            committed_candidate_ids=tuple(session.committed_candidate_ids),
            epoch_results=tuple(epochs),
            compute_summary=session.compute_ledger.summary(),
            frontier_ledger=session.frontier.to_dict(),
            commit_authority_mode=self.commit_authority_mode,
            commit_authority_decision_hashes=tuple(
                result.commit_authority_decision_hash
                for result in epochs
                if result.commit_authority_decision_hash is not None
            ),
        )


__all__ = ["CommonIRGRunner", "EpochResult", "PruningRunResult"]
