"""Shared interfaces and invariant helpers for IRG response versions.

A response version is not a pruning-policy menu.  It is a retained-information
model whose optimizer, candidate family, measurement requirements, and refresh
rule are induced by that information.  Numerical solver fidelity is carried in
``ResponseVersionSpec.fidelity`` and never changes the scientific version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..candidates import ActionCandidate
from ..contracts import (
    ActionPrediction,
    ContextSelector,
    MeasurementKind,
    MeasurementOutcome,
    MeasurementProvenance,
    MeasurementRequest,
    ModelValidityDecision,
    ProjectionReceipt,
    HistoricalResponseProfileName,
    ResponseVersionSpec,
    content_hash,
)
from ..costs import CostEstimate, CostVector
from ..state import PruningSessionState
from ..trust_geometry import GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY


def readonly_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
    return array


def readonly_bool_vector(
    values: Sequence[bool] | np.ndarray, *, size: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=bool).reshape(-1).copy()
    if array.size != int(size):
        raise ValueError(f"{name} must align with the route axis")
    array.setflags(write=False)
    return array


def readonly_matrix(
    values: Sequence[Sequence[float]] | np.ndarray, *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    if array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
    return array


def validate_route_state(
    *,
    route_ids: Sequence[str],
    route_costs: Sequence[float] | np.ndarray,
    protected: Sequence[bool] | np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    identifiers = tuple(str(item) for item in route_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("route_ids must be finite, nonempty, and unique")
    costs = readonly_vector(route_costs, name="route_costs")
    if costs.size != len(identifiers) or np.any(costs <= 0.0):
        raise ValueError("route_costs must be positive and align with route_ids")
    protected_array = readonly_bool_vector(
        protected, size=len(identifiers), name="protected"
    )
    return identifiers, costs, protected_array


def released_cost_vector(
    *,
    primary_coordinate: str,
    route_costs: np.ndarray,
    removed_indices: Sequence[int],
) -> CostVector:
    indices = np.asarray(removed_indices, dtype=np.int64).reshape(-1)
    total = float(np.sum(route_costs[indices])) if indices.size else 0.0
    count = float(indices.size)
    coordinates: dict[str, Any] = {"route_count": count}
    if primary_coordinate in {
        "route_count",
        "parameter_count",
        "flop_count",
        "memory_bytes",
        "latency_ms",
    }:
        coordinates[primary_coordinate] = total
        return CostVector(**coordinates)
    return CostVector(route_count=count, custom={str(primary_coordinate): total})


def available_route_mask(
    session: PruningSessionState, protected: np.ndarray
) -> np.ndarray:
    retained = session.current_mask.to_bool_array()
    if retained.size != protected.size:
        raise ValueError(
            "version state and session mask use different route dimensions"
        )
    available = retained & ~protected
    # The common runner sets this transient filter while invoking a version
    # solver once per frozen geometric partition.  Solvers therefore never
    # rank raw route scores across incommensurate partitions.
    from ..geometric_partitions import GEOMETRIC_PARTITION_FILTER_CACHE_KEY

    active = session.cache.get(GEOMETRIC_PARTITION_FILTER_CACHE_KEY)
    if active is not None:
        allowed = np.zeros(available.size, dtype=bool)
        indices = np.asarray(tuple(int(value) for value in active), dtype=np.int64)
        if (
            indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= available.size)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("active geometric partition route filter is invalid")
        allowed[indices] = True
        available &= allowed
    return available


def response_state_is_current(session: PruningSessionState, state: object) -> bool:
    """Whether a measured local chart may be reused at the current mask.

    Direct equality is the ordinary freshly measured case.  The transport
    case is allowed only when exact transition receipts have advanced the
    chart from its preserved measurement anchor to the current complete mask.
    """

    active_calibration = session.cache.get("irg_calibration_uncertainty_state_v1")
    active_calibration_id = str(getattr(active_calibration, "calibration_state_id", ""))
    intrinsic_state = getattr(state, "intrinsic_state", None)
    chart_calibration_id = str(
        getattr(
            state,
            "calibration_state_id",
            getattr(intrinsic_state, "calibration_state_id", ""),
        )
    )
    if active_calibration_id and chart_calibration_id != active_calibration_id:
        return False
    measured = str(getattr(state, "measured_mask_hash", ""))
    current = session.current_mask.mask_hash
    if measured == current:
        return True
    decision = session.decision
    return bool(
        measured
        and decision.geometry_anchor_mask_hash == measured
        and decision.geometry_current_mask_hash == current
        and decision.geometry_transition_receipt_hashes
    )


def outcome_kind(outcome: MeasurementOutcome) -> str:
    kind = str(outcome.metadata.get("kind", ""))
    if not kind:
        raise ValueError(
            "measurement outcome metadata must include its measurement kind"
        )
    return kind


def default_provenance(version: HistoricalResponseProfileName) -> MeasurementProvenance:
    return MeasurementProvenance(
        producer_id="mechanistic_systems.circuits.irg",
        producer_version="1",
        code_hash=content_hash({"package": "irg", "version": version}),
    )


def default_jacobian_request(
    *,
    session: PruningSessionState,
    version: HistoricalResponseProfileName,
    requested_count: int,
    expected_backward_passes: float = 1.0,
    measurement_kind: MeasurementKind = "jacobian_rows",
    payload: Mapping[str, Any] | None = None,
) -> MeasurementRequest:
    selector = ContextSelector(
        selector_id="irg_active_contrast_context_selector_v1",
        requested_count=max(1, int(requested_count)),
        random_seed=session.random_seed + session.iteration,
        parameters={"allocation": "frontier_relevance_per_realized_cost"},
    )
    request_payload = {"mask_hash": session.current_mask.mask_hash, **dict(payload or {})}
    if (
        measurement_kind == "response_moments"
        and session.identity.global_trust_geometry is not None
    ):
        request_payload.setdefault(
            "global_trust_geometry_spec",
            session.identity.global_trust_geometry,
        )
        request_payload.setdefault("retain_full_route_global_trust", True)
        requested_rank = session.cache.get(
            GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY
        )
        if requested_rank is not None:
            request_payload.setdefault(
                "global_trust_requested_rank", int(requested_rank)
            )
    if (
        measurement_kind == "response_moments"
        and session.identity.functional_atlas is not None
    ):
        request_payload.setdefault(
            "functional_atlas_spec",
            session.identity.functional_atlas,
        )
    if (
        measurement_kind == "response_moments"
        and session.identity.interaction_schedule is not None
    ):
        request_payload.setdefault(
            "interaction_schedule_spec",
            session.identity.interaction_schedule,
        )
    return MeasurementRequest.create(
        kind=measurement_kind,
        state_hash=session.state_hash,
        version=version,
        context_selector=selector,
        expected_cost=CostEstimate(backward_passes=float(expected_backward_passes)),
        expected_decision_value=1.0,
        reuse_keys=(
            session.identity.model_state_hash,
            session.current_mask.mask_hash,
            session.identity.metric.metric_id,
            session.identity.calibration.policy_id,
        ),
        provenance=default_provenance(version),
        payload=request_payload,
    )


def cached_matrix_response_validity(
    session: PruningSessionState,
    *,
    version_label: str,
    reuse_threshold: float = 0.10,
    refresh_threshold: float = 0.35,
) -> ModelValidityDecision | None:
    """Translate the latest paired matrix-response receipt into a version decision.

    The receipt calibrates the transition that produced the current complete
    mask.  It is therefore bound by candidate-mask identity rather than by the
    post-commit session-state hash.  Stale receipts are ignored instead of
    silently governing a different mask.
    """

    from ..response_geometry import (
        RESPONSE_TRANSITION_CACHE_KEY,
        ResponseEllipsoidTrustRegion,
        ResponseTransitionReceipt,
    )

    receipt = session.cache.get(RESPONSE_TRANSITION_CACHE_KEY)
    if receipt is None:
        return None
    if not isinstance(receipt, ResponseTransitionReceipt):
        raise TypeError("matrix-response cache contains an invalid receipt type")
    if receipt.candidate_mask_hash != session.current_mask.mask_hash:
        return None
    decision = ResponseEllipsoidTrustRegion(
        reuse_threshold=float(reuse_threshold),
        refresh_threshold=float(refresh_threshold),
    ).evaluate(receipt)
    return ModelValidityDecision(
        action=decision.action,
        reason=f"{version_label} matrix response trust region: {decision.reason}",
        recommended_scale=decision.recommended_scale,
        diagnostics={
            **dict(decision.diagnostics),
            "version_label": str(version_label),
            "receipt_bound_to_current_mask": True,
            "normalization_mode": receipt.validity.metadata.get(
                "normalization_mode", "unknown"
            ),
            "response_safe_region_hash": receipt.validity.metadata.get(
                "response_constraint_hash", ""
            ),
        },
    )


@dataclass(frozen=True)
class GeneratedCandidateSet:
    candidates: tuple[ActionCandidate, ...]
    predictions: tuple[ActionPrediction, ...]

    def __post_init__(self) -> None:
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        prediction_ids = tuple(item.candidate_id for item in self.predictions)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("generated candidate set contains duplicate candidates")
        if set(candidate_ids) != set(prediction_ids):
            raise ValueError(
                "every generated candidate requires exactly one prediction"
            )


@runtime_checkable
class VersionState(Protocol):
    version: HistoricalResponseProfileName

    @property
    def state_hash(self) -> str: ...


@runtime_checkable
class ResponseVersion(Protocol):
    name: HistoricalResponseProfileName
    spec: ResponseVersionSpec

    def required_measurements(
        self, session: PruningSessionState
    ) -> list[MeasurementRequest]: ...

    def update_response_state(
        self, state: VersionState, outcome: MeasurementOutcome
    ) -> VersionState: ...

    def generate_candidates(
        self,
        session: PruningSessionState,
        budget_targets: Sequence[float],
    ) -> list[ActionCandidate]: ...

    def predict_action(
        self,
        candidate: ActionCandidate,
        session: PruningSessionState,
    ) -> ActionPrediction: ...

    def assess_model_validity(
        self, session: PruningSessionState
    ) -> ModelValidityDecision: ...

    def project_to_lower_version(
        self,
        state: VersionState,
        target: HistoricalResponseProfileName,
    ) -> tuple[VersionState, tuple[ProjectionReceipt, ...]]: ...


__all__ = [
    "GeneratedCandidateSet",
    "ResponseVersion",
    "VersionState",
    "available_route_mask",
    "cached_matrix_response_validity",
    "default_jacobian_request",
    "default_provenance",
    "outcome_kind",
    "readonly_bool_vector",
    "readonly_matrix",
    "readonly_vector",
    "response_state_is_current",
    "released_cost_vector",
    "validate_route_state",
]
