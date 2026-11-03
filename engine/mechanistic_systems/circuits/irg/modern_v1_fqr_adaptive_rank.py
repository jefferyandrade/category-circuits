"""Outcome-independent FD-rank authority for canonical F/Q/R proposal D.

This module owns no scorer, KKT solver, C-stream, oracle, or mask mutation.
It registers pre-outcome rank liveness, binds exact D-prefix geometry produced
by an executor, and is the sole issuer of the certificate that permits a
rank-specific provisional KKT family to be frozen before C opens.

The historical v1 surface remains replayable: execution starts at rank 8 and
promotes through ``(8, 16, 32)`` only when the current rank's deterministic FD
residual envelope can still change a winning KKT face, complete hard-F/Q
feasibility, or the next irreversible decision.

The canonical endogenized v2 surface removes rank 8 as a scientific choice.
It registers the route-clipped dyadic rank lattice as an execution
approximation, filters that lattice by pre-outcome phase liveness only, and
then fans one registered D capture/VJP stream into every currently identifiable
rank-local FD state.  Its device-liveness ledger therefore accumulates every
persistent rank-local state in an admitted prefix while counting the shared
capture/VJP packet and serial rank-local workspace exactly once.  No rank is
selected before D, no lower-rank zero-padding is allowed, and C stays closed
until the source decision is invariant under all registered sampling, FD, and
numerical uncertainty.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Integral
from typing import Final, Mapping, NoReturn, Sequence, TypedDict

from .contracts import content_hash
from .modern_v1_relaxed_kkt import (
    V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT,
)
from .modern_v1_stratified_fd_gpu import forecast_stratified_fd_gpu_memory


CANONICAL_FQR_ADAPTIVE_RANK_LADDER: Final = (8, 16, 32)
FQR_ADAPTIVE_RANK_MEMORY_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_memory_forecast_v1"
)
FQR_ADAPTIVE_RANK_POLICY_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_policy_v1"
)
FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_branch_decision_v1"
)
FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_branch_decision_v2"
)
FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_decision_certificate_v1"
)
FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_decision_certificate_v2"
)
FQR_ADAPTIVE_RANK_REPLAY_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_replay_record_v1"
)
FQR_ADAPTIVE_RANK_LEDGER_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_replay_ledger_v1"
)
FQR_ADAPTIVE_RANK_STABILITY_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_stability_certificate_v1"
)
FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_stability_certificate_v2"
)
FQR_ADAPTIVE_RANK_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_no_c_refusal_v1"
)
FQR_D_DECISION_ACQUISITION_GUIDANCE_SCHEMA: Final = (
    "partition_free_v1_fqr_D_decision_acquisition_guidance_v1"
)
FQR_ENDOGENOUS_RANK_MEMORY_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_memory_forecast_v2"
)
FQR_ENDOGENOUS_RANK_POLICY_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_policy_v2"
)
FQR_ENDOGENOUS_RANK_D_PREFIX_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_D_prefix_v2"
)
FQR_ENDOGENOUS_RANK_FANOUT_STATE_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_fanout_state_v2"
)
FQR_ENDOGENOUS_RANK_FANOUT_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_same_VJP_fanout_v2"
)
FQR_ENDOGENOUS_RANK_NO_C_REFUSAL_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_endogenous_rank_no_C_refusal_v2"
)

_MEMORY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-memory-"
_POLICY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-policy-"
_UNCERTAINTY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-set-"
_BRANCH_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-branch-"
_RANK_DECISION_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-decision-"
_REPLAY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-replay-"
_DECISION_FAMILY_PREFIX: Final = (
    "partition-free-v1-fqr-adaptive-rank-decision-family-"
)
_LEDGER_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-ledger-"
_STABILITY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-stability-"
_REFUSAL_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-refusal-"
_D_GUIDANCE_PREFIX: Final = "partition-free-v1-fqr-D-decision-guidance-"
_ENDOGENOUS_MEMORY_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-memory-v2-"
)
_ENDOGENOUS_POLICY_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-policy-v2-"
)
_ENDOGENOUS_D_PREFIX_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-D-prefix-v2-"
)
_ENDOGENOUS_FANOUT_STATE_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-fanout-state-v2-"
)
_ENDOGENOUS_FANOUT_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-same-VJP-fanout-v2-"
)
_ENDOGENOUS_REFUSAL_V2_PREFIX: Final = (
    "partition-free-v1-fqr-endogenous-rank-no-C-refusal-v2-"
)

_BRANCH_AXIS: Final = ("spectral", "diagonal_l1")
_STABILITY_ISSUANCE_TOKEN: Final = object()
_SOURCE_DERIVATION_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_D_PREFIX_V2_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_FANOUT_STATE_V2_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_FANOUT_V2_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2: Final = (
    "simultaneous_persistent_rank_prefix_with_serial_shared_workspace_v2"
)


def canonical_fqr_adaptive_rank_ladder_for_route_count(
    route_count: int,
) -> tuple[int, ...]:
    """Return the outcome-independent canonical rank ladder for this route axis."""

    routes = _positive_int(route_count, name="route count")
    return tuple(
        rank for rank in CANONICAL_FQR_ADAPTIVE_RANK_LADDER if rank <= routes
    ) or (routes,)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _identity_axis(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _phase_liveness_proof_hash(
    *,
    route_count: int,
    rank: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
    D_packet_staging_device_bytes: int,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_fqr_adaptive_rank_phase_liveness_v1",
            "route_count": route_count,
            "rank": rank,
            "legality_block_count": legality_block_count,
            "rounding_tile_size": rounding_tile_size,
            "maximum_hard_action_count": maximum_hard_action_count,
            "D_packet_staging_device_bytes": D_packet_staging_device_bytes,
            "fd_forecast_owner": (
                "modern_v1_stratified_fd_gpu.forecast_stratified_fd_gpu_memory"
            ),
            "kkt_forecast_owner": (
                "modern_v1_relaxed_kkt.estimate_v1_relaxed_kkt_cuda_hbm"
            ),
            "kkt_fused_route_vector_count": (
                V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
            ),
            "rounding_workspace_scalar_factor": 12,
            "phase_order": (
                "D_FD_acquisition_then_transient_recompression_eviction_then_"
                "retained_geometry_KKT_rounding_then_all_D_KKT_eviction_then_C"
            ),
            "fd_recompression_workspace_evicted_before_kkt": True,
            "D_packet_staging_colive_with_fd_accumulators": True,
            "retained_rank_major_geometry_live_during_kkt": True,
            "fused_kkt_vectors_duals_tiles_and_rounding_forecasted": True,
            "kkt_and_rank_geometry_evicted_before_C": True,
            "native_C_overlap_bytes": 0,
            "C_open_authority_before_rank_stability": False,
        },
        prefix="partition-free-v1-fqr-adaptive-rank-phase-liveness-",
    )


class V1FQRAdaptiveRankRefusalCode(str, Enum):
    INITIAL_RANK_EXCEEDS_MEMORY = "initial_rank_exceeds_memory"
    POLICY_NOT_REGISTERED = "policy_not_registered"
    LEGACY_FIXED_RANK_NO_AUTHORITY = "legacy_fixed_rank_no_authority"
    INCOMPLETE_RANK_REPLAY = "incomplete_rank_replay"
    D_PREFIX_DRIFT = "d_prefix_drift"
    NON_RAW_REPLAY = "non_raw_replay"
    ZERO_PADDING = "zero_padding"
    C_OPENED_BEFORE_STABILITY = "c_opened_before_stability"
    INCOMPLETE_HARD_LADDER = "incomplete_hard_ladder"
    RANK_PROMOTION_REQUIRED = "rank_promotion_required"
    RESIDUAL_WITNESS_INVALID = "residual_witness_invalid"
    DECISION_FAMILY_AMBIGUOUS = "decision_family_ambiguous"
    STALE_CERTIFICATE = "stale_certificate"


class V1FQRDDecisionAmbiguitySource(str, Enum):
    """Mutually exclusive owner of the next lawful D-side computation."""

    D_SAMPLING = "D_sampling"
    FD_RANK_TRUNCATION = "FD_rank_truncation"
    KKT_NUMERICAL = "KKT_numerical"
    KKT_OPERATOR_SENSITIVITY = "KKT_operator_sensitivity"
    CROSS_BRANCH_DECISION = "cross_branch_decision"
    NONE = "none"


class V1FQRDDecisionAcquisitionAction(str, Enum):
    """Registered computation that can contract the dominant ambiguity."""

    ACQUIRE_NEXT_REGISTERED_D_PREFIX = "acquire_next_registered_D_prefix"
    REPLAY_IDENTICAL_D_PREFIX_AT_HIGHER_FD_RANK = (
        "replay_identical_D_prefix_at_higher_FD_rank"
    )
    CONTINUE_REGISTERED_PDHG = "continue_registered_PDHG"
    BUILD_FULL_OPERATOR_SENSITIVITY_PROOF = (
        "build_full_operator_sensitivity_proof"
    )
    REFUSE_NONUNIQUE_CROSS_BRANCH_DECISION = (
        "refuse_nonunique_cross_branch_decision"
    )
    OPEN_C_FROM_STABLE_D_CERTIFICATE = "open_C_from_stable_D_certificate"
    EXACT_REGISTERED_D_CENSUS = "exact_registered_D_census"


@dataclass(frozen=True, slots=True)
class V1FQRDDecisionStabilityAcquisitionGuidance:
    """Authority-inert, hash-replayable guidance for unresolved D stability.

    The receipt does not decide a mask or open C.  It only names which already
    registered computation can reduce the uncertainty that currently blocks
    the D decision-stability theorem.  In particular, an unavailable interval
    theorem cannot be replaced by a K/O rejection or by a caller-selected
    sample count.
    """

    policy_hash: str
    current_mask_hash: str
    route_axis_hash: str
    proposal_generation: int
    registered_d_prefix_identity_hash: str
    observed_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    status_by_source: tuple[tuple[str, str], ...]
    dominant_ambiguity: V1FQRDDecisionAmbiguitySource
    required_action: V1FQRDDecisionAcquisitionAction
    source_receipt_hashes: tuple[str, ...]
    next_registered_d_stratum_counts: tuple[int, int] | None = None
    next_admitted_fd_rank: int | None = None
    population_counts_available: bool = True
    exact_census_is_fallback: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FQR_D_DECISION_ACQUISITION_GUIDANCE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "current_mask_hash",
            "route_axis_hash",
            "registered_d_prefix_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        observed = tuple(
            _nonnegative_int(value, name="observed D stratum count")
            for value in self.observed_stratum_counts
        )
        population = tuple(
            _nonnegative_int(value, name="D population stratum count")
            for value in self.population_stratum_counts
        )
        if (
            len(observed) != 2
            or len(population) != 2
        ):
            raise ValueError("D decision guidance has invalid population counts")
        if type(self.population_counts_available) is not bool:
            raise TypeError("D population-count availability must be an exact bool")
        if self.population_counts_available:
            if any(value == 0 for value in population) or any(
                value > limit
                for value, limit in zip(observed, population, strict=True)
            ):
                raise ValueError("D decision guidance has invalid population counts")
        elif population != (0, 0):
            raise ValueError(
                "unavailable D population counts must use the explicit zero sentinel"
            )
        status = tuple(
            (
                _identity(source, name="D ambiguity source"),
                _identity(value, name="D ambiguity status"),
            )
            for source, value in self.status_by_source
        )
        expected_sources = (
            "D_sampling",
            "FD_rank_truncation",
            "KKT_numerical",
            "KKT_operator_sensitivity",
            "cross_branch_decision",
        )
        if tuple(source for source, _value in status) != expected_sources:
            raise ValueError("D decision guidance source axis is not canonical")
        if type(self.dominant_ambiguity) is not V1FQRDDecisionAmbiguitySource:
            raise TypeError("D decision guidance requires an exact ambiguity source")
        if type(self.required_action) is not V1FQRDDecisionAcquisitionAction:
            raise TypeError("D decision guidance requires an exact acquisition action")
        hashes = tuple(
            _identity(value, name="D decision source receipt hash")
            for value in self.source_receipt_hashes
        )
        if len(set(hashes)) != len(hashes):
            raise ValueError("D decision guidance repeated a source receipt")
        next_counts = self.next_registered_d_stratum_counts
        if next_counts is not None:
            if not self.population_counts_available:
                raise ValueError(
                    "unknown D population counts cannot authorize a next prefix"
                )
            normalized_next = tuple(
                _positive_int(value, name="next registered D stratum count")
                for value in next_counts
            )
            if (
                len(normalized_next) != 2
                or any(
                    value < before or value > limit
                    for value, before, limit in zip(
                        normalized_next, observed, population, strict=True
                    )
                )
                or normalized_next == observed
            ):
                raise ValueError("next registered D prefix does not extend this prefix")
            next_counts = normalized_next
        next_rank = self.next_admitted_fd_rank
        if next_rank is not None:
            next_rank = _positive_int(next_rank, name="next admitted FD rank")
        authority_flags = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            type(self.exact_census_is_fallback) is not bool
            or not self.exact_census_is_fallback
            or any(type(value) is not bool for value in authority_flags)
            or any(authority_flags)
            or self.schema != FQR_D_DECISION_ACQUISITION_GUIDANCE_SCHEMA
        ):
            raise PermissionError("D decision guidance gained downstream authority")
        if self.dominant_ambiguity is V1FQRDDecisionAmbiguitySource.NONE:
            if self.required_action is not (
                V1FQRDDecisionAcquisitionAction.OPEN_C_FROM_STABLE_D_CERTIFICATE
            ):
                raise ValueError("resolved D guidance must name stable-C admission")
        elif self.required_action is (
            V1FQRDDecisionAcquisitionAction.OPEN_C_FROM_STABLE_D_CERTIFICATE
        ):
            raise ValueError("unresolved D guidance cannot name stable-C admission")
        object.__setattr__(self, "observed_stratum_counts", observed)
        object.__setattr__(self, "population_stratum_counts", population)
        object.__setattr__(self, "status_by_source", status)
        object.__setattr__(self, "source_receipt_hashes", hashes)
        object.__setattr__(self, "next_registered_d_stratum_counts", next_counts)
        object.__setattr__(self, "next_admitted_fd_rank", next_rank)

    def to_payload(self) -> dict[str, object]:
        """Return the exact JSON-safe, authority-inert replay payload."""

        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "proposal_generation": self.proposal_generation,
            "registered_d_prefix_identity_hash": (
                self.registered_d_prefix_identity_hash
            ),
            "observed_stratum_counts": self.observed_stratum_counts,
            "population_stratum_counts": self.population_stratum_counts,
            "population_counts_available": self.population_counts_available,
            "status_by_source": self.status_by_source,
            "dominant_ambiguity": self.dominant_ambiguity.value,
            "required_action": self.required_action.value,
            "source_receipt_hashes": self.source_receipt_hashes,
            "next_registered_d_stratum_counts": (
                self.next_registered_d_stratum_counts
            ),
            "next_admitted_fd_rank": self.next_admitted_fd_rank,
            "exact_census_is_fallback": self.exact_census_is_fallback,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "V1FQRDDecisionStabilityAcquisitionGuidance":
        """Replay one payload without allowing missing or surplus fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("D decision guidance payload must be a mapping")
        expected = {
            "schema",
            "policy_hash",
            "current_mask_hash",
            "route_axis_hash",
            "proposal_generation",
            "registered_d_prefix_identity_hash",
            "observed_stratum_counts",
            "population_stratum_counts",
            "population_counts_available",
            "status_by_source",
            "dominant_ambiguity",
            "required_action",
            "source_receipt_hashes",
            "next_registered_d_stratum_counts",
            "next_admitted_fd_rank",
            "exact_census_is_fallback",
            "C_stream_open_authority",
            "calibration_K_open_authority",
            "paired_oracle_O_open_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_authority",
        }
        if set(payload) != expected:
            raise ValueError("D decision guidance payload fields are not exact")
        try:
            dominant = V1FQRDDecisionAmbiguitySource(
                payload["dominant_ambiguity"]
            )
            action = V1FQRDDecisionAcquisitionAction(payload["required_action"])
        except (TypeError, ValueError) as error:
            raise ValueError("D decision guidance enum value is invalid") from error
        next_counts_raw = payload["next_registered_d_stratum_counts"]
        value = cls(
            schema=payload["schema"],
            policy_hash=payload["policy_hash"],
            current_mask_hash=payload["current_mask_hash"],
            route_axis_hash=payload["route_axis_hash"],
            proposal_generation=payload["proposal_generation"],
            registered_d_prefix_identity_hash=payload[
                "registered_d_prefix_identity_hash"
            ],
            observed_stratum_counts=tuple(payload["observed_stratum_counts"]),
            population_stratum_counts=tuple(payload["population_stratum_counts"]),
            population_counts_available=payload["population_counts_available"],
            status_by_source=tuple(
                tuple(item) for item in payload["status_by_source"]
            ),
            dominant_ambiguity=dominant,
            required_action=action,
            source_receipt_hashes=tuple(payload["source_receipt_hashes"]),
            next_registered_d_stratum_counts=(
                None if next_counts_raw is None else tuple(next_counts_raw)
            ),
            next_admitted_fd_rank=payload["next_admitted_fd_rank"],
            exact_census_is_fallback=payload["exact_census_is_fallback"],
            C_stream_open_authority=payload["C_stream_open_authority"],
            calibration_K_open_authority=payload["calibration_K_open_authority"],
            paired_oracle_O_open_authority=payload[
                "paired_oracle_O_open_authority"
            ],
            mask_commit_authority=payload["mask_commit_authority"],
            terminal_authority=payload["terminal_authority"],
            pareto_authority=payload["pareto_authority"],
        )
        return value

    @property
    def guidance_hash(self) -> str:
        return content_hash(self.to_payload(), prefix=_D_GUIDANCE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankNoCRefusal:
    code: V1FQRAdaptiveRankRefusalCode
    reason: str
    policy_hash: str | None
    ledger_hash: str | None = None
    current_mask_hash: str | None = None
    proposal_generation: int | None = None
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_REFUSAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.code) is not V1FQRAdaptiveRankRefusalCode:
            raise TypeError("adaptive-rank refusal requires its exact code")
        object.__setattr__(self, "reason", _identity(self.reason, name="reason"))
        for name in ("policy_hash", "ledger_hash", "current_mask_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        if self.proposal_generation is not None:
            object.__setattr__(
                self,
                "proposal_generation",
                _nonnegative_int(
                    self.proposal_generation,
                    name="proposal generation",
                ),
            )
        flags = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            raise PermissionError("adaptive-rank refusal gained downstream authority")
        if self.schema != FQR_ADAPTIVE_RANK_REFUSAL_SCHEMA:
            raise ValueError("adaptive-rank refusal schema is invalid")

    @property
    def refusal_hash(self) -> str:
        return content_hash(self, prefix=_REFUSAL_PREFIX)


class V1FQRAdaptiveRankRefusal(RuntimeError):
    def __init__(self, receipt: V1FQRAdaptiveRankNoCRefusal) -> None:
        if type(receipt) is not V1FQRAdaptiveRankNoCRefusal:
            raise TypeError("adaptive-rank exception requires a typed refusal")
        self.receipt = receipt
        self.durable_checkpoint_payload: bytes | None = None
        self.durable_checkpoint_payload_hash: str | None = None
        super().__init__(f"{receipt.code.value}: {receipt.reason}")

    def bind_durable_checkpoint(self, payload: bytes, payload_hash: str) -> None:
        """Attach one inert restart artifact without changing refusal authority."""

        if type(payload) is not bytes or not payload:
            raise TypeError("adaptive-rank durable checkpoint must be nonempty bytes")
        digest = _identity(payload_hash, name="durable checkpoint payload hash")
        if self.durable_checkpoint_payload is not None:
            if (
                self.durable_checkpoint_payload != payload
                or self.durable_checkpoint_payload_hash != digest
            ):
                raise ValueError("adaptive-rank refusal checkpoint was rebound")
            return
        self.durable_checkpoint_payload = payload
        self.durable_checkpoint_payload_hash = digest


def refuse_v1_fqr_adaptive_rank(
    code: V1FQRAdaptiveRankRefusalCode,
    reason: str,
    *,
    policy_hash: str | None,
    ledger_hash: str | None = None,
    current_mask_hash: str | None = None,
    proposal_generation: int | None = None,
) -> NoReturn:
    raise V1FQRAdaptiveRankRefusal(
        V1FQRAdaptiveRankNoCRefusal(
            code=code,
            reason=reason,
            policy_hash=policy_hash,
            ledger_hash=ledger_hash,
            current_mask_hash=current_mask_hash,
            proposal_generation=proposal_generation,
        )
    )


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankMemoryForecast:
    rank: int
    route_count: int
    stratum_count: int
    response_rows: int
    device_total_memory_bytes: int
    device_reserved_memory_bytes: int
    device_memory_ceiling_bytes: int
    available_rank_workspace_bytes: int
    live_stratum_fd_device_bytes: int
    combined_pre_recompression_factor_device_bytes: int
    combined_output_state_device_bytes: int
    gram_workspace_device_bytes: int
    fd_peak_device_bytes: int
    D_packet_staging_device_bytes: int
    D_acquisition_peak_device_bytes: int
    retained_rank_major_geometry_device_bytes: int
    fused_kkt_problem_device_bytes: int
    fused_kkt_solver_workspace_device_bytes: int
    deterministic_rounding_workspace_device_bytes: int
    post_D_kkt_peak_device_bytes: int
    registered_phase_peak_device_bytes: int
    kkt_legality_block_count: int
    kkt_rounding_tile_size: int
    maximum_hard_action_count: int
    phase_liveness_proof_hash: str
    admitted: bool
    device_memory_ceiling_numerator: int = 85
    device_memory_ceiling_denominator: int = 100
    allocation_free: bool = True
    registered_before_D_outcomes: bool = True
    filter_uses_only_frozen_memory_coordinates: bool = True
    fd_recompression_workspace_evicted_before_kkt: bool = True
    kkt_and_rank_geometry_evicted_before_C: bool = True
    native_C_overlap_bytes: int = 0
    schema: str = FQR_ADAPTIVE_RANK_MEMORY_SCHEMA

    def __post_init__(self) -> None:
        rank = _positive_int(self.rank, name="FD rank")
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(self.stratum_count, name="stratum count")
        rows = _positive_int(self.response_rows, name="response rows")
        blocks = _positive_int(
            self.kkt_legality_block_count,
            name="KKT legality block count",
        )
        tile = _positive_int(
            self.kkt_rounding_tile_size,
            name="KKT rounding tile size",
        )
        maximum_action = _positive_int(
            self.maximum_hard_action_count,
            name="maximum hard action count",
        )
        packet_staging = _nonnegative_int(
            self.D_packet_staging_device_bytes,
            name="D packet staging device bytes",
        )
        total = _positive_int(
            self.device_total_memory_bytes,
            name="device total memory",
        )
        reserved = _nonnegative_int(
            self.device_reserved_memory_bytes,
            name="device reserved memory",
        )
        numerator = _positive_int(
            self.device_memory_ceiling_numerator,
            name="memory ceiling numerator",
        )
        denominator = _positive_int(
            self.device_memory_ceiling_denominator,
            name="memory ceiling denominator",
        )
        if (numerator, denominator) != (85, 100):
            raise ValueError("canonical adaptive FD rank requires the 85% HBM ceiling")
        ceiling = (numerator * total) // denominator
        available = ceiling - reserved
        if available <= 0:
            raise MemoryError("no FD-rank workspace remains below the 85% HBM ceiling")
        base = forecast_stratified_fd_gpu_memory(
            route_count=routes,
            stratum_count=strata,
            response_rows=rows,
            per_stratum_sketch_rank=rank,
            combined_maximum_rank=rank,
        )
        retained_geometry = base.combined_output_state_device_bytes
        # Explicit route-scale KKT liveness model.  Problem tensors include
        # rank-major H, resource/work/residual/sqrt-residual/objective vectors,
        # eligibility, and compact block metadata.  Solver/transient workspace
        # mirrors the landed fused-KKT estimator's public FP64 route-vector
        # bound.
        # Rounding mirrors its 12-scalar bound over one tile plus the retained
        # endpoint face.  These are future allocations and therefore may not be
        # hidden inside allocator bytes already reserved at registration.
        kkt_problem = (
            (rank + 5) * routes * 8
            + routes
            + blocks * 4 * 8
        )
        vector_count = V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
        kkt_solver = (
            vector_count * routes * 8
            + (rank + 4 * blocks + vector_count) * 8
        )
        winner_count = min(routes, maximum_action)
        rounding = 12 * (winner_count + min(routes, tile)) * 8
        post_D_peak = retained_geometry + kkt_problem + kkt_solver + rounding
        D_acquisition_peak = base.peak_device_bytes + packet_staging
        registered_peak = max(D_acquisition_peak, post_D_peak)
        expected = {
            "device_memory_ceiling_bytes": ceiling,
            "available_rank_workspace_bytes": available,
            "live_stratum_fd_device_bytes": base.live_stratum_fd_device_bytes,
            "combined_pre_recompression_factor_device_bytes": (
                base.combined_pre_recompression_factor_device_bytes
            ),
            "combined_output_state_device_bytes": (
                base.combined_output_state_device_bytes
            ),
            "gram_workspace_device_bytes": base.gram_workspace_device_bytes,
            "fd_peak_device_bytes": base.peak_device_bytes,
            "D_packet_staging_device_bytes": packet_staging,
            "D_acquisition_peak_device_bytes": D_acquisition_peak,
            "retained_rank_major_geometry_device_bytes": retained_geometry,
            "fused_kkt_problem_device_bytes": kkt_problem,
            "fused_kkt_solver_workspace_device_bytes": kkt_solver,
            "deterministic_rounding_workspace_device_bytes": rounding,
            "post_D_kkt_peak_device_bytes": post_D_peak,
            "registered_phase_peak_device_bytes": registered_peak,
        }
        for name, expected_value in expected.items():
            if _nonnegative_int(getattr(self, name), name=name) != expected_value:
                raise ValueError(f"adaptive-rank {name} does not replay")
        admitted = registered_peak <= available
        flags = (
            self.admitted,
            self.allocation_free,
            self.registered_before_D_outcomes,
            self.filter_uses_only_frozen_memory_coordinates,
            self.fd_recompression_workspace_evicted_before_kkt,
            self.kkt_and_rank_geometry_evicted_before_C,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive-rank forecast flags must be exact booleans")
        if self.admitted is not admitted:
            raise ValueError("adaptive-rank memory admission does not replay")
        if not all(flags[1:]):
            raise PermissionError("adaptive-rank memory forecast crossed its boundary")
        if _nonnegative_int(
            self.native_C_overlap_bytes,
            name="native C overlap bytes",
        ) != 0:
            raise PermissionError(
                "canonical adaptive rank requires KKT/rank geometry eviction before C"
            )
        supplied_liveness = _identity(
            self.phase_liveness_proof_hash,
            name="phase liveness proof hash",
        )
        expected_liveness = _phase_liveness_proof_hash(
            route_count=routes,
            rank=rank,
            legality_block_count=blocks,
            rounding_tile_size=tile,
            maximum_hard_action_count=maximum_action,
            D_packet_staging_device_bytes=packet_staging,
        )
        if supplied_liveness != expected_liveness:
            raise ValueError("adaptive-rank phase-liveness proof does not replay")
        object.__setattr__(self, "phase_liveness_proof_hash", supplied_liveness)
        if self.schema != FQR_ADAPTIVE_RANK_MEMORY_SCHEMA:
            raise ValueError("adaptive-rank memory schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "stratum_count", strata)
        object.__setattr__(self, "response_rows", rows)
        object.__setattr__(self, "kkt_legality_block_count", blocks)
        object.__setattr__(self, "kkt_rounding_tile_size", tile)
        object.__setattr__(self, "maximum_hard_action_count", maximum_action)
        object.__setattr__(self, "device_total_memory_bytes", total)
        object.__setattr__(self, "device_reserved_memory_bytes", reserved)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_MEMORY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankPolicy:
    policy_id: str
    route_count: int
    stratum_count: int
    response_rows: int
    requested_rank_ladder: tuple[int, ...]
    admitted_rank_ladder: tuple[int, ...]
    memory_forecasts: tuple[V1FQRAdaptiveRankMemoryForecast, ...]
    registered_before_D_outcomes: bool = True
    empirical_outcomes_used_for_filtering: bool = False
    rank_growth_requires_raw_D_prefix_reexecution: bool = True
    rank_promotion_only_on_residual_ambiguity: bool = True
    stop_at_first_robust_decision: bool = True
    lower_rank_zero_padding_authority: bool = False
    C_may_open_before_stability: bool = False
    legacy_fixed_rank_commit_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_POLICY_SCHEMA
    _policy_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(self.stratum_count, name="stratum count")
        rows = _positive_int(self.response_rows, name="response rows")
        requested = tuple(
            _positive_int(value, name="requested FD rank")
            for value in self.requested_rank_ladder
        )
        if requested != canonical_fqr_adaptive_rank_ladder_for_route_count(routes):
            raise ValueError(
                "canonical adaptive FD-rank ladder must be clipped only by route count"
            )
        forecasts = tuple(self.memory_forecasts)
        if (
            len(forecasts) != len(requested)
            or any(type(value) is not V1FQRAdaptiveRankMemoryForecast for value in forecasts)
            or tuple(value.rank for value in forecasts) != requested
            or any(
                (value.route_count, value.stratum_count, value.response_rows)
                != (routes, strata, rows)
                for value in forecasts
            )
        ):
            raise ValueError("adaptive-rank forecasts do not cover the registered ladder")
        admitted = tuple(value.rank for value in forecasts if value.admitted)
        supplied = tuple(
            _positive_int(value, name="admitted FD rank")
            for value in self.admitted_rank_ladder
        )
        if supplied != admitted:
            raise ValueError("adaptive-rank admitted ladder does not replay memory filtering")
        initial_rank = requested[0]
        if not admitted or admitted[0] != initial_rank:
            refuse_v1_fqr_adaptive_rank(
                V1FQRAdaptiveRankRefusalCode.INITIAL_RANK_EXCEEDS_MEMORY,
                (
                    f"initial rank {initial_rank} is not admitted below "
                    "the frozen 85% HBM ceiling"
                ),
                policy_hash=None,
            )
        flags = (
            self.registered_before_D_outcomes,
            self.empirical_outcomes_used_for_filtering,
            self.rank_growth_requires_raw_D_prefix_reexecution,
            self.rank_promotion_only_on_residual_ambiguity,
            self.stop_at_first_robust_decision,
            self.lower_rank_zero_padding_authority,
            self.C_may_open_before_stability,
            self.legacy_fixed_rank_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive-rank policy flags must be exact booleans")
        if flags != (True, False, True, True, True, False, False, False):
            raise PermissionError("adaptive-rank policy crossed its causal boundary")
        if self.schema != FQR_ADAPTIVE_RANK_POLICY_SCHEMA:
            raise ValueError("adaptive-rank policy schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "stratum_count", strata)
        object.__setattr__(self, "response_rows", rows)
        object.__setattr__(self, "requested_rank_ladder", requested)
        object.__setattr__(self, "admitted_rank_ladder", admitted)
        object.__setattr__(self, "memory_forecasts", forecasts)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "route_count": self.route_count,
            "stratum_count": self.stratum_count,
            "response_rows": self.response_rows,
            "requested_rank_ladder": self.requested_rank_ladder,
            "admitted_rank_ladder": self.admitted_rank_ladder,
            "memory_forecast_hashes": tuple(
                value.forecast_hash for value in self.memory_forecasts
            ),
            "registered_before_D_outcomes": self.registered_before_D_outcomes,
            "empirical_outcomes_used_for_filtering": (
                self.empirical_outcomes_used_for_filtering
            ),
            "rank_growth_requires_raw_D_prefix_reexecution": (
                self.rank_growth_requires_raw_D_prefix_reexecution
            ),
            "rank_promotion_only_on_residual_ambiguity": (
                self.rank_promotion_only_on_residual_ambiguity
            ),
            "stop_at_first_robust_decision": self.stop_at_first_robust_decision,
            "lower_rank_zero_padding_authority": (
                self.lower_rank_zero_padding_authority
            ),
            "C_may_open_before_stability": self.C_may_open_before_stability,
            "legacy_fixed_rank_commit_authority": (
                self.legacy_fixed_rank_commit_authority
            ),
        }

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    @property
    def uncertainty_set_hash(self) -> str:
        return content_hash(
            {
                "policy_hash": self.policy_hash,
                "requested_rank_ladder": self.requested_rank_ladder,
                "admitted_rank_ladder": self.admitted_rank_ladder,
                "same_registered_D_prefix_required": True,
                "rank_local_fd_residual_margin_witness_required": True,
                "rank_promotion_only_on_residual_ambiguity": True,
                "stop_at_first_robust_decision": True,
                "cross_rank_agreement_is_not_C_authority": True,
                "max_rank_unresolved_is_no_C_refusal": True,
                "deterministic_uncertainty_spends_alpha": False,
            },
            prefix=_UNCERTAINTY_PREFIX,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_POLICY_PREFIX)
        if self._policy_hash is not None and self._policy_hash != observed:
            raise ValueError("adaptive-rank policy was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("adaptive-rank policy hash does not match")
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def register_v1_fqr_adaptive_rank_policy(
    *,
    policy_id: str,
    route_count: int,
    stratum_count: int,
    response_rows: int,
    device_total_memory_bytes: int,
    device_reserved_memory_bytes: int,
    kkt_legality_block_count: int = 16,
    kkt_rounding_tile_size: int = 1 << 20,
    maximum_hard_action_count: int = 4_000_000,
    D_packet_staging_device_bytes: int = 0,
) -> V1FQRAdaptiveRankPolicy:
    forecasts: list[V1FQRAdaptiveRankMemoryForecast] = []
    routes = _positive_int(route_count, name="route count")
    strata = _positive_int(stratum_count, name="stratum count")
    rows = _positive_int(response_rows, name="response rows")
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    reserved = _nonnegative_int(
        device_reserved_memory_bytes,
        name="device reserved memory",
    )
    blocks = _positive_int(
        kkt_legality_block_count,
        name="KKT legality block count",
    )
    tile = _positive_int(
        kkt_rounding_tile_size,
        name="KKT rounding tile size",
    )
    maximum_action = _positive_int(
        maximum_hard_action_count,
        name="maximum hard action count",
    )
    packet_staging = _nonnegative_int(
        D_packet_staging_device_bytes,
        name="D packet staging device bytes",
    )
    ceiling = (85 * total) // 100
    if ceiling - reserved <= 0:
        refuse_v1_fqr_adaptive_rank(
            V1FQRAdaptiveRankRefusalCode.INITIAL_RANK_EXCEEDS_MEMORY,
            "no FD-rank workspace remains below the frozen 85% HBM ceiling",
            policy_hash=None,
        )
    requested = canonical_fqr_adaptive_rank_ladder_for_route_count(routes)
    for rank in requested:
        base = forecast_stratified_fd_gpu_memory(
            route_count=routes,
            stratum_count=strata,
            response_rows=rows,
            per_stratum_sketch_rank=rank,
            combined_maximum_rank=rank,
        )
        retained_geometry = base.combined_output_state_device_bytes
        kkt_problem = (rank + 5) * routes * 8 + routes + blocks * 4 * 8
        vector_count = V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
        kkt_solver = (
            vector_count * routes * 8
            + (rank + 4 * blocks + vector_count) * 8
        )
        winner_count = min(routes, maximum_action)
        rounding = 12 * (winner_count + min(routes, tile)) * 8
        post_D_peak = retained_geometry + kkt_problem + kkt_solver + rounding
        D_acquisition_peak = base.peak_device_bytes + packet_staging
        registered_peak = max(D_acquisition_peak, post_D_peak)
        available = ceiling - reserved
        forecasts.append(
            V1FQRAdaptiveRankMemoryForecast(
                rank=rank,
                route_count=routes,
                stratum_count=strata,
                response_rows=rows,
                device_total_memory_bytes=total,
                device_reserved_memory_bytes=reserved,
                device_memory_ceiling_bytes=ceiling,
                available_rank_workspace_bytes=available,
                live_stratum_fd_device_bytes=base.live_stratum_fd_device_bytes,
                combined_pre_recompression_factor_device_bytes=(
                    base.combined_pre_recompression_factor_device_bytes
                ),
                combined_output_state_device_bytes=(
                    base.combined_output_state_device_bytes
                ),
                gram_workspace_device_bytes=base.gram_workspace_device_bytes,
                fd_peak_device_bytes=base.peak_device_bytes,
                D_packet_staging_device_bytes=packet_staging,
                D_acquisition_peak_device_bytes=D_acquisition_peak,
                retained_rank_major_geometry_device_bytes=retained_geometry,
                fused_kkt_problem_device_bytes=kkt_problem,
                fused_kkt_solver_workspace_device_bytes=kkt_solver,
                deterministic_rounding_workspace_device_bytes=rounding,
                post_D_kkt_peak_device_bytes=post_D_peak,
                registered_phase_peak_device_bytes=registered_peak,
                kkt_legality_block_count=blocks,
                kkt_rounding_tile_size=tile,
                maximum_hard_action_count=maximum_action,
                phase_liveness_proof_hash=_phase_liveness_proof_hash(
                    route_count=routes,
                    rank=rank,
                    legality_block_count=blocks,
                    rounding_tile_size=tile,
                    maximum_hard_action_count=maximum_action,
                    D_packet_staging_device_bytes=packet_staging,
                ),
                admitted=available > 0 and registered_peak <= available,
            )
        )
    admitted = tuple(value.rank for value in forecasts if value.admitted)
    return V1FQRAdaptiveRankPolicy(
        policy_id=policy_id,
        route_count=routes,
        stratum_count=strata,
        response_rows=rows,
        requested_rank_ladder=requested,
        admitted_rank_ladder=admitted,
        memory_forecasts=tuple(forecasts),
    )


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankBranchDecision:
    branch: str
    ordered_packed_action_hashes: tuple[str, ...]
    admissible_packed_action_hashes: tuple[str, ...]
    selected_packed_action_hash: str | None
    hard_ladder_hash: str
    feasibility_receipt_hash: str
    decision_receipt_hash: str
    fd_residual_envelope_receipt_hash: str
    fd_residual_spectral_upper_bound: float
    fd_residual_diagonal_upper_bounds_hash: str
    winning_face_margin_lower_bound: float
    winning_face_residual_perturbation_upper_bound: float
    hard_fq_feasibility_margin_lower_bound: float
    hard_fq_residual_perturbation_upper_bound: float
    selected_vs_runner_up_margin_lower_bound: float
    selected_vs_runner_up_residual_perturbation_upper_bound: float
    next_larger_infeasible_margin_lower_bound: float
    next_larger_residual_perturbation_upper_bound: float
    selected_competitor_margin_receipt_hash: str
    next_larger_feasibility_boundary_receipt_hash: str
    complete_hard_ladder_replayed: bool = True
    complete_residual_envelope_replayed: bool = True
    outward_numerical_enclosure_used: bool = True
    authoritative_typed_source_derivation_available: bool = False
    authoritative_typed_source_derivation_receipt_hash: str | None = None
    provisional_pre_C: bool = True
    relaxed_face_source_stable: bool = False
    rounded_endpoint_set_source_stable: bool = False
    hard_fq_status_axis_source_stable: bool = False
    stable_branch_identity_hash: str | None = None
    endpoint_status_axis_hash: str | None = None
    source_derivation_ambiguity_codes: tuple[str, ...] = ()
    _source_derivation_issuance_token: InitVar[object | None] = None
    schema: str = FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA

    def __post_init__(self, _source_derivation_issuance_token: object | None) -> None:
        branch = _identity(self.branch, name="KKT residual branch")
        if branch not in _BRANCH_AXIS:
            raise ValueError("adaptive-rank branch is unknown")
        actions = _identity_axis(
            self.ordered_packed_action_hashes,
            name="ordered packed action hashes",
        )
        admissible = tuple(
            _identity(value, name="admissible packed action hash")
            for value in self.admissible_packed_action_hashes
        )
        if len(set(admissible)) != len(admissible) or any(
            value not in actions for value in admissible
        ):
            raise ValueError("admissible action set is not a unique ladder subset")
        if tuple(value for value in actions if value in set(admissible)) != admissible:
            raise ValueError("admissible actions must preserve hard-ladder order")
        selected = self.selected_packed_action_hash
        if selected is not None:
            selected = _identity(selected, name="selected packed action hash")
            if selected not in admissible:
                raise ValueError("selected action is not in the admissible hard ladder")
        for name in (
            "hard_ladder_hash",
            "feasibility_receipt_hash",
            "decision_receipt_hash",
            "fd_residual_envelope_receipt_hash",
            "fd_residual_diagonal_upper_bounds_hash",
            "selected_competitor_margin_receipt_hash",
            "next_larger_feasibility_boundary_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "fd_residual_spectral_upper_bound",
            "winning_face_margin_lower_bound",
            "winning_face_residual_perturbation_upper_bound",
            "hard_fq_feasibility_margin_lower_bound",
            "hard_fq_residual_perturbation_upper_bound",
            "selected_vs_runner_up_margin_lower_bound",
            "selected_vs_runner_up_residual_perturbation_upper_bound",
            "next_larger_infeasible_margin_lower_bound",
            "next_larger_residual_perturbation_upper_bound",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        legacy_flags = (
            self.complete_hard_ladder_replayed,
            self.complete_residual_envelope_replayed,
            self.outward_numerical_enclosure_used,
            self.authoritative_typed_source_derivation_available,
            self.provisional_pre_C,
        )
        source_flags = (
            self.relaxed_face_source_stable,
            self.rounded_endpoint_set_source_stable,
            self.hard_fq_status_axis_source_stable,
        )
        if any(type(value) is not bool for value in (*legacy_flags, *source_flags)):
            raise TypeError("adaptive-rank derivation flags must be exact booleans")
        if legacy_flags[:3] != (True, True, True) or self.provisional_pre_C is not True:
            raise PermissionError("adaptive-rank branch decision crossed pre-C scope")
        ambiguities = tuple(
            _identity(value, name="source derivation ambiguity code")
            for value in self.source_derivation_ambiguity_codes
        )
        if len(set(ambiguities)) != len(ambiguities):
            raise ValueError("source derivation ambiguity codes must be unique")
        source_hash = self.authoritative_typed_source_derivation_receipt_hash
        stable_identity = self.stable_branch_identity_hash
        endpoint_axis = self.endpoint_status_axis_hash
        if self.schema == FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA:
            if legacy_flags[3] is not False or source_hash is not None:
                raise PermissionError(
                    "an arbitrary derivation hash cannot impersonate a typed source builder"
                )
            if any(source_flags) or stable_identity is not None or endpoint_axis is not None:
                raise PermissionError("legacy branch decision gained source authority")
            if ambiguities:
                raise ValueError("legacy branch decisions cannot name v2 ambiguity codes")
        elif self.schema == FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA:
            if (
                _source_derivation_issuance_token is not _SOURCE_DERIVATION_ISSUANCE_TOKEN
            ):
                raise PermissionError(
                    "source-derived branch authority requires the registered builder"
                )
            if legacy_flags[3] is not True:
                raise PermissionError("v2 branch decision lacks typed source derivation")
            source_hash = _identity(source_hash, name="source derivation receipt hash")
            endpoint_axis = _identity(endpoint_axis, name="endpoint status axis hash")
            if all(source_flags):
                stable_identity = _identity(
                    stable_identity,
                    name="stable branch identity hash",
                )
                if ambiguities:
                    raise ValueError("stable source-derived branch retained ambiguity")
            elif stable_identity is not None:
                raise ValueError("ambiguous source-derived branch gained a stable identity")
            elif not ambiguities:
                raise ValueError("ambiguous source-derived branch lacks a typed reason")
        else:
            raise ValueError("adaptive-rank branch decision schema is invalid")
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "ordered_packed_action_hashes", actions)
        object.__setattr__(self, "admissible_packed_action_hashes", admissible)
        object.__setattr__(self, "selected_packed_action_hash", selected)
        object.__setattr__(
            self,
            "authoritative_typed_source_derivation_receipt_hash",
            source_hash,
        )
        object.__setattr__(self, "stable_branch_identity_hash", stable_identity)
        object.__setattr__(self, "endpoint_status_axis_hash", endpoint_axis)
        object.__setattr__(self, "source_derivation_ambiguity_codes", ambiguities)

    @property
    def decision_hash(self) -> str:
        payload: dict[str, object] = {
            "schema": self.schema,
            "branch": self.branch,
            "ordered_packed_action_hashes": self.ordered_packed_action_hashes,
            "admissible_packed_action_hashes": self.admissible_packed_action_hashes,
            "selected_packed_action_hash": self.selected_packed_action_hash,
            "hard_ladder_hash": self.hard_ladder_hash,
            "feasibility_receipt_hash": self.feasibility_receipt_hash,
            "decision_receipt_hash": self.decision_receipt_hash,
            "fd_residual_envelope_receipt_hash": (
                self.fd_residual_envelope_receipt_hash
            ),
            "fd_residual_spectral_upper_bound": (
                self.fd_residual_spectral_upper_bound
            ),
            "fd_residual_diagonal_upper_bounds_hash": (
                self.fd_residual_diagonal_upper_bounds_hash
            ),
            "winning_face_margin_lower_bound": self.winning_face_margin_lower_bound,
            "winning_face_residual_perturbation_upper_bound": (
                self.winning_face_residual_perturbation_upper_bound
            ),
            "hard_fq_feasibility_margin_lower_bound": (
                self.hard_fq_feasibility_margin_lower_bound
            ),
            "hard_fq_residual_perturbation_upper_bound": (
                self.hard_fq_residual_perturbation_upper_bound
            ),
            "selected_vs_runner_up_margin_lower_bound": (
                self.selected_vs_runner_up_margin_lower_bound
            ),
            "selected_vs_runner_up_residual_perturbation_upper_bound": (
                self.selected_vs_runner_up_residual_perturbation_upper_bound
            ),
            "next_larger_infeasible_margin_lower_bound": (
                self.next_larger_infeasible_margin_lower_bound
            ),
            "next_larger_residual_perturbation_upper_bound": (
                self.next_larger_residual_perturbation_upper_bound
            ),
            "selected_competitor_margin_receipt_hash": (
                self.selected_competitor_margin_receipt_hash
            ),
            "next_larger_feasibility_boundary_receipt_hash": (
                self.next_larger_feasibility_boundary_receipt_hash
            ),
            "complete_hard_ladder_replayed": self.complete_hard_ladder_replayed,
            "complete_residual_envelope_replayed": (
                self.complete_residual_envelope_replayed
            ),
            "outward_numerical_enclosure_used": (
                self.outward_numerical_enclosure_used
            ),
            "authoritative_typed_source_derivation_available": (
                self.authoritative_typed_source_derivation_available
            ),
            "authoritative_typed_source_derivation_receipt_hash": (
                self.authoritative_typed_source_derivation_receipt_hash
            ),
            "provisional_pre_C": self.provisional_pre_C,
        }
        if self.schema == FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA:
            payload.update(
                {
                    "relaxed_face_source_stable": self.relaxed_face_source_stable,
                    "rounded_endpoint_set_source_stable": (
                        self.rounded_endpoint_set_source_stable
                    ),
                    "hard_fq_status_axis_source_stable": (
                        self.hard_fq_status_axis_source_stable
                    ),
                    "stable_branch_identity_hash": self.stable_branch_identity_hash,
                    "endpoint_status_axis_hash": self.endpoint_status_axis_hash,
                    "source_derivation_ambiguity_codes": (
                        self.source_derivation_ambiguity_codes
                    ),
                }
            )
        return content_hash(payload, prefix=_BRANCH_PREFIX)

    @property
    def winning_face_robust(self) -> bool:
        return (
            self.winning_face_margin_lower_bound
            > self.winning_face_residual_perturbation_upper_bound
        )

    @property
    def hard_fq_feasibility_robust(self) -> bool:
        return (
            self.hard_fq_feasibility_margin_lower_bound
            > self.hard_fq_residual_perturbation_upper_bound
        )

    @property
    def next_irreversible_decision_robust(self) -> bool:
        return (
            bool(self.admissible_packed_action_hashes)
            and self.selected_packed_action_hash
            == self.admissible_packed_action_hashes[-1]
            and self.selected_vs_runner_up_margin_lower_bound
            > self.selected_vs_runner_up_residual_perturbation_upper_bound
            and self.next_larger_infeasible_margin_lower_bound
            > self.next_larger_residual_perturbation_upper_bound
        )

    @property
    def rank_local_residual_decision_stable(self) -> bool:
        if self.schema == FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA:
            # Caller-supplied v1 scalars are proposal-only diagnostics.
            return False
        return (
            self.authoritative_typed_source_derivation_available
            and self.authoritative_typed_source_derivation_receipt_hash is not None
            and self.relaxed_face_source_stable
            and self.rounded_endpoint_set_source_stable
            and self.hard_fq_status_axis_source_stable
            and self.stable_branch_identity_hash is not None
            and not self.source_derivation_ambiguity_codes
        )

    @property
    def provisional_margin_clear(self) -> bool:
        return (
            self.winning_face_robust
            and self.hard_fq_feasibility_robust
            and self.next_irreversible_decision_robust
        )


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankDecisionCertificate:
    policy_hash: str
    rank: int
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    branch_decisions: tuple[V1FQRAdaptiveRankBranchDecision, ...]
    rank_local_residual_decision_stable: bool
    robust_next_decision_hash: str | None
    C_stream_open_authority: bool
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    robust_branch_family_hash: str | None = None
    source_pre_c_branch_family_hash: str | None = None
    schema: str = FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in ("policy_hash", "current_mask_hash", "route_axis_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _positive_int(self.rank, name="FD rank")
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        branches = tuple(self.branch_decisions)
        if (
            any(type(value) is not V1FQRAdaptiveRankBranchDecision for value in branches)
            or tuple(value.branch for value in branches) != _BRANCH_AXIS
        ):
            refuse_v1_fqr_adaptive_rank(
                V1FQRAdaptiveRankRefusalCode.INCOMPLETE_HARD_LADDER,
                "rank-local authority requires exact spectral and diagonal-l1 "
                "residual-branch witnesses",
                policy_hash=self.policy_hash,
                current_mask_hash=self.current_mask_hash,
                proposal_generation=generation,
            )
        branch_schemas = tuple(value.schema for value in branches)
        source_family = self.source_pre_c_branch_family_hash
        if self.schema == FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_SCHEMA:
            if branch_schemas != (FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA,) * 2:
                raise ValueError("legacy rank decision requires legacy branches")
            decisions = {
                value.selected_packed_action_hash
                for value in branches
                if value.rank_local_residual_decision_stable
            }
            derived_stable = (
                all(value.rank_local_residual_decision_stable for value in branches)
                and len(decisions) == 1
                and None not in decisions
            )
            expected_family = None
            expected_robust = next(iter(decisions)) if derived_stable else None
            if source_family is not None:
                raise ValueError("legacy rank decision gained a pre-C family hash")
        elif self.schema == FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA:
            if branch_schemas != (FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA,) * 2:
                raise ValueError("v2 rank decision requires two v2 source branches")
            source_family = _identity(
                source_family,
                name="source pre-C branch family hash",
            )
            derived_stable = all(
                value.rank_local_residual_decision_stable for value in branches
            )
            expected_family = (
                content_hash(
                    {
                        "schema": (
                            "partition_free_v1_fqr_adaptive_rank_robust_"
                            "two_branch_family_v2"
                        ),
                        "branch_axis": _BRANCH_AXIS,
                        "stable_branch_identity_hashes": tuple(
                            value.stable_branch_identity_hash for value in branches
                        ),
                        "ordered_action_axes": tuple(
                            value.ordered_packed_action_hashes for value in branches
                        ),
                        "endpoint_status_axis_hashes": tuple(
                            value.endpoint_status_axis_hash for value in branches
                        ),
                        "source_derivation_receipt_hashes": tuple(
                            value.authoritative_typed_source_derivation_receipt_hash
                            for value in branches
                        ),
                        "source_pre_c_branch_family_hash": source_family,
                    },
                    prefix=_DECISION_FAMILY_PREFIX,
                )
                if derived_stable
                else None
            )
            # The v2 object freezes a two-branch family.  It deliberately does
            # not force both convex branches to nominate the same action.
            expected_robust = expected_family
        else:
            raise ValueError("rank-local decision certificate schema is invalid")
        if type(self.rank_local_residual_decision_stable) is not bool:
            raise TypeError("rank-local residual stability must be an exact boolean")
        if self.rank_local_residual_decision_stable is not derived_stable:
            raise ValueError("rank-local residual stability does not replay its margins")
        robust = self.robust_next_decision_hash
        if robust is not None:
            robust = _identity(robust, name="robust next decision hash")
        if robust != expected_robust:
            raise ValueError("robust next decision family does not replay its branches")
        supplied_family = self.robust_branch_family_hash
        if supplied_family is not None:
            supplied_family = _identity(
                supplied_family,
                name="robust branch family hash",
            )
        if supplied_family != expected_family:
            raise ValueError("robust branch family hash does not replay its branches")
        flags = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("rank-local decision authority flags must be exact booleans")
        if flags != (derived_stable, False, False, False, False):
            raise PermissionError("rank-local decision certificate crossed its boundary")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "branch_decisions", branches)
        object.__setattr__(self, "robust_next_decision_hash", robust)
        object.__setattr__(self, "robust_branch_family_hash", supplied_family)
        object.__setattr__(self, "source_pre_c_branch_family_hash", source_family)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "rank": self.rank,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "branch_decision_hashes": tuple(
                value.decision_hash for value in self.branch_decisions
            ),
            "rank_local_residual_decision_stable": (
                self.rank_local_residual_decision_stable
            ),
            "robust_next_decision_hash": self.robust_next_decision_hash,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }
        if self.schema == FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA:
            payload["robust_branch_family_hash"] = self.robust_branch_family_hash
            payload["source_pre_c_branch_family_hash"] = (
                self.source_pre_c_branch_family_hash
            )
        return payload

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RANK_DECISION_PREFIX)
        if self._certificate_hash is not None and self._certificate_hash != observed:
            raise ValueError("rank-local decision certificate was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("rank-local decision certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankReplayRecord:
    policy_hash: str
    rank: int
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    cumulative_stratum_counts: tuple[int, ...]
    D_context_axis_hash: str
    D_draw_axis_hash: str
    D_source_measurement_axis_hash: str
    D_weight_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    sampled_geometry_source_hash: str
    kkt_problem_hash: str
    kkt_result_hash: str
    provisional_branch_family_hash: str
    decision_certificate: V1FQRAdaptiveRankDecisionCertificate
    replayed_from_registered_raw_D_source: bool = True
    lower_rank_snapshot_restored: bool = False
    zero_padding_used: bool = False
    D_population_or_order_changed: bool = False
    D_outcomes_extended: bool = False
    C_outcomes_seen: bool = False
    provisional_pre_C_only: bool = True
    schema: str = FQR_ADAPTIVE_RANK_REPLAY_SCHEMA
    _record_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "current_mask_hash",
            "route_axis_hash",
            "D_context_axis_hash",
            "D_draw_axis_hash",
            "D_source_measurement_axis_hash",
            "D_weight_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "sampled_geometry_source_hash",
            "kkt_problem_hash",
            "kkt_result_hash",
            "provisional_branch_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "rank", _positive_int(self.rank, name="FD rank"))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        counts = tuple(
            _positive_int(value, name="D cumulative stratum count")
            for value in self.cumulative_stratum_counts
        )
        if len(counts) != 2:
            raise ValueError("canonical adaptive rank requires both D strata")
        certificate = self.decision_certificate
        if type(certificate) is not V1FQRAdaptiveRankDecisionCertificate:
            raise TypeError("rank replay requires its exact decision certificate")
        certificate.assert_integrity()
        if (
            certificate.policy_hash != self.policy_hash
            or certificate.rank != self.rank
            or certificate.current_mask_hash != self.current_mask_hash
            or certificate.route_axis_hash != self.route_axis_hash
            or certificate.route_count != self.route_count
            or certificate.proposal_generation != self.proposal_generation
        ):
            raise ValueError("rank replay decision certificate has stale coordinates")
        if (
            certificate.schema == FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
            and certificate.source_pre_c_branch_family_hash
            != self.provisional_branch_family_hash
        ):
            raise ValueError("rank replay decision certificate names another pre-C family")
        flags = (
            self.replayed_from_registered_raw_D_source,
            self.lower_rank_snapshot_restored,
            self.zero_padding_used,
            self.D_population_or_order_changed,
            self.D_outcomes_extended,
            self.C_outcomes_seen,
            self.provisional_pre_C_only,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive-rank replay flags must be exact booleans")
        if flags != (True, False, False, False, False, False, True):
            code = (
                V1FQRAdaptiveRankRefusalCode.ZERO_PADDING
                if self.zero_padding_used or self.lower_rank_snapshot_restored
                else V1FQRAdaptiveRankRefusalCode.C_OPENED_BEFORE_STABILITY
                if self.C_outcomes_seen
                else V1FQRAdaptiveRankRefusalCode.NON_RAW_REPLAY
            )
            refuse_v1_fqr_adaptive_rank(
                code,
                "rank replay did not reexecute the unchanged registered raw D prefix",
                policy_hash=self.policy_hash,
                current_mask_hash=self.current_mask_hash,
                proposal_generation=self.proposal_generation,
            )
        if self.schema != FQR_ADAPTIVE_RANK_REPLAY_SCHEMA:
            raise ValueError("adaptive-rank replay schema is invalid")
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "rank": self.rank,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "cumulative_stratum_counts": self.cumulative_stratum_counts,
            "D_context_axis_hash": self.D_context_axis_hash,
            "D_draw_axis_hash": self.D_draw_axis_hash,
            "D_source_measurement_axis_hash": self.D_source_measurement_axis_hash,
            "D_weight_hash": self.D_weight_hash,
            "fd_design_hash": self.fd_design_hash,
            "fd_snapshot_hash": self.fd_snapshot_hash,
            "sampled_geometry_source_hash": self.sampled_geometry_source_hash,
            "kkt_problem_hash": self.kkt_problem_hash,
            "kkt_result_hash": self.kkt_result_hash,
            "provisional_branch_family_hash": self.provisional_branch_family_hash,
            "decision_certificate_hash": self.decision_certificate.certificate_hash,
            "replayed_from_registered_raw_D_source": (
                self.replayed_from_registered_raw_D_source
            ),
            "lower_rank_snapshot_restored": self.lower_rank_snapshot_restored,
            "zero_padding_used": self.zero_padding_used,
            "D_population_or_order_changed": self.D_population_or_order_changed,
            "D_outcomes_extended": self.D_outcomes_extended,
            "C_outcomes_seen": self.C_outcomes_seen,
            "provisional_pre_C_only": self.provisional_pre_C_only,
        }

    @property
    def record_hash(self) -> str:
        cached = self._record_hash
        return self.assert_integrity() if cached is None else cached

    @property
    def physical_decision_family_hash(self) -> str:
        return content_hash(
            {
                "rank_local_decision_certificate_hash": (
                    self.decision_certificate.certificate_hash
                ),
                "robust_next_decision_hash": (
                    self.decision_certificate.robust_next_decision_hash
                ),
            },
            prefix=_DECISION_FAMILY_PREFIX,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_REPLAY_PREFIX)
        if self._record_hash is not None and self._record_hash != observed:
            raise ValueError("adaptive-rank replay record was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("adaptive-rank replay record hash does not match")
        object.__setattr__(self, "_record_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankReplayLedger:
    policy: V1FQRAdaptiveRankPolicy
    records: tuple[V1FQRAdaptiveRankReplayRecord, ...]
    executed_contiguous_rank_prefix: bool = True
    stopped_at_first_robust_rank: bool = True
    C_outcomes_seen: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_LEDGER_SCHEMA
    _ledger_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.policy) is not V1FQRAdaptiveRankPolicy:
            raise TypeError("adaptive-rank ledger requires its exact policy")
        policy_hash = self.policy.assert_integrity()
        records = tuple(self.records)
        if any(type(value) is not V1FQRAdaptiveRankReplayRecord for value in records):
            raise TypeError("adaptive-rank ledger requires exact replay records")
        observed_ranks = tuple(value.rank for value in records)
        expected_ranks = self.policy.admitted_rank_ladder[: len(records)]
        if not records or observed_ranks != expected_ranks:
            refuse_v1_fqr_adaptive_rank(
                V1FQRAdaptiveRankRefusalCode.INCOMPLETE_RANK_REPLAY,
                "executed FD ranks must be one nonempty contiguous prefix of "
                "the frozen memory-admitted ladder beginning at rank 8",
                policy_hash=policy_hash,
            )
        if any(value.policy_hash != policy_hash for value in records):
            raise ValueError("adaptive-rank replay record names another policy")
        shared = {
            (
                value.current_mask_hash,
                value.route_axis_hash,
                value.route_count,
                value.proposal_generation,
                value.cumulative_stratum_counts,
                value.D_context_axis_hash,
                value.D_draw_axis_hash,
                value.D_source_measurement_axis_hash,
                value.D_weight_hash,
            )
            for value in records
        }
        if len(shared) != 1:
            refuse_v1_fqr_adaptive_rank(
                V1FQRAdaptiveRankRefusalCode.D_PREFIX_DRIFT,
                "adaptive ranks did not replay one identical sealed D prefix",
                policy_hash=policy_hash,
            )
        if any(
            value.decision_certificate.rank_local_residual_decision_stable
            for value in records[:-1]
        ):
            raise ValueError(
                "adaptive rank replay continued after the first robust decision"
            )
        flags = (
            self.executed_contiguous_rank_prefix,
            self.stopped_at_first_robust_rank,
            self.C_outcomes_seen,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive-rank ledger flags must be exact booleans")
        if flags != (True, True, False, False, False, False):
            raise PermissionError("adaptive-rank replay ledger crossed the pre-C boundary")
        if self.schema != FQR_ADAPTIVE_RANK_LEDGER_SCHEMA:
            raise ValueError("adaptive-rank ledger schema is invalid")
        object.__setattr__(self, "records", records)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy.policy_hash,
            "record_hashes": tuple(value.record_hash for value in self.records),
            "executed_contiguous_rank_prefix": self.executed_contiguous_rank_prefix,
            "stopped_at_first_robust_rank": self.stopped_at_first_robust_rank,
            "C_outcomes_seen": self.C_outcomes_seen,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def next_rank(self) -> int | None:
        index = len(self.records)
        return (
            None
            if index >= len(self.policy.admitted_rank_ladder)
            else self.policy.admitted_rank_ladder[index]
        )

    @property
    def ledger_hash(self) -> str:
        cached = self._ledger_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_LEDGER_PREFIX)
        if self._ledger_hash is not None and self._ledger_hash != observed:
            raise ValueError("adaptive-rank replay ledger was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("adaptive-rank replay ledger hash does not match")
        object.__setattr__(self, "_ledger_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankStabilityCertificate:
    policy_hash: str
    uncertainty_set_hash: str
    replay_ledger_hash: str
    admitted_rank_ladder: tuple[int, ...]
    executed_rank_prefix: tuple[int, ...]
    replay_record_hashes: tuple[str, ...]
    rank_local_decision_certificate_hash: str
    robust_next_decision_hash: str
    frozen_rank: int
    frozen_replay_record_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    _issuance_token: InitVar[object]
    robust_branch_family_hash: str | None = None
    source_pre_c_branch_family_hash: str | None = None
    rank_local_fd_residual_bound_proof: bool = True
    complete_two_branch_hard_ladders_replayed: bool = True
    stopped_at_first_robust_rank: bool = True
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_STABILITY_SCHEMA
    _certificate_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _STABILITY_ISSUANCE_TOKEN:
            raise PermissionError(
                "adaptive-rank C authority requires the registered stability "
                "issuer, not direct dataclass construction"
            )
        for name in (
            "policy_hash",
            "uncertainty_set_hash",
            "replay_ledger_hash",
            "rank_local_decision_certificate_hash",
            "robust_next_decision_hash",
            "frozen_replay_record_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ranks = tuple(
            _positive_int(value, name="admitted FD rank")
            for value in self.admitted_rank_ladder
        )
        executed = tuple(
            _positive_int(value, name="executed FD rank")
            for value in self.executed_rank_prefix
        )
        hashes = tuple(
            _identity(value, name="rank replay record hash")
            for value in self.replay_record_hashes
        )
        if (
            not ranks
            or not executed
            or executed != ranks[: len(executed)]
            or len(hashes) != len(executed)
        ):
            raise ValueError("adaptive-rank stability certificate has an incomplete rank axis")
        frozen = _positive_int(self.frozen_rank, name="frozen FD rank")
        if frozen != executed[-1] or self.frozen_replay_record_hash != hashes[-1]:
            raise ValueError("canonical adaptive rank freezes the first robust replay")
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        flags = (
            self.rank_local_fd_residual_bound_proof,
            self.complete_two_branch_hard_ladders_replayed,
            self.stopped_at_first_robust_rank,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive-rank stability flags must be exact booleans")
        if flags != (True, True, True, True, False, False, False, False):
            raise PermissionError("adaptive-rank stability crossed its authority boundary")
        family = self.robust_branch_family_hash
        source_family = self.source_pre_c_branch_family_hash
        if self.schema == FQR_ADAPTIVE_RANK_STABILITY_SCHEMA:
            if family is not None:
                raise ValueError("legacy adaptive-rank stability gained a family hash")
            if source_family is not None:
                raise ValueError("legacy stability gained a pre-C family hash")
        elif self.schema == FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA:
            family = _identity(family, name="robust branch family hash")
            source_family = _identity(
                source_family,
                name="source pre-C branch family hash",
            )
            if family != self.robust_next_decision_hash:
                raise ValueError("v2 robust decision must be its two-branch family")
        else:
            raise ValueError("adaptive-rank stability schema is invalid")
        object.__setattr__(self, "admitted_rank_ladder", ranks)
        object.__setattr__(self, "executed_rank_prefix", executed)
        object.__setattr__(self, "replay_record_hashes", hashes)
        object.__setattr__(self, "frozen_rank", frozen)
        object.__setattr__(self, "robust_branch_family_hash", family)
        object.__setattr__(self, "source_pre_c_branch_family_hash", source_family)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "policy_hash",
                "uncertainty_set_hash",
                "replay_ledger_hash",
                "admitted_rank_ladder",
                "executed_rank_prefix",
                "replay_record_hashes",
                "rank_local_decision_certificate_hash",
                "robust_next_decision_hash",
                "frozen_rank",
                "frozen_replay_record_hash",
                "current_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "rank_local_fd_residual_bound_proof",
                "complete_two_branch_hard_ladders_replayed",
                "stopped_at_first_robust_rank",
                "C_stream_open_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        }
        if self.schema == FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA:
            payload["robust_branch_family_hash"] = self.robust_branch_family_hash
            payload["source_pre_c_branch_family_hash"] = (
                self.source_pre_c_branch_family_hash
            )
        return payload

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_STABILITY_PREFIX)
        if self._certificate_hash is not None and self._certificate_hash != observed:
            raise ValueError("adaptive-rank stability certificate was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("adaptive-rank stability certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


def _rank_instability_detail(
    *,
    decision: V1FQRAdaptiveRankDecisionCertificate,
    next_rank: int | None,
) -> str:
    branch_details: list[str] = []
    for branch in decision.branch_decisions:
        codes = tuple(branch.source_derivation_ambiguity_codes)
        if codes:
            branch_details.append(f"{branch.branch}={','.join(codes)}")
        elif not branch.authoritative_typed_source_derivation_available:
            branch_details.append(f"{branch.branch}=typed_source_derivation_missing")
        elif not branch.hard_fq_status_axis_source_stable:
            branch_details.append(f"{branch.branch}=hard_fq_status_axis_unstable")
        elif not branch.relaxed_face_source_stable:
            branch_details.append(f"{branch.branch}=relaxed_face_unstable")
        elif not branch.rounded_endpoint_set_source_stable:
            branch_details.append(f"{branch.branch}=rounded_endpoint_set_unstable")
        else:
            branch_details.append(f"{branch.branch}=unknown_residual_ambiguity")
    branch_suffix = (
        "; unresolved branch ambiguity codes: " + "; ".join(branch_details)
        if branch_details
        else "; unresolved branch ambiguity codes: unavailable"
    )
    if next_rank is not None:
        return (
            f"replay the identical sealed raw D prefix at rank {next_rank}"
            f"{branch_suffix}"
        )
    return (
        "maximum memory-admitted FD rank retains residual decision ambiguity"
        f"{branch_suffix}"
    )


def certify_v1_fqr_adaptive_rank_stability(
    *,
    policy: V1FQRAdaptiveRankPolicy,
    ledger: V1FQRAdaptiveRankReplayLedger,
) -> V1FQRAdaptiveRankStabilityCertificate:
    if type(policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("adaptive-rank certification requires the exact policy")
    if type(ledger) is not V1FQRAdaptiveRankReplayLedger:
        raise TypeError("adaptive-rank certification requires the exact replay ledger")
    policy_hash = policy.assert_integrity()
    ledger_hash = ledger.assert_integrity()
    if ledger.policy.policy_hash != policy_hash:
        refuse_v1_fqr_adaptive_rank(
            V1FQRAdaptiveRankRefusalCode.STALE_CERTIFICATE,
            "rank replay ledger belongs to another policy",
            policy_hash=policy_hash,
            ledger_hash=ledger_hash,
        )
    frozen = ledger.records[-1]
    decision = frozen.decision_certificate
    if not decision.rank_local_residual_decision_stable:
        code = (
            V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED
            if ledger.next_rank is not None
            else V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
        )
        detail = _rank_instability_detail(
            decision=decision,
            next_rank=ledger.next_rank,
        )
        refuse_v1_fqr_adaptive_rank(
            code,
            detail,
            policy_hash=policy_hash,
            ledger_hash=ledger_hash,
            current_mask_hash=frozen.current_mask_hash,
            proposal_generation=frozen.proposal_generation,
        )
    if decision.robust_next_decision_hash is None:
        raise ValueError("stable rank-local witness lacks its robust next decision")
    return V1FQRAdaptiveRankStabilityCertificate(
        policy_hash=policy_hash,
        uncertainty_set_hash=policy.uncertainty_set_hash,
        replay_ledger_hash=ledger_hash,
        admitted_rank_ladder=policy.admitted_rank_ladder,
        executed_rank_prefix=tuple(value.rank for value in ledger.records),
        replay_record_hashes=tuple(value.record_hash for value in ledger.records),
        rank_local_decision_certificate_hash=decision.certificate_hash,
        robust_next_decision_hash=decision.robust_next_decision_hash,
        frozen_rank=frozen.rank,
        frozen_replay_record_hash=frozen.record_hash,
        current_mask_hash=frozen.current_mask_hash,
        route_axis_hash=frozen.route_axis_hash,
        route_count=frozen.route_count,
        proposal_generation=frozen.proposal_generation,
        _issuance_token=_STABILITY_ISSUANCE_TOKEN,
        robust_branch_family_hash=decision.robust_branch_family_hash,
        source_pre_c_branch_family_hash=(
            decision.source_pre_c_branch_family_hash
        ),
        schema=(
            FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA
            if decision.schema == FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
            else FQR_ADAPTIVE_RANK_STABILITY_SCHEMA
        ),
    )


# ---------------------------------------------------------------------------
# Endogenous rank v2
# ---------------------------------------------------------------------------
#
# These types deliberately do not reinterpret the v1 `(8, 16, 32)` replay
# surface above.  V1 remains byte-for-byte readable and keeps its historical
# raw-D rank reexecution semantics.  V2 makes the dyadic rank lattice an
# execution approximation only: physical liveness filters it before D, while
# a registered D prefix determines which admitted ranks are identifiable.
# No rank is selected before D.


class V1FQREndogenousRankNoCCode(str, Enum):
    """Typed reasons that the v2 rank seam cannot lawfully open C."""

    PHYSICAL_LIVENESS_EXHAUSTED = "physical_liveness_exhausted"
    D_IDENTIFIABILITY_INCOMPLETE = "D_identifiability_incomplete"
    D_SAMPLING_AMBIGUOUS = "D_sampling_ambiguous"
    FD_REFINEMENT_AVAILABLE = "FD_refinement_available"
    KKT_NUMERICAL_AMBIGUOUS = "KKT_numerical_ambiguous"
    MAXIMUM_ADMITTED_RANK_AMBIGUOUS = "maximum_admitted_rank_ambiguous"
    SOURCE_DECISION_IMAGE_INCOMPLETE = "source_decision_image_incomplete"
    SAME_VJP_FANOUT_INVALID = "same_VJP_fanout_invalid"


class V1FQREndogenousRankNextAction(str, Enum):
    """Only computations allowed to resolve a v2 pre-C rank refusal."""

    ACQUIRE_MORE_D_FOR_IDENTIFIABILITY = (
        "acquire_more_D_for_identifiability"
    )
    ACQUIRE_MORE_D_FOR_SAMPLING = "acquire_more_D_for_sampling"
    REFINE_RANK_FROM_SAME_VJP_FANOUT = "refine_rank_from_same_VJP_fanout"
    CONTINUE_REGISTERED_KKT_SOLVER = "continue_registered_KKT_solver"
    PHYSICAL_NO_C_REFUSAL = "physical_no_C_refusal"


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankNoCRefusalV2:
    """Array-free refusal; unresolved or exhausted rank evidence never opens C."""

    code: V1FQREndogenousRankNoCCode
    reason: str
    required_action: V1FQREndogenousRankNextAction
    policy_hash: str | None = None
    registered_d_prefix_hash: str | None = None
    same_vjp_fanout_hash: str | None = None
    current_rank: int | None = None
    next_rank: int | None = None
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FQR_ENDOGENOUS_RANK_NO_C_REFUSAL_V2_SCHEMA

    def __post_init__(self) -> None:
        if type(self.code) is not V1FQREndogenousRankNoCCode:
            raise TypeError("endogenous-rank refusal requires an exact code")
        if type(self.required_action) is not V1FQREndogenousRankNextAction:
            raise TypeError("endogenous-rank refusal requires an exact next action")
        object.__setattr__(self, "reason", _identity(self.reason, name="reason"))
        for name in (
            "policy_hash",
            "registered_d_prefix_hash",
            "same_vjp_fanout_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        for name in ("current_rank", "next_rank"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_int(value, name=name))
        flags = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("endogenous-rank refusal flags must be exact booleans")
        if any(flags):
            raise PermissionError("an endogenous-rank refusal cannot open C")
        if self.schema != FQR_ENDOGENOUS_RANK_NO_C_REFUSAL_V2_SCHEMA:
            raise ValueError("endogenous-rank refusal schema is invalid")

    @property
    def refusal_hash(self) -> str:
        return content_hash(self, prefix=_ENDOGENOUS_REFUSAL_V2_PREFIX)


def canonical_fqr_endogenous_rank_lattice_for_route_count(
    route_count: int,
) -> tuple[int, ...]:
    """Return the route-clipped natural dyadic execution lattice.

    The lattice is not a scientific hyperparameter and carries no selection
    authority.  Powers of two permit nested rank-local FD state allocation;
    a non-dyadic terminal route dimension is included only to make the finite
    axis representable before physical liveness filtering.
    """

    routes = _positive_int(route_count, name="route count")
    ranks: list[int] = []
    rank = 1
    while rank < routes:
        ranks.append(rank)
        rank *= 2
    ranks.append(routes)
    return tuple(ranks)


class _EndogenousRankLivenessComponentsV2(TypedDict):
    rank_prefix: tuple[int, ...]
    live_stratum_fd_device_bytes: int
    combined_pre_recompression_factor_device_bytes: int
    combined_output_state_device_bytes: int
    gram_workspace_device_bytes: int
    fd_peak_device_bytes: int
    D_packet_staging_device_bytes: int
    cumulative_live_stratum_fd_device_bytes: int
    shared_fd_transient_workspace_device_bytes: int
    D_acquisition_peak_device_bytes: int
    retained_rank_major_geometry_device_bytes: int
    cumulative_retained_rank_major_geometry_device_bytes: int
    fused_kkt_problem_device_bytes: int
    fused_kkt_solver_workspace_device_bytes: int
    deterministic_rounding_workspace_device_bytes: int
    shared_rank_local_kkt_execution_device_bytes: int
    post_D_kkt_peak_device_bytes: int
    registered_phase_peak_device_bytes: int
    physical_workspace_ceiling_bytes: int


def _endogenous_rank_liveness_components_v2(
    *,
    route_count: int,
    rank: int,
    stratum_count: int,
    response_rows: int,
    physical_workspace_ceiling_bytes: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
    D_packet_staging_device_bytes: int,
) -> _EndogenousRankLivenessComponentsV2:
    """Replay the simultaneous-prefix lifetime for one execution rank.

    A capture/VJP packet is consumed once and serially fanned into independent
    FD states for every rank in the prefix.  Those rank-local states are
    persistent through acquisition, so their bytes add.  Recompression and
    KKT/rounding execute one rank at a time, so their transient workspace is
    the prefix maximum, not the prefix sum.  Final rank-major geometries remain
    resident until the registered rank-local KKT family has been evaluated and
    therefore also add across the prefix.

    This is an execution-liveness statement only.  It cannot select a rank or
    alter which source decision is scientifically admissible.
    """

    routes = _positive_int(route_count, name="route count")
    target_rank = _positive_int(rank, name="FD rank")
    strata = _positive_int(stratum_count, name="stratum count")
    rows = _positive_int(response_rows, name="response rows")
    ceiling = _positive_int(
        physical_workspace_ceiling_bytes,
        name="physical workspace ceiling bytes",
    )
    blocks = _positive_int(
        legality_block_count,
        name="KKT legality block count",
    )
    tile = _positive_int(rounding_tile_size, name="KKT rounding tile size")
    maximum_action = _positive_int(
        maximum_hard_action_count,
        name="maximum hard action count",
    )
    staging = _nonnegative_int(
        D_packet_staging_device_bytes,
        name="D packet staging device bytes",
    )
    lattice = canonical_fqr_endogenous_rank_lattice_for_route_count(routes)
    if target_rank not in lattice:
        raise ValueError("FD rank must belong to the endogenous execution lattice")
    rank_prefix = lattice[: lattice.index(target_rank) + 1]
    winner_count = min(routes, maximum_action)

    per_rank: list[dict[str, int]] = []
    for prefix_rank in rank_prefix:
        base = forecast_stratified_fd_gpu_memory(
            route_count=routes,
            stratum_count=strata,
            response_rows=rows,
            per_stratum_sketch_rank=prefix_rank,
            combined_maximum_rank=prefix_rank,
        )
        kkt_problem = (prefix_rank + 5) * routes * 8 + routes + blocks * 4 * 8
        kkt_solver = (
            V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT * routes * 8
            + (
                prefix_rank
                + 4 * blocks
                + V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
            )
            * 8
        )
        rounding = 12 * (winner_count + min(routes, tile)) * 8
        per_rank.append(
            {
                "rank": prefix_rank,
                "live": base.live_stratum_fd_device_bytes,
                "pre": base.combined_pre_recompression_factor_device_bytes,
                "output": base.combined_output_state_device_bytes,
                "gram": base.gram_workspace_device_bytes,
                "fd_peak": base.peak_device_bytes,
                "fd_transient": (
                    base.peak_device_bytes
                    - base.live_stratum_fd_device_bytes
                ),
                "kkt_problem": kkt_problem,
                "kkt_solver": kkt_solver,
                "rounding": rounding,
                "kkt_execution": kkt_problem + kkt_solver + rounding,
            }
        )

    current = per_rank[-1]
    cumulative_live = sum(value["live"] for value in per_rank)
    shared_fd_transient = max(value["fd_transient"] for value in per_rank)
    cumulative_retained = sum(value["output"] for value in per_rank)
    shared_kkt_execution = max(value["kkt_execution"] for value in per_rank)
    D_peak = cumulative_live + shared_fd_transient + staging
    post_D_peak = cumulative_retained + shared_kkt_execution
    phase_peak = max(D_peak, post_D_peak)
    return {
        "rank_prefix": rank_prefix,
        "live_stratum_fd_device_bytes": current["live"],
        "combined_pre_recompression_factor_device_bytes": current["pre"],
        "combined_output_state_device_bytes": current["output"],
        "gram_workspace_device_bytes": current["gram"],
        "fd_peak_device_bytes": current["fd_peak"],
        "D_packet_staging_device_bytes": staging,
        "cumulative_live_stratum_fd_device_bytes": cumulative_live,
        "shared_fd_transient_workspace_device_bytes": shared_fd_transient,
        "D_acquisition_peak_device_bytes": D_peak,
        "retained_rank_major_geometry_device_bytes": current["output"],
        "cumulative_retained_rank_major_geometry_device_bytes": (
            cumulative_retained
        ),
        "fused_kkt_problem_device_bytes": current["kkt_problem"],
        "fused_kkt_solver_workspace_device_bytes": current["kkt_solver"],
        "deterministic_rounding_workspace_device_bytes": current["rounding"],
        "shared_rank_local_kkt_execution_device_bytes": shared_kkt_execution,
        "post_D_kkt_peak_device_bytes": post_D_peak,
        "registered_phase_peak_device_bytes": phase_peak,
        "physical_workspace_ceiling_bytes": ceiling,
    }


def _endogenous_phase_liveness_proof_hash_v2(
    *,
    route_count: int,
    rank: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
    D_packet_staging_device_bytes: int,
    physical_workspace_ceiling_bytes: int,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_fqr_endogenous_phase_liveness_v2",
            "route_count": route_count,
            "rank": rank,
            "legality_block_count": legality_block_count,
            "rounding_tile_size": rounding_tile_size,
            "maximum_hard_action_count": maximum_hard_action_count,
            "D_packet_staging_device_bytes": D_packet_staging_device_bytes,
            "physical_workspace_ceiling_bytes": physical_workspace_ceiling_bytes,
            "fd_forecast_owner": (
                "modern_v1_stratified_fd_gpu.forecast_stratified_fd_gpu_memory"
            ),
            "kkt_forecast_owner": (
                "modern_v1_relaxed_kkt."
                "V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT"
            ),
            "kkt_fused_route_vector_count": (
                V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
            ),
            "rank_lattice_role": "execution_approximation_only",
            "pre_D_rank_selection": None,
            "same_capture_VJP_fanout_required": True,
            "independent_rank_local_FD_states_required": True,
            "rank_local_device_lifetime": (
                _ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2
            ),
            "persistent_rank_local_state_accounting": "cumulative_prefix",
            "shared_D_packet_live_copy_count": 1,
            "shared_transient_workspace_live_copy_count": 1,
            "lower_rank_zero_padding_authority": False,
            "phase_order": (
                "D_same_VJP_rank_fanout_then_transient_recompression_eviction_"
                "then_rank_local_KKT_rounding_then_all_D_KKT_eviction_then_C"
            ),
        },
        prefix="partition-free-v1-fqr-endogenous-phase-liveness-v2-",
    )


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankMemoryForecastV2:
    """Pre-outcome phase liveness for a simultaneous rank prefix.

    The scalar rank-local fields retain the resource breakdown for ``rank``.
    The cumulative/shared fields are the admission authority: persistent FD
    and retained-geometry state add across ``rank_prefix``; packet, FD
    transient, and rank-local KKT workspace have one live copy.
    """

    rank: int
    rank_prefix: tuple[int, ...]
    route_count: int
    stratum_count: int
    response_rows: int
    physical_workspace_ceiling_bytes: int
    live_stratum_fd_device_bytes: int
    combined_pre_recompression_factor_device_bytes: int
    combined_output_state_device_bytes: int
    gram_workspace_device_bytes: int
    fd_peak_device_bytes: int
    D_packet_staging_device_bytes: int
    cumulative_live_stratum_fd_device_bytes: int
    shared_fd_transient_workspace_device_bytes: int
    D_acquisition_peak_device_bytes: int
    retained_rank_major_geometry_device_bytes: int
    cumulative_retained_rank_major_geometry_device_bytes: int
    fused_kkt_problem_device_bytes: int
    fused_kkt_solver_workspace_device_bytes: int
    deterministic_rounding_workspace_device_bytes: int
    shared_rank_local_kkt_execution_device_bytes: int
    post_D_kkt_peak_device_bytes: int
    registered_phase_peak_device_bytes: int
    kkt_legality_block_count: int
    kkt_rounding_tile_size: int
    maximum_hard_action_count: int
    phase_liveness_proof_hash: str
    admitted: bool
    kkt_fused_route_vector_count: int = (
        V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
    )
    allocation_free: bool = True
    registered_before_D_outcomes: bool = True
    rank_lattice_is_execution_only: bool = True
    rank_local_device_lifetime: str = (
        _ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2
    )
    shared_D_packet_live_copy_count: int = 1
    shared_transient_workspace_live_copy_count: int = 1
    schema: str = FQR_ENDOGENOUS_RANK_MEMORY_V2_SCHEMA

    def __post_init__(self) -> None:
        rank = _positive_int(self.rank, name="FD rank")
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(self.stratum_count, name="stratum count")
        rows = _positive_int(self.response_rows, name="response rows")
        ceiling = _positive_int(
            self.physical_workspace_ceiling_bytes,
            name="physical workspace ceiling bytes",
        )
        blocks = _positive_int(
            self.kkt_legality_block_count,
            name="KKT legality block count",
        )
        tile = _positive_int(
            self.kkt_rounding_tile_size,
            name="KKT rounding tile size",
        )
        maximum_action = _positive_int(
            self.maximum_hard_action_count,
            name="maximum hard action count",
        )
        staging = _nonnegative_int(
            self.D_packet_staging_device_bytes,
            name="D packet staging device bytes",
        )
        if rank > routes:
            raise ValueError("FD rank cannot exceed the route dimension")
        if self.kkt_fused_route_vector_count != (
            V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
        ):
            raise ValueError("v2 rank forecast is not bound to the fused KKT owner")
        replay = _endogenous_rank_liveness_components_v2(
            route_count=routes,
            rank=rank,
            stratum_count=strata,
            response_rows=rows,
            physical_workspace_ceiling_bytes=ceiling,
            legality_block_count=blocks,
            rounding_tile_size=tile,
            maximum_hard_action_count=maximum_action,
            D_packet_staging_device_bytes=staging,
        )
        rank_prefix = tuple(
            _positive_int(value, name="rank-prefix FD rank")
            for value in self.rank_prefix
        )
        if rank_prefix != replay["rank_prefix"]:
            raise ValueError("endogenous-rank prefix does not replay")
        expected = {
            "live_stratum_fd_device_bytes": replay[
                "live_stratum_fd_device_bytes"
            ],
            "combined_pre_recompression_factor_device_bytes": (
                replay["combined_pre_recompression_factor_device_bytes"]
            ),
            "combined_output_state_device_bytes": (
                replay["combined_output_state_device_bytes"]
            ),
            "gram_workspace_device_bytes": replay["gram_workspace_device_bytes"],
            "fd_peak_device_bytes": replay["fd_peak_device_bytes"],
            "cumulative_live_stratum_fd_device_bytes": replay[
                "cumulative_live_stratum_fd_device_bytes"
            ],
            "shared_fd_transient_workspace_device_bytes": replay[
                "shared_fd_transient_workspace_device_bytes"
            ],
            "D_acquisition_peak_device_bytes": replay[
                "D_acquisition_peak_device_bytes"
            ],
            "retained_rank_major_geometry_device_bytes": replay[
                "retained_rank_major_geometry_device_bytes"
            ],
            "cumulative_retained_rank_major_geometry_device_bytes": replay[
                "cumulative_retained_rank_major_geometry_device_bytes"
            ],
            "fused_kkt_problem_device_bytes": replay[
                "fused_kkt_problem_device_bytes"
            ],
            "fused_kkt_solver_workspace_device_bytes": replay[
                "fused_kkt_solver_workspace_device_bytes"
            ],
            "deterministic_rounding_workspace_device_bytes": replay[
                "deterministic_rounding_workspace_device_bytes"
            ],
            "shared_rank_local_kkt_execution_device_bytes": replay[
                "shared_rank_local_kkt_execution_device_bytes"
            ],
            "post_D_kkt_peak_device_bytes": replay[
                "post_D_kkt_peak_device_bytes"
            ],
            "registered_phase_peak_device_bytes": replay[
                "registered_phase_peak_device_bytes"
            ],
        }
        for name, expected_value in expected.items():
            if _nonnegative_int(getattr(self, name), name=name) != expected_value:
                raise ValueError(f"endogenous-rank {name} does not replay")
        phase_peak = replay["registered_phase_peak_device_bytes"]
        admitted = phase_peak <= ceiling
        if type(self.admitted) is not bool or self.admitted is not admitted:
            raise ValueError("endogenous-rank physical admission does not replay")
        flags = (
            self.allocation_free,
            self.registered_before_D_outcomes,
            self.rank_lattice_is_execution_only,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            True,
        ):
            raise PermissionError("endogenous-rank forecast crossed its boundary")
        if self.rank_local_device_lifetime != (
            _ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2
        ):
            raise ValueError("endogenous-rank device lifetime is invalid")
        if (
            _positive_int(
                self.shared_D_packet_live_copy_count,
                name="shared D-packet live-copy count",
            )
            != 1
            or _positive_int(
                self.shared_transient_workspace_live_copy_count,
                name="shared transient-workspace live-copy count",
            )
            != 1
        ):
            raise ValueError("shared VJP workspace must have exactly one live copy")
        expected_liveness = _endogenous_phase_liveness_proof_hash_v2(
            route_count=routes,
            rank=rank,
            legality_block_count=blocks,
            rounding_tile_size=tile,
            maximum_hard_action_count=maximum_action,
            D_packet_staging_device_bytes=staging,
            physical_workspace_ceiling_bytes=ceiling,
        )
        if self.phase_liveness_proof_hash != expected_liveness:
            raise ValueError("endogenous-rank phase-liveness proof does not replay")
        if self.schema != FQR_ENDOGENOUS_RANK_MEMORY_V2_SCHEMA:
            raise ValueError("endogenous-rank memory schema is invalid")
        object.__setattr__(self, "rank_prefix", rank_prefix)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_ENDOGENOUS_MEMORY_V2_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankPolicyV2:
    """Physical rank admission with no empirical or pre-D rank selection."""

    policy_id: str
    route_count: int
    positive_weight_stratum_ids: tuple[str, ...]
    response_rows: int
    execution_rank_lattice: tuple[int, ...]
    admitted_rank_lattice: tuple[int, ...]
    memory_forecasts: tuple[V1FQREndogenousRankMemoryForecastV2, ...]
    physical_workspace_ceiling_bytes: int
    pre_D_selected_rank: None = None
    rank_lattice_is_execution_only: bool = True
    admitted_by_preoutcome_liveness_only: bool = True
    rank_selected_before_D: bool = False
    empirical_outcomes_used_for_admission: bool = False
    identifiability_rule: str = "every_positive_weight_stratum_n_s_ge_rank_plus_one_v2"
    same_capture_VJP_fanout_required: bool = True
    independent_rank_local_FD_states_required: bool = True
    simultaneous_persistent_rank_prefix_required: bool = True
    rank_local_device_lifetime: str = (
        _ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2
    )
    lower_rank_zero_padding_authority: bool = False
    C_may_open_before_source_decision_invariance: bool = False
    schema: str = FQR_ENDOGENOUS_RANK_POLICY_V2_SCHEMA
    _policy_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        routes = _positive_int(self.route_count, name="route count")
        strata = _identity_axis(
            self.positive_weight_stratum_ids,
            name="positive-weight stratum ids",
        )
        rows = _positive_int(self.response_rows, name="response rows")
        ceiling = _positive_int(
            self.physical_workspace_ceiling_bytes,
            name="physical workspace ceiling bytes",
        )
        lattice = tuple(
            _positive_int(value, name="execution FD rank")
            for value in self.execution_rank_lattice
        )
        expected_lattice = canonical_fqr_endogenous_rank_lattice_for_route_count(
            routes
        )
        if lattice != expected_lattice:
            raise ValueError("v2 requires the route-clipped natural dyadic lattice")
        forecasts = tuple(self.memory_forecasts)
        if (
            len(forecasts) != len(lattice)
            or any(
                type(value) is not V1FQREndogenousRankMemoryForecastV2
                for value in forecasts
            )
            or tuple(value.rank for value in forecasts) != lattice
            or any(
                (
                    value.route_count,
                    value.stratum_count,
                    value.response_rows,
                    value.physical_workspace_ceiling_bytes,
                )
                != (routes, len(strata), rows, ceiling)
                for value in forecasts
            )
        ):
            raise ValueError("v2 rank forecasts do not cover the execution lattice")
        admitted = tuple(value.rank for value in forecasts if value.admitted)
        supplied = tuple(
            _positive_int(value, name="admitted FD rank")
            for value in self.admitted_rank_lattice
        )
        if supplied != admitted or admitted != lattice[: len(admitted)]:
            raise ValueError("v2 admitted ranks do not replay physical liveness")
        if not admitted:
            raise ValueError("a v2 policy requires at least one physically live rank")
        flags = (
            self.rank_lattice_is_execution_only,
            self.admitted_by_preoutcome_liveness_only,
            self.rank_selected_before_D,
            self.empirical_outcomes_used_for_admission,
            self.same_capture_VJP_fanout_required,
            self.independent_rank_local_FD_states_required,
            self.simultaneous_persistent_rank_prefix_required,
            self.lower_rank_zero_padding_authority,
            self.C_may_open_before_source_decision_invariance,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("v2 rank-policy flags must be exact booleans")
        if flags != (True, True, False, False, True, True, True, False, False):
            raise PermissionError("v2 rank policy crossed its causal boundary")
        if self.pre_D_selected_rank is not None:
            raise PermissionError("v2 cannot select a rank before D")
        if self.identifiability_rule != (
            "every_positive_weight_stratum_n_s_ge_rank_plus_one_v2"
        ):
            raise ValueError("v2 rank identifiability rule is invalid")
        if self.rank_local_device_lifetime != (
            _ENDOGENOUS_RANK_LOCAL_DEVICE_LIFETIME_V2
        ):
            raise ValueError("v2 rank-policy device lifetime is invalid")
        if self.schema != FQR_ENDOGENOUS_RANK_POLICY_V2_SCHEMA:
            raise ValueError("v2 rank-policy schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "positive_weight_stratum_ids", strata)
        object.__setattr__(self, "response_rows", rows)
        object.__setattr__(self, "execution_rank_lattice", lattice)
        object.__setattr__(self, "admitted_rank_lattice", admitted)
        object.__setattr__(self, "memory_forecasts", forecasts)
        object.__setattr__(self, "physical_workspace_ceiling_bytes", ceiling)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "route_count": self.route_count,
            "positive_weight_stratum_ids": self.positive_weight_stratum_ids,
            "response_rows": self.response_rows,
            "execution_rank_lattice": self.execution_rank_lattice,
            "admitted_rank_lattice": self.admitted_rank_lattice,
            "memory_forecast_hashes": tuple(
                value.forecast_hash for value in self.memory_forecasts
            ),
            "physical_workspace_ceiling_bytes": (
                self.physical_workspace_ceiling_bytes
            ),
            "pre_D_selected_rank": self.pre_D_selected_rank,
            "rank_lattice_is_execution_only": self.rank_lattice_is_execution_only,
            "admitted_by_preoutcome_liveness_only": (
                self.admitted_by_preoutcome_liveness_only
            ),
            "rank_selected_before_D": self.rank_selected_before_D,
            "empirical_outcomes_used_for_admission": (
                self.empirical_outcomes_used_for_admission
            ),
            "identifiability_rule": self.identifiability_rule,
            "same_capture_VJP_fanout_required": (
                self.same_capture_VJP_fanout_required
            ),
            "independent_rank_local_FD_states_required": (
                self.independent_rank_local_FD_states_required
            ),
            "simultaneous_persistent_rank_prefix_required": (
                self.simultaneous_persistent_rank_prefix_required
            ),
            "rank_local_device_lifetime": self.rank_local_device_lifetime,
            "lower_rank_zero_padding_authority": (
                self.lower_rank_zero_padding_authority
            ),
            "C_may_open_before_source_decision_invariance": (
                self.C_may_open_before_source_decision_invariance
            ),
        }

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ENDOGENOUS_POLICY_V2_PREFIX)
        cached = self._policy_hash
        if cached is not None and cached != observed:
            raise ValueError("v2 endogenous-rank policy was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("v2 endogenous-rank policy hash does not match")
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def register_v1_fqr_endogenous_rank_policy_v2(
    *,
    policy_id: str,
    route_count: int,
    positive_weight_stratum_ids: Sequence[str],
    response_rows: int,
    physical_workspace_ceiling_bytes: int,
    kkt_legality_block_count: int = 16,
    kkt_rounding_tile_size: int = 1 << 20,
    maximum_hard_action_count: int | None = None,
    D_packet_staging_device_bytes: int = 0,
) -> V1FQREndogenousRankPolicyV2 | V1FQREndogenousRankNoCRefusalV2:
    """Register physical rank liveness without selecting a rank before D."""

    policy_name = _identity(policy_id, name="policy id")
    routes = _positive_int(route_count, name="route count")
    strata = _identity_axis(
        positive_weight_stratum_ids,
        name="positive-weight stratum ids",
    )
    rows = _positive_int(response_rows, name="response rows")
    ceiling = _positive_int(
        physical_workspace_ceiling_bytes,
        name="physical workspace ceiling bytes",
    )
    blocks = _positive_int(
        kkt_legality_block_count,
        name="KKT legality block count",
    )
    tile = _positive_int(
        kkt_rounding_tile_size,
        name="KKT rounding tile size",
    )
    maximum_action = (
        routes
        if maximum_hard_action_count is None
        else _positive_int(
            maximum_hard_action_count,
            name="maximum hard action count",
        )
    )
    staging = _nonnegative_int(
        D_packet_staging_device_bytes,
        name="D packet staging device bytes",
    )
    lattice = canonical_fqr_endogenous_rank_lattice_for_route_count(routes)
    forecasts: list[V1FQREndogenousRankMemoryForecastV2] = []
    for rank in lattice:
        replay = _endogenous_rank_liveness_components_v2(
            route_count=routes,
            rank=rank,
            stratum_count=len(strata),
            response_rows=rows,
            physical_workspace_ceiling_bytes=ceiling,
            legality_block_count=blocks,
            rounding_tile_size=tile,
            maximum_hard_action_count=maximum_action,
            D_packet_staging_device_bytes=staging,
        )
        phase_peak = int(replay["registered_phase_peak_device_bytes"])
        forecasts.append(
            V1FQREndogenousRankMemoryForecastV2(
                rank=rank,
                rank_prefix=replay["rank_prefix"],
                route_count=routes,
                stratum_count=len(strata),
                response_rows=rows,
                physical_workspace_ceiling_bytes=ceiling,
                live_stratum_fd_device_bytes=replay[
                    "live_stratum_fd_device_bytes"
                ],
                combined_pre_recompression_factor_device_bytes=(
                    replay["combined_pre_recompression_factor_device_bytes"]
                ),
                combined_output_state_device_bytes=(
                    replay["combined_output_state_device_bytes"]
                ),
                gram_workspace_device_bytes=replay[
                    "gram_workspace_device_bytes"
                ],
                fd_peak_device_bytes=replay["fd_peak_device_bytes"],
                D_packet_staging_device_bytes=staging,
                cumulative_live_stratum_fd_device_bytes=(
                    replay["cumulative_live_stratum_fd_device_bytes"]
                ),
                shared_fd_transient_workspace_device_bytes=(
                    replay["shared_fd_transient_workspace_device_bytes"]
                ),
                D_acquisition_peak_device_bytes=(
                    replay["D_acquisition_peak_device_bytes"]
                ),
                retained_rank_major_geometry_device_bytes=(
                    replay["retained_rank_major_geometry_device_bytes"]
                ),
                cumulative_retained_rank_major_geometry_device_bytes=(
                    replay[
                        "cumulative_retained_rank_major_geometry_device_bytes"
                    ]
                ),
                fused_kkt_problem_device_bytes=(
                    replay["fused_kkt_problem_device_bytes"]
                ),
                fused_kkt_solver_workspace_device_bytes=(
                    replay["fused_kkt_solver_workspace_device_bytes"]
                ),
                deterministic_rounding_workspace_device_bytes=(
                    replay["deterministic_rounding_workspace_device_bytes"]
                ),
                shared_rank_local_kkt_execution_device_bytes=(
                    replay["shared_rank_local_kkt_execution_device_bytes"]
                ),
                post_D_kkt_peak_device_bytes=(
                    replay["post_D_kkt_peak_device_bytes"]
                ),
                registered_phase_peak_device_bytes=phase_peak,
                kkt_legality_block_count=blocks,
                kkt_rounding_tile_size=tile,
                maximum_hard_action_count=maximum_action,
                phase_liveness_proof_hash=(
                    _endogenous_phase_liveness_proof_hash_v2(
                        route_count=routes,
                        rank=rank,
                        legality_block_count=blocks,
                        rounding_tile_size=tile,
                        maximum_hard_action_count=maximum_action,
                        D_packet_staging_device_bytes=staging,
                        physical_workspace_ceiling_bytes=ceiling,
                    )
                ),
                admitted=phase_peak <= ceiling,
            )
        )
    admitted = tuple(value.rank for value in forecasts if value.admitted)
    if not admitted:
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.PHYSICAL_LIVENESS_EXHAUSTED,
            reason=(
                "even rank one exceeds the registered pre-outcome physical "
                "workspace ceiling"
            ),
            required_action=V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL,
        )
    return V1FQREndogenousRankPolicyV2(
        policy_id=policy_name,
        route_count=routes,
        positive_weight_stratum_ids=strata,
        response_rows=rows,
        execution_rank_lattice=lattice,
        admitted_rank_lattice=admitted,
        memory_forecasts=tuple(forecasts),
        physical_workspace_ceiling_bytes=ceiling,
    )


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankDPrefixV2:
    """Registered D prefix and all ranks identifiable from that prefix."""

    policy_hash: str
    current_mask_hash: str
    route_axis_hash: str
    proposal_generation: int
    registered_d_prefix_identity_hash: str
    D_context_axis_hash: str
    D_draw_axis_hash: str
    observed_stratum_counts: tuple[tuple[str, int], ...]
    population_stratum_counts: tuple[tuple[str, int], ...]
    identifiable_admitted_ranks: tuple[int, ...]
    full_D_census: bool
    _issuance_token: InitVar[object]
    every_positive_weight_stratum_required: bool = True
    identifiability_rule: str = "every_positive_weight_stratum_n_s_ge_rank_plus_one_v2"
    rank_selected: None = None
    C_stream_open_authority: bool = False
    schema: str = FQR_ENDOGENOUS_RANK_D_PREFIX_V2_SCHEMA
    _prefix_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ENDOGENOUS_D_PREFIX_V2_ISSUANCE_TOKEN:
            raise PermissionError("v2 D prefix requires its registered issuer")
        for name in (
            "policy_hash",
            "current_mask_hash",
            "route_axis_hash",
            "registered_d_prefix_identity_hash",
            "D_context_axis_hash",
            "D_draw_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for axis_name in ("observed_stratum_counts", "population_stratum_counts"):
            axis = tuple(
                (
                    _identity(stratum, name=f"{axis_name} stratum"),
                    _nonnegative_int(count, name=f"{axis_name} count"),
                )
                for stratum, count in getattr(self, axis_name)
            )
            if not axis or len({stratum for stratum, _count in axis}) != len(axis):
                raise ValueError(f"{axis_name} must have one unique stratum axis")
            object.__setattr__(self, axis_name, axis)
        if tuple(value[0] for value in self.observed_stratum_counts) != tuple(
            value[0] for value in self.population_stratum_counts
        ):
            raise ValueError("observed and population D strata differ")
        if any(
            observed > population or population == 0
            for (_stratum, observed), (_same, population) in zip(
                self.observed_stratum_counts,
                self.population_stratum_counts,
                strict=True,
            )
        ):
            raise ValueError("D prefix counts exceed or lack their population")
        ranks = tuple(
            _positive_int(value, name="identifiable admitted rank")
            for value in self.identifiable_admitted_ranks
        )
        if ranks != tuple(sorted(set(ranks))):
            raise ValueError("identifiable ranks must be unique and increasing")
        object.__setattr__(self, "identifiable_admitted_ranks", ranks)
        census = all(
            observed == population
            for (_stratum, observed), (_same, population) in zip(
                self.observed_stratum_counts,
                self.population_stratum_counts,
                strict=True,
            )
        )
        if type(self.full_D_census) is not bool or self.full_D_census is not census:
            raise ValueError("full-D-census status does not replay its counts")
        if (
            type(self.every_positive_weight_stratum_required) is not bool
            or self.every_positive_weight_stratum_required is not True
            or self.identifiability_rule
            != "every_positive_weight_stratum_n_s_ge_rank_plus_one_v2"
            or self.rank_selected is not None
            or type(self.C_stream_open_authority) is not bool
            or self.C_stream_open_authority
        ):
            raise PermissionError("v2 D-prefix receipt crossed its authority")
        if self.schema != FQR_ENDOGENOUS_RANK_D_PREFIX_V2_SCHEMA:
            raise ValueError("v2 D-prefix schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "proposal_generation": self.proposal_generation,
            "registered_d_prefix_identity_hash": (
                self.registered_d_prefix_identity_hash
            ),
            "D_context_axis_hash": self.D_context_axis_hash,
            "D_draw_axis_hash": self.D_draw_axis_hash,
            "observed_stratum_counts": self.observed_stratum_counts,
            "population_stratum_counts": self.population_stratum_counts,
            "identifiable_admitted_ranks": self.identifiable_admitted_ranks,
            "full_D_census": self.full_D_census,
            "every_positive_weight_stratum_required": (
                self.every_positive_weight_stratum_required
            ),
            "identifiability_rule": self.identifiability_rule,
            "rank_selected": self.rank_selected,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def prefix_hash(self) -> str:
        cached = self._prefix_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ENDOGENOUS_D_PREFIX_V2_PREFIX)
        cached = self._prefix_hash
        if cached is not None and cached != observed:
            raise ValueError("v2 D-prefix receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("v2 D-prefix hash does not match")
        object.__setattr__(self, "_prefix_hash", observed)
        return observed


def register_v1_fqr_endogenous_rank_d_prefix_v2(
    *,
    policy: V1FQREndogenousRankPolicyV2,
    current_mask_hash: str,
    route_axis_hash: str,
    proposal_generation: int,
    registered_d_prefix_identity_hash: str,
    D_context_axis_hash: str,
    D_draw_axis_hash: str,
    observed_stratum_counts: Mapping[str, int],
    population_stratum_counts: Mapping[str, int],
) -> V1FQREndogenousRankDPrefixV2:
    """Expose every admitted rank identifiable at this exact D prefix."""

    if type(policy) is not V1FQREndogenousRankPolicyV2:
        raise TypeError("v2 D-prefix registration requires the exact v2 policy")
    policy.assert_integrity()
    stratum_axis = policy.positive_weight_stratum_ids
    if set(observed_stratum_counts) != set(stratum_axis) or set(
        population_stratum_counts
    ) != set(stratum_axis):
        raise ValueError("D prefix must cover every positive-weight stratum exactly")
    observed = tuple(
        (
            stratum,
            _nonnegative_int(
                observed_stratum_counts[stratum],
                name=f"observed count for {stratum}",
            ),
        )
        for stratum in stratum_axis
    )
    population = tuple(
        (
            stratum,
            _positive_int(
                population_stratum_counts[stratum],
                name=f"population count for {stratum}",
            ),
        )
        for stratum in stratum_axis
    )
    if any(
        count > limit
        for (_stratum, count), (_same, limit) in zip(
            observed,
            population,
            strict=True,
        )
    ):
        raise ValueError("observed D prefix exceeds the registered population")
    minimum_count = min(count for _stratum, count in observed)
    identifiable = tuple(
        rank
        for rank in policy.admitted_rank_lattice
        if minimum_count >= rank + 1
    )
    return V1FQREndogenousRankDPrefixV2(
        policy_hash=policy.policy_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        proposal_generation=proposal_generation,
        registered_d_prefix_identity_hash=registered_d_prefix_identity_hash,
        D_context_axis_hash=D_context_axis_hash,
        D_draw_axis_hash=D_draw_axis_hash,
        observed_stratum_counts=observed,
        population_stratum_counts=population,
        identifiable_admitted_ranks=identifiable,
        full_D_census=observed == population,
        _issuance_token=_ENDOGENOUS_D_PREFIX_V2_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankFanoutStateV2:
    """One independent rank-local FD state fed by the shared VJP lineage."""

    rank: int
    fd_state_hash: str
    source_capture_vjp_lineage_hash: str
    registered_d_prefix_hash: str
    _issuance_token: InitVar[object]
    built_directly_from_shared_row_stream: bool = True
    lower_rank_zero_padding_used: bool = False
    raw_D_rank_reexecution_used: bool = False
    schema: str = FQR_ENDOGENOUS_RANK_FANOUT_STATE_V2_SCHEMA

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ENDOGENOUS_FANOUT_STATE_V2_ISSUANCE_TOKEN:
            raise PermissionError("v2 fanout state requires its registered issuer")
        object.__setattr__(self, "rank", _positive_int(self.rank, name="FD rank"))
        for name in (
            "fd_state_hash",
            "source_capture_vjp_lineage_hash",
            "registered_d_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        flags = (
            self.built_directly_from_shared_row_stream,
            self.lower_rank_zero_padding_used,
            self.raw_D_rank_reexecution_used,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("v2 fanout-state flags must be exact booleans")
        if flags != (True, False, False):
            raise PermissionError("v2 rank-local state is not a direct shared-VJP fanout")
        if self.schema != FQR_ENDOGENOUS_RANK_FANOUT_STATE_V2_SCHEMA:
            raise ValueError("v2 fanout-state schema is invalid")

    @property
    def state_hash(self) -> str:
        return content_hash(self, prefix=_ENDOGENOUS_FANOUT_STATE_V2_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQREndogenousRankSameVJPFanoutV2:
    """One packet lineage fanned into every identifiable rank-local FD state."""

    policy_hash: str
    registered_d_prefix_hash: str
    capture_vjp_packet_hashes: tuple[str, ...]
    source_capture_vjp_lineage_hash: str
    rank_local_states: tuple[V1FQREndogenousRankFanoutStateV2, ...]
    _issuance_token: InitVar[object]
    raw_D_rank_reexecution_count: int = 0
    same_capture_VJP_fanned_to_every_identifiable_rank: bool = True
    independent_rank_local_FD_states: bool = True
    lower_rank_zero_padding_used: bool = False
    C_stream_open_authority: bool = False
    schema: str = FQR_ENDOGENOUS_RANK_FANOUT_V2_SCHEMA
    _fanout_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ENDOGENOUS_FANOUT_V2_ISSUANCE_TOKEN:
            raise PermissionError("v2 same-VJP fanout requires its registered issuer")
        for name in (
            "policy_hash",
            "registered_d_prefix_hash",
            "source_capture_vjp_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        packets = _identity_axis(
            self.capture_vjp_packet_hashes,
            name="capture/VJP packet hashes",
        )
        states = tuple(self.rank_local_states)
        if not states or any(
            type(value) is not V1FQREndogenousRankFanoutStateV2
            for value in states
        ):
            raise TypeError("v2 fanout requires exact rank-local states")
        if tuple(value.rank for value in states) != tuple(
            sorted({value.rank for value in states})
        ):
            raise ValueError("v2 fanout rank-local states must be unique and increasing")
        if len({value.fd_state_hash for value in states}) != len(states):
            raise ValueError("rank-local FD states must be independently identified")
        if any(
            value.source_capture_vjp_lineage_hash
            != self.source_capture_vjp_lineage_hash
            or value.registered_d_prefix_hash != self.registered_d_prefix_hash
            for value in states
        ):
            raise ValueError("rank-local FD states do not share one VJP lineage")
        expected_lineage = content_hash(
            {
                "registered_d_prefix_hash": self.registered_d_prefix_hash,
                "capture_vjp_packet_hashes": packets,
                "one_capture_vjp_per_packet_fanned_to_all_ranks": True,
            },
            prefix="partition-free-v1-fqr-endogenous-capture-VJP-lineage-v2-",
        )
        if self.source_capture_vjp_lineage_hash != expected_lineage:
            raise ValueError("v2 capture/VJP lineage does not replay its packets")
        if _nonnegative_int(
            self.raw_D_rank_reexecution_count,
            name="raw-D rank reexecution count",
        ) != 0:
            raise PermissionError("v2 forbids raw-D reexecution per FD rank")
        flags = (
            self.same_capture_VJP_fanned_to_every_identifiable_rank,
            self.independent_rank_local_FD_states,
            self.lower_rank_zero_padding_used,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("v2 fanout flags must be exact booleans")
        if flags != (True, True, False, False):
            raise PermissionError("v2 same-VJP fanout crossed its authority")
        if self.schema != FQR_ENDOGENOUS_RANK_FANOUT_V2_SCHEMA:
            raise ValueError("v2 same-VJP fanout schema is invalid")
        object.__setattr__(self, "capture_vjp_packet_hashes", packets)
        object.__setattr__(self, "rank_local_states", states)
        self.assert_integrity()

    @property
    def fanned_ranks(self) -> tuple[int, ...]:
        return tuple(value.rank for value in self.rank_local_states)

    @property
    def dense_vjp_packet_execution_count(self) -> int:
        return len(self.capture_vjp_packet_hashes)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "capture_vjp_packet_hashes": self.capture_vjp_packet_hashes,
            "source_capture_vjp_lineage_hash": (
                self.source_capture_vjp_lineage_hash
            ),
            "rank_local_state_hashes": tuple(
                value.state_hash for value in self.rank_local_states
            ),
            "fanned_ranks": self.fanned_ranks,
            "dense_vjp_packet_execution_count": (
                self.dense_vjp_packet_execution_count
            ),
            "raw_D_rank_reexecution_count": self.raw_D_rank_reexecution_count,
            "same_capture_VJP_fanned_to_every_identifiable_rank": (
                self.same_capture_VJP_fanned_to_every_identifiable_rank
            ),
            "independent_rank_local_FD_states": (
                self.independent_rank_local_FD_states
            ),
            "lower_rank_zero_padding_used": self.lower_rank_zero_padding_used,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def fanout_hash(self) -> str:
        cached = self._fanout_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ENDOGENOUS_FANOUT_V2_PREFIX)
        cached = self._fanout_hash
        if cached is not None and cached != observed:
            raise ValueError("v2 same-VJP fanout was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("v2 same-VJP fanout hash does not match")
        object.__setattr__(self, "_fanout_hash", observed)
        return observed


def register_v1_fqr_same_vjp_rank_fanout_v2(
    *,
    policy: V1FQREndogenousRankPolicyV2,
    d_prefix: V1FQREndogenousRankDPrefixV2,
    capture_vjp_packet_hashes: Sequence[str],
    rank_local_fd_state_hashes: Mapping[int, str],
) -> V1FQREndogenousRankSameVJPFanoutV2 | V1FQREndogenousRankNoCRefusalV2:
    """Bind one D capture/VJP packet lineage to all identifiable FD ranks."""

    if type(policy) is not V1FQREndogenousRankPolicyV2:
        raise TypeError("v2 fanout registration requires the exact v2 policy")
    if type(d_prefix) is not V1FQREndogenousRankDPrefixV2:
        raise TypeError("v2 fanout registration requires the exact D prefix")
    policy_hash = policy.assert_integrity()
    prefix_hash = d_prefix.assert_integrity()
    if d_prefix.policy_hash != policy_hash:
        raise ValueError("v2 D prefix belongs to another policy")
    ranks = d_prefix.identifiable_admitted_ranks
    if not ranks:
        next_rank = policy.admitted_rank_lattice[0]
        action = (
            V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL
            if d_prefix.full_D_census
            else V1FQREndogenousRankNextAction.ACQUIRE_MORE_D_FOR_IDENTIFIABILITY
        )
        return V1FQREndogenousRankNoCRefusalV2(
            code=(
                V1FQREndogenousRankNoCCode.PHYSICAL_LIVENESS_EXHAUSTED
                if d_prefix.full_D_census
                else V1FQREndogenousRankNoCCode.D_IDENTIFIABILITY_INCOMPLETE
            ),
            reason=(
                "no physically admitted rank is identifiable at the registered "
                "D prefix under n_s >= r + 1"
            ),
            required_action=action,
            policy_hash=policy_hash,
            registered_d_prefix_hash=prefix_hash,
            next_rank=next_rank,
        )
    supplied_ranks = tuple(sorted(rank_local_fd_state_hashes))
    if supplied_ranks != ranks:
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.SAME_VJP_FANOUT_INVALID,
            reason=(
                "same-VJP fanout must materialize every and only currently "
                "identifiable admitted rank"
            ),
            required_action=(
                V1FQREndogenousRankNextAction.REFINE_RANK_FROM_SAME_VJP_FANOUT
            ),
            policy_hash=policy_hash,
            registered_d_prefix_hash=prefix_hash,
        )
    packets = _identity_axis(
        capture_vjp_packet_hashes,
        name="capture/VJP packet hashes",
    )
    lineage = content_hash(
        {
            "registered_d_prefix_hash": prefix_hash,
            "capture_vjp_packet_hashes": packets,
            "one_capture_vjp_per_packet_fanned_to_all_ranks": True,
        },
        prefix="partition-free-v1-fqr-endogenous-capture-VJP-lineage-v2-",
    )
    states = tuple(
        V1FQREndogenousRankFanoutStateV2(
            rank=rank,
            fd_state_hash=_identity(
                rank_local_fd_state_hashes[rank],
                name=f"rank-{rank} FD-state hash",
            ),
            source_capture_vjp_lineage_hash=lineage,
            registered_d_prefix_hash=prefix_hash,
            _issuance_token=_ENDOGENOUS_FANOUT_STATE_V2_ISSUANCE_TOKEN,
        )
        for rank in ranks
    )
    return V1FQREndogenousRankSameVJPFanoutV2(
        policy_hash=policy_hash,
        registered_d_prefix_hash=prefix_hash,
        capture_vjp_packet_hashes=packets,
        source_capture_vjp_lineage_hash=lineage,
        rank_local_states=states,
        _issuance_token=_ENDOGENOUS_FANOUT_V2_ISSUANCE_TOKEN,
    )


__all__ = [
    "CANONICAL_FQR_ADAPTIVE_RANK_LADDER",
    "FQR_ADAPTIVE_RANK_BRANCH_DECISION_SCHEMA",
    "FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA",
    "FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_SCHEMA",
    "FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA",
    "FQR_ADAPTIVE_RANK_LEDGER_SCHEMA",
    "FQR_ADAPTIVE_RANK_MEMORY_SCHEMA",
    "FQR_ADAPTIVE_RANK_POLICY_SCHEMA",
    "FQR_ADAPTIVE_RANK_REFUSAL_SCHEMA",
    "FQR_ADAPTIVE_RANK_REPLAY_SCHEMA",
    "FQR_ADAPTIVE_RANK_STABILITY_SCHEMA",
    "FQR_ADAPTIVE_RANK_STABILITY_V2_SCHEMA",
    "FQR_D_DECISION_ACQUISITION_GUIDANCE_SCHEMA",
    "FQR_ENDOGENOUS_RANK_D_PREFIX_V2_SCHEMA",
    "FQR_ENDOGENOUS_RANK_FANOUT_STATE_V2_SCHEMA",
    "FQR_ENDOGENOUS_RANK_FANOUT_V2_SCHEMA",
    "FQR_ENDOGENOUS_RANK_MEMORY_V2_SCHEMA",
    "FQR_ENDOGENOUS_RANK_NO_C_REFUSAL_V2_SCHEMA",
    "FQR_ENDOGENOUS_RANK_POLICY_V2_SCHEMA",
    "V1FQRAdaptiveRankBranchDecision",
    "V1FQRAdaptiveRankDecisionCertificate",
    "V1FQRAdaptiveRankMemoryForecast",
    "V1FQRAdaptiveRankNoCRefusal",
    "V1FQRAdaptiveRankPolicy",
    "V1FQRAdaptiveRankRefusal",
    "V1FQRAdaptiveRankRefusalCode",
    "V1FQRAdaptiveRankReplayLedger",
    "V1FQRAdaptiveRankReplayRecord",
    "V1FQRAdaptiveRankStabilityCertificate",
    "V1FQRDDecisionAcquisitionAction",
    "V1FQRDDecisionAmbiguitySource",
    "V1FQRDDecisionStabilityAcquisitionGuidance",
    "V1FQREndogenousRankDPrefixV2",
    "V1FQREndogenousRankFanoutStateV2",
    "V1FQREndogenousRankMemoryForecastV2",
    "V1FQREndogenousRankNextAction",
    "V1FQREndogenousRankNoCCode",
    "V1FQREndogenousRankNoCRefusalV2",
    "V1FQREndogenousRankPolicyV2",
    "V1FQREndogenousRankSameVJPFanoutV2",
    "certify_v1_fqr_adaptive_rank_stability",
    "canonical_fqr_adaptive_rank_ladder_for_route_count",
    "canonical_fqr_endogenous_rank_lattice_for_route_count",
    "refuse_v1_fqr_adaptive_rank",
    "register_v1_fqr_adaptive_rank_policy",
    "register_v1_fqr_endogenous_rank_d_prefix_v2",
    "register_v1_fqr_endogenous_rank_policy_v2",
    "register_v1_fqr_same_vjp_rank_fanout_v2",
]
