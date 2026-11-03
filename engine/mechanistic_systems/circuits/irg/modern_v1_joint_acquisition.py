"""Resumable evidence acquisition contracts for the modern-v1 pruning step.

The scheduler in this module is an inert control-plane joint.  It compares *predicted
contractions of uncertainty about the next irreversible action* per device
second, but it neither observes an acquisition outcome nor chooses, evaluates,
or commits a mask.  In particular, the common regret currency is only a way to
allocate measurement precision.  It never permits sparsity value to compensate
for unresolved paired-oracle loss evidence: the oracle remains a hard gate.

A separate outcome-bound receipt advances proposal evidence using train-only
request/source/outcome identities.  A further, evidence-bound decision receipt
may authorize candidate freeze or explicitly refuse it.  Scheduler selection
alone carries no such authority.

The phase boundary is deliberately asymmetric.  Proposal measurements may
refine corrected attribution or response geometry before a candidate exists.
After a candidate is frozen, only its calibration measurements are legal; once
the joint calibration state is frozen, only paired-oracle measurements are
legal.  Terminal state admits no acquisition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral, Real
from typing import Final, Literal, TypeAlias, cast

from .contracts import content_hash

V1JointAcquisitionPhase: TypeAlias = Literal[
    "proposal",
    "calibration",
    "oracle",
    "terminal",
]
V1JointAcquisitionRole: TypeAlias = Literal[
    "corrected_attribution",
    "response_exactification",
    "corrected_attribution_response_joint",
    "selected_shell_fq_response_exactification",
    "canonical_fqr_C_certification",
    "candidate_calibration",
    "dense_calibration",
    "paired_oracle",
]
V1ProposalEvidenceRole: TypeAlias = Literal[
    "corrected_attribution",
    "response_exactification",
]
V1JointAcquisitionScheduleAction: TypeAlias = Literal[
    "acquire",
    "refuse_no_legal_action",
]
V1ProposalFreezeAction: TypeAlias = Literal[
    "authorize_candidate_freeze",
    "refuse_candidate_freeze",
]

_STATE_SCHEMA: Final = "partition_free_v1_joint_acquisition_state_v2"
_FORECAST_SCHEMA: Final = "partition_free_v1_next_action_acquisition_forecast_v2"
_RECEIPT_SCHEMA: Final = "partition_free_v1_joint_acquisition_schedule_receipt_v2"
_UPDATE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_proposal_acquisition_update_receipt_v1"
)
_FREEZE_EVIDENCE_SCHEMA: Final = "partition_free_v1_proposal_freeze_evidence_v2"
_FREEZE_DECISION_SCHEMA: Final = "partition_free_v1_proposal_freeze_decision_receipt_v2"
_REPLAY_STEP_SCHEMA: Final = "partition_free_v1_proposal_acquisition_replay_step_v1"
_VERIFIED_EVIDENCE_SCHEMA: Final = (
    "partition_free_v1_verified_proposal_evidence_state_v1"
)
_PRECISION_CURRENCY: Final = "next_irreversible_action_regret_v1"
_FORECAST_INFORMATION_SCOPE: Final = "predictable_pre_measurement_information_only"
_FREEZE_EVIDENCE_TOKEN: Final = object()
_VERIFIED_EVIDENCE_TOKEN: Final = object()

_ROLES_BY_PHASE: Final[dict[str, frozenset[str]]] = {
    "proposal": frozenset(
        {
            "corrected_attribution",
            "response_exactification",
            "corrected_attribution_response_joint",
            "selected_shell_fq_response_exactification",
            "canonical_fqr_C_certification",
        }
    ),
    "calibration": frozenset({"candidate_calibration", "dense_calibration"}),
    "oracle": frozenset({"paired_oracle"}),
    "terminal": frozenset(),
}
_ROLE_ORDER: Final[dict[str, int]] = {
    "corrected_attribution": 0,
    "response_exactification": 1,
    "corrected_attribution_response_joint": 2,
    "selected_shell_fq_response_exactification": 3,
    "canonical_fqr_C_certification": 4,
    "candidate_calibration": 5,
    "dense_calibration": 6,
    "paired_oracle": 7,
}
_PROPOSAL_EVIDENCE_ROLES_BY_ACQUISITION: Final[
    dict[str, tuple[V1ProposalEvidenceRole, ...]]
] = {
    "corrected_attribution": ("corrected_attribution",),
    "response_exactification": ("response_exactification",),
    "corrected_attribution_response_joint": (
        "corrected_attribution",
        "response_exactification",
    ),
    "selected_shell_fq_response_exactification": ("response_exactification",),
    "canonical_fqr_C_certification": ("response_exactification",),
}


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _optional_nonempty(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_positive(value: object, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonempty_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a nonempty sequence of strings")
    result = tuple(_nonempty(item, label) for item in value)
    if not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _context_draw_ids_hash(
    context_ids: tuple[str, ...],
    draw_ids: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_train_context_draw_identity_v1",
            "ordered_pairs": tuple(zip(context_ids, draw_ids, strict=True)),
        },
        prefix="partition-free-v1-train-context-draw-ids-",
    )


def _validate_phase_binding(
    *,
    phase: object,
    candidate_identity_hash: object,
    frozen_calibration_hash: object,
) -> tuple[V1JointAcquisitionPhase, str | None, str | None]:
    if phase not in _ROLES_BY_PHASE:
        raise ValueError("unknown modern-v1 joint-acquisition phase")
    resolved_phase = cast(V1JointAcquisitionPhase, phase)
    candidate = _optional_nonempty(
        candidate_identity_hash,
        "candidate_identity_hash",
    )
    calibration = _optional_nonempty(
        frozen_calibration_hash,
        "frozen_calibration_hash",
    )
    if resolved_phase == "proposal":
        if candidate is not None or calibration is not None:
            raise ValueError(
                "proposal phase forbids a frozen candidate and calibration"
            )
    elif resolved_phase == "calibration":
        if candidate is None or calibration is not None:
            raise ValueError(
                "calibration phase requires a frozen candidate and forbids an "
                "already-frozen calibration"
            )
    elif resolved_phase == "oracle" and (candidate is None or calibration is None):
        raise ValueError(
            "oracle phase requires both a frozen candidate and frozen calibration"
        )
    elif resolved_phase == "terminal" and ((candidate is None) != (calibration is None)):
        raise ValueError(
            "terminal phase must retain both candidate/calibration identities or neither"
        )
    return resolved_phase, candidate, calibration


@dataclass(frozen=True, slots=True)
class V1JointAcquisitionState:
    """Immutable phase and identity boundary for one acquisition decision."""

    campaign_binding_hash: str
    current_mask_hash: str
    action_regret_metric_hash: str
    phase_evidence_state_hash: str
    phase: V1JointAcquisitionPhase
    logical_sequence: int
    candidate_identity_hash: str | None = None
    frozen_calibration_hash: str | None = None
    precision_currency: str = _PRECISION_CURRENCY
    oracle_hard_gate: bool = True
    loss_sparsity_trade_allowed: bool = False
    schema: str = _STATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_binding_hash",
            "current_mask_hash",
            "action_regret_metric_hash",
            "phase_evidence_state_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        phase, candidate, calibration = _validate_phase_binding(
            phase=self.phase,
            candidate_identity_hash=self.candidate_identity_hash,
            frozen_calibration_hash=self.frozen_calibration_hash,
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "candidate_identity_hash", candidate)
        object.__setattr__(self, "frozen_calibration_hash", calibration)
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        if (
            self.precision_currency != _PRECISION_CURRENCY
            or self.oracle_hard_gate is not True
            or self.loss_sparsity_trade_allowed is not False
            or self.schema != _STATE_SCHEMA
        ):
            raise ValueError(
                "joint acquisition must preserve the regret-currency/oracle firewall"
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-joint-acquisition-state-")


@dataclass(frozen=True, slots=True)
class V1NextActionAcquisitionForecast:
    """Predictable value-of-information forecast for one physical acquisition.

    ``predicted_action_regret_contraction`` is a reduction in uncertainty about
    which next action should be taken.  It is not predicted sparsity reward or
    realized loss and therefore cannot offset the paired-oracle gate.
    """

    state_hash: str
    phase: V1JointAcquisitionPhase
    role: V1JointAcquisitionRole
    action_regret_metric_hash: str
    phase_evidence_state_hash: str
    acquisition_design_hash: str
    forecast_basis_hash: str
    acquisition_units: int
    predicted_action_regret_contraction: float
    predicted_device_seconds: float
    candidate_identity_hash: str | None = None
    frozen_calibration_hash: str | None = None
    forecast_information_scope: str = _FORECAST_INFORMATION_SCOPE
    outcome_independent: bool = True
    precision_coordination_only: bool = True
    oracle_hard_gate: bool = True
    loss_sparsity_trade_allowed: bool = False
    mask_commit_authority: bool = False
    schema: str = _FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "state_hash",
            "action_regret_metric_hash",
            "phase_evidence_state_hash",
            "acquisition_design_hash",
            "forecast_basis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        phase, candidate, calibration = _validate_phase_binding(
            phase=self.phase,
            candidate_identity_hash=self.candidate_identity_hash,
            frozen_calibration_hash=self.frozen_calibration_hash,
        )
        if self.role not in _ROLE_ORDER:
            raise ValueError("unknown modern-v1 acquisition role")
        if self.role not in _ROLES_BY_PHASE[phase]:
            raise ValueError(f"{self.role} acquisition is ineligible in {phase} phase")
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "candidate_identity_hash", candidate)
        object.__setattr__(self, "frozen_calibration_hash", calibration)
        object.__setattr__(
            self,
            "acquisition_units",
            _positive_int(self.acquisition_units, "acquisition_units"),
        )
        object.__setattr__(
            self,
            "predicted_action_regret_contraction",
            _finite_nonnegative(
                self.predicted_action_regret_contraction,
                "predicted_action_regret_contraction",
            ),
        )
        object.__setattr__(
            self,
            "predicted_device_seconds",
            _finite_positive(
                self.predicted_device_seconds,
                "predicted_device_seconds",
            ),
        )
        if (
            self.forecast_information_scope != _FORECAST_INFORMATION_SCOPE
            or self.outcome_independent is not True
            or self.precision_coordination_only is not True
            or self.oracle_hard_gate is not True
            or self.loss_sparsity_trade_allowed is not False
            or self.mask_commit_authority is not False
            or self.schema != _FORECAST_SCHEMA
        ):
            raise ValueError(
                "forecast crossed its predictable precision-only authority firewall"
            )

    @property
    def utility_per_device_second(self) -> float:
        return (
            float(self.predicted_action_regret_contraction)
            / float(self.predicted_device_seconds)
        )

    @property
    def content_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-next-action-acquisition-forecast-",
        )


@dataclass(frozen=True, slots=True)
class V1JointAcquisitionScheduleReceipt:
    """Content-addressed selection or typed refusal from the pure scheduler."""

    state_hash: str
    phase: V1JointAcquisitionPhase
    logical_sequence: int
    action_regret_metric_hash: str
    phase_evidence_state_hash: str
    candidate_identity_hash: str | None
    frozen_calibration_hash: str | None
    action: V1JointAcquisitionScheduleAction
    considered_forecast_hashes: tuple[str, ...]
    selected_forecast_hash: str | None
    selected_role: V1JointAcquisitionRole | None
    refusal_reason: str | None
    forecast_information_scope: str = _FORECAST_INFORMATION_SCOPE
    outcome_independent: bool = True
    precision_coordination_only: bool = True
    oracle_hard_gate: bool = True
    loss_sparsity_trade_allowed: bool = False
    mask_commit_authority: bool = False
    schema: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_hash", _nonempty(self.state_hash, "state_hash"))
        object.__setattr__(
            self,
            "action_regret_metric_hash",
            _nonempty(self.action_regret_metric_hash, "action_regret_metric_hash"),
        )
        object.__setattr__(
            self,
            "phase_evidence_state_hash",
            _nonempty(self.phase_evidence_state_hash, "phase_evidence_state_hash"),
        )
        phase, candidate, calibration = _validate_phase_binding(
            phase=self.phase,
            candidate_identity_hash=self.candidate_identity_hash,
            frozen_calibration_hash=self.frozen_calibration_hash,
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "candidate_identity_hash", candidate)
        object.__setattr__(self, "frozen_calibration_hash", calibration)
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        hashes = tuple(
            _nonempty(value, "considered_forecast_hashes")
            for value in self.considered_forecast_hashes
        )
        if hashes != tuple(sorted(hashes)) or len(set(hashes)) != len(hashes):
            raise ValueError("considered forecast hashes must be sorted and unique")
        object.__setattr__(self, "considered_forecast_hashes", hashes)
        selected_hash = _optional_nonempty(
            self.selected_forecast_hash,
            "selected_forecast_hash",
        )
        reason = _optional_nonempty(self.refusal_reason, "refusal_reason")
        if self.action == "acquire":
            if (
                phase == "terminal"
                or selected_hash is None
                or selected_hash not in hashes
                or self.selected_role not in _ROLES_BY_PHASE[phase]
                or reason is not None
            ):
                raise ValueError("acquisition receipt has an incoherent selection")
        elif self.action == "refuse_no_legal_action":
            if (
                selected_hash is not None
                or self.selected_role is not None
                or reason is None
            ):
                raise ValueError("refusal receipt must carry only a refusal reason")
        else:
            raise ValueError("unknown joint-acquisition schedule action")
        object.__setattr__(self, "selected_forecast_hash", selected_hash)
        object.__setattr__(self, "refusal_reason", reason)
        if (
            self.forecast_information_scope != _FORECAST_INFORMATION_SCOPE
            or self.outcome_independent is not True
            or self.precision_coordination_only is not True
            or self.oracle_hard_gate is not True
            or self.loss_sparsity_trade_allowed is not False
            or self.mask_commit_authority is not False
            or self.schema != _RECEIPT_SCHEMA
        ):
            raise ValueError("schedule receipt crossed its no-commit oracle firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-joint-acquisition-schedule-",
        )


@dataclass(frozen=True, slots=True)
class V1ProposalAcquisitionUpdateReceipt:
    """Outcome-bound, train-only update following one proposal acquisition.

    This receipt is deliberately distinct from the scheduler receipt.  The
    scheduler chooses an outcome-independent physical acquisition; this object binds
    the returned outcome *by hash* to the exact request, source, train draws,
    and sufficient-state transition.  It neither selects nor freezes a
    candidate and cannot mutate or commit a mask.
    """

    prior_joint_state_hash: str
    prior_phase_evidence_state_hash: str
    prior_logical_sequence: int
    scheduler_receipt_hash: str
    selected_forecast_hash: str
    selected_acquisition_role: V1JointAcquisitionRole
    selected_evidence_roles: tuple[V1ProposalEvidenceRole, ...]
    acquisition_units: int
    acquired_train_context_ids: tuple[str, ...]
    acquired_train_draw_ids: tuple[str, ...]
    acquired_train_context_draw_ids_hash: str
    request_hash: str
    source_hash: str
    outcome_hash: str
    prior_sufficient_state_hash: str
    updated_sufficient_state_hash: str
    updated_phase_evidence_state_hash: str
    updated_logical_sequence: int
    updated_joint_state_hash: str
    phase: V1JointAcquisitionPhase = "proposal"
    outcome_bound: bool = True
    scheduler_was_outcome_independent: bool = True
    candidate_or_mask_authority: bool = False
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    schema: str = _UPDATE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "prior_joint_state_hash",
            "prior_phase_evidence_state_hash",
            "scheduler_receipt_hash",
            "selected_forecast_hash",
            "acquired_train_context_draw_ids_hash",
            "request_hash",
            "source_hash",
            "outcome_hash",
            "prior_sufficient_state_hash",
            "updated_sufficient_state_hash",
            "updated_phase_evidence_state_hash",
            "updated_joint_state_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        prior_sequence = _nonnegative_int(
            self.prior_logical_sequence,
            "prior_logical_sequence",
        )
        updated_sequence = _nonnegative_int(
            self.updated_logical_sequence,
            "updated_logical_sequence",
        )
        if updated_sequence != prior_sequence + 1:
            raise ValueError(
                "proposal update must advance logical_sequence exactly once"
            )
        object.__setattr__(self, "prior_logical_sequence", prior_sequence)
        object.__setattr__(self, "updated_logical_sequence", updated_sequence)
        if self.phase != "proposal":
            raise ValueError(
                "proposal acquisition updates are legal only in proposal phase"
            )
        expected_roles = _PROPOSAL_EVIDENCE_ROLES_BY_ACQUISITION.get(
            self.selected_acquisition_role
        )
        if expected_roles is None:
            raise ValueError("proposal update requires a proposal acquisition role")
        roles = tuple(self.selected_evidence_roles)
        if roles != expected_roles:
            raise ValueError(
                "selected evidence roles must exactly disclose the acquisition role"
            )
        object.__setattr__(self, "selected_evidence_roles", roles)
        units = _positive_int(self.acquisition_units, "acquisition_units")
        object.__setattr__(self, "acquisition_units", units)
        context_ids = _nonempty_string_tuple(
            self.acquired_train_context_ids,
            "acquired_train_context_ids",
        )
        draw_ids = _nonempty_string_tuple(
            self.acquired_train_draw_ids,
            "acquired_train_draw_ids",
        )
        if len(context_ids) != units or len(draw_ids) != units:
            raise ValueError(
                "train context/draw identities must match acquisition_units"
            )
        if len(set(draw_ids)) != len(draw_ids):
            raise ValueError("train draw identities must be unique")
        pairs = tuple(zip(context_ids, draw_ids, strict=True))
        if len(set(pairs)) != len(pairs):
            raise ValueError("train context/draw pairs must be unique")
        expected_ids_hash = _context_draw_ids_hash(context_ids, draw_ids)
        if self.acquired_train_context_draw_ids_hash != expected_ids_hash:
            raise ValueError("train context/draw identity hash does not match IDs")
        object.__setattr__(self, "acquired_train_context_ids", context_ids)
        object.__setattr__(self, "acquired_train_draw_ids", draw_ids)
        if self.prior_sufficient_state_hash == self.updated_sufficient_state_hash:
            raise ValueError(
                "an acquisition must advance its sufficient-state identity"
            )
        if (
            self.prior_phase_evidence_state_hash
            == self.updated_phase_evidence_state_hash
        ):
            raise ValueError("an acquisition must advance its phase-evidence identity")
        if (
            self.outcome_bound is not True
            or self.scheduler_was_outcome_independent is not True
            or self.candidate_or_mask_authority is not False
            or self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.schema != _UPDATE_RECEIPT_SCHEMA
        ):
            raise ValueError(
                "proposal update crossed its outcome or authority firewall"
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-proposal-acquisition-update-",
        )


@dataclass(frozen=True, slots=True)
class V1ProposalAcquisitionReplayStep:
    """Complete predictable and outcome-bound graph for one proposal update."""

    considered_forecasts: tuple[V1NextActionAcquisitionForecast, ...]
    selected_forecast: V1NextActionAcquisitionForecast
    scheduler_receipt: V1JointAcquisitionScheduleReceipt
    update_receipt: V1ProposalAcquisitionUpdateReceipt
    schema: str = _REPLAY_STEP_SCHEMA

    def __post_init__(self) -> None:
        forecasts = tuple(self.considered_forecasts)
        if not forecasts or any(
            type(item) is not V1NextActionAcquisitionForecast for item in forecasts
        ):
            raise TypeError(
                "proposal replay requires a nonempty exact forecast tuple"
            )
        if type(self.selected_forecast) is not V1NextActionAcquisitionForecast:
            raise TypeError("proposal replay requires an exact selected forecast")
        if type(self.scheduler_receipt) is not V1JointAcquisitionScheduleReceipt:
            raise TypeError("proposal replay requires an exact scheduler receipt")
        if type(self.update_receipt) is not V1ProposalAcquisitionUpdateReceipt:
            raise TypeError("proposal replay requires an exact update receipt")
        forecasts = tuple(sorted(forecasts, key=lambda item: item.content_hash))
        forecast_hashes = tuple(item.content_hash for item in forecasts)
        if len(set(forecast_hashes)) != len(forecast_hashes):
            raise ValueError("proposal replay forecasts must be distinct")
        if self.selected_forecast.content_hash not in forecast_hashes:
            raise ValueError("selected forecast is absent from the replay forecast set")
        if (
            self.scheduler_receipt.selected_forecast_hash
            != self.selected_forecast.content_hash
            or self.update_receipt.selected_forecast_hash
            != self.selected_forecast.content_hash
            or self.update_receipt.scheduler_receipt_hash
            != self.scheduler_receipt.receipt_hash
        ):
            raise ValueError("proposal replay step has a broken receipt graph")
        if self.schema != _REPLAY_STEP_SCHEMA:
            raise ValueError("unsupported proposal replay-step schema")
        object.__setattr__(self, "considered_forecasts", forecasts)

    @property
    def content_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-proposal-acquisition-replay-step-",
        )


def _proposal_receipt_graph_hash(
    *,
    initial_joint_state_hash: str,
    current_joint_state_hash: str,
    current_phase_evidence_state_hash: str,
    current_logical_sequence: int,
    current_sufficient_state_hash: str,
    initialization_receipt_hash: str,
    replay_step_hashes: tuple[str, ...],
    update_receipt_hashes: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_proposal_receipt_graph_v1",
            "initial_joint_state_hash": initial_joint_state_hash,
            "current_joint_state_hash": current_joint_state_hash,
            "current_phase_evidence_state_hash": (
                current_phase_evidence_state_hash
            ),
            "current_logical_sequence": current_logical_sequence,
            "current_sufficient_state_hash": current_sufficient_state_hash,
            "initialization_receipt_hash": initialization_receipt_hash,
            "replay_step_hashes": replay_step_hashes,
            "update_receipt_hashes": update_receipt_hashes,
        },
        prefix="partition-free-v1-proposal-receipt-graph-",
    )


@dataclass(frozen=True, slots=True, init=False)
class V1VerifiedProposalEvidenceState:
    """Factory-issued capability proving deterministic proposal-graph replay."""

    initial_joint_state_hash: str
    current_joint_state_hash: str
    current_phase_evidence_state_hash: str
    current_logical_sequence: int
    current_sufficient_state_hash: str
    initialization_receipt_hash: str
    replay_step_hashes: tuple[str, ...]
    update_receipt_hashes: tuple[str, ...]
    verified_receipt_graph_hash: str
    receipt_graph_replayed: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _VERIFIED_EVIDENCE_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _VERIFIED_EVIDENCE_TOKEN:
            raise TypeError(
                "verified proposal evidence can only be issued by deterministic replay"
            )
        expected = tuple(
            name
            for name in self.__dataclass_fields__
            if name
            not in {
                "receipt_graph_replayed",
                "candidate_or_mask_authority",
                "schema",
            }
        )
        if set(values) != set(expected):
            raise TypeError("verified proposal-evidence fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "receipt_graph_replayed", True)
        object.__setattr__(self, "candidate_or_mask_authority", False)
        object.__setattr__(self, "schema", _VERIFIED_EVIDENCE_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "initial_joint_state_hash",
            "current_joint_state_hash",
            "current_phase_evidence_state_hash",
            "current_sufficient_state_hash",
            "initialization_receipt_hash",
            "verified_receipt_graph_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        sequence = _nonnegative_int(
            self.current_logical_sequence,
            "current_logical_sequence",
        )
        step_hashes = tuple(
            _nonempty(value, "replay_step_hash") for value in self.replay_step_hashes
        )
        update_hashes = tuple(
            _nonempty(value, "update_receipt_hash")
            for value in self.update_receipt_hashes
        )
        if len(step_hashes) != len(update_hashes):
            raise ValueError("replay-step and update-receipt axes must align")
        expected_graph_hash = _proposal_receipt_graph_hash(
            initial_joint_state_hash=self.initial_joint_state_hash,
            current_joint_state_hash=self.current_joint_state_hash,
            current_phase_evidence_state_hash=(
                self.current_phase_evidence_state_hash
            ),
            current_logical_sequence=sequence,
            current_sufficient_state_hash=self.current_sufficient_state_hash,
            initialization_receipt_hash=self.initialization_receipt_hash,
            replay_step_hashes=step_hashes,
            update_receipt_hashes=update_hashes,
        )
        if self.verified_receipt_graph_hash != expected_graph_hash:
            raise ValueError("verified proposal receipt-graph hash is inconsistent")
        if (
            self.receipt_graph_replayed is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _VERIFIED_EVIDENCE_SCHEMA
        ):
            raise ValueError("verified proposal evidence crossed its authority boundary")
        object.__setattr__(self, "current_logical_sequence", sequence)
        object.__setattr__(self, "replay_step_hashes", step_hashes)
        object.__setattr__(self, "update_receipt_hashes", update_hashes)

    @property
    def content_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-verified-proposal-evidence-",
        )


@dataclass(frozen=True, slots=True, init=False)
class V1ProposalFreezeEvidence:
    """Factory-issued external criterion result bound to replayed evidence.

    The joint controller does not know the domain-specific freeze functional.
    A production evaluator must replay that functional and provide a
    content-addressed verification receipt.  This object binds that external
    verification to the in-process replay capability; it does not claim that
    a bare caller-supplied boolean proves the criterion.
    """

    state_hash: str
    phase_evidence_state_hash: str
    logical_sequence: int
    sufficient_state_hash: str
    verified_receipt_graph_hash: str
    verified_evidence_state_hash: str
    freeze_criterion_hash: str
    criterion_evaluator_identity_hash: str
    criterion_verification_receipt_hash: str
    evidence_evaluation_hash: str
    criterion_satisfied: bool
    proposed_candidate_identity_hash: str | None
    phase: V1JointAcquisitionPhase = "proposal"
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    candidate_or_mask_authority: bool = False
    external_criterion_verification_required: bool = True
    schema: str = _FREEZE_EVIDENCE_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _FREEZE_EVIDENCE_TOKEN:
            raise TypeError(
                "proposal freeze evidence can only be issued from replay-bound "
                "external criterion verification"
            )
        expected = tuple(
            name
            for name in self.__dataclass_fields__
            if name
            not in {
                "phase",
                "validation_outcomes_observed",
                "train_optimization_data_only",
                "candidate_or_mask_authority",
                "external_criterion_verification_required",
                "schema",
            }
        )
        if set(values) != set(expected):
            raise TypeError("proposal freeze-evidence fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "phase", "proposal")
        object.__setattr__(self, "validation_outcomes_observed", False)
        object.__setattr__(self, "train_optimization_data_only", True)
        object.__setattr__(self, "candidate_or_mask_authority", False)
        object.__setattr__(self, "external_criterion_verification_required", True)
        object.__setattr__(self, "schema", _FREEZE_EVIDENCE_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "state_hash",
            "phase_evidence_state_hash",
            "sufficient_state_hash",
            "verified_receipt_graph_hash",
            "verified_evidence_state_hash",
            "freeze_criterion_hash",
            "criterion_evaluator_identity_hash",
            "criterion_verification_receipt_hash",
            "evidence_evaluation_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        if self.phase != "proposal":
            raise ValueError(
                "candidate-freeze evidence is legal only in proposal phase"
            )
        if type(self.criterion_satisfied) is not bool:
            raise TypeError("criterion_satisfied must be a boolean")
        candidate = _optional_nonempty(
            self.proposed_candidate_identity_hash,
            "proposed_candidate_identity_hash",
        )
        if self.criterion_satisfied is True and candidate is None:
            raise ValueError("satisfied freeze evidence requires a proposed candidate")
        if self.criterion_satisfied is False and candidate is not None:
            raise ValueError("unsatisfied freeze evidence cannot name a candidate")
        object.__setattr__(self, "proposed_candidate_identity_hash", candidate)
        if (
            self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.candidate_or_mask_authority is not False
            or self.external_criterion_verification_required is not True
            or self.schema != _FREEZE_EVIDENCE_SCHEMA
        ):
            raise ValueError(
                "freeze evidence crossed its train-only authority firewall"
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-proposal-freeze-evidence-")


@dataclass(frozen=True, slots=True)
class V1ProposalFreezeDecisionReceipt:
    """Evidence-bound candidate-freeze authorization or typed refusal."""

    state_hash: str
    phase_evidence_state_hash: str
    logical_sequence: int
    sufficient_state_hash: str
    verified_receipt_graph_hash: str
    freeze_criterion_hash: str
    criterion_evaluator_identity_hash: str
    criterion_verification_receipt_hash: str
    evidence_hash: str
    evidence_evaluation_hash: str
    action: V1ProposalFreezeAction
    candidate_identity_hash: str | None
    refusal_reason: str | None
    phase: V1JointAcquisitionPhase = "proposal"
    scheduler_selection_authorizes_freeze: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    schema: str = _FREEZE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "state_hash",
            "phase_evidence_state_hash",
            "sufficient_state_hash",
            "verified_receipt_graph_hash",
            "freeze_criterion_hash",
            "criterion_evaluator_identity_hash",
            "criterion_verification_receipt_hash",
            "evidence_hash",
            "evidence_evaluation_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        if self.phase != "proposal":
            raise ValueError(
                "candidate-freeze decisions are legal only in proposal phase"
            )
        candidate = _optional_nonempty(
            self.candidate_identity_hash,
            "candidate_identity_hash",
        )
        reason = _optional_nonempty(self.refusal_reason, "refusal_reason")
        if self.action == "authorize_candidate_freeze":
            if (
                candidate is None
                or reason is not None
                or self.candidate_freeze_authority is not True
            ):
                raise ValueError(
                    "freeze authorization requires only a candidate identity"
                )
        elif self.action == "refuse_candidate_freeze":
            if (
                candidate is not None
                or reason is None
                or self.candidate_freeze_authority is not False
            ):
                raise ValueError("freeze refusal requires only a refusal reason")
        else:
            raise ValueError("unknown proposal freeze action")
        object.__setattr__(self, "candidate_identity_hash", candidate)
        object.__setattr__(self, "refusal_reason", reason)
        if (
            self.scheduler_selection_authorizes_freeze is not False
            or self.mask_commit_authority is not False
            or self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.schema != _FREEZE_DECISION_SCHEMA
        ):
            raise ValueError("freeze decision crossed its scheduler or mask firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-proposal-freeze-decision-",
        )


def _assert_forecast_bound(
    state: V1JointAcquisitionState,
    forecast: V1NextActionAcquisitionForecast,
) -> None:
    if forecast.state_hash != state.content_hash:
        raise ValueError("acquisition forecast is not bound to the current state")
    if forecast.phase != state.phase:
        raise ValueError("acquisition forecast phase differs from current state")
    if forecast.action_regret_metric_hash != state.action_regret_metric_hash:
        raise ValueError("acquisition forecasts must share the state's regret metric")
    if forecast.phase_evidence_state_hash != state.phase_evidence_state_hash:
        raise ValueError(
            "acquisition forecast evidence-state identity differs from current state"
        )
    if forecast.candidate_identity_hash != state.candidate_identity_hash:
        raise ValueError("acquisition forecast candidate identity differs from state")
    if forecast.frozen_calibration_hash != state.frozen_calibration_hash:
        raise ValueError("acquisition forecast calibration identity differs from state")


def _exact_utility_per_device_second(
    forecast: V1NextActionAcquisitionForecast,
) -> Fraction:
    """Compare binary64 forecast ratios without overflow or underflow.

    Forecast coordinates are already frozen as finite binary64 values.  Their
    exact rational images therefore preserve the registered numbers while
    avoiding a rounded division in the authority-bearing selection key.
    """

    return Fraction.from_float(
        float(forecast.predicted_action_regret_contraction)
    ) / Fraction.from_float(float(forecast.predicted_device_seconds))


def schedule_v1_joint_acquisition(
    state: V1JointAcquisitionState,
    forecasts: Sequence[V1NextActionAcquisitionForecast],
) -> V1JointAcquisitionScheduleReceipt:
    """Select the predictable legal measurement with greatest contraction/cost.

    The function has no outcome argument and emits no accept/reject or mask
    authority.  Zero-contraction forecasts remain valid audit artifacts but do
    not justify a positive-cost acquisition.  Input ordering cannot affect the
    selected forecast or receipt identity.
    """

    if type(state) is not V1JointAcquisitionState:
        raise TypeError("joint acquisition scheduling requires exact state type")
    options = tuple(forecasts)
    if any(type(item) is not V1NextActionAcquisitionForecast for item in options):
        raise TypeError("joint acquisition scheduling requires exact forecast types")
    for item in options:
        _assert_forecast_bound(state, item)
    option_hashes = tuple(sorted(item.content_hash for item in options))
    if len(set(option_hashes)) != len(option_hashes):
        raise ValueError("duplicate acquisition forecasts are not distinct options")
    if state.phase == "terminal" and options:
        raise ValueError("terminal phase forbids acquisition forecasts")

    positive_options = tuple(
        item
        for item in options
        if float(item.predicted_action_regret_contraction) > 0.0
    )
    if not positive_options:
        reason = (
            "terminal_phase_has_no_legal_acquisition"
            if state.phase == "terminal"
            else (
                "no_legal_acquisition_forecast"
                if not options
                else "no_positive_action_regret_contraction"
            )
        )
        return V1JointAcquisitionScheduleReceipt(
            state_hash=state.content_hash,
            phase=state.phase,
            logical_sequence=state.logical_sequence,
            action_regret_metric_hash=state.action_regret_metric_hash,
            phase_evidence_state_hash=state.phase_evidence_state_hash,
            candidate_identity_hash=state.candidate_identity_hash,
            frozen_calibration_hash=state.frozen_calibration_hash,
            action="refuse_no_legal_action",
            considered_forecast_hashes=option_hashes,
            selected_forecast_hash=None,
            selected_role=None,
            refusal_reason=reason,
        )

    selected = sorted(
        positive_options,
        key=lambda item: (
            -_exact_utility_per_device_second(item),
            -float(item.predicted_action_regret_contraction),
            _ROLE_ORDER[item.role],
            int(item.acquisition_units),
            item.content_hash,
        ),
    )[0]
    return V1JointAcquisitionScheduleReceipt(
        state_hash=state.content_hash,
        phase=state.phase,
        logical_sequence=state.logical_sequence,
        action_regret_metric_hash=state.action_regret_metric_hash,
        phase_evidence_state_hash=state.phase_evidence_state_hash,
        candidate_identity_hash=state.candidate_identity_hash,
        frozen_calibration_hash=state.frozen_calibration_hash,
        action="acquire",
        considered_forecast_hashes=option_hashes,
        selected_forecast_hash=selected.content_hash,
        selected_role=selected.role,
        refusal_reason=None,
    )


def _assert_schedule_receipt_bound(
    state: V1JointAcquisitionState,
    forecast: V1NextActionAcquisitionForecast,
    receipt: V1JointAcquisitionScheduleReceipt,
    considered_forecasts: Sequence[V1NextActionAcquisitionForecast],
) -> None:
    if receipt.state_hash != state.content_hash:
        raise ValueError("scheduler receipt is stale for the current joint state")
    if (
        receipt.phase != state.phase
        or receipt.logical_sequence != state.logical_sequence
        or receipt.action_regret_metric_hash != state.action_regret_metric_hash
        or receipt.phase_evidence_state_hash != state.phase_evidence_state_hash
        or receipt.candidate_identity_hash != state.candidate_identity_hash
        or receipt.frozen_calibration_hash != state.frozen_calibration_hash
    ):
        raise ValueError(
            "scheduler receipt progress identity differs from current state"
        )
    if receipt.action != "acquire":
        raise ValueError("a refusal receipt cannot execute an acquisition")
    if (
        receipt.selected_forecast_hash != forecast.content_hash
        or receipt.selected_role != forecast.role
    ):
        raise ValueError("scheduler receipt does not select the supplied forecast")
    replayed_receipt = schedule_v1_joint_acquisition(state, considered_forecasts)
    if replayed_receipt.receipt_hash != receipt.receipt_hash:
        raise ValueError(
            "scheduler receipt is not the replayed optimum over considered forecasts"
        )


def record_v1_proposal_acquisition_update(
    *,
    state: V1JointAcquisitionState,
    considered_forecasts: Sequence[V1NextActionAcquisitionForecast],
    forecast: V1NextActionAcquisitionForecast,
    scheduler_receipt: V1JointAcquisitionScheduleReceipt,
    acquired_train_context_ids: Sequence[str],
    acquired_train_draw_ids: Sequence[str],
    request_hash: str,
    source_hash: str,
    outcome_hash: str,
    prior_sufficient_state_hash: str,
    updated_sufficient_state_hash: str,
    updated_phase_evidence_state_hash: str,
) -> V1ProposalAcquisitionUpdateReceipt:
    """Bind a proposal measurement outcome and sufficient-state transition.

    No raw outcome is accepted.  The outcome's content address is bound after
    execution while the scheduler remains a separately hashed, outcome-independent
    input.  The returned receipt carries no candidate or mask authority.
    """

    if type(state) is not V1JointAcquisitionState:
        raise TypeError("proposal updates require exact joint state type")
    if type(forecast) is not V1NextActionAcquisitionForecast:
        raise TypeError("proposal updates require exact forecast type")
    if type(scheduler_receipt) is not V1JointAcquisitionScheduleReceipt:
        raise TypeError("proposal updates require exact scheduler receipt type")
    if state.phase != "proposal":
        raise ValueError(
            "proposal acquisition updates are legal only in proposal phase"
        )
    _assert_forecast_bound(state, forecast)
    _assert_schedule_receipt_bound(
        state,
        forecast,
        scheduler_receipt,
        considered_forecasts,
    )
    if forecast.role not in _PROPOSAL_EVIDENCE_ROLES_BY_ACQUISITION:
        raise ValueError("proposal update requires a proposal acquisition role")
    context_ids = _nonempty_string_tuple(
        acquired_train_context_ids,
        "acquired_train_context_ids",
    )
    draw_ids = _nonempty_string_tuple(
        acquired_train_draw_ids,
        "acquired_train_draw_ids",
    )
    if (
        len(context_ids) != forecast.acquisition_units
        or len(draw_ids) != forecast.acquisition_units
    ):
        raise ValueError(
            "executed train identities must match forecast acquisition_units"
        )
    updated_evidence_hash = _nonempty(
        updated_phase_evidence_state_hash,
        "updated_phase_evidence_state_hash",
    )
    updated_state = replace(
        state,
        logical_sequence=state.logical_sequence + 1,
        phase_evidence_state_hash=updated_evidence_hash,
    )
    return V1ProposalAcquisitionUpdateReceipt(
        prior_joint_state_hash=state.content_hash,
        prior_phase_evidence_state_hash=state.phase_evidence_state_hash,
        prior_logical_sequence=state.logical_sequence,
        scheduler_receipt_hash=scheduler_receipt.receipt_hash,
        selected_forecast_hash=forecast.content_hash,
        selected_acquisition_role=forecast.role,
        selected_evidence_roles=_PROPOSAL_EVIDENCE_ROLES_BY_ACQUISITION[forecast.role],
        acquisition_units=forecast.acquisition_units,
        acquired_train_context_ids=context_ids,
        acquired_train_draw_ids=draw_ids,
        acquired_train_context_draw_ids_hash=_context_draw_ids_hash(
            context_ids,
            draw_ids,
        ),
        request_hash=request_hash,
        source_hash=source_hash,
        outcome_hash=outcome_hash,
        prior_sufficient_state_hash=prior_sufficient_state_hash,
        updated_sufficient_state_hash=updated_sufficient_state_hash,
        updated_phase_evidence_state_hash=updated_evidence_hash,
        updated_logical_sequence=updated_state.logical_sequence,
        updated_joint_state_hash=updated_state.content_hash,
    )


def advance_v1_joint_acquisition_state(
    state: V1JointAcquisitionState,
    update_receipt: V1ProposalAcquisitionUpdateReceipt,
    *,
    considered_forecasts: Sequence[V1NextActionAcquisitionForecast],
    forecast: V1NextActionAcquisitionForecast,
    scheduler_receipt: V1JointAcquisitionScheduleReceipt,
) -> V1JointAcquisitionState:
    """Replay and apply exactly one fresh proposal update.

    The full predictable forecast set is mandatory.  Replaying the scheduler
    here prevents a well-shaped receipt from selecting a non-optimal legal
    forecast and prevents direct receipt construction from bypassing the
    authority-bearing transition boundary.
    """

    if type(state) is not V1JointAcquisitionState:
        raise TypeError("state advancement requires exact joint state type")
    if type(update_receipt) is not V1ProposalAcquisitionUpdateReceipt:
        raise TypeError("state advancement requires exact proposal update receipt type")
    if state.phase != "proposal":
        raise ValueError(
            "proposal acquisition updates are legal only in proposal phase"
        )
    replayed_receipt = record_v1_proposal_acquisition_update(
        state=state,
        considered_forecasts=considered_forecasts,
        forecast=forecast,
        scheduler_receipt=scheduler_receipt,
        acquired_train_context_ids=update_receipt.acquired_train_context_ids,
        acquired_train_draw_ids=update_receipt.acquired_train_draw_ids,
        request_hash=update_receipt.request_hash,
        source_hash=update_receipt.source_hash,
        outcome_hash=update_receipt.outcome_hash,
        prior_sufficient_state_hash=update_receipt.prior_sufficient_state_hash,
        updated_sufficient_state_hash=update_receipt.updated_sufficient_state_hash,
        updated_phase_evidence_state_hash=(
            update_receipt.updated_phase_evidence_state_hash
        ),
    )
    if replayed_receipt.receipt_hash != update_receipt.receipt_hash:
        raise ValueError(
            "proposal update receipt does not match its replayed receipt graph"
        )
    if (
        update_receipt.prior_joint_state_hash != state.content_hash
        or update_receipt.prior_phase_evidence_state_hash
        != state.phase_evidence_state_hash
        or update_receipt.prior_logical_sequence != state.logical_sequence
    ):
        raise ValueError("proposal update receipt is stale for the current joint state")
    updated_state = replace(
        state,
        logical_sequence=update_receipt.updated_logical_sequence,
        phase_evidence_state_hash=update_receipt.updated_phase_evidence_state_hash,
    )
    if updated_state.content_hash != update_receipt.updated_joint_state_hash:
        raise ValueError(
            "proposal update receipt does not bind the resulting joint state"
        )
    return updated_state


def replay_v1_proposal_evidence_state(
    *,
    initial_state: V1JointAcquisitionState,
    initial_sufficient_state_hash: str,
    initialization_receipt_hash: str,
    replay_steps: Sequence[V1ProposalAcquisitionReplayStep],
) -> V1VerifiedProposalEvidenceState:
    """Replay the complete proposal receipt graph and issue a capability.

    The capability is intentionally not serial authority by itself.  Production
    resume reconstructs it by replay after validating the checkpoint and the
    content-addressed request/source/outcome artifacts named by each update.
    """

    if type(initial_state) is not V1JointAcquisitionState:
        raise TypeError("proposal replay requires an exact initial joint state")
    if initial_state.phase != "proposal":
        raise ValueError("proposal replay requires an initial proposal-phase state")
    sufficient_state_hash = _nonempty(
        initial_sufficient_state_hash,
        "initial_sufficient_state_hash",
    )
    initialization_hash = _nonempty(
        initialization_receipt_hash,
        "initialization_receipt_hash",
    )
    steps = tuple(replay_steps)
    if any(type(item) is not V1ProposalAcquisitionReplayStep for item in steps):
        raise TypeError("proposal replay requires exact replay-step types")

    current_state = initial_state
    step_hashes: list[str] = []
    update_hashes: list[str] = []
    for step in steps:
        update = step.update_receipt
        if update.prior_sufficient_state_hash != sufficient_state_hash:
            raise ValueError(
                "proposal replay sufficient-state lineage is discontinuous"
            )
        current_state = advance_v1_joint_acquisition_state(
            current_state,
            update,
            considered_forecasts=step.considered_forecasts,
            forecast=step.selected_forecast,
            scheduler_receipt=step.scheduler_receipt,
        )
        sufficient_state_hash = update.updated_sufficient_state_hash
        step_hashes.append(step.content_hash)
        update_hashes.append(update.receipt_hash)

    replay_step_hashes = tuple(step_hashes)
    update_receipt_hashes = tuple(update_hashes)
    graph_hash = _proposal_receipt_graph_hash(
        initial_joint_state_hash=initial_state.content_hash,
        current_joint_state_hash=current_state.content_hash,
        current_phase_evidence_state_hash=current_state.phase_evidence_state_hash,
        current_logical_sequence=current_state.logical_sequence,
        current_sufficient_state_hash=sufficient_state_hash,
        initialization_receipt_hash=initialization_hash,
        replay_step_hashes=replay_step_hashes,
        update_receipt_hashes=update_receipt_hashes,
    )
    return V1VerifiedProposalEvidenceState(
        _issuer_token=_VERIFIED_EVIDENCE_TOKEN,
        initial_joint_state_hash=initial_state.content_hash,
        current_joint_state_hash=current_state.content_hash,
        current_phase_evidence_state_hash=current_state.phase_evidence_state_hash,
        current_logical_sequence=current_state.logical_sequence,
        current_sufficient_state_hash=sufficient_state_hash,
        initialization_receipt_hash=initialization_hash,
        replay_step_hashes=replay_step_hashes,
        update_receipt_hashes=update_receipt_hashes,
        verified_receipt_graph_hash=graph_hash,
    )


def issue_v1_proposal_freeze_evidence(
    *,
    state: V1JointAcquisitionState,
    verified_evidence_state: V1VerifiedProposalEvidenceState,
    freeze_criterion_hash: str,
    criterion_evaluator_identity_hash: str,
    criterion_verification_receipt_hash: str,
    evidence_evaluation_hash: str,
    criterion_satisfied: bool,
    proposed_candidate_identity_hash: str | None,
) -> V1ProposalFreezeEvidence:
    """Bind an externally replayed freeze evaluation to proposal evidence.

    The named evaluator and its verification receipt are production-seal
    inputs.  Artifact existence and domain-specific criterion replay remain the
    responsibility of that sealed evaluator and the independent verifier; this
    control-plane function prevents a bare, directly constructed boolean from
    carrying candidate-freeze authority.
    """

    if type(state) is not V1JointAcquisitionState:
        raise TypeError("freeze evidence issuance requires an exact joint state")
    if type(verified_evidence_state) is not V1VerifiedProposalEvidenceState:
        raise TypeError(
            "freeze evidence issuance requires replay-verified proposal evidence"
        )
    if state.phase != "proposal":
        raise ValueError("freeze evidence is legal only in proposal phase")
    if (
        verified_evidence_state.current_joint_state_hash != state.content_hash
        or verified_evidence_state.current_phase_evidence_state_hash
        != state.phase_evidence_state_hash
        or verified_evidence_state.current_logical_sequence
        != state.logical_sequence
    ):
        raise ValueError("verified proposal evidence is stale for freeze issuance")
    return V1ProposalFreezeEvidence(
        _issuer_token=_FREEZE_EVIDENCE_TOKEN,
        state_hash=state.content_hash,
        phase_evidence_state_hash=state.phase_evidence_state_hash,
        logical_sequence=state.logical_sequence,
        sufficient_state_hash=(
            verified_evidence_state.current_sufficient_state_hash
        ),
        verified_receipt_graph_hash=(
            verified_evidence_state.verified_receipt_graph_hash
        ),
        verified_evidence_state_hash=verified_evidence_state.content_hash,
        freeze_criterion_hash=freeze_criterion_hash,
        criterion_evaluator_identity_hash=criterion_evaluator_identity_hash,
        criterion_verification_receipt_hash=criterion_verification_receipt_hash,
        evidence_evaluation_hash=evidence_evaluation_hash,
        criterion_satisfied=criterion_satisfied,
        proposed_candidate_identity_hash=proposed_candidate_identity_hash,
    )


def decide_v1_proposal_freeze(
    state: V1JointAcquisitionState,
    evidence: V1ProposalFreezeEvidence,
    *,
    verified_evidence_state: V1VerifiedProposalEvidenceState,
) -> V1ProposalFreezeDecisionReceipt:
    """Issue a candidate-freeze decision from current evidence, never schedule.

    A scheduler receipt is intentionally not an accepted input type.  Selecting
    or even executing another measurement therefore cannot itself authorize a
    candidate freeze; only an evidence evaluation bound to the current
    phase-evidence identity can do so.
    """

    if type(state) is not V1JointAcquisitionState:
        raise TypeError("proposal freeze decisions require exact joint state type")
    if type(evidence) is not V1ProposalFreezeEvidence:
        raise TypeError("proposal freeze decisions require exact freeze evidence type")
    if type(verified_evidence_state) is not V1VerifiedProposalEvidenceState:
        raise TypeError(
            "proposal freeze decisions require replay-verified proposal evidence"
        )
    if state.phase != "proposal":
        raise ValueError("candidate-freeze decisions are legal only in proposal phase")
    if (
        evidence.state_hash != state.content_hash
        or evidence.phase_evidence_state_hash != state.phase_evidence_state_hash
        or evidence.logical_sequence != state.logical_sequence
    ):
        raise ValueError("candidate-freeze evidence is stale for the current state")
    if (
        verified_evidence_state.current_joint_state_hash != state.content_hash
        or verified_evidence_state.current_phase_evidence_state_hash
        != state.phase_evidence_state_hash
        or verified_evidence_state.current_logical_sequence
        != state.logical_sequence
        or evidence.sufficient_state_hash
        != verified_evidence_state.current_sufficient_state_hash
        or evidence.verified_receipt_graph_hash
        != verified_evidence_state.verified_receipt_graph_hash
        or evidence.verified_evidence_state_hash
        != verified_evidence_state.content_hash
    ):
        raise ValueError(
            "freeze evidence is not bound to the replay-verified receipt graph"
        )
    authorized = evidence.criterion_satisfied is True
    return V1ProposalFreezeDecisionReceipt(
        state_hash=state.content_hash,
        phase_evidence_state_hash=state.phase_evidence_state_hash,
        logical_sequence=state.logical_sequence,
        sufficient_state_hash=evidence.sufficient_state_hash,
        verified_receipt_graph_hash=(
            verified_evidence_state.verified_receipt_graph_hash
        ),
        freeze_criterion_hash=evidence.freeze_criterion_hash,
        criterion_evaluator_identity_hash=(
            evidence.criterion_evaluator_identity_hash
        ),
        criterion_verification_receipt_hash=(
            evidence.criterion_verification_receipt_hash
        ),
        evidence_hash=evidence.content_hash,
        evidence_evaluation_hash=evidence.evidence_evaluation_hash,
        action=(
            "authorize_candidate_freeze"
            if authorized
            else "refuse_candidate_freeze"
        ),
        candidate_identity_hash=(
            evidence.proposed_candidate_identity_hash if authorized else None
        ),
        refusal_reason=(
            None if authorized else "proposal_freeze_criterion_not_satisfied"
        ),
        candidate_freeze_authority=authorized,
    )


__all__ = [
    "V1ProposalAcquisitionReplayStep",
    "V1ProposalAcquisitionUpdateReceipt",
    "V1ProposalEvidenceRole",
    "V1ProposalFreezeAction",
    "V1ProposalFreezeDecisionReceipt",
    "V1ProposalFreezeEvidence",
    "V1VerifiedProposalEvidenceState",
    "V1JointAcquisitionPhase",
    "V1JointAcquisitionRole",
    "V1JointAcquisitionScheduleAction",
    "V1JointAcquisitionScheduleReceipt",
    "V1JointAcquisitionState",
    "V1NextActionAcquisitionForecast",
    "advance_v1_joint_acquisition_state",
    "decide_v1_proposal_freeze",
    "issue_v1_proposal_freeze_evidence",
    "record_v1_proposal_acquisition_update",
    "replay_v1_proposal_evidence_state",
    "schedule_v1_joint_acquisition",
]
