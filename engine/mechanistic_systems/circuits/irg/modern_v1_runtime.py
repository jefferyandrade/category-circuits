"""Concrete orchestration, resume, and terminal firewall for modern-v1.

The runtime composes the immutable campaign binding, fresh acquisition factory,
pure controller, direct paired-mask provider, and one post-freeze terminal
reveal.  It exposes an explicit internal development entry point only.  Every
artifact remains ``claim_execution_certified=False`` and the public selector is
untouched.

Logical draw identities, masks, intervals, decisions, and receipts do not
depend on physical batch or prefetch settings.  A sampling plan is checkpointed
*before* its outcome is requested, so interruption and resume reuse the exact
system-entropy draw rather than drawing again.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import stat
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Protocol

import cloudpickle
import numpy as np

from mechanistic_systems.backends.factored_calibration import (
    CalibrationConvergenceError,
)

from .contracts import (
    ContextSelector,
    CostEstimate,
    MeasurementProvenance,
    MeasurementRequest,
    content_hash,
)
from .modern_v1_acquisition import (
    V1FQAuthorizedAcquisition,
    V1RelinearizedAcquisition,
)
from .modern_v1_controller import (
    V1ActiveCandidate,
    V1CandidateCalibrationFreeze,
    V1CandidateDamageThreshold,
    V1ControllerState,
    V1StepSamplingLaw,
    _is_label_native_acquisition,
    advance_v1_controller_calibration_horizon_skip,
    advance_v1_controller_look,
    advance_v1_controller_paired_bce_decision,
    attach_v1_acquisition,
    calibration_freeze_from_certificate,
    freeze_v1_candidate,
    register_v1_exact_pareto_prefix_rung,
    seal_v1_pareto_milestone_after_checkpoint,
    update_v1_controller_paired_bce_state,
)
from .modern_v1_core import MODERN_V1_WORKFLOW_IDENTITY, V1EvidencePacket
from .modern_v1_fqr_runtime_acquisition import V1FQRAuthorizedAcquisition
from .modern_v1_signed_work_runtime_acquisition import (
    V1SignedWorkAuthorizedAcquisition,
    V1SignedWorkAuthorizedAcquisitionV2,
)
from .modern_v1_signed_work_label_native_runtime_acquisition import (
    export_v1_label_native_runtime_plan_checkpoint_payload,
    restore_v1_label_native_runtime_plan_checkpoint_payload,
)
from .modern_v1_fqr_horizon import (
    FQR_NO_CANDIDATE_HORIZON_STOP_REASON,
    V1ControllerProposalHorizonWitness,
    V1FQRNoCandidateHorizonProofReceipt,
    v1_fqr_no_candidate_freeze_evaluation_hash,
    v1_fqr_no_candidate_freeze_verification_hash,
)
from .modern_v1_joint_acquisition import (
    V1JointAcquisitionScheduleReceipt,
    V1JointAcquisitionRole,
    V1JointAcquisitionState,
    V1NextActionAcquisitionForecast,
    V1ProposalAcquisitionReplayStep,
    V1ProposalAcquisitionUpdateReceipt,
    V1ProposalFreezeDecisionReceipt,
    V1ProposalFreezeEvidence,
    V1VerifiedProposalEvidenceState,
    advance_v1_joint_acquisition_state,
    decide_v1_proposal_freeze,
    issue_v1_proposal_freeze_evidence,
    record_v1_proposal_acquisition_update,
    replay_v1_proposal_evidence_state,
    schedule_v1_joint_acquisition,
)
from .modern_v1_pareto_milestones import (
    REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED,
    V1ExactPrefixRungRegistration,
    V1ParetoMilestoneCheckpointBinding,
    V1ParetoMilestonePending,
    V1ParetoMilestoneReceipt,
    V1ParetoMilestoneRegistration,
    V1ScientificHorizonDisposition,
    classify_v1_scientific_horizon,
)
from .modern_v1_paired_bce_acquisition import (
    V1PairedBCEAcquisitionRegistration,
    V1PairedBCEAcquisitionState,
    V1PairedBCEBettingReplay,
    V1PairedBCECalibrationHorizonSkipDecision,
    V1PairedBCEChunkReceipt,
    V1PairedBCEChunkRequest,
    V1PairedBCEDecision,
    commit_paired_bce_chunk,
    decide_paired_bce_acquisition,
    validate_paired_bce_calibration_horizon_refusal_receipt,
)
from .modern_v1_checkpoint_codec import (
    V1RuntimeProposalStateSidecarPayload,
    V1RuntimeProposalStateSidecarReceipt,
    inspect_v1_runtime_checkpoint,
)
from .modern_v1_receipts import (
    V1ControllerSnapshotReceipt,
    V1PerformanceReceipt,
    V1RunCompletionReceipt,
    V1TerminalComparisonFamilyFreezeReceiptV2,
    V1TerminalComparisonFamilyObservationReceiptV2,
    V1TerminalComparisonFamilyReceiptV2,
    V1TerminalComparisonFamilyRequestReceiptV2,
    V1TerminalFreezeReceipt,
    V1TerminalValidationFamilyFreezeReceipt,
    V1TerminalValidationFamilyObservationReceipt,
    V1TerminalValidationFamilyReceipt,
    V1TerminalValidationFamilyRequestReceipt,
    V1TerminalValidationReceipt,
)
from .modern_v1_staged_alpha import (
    V1StagedAlphaDecision,
    V1StagedAlphaEntropyTape,
    V1StagedAlphaSamplingPlan,
    draw_v1_staged_alpha_sampling_plan_from_tape,
    issue_v1_staged_alpha_entropy_tape,
    issue_v1_staged_alpha_packet,
)
from mechanistic_systems.measurement.request_providers import PairedMaskOracleProvider

_RUNTIME_CONFIG_SCHEMA: Final = "partition_free_v1_runtime_execution_config_v1"
_PENDING_SCHEMA: Final = "partition_free_v1_runtime_pending_look_v2"
_PENDING_PAIRED_BCE_CHUNK_SCHEMA: Final = (
    "partition_free_v1_runtime_pending_paired_bce_chunk_v1"
)
_PAIRED_BCE_CHUNK_OUTCOME_SCHEMA: Final = (
    "partition_free_v1_runtime_paired_bce_chunk_outcome_v1"
)
_PROPOSAL_SUFFICIENT_STATE_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_sufficient_state_descriptor_v1"
)
_PROPOSAL_REQUEST_PLAN_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_request_plan_v1"
)
_PROPOSAL_REQUEST_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_acquisition_request_v1"
)
_PROPOSAL_OUTCOME_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_acquisition_outcome_v1"
)
_PROPOSAL_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_acquisition_progress_v2"
)
_PROPOSAL_FREEZE_EVALUATION_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_freeze_evaluation_v1"
)
_COMPLETED_PROPOSAL_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_runtime_completed_proposal_closure_receipt_v1"
)
_COMPLETED_PROPOSAL_CLOSURE_V2_SCHEMA: Final = (
    "partition_free_v1_runtime_completed_proposal_closure_receipt_v2"
)
_COMPLETED_PROPOSAL_CLOSURE_V3_SCHEMA: Final = (
    "partition_free_v1_runtime_completed_proposal_closure_receipt_v3"
)
_COMPLETED_PROPOSAL_CLOSURE_V4_SCHEMA: Final = (
    "partition_free_v1_runtime_completed_proposal_closure_receipt_v4"
)
_PENDING_PROPOSAL_SCHEMA: Final = (
    "partition_free_v1_runtime_pending_proposal_acquisition_v1"
)
_RUNTIME_STATE_SCHEMA: Final = "partition_free_v1_runtime_state_v2"
_LEGACY_CHECKPOINT_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v1"
_CHECKPOINT_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v2"
_CHECKPOINT_V3_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v3"
_CHECKPOINT_JOURNAL_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v1"
_CHECKPOINT_JOURNAL_V2_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v2"
)
_CHECKPOINT_JOURNAL_V3_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v3"
)
_CHECKPOINT_JOURNAL_V4_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v4"
)
_CHECKPOINT_JOURNAL_V5_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v5"
)
_CHECKPOINT_JOURNAL_V6_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v6"
)
_CHECKPOINT_JOURNAL_V7_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_journal_v7"
)
_CHECKPOINT_CONTROLLER_SCHEMA: Final = (
    "partition_free_v1_runtime_checkpoint_controller_projection_v1"
)
_ACQUISITION_SIDECAR_SCHEMA: Final = (
    "partition_free_v1_runtime_acquisition_sidecar_v1"
)
_LEGACY_CHECKPOINT_MAGIC: Final = b"MV1CKPT1"
_CHECKPOINT_MAGIC: Final = b"MV1CKPT2"
_CHECKPOINT_V3_MAGIC: Final = b"MV1CKPT3"
_LIVE_CHECKPOINT_MAGICS: Final = frozenset(
    {
        _LEGACY_CHECKPOINT_MAGIC,
        _CHECKPOINT_MAGIC,
        _CHECKPOINT_V3_MAGIC,
    }
)
_INSPECTED_LIVE_CHECKPOINT_MAGICS: Final = frozenset(
    {
        _CHECKPOINT_MAGIC,
        _CHECKPOINT_V3_MAGIC,
    }
)
_ACQUISITION_SIDECAR_MAGIC: Final = b"MV1ACQ1"
_ACQUISITION_SIDECAR_HEADER_BYTES: Final = 4096
_PROPOSAL_STATE_SIDECAR_MAGIC: Final = b"MV1PROP1"
_PROPOSAL_STATE_SIDECAR_HEADER_BYTES: Final = 4096
_PROPOSAL_STATE_SIDECAR_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_state_sidecar_v2"
)
_PROPOSAL_STATE_CODEC_ID: Final = (
    "modern_v1_cloudpickle_protocol_5_exact_runtime_dto_v2"
)
V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_matrix_v1"
)
_TERMINAL_VALIDATION_FAMILY_MATRIX_MAGIC: Final = b"MV1TERM1"
_TERMINAL_VALIDATION_FAMILY_ARTIFACT_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_artifact_reference_v1"
)
_TERMINAL_VALIDATION_FAMILY_VALUES_PREFIX: Final = (
    "partition-free-v1-terminal-validation-family-values-"
)
_TERMINAL_VALIDATION_FAMILY_LANE_VALUES_PREFIX: Final = (
    "partition-free-v1-terminal-validation-family-lane-values-"
)
V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_matrix_v2"
)
_TERMINAL_COMPARISON_FAMILY_MATRIX_MAGIC_V2: Final = b"MV1TERM2"
_TERMINAL_COMPARISON_FAMILY_ARTIFACT_REFERENCE_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_artifact_reference_v2"
)
_CHECKPOINT_SECURITY_BOUNDARY: Final = (
    "trusted_local_only_do_not_unpickle_untrusted_bytes"
)
_MAX_FRAME_HEADER_BYTES: Final = 1 << 20

from .modern_v1_operator_acquisition import V1OperatorAcquisition

V1RuntimeAcquisition = (
    V1OperatorAcquisition
    | V1RelinearizedAcquisition
    | V1FQAuthorizedAcquisition
    | V1FQRAuthorizedAcquisition
    | V1SignedWorkAuthorizedAcquisition
    | V1SignedWorkAuthorizedAcquisitionV2
)
_RUNTIME_ACQUISITION_TYPES: Final = (
    V1OperatorAcquisition,
    V1RelinearizedAcquisition,
    V1FQAuthorizedAcquisition,
    V1FQRAuthorizedAcquisition,
    V1SignedWorkAuthorizedAcquisition,
    V1SignedWorkAuthorizedAcquisitionV2,
)

AcquisitionFactory = Callable[[V1ControllerState], V1RuntimeAcquisition]
ThresholdFactory = Callable[[V1ControllerState], V1CandidateDamageThreshold]
SamplingLawFactory = Callable[[V1ControllerState], V1StepSamplingLaw]
TerminalValidator = Callable[[str, str], Sequence[float] | np.ndarray]


@dataclass(frozen=True, slots=True)
class V1TerminalValidationFamilyMatrixOutcome:
    """One backend common-panel result before durable publication.

    This value is deliberately not checkpoint authority.  Runtime first
    validates it against the frozen request, atomically publishes the raw
    matrix, and replays those bytes before constructing an observation
    receipt.
    """

    validation_values: np.ndarray
    common_panel_evaluator_receipt_hash: str

    def __post_init__(self) -> None:
        values = np.asarray(self.validation_values)
        if (
            values.ndim != 2
            or values.size == 0
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or bool(np.any(values < 0.0))
        ):
            raise ValueError(
                "terminal family outcome requires a finite nonnegative matrix"
            )
        frozen = np.ascontiguousarray(values, dtype="<f8")
        frozen.setflags(write=False)
        object.__setattr__(self, "validation_values", frozen)
        object.__setattr__(
            self,
            "common_panel_evaluator_receipt_hash",
            _nonempty(
                self.common_panel_evaluator_receipt_hash,
                "common_panel_evaluator_receipt_hash",
            ),
        )


@dataclass(frozen=True, slots=True)
class V1TerminalComparisonFamilyMatrixOutcomeV2:
    """One unrevealed three-role common-panel result before publication."""

    validation_values: np.ndarray
    common_panel_evaluator_receipt_hash: str

    def __post_init__(self) -> None:
        values = np.asarray(self.validation_values)
        if (
            values.ndim != 2
            or values.size == 0
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or bool(np.any(values < 0.0))
        ):
            raise ValueError(
                "terminal comparison outcome requires a finite nonnegative matrix"
            )
        frozen = np.ascontiguousarray(values, dtype="<f8")
        frozen.setflags(write=False)
        object.__setattr__(self, "validation_values", frozen)
        object.__setattr__(
            self,
            "common_panel_evaluator_receipt_hash",
            _nonempty(
                self.common_panel_evaluator_receipt_hash,
                "common_panel_evaluator_receipt_hash",
            ),
        )


TerminalFamilyValidator = Callable[
    [V1TerminalValidationFamilyRequestReceipt],
    V1TerminalValidationFamilyMatrixOutcome,
]
TerminalComparisonFamilyValidatorV2 = Callable[
    [V1TerminalComparisonFamilyRequestReceiptV2],
    V1TerminalComparisonFamilyMatrixOutcomeV2,
]

_CANDIDATE_CALIBRATION_FREEZE_METHOD: Final = "freeze_candidate_calibration"
_CANDIDATE_CALIBRATION_VERIFY_METHOD: Final = "verify_candidate_calibration_freeze"


class V1RuntimeProposalOrderIndeterminate(PermissionError):
    """Typed terminal result for a generation with no legal next acquisition."""

    def __init__(
        self,
        *,
        progress_hash: str,
        scheduler_receipt_hash: str,
        refusal_reason: str,
    ) -> None:
        self.progress_hash = _nonempty(progress_hash, "progress_hash")
        self.scheduler_receipt_hash = _nonempty(
            scheduler_receipt_hash,
            "scheduler_receipt_hash",
        )
        self.refusal_reason = _nonempty(refusal_reason, "refusal_reason")
        super().__init__(
            "proposal_order_indeterminate: staged proposal scheduling has "
            f"terminally refused ({self.refusal_reason})"
        )


class V1RuntimeScientificHorizonIndeterminate(PermissionError):
    """Typed refusal to convert a finite resource stop into terminal science."""

    def __init__(self, disposition: V1ScientificHorizonDisposition) -> None:
        if type(disposition) is not V1ScientificHorizonDisposition:
            raise TypeError("scientific-horizon refusal requires an exact disposition")
        self.stop_reason = disposition.stop_reason
        self.horizon_class = disposition.horizon_class
        self.disposition_hash = disposition.disposition_hash
        super().__init__(
            "scientific_horizon_indeterminate: terminal extraction and validation "
            f"remain closed for stop reason {self.stop_reason!r}"
        )


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _positive(value: object, label: str) -> int:
    result = _nonnegative(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonempty_string_tuple(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    result = tuple(_nonempty(item, label) for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _sha256_hex(value: object, label: str) -> str:
    result = _nonempty(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


@dataclass(frozen=True, slots=True)
class V1RuntimeExecutionConfig:
    """Runtime identity with nonlogical physical packing coordinates separated."""

    executor_identity_hash: str
    source_seal_hash: str
    environment_hash: str
    provider_provenance: MeasurementProvenance
    physical_batch_size: int = 1
    prefetch_depth: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _RUNTIME_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        for name in ("executor_identity_hash", "source_seal_hash", "environment_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.provider_provenance) is not MeasurementProvenance:
            raise TypeError("runtime config requires exact provider provenance")
        object.__setattr__(
            self,
            "physical_batch_size",
            _positive(self.physical_batch_size, "physical_batch_size"),
        )
        object.__setattr__(
            self,
            "prefetch_depth",
            _nonnegative(self.prefetch_depth, "prefetch_depth"),
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "runtime config metadata"))
        from .modern_v1_operator_run_binding import validate_config_binding
        validate_config_binding(self.metadata,self.source_seal_hash)
        if self.schema != _RUNTIME_CONFIG_SCHEMA:
            raise ValueError("unsupported modern-v1 runtime-config schema")

    @property
    def logical_identity_hash(self) -> str:
        """Hash of coordinates allowed to affect logical receipts."""

        return content_hash(
            {
                "schema": self.schema,
                "executor_identity_hash": self.executor_identity_hash,
                "source_seal_hash": self.source_seal_hash,
                "environment_hash": self.environment_hash,
                "provider_provenance": self.provider_provenance,
            },
            prefix="partition-free-v1-runtime-logical-config-",
        )

    @property
    def physical_execution_hash(self) -> str:
        return content_hash(
            {
                "logical_identity_hash": self.logical_identity_hash,
                "physical_batch_size": self.physical_batch_size,
                "prefetch_depth": self.prefetch_depth,
                "metadata": self.metadata,
            },
            prefix="partition-free-v1-runtime-physical-config-",
        )

    @property
    def performance_source_environment_hash(self) -> str:
        """Exact source/environment/packing identity for performance receipts."""

        return content_hash(
            {
                "source_seal_hash": self.source_seal_hash,
                "environment_hash": self.environment_hash,
                "executor_identity_hash": self.executor_identity_hash,
                "runtime_logical_config_hash": self.logical_identity_hash,
                "runtime_physical_config_hash": self.physical_execution_hash,
            },
            prefix="partition-free-v1-performance-source-environment-",
        )


def _draw_ids(controller: V1ControllerState, *, look_index: int, count: int) -> tuple[str, ...]:
    """Logical counter identities; physical context IDs may repeat."""

    return tuple(
        (
            f"{controller.controller_run_id}:attempt={controller.candidate_attempt_count}:"
            f"look={look_index}:draw={index}"
        )
        for index in range(count)
    )


def _active_provider_evaluator(provider: PairedMaskOracleProvider) -> Callable[..., Any]:
    evaluator = (
        provider.parent_aware_evaluator
        if provider.parent_aware_evaluator is not None
        else provider.evaluator
    )
    if evaluator is None:
        raise RuntimeError("paired-mask provider lost its active evaluator")
    return evaluator


def _evaluator_hook(
    provider: PairedMaskOracleProvider,
    name: str,
) -> Callable[..., Any] | None:
    observed = getattr(_active_provider_evaluator(provider), name, None)
    if observed is None:
        return None
    if not callable(observed):
        raise TypeError(f"paired-mask evaluator {name} hook is not callable")
    return observed


def _materialize_candidate_calibration_freeze(
    *,
    state: V1RuntimeState,
    provider: PairedMaskOracleProvider,
) -> V1CandidateCalibrationFreeze | None:
    required = state.config.metadata.get("candidate_calibration_freeze_required") is True
    hook = _evaluator_hook(provider, _CANDIDATE_CALIBRATION_FREEZE_METHOD)
    if hook is None:
        if required:
            raise PermissionError(
                "runtime requires candidate calibration freeze materialization"
            )
        return None
    result = hook(state.controller)
    if type(result) is not V1CandidateCalibrationFreeze:
        raise TypeError(
            "candidate calibration freezer must return an exact typed freeze"
        )
    return result


def _verify_candidate_calibration_pre_oracle(
    *,
    state: V1RuntimeState,
    provider: PairedMaskOracleProvider,
) -> V1CandidateCalibrationFreeze | None:
    active = state.controller.active_candidate
    if active is None:
        raise PermissionError("runtime has no active candidate calibration to verify")
    required = (
        state.config.metadata.get("candidate_calibration_freeze_required") is True
        or active.step_freeze.certificate_freeze.metadata.get(
            "candidate_calibration_freeze_required",
            False,
        )
        is True
    )
    freeze = calibration_freeze_from_certificate(
        active.step_freeze.certificate_freeze,
        required=required,
    )
    if freeze is None:
        return None
    hook = _evaluator_hook(provider, _CANDIDATE_CALIBRATION_VERIFY_METHOD)
    if hook is None:
        raise PermissionError(
            "candidate calibration freeze has no sealed evaluator verifier"
        )
    result = hook(freeze)
    if result is not None:
        raise ValueError("candidate calibration verifier must return None")
    return freeze


def _pending_request_and_evidence(
    *,
    config: V1RuntimeExecutionConfig,
    controller: V1ControllerState,
    active: V1ActiveCandidate,
    look_index: int,
    plan: V1StagedAlphaSamplingPlan,
) -> tuple[MeasurementRequest, V1EvidencePacket]:
    """Reconstruct the only lawful pre-outcome request/evidence pair."""

    freeze = active.step_freeze.certificate_freeze
    calibration_freeze = calibration_freeze_from_certificate(
        freeze,
        required=freeze.metadata.get(
            "candidate_calibration_freeze_required",
            False,
        )
        is True,
    )
    calibration_freeze_hash = (
        None if calibration_freeze is None else calibration_freeze.freeze_hash
    )
    calibration_contract_declared = (
        "candidate_calibration_freeze_required" in freeze.metadata
    )
    if (
        plan.population_context_ids != active.sampling_law.population_context_ids
        or plan.sampling_masses_u64 != active.sampling_law.sampling_masses_u64
    ):
        raise ValueError("sampling plan differs from the frozen exact uint64 law")
    count = len(plan.draw_ids)
    request_payload: dict[str, Any] = {
        "candidate_mask_hash": freeze.candidate_mask_hash,
        "parent_mask_hash": freeze.parent_mask_hash,
        "candidate_definition_hash": freeze.candidate_definition_hash,
        "null_id": freeze.null_id,
        "loss_id": freeze.loss_id,
        "sampling_law_hash": freeze.sampling_law_hash,
        "importance_weight_hash": freeze.importance_weight_hash,
        "certificate_factor_hash": freeze.certificate_factor_hash,
        "pre_draw_filtration_hash": freeze.pre_draw_filtration_hash,
        "stream_role": "calibration_certificate",
        "draw_ids": plan.draw_ids,
        "source_context_ids": plan.source_context_ids,
        "draw_reuse_status": "fresh_candidate_certificate",
        "population_context_ids": active.sampling_law.population_context_ids,
        "target_population_probabilities": (
            active.sampling_law.target_population_probabilities
        ),
        "sampling_population_probabilities": (
            plan.sampling_population_probabilities
        ),
        "sampling_plan_hash": plan.plan_hash,
        "runtime_logical_config_hash": config.logical_identity_hash,
    }
    evidence_metadata: dict[str, Any] = {
        "sampling_plan_hash": plan.plan_hash,
        "runtime_logical_config_hash": config.logical_identity_hash,
        "outcome_state": "not_requested",
    }
    if calibration_contract_declared:
        calibration_lineage = {
            "candidate_calibration_freeze_hash": calibration_freeze_hash,
            "candidate_calibration_freeze_required": (
                calibration_freeze is not None
            ),
        }
        request_payload.update(calibration_lineage)
        evidence_metadata.update(calibration_lineage)
    request = MeasurementRequest.create(
        kind="hard_mask_oracle",
        state_hash=freeze.parent_mask_hash,
        version=MODERN_V1_WORKFLOW_IDENTITY,
        context_selector=ContextSelector(
            selector_id=(
                f"{controller.controller_run_id}-attempt-{controller.candidate_attempt_count}"
                f"-look-{look_index}"
            ),
            requested_count=count,
        ),
        candidate_ids=(freeze.candidate_mask_hash,),
        expected_cost=CostEstimate(forward_passes=2),
        provenance=config.provider_provenance,
        payload=request_payload,
    )
    evidence = V1EvidencePacket(
        packet_id=(
            f"{controller.controller_run_id}-attempt-{controller.candidate_attempt_count}"
            f"-look-{look_index}-packet"
        ),
        role="calibration_certificate",
        draw_ids=plan.draw_ids,
        source_context_ids=plan.source_context_ids,
        registration_hash=freeze.registration_hash,
        parent_mask_hash=freeze.parent_mask_hash,
        candidate_mask_hash=freeze.candidate_mask_hash,
        candidate_definition_hash=freeze.candidate_definition_hash,
        null_id=freeze.null_id,
        loss_id=freeze.loss_id,
        sampling_law_hash=freeze.sampling_law_hash,
        importance_weight_hash=freeze.importance_weight_hash,
        certificate_factor_hash=freeze.certificate_factor_hash,
        pre_draw_filtration_hash=freeze.pre_draw_filtration_hash,
        outcome_observed=False,
        metadata=evidence_metadata,
    )
    return request, evidence


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalSufficientStateDescriptor:
    """Small durable reference to one backend-owned proposal sufficient state.

    Route-wide arrays are intentionally not embedded here.  A production
    backend can name an immutable framed sidecar in ``storage_descriptor``;
    tests and small synthetic backends can use an in-memory content address.
    The runtime treats both representations identically and never infers
    authority from storage location alone.
    """

    sufficient_state_hash: str
    representation_hash: str
    storage_descriptor: Mapping[str, Any]
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _PROPOSAL_SUFFICIENT_STATE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("sufficient_state_hash", "representation_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not isinstance(self.storage_descriptor, Mapping):
            raise TypeError("proposal sufficient-state storage descriptor must be a mapping")
        object.__setattr__(
            self,
            "storage_descriptor",
            _freeze(self.storage_descriptor, "proposal sufficient-state storage descriptor"),
        )
        if (
            self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _PROPOSAL_SUFFICIENT_STATE_SCHEMA
        ):
            raise ValueError("proposal sufficient state crossed its train-only authority firewall")

    @property
    def descriptor_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-sufficient-state-",
        )

    def __hash__(self) -> int:
        return hash(self.descriptor_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalRequestPlan:
    """Outcome-independent train draw/source plan supplied by a proposal backend."""

    train_context_ids: tuple[str, ...]
    train_draw_ids: tuple[str, ...]
    source_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    outcome_independent: bool = True
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _PROPOSAL_REQUEST_PLAN_SCHEMA

    def __post_init__(self) -> None:
        contexts = _nonempty_string_tuple(self.train_context_ids, "train_context_ids")
        draws = _nonempty_string_tuple(self.train_draw_ids, "train_draw_ids")
        if len(contexts) != len(draws):
            raise ValueError("proposal train context/draw plans must have equal length")
        if len(set(draws)) != len(draws):
            raise ValueError("proposal train draw IDs must be globally unique within a plan")
        object.__setattr__(self, "train_context_ids", contexts)
        object.__setattr__(self, "train_draw_ids", draws)
        object.__setattr__(self, "source_hash", _nonempty(self.source_hash, "source_hash"))
        if not isinstance(self.payload, Mapping):
            raise TypeError("proposal request-plan payload must be a mapping")
        object.__setattr__(self, "payload", _freeze(self.payload, "proposal request-plan payload"))
        if (
            self.outcome_independent is not True
            or self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _PROPOSAL_REQUEST_PLAN_SCHEMA
        ):
            raise ValueError("proposal request plan crossed its outcome or authority firewall")

    @property
    def acquisition_units(self) -> int:
        return len(self.train_draw_ids)

    @property
    def plan_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-request-plan-",
        )

    def __hash__(self) -> int:
        return hash(self.plan_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalAcquisitionRequest:
    """Exact persisted proposal request issued before any outcome exists."""

    campaign_binding_hash: str
    controller_run_id: str
    controller_state_hash: str
    controller_snapshot_hash: str
    current_mask_hash: str
    acquisition_generation_index: int
    mask_lineage_hash: str
    prior_progress_hash: str
    prior_joint_state_hash: str
    prior_phase_evidence_state_hash: str
    prior_logical_sequence: int
    prior_sufficient_state_hash: str
    prior_sufficient_state_descriptor_hash: str
    forecast_set_hash: str
    scheduler_receipt_hash: str
    selected_forecast_hash: str
    selected_role: V1JointAcquisitionRole
    acquisition_units: int
    train_context_ids: tuple[str, ...]
    train_draw_ids: tuple[str, ...]
    source_hash: str
    request_plan_hash: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    outcome_independent: bool = True
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _PROPOSAL_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_binding_hash",
            "controller_run_id",
            "controller_state_hash",
            "controller_snapshot_hash",
            "current_mask_hash",
            "mask_lineage_hash",
            "prior_progress_hash",
            "prior_joint_state_hash",
            "prior_phase_evidence_state_hash",
            "prior_sufficient_state_hash",
            "prior_sufficient_state_descriptor_hash",
            "forecast_set_hash",
            "scheduler_receipt_hash",
            "selected_forecast_hash",
            "selected_role",
            "source_hash",
            "request_plan_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "acquisition_generation_index",
            _nonnegative(self.acquisition_generation_index, "acquisition_generation_index"),
        )
        object.__setattr__(
            self,
            "prior_logical_sequence",
            _nonnegative(self.prior_logical_sequence, "prior_logical_sequence"),
        )
        units = _positive(self.acquisition_units, "acquisition_units")
        object.__setattr__(self, "acquisition_units", units)
        contexts = _nonempty_string_tuple(self.train_context_ids, "train_context_ids")
        draws = _nonempty_string_tuple(self.train_draw_ids, "train_draw_ids")
        if len(contexts) != units or len(draws) != units:
            raise ValueError("proposal request draw identities must match acquisition_units")
        if len(set(draws)) != len(draws):
            raise ValueError("proposal request train draw IDs must be unique")
        object.__setattr__(self, "train_context_ids", contexts)
        object.__setattr__(self, "train_draw_ids", draws)
        if not isinstance(self.payload, Mapping):
            raise TypeError("proposal request payload must be a mapping")
        object.__setattr__(self, "payload", _freeze(self.payload, "proposal request payload"))
        if (
            self.outcome_independent is not True
            or self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _PROPOSAL_REQUEST_SCHEMA
        ):
            raise ValueError("proposal request crossed its pre-outcome authority firewall")

    @property
    def request_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-request-",
        )

    def __hash__(self) -> int:
        return hash(self.request_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalAcquisitionOutcome:
    """Content addresses returned by one train-only proposal execution."""

    request_hash: str
    source_hash: str
    outcome_hash: str
    prior_sufficient_state_hash: str
    executed_train_context_ids: tuple[str, ...]
    executed_train_draw_ids: tuple[str, ...]
    updated_sufficient_state: V1RuntimeProposalSufficientStateDescriptor
    updated_phase_evidence_state_hash: str
    outcome_bound: bool = True
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _PROPOSAL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "source_hash",
            "outcome_hash",
            "prior_sufficient_state_hash",
            "updated_phase_evidence_state_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        contexts = _nonempty_string_tuple(
            self.executed_train_context_ids,
            "executed_train_context_ids",
        )
        draws = _nonempty_string_tuple(
            self.executed_train_draw_ids,
            "executed_train_draw_ids",
        )
        if len(contexts) != len(draws) or len(set(draws)) != len(draws):
            raise ValueError("executed proposal train draw identities are invalid")
        object.__setattr__(self, "executed_train_context_ids", contexts)
        object.__setattr__(self, "executed_train_draw_ids", draws)
        if type(self.updated_sufficient_state) is not V1RuntimeProposalSufficientStateDescriptor:
            raise TypeError("proposal outcome requires an exact updated sufficient state")
        if self.updated_sufficient_state.sufficient_state_hash == self.prior_sufficient_state_hash:
            raise ValueError("proposal outcome must advance the sufficient-state identity")
        if (
            self.outcome_bound is not True
            or self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _PROPOSAL_OUTCOME_SCHEMA
        ):
            raise ValueError("proposal outcome crossed its train-only authority firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-outcome-",
        )

    def __hash__(self) -> int:
        return hash(self.receipt_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalFreezeEvaluation:
    """External criterion-verification receipt returned by a sealed evaluator."""

    verified_receipt_graph_hash: str
    freeze_criterion_hash: str
    criterion_evaluator_identity_hash: str
    criterion_verification_receipt_hash: str
    evidence_evaluation_hash: str
    criterion_satisfied: bool
    proposed_candidate_identity_hash: str | None
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    mask_commit_authority: bool = False
    schema: str = _PROPOSAL_FREEZE_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "verified_receipt_graph_hash",
            "freeze_criterion_hash",
            "criterion_evaluator_identity_hash",
            "criterion_verification_receipt_hash",
            "evidence_evaluation_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.criterion_satisfied) is not bool:
            raise TypeError("proposal freeze criterion result must be a boolean")
        candidate = self.proposed_candidate_identity_hash
        if candidate is not None:
            candidate = _nonempty(candidate, "proposed_candidate_identity_hash")
        if (candidate is not None) is not self.criterion_satisfied:
            raise ValueError("proposal freeze evaluation candidate identity disagrees with result")
        object.__setattr__(self, "proposed_candidate_identity_hash", candidate)
        if (
            self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.mask_commit_authority is not False
            or self.schema != _PROPOSAL_FREEZE_EVALUATION_SCHEMA
        ):
            raise ValueError("proposal freeze evaluation crossed its authority firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-freeze-evaluation-",
        )

    def __hash__(self) -> int:
        return hash(self.receipt_hash)


def _proposal_mask_lineage_hash(
    mask_lineage_hashes: tuple[str, ...],
    transition_receipt_hashes: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_mask_lineage_v1",
            "mask_lineage_hashes": mask_lineage_hashes,
            "transition_receipt_hashes": transition_receipt_hashes,
        },
        prefix="partition-free-v1-runtime-proposal-mask-lineage-",
    )


def _proposal_candidate_identity_hash(
    *,
    campaign_binding_hash: str,
    generation_index: int,
    current_mask_hash: str,
    committed_mask_lineage_hashes: tuple[str, ...],
    committed_mask_transition_receipt_hashes: tuple[str, ...],
    acquisition_hash: str,
    acquisition_receipt_hash: str,
    ordered_candidate_ids: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_authorized_candidate_queue_v1",
            "campaign_binding_hash": campaign_binding_hash,
            "generation_index": generation_index,
            "current_mask_hash": current_mask_hash,
            "committed_mask_lineage_hashes": committed_mask_lineage_hashes,
            "committed_mask_transition_receipt_hashes": (
                committed_mask_transition_receipt_hashes
            ),
            "acquisition_hash": acquisition_hash,
            "acquisition_receipt_hash": acquisition_receipt_hash,
            "ordered_candidate_ids": ordered_candidate_ids,
        },
        prefix="partition-free-v1-runtime-authorized-candidate-queue-",
    )


def _proposal_closure_controller_state_hash(
    *,
    snapshot: V1ControllerSnapshotReceipt,
    controller_schema: str,
    initial_mask_hash: str,
    acquisition_hash: str | None,
    globally_used_draw_ids: tuple[str, ...],
) -> str:
    expected_draw_hash = content_hash(
        globally_used_draw_ids,
        prefix="partition-free-v1-globally-used-draw-ids-",
    )
    if (
        snapshot.globally_used_draw_ids_hash != expected_draw_hash
        or snapshot.active_candidate_id is not None
        or snapshot.active_step_freeze_hash is not None
        or snapshot.active_ledger_hash is not None
        or (
            snapshot.phase == "needs_acquisition"
            and (
                snapshot.acquisition_receipt_hash is not None
                or acquisition_hash is not None
                or snapshot.stop_reason is not None
            )
        )
        or (
            snapshot.phase == "queue_ready"
            and (
                snapshot.acquisition_receipt_hash is None
                or acquisition_hash is None
                or snapshot.stop_reason is not None
            )
        )
        or (
            snapshot.phase == "stopped"
            and (
                snapshot.acquisition_receipt_hash is not None
                or acquisition_hash is not None
                or snapshot.stop_reason != "global_queue_exhausted"
            )
        )
    ):
        raise ValueError("completed proposal closure snapshot has invalid controller phase")
    projection: dict[str, Any] = {
        "schema": controller_schema,
        "campaign_binding_hash": snapshot.campaign_binding_hash,
        "controller_run_id": snapshot.controller_run_id,
        "initial_mask_hash": initial_mask_hash,
        "current_mask_hash": snapshot.current_mask_hash,
        "phase": snapshot.phase,
        "acquisition_hash": acquisition_hash,
        "queue_cursor": snapshot.queue_cursor,
        "active_candidate_hash": None,
        "accepted_trajectory_hash": snapshot.accepted_trajectory_hash,
        "attempt_receipt_hashes": snapshot.attempt_receipt_hashes,
        "transition_receipt_hashes": snapshot.transition_receipt_hashes,
        "acquisition_receipt_hashes": snapshot.acquisition_receipt_hashes,
        "globally_used_draw_ids": globally_used_draw_ids,
        "stop_reason": snapshot.stop_reason,
        "logical_sequence": snapshot.logical_sequence,
    }
    if snapshot.label_native_runtime_plan_hashes:
        projection.update(
            {
                "label_native_runtime_plan_hash": (
                    snapshot.label_native_runtime_plan_hash
                ),
                "label_native_runtime_plan_hashes": (
                    snapshot.label_native_runtime_plan_hashes
                ),
                "label_native_rejected_plan_indices": (
                    snapshot.label_native_rejected_plan_indices
                ),
                "label_native_selected_acquisition_receipt_hash": (
                    snapshot.label_native_selected_acquisition_receipt_hash
                ),
            }
        )
    return content_hash(
        projection,
        prefix="partition-free-v1-controller-state-",
    )


def _proposal_initialization_receipt_hash(
    *,
    campaign_contract_hash: str,
    campaign_binding_hash: str,
    controller_run_id: str,
    proposal_backend_identity_hash: str,
    controller_state_hash: str,
    controller_snapshot_hash: str,
    controller_logical_sequence: int,
    current_mask_hash: str,
    route_axis_hash: str,
    acquisition_generation_index: int,
    mask_lineage_hash: str,
    action_regret_metric_hash: str,
    freeze_criterion_hash: str,
    initial_joint_state_hash: str,
    initial_sufficient_state_descriptor_hash: str,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_initialization_receipt_v2",
            "campaign_contract_hash": campaign_contract_hash,
            "campaign_binding_hash": campaign_binding_hash,
            "controller_run_id": controller_run_id,
            "proposal_backend_identity_hash": proposal_backend_identity_hash,
            "controller_state_hash": controller_state_hash,
            "controller_snapshot_hash": controller_snapshot_hash,
            "controller_logical_sequence": controller_logical_sequence,
            "current_mask_hash": current_mask_hash,
            "route_axis_hash": route_axis_hash,
            "acquisition_generation_index": acquisition_generation_index,
            "mask_lineage_hash": mask_lineage_hash,
            "action_regret_metric_hash": action_regret_metric_hash,
            "freeze_criterion_hash": freeze_criterion_hash,
            "initial_joint_state_hash": initial_joint_state_hash,
            "initial_sufficient_state_descriptor_hash": (
                initial_sufficient_state_descriptor_hash
            ),
        },
        prefix="partition-free-v1-runtime-proposal-initialization-",
    )


def _proposal_forecast_set_hash(
    forecasts: tuple[V1NextActionAcquisitionForecast, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_forecast_set_v1",
            "ordered_forecast_hashes": tuple(item.content_hash for item in forecasts),
        },
        prefix="partition-free-v1-runtime-proposal-forecast-set-",
    )


def _proposal_initial_frontier_hash(
    *,
    initialization_receipt_hash: str,
    proposal_backend_identity_hash: str,
    joint_state_hash: str,
    sufficient_state_descriptor_hash: str,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_frontier_initial_v2",
            "initialization_receipt_hash": initialization_receipt_hash,
            "proposal_backend_identity_hash": proposal_backend_identity_hash,
            "joint_state_hash": joint_state_hash,
            "sufficient_state_descriptor_hash": sufficient_state_descriptor_hash,
        },
        prefix="partition-free-v1-runtime-proposal-frontier-initial-",
    )


def _proposal_frontier_after_terminal_refusal(
    prior_frontier_hash: str,
    *,
    receipt: V1JointAcquisitionScheduleReceipt,
    forecasts: tuple[V1NextActionAcquisitionForecast, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_frontier_terminal_refusal_v1",
            "prior_frontier_hash": prior_frontier_hash,
            "scheduler_receipt_hash": receipt.receipt_hash,
            "forecast_set_hash": _proposal_forecast_set_hash(forecasts),
            "ordered_forecast_hashes": tuple(
                item.content_hash for item in forecasts
            ),
            "refusal_reason": receipt.refusal_reason,
        },
        prefix="partition-free-v1-runtime-proposal-frontier-terminal-refusal-",
    )


def _proposal_frontier_after_freeze_decision(
    prior_frontier_hash: str,
    *,
    evaluation: V1RuntimeProposalFreezeEvaluation,
    evidence: V1ProposalFreezeEvidence,
    decision: V1ProposalFreezeDecisionReceipt,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_frontier_freeze_event_v1",
            "prior_frontier_hash": prior_frontier_hash,
            "freeze_evaluation_receipt_hash": evaluation.receipt_hash,
            "freeze_evidence_hash": evidence.content_hash,
            "freeze_decision_receipt_hash": decision.receipt_hash,
            "joint_state_hash": decision.state_hash,
            "verified_receipt_graph_hash": decision.verified_receipt_graph_hash,
        },
        prefix="partition-free-v1-runtime-proposal-frontier-freeze-",
    )


def _proposal_frontier_after_update(
    prior_frontier_hash: str,
    *,
    completed_pending: V1RuntimePendingProposalAcquisition,
    outcome: V1RuntimeProposalAcquisitionOutcome,
    replay_step: V1ProposalAcquisitionReplayStep,
    update: V1ProposalAcquisitionUpdateReceipt,
    updated_joint_state_hash: str,
    updated_sufficient_state_descriptor_hash: str,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_frontier_update_event_v1",
            "prior_frontier_hash": prior_frontier_hash,
            "completed_pending_hash": completed_pending.pending_hash,
            "outcome_receipt_hash": outcome.receipt_hash,
            "replay_step_hash": replay_step.content_hash,
            "update_receipt_hash": update.receipt_hash,
            "updated_joint_state_hash": updated_joint_state_hash,
            "updated_sufficient_state_descriptor_hash": (
                updated_sufficient_state_descriptor_hash
            ),
            "new_train_draw_ids": update.acquired_train_draw_ids,
        },
        prefix="partition-free-v1-runtime-proposal-frontier-update-",
    )


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalAcquisitionProgress:
    """Runtime-owned evidence frontier for one relinearization generation."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    controller_state_hash: str
    controller_snapshot_hash: str
    controller_logical_sequence: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_generation_index: int
    mask_lineage_hashes: tuple[str, ...]
    transition_receipt_hashes: tuple[str, ...]
    mask_lineage_hash: str
    action_regret_metric_hash: str
    freeze_criterion_hash: str
    initial_phase_evidence_state_hash: str
    initial_joint_state_hash: str
    initialization_receipt_hash: str
    joint_state: V1JointAcquisitionState
    initial_sufficient_state: V1RuntimeProposalSufficientStateDescriptor
    current_sufficient_state: V1RuntimeProposalSufficientStateDescriptor
    update_receipts: tuple[V1ProposalAcquisitionUpdateReceipt, ...] = ()
    replay_steps: tuple[V1ProposalAcquisitionReplayStep, ...] = ()
    completed_pending_requests: tuple[V1RuntimePendingProposalAcquisition, ...] = ()
    outcome_receipts: tuple[V1RuntimeProposalAcquisitionOutcome, ...] = ()
    freeze_evaluation_receipts: tuple[V1RuntimeProposalFreezeEvaluation, ...] = ()
    freeze_evidence_receipts: tuple[V1ProposalFreezeEvidence, ...] = ()
    freeze_decision_receipts: tuple[V1ProposalFreezeDecisionReceipt, ...] = ()
    proposal_train_draw_ids: tuple[str, ...] = ()
    terminal_refusal_receipt: V1JointAcquisitionScheduleReceipt | None = None
    terminal_refusal_forecasts: tuple[V1NextActionAcquisitionForecast, ...] = ()
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    calibration_authority: bool = False
    validation_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = _PROPOSAL_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "proposal_backend_identity_hash",
            "controller_state_hash",
            "controller_snapshot_hash",
            "current_mask_hash",
            "route_axis_hash",
            "mask_lineage_hash",
            "action_regret_metric_hash",
            "freeze_criterion_hash",
            "initial_phase_evidence_state_hash",
            "initial_joint_state_hash",
            "initialization_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        generation = _nonnegative(
            self.acquisition_generation_index,
            "acquisition_generation_index",
        )
        object.__setattr__(self, "acquisition_generation_index", generation)
        object.__setattr__(
            self,
            "controller_logical_sequence",
            _nonnegative(
                self.controller_logical_sequence,
                "controller_logical_sequence",
            ),
        )
        lineage = _nonempty_string_tuple(
            self.mask_lineage_hashes,
            "mask_lineage_hashes",
        )
        transitions = _nonempty_string_tuple(
            self.transition_receipt_hashes,
            "transition_receipt_hashes",
            allow_empty=True,
        )
        if (
            len(transitions) != len(lineage) - 1
            or generation != len(transitions)
            or lineage[-1] != self.current_mask_hash
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("proposal progress has invalid committed mask ancestry")
        expected_lineage_hash = _proposal_mask_lineage_hash(lineage, transitions)
        if self.mask_lineage_hash != expected_lineage_hash:
            raise ValueError("proposal progress mask-lineage hash differs from its ancestry")
        object.__setattr__(self, "mask_lineage_hashes", lineage)
        object.__setattr__(self, "transition_receipt_hashes", transitions)
        if type(self.joint_state) is not V1JointAcquisitionState:
            raise TypeError("proposal progress requires an exact joint-acquisition state")
        if (
            self.joint_state.phase != "proposal"
            or self.joint_state.campaign_binding_hash != self.campaign_binding_hash
            or self.joint_state.current_mask_hash != self.current_mask_hash
            or self.joint_state.action_regret_metric_hash != self.action_regret_metric_hash
        ):
            raise ValueError("proposal progress joint state has stale campaign/mask semantics")
        for value, label in (
            (self.initial_sufficient_state, "initial sufficient state"),
            (self.current_sufficient_state, "current sufficient state"),
        ):
            if type(value) is not V1RuntimeProposalSufficientStateDescriptor:
                raise TypeError(f"proposal progress {label} has the wrong type")
        initial_joint = replace(
            self.joint_state,
            logical_sequence=0,
            phase_evidence_state_hash=self.initial_phase_evidence_state_hash,
        )
        if initial_joint.content_hash != self.initial_joint_state_hash:
            raise ValueError("proposal progress initial joint-state identity is not reproducible")
        expected_initialization_hash = _proposal_initialization_receipt_hash(
            campaign_contract_hash=self.campaign_contract_hash,
            campaign_binding_hash=self.campaign_binding_hash,
            controller_run_id=self.controller_run_id,
            proposal_backend_identity_hash=self.proposal_backend_identity_hash,
            controller_state_hash=self.controller_state_hash,
            controller_snapshot_hash=self.controller_snapshot_hash,
            controller_logical_sequence=self.controller_logical_sequence,
            current_mask_hash=self.current_mask_hash,
            route_axis_hash=self.route_axis_hash,
            acquisition_generation_index=self.acquisition_generation_index,
            mask_lineage_hash=self.mask_lineage_hash,
            action_regret_metric_hash=self.action_regret_metric_hash,
            freeze_criterion_hash=self.freeze_criterion_hash,
            initial_joint_state_hash=self.initial_joint_state_hash,
            initial_sufficient_state_descriptor_hash=(
                self.initial_sufficient_state.descriptor_hash
            ),
        )
        if self.initialization_receipt_hash != expected_initialization_hash:
            raise ValueError("proposal initialization receipt is not reproducible")
        updates = tuple(self.update_receipts)
        if any(type(item) is not V1ProposalAcquisitionUpdateReceipt for item in updates):
            raise TypeError("proposal progress update history contains a nonreceipt")
        outcomes = tuple(self.outcome_receipts)
        if any(type(item) is not V1RuntimeProposalAcquisitionOutcome for item in outcomes):
            raise TypeError("proposal progress outcome history contains a nonreceipt")
        replay_steps = tuple(self.replay_steps)
        if any(type(item) is not V1ProposalAcquisitionReplayStep for item in replay_steps):
            raise TypeError("proposal progress replay history contains a nonstep")
        completed_pending = tuple(self.completed_pending_requests)
        if any(
            type(item) is not V1RuntimePendingProposalAcquisition
            for item in completed_pending
        ):
            raise TypeError("proposal progress request history contains a nonrequest")
        if not (
            len(outcomes)
            == len(updates)
            == len(replay_steps)
            == len(completed_pending)
        ):
            raise ValueError("proposal progress receipt-graph axes have different lengths")
        for hashes, label in (
            (tuple(item.receipt_hash for item in updates), "update"),
            (tuple(item.content_hash for item in replay_steps), "replay-step"),
            (tuple(item.pending_hash for item in completed_pending), "request"),
            (tuple(item.receipt_hash for item in outcomes), "outcome"),
        ):
            if len(set(hashes)) != len(hashes):
                raise ValueError(f"proposal progress {label} history contains duplicates")
        decisions = tuple(self.freeze_decision_receipts)
        if any(type(item) is not V1ProposalFreezeDecisionReceipt for item in decisions):
            raise TypeError("proposal progress freeze history contains a nonreceipt")
        evaluations = tuple(self.freeze_evaluation_receipts)
        if any(type(item) is not V1RuntimeProposalFreezeEvaluation for item in evaluations):
            raise TypeError("proposal progress freeze history contains a nonevaluation")
        evidence_receipts = tuple(self.freeze_evidence_receipts)
        if any(type(item) is not V1ProposalFreezeEvidence for item in evidence_receipts):
            raise TypeError("proposal progress freeze history contains a nonevidence value")
        if len(evaluations) != len(evidence_receipts) or len(evaluations) != len(decisions):
            raise ValueError("proposal progress freeze receipt-graph axes have different lengths")
        decision_hashes = tuple(item.receipt_hash for item in decisions)
        if len(set(decision_hashes)) != len(decision_hashes):
            raise ValueError("proposal progress freeze decision history contains duplicates")
        state_index_by_hash = {self.initial_joint_state_hash: 0}
        for index, receipt in enumerate(updates, start=1):
            if receipt.updated_joint_state_hash in state_index_by_hash:
                raise ValueError("proposal progress revisited an earlier joint state")
            state_index_by_hash[receipt.updated_joint_state_hash] = index
        freeze_indices_by_state_index: dict[int, list[int]] = {}
        for index, decision in enumerate(decisions):
            state_index = state_index_by_hash.get(decision.state_hash)
            if state_index is not None:
                freeze_indices_by_state_index.setdefault(state_index, []).append(index)
        expected_joint_hash = self.initial_joint_state_hash
        expected_phase_hash = self.initial_phase_evidence_state_hash
        expected_sufficient_hash = self.initial_sufficient_state.sufficient_state_hash
        expected_sequence = 0
        joint_history: dict[str, tuple[str, int, str, int]] = {
            expected_joint_hash: (
                expected_phase_hash,
                expected_sequence,
                expected_sufficient_hash,
                0,
            )
        }
        flattened_draws: list[str] = []
        expected_descriptor_hash = self.initial_sufficient_state.descriptor_hash
        expected_frontier_hash = _proposal_initial_frontier_hash(
            initialization_receipt_hash=self.initialization_receipt_hash,
            proposal_backend_identity_hash=self.proposal_backend_identity_hash,
            joint_state_hash=self.initial_joint_state_hash,
            sufficient_state_descriptor_hash=expected_descriptor_hash,
        )
        for history_index, (receipt, step, completed, outcome) in enumerate(
            zip(updates, replay_steps, completed_pending, outcomes, strict=True),
            start=1,
        ):
            prior_index = history_index - 1
            for index in freeze_indices_by_state_index.get(prior_index, []):
                expected_frontier_hash = _proposal_frontier_after_freeze_decision(
                    expected_frontier_hash,
                    evaluation=evaluations[index],
                    evidence=evidence_receipts[index],
                    decision=decisions[index],
                )
            if (
                completed.prior_progress_hash != expected_frontier_hash
                or completed.request.prior_progress_hash
                != expected_frontier_hash
                or step.update_receipt.receipt_hash != receipt.receipt_hash
                or step.considered_forecasts != completed.forecasts
                or step.selected_forecast.content_hash
                != completed.scheduler_receipt.selected_forecast_hash
                or step.scheduler_receipt.receipt_hash
                != completed.scheduler_receipt.receipt_hash
                or completed.request.request_hash != receipt.request_hash
                or receipt.prior_joint_state_hash != expected_joint_hash
                or receipt.prior_phase_evidence_state_hash != expected_phase_hash
                or receipt.prior_logical_sequence != expected_sequence
                or receipt.prior_sufficient_state_hash != expected_sufficient_hash
                or outcome.prior_sufficient_state_hash != expected_sufficient_hash
                or receipt.request_hash != outcome.request_hash
                or receipt.source_hash != outcome.source_hash
                or receipt.outcome_hash != outcome.outcome_hash
                or receipt.acquired_train_context_ids
                != outcome.executed_train_context_ids
                or receipt.acquired_train_draw_ids != outcome.executed_train_draw_ids
                or receipt.updated_sufficient_state_hash
                != outcome.updated_sufficient_state.sufficient_state_hash
                or receipt.updated_phase_evidence_state_hash
                != outcome.updated_phase_evidence_state_hash
            ):
                raise ValueError("proposal progress update receipt graph is discontinuous")
            expected_joint_hash = receipt.updated_joint_state_hash
            expected_phase_hash = receipt.updated_phase_evidence_state_hash
            expected_sequence = receipt.updated_logical_sequence
            expected_sufficient_hash = receipt.updated_sufficient_state_hash
            expected_descriptor_hash = outcome.updated_sufficient_state.descriptor_hash
            reproduced = replace(
                self.joint_state,
                logical_sequence=expected_sequence,
                phase_evidence_state_hash=expected_phase_hash,
            )
            if reproduced.content_hash != expected_joint_hash:
                raise ValueError("proposal progress update joint state is not reproducible")
            if expected_joint_hash in joint_history:
                raise ValueError("proposal progress revisited an earlier joint state")
            joint_history[expected_joint_hash] = (
                expected_phase_hash,
                expected_sequence,
                expected_sufficient_hash,
                history_index,
            )
            flattened_draws.extend(receipt.acquired_train_draw_ids)
            expected_frontier_hash = _proposal_frontier_after_update(
                expected_frontier_hash,
                completed_pending=completed,
                outcome=outcome,
                replay_step=step,
                update=receipt,
                updated_joint_state_hash=expected_joint_hash,
                updated_sufficient_state_descriptor_hash=expected_descriptor_hash,
            )
        final_history_index = len(updates)
        for index in freeze_indices_by_state_index.get(final_history_index, []):
            expected_frontier_hash = _proposal_frontier_after_freeze_decision(
                expected_frontier_hash,
                evaluation=evaluations[index],
                evidence=evidence_receipts[index],
                decision=decisions[index],
            )
        if (
            self.joint_state.content_hash != expected_joint_hash
            or self.joint_state.phase_evidence_state_hash != expected_phase_hash
            or self.joint_state.logical_sequence != expected_sequence
            or self.current_sufficient_state.sufficient_state_hash
            != expected_sufficient_hash
            or self.current_sufficient_state.descriptor_hash != expected_descriptor_hash
        ):
            raise ValueError("proposal progress current frontier does not close its receipt graph")
        verified_current = replay_v1_proposal_evidence_state(
            initial_state=initial_joint,
            initial_sufficient_state_hash=(
                self.initial_sufficient_state.sufficient_state_hash
            ),
            initialization_receipt_hash=self.initialization_receipt_hash,
            replay_steps=replay_steps,
        )
        if (
            verified_current.current_joint_state_hash != self.joint_state.content_hash
            or verified_current.current_phase_evidence_state_hash
            != self.joint_state.phase_evidence_state_hash
            or verified_current.current_logical_sequence
            != self.joint_state.logical_sequence
            or verified_current.current_sufficient_state_hash
            != self.current_sufficient_state.sufficient_state_hash
        ):
            raise ValueError("proposal progress failed deterministic receipt-graph replay")
        draws = _nonempty_string_tuple(
            self.proposal_train_draw_ids,
            "proposal_train_draw_ids",
            allow_empty=True,
        )
        if draws != tuple(flattened_draws) or len(set(draws)) != len(draws):
            raise ValueError("proposal train draw ledger differs from update receipt history")
        last_history_index = -1
        authorization_count = 0
        for evaluation, evidence, decision in zip(
            evaluations,
            evidence_receipts,
            decisions,
            strict=True,
        ):
            history = joint_history.get(decision.state_hash)
            if history is None:
                raise ValueError("proposal freeze decision names a foreign joint state")
            phase_hash, sequence, sufficient_hash, history_index = history
            if (
                decision.phase_evidence_state_hash != phase_hash
                or decision.logical_sequence != sequence
                or decision.sufficient_state_hash != sufficient_hash
                or decision.freeze_criterion_hash != self.freeze_criterion_hash
                or evaluation.freeze_criterion_hash != self.freeze_criterion_hash
                or evaluation.verified_receipt_graph_hash
                != decision.verified_receipt_graph_hash
                or evidence.state_hash != decision.state_hash
                or evidence.phase_evidence_state_hash != phase_hash
                or evidence.logical_sequence != sequence
                or evidence.sufficient_state_hash != sufficient_hash
                or evidence.verified_receipt_graph_hash
                != decision.verified_receipt_graph_hash
                or evidence.freeze_criterion_hash != decision.freeze_criterion_hash
                or evidence.criterion_evaluator_identity_hash
                != evaluation.criterion_evaluator_identity_hash
                or evidence.criterion_verification_receipt_hash
                != evaluation.criterion_verification_receipt_hash
                or evidence.evidence_evaluation_hash
                != evaluation.evidence_evaluation_hash
                or evidence.criterion_satisfied != evaluation.criterion_satisfied
                or evidence.proposed_candidate_identity_hash
                != evaluation.proposed_candidate_identity_hash
                or decision.criterion_evaluator_identity_hash
                != evaluation.criterion_evaluator_identity_hash
                or decision.criterion_verification_receipt_hash
                != evaluation.criterion_verification_receipt_hash
                or decision.evidence_hash != evidence.content_hash
                or history_index < last_history_index
            ):
                raise ValueError("proposal freeze decision has stale evidence lineage")
            verified_at_decision = replay_v1_proposal_evidence_state(
                initial_state=initial_joint,
                initial_sufficient_state_hash=(
                    self.initial_sufficient_state.sufficient_state_hash
                ),
                initialization_receipt_hash=self.initialization_receipt_hash,
                replay_steps=replay_steps[:history_index],
            )
            if (
                decision.verified_receipt_graph_hash
                != verified_at_decision.verified_receipt_graph_hash
                or evidence.verified_evidence_state_hash
                != verified_at_decision.content_hash
            ):
                raise ValueError("proposal freeze decision lacks verified receipt-graph identity")
            last_history_index = history_index
            if decision.action == "authorize_candidate_freeze":
                authorization_count += 1
                if history_index != len(updates) or decision is not decisions[-1]:
                    raise ValueError("candidate-freeze authorization must close current evidence")
        if authorization_count > 1:
            raise ValueError("proposal progress has multiple candidate-freeze authorizations")
        refusal = self.terminal_refusal_receipt
        refusal_forecasts = tuple(self.terminal_refusal_forecasts)
        if refusal is None:
            if refusal_forecasts:
                raise ValueError("terminal proposal refusal forecasts lack their receipt")
        else:
            if type(refusal) is not V1JointAcquisitionScheduleReceipt:
                raise TypeError("terminal proposal refusal has the wrong receipt type")
            if any(
                type(item) is not V1NextActionAcquisitionForecast
                for item in refusal_forecasts
            ):
                raise TypeError("terminal proposal refusal contains a nonforecast")
            canonical_forecasts = tuple(
                sorted(refusal_forecasts, key=lambda item: item.content_hash)
            )
            if refusal_forecasts != canonical_forecasts:
                raise ValueError("terminal proposal refusal forecasts are not canonical")
            replayed_refusal = schedule_v1_joint_acquisition(
                self.joint_state,
                refusal_forecasts,
            )
            if (
                refusal.action != "refuse_no_legal_action"
                or replayed_refusal.receipt_hash != refusal.receipt_hash
                or refusal.state_hash != self.joint_state.content_hash
                or refusal.considered_forecast_hashes
                != tuple(item.content_hash for item in refusal_forecasts)
                or authorization_count
            ):
                raise ValueError("terminal proposal refusal does not replay exactly")
            expected_frontier_hash = _proposal_frontier_after_terminal_refusal(
                expected_frontier_hash,
                receipt=refusal,
                forecasts=refusal_forecasts,
            )
        if self.frontier_hash != expected_frontier_hash:
            raise ValueError("proposal progress frontier event chain is not reproducible")
        object.__setattr__(self, "update_receipts", updates)
        object.__setattr__(self, "replay_steps", replay_steps)
        object.__setattr__(self, "completed_pending_requests", completed_pending)
        object.__setattr__(self, "outcome_receipts", outcomes)
        object.__setattr__(self, "freeze_evaluation_receipts", evaluations)
        object.__setattr__(self, "freeze_evidence_receipts", evidence_receipts)
        object.__setattr__(self, "freeze_decision_receipts", decisions)
        object.__setattr__(self, "proposal_train_draw_ids", draws)
        object.__setattr__(self, "terminal_refusal_forecasts", refusal_forecasts)
        if (
            self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.calibration_authority is not False
            or self.validation_authority is not False
            or self.mask_commit_authority is not False
            or self.schema != _PROPOSAL_PROGRESS_SCHEMA
        ):
            raise ValueError("proposal progress crossed its acquisition authority firewall")

    @property
    def frontier_hash(self) -> str:
        frontier = _proposal_initial_frontier_hash(
            initialization_receipt_hash=self.initialization_receipt_hash,
            proposal_backend_identity_hash=self.proposal_backend_identity_hash,
            joint_state_hash=self.initial_joint_state_hash,
            sufficient_state_descriptor_hash=(
                self.initial_sufficient_state.descriptor_hash
            ),
        )
        state_index_by_hash = {self.initial_joint_state_hash: 0}
        for index, update in enumerate(self.update_receipts, start=1):
            state_index_by_hash[update.updated_joint_state_hash] = index
        freeze_by_state: dict[
            int,
            list[
                tuple[
                    V1RuntimeProposalFreezeEvaluation,
                    V1ProposalFreezeEvidence,
                    V1ProposalFreezeDecisionReceipt,
                ]
            ],
        ] = {}
        for evaluation, evidence, decision in zip(
            self.freeze_evaluation_receipts,
            self.freeze_evidence_receipts,
            self.freeze_decision_receipts,
            strict=True,
        ):
            index = state_index_by_hash[decision.state_hash]
            freeze_by_state.setdefault(index, []).append(
                (evaluation, evidence, decision)
            )
        for index in range(len(self.update_receipts) + 1):
            for evaluation, evidence, decision in freeze_by_state.get(index, []):
                frontier = _proposal_frontier_after_freeze_decision(
                    frontier,
                    evaluation=evaluation,
                    evidence=evidence,
                    decision=decision,
                )
            if index == len(self.update_receipts):
                break
            frontier = _proposal_frontier_after_update(
                frontier,
                completed_pending=self.completed_pending_requests[index],
                outcome=self.outcome_receipts[index],
                replay_step=self.replay_steps[index],
                update=self.update_receipts[index],
                updated_joint_state_hash=(
                    self.update_receipts[index].updated_joint_state_hash
                ),
                updated_sufficient_state_descriptor_hash=(
                    self.outcome_receipts[index].updated_sufficient_state.descriptor_hash
                ),
            )
        if self.terminal_refusal_receipt is not None:
            frontier = _proposal_frontier_after_terminal_refusal(
                frontier,
                receipt=self.terminal_refusal_receipt,
                forecasts=self.terminal_refusal_forecasts,
            )
        return frontier

    @property
    def progress_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-proposal-progress-",
        )

    @property
    def authorized_freeze_decision(self) -> V1ProposalFreezeDecisionReceipt | None:
        if not self.freeze_decision_receipts:
            return None
        decision = self.freeze_decision_receipts[-1]
        return decision if decision.action == "authorize_candidate_freeze" else None

    @property
    def proposal_order_indeterminate(self) -> bool:
        return self.terminal_refusal_receipt is not None

    @property
    def verified_evidence_state(self) -> V1VerifiedProposalEvidenceState:
        """Replay the full predictable/outcome receipt graph on demand."""

        initial_state = replace(
            self.joint_state,
            logical_sequence=0,
            phase_evidence_state_hash=self.initial_phase_evidence_state_hash,
        )
        return replay_v1_proposal_evidence_state(
            initial_state=initial_state,
            initial_sufficient_state_hash=(
                self.initial_sufficient_state.sufficient_state_hash
            ),
            initialization_receipt_hash=self.initialization_receipt_hash,
            replay_steps=self.replay_steps,
        )

    def __hash__(self) -> int:
        return hash(self.progress_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeCompletedProposalClosureReceipt:
    """Permanent content-addressed pairing of a freeze and attached acquisition.

    The progress/frontier fields are durable commitments, not a retained copy
    of the full proposal evidence graph.  Full historical evidence replay
    therefore requires the original proposal sidecar or an external archive.
    This compact closure instead keeps the exact decision and enough small
    controller/candidate cross-links to reject lineage substitution on resume.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    acquisition_generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    mask_lineage_hash: str
    pre_attach_controller_state_hash: str
    pre_attach_controller_snapshot: V1ControllerSnapshotReceipt
    proposal_progress_hash: str
    proposal_frontier_hash: str
    proposal_joint_state_hash: str
    proposal_phase_evidence_state_hash: str
    proposal_sufficient_state_hash: str
    proposal_logical_sequence: int
    verified_receipt_graph_hash: str
    freeze_criterion_hash: str
    freeze_evidence_hash: str
    freeze_evidence_evaluation_hash: str
    criterion_verification_receipt_hash: str
    freeze_decision: V1ProposalFreezeDecisionReceipt
    acquisition_hash: str
    acquisition_receipt_hash: str
    ordered_candidate_ids: tuple[str, ...]
    candidate_identity_hash: str
    post_attach_controller_state_hash: str
    post_attach_controller_snapshot: V1ControllerSnapshotReceipt
    proposal_order_alpha_schedule_hash: str | None = None
    proposal_order_alpha_ledger_before_hash: str | None = None
    proposal_order_design_hash: str | None = None
    proposal_order_base_serfling_design_hash: str | None = None
    proposal_order_alpha_ledger_after_hash: str | None = None
    proposal_order_alpha_ledger_after_replay: Mapping[str, Any] | None = None
    proposal_order_envelope_certificate_hash: str | None = None
    proposal_order_action_family_registry_hash: str | None = None
    sampled_v7_D_alpha_family_plan_hash: str | None = None
    sampled_v7_D_alpha_family_registry_receipt_hash: str | None = None
    sampled_v7_D_alpha_family_registry_replay: Mapping[str, Any] | None = None
    sampled_v7_D_alpha_head_CAS_receipt_hash: str | None = None
    sampled_v7_D_alpha_published_ledger_head_hash: str | None = None
    sampled_v7_D_alpha_durable_controller_head_closed: bool | None = None
    validation_outcomes_observed: bool = False
    train_optimization_data_only: bool = True
    mask_commit_authority: bool = False
    schema: str = _COMPLETED_PROPOSAL_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "proposal_backend_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "mask_lineage_hash",
            "pre_attach_controller_state_hash",
            "proposal_progress_hash",
            "proposal_frontier_hash",
            "proposal_joint_state_hash",
            "proposal_phase_evidence_state_hash",
            "proposal_sufficient_state_hash",
            "verified_receipt_graph_hash",
            "freeze_criterion_hash",
            "freeze_evidence_hash",
            "freeze_evidence_evaluation_hash",
            "criterion_verification_receipt_hash",
            "acquisition_hash",
            "acquisition_receipt_hash",
            "candidate_identity_hash",
            "post_attach_controller_state_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        generation = _nonnegative(
            self.acquisition_generation_index,
            "acquisition_generation_index",
        )
        object.__setattr__(self, "acquisition_generation_index", generation)
        object.__setattr__(
            self,
            "proposal_logical_sequence",
            _nonnegative(
                self.proposal_logical_sequence,
                "proposal_logical_sequence",
            ),
        )
        ordered_candidate_ids = tuple(
            _nonempty(value, "ordered_candidate_id")
            for value in self.ordered_candidate_ids
        )
        if len(set(ordered_candidate_ids)) != len(ordered_candidate_ids):
            raise ValueError("completed proposal closure candidate IDs are not unique")
        object.__setattr__(self, "ordered_candidate_ids", ordered_candidate_ids)
        alpha_coordinate_names = (
            "proposal_order_alpha_schedule_hash",
            "proposal_order_alpha_ledger_before_hash",
            "proposal_order_design_hash",
            "proposal_order_alpha_ledger_after_hash",
            "proposal_order_alpha_ledger_after_replay",
            "proposal_order_envelope_certificate_hash",
            "proposal_order_action_family_registry_hash",
        )
        alpha_coordinates = tuple(
            getattr(self, name) for name in alpha_coordinate_names
        )
        alpha_bound = all(value is not None for value in alpha_coordinates)
        if any(value is not None for value in alpha_coordinates) and not alpha_bound:
            raise ValueError(
                "completed proposal closure alpha coordinates must be all-or-none"
            )
        if alpha_bound:
            for name in alpha_coordinate_names:
                if name != "proposal_order_alpha_ledger_after_replay":
                    object.__setattr__(
                        self,
                        name,
                        _nonempty(getattr(self, name), name),
                    )
            replay = self.proposal_order_alpha_ledger_after_replay
            if not isinstance(replay, Mapping):
                raise TypeError(
                    "completed proposal closure alpha replay must be one mapping"
                )
            from .modern_v1_proposal_alpha_ledger import (
                restore_v1_proposal_order_alpha_ledger_replay,
            )

            restored = restore_v1_proposal_order_alpha_ledger_replay(replay)
            base_design_hash = self.proposal_order_base_serfling_design_hash
            if base_design_hash is not None:
                base_design_hash = _nonempty(
                    base_design_hash,
                    "proposal_order_base_serfling_design_hash",
                )
                object.__setattr__(
                    self,
                    "proposal_order_base_serfling_design_hash",
                    base_design_hash,
                )
            charged_design_hash = (
                base_design_hash
                if base_design_hash is not None
                else self.proposal_order_design_hash
            )
            if (
                restored.schedule_hash
                != self.proposal_order_alpha_schedule_hash
                or restored.ledger_hash
                != self.proposal_order_alpha_ledger_after_hash
                or restored.proposal_generation != generation + 1
                or restored.prior_state_hash
                != self.proposal_order_alpha_ledger_before_hash
                or restored.registered_design_hashes[-1]
                != charged_design_hash
                or restored.campaign_contract_hash != self.campaign_contract_hash
                or restored.controller_run_hash != self.controller_run_id
            ):
                raise ValueError(
                    "completed proposal closure alpha replay has stale coordinates"
                )
            object.__setattr__(
                self,
                "proposal_order_alpha_ledger_after_replay",
                _freeze(replay, "completed proposal closure alpha replay"),
            )
        alpha_family_coordinate_names = (
            "sampled_v7_D_alpha_family_plan_hash",
            "sampled_v7_D_alpha_family_registry_receipt_hash",
            "sampled_v7_D_alpha_family_registry_replay",
            "sampled_v7_D_alpha_head_CAS_receipt_hash",
            "sampled_v7_D_alpha_published_ledger_head_hash",
            "sampled_v7_D_alpha_durable_controller_head_closed",
        )
        alpha_family_coordinates = tuple(
            getattr(self, name) for name in alpha_family_coordinate_names
        )
        alpha_family_bound = all(
            value is not None for value in alpha_family_coordinates
        )
        if (
            any(value is not None for value in alpha_family_coordinates)
            and not alpha_family_bound
        ):
            raise ValueError(
                "completed proposal closure A/F/J alpha-family coordinates "
                "must be all-or-none"
            )
        if alpha_family_bound:
            if not alpha_bound or self.proposal_order_base_serfling_design_hash is None:
                raise ValueError(
                    "A/F/J alpha-family closure requires the charged base design"
                )
            for name in alpha_family_coordinate_names:
                if name not in {
                    "sampled_v7_D_alpha_family_registry_replay",
                    "sampled_v7_D_alpha_durable_controller_head_closed",
                }:
                    object.__setattr__(
                        self,
                        name,
                        _nonempty(getattr(self, name), name),
                    )
            if self.sampled_v7_D_alpha_durable_controller_head_closed is not True:
                raise ValueError(
                    "A/F/J alpha-family closure lacks its durable controller head"
                )
            registry_replay = self.sampled_v7_D_alpha_family_registry_replay
            if not isinstance(registry_replay, Mapping):
                raise TypeError(
                    "A/F/J alpha-family closure replay must be one mapping"
                )
            from .modern_v1_sampled_v7_d_alpha_family_registry import (
                restore_v1_sampled_v7_d_alpha_family_registry_replay,
            )

            registry = restore_v1_sampled_v7_d_alpha_family_registry_replay(
                registry_replay
            )
            if (
                registry.receipt_hash
                != self.sampled_v7_D_alpha_family_registry_receipt_hash
                or registry.alpha_family_plan_hash
                != self.sampled_v7_D_alpha_family_plan_hash
                or registry.campaign_contract_hash != self.campaign_contract_hash
                or registry.controller_run_hash != self.controller_run_id
                or registry.proposal_generation != generation
                or registry.ledger_before_hash
                != self.proposal_order_alpha_ledger_before_hash
                or registry.ledger_after_hash
                != self.proposal_order_alpha_ledger_after_hash
                or registry.charged_base_design_hash
                != self.proposal_order_base_serfling_design_hash
                or registry.hybrid_wrapper_design_hash
                != self.proposal_order_design_hash
                or self.sampled_v7_D_alpha_published_ledger_head_hash
                != self.proposal_order_alpha_ledger_after_hash
            ):
                raise ValueError(
                    "completed proposal closure A/F/J alpha-family replay forked"
                )
            object.__setattr__(
                self,
                "sampled_v7_D_alpha_family_registry_replay",
                _freeze(
                    registry_replay,
                    "completed proposal closure A/F/J alpha-family replay",
                ),
            )
        if type(self.freeze_decision) is not V1ProposalFreezeDecisionReceipt:
            raise TypeError("completed proposal closure requires an exact freeze decision")
        self.freeze_decision.__post_init__()
        if (
            self.freeze_decision.action != "authorize_candidate_freeze"
            or self.freeze_decision.candidate_identity_hash
            != self.candidate_identity_hash
            or self.freeze_decision.verified_receipt_graph_hash
            != self.verified_receipt_graph_hash
            or self.freeze_decision.state_hash != self.proposal_joint_state_hash
            or self.freeze_decision.phase_evidence_state_hash
            != self.proposal_phase_evidence_state_hash
            or self.freeze_decision.sufficient_state_hash
            != self.proposal_sufficient_state_hash
            or self.freeze_decision.logical_sequence
            != self.proposal_logical_sequence
            or self.freeze_decision.freeze_criterion_hash
            != self.freeze_criterion_hash
            or self.freeze_decision.evidence_hash != self.freeze_evidence_hash
            or self.freeze_decision.evidence_evaluation_hash
            != self.freeze_evidence_evaluation_hash
            or self.freeze_decision.criterion_verification_receipt_hash
            != self.criterion_verification_receipt_hash
        ):
            raise ValueError("completed proposal closure has stale freeze authorization")
        for snapshot, label in (
            (self.pre_attach_controller_snapshot, "pre-attach"),
            (self.post_attach_controller_snapshot, "post-attach"),
        ):
            if type(snapshot) is not V1ControllerSnapshotReceipt:
                raise TypeError(
                    f"completed proposal closure {label} snapshot is not exact"
                )
            snapshot.__post_init__()
        pre = self.pre_attach_controller_snapshot
        post = self.post_attach_controller_snapshot
        common_expected = {
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "current_mask_hash": self.current_mask_hash,
            "acquisition_generation_index": generation,
            "accepted_step_count": generation,
        }
        if any(
            getattr(snapshot, name) != expected
            for snapshot in (pre, post)
            for name, expected in common_expected.items()
        ):
            raise ValueError("completed proposal closure snapshots have stale run lineage")
        if (
            pre.phase != "needs_acquisition"
            or pre.queue_cursor != 0
            or len(pre.acquisition_receipt_hashes) != generation
            or post.phase not in {"queue_ready", "stopped"}
            or post.queue_cursor != 0
            or post.logical_sequence != pre.logical_sequence + 1
            or post.acquisition_receipt_hashes
            != pre.acquisition_receipt_hashes + (self.acquisition_receipt_hash,)
            or post.attempt_receipt_hashes != pre.attempt_receipt_hashes
            or post.transition_receipt_hashes != pre.transition_receipt_hashes
            or post.accepted_trajectory_hash != pre.accepted_trajectory_hash
            or post.candidate_attempt_count != pre.candidate_attempt_count
            or post.cumulative_removed_count != pre.cumulative_removed_count
            or post.globally_used_draw_ids_hash != pre.globally_used_draw_ids_hash
            or (
                post.phase == "queue_ready"
                and post.acquisition_receipt_hash != self.acquisition_receipt_hash
            )
            or (
                post.phase == "stopped"
                and post.stop_reason != "global_queue_exhausted"
            )
        ):
            raise ValueError("completed proposal closure does not replay controller attach")
        if (
            self.validation_outcomes_observed is not False
            or self.train_optimization_data_only is not True
            or self.mask_commit_authority is not False
            or self.schema
            != (
                _COMPLETED_PROPOSAL_CLOSURE_V4_SCHEMA
                if alpha_family_bound
                else _COMPLETED_PROPOSAL_CLOSURE_V3_SCHEMA
                if alpha_bound
                and self.proposal_order_base_serfling_design_hash is not None
                else _COMPLETED_PROPOSAL_CLOSURE_V2_SCHEMA
                if alpha_bound
                else _COMPLETED_PROPOSAL_CLOSURE_SCHEMA
            )
        ):
            raise ValueError("completed proposal closure crossed its authority firewall")

    @property
    def pre_attach_controller_snapshot_hash(self) -> str:
        return self.pre_attach_controller_snapshot.receipt_hash

    @property
    def post_attach_controller_snapshot_hash(self) -> str:
        return self.post_attach_controller_snapshot.receipt_hash

    @property
    def freeze_decision_hash(self) -> str:
        return self.freeze_decision.receipt_hash

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-completed-proposal-closure-",
        )

    def __hash__(self) -> int:
        return hash(self.receipt_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimePendingProposalAcquisition:
    """Full predictable proposal decision persisted before train execution."""

    prior_progress_hash: str
    forecasts: tuple[V1NextActionAcquisitionForecast, ...]
    forecast_set_hash: str
    scheduler_receipt: V1JointAcquisitionScheduleReceipt
    request_plan: V1RuntimeProposalRequestPlan
    request: V1RuntimeProposalAcquisitionRequest
    outcome_observed: bool = False
    candidate_or_mask_authority: bool = False
    schema: str = _PENDING_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prior_progress_hash",
            _nonempty(self.prior_progress_hash, "prior_progress_hash"),
        )
        forecasts = tuple(self.forecasts)
        if any(type(item) is not V1NextActionAcquisitionForecast for item in forecasts):
            raise TypeError("pending proposal contains a nonforecast value")
        forecasts = tuple(sorted(forecasts, key=lambda item: item.content_hash))
        forecast_hashes = tuple(item.content_hash for item in forecasts)
        if not forecasts or len(set(forecast_hashes)) != len(forecast_hashes):
            raise ValueError("pending proposal requires a nonempty distinct forecast set")
        object.__setattr__(self, "forecasts", forecasts)
        expected_set_hash = _proposal_forecast_set_hash(forecasts)
        if self.forecast_set_hash != expected_set_hash:
            raise ValueError("pending proposal forecast-set hash differs from its full set")
        if type(self.scheduler_receipt) is not V1JointAcquisitionScheduleReceipt:
            raise TypeError("pending proposal requires an exact scheduler receipt")
        if type(self.request_plan) is not V1RuntimeProposalRequestPlan:
            raise TypeError("pending proposal requires an exact request plan")
        if type(self.request) is not V1RuntimeProposalAcquisitionRequest:
            raise TypeError("pending proposal requires an exact outcome-independent request")
        if (
            self.scheduler_receipt.action != "acquire"
            or self.scheduler_receipt.considered_forecast_hashes
            != tuple(sorted(forecast_hashes))
            or self.request.prior_progress_hash != self.prior_progress_hash
            or self.request.forecast_set_hash != self.forecast_set_hash
            or self.request.scheduler_receipt_hash != self.scheduler_receipt.receipt_hash
            or self.request.selected_forecast_hash
            != self.scheduler_receipt.selected_forecast_hash
            or self.request.selected_role != self.scheduler_receipt.selected_role
            or self.request.request_plan_hash != self.request_plan.plan_hash
            or self.request.train_context_ids != self.request_plan.train_context_ids
            or self.request.train_draw_ids != self.request_plan.train_draw_ids
            or self.request.source_hash != self.request_plan.source_hash
            or self.request.payload != self.request_plan.payload
        ):
            raise ValueError("pending proposal request differs from scheduler selection")
        selected = next(
            (item for item in forecasts if item.content_hash == self.request.selected_forecast_hash),
            None,
        )
        if (
            selected is None
            or selected.acquisition_units != self.request.acquisition_units
            or selected.role != self.request.selected_role
        ):
            raise ValueError("pending proposal request differs from its selected forecast")
        if (
            self.outcome_observed is not False
            or self.candidate_or_mask_authority is not False
            or self.schema != _PENDING_PROPOSAL_SCHEMA
        ):
            raise ValueError("pending proposal crossed its pre-outcome authority firewall")

    @property
    def pending_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-pending-proposal-",
        )

    def __hash__(self) -> int:
        return hash(self.pending_hash)


def _assert_proposal_progress_bound_to_controller(
    progress: V1RuntimeProposalAcquisitionProgress,
    controller: V1ControllerState,
) -> None:
    contract = controller.binding.contract
    horizon = getattr(controller, "fqr_no_candidate_horizon_witness", None)
    controller_state_hash = (
        controller.state_hash
        if horizon is None
        else horizon.pre_stop_controller_state_hash
    )
    controller_snapshot_hash = (
        controller.snapshot_receipt().receipt_hash
        if horizon is None
        else horizon.pre_stop_controller_snapshot_hash
    )
    controller_sequence = (
        controller.logical_sequence
        if horizon is None
        else horizon.pre_stop_logical_sequence
    )
    expected = {
        "campaign_contract_hash": contract.campaign_contract_hash,
        "campaign_binding_hash": controller.binding.binding_hash,
        "controller_run_id": controller.controller_run_id,
        "controller_state_hash": controller_state_hash,
        "controller_snapshot_hash": controller_snapshot_hash,
        "controller_logical_sequence": controller_sequence,
        "current_mask_hash": controller.current_mask.mask_hash,
        "route_axis_hash": controller.current_mask.route_axis_hash,
        "acquisition_generation_index": controller.acquisition_generation_index,
        "mask_lineage_hashes": controller.mask_lineage_hashes,
        "transition_receipt_hashes": controller.transition_receipt_hashes,
    }
    if any(getattr(progress, name) != value for name, value in expected.items()):
        raise ValueError("runtime proposal progress has stale controller/mask lineage")
    if set(progress.proposal_train_draw_ids).intersection(
        controller.globally_used_draw_ids
    ):
        raise ValueError("proposal train draws overlap the paired-oracle draw ledger")
    if horizon is not None:
        _assert_fqr_horizon_witness_bound_to_progress(horizon, progress)


def _assert_fqr_horizon_witness_bound_to_progress(
    witness: V1ControllerProposalHorizonWitness,
    progress: V1RuntimeProposalAcquisitionProgress,
) -> None:
    """Replay the typed horizon binding without requiring backend process caches."""

    if type(witness) is not V1ControllerProposalHorizonWitness:
        raise TypeError("proposal horizon requires the exact controller witness")
    if type(progress) is not V1RuntimeProposalAcquisitionProgress:
        raise TypeError("proposal horizon requires exact retained proposal progress")
    proof = witness.proof
    proof.__post_init__()
    verified = progress.verified_evidence_state
    expected = {
        "campaign_contract_hash": progress.campaign_contract_hash,
        "campaign_binding_hash": progress.campaign_binding_hash,
        "controller_run_id": progress.controller_run_id,
        "proposal_backend_identity_hash": progress.proposal_backend_identity_hash,
        "generation_index": progress.acquisition_generation_index,
        "current_mask_hash": progress.current_mask_hash,
        "route_axis_hash": progress.route_axis_hash,
        "mask_lineage_hash": progress.mask_lineage_hash,
        "terminal_proposal_progress_hash": progress.progress_hash,
        "terminal_proposal_frontier_hash": progress.frontier_hash,
        "proposal_joint_state_hash": progress.joint_state.content_hash,
        "proposal_phase_evidence_state_hash": (
            progress.joint_state.phase_evidence_state_hash
        ),
        "proposal_sufficient_state_hash": (
            progress.current_sufficient_state.sufficient_state_hash
        ),
        "proposal_sufficient_state_descriptor_hash": (
            progress.current_sufficient_state.descriptor_hash
        ),
        "verified_receipt_graph_hash": verified.verified_receipt_graph_hash,
    }
    if any(getattr(witness, name) != value for name, value in expected.items()):
        raise ValueError("proposal horizon witness has stale retained-progress lineage")
    if (
        not progress.freeze_evaluation_receipts
        or len(progress.freeze_evaluation_receipts)
        != len(progress.freeze_evidence_receipts)
        or len(progress.freeze_evaluation_receipts)
        != len(progress.freeze_decision_receipts)
    ):
        raise ValueError("proposal horizon lacks one aligned freeze-refusal history")
    evaluation = progress.freeze_evaluation_receipts[-1]
    evidence = progress.freeze_evidence_receipts[-1]
    decision = progress.freeze_decision_receipts[-1]
    expected_verification = v1_fqr_no_candidate_freeze_verification_hash(
        proof=proof,
        verified_receipt_graph_hash=verified.verified_receipt_graph_hash,
        freeze_criterion_hash=progress.freeze_criterion_hash,
        sufficient_state_hash=progress.current_sufficient_state.sufficient_state_hash,
    )
    expected_evaluation = v1_fqr_no_candidate_freeze_evaluation_hash(
        proof=proof,
        verification_hash=expected_verification,
    )
    if (
        evaluation.receipt_hash != witness.freeze_evaluation_receipt_hash
        or evidence.content_hash != witness.freeze_evidence_receipt_hash
        or decision.receipt_hash != witness.freeze_decision_receipt_hash
        or evaluation.criterion_verification_receipt_hash != expected_verification
        or evaluation.evidence_evaluation_hash != expected_evaluation
        or evaluation.criterion_satisfied is not False
        or evaluation.proposed_candidate_identity_hash is not None
        or evidence.criterion_verification_receipt_hash != expected_verification
        or evidence.evidence_evaluation_hash != expected_evaluation
        or evidence.criterion_satisfied is not False
        or evidence.proposed_candidate_identity_hash is not None
        or decision.criterion_verification_receipt_hash != expected_verification
        or decision.evidence_evaluation_hash != expected_evaluation
        or decision.action != "refuse_candidate_freeze"
        or decision.candidate_identity_hash is not None
        or progress.authorized_freeze_decision is not None
        or progress.terminal_refusal_receipt is not None
    ):
        raise ValueError("proposal horizon is not the exact proof-bound freeze refusal")
    storage = progress.current_sufficient_state.storage_descriptor
    if (
        storage.get("proposal_inference_stage") != "fqr_C_stable_no_candidate"
        or storage.get("fqr_C_state_payload_hash")
        != proof.fqr_c_state_payload_hash
        or storage.get("fqr_graph_hash") != proof.fqr_graph_hash
        or storage.get("fqr_policy_hash") != proof.fqr_policy_hash
        or storage.get("fqr_progress_receipt_hash")
        != proof.fqr_progress_receipt_hash
        or storage.get("fqr_readiness_receipt_hash")
        != proof.fqr_readiness_receipt_hash
        or storage.get("fqr_accumulator_snapshot_hash")
        != proof.fqr_accumulator_snapshot_hash
        or storage.get("fqr_native_execution_binding_chain_hash")
        != proof.native_execution_binding_chain_hash
        or storage.get("canonical_fqr_candidate_queue_authority") is not False
        or storage.get("candidate_freeze_authority") is not False
        or storage.get("calibration_K_open_authority") is not False
        or storage.get("paired_oracle_O_open_authority") is not False
        or storage.get("mask_commit_authority") is not False
        or storage.get("validation_outcomes_observed") is not False
    ):
        raise ValueError("proposal horizon retained sidecar differs from its compact proof")


def _assert_pending_proposal_bound(
    progress: V1RuntimeProposalAcquisitionProgress,
    pending: V1RuntimePendingProposalAcquisition,
) -> None:
    if pending.prior_progress_hash != progress.frontier_hash:
        raise ValueError("runtime pending proposal is stale for current progress")
    recomputed = schedule_v1_joint_acquisition(
        progress.joint_state,
        pending.forecasts,
    )
    if recomputed.receipt_hash != pending.scheduler_receipt.receipt_hash:
        raise ValueError("runtime pending proposal did not preserve exact max-utility scheduling")
    request = pending.request
    expected = {
        "campaign_binding_hash": progress.campaign_binding_hash,
        "controller_run_id": progress.controller_run_id,
        "controller_state_hash": progress.controller_state_hash,
        "controller_snapshot_hash": progress.controller_snapshot_hash,
        "current_mask_hash": progress.current_mask_hash,
        "acquisition_generation_index": progress.acquisition_generation_index,
        "mask_lineage_hash": progress.mask_lineage_hash,
        "prior_joint_state_hash": progress.joint_state.content_hash,
        "prior_phase_evidence_state_hash": (
            progress.joint_state.phase_evidence_state_hash
        ),
        "prior_logical_sequence": progress.joint_state.logical_sequence,
        "prior_sufficient_state_hash": (
            progress.current_sufficient_state.sufficient_state_hash
        ),
        "prior_sufficient_state_descriptor_hash": (
            progress.current_sufficient_state.descriptor_hash
        ),
    }
    if any(getattr(request, name) != value for name, value in expected.items()):
        raise ValueError("runtime pending proposal request has stale progress lineage")
    new_draws = set(request.train_draw_ids)
    if new_draws.intersection(progress.proposal_train_draw_ids):
        raise ValueError("runtime pending proposal would reuse a proposal train draw ID")


@dataclass(frozen=True, slots=True)
class V1RuntimePendingLook:
    """System-entropy sampling plan persisted before provider execution."""

    controller_snapshot_hash: str
    active_candidate_hash: str
    look_index: int
    sampling_plan: V1StagedAlphaSamplingPlan
    request: MeasurementRequest
    evidence_packet: V1EvidencePacket
    schema: str = _PENDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_snapshot_hash",
            _nonempty(self.controller_snapshot_hash, "controller_snapshot_hash"),
        )
        object.__setattr__(
            self,
            "active_candidate_hash",
            _nonempty(self.active_candidate_hash, "active_candidate_hash"),
        )
        object.__setattr__(self, "look_index", _nonnegative(self.look_index, "look_index"))
        if type(self.sampling_plan) is not V1StagedAlphaSamplingPlan:
            raise TypeError("pending look requires an exact sampling plan")
        if type(self.request) is not MeasurementRequest:
            raise TypeError("pending look requires an exact measurement request")
        if type(self.evidence_packet) is not V1EvidencePacket:
            raise TypeError("pending look requires an exact evidence packet")
        if (
            not self.sampling_plan.runtime_generated
            or self.request.kind != "hard_mask_oracle"
            or tuple(self.request.payload.get("draw_ids", ())) != self.sampling_plan.draw_ids
            or tuple(self.request.payload.get("source_context_ids", ()))
            != self.sampling_plan.source_context_ids
            or tuple(
                self.request.payload.get("sampling_population_probabilities", ())
            )
            != self.sampling_plan.sampling_population_probabilities
            or self.evidence_packet.draw_ids != self.sampling_plan.draw_ids
            or self.evidence_packet.source_context_ids != self.sampling_plan.source_context_ids
        ):
            raise ValueError("pending look lacks exact system-entropy draw lineage")
        if self.schema != _PENDING_SCHEMA:
            raise ValueError("unsupported modern-v1 pending-look schema")

    @property
    def pending_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-runtime-pending-look-")


@dataclass(frozen=True, slots=True)
class V1RuntimePendingPairedBCEChunk:
    """Outcome-independent paired-BCE oracle chunk persisted before row execution."""

    controller_snapshot_hash: str
    active_candidate_hash: str
    registration_hash: str
    prior_paired_bce_state_hash: str
    request: V1PairedBCEChunkRequest
    context_ids_in_registered_slice: tuple[str, ...]
    dynamic_oracle_receipt_hash: str | None = None
    betting_replay_reference_hash: str | None = None
    row_payload_externalized: bool = True
    outcome_observed: bool = False
    candidate_or_mask_authority: bool = False
    schema: str = _PENDING_PAIRED_BCE_CHUNK_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "controller_snapshot_hash",
            "active_candidate_hash",
            "registration_hash",
            "prior_paired_bce_state_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.request) is not V1PairedBCEChunkRequest:
            raise TypeError("pending paired-BCE chunk requires an exact request")
        self.request.assert_integrity()
        contexts = _nonempty_string_tuple(
            self.context_ids_in_registered_slice,
            "context_ids_in_registered_slice",
        )
        if len(contexts) != self.request.row_count:
            raise ValueError("pending paired-BCE context slice length differs from request")
        replay_ref = self.betting_replay_reference_hash
        if replay_ref is not None:
            replay_ref = _sha256_hex(
                replay_ref,
                "betting_replay_reference_hash",
            )
            object.__setattr__(
                self,
                "betting_replay_reference_hash",
                replay_ref,
            )
        dynamic_hash = self.dynamic_oracle_receipt_hash
        if dynamic_hash is not None:
            dynamic_hash = _sha256_hex(
                dynamic_hash,
                "dynamic_oracle_receipt_hash",
            )
            object.__setattr__(self, "dynamic_oracle_receipt_hash", dynamic_hash)
        if (
            self.request.registration_hash != self.registration_hash
            or self.request.prior_state_hash != self.prior_paired_bce_state_hash
            or self.request.dynamic_oracle_receipt_hash != dynamic_hash
        ):
            raise ValueError("pending paired-BCE request has stale registration/state")
        if (
            self.row_payload_externalized is not True
            or self.outcome_observed is not False
            or self.candidate_or_mask_authority is not False
            or self.schema != _PENDING_PAIRED_BCE_CHUNK_SCHEMA
        ):
            raise ValueError("pending paired-BCE chunk crossed its outcome firewall")
        object.__setattr__(self, "context_ids_in_registered_slice", contexts)

    @property
    def pending_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-pending-paired-bce-chunk-",
        )

    def __hash__(self) -> int:
        return hash(self.pending_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimePairedBCEChunkOutcome:
    """Compact outcome receipt for one externally evaluated paired-BCE chunk."""

    pending_hash: str
    request_hash: str
    receipt: V1PairedBCEChunkReceipt
    committed_state: V1PairedBCEAcquisitionState
    row_payload_externalized: bool = True
    candidate_or_mask_authority: bool = False
    schema: str = _PAIRED_BCE_CHUNK_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "pending_hash", _nonempty(self.pending_hash, "pending_hash"))
        object.__setattr__(self, "request_hash", _nonempty(self.request_hash, "request_hash"))
        if type(self.receipt) is not V1PairedBCEChunkReceipt:
            raise TypeError("paired-BCE chunk outcome requires an exact receipt")
        if type(self.committed_state) is not V1PairedBCEAcquisitionState:
            raise TypeError("paired-BCE chunk outcome requires an exact committed state")
        self.receipt.assert_integrity()
        self.committed_state.assert_integrity()
        if self.receipt.request_hash != self.request_hash:
            raise ValueError("paired-BCE chunk outcome receipt names another request")
        if (
            self.receipt.dynamic_oracle_receipt_hash
            != self.committed_state.dynamic_oracle_receipt_hash
        ):
            raise ValueError("paired-BCE chunk outcome crossed dynamic O frames")
        if (
            self.row_payload_externalized is not True
            or self.candidate_or_mask_authority is not False
            or self.schema != _PAIRED_BCE_CHUNK_OUTCOME_SCHEMA
        ):
            raise ValueError("paired-BCE chunk outcome crossed its authority firewall")

    @property
    def outcome_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-paired-bce-chunk-outcome-",
        )

    def __hash__(self) -> int:
        return hash(self.outcome_hash)


@dataclass(frozen=True, slots=True)
class V1RuntimeState:
    """Complete resumable orchestration state, including terminal firewall."""

    config: V1RuntimeExecutionConfig
    controller: V1ControllerState
    entropy_tape: V1StagedAlphaEntropyTape
    pending_look: V1RuntimePendingLook | None = None
    pending_paired_bce_chunk: V1RuntimePendingPairedBCEChunk | None = None
    terminal_freeze: V1TerminalFreezeReceipt | None = None
    terminal_validation: V1TerminalValidationReceipt | None = None
    completion: V1RunCompletionReceipt | None = None
    performance: V1PerformanceReceipt | None = None
    schema: str = _RUNTIME_STATE_SCHEMA
    proposal_progress: V1RuntimeProposalAcquisitionProgress | None = None
    pending_proposal_acquisition: V1RuntimePendingProposalAcquisition | None = None
    completed_proposal_closures: tuple[
        V1RuntimeCompletedProposalClosureReceipt, ...
    ] = ()
    staged_proposal_backend_identity_hash: str | None = None
    terminal_validation_family_freeze: (
        V1TerminalValidationFamilyFreezeReceipt | None
    ) = None
    terminal_validation_family_request: (
        V1TerminalValidationFamilyRequestReceipt | None
    ) = None
    terminal_validation_family_observation: (
        V1TerminalValidationFamilyObservationReceipt | None
    ) = None
    terminal_validation_family: V1TerminalValidationFamilyReceipt | None = None
    terminal_comparison_family_freeze: (
        V1TerminalComparisonFamilyFreezeReceiptV2 | None
    ) = None
    terminal_comparison_family_request: (
        V1TerminalComparisonFamilyRequestReceiptV2 | None
    ) = None
    terminal_comparison_family_observation: (
        V1TerminalComparisonFamilyObservationReceiptV2 | None
    ) = None
    terminal_comparison_family: V1TerminalComparisonFamilyReceiptV2 | None = None

    def __post_init__(self) -> None:
        if type(self.config) is not V1RuntimeExecutionConfig:
            raise TypeError("runtime state requires an exact execution config")
        if type(self.controller) is not V1ControllerState:
            raise TypeError("runtime state requires an exact controller state")
        if type(self.entropy_tape) is not V1StagedAlphaEntropyTape:
            raise TypeError("runtime state requires an exact staged-alpha entropy tape")
        completed_closures = tuple(self.completed_proposal_closures)
        staged_backend = self.staged_proposal_backend_identity_hash
        if staged_backend is not None:
            staged_backend = _nonempty(
                staged_backend,
                "staged_proposal_backend_identity_hash",
            )
            object.__setattr__(
                self,
                "staged_proposal_backend_identity_hash",
                staged_backend,
            )
        if any(
            type(item) is not V1RuntimeCompletedProposalClosureReceipt
            for item in completed_closures
        ):
            raise TypeError("completed proposal closure ledger contains a nonreceipt")
        for closure in completed_closures:
            closure.__post_init__()
        closure_hashes = tuple(item.receipt_hash for item in completed_closures)
        if len(set(closure_hashes)) != len(closure_hashes):
            raise ValueError("completed proposal closure ledger contains duplicates")
        if completed_closures and len(completed_closures) != len(
            self.controller.acquisition_receipt_hashes
        ):
            raise ValueError(
                "staged proposal closure ledger and acquisition lineage disagree"
            )
        backend_identities = {
            item.proposal_backend_identity_hash for item in completed_closures
        }
        if len(backend_identities) > 1:
            raise ValueError("staged proposal closure ledger mixes backend identities")
        if completed_closures and backend_identities != {staged_backend}:
            raise ValueError("staged proposal closure changed its backend anchor")
        controller_lineage = self.controller.mask_lineage_hashes
        transition_hashes = self.controller.transition_receipt_hashes
        for generation, closure in enumerate(completed_closures):
            alpha_bound = closure.proposal_order_alpha_schedule_hash is not None
            if generation == 0:
                if alpha_bound:
                    from .modern_v1_proposal_alpha_ledger import (
                        initialize_v1_proposal_order_alpha_ledger,
                        restore_v1_proposal_order_alpha_ledger_replay,
                    )

                    restored = restore_v1_proposal_order_alpha_ledger_replay(
                        closure.proposal_order_alpha_ledger_after_replay  # type: ignore[arg-type]
                    )
                    genesis = initialize_v1_proposal_order_alpha_ledger(
                        restored.schedule
                    )
                    if (
                        genesis.ledger_hash
                        != closure.proposal_order_alpha_ledger_before_hash
                    ):
                        raise ValueError(
                            "generation-zero proposal closure does not name the "
                            "registered genesis alpha ledger"
                        )
            else:
                prior_closure = completed_closures[generation - 1]
                prior_alpha_bound = (
                    prior_closure.proposal_order_alpha_schedule_hash is not None
                )
                if alpha_bound != prior_alpha_bound:
                    raise ValueError(
                        "completed proposal closures mix alpha-bound and legacy chains"
                    )
                if alpha_bound and (
                    closure.proposal_order_alpha_schedule_hash
                    != prior_closure.proposal_order_alpha_schedule_hash
                    or closure.proposal_order_alpha_ledger_before_hash
                    != prior_closure.proposal_order_alpha_ledger_after_hash
                ):
                    raise ValueError(
                        "completed proposal closures fork or gap the proposal-alpha chain"
                    )
            prefix_lineage = controller_lineage[: generation + 1]
            prefix_transitions = transition_hashes[:generation]
            attempt_count = len(
                closure.pre_attach_controller_snapshot.attempt_receipt_hashes
            )
            attempt_prefix = self.controller.attempt_receipts[:attempt_count]
            attempt_hash_prefix = tuple(item.receipt_hash for item in attempt_prefix)
            prefix_draw_ids = tuple(
                draw_id
                for attempt in attempt_prefix
                for draw_id in attempt.draw_ids
            )
            expected_pre_state_hash = _proposal_closure_controller_state_hash(
                snapshot=closure.pre_attach_controller_snapshot,
                controller_schema=self.controller.schema,
                initial_mask_hash=self.controller.initial_mask.mask_hash,
                acquisition_hash=None,
                globally_used_draw_ids=prefix_draw_ids,
            )
            expected_post_state_hash = _proposal_closure_controller_state_hash(
                snapshot=closure.post_attach_controller_snapshot,
                controller_schema=self.controller.schema,
                initial_mask_hash=self.controller.initial_mask.mask_hash,
                acquisition_hash=(
                    closure.acquisition_hash
                    if closure.post_attach_controller_snapshot.phase == "queue_ready"
                    else None
                ),
                globally_used_draw_ids=prefix_draw_ids,
            )
            if (
                closure.acquisition_generation_index != generation
                or closure.campaign_contract_hash
                != self.controller.binding.contract.campaign_contract_hash
                or closure.campaign_binding_hash
                != self.controller.binding.binding_hash
                or closure.controller_run_id != self.controller.controller_run_id
                or closure.current_mask_hash != controller_lineage[generation]
                or closure.route_axis_hash != self.controller.current_mask.route_axis_hash
                or closure.mask_lineage_hash
                != _proposal_mask_lineage_hash(prefix_lineage, prefix_transitions)
                or closure.candidate_identity_hash
                != _proposal_candidate_identity_hash(
                    campaign_binding_hash=closure.campaign_binding_hash,
                    generation_index=generation,
                    current_mask_hash=closure.current_mask_hash,
                    committed_mask_lineage_hashes=prefix_lineage,
                    committed_mask_transition_receipt_hashes=prefix_transitions,
                    acquisition_hash=closure.acquisition_hash,
                    acquisition_receipt_hash=closure.acquisition_receipt_hash,
                    ordered_candidate_ids=closure.ordered_candidate_ids,
                )
                or closure.acquisition_receipt_hash
                != self.controller.acquisition_receipt_hashes[generation]
                or closure.pre_attach_controller_snapshot.acquisition_receipt_hashes
                != self.controller.acquisition_receipt_hashes[:generation]
                or closure.post_attach_controller_snapshot.acquisition_receipt_hashes
                != self.controller.acquisition_receipt_hashes[: generation + 1]
                or closure.pre_attach_controller_snapshot.transition_receipt_hashes
                != transition_hashes[:generation]
                or closure.post_attach_controller_snapshot.transition_receipt_hashes
                != transition_hashes[:generation]
                or attempt_hash_prefix
                != closure.pre_attach_controller_snapshot.attempt_receipt_hashes
                or attempt_hash_prefix
                != closure.post_attach_controller_snapshot.attempt_receipt_hashes
                or closure.pre_attach_controller_state_hash
                != expected_pre_state_hash
                or closure.post_attach_controller_state_hash
                != expected_post_state_hash
            ):
                raise ValueError(
                    "completed proposal closure has stale positional acquisition lineage"
                )
        if completed_closures:
            latest = completed_closures[-1]
            if self.controller.logical_sequence == (
                latest.post_attach_controller_snapshot.logical_sequence
            ) and (
                latest.post_attach_controller_state_hash != self.controller.state_hash
                or latest.post_attach_controller_snapshot_hash
                != self.controller.snapshot_receipt().receipt_hash
            ):
                raise ValueError("latest proposal closure has stale post-attach controller")
            live_acquisition = self.controller.acquisition
            if live_acquisition is not None and (
                live_acquisition.generation_index
                == latest.acquisition_generation_index
            ) and (
                latest.acquisition_hash != live_acquisition.acquisition_hash
                or latest.acquisition_receipt_hash
                != live_acquisition.receipt.receipt_hash
            ):
                raise ValueError("latest proposal closure names another live acquisition")
        object.__setattr__(
            self,
            "completed_proposal_closures",
            completed_closures,
        )
        contract = self.controller.binding.contract
        if (
            self.entropy_tape.campaign_contract_hash != contract.campaign_contract_hash
            or self.entropy_tape.controller_run_id != self.controller.controller_run_id
            or self.entropy_tape.executor_identity_hash != self.config.executor_identity_hash
            or self.entropy_tape.maximum_candidate_count != contract.maximum_candidate_count
            or self.entropy_tape.draws_per_candidate != contract.cumulative_draw_counts[-1]
        ):
            raise ValueError("runtime entropy tape differs from its campaign/run/executor bounds")
        if self.pending_look is not None:
            if type(self.pending_look) is not V1RuntimePendingLook:
                raise TypeError("runtime pending look has the wrong type")
            active = self.controller.active_candidate
            if (
                self.controller.phase != "candidate_active"
                or active is None
                or self.pending_look.controller_snapshot_hash
                != self.controller.snapshot_receipt().receipt_hash
                or self.pending_look.active_candidate_hash != active.active_hash
                or self.pending_look.look_index != len(active.ledger.packets)
            ):
                raise ValueError("runtime pending look is stale")
            counts = active.step_freeze.cumulative_draw_counts
            look_index = self.pending_look.look_index
            if look_index >= len(counts):
                raise ValueError("runtime pending look is outside the registered schedule")
            first = 0 if look_index == 0 else counts[look_index - 1]
            count = counts[look_index] - first
            expected_draw_ids = _draw_ids(
                self.controller,
                look_index=look_index,
                count=count,
            )
            expected_plan = draw_v1_staged_alpha_sampling_plan_from_tape(
                step_freeze=active.step_freeze,
                entropy_tape=self.entropy_tape,
                candidate_attempt_index=self.controller.candidate_attempt_count,
                first_draw_index=first,
                population_context_ids=active.sampling_law.population_context_ids,
                sampling_masses_u64=active.sampling_law.sampling_masses_u64,
                draw_ids=expected_draw_ids,
                executor_identity_hash=self.config.executor_identity_hash,
            )
            if (
                self.pending_look.sampling_plan.plan_hash != expected_plan.plan_hash
                or set(self.pending_look.sampling_plan.draw_ids).intersection(
                    self.controller.globally_used_draw_ids
                )
            ):
                raise ValueError(
                    "runtime pending look differs from its exact persisted entropy draw"
                )
            expected_request, expected_evidence = _pending_request_and_evidence(
                config=self.config,
                controller=self.controller,
                active=active,
                look_index=look_index,
                plan=expected_plan,
            )
            if (
                self.pending_look.request.request_id != expected_request.request_id
                or self.pending_look.evidence_packet.packet_hash
                != expected_evidence.packet_hash
            ):
                raise ValueError(
                    "runtime pending look differs from its exact pre-outcome request lineage"
                )
        if self.pending_paired_bce_chunk is not None:
            pending_paired = self.pending_paired_bce_chunk
            if type(pending_paired) is not V1RuntimePendingPairedBCEChunk:
                raise TypeError("runtime pending paired-BCE chunk has the wrong type")
            active = self.controller.active_candidate
            if (
                self.controller.phase != "candidate_active"
                or active is None
                or active.paired_bce_registration is None
                or active.paired_bce_state is None
                or pending_paired.controller_snapshot_hash
                != self.controller.snapshot_receipt().receipt_hash
                or pending_paired.active_candidate_hash != active.active_hash
                or pending_paired.registration_hash
                != active.paired_bce_registration.registration_hash
                or pending_paired.prior_paired_bce_state_hash
                != active.paired_bce_state.state_hash
                or pending_paired.request.registration_hash
                != active.paired_bce_registration.registration_hash
                or pending_paired.request.prior_state_hash
                != active.paired_bce_state.state_hash
                or pending_paired.dynamic_oracle_receipt_hash
                != active.paired_bce_registration.dynamic_oracle_receipt_hash
                or pending_paired.request.dynamic_oracle_receipt_hash
                != active.paired_bce_registration.dynamic_oracle_receipt_hash
            ):
                raise ValueError("runtime pending paired-BCE chunk is stale")
        if self.proposal_progress is not None:
            if type(self.proposal_progress) is not V1RuntimeProposalAcquisitionProgress:
                raise TypeError("runtime proposal progress has the wrong type")
            proposal_horizon = getattr(
                self.controller,
                "fqr_no_candidate_horizon_witness",
                None,
            )
            active_relinearization = self.controller.phase == "needs_acquisition"
            witnessed_horizon = (
                self.controller.phase == "stopped"
                and type(proposal_horizon) is V1ControllerProposalHorizonWitness
            )
            if not (active_relinearization or witnessed_horizon):
                raise ValueError(
                    "runtime proposal evidence requires active relinearization or "
                    "an exact stable-negative F/Q/R horizon witness"
                )
            _assert_proposal_progress_bound_to_controller(
                self.proposal_progress,
                self.controller,
            )
            if len(completed_closures) != self.proposal_progress.acquisition_generation_index:
                raise ValueError(
                    "live staged proposal generation differs from its closure ledger"
                )
            if (
                self.proposal_progress.proposal_backend_identity_hash
                != staged_backend
            ):
                raise ValueError("live proposal generation changed its backend identity")
            if any(
                item.receipt_hash
                in {closure.freeze_decision_hash for closure in completed_closures}
                for item in self.proposal_progress.freeze_decision_receipts
            ):
                raise ValueError("live proposal generation reuses a completed freeze decision")
            if not witnessed_horizon and any(
                item is not None
                for item in (
                    self.terminal_freeze,
                    self.terminal_validation,
                    self.completion,
                    self.performance,
                    self.terminal_validation_family_freeze,
                    self.terminal_validation_family_request,
                    self.terminal_validation_family_observation,
                    self.terminal_validation_family,
                    self.terminal_comparison_family_freeze,
                    self.terminal_comparison_family_request,
                    self.terminal_comparison_family_observation,
                    self.terminal_comparison_family,
                )
            ):
                raise ValueError("runtime proposal acquisition crossed the terminal firewall")
            if witnessed_horizon and self.pending_proposal_acquisition is not None:
                raise ValueError(
                    "stable-negative F/Q/R horizon retained an unobserved proposal request"
                )
        if self.pending_proposal_acquisition is not None:
            if type(
                self.pending_proposal_acquisition
            ) is not V1RuntimePendingProposalAcquisition:
                raise TypeError("runtime pending proposal acquisition has the wrong type")
            if self.proposal_progress is None:
                raise ValueError("runtime pending proposal lacks its proposal progress")
            _assert_pending_proposal_bound(
                self.proposal_progress,
                self.pending_proposal_acquisition,
            )
            if set(
                self.pending_proposal_acquisition.request.train_draw_ids
            ).intersection(self.controller.globally_used_draw_ids):
                raise ValueError("proposal train draws overlap the paired-oracle draw ledger")
        pending_count = sum(
            item is not None
            for item in (
                self.pending_look,
                self.pending_paired_bce_chunk,
                self.pending_proposal_acquisition,
            )
        )
        if pending_count > 1:
            raise ValueError("runtime cannot execute multiple pending outcomes concurrently")
        for value, expected, label in (
            (self.terminal_freeze, V1TerminalFreezeReceipt, "terminal freeze"),
            (self.terminal_validation, V1TerminalValidationReceipt, "terminal validation"),
            (
                self.terminal_validation_family_freeze,
                V1TerminalValidationFamilyFreezeReceipt,
                "terminal validation family freeze",
            ),
            (
                self.terminal_validation_family_request,
                V1TerminalValidationFamilyRequestReceipt,
                "terminal validation family request",
            ),
            (
                self.terminal_validation_family_observation,
                V1TerminalValidationFamilyObservationReceipt,
                "terminal validation family observation",
            ),
            (
                self.terminal_validation_family,
                V1TerminalValidationFamilyReceipt,
                "terminal validation family",
            ),
            (
                self.terminal_comparison_family_freeze,
                V1TerminalComparisonFamilyFreezeReceiptV2,
                "terminal comparison family freeze v2",
            ),
            (
                self.terminal_comparison_family_request,
                V1TerminalComparisonFamilyRequestReceiptV2,
                "terminal comparison family request v2",
            ),
            (
                self.terminal_comparison_family_observation,
                V1TerminalComparisonFamilyObservationReceiptV2,
                "terminal comparison family observation v2",
            ),
            (
                self.terminal_comparison_family,
                V1TerminalComparisonFamilyReceiptV2,
                "terminal comparison family receipt v2",
            ),
            (self.completion, V1RunCompletionReceipt, "completion"),
            (self.performance, V1PerformanceReceipt, "performance"),
        ):
            if value is not None and type(value) is not expected:
                raise TypeError(f"runtime {label} has the wrong type")
        snapshot = self.controller.snapshot_receipt()
        if self.terminal_freeze is not None:
            freeze = self.terminal_freeze
            expected_freeze = {
                "campaign_contract_hash": contract.campaign_contract_hash,
                "campaign_binding_hash": self.controller.binding.binding_hash,
                "controller_run_id": self.controller.controller_run_id,
                "controller_snapshot_hash": snapshot.receipt_hash,
                "terminal_mask_hash": self.controller.current_mask.mask_hash,
                "accepted_trajectory_hash": (
                    self.controller.accepted_trajectory_hash
                ),
                "dense_reference_hash": contract.dense_reference_hash,
                "terminal_validation_population_hash": (
                    contract.terminal_validation_population_hash
                ),
                "acquisition_receipt_hashes": self.controller.acquisition_receipt_hashes,
                "attempt_receipt_hashes": self.controller.attempt_receipt_hashes,
                "transition_receipt_hashes": self.controller.transition_receipt_hashes,
                "stop_reason": self.controller.stop_reason,
                "freeze_sequence": self.controller.logical_sequence + 1,
            }
            if self.controller.phase != "stopped" or any(
                getattr(freeze, name) != expected
                for name, expected in expected_freeze.items()
            ):
                raise ValueError("terminal freeze differs from the exact stopped controller")
        if self.terminal_validation is not None:
            validation = self.terminal_validation
            if (
                self.terminal_freeze is None
                or validation.terminal_freeze_hash != self.terminal_freeze.receipt_hash
                or validation.calibration_identity != contract.calibration_identity
                or validation.terminal_mask_hash
                != self.terminal_freeze.terminal_mask_hash
                or validation.reveal_sequence != self.terminal_freeze.freeze_sequence + 1
            ):
                raise ValueError("terminal validation lacks its exact freeze")
            expected_request_hash = content_hash(
                {
                    "terminal_freeze_hash": self.terminal_freeze.receipt_hash,
                    "terminal_mask_hash": self.terminal_freeze.terminal_mask_hash,
                    "terminal_validation_population_hash": (
                        self.terminal_freeze.terminal_validation_population_hash
                    ),
                    "calibration_identity": contract.calibration_identity,
                },
                prefix="partition-free-v1-terminal-request-",
            )
            expected_outcome_hash = content_hash(
                {
                    "terminal_request_hash": expected_request_hash,
                    "terminal_freeze_hash": self.terminal_freeze.receipt_hash,
                    "validation_values_hash": validation.validation_values_hash,
                },
                prefix="partition-free-v1-terminal-outcome-",
            )
            if (
                validation.terminal_request_hash != expected_request_hash
                or validation.terminal_outcome_hash != expected_outcome_hash
            ):
                raise ValueError("terminal validation request/outcome lineage is invalid")
        family_coordinates = (
            self.terminal_validation_family_freeze,
            self.terminal_validation_family_request,
            self.terminal_validation_family_observation,
            self.terminal_validation_family,
        )
        comparison_coordinates = (
            self.terminal_comparison_family_freeze,
            self.terminal_comparison_family_request,
            self.terminal_comparison_family_observation,
            self.terminal_comparison_family,
        )
        for item in comparison_coordinates:
            if item is not None:
                item.__post_init__()
        if self.terminal_validation is not None and any(
            item is not None
            for item in (*family_coordinates, *comparison_coordinates)
        ):
            raise ValueError(
                "legacy and terminal-family validation are mutually exclusive"
            )
        if any(item is not None for item in family_coordinates) and any(
            item is not None for item in comparison_coordinates
        ):
            raise ValueError(
                "terminal validation family v1 and comparison family v2 are mutually exclusive"
            )
        seen_family_gap = False
        for item in family_coordinates:
            if item is None:
                seen_family_gap = True
            elif seen_family_gap:
                raise ValueError("terminal validation family chain skips a causal joint")
        family_freeze = self.terminal_validation_family_freeze
        if family_freeze is not None:
            terminal_freeze = self.terminal_freeze
            milestone = self.controller.pareto_milestone_receipt
            initial_mask = self.controller.initial_mask
            current_mask = self.controller.current_mask
            if (
                terminal_freeze is None
                or family_freeze.terminal_freeze_hash
                != terminal_freeze.receipt_hash
                or family_freeze.freeze_sequence
                != terminal_freeze.freeze_sequence + 1
                or family_freeze.terminal_validation_population_hash
                != terminal_freeze.terminal_validation_population_hash
                or family_freeze.dense_reference_hash
                != terminal_freeze.dense_reference_hash
                or family_freeze.dense_mask_hash != initial_mask.mask_hash
                or family_freeze.final_mask_hash != current_mask.mask_hash
                or family_freeze.route_axis_hash != current_mask.route_axis_hash
                or family_freeze.route_count != current_mask.route_count
                or family_freeze.role_retained_route_counts[0]
                != initial_mask.retained_count
                or family_freeze.role_retained_route_counts[2]
                != current_mask.retained_count
                or milestone is None
                or family_freeze.pareto_milestone_receipt.receipt_hash
                != milestone.receipt_hash
            ):
                raise ValueError(
                    "terminal validation family freeze differs from its frozen run"
                )
        family_request = self.terminal_validation_family_request
        if family_request is not None and (
            family_freeze is None
            or family_request.family_freeze.receipt_hash
            != family_freeze.receipt_hash
        ):
            raise ValueError("terminal validation family request lacks its exact freeze")
        family_observation = self.terminal_validation_family_observation
        if family_observation is not None and (
            family_request is None
            or family_observation.request.request_hash != family_request.request_hash
        ):
            raise ValueError(
                "terminal validation family observation lacks its exact request"
            )
        family_receipt = self.terminal_validation_family
        if family_receipt is not None and (
            family_observation is None
            or family_receipt.observation.terminal_outcome_hash
            != family_observation.terminal_outcome_hash
        ):
            raise ValueError("terminal validation family lacks its exact observation")
        seen_comparison_gap = False
        for item in comparison_coordinates:
            if item is None:
                seen_comparison_gap = True
            elif seen_comparison_gap:
                raise ValueError("terminal comparison family chain skips a causal joint")
        comparison_freeze = self.terminal_comparison_family_freeze
        if comparison_freeze is not None:
            terminal_freeze = self.terminal_freeze
            initial_mask = self.controller.initial_mask
            current_mask = self.controller.current_mask
            if (
                terminal_freeze is None
                or comparison_freeze.terminal_freeze_hash
                != terminal_freeze.receipt_hash
                or comparison_freeze.freeze_sequence
                != terminal_freeze.freeze_sequence + 1
                or comparison_freeze.terminal_validation_population_hash
                != terminal_freeze.terminal_validation_population_hash
                or comparison_freeze.dense_mask_hash != initial_mask.mask_hash
                or comparison_freeze.final_v1_mask_hash != current_mask.mask_hash
                or comparison_freeze.route_axis_hash != current_mask.route_axis_hash
                or comparison_freeze.route_count != current_mask.route_count
                or comparison_freeze.role_retained_route_counts[0]
                != initial_mask.retained_count
                or comparison_freeze.role_retained_route_counts[2]
                != current_mask.retained_count
            ):
                raise ValueError(
                    "terminal comparison family freeze differs from its frozen run"
                )
        comparison_request = self.terminal_comparison_family_request
        if comparison_request is not None and (
            comparison_freeze is None
            or comparison_request.family_freeze.receipt_hash
            != comparison_freeze.receipt_hash
        ):
            raise ValueError("terminal comparison request lacks its exact freeze")
        comparison_observation = self.terminal_comparison_family_observation
        if comparison_observation is not None and (
            comparison_request is None
            or comparison_observation.request.request_hash
            != comparison_request.request_hash
        ):
            raise ValueError("terminal comparison observation lacks its exact request")
        comparison_receipt = self.terminal_comparison_family
        if comparison_receipt is not None and (
            comparison_observation is None
            or comparison_receipt.observation.terminal_outcome_hash
            != comparison_observation.terminal_outcome_hash
        ):
            raise ValueError("terminal comparison family lacks its exact observation")
        terminal_authority = (
            self.terminal_validation
            if self.terminal_validation is not None
            else (
                self.terminal_validation_family
                if self.terminal_validation_family is not None
                else self.terminal_comparison_family
            )
        )
        if self.performance is not None:
            if terminal_authority is None:
                raise ValueError("performance receipt requires completed terminal validation")
            if (
                self.performance.controller_run_id != self.controller.controller_run_id
                or self.performance.source_environment_hash
                != self.config.performance_source_environment_hash
            ):
                raise ValueError("performance receipt names another run/source/environment")
        if self.completion is not None:
            completion = self.completion
            if (
                terminal_authority is None
                or self.terminal_freeze is None
                or completion.campaign_contract_hash != contract.campaign_contract_hash
                or completion.campaign_binding_hash != self.controller.binding.binding_hash
                or completion.controller_run_id != self.controller.controller_run_id
                or completion.final_controller_snapshot_hash != snapshot.receipt_hash
                or completion.terminal_freeze_hash != self.terminal_freeze.receipt_hash
                or completion.terminal_validation_hash
                != terminal_authority.receipt_hash
                or completion.terminal_mask_hash != self.controller.current_mask.mask_hash
                or completion.accepted_step_count != self.controller.accepted_step_count
                or completion.candidate_attempt_count
                != self.controller.candidate_attempt_count
                or completion.performance_receipt_hash
                != (None if self.performance is None else self.performance.receipt_hash)
                or completion.source_seal_hash != self.config.source_seal_hash
                or completion.environment_hash != self.config.environment_hash
                or completion.completion_sequence
                != terminal_authority.reveal_sequence + 1
            ):
                raise ValueError("runtime completion differs from its exact completed run")
        if (
            self.pending_look is not None
            or self.pending_paired_bce_chunk is not None
            or self.pending_proposal_acquisition is not None
        ) and self.terminal_freeze is not None:
            raise ValueError("runtime cannot retain pending selection work after terminal freeze")
        has_staged_authority = bool(
            self.proposal_progress is not None
            or self.pending_proposal_acquisition is not None
            or completed_closures
        )
        if has_staged_authority is not (staged_backend is not None):
            raise ValueError("runtime staged proposal backend anchor is absent or stale")
        if self.schema != _RUNTIME_STATE_SCHEMA:
            raise ValueError("unsupported modern-v1 runtime-state schema")

    @property
    def completed_proposal_freeze_decisions(
        self,
    ) -> tuple[V1ProposalFreezeDecisionReceipt, ...]:
        """Read-only compatibility view; closures remain the authority."""

        return tuple(item.freeze_decision for item in self.completed_proposal_closures)

    @property
    def runtime_state_hash(self) -> str:
        completed_closures = self.completed_proposal_closures
        closure_hashes = tuple(item.receipt_hash for item in completed_closures)
        projection: dict[str, Any] = {
            "schema": self.schema,
            "logical_config_hash": self.config.logical_identity_hash,
            "entropy_tape_hash": self.entropy_tape.tape_hash,
            "controller_state_hash": self.controller.state_hash,
            "pending_hash": None if self.pending_look is None else self.pending_look.pending_hash,
            "terminal_freeze_hash": (
                None if self.terminal_freeze is None else self.terminal_freeze.receipt_hash
            ),
            "terminal_validation_hash": (
                None
                if self.terminal_validation is None
                else self.terminal_validation.receipt_hash
            ),
            "completion_hash": None if self.completion is None else self.completion.receipt_hash,
            "performance_hash": (
                None if self.performance is None else self.performance.receipt_hash
            ),
        }
        # Preserve the exact historical v2 state-hash projection whenever the
        # staged proposal protocol is absent, so existing synthetic/checkpoint
        # identities are not rewritten merely by adding None fields.
        if (
            self.proposal_progress is not None
            or self.pending_proposal_acquisition is not None
            or completed_closures
        ):
            projection.update(
                {
                    "proposal_progress_hash": (
                        None
                        if self.proposal_progress is None
                        else self.proposal_progress.progress_hash
                    ),
                    "pending_proposal_acquisition_hash": (
                        None
                        if self.pending_proposal_acquisition is None
                        else self.pending_proposal_acquisition.pending_hash
                    ),
                    "completed_proposal_closure_hashes": closure_hashes,
                    "staged_proposal_backend_identity_hash": (
                        self.staged_proposal_backend_identity_hash
                    ),
                }
            )
        if self.pending_paired_bce_chunk is not None:
            projection["pending_paired_bce_chunk_hash"] = (
                self.pending_paired_bce_chunk.pending_hash
            )
        if any(
            item is not None
            for item in (
                self.terminal_validation_family_freeze,
                self.terminal_validation_family_request,
                self.terminal_validation_family_observation,
                self.terminal_validation_family,
            )
        ):
            projection.update(
                {
                    "terminal_validation_family_freeze_hash": (
                        None
                        if self.terminal_validation_family_freeze is None
                        else self.terminal_validation_family_freeze.receipt_hash
                    ),
                    "terminal_validation_family_request_hash": (
                        None
                        if self.terminal_validation_family_request is None
                        else self.terminal_validation_family_request.request_hash
                    ),
                    "terminal_validation_family_observation_hash": (
                        None
                        if self.terminal_validation_family_observation is None
                        else self.terminal_validation_family_observation.terminal_outcome_hash
                    ),
                    "terminal_validation_family_hash": (
                        None
                        if self.terminal_validation_family is None
                        else self.terminal_validation_family.receipt_hash
                    ),
                }
            )
        if any(
            item is not None
            for item in (
                self.terminal_comparison_family_freeze,
                self.terminal_comparison_family_request,
                self.terminal_comparison_family_observation,
                self.terminal_comparison_family,
            )
        ):
            projection.update(
                {
                    "terminal_comparison_family_freeze_hash": (
                        None
                        if self.terminal_comparison_family_freeze is None
                        else self.terminal_comparison_family_freeze.receipt_hash
                    ),
                    "terminal_comparison_family_request_hash": (
                        None
                        if self.terminal_comparison_family_request is None
                        else self.terminal_comparison_family_request.request_hash
                    ),
                    "terminal_comparison_family_observation_hash": (
                        None
                        if self.terminal_comparison_family_observation is None
                        else self.terminal_comparison_family_observation.terminal_outcome_hash
                    ),
                    "terminal_comparison_family_hash": (
                        None
                        if self.terminal_comparison_family is None
                        else self.terminal_comparison_family.receipt_hash
                    ),
                }
            )
        return content_hash(
            projection,
            prefix="partition-free-v1-runtime-state-",
        )

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def selector_promoted(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1RuntimeAcquisitionSidecarReceipt:
    """Durable content address for one immutable live acquisition payload."""

    relative_path: str
    payload_sha256: str
    payload_size_bytes: int
    framed_size_bytes: int
    acquisition_hash: str
    acquisition_receipt_hash: str
    campaign_binding_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    schema: str = _ACQUISITION_SIDECAR_SCHEMA

    def __post_init__(self) -> None:
        relative = Path(_nonempty(self.relative_path, "relative_path"))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValueError("acquisition sidecar path must be a safe two-part relative path")
        object.__setattr__(self, "relative_path", relative.as_posix())
        object.__setattr__(
            self,
            "payload_sha256",
            _sha256_hex(self.payload_sha256, "payload_sha256"),
        )
        payload_size = _positive(self.payload_size_bytes, "payload_size_bytes")
        framed_size = _positive(self.framed_size_bytes, "framed_size_bytes")
        if framed_size <= payload_size:
            raise ValueError("acquisition sidecar framing size is inconsistent")
        object.__setattr__(self, "payload_size_bytes", payload_size)
        object.__setattr__(self, "framed_size_bytes", framed_size)
        for name in (
            "acquisition_hash",
            "acquisition_receipt_hash",
            "campaign_binding_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative(self.generation_index, "generation_index"),
        )
        if self.schema != _ACQUISITION_SIDECAR_SCHEMA:
            raise ValueError("unsupported modern-v1 acquisition-sidecar schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-runtime-acquisition-sidecar-",
        )


@dataclass(frozen=True, slots=True)
class V1RuntimeCheckpointReceipt:
    checkpoint_path: str
    payload_sha256: str
    payload_size_bytes: int
    runtime_state_hash: str
    controller_snapshot_hash: str
    pending_look_hash: str | None
    acquisition_sidecar: V1RuntimeAcquisitionSidecarReceipt | None = None
    proposal_state_sidecar: V1RuntimeProposalStateSidecarReceipt | None = None
    completed_proposal_closure_hashes: tuple[str, ...] = ()
    staged_proposal_backend_identity_hash: str | None = None
    pending_paired_bce_chunk_hash: str | None = None
    terminal_validation_family_freeze_hash: str | None = None
    terminal_validation_family_request_hash: str | None = None
    terminal_validation_family_artifact: Mapping[str, Any] | None = None
    terminal_comparison_family_freeze_hash: str | None = None
    terminal_comparison_family_request_hash: str | None = None
    terminal_comparison_family_artifact: Mapping[str, Any] | None = None
    schema: str = _CHECKPOINT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_path",
            "payload_sha256",
            "runtime_state_hash",
            "controller_snapshot_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "payload_size_bytes",
            _positive(self.payload_size_bytes, "payload_size_bytes"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            _sha256_hex(self.payload_sha256, "payload_sha256"),
        )
        if self.pending_look_hash is not None:
            object.__setattr__(
                self,
                "pending_look_hash",
                _nonempty(self.pending_look_hash, "pending_look_hash"),
            )
        if self.acquisition_sidecar is not None and type(
            self.acquisition_sidecar
        ) is not V1RuntimeAcquisitionSidecarReceipt:
            raise TypeError("checkpoint acquisition sidecar has the wrong type")
        if self.proposal_state_sidecar is not None and type(
            self.proposal_state_sidecar
        ) is not V1RuntimeProposalStateSidecarReceipt:
            raise TypeError("checkpoint proposal-state sidecar has the wrong type")
        completed_hashes = _nonempty_string_tuple(
            self.completed_proposal_closure_hashes,
            "completed_proposal_closure_hashes",
            allow_empty=True,
        )
        if len(set(completed_hashes)) != len(completed_hashes):
            raise ValueError("checkpoint proposal freeze hashes contain duplicates")
        object.__setattr__(
            self,
            "completed_proposal_closure_hashes",
            completed_hashes,
        )
        staged_backend = self.staged_proposal_backend_identity_hash
        if staged_backend is not None:
            object.__setattr__(
                self,
                "staged_proposal_backend_identity_hash",
                _nonempty(staged_backend, "staged_proposal_backend_identity_hash"),
            )
        paired_pending = self.pending_paired_bce_chunk_hash
        if paired_pending is not None:
            object.__setattr__(
                self,
                "pending_paired_bce_chunk_hash",
                _nonempty(paired_pending, "pending_paired_bce_chunk_hash"),
            )
        family_freeze_hash = self.terminal_validation_family_freeze_hash
        if family_freeze_hash is not None:
            object.__setattr__(
                self,
                "terminal_validation_family_freeze_hash",
                _nonempty(
                    family_freeze_hash,
                    "terminal_validation_family_freeze_hash",
                ),
            )
        family_request_hash = self.terminal_validation_family_request_hash
        if family_request_hash is not None:
            object.__setattr__(
                self,
                "terminal_validation_family_request_hash",
                _nonempty(
                    family_request_hash,
                    "terminal_validation_family_request_hash",
                ),
            )
        family_artifact = self.terminal_validation_family_artifact
        if family_artifact is not None:
            family_artifact = _validate_terminal_family_artifact_reference(
                family_artifact
            )
            object.__setattr__(
                self,
                "terminal_validation_family_artifact",
                MappingProxyType(dict(family_artifact)),
            )
        comparison_freeze_hash = self.terminal_comparison_family_freeze_hash
        if comparison_freeze_hash is not None:
            object.__setattr__(
                self,
                "terminal_comparison_family_freeze_hash",
                _nonempty(
                    comparison_freeze_hash,
                    "terminal_comparison_family_freeze_hash",
                ),
            )
        comparison_request_hash = self.terminal_comparison_family_request_hash
        if comparison_request_hash is not None:
            object.__setattr__(
                self,
                "terminal_comparison_family_request_hash",
                _nonempty(
                    comparison_request_hash,
                    "terminal_comparison_family_request_hash",
                ),
            )
        comparison_artifact = self.terminal_comparison_family_artifact
        if comparison_artifact is not None:
            comparison_artifact = _validate_terminal_comparison_artifact_reference_v2(
                comparison_artifact
            )
            object.__setattr__(
                self,
                "terminal_comparison_family_artifact",
                MappingProxyType(dict(comparison_artifact)),
            )
        if self.schema not in {
            _LEGACY_CHECKPOINT_SCHEMA,
            _CHECKPOINT_SCHEMA,
            _CHECKPOINT_V3_SCHEMA,
        }:
            raise ValueError("unsupported modern-v1 checkpoint schema")
        has_staged_state = bool(
            self.proposal_state_sidecar is not None or completed_hashes
        )
        has_paired_pending = self.pending_paired_bce_chunk_hash is not None
        has_family = self.terminal_validation_family_freeze_hash is not None
        has_comparison = self.terminal_comparison_family_freeze_hash is not None
        if (self.schema == _CHECKPOINT_V3_SCHEMA) is not (
            has_staged_state or has_paired_pending or has_family or has_comparison
        ):
            raise ValueError("checkpoint schema and v3 live state disagree")
        if has_staged_state is not (staged_backend is not None):
            raise ValueError("checkpoint staged proposal state and backend anchor disagree")
        if self.schema != _CHECKPOINT_V3_SCHEMA and has_paired_pending:
            raise ValueError("legacy checkpoint receipt cannot carry paired-BCE state")
        if family_request_hash is not None and not has_family:
            raise ValueError("terminal family request lacks its freeze identity")
        if (family_artifact is not None) and family_request_hash is None:
            raise ValueError("terminal family artifact lacks its request identity")
        if self.schema != _CHECKPOINT_V3_SCHEMA and has_family:
            raise ValueError("legacy checkpoint receipt cannot carry terminal-family state")
        if has_family and has_comparison:
            raise ValueError("checkpoint cannot mix terminal family v1 and comparison v2")
        if comparison_request_hash is not None and not has_comparison:
            raise ValueError("terminal comparison request lacks its freeze identity")
        if comparison_artifact is not None and comparison_request_hash is None:
            raise ValueError("terminal comparison artifact lacks its request identity")
        if self.schema != _CHECKPOINT_V3_SCHEMA and has_comparison:
            raise ValueError("legacy checkpoint cannot carry terminal comparison state")

    @property
    def receipt_hash(self) -> str:
        # Preserve the exact historical receipt identity when no proposal
        # state exists; an optional-None dataclass field must not rewrite v2.
        projection: dict[str, Any] = {
            "checkpoint_path": self.checkpoint_path,
            "payload_sha256": self.payload_sha256,
            "payload_size_bytes": self.payload_size_bytes,
            "runtime_state_hash": self.runtime_state_hash,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "pending_look_hash": self.pending_look_hash,
            "acquisition_sidecar": self.acquisition_sidecar,
            "schema": self.schema,
        }
        if self.schema == _CHECKPOINT_V3_SCHEMA:
            projection["proposal_state_sidecar"] = self.proposal_state_sidecar
            projection["completed_proposal_closure_hashes"] = (
                self.completed_proposal_closure_hashes
            )
            projection["staged_proposal_backend_identity_hash"] = (
                self.staged_proposal_backend_identity_hash
            )
            if self.pending_paired_bce_chunk_hash is not None:
                projection["pending_paired_bce_chunk_hash"] = (
                    self.pending_paired_bce_chunk_hash
                )
            if self.terminal_validation_family_freeze_hash is not None:
                projection["terminal_validation_family_freeze_hash"] = (
                    self.terminal_validation_family_freeze_hash
                )
                projection["terminal_validation_family_request_hash"] = (
                    self.terminal_validation_family_request_hash
                )
                projection["terminal_validation_family_artifact"] = (
                    self.terminal_validation_family_artifact
                )
            if self.terminal_comparison_family_freeze_hash is not None:
                projection["terminal_comparison_family_freeze_hash"] = (
                    self.terminal_comparison_family_freeze_hash
                )
                projection["terminal_comparison_family_request_hash"] = (
                    self.terminal_comparison_family_request_hash
                )
                projection["terminal_comparison_family_artifact"] = (
                    self.terminal_comparison_family_artifact
                )
        return content_hash(
            projection,
            prefix="partition-free-v1-runtime-checkpoint-",
        )


@dataclass(frozen=True, slots=True)
class _V1ControllerCheckpointProjection:
    """Controller state with its immutable acquisition replaced by a receipt."""

    binding: Any
    controller_run_id: str
    initial_mask: Any
    current_mask: Any
    phase: str
    queue_cursor: int
    active_candidate: V1ActiveCandidate | None
    accepted_trajectory: Any
    attempt_receipts: tuple[Any, ...]
    transition_receipts: tuple[Any, ...]
    acquisition_receipt_hashes: tuple[str, ...]
    globally_used_draw_ids: tuple[str, ...]
    stop_reason: Any
    logical_sequence: int
    controller_schema: str
    acquisition_sidecar: V1RuntimeAcquisitionSidecarReceipt | None
    controller_state_hash: str
    controller_snapshot_hash: str
    accepted_oracle_trajectory: Any = None
    pareto_milestone_registration: V1ParetoMilestoneRegistration | None = None
    live_exact_prefix_rung_registration: V1ExactPrefixRungRegistration | None = None
    pending_pareto_milestone: V1ParetoMilestonePending | None = None
    pareto_milestone_receipt: V1ParetoMilestoneReceipt | None = None
    fqr_no_candidate_horizon_witness: V1ControllerProposalHorizonWitness | None = None
    label_native_runtime_plan_checkpoint_payload: Mapping[str, object] | None = None
    label_native_runtime_plan_hash: str | None = None
    label_native_runtime_plan_hashes: tuple[str, ...] = ()
    label_native_rejected_plan_indices: tuple[int, ...] = ()
    label_native_selected_acquisition_receipt_hash: str | None = None
    schema: str = _CHECKPOINT_CONTROLLER_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_run_id",
            _nonempty(self.controller_run_id, "controller_run_id"),
        )
        for name in ("controller_schema", "controller_state_hash", "controller_snapshot_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "queue_cursor", _nonnegative(self.queue_cursor, "queue_cursor"))
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative(self.logical_sequence, "logical_sequence"),
        )
        if self.phase not in {"needs_acquisition", "queue_ready", "candidate_active", "stopped"}:
            raise ValueError("checkpoint controller projection has an invalid phase")
        requires_acquisition = self.phase in {"queue_ready", "candidate_active"}
        if (self.acquisition_sidecar is not None) is not requires_acquisition:
            raise ValueError("checkpoint controller phase and acquisition sidecar disagree")
        registration = getattr(self, "pareto_milestone_registration", None)
        rung = getattr(self, "live_exact_prefix_rung_registration", None)
        pending = getattr(self, "pending_pareto_milestone", None)
        receipt = getattr(self, "pareto_milestone_receipt", None)
        if registration is None:
            if any(value is not None for value in (rung, pending, receipt)):
                raise ValueError(
                    "checkpoint carries Pareto milestone state without registration"
                )
        elif type(registration) is not V1ParetoMilestoneRegistration:
            raise TypeError("checkpoint Pareto milestone registration has the wrong type")
        else:
            expected_types = (
                (rung, V1ExactPrefixRungRegistration, "exact-prefix rung"),
                (pending, V1ParetoMilestonePending, "pending Pareto milestone"),
                (receipt, V1ParetoMilestoneReceipt, "Pareto milestone receipt"),
            )
            for value, expected, label in expected_types:
                if value is not None and type(value) is not expected:
                    raise TypeError(f"checkpoint {label} has the wrong type")
        horizon = getattr(self, "fqr_no_candidate_horizon_witness", None)
        if horizon is not None and type(
            horizon
        ) is not V1ControllerProposalHorizonWitness:
            raise TypeError("checkpoint F/Q/R horizon witness has the wrong type")
        if (horizon is not None) != (
            self.stop_reason == FQR_NO_CANDIDATE_HORIZON_STOP_REASON
        ):
            raise ValueError("checkpoint F/Q/R horizon reason and witness disagree")
        plan_hash = getattr(self, "label_native_runtime_plan_hash", None)
        if plan_hash is not None:
            plan_hash = _nonempty(
                plan_hash,
                "label_native_runtime_plan_hash",
            )
        plan_hashes = tuple(
            _nonempty(value, "label_native_runtime_plan_hash")
            for value in getattr(
                self,
                "label_native_runtime_plan_hashes",
                (),
            )
        )
        if len(set(plan_hashes)) != len(plan_hashes):
            raise ValueError(
                "checkpoint label-native runtime plan history must be unique"
            )
        rejected_plan_indices = tuple(
            _nonnegative(value, "label_native_rejected_plan_index")
            for value in getattr(
                self,
                "label_native_rejected_plan_indices",
                (),
            )
        )
        selected_receipt_hash = getattr(
            self,
            "label_native_selected_acquisition_receipt_hash",
            None,
        )
        if selected_receipt_hash is not None:
            selected_receipt_hash = _nonempty(
                selected_receipt_hash,
                "label_native_selected_acquisition_receipt_hash",
            )
        plan_payload = getattr(
            self,
            "label_native_runtime_plan_checkpoint_payload",
            None,
        )
        if plan_payload is not None:
            restored_plan = (
                restore_v1_label_native_runtime_plan_checkpoint_payload(
                    plan_payload
                )
            )
            if plan_hash is None or restored_plan.plan_hash != plan_hash:
                raise ValueError(
                    "checkpoint label-native plan payload differs from its hash"
                )
        elif plan_hash is not None:
            raise ValueError(
                "checkpoint live label-native plan lacks its compact payload"
            )
        has_label_native = bool(
            plan_hash is not None
            or plan_hashes
            or rejected_plan_indices
            or selected_receipt_hash is not None
        )
        if has_label_native:
            if not plan_hashes:
                raise ValueError(
                    "checkpoint label-native state lacks immutable plan history"
                )
            if plan_hash is None:
                if rejected_plan_indices or selected_receipt_hash is not None:
                    raise ValueError(
                        "checkpoint retired label-native plan retains live coordinates"
                    )
            else:
                if (
                    plan_hashes[-1] != plan_hash
                    or rejected_plan_indices != tuple(range(self.queue_cursor))
                ):
                    raise ValueError(
                        "checkpoint label-native cursor/rejections differ from its plan"
                    )
                if self.phase in {"queue_ready", "candidate_active"}:
                    sidecar = self.acquisition_sidecar
                    if (
                        selected_receipt_hash is None
                        or sidecar is None
                        or selected_receipt_hash
                        != sidecar.acquisition_receipt_hash
                    ):
                        raise ValueError(
                            "checkpoint label-native queue lacks its selected endpoint"
                        )
                elif selected_receipt_hash is not None:
                    raise ValueError(
                        "checkpoint label-native nonqueue phase retained an endpoint"
                    )
            if selected_receipt_hash is not None and (
                selected_receipt_hash not in self.acquisition_receipt_hashes
            ):
                raise ValueError(
                    "checkpoint label-native endpoint is absent from acquisition history"
                )
        object.__setattr__(
            self,
            "label_native_runtime_plan_checkpoint_payload",
            plan_payload,
        )
        object.__setattr__(self, "label_native_runtime_plan_hash", plan_hash)
        object.__setattr__(
            self,
            "label_native_runtime_plan_hashes",
            plan_hashes,
        )
        object.__setattr__(
            self,
            "label_native_rejected_plan_indices",
            rejected_plan_indices,
        )
        object.__setattr__(
            self,
            "label_native_selected_acquisition_receipt_hash",
            selected_receipt_hash,
        )
        if self.schema != _CHECKPOINT_CONTROLLER_SCHEMA:
            raise ValueError("unsupported checkpoint controller-projection schema")


@dataclass(frozen=True, slots=True)
class _V1RuntimeCheckpointJournal:
    """Small frequently rewritten projection of one exact runtime state."""

    config: V1RuntimeExecutionConfig
    controller: _V1ControllerCheckpointProjection
    entropy_tape: V1StagedAlphaEntropyTape
    pending_look: V1RuntimePendingLook | None
    terminal_freeze: V1TerminalFreezeReceipt | None
    terminal_validation: V1TerminalValidationReceipt | None
    completion: V1RunCompletionReceipt | None
    performance: V1PerformanceReceipt | None
    runtime_schema: str
    runtime_state_hash: str
    proposal_state_sidecar: V1RuntimeProposalStateSidecarReceipt | None = None
    staged_proposal_backend_identity_hash: str | None = None
    completed_proposal_closures: tuple[
        V1RuntimeCompletedProposalClosureReceipt, ...
    ] = ()
    pending_paired_bce_chunk: V1RuntimePendingPairedBCEChunk | None = None
    terminal_validation_family_freeze: (
        V1TerminalValidationFamilyFreezeReceipt | None
    ) = None
    terminal_validation_family_request: (
        V1TerminalValidationFamilyRequestReceipt | None
    ) = None
    terminal_validation_family_observation: (
        V1TerminalValidationFamilyObservationReceipt | None
    ) = None
    terminal_validation_family: V1TerminalValidationFamilyReceipt | None = None
    terminal_comparison_family_freeze: (
        V1TerminalComparisonFamilyFreezeReceiptV2 | None
    ) = None
    terminal_comparison_family_request: (
        V1TerminalComparisonFamilyRequestReceiptV2 | None
    ) = None
    terminal_comparison_family_observation: (
        V1TerminalComparisonFamilyObservationReceiptV2 | None
    ) = None
    terminal_comparison_family: V1TerminalComparisonFamilyReceiptV2 | None = None
    schema: str = _CHECKPOINT_JOURNAL_SCHEMA

    def __getstate__(self) -> list[Any]:
        """Emit historical arities, adding paired-BCE state only when present."""

        common = [
            self.config,
            self.controller,
            self.entropy_tape,
            self.pending_look,
            self.terminal_freeze,
            self.terminal_validation,
            self.completion,
            self.performance,
            self.runtime_schema,
            self.runtime_state_hash,
        ]
        family_coordinates = (
            self.terminal_validation_family_freeze,
            self.terminal_validation_family_request,
            self.terminal_validation_family_observation,
            self.terminal_validation_family,
        )
        comparison_coordinates = (
            self.terminal_comparison_family_freeze,
            self.terminal_comparison_family_request,
            self.terminal_comparison_family_observation,
            self.terminal_comparison_family,
        )
        if any(item is not None for item in comparison_coordinates):
            return [
                *common,
                self.proposal_state_sidecar,
                self.staged_proposal_backend_identity_hash,
                self.completed_proposal_closures,
                self.pending_paired_bce_chunk,
                *family_coordinates,
                *comparison_coordinates,
                self.schema,
            ]
        if any(item is not None for item in family_coordinates):
            return [
                *common,
                self.proposal_state_sidecar,
                self.staged_proposal_backend_identity_hash,
                self.completed_proposal_closures,
                self.pending_paired_bce_chunk,
                *family_coordinates,
                self.schema,
            ]
        if (
            self.proposal_state_sidecar is None
            and not self.completed_proposal_closures
            and self.pending_paired_bce_chunk is None
        ):
            return [*common, self.schema]
        if self.pending_paired_bce_chunk is None:
            return [
                *common,
                self.proposal_state_sidecar,
                self.staged_proposal_backend_identity_hash,
                self.completed_proposal_closures,
                self.schema,
            ]
        return [
            *common,
            self.proposal_state_sidecar,
            self.staged_proposal_backend_identity_hash,
            self.completed_proposal_closures,
            self.pending_paired_bce_chunk,
            self.schema,
        ]

    def __setstate__(self, state: Sequence[Any]) -> None:
        """Read both landed journal arities without positional slot drift."""

        values = tuple(state)
        if len(values) == 11:
            common = values[:10]
            proposal_state_sidecar = None
            staged_proposal_backend_identity_hash = None
            completed_proposal_closures = ()
            schema = values[10]
        elif len(values) == 12:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = None
            completed_proposal_closures = ()
            schema = values[11]
        elif len(values) == 13:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = None
            completed_proposal_closures = values[11]
            schema = values[12]
        elif len(values) == 14:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = values[11]
            completed_proposal_closures = values[12]
            pending_paired_bce_chunk = None
            schema = values[13]
        elif len(values) == 15:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = values[11]
            completed_proposal_closures = values[12]
            pending_paired_bce_chunk = values[13]
            schema = values[14]
        elif len(values) == 19:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = values[11]
            completed_proposal_closures = values[12]
            pending_paired_bce_chunk = values[13]
            terminal_validation_family_freeze = values[14]
            terminal_validation_family_request = values[15]
            terminal_validation_family_observation = values[16]
            terminal_validation_family = values[17]
            schema = values[18]
        elif len(values) == 23:
            common = values[:10]
            proposal_state_sidecar = values[10]
            staged_proposal_backend_identity_hash = values[11]
            completed_proposal_closures = values[12]
            pending_paired_bce_chunk = values[13]
            terminal_validation_family_freeze = values[14]
            terminal_validation_family_request = values[15]
            terminal_validation_family_observation = values[16]
            terminal_validation_family = values[17]
            terminal_comparison_family_freeze = values[18]
            terminal_comparison_family_request = values[19]
            terminal_comparison_family_observation = values[20]
            terminal_comparison_family = values[21]
            schema = values[22]
        else:
            raise ValueError("unsupported modern-v1 checkpoint-journal arity")
        if len(values) in {11, 12, 13}:
            pending_paired_bce_chunk = None
        if len(values) not in {19, 23}:
            terminal_validation_family_freeze = None
            terminal_validation_family_request = None
            terminal_validation_family_observation = None
            terminal_validation_family = None
        if len(values) != 23:
            terminal_comparison_family_freeze = None
            terminal_comparison_family_request = None
            terminal_comparison_family_observation = None
            terminal_comparison_family = None
        names = (
            "config",
            "controller",
            "entropy_tape",
            "pending_look",
            "terminal_freeze",
            "terminal_validation",
            "completion",
            "performance",
            "runtime_schema",
            "runtime_state_hash",
        )
        for name, value in zip(names, common, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "proposal_state_sidecar",
            proposal_state_sidecar,
        )
        object.__setattr__(
            self,
            "staged_proposal_backend_identity_hash",
            staged_proposal_backend_identity_hash,
        )
        object.__setattr__(
            self,
            "completed_proposal_closures",
            completed_proposal_closures,
        )
        object.__setattr__(
            self,
            "pending_paired_bce_chunk",
            pending_paired_bce_chunk,
        )
        object.__setattr__(
            self,
            "terminal_validation_family_freeze",
            terminal_validation_family_freeze,
        )
        object.__setattr__(
            self,
            "terminal_validation_family_request",
            terminal_validation_family_request,
        )
        object.__setattr__(
            self,
            "terminal_validation_family_observation",
            terminal_validation_family_observation,
        )
        object.__setattr__(
            self,
            "terminal_validation_family",
            terminal_validation_family,
        )
        object.__setattr__(
            self,
            "terminal_comparison_family_freeze",
            terminal_comparison_family_freeze,
        )
        object.__setattr__(
            self,
            "terminal_comparison_family_request",
            terminal_comparison_family_request,
        )
        object.__setattr__(
            self,
            "terminal_comparison_family_observation",
            terminal_comparison_family_observation,
        )
        object.__setattr__(
            self,
            "terminal_comparison_family",
            terminal_comparison_family,
        )
        object.__setattr__(self, "schema", schema)

    def __post_init__(self) -> None:
        if type(self.config) is not V1RuntimeExecutionConfig:
            raise TypeError("checkpoint journal requires an exact runtime config")
        if type(self.controller) is not _V1ControllerCheckpointProjection:
            raise TypeError("checkpoint journal requires an exact controller projection")
        if type(self.entropy_tape) is not V1StagedAlphaEntropyTape:
            raise TypeError("checkpoint journal requires an exact entropy tape")
        object.__setattr__(self, "runtime_schema", _nonempty(self.runtime_schema, "runtime_schema"))
        object.__setattr__(
            self,
            "runtime_state_hash",
            _nonempty(self.runtime_state_hash, "runtime_state_hash"),
        )
        completed = tuple(self.completed_proposal_closures)
        if any(
            type(item) is not V1RuntimeCompletedProposalClosureReceipt
            for item in completed
        ):
            raise TypeError("checkpoint journal closure ledger contains a nonreceipt")
        for closure in completed:
            closure.__post_init__()
        object.__setattr__(self, "completed_proposal_closures", completed)
        if self.proposal_state_sidecar is not None:
            if type(
                self.proposal_state_sidecar
            ) is not V1RuntimeProposalStateSidecarReceipt:
                raise TypeError("checkpoint journal proposal-state receipt is not exact")
        if self.pending_paired_bce_chunk is not None and type(
            self.pending_paired_bce_chunk
        ) is not V1RuntimePendingPairedBCEChunk:
            raise TypeError("checkpoint journal pending paired-BCE chunk is not exact")
        for value, expected, label in (
            (
                self.terminal_validation_family_freeze,
                V1TerminalValidationFamilyFreezeReceipt,
                "family freeze",
            ),
            (
                self.terminal_validation_family_request,
                V1TerminalValidationFamilyRequestReceipt,
                "family request",
            ),
            (
                self.terminal_validation_family_observation,
                V1TerminalValidationFamilyObservationReceipt,
                "family observation",
            ),
            (
                self.terminal_validation_family,
                V1TerminalValidationFamilyReceipt,
                "family receipt",
            ),
            (
                self.terminal_comparison_family_freeze,
                V1TerminalComparisonFamilyFreezeReceiptV2,
                "comparison family freeze v2",
            ),
            (
                self.terminal_comparison_family_request,
                V1TerminalComparisonFamilyRequestReceiptV2,
                "comparison family request v2",
            ),
            (
                self.terminal_comparison_family_observation,
                V1TerminalComparisonFamilyObservationReceiptV2,
                "comparison family observation v2",
            ),
            (
                self.terminal_comparison_family,
                V1TerminalComparisonFamilyReceiptV2,
                "comparison family receipt v2",
            ),
        ):
            if value is not None and type(value) is not expected:
                raise TypeError(f"checkpoint journal {label} is not exact")
        family_coordinates = (
            self.terminal_validation_family_freeze,
            self.terminal_validation_family_request,
            self.terminal_validation_family_observation,
            self.terminal_validation_family,
        )
        family_aware = any(item is not None for item in family_coordinates)
        comparison_coordinates = (
            self.terminal_comparison_family_freeze,
            self.terminal_comparison_family_request,
            self.terminal_comparison_family_observation,
            self.terminal_comparison_family,
        )
        for item in comparison_coordinates:
            if item is not None:
                item.__post_init__()
        comparison_aware = any(
            item is not None for item in comparison_coordinates
        )
        if family_aware and comparison_aware:
            raise ValueError(
                "checkpoint journal mixes terminal family v1 and comparison v2"
            )
        if self.schema == _CHECKPOINT_JOURNAL_SCHEMA:
            if (
                self.proposal_state_sidecar is not None
                or self.staged_proposal_backend_identity_hash is not None
                or completed
                or self.pending_paired_bce_chunk is not None
            ):
                raise ValueError("historical journal contains staged proposal authority")
        elif self.schema == _CHECKPOINT_JOURNAL_V2_SCHEMA:
            if (
                self.proposal_state_sidecar is None
                or self.staged_proposal_backend_identity_hash is not None
                or completed
                or self.pending_paired_bce_chunk is not None
            ):
                raise ValueError("checkpoint journal-v2 has invalid staged state")
        elif self.schema == _CHECKPOINT_JOURNAL_V3_SCHEMA:
            raise ValueError("decision-only checkpoint journal-v3 is inspection-only")
        elif self.schema == _CHECKPOINT_JOURNAL_V4_SCHEMA:
            if self.pending_paired_bce_chunk is not None:
                raise ValueError("checkpoint journal-v4 cannot contain paired-BCE state")
            if self.proposal_state_sidecar is None and not completed:
                raise ValueError("checkpoint journal-v4 lacks staged state")
            backend = _nonempty(
                self.staged_proposal_backend_identity_hash,
                "staged_proposal_backend_identity_hash",
            )
            if (
                self.proposal_state_sidecar is not None
                and self.proposal_state_sidecar.proposal_backend_identity_hash
                != backend
            ) or any(
                item.proposal_backend_identity_hash != backend
                for item in completed
            ):
                raise ValueError("checkpoint journal-v4 changed its backend anchor")
        elif self.schema == _CHECKPOINT_JOURNAL_V5_SCHEMA:
            if self.pending_paired_bce_chunk is None:
                raise ValueError("checkpoint journal-v5 lacks paired-BCE state")
            if self.proposal_state_sidecar is not None or completed:
                backend = _nonempty(
                    self.staged_proposal_backend_identity_hash,
                    "staged_proposal_backend_identity_hash",
                )
                if (
                    self.proposal_state_sidecar is not None
                    and self.proposal_state_sidecar.proposal_backend_identity_hash
                    != backend
                ) or any(
                    item.proposal_backend_identity_hash != backend
                    for item in completed
                ):
                    raise ValueError("checkpoint journal-v5 changed its backend anchor")
            elif self.staged_proposal_backend_identity_hash is not None:
                raise ValueError(
                    "checkpoint journal-v5 has a staged backend without staged state"
                )
        elif self.schema == _CHECKPOINT_JOURNAL_V6_SCHEMA:
            if not family_aware:
                raise ValueError("checkpoint journal-v6 lacks terminal-family state")
            seen_gap = False
            for item in family_coordinates:
                if item is None:
                    seen_gap = True
                elif seen_gap:
                    raise ValueError(
                        "checkpoint journal-v6 skips a terminal-family causal joint"
                    )
            if self.proposal_state_sidecar is not None or completed:
                backend = _nonempty(
                    self.staged_proposal_backend_identity_hash,
                    "staged_proposal_backend_identity_hash",
                )
                if (
                    self.proposal_state_sidecar is not None
                    and self.proposal_state_sidecar.proposal_backend_identity_hash
                    != backend
                ) or any(
                    item.proposal_backend_identity_hash != backend
                    for item in completed
                ):
                    raise ValueError("checkpoint journal-v6 changed its backend anchor")
            elif self.staged_proposal_backend_identity_hash is not None:
                raise ValueError(
                    "checkpoint journal-v6 has a staged backend without staged state"
                )
        elif self.schema == _CHECKPOINT_JOURNAL_V7_SCHEMA:
            if not comparison_aware or family_aware:
                raise ValueError(
                    "checkpoint journal-v7 lacks an exclusive comparison-family chain"
                )
            seen_gap = False
            for item in comparison_coordinates:
                if item is None:
                    seen_gap = True
                elif seen_gap:
                    raise ValueError(
                        "checkpoint journal-v7 skips a comparison-family causal joint"
                    )
            if self.proposal_state_sidecar is not None or completed:
                backend = _nonempty(
                    self.staged_proposal_backend_identity_hash,
                    "staged_proposal_backend_identity_hash",
                )
                if (
                    self.proposal_state_sidecar is not None
                    and self.proposal_state_sidecar.proposal_backend_identity_hash
                    != backend
                ) or any(
                    item.proposal_backend_identity_hash != backend
                    for item in completed
                ):
                    raise ValueError("checkpoint journal-v7 changed its backend anchor")
            elif self.staged_proposal_backend_identity_hash is not None:
                raise ValueError(
                    "checkpoint journal-v7 has a staged backend without staged state"
                )
        else:
            raise ValueError("unsupported modern-v1 checkpoint-journal schema")
        if self.schema != _CHECKPOINT_JOURNAL_V6_SCHEMA and family_aware:
            raise ValueError("historical journal contains terminal-family authority")
        if self.schema != _CHECKPOINT_JOURNAL_V7_SCHEMA and comparison_aware:
            raise ValueError("historical journal contains comparison-family authority")


class V1ProposalForecastFactory(Protocol):
    """Predictable, outcome-independent proposal forecast hook."""

    def __call__(
        self,
        progress: V1RuntimeProposalAcquisitionProgress,
    ) -> Sequence[V1NextActionAcquisitionForecast]: ...


class V1ProposalRequestPlanFactory(Protocol):
    """Select exact train draws only after the pure scheduler has decided."""

    def __call__(
        self,
        progress: V1RuntimeProposalAcquisitionProgress,
        selected_forecast: V1NextActionAcquisitionForecast,
        scheduler_receipt: V1JointAcquisitionScheduleReceipt,
    ) -> V1RuntimeProposalRequestPlan: ...


class V1ProposalAcquisitionExecutor(Protocol):
    """Execute one already-persisted outcome-independent train request."""

    def __call__(
        self,
        request: V1RuntimeProposalAcquisitionRequest,
    ) -> V1RuntimeProposalAcquisitionOutcome: ...


class V1ProposalFreezeEvaluator(Protocol):
    """Evaluate a frozen stopping criterion over the current receipt graph."""

    def __call__(
        self,
        progress: V1RuntimeProposalAcquisitionProgress,
    ) -> V1RuntimeProposalFreezeEvaluation: ...


class V1FQRNoCandidateHorizonProofBuilder(Protocol):
    """Revalidate retained F/Q/R C state and reconstruct its compact proof."""

    def __call__(
        self,
        progress: V1RuntimeProposalAcquisitionProgress,
    ) -> V1FQRNoCandidateHorizonProofReceipt: ...


class V1ProposalPureFinalizer(Protocol):
    """Purely materialize the authorized acquisition without new measurements."""

    def __call__(
        self,
        progress: V1RuntimeProposalAcquisitionProgress,
        decision: V1ProposalFreezeDecisionReceipt,
    ) -> V1RuntimeAcquisition: ...


class V1PairedBCEChunkFactory(Protocol):
    """Build one predictable, outcome-independent paired-BCE chunk request."""

    def __call__(
        self,
        *,
        controller: V1ControllerState,
        active_candidate: V1ActiveCandidate,
        registration: V1PairedBCEAcquisitionRegistration,
        paired_bce_state: V1PairedBCEAcquisitionState,
    ) -> V1RuntimePendingPairedBCEChunk: ...


class V1PairedBCEChunkExecutor(Protocol):
    """Execute one already-persisted paired-BCE row chunk."""

    def __call__(
        self,
        pending: V1RuntimePendingPairedBCEChunk,
    ) -> V1RuntimePairedBCEChunkOutcome: ...


class V1PairedBCEBettingReplayFactory(Protocol):
    """Rehydrate append-only external row prefixes for partial decisions."""

    def __call__(
        self,
        *,
        registration: V1PairedBCEAcquisitionRegistration,
        paired_bce_state: V1PairedBCEAcquisitionState,
    ) -> V1PairedBCEBettingReplay | None: ...


class V1PairedBCERegistrationFactory(Protocol):
    """Freeze the candidate-specific paired-BCE registration pre-outcome."""

    def __call__(
        self,
        controller: V1ControllerState,
        threshold: V1CandidateDamageThreshold,
        calibration_freeze: V1CandidateCalibrationFreeze,
    ) -> V1PairedBCEAcquisitionRegistration: ...


class V1CandidateCalibrationFreezeFactory(Protocol):
    """Materialize the exact candidate calibration freeze before oracle rows."""

    def __call__(
        self,
        state: V1RuntimeState,
        threshold: V1CandidateDamageThreshold,
    ) -> V1CandidateCalibrationFreeze: ...


class V1CalibrationHorizonSkipDecisionFactory(Protocol):
    """Bind one validated causal K-horizon refusal to its alpha coordinate."""

    def __call__(
        self,
        *,
        state: V1RuntimeState,
        threshold: V1CandidateDamageThreshold,
        refusal_receipt: Mapping[str, Any],
    ) -> V1PairedBCECalibrationHorizonSkipDecision: ...


def v1_runtime_proposal_order_indeterminate(state: V1RuntimeState) -> bool:
    """Return the exact durable terminal-scheduler-refusal predicate."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal-order predicate requires an exact runtime state")
    progress = state.proposal_progress
    return progress is not None and progress.proposal_order_indeterminate


def require_v1_runtime_proposal_order_resolved(state: V1RuntimeState) -> None:
    """Raise the typed abort signal after callers have checkpointed refusal."""

    if not v1_runtime_proposal_order_indeterminate(state):
        return
    progress = state.proposal_progress
    assert progress is not None and progress.terminal_refusal_receipt is not None
    receipt = progress.terminal_refusal_receipt
    assert receipt.refusal_reason is not None
    raise V1RuntimeProposalOrderIndeterminate(
        progress_hash=progress.progress_hash,
        scheduler_receipt_hash=receipt.receipt_hash,
        refusal_reason=receipt.refusal_reason,
    )


def initialize_v1_runtime_proposal_acquisition(
    state: V1RuntimeState,
    *,
    action_regret_metric_hash: str,
    proposal_backend_identity_hash: str,
    phase_evidence_state_hash: str,
    freeze_criterion_hash: str,
    initial_sufficient_state: V1RuntimeProposalSufficientStateDescriptor,
) -> V1RuntimeState:
    """Open one runtime-owned, train-only proposal acquisition generation."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal acquisition initialization requires an exact runtime state")
    if type(initial_sufficient_state) is not V1RuntimeProposalSufficientStateDescriptor:
        raise TypeError("proposal acquisition requires an exact initial sufficient state")
    if state.controller.phase != "needs_acquisition":
        raise PermissionError("proposal acquisition is legal only while relinearizing")
    if state.proposal_progress is not None or state.pending_proposal_acquisition is not None:
        raise PermissionError("runtime already owns a proposal acquisition generation")
    if state.pending_look is not None or state.terminal_freeze is not None:
        raise PermissionError("proposal acquisition cannot cross oracle or terminal work")
    controller = state.controller
    backend_identity = _nonempty(
        proposal_backend_identity_hash,
        "proposal_backend_identity_hash",
    )
    if len(state.completed_proposal_closures) != len(
        controller.acquisition_receipt_hashes
    ):
        raise PermissionError(
            "staged proposal acquisition forbids mixed legacy one-shot provenance"
        )
    if (
        state.completed_proposal_closures
        and backend_identity
        != state.completed_proposal_closures[0].proposal_backend_identity_hash
    ):
        raise PermissionError(
            "staged proposal acquisition cannot change its sealed backend identity"
        )
    joint = V1JointAcquisitionState(
        campaign_binding_hash=controller.binding.binding_hash,
        current_mask_hash=controller.current_mask.mask_hash,
        action_regret_metric_hash=_nonempty(
            action_regret_metric_hash,
            "action_regret_metric_hash",
        ),
        phase_evidence_state_hash=_nonempty(
            phase_evidence_state_hash,
            "phase_evidence_state_hash",
        ),
        phase="proposal",
        logical_sequence=0,
    )
    lineage = controller.mask_lineage_hashes
    transitions = controller.transition_receipt_hashes
    lineage_hash = _proposal_mask_lineage_hash(lineage, transitions)
    freeze_hash = _nonempty(
        freeze_criterion_hash,
        "freeze_criterion_hash",
    )
    initialization_receipt_hash = _proposal_initialization_receipt_hash(
        campaign_contract_hash=controller.binding.contract.campaign_contract_hash,
        campaign_binding_hash=controller.binding.binding_hash,
        controller_run_id=controller.controller_run_id,
        proposal_backend_identity_hash=backend_identity,
        controller_state_hash=controller.state_hash,
        controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
        controller_logical_sequence=controller.logical_sequence,
        current_mask_hash=controller.current_mask.mask_hash,
        route_axis_hash=controller.current_mask.route_axis_hash,
        acquisition_generation_index=controller.acquisition_generation_index,
        mask_lineage_hash=lineage_hash,
        action_regret_metric_hash=joint.action_regret_metric_hash,
        freeze_criterion_hash=freeze_hash,
        initial_joint_state_hash=joint.content_hash,
        initial_sufficient_state_descriptor_hash=initial_sufficient_state.descriptor_hash,
    )
    progress = V1RuntimeProposalAcquisitionProgress(
        campaign_contract_hash=controller.binding.contract.campaign_contract_hash,
        campaign_binding_hash=controller.binding.binding_hash,
        controller_run_id=controller.controller_run_id,
        proposal_backend_identity_hash=backend_identity,
        controller_state_hash=controller.state_hash,
        controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
        controller_logical_sequence=controller.logical_sequence,
        current_mask_hash=controller.current_mask.mask_hash,
        route_axis_hash=controller.current_mask.route_axis_hash,
        acquisition_generation_index=controller.acquisition_generation_index,
        mask_lineage_hashes=lineage,
        transition_receipt_hashes=transitions,
        mask_lineage_hash=lineage_hash,
        action_regret_metric_hash=joint.action_regret_metric_hash,
        freeze_criterion_hash=freeze_hash,
        initial_phase_evidence_state_hash=joint.phase_evidence_state_hash,
        initial_joint_state_hash=joint.content_hash,
        initialization_receipt_hash=initialization_receipt_hash,
        joint_state=joint,
        initial_sufficient_state=initial_sufficient_state,
        current_sufficient_state=initial_sufficient_state,
    )
    return replace(
        state,
        proposal_progress=progress,
        staged_proposal_backend_identity_hash=backend_identity,
    )


def prepare_v1_runtime_proposal_acquisition(
    state: V1RuntimeState,
    *,
    forecast_factory: V1ProposalForecastFactory,
    request_plan_factory: V1ProposalRequestPlanFactory,
) -> tuple[V1RuntimeState, V1JointAcquisitionScheduleReceipt]:
    """Persist one exact max-utility proposal request before execution."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal preparation requires an exact runtime state")
    progress = state.proposal_progress
    if progress is None or state.controller.phase != "needs_acquisition":
        raise PermissionError("runtime has no live proposal acquisition generation")
    if state.pending_look is not None:
        raise PermissionError("proposal and paired-oracle requests are mutually exclusive")
    if state.pending_proposal_acquisition is not None:
        raise PermissionError("runtime already has a persisted proposal request")
    if progress.proposal_order_indeterminate:
        require_v1_runtime_proposal_order_resolved(state)
    if progress.authorized_freeze_decision is not None:
        raise PermissionError("proposal evidence is already frozen for pure finalization")
    if not callable(forecast_factory) or not callable(request_plan_factory):
        raise TypeError("proposal preparation requires callable typed hooks")
    raw_forecasts = forecast_factory(progress)
    if isinstance(raw_forecasts, (str, bytes)) or not isinstance(raw_forecasts, Sequence):
        raise TypeError("proposal forecast hook must return a sequence")
    forecasts = tuple(raw_forecasts)
    if any(type(item) is not V1NextActionAcquisitionForecast for item in forecasts):
        raise TypeError("proposal forecast hook returned a nonforecast value")
    forecasts = tuple(sorted(forecasts, key=lambda item: item.content_hash))
    receipt = schedule_v1_joint_acquisition(progress.joint_state, forecasts)
    if receipt.action != "acquire":
        if receipt.action != "refuse_no_legal_action":
            raise ValueError("proposal scheduler returned an unsupported action")
        # Persist the full predictable decision before exposing the terminal
        # predicate.  No request plan or executor is consulted on refusal.
        refused_progress = replace(
            progress,
            terminal_refusal_receipt=receipt,
            terminal_refusal_forecasts=forecasts,
        )
        return replace(state, proposal_progress=refused_progress), receipt
    selected = next(
        item
        for item in forecasts
        if item.content_hash == receipt.selected_forecast_hash
    )
    plan = request_plan_factory(progress, selected, receipt)
    if type(plan) is not V1RuntimeProposalRequestPlan:
        raise TypeError("proposal request-plan hook returned the wrong type")
    if plan.acquisition_units != selected.acquisition_units:
        raise ValueError("proposal request plan differs from selected acquisition units")
    new_draws = set(plan.train_draw_ids)
    if new_draws.intersection(progress.proposal_train_draw_ids):
        raise ValueError("proposal request plan reuses an earlier proposal train draw ID")
    if new_draws.intersection(state.controller.globally_used_draw_ids):
        raise ValueError("proposal request plan overlaps the paired-oracle draw ledger")
    forecast_set_hash = _proposal_forecast_set_hash(forecasts)
    request = V1RuntimeProposalAcquisitionRequest(
        campaign_binding_hash=progress.campaign_binding_hash,
        controller_run_id=progress.controller_run_id,
        controller_state_hash=progress.controller_state_hash,
        controller_snapshot_hash=progress.controller_snapshot_hash,
        current_mask_hash=progress.current_mask_hash,
        acquisition_generation_index=progress.acquisition_generation_index,
        mask_lineage_hash=progress.mask_lineage_hash,
        prior_progress_hash=progress.frontier_hash,
        prior_joint_state_hash=progress.joint_state.content_hash,
        prior_phase_evidence_state_hash=progress.joint_state.phase_evidence_state_hash,
        prior_logical_sequence=progress.joint_state.logical_sequence,
        prior_sufficient_state_hash=progress.current_sufficient_state.sufficient_state_hash,
        prior_sufficient_state_descriptor_hash=(
            progress.current_sufficient_state.descriptor_hash
        ),
        forecast_set_hash=forecast_set_hash,
        scheduler_receipt_hash=receipt.receipt_hash,
        selected_forecast_hash=selected.content_hash,
        selected_role=selected.role,
        acquisition_units=selected.acquisition_units,
        train_context_ids=plan.train_context_ids,
        train_draw_ids=plan.train_draw_ids,
        source_hash=plan.source_hash,
        request_plan_hash=plan.plan_hash,
        payload=plan.payload,
    )
    pending = V1RuntimePendingProposalAcquisition(
        prior_progress_hash=progress.frontier_hash,
        forecasts=forecasts,
        forecast_set_hash=forecast_set_hash,
        scheduler_receipt=receipt,
        request_plan=plan,
        request=request,
    )
    return replace(state, pending_proposal_acquisition=pending), receipt


def execute_v1_runtime_pending_proposal_acquisition(
    state: V1RuntimeState,
    *,
    executor: V1ProposalAcquisitionExecutor,
) -> V1RuntimeState:
    """Execute one persisted proposal request and advance its receipt graph."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal execution requires an exact runtime state")
    progress = state.proposal_progress
    pending = state.pending_proposal_acquisition
    if progress is not None and progress.proposal_order_indeterminate:
        require_v1_runtime_proposal_order_resolved(state)
    if progress is None or pending is None:
        raise PermissionError("runtime has no persisted proposal request")
    if not callable(executor):
        raise TypeError("proposal execution requires a callable typed hook")
    # Re-run every predictable authority-bearing check immediately before the
    # backend can observe the request.  A directly constructed scheduler
    # receipt therefore cannot select a nonmaximal forecast.
    _assert_proposal_progress_bound_to_controller(progress, state.controller)
    _assert_pending_proposal_bound(progress, pending)
    outcome = executor(pending.request)
    if type(outcome) is not V1RuntimeProposalAcquisitionOutcome:
        raise TypeError("proposal executor returned the wrong outcome receipt type")
    request = pending.request
    if (
        outcome.request_hash != request.request_hash
        or outcome.source_hash != request.source_hash
        or outcome.prior_sufficient_state_hash != request.prior_sufficient_state_hash
        or outcome.executed_train_context_ids != request.train_context_ids
        or outcome.executed_train_draw_ids != request.train_draw_ids
    ):
        raise ValueError("proposal outcome differs from its exact persisted request")
    selected = next(
        item
        for item in pending.forecasts
        if item.content_hash == request.selected_forecast_hash
    )
    update = record_v1_proposal_acquisition_update(
        state=progress.joint_state,
        considered_forecasts=pending.forecasts,
        forecast=selected,
        scheduler_receipt=pending.scheduler_receipt,
        acquired_train_context_ids=request.train_context_ids,
        acquired_train_draw_ids=request.train_draw_ids,
        request_hash=request.request_hash,
        source_hash=outcome.source_hash,
        outcome_hash=outcome.outcome_hash,
        prior_sufficient_state_hash=request.prior_sufficient_state_hash,
        updated_sufficient_state_hash=(
            outcome.updated_sufficient_state.sufficient_state_hash
        ),
        updated_phase_evidence_state_hash=outcome.updated_phase_evidence_state_hash,
    )
    replay_step = V1ProposalAcquisitionReplayStep(
        considered_forecasts=pending.forecasts,
        selected_forecast=selected,
        scheduler_receipt=pending.scheduler_receipt,
        update_receipt=update,
    )
    joint = advance_v1_joint_acquisition_state(
        progress.joint_state,
        update,
        considered_forecasts=pending.forecasts,
        forecast=selected,
        scheduler_receipt=pending.scheduler_receipt,
    )
    updated_progress = replace(
        progress,
        joint_state=joint,
        current_sufficient_state=outcome.updated_sufficient_state,
        update_receipts=progress.update_receipts + (update,),
        replay_steps=progress.replay_steps + (replay_step,),
        completed_pending_requests=(
            progress.completed_pending_requests + (pending,)
        ),
        outcome_receipts=progress.outcome_receipts + (outcome,),
        proposal_train_draw_ids=(
            progress.proposal_train_draw_ids + request.train_draw_ids
        ),
    )
    return replace(
        state,
        proposal_progress=updated_progress,
        pending_proposal_acquisition=None,
    )


def decide_v1_runtime_proposal_freeze(
    state: V1RuntimeState,
    *,
    evaluator: V1ProposalFreezeEvaluator,
) -> tuple[V1RuntimeState, V1ProposalFreezeDecisionReceipt]:
    """Replay all proposal evidence, then evaluate the frozen stop criterion."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal freeze requires an exact runtime state")
    progress = state.proposal_progress
    if progress is None or state.controller.phase != "needs_acquisition":
        raise PermissionError("runtime has no live proposal acquisition generation")
    if state.pending_proposal_acquisition is not None:
        raise PermissionError("proposal freeze requires no unobserved pending request")
    if progress.proposal_order_indeterminate:
        require_v1_runtime_proposal_order_resolved(state)
    if progress.authorized_freeze_decision is not None:
        raise PermissionError("proposal evidence already has freeze authorization")
    if not callable(evaluator):
        raise TypeError("proposal freeze requires a callable evidence evaluator")
    verified = progress.verified_evidence_state
    evaluation = evaluator(progress)
    if type(evaluation) is not V1RuntimeProposalFreezeEvaluation:
        raise TypeError("proposal freeze evaluator returned the wrong evaluation type")
    if (
        evaluation.freeze_criterion_hash != progress.freeze_criterion_hash
        or evaluation.verified_receipt_graph_hash
        != verified.verified_receipt_graph_hash
    ):
        raise ValueError("proposal freeze evidence has stale criterion/receipt lineage")
    evidence = issue_v1_proposal_freeze_evidence(
        state=progress.joint_state,
        verified_evidence_state=verified,
        freeze_criterion_hash=evaluation.freeze_criterion_hash,
        criterion_evaluator_identity_hash=(
            evaluation.criterion_evaluator_identity_hash
        ),
        criterion_verification_receipt_hash=(
            evaluation.criterion_verification_receipt_hash
        ),
        evidence_evaluation_hash=evaluation.evidence_evaluation_hash,
        criterion_satisfied=evaluation.criterion_satisfied,
        proposed_candidate_identity_hash=(
            evaluation.proposed_candidate_identity_hash
        ),
    )
    decision = decide_v1_proposal_freeze(
        progress.joint_state,
        evidence,
        verified_evidence_state=verified,
    )
    updated_progress = replace(
        progress,
        freeze_evaluation_receipts=(
            progress.freeze_evaluation_receipts + (evaluation,)
        ),
        freeze_evidence_receipts=progress.freeze_evidence_receipts + (evidence,),
        freeze_decision_receipts=progress.freeze_decision_receipts + (decision,),
    )
    return replace(state, proposal_progress=updated_progress), decision


def finalize_v1_runtime_fqr_no_candidate_horizon(
    state: V1RuntimeState,
    *,
    proof_builder: V1FQRNoCandidateHorizonProofBuilder,
) -> V1RuntimeState:
    """Close one stable-negative F/Q/R generation without fabricating a queue."""

    if type(state) is not V1RuntimeState:
        raise TypeError("F/Q/R no-candidate horizon requires an exact runtime state")
    progress = state.proposal_progress
    if progress is None or state.controller.phase != "needs_acquisition":
        raise PermissionError(
            "F/Q/R no-candidate horizon requires live relinearization evidence"
        )
    if state.pending_proposal_acquisition is not None:
        raise PermissionError(
            "F/Q/R no-candidate horizon cannot retain an unobserved request"
        )
    if not callable(proof_builder):
        raise TypeError("F/Q/R no-candidate horizon requires a proof builder")
    proof = proof_builder(progress)
    if type(proof) is not V1FQRNoCandidateHorizonProofReceipt:
        raise TypeError("F/Q/R horizon proof builder returned the wrong type")
    proof.__post_init__()
    if (
        proof.campaign_contract_hash != progress.campaign_contract_hash
        or proof.campaign_binding_hash != progress.campaign_binding_hash
        or proof.controller_run_id != progress.controller_run_id
        or proof.proposal_backend_identity_hash
        != progress.proposal_backend_identity_hash
        or proof.generation_index != progress.acquisition_generation_index
        or proof.current_mask_hash != progress.current_mask_hash
        or proof.route_axis_hash != progress.route_axis_hash
        or proof.mask_lineage_hash != progress.mask_lineage_hash
        or proof.proposal_sufficient_state_hash
        != progress.current_sufficient_state.sufficient_state_hash
        or proof.proposal_sufficient_state_descriptor_hash
        != progress.current_sufficient_state.descriptor_hash
    ):
        raise ValueError("F/Q/R no-candidate proof has stale runtime coordinates")
    if (
        not progress.freeze_evaluation_receipts
        or len(progress.freeze_evaluation_receipts)
        != len(progress.freeze_evidence_receipts)
        or len(progress.freeze_evaluation_receipts)
        != len(progress.freeze_decision_receipts)
    ):
        raise PermissionError(
            "F/Q/R no-candidate horizon requires its proof-bound freeze "
            "refusal at the current stable-negative frontier"
        )
    evaluation = progress.freeze_evaluation_receipts[-1]
    evidence = progress.freeze_evidence_receipts[-1]
    decision = progress.freeze_decision_receipts[-1]
    current_joint_hash = progress.joint_state.content_hash
    current_sufficient_hash = (
        progress.current_sufficient_state.sufficient_state_hash
    )
    current_verified = progress.verified_evidence_state
    expected_verification_hash = v1_fqr_no_candidate_freeze_verification_hash(
        proof=proof,
        verified_receipt_graph_hash=(
            current_verified.verified_receipt_graph_hash
        ),
        freeze_criterion_hash=progress.freeze_criterion_hash,
        sufficient_state_hash=current_sufficient_hash,
    )
    expected_evaluation_hash = v1_fqr_no_candidate_freeze_evaluation_hash(
        proof=proof,
        verification_hash=expected_verification_hash,
    )
    if (
        decision.state_hash != current_joint_hash
        or evidence.state_hash != current_joint_hash
        or decision.phase_evidence_state_hash
        != progress.joint_state.phase_evidence_state_hash
        or evidence.phase_evidence_state_hash
        != progress.joint_state.phase_evidence_state_hash
        or decision.logical_sequence != progress.joint_state.logical_sequence
        or evidence.logical_sequence != progress.joint_state.logical_sequence
        or decision.sufficient_state_hash != current_sufficient_hash
        or evidence.sufficient_state_hash != current_sufficient_hash
        or evaluation.verified_receipt_graph_hash
        != current_verified.verified_receipt_graph_hash
        or decision.verified_receipt_graph_hash
        != current_verified.verified_receipt_graph_hash
        or evidence.verified_receipt_graph_hash
        != current_verified.verified_receipt_graph_hash
        or evaluation.criterion_verification_receipt_hash
        != expected_verification_hash
        or decision.criterion_verification_receipt_hash
        != expected_verification_hash
        or evidence.criterion_verification_receipt_hash
        != expected_verification_hash
        or evaluation.evidence_evaluation_hash != expected_evaluation_hash
        or decision.evidence_evaluation_hash != expected_evaluation_hash
        or evidence.evidence_evaluation_hash != expected_evaluation_hash
        or evaluation.criterion_satisfied is not False
        or evidence.criterion_satisfied is not False
        or decision.action != "refuse_candidate_freeze"
    ):
        raise PermissionError(
            "F/Q/R no-candidate horizon requires its proof-bound freeze "
            "refusal at the current stable-negative frontier"
        )
    pre_controller = state.controller
    pre_snapshot = pre_controller.snapshot_receipt()
    witness = V1ControllerProposalHorizonWitness(
        proof=proof,
        campaign_contract_hash=progress.campaign_contract_hash,
        campaign_binding_hash=progress.campaign_binding_hash,
        controller_run_id=progress.controller_run_id,
        proposal_backend_identity_hash=progress.proposal_backend_identity_hash,
        generation_index=progress.acquisition_generation_index,
        current_mask_hash=progress.current_mask_hash,
        route_axis_hash=progress.route_axis_hash,
        mask_lineage_hash=progress.mask_lineage_hash,
        pre_stop_controller_state_hash=pre_controller.state_hash,
        pre_stop_controller_snapshot_hash=pre_snapshot.receipt_hash,
        pre_stop_logical_sequence=pre_controller.logical_sequence,
        terminal_proposal_progress_hash=progress.progress_hash,
        terminal_proposal_frontier_hash=progress.frontier_hash,
        proposal_joint_state_hash=progress.joint_state.content_hash,
        proposal_phase_evidence_state_hash=(
            progress.joint_state.phase_evidence_state_hash
        ),
        proposal_sufficient_state_hash=(
            progress.current_sufficient_state.sufficient_state_hash
        ),
        proposal_sufficient_state_descriptor_hash=(
            progress.current_sufficient_state.descriptor_hash
        ),
        verified_receipt_graph_hash=(
            progress.verified_evidence_state.verified_receipt_graph_hash
        ),
        freeze_evaluation_receipt_hash=evaluation.receipt_hash,
        freeze_evidence_receipt_hash=evidence.content_hash,
        freeze_decision_receipt_hash=decision.receipt_hash,
        post_stop_logical_sequence=pre_controller.logical_sequence + 1,
    )
    _assert_fqr_horizon_witness_bound_to_progress(witness, progress)
    from .modern_v1_controller import (
        stop_v1_controller_at_fqr_no_candidate_horizon,
    )

    controller = stop_v1_controller_at_fqr_no_candidate_horizon(
        pre_controller,
        witness=witness,
    )
    return replace(
        state,
        controller=controller,
        proposal_progress=progress,
        pending_proposal_acquisition=None,
    )


def v1_runtime_proposal_candidate_identity_hash(
    acquisition: V1RuntimeAcquisition,
) -> str:
    """Identity of the exact deterministic candidate queue authorized to attach."""

    if type(acquisition) not in _RUNTIME_ACQUISITION_TYPES:
        raise TypeError("proposal candidate identity requires a supported acquisition")
    return _proposal_candidate_identity_hash(
        campaign_binding_hash=acquisition.binding.binding_hash,
        generation_index=acquisition.generation_index,
        current_mask_hash=acquisition.current_mask.mask_hash,
        committed_mask_lineage_hashes=acquisition.committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            acquisition.committed_mask_transition_receipt_hashes
        ),
        acquisition_hash=acquisition.acquisition_hash,
        acquisition_receipt_hash=acquisition.receipt.receipt_hash,
        ordered_candidate_ids=acquisition.receipt.ordered_candidate_ids,
    )


def _attach_v1_runtime_acquisition(
    controller: V1ControllerState,
    acquisition: V1RuntimeAcquisition,
) -> V1ControllerState:
    """Attach one sealed v1 order, preregistering only its exact scalar rung."""

    registration = controller.pareto_milestone_registration
    exact_rung: V1ExactPrefixRungRegistration | None = None
    if (
        registration is not None
        and controller.pareto_milestone_receipt is None
        and controller.current_mask.retained_count
        > registration.comparator_retained_route_count
    ):
        comparator_count = registration.comparator_retained_route_count
        first_crossing = next(
            (
                index
                for index, proposal in enumerate(acquisition.queue)
                if proposal.resulting_mask.retained_count <= comparator_count
            ),
            None,
        )
        if (
            first_crossing is not None
            and acquisition.queue[first_crossing].resulting_mask.retained_count
            == comparator_count
        ):
            # The acquisition receipt binds the complete v1-generated order
            # and its source lineage.  The comparator supplies cardinality
            # only; no historical mask/order/trajectory is consulted.
            exact_rung = register_v1_exact_pareto_prefix_rung(
                controller,
                acquisition,
                queue_index=first_crossing,
                v1_order_receipt_hash=acquisition.receipt.receipt_hash,
            )
    return attach_v1_acquisition(
        controller,
        acquisition,
        exact_prefix_rung_registration=exact_rung,
    )


def finalize_and_attach_v1_runtime_proposal_acquisition(
    state: V1RuntimeState,
    *,
    finalizer: V1ProposalPureFinalizer,
) -> V1RuntimeState:
    """Purely materialize and attach only the replay-authorized acquisition."""

    if type(state) is not V1RuntimeState:
        raise TypeError("proposal finalization requires an exact runtime state")
    progress = state.proposal_progress
    if progress is None or state.controller.phase != "needs_acquisition":
        raise PermissionError("runtime has no live proposal acquisition generation")
    if state.pending_proposal_acquisition is not None:
        raise PermissionError("proposal finalization requires no pending train outcome")
    if progress.proposal_order_indeterminate:
        require_v1_runtime_proposal_order_resolved(state)
    decision = progress.authorized_freeze_decision
    if decision is None:
        raise PermissionError("proposal finalization requires evidence-bound authorization")
    verified = progress.verified_evidence_state
    if decision.verified_receipt_graph_hash != verified.verified_receipt_graph_hash:
        raise ValueError("proposal finalization authorization has stale receipt lineage")
    if not callable(finalizer):
        raise TypeError("proposal finalization requires a callable pure finalizer")
    acquisition = finalizer(progress, decision)
    if type(acquisition) not in _RUNTIME_ACQUISITION_TYPES:
        raise TypeError("proposal finalizer returned the wrong acquisition type")
    if (
        decision.candidate_identity_hash
        != v1_runtime_proposal_candidate_identity_hash(acquisition)
    ):
        raise ValueError("finalized candidate queue differs from freeze authorization")
    pre_attach_controller = state.controller
    pre_attach_snapshot = pre_attach_controller.snapshot_receipt()
    controller = _attach_v1_runtime_acquisition(
        pre_attach_controller,
        acquisition,
    )
    post_attach_snapshot = controller.snapshot_receipt()
    assert decision.candidate_identity_hash is not None
    initial_storage = progress.initial_sufficient_state.storage_descriptor
    raw_order_registration = initial_storage.get(
        "proposal_order_inference_registration"
    )
    proposal_order_coordinates: dict[str, Any] = {}
    closure_schema = _COMPLETED_PROPOSAL_CLOSURE_SCHEMA
    if raw_order_registration is not None:
        if not isinstance(raw_order_registration, Mapping):
            raise TypeError(
                "proposal-order inference registration must be one mapping"
            )
        registration = dict(raw_order_registration)
        required_registration_fields = {
            "proposal_order_alpha_schedule_hash",
            "proposal_order_alpha_ledger_before_hash",
            "proposal_order_design_hash",
            "proposal_order_base_serfling_design_hash",
            "proposal_order_alpha_ledger_after_hash",
            "proposal_order_alpha_ledger_after_replay",
            "proposal_order_envelope_certificate_hash",
            "proposal_order_action_family_registry_hash",
        }
        if not required_registration_fields.issubset(registration):
            raise ValueError(
                "proposal-order inference registration lacks closure coordinates"
            )
        proposal_order_coordinates = {
            name: registration[name] for name in required_registration_fields
        }
        closure_schema = _COMPLETED_PROPOSAL_CLOSURE_V3_SCHEMA
        alpha_family_registration_fields = {
            "sampled_v7_D_alpha_family_plan_hash",
            "sampled_v7_D_alpha_family_registry_receipt_hash",
            "sampled_v7_D_alpha_family_registry_replay",
            "sampled_v7_D_alpha_head_CAS_receipt_hash",
            "sampled_v7_D_alpha_published_ledger_head_hash",
            "sampled_v7_D_alpha_durable_controller_head_closed",
        }
        alpha_family_values = tuple(
            registration.get(name) for name in alpha_family_registration_fields
        )
        if any(value is not None for value in alpha_family_values):
            if not all(value is not None for value in alpha_family_values):
                raise ValueError(
                    "proposal-order registration has a partial A/F/J alpha family"
                )
            proposal_order_coordinates.update(
                {
                    name: registration[name]
                    for name in alpha_family_registration_fields
                }
            )
            closure_schema = _COMPLETED_PROPOSAL_CLOSURE_V4_SCHEMA
    closure = V1RuntimeCompletedProposalClosureReceipt(
        campaign_contract_hash=progress.campaign_contract_hash,
        campaign_binding_hash=progress.campaign_binding_hash,
        controller_run_id=progress.controller_run_id,
        proposal_backend_identity_hash=progress.proposal_backend_identity_hash,
        acquisition_generation_index=progress.acquisition_generation_index,
        current_mask_hash=progress.current_mask_hash,
        route_axis_hash=progress.route_axis_hash,
        mask_lineage_hash=progress.mask_lineage_hash,
        pre_attach_controller_state_hash=pre_attach_controller.state_hash,
        pre_attach_controller_snapshot=pre_attach_snapshot,
        proposal_progress_hash=progress.progress_hash,
        proposal_frontier_hash=progress.frontier_hash,
        proposal_joint_state_hash=progress.joint_state.content_hash,
        proposal_phase_evidence_state_hash=(
            progress.joint_state.phase_evidence_state_hash
        ),
        proposal_sufficient_state_hash=(
            progress.current_sufficient_state.sufficient_state_hash
        ),
        proposal_logical_sequence=progress.joint_state.logical_sequence,
        verified_receipt_graph_hash=verified.verified_receipt_graph_hash,
        freeze_criterion_hash=decision.freeze_criterion_hash,
        freeze_evidence_hash=decision.evidence_hash,
        freeze_evidence_evaluation_hash=decision.evidence_evaluation_hash,
        criterion_verification_receipt_hash=(
            decision.criterion_verification_receipt_hash
        ),
        freeze_decision=decision,
        acquisition_hash=acquisition.acquisition_hash,
        acquisition_receipt_hash=acquisition.receipt.receipt_hash,
        ordered_candidate_ids=acquisition.receipt.ordered_candidate_ids,
        candidate_identity_hash=decision.candidate_identity_hash,
        post_attach_controller_state_hash=controller.state_hash,
        post_attach_controller_snapshot=post_attach_snapshot,
        **proposal_order_coordinates,
        schema=closure_schema,
    )
    if len(controller.acquisition_receipt_hashes) != (
        len(state.completed_proposal_closures) + 1
    ):
        raise ValueError("finalized staged acquisition has mixed provenance")
    return replace(
        state,
        controller=controller,
        proposal_progress=None,
        pending_proposal_acquisition=None,
        completed_proposal_closures=(
            state.completed_proposal_closures + (closure,)
        ),
    )


def initialize_v1_runtime(
    *,
    config: V1RuntimeExecutionConfig,
    controller: V1ControllerState,
    entropy_tape: V1StagedAlphaEntropyTape | None = None,
) -> V1RuntimeState:
    if type(config) is not V1RuntimeExecutionConfig or type(controller) is not V1ControllerState:
        raise TypeError("runtime initialization requires exact config/controller objects")
    contract = controller.binding.contract
    tape = entropy_tape
    if tape is None:
        tape = issue_v1_staged_alpha_entropy_tape(
            campaign_contract_hash=contract.campaign_contract_hash,
            controller_run_id=controller.controller_run_id,
            executor_identity_hash=config.executor_identity_hash,
            maximum_candidate_count=contract.maximum_candidate_count,
            draws_per_candidate=contract.cumulative_draw_counts[-1],
        )
    return V1RuntimeState(config=config, controller=controller, entropy_tape=tape)


def prepare_v1_runtime_look(state: V1RuntimeState) -> V1RuntimeState:
    """Draw and persist the exact system-entropy sampling plan pre-outcome."""

    if type(state) is not V1RuntimeState:
        raise TypeError("runtime look preparation requires an exact state")
    if state.pending_look is not None:
        return state
    controller = state.controller
    active = controller.active_candidate
    if controller.phase != "candidate_active" or active is None:
        raise PermissionError("runtime has no active candidate to sample")
    look_index = len(active.ledger.packets)
    counts = active.step_freeze.cumulative_draw_counts
    if look_index >= len(counts):
        raise ValueError("active candidate already exhausted its registered look schedule")
    previous = 0 if look_index == 0 else counts[look_index - 1]
    count = counts[look_index] - previous
    draw_ids = _draw_ids(controller, look_index=look_index, count=count)
    if set(draw_ids).intersection(controller.globally_used_draw_ids):
        raise ValueError("runtime logical draw counter would reuse a prior draw ID")
    plan = draw_v1_staged_alpha_sampling_plan_from_tape(
        step_freeze=active.step_freeze,
        entropy_tape=state.entropy_tape,
        candidate_attempt_index=controller.candidate_attempt_count,
        first_draw_index=previous,
        population_context_ids=active.sampling_law.population_context_ids,
        sampling_masses_u64=active.sampling_law.sampling_masses_u64,
        draw_ids=draw_ids,
        executor_identity_hash=state.config.executor_identity_hash,
    )
    request, evidence = _pending_request_and_evidence(
        config=state.config,
        controller=controller,
        active=active,
        look_index=look_index,
        plan=plan,
    )
    pending = V1RuntimePendingLook(
        controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
        active_candidate_hash=active.active_hash,
        look_index=look_index,
        sampling_plan=plan,
        request=request,
        evidence_packet=evidence,
    )
    return replace(state, pending_look=pending)


def execute_v1_runtime_pending_look(
    state: V1RuntimeState,
    *,
    provider: PairedMaskOracleProvider,
) -> tuple[V1RuntimeState, V1StagedAlphaDecision]:
    """Execute exactly one persisted draw plan and apply its decision."""

    if type(state) is not V1RuntimeState or type(provider) is not PairedMaskOracleProvider:
        raise TypeError("runtime look execution requires exact state and paired-mask provider")
    pending = state.pending_look
    active = state.controller.active_candidate
    if pending is None or active is None:
        raise PermissionError("runtime has no persisted pending look")
    if tuple(provider.context_ids) != active.sampling_law.population_context_ids:
        raise ValueError("direct provider context population differs from frozen sampling law")
    if provider.calibration_state_id != state.controller.binding.contract.calibration_identity:
        raise ValueError("direct provider calibration differs from the campaign contract")
    calibration_freeze = _verify_candidate_calibration_pre_oracle(
        state=state,
        provider=provider,
    )
    expected_calibration_freeze_hash = (
        None if calibration_freeze is None else calibration_freeze.freeze_hash
    )
    if (
        pending.request.payload.get("candidate_calibration_freeze_hash")
        != expected_calibration_freeze_hash
        or pending.evidence_packet.metadata.get(
            "candidate_calibration_freeze_hash"
        )
        != expected_calibration_freeze_hash
    ):
        raise ValueError("pending oracle lineage changed its calibration freeze")
    # The exact affine state is verified above.  Only after that causal joint
    # may the provider perform an oracle-panel forward.
    raw = provider.execute(pending.request)
    observed_evidence = replace(
        pending.evidence_packet,
        outcome_observed=True,
        metadata={
            **dict(pending.evidence_packet.metadata),
            "outcome_state": "observed",
            "outcome_integrity_hash": raw.integrity_hash,
        },
    )
    envelope = issue_v1_staged_alpha_packet(
        coordinates=state.controller.binding.coordinates,
        step_freeze=active.step_freeze,
        evidence_packet=observed_evidence,
        look_index=pending.look_index,
        sampling_plan=pending.sampling_plan,
        provider=provider,
        request=pending.request,
        raw_outcome=raw,
    )
    controller, decision = advance_v1_controller_look(state.controller, envelope)
    return replace(state, controller=controller, pending_look=None), decision


def advance_v1_runtime_paired_bce_candidate_freeze(
    state: V1RuntimeState,
    *,
    threshold_factory: ThresholdFactory,
    sampling_law_factory: SamplingLawFactory,
    calibration_freeze_factory: V1CandidateCalibrationFreezeFactory,
    paired_bce_registration_factory: V1PairedBCERegistrationFactory,
    calibration_horizon_skip_factory: (
        V1CalibrationHorizonSkipDecisionFactory | None
    ) = None,
) -> tuple[
    V1RuntimeState,
    V1PairedBCECalibrationHorizonSkipDecision | None,
]:
    """Freeze a paired-BCE candidate or retire one exact causal K refusal.

    Calibration precedes sampling-law and O-registration materialization.  A
    validated terminal ``K=N-1`` refusal can therefore consume one immutable
    candidate/alpha coordinate without opening either downstream surface.
    """

    if type(state) is not V1RuntimeState:
        raise TypeError("paired-BCE candidate freeze requires an exact runtime state")
    if state.controller.phase != "queue_ready":
        raise PermissionError("paired-BCE candidate freeze requires a ready queue")
    if (
        state.pending_look is not None
        or state.pending_paired_bce_chunk is not None
        or state.pending_proposal_acquisition is not None
        or state.terminal_freeze is not None
    ):
        raise PermissionError("paired-BCE candidate freeze cannot cross pending work")
    for label, hook in (
        ("threshold_factory", threshold_factory),
        ("calibration_freeze_factory", calibration_freeze_factory),
        ("paired_bce_registration_factory", paired_bce_registration_factory),
    ):
        if not callable(hook):
            raise TypeError(f"{label} must be callable")
    if not callable(sampling_law_factory):
        raise TypeError("sampling_law_factory must be callable")
    if calibration_horizon_skip_factory is not None and not callable(
        calibration_horizon_skip_factory
    ):
        raise TypeError("calibration_horizon_skip_factory must be callable")
    threshold = threshold_factory(state.controller)
    if type(threshold) is not V1CandidateDamageThreshold:
        raise TypeError("paired-BCE threshold factory returned the wrong type")
    try:
        calibration_freeze = calibration_freeze_factory(state, threshold)
    except CalibrationConvergenceError as error:
        if type(error) is not CalibrationConvergenceError:
            raise
        raw_refusal = getattr(error, "causal_horizon_refusal_receipt", None)
        if raw_refusal is None or calibration_horizon_skip_factory is None:
            raise
        try:
            refusal = validate_paired_bce_calibration_horizon_refusal_receipt(
                raw_refusal
            )
        except (TypeError, ValueError) as invalid:
            raise ValueError(
                "calibration failure carried an invalid causal horizon refusal"
            ) from invalid
        decision = calibration_horizon_skip_factory(
            state=state,
            threshold=threshold,
            refusal_receipt=refusal,
        )
        if type(decision) is not V1PairedBCECalibrationHorizonSkipDecision:
            raise TypeError(
                "calibration-horizon skip factory returned the wrong type"
            )
        decision.assert_integrity()
        if dict(decision.causal_horizon_refusal_receipt) != dict(refusal):
            raise ValueError(
                "calibration-horizon skip factory replaced its validated refusal"
            )
        controller, admitted = advance_v1_controller_calibration_horizon_skip(
            state.controller,
            decision,
        )
        return replace(state, controller=controller), admitted
    if type(calibration_freeze) is not V1CandidateCalibrationFreeze:
        raise TypeError("paired-BCE calibration freeze factory returned the wrong type")
    sampling_law = sampling_law_factory(state.controller)
    if type(sampling_law) is not V1StepSamplingLaw:
        raise TypeError("paired-BCE sampling-law factory returned the wrong type")
    registration = paired_bce_registration_factory(
        state.controller,
        threshold,
        calibration_freeze,
    )
    if type(registration) is not V1PairedBCEAcquisitionRegistration:
        raise TypeError("paired-BCE registration factory returned the wrong type")
    controller = freeze_v1_candidate(
        state.controller,
        threshold=threshold,
        sampling_law=sampling_law,
        candidate_calibration_freeze=calibration_freeze,
        paired_bce_registration=registration,
    )
    return replace(state, controller=controller), None


def freeze_v1_runtime_paired_bce_candidate(
    state: V1RuntimeState,
    *,
    threshold_factory: ThresholdFactory,
    sampling_law_factory: SamplingLawFactory,
    calibration_freeze_factory: V1CandidateCalibrationFreezeFactory,
    paired_bce_registration_factory: V1PairedBCERegistrationFactory,
) -> V1RuntimeState:
    """Freeze a queue candidate; solver failures remain fatal on this API."""

    frozen, decision = advance_v1_runtime_paired_bce_candidate_freeze(
        state,
        threshold_factory=threshold_factory,
        sampling_law_factory=sampling_law_factory,
        calibration_freeze_factory=calibration_freeze_factory,
        paired_bce_registration_factory=paired_bce_registration_factory,
    )
    if decision is not None:  # pragma: no cover - no skip factory was supplied.
        raise RuntimeError("legacy paired-BCE freeze unexpectedly skipped a candidate")
    return frozen


def _active_paired_bce_handles(
    state: V1RuntimeState,
) -> tuple[
    V1ActiveCandidate,
    V1PairedBCEAcquisitionRegistration,
    V1PairedBCEAcquisitionState,
]:
    if type(state) is not V1RuntimeState:
        raise TypeError("paired-BCE runtime operation requires an exact state")
    active = state.controller.active_candidate
    if state.controller.phase != "candidate_active" or active is None:
        raise PermissionError("runtime has no active candidate paired-BCE oracle")
    registration = active.paired_bce_registration
    paired_state = active.paired_bce_state
    if registration is None or paired_state is None:
        raise PermissionError("active candidate is not registered for paired-BCE oracle")
    return active, registration, paired_state


def prepare_v1_runtime_paired_bce_chunk(
    state: V1RuntimeState,
    *,
    chunk_factory: V1PairedBCEChunkFactory,
) -> V1RuntimeState:
    """Persist one exact paired-BCE chunk request before row outcomes exist."""

    if type(state) is not V1RuntimeState:
        raise TypeError("paired-BCE preparation requires an exact runtime state")
    if state.terminal_freeze is not None:
        raise PermissionError("selection runtime is closed after terminal freeze")
    if state.pending_paired_bce_chunk is not None:
        return state
    if state.pending_look is not None or state.pending_proposal_acquisition is not None:
        raise PermissionError("paired-BCE chunks cannot overlap another pending outcome")
    if not callable(chunk_factory):
        raise TypeError("paired-BCE preparation requires a callable chunk factory")
    active, registration, paired_state = _active_paired_bce_handles(state)
    current_decision = decide_paired_bce_acquisition(
        registration=registration,
        state=paired_state,
    )
    if current_decision.status != "acquire_more":
        raise PermissionError("paired-BCE oracle is already terminal")
    pending = chunk_factory(
        controller=state.controller,
        active_candidate=active,
        registration=registration,
        paired_bce_state=paired_state,
    )
    if type(pending) is not V1RuntimePendingPairedBCEChunk:
        raise TypeError("paired-BCE chunk factory returned the wrong type")
    expected = {
        "controller_snapshot_hash": state.controller.snapshot_receipt().receipt_hash,
        "active_candidate_hash": active.active_hash,
        "registration_hash": registration.registration_hash,
        "prior_paired_bce_state_hash": paired_state.state_hash,
        "dynamic_oracle_receipt_hash": registration.dynamic_oracle_receipt_hash,
    }
    if any(getattr(pending, name) != value for name, value in expected.items()):
        raise ValueError("paired-BCE pending chunk has stale active-candidate lineage")
    return replace(state, pending_paired_bce_chunk=pending)


def execute_v1_runtime_pending_paired_bce_chunk(
    state: V1RuntimeState,
    *,
    executor: V1PairedBCEChunkExecutor,
) -> V1RuntimeState:
    """Execute one persisted paired-BCE chunk and store compact state only."""

    if type(state) is not V1RuntimeState:
        raise TypeError("paired-BCE execution requires an exact runtime state")
    pending = state.pending_paired_bce_chunk
    if pending is None:
        raise PermissionError("runtime has no persisted paired-BCE chunk")
    if not callable(executor):
        raise TypeError("paired-BCE execution requires a callable executor")
    active, registration, paired_state = _active_paired_bce_handles(state)
    if (
        pending.controller_snapshot_hash != state.controller.snapshot_receipt().receipt_hash
        or pending.active_candidate_hash != active.active_hash
        or pending.registration_hash != registration.registration_hash
        or pending.prior_paired_bce_state_hash != paired_state.state_hash
        or pending.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
        or pending.request.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
    ):
        raise ValueError("pending paired-BCE chunk is stale before execution")
    outcome = executor(pending)
    if type(outcome) is not V1RuntimePairedBCEChunkOutcome:
        raise TypeError("paired-BCE executor returned the wrong outcome type")
    if (
        outcome.pending_hash != pending.pending_hash
        or outcome.request_hash != pending.request.request_hash
    ):
        raise ValueError("paired-BCE outcome names another pending request")
    expected_state = commit_paired_bce_chunk(
        registration=registration,
        state=paired_state,
        request=pending.request,
        receipt=outcome.receipt,
    )
    if outcome.committed_state.state_hash != expected_state.state_hash:
        raise ValueError("paired-BCE outcome committed state is not reproducible")
    controller = update_v1_controller_paired_bce_state(
        state.controller,
        outcome.committed_state,
    )
    return replace(
        state,
        controller=controller,
        pending_paired_bce_chunk=None,
    )


def decide_v1_runtime_paired_bce_oracle(
    state: V1RuntimeState,
    *,
    betting_replay_factory: V1PairedBCEBettingReplayFactory | None = None,
) -> tuple[V1RuntimeState, V1PairedBCEDecision]:
    """Evaluate paired-BCE state and let the controller admit terminal decisions."""

    if type(state) is not V1RuntimeState:
        raise TypeError("paired-BCE decision requires an exact runtime state")
    if state.pending_paired_bce_chunk is not None:
        raise PermissionError("paired-BCE decision requires no unobserved pending chunk")
    active, registration, paired_state = _active_paired_bce_handles(state)
    del active
    replay = None
    if betting_replay_factory is not None:
        if not callable(betting_replay_factory):
            raise TypeError("paired-BCE betting replay factory must be callable")
        replay = betting_replay_factory(
            registration=registration,
            paired_bce_state=paired_state,
        )
        if replay is not None and type(replay) is not V1PairedBCEBettingReplay:
            raise TypeError("paired-BCE betting replay factory returned the wrong type")
    decision = decide_paired_bce_acquisition(
        registration=registration,
        state=paired_state,
        betting_replay=replay,
    )
    if decision.status == "acquire_more":
        return state, decision
    controller, admitted = advance_v1_controller_paired_bce_decision(
        state.controller,
        decision,
    )
    return replace(state, controller=controller), admitted


def advance_v1_runtime_nonproposal_once(
    state: V1RuntimeState,
    *,
    threshold_factory: ThresholdFactory,
    sampling_law_factory: SamplingLawFactory,
    provider: PairedMaskOracleProvider,
) -> tuple[V1RuntimeState, V1StagedAlphaDecision | None]:
    """Advance only an already-attached candidate/oracle controller joint.

    Canonical staged acquisition callers use this surface so no unused
    one-shot acquisition callable can accidentally regain acquisition
    authority.
    """

    if type(state) is not V1RuntimeState:
        raise TypeError("nonproposal advancement requires an exact runtime state")
    if state.terminal_freeze is not None:
        raise PermissionError("selection runtime is closed after terminal freeze")
    if state.controller.phase == "needs_acquisition":
        raise PermissionError(
            "nonproposal advancement cannot materialize an acquisition"
        )
    if (
        state.proposal_progress is not None
        or state.pending_proposal_acquisition is not None
    ):
        raise PermissionError(
            "resumable proposal acquisition requires its explicit staged runtime API"
        )
    if state.pending_look is not None:
        return execute_v1_runtime_pending_look(state, provider=provider)
    if state.pending_paired_bce_chunk is not None:
        raise PermissionError(
            "paired-BCE oracle acquisition requires its explicit runtime API"
        )
    controller = state.controller
    if controller.phase == "queue_ready":
        threshold = threshold_factory(controller)
        sampling_law = sampling_law_factory(controller)
        calibration_freeze = _materialize_candidate_calibration_freeze(
            state=state,
            provider=provider,
        )
        return (
            replace(
                state,
                controller=freeze_v1_candidate(
                    controller,
                    threshold=threshold,
                    sampling_law=sampling_law,
                    candidate_calibration_freeze=calibration_freeze,
                ),
            ),
            None,
        )
    if controller.phase == "candidate_active":
        prepared = prepare_v1_runtime_look(state)
        return execute_v1_runtime_pending_look(prepared, provider=provider)
    if controller.phase == "stopped":
        return state, None
    raise RuntimeError("unreachable modern-v1 controller phase")


def advance_v1_runtime_once(
    state: V1RuntimeState,
    *,
    acquisition_factory: AcquisitionFactory,
    threshold_factory: ThresholdFactory,
    sampling_law_factory: SamplingLawFactory,
    provider: PairedMaskOracleProvider,
) -> tuple[V1RuntimeState, V1StagedAlphaDecision | None]:
    """Advance one explicit orchestration joint without implicit terminal reveal."""

    if state.terminal_freeze is not None:
        raise PermissionError("selection runtime is closed after terminal freeze")
    if (
        state.proposal_progress is not None
        or state.pending_proposal_acquisition is not None
    ):
        raise PermissionError(
            "resumable proposal acquisition requires its explicit staged runtime API"
        )
    if state.pending_look is not None:
        return execute_v1_runtime_pending_look(state, provider=provider)
    if state.pending_paired_bce_chunk is not None:
        raise PermissionError(
            "paired-BCE oracle acquisition requires its explicit runtime API"
        )
    controller = state.controller
    if controller.phase == "needs_acquisition":
        from .modern_v1_operator_factory import OperatorAcquisitionFactory
        if type(acquisition_factory) is OperatorAcquisitionFactory:
            from .modern_v1_operator_run_binding import validate_bound_operator_sources
            validate_bound_operator_sources(state.config,acquisition_factory._model.sources,factory=acquisition_factory)
        acquisition = acquisition_factory(controller)
        return replace(
            state,
            controller=_attach_v1_runtime_acquisition(controller, acquisition),
        ), None
    return advance_v1_runtime_nonproposal_once(
        state,
        threshold_factory=threshold_factory,
        sampling_law_factory=sampling_law_factory,
        provider=provider,
    )


def run_v1_runtime_to_stop(
    state: V1RuntimeState,
    *,
    acquisition_factory: AcquisitionFactory,
    threshold_factory: ThresholdFactory,
    sampling_law_factory: SamplingLawFactory,
    provider: PairedMaskOracleProvider,
    maximum_transitions: int = 100000,
) -> tuple[V1RuntimeState, tuple[V1StagedAlphaDecision, ...]]:
    """Run the complete selection loop to an explicit registered stop."""

    decisions: list[V1StagedAlphaDecision] = []
    current = state
    for _ in range(_positive(maximum_transitions, "maximum_transitions")):
        if current.controller.phase == "stopped":
            return current, tuple(decisions)
        current, decision = advance_v1_runtime_once(
            current,
            acquisition_factory=acquisition_factory,
            threshold_factory=threshold_factory,
            sampling_law_factory=sampling_law_factory,
            provider=provider,
        )
        if decision is not None:
            decisions.append(decision)
    raise RuntimeError("modern-v1 runtime exceeded its explicit orchestration bound")


def freeze_v1_terminal(
    state: V1RuntimeState,
    *,
    extraction_state_hash: str,
    extraction_calibration_hash: str,
) -> V1RuntimeState:
    """Close selection and freeze extraction/calibration before terminal rows exist."""

    if state.controller.phase != "stopped" or state.pending_look is not None:
        raise PermissionError("terminal freeze requires a stopped, nonpending controller")
    require_v1_runtime_scientific_endpoint(state)
    if state.terminal_freeze is not None:
        if state.terminal_freeze.extraction_state_hash != extraction_state_hash:
            raise ValueError("terminal extraction state is already frozen differently")
        if state.terminal_freeze.extraction_calibration_hash != extraction_calibration_hash:
            raise ValueError("terminal extraction calibration is already frozen differently")
        return state
    snapshot = state.controller.snapshot_receipt()
    receipt = V1TerminalFreezeReceipt(
        campaign_contract_hash=state.controller.binding.contract.campaign_contract_hash,
        campaign_binding_hash=state.controller.binding.binding_hash,
        controller_run_id=state.controller.controller_run_id,
        controller_snapshot_hash=snapshot.receipt_hash,
        terminal_mask_hash=state.controller.current_mask.mask_hash,
        accepted_trajectory_hash=(
            state.controller.accepted_trajectory_hash
        ),
        extraction_state_hash=_nonempty(extraction_state_hash, "extraction_state_hash"),
        extraction_calibration_hash=_nonempty(
            extraction_calibration_hash,
            "extraction_calibration_hash",
        ),
        dense_reference_hash=state.controller.binding.contract.dense_reference_hash,
        terminal_validation_population_hash=(
            state.controller.binding.contract.terminal_validation_population_hash
        ),
        acquisition_receipt_hashes=state.controller.acquisition_receipt_hashes,
        attempt_receipt_hashes=state.controller.attempt_receipt_hashes,
        transition_receipt_hashes=state.controller.transition_receipt_hashes,
        stop_reason=str(state.controller.stop_reason),
        freeze_sequence=state.controller.logical_sequence + 1,
    )
    return replace(state, terminal_freeze=receipt)


def require_v1_runtime_scientific_endpoint(
    state: V1RuntimeState,
) -> V1ScientificHorizonDisposition:
    """Require a scientific endpoint before any terminal extraction or reveal.

    Step/candidate limits, ordinary queue exhaustion, proposal-order refusal,
    and an owner-requested administrative stop remain useful checkpoints, but
    none is evidence that the loss-constrained pruning frontier was reached.
    """

    if type(state) is not V1RuntimeState:
        raise TypeError("scientific-horizon gate requires an exact runtime state")
    if state.controller.phase != "stopped":
        raise PermissionError("scientific-horizon gate requires a stopped controller")
    if state.controller.pending_pareto_milestone is not None:
        raise PermissionError(
            "pending Pareto milestone must cross its checkpoint seam before terminal work"
        )
    # Legacy controller/checkpoint readers retain their historical generic
    # interpretation of the registered floor.  A scientific Pareto campaign,
    # however, has explicitly registered a matched-density milestone and must
    # demonstrate operational saturation rather than stop at an arbitrary
    # configured density cap.  Until a typed natural-legal-floor proof exists,
    # that reason is therefore indeterminate at this terminal authority gate.
    registered_pareto_floor_without_natural_proof = (
        state.controller.stop_reason
        == REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED
        and state.controller.pareto_milestone_registration is not None
    )
    disposition = classify_v1_scientific_horizon(
        state.controller.stop_reason,
        registered_minimum_is_scientific_endpoint=(
            not registered_pareto_floor_without_natural_proof
        ),
    )
    if not disposition.scientific_endpoint:
        raise V1RuntimeScientificHorizonIndeterminate(disposition)
    return disposition


def reveal_v1_terminal_once(
    state: V1RuntimeState,
    *,
    validator: TerminalValidator,
) -> V1RuntimeState:
    """Materialize one terminal population only after the exact freeze."""

    freeze = state.terminal_freeze
    if freeze is None:
        raise PermissionError("terminal validation is invisible before terminal freeze")
    if state.terminal_validation is not None:
        raise PermissionError("terminal validation has already been revealed")
    request_hash = content_hash(
        {
            "terminal_freeze_hash": freeze.receipt_hash,
            "terminal_mask_hash": freeze.terminal_mask_hash,
            "terminal_validation_population_hash": (
                freeze.terminal_validation_population_hash
            ),
            "calibration_identity": state.controller.binding.contract.calibration_identity,
        },
        prefix="partition-free-v1-terminal-request-",
    )
    values = np.asarray(
        validator(
            freeze.terminal_mask_hash,
            freeze.terminal_validation_population_hash,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or bool(np.any(values < 0.0)):
        raise ValueError("terminal validator must return finite nonnegative rows")
    values = np.ascontiguousarray(values, dtype="<f8")
    values_hash = content_hash(
        {"shape": values.shape, "payload": values.tobytes(order="C")},
        prefix="partition-free-v1-terminal-values-",
    )
    outcome_hash = content_hash(
        {
            "terminal_request_hash": request_hash,
            "terminal_freeze_hash": freeze.receipt_hash,
            "validation_values_hash": values_hash,
        },
        prefix="partition-free-v1-terminal-outcome-",
    )
    receipt = V1TerminalValidationReceipt(
        terminal_freeze_hash=freeze.receipt_hash,
        terminal_request_hash=request_hash,
        terminal_outcome_hash=outcome_hash,
        validation_values_hash=values_hash,
        validation_value_count=int(values.size),
        calibration_identity=state.controller.binding.contract.calibration_identity,
        terminal_mask_hash=freeze.terminal_mask_hash,
        reveal_sequence=freeze.freeze_sequence + 1,
        metrics={
            "mean": float(np.mean(values, dtype=np.float64)),
            "maximum": float(np.max(values)),
            "root_mean_square": float(
                np.sqrt(np.mean(np.square(values), dtype=np.float64))
            ),
        },
    )
    return replace(state, terminal_validation=receipt)


def freeze_v1_terminal_validation_family(
    state: V1RuntimeState,
    *,
    family_freeze: V1TerminalValidationFamilyFreezeReceipt,
) -> V1RuntimeState:
    """Attach one outcome-independent dense/milestone/final family after freeze."""

    if type(state) is not V1RuntimeState:
        raise TypeError("terminal family freeze requires an exact runtime state")
    if type(family_freeze) is not V1TerminalValidationFamilyFreezeReceipt:
        raise TypeError("terminal family freeze requires an exact receipt")
    family_freeze.__post_init__()
    if state.terminal_freeze is None:
        raise PermissionError("terminal family is invisible before terminal freeze")
    if state.terminal_validation is not None:
        raise PermissionError("legacy terminal validation already owns the reveal branch")
    if state.terminal_validation_family_freeze is not None:
        if (
            state.terminal_validation_family_freeze.receipt_hash
            != family_freeze.receipt_hash
        ):
            raise ValueError("terminal validation family is already frozen differently")
        return state
    return replace(state, terminal_validation_family_freeze=family_freeze)


def prepare_v1_terminal_validation_family_request(
    state: V1RuntimeState,
    *,
    observation_artifact_relative_path: str,
    observation_artifact_schema: str = (
        V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA
    ),
) -> V1RuntimeState:
    """Freeze an outcome-free common-panel request for a durable checkpoint."""

    family_freeze = state.terminal_validation_family_freeze
    if family_freeze is None:
        raise PermissionError("terminal family request requires its exact family freeze")
    if state.terminal_validation_family_observation is not None:
        raise PermissionError("terminal family request cannot change after observation")
    if (
        observation_artifact_schema
        != V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA
    ):
        raise ValueError("terminal family request uses an unsupported matrix schema")
    request = V1TerminalValidationFamilyRequestReceipt(
        family_freeze=family_freeze,
        observation_artifact_relative_path=observation_artifact_relative_path,
        observation_artifact_schema=observation_artifact_schema,
        request_sequence=family_freeze.freeze_sequence + 1,
    )
    existing = state.terminal_validation_family_request
    if existing is not None:
        if existing.request_hash != request.request_hash:
            raise ValueError("terminal validation family request is already frozen differently")
        return state
    return replace(state, terminal_validation_family_request=request)


def _terminal_family_matrix_semantics(
    request: V1TerminalValidationFamilyRequestReceipt,
    values: np.ndarray,
    *,
    common_panel_evaluator_receipt_hash: str,
) -> dict[str, Any]:
    freeze = request.family_freeze
    matrix = np.ascontiguousarray(values, dtype="<f8")
    expected_shape = (
        len(freeze.unique_lane_mask_hashes),
        freeze.terminal_validation_context_count,
    )
    if (
        matrix.shape != expected_shape
        or not np.isfinite(matrix).all()
        or bool(np.any(matrix < 0.0))
    ):
        raise ValueError("terminal family matrix differs from its frozen lane/panel shape")
    payload = matrix.tobytes(order="C")
    joint_hash = content_hash(
        {"shape": matrix.shape, "payload": payload},
        prefix=_TERMINAL_VALIDATION_FAMILY_VALUES_PREFIX,
    )
    lane_hashes = tuple(
        content_hash(
            {"shape": row.shape, "payload": row.tobytes(order="C")},
            prefix=_TERMINAL_VALIDATION_FAMILY_LANE_VALUES_PREFIX,
        )
        for row in matrix
    )
    counts = (matrix.shape[1],) * matrix.shape[0]
    metrics = tuple(
        {
            "mean": float(np.mean(row, dtype=np.float64)),
            "maximum": float(np.max(row)),
            "root_mean_square": float(
                np.sqrt(np.mean(np.square(row), dtype=np.float64))
            ),
        }
        for row in matrix
    )
    if any(
        not math.isfinite(value)
        for lane_metrics in metrics
        for value in lane_metrics.values()
    ):
        raise ValueError("terminal family matrix metrics overflow their finite domain")
    evaluator_hash = _nonempty(
        common_panel_evaluator_receipt_hash,
        "common_panel_evaluator_receipt_hash",
    )
    semantic_hash = V1TerminalValidationFamilyObservationReceipt.derive_artifact_semantic_hash(
        terminal_request_hash=request.request_hash,
        joint_validation_values_hash=joint_hash,
        joint_validation_shape=matrix.shape,
        unique_lane_validation_values_hashes=lane_hashes,
        unique_lane_validation_value_counts=counts,
        unique_lane_metrics=metrics,
        common_panel_evaluator_receipt_hash=evaluator_hash,
    )
    return {
        "joint_validation_values_hash": joint_hash,
        "joint_validation_shape": matrix.shape,
        "unique_lane_validation_values_hashes": lane_hashes,
        "unique_lane_validation_value_counts": counts,
        "unique_lane_metrics": metrics,
        "common_panel_evaluator_receipt_hash": evaluator_hash,
        "observation_artifact_semantic_hash": semantic_hash,
        "payload": payload,
    }


def _terminal_family_matrix_header(
    request: V1TerminalValidationFamilyRequestReceipt,
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    payload = semantics["payload"]
    if type(payload) is not bytes:
        raise TypeError("terminal family matrix payload is not exact bytes")
    return {
        "schema": V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA,
        "payload_kind": "raw_little_endian_float64_c_matrix",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "terminal_request_hash": request.request_hash,
        "family_freeze_hash": request.family_freeze.receipt_hash,
        "joint_validation_values_hash": semantics["joint_validation_values_hash"],
        "joint_validation_shape": list(semantics["joint_validation_shape"]),
        "unique_lane_validation_values_hashes": list(
            semantics["unique_lane_validation_values_hashes"]
        ),
        "unique_lane_validation_value_counts": list(
            semantics["unique_lane_validation_value_counts"]
        ),
        "unique_lane_metrics": [
            dict(item) for item in semantics["unique_lane_metrics"]
        ],
        "common_panel_evaluator_receipt_hash": semantics[
            "common_panel_evaluator_receipt_hash"
        ],
        "observation_artifact_semantic_hash": semantics[
            "observation_artifact_semantic_hash"
        ],
        "security_boundary": _CHECKPOINT_SECURITY_BOUNDARY,
    }


def _terminal_family_artifact_path(
    artifact_root: str | os.PathLike[str],
    relative_path: str,
    *,
    create_parents: bool,
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("terminal family artifact path is not canonical relative POSIX")
    root = Path(artifact_root)
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError("terminal family artifact root is absent") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise ValueError("terminal family artifact root must be a nonsymlink directory")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if create_parents:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            else:
                _fsync_directory(current.parent)
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValueError("terminal family artifact parent is absent") from exc
        if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
            raise ValueError("terminal family artifact parent is symlinked or non-directory")
    destination = current / relative.name
    if destination.resolve(strict=False).parent != current.resolve(strict=True):
        raise ValueError("terminal family artifact escapes its registered parent")
    return destination


def _terminal_family_observation_from_artifact(
    request: V1TerminalValidationFamilyRequestReceipt,
    *,
    header: Mapping[str, Any],
    artifact_sha256: str,
    artifact_size_bytes: int,
) -> V1TerminalValidationFamilyObservationReceipt:
    return V1TerminalValidationFamilyObservationReceipt(
        request=request,
        joint_validation_values_hash=str(header["joint_validation_values_hash"]),
        joint_validation_shape=tuple(header["joint_validation_shape"]),
        unique_lane_validation_values_hashes=tuple(
            header["unique_lane_validation_values_hashes"]
        ),
        unique_lane_validation_value_counts=tuple(
            header["unique_lane_validation_value_counts"]
        ),
        unique_lane_metrics=tuple(header["unique_lane_metrics"]),
        common_panel_evaluator_receipt_hash=str(
            header["common_panel_evaluator_receipt_hash"]
        ),
        observation_artifact_relative_path=(
            request.observation_artifact_relative_path
        ),
        observation_artifact_sha256=artifact_sha256,
        observation_artifact_size_bytes=artifact_size_bytes,
        observation_artifact_semantic_hash=str(
            header["observation_artifact_semantic_hash"]
        ),
        observation_sequence=request.request_sequence + 1,
    )


def read_v1_terminal_validation_family_matrix_artifact(
    artifact_root: str | os.PathLike[str],
    request: V1TerminalValidationFamilyRequestReceipt,
    *,
    expected_observation: V1TerminalValidationFamilyObservationReceipt | None = None,
) -> tuple[V1TerminalValidationFamilyObservationReceipt, np.ndarray]:
    """Verify and replay one immutable common-panel matrix without evaluation."""

    if type(request) is not V1TerminalValidationFamilyRequestReceipt:
        raise TypeError("terminal family artifact replay requires an exact request")
    request.__post_init__()
    if (
        request.observation_artifact_schema
        != V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA
    ):
        raise ValueError("terminal family matrix schema is unsupported")
    source = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=False,
    )
    try:
        mode = source.lstat().st_mode
    except OSError as exc:
        raise ValueError("terminal family matrix artifact is absent") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ValueError("terminal family matrix artifact must be a nonsymlink regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        prefix_size = len(_TERMINAL_VALIDATION_FAMILY_MATRIX_MAGIC) + 8
        prefix = os.read(descriptor, prefix_size)
        if (
            len(prefix) != prefix_size
            or prefix[: len(_TERMINAL_VALIDATION_FAMILY_MATRIX_MAGIC)]
            != _TERMINAL_VALIDATION_FAMILY_MATRIX_MAGIC
        ):
            raise ValueError("terminal family matrix artifact magic is invalid")
        header_size = struct.unpack(">Q", prefix[-8:])[0]
        if header_size <= 0 or header_size > _MAX_FRAME_HEADER_BYTES:
            raise ValueError("terminal family matrix artifact header is invalid")
        header_bytes = os.read(descriptor, header_size)
        try:
            header = json.loads(
                header_bytes.decode("utf-8"),
                object_pairs_hook=lambda pairs: (
                    dict(pairs)
                    if len({key for key, _value in pairs}) == len(pairs)
                    else (_ for _ in ()).throw(ValueError("duplicate JSON key"))
                ),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("terminal family matrix artifact JSON is invalid") from exc
        if type(header) is not dict:
            raise ValueError("terminal family matrix artifact header is not an object")
        expected_fields = {
            "schema",
            "payload_kind",
            "payload_sha256",
            "payload_size_bytes",
            "terminal_request_hash",
            "family_freeze_hash",
            "joint_validation_values_hash",
            "joint_validation_shape",
            "unique_lane_validation_values_hashes",
            "unique_lane_validation_value_counts",
            "unique_lane_metrics",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
            "security_boundary",
        }
        payload_size = header.get("payload_size_bytes")
        if (
            set(header) != expected_fields
            or header.get("schema")
            != V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA
            or header.get("payload_kind")
            != "raw_little_endian_float64_c_matrix"
            or header.get("terminal_request_hash") != request.request_hash
            or header.get("family_freeze_hash")
            != request.family_freeze.receipt_hash
            or header.get("security_boundary") != _CHECKPOINT_SECURITY_BOUNDARY
            or isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or payload_size <= 0
            or before.st_size != prefix_size + header_size + payload_size
        ):
            raise ValueError("terminal family matrix artifact header differs")
        payload = b""
        while len(payload) < payload_size:
            chunk = os.read(descriptor, payload_size - len(payload))
            if not chunk:
                raise ValueError("terminal family matrix artifact payload is truncated")
            payload += chunk
        if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
            raise ValueError("terminal family matrix artifact payload digest differs")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("terminal family matrix artifact changed during replay")
    finally:
        os.close(descriptor)
    shape = tuple(header.get("joint_validation_shape", ()))
    if (
        len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or math.prod(shape) * 8 != len(payload)
    ):
        raise ValueError("terminal family matrix artifact shape differs")
    matrix = np.frombuffer(payload, dtype="<f8").reshape(shape).copy(order="C")
    semantics = _terminal_family_matrix_semantics(
        request,
        matrix,
        common_panel_evaluator_receipt_hash=str(
            header.get("common_panel_evaluator_receipt_hash")
        ),
    )
    if header != _terminal_family_matrix_header(request, semantics):
        raise ValueError("terminal family matrix artifact semantic replay differs")
    whole_sha256 = hashlib.sha256(prefix + header_bytes + payload).hexdigest()
    observation = _terminal_family_observation_from_artifact(
        request,
        header=header,
        artifact_sha256=whole_sha256,
        artifact_size_bytes=len(prefix) + len(header_bytes) + len(payload),
    )
    if expected_observation is not None:
        if type(expected_observation) is not V1TerminalValidationFamilyObservationReceipt:
            raise TypeError("expected terminal family observation is not exact")
        if expected_observation.terminal_outcome_hash != observation.terminal_outcome_hash:
            raise ValueError("terminal family matrix artifact differs from its observation")
    matrix.setflags(write=False)
    return observation, matrix


def publish_v1_terminal_validation_family_matrix_artifact(
    artifact_root: str | os.PathLike[str],
    request: V1TerminalValidationFamilyRequestReceipt,
    outcome: V1TerminalValidationFamilyMatrixOutcome,
) -> V1TerminalValidationFamilyObservationReceipt:
    """Durably publish once; valid existing bytes replay, all drift refuses."""

    if type(outcome) is not V1TerminalValidationFamilyMatrixOutcome:
        raise TypeError("terminal family publication requires an exact matrix outcome")
    semantics = _terminal_family_matrix_semantics(
        request,
        outcome.validation_values,
        common_panel_evaluator_receipt_hash=(
            outcome.common_panel_evaluator_receipt_hash
        ),
    )
    header = _terminal_family_matrix_header(request, semantics)
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload = semantics["payload"]
    framed = (
        _TERMINAL_VALIDATION_FAMILY_MATRIX_MAGIC
        + struct.pack(">Q", len(header_bytes))
        + header_bytes
        + payload
    )
    destination = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=True,
    )
    if destination.exists() or destination.is_symlink():
        observed, _matrix = read_v1_terminal_validation_family_matrix_artifact(
            artifact_root,
            request,
        )
        expected = _terminal_family_observation_from_artifact(
            request,
            header=header,
            artifact_sha256=hashlib.sha256(framed).hexdigest(),
            artifact_size_bytes=len(framed),
        )
        if observed.terminal_outcome_hash != expected.terminal_outcome_hash:
            raise FileExistsError(
                "terminal family matrix artifact already exists with other semantics"
            )
        return observed
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(framed)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            observed, _matrix = read_v1_terminal_validation_family_matrix_artifact(
                artifact_root,
                request,
            )
            expected = _terminal_family_observation_from_artifact(
                request,
                header=header,
                artifact_sha256=hashlib.sha256(framed).hexdigest(),
                artifact_size_bytes=len(framed),
            )
            if observed.terminal_outcome_hash != expected.terminal_outcome_hash:
                raise FileExistsError(
                    "terminal family matrix publication raced with other semantics"
                )
            return observed
        _fsync_directory(destination.parent)
    finally:
        temporary_existed = temporary.exists()
        temporary.unlink(missing_ok=True)
        if temporary_existed:
            _fsync_directory(destination.parent)
    observation, _matrix = read_v1_terminal_validation_family_matrix_artifact(
        artifact_root,
        request,
    )
    return observation


def reveal_v1_terminal_validation_family_once(
    state: V1RuntimeState,
    *,
    validator: TerminalFamilyValidator,
    artifact_root: str | os.PathLike[str],
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt,
) -> V1RuntimeState:
    """Reveal only after restoring the exact outcome-free request checkpoint."""

    request = state.terminal_validation_family_request
    if request is None:
        raise PermissionError("terminal family reveal requires a frozen request")
    if state.terminal_validation_family_observation is not None:
        raise PermissionError("terminal family has already been observed")
    if type(request_checkpoint_receipt) is not V1RuntimeCheckpointReceipt:
        raise TypeError("terminal family reveal requires an exact checkpoint receipt")
    checkpoint_path = Path(request_checkpoint_receipt.checkpoint_path)
    if checkpoint_path.parent.resolve() != Path(artifact_root).resolve():
        raise ValueError("terminal family artifact root differs from request checkpoint root")
    if (
        request_checkpoint_receipt.runtime_state_hash != state.runtime_state_hash
        or request_checkpoint_receipt.controller_snapshot_hash
        != state.controller.snapshot_receipt().receipt_hash
    ):
        raise ValueError("terminal family request checkpoint names another runtime state")
    restored, replay_receipt = read_v1_runtime_checkpoint(
        checkpoint_path,
        trusted_local=True,
    )
    restored_request = restored.terminal_validation_family_request
    if (
        replay_receipt.payload_sha256 != request_checkpoint_receipt.payload_sha256
        or replay_receipt.runtime_state_hash != request_checkpoint_receipt.runtime_state_hash
        or restored_request is None
        or restored_request.request_hash != request.request_hash
        or restored.terminal_validation_family_observation is not None
    ):
        raise ValueError("terminal family request checkpoint did not restore exactly")
    destination = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=True,
    )
    if destination.exists() or destination.is_symlink():
        observation, _matrix = read_v1_terminal_validation_family_matrix_artifact(
            artifact_root,
            request,
        )
    else:
        outcome = validator(request)
        if type(outcome) is not V1TerminalValidationFamilyMatrixOutcome:
            raise TypeError("terminal family validator returned the wrong outcome type")
        observation = publish_v1_terminal_validation_family_matrix_artifact(
            artifact_root,
            request,
            outcome,
        )
    return replace(state, terminal_validation_family_observation=observation)


def complete_v1_terminal_validation_family(state: V1RuntimeState) -> V1RuntimeState:
    """Promote a replayable observation into matched-density authority."""

    observation = state.terminal_validation_family_observation
    if observation is None:
        raise PermissionError("terminal family completion requires an observation")
    if state.terminal_validation_family is not None:
        return state
    receipt = V1TerminalValidationFamilyReceipt(
        observation=observation,
        reveal_sequence=observation.observation_sequence,
    )
    return replace(state, terminal_validation_family=receipt)


def freeze_v1_terminal_comparison_family_v2(
    state: V1RuntimeState,
    *,
    family_freeze: V1TerminalComparisonFamilyFreezeReceiptV2,
) -> V1RuntimeState:
    """Attach the outcome-free dense/current-v0/final-v1 family."""

    if type(state) is not V1RuntimeState:
        raise TypeError("terminal comparison freeze requires an exact runtime state")
    if type(family_freeze) is not V1TerminalComparisonFamilyFreezeReceiptV2:
        raise TypeError("terminal comparison freeze requires the exact v2 receipt")
    family_freeze.__post_init__()
    if state.terminal_freeze is None:
        raise PermissionError("terminal comparison is invisible before terminal freeze")
    if state.terminal_validation is not None or any(
        item is not None
        for item in (
            state.terminal_validation_family_freeze,
            state.terminal_validation_family_request,
            state.terminal_validation_family_observation,
            state.terminal_validation_family,
        )
    ):
        raise PermissionError("another terminal validation branch already owns reveal")
    existing = state.terminal_comparison_family_freeze
    if existing is not None:
        if existing.receipt_hash != family_freeze.receipt_hash:
            raise ValueError("terminal comparison family is already frozen differently")
        return state
    return replace(state, terminal_comparison_family_freeze=family_freeze)


def prepare_v1_terminal_comparison_family_request_v2(
    state: V1RuntimeState,
    *,
    observation_artifact_relative_path: str,
    observation_artifact_schema: str = (
        V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
    ),
) -> V1RuntimeState:
    """Freeze the outcome-free three-role request before any validation read."""

    family_freeze = state.terminal_comparison_family_freeze
    if family_freeze is None:
        raise PermissionError("terminal comparison request requires its exact freeze")
    if state.terminal_comparison_family_observation is not None:
        raise PermissionError("terminal comparison request cannot change after reveal")
    if (
        observation_artifact_schema
        != V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
    ):
        raise ValueError("terminal comparison request uses an unsupported matrix schema")
    request = V1TerminalComparisonFamilyRequestReceiptV2(
        family_freeze=family_freeze,
        observation_artifact_relative_path=observation_artifact_relative_path,
        observation_artifact_schema=observation_artifact_schema,
        request_sequence=family_freeze.freeze_sequence + 1,
    )
    existing = state.terminal_comparison_family_request
    if existing is not None:
        if existing.request_hash != request.request_hash:
            raise ValueError("terminal comparison request is already frozen differently")
        return state
    return replace(state, terminal_comparison_family_request=request)


def _terminal_comparison_matrix_semantics_v2(
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    values: np.ndarray,
    *,
    common_panel_evaluator_receipt_hash: str,
) -> dict[str, Any]:
    matrix = np.ascontiguousarray(values, dtype="<f8")
    expected_shape = (
        len(request.family_freeze.unique_lane_mask_hashes),
        request.family_freeze.terminal_validation_context_count,
    )
    if (
        matrix.shape != expected_shape
        or not np.isfinite(matrix).all()
        or bool(np.any(matrix < 0.0))
    ):
        raise ValueError(
            "terminal comparison matrix differs from its frozen lane/panel shape"
        )
    provisional = V1TerminalComparisonFamilyObservationReceiptV2.from_replayed_matrix(
        request=request,
        validation_rows=matrix,
        common_panel_evaluator_receipt_hash=(
            common_panel_evaluator_receipt_hash
        ),
        observation_artifact_relative_path=(
            request.observation_artifact_relative_path
        ),
        observation_artifact_sha256="0" * 64,
        observation_artifact_size_bytes=1,
    )
    return {
        "joint_validation_values_hash": provisional.joint_validation_values_hash,
        "joint_validation_shape": provisional.joint_validation_shape,
        "unique_lane_validation_values_hashes": (
            provisional.unique_lane_validation_values_hashes
        ),
        "unique_lane_validation_value_counts": (
            provisional.unique_lane_validation_value_counts
        ),
        "unique_lane_metrics": provisional.unique_lane_metrics,
        "paired_mean_deltas": dict(provisional.paired_mean_deltas),
        "paired_validation_difference_hashes": dict(
            provisional.paired_validation_difference_hashes
        ),
        "paired_delta_derivation": provisional.paired_delta_derivation,
        "common_panel_evaluator_receipt_hash": (
            provisional.common_panel_evaluator_receipt_hash
        ),
        "observation_artifact_semantic_hash": (
            provisional.observation_artifact_semantic_hash
        ),
        "payload": matrix.tobytes(order="C"),
    }


def _terminal_comparison_matrix_header_v2(
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    semantics: Mapping[str, Any],
) -> dict[str, Any]:
    payload = semantics["payload"]
    if type(payload) is not bytes:
        raise TypeError("terminal comparison matrix payload is not exact bytes")
    return {
        "schema": V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2,
        "payload_kind": "raw_little_endian_float64_c_matrix",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_size_bytes": len(payload),
        "terminal_request_hash": request.request_hash,
        "family_freeze_hash": request.family_freeze.receipt_hash,
        "roles": list(request.family_freeze.roles),
        "role_to_unique_lane": list(request.family_freeze.role_to_unique_lane),
        "joint_validation_values_hash": semantics["joint_validation_values_hash"],
        "joint_validation_shape": list(semantics["joint_validation_shape"]),
        "unique_lane_validation_values_hashes": list(
            semantics["unique_lane_validation_values_hashes"]
        ),
        "unique_lane_validation_value_counts": list(
            semantics["unique_lane_validation_value_counts"]
        ),
        "unique_lane_metrics": [
            dict(item) for item in semantics["unique_lane_metrics"]
        ],
        "paired_mean_deltas": dict(semantics["paired_mean_deltas"]),
        "paired_validation_difference_hashes": dict(
            semantics["paired_validation_difference_hashes"]
        ),
        "paired_delta_derivation": semantics["paired_delta_derivation"],
        "common_panel_evaluator_receipt_hash": semantics[
            "common_panel_evaluator_receipt_hash"
        ],
        "observation_artifact_semantic_hash": semantics[
            "observation_artifact_semantic_hash"
        ],
        "security_boundary": _CHECKPOINT_SECURITY_BOUNDARY,
    }


def _terminal_comparison_observation_from_header_v2(
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    *,
    header: Mapping[str, Any],
    artifact_sha256: str,
    artifact_size_bytes: int,
) -> V1TerminalComparisonFamilyObservationReceiptV2:
    return V1TerminalComparisonFamilyObservationReceiptV2(
        request=request,
        joint_validation_values_hash=str(header["joint_validation_values_hash"]),
        joint_validation_shape=tuple(header["joint_validation_shape"]),
        unique_lane_validation_values_hashes=tuple(
            header["unique_lane_validation_values_hashes"]
        ),
        unique_lane_validation_value_counts=tuple(
            header["unique_lane_validation_value_counts"]
        ),
        unique_lane_metrics=tuple(header["unique_lane_metrics"]),
        paired_mean_deltas=dict(header["paired_mean_deltas"]),
        paired_validation_difference_hashes=dict(
            header["paired_validation_difference_hashes"]
        ),
        common_panel_evaluator_receipt_hash=str(
            header["common_panel_evaluator_receipt_hash"]
        ),
        observation_artifact_relative_path=(
            request.observation_artifact_relative_path
        ),
        observation_artifact_sha256=artifact_sha256,
        observation_artifact_size_bytes=artifact_size_bytes,
        observation_artifact_semantic_hash=str(
            header["observation_artifact_semantic_hash"]
        ),
        observation_sequence=request.request_sequence + 1,
    )


def read_v1_terminal_comparison_family_matrix_artifact_v2(
    artifact_root: str | os.PathLike[str],
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    *,
    expected_observation: V1TerminalComparisonFamilyObservationReceiptV2 | None = None,
) -> tuple[V1TerminalComparisonFamilyObservationReceiptV2, np.ndarray]:
    """Verify and replay the immutable MV1TERM2 panel without evaluation."""

    if type(request) is not V1TerminalComparisonFamilyRequestReceiptV2:
        raise TypeError("terminal comparison replay requires the exact v2 request")
    request.__post_init__()
    if (
        request.observation_artifact_schema
        != V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
    ):
        raise ValueError("terminal comparison matrix schema is unsupported")
    source = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=False,
    )
    try:
        mode = source.lstat().st_mode
    except OSError as exc:
        raise ValueError("terminal comparison matrix artifact is absent") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ValueError(
            "terminal comparison matrix artifact must be a nonsymlink regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before = os.fstat(descriptor)
        prefix_size = len(_TERMINAL_COMPARISON_FAMILY_MATRIX_MAGIC_V2) + 8
        prefix = os.read(descriptor, prefix_size)
        if (
            len(prefix) != prefix_size
            or prefix[: len(_TERMINAL_COMPARISON_FAMILY_MATRIX_MAGIC_V2)]
            != _TERMINAL_COMPARISON_FAMILY_MATRIX_MAGIC_V2
        ):
            raise ValueError("terminal comparison matrix artifact magic is invalid")
        header_size = struct.unpack(">Q", prefix[-8:])[0]
        if header_size <= 0 or header_size > _MAX_FRAME_HEADER_BYTES:
            raise ValueError("terminal comparison matrix artifact header is invalid")
        header_bytes = os.read(descriptor, header_size)
        try:
            header = json.loads(
                header_bytes.decode("utf-8"),
                object_pairs_hook=lambda pairs: (
                    dict(pairs)
                    if len({key for key, _value in pairs}) == len(pairs)
                    else (_ for _ in ()).throw(ValueError("duplicate JSON key"))
                ),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("terminal comparison matrix JSON is invalid") from exc
        # Header-shape validation must not allocate or traverse a synthetic
        # validation matrix.  At the production 50k panel size, even a
        # shape-only check would otherwise perform an avoidable second
        # statistics pass during every replay.
        expected_fields = {
            "schema",
            "payload_kind",
            "payload_sha256",
            "payload_size_bytes",
            "terminal_request_hash",
            "family_freeze_hash",
            "roles",
            "role_to_unique_lane",
            "joint_validation_values_hash",
            "joint_validation_shape",
            "unique_lane_validation_values_hashes",
            "unique_lane_validation_value_counts",
            "unique_lane_metrics",
            "paired_mean_deltas",
            "paired_validation_difference_hashes",
            "paired_delta_derivation",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
            "security_boundary",
        }
        payload_size = header.get("payload_size_bytes") if isinstance(header, dict) else None
        if (
            type(header) is not dict
            or set(header) != expected_fields
            or header.get("schema")
            != V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
            or header.get("payload_kind")
            != "raw_little_endian_float64_c_matrix"
            or header.get("terminal_request_hash") != request.request_hash
            or header.get("family_freeze_hash")
            != request.family_freeze.receipt_hash
            or header.get("security_boundary") != _CHECKPOINT_SECURITY_BOUNDARY
            or isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or payload_size <= 0
            or before.st_size != prefix_size + header_size + payload_size
        ):
            raise ValueError("terminal comparison matrix artifact header differs")
        payload = b""
        while len(payload) < payload_size:
            chunk = os.read(descriptor, payload_size - len(payload))
            if not chunk:
                raise ValueError("terminal comparison matrix payload is truncated")
            payload += chunk
        if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
            raise ValueError("terminal comparison matrix payload digest differs")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("terminal comparison matrix changed during replay")
    finally:
        os.close(descriptor)
    shape = tuple(header.get("joint_validation_shape", ()))
    if (
        len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or math.prod(shape) * 8 != len(payload)
    ):
        raise ValueError("terminal comparison matrix shape differs")
    matrix = np.frombuffer(payload, dtype="<f8").reshape(shape).copy(order="C")
    semantics = _terminal_comparison_matrix_semantics_v2(
        request,
        matrix,
        common_panel_evaluator_receipt_hash=str(
            header.get("common_panel_evaluator_receipt_hash")
        ),
    )
    if header != _terminal_comparison_matrix_header_v2(request, semantics):
        raise ValueError("terminal comparison matrix semantic replay differs")
    framed_size = len(prefix) + len(header_bytes) + len(payload)
    observation = _terminal_comparison_observation_from_header_v2(
        request,
        header=header,
        artifact_sha256=hashlib.sha256(prefix + header_bytes + payload).hexdigest(),
        artifact_size_bytes=framed_size,
    )
    if expected_observation is not None:
        if type(expected_observation) is not V1TerminalComparisonFamilyObservationReceiptV2:
            raise TypeError("expected terminal comparison observation is not exact")
        if expected_observation.terminal_outcome_hash != observation.terminal_outcome_hash:
            raise ValueError("terminal comparison matrix differs from its observation")
    matrix.setflags(write=False)
    return observation, matrix


def publish_v1_terminal_comparison_family_matrix_artifact_v2(
    artifact_root: str | os.PathLike[str],
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    outcome: V1TerminalComparisonFamilyMatrixOutcomeV2,
) -> V1TerminalComparisonFamilyObservationReceiptV2:
    """Durably publish MV1TERM2 once; drift never clobbers existing bytes."""

    if type(outcome) is not V1TerminalComparisonFamilyMatrixOutcomeV2:
        raise TypeError("terminal comparison publication requires the exact v2 outcome")
    semantics = _terminal_comparison_matrix_semantics_v2(
        request,
        outcome.validation_values,
        common_panel_evaluator_receipt_hash=(
            outcome.common_panel_evaluator_receipt_hash
        ),
    )
    header = _terminal_comparison_matrix_header_v2(request, semantics)
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload = semantics["payload"]
    framed = (
        _TERMINAL_COMPARISON_FAMILY_MATRIX_MAGIC_V2
        + struct.pack(">Q", len(header_bytes))
        + header_bytes
        + payload
    )
    destination = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=True,
    )
    expected = _terminal_comparison_observation_from_header_v2(
        request,
        header=header,
        artifact_sha256=hashlib.sha256(framed).hexdigest(),
        artifact_size_bytes=len(framed),
    )
    if destination.exists() or destination.is_symlink():
        observed, _matrix = read_v1_terminal_comparison_family_matrix_artifact_v2(
            artifact_root, request
        )
        if observed.terminal_outcome_hash != expected.terminal_outcome_hash:
            raise FileExistsError(
                "terminal comparison matrix already exists with other semantics"
            )
        return observed
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(framed)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            observed, _matrix = read_v1_terminal_comparison_family_matrix_artifact_v2(
                artifact_root, request
            )
            if observed.terminal_outcome_hash != expected.terminal_outcome_hash:
                raise FileExistsError(
                    "terminal comparison publication raced with other semantics"
                )
            return observed
        _fsync_directory(destination.parent)
    finally:
        temporary_existed = temporary.exists()
        temporary.unlink(missing_ok=True)
        if temporary_existed:
            _fsync_directory(destination.parent)
    observation, _matrix = read_v1_terminal_comparison_family_matrix_artifact_v2(
        artifact_root, request
    )
    return observation


def reveal_v1_terminal_comparison_family_once_v2(
    state: V1RuntimeState,
    *,
    validator: TerminalComparisonFamilyValidatorV2,
    artifact_root: str | os.PathLike[str],
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt,
) -> V1RuntimeState:
    """Reveal once after exact replay of the outcome-free request checkpoint."""

    request = state.terminal_comparison_family_request
    if request is None:
        raise PermissionError("terminal comparison reveal requires a frozen request")
    if state.terminal_comparison_family_observation is not None:
        raise PermissionError("terminal comparison family has already been observed")
    if type(request_checkpoint_receipt) is not V1RuntimeCheckpointReceipt:
        raise TypeError("terminal comparison reveal requires an exact checkpoint receipt")
    checkpoint_path = Path(request_checkpoint_receipt.checkpoint_path)
    if checkpoint_path.parent.resolve() != Path(artifact_root).resolve():
        raise ValueError("terminal comparison root differs from checkpoint root")
    if (
        request_checkpoint_receipt.runtime_state_hash != state.runtime_state_hash
        or request_checkpoint_receipt.controller_snapshot_hash
        != state.controller.snapshot_receipt().receipt_hash
        or request_checkpoint_receipt.terminal_comparison_family_request_hash
        != request.request_hash
    ):
        raise ValueError("terminal comparison checkpoint names another request state")
    restored, replay_receipt = read_v1_runtime_checkpoint(
        checkpoint_path, trusted_local=True
    )
    restored_request = restored.terminal_comparison_family_request
    if (
        replay_receipt.payload_sha256 != request_checkpoint_receipt.payload_sha256
        or replay_receipt.runtime_state_hash != request_checkpoint_receipt.runtime_state_hash
        or restored_request is None
        or restored_request.request_hash != request.request_hash
        or restored.terminal_comparison_family_observation is not None
    ):
        raise ValueError("terminal comparison request checkpoint did not restore exactly")
    destination = _terminal_family_artifact_path(
        artifact_root,
        request.observation_artifact_relative_path,
        create_parents=True,
    )
    if destination.exists() or destination.is_symlink():
        observation, _matrix = read_v1_terminal_comparison_family_matrix_artifact_v2(
            artifact_root, request
        )
    else:
        outcome = validator(request)
        if type(outcome) is not V1TerminalComparisonFamilyMatrixOutcomeV2:
            raise TypeError("terminal comparison validator returned the wrong outcome")
        observation = publish_v1_terminal_comparison_family_matrix_artifact_v2(
            artifact_root, request, outcome
        )
    return replace(state, terminal_comparison_family_observation=observation)


def complete_v1_terminal_comparison_family_v2(
    state: V1RuntimeState,
) -> V1RuntimeState:
    """Promote only a replayed MV1TERM2 observation into comparison authority."""

    observation = state.terminal_comparison_family_observation
    if observation is None:
        raise PermissionError("terminal comparison completion requires an observation")
    if state.terminal_comparison_family is not None:
        return state
    receipt = V1TerminalComparisonFamilyReceiptV2(
        observation=observation,
        reveal_sequence=observation.observation_sequence,
    )
    return replace(state, terminal_comparison_family=receipt)


def _terminal_family_artifact_reference_payload(
    observation: V1TerminalValidationFamilyObservationReceipt,
) -> dict[str, Any]:
    request = observation.request
    return {
        "schema": _TERMINAL_VALIDATION_FAMILY_ARTIFACT_REFERENCE_SCHEMA,
        "relative_path": observation.observation_artifact_relative_path,
        "artifact_schema": request.observation_artifact_schema,
        "artifact_sha256": observation.observation_artifact_sha256,
        "artifact_size_bytes": observation.observation_artifact_size_bytes,
        "terminal_request_hash": request.request_hash,
        "family_freeze_hash": request.family_freeze.receipt_hash,
        "joint_validation_values_hash": observation.joint_validation_values_hash,
        "joint_validation_shape": list(observation.joint_validation_shape),
        "unique_lane_validation_values_hashes": list(
            observation.unique_lane_validation_values_hashes
        ),
        "unique_lane_validation_value_counts": list(
            observation.unique_lane_validation_value_counts
        ),
        "unique_lane_metrics": [
            dict(item) for item in observation.unique_lane_metrics
        ],
        "common_panel_evaluator_receipt_hash": (
            observation.common_panel_evaluator_receipt_hash
        ),
        "observation_artifact_semantic_hash": (
            observation.observation_artifact_semantic_hash
        ),
        "terminal_observation_hash": observation.terminal_outcome_hash,
    }


def _terminal_comparison_artifact_reference_payload_v2(
    observation: V1TerminalComparisonFamilyObservationReceiptV2,
) -> dict[str, Any]:
    request = observation.request
    return {
        "schema": _TERMINAL_COMPARISON_FAMILY_ARTIFACT_REFERENCE_SCHEMA_V2,
        "relative_path": observation.observation_artifact_relative_path,
        "artifact_schema": request.observation_artifact_schema,
        "artifact_sha256": observation.observation_artifact_sha256,
        "artifact_size_bytes": observation.observation_artifact_size_bytes,
        "terminal_request_hash": request.request_hash,
        "family_freeze_hash": request.family_freeze.receipt_hash,
        "roles": list(request.family_freeze.roles),
        "role_to_unique_lane": list(request.family_freeze.role_to_unique_lane),
        "joint_validation_values_hash": observation.joint_validation_values_hash,
        "joint_validation_shape": list(observation.joint_validation_shape),
        "unique_lane_validation_values_hashes": list(
            observation.unique_lane_validation_values_hashes
        ),
        "unique_lane_validation_value_counts": list(
            observation.unique_lane_validation_value_counts
        ),
        "unique_lane_metrics": [
            dict(item) for item in observation.unique_lane_metrics
        ],
        "paired_mean_deltas": dict(observation.paired_mean_deltas),
        "paired_validation_difference_hashes": dict(
            observation.paired_validation_difference_hashes
        ),
        "paired_delta_derivation": observation.paired_delta_derivation,
        "common_panel_evaluator_receipt_hash": (
            observation.common_panel_evaluator_receipt_hash
        ),
        "observation_artifact_semantic_hash": (
            observation.observation_artifact_semantic_hash
        ),
        "terminal_observation_hash": observation.terminal_outcome_hash,
    }


def _assert_terminal_family_checkpoint_artifact_path(
    checkpoint_path: Path,
    request: V1TerminalValidationFamilyRequestReceipt,
) -> None:
    relative = PurePosixPath(request.observation_artifact_relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0] != f"{checkpoint_path.name}.terminal_validation"
        or not relative.parts[1].endswith(".mv1term")
    ):
        raise ValueError(
            "terminal family matrix must use its exact checkpoint-owned directory"
        )


def _assert_terminal_comparison_checkpoint_artifact_path_v2(
    checkpoint_path: Path,
    request: V1TerminalComparisonFamilyRequestReceiptV2,
) -> None:
    relative = PurePosixPath(request.observation_artifact_relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0] != f"{checkpoint_path.name}.terminal_comparison"
        or not relative.parts[1].endswith(".mv1term2")
    ):
        raise ValueError(
            "terminal comparison matrix must use its checkpoint-owned directory"
        )


def _validate_terminal_family_artifact_reference(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("terminal family artifact reference must be a mapping")
    reference = dict(value)
    expected_fields = {
        "schema",
        "relative_path",
        "artifact_schema",
        "artifact_sha256",
        "artifact_size_bytes",
        "terminal_request_hash",
        "family_freeze_hash",
        "joint_validation_values_hash",
        "joint_validation_shape",
        "unique_lane_validation_values_hashes",
        "unique_lane_validation_value_counts",
        "unique_lane_metrics",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    }
    if set(reference) != expected_fields:
        raise ValueError("terminal family artifact reference fields differ")
    if (
        reference["schema"]
        != _TERMINAL_VALIDATION_FAMILY_ARTIFACT_REFERENCE_SCHEMA
        or reference["artifact_schema"]
        != V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA
    ):
        raise ValueError("terminal family artifact reference schema differs")
    raw_relative = _nonempty(
        reference["relative_path"],
        "terminal family relative_path",
    )
    relative = PurePosixPath(raw_relative)
    if (
        relative.is_absolute()
        or relative.as_posix() != raw_relative
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("terminal family artifact reference path is not canonical")
    _sha256_hex(reference["artifact_sha256"], "terminal family artifact_sha256")
    size = reference["artifact_size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("terminal family artifact_size_bytes must be positive")
    for name in (
        "terminal_request_hash",
        "family_freeze_hash",
        "joint_validation_values_hash",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    ):
        _nonempty(reference[name], name)
    shape = reference["joint_validation_shape"]
    lane_hashes = reference["unique_lane_validation_values_hashes"]
    lane_counts = reference["unique_lane_validation_value_counts"]
    metrics = reference["unique_lane_metrics"]
    if (
        type(shape) not in {list, tuple}
        or len(shape) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape
        )
        or type(lane_hashes) not in {list, tuple}
        or type(lane_counts) not in {list, tuple}
        or type(metrics) not in {list, tuple}
        or not (
            len(lane_hashes) == len(lane_counts) == len(metrics) == shape[0]
        )
    ):
        raise ValueError("terminal family artifact reference lane shape differs")
    reference["joint_validation_shape"] = list(shape)
    reference["unique_lane_validation_values_hashes"] = list(lane_hashes)
    reference["unique_lane_validation_value_counts"] = list(lane_counts)
    reference["unique_lane_metrics"] = [dict(item) for item in metrics]
    expected_semantic_hash = (
        V1TerminalValidationFamilyObservationReceipt.derive_artifact_semantic_hash(
            terminal_request_hash=reference["terminal_request_hash"],
            joint_validation_values_hash=reference["joint_validation_values_hash"],
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=lane_hashes,
            unique_lane_validation_value_counts=lane_counts,
            unique_lane_metrics=metrics,
            common_panel_evaluator_receipt_hash=reference[
                "common_panel_evaluator_receipt_hash"
            ],
        )
    )
    if reference["observation_artifact_semantic_hash"] != expected_semantic_hash:
        raise ValueError("terminal family artifact reference semantic hash differs")
    return reference


def _validate_terminal_comparison_artifact_reference_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("terminal comparison artifact reference must be a mapping")
    reference = dict(value)
    expected_fields = {
        "schema",
        "relative_path",
        "artifact_schema",
        "artifact_sha256",
        "artifact_size_bytes",
        "terminal_request_hash",
        "family_freeze_hash",
        "roles",
        "role_to_unique_lane",
        "joint_validation_values_hash",
        "joint_validation_shape",
        "unique_lane_validation_values_hashes",
        "unique_lane_validation_value_counts",
        "unique_lane_metrics",
        "paired_mean_deltas",
        "paired_validation_difference_hashes",
        "paired_delta_derivation",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    }
    if set(reference) != expected_fields:
        raise ValueError("terminal comparison artifact reference fields differ")
    if (
        reference["schema"]
        != _TERMINAL_COMPARISON_FAMILY_ARTIFACT_REFERENCE_SCHEMA_V2
        or reference["artifact_schema"]
        != V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
    ):
        raise ValueError("terminal comparison artifact reference schema differs")
    _sha256_hex(reference["artifact_sha256"], "artifact_sha256")
    _positive(reference["artifact_size_bytes"], "artifact_size_bytes")
    for name in (
        "relative_path",
        "terminal_request_hash",
        "family_freeze_hash",
        "joint_validation_values_hash",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    ):
        _nonempty(reference[name], name)
    shape = reference["joint_validation_shape"]
    lane_hashes = reference["unique_lane_validation_values_hashes"]
    lane_counts = reference["unique_lane_validation_value_counts"]
    metrics = reference["unique_lane_metrics"]
    if (
        type(shape) not in {list, tuple}
        or len(shape) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape
        )
        or type(lane_hashes) not in {list, tuple}
        or type(lane_counts) not in {list, tuple}
        or type(metrics) not in {list, tuple}
        or not (
            len(lane_hashes) == len(lane_counts) == len(metrics) == shape[0]
        )
        or any(item != shape[1] for item in lane_counts)
    ):
        raise ValueError("terminal comparison artifact lane shape differs")
    if (
        tuple(reference["roles"])
        != ("dense_reference", "current_v0_comparator", "terminal_v1")
        or type(reference["role_to_unique_lane"]) not in {list, tuple}
        or len(reference["role_to_unique_lane"]) != 3
        or any(
            isinstance(lane, bool)
            or not isinstance(lane, int)
            or lane < 0
            or lane >= shape[0]
            for lane in reference["role_to_unique_lane"]
        )
    ):
        raise ValueError("terminal comparison artifact role map differs")
    deltas = reference["paired_mean_deltas"]
    pair_hashes = reference["paired_validation_difference_hashes"]
    expected_pairs = {
        "current_v0_comparator_minus_dense_reference",
        "terminal_v1_minus_dense_reference",
        "terminal_v1_minus_current_v0_comparator",
    }
    if (
        not isinstance(deltas, Mapping)
        or not isinstance(pair_hashes, Mapping)
        or set(deltas) != expected_pairs
        or set(pair_hashes) != expected_pairs
        or reference["paired_delta_derivation"]
        != "mean_of_pointwise_common_panel_row_differences_float64_v1"
    ):
        raise ValueError("terminal comparison paired reference fields differ")
    for value in deltas.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("terminal comparison paired delta is nonfinite")
    for value in pair_hashes.values():
        _nonempty(value, "paired validation difference hash")
    expected_semantic_hash = (
        V1TerminalComparisonFamilyObservationReceiptV2.derive_artifact_semantic_hash(
            terminal_request_hash=reference["terminal_request_hash"],
            joint_validation_values_hash=reference["joint_validation_values_hash"],
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=lane_hashes,
            unique_lane_validation_value_counts=lane_counts,
            unique_lane_metrics=metrics,
            paired_mean_deltas=deltas,
            paired_validation_difference_hashes=pair_hashes,
            common_panel_evaluator_receipt_hash=reference[
                "common_panel_evaluator_receipt_hash"
            ],
        )
    )
    if reference["observation_artifact_semantic_hash"] != expected_semantic_hash:
        raise ValueError("terminal comparison artifact semantic hash differs")
    reference["joint_validation_shape"] = list(shape)
    reference["unique_lane_validation_values_hashes"] = list(lane_hashes)
    reference["unique_lane_validation_value_counts"] = list(lane_counts)
    reference["unique_lane_metrics"] = [dict(item) for item in metrics]
    reference["paired_mean_deltas"] = dict(deltas)
    reference["paired_validation_difference_hashes"] = dict(pair_hashes)
    reference["roles"] = list(reference["roles"])
    reference["role_to_unique_lane"] = list(reference["role_to_unique_lane"])
    return reference


def attach_v1_performance_receipt(
    state: V1RuntimeState,
    receipt: V1PerformanceReceipt,
) -> V1RuntimeState:
    if type(receipt) is not V1PerformanceReceipt:
        raise TypeError("runtime performance attachment requires an exact receipt")
    if (
        state.terminal_validation is None
        and state.terminal_validation_family is None
        and state.terminal_comparison_family is None
    ):
        raise PermissionError("performance receipt requires completed terminal validation")
    if state.completion is not None:
        raise PermissionError("performance receipt cannot alter a completed runtime")
    if receipt.controller_run_id != state.controller.controller_run_id:
        raise ValueError("performance receipt names another controller run")
    if receipt.source_environment_hash != state.config.performance_source_environment_hash:
        raise ValueError("performance receipt names another source/environment/packing identity")
    if state.performance is not None and state.performance.receipt_hash != receipt.receipt_hash:
        raise ValueError("runtime already has a different performance receipt")
    return replace(state, performance=receipt)


def complete_v1_runtime(state: V1RuntimeState) -> V1RuntimeState:
    """Emit a genuine-controller development completion receipt, never promotion."""

    terminal_authority = (
        state.terminal_validation
        if state.terminal_validation is not None
        else (
            state.terminal_validation_family
            if state.terminal_validation_family is not None
            else state.terminal_comparison_family
        )
    )
    if state.terminal_freeze is None or terminal_authority is None:
        raise PermissionError("runtime completion requires freeze and one terminal reveal")
    if state.completion is not None:
        return state
    snapshot = state.controller.snapshot_receipt()
    receipt = V1RunCompletionReceipt(
        campaign_contract_hash=state.controller.binding.contract.campaign_contract_hash,
        campaign_binding_hash=state.controller.binding.binding_hash,
        controller_run_id=state.controller.controller_run_id,
        final_controller_snapshot_hash=snapshot.receipt_hash,
        terminal_freeze_hash=state.terminal_freeze.receipt_hash,
        terminal_validation_hash=terminal_authority.receipt_hash,
        terminal_mask_hash=state.controller.current_mask.mask_hash,
        accepted_step_count=state.controller.accepted_step_count,
        candidate_attempt_count=state.controller.candidate_attempt_count,
        performance_receipt_hash=(
            None if state.performance is None else state.performance.receipt_hash
        ),
        source_seal_hash=state.config.source_seal_hash,
        environment_hash=state.config.environment_hash,
        completion_sequence=terminal_authority.reveal_sequence + 1,
    )
    return replace(state, completion=receipt)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_framed_payload(
    *,
    destination: Path,
    magic: bytes,
    header: Mapping[str, Any],
    payload: bytes,
) -> None:
    header_bytes = json.dumps(
        dict(header),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not header_bytes or len(header_bytes) > _MAX_FRAME_HEADER_BYTES:
        raise ValueError("modern-v1 checkpoint header exceeds its framing bound")
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(magic)
        handle.write(struct.pack(">Q", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)


class _DigestingWriter:
    """File facade that hashes the exact pickle stream without retaining it."""

    __slots__ = ("handle", "digest", "size")

    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, payload: Any) -> int:
        # Protocol-5 picklers pass large NumPy storage as ``PickleBuffer``.
        # It implements the buffer protocol but deliberately has no ``len``;
        # ``memoryview.nbytes`` is the exact byte count for bytes, views, and
        # PickleBuffer alike without materializing another route-scale copy.
        view = memoryview(payload)
        try:
            expected = view.nbytes
            count = self.handle.write(view)
            if count != expected:
                raise OSError("short write while serializing acquisition sidecar")
            self.digest.update(view)
            self.size += count
            return count
        finally:
            view.release()


@dataclass(frozen=True, slots=True)
class _SidecarCacheEntry:
    receipt: V1RuntimeAcquisitionSidecarReceipt
    stat_signature: tuple[int, int, int, int]


_SIDECAR_CACHE: dict[str, _SidecarCacheEntry] = {}


@dataclass(frozen=True, slots=True)
class _ProposalSidecarCacheEntry:
    receipt: V1RuntimeProposalStateSidecarReceipt
    stat_signature: tuple[int, int, int, int]


_PROPOSAL_SIDECAR_CACHE: dict[str, _ProposalSidecarCacheEntry] = {}


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    observed = path.stat()
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
    )


def _sidecar_reference_payload(
    receipt: V1RuntimeAcquisitionSidecarReceipt,
) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "relative_path": receipt.relative_path,
        "payload_sha256": receipt.payload_sha256,
        "payload_size_bytes": receipt.payload_size_bytes,
        "framed_size_bytes": receipt.framed_size_bytes,
        "acquisition_hash": receipt.acquisition_hash,
        "acquisition_receipt_hash": receipt.acquisition_receipt_hash,
        "campaign_binding_hash": receipt.campaign_binding_hash,
        "generation_index": receipt.generation_index,
        "current_mask_hash": receipt.current_mask_hash,
        "route_axis_hash": receipt.route_axis_hash,
    }


def _proposal_sidecar_reference_payload(
    receipt: V1RuntimeProposalStateSidecarReceipt,
) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "relative_path": receipt.relative_path,
        "payload_sha256": receipt.payload_sha256,
        "payload_size_bytes": receipt.payload_size_bytes,
        "framed_size_bytes": receipt.framed_size_bytes,
        "codec_id": receipt.codec_id,
        "representation_hash": receipt.representation_hash,
        "sufficient_state_hash": receipt.sufficient_state_hash,
        "storage_descriptor_hash": receipt.storage_descriptor_hash,
        "proposal_progress_hash": receipt.proposal_progress_hash,
        "campaign_contract_hash": receipt.campaign_contract_hash,
        "campaign_binding_hash": receipt.campaign_binding_hash,
        "controller_run_id": receipt.controller_run_id,
        "proposal_backend_identity_hash": receipt.proposal_backend_identity_hash,
        "controller_state_hash": receipt.controller_state_hash,
        "controller_snapshot_hash": receipt.controller_snapshot_hash,
        "current_mask_hash": receipt.current_mask_hash,
        "route_axis_hash": receipt.route_axis_hash,
        "acquisition_generation": receipt.acquisition_generation,
        "mask_lineage_hash": receipt.mask_lineage_hash,
        "transition_lineage_hash": receipt.transition_lineage_hash,
        "joint_state_hash": receipt.joint_state_hash,
        "phase_evidence_state_hash": receipt.phase_evidence_state_hash,
        "train_optimization_data_only": receipt.train_optimization_data_only,
        "validation_outcomes_observed": receipt.validation_outcomes_observed,
        "candidate_or_mask_authority": receipt.candidate_or_mask_authority,
        "receipt_hash": receipt.receipt_hash,
    }


def _proposal_storage_descriptor_hash(
    descriptor: V1RuntimeProposalSufficientStateDescriptor,
) -> str:
    return content_hash(
        descriptor.storage_descriptor,
        prefix="partition-free-v1-runtime-proposal-storage-descriptor-",
    )


def _proposal_transition_lineage_hash(
    transition_receipt_hashes: tuple[str, ...],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_transition_lineage_v1",
            "transition_receipt_hashes": transition_receipt_hashes,
        },
        prefix="partition-free-v1-runtime-proposal-transition-lineage-",
    )


def _proposal_sidecar_semantics_from_progress(
    progress: V1RuntimeProposalAcquisitionProgress,
    controller: V1ControllerState,
    *,
    controller_state_hash: str,
) -> dict[str, Any]:
    descriptor = progress.current_sufficient_state
    return {
        "codec_id": _PROPOSAL_STATE_CODEC_ID,
        "representation_hash": descriptor.representation_hash,
        "sufficient_state_hash": descriptor.sufficient_state_hash,
        "storage_descriptor_hash": _proposal_storage_descriptor_hash(descriptor),
        "proposal_progress_hash": progress.progress_hash,
        "campaign_contract_hash": progress.campaign_contract_hash,
        "campaign_binding_hash": progress.campaign_binding_hash,
        "controller_run_id": progress.controller_run_id,
        "proposal_backend_identity_hash": progress.proposal_backend_identity_hash,
        "controller_state_hash": controller_state_hash,
        "controller_snapshot_hash": (
            controller.snapshot_receipt().receipt_hash
        ),
        "current_mask_hash": progress.current_mask_hash,
        "route_axis_hash": progress.route_axis_hash,
        "acquisition_generation": progress.acquisition_generation_index,
        "mask_lineage_hash": progress.mask_lineage_hash,
        "transition_lineage_hash": _proposal_transition_lineage_hash(
            progress.transition_receipt_hashes
        ),
        "joint_state_hash": progress.joint_state.content_hash,
        "phase_evidence_state_hash": (
            progress.joint_state.phase_evidence_state_hash
        ),
        "train_optimization_data_only": True,
        "validation_outcomes_observed": False,
        "candidate_or_mask_authority": False,
    }


def _proposal_sidecar_semantics(
    state: V1RuntimeState,
    *,
    controller_state_hash: str,
) -> dict[str, Any]:
    progress = state.proposal_progress
    if progress is None:
        raise ValueError("proposal-state sidecar requires live proposal progress")
    return _proposal_sidecar_semantics_from_progress(
        progress,
        state.controller,
        controller_state_hash=controller_state_hash,
    )


def _proposal_sidecar_receipt(
    *,
    relative_path: str,
    payload_sha256: str,
    payload_size_bytes: int,
    framed_size_bytes: int,
    semantics: Mapping[str, Any],
) -> V1RuntimeProposalStateSidecarReceipt:
    logical: dict[str, Any] = {
        "schema": _PROPOSAL_STATE_SIDECAR_SCHEMA,
        "relative_path": relative_path,
        "payload_sha256": payload_sha256,
        "payload_size_bytes": payload_size_bytes,
        "framed_size_bytes": framed_size_bytes,
        **dict(semantics),
    }
    logical["receipt_hash"] = content_hash(
        logical,
        prefix="partition-free-v1-runtime-proposal-state-sidecar-",
    )
    return V1RuntimeProposalStateSidecarReceipt(**logical)


def _proposal_sidecar_receipt_from_header(
    *,
    header: Mapping[str, Any],
    relative_path: str,
    framed_size_bytes: int,
) -> V1RuntimeProposalStateSidecarReceipt:
    logical = dict(header)
    logical.pop("security_boundary", None)
    logical["relative_path"] = relative_path
    logical["framed_size_bytes"] = framed_size_bytes
    return V1RuntimeProposalStateSidecarReceipt(**logical)


def _read_framed_header(
    source: Path,
    *,
    expected_magic: bytes,
    label: str,
) -> tuple[dict[str, Any], int, int]:
    failure = f"{label} failed pre-unpickle integrity verification"
    with source.open("rb") as handle:
        magic = handle.read(len(expected_magic))
        size_bytes = handle.read(8)
        if magic != expected_magic or len(size_bytes) != 8:
            raise ValueError(f"{label} magic is invalid")
        header_size = struct.unpack(">Q", size_bytes)[0]
        if header_size <= 0 or header_size > _MAX_FRAME_HEADER_BYTES:
            raise ValueError(failure)
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(failure)
        try:
            raw_header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(failure) from exc
        if not isinstance(raw_header, dict):
            raise ValueError(failure)
        payload_size = raw_header.get("payload_size_bytes")
        payload_offset = len(expected_magic) + 8 + header_size
        file_size = os.fstat(handle.fileno()).st_size
        if (
            isinstance(payload_size, bool)
            or not isinstance(payload_size, int)
            or payload_size <= 0
            or file_size != payload_offset + payload_size
        ):
            raise ValueError(failure)
    return raw_header, payload_offset, payload_size


def _verified_payload_digest(
    source: Path,
    *,
    payload_offset: int,
    payload_size: int,
    expected_sha256: str,
    label: str,
) -> str:
    expected = _sha256_hex(expected_sha256, f"{label} payload_sha256")
    failure = f"{label} failed pre-unpickle integrity verification"
    with source.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mapped)[payload_offset : payload_offset + payload_size]
        try:
            observed = hashlib.sha256(view).hexdigest()
        finally:
            view.release()
            mapped.close()
    if observed != expected:
        raise ValueError(failure)
    return observed


def _load_verified_framed_payload(
    source: Path,
    *,
    expected_magic: bytes,
    expected_schema: str,
    label: str,
) -> tuple[Any, dict[str, Any], str, int]:
    header, payload_offset, payload_size = _read_framed_header(
        source,
        expected_magic=expected_magic,
        label=label,
    )
    failure = f"{label} failed pre-unpickle integrity verification"
    if header.get("schema") != expected_schema:
        raise ValueError(failure)
    expected_digest = header.get("payload_sha256")
    if type(expected_digest) is not str:
        raise ValueError(failure)
    with source.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        view = memoryview(mapped)[payload_offset : payload_offset + payload_size]
        try:
            digest = hashlib.sha256(view).hexdigest()
            if digest != expected_digest:
                raise ValueError(failure)
            value = cloudpickle.loads(view)
        finally:
            view.release()
            mapped.close()
    return value, header, digest, payload_size


def _acquisition_sidecar_semantics(
    acquisition: V1RuntimeAcquisition,
) -> dict[str, Any]:
    return {
        "acquisition_hash": acquisition.receipt.acquisition_hash,
        "acquisition_receipt_hash": acquisition.receipt.receipt_hash,
        "campaign_binding_hash": acquisition.binding.binding_hash,
        "generation_index": acquisition.generation_index,
        "current_mask_hash": acquisition.current_mask.mask_hash,
        "route_axis_hash": acquisition.current_mask.route_axis_hash,
    }


def _assert_sidecar_matches_acquisition(
    receipt: V1RuntimeAcquisitionSidecarReceipt,
    acquisition: V1RuntimeAcquisition,
) -> None:
    expected = _acquisition_sidecar_semantics(acquisition)
    if any(getattr(receipt, name) != value for name, value in expected.items()):
        raise ValueError("acquisition sidecar differs from the live acquisition")


def _sidecar_receipt_from_header(
    *,
    header: Mapping[str, Any],
    relative_path: str,
    framed_size_bytes: int,
) -> V1RuntimeAcquisitionSidecarReceipt:
    return V1RuntimeAcquisitionSidecarReceipt(
        relative_path=relative_path,
        payload_sha256=header.get("payload_sha256"),
        payload_size_bytes=header.get("payload_size_bytes"),
        framed_size_bytes=framed_size_bytes,
        acquisition_hash=header.get("acquisition_hash"),
        acquisition_receipt_hash=header.get("acquisition_receipt_hash"),
        campaign_binding_hash=header.get("campaign_binding_hash"),
        generation_index=header.get("generation_index"),
        current_mask_hash=header.get("current_mask_hash"),
        route_axis_hash=header.get("route_axis_hash"),
        schema=header.get("schema"),
    )


def _write_acquisition_sidecar(
    *,
    destination: Path,
    relative_path: str,
    acquisition: V1RuntimeAcquisition,
) -> V1RuntimeAcquisitionSidecarReceipt:
    if acquisition.acquisition_hash != acquisition.receipt.acquisition_hash:
        raise ValueError("live acquisition differs from its sealed acquisition receipt")
    temporary = destination.with_name(destination.name + ".tmp")
    semantics = _acquisition_sidecar_semantics(acquisition)
    try:
        with temporary.open("w+b") as handle:
            handle.write(_ACQUISITION_SIDECAR_MAGIC)
            handle.write(struct.pack(">Q", _ACQUISITION_SIDECAR_HEADER_BYTES))
            handle.write(b" " * _ACQUISITION_SIDECAR_HEADER_BYTES)
            writer = _DigestingWriter(handle)
            if type(acquisition) is V1OperatorAcquisition:
                from .modern_v1_operator_checkpoint import export_operator_acquisition_payload
                cloudpickle.dump(export_operator_acquisition_payload(acquisition), writer, protocol=5)
            else:
                cloudpickle.dump(acquisition, writer, protocol=5)
            header = {
                "schema": _ACQUISITION_SIDECAR_SCHEMA,
                "payload_sha256": writer.digest.hexdigest(),
                "payload_size_bytes": writer.size,
                **semantics,
                "security_boundary": "trusted_local_only_do_not_unpickle_untrusted_bytes",
            }
            header_bytes = json.dumps(
                header,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(header_bytes) > _ACQUISITION_SIDECAR_HEADER_BYTES:
                raise ValueError(
                    "acquisition sidecar header exceeds its fixed framing bound"
                )
            handle.seek(len(_ACQUISITION_SIDECAR_MAGIC) + 8)
            handle.write(header_bytes)
            handle.write(
                b" " * (_ACQUISITION_SIDECAR_HEADER_BYTES - len(header_bytes))
            )
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    framed_size = (
        len(_ACQUISITION_SIDECAR_MAGIC)
        + 8
        + _ACQUISITION_SIDECAR_HEADER_BYTES
        + writer.size
    )
    if temporary.stat().st_size != framed_size:
        raise RuntimeError("acquisition sidecar framing changed during serialization")
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    return V1RuntimeAcquisitionSidecarReceipt(
        relative_path=relative_path,
        payload_sha256=writer.digest.hexdigest(),
        payload_size_bytes=writer.size,
        framed_size_bytes=framed_size,
        **semantics,
    )


def _ensure_acquisition_sidecar(
    checkpoint_path: Path,
    acquisition: V1RuntimeAcquisition,
) -> V1RuntimeAcquisitionSidecarReceipt:
    if type(acquisition) not in _RUNTIME_ACQUISITION_TYPES:
        raise TypeError("checkpoint sidecar requires a supported acquisition")
    directory_name = checkpoint_path.name + ".acquisitions"
    directory = checkpoint_path.parent / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        hashlib.sha256(acquisition.receipt.acquisition_hash.encode("utf-8")).hexdigest()
        + ".mv1acq"
    )
    relative_path = (Path(directory_name) / filename).as_posix()
    destination = directory / filename
    cache_key = str(destination.resolve())
    cached = _SIDECAR_CACHE.get(cache_key)
    if destination.is_file():
        header, payload_offset, payload_size = _read_framed_header(
            destination,
            expected_magic=_ACQUISITION_SIDECAR_MAGIC,
            label="modern-v1 acquisition sidecar",
        )
        observed = _sidecar_receipt_from_header(
            header=header,
            relative_path=relative_path,
            framed_size_bytes=destination.stat().st_size,
        )
        _assert_sidecar_matches_acquisition(observed, acquisition)
        signature = _stat_signature(destination)
        if cached is not None and cached.receipt == observed and cached.stat_signature == signature:
            return observed
        _verified_payload_digest(
            destination,
            payload_offset=payload_offset,
            payload_size=payload_size,
            expected_sha256=observed.payload_sha256,
            label="modern-v1 acquisition sidecar",
        )
        _SIDECAR_CACHE[cache_key] = _SidecarCacheEntry(observed, signature)
        return observed
    receipt = _write_acquisition_sidecar(
        destination=destination,
        relative_path=relative_path,
        acquisition=acquisition,
    )
    _SIDECAR_CACHE[cache_key] = _SidecarCacheEntry(
        receipt,
        _stat_signature(destination),
    )
    return receipt


def _proposal_state_payload(
    state: V1RuntimeState,
) -> V1RuntimeProposalStateSidecarPayload:
    progress = state.proposal_progress
    if progress is None:
        raise ValueError("proposal-state payload requires live proposal progress")
    pending = state.pending_proposal_acquisition
    return V1RuntimeProposalStateSidecarPayload(
        proposal_progress_payload=progress,
        pending_proposal_acquisition_payload=pending,
        proposal_progress_hash=progress.progress_hash,
        pending_proposal_acquisition_hash=(
            None if pending is None else pending.pending_hash
        ),
        proposal_backend_identity_hash=progress.proposal_backend_identity_hash,
        representation_hash=progress.current_sufficient_state.representation_hash,
    )


def _revalidate_pending_proposal_graph(
    pending: V1RuntimePendingProposalAcquisition,
) -> None:
    pending.__post_init__()
    for forecast in pending.forecasts:
        forecast.__post_init__()
    pending.scheduler_receipt.__post_init__()
    pending.request_plan.__post_init__()
    pending.request.__post_init__()
    pending.__post_init__()


def _revalidate_proposal_progress_graph(
    progress: V1RuntimeProposalAcquisitionProgress,
) -> None:
    progress.__post_init__()
    progress.joint_state.__post_init__()
    progress.initial_sufficient_state.__post_init__()
    progress.current_sufficient_state.__post_init__()
    for update in progress.update_receipts:
        update.__post_init__()
    for step in progress.replay_steps:
        step.__post_init__()
    for completed in progress.completed_pending_requests:
        _revalidate_pending_proposal_graph(completed)
    for outcome in progress.outcome_receipts:
        outcome.updated_sufficient_state.__post_init__()
        outcome.__post_init__()
    for evaluation in progress.freeze_evaluation_receipts:
        evaluation.__post_init__()
    for evidence in progress.freeze_evidence_receipts:
        evidence.__post_init__()
    for decision in progress.freeze_decision_receipts:
        decision.__post_init__()
    if progress.terminal_refusal_receipt is not None:
        progress.terminal_refusal_receipt.__post_init__()
    for forecast in progress.terminal_refusal_forecasts:
        forecast.__post_init__()
    progress.__post_init__()
    progress.verified_evidence_state


def _validate_live_proposal_sidecar_payload(
    value: Any,
    *,
    expected_receipt: V1RuntimeProposalStateSidecarReceipt,
) -> tuple[
    V1RuntimeProposalAcquisitionProgress,
    V1RuntimePendingProposalAcquisition | None,
]:
    if type(value) is not V1RuntimeProposalStateSidecarPayload:
        raise TypeError("proposal-state sidecar payload has the wrong type")
    value.__post_init__()
    progress = value.proposal_progress_payload
    pending = value.pending_proposal_acquisition_payload
    if type(progress) is not V1RuntimeProposalAcquisitionProgress:
        raise TypeError("proposal-state sidecar progress has the wrong type")
    _revalidate_proposal_progress_graph(progress)
    if pending is not None:
        if type(pending) is not V1RuntimePendingProposalAcquisition:
            raise TypeError("proposal-state sidecar pending request has the wrong type")
        _revalidate_pending_proposal_graph(pending)
        _assert_pending_proposal_bound(progress, pending)
    pending_hash = None if pending is None else pending.pending_hash
    descriptor = progress.current_sufficient_state
    if (
        value.proposal_progress_hash != progress.progress_hash
        or value.pending_proposal_acquisition_hash != pending_hash
        or value.proposal_backend_identity_hash
        != progress.proposal_backend_identity_hash
        or value.representation_hash != descriptor.representation_hash
        or expected_receipt.proposal_progress_hash != progress.progress_hash
        or expected_receipt.proposal_backend_identity_hash
        != progress.proposal_backend_identity_hash
        or expected_receipt.representation_hash != descriptor.representation_hash
        or expected_receipt.sufficient_state_hash != descriptor.sufficient_state_hash
        or expected_receipt.storage_descriptor_hash
        != _proposal_storage_descriptor_hash(descriptor)
    ):
        raise ValueError("proposal-state payload differs from its exact framed receipt")
    return progress, pending


def _write_proposal_state_sidecar(
    *,
    destination: Path,
    relative_path: str,
    payload: V1RuntimeProposalStateSidecarPayload,
    semantics: Mapping[str, Any],
) -> V1RuntimeProposalStateSidecarReceipt:
    temporary = destination.with_name(destination.name + ".tmp")
    writer: _DigestingWriter | None = None
    try:
        with temporary.open("w+b") as handle:
            handle.write(_PROPOSAL_STATE_SIDECAR_MAGIC)
            handle.write(struct.pack(">Q", _PROPOSAL_STATE_SIDECAR_HEADER_BYTES))
            handle.write(b" " * _PROPOSAL_STATE_SIDECAR_HEADER_BYTES)
            writer = _DigestingWriter(handle)
            cloudpickle.dump(payload, writer, protocol=5)
            framed_size = (
                len(_PROPOSAL_STATE_SIDECAR_MAGIC)
                + 8
                + _PROPOSAL_STATE_SIDECAR_HEADER_BYTES
                + writer.size
            )
            receipt = _proposal_sidecar_receipt(
                relative_path=relative_path,
                payload_sha256=writer.digest.hexdigest(),
                payload_size_bytes=writer.size,
                framed_size_bytes=framed_size,
                semantics=semantics,
            )
            header = _proposal_sidecar_reference_payload(receipt)
            header.pop("relative_path")
            header.pop("framed_size_bytes")
            header["security_boundary"] = _CHECKPOINT_SECURITY_BOUNDARY
            header_bytes = json.dumps(
                header,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if len(header_bytes) > _PROPOSAL_STATE_SIDECAR_HEADER_BYTES:
                raise ValueError(
                    "proposal-state sidecar header exceeds its fixed framing bound"
                )
            handle.seek(len(_PROPOSAL_STATE_SIDECAR_MAGIC) + 8)
            handle.write(header_bytes)
            handle.write(
                b" "
                * (_PROPOSAL_STATE_SIDECAR_HEADER_BYTES - len(header_bytes))
            )
            handle.flush()
            os.fsync(handle.fileno())
        if writer is None or temporary.stat().st_size != receipt.framed_size_bytes:
            raise RuntimeError("proposal-state sidecar framing changed during serialization")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return receipt
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_proposal_state_sidecar(
    checkpoint_path: Path,
    state: V1RuntimeState,
    *,
    controller_state_hash: str,
) -> V1RuntimeProposalStateSidecarReceipt:
    progress = state.proposal_progress
    if progress is None:
        raise ValueError("checkpoint proposal sidecar requires live progress")
    # Re-run every live authority and graph invariant before serializing any
    # resumable bytes.  Trusted local is an origin boundary, not validation.
    _revalidate_proposal_progress_graph(progress)
    if state.pending_proposal_acquisition is not None:
        _revalidate_pending_proposal_graph(state.pending_proposal_acquisition)
    state.__post_init__()
    payload = _proposal_state_payload(state)
    semantics = _proposal_sidecar_semantics(
        state,
        controller_state_hash=controller_state_hash,
    )
    pending_hash = payload.pending_proposal_acquisition_hash
    address = content_hash(
        {
            "schema": _PROPOSAL_STATE_SIDECAR_SCHEMA,
            "proposal_progress_hash": payload.proposal_progress_hash,
            "pending_proposal_acquisition_hash": pending_hash,
            "semantics": semantics,
        }
    )
    directory_name = checkpoint_path.name + ".proposal_states"
    directory = checkpoint_path.parent / directory_name
    if directory.is_symlink():
        raise ValueError("checkpoint proposal-state directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint proposal-state directory is not a regular directory")
    filename = address + ".mv1prop"
    relative_path = (Path(directory_name) / filename).as_posix()
    destination = directory / filename
    if destination.is_symlink():
        raise ValueError("checkpoint proposal-state sidecar must not be a symlink")
    cache_key = str(destination.resolve())
    cached = _PROPOSAL_SIDECAR_CACHE.get(cache_key)
    if destination.is_file():
        signature = _stat_signature(destination)
        if cached is not None and cached.stat_signature == signature:
            observed_semantics = {
                name: getattr(cached.receipt, name) for name in semantics
            }
            if (
                cached.receipt.relative_path == relative_path
                and observed_semantics == dict(semantics)
            ):
                return cached.receipt
        header, _payload_offset, _payload_size = _read_framed_header(
            destination,
            expected_magic=_PROPOSAL_STATE_SIDECAR_MAGIC,
            label="modern-v1 proposal-state sidecar",
        )
        observed = _proposal_sidecar_receipt_from_header(
            header=header,
            relative_path=relative_path,
            framed_size_bytes=destination.stat().st_size,
        )
        observed_semantics = {name: getattr(observed, name) for name in semantics}
        if observed_semantics != dict(semantics):
            raise ValueError("proposal-state sidecar differs from live proposal semantics")
        value, loaded_header, digest, payload_size = _load_verified_framed_payload(
            destination,
            expected_magic=_PROPOSAL_STATE_SIDECAR_MAGIC,
            expected_schema=_PROPOSAL_STATE_SIDECAR_SCHEMA,
            label="modern-v1 proposal-state sidecar",
        )
        loaded = _proposal_sidecar_receipt_from_header(
            header=loaded_header,
            relative_path=relative_path,
            framed_size_bytes=destination.stat().st_size,
        )
        if (
            loaded != observed
            or digest != observed.payload_sha256
            or payload_size != observed.payload_size_bytes
        ):
            raise ValueError("proposal-state sidecar changed during verified reuse")
        loaded_progress, loaded_pending = _validate_live_proposal_sidecar_payload(
            value,
            expected_receipt=observed,
        )
        if (
            loaded_progress.progress_hash != payload.proposal_progress_hash
            or (None if loaded_pending is None else loaded_pending.pending_hash)
            != pending_hash
        ):
            raise ValueError("existing proposal-state sidecar has another live payload")
        _PROPOSAL_SIDECAR_CACHE[cache_key] = _ProposalSidecarCacheEntry(
            observed,
            signature,
        )
        return observed
    receipt = _write_proposal_state_sidecar(
        destination=destination,
        relative_path=relative_path,
        payload=payload,
        semantics=semantics,
    )
    _PROPOSAL_SIDECAR_CACHE[cache_key] = _ProposalSidecarCacheEntry(
        receipt,
        _stat_signature(destination),
    )
    return receipt


def _prune_unreferenced_acquisition_sidecars(
    checkpoint_path: Path,
    *,
    referenced: V1RuntimeAcquisitionSidecarReceipt | None,
) -> None:
    """Remove acquisition generations no longer named by the durable journal.

    A fresh acquisition can be many gigabytes at ImageNet scale.  Its sidecar
    must therefore have checkpoint lifetime, rather than trajectory lifetime.
    This cleanup runs only *after* the replacement journal is durably
    published: a crash before publication leaves the prior referenced sidecar
    intact, while a crash after publication can at worst leave garbage for the
    next checkpoint write to collect.
    """

    directory = checkpoint_path.parent / (checkpoint_path.name + ".acquisitions")
    if directory.is_symlink():
        raise ValueError("checkpoint acquisition-sidecar directory must not be a symlink")
    if not directory.is_dir():
        return
    keep = None
    if referenced is not None:
        keep = _resolved_sidecar_path(checkpoint_path, referenced)
    removed = False
    for candidate in directory.glob("*.mv1acq"):
        if keep is not None and candidate.resolve() == keep:
            continue
        candidate.unlink(missing_ok=True)
        _SIDECAR_CACHE.pop(str(candidate.resolve()), None)
        removed = True
    if removed:
        _fsync_directory(directory)


def _resolved_proposal_sidecar_path(
    checkpoint_path: Path,
    receipt: V1RuntimeProposalStateSidecarReceipt,
) -> Path:
    relative = Path(receipt.relative_path)
    expected_name = checkpoint_path.name + ".proposal_states"
    if (
        relative.is_absolute()
        or relative.as_posix() != receipt.relative_path
        or len(relative.parts) != 2
        or relative.parts[0] != expected_name
        or relative.parts[1] in {"", ".", ".."}
        or not relative.parts[1].endswith(".mv1prop")
    ):
        raise ValueError("checkpoint proposal-state sidecar leaves its owned directory")
    directory = checkpoint_path.parent / expected_name
    source = directory / relative.parts[1]
    if directory.is_symlink() or source.is_symlink():
        raise ValueError("checkpoint proposal-state sidecar path is symlinked")
    resolved_directory = directory.resolve()
    resolved = source.resolve()
    if resolved.parent != resolved_directory or resolved.name != relative.parts[1]:
        raise ValueError("checkpoint proposal-state sidecar leaves its owned directory")
    return resolved


def _prune_unreferenced_proposal_state_sidecars(
    checkpoint_path: Path,
    *,
    referenced: V1RuntimeProposalStateSidecarReceipt | None,
) -> None:
    """Collect proposal states only after their replacement journal is durable."""

    directory = checkpoint_path.parent / (checkpoint_path.name + ".proposal_states")
    if directory.is_symlink():
        raise ValueError("checkpoint proposal-state directory must not be a symlink")
    if not directory.is_dir():
        return
    keep = (
        None
        if referenced is None
        else _resolved_proposal_sidecar_path(checkpoint_path, referenced)
    )
    removed = False
    for candidate in directory.glob("*.mv1prop"):
        if candidate.is_symlink():
            candidate.unlink()
            removed = True
            continue
        if keep is not None and candidate.resolve() == keep:
            continue
        candidate.unlink(missing_ok=True)
        _PROPOSAL_SIDECAR_CACHE.pop(str(candidate.resolve()), None)
        removed = True
    if removed:
        _fsync_directory(directory)


def _checkpoint_controller_state_hash(controller: V1ControllerState) -> str:
    """Reproduce ``state_hash`` from the already-sealed acquisition receipt.

    ``V1RelinearizedAcquisition.__post_init__`` binds that receipt to the exact
    acquisition hash.  Reusing it here avoids rehashing every route-wide array
    on each small journal rewrite.
    """

    acquisition = controller.acquisition
    projection: dict[str, Any] = {
        "schema": controller.schema,
        "campaign_binding_hash": controller.binding.binding_hash,
        "controller_run_id": controller.controller_run_id,
        "initial_mask_hash": controller.initial_mask.mask_hash,
        "current_mask_hash": controller.current_mask.mask_hash,
        "phase": controller.phase,
        "acquisition_hash": (
            None if acquisition is None else acquisition.receipt.acquisition_hash
        ),
        "queue_cursor": controller.queue_cursor,
        "active_candidate_hash": (
            None
            if controller.active_candidate is None
            else controller.active_candidate.active_hash
        ),
        "accepted_trajectory_hash": (
            controller.accepted_trajectory_hash
        ),
        "attempt_receipt_hashes": controller.attempt_receipt_hashes,
        "transition_receipt_hashes": controller.transition_receipt_hashes,
        "acquisition_receipt_hashes": controller.acquisition_receipt_hashes,
        "globally_used_draw_ids": controller.globally_used_draw_ids,
        "stop_reason": controller.stop_reason,
        "logical_sequence": controller.logical_sequence,
    }
    if controller.label_native_runtime_plan_hashes:
        projection.update(
            {
                "label_native_runtime_plan_hash": (
                    None
                    if controller.label_native_runtime_plan is None
                    else controller.label_native_runtime_plan.plan_hash
                ),
                "label_native_runtime_plan_hashes": (
                    controller.label_native_runtime_plan_hashes
                ),
                "label_native_rejected_plan_indices": (
                    controller.label_native_rejected_plan_indices
                ),
                "label_native_selected_acquisition_receipt_hash": (
                    acquisition.receipt.receipt_hash
                    if _is_label_native_acquisition(acquisition)
                    else None
                ),
            }
        )
    if controller.pareto_milestone_registration is not None:
        projection.update(
            {
                "pareto_milestone_registration_hash": (
                    controller.pareto_milestone_registration.registration_hash
                ),
                "live_exact_prefix_rung_registration_hash": (
                    None
                    if controller.live_exact_prefix_rung_registration is None
                    else controller.live_exact_prefix_rung_registration.registration_hash
                ),
                "pending_pareto_milestone_hash": (
                    None
                    if controller.pending_pareto_milestone is None
                    else controller.pending_pareto_milestone.pending_hash
                ),
                "pareto_milestone_receipt_hash": (
                    None
                    if controller.pareto_milestone_receipt is None
                    else controller.pareto_milestone_receipt.receipt_hash
                ),
            }
        )
    if controller.fqr_no_candidate_horizon_witness is not None:
        projection["fqr_no_candidate_horizon_witness_hash"] = (
            controller.fqr_no_candidate_horizon_witness.witness_hash
        )
    return content_hash(
        projection,
        prefix="partition-free-v1-controller-state-",
    )


def _checkpoint_runtime_state_hash(
    state: V1RuntimeState,
    *,
    controller_state_hash: str,
) -> str:
    projection: dict[str, Any] = {
        "schema": state.schema,
        "logical_config_hash": state.config.logical_identity_hash,
        "entropy_tape_hash": state.entropy_tape.tape_hash,
        "controller_state_hash": controller_state_hash,
        "pending_hash": (
            None if state.pending_look is None else state.pending_look.pending_hash
        ),
        "terminal_freeze_hash": (
            None
            if state.terminal_freeze is None
            else state.terminal_freeze.receipt_hash
        ),
        "terminal_validation_hash": (
            None
            if state.terminal_validation is None
            else state.terminal_validation.receipt_hash
        ),
        "completion_hash": (
            None if state.completion is None else state.completion.receipt_hash
        ),
        "performance_hash": (
            None if state.performance is None else state.performance.receipt_hash
        ),
    }
    completed_hashes = tuple(
        item.receipt_hash for item in state.completed_proposal_closures
    )
    if state.proposal_progress is not None or completed_hashes:
        projection.update(
            {
                "proposal_progress_hash": (
                    None
                    if state.proposal_progress is None
                    else state.proposal_progress.progress_hash
                ),
                "pending_proposal_acquisition_hash": (
                    None
                    if state.pending_proposal_acquisition is None
                    else state.pending_proposal_acquisition.pending_hash
                ),
                "completed_proposal_closure_hashes": completed_hashes,
                "staged_proposal_backend_identity_hash": (
                    state.staged_proposal_backend_identity_hash
                ),
            }
        )
    if state.pending_paired_bce_chunk is not None:
        projection["pending_paired_bce_chunk_hash"] = (
            state.pending_paired_bce_chunk.pending_hash
        )
    if any(
        item is not None
        for item in (
            state.terminal_validation_family_freeze,
            state.terminal_validation_family_request,
            state.terminal_validation_family_observation,
            state.terminal_validation_family,
        )
    ):
        projection.update(
            {
                "terminal_validation_family_freeze_hash": (
                    None
                    if state.terminal_validation_family_freeze is None
                    else state.terminal_validation_family_freeze.receipt_hash
                ),
                "terminal_validation_family_request_hash": (
                    None
                    if state.terminal_validation_family_request is None
                    else state.terminal_validation_family_request.request_hash
                ),
                "terminal_validation_family_observation_hash": (
                    None
                    if state.terminal_validation_family_observation is None
                    else state.terminal_validation_family_observation.terminal_outcome_hash
                ),
                "terminal_validation_family_hash": (
                    None
                    if state.terminal_validation_family is None
                    else state.terminal_validation_family.receipt_hash
                ),
            }
        )
    if any(
        item is not None
        for item in (
            state.terminal_comparison_family_freeze,
            state.terminal_comparison_family_request,
            state.terminal_comparison_family_observation,
            state.terminal_comparison_family,
        )
    ):
        projection.update(
            {
                "terminal_comparison_family_freeze_hash": (
                    None
                    if state.terminal_comparison_family_freeze is None
                    else state.terminal_comparison_family_freeze.receipt_hash
                ),
                "terminal_comparison_family_request_hash": (
                    None
                    if state.terminal_comparison_family_request is None
                    else state.terminal_comparison_family_request.request_hash
                ),
                "terminal_comparison_family_observation_hash": (
                    None
                    if state.terminal_comparison_family_observation is None
                    else state.terminal_comparison_family_observation.terminal_outcome_hash
                ),
                "terminal_comparison_family_hash": (
                    None
                    if state.terminal_comparison_family is None
                    else state.terminal_comparison_family.receipt_hash
                ),
            }
        )
    return content_hash(
        projection,
        prefix="partition-free-v1-runtime-state-",
    )


def _project_runtime_checkpoint(
    state: V1RuntimeState,
    *,
    checkpoint_path: Path,
) -> _V1RuntimeCheckpointJournal:
    if state.proposal_progress is not None:
        _revalidate_proposal_progress_graph(state.proposal_progress)
        if state.pending_proposal_acquisition is not None:
            _revalidate_pending_proposal_graph(state.pending_proposal_acquisition)
    if (
        state.proposal_progress is not None
        or state.completed_proposal_closures
        or state.terminal_validation_family_freeze is not None
        or state.terminal_comparison_family_freeze is not None
    ):
        state.__post_init__()
    controller = state.controller
    from .modern_v1_operator_run_binding import validate_history_binding
    validate_history_binding(state.config.metadata,state.config.source_seal_hash,controller.acquisition_receipt_hashes)
    if type(controller.acquisition) is V1OperatorAcquisition:
        from .modern_v1_operator_run_binding import validate_bound_operator_sources
        acquisition=controller.acquisition
        proposal=(acquisition._no_candidate_proposal if acquisition.candidate is None
                  else acquisition.candidate._producer.proposal)
        validate_bound_operator_sources(state.config,proposal.discovery.sources)
    if _is_label_native_acquisition(controller.acquisition):
        raise ValueError(
            "a live label-native selected endpoint has no generic acquisition "
            "sidecar codec; checkpoint after its controller decision"
        )
    controller_state_hash = _checkpoint_controller_state_hash(controller)
    runtime_state_hash = _checkpoint_runtime_state_hash(
        state,
        controller_state_hash=controller_state_hash,
    )
    acquisition_sidecar = (
        None
        if controller.acquisition is None
        else _ensure_acquisition_sidecar(checkpoint_path, controller.acquisition)
    )
    proposal_state_sidecar = (
        None
        if state.proposal_progress is None
        else _ensure_proposal_state_sidecar(
            checkpoint_path,
            state,
            controller_state_hash=controller_state_hash,
        )
    )
    from .modern_v1_operator_checkpoint import project_operator_active_candidate
    projection = _V1ControllerCheckpointProjection(
        binding=controller.binding,
        controller_run_id=controller.controller_run_id,
        initial_mask=controller.initial_mask,
        current_mask=controller.current_mask,
        phase=controller.phase,
        queue_cursor=controller.queue_cursor,
        active_candidate=project_operator_active_candidate(controller.active_candidate),
        accepted_trajectory=controller.accepted_trajectory,
        accepted_oracle_trajectory=controller.accepted_oracle_trajectory,
        attempt_receipts=controller.attempt_receipts,
        transition_receipts=controller.transition_receipts,
        acquisition_receipt_hashes=controller.acquisition_receipt_hashes,
        globally_used_draw_ids=controller.globally_used_draw_ids,
        stop_reason=controller.stop_reason,
        logical_sequence=controller.logical_sequence,
        controller_schema=controller.schema,
        acquisition_sidecar=acquisition_sidecar,
        controller_state_hash=controller_state_hash,
        controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
        pareto_milestone_registration=controller.pareto_milestone_registration,
        live_exact_prefix_rung_registration=(
            controller.live_exact_prefix_rung_registration
        ),
        pending_pareto_milestone=controller.pending_pareto_milestone,
        pareto_milestone_receipt=controller.pareto_milestone_receipt,
        fqr_no_candidate_horizon_witness=(
            controller.fqr_no_candidate_horizon_witness
        ),
        label_native_runtime_plan_checkpoint_payload=(
            None
            if controller.label_native_runtime_plan is None
            else export_v1_label_native_runtime_plan_checkpoint_payload(
                controller.label_native_runtime_plan
            )
        ),
        label_native_runtime_plan_hash=(
            None
            if controller.label_native_runtime_plan is None
            else controller.label_native_runtime_plan.plan_hash
        ),
        label_native_runtime_plan_hashes=(
            controller.label_native_runtime_plan_hashes
        ),
        label_native_rejected_plan_indices=(
            controller.label_native_rejected_plan_indices
        ),
        label_native_selected_acquisition_receipt_hash=(
            controller.acquisition.receipt.receipt_hash
            if _is_label_native_acquisition(controller.acquisition)
            else None
        ),
    )
    return _V1RuntimeCheckpointJournal(
        config=state.config,
        controller=projection,
        entropy_tape=state.entropy_tape,
        pending_look=state.pending_look,
        terminal_freeze=state.terminal_freeze,
        terminal_validation=state.terminal_validation,
        completion=state.completion,
        performance=state.performance,
        runtime_schema=state.schema,
        runtime_state_hash=runtime_state_hash,
        proposal_state_sidecar=proposal_state_sidecar,
        staged_proposal_backend_identity_hash=(
            state.staged_proposal_backend_identity_hash
        ),
        completed_proposal_closures=(
            state.completed_proposal_closures
        ),
        pending_paired_bce_chunk=state.pending_paired_bce_chunk,
        terminal_validation_family_freeze=(
            state.terminal_validation_family_freeze
        ),
        terminal_validation_family_request=(
            state.terminal_validation_family_request
        ),
        terminal_validation_family_observation=(
            state.terminal_validation_family_observation
        ),
        terminal_validation_family=state.terminal_validation_family,
        terminal_comparison_family_freeze=(
            state.terminal_comparison_family_freeze
        ),
        terminal_comparison_family_request=(
            state.terminal_comparison_family_request
        ),
        terminal_comparison_family_observation=(
            state.terminal_comparison_family_observation
        ),
        terminal_comparison_family=state.terminal_comparison_family,
        schema=(
            _CHECKPOINT_JOURNAL_V7_SCHEMA
            if state.terminal_comparison_family_freeze is not None
            else (
                _CHECKPOINT_JOURNAL_V6_SCHEMA
                if state.terminal_validation_family_freeze is not None
                else (
                    _CHECKPOINT_JOURNAL_SCHEMA
                    if (
                        proposal_state_sidecar is None
                        and not state.completed_proposal_closures
                        and state.pending_paired_bce_chunk is None
                    )
                    else (
                        _CHECKPOINT_JOURNAL_V5_SCHEMA
                        if state.pending_paired_bce_chunk is not None
                        else _CHECKPOINT_JOURNAL_V4_SCHEMA
                    )
                )
            )
        ),
    )


def _resolved_sidecar_path(
    checkpoint_path: Path,
    receipt: V1RuntimeAcquisitionSidecarReceipt,
) -> Path:
    relative = Path(receipt.relative_path)
    expected_directory = (
        checkpoint_path.parent / (checkpoint_path.name + ".acquisitions")
    ).resolve()
    resolved = (checkpoint_path.parent / relative).resolve()
    if resolved.parent != expected_directory or resolved.name != relative.name:
        raise ValueError("checkpoint acquisition sidecar leaves its output directory")
    return resolved


def _assert_sidecar_matches_projection(
    receipt: V1RuntimeAcquisitionSidecarReceipt,
    projection: _V1ControllerCheckpointProjection,
) -> None:
    binding_hash = projection.binding.binding_hash
    current_mask_hash = projection.current_mask.mask_hash
    route_axis_hash = projection.current_mask.route_axis_hash
    acquisition_generation = len(projection.transition_receipts)
    expected_receipt_hash = (
        None
        if not projection.acquisition_receipt_hashes
        else projection.acquisition_receipt_hashes[-1]
    )
    if (
        receipt.campaign_binding_hash != binding_hash
        or receipt.current_mask_hash != current_mask_hash
        or receipt.route_axis_hash != route_axis_hash
        or receipt.generation_index != acquisition_generation
        or receipt.acquisition_receipt_hash != expected_receipt_hash
    ):
        raise ValueError("checkpoint acquisition sidecar has stale semantic lineage")


def _read_acquisition_sidecar(
    checkpoint_path: Path,
    expected: V1RuntimeAcquisitionSidecarReceipt,
    projection: _V1ControllerCheckpointProjection,
    *, operator_sources=None,
) -> V1RuntimeAcquisition:
    _assert_sidecar_matches_projection(expected, projection)
    source = _resolved_sidecar_path(checkpoint_path, expected)
    if not source.is_file() or source.stat().st_size != expected.framed_size_bytes:
        raise ValueError(
            "modern-v1 acquisition sidecar failed pre-unpickle integrity verification"
        )
    pre_header, _payload_offset, _payload_size = _read_framed_header(
        source,
        expected_magic=_ACQUISITION_SIDECAR_MAGIC,
        label="modern-v1 acquisition sidecar",
    )
    pre_observed = _sidecar_receipt_from_header(
        header=pre_header,
        relative_path=expected.relative_path,
        framed_size_bytes=source.stat().st_size,
    )
    if pre_observed != expected:
        raise ValueError("checkpoint acquisition sidecar differs from its journal reference")
    value, header, digest, payload_size = _load_verified_framed_payload(
        source,
        expected_magic=_ACQUISITION_SIDECAR_MAGIC,
        expected_schema=_ACQUISITION_SIDECAR_SCHEMA,
        label="modern-v1 acquisition sidecar",
    )
    observed = _sidecar_receipt_from_header(
        header=header,
        relative_path=expected.relative_path,
        framed_size_bytes=source.stat().st_size,
    )
    if (
        observed != expected
        or digest != expected.payload_sha256
        or payload_size != expected.payload_size_bytes
    ):
        raise ValueError("checkpoint acquisition sidecar differs from its journal reference")
    if type(value) is bytes:
        if operator_sources is None:
            raise ValueError('operator checkpoint requires freshly verified execution sources')
        from .modern_v1_operator_checkpoint import restore_operator_acquisition_payload
        value=restore_operator_acquisition_payload(value,
            expected_payload_sha256=hashlib.sha256(value).hexdigest(),
            expected_acquisition_hash=expected.acquisition_hash,sources=operator_sources,
            binding=projection.binding,current_mask=projection.current_mask,
            initial_mask=projection.initial_mask,trusted_local=True)
    if type(value) not in _RUNTIME_ACQUISITION_TYPES:
        raise TypeError("acquisition sidecar payload has the wrong type")
    if type(value) is V1OperatorAcquisition:
        value.assert_integrity()
    else:
        value.__post_init__()
    _assert_sidecar_matches_acquisition(expected, value)
    _SIDECAR_CACHE[str(source.resolve())] = _SidecarCacheEntry(
        expected,
        _stat_signature(source),
    )
    return value


def _read_proposal_state_sidecar(
    checkpoint_path: Path,
    expected: V1RuntimeProposalStateSidecarReceipt,
    controller: V1ControllerState,
) -> tuple[
    V1RuntimeProposalAcquisitionProgress,
    V1RuntimePendingProposalAcquisition | None,
]:
    expected.__post_init__()
    source = _resolved_proposal_sidecar_path(checkpoint_path, expected)
    if not source.is_file() or source.stat().st_size != expected.framed_size_bytes:
        raise ValueError(
            "modern-v1 proposal-state sidecar failed pre-unpickle integrity verification"
        )
    pre_header, _payload_offset, _payload_size = _read_framed_header(
        source,
        expected_magic=_PROPOSAL_STATE_SIDECAR_MAGIC,
        label="modern-v1 proposal-state sidecar",
    )
    pre_observed = _proposal_sidecar_receipt_from_header(
        header=pre_header,
        relative_path=expected.relative_path,
        framed_size_bytes=source.stat().st_size,
    )
    if pre_observed != expected:
        raise ValueError(
            "checkpoint proposal-state sidecar differs from its journal reference"
        )
    value, header, digest, payload_size = _load_verified_framed_payload(
        source,
        expected_magic=_PROPOSAL_STATE_SIDECAR_MAGIC,
        expected_schema=_PROPOSAL_STATE_SIDECAR_SCHEMA,
        label="modern-v1 proposal-state sidecar",
    )
    observed = _proposal_sidecar_receipt_from_header(
        header=header,
        relative_path=expected.relative_path,
        framed_size_bytes=source.stat().st_size,
    )
    if (
        observed != expected
        or digest != expected.payload_sha256
        or payload_size != expected.payload_size_bytes
    ):
        raise ValueError(
            "checkpoint proposal-state sidecar differs from its journal reference"
        )
    progress, pending = _validate_live_proposal_sidecar_payload(
        value,
        expected_receipt=expected,
    )
    controller_state_hash = _checkpoint_controller_state_hash(controller)
    expected_semantics = _proposal_sidecar_semantics_from_progress(
        progress,
        controller,
        controller_state_hash=controller_state_hash,
    )
    observed_semantics = {
        name: getattr(expected, name) for name in expected_semantics
    }
    if observed_semantics != expected_semantics:
        raise ValueError(
            "checkpoint proposal-state receipt has stale controller/evidence lineage"
        )
    _assert_proposal_progress_bound_to_controller(progress, controller)
    if pending is not None:
        _assert_pending_proposal_bound(progress, pending)
    _PROPOSAL_SIDECAR_CACHE[str(source.resolve())] = _ProposalSidecarCacheEntry(
        expected,
        _stat_signature(source),
    )
    return progress, pending


def _restore_runtime_checkpoint(
    journal: _V1RuntimeCheckpointJournal,
    *,
    checkpoint_path: Path,
    operator_sources=None,
    restore_operator_mask=False,
) -> V1RuntimeState:
    journal.__post_init__()
    projection = journal.controller
    projection.__post_init__()
    from .modern_v1_operator_run_binding import KEY,validate_bound_operator_sources,validate_history_binding
    validate_history_binding(journal.config.metadata,journal.config.source_seal_hash,projection.acquisition_receipt_hashes)
    if KEY in journal.config.metadata:
        if restore_operator_mask:
            from .modern_v1_operator_run_binding import restore_operator_current_mask
            validate_bound_operator_sources(journal.config,operator_sources)
            restore_operator_current_mask(operator_sources,projection.current_mask)
        validate_bound_operator_sources(journal.config,operator_sources,current_mask=projection.current_mask)
    acquisition = (
        None
        if projection.acquisition_sidecar is None
        else _read_acquisition_sidecar(
            checkpoint_path,
            projection.acquisition_sidecar,
            projection,
            operator_sources=operator_sources,
        )
    )
    if (
        getattr(
            projection,
            "label_native_selected_acquisition_receipt_hash",
            None,
        )
        is not None
    ):
        raise ValueError(
            "checkpoint contains an unsupported live label-native endpoint"
        )
    plan_payload = getattr(
        projection,
        "label_native_runtime_plan_checkpoint_payload",
        None,
    )
    label_native_runtime_plan = (
        None
        if plan_payload is None
        else restore_v1_label_native_runtime_plan_checkpoint_payload(
            plan_payload
        )
    )
    from .modern_v1_operator_checkpoint import restore_operator_active_candidate
    controller = V1ControllerState(
        binding=projection.binding,
        controller_run_id=projection.controller_run_id,
        initial_mask=projection.initial_mask,
        current_mask=projection.current_mask,
        phase=projection.phase,
        acquisition=acquisition,
        queue_cursor=projection.queue_cursor,
        label_native_runtime_plan=label_native_runtime_plan,
        label_native_runtime_plan_hashes=getattr(
            projection,
            "label_native_runtime_plan_hashes",
            (),
        ),
        label_native_rejected_plan_indices=getattr(
            projection,
            "label_native_rejected_plan_indices",
            (),
        ),
        active_candidate=restore_operator_active_candidate(projection.active_candidate,acquisition),
        accepted_trajectory=projection.accepted_trajectory,
        accepted_oracle_trajectory=getattr(
            projection,
            "accepted_oracle_trajectory",
            None,
        ),
        attempt_receipts=projection.attempt_receipts,
        transition_receipts=projection.transition_receipts,
        acquisition_receipt_hashes=projection.acquisition_receipt_hashes,
        globally_used_draw_ids=projection.globally_used_draw_ids,
        pareto_milestone_registration=getattr(
            projection,
            "pareto_milestone_registration",
            None,
        ),
        live_exact_prefix_rung_registration=getattr(
            projection,
            "live_exact_prefix_rung_registration",
            None,
        ),
        pending_pareto_milestone=getattr(
            projection,
            "pending_pareto_milestone",
            None,
        ),
        pareto_milestone_receipt=getattr(
            projection,
            "pareto_milestone_receipt",
            None,
        ),
        fqr_no_candidate_horizon_witness=getattr(
            projection,
            "fqr_no_candidate_horizon_witness",
            None,
        ),
        stop_reason=projection.stop_reason,
        logical_sequence=projection.logical_sequence,
        schema=projection.controller_schema,
    )
    restored_controller_hash = _checkpoint_controller_state_hash(controller)
    if (
        restored_controller_hash != projection.controller_state_hash
        or controller.snapshot_receipt().receipt_hash
        != projection.controller_snapshot_hash
    ):
        raise ValueError("checkpoint controller differs from its compact projection")
    proposal_progress: V1RuntimeProposalAcquisitionProgress | None = None
    pending_proposal: V1RuntimePendingProposalAcquisition | None = None
    if journal.proposal_state_sidecar is not None:
        proposal_progress, pending_proposal = _read_proposal_state_sidecar(
            checkpoint_path,
            journal.proposal_state_sidecar,
            controller,
        )
    if journal.terminal_validation_family_observation is not None:
        read_v1_terminal_validation_family_matrix_artifact(
            checkpoint_path.parent,
            journal.terminal_validation_family_observation.request,
            expected_observation=(
                journal.terminal_validation_family_observation
            ),
        )
    if journal.terminal_comparison_family_observation is not None:
        read_v1_terminal_comparison_family_matrix_artifact_v2(
            checkpoint_path.parent,
            journal.terminal_comparison_family_observation.request,
            expected_observation=(
                journal.terminal_comparison_family_observation
            ),
        )
    state = V1RuntimeState(
        config=journal.config,
        controller=controller,
        entropy_tape=journal.entropy_tape,
        pending_look=journal.pending_look,
        pending_paired_bce_chunk=journal.pending_paired_bce_chunk,
        terminal_freeze=journal.terminal_freeze,
        terminal_validation=journal.terminal_validation,
        completion=journal.completion,
        performance=journal.performance,
        schema=journal.runtime_schema,
        proposal_progress=proposal_progress,
        pending_proposal_acquisition=pending_proposal,
        staged_proposal_backend_identity_hash=(
            journal.staged_proposal_backend_identity_hash
        ),
        completed_proposal_closures=(
            journal.completed_proposal_closures
        ),
        terminal_validation_family_freeze=(
            journal.terminal_validation_family_freeze
        ),
        terminal_validation_family_request=(
            journal.terminal_validation_family_request
        ),
        terminal_validation_family_observation=(
            journal.terminal_validation_family_observation
        ),
        terminal_validation_family=journal.terminal_validation_family,
        terminal_comparison_family_freeze=(
            journal.terminal_comparison_family_freeze
        ),
        terminal_comparison_family_request=(
            journal.terminal_comparison_family_request
        ),
        terminal_comparison_family_observation=(
            journal.terminal_comparison_family_observation
        ),
        terminal_comparison_family=journal.terminal_comparison_family,
    )
    restored_runtime_hash = _checkpoint_runtime_state_hash(
        state,
        controller_state_hash=restored_controller_hash,
    )
    if restored_runtime_hash != journal.runtime_state_hash:
        raise ValueError("checkpoint runtime differs from its compact journal")
    return state


def write_v1_runtime_checkpoint(
    path: str | os.PathLike[str],
    state: V1RuntimeState,
) -> V1RuntimeCheckpointReceipt:
    """Write an atomic journal after every referenced sidecar is durable.

    The immutable acquisition is serialized once per content address.  Every
    unchanged proposal evidence frontier is likewise reused by content
    address.  Sidecar durability always precedes journal publication, and
    garbage collection follows the durable main-file replacement.
    """

    if type(state) is not V1RuntimeState:
        raise TypeError("checkpoint write requires an exact runtime state")
    if state.pending_proposal_acquisition is not None and state.proposal_progress is None:
        raise ValueError("checkpoint pending proposal lacks cumulative proposal progress")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if state.terminal_validation_family_request is not None:
        _assert_terminal_family_checkpoint_artifact_path(
            destination,
            state.terminal_validation_family_request,
        )
        if state.terminal_validation_family_observation is None:
            terminal_family_artifact_path = _terminal_family_artifact_path(
                destination.parent,
                state.terminal_validation_family_request.observation_artifact_relative_path,
                create_parents=True,
            )
            if (
                terminal_family_artifact_path.exists()
                or terminal_family_artifact_path.is_symlink()
            ):
                raise FileExistsError(
                    "terminal family request checkpoint requires pre-outcome "
                    "artifact absence"
                )
    if state.terminal_comparison_family_request is not None:
        _assert_terminal_comparison_checkpoint_artifact_path_v2(
            destination,
            state.terminal_comparison_family_request,
        )
        if state.terminal_comparison_family_observation is None:
            comparison_artifact_path = _terminal_family_artifact_path(
                destination.parent,
                state.terminal_comparison_family_request.observation_artifact_relative_path,
                create_parents=True,
            )
            if comparison_artifact_path.exists() or comparison_artifact_path.is_symlink():
                raise FileExistsError(
                    "terminal comparison request checkpoint requires pre-outcome "
                    "artifact absence"
                )
    journal = _project_runtime_checkpoint(state, checkpoint_path=destination)
    family_observation = journal.terminal_validation_family_observation
    family_artifact = (
        None
        if family_observation is None
        else _terminal_family_artifact_reference_payload(family_observation)
    )
    if family_observation is not None:
        read_v1_terminal_validation_family_matrix_artifact(
            destination.parent,
            family_observation.request,
            expected_observation=family_observation,
        )
    comparison_observation = journal.terminal_comparison_family_observation
    comparison_artifact = (
        None
        if comparison_observation is None
        else _terminal_comparison_artifact_reference_payload_v2(
            comparison_observation
        )
    )
    if comparison_observation is not None:
        read_v1_terminal_comparison_family_matrix_artifact_v2(
            destination.parent,
            comparison_observation.request,
            expected_observation=comparison_observation,
        )
    payload = cloudpickle.dumps(journal, protocol=5)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = journal.controller.acquisition_sidecar
    proposal_sidecar = journal.proposal_state_sidecar
    family_aware = journal.terminal_validation_family_freeze is not None
    comparison_aware = journal.terminal_comparison_family_freeze is not None
    proposal_aware = (
        proposal_sidecar is not None
        or bool(journal.completed_proposal_closures)
        or journal.pending_paired_bce_chunk is not None
    )
    v3_aware = proposal_aware or family_aware or comparison_aware
    header = {
        "schema": _CHECKPOINT_V3_SCHEMA if v3_aware else _CHECKPOINT_SCHEMA,
        "payload_kind": "compact_runtime_journal",
        "payload_sha256": digest,
        "payload_size_bytes": len(payload),
        "runtime_state_hash": journal.runtime_state_hash,
        "controller_snapshot_hash": state.controller.snapshot_receipt().receipt_hash,
        "pending_look_hash": (
            None if state.pending_look is None else state.pending_look.pending_hash
        ),
        "acquisition_sidecar": (
            None if sidecar is None else _sidecar_reference_payload(sidecar)
        ),
        "security_boundary": _CHECKPOINT_SECURITY_BOUNDARY,
    }
    if v3_aware:
        header["proposal_state_sidecar"] = (
            None
            if proposal_sidecar is None
            else _proposal_sidecar_reference_payload(proposal_sidecar)
        )
        header["completed_proposal_closure_hashes"] = tuple(
            item.receipt_hash
            for item in journal.completed_proposal_closures
        )
        header["staged_proposal_backend_identity_hash"] = (
            journal.staged_proposal_backend_identity_hash
        )
        if journal.pending_paired_bce_chunk is not None:
            header["pending_paired_bce_chunk_hash"] = (
                journal.pending_paired_bce_chunk.pending_hash
            )
        if family_aware:
            header["terminal_validation_family_freeze_hash"] = (
                journal.terminal_validation_family_freeze.receipt_hash
            )
            header["terminal_validation_family_request_hash"] = (
                None
                if journal.terminal_validation_family_request is None
                else journal.terminal_validation_family_request.request_hash
            )
            header["terminal_validation_family_artifact"] = family_artifact
        if comparison_aware:
            header["terminal_comparison_family_freeze_hash"] = (
                journal.terminal_comparison_family_freeze.receipt_hash
            )
            header["terminal_comparison_family_request_hash"] = (
                None
                if journal.terminal_comparison_family_request is None
                else journal.terminal_comparison_family_request.request_hash
            )
            header["terminal_comparison_family_artifact"] = comparison_artifact
    _atomic_write_framed_payload(
        destination=destination,
        magic=_CHECKPOINT_V3_MAGIC if v3_aware else _CHECKPOINT_MAGIC,
        header=header,
        payload=payload,
    )
    _prune_unreferenced_acquisition_sidecars(
        destination,
        referenced=sidecar,
    )
    _prune_unreferenced_proposal_state_sidecars(
        destination,
        referenced=proposal_sidecar,
    )
    return V1RuntimeCheckpointReceipt(
        checkpoint_path=str(destination),
        payload_sha256=digest,
        payload_size_bytes=len(payload),
        runtime_state_hash=journal.runtime_state_hash,
        controller_snapshot_hash=state.controller.snapshot_receipt().receipt_hash,
        pending_look_hash=(
            None if state.pending_look is None else state.pending_look.pending_hash
        ),
        acquisition_sidecar=sidecar,
        proposal_state_sidecar=proposal_sidecar,
        completed_proposal_closure_hashes=tuple(
            item.receipt_hash
            for item in journal.completed_proposal_closures
        ),
        staged_proposal_backend_identity_hash=(
            journal.staged_proposal_backend_identity_hash
        ),
        pending_paired_bce_chunk_hash=(
            None
            if journal.pending_paired_bce_chunk is None
            else journal.pending_paired_bce_chunk.pending_hash
        ),
        terminal_validation_family_freeze_hash=(
            None
            if journal.terminal_validation_family_freeze is None
            else journal.terminal_validation_family_freeze.receipt_hash
        ),
        terminal_validation_family_request_hash=(
            None
            if journal.terminal_validation_family_request is None
            else journal.terminal_validation_family_request.request_hash
        ),
        terminal_validation_family_artifact=family_artifact,
        terminal_comparison_family_freeze_hash=(
            None
            if journal.terminal_comparison_family_freeze is None
            else journal.terminal_comparison_family_freeze.receipt_hash
        ),
        terminal_comparison_family_request_hash=(
            None
            if journal.terminal_comparison_family_request is None
            else journal.terminal_comparison_family_request.request_hash
        ),
        terminal_comparison_family_artifact=comparison_artifact,
        schema=_CHECKPOINT_V3_SCHEMA if v3_aware else _CHECKPOINT_SCHEMA,
    )


def _validate_checkpoint_state_header(
    state: V1RuntimeState,
    header: Mapping[str, Any],
) -> None:
    controller_state_hash = _checkpoint_controller_state_hash(state.controller)
    runtime_state_hash = _checkpoint_runtime_state_hash(
        state,
        controller_state_hash=controller_state_hash,
    )
    family_freeze_hash = (
        None
        if state.terminal_validation_family_freeze is None
        else state.terminal_validation_family_freeze.receipt_hash
    )
    family_request_hash = (
        None
        if state.terminal_validation_family_request is None
        else state.terminal_validation_family_request.request_hash
    )
    family_artifact = (
        None
        if state.terminal_validation_family_observation is None
        else _terminal_family_artifact_reference_payload(
            state.terminal_validation_family_observation
        )
    )
    comparison_freeze_hash = (
        None
        if state.terminal_comparison_family_freeze is None
        else state.terminal_comparison_family_freeze.receipt_hash
    )
    comparison_request_hash = (
        None
        if state.terminal_comparison_family_request is None
        else state.terminal_comparison_family_request.request_hash
    )
    comparison_artifact = (
        None
        if state.terminal_comparison_family_observation is None
        else _terminal_comparison_artifact_reference_payload_v2(
            state.terminal_comparison_family_observation
        )
    )
    if (
        runtime_state_hash != header.get("runtime_state_hash")
        or state.controller.snapshot_receipt().receipt_hash
        != header.get("controller_snapshot_hash")
        or (None if state.pending_look is None else state.pending_look.pending_hash)
        != header.get("pending_look_hash")
        or (
            None
            if state.pending_paired_bce_chunk is None
            else state.pending_paired_bce_chunk.pending_hash
        )
        != header.get("pending_paired_bce_chunk_hash")
        or family_freeze_hash
        != header.get("terminal_validation_family_freeze_hash")
        or family_request_hash
        != header.get("terminal_validation_family_request_hash")
        or family_artifact
        != header.get("terminal_validation_family_artifact")
        or comparison_freeze_hash
        != header.get("terminal_comparison_family_freeze_hash")
        or comparison_request_hash
        != header.get("terminal_comparison_family_request_hash")
        or comparison_artifact
        != header.get("terminal_comparison_family_artifact")
    ):
        raise ValueError("modern-v1 checkpoint content differs from its verified header")


def read_v1_runtime_checkpoint(
    path: str | os.PathLike[str],
    *,
    trusted_local: bool,
    operator_sources=None,
    restore_operator_mask: bool = False,
) -> tuple[V1RuntimeState, V1RuntimeCheckpointReceipt]:
    """Verify framed bytes and replay exact live state before granting resume."""

    if trusted_local is not True:
        raise PermissionError("modern-v1 checkpoints may only be loaded as trusted local bytes")
    if type(restore_operator_mask) is not bool:
        raise TypeError('operator mask reconstruction flag must be an exact bool')
    source = Path(path)
    try:
        source_mode = source.lstat().st_mode
    except OSError as exc:
        raise ValueError("modern-v1 checkpoint is absent or inaccessible") from exc
    if not stat.S_ISREG(source_mode):
        raise ValueError("modern-v1 checkpoint must be a nonsymlink regular file")
    with source.open("rb") as handle:
        magic = handle.read(
            max(
                len(_CHECKPOINT_V3_MAGIC),
                len(_CHECKPOINT_MAGIC),
                len(_LEGACY_CHECKPOINT_MAGIC),
            )
        )
    if magic not in _LIVE_CHECKPOINT_MAGICS:
        raise ValueError("modern-v1 checkpoint magic is invalid")

    inspected = None
    if magic in _INSPECTED_LIVE_CHECKPOINT_MAGICS:
        # Every compact live checkpoint format for which the codec has an
        # authoritative journal projection first crosses its inert semantic-
        # replay barrier.  In particular, MV1CKPT2 may carry a compact F/Q
        # acquisition sidecar whose integrity is not established by digest
        # verification alone.  No live repository class is constructed and no
        # ``cloudpickle.loads`` call is reachable before this inspection
        # succeeds for the main journal and all referenced sidecars.
        #
        # MV1CKPT1 is intentionally outside this closed set: the inspector's
        # legacy cohort is the sealed pre-v16 historical layout, whereas the
        # live loader must continue to resume its valid current-schema
        # monolithic compatibility checkpoints exactly as before.
        inspected = inspect_v1_runtime_checkpoint(source, trusted_local=True)
        if inspected.checkpoint_magic != magic.decode("ascii"):
            raise ValueError("checkpoint failed exact inert inspection")

    if magic == _LEGACY_CHECKPOINT_MAGIC:
        value, header, digest, payload_size = _load_verified_framed_payload(
            source,
            expected_magic=_LEGACY_CHECKPOINT_MAGIC,
            expected_schema=_LEGACY_CHECKPOINT_SCHEMA,
            label="modern-v1 checkpoint",
        )
        if type(value) is not V1RuntimeState:
            raise TypeError("checkpoint payload is not an exact modern-v1 runtime state")
        value.__post_init__()
        _validate_checkpoint_state_header(value, header)
        receipt = V1RuntimeCheckpointReceipt(
            checkpoint_path=str(source),
            payload_sha256=digest,
            payload_size_bytes=payload_size,
            runtime_state_hash=str(header["runtime_state_hash"]),
            controller_snapshot_hash=value.controller.snapshot_receipt().receipt_hash,
            pending_look_hash=(
                None if value.pending_look is None else value.pending_look.pending_hash
            ),
            schema=_LEGACY_CHECKPOINT_SCHEMA,
        )
        return value, receipt
    proposal_aware = magic == _CHECKPOINT_V3_MAGIC
    value, header, digest, payload_size = _load_verified_framed_payload(
        source,
        expected_magic=_CHECKPOINT_V3_MAGIC if proposal_aware else _CHECKPOINT_MAGIC,
        expected_schema=_CHECKPOINT_V3_SCHEMA if proposal_aware else _CHECKPOINT_SCHEMA,
        label="modern-v1 checkpoint",
    )
    if header.get("payload_kind") != "compact_runtime_journal":
        raise ValueError("modern-v1 checkpoint has an unsupported payload kind")
    if type(value) is not _V1RuntimeCheckpointJournal:
        raise TypeError("checkpoint payload is not an exact modern-v1 runtime journal")
    sidecar = value.controller.acquisition_sidecar
    expected_sidecar_header = (
        None if sidecar is None else _sidecar_reference_payload(sidecar)
    )
    if header.get("acquisition_sidecar") != expected_sidecar_header:
        raise ValueError("checkpoint header and acquisition-sidecar journal differ")
    proposal_sidecar = value.proposal_state_sidecar
    expected_pending_paired_hash = (
        None
        if value.pending_paired_bce_chunk is None
        else value.pending_paired_bce_chunk.pending_hash
    )
    expected_family_freeze_hash = (
        None
        if value.terminal_validation_family_freeze is None
        else value.terminal_validation_family_freeze.receipt_hash
    )
    expected_family_request_hash = (
        None
        if value.terminal_validation_family_request is None
        else value.terminal_validation_family_request.request_hash
    )
    expected_family_artifact = (
        None
        if value.terminal_validation_family_observation is None
        else _terminal_family_artifact_reference_payload(
            value.terminal_validation_family_observation
        )
    )
    expected_comparison_freeze_hash = (
        None
        if value.terminal_comparison_family_freeze is None
        else value.terminal_comparison_family_freeze.receipt_hash
    )
    expected_comparison_request_hash = (
        None
        if value.terminal_comparison_family_request is None
        else value.terminal_comparison_family_request.request_hash
    )
    expected_comparison_artifact = (
        None
        if value.terminal_comparison_family_observation is None
        else _terminal_comparison_artifact_reference_payload_v2(
            value.terminal_comparison_family_observation
        )
    )
    if proposal_aware:
        if (
            proposal_sidecar is None
            and not value.completed_proposal_closures
            and expected_pending_paired_hash is None
            and expected_family_freeze_hash is None
            and expected_comparison_freeze_hash is None
        ):
            raise ValueError(
                "checkpoint-v3 journal lacks staged, paired, or terminal-family state"
            )
        expected_proposal_header = (
            None
            if proposal_sidecar is None
            else _proposal_sidecar_reference_payload(proposal_sidecar)
        )
        if header.get("proposal_state_sidecar") != expected_proposal_header:
            raise ValueError(
                "checkpoint header and proposal-state journal differ"
            )
        if header.get("completed_proposal_closure_hashes") != list(
            item.receipt_hash
            for item in value.completed_proposal_closures
        ):
            raise ValueError(
                "checkpoint header and completed proposal ledger differ"
            )
        if header.get("staged_proposal_backend_identity_hash") != (
            value.staged_proposal_backend_identity_hash
        ):
            raise ValueError("checkpoint header and staged backend anchor differ")
        if header.get("pending_paired_bce_chunk_hash") != expected_pending_paired_hash:
            raise ValueError("checkpoint header and pending paired-BCE journal differ")
        if (
            header.get("terminal_validation_family_freeze_hash")
            != expected_family_freeze_hash
            or header.get("terminal_validation_family_request_hash")
            != expected_family_request_hash
            or header.get("terminal_validation_family_artifact")
            != expected_family_artifact
        ):
            raise ValueError("checkpoint header and terminal-family journal differ")
        if (
            header.get("terminal_comparison_family_freeze_hash")
            != expected_comparison_freeze_hash
            or header.get("terminal_comparison_family_request_hash")
            != expected_comparison_request_hash
            or header.get("terminal_comparison_family_artifact")
            != expected_comparison_artifact
        ):
            raise ValueError(
                "checkpoint header and terminal-comparison journal differ"
            )
    elif proposal_sidecar is not None:
        raise ValueError("checkpoint-v2 journal contains proposal-state authority")
    state = _restore_runtime_checkpoint(value, checkpoint_path=source,operator_sources=operator_sources,
        restore_operator_mask=restore_operator_mask)
    _validate_checkpoint_state_header(state, header)
    if inspected is None:  # pragma: no cover - compact magics are closed above.
        raise AssertionError("compact checkpoint bypassed inert inspection")
    if (
        inspected.runtime_state_hash != state.runtime_state_hash
        or inspected.controller_state_hash != state.controller.state_hash
        or inspected.controller_snapshot_hash
        != state.controller.snapshot_receipt().receipt_hash
    ):
        raise ValueError("live resume differs from inspect-only identity replay")
    receipt = V1RuntimeCheckpointReceipt(
        checkpoint_path=str(source),
        payload_sha256=digest,
        payload_size_bytes=payload_size,
        runtime_state_hash=value.runtime_state_hash,
        controller_snapshot_hash=state.controller.snapshot_receipt().receipt_hash,
        pending_look_hash=(
            None if state.pending_look is None else state.pending_look.pending_hash
        ),
        acquisition_sidecar=sidecar,
        proposal_state_sidecar=proposal_sidecar,
        completed_proposal_closure_hashes=tuple(
            item.receipt_hash
            for item in value.completed_proposal_closures
        ),
        staged_proposal_backend_identity_hash=(
            value.staged_proposal_backend_identity_hash
        ),
        pending_paired_bce_chunk_hash=expected_pending_paired_hash,
        terminal_validation_family_freeze_hash=expected_family_freeze_hash,
        terminal_validation_family_request_hash=expected_family_request_hash,
        terminal_validation_family_artifact=expected_family_artifact,
        terminal_comparison_family_freeze_hash=(
            expected_comparison_freeze_hash
        ),
        terminal_comparison_family_request_hash=(
            expected_comparison_request_hash
        ),
        terminal_comparison_family_artifact=expected_comparison_artifact,
        schema=_CHECKPOINT_V3_SCHEMA if proposal_aware else _CHECKPOINT_SCHEMA,
    )
    return state, receipt


def _regular_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("checkpoint artifact must be a nonsymlink regular file")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    finally:
        os.close(descriptor)


def checkpoint_and_seal_v1_runtime_pareto_milestone(
    path: str | os.PathLike[str],
    state: V1RuntimeState,
) -> tuple[V1RuntimeState, V1RuntimeCheckpointReceipt]:
    """Checkpoint progress and close a pending matched-density seam durably.

    Ordinary progress takes one checkpoint write.  When the just-accepted
    transition lands on the registered comparator retained count, this helper
    first publishes and restores that exact pending state, preserves a
    content-addressed checkpoint copy under the checkpoint-owned milestone
    directory, binds the restored artifact into the validation-independent milestone
    receipt, and only then publishes/restores the continuing controller.  A
    crash may therefore leave either a normal pending checkpoint (which is
    safely sealable on resume) or the already sealed continuation; it can
    never skip the checkpoint seam.
    """

    if type(state) is not V1RuntimeState:
        raise TypeError("Pareto-aware checkpointing requires an exact runtime state")
    destination = Path(path)
    pending = state.controller.pending_pareto_milestone
    if pending is None:
        return state, write_v1_runtime_checkpoint(destination, state)

    pending_receipt = write_v1_runtime_checkpoint(destination, state)
    restored_pending, observed_pending = read_v1_runtime_checkpoint(
        destination,
        trusted_local=True,
    )
    restored_marker = restored_pending.controller.pending_pareto_milestone
    if (
        observed_pending != pending_receipt
        or restored_pending.runtime_state_hash != state.runtime_state_hash
        or restored_marker is None
        or restored_marker.pending_hash != pending.pending_hash
    ):
        raise ValueError("pending Pareto checkpoint failed exact restore verification")

    archive_directory = destination.parent / (
        destination.name + ".pareto_milestones"
    )
    if archive_directory.exists():
        if archive_directory.is_symlink() or not archive_directory.is_dir():
            raise ValueError("Pareto milestone checkpoint directory is not a real directory")
    else:
        archive_directory.mkdir(parents=False, exist_ok=False)
        _fsync_directory(destination.parent)
    archive_path = archive_directory / f"{pending.pending_hash}.mv1ckpt"
    if archive_path.exists() or archive_path.is_symlink():
        archived_state, archive_receipt = read_v1_runtime_checkpoint(
            archive_path,
            trusted_local=True,
        )
    else:
        archive_receipt = write_v1_runtime_checkpoint(
            archive_path,
            restored_pending,
        )
        archived_state, observed_archive = read_v1_runtime_checkpoint(
            archive_path,
            trusted_local=True,
        )
        if observed_archive != archive_receipt:
            raise ValueError("Pareto milestone archive receipt changed across restore")
    archived_marker = archived_state.controller.pending_pareto_milestone
    if (
        archived_state.runtime_state_hash != restored_pending.runtime_state_hash
        or archived_marker is None
        or archived_marker.pending_hash != pending.pending_hash
    ):
        raise ValueError("Pareto milestone archive contains another pending transition")

    checkpoint_binding = V1ParetoMilestoneCheckpointBinding(
        pending_milestone_hash=pending.pending_hash,
        controller_state_hash=archived_state.controller.state_hash,
        controller_snapshot_receipt_hash=(
            archived_state.controller.snapshot_receipt().receipt_hash
        ),
        checkpoint_artifact_hash=_regular_file_sha256(archive_path),
        checkpoint_manifest_hash=archive_receipt.receipt_hash,
        checkpoint_codec_schema=archive_receipt.schema,
    )
    sealed_controller = seal_v1_pareto_milestone_after_checkpoint(
        archived_state.controller,
        checkpoint_binding=checkpoint_binding,
    )
    sealed_state = replace(archived_state, controller=sealed_controller)
    sealed_receipt = write_v1_runtime_checkpoint(destination, sealed_state)
    restored_sealed, observed_sealed = read_v1_runtime_checkpoint(
        destination,
        trusted_local=True,
    )
    sealed_milestone = restored_sealed.controller.pareto_milestone_receipt
    if (
        observed_sealed != sealed_receipt
        or restored_sealed.runtime_state_hash != sealed_state.runtime_state_hash
        or restored_sealed.controller.pending_pareto_milestone is not None
        or sealed_milestone is None
        or sealed_milestone.checkpoint_binding.binding_hash
        != checkpoint_binding.binding_hash
    ):
        raise ValueError("sealed Pareto checkpoint failed exact restore verification")
    return restored_sealed, observed_sealed


__all__ = [
    "AcquisitionFactory",
    "SamplingLawFactory",
    "TerminalFamilyValidator",
    "TerminalValidator",
    "ThresholdFactory",
    "V1CandidateCalibrationFreezeFactory",
    "V1CalibrationHorizonSkipDecisionFactory",
    "V1PairedBCEBettingReplayFactory",
    "V1PairedBCEChunkExecutor",
    "V1PairedBCEChunkFactory",
    "V1PairedBCERegistrationFactory",
    "V1ProposalAcquisitionExecutor",
    "V1ProposalForecastFactory",
    "V1FQRNoCandidateHorizonProofBuilder",
    "V1ProposalFreezeEvaluator",
    "V1ProposalPureFinalizer",
    "V1ProposalRequestPlanFactory",
    "V1RuntimeAcquisitionSidecarReceipt",
    "V1RuntimeAcquisition",
    "V1RuntimeCheckpointReceipt",
    "V1RuntimeCompletedProposalClosureReceipt",
    "V1RuntimeExecutionConfig",
    "V1RuntimePairedBCEChunkOutcome",
    "V1RuntimePendingLook",
    "V1RuntimePendingPairedBCEChunk",
    "V1RuntimePendingProposalAcquisition",
    "V1RuntimeProposalOrderIndeterminate",
    "V1RuntimeScientificHorizonIndeterminate",
    "V1RuntimeProposalAcquisitionOutcome",
    "V1RuntimeProposalAcquisitionProgress",
    "V1RuntimeProposalAcquisitionRequest",
    "V1RuntimeProposalFreezeEvaluation",
    "V1RuntimeProposalRequestPlan",
    "V1RuntimeProposalSufficientStateDescriptor",
    "V1RuntimeState",
    "V1TerminalComparisonFamilyMatrixOutcomeV2",
    "V1TerminalValidationFamilyMatrixOutcome",
    "V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2",
    "V1_TERMINAL_VALIDATION_FAMILY_MATRIX_ARTIFACT_SCHEMA",
    "advance_v1_runtime_once",
    "advance_v1_runtime_paired_bce_candidate_freeze",
    "advance_v1_runtime_nonproposal_once",
    "attach_v1_performance_receipt",
    "complete_v1_runtime",
    "complete_v1_terminal_comparison_family_v2",
    "complete_v1_terminal_validation_family",
    "checkpoint_and_seal_v1_runtime_pareto_milestone",
    "decide_v1_runtime_paired_bce_oracle",
    "decide_v1_runtime_proposal_freeze",
    "execute_v1_runtime_pending_paired_bce_chunk",
    "execute_v1_runtime_pending_proposal_acquisition",
    "execute_v1_runtime_pending_look",
    "finalize_and_attach_v1_runtime_proposal_acquisition",
    "finalize_v1_runtime_fqr_no_candidate_horizon",
    "freeze_v1_terminal",
    "freeze_v1_terminal_comparison_family_v2",
    "freeze_v1_terminal_validation_family",
    "freeze_v1_runtime_paired_bce_candidate",
    "initialize_v1_runtime",
    "initialize_v1_runtime_proposal_acquisition",
    "prepare_v1_runtime_proposal_acquisition",
    "prepare_v1_runtime_paired_bce_chunk",
    "prepare_v1_runtime_look",
    "prepare_v1_terminal_comparison_family_request_v2",
    "prepare_v1_terminal_validation_family_request",
    "publish_v1_terminal_comparison_family_matrix_artifact_v2",
    "publish_v1_terminal_validation_family_matrix_artifact",
    "read_v1_runtime_checkpoint",
    "require_v1_runtime_proposal_order_resolved",
    "require_v1_runtime_scientific_endpoint",
    "reveal_v1_terminal_once",
    "reveal_v1_terminal_comparison_family_once_v2",
    "reveal_v1_terminal_validation_family_once",
    "read_v1_terminal_comparison_family_matrix_artifact_v2",
    "read_v1_terminal_validation_family_matrix_artifact",
    "run_v1_runtime_to_stop",
    "v1_runtime_proposal_candidate_identity_hash",
    "v1_runtime_proposal_order_indeterminate",
    "write_v1_runtime_checkpoint",
]
