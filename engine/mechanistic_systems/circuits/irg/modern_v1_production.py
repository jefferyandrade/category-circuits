"""Production admission and durable execution for genuine modern-v1.

This module is the single composition joint between the immutable Option-A
campaign, a concrete model/data backend, the partition-free controller, exact
paired direct-increment provider, resumable checkpoints, and the terminal
single-reveal firewall.  It is deliberately an internal development entry
point: successful execution emits a genuine controller artifact but never
opens ``irg/v1``, replaces the unavailable public certificate, or asserts a
scientific result.

Decision-bearing coordinates live in :class:`V1OptionACampaignContract` and in
content-addressed factories declared by :class:`V1ProductionRuntimeManifest`.
Environment variables and command-line callers may choose paths and physical
packing only; they cannot rewrite the campaign, threshold, sampling, queue, or
terminal policies after outcomes exist.
"""

from __future__ import annotations

import time

import contextlib

import hashlib
import inspect
import json
import math
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from .contracts import MeasurementProvenance, canonical_json, content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from .modern_v1_campaign_contract import (
    V1OptionACampaignBinding,
    v1_dense_referent_mask_hash,
)
from .modern_v1_controller import (
    V1CandidateCalibrationFreeze,
    V1ControllerState,
    V1FrozenAffineCalibration,
    initialize_v1_controller,
)
from .modern_v1_pareto_milestones import (
    V1ParetoMilestoneReceipt,
    V1ParetoMilestoneRegistration,
)
from .modern_v1_paired_bce_acquisition import (
    ANALYTIC_FIXED_AFFINE_PAIRED_BCE_KIND,
    CENSUS_ONLY_PAIRED_BCE_KIND,
    PAIRED_BCE_STRATA,
    V1PairedBCEAcquisitionRegistration,
    V1PairedBCEAcquisitionState,
    V1PairedBCEBettingReplay,
    V1PairedBCECalibrationHorizonSkipDecision,
    V1PairedBCEChunkRequest,
    V1PairedBCEChunkReceipt,
    V1PairedBCEDecision,
    V1PairedBCEDynamicOracleReceipt,
    V1PairedBCEStratumRegistration,
    V1PairedBCEVerifiedSealedOrder,
    build_paired_bce_acquisition_registration,
    commit_paired_bce_chunk,
    make_next_paired_bce_chunk_request,
    next_paired_bce_chunk_coordinates,
    paired_bce_attempt_alpha_coordinate_hash,
    paired_bce_loss_ceiling_identity_hash,
    paired_bce_sealed_order_hash,
    reduce_paired_bce_chunk,
    validate_paired_bce_chunk_request_preoutcome,
    verify_paired_bce_sealed_order,
    register_paired_bce_dynamic_oracle,
    register_paired_bce_oracle_role_separation,
)
from .modern_v1_receipts import (
    V1PerformanceReceipt,
    V1TerminalComparisonFamilyFreezeReceiptV2,
    V1TerminalComparisonFamilyRequestReceiptV2,
    V1TerminalValidationFamilyFreezeReceipt,
    V1TerminalValidationFamilyRequestReceipt,
)
from .modern_v1_response_exactification import (
    ORACLE_ATTEMPT_ALPHA_AXIS_SCHEMA,
    V1MasterDecisionAlphaRegistration,
)
from .modern_v1_runtime import (
    AcquisitionFactory,
    SamplingLawFactory,
    TerminalValidator,
    ThresholdFactory,
    V1RuntimeCheckpointReceipt,
    V1RuntimeCompletedProposalClosureReceipt,
    V1RuntimeExecutionConfig,
    V1RuntimePairedBCEChunkOutcome,
    V1RuntimePendingPairedBCEChunk,
    V1RuntimeProposalSufficientStateDescriptor,
    V1RuntimeState,
    V1TerminalComparisonFamilyMatrixOutcomeV2,
    V1TerminalValidationFamilyMatrixOutcome,
    V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2,
    advance_v1_runtime_nonproposal_once,
    advance_v1_runtime_once,
    advance_v1_runtime_paired_bce_candidate_freeze,
    attach_v1_performance_receipt,
    checkpoint_and_seal_v1_runtime_pareto_milestone,
    complete_v1_runtime,
    complete_v1_terminal_comparison_family_v2,
    complete_v1_terminal_validation_family,
    decide_v1_runtime_proposal_freeze,
    decide_v1_runtime_paired_bce_oracle,
    execute_v1_runtime_pending_look,
    execute_v1_runtime_pending_paired_bce_chunk,
    execute_v1_runtime_pending_proposal_acquisition,
    finalize_and_attach_v1_runtime_proposal_acquisition,
    finalize_v1_runtime_fqr_no_candidate_horizon,
    freeze_v1_runtime_paired_bce_candidate,
    freeze_v1_terminal,
    freeze_v1_terminal_comparison_family_v2,
    freeze_v1_terminal_validation_family,
    initialize_v1_runtime,
    initialize_v1_runtime_proposal_acquisition,
    prepare_v1_runtime_paired_bce_chunk,
    prepare_v1_runtime_proposal_acquisition,
    prepare_v1_runtime_look,
    prepare_v1_terminal_comparison_family_request_v2,
    prepare_v1_terminal_validation_family_request,
    read_v1_runtime_checkpoint,
    reveal_v1_terminal_once,
    reveal_v1_terminal_comparison_family_once_v2,
    reveal_v1_terminal_validation_family_once,
    require_v1_runtime_proposal_order_resolved,
    require_v1_runtime_scientific_endpoint,
    v1_runtime_proposal_order_indeterminate,
    write_v1_runtime_checkpoint,
)
from .modern_v1_staged_alpha import V1StagedAlphaDecision, V1StagedAlphaEntropyTape
from mechanistic_systems.measurement.request_providers import PairedMaskOracleProvider

_MANIFEST_SCHEMA: Final = "partition_free_v1_production_runtime_manifest_v3"
_BUNDLE_SCHEMA: Final = "partition_free_v1_production_runtime_bundle_v3"
_INDEX_SCHEMA: Final = "partition_free_v1_production_run_index_v3"
_OUTPUT_SCHEMA: Final = "partition_free_v1_production_output_receipt_v3"
_CALLABLE_SEAL_SCHEMA: Final = "partition_free_v1_production_callable_seal_v1"
_PROPOSAL_BACKEND_SEAL_SCHEMA: Final = (
    "partition_free_v1_production_proposal_backend_seal_v1"
)
_PRODUCTION_RESUME_BINDING_SCHEMA: Final = (
    "partition_free_v1_production_resume_binding_v1"
)
_PRODUCTION_RESUME_BINDING_FILENAME: Final = (
    "modern_v1_production_resume_binding.json"
)
_PROPOSAL_QUALIFICATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_production_proposal_qualification_receipt_v1"
)
_PROPOSAL_QUALIFICATION_RECEIPT_FILENAME: Final = (
    "modern_v1_proposal_qualification_receipt.json"
)
_PROPOSAL_QUALIFICATION_CHECKPOINT_FILENAME: Final = (
    "modern_v1_proposal_qualification.ckpt"
)
_TRAIN_ONLY_DECISION_QUALIFICATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_train_only_decision_qualification_receipt_v1"
)
_TRAIN_ONLY_DECISION_QUALIFICATION_RECEIPT_FILENAME: Final = (
    "modern_v1_train_only_decision_qualification_receipt.json"
)
_TRAIN_ONLY_DECISION_QUALIFICATION_CHECKPOINT_FILENAME: Final = (
    "modern_v1_train_only_decision_qualification.ckpt"
)
_TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_SCHEMA: Final = (
    "partition_free_v1_train_only_decision_qualification_binding_v1"
)
_TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_FILENAME: Final = (
    "modern_v1_train_only_decision_qualification_binding.json"
)
_PAIRED_ORACLE_SEAL_SCHEMA: Final = "partition_free_v1_paired_oracle_seal_v2"
_TERMINAL_EXTRACTION_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_terminal_extraction_closure_v1"
)
_TERMINAL_EXTRACTION_CLOSURE_FILE_SCHEMA: Final = (
    "partition_free_v1_terminal_extraction_closure_file_v1"
)
_PAIRED_BCE_PRODUCTION_ORACLE_SCHEMA: Final = (
    "partition_free_v1_production_paired_bce_oracle_v1"
)
_PAIRED_BCE_ROW_ARTIFACT_SCHEMA: Final = (
    "partition_free_v1_production_paired_bce_external_row_artifact_v1"
)
_EVALUATOR_ORACLE_ATTEMPT_ORDER_RECEIPTS_METHOD: Final = (
    "oracle_attempt_order_receipts"
)
_EVALUATOR_ACTIVE_DYNAMIC_ORACLE_FRAME_METHOD: Final = (
    "active_paired_bce_oracle_sampling_frame"
)
_EVALUATOR_RETIRE_COMMITTED_GENERATION_STATE_METHOD: Final = (
    "retire_committed_generation_state"
)
_TERMINAL_EXTRACTION_CLOSURE_FILENAME: Final = (
    "modern_v1_terminal_extraction_closure.json"
)
_TERMINAL_VALIDATION_FAMILY_DESIGN_SCHEMA: Final = (
    "partition_free_v1_production_terminal_validation_family_design_v1"
)
_TERMINAL_VALIDATION_FAMILY_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_production_terminal_validation_family_execution_v2"
)
_TERMINAL_VALIDATION_FAMILY_METADATA_KEY: Final = "terminal_validation_family"
_TERMINAL_VALIDATION_FAMILY_MATRIX_FILENAME: Final = "common_panel.mv1term"
_TERMINAL_COMPARISON_FAMILY_DESIGN_SCHEMA_V2: Final = (
    "partition_free_v1_production_terminal_comparison_family_design_v2"
)
_TERMINAL_COMPARISON_FAMILY_EXECUTION_SCHEMA_V2: Final = (
    "partition_free_v1_production_terminal_comparison_family_execution_v2"
)
_TERMINAL_COMPARISON_FAMILY_MATRIX_FILENAME_V2: Final = "common_panel.mv1term2"
_EVALUATOR_STATE_RESTORATION_METHOD: Final = "restore_controller_state"
_EVALUATOR_CANDIDATE_CALIBRATION_FREEZE_METHOD: Final = (
    "freeze_candidate_calibration"
)
_EVALUATOR_CANDIDATE_CALIBRATION_VERIFY_METHOD: Final = (
    "verify_candidate_calibration_freeze"
)

BackendKind = Literal["synthetic_fixture", "alexnet_production"]
ProposalProtocolKind = Literal[
    "legacy_synthetic_one_shot_v1",
    "operator_binary_d_c_v1",
    "staged_train_only_proposal_v1",
]
ProductionCallableRole = Literal[
    "acquisition_factory",
    "threshold_factory",
    "sampling_law_factory",
    "direct_oracle_evaluator",
    "proposal_initial_state_factory",
    "proposal_forecast_factory",
    "proposal_request_plan_factory",
    "proposal_acquisition_executor",
    "proposal_freeze_evaluator",
    "proposal_acquisition_finalizer",
    "proposal_resume_validator",
    "terminal_validator",
    "terminal_extraction_finalizer",
    "performance_receipt_factory",
    "attempt_structural_disclosure",
]
PerformanceReceiptFactory = Callable[[V1RuntimeState], V1PerformanceReceipt]
ProductionDecision = (
    V1StagedAlphaDecision
    | V1PairedBCEDecision
    | V1PairedBCECalibrationHorizonSkipDecision
)
_PRODUCTION_CALLABLE_ROLES: Final = frozenset(
    {
        "acquisition_factory",
        "threshold_factory",
        "sampling_law_factory",
        "direct_oracle_evaluator",
        "proposal_initial_state_factory",
        "proposal_forecast_factory",
        "proposal_request_plan_factory",
        "proposal_acquisition_executor",
        "proposal_freeze_evaluator",
        "proposal_acquisition_finalizer",
        "proposal_resume_validator",
        "terminal_validator",
        "terminal_extraction_finalizer",
        "performance_receipt_factory",
        "attempt_structural_disclosure",
    }
)
_FORBIDDEN_CALLABLE_SPEC_FRAGMENTS: Final = (
    "registered_resnet18_pilot",
    "trust_walk",
    "modern_v1_validation_20260813",
)


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _sha256_hex(value: object, label: str) -> str:
    result = _nonempty(value, label)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_lexists(path: Path) -> bool:
    """Return whether ``path`` names a directory entry without following links."""

    return os.path.lexists(os.fspath(path))


def _require_nonsymlink_regular_file(path: Path, *, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{label} is not a non-symlink regular file")
    return observed


def _require_nonsymlink_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise ValueError(f"{label} is not a non-symlink directory")
    return observed


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_materialize_exact_file(
    source: Path,
    destination: Path,
    *,
    label: str,
) -> None:
    """Install an exact immutable copy without ever replacing a destination.

    Resume dependencies are content-addressed append-only records.  A target
    already present is therefore accepted only when it is byte-identical to
    the source.  Missing targets are copied to a same-directory temporary,
    fsynced, and installed with a no-clobber hard-link operation.  Consequently
    a crash can leave at most an inert temporary or a valid complete artifact;
    it cannot publish a partial dependency under its authority-bearing name.
    """

    source_stat = _require_nonsymlink_regular_file(source, label=f"source {label}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_nonsymlink_directory(
        destination.parent,
        label=f"destination directory for {label}",
    )

    if _path_lexists(destination):
        destination_stat = _require_nonsymlink_regular_file(
            destination,
            label=f"destination {label}",
        )
        if (
            source_stat.st_dev == destination_stat.st_dev
            and source_stat.st_ino == destination_stat.st_ino
        ):
            return
        if (
            source_stat.st_size != destination_stat.st_size
            or _sha256_file(source) != _sha256_file(destination)
        ):
            raise ValueError(f"destination {label} conflicts with its resume source")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.resume-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    source_digest = hashlib.sha256()
    copied = 0
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_descriptor = os.open(source, source_flags)
        try:
            opened_source_stat = os.fstat(source_descriptor)
            if (
                not stat.S_ISREG(opened_source_stat.st_mode)
                or opened_source_stat.st_dev != source_stat.st_dev
                or opened_source_stat.st_ino != source_stat.st_ino
            ):
                raise ValueError(f"source {label} changed during resume import")
            with os.fdopen(descriptor, "wb", closefd=True) as target_handle:
                descriptor = -1
                while True:
                    block = os.read(source_descriptor, 1024 * 1024)
                    if not block:
                        break
                    target_handle.write(block)
                    source_digest.update(block)
                    copied += len(block)
                target_handle.flush()
                os.fsync(target_handle.fileno())
        finally:
            os.close(source_descriptor)

        final_source_stat = _require_nonsymlink_regular_file(
            source,
            label=f"source {label}",
        )
        if (
            final_source_stat.st_dev != source_stat.st_dev
            or final_source_stat.st_ino != source_stat.st_ino
            or final_source_stat.st_size != source_stat.st_size
            or final_source_stat.st_mtime_ns != source_stat.st_mtime_ns
            or final_source_stat.st_ctime_ns != source_stat.st_ctime_ns
            or copied != source_stat.st_size
            or _sha256_file(temporary) != source_digest.hexdigest()
            or _sha256_file(source) != source_digest.hexdigest()
        ):
            raise ValueError(f"source {label} changed during resume import")

        try:
            os.link(temporary, destination)
        except FileExistsError:
            destination_stat = _require_nonsymlink_regular_file(
                destination,
                label=f"destination {label}",
            )
            if (
                destination_stat.st_size != copied
                or _sha256_file(destination) != source_digest.hexdigest()
            ):
                raise ValueError(
                    f"destination {label} raced with a conflicting artifact"
                )
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(payload).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _observed_callable_source(value: Callable[..., Any]) -> tuple[str, str]:
    """Return the exact inspectable symbol and source-file SHA for one callable.

    Production components must be reviewable Python symbols.  Native/builtin or
    dynamically generated callables without an inspectable source file refuse;
    their behavior cannot be tied to the immutable source target.
    """

    if not callable(value):
        raise TypeError("production component must be callable")
    unwrapped = inspect.unwrap(value)
    if inspect.ismethod(unwrapped):
        target = unwrapped.__func__
    elif inspect.isfunction(unwrapped):
        target = unwrapped
    else:
        target = type(unwrapped)
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if type(module) is not str or not module or type(qualname) is not str or not qualname:
        raise ValueError("production callable lacks a stable module-qualified symbol")
    callable_spec = f"{module}:{qualname}"
    lowered = callable_spec.lower()
    if any(fragment in lowered for fragment in _FORBIDDEN_CALLABLE_SPEC_FRAGMENTS):
        raise ValueError("retired/v0 walker callable cannot enter a modern-v1 bundle")
    source = inspect.getsourcefile(target)
    if source is None:
        try:
            source = inspect.getfile(target)
        except (TypeError, OSError) as exc:
            raise ValueError("production callable has no inspectable source file") from exc
    source_path = Path(source).resolve()
    if not source_path.is_file():
        raise ValueError("production callable source file is unavailable")
    return callable_spec, _sha256_file(source_path)


@dataclass(frozen=True, slots=True)
class V1ProductionCallableSeal:
    """Bind an executable component to its observed symbol, source, and config.

    The callable object itself is runtime-only and excluded from the identity.
    Its module-qualified symbol and entire source-file SHA are derived rather
    than caller asserted.  ``configuration_hash`` binds artifact/model/config
    state that is intentionally outside the Python source file.
    """

    role: ProductionCallableRole
    callable_value: Callable[..., Any] = field(repr=False, compare=False)
    configuration_hash: str = ""
    callable_spec: str = field(init=False)
    source_file_sha256: str = field(init=False)
    schema: str = _CALLABLE_SEAL_SCHEMA

    def __post_init__(self) -> None:
        if self.role not in _PRODUCTION_CALLABLE_ROLES:
            raise ValueError("unsupported modern-v1 production callable role")
        object.__setattr__(
            self,
            "configuration_hash",
            _nonempty(self.configuration_hash, "configuration_hash"),
        )
        callable_spec, source_sha256 = _observed_callable_source(self.callable_value)
        object.__setattr__(self, "callable_spec", callable_spec)
        object.__setattr__(self, "source_file_sha256", source_sha256)
        if self.schema != _CALLABLE_SEAL_SCHEMA:
            raise ValueError("unsupported modern-v1 production callable-seal schema")

    @property
    def identity_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "role": self.role,
                "callable_spec": self.callable_spec,
                "source_file_sha256": self.source_file_sha256,
                "configuration_hash": self.configuration_hash,
            },
            prefix="partition-free-v1-production-callable-",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.callable_value(*args, **kwargs)


def seal_v1_production_callable(
    *,
    role: ProductionCallableRole,
    callable_value: Callable[..., Any],
    configuration_hash: str,
) -> V1ProductionCallableSeal:
    return V1ProductionCallableSeal(
        role=role,
        callable_value=callable_value,
        configuration_hash=configuration_hash,
    )


@dataclass(frozen=True, slots=True)
class V1ProductionProposalBackendSeal:
    """Composite identity for the complete train-only proposal capability.

    Individual callable seals remain independently reviewable, while this seal
    prevents a caller from recombining or relabelling them as another proposal
    backend.  Its identity also binds the initial sufficient-state ABI and the
    campaign/run/model coordinates in which the capability is authorized.
    """

    backend_kind: BackendKind
    backend_manifest_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    model_data_seal_hash: str
    runtime_logical_config_hash: str
    action_regret_metric_hash: str
    initial_phase_evidence_state_hash: str
    freeze_criterion_hash: str
    train_population_hash: str
    sufficient_state_representation_hash: str
    initial_state_factory: V1ProductionCallableSeal
    forecast_factory: V1ProductionCallableSeal
    request_plan_factory: V1ProductionCallableSeal
    acquisition_executor: V1ProductionCallableSeal
    freeze_evaluator: V1ProductionCallableSeal
    acquisition_finalizer: V1ProductionCallableSeal
    resume_validator: V1ProductionCallableSeal | None = None
    proposal_order_alpha_schedule_hash: str | None = None
    train_optimization_data_only: bool = True
    validation_outcomes_observed: bool = False
    candidate_or_mask_authority: bool = False
    schema: str = _PROPOSAL_BACKEND_SEAL_SCHEMA

    def __post_init__(self) -> None:
        if self.backend_kind not in {"synthetic_fixture", "alexnet_production"}:
            raise ValueError("proposal backend seal has an unsupported backend kind")
        for name in (
            "backend_manifest_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "model_data_seal_hash",
            "runtime_logical_config_hash",
            "action_regret_metric_hash",
            "initial_phase_evidence_state_hash",
            "freeze_criterion_hash",
            "train_population_hash",
            "sufficient_state_representation_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for component_value, component_role in (
            (self.initial_state_factory, "proposal_initial_state_factory"),
            (self.forecast_factory, "proposal_forecast_factory"),
            (self.request_plan_factory, "proposal_request_plan_factory"),
            (self.acquisition_executor, "proposal_acquisition_executor"),
            (self.freeze_evaluator, "proposal_freeze_evaluator"),
            (self.acquisition_finalizer, "proposal_acquisition_finalizer"),
        ):
            if (
                type(component_value) is not V1ProductionCallableSeal
                or component_value.role != component_role
            ):
                raise TypeError(
                    f"{component_role} must be an exact role-matched callable seal"
                )
        if self.resume_validator is not None and (
            type(self.resume_validator) is not V1ProductionCallableSeal
            or self.resume_validator.role != "proposal_resume_validator"
        ):
            raise TypeError(
                "proposal_resume_validator must be an exact role-matched callable seal"
            )
        if self.proposal_order_alpha_schedule_hash is not None:
            object.__setattr__(
                self,
                "proposal_order_alpha_schedule_hash",
                _nonempty(
                    self.proposal_order_alpha_schedule_hash,
                    "proposal_order_alpha_schedule_hash",
                ),
            )
        if (
            self.train_optimization_data_only is not True
            or self.validation_outcomes_observed is not False
            or self.candidate_or_mask_authority is not False
            or self.schema != _PROPOSAL_BACKEND_SEAL_SCHEMA
        ):
            raise ValueError("proposal backend crossed its train-only authority boundary")

    @property
    def identity_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "protocol": "staged_train_only_proposal_v1",
                "backend_kind": self.backend_kind,
                "backend_manifest_hash": self.backend_manifest_hash,
                "campaign_binding_hash": self.campaign_binding_hash,
                "controller_run_id": self.controller_run_id,
                "model_data_seal_hash": self.model_data_seal_hash,
                "runtime_logical_config_hash": self.runtime_logical_config_hash,
                "action_regret_metric_hash": self.action_regret_metric_hash,
                "initial_phase_evidence_state_hash": (
                    self.initial_phase_evidence_state_hash
                ),
                "freeze_criterion_hash": self.freeze_criterion_hash,
                "train_population_hash": self.train_population_hash,
                "sufficient_state_representation_hash": (
                    self.sufficient_state_representation_hash
                ),
                "proposal_initial_state_factory_hash": (
                    self.initial_state_factory.identity_hash
                ),
                "proposal_forecast_factory_hash": self.forecast_factory.identity_hash,
                "proposal_request_plan_factory_hash": (
                    self.request_plan_factory.identity_hash
                ),
                "proposal_acquisition_executor_hash": (
                    self.acquisition_executor.identity_hash
                ),
                "proposal_freeze_evaluator_hash": self.freeze_evaluator.identity_hash,
                "proposal_acquisition_finalizer_hash": (
                    self.acquisition_finalizer.identity_hash
                ),
                "proposal_resume_validator_hash": (
                    None
                    if self.resume_validator is None
                    else self.resume_validator.identity_hash
                ),
                "proposal_order_alpha_schedule_hash": (
                    self.proposal_order_alpha_schedule_hash
                ),
                "proposal_initial_state_input_kind": (
                    "runtime_state"
                    if self.proposal_order_alpha_schedule_hash is not None
                    else "controller_state"
                ),
                "train_optimization_data_only": True,
                "validation_outcomes_observed": False,
                "candidate_or_mask_authority": False,
            },
            prefix="partition-free-v1-production-proposal-backend-",
        )

    @property
    def manifest_metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "proposal_backend_identity_hash": self.identity_hash,
                "action_regret_metric_hash": self.action_regret_metric_hash,
                "initial_phase_evidence_state_hash": (
                    self.initial_phase_evidence_state_hash
                ),
                "freeze_criterion_hash": self.freeze_criterion_hash,
                "train_population_hash": self.train_population_hash,
                "sufficient_state_representation_hash": (
                    self.sufficient_state_representation_hash
                ),
                "proposal_initial_state_factory_hash": (
                    self.initial_state_factory.identity_hash
                ),
                "proposal_forecast_factory_hash": self.forecast_factory.identity_hash,
                "proposal_request_plan_factory_hash": (
                    self.request_plan_factory.identity_hash
                ),
                "proposal_acquisition_executor_hash": (
                    self.acquisition_executor.identity_hash
                ),
                "proposal_freeze_evaluator_hash": self.freeze_evaluator.identity_hash,
                "proposal_acquisition_finalizer_hash": (
                    self.acquisition_finalizer.identity_hash
                ),
                "proposal_resume_validator_hash": (
                    None
                    if self.resume_validator is None
                    else self.resume_validator.identity_hash
                ),
                "proposal_order_alpha_schedule_hash": (
                    self.proposal_order_alpha_schedule_hash
                ),
                "proposal_initial_state_input_kind": (
                    "runtime_state"
                    if self.proposal_order_alpha_schedule_hash is not None
                    else "controller_state"
                ),
                "train_optimization_data_only": True,
                "validation_outcomes_observed": False,
                "candidate_or_mask_authority": False,
            }
        )


def seal_v1_production_proposal_backend(
    *,
    backend_kind: BackendKind,
    backend_manifest_hash: str,
    campaign_binding_hash: str,
    controller_run_id: str,
    model_data_seal_hash: str,
    runtime_logical_config_hash: str,
    action_regret_metric_hash: str,
    initial_phase_evidence_state_hash: str,
    freeze_criterion_hash: str,
    train_population_hash: str,
    sufficient_state_representation_hash: str,
    initial_state_factory: V1ProductionCallableSeal,
    forecast_factory: V1ProductionCallableSeal,
    request_plan_factory: V1ProductionCallableSeal,
    acquisition_executor: V1ProductionCallableSeal,
    freeze_evaluator: V1ProductionCallableSeal,
    acquisition_finalizer: V1ProductionCallableSeal,
    resume_validator: V1ProductionCallableSeal | None = None,
    proposal_order_alpha_schedule_hash: str | None = None,
) -> V1ProductionProposalBackendSeal:
    return V1ProductionProposalBackendSeal(
        backend_kind=backend_kind,
        backend_manifest_hash=backend_manifest_hash,
        campaign_binding_hash=campaign_binding_hash,
        controller_run_id=controller_run_id,
        model_data_seal_hash=model_data_seal_hash,
        runtime_logical_config_hash=runtime_logical_config_hash,
        action_regret_metric_hash=action_regret_metric_hash,
        initial_phase_evidence_state_hash=initial_phase_evidence_state_hash,
        freeze_criterion_hash=freeze_criterion_hash,
        train_population_hash=train_population_hash,
        sufficient_state_representation_hash=sufficient_state_representation_hash,
        initial_state_factory=initial_state_factory,
        forecast_factory=forecast_factory,
        request_plan_factory=request_plan_factory,
        acquisition_executor=acquisition_executor,
        freeze_evaluator=freeze_evaluator,
        acquisition_finalizer=acquisition_finalizer,
        resume_validator=resume_validator,
        proposal_order_alpha_schedule_hash=proposal_order_alpha_schedule_hash,
    )


@dataclass(frozen=True, slots=True)
class V1TerminalExtractionClosure:
    """Validation-independent final-mask extraction/calibration installation receipt.

    A backend finalizer may materialize files or install an in-process terminal
    calibration, but the production orchestrator accepts only this compact
    immutable result.  The stopped controller snapshot and complete acquisition
    receipt lineage make it impossible to reuse a closure for another terminal
    mask or another dynamically relinearized trajectory.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    controller_snapshot_hash: str
    terminal_mask_hash: str
    acquisition_receipt_hashes: tuple[str, ...]
    extraction_state_hash: str
    extraction_calibration_hash: str
    validation_independent: bool = True
    terminal_rows_visible: bool = False
    schema: str = _TERMINAL_EXTRACTION_CLOSURE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "controller_snapshot_hash",
            "terminal_mask_hash",
            "extraction_state_hash",
            "extraction_calibration_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        acquisition_receipt_hashes = tuple(
            _nonempty(value, "acquisition_receipt_hashes")
            for value in self.acquisition_receipt_hashes
        )
        if len(set(acquisition_receipt_hashes)) != len(acquisition_receipt_hashes):
            raise ValueError("terminal extraction acquisition lineage must be unique")
        object.__setattr__(
            self,
            "acquisition_receipt_hashes",
            acquisition_receipt_hashes,
        )
        if (
            self.schema != _TERMINAL_EXTRACTION_CLOSURE_SCHEMA
            or self.validation_independent is not True
            or self.terminal_rows_visible is not False
        ):
            raise ValueError(
                "terminal extraction closure crossed the validation-independent firewall"
            )

    @property
    def closure_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-extraction-closure-",
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "closure_hash": self.closure_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "controller_snapshot_hash": self.controller_snapshot_hash,
            "terminal_mask_hash": self.terminal_mask_hash,
            "acquisition_receipt_hashes": self.acquisition_receipt_hashes,
            "extraction_state_hash": self.extraction_state_hash,
            "extraction_calibration_hash": self.extraction_calibration_hash,
            "validation_independent": True,
            "terminal_rows_visible": False,
        }


TerminalExtractionFinalizer = Callable[
    [V1ControllerState],
    V1TerminalExtractionClosure,
]


@dataclass(frozen=True, slots=True)
class V1ProductionTerminalValidationFamilyDesign:
    """Pre-outcome owner for one dense/matched-density/final reveal.

    The design binds the complete validation population/order and metric
    semantics before the pruning trajectory opens them.  Exact masks and
    affine fits do not exist at bundle admission time; production recovers
    those later only from the immutable Pareto archive and the accepted
    train-only oracle trajectory.
    """

    terminal_validation_population_hash: str
    terminal_validation_context_ids_hash: str
    terminal_validation_context_count: int
    terminal_metric_semantics_hash: str
    terminal_evaluator_identity_hash: str
    dense_reference_hash: str
    validation_independent: bool = True
    terminal_rows_visible: bool = False
    terminal_outcomes_used_for_selection: bool = False
    single_common_panel_required: bool = True
    schema: str = _TERMINAL_VALIDATION_FAMILY_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "terminal_validation_population_hash",
            "terminal_validation_context_ids_hash",
            "terminal_metric_semantics_hash",
            "terminal_evaluator_identity_hash",
            "dense_reference_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "terminal_validation_context_count",
            _positive(
                self.terminal_validation_context_count,
                "terminal_validation_context_count",
            ),
        )
        if (
            self.validation_independent is not True
            or self.terminal_rows_visible is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.single_common_panel_required is not True
            or self.schema != _TERMINAL_VALIDATION_FAMILY_DESIGN_SCHEMA
        ):
            raise ValueError(
                "production terminal-family design crossed its pre-outcome firewall"
            )

    @property
    def design_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-terminal-family-design-",
        )

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class V1ProductionTerminalComparisonFamilyDesignV2:
    """Outcome-free dense/current-v0/final-v1 terminal-panel contract.

    ``pending_dense_reference_design_hash`` identifies what will be measured;
    ``dense_mask_hash`` identifies the concrete hard mask used to measure it.
    The two coordinates are deliberately distinct.  No terminal scalar,
    tolerance, or historical validation artifact belongs to this design.
    """

    terminal_validation_population_hash: str
    terminal_validation_context_ids_hash: str
    terminal_validation_context_count: int
    terminal_metric_semantics_hash: str
    terminal_evaluator_identity_hash: str
    pending_dense_reference_design_hash: str
    dense_mask_hash: str
    current_v0_mask_hash: str
    current_v0_mask_artifact_sha256: str
    current_v0_mask_artifact_semantic_hash: str
    current_v0_mask_artifact_byte_count: int
    current_v0_retained_route_count: int
    dense_current_v0_calibration_family_freeze_hash: str
    dense_calibration_freeze_hash: str
    current_v0_calibration_freeze_hash: str
    terminal_roles: tuple[str, str, str] = (
        "dense_reference",
        "current_v0_comparator",
        "terminal_v1",
    )
    dense_reference_mode: str = "pending_joint_terminal_measurement_v2"
    validation_independent: bool = True
    outcome_fields_absent: bool = True
    terminal_rows_visible: bool = False
    terminal_outcomes_opened: bool = False
    terminal_outcomes_used_for_selection: bool = False
    single_common_panel_required: bool = True
    historical_validation_artifact_loaded: bool = False
    pareto_milestone_required: bool = False
    schema: str = _TERMINAL_COMPARISON_FAMILY_DESIGN_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "terminal_validation_population_hash",
            "terminal_validation_context_ids_hash",
            "terminal_metric_semantics_hash",
            "terminal_evaluator_identity_hash",
            "pending_dense_reference_design_hash",
            "dense_mask_hash",
            "current_v0_mask_hash",
            "current_v0_mask_artifact_semantic_hash",
            "dense_current_v0_calibration_family_freeze_hash",
            "dense_calibration_freeze_hash",
            "current_v0_calibration_freeze_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        context_count = _positive(
            self.terminal_validation_context_count,
            "terminal_validation_context_count",
        )
        artifact_sha256 = _sha256_hex(
            self.current_v0_mask_artifact_sha256,
            "current_v0_mask_artifact_sha256",
        )
        artifact_bytes = _positive(
            self.current_v0_mask_artifact_byte_count,
            "current_v0_mask_artifact_byte_count",
        )
        retained = _positive(
            self.current_v0_retained_route_count,
            "current_v0_retained_route_count",
        )
        if (
            tuple(self.terminal_roles)
            != ("dense_reference", "current_v0_comparator", "terminal_v1")
            or self.current_v0_mask_artifact_semantic_hash
            != self.current_v0_mask_hash
            or self.current_v0_mask_hash == self.dense_mask_hash
            or self.dense_reference_mode
            != "pending_joint_terminal_measurement_v2"
            or self.validation_independent is not True
            or self.outcome_fields_absent is not True
            or self.terminal_rows_visible is not False
            or self.terminal_outcomes_opened is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.single_common_panel_required is not True
            or self.historical_validation_artifact_loaded is not False
            or self.pareto_milestone_required is not False
            or self.schema != _TERMINAL_COMPARISON_FAMILY_DESIGN_SCHEMA_V2
        ):
            raise ValueError(
                "production terminal-comparison design crossed its pre-outcome firewall"
            )
        object.__setattr__(self, "terminal_validation_context_count", context_count)
        object.__setattr__(self, "current_v0_mask_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "current_v0_mask_artifact_byte_count", artifact_bytes)
        object.__setattr__(self, "current_v0_retained_route_count", retained)
        object.__setattr__(
            self,
            "terminal_roles",
            ("dense_reference", "current_v0_comparator", "terminal_v1"),
        )

    @property
    def design_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-terminal-comparison-design-v2-",
        )

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class V1ProductionTerminalValidationFamilyExecution:
    """Exact masks and train-only affine states supplied to one backend call."""

    request: V1TerminalValidationFamilyRequestReceipt
    unique_lane_masks: tuple[HardMask, ...]
    unique_lane_calibrations: tuple[V1FrozenAffineCalibration, ...]
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt
    pareto_milestone_archive_checkpoint_receipt_hash: str
    pareto_milestone_archive_runtime_state_hash: str
    validation_rows_opened: bool = False
    schema: str = _TERMINAL_VALIDATION_FAMILY_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.request) is not V1TerminalValidationFamilyRequestReceipt:
            raise TypeError("terminal-family execution requires its exact request")
        self.request.__post_init__()
        checkpoint = self.request_checkpoint_receipt
        if type(checkpoint) is not V1RuntimeCheckpointReceipt:
            raise TypeError(
                "terminal-family execution requires its exact request checkpoint"
            )
        checkpoint.__post_init__()
        if (
            checkpoint.terminal_validation_family_freeze_hash
            != self.request.family_freeze.receipt_hash
            or checkpoint.terminal_validation_family_request_hash
            != self.request.request_hash
            or checkpoint.terminal_validation_family_artifact is not None
        ):
            raise PermissionError(
                "terminal-family execution lacks an outcome-free request checkpoint"
            )
        masks = tuple(self.unique_lane_masks)
        calibrations = tuple(self.unique_lane_calibrations)
        freeze = self.request.family_freeze
        if (
            len(masks) != len(calibrations)
            or len(masks) != len(freeze.unique_lane_mask_hashes)
            or any(type(mask) is not HardMask for mask in masks)
            or any(
                type(calibration) is not V1FrozenAffineCalibration
                for calibration in calibrations
            )
        ):
            raise TypeError(
                "terminal-family execution requires one exact mask/calibration per lane"
            )
        for index, (mask, calibration) in enumerate(
            zip(masks, calibrations, strict=True)
        ):
            if (
                mask.mask_hash != freeze.unique_lane_mask_hashes[index]
                or mask.route_axis_hash != freeze.route_axis_hash
                or mask.route_count != freeze.route_count
                or calibration.mask_hash != mask.mask_hash
                or calibration.calibration_hash
                != freeze.unique_lane_calibration_hashes[index]
                or calibration.frozen_calibration_hash
                != freeze.unique_lane_calibration_freeze_hashes[index]
                or dict(calibration.to_payload())
                != dict(freeze.unique_lane_calibration_freeze_payloads[index])
            ):
                raise ValueError(
                    "terminal-family execution lane differs from its frozen request"
                )
        role_counts = tuple(
            masks[lane].retained_count for lane in freeze.role_to_unique_lane
        )
        if role_counts != freeze.role_retained_route_counts:
            raise ValueError(
                "terminal-family execution masks differ from frozen role counts"
            )
        object.__setattr__(
            self,
            "pareto_milestone_archive_checkpoint_receipt_hash",
            _nonempty(
                self.pareto_milestone_archive_checkpoint_receipt_hash,
                "pareto_milestone_archive_checkpoint_receipt_hash",
            ),
        )
        object.__setattr__(
            self,
            "pareto_milestone_archive_runtime_state_hash",
            _nonempty(
                self.pareto_milestone_archive_runtime_state_hash,
                "pareto_milestone_archive_runtime_state_hash",
            ),
        )
        if (
            self.validation_rows_opened is not False
            or self.schema != _TERMINAL_VALIDATION_FAMILY_EXECUTION_SCHEMA
        ):
            raise ValueError("terminal-family execution was constructed post-outcome")
        object.__setattr__(self, "unique_lane_masks", masks)
        object.__setattr__(self, "unique_lane_calibrations", calibrations)


@dataclass(frozen=True, slots=True)
class V1ProductionTerminalComparisonFamilyExecutionV2:
    """Exact three-role material supplied to one v2 common-panel invocation."""

    request: V1TerminalComparisonFamilyRequestReceiptV2
    unique_lane_masks: tuple[HardMask, ...]
    unique_lane_calibrations: tuple[V1FrozenAffineCalibration, ...]
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt
    current_v0_mask_artifact_sha256: str
    current_v0_mask_artifact_semantic_hash: str
    current_v0_mask_artifact_byte_count: int
    validation_rows_opened: bool = False
    schema: str = _TERMINAL_COMPARISON_FAMILY_EXECUTION_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.request) is not V1TerminalComparisonFamilyRequestReceiptV2:
            raise TypeError("terminal comparison execution requires its exact v2 request")
        self.request.__post_init__()
        checkpoint = self.request_checkpoint_receipt
        if type(checkpoint) is not V1RuntimeCheckpointReceipt:
            raise TypeError(
                "terminal comparison execution requires its exact request checkpoint"
            )
        checkpoint.__post_init__()
        if (
            checkpoint.terminal_comparison_family_freeze_hash
            != self.request.family_freeze.receipt_hash
            or checkpoint.terminal_comparison_family_request_hash
            != self.request.request_hash
            or checkpoint.terminal_comparison_family_artifact is not None
        ):
            raise PermissionError(
                "terminal comparison execution lacks an outcome-free request checkpoint"
            )
        masks = tuple(self.unique_lane_masks)
        calibrations = tuple(self.unique_lane_calibrations)
        freeze = self.request.family_freeze
        if (
            len(masks) != len(calibrations)
            or len(masks) != len(freeze.unique_lane_mask_hashes)
            or any(type(mask) is not HardMask for mask in masks)
            or any(
                type(calibration) is not V1FrozenAffineCalibration
                for calibration in calibrations
            )
        ):
            raise TypeError(
                "terminal comparison execution requires one exact mask/calibration per lane"
            )
        for index, (mask, calibration) in enumerate(
            zip(masks, calibrations, strict=True)
        ):
            if (
                mask.mask_hash != freeze.unique_lane_mask_hashes[index]
                or mask.route_axis_hash != freeze.route_axis_hash
                or mask.route_count != freeze.route_count
                or calibration.mask_hash != mask.mask_hash
                or calibration.calibration_hash
                != freeze.unique_lane_calibration_hashes[index]
                or calibration.frozen_calibration_hash
                != freeze.unique_lane_calibration_freeze_hashes[index]
                or dict(calibration.to_payload())
                != dict(freeze.unique_lane_calibration_freeze_payloads[index])
            ):
                raise ValueError(
                    "terminal comparison execution lane differs from its frozen request"
                )
        role_counts = tuple(
            masks[lane].retained_count for lane in freeze.role_to_unique_lane
        )
        if role_counts != freeze.role_retained_route_counts:
            raise ValueError(
                "terminal comparison execution masks differ from frozen role counts"
            )
        artifact_sha256 = _sha256_hex(
            self.current_v0_mask_artifact_sha256,
            "current_v0_mask_artifact_sha256",
        )
        artifact_semantic = _nonempty(
            self.current_v0_mask_artifact_semantic_hash,
            "current_v0_mask_artifact_semantic_hash",
        )
        artifact_bytes = _positive(
            self.current_v0_mask_artifact_byte_count,
            "current_v0_mask_artifact_byte_count",
        )
        current_v0_lane = freeze.lane_for_role("current_v0_comparator")
        if (
            artifact_sha256 != freeze.current_v0_mask_artifact_sha256
            or artifact_semantic != freeze.current_v0_mask_artifact_semantic_hash
            or artifact_bytes != freeze.current_v0_mask_artifact_byte_count
            or hashlib.sha256(masks[current_v0_lane].packed_bits).hexdigest()
            != artifact_sha256
            or len(masks[current_v0_lane].packed_bits) != artifact_bytes
            or masks[current_v0_lane].mask_hash != artifact_semantic
        ):
            raise ValueError(
                "terminal comparison current-v0 bytes differ from the packed artifact binding"
            )
        if (
            self.validation_rows_opened is not False
            or self.schema != _TERMINAL_COMPARISON_FAMILY_EXECUTION_SCHEMA_V2
        ):
            raise ValueError("terminal comparison execution was constructed post-outcome")
        object.__setattr__(self, "unique_lane_masks", masks)
        object.__setattr__(self, "unique_lane_calibrations", calibrations)
        object.__setattr__(self, "current_v0_mask_artifact_sha256", artifact_sha256)
        object.__setattr__(
            self,
            "current_v0_mask_artifact_semantic_hash",
            artifact_semantic,
        )
        object.__setattr__(
            self,
            "current_v0_mask_artifact_byte_count",
            artifact_bytes,
        )


@dataclass(frozen=True, slots=True)
class _BoundProductionTerminalFamilyValidator:
    validator: V1ProductionCallableSeal
    execution: V1ProductionTerminalValidationFamilyExecution

    def __call__(
        self,
        request: V1TerminalValidationFamilyRequestReceipt,
    ) -> V1TerminalValidationFamilyMatrixOutcome:
        if (
            type(request) is not V1TerminalValidationFamilyRequestReceipt
            or request.request_hash != self.execution.request.request_hash
        ):
            raise ValueError("terminal-family backend received another frozen request")
        outcome = self.validator(self.execution)
        if type(outcome) is not V1TerminalValidationFamilyMatrixOutcome:
            raise TypeError("terminal-family backend returned another outcome type")
        return outcome


@dataclass(frozen=True, slots=True)
class _BoundProductionTerminalComparisonFamilyValidatorV2:
    validator: V1ProductionCallableSeal
    execution: V1ProductionTerminalComparisonFamilyExecutionV2

    def __call__(
        self,
        request: V1TerminalComparisonFamilyRequestReceiptV2,
    ) -> V1TerminalComparisonFamilyMatrixOutcomeV2:
        if (
            type(request) is not V1TerminalComparisonFamilyRequestReceiptV2
            or request.request_hash != self.execution.request.request_hash
        ):
            raise ValueError(
                "terminal comparison backend received another frozen request"
            )
        outcome = self.validator(self.execution)
        if type(outcome) is not V1TerminalComparisonFamilyMatrixOutcomeV2:
            raise TypeError("terminal comparison backend returned another outcome type")
        return outcome


def _validate_terminal_extraction_closure(
    controller: V1ControllerState,
    closure: V1TerminalExtractionClosure,
) -> None:
    if type(controller) is not V1ControllerState or controller.phase != "stopped":
        raise PermissionError(
            "terminal extraction closure requires the exact stopped controller"
        )
    if type(closure) is not V1TerminalExtractionClosure:
        raise TypeError("terminal extraction finalizer returned another closure type")
    snapshot = controller.snapshot_receipt()
    expected = {
        "campaign_contract_hash": controller.binding.contract.campaign_contract_hash,
        "campaign_binding_hash": controller.binding.binding_hash,
        "controller_run_id": controller.controller_run_id,
        "controller_snapshot_hash": snapshot.receipt_hash,
        "terminal_mask_hash": controller.current_mask.mask_hash,
        "acquisition_receipt_hashes": controller.acquisition_receipt_hashes,
    }
    for name, value in expected.items():
        if getattr(closure, name) != value:
            raise ValueError(f"terminal extraction closure disagrees on {name}")


def _active_paired_oracle_evaluator(
    provider: PairedMaskOracleProvider,
) -> Callable[..., Any]:
    legacy_evaluator = provider.evaluator
    parent_aware_evaluator = provider.parent_aware_evaluator
    if (legacy_evaluator is None) == (parent_aware_evaluator is None):
        raise ValueError("paired-oracle provider does not have exactly one evaluator")
    evaluator = (
        parent_aware_evaluator
        if parent_aware_evaluator is not None
        else legacy_evaluator
    )
    if evaluator is None:  # pragma: no cover - guarded above
        raise RuntimeError("paired-oracle provider lost its active evaluator")
    return evaluator


def _sealed_evaluator_hook(
    *,
    evaluator: Callable[..., Any],
    evaluator_seal: V1ProductionCallableSeal,
    method_name: str,
    label: str,
    require_bound_to_evaluator: bool = False,
) -> tuple[Callable[..., object], str] | None:
    declared = inspect.getattr_static(
        type(evaluator),
        method_name,
        None,
    )
    observed = getattr(evaluator, method_name, None)
    if declared is None:
        if observed is not None:
            raise ValueError(
                f"paired-oracle evaluator has an unsealed instance {label} hook"
            )
        return None
    if not callable(observed):
        raise TypeError(f"paired-oracle evaluator {label} hook is not callable")
    if require_bound_to_evaluator and getattr(observed, "__self__", None) is not evaluator:
        raise ValueError(
            f"paired-oracle evaluator {label} hook is bound to another evaluator"
        )
    callable_spec, source_sha256 = _observed_callable_source(observed)
    if source_sha256 != evaluator_seal.source_file_sha256:
        raise ValueError(
            f"paired-oracle evaluator {label} hook is outside its source seal"
        )
    return observed, callable_spec


def _expected_sealed_evaluator_hook(
    *,
    evaluator: Callable[..., Any],
    evaluator_seal: V1ProductionCallableSeal,
    method_name: str,
    label: str,
    expected_spec: str | None,
    require_bound_to_evaluator: bool = False,
) -> tuple[Callable[..., object], str] | None:
    observed = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=evaluator_seal,
        method_name=method_name,
        label=label,
        require_bound_to_evaluator=require_bound_to_evaluator,
    )
    if expected_spec is None:
        if observed is not None:
            raise ValueError(
                f"paired-oracle evaluator {label} hook appeared after seal"
            )
        return None
    if observed is None:
        raise ValueError(
            f"paired-oracle evaluator {label} hook disappeared after seal"
        )
    if observed[1] != expected_spec:
        raise ValueError(
            f"paired-oracle evaluator {label} hook callable spec changed after seal"
        )
    return observed


def _sealed_evaluator_state_restoration_hook(
    *,
    evaluator: Callable[..., Any],
    evaluator_seal: V1ProductionCallableSeal,
) -> tuple[Callable[[V1ControllerState], object], str] | None:
    observed = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=evaluator_seal,
        method_name=_EVALUATOR_STATE_RESTORATION_METHOD,
        label="restoration",
    )
    return observed  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class V1ProductionPairedOracleSeal:
    """Content-address the exact paired direct-increment provider composition."""

    provider: PairedMaskOracleProvider = field(repr=False, compare=False)
    evaluator_seal: V1ProductionCallableSeal
    context_population_hash: str
    provider_context_ids_hash: str = field(init=False)
    calibration_identity: str = field(init=False)
    realized_cost_hash: str = field(init=False)
    evaluator_kind: Literal["legacy", "parent_aware"] = field(init=False)
    evaluator_state_restoration_hook_spec: str | None = field(init=False)
    evaluator_candidate_calibration_freeze_hook_spec: str | None = field(init=False)
    evaluator_candidate_calibration_verify_hook_spec: str | None = field(init=False)
    evaluator_oracle_attempt_order_receipts_hook_spec: str | None = field(init=False)
    evaluator_active_dynamic_oracle_frame_hook_spec: str | None = field(init=False)
    evaluator_exact_k_paired_bce_hook_spec: str | None = field(init=False)
    evaluator_request_aware_paired_bce_hook_spec: str | None = field(init=False)
    evaluator_retire_committed_generation_state_hook_spec: str | None = field(
        init=False
    )
    schema: str = _PAIRED_ORACLE_SEAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.provider) is not PairedMaskOracleProvider:
            raise TypeError("paired-oracle seal requires an exact provider")
        if type(self.evaluator_seal) is not V1ProductionCallableSeal:
            raise TypeError("paired-oracle seal requires an exact evaluator seal")
        if self.evaluator_seal.role != "direct_oracle_evaluator":
            raise ValueError("paired-oracle evaluator seal has another role")
        legacy_evaluator = self.provider.evaluator
        parent_aware_evaluator = self.provider.parent_aware_evaluator
        active_evaluator = _active_paired_oracle_evaluator(self.provider)
        if self.evaluator_seal.callable_value is not active_evaluator:
            raise ValueError("paired-oracle seal names another evaluator object")
        object.__setattr__(
            self,
            "evaluator_kind",
            "parent_aware" if parent_aware_evaluator is not None else "legacy",
        )
        restoration_hook = _sealed_evaluator_state_restoration_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
        )
        object.__setattr__(
            self,
            "evaluator_state_restoration_hook_spec",
            None if restoration_hook is None else restoration_hook[1],
        )
        calibration_freeze_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name=_EVALUATOR_CANDIDATE_CALIBRATION_FREEZE_METHOD,
            label="candidate-calibration-freeze",
        )
        calibration_verify_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name=_EVALUATOR_CANDIDATE_CALIBRATION_VERIFY_METHOD,
            label="candidate-calibration-verify",
        )
        if (calibration_freeze_hook is None) != (calibration_verify_hook is None):
            raise ValueError(
                "paired-oracle evaluator calibration freeze/verify hooks must be paired"
            )
        object.__setattr__(
            self,
            "evaluator_candidate_calibration_freeze_hook_spec",
            None if calibration_freeze_hook is None else calibration_freeze_hook[1],
        )
        object.__setattr__(
            self,
            "evaluator_candidate_calibration_verify_hook_spec",
            None if calibration_verify_hook is None else calibration_verify_hook[1],
        )
        order_receipts_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name=_EVALUATOR_ORACLE_ATTEMPT_ORDER_RECEIPTS_METHOD,
            label="oracle-attempt-order-receipts",
        )
        object.__setattr__(
            self,
            "evaluator_oracle_attempt_order_receipts_hook_spec",
            None if order_receipts_hook is None else order_receipts_hook[1],
        )
        dynamic_frame_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name=_EVALUATOR_ACTIVE_DYNAMIC_ORACLE_FRAME_METHOD,
            label="active-dynamic-oracle-frame",
        )
        object.__setattr__(
            self,
            "evaluator_active_dynamic_oracle_frame_hook_spec",
            None if dynamic_frame_hook is None else dynamic_frame_hook[1],
        )
        exact_k_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name="evaluate_paired_bce_calibration_prefix_rows",
            label="exact-k-paired-bce-calibration-prefix",
            require_bound_to_evaluator=True,
        )
        object.__setattr__(
            self,
            "evaluator_exact_k_paired_bce_hook_spec",
            None if exact_k_hook is None else exact_k_hook[1],
        )
        request_aware_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name="evaluate_paired_bce_chunk_request",
            label="request-aware-paired-bce-chunk",
            require_bound_to_evaluator=True,
        )
        object.__setattr__(
            self,
            "evaluator_request_aware_paired_bce_hook_spec",
            None if request_aware_hook is None else request_aware_hook[1],
        )
        retirement_hook = _sealed_evaluator_hook(
            evaluator=active_evaluator,
            evaluator_seal=self.evaluator_seal,
            method_name=_EVALUATOR_RETIRE_COMMITTED_GENERATION_STATE_METHOD,
            label="committed-generation-retirement",
        )
        object.__setattr__(
            self,
            "evaluator_retire_committed_generation_state_hook_spec",
            None if retirement_hook is None else retirement_hook[1],
        )
        object.__setattr__(
            self,
            "context_population_hash",
            _nonempty(self.context_population_hash, "context_population_hash"),
        )
        object.__setattr__(
            self,
            "provider_context_ids_hash",
            content_hash(
                self.provider.context_ids,
                prefix="partition-free-v1-provider-context-ids-",
            ),
        )
        object.__setattr__(
            self,
            "calibration_identity",
            _nonempty(self.provider.calibration_state_id, "provider calibration identity"),
        )
        object.__setattr__(
            self,
            "realized_cost_hash",
            content_hash(
                self.provider.realized_cost,
                prefix="partition-free-v1-provider-realized-cost-",
            ),
        )
        if self.schema != _PAIRED_ORACLE_SEAL_SCHEMA:
            raise ValueError("unsupported modern-v1 paired-oracle seal schema")

    @property
    def provider_hash(self) -> str:
        payload: dict[str, object] = {
            "schema": self.schema,
            "evaluator_identity_hash": self.evaluator_seal.identity_hash,
            "provider_context_ids_hash": self.provider_context_ids_hash,
            "context_population_hash": self.context_population_hash,
            "calibration_identity": self.calibration_identity,
            "realized_cost_hash": self.realized_cost_hash,
            "evaluator_kind": self.evaluator_kind,
            "evaluator_state_restoration_hook_spec": (
                self.evaluator_state_restoration_hook_spec
            ),
            "evaluator_candidate_calibration_freeze_hook_spec": (
                self.evaluator_candidate_calibration_freeze_hook_spec
            ),
            "evaluator_candidate_calibration_verify_hook_spec": (
                self.evaluator_candidate_calibration_verify_hook_spec
            ),
            "paired_common_contexts": True,
            "candidate_damage_rows_direct": True,
        }
        if self.evaluator_oracle_attempt_order_receipts_hook_spec is not None:
            payload["evaluator_oracle_attempt_order_receipts_hook_spec"] = (
                self.evaluator_oracle_attempt_order_receipts_hook_spec
            )
        if self.evaluator_active_dynamic_oracle_frame_hook_spec is not None:
            payload["evaluator_active_dynamic_oracle_frame_hook_spec"] = (
                self.evaluator_active_dynamic_oracle_frame_hook_spec
            )
        if self.evaluator_exact_k_paired_bce_hook_spec is not None:
            payload["evaluator_exact_k_paired_bce_hook_spec"] = (
                self.evaluator_exact_k_paired_bce_hook_spec
            )
        if self.evaluator_request_aware_paired_bce_hook_spec is not None:
            payload["evaluator_request_aware_paired_bce_hook_spec"] = (
                self.evaluator_request_aware_paired_bce_hook_spec
            )
        if self.evaluator_retire_committed_generation_state_hook_spec is not None:
            payload["evaluator_retire_committed_generation_state_hook_spec"] = (
                self.evaluator_retire_committed_generation_state_hook_spec
            )
        return content_hash(payload, prefix="partition-free-v1-paired-oracle-provider-")


def seal_v1_paired_oracle_provider(
    *,
    provider: PairedMaskOracleProvider,
    context_population_hash: str,
    evaluator_configuration_hash: str,
) -> V1ProductionPairedOracleSeal:
    if type(provider) is not PairedMaskOracleProvider:
        raise TypeError("paired-oracle sealing requires an exact provider")
    evaluator_value = _active_paired_oracle_evaluator(provider)
    evaluator = seal_v1_production_callable(
        role="direct_oracle_evaluator",
        callable_value=evaluator_value,
        configuration_hash=evaluator_configuration_hash,
    )
    return V1ProductionPairedOracleSeal(
        provider=provider,
        evaluator_seal=evaluator,
        context_population_hash=context_population_hash,
    )


@dataclass(frozen=True, slots=True)
class V1ProductionRuntimeManifest:
    """Pre-outcome admission record for one concrete backend composition."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    backend_kind: BackendKind
    proposal_protocol: ProposalProtocolKind
    backend_manifest_hash: str
    study_contract_hash: str
    measurement_contract_hash: str
    model_data_seal_hash: str
    route_axis_hash: str
    initial_mask_hash: str
    runtime_logical_config_hash: str
    runtime_physical_config_hash: str
    entropy_tape_hash: str
    acquisition_factory_hash: str | None
    proposal_backend_identity_hash: str | None
    threshold_factory_hash: str
    sampling_law_factory_hash: str
    direct_oracle_provider_hash: str
    provider_context_ids_hash: str
    provider_context_population_hash: str
    provider_calibration_identity: str
    terminal_validator_hash: str
    extraction_state_hash: str | None
    extraction_calibration_hash: str | None
    terminal_extraction_finalizer_hash: str | None = None
    performance_factory_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    genuine_partition_free_controller: bool = True
    direct_rule_over_v0_adapter: bool = False
    retired_trust_walk: bool = False
    selector_fail_closed: bool = True
    claim_execution_certified: bool = False
    schema: str = _MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "backend_manifest_hash",
            "study_contract_hash",
            "measurement_contract_hash",
            "model_data_seal_hash",
            "route_axis_hash",
            "initial_mask_hash",
            "runtime_logical_config_hash",
            "runtime_physical_config_hash",
            "entropy_tape_hash",
            "threshold_factory_hash",
            "sampling_law_factory_hash",
            "direct_oracle_provider_hash",
            "provider_context_ids_hash",
            "provider_context_population_hash",
            "provider_calibration_identity",
            "terminal_validator_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.backend_kind not in {"synthetic_fixture", "alexnet_production"}:
            raise ValueError("production manifest has an unsupported backend kind")
        if self.proposal_protocol not in {
            "legacy_synthetic_one_shot_v1",
            "operator_binary_d_c_v1",
            "staged_train_only_proposal_v1",
        }:
            raise ValueError("production manifest has an unsupported proposal protocol")
        if self.proposal_protocol in {"legacy_synthetic_one_shot_v1","operator_binary_d_c_v1"}:
            if self.proposal_protocol == "legacy_synthetic_one_shot_v1" and self.backend_kind != "synthetic_fixture":
                raise ValueError(
                    "legacy one-shot proposal authority is restricted to explicit "
                    "synthetic fixtures"
                )
            object.__setattr__(
                self,
                "acquisition_factory_hash",
                _nonempty(self.acquisition_factory_hash, "acquisition_factory_hash"),
            )
            if self.proposal_backend_identity_hash is not None:
                raise ValueError("legacy proposal manifest must not name a staged backend")
        else:
            if self.acquisition_factory_hash is not None:
                raise ValueError(
                    "staged proposal manifest forbids executable one-shot authority"
                )
            object.__setattr__(
                self,
                "proposal_backend_identity_hash",
                _nonempty(
                    self.proposal_backend_identity_hash,
                    "proposal_backend_identity_hash",
                ),
            )
        if self.performance_factory_hash is not None:
            object.__setattr__(
                self,
                "performance_factory_hash",
                _nonempty(self.performance_factory_hash, "performance_factory_hash"),
            )
        if self.terminal_extraction_finalizer_hash is None:
            object.__setattr__(
                self,
                "extraction_state_hash",
                _nonempty(self.extraction_state_hash, "extraction_state_hash"),
            )
            object.__setattr__(
                self,
                "extraction_calibration_hash",
                _nonempty(
                    self.extraction_calibration_hash,
                    "extraction_calibration_hash",
                ),
            )
        else:
            object.__setattr__(
                self,
                "terminal_extraction_finalizer_hash",
                _nonempty(
                    self.terminal_extraction_finalizer_hash,
                    "terminal_extraction_finalizer_hash",
                ),
            )
            if (
                self.extraction_state_hash is not None
                or self.extraction_calibration_hash is not None
            ):
                raise ValueError(
                    "dynamic terminal extraction must not retain static freeze hashes"
                )
        if (
            self.schema != _MANIFEST_SCHEMA
            or self.genuine_partition_free_controller is not True
            or self.direct_rule_over_v0_adapter is not False
            or self.retired_trust_walk is not False
            or self.selector_fail_closed is not True
            or self.claim_execution_certified is not False
        ):
            raise ValueError("production manifest crossed the modern-v1 authority boundary")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "production metadata"))

    @property
    def manifest_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-production-manifest-")


@dataclass(frozen=True, slots=True)
class V1ProductionRuntimeBundle:
    """Exact executable objects admitted by one production manifest.

    Callables are intentionally excluded from the manifest hash.  Their source
    and configuration identities are supplied as sealed hashes and the bundle
    validates all concrete objects that can be checked before any outcome is
    requested.
    """

    manifest: V1ProductionRuntimeManifest
    binding: V1OptionACampaignBinding
    initial_mask: HardMask
    runtime_config: V1RuntimeExecutionConfig
    entropy_tape: V1StagedAlphaEntropyTape
    acquisition_factory: V1ProductionCallableSeal | None
    threshold_factory: V1ProductionCallableSeal
    sampling_law_factory: V1ProductionCallableSeal
    provider: PairedMaskOracleProvider
    provider_seal: V1ProductionPairedOracleSeal
    terminal_validator: V1ProductionCallableSeal
    pareto_milestone_registration: V1ParetoMilestoneRegistration | None = None
    proposal_backend_seal: V1ProductionProposalBackendSeal | None = None
    proposal_initial_state_factory: V1ProductionCallableSeal | None = None
    proposal_forecast_factory: V1ProductionCallableSeal | None = None
    proposal_request_plan_factory: V1ProductionCallableSeal | None = None
    proposal_acquisition_executor: V1ProductionCallableSeal | None = None
    proposal_freeze_evaluator: V1ProductionCallableSeal | None = None
    proposal_acquisition_finalizer: V1ProductionCallableSeal | None = None
    proposal_resume_validator: V1ProductionCallableSeal | None = None
    terminal_extraction_finalizer: V1ProductionCallableSeal | None = None
    performance_receipt_factory: V1ProductionCallableSeal | None = None
    # Optional attempt-level structural disclosure hook (action-decomposition
    # bookkeeping).  Disclosure only: it never enters a decision, the
    # manifest (a hashed family) does not know it, and ``bundle_hash`` binds its
    # identity only when it is present, so bundles without it hash as before.
    attempt_structural_disclosure: V1ProductionCallableSeal | None = None
    schema: str = _BUNDLE_SCHEMA
    terminal_validation_family_design: (
        V1ProductionTerminalValidationFamilyDesign
        | V1ProductionTerminalComparisonFamilyDesignV2
        | None
    ) = None
    terminal_comparison_current_v0_mask: HardMask | None = None
    terminal_comparison_current_v0_calibration_freeze: (
        V1CandidateCalibrationFreeze | None
    ) = None

    def __post_init__(self) -> None:
        if type(self.manifest) is not V1ProductionRuntimeManifest:
            raise TypeError("production bundle requires an exact manifest")
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("production bundle requires an exact campaign binding")
        if type(self.initial_mask) is not HardMask:
            raise TypeError("production bundle requires an exact initial hard mask")
        if type(self.runtime_config) is not V1RuntimeExecutionConfig:
            raise TypeError("production bundle requires an exact runtime config")
        if type(self.entropy_tape) is not V1StagedAlphaEntropyTape:
            raise TypeError("production bundle requires an exact staged-alpha entropy tape")
        if type(self.provider) is not PairedMaskOracleProvider:
            raise TypeError("production bundle requires the exact paired-mask oracle provider")
        if type(self.provider_seal) is not V1ProductionPairedOracleSeal:
            raise TypeError("production bundle requires an exact paired-oracle seal")
        if self.provider_seal.provider is not self.provider:
            raise ValueError("production paired-oracle seal names another provider object")
        required_seals = (
            (self.threshold_factory, "threshold_factory"),
            (self.sampling_law_factory, "sampling_law_factory"),
            (self.terminal_validator, "terminal_validator"),
        )
        for value, role in required_seals:
            if type(value) is not V1ProductionCallableSeal or value.role != role:
                raise TypeError(f"{role} must be an exact role-matched callable seal")
        protocol = self.manifest.proposal_protocol
        if protocol in {"legacy_synthetic_one_shot_v1","operator_binary_d_c_v1"}:
            if (
                type(self.acquisition_factory) is not V1ProductionCallableSeal
                or self.acquisition_factory.role != "acquisition_factory"
            ):
                raise TypeError(
                    "legacy synthetic protocol requires an exact acquisition factory seal"
                )
            if self.proposal_backend_seal is not None:
                raise ValueError("legacy synthetic protocol must not retain a staged backend")
        elif self.acquisition_factory is not None:
            raise ValueError(
                "staged proposal protocol forbids executable one-shot authority"
            )
        if protocol=='operator_binary_d_c_v1':
            _bound_operator_factory(self)
        if self.attempt_structural_disclosure is not None and (
            type(self.attempt_structural_disclosure) is not V1ProductionCallableSeal
            or self.attempt_structural_disclosure.role != "attempt_structural_disclosure"
        ):
            raise TypeError(
                "attempt_structural_disclosure must be an exact role-matched callable seal"
            )
        proposal_components = {
            "proposal_backend_seal": self.proposal_backend_seal,
            "proposal_initial_state_factory": self.proposal_initial_state_factory,
            "proposal_forecast_factory": self.proposal_forecast_factory,
            "proposal_request_plan_factory": self.proposal_request_plan_factory,
            "proposal_acquisition_executor": self.proposal_acquisition_executor,
            "proposal_freeze_evaluator": self.proposal_freeze_evaluator,
            "proposal_acquisition_finalizer": self.proposal_acquisition_finalizer,
        }
        proposal_enabled = protocol == "staged_train_only_proposal_v1"
        if proposal_enabled and any(value is None for value in proposal_components.values()):
            raise ValueError("staged proposal protocol must be supplied as one complete bundle")
        if not proposal_enabled and any(
            value is not None for value in proposal_components.values()
        ):
            raise ValueError("legacy proposal protocol must not retain staged hooks")
        if self.proposal_backend_seal is not None and type(
            self.proposal_backend_seal
        ) is not V1ProductionProposalBackendSeal:
            raise TypeError("proposal_backend_seal has the wrong type")
        for optional_value, optional_role in (
            (self.proposal_initial_state_factory, "proposal_initial_state_factory"),
            (self.proposal_forecast_factory, "proposal_forecast_factory"),
            (self.proposal_request_plan_factory, "proposal_request_plan_factory"),
            (self.proposal_acquisition_executor, "proposal_acquisition_executor"),
            (self.proposal_freeze_evaluator, "proposal_freeze_evaluator"),
            (self.proposal_acquisition_finalizer, "proposal_acquisition_finalizer"),
            (self.proposal_resume_validator, "proposal_resume_validator"),
        ):
            if optional_value is not None and (
                type(optional_value) is not V1ProductionCallableSeal
                or optional_value.role != optional_role
            ):
                raise TypeError(
                    f"{optional_role} must be an exact role-matched callable seal"
                )
        if self.proposal_backend_seal is None:
            if self.proposal_resume_validator is not None:
                raise ValueError(
                    "proposal resume validator lacks its composite backend seal"
                )
        elif self.proposal_backend_seal.resume_validator is not (
            self.proposal_resume_validator
        ):
            raise ValueError(
                "proposal resume-validator identity disagrees with its backend seal"
            )
        if self.performance_receipt_factory is not None and (
            type(self.performance_receipt_factory) is not V1ProductionCallableSeal
            or self.performance_receipt_factory.role != "performance_receipt_factory"
        ):
            raise TypeError("performance_receipt_factory must be an exact role-matched seal")
        if self.terminal_extraction_finalizer is not None and (
            type(self.terminal_extraction_finalizer) is not V1ProductionCallableSeal
            or self.terminal_extraction_finalizer.role
            != "terminal_extraction_finalizer"
        ):
            raise TypeError(
                "terminal_extraction_finalizer must be an exact role-matched seal"
            )
        family_design = self.terminal_validation_family_design
        family_metadata = self.manifest.metadata.get(
            _TERMINAL_VALIDATION_FAMILY_METADATA_KEY
        )
        current_v0_mask = self.terminal_comparison_current_v0_mask
        current_v0_calibration = (
            self.terminal_comparison_current_v0_calibration_freeze
        )
        if family_design is None:
            if family_metadata is not None:
                raise ValueError(
                    "production manifest registers a terminal family without its "
                    "typed design"
                )
            if current_v0_mask is not None or current_v0_calibration is not None:
                raise ValueError(
                    "production bundle carries terminal-comparison material without "
                    "its v2 design"
                )
        else:
            if type(family_design) not in (
                V1ProductionTerminalValidationFamilyDesign,
                V1ProductionTerminalComparisonFamilyDesignV2,
            ):
                raise TypeError("production terminal-family design has the wrong type")
            family_design.__post_init__()
            if (
                not isinstance(family_metadata, Mapping)
                or dict(family_metadata) != dict(family_design.to_payload())
            ):
                raise ValueError(
                    "production manifest and terminal-family design differ"
                )
            if (
                family_design.terminal_evaluator_identity_hash
                != self.terminal_validator.identity_hash
                or family_design.terminal_validation_population_hash
                != self.binding.contract.terminal_validation_population_hash
                or self.initial_mask.retained_count != self.initial_mask.route_count
            ):
                raise ValueError(
                    "terminal-family design differs from evaluator/campaign/dense mask"
                )
            if type(family_design) is V1ProductionTerminalValidationFamilyDesign:
                if (
                    family_design.dense_reference_hash
                    != self.binding.contract.dense_reference_hash
                ):
                    raise ValueError(
                        "legacy terminal-family design names another dense reference"
                    )
                if self.pareto_milestone_registration is None:
                    raise ValueError(
                        "terminal-family production requires a registered Pareto milestone"
                    )
                if not self.has_paired_bce_oracle_protocol:
                    raise PermissionError(
                        "terminal-family production requires persisted paired-BCE "
                        "calibration authority"
                    )
                if current_v0_mask is not None or current_v0_calibration is not None:
                    raise ValueError(
                        "legacy terminal-family production cannot carry v2 comparator material"
                    )
            else:
                if not self.has_paired_bce_oracle_protocol:
                    raise PermissionError(
                        "terminal-family production requires persisted paired-BCE "
                        "calibration authority"
                    )
                if (
                    current_v0_mask is None
                    or type(current_v0_mask) is not HardMask
                    or current_v0_calibration is None
                    or type(current_v0_calibration)
                    is not V1CandidateCalibrationFreeze
                ):
                    raise TypeError(
                        "terminal-comparison v2 requires its exact current-v0 mask "
                        "and train-only calibration freeze"
                    )
                current_v0_calibration.__post_init__()
                dense_calibration = current_v0_calibration.dense_calibration
                v0_calibration = current_v0_calibration.candidate_calibration
                if (
                    family_design.pending_dense_reference_design_hash
                    != self.binding.contract.dense_reference_hash
                    or family_design.dense_mask_hash != self.initial_mask.mask_hash
                    or current_v0_mask.route_axis_hash
                    != self.initial_mask.route_axis_hash
                    or current_v0_mask.route_count != self.initial_mask.route_count
                    or current_v0_mask.mask_hash
                    != family_design.current_v0_mask_hash
                    or current_v0_mask.retained_count
                    != family_design.current_v0_retained_route_count
                    or len(current_v0_mask.packed_bits)
                    != family_design.current_v0_mask_artifact_byte_count
                    or hashlib.sha256(current_v0_mask.packed_bits).hexdigest()
                    != family_design.current_v0_mask_artifact_sha256
                    or current_v0_calibration.freeze_hash
                    != family_design.dense_current_v0_calibration_family_freeze_hash
                    or current_v0_calibration.dense_referent_mask_hash
                    != self.initial_mask.mask_hash
                    or current_v0_calibration.candidate_mask_hash
                    != current_v0_mask.mask_hash
                    or dense_calibration.mask_hash != self.initial_mask.mask_hash
                    or v0_calibration.mask_hash != current_v0_mask.mask_hash
                    or dense_calibration.frozen_calibration_hash
                    != family_design.dense_calibration_freeze_hash
                    or v0_calibration.frozen_calibration_hash
                    != family_design.current_v0_calibration_freeze_hash
                    or current_v0_calibration.validation_independent is not True
                    or dense_calibration.terminal_rows_used_for_fit is not False
                    or v0_calibration.terminal_rows_used_for_fit is not False
                ):
                    raise ValueError(
                        "terminal-comparison v2 differs from its pending dense/current-v0 "
                        "mask or train-only calibration owners"
                    )
        contract = self.binding.contract
        manifest = self.manifest
        milestone = self.pareto_milestone_registration
        milestone_metadata = manifest.metadata.get("pareto_milestone_registration")
        if milestone is None:
            if milestone_metadata is not None:
                raise ValueError(
                    "production manifest registers a Pareto milestone without its typed object"
                )
        else:
            if type(milestone) is not V1ParetoMilestoneRegistration:
                raise TypeError("production Pareto milestone registration has the wrong type")
            if not isinstance(milestone_metadata, Mapping) or dict(
                milestone_metadata
            ) != dict(milestone.to_payload()):
                raise ValueError(
                    "production manifest and typed Pareto milestone registration differ"
                )
            if (
                milestone.campaign_contract_hash != contract.campaign_contract_hash
                or milestone.route_axis_hash != self.initial_mask.route_axis_hash
                or milestone.route_count != self.initial_mask.route_count
                or self.initial_mask.retained_count
                <= milestone.comparator_retained_route_count
                or self.initial_mask.retained_count
                - milestone.comparator_retained_route_count
                > contract.maximum_removed_count
            ):
                raise ValueError(
                    "production Pareto milestone lies outside the admitted campaign axis"
                )
        expected_context_ids_hash = content_hash(
            self.provider.context_ids,
            prefix="partition-free-v1-provider-context-ids-",
        )
        checks = {
            "campaign_contract_hash": contract.campaign_contract_hash,
            "campaign_binding_hash": self.binding.binding_hash,
            "route_axis_hash": self.initial_mask.route_axis_hash,
            "initial_mask_hash": self.initial_mask.mask_hash,
            "runtime_logical_config_hash": self.runtime_config.logical_identity_hash,
            "runtime_physical_config_hash": self.runtime_config.physical_execution_hash,
            "entropy_tape_hash": self.entropy_tape.tape_hash,
            "provider_context_ids_hash": expected_context_ids_hash,
            "provider_calibration_identity": self.provider.calibration_state_id,
            "acquisition_factory_hash": (
                None
                if self.acquisition_factory is None
                else self.acquisition_factory.identity_hash
            ),
            "proposal_backend_identity_hash": (
                None
                if self.proposal_backend_seal is None
                else self.proposal_backend_seal.identity_hash
            ),
            "threshold_factory_hash": self.threshold_factory.identity_hash,
            "sampling_law_factory_hash": self.sampling_law_factory.identity_hash,
            "direct_oracle_provider_hash": self.provider_seal.provider_hash,
            "terminal_validator_hash": self.terminal_validator.identity_hash,
        }
        for name, expected in checks.items():
            if getattr(manifest, name) != expected:
                raise ValueError(f"production manifest disagrees on {name}")
        if (
            self.entropy_tape.campaign_contract_hash != contract.campaign_contract_hash
            or self.entropy_tape.controller_run_id != manifest.controller_run_id
            or self.entropy_tape.executor_identity_hash != self.runtime_config.executor_identity_hash
            or self.entropy_tape.maximum_candidate_count != contract.maximum_candidate_count
            or self.entropy_tape.draws_per_candidate != contract.cumulative_draw_counts[-1]
        ):
            raise ValueError("production entropy tape differs from the sealed campaign/run")
        if manifest.route_axis_hash != contract.route_axis_hash:
            raise ValueError("production route axis differs from the campaign contract")
        proposal_metadata = self._proposal_protocol_metadata()
        if proposal_enabled:
            assert self.proposal_backend_seal is not None
            assert self.proposal_initial_state_factory is not None
            assert self.proposal_forecast_factory is not None
            assert self.proposal_request_plan_factory is not None
            assert self.proposal_acquisition_executor is not None
            assert self.proposal_freeze_evaluator is not None
            assert self.proposal_acquisition_finalizer is not None
            backend = self.proposal_backend_seal
            expected_backend_coordinates = {
                "backend_kind": manifest.backend_kind,
                "backend_manifest_hash": manifest.backend_manifest_hash,
                "campaign_binding_hash": self.binding.binding_hash,
                "controller_run_id": manifest.controller_run_id,
                "model_data_seal_hash": manifest.model_data_seal_hash,
                "runtime_logical_config_hash": (
                    self.runtime_config.logical_identity_hash
                ),
            }
            if any(
                getattr(backend, name) != expected
                for name, expected in expected_backend_coordinates.items()
            ):
                raise ValueError(
                    "composite proposal backend seal has stale run/backend coordinates"
                )
            expected_component_hashes = {
                "initial_state_factory": (
                    self.proposal_initial_state_factory.identity_hash,
                    backend.initial_state_factory.identity_hash,
                ),
                "forecast_factory": (
                    self.proposal_forecast_factory.identity_hash,
                    backend.forecast_factory.identity_hash,
                ),
                "request_plan_factory": (
                    self.proposal_request_plan_factory.identity_hash,
                    backend.request_plan_factory.identity_hash,
                ),
                "acquisition_executor": (
                    self.proposal_acquisition_executor.identity_hash,
                    backend.acquisition_executor.identity_hash,
                ),
                "freeze_evaluator": (
                    self.proposal_freeze_evaluator.identity_hash,
                    backend.freeze_evaluator.identity_hash,
                ),
                "acquisition_finalizer": (
                    self.proposal_acquisition_finalizer.identity_hash,
                    backend.acquisition_finalizer.identity_hash,
                ),
            }
            if any(left != right for left, right in expected_component_hashes.values()):
                raise ValueError(
                    "individual proposal capability differs from its composite seal"
                )
            if dict(proposal_metadata) != dict(backend.manifest_metadata):
                raise ValueError("production manifest disagrees on staged proposal protocol")
        elif proposal_metadata:
            raise ValueError("staged proposal manifest metadata lacks executable hooks")
        if manifest.model_data_seal_hash != contract.model_seal_hash:
            raise ValueError("production model/data seal differs from the campaign model seal")
        if manifest.provider_context_population_hash != contract.context_population_hash:
            raise ValueError("provider population differs from the campaign context population")
        if self.provider_seal.context_population_hash != contract.context_population_hash:
            raise ValueError("paired-oracle seal differs from the campaign context population")
        if self.provider_seal.provider_context_ids_hash != expected_context_ids_hash:
            raise ValueError("paired-oracle seal differs from the concrete provider contexts")
        if self.provider_seal.calibration_identity != self.provider.calibration_state_id:
            raise ValueError("paired-oracle seal differs from the concrete provider calibration")
        if self.runtime_config.metadata.get(
            "candidate_calibration_freeze_required"
        ) is True and (
            self.provider_seal.evaluator_candidate_calibration_freeze_hook_spec
            is None
            or self.provider_seal.evaluator_candidate_calibration_verify_hook_spec
            is None
        ):
            raise ValueError(
                "freeze-required runtime lacks sealed candidate calibration hooks"
            )
        if self.manifest.metadata.get("paired_bce_oracle") is not None:
            if self.provider_seal.evaluator_kind != "parent_aware":
                raise ValueError(
                    "paired-BCE production requires a parent-aware direct oracle"
                )
            if (
                self.provider_seal.evaluator_candidate_calibration_freeze_hook_spec
                is None
                or self.provider_seal.evaluator_candidate_calibration_verify_hook_spec
                is None
            ):
                raise ValueError(
                    "paired-BCE production requires sealed candidate calibration hooks"
                )
            if (
                self.manifest.backend_kind == "alexnet_production"
                and (
                    self.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec
                    is None
                    or self.provider_seal
                    .evaluator_retire_committed_generation_state_hook_spec
                    is None
                )
            ):
                raise ValueError(
                    "AlexNet paired-BCE production requires a sealed evaluator "
                    "generation-local dynamic O-frame and committed-generation "
                    "retirement hooks"
                )
            _paired_bce_metadata(self)
        if manifest.provider_calibration_identity != contract.calibration_identity:
            raise ValueError("provider calibration differs from the campaign calibration")
        if (
            (self.performance_receipt_factory is None)
            != (manifest.performance_factory_hash is None)
        ):
            raise ValueError("performance factory callable/hash presence disagrees")
        if (
            self.performance_receipt_factory is not None
            and manifest.performance_factory_hash
            != self.performance_receipt_factory.identity_hash
        ):
            raise ValueError("production manifest disagrees on performance_factory_hash")
        if (
            (self.terminal_extraction_finalizer is None)
            != (manifest.terminal_extraction_finalizer_hash is None)
        ):
            raise ValueError("terminal extraction finalizer callable/hash presence disagrees")
        if (
            self.terminal_extraction_finalizer is not None
            and manifest.terminal_extraction_finalizer_hash
            != self.terminal_extraction_finalizer.identity_hash
        ):
            raise ValueError(
                "production manifest disagrees on terminal_extraction_finalizer_hash"
            )
        if self.schema != _BUNDLE_SCHEMA:
            raise ValueError("unsupported modern-v1 production-bundle schema")

    @property
    def bundle_hash(self) -> str:
        payload: dict[str, Any] = {
                "schema": self.schema,
                "manifest_hash": self.manifest.manifest_hash,
                "binding_hash": self.binding.binding_hash,
                "initial_mask_hash": self.initial_mask.mask_hash,
                "runtime_logical_config_hash": self.runtime_config.logical_identity_hash,
                "runtime_physical_config_hash": self.runtime_config.physical_execution_hash,
                "entropy_tape_hash": self.entropy_tape.tape_hash,
                "proposal_protocol": self.manifest.proposal_protocol,
                "acquisition_factory_hash": (
                    None
                    if self.acquisition_factory is None
                    else self.acquisition_factory.identity_hash
                ),
                "threshold_factory_hash": self.threshold_factory.identity_hash,
                "sampling_law_factory_hash": self.sampling_law_factory.identity_hash,
                "direct_oracle_provider_hash": self.provider_seal.provider_hash,
                "terminal_validator_hash": self.terminal_validator.identity_hash,
                "proposal_backend_identity_hash": (
                    None
                    if self.proposal_backend_seal is None
                    else self.proposal_backend_seal.identity_hash
                ),
                "terminal_extraction_finalizer_hash": (
                    None
                    if self.terminal_extraction_finalizer is None
                    else self.terminal_extraction_finalizer.identity_hash
                ),
                "performance_factory_hash": (
                    None
                    if self.performance_receipt_factory is None
                    else self.performance_receipt_factory.identity_hash
                ),
            }
        if self.attempt_structural_disclosure is not None:
            # Bound only when present: bundles and checkpoints without the
            # disclosure hook hash exactly as before (no hashed-family change).
            payload["attempt_structural_disclosure_hash"] = (
                self.attempt_structural_disclosure.identity_hash
            )
        if self.pareto_milestone_registration is not None:
            payload["pareto_milestone_registration_hash"] = (
                self.pareto_milestone_registration.registration_hash
            )
        if self.terminal_validation_family_design is not None:
            payload["terminal_validation_family_design_hash"] = (
                self.terminal_validation_family_design.design_hash
            )
        if self.terminal_comparison_current_v0_mask is not None:
            payload["terminal_comparison_current_v0_mask_hash"] = (
                self.terminal_comparison_current_v0_mask.mask_hash
            )
            payload["terminal_comparison_current_v0_mask_packed_sha256"] = (
                hashlib.sha256(
                    self.terminal_comparison_current_v0_mask.packed_bits
                ).hexdigest()
            )
        if self.terminal_comparison_current_v0_calibration_freeze is not None:
            payload["terminal_comparison_current_v0_calibration_freeze_hash"] = (
                self.terminal_comparison_current_v0_calibration_freeze.freeze_hash
            )
        return content_hash(
            payload,
            prefix="partition-free-v1-production-bundle-",
        )

    @property
    def resume_identity_hash(self) -> str:
        """Logical executable identity that must remain fixed across restart.

        Physical packing is intentionally absent: non-performance runs may
        lawfully resume with different batching/prefetch.  Every decision- or
        evidence-bearing capability, including the proposal protocol/backend,
        remains bound.
        """

        payload: dict[str, Any] = {
                "schema": _PRODUCTION_RESUME_BINDING_SCHEMA,
                "campaign_contract_hash": (
                    self.binding.contract.campaign_contract_hash
                ),
                "campaign_binding_hash": self.binding.binding_hash,
                "controller_run_id": self.manifest.controller_run_id,
                "backend_kind": self.manifest.backend_kind,
                "backend_manifest_hash": self.manifest.backend_manifest_hash,
                "model_data_seal_hash": self.manifest.model_data_seal_hash,
                "route_axis_hash": self.manifest.route_axis_hash,
                "initial_mask_hash": self.initial_mask.mask_hash,
                "runtime_logical_config_hash": (
                    self.runtime_config.logical_identity_hash
                ),
                "entropy_tape_hash": self.entropy_tape.tape_hash,
                "proposal_protocol": self.manifest.proposal_protocol,
                "acquisition_factory_hash": (
                    None
                    if self.acquisition_factory is None
                    else self.acquisition_factory.identity_hash
                ),
                "proposal_backend_identity_hash": (
                    None
                    if self.proposal_backend_seal is None
                    else self.proposal_backend_seal.identity_hash
                ),
                "threshold_factory_hash": self.threshold_factory.identity_hash,
                "sampling_law_factory_hash": (
                    self.sampling_law_factory.identity_hash
                ),
                "direct_oracle_provider_hash": self.provider_seal.provider_hash,
                "terminal_validator_hash": self.terminal_validator.identity_hash,
                "terminal_extraction_finalizer_hash": (
                    None
                    if self.terminal_extraction_finalizer is None
                    else self.terminal_extraction_finalizer.identity_hash
                ),
                "performance_factory_hash": (
                    None
                    if self.performance_receipt_factory is None
                    else self.performance_receipt_factory.identity_hash
                ),
            }
        if self.attempt_structural_disclosure is not None:
            # Bound only when present: bundles and checkpoints without the
            # disclosure hook hash exactly as before (no hashed-family change).
            payload["attempt_structural_disclosure_hash"] = (
                self.attempt_structural_disclosure.identity_hash
            )
        if self.pareto_milestone_registration is not None:
            payload["pareto_milestone_registration_hash"] = (
                self.pareto_milestone_registration.registration_hash
            )
        if self.terminal_validation_family_design is not None:
            payload["terminal_validation_family_design_hash"] = (
                self.terminal_validation_family_design.design_hash
            )
        if self.terminal_comparison_current_v0_mask is not None:
            payload["terminal_comparison_current_v0_mask_hash"] = (
                self.terminal_comparison_current_v0_mask.mask_hash
            )
            payload["terminal_comparison_current_v0_mask_packed_sha256"] = (
                hashlib.sha256(
                    self.terminal_comparison_current_v0_mask.packed_bits
                ).hexdigest()
            )
        if self.terminal_comparison_current_v0_calibration_freeze is not None:
            payload["terminal_comparison_current_v0_calibration_freeze_hash"] = (
                self.terminal_comparison_current_v0_calibration_freeze.freeze_hash
            )
        return content_hash(
            payload,
            prefix="partition-free-v1-production-resume-identity-",
        )

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def selector_promoted(self) -> bool:
        return False

    def _proposal_protocol_metadata(self) -> Mapping[str, Any]:
        metadata = self.manifest.metadata.get("staged_proposal_acquisition")
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise TypeError("staged proposal manifest metadata must be a mapping")
        return metadata

    @property
    def has_staged_proposal_protocol(self) -> bool:
        return self.manifest.proposal_protocol == "staged_train_only_proposal_v1"

    @property
    def has_paired_bce_oracle_protocol(self) -> bool:
        return self.manifest.metadata.get("paired_bce_oracle") is not None

    @property
    def has_terminal_validation_family_protocol(self) -> bool:
        return self.terminal_validation_family_design is not None

    @property
    def has_legacy_terminal_validation_family_protocol(self) -> bool:
        return type(self.terminal_validation_family_design) is (
            V1ProductionTerminalValidationFamilyDesign
        )

    @property
    def has_terminal_comparison_family_protocol_v2(self) -> bool:
        return type(self.terminal_validation_family_design) is (
            V1ProductionTerminalComparisonFamilyDesignV2
        )

    @property
    def proposal_protocol_hash(self) -> str | None:
        if not self.has_staged_proposal_protocol:
            return None
        assert self.proposal_backend_seal is not None
        return self.proposal_backend_seal.identity_hash


@dataclass(frozen=True, slots=True)
class V1ProductionRunIndex:
    """Machine-readable index of a completed genuine-controller development run."""

    production_manifest_hash: str
    production_bundle_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_protocol: ProposalProtocolKind
    proposal_backend_identity_hash: str | None
    final_runtime_state_hash: str
    final_controller_snapshot_hash: str
    terminal_extraction_closure_hash: str
    terminal_freeze_hash: str
    terminal_validation_hash: str
    completion_receipt_hash: str
    terminal_mask_hash: str
    acquisition_receipt_hashes: tuple[str, ...]
    attempt_receipt_hashes: tuple[str, ...]
    transition_receipt_hashes: tuple[str, ...]
    decision_receipt_hashes: tuple[str, ...]
    completed_proposal_closure_hashes: tuple[str, ...]
    final_checkpoint_payload_sha256: str
    performance_receipt_hash: str | None
    resumed_from_checkpoint_payload_sha256: str | None
    claim_execution_certified: bool = False
    selector_promoted: bool = False
    scientific_superiority_claim: bool = False
    schema: str = _INDEX_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "production_manifest_hash",
            "production_bundle_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "final_runtime_state_hash",
            "final_controller_snapshot_hash",
            "terminal_extraction_closure_hash",
            "terminal_freeze_hash",
            "terminal_validation_hash",
            "completion_receipt_hash",
            "terminal_mask_hash",
            "final_checkpoint_payload_sha256",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.proposal_protocol not in {
            "legacy_synthetic_one_shot_v1",
            "operator_binary_d_c_v1",
            "staged_train_only_proposal_v1",
        }:
            raise ValueError("production run index has an unsupported proposal protocol")
        if self.proposal_protocol in {"legacy_synthetic_one_shot_v1","operator_binary_d_c_v1"}:
            if self.proposal_backend_identity_hash is not None:
                raise ValueError("legacy run index must not name a staged proposal backend")
        else:
            object.__setattr__(
                self,
                "proposal_backend_identity_hash",
                _nonempty(
                    self.proposal_backend_identity_hash,
                    "proposal_backend_identity_hash",
                ),
            )
        for name in (
            "performance_receipt_hash",
            "resumed_from_checkpoint_payload_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        for name in (
            "acquisition_receipt_hashes",
            "attempt_receipt_hashes",
            "transition_receipt_hashes",
            "decision_receipt_hashes",
            "completed_proposal_closure_hashes",
        ):
            values = tuple(_nonempty(value, name) for value in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique")
            object.__setattr__(self, name, values)
        if self.proposal_protocol in {"legacy_synthetic_one_shot_v1","operator_binary_d_c_v1"}:
            if self.completed_proposal_closure_hashes:
                raise ValueError("legacy run index contains staged proposal closures")
        elif len(self.completed_proposal_closure_hashes) != len(
            self.acquisition_receipt_hashes
        ):
            raise ValueError(
                "staged run index acquisition lineage lacks its proposal closure"
            )
        if len(self.final_checkpoint_payload_sha256) != 64:
            raise ValueError("final checkpoint SHA-256 has the wrong length")
        if (
            self.schema != _INDEX_SCHEMA
            or self.claim_execution_certified is not False
            or self.selector_promoted is not False
            or self.scientific_superiority_claim is not False
        ):
            raise ValueError("production run index crossed its evidence ceiling")

    @property
    def index_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-production-run-index-")

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "index_hash": self.index_hash,
            "production_manifest_hash": self.production_manifest_hash,
            "production_bundle_hash": self.production_bundle_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "proposal_protocol": self.proposal_protocol,
            "proposal_backend_identity_hash": self.proposal_backend_identity_hash,
            "final_runtime_state_hash": self.final_runtime_state_hash,
            "final_controller_snapshot_hash": self.final_controller_snapshot_hash,
            "terminal_extraction_closure_hash": (
                self.terminal_extraction_closure_hash
            ),
            "terminal_freeze_hash": self.terminal_freeze_hash,
            "terminal_validation_hash": self.terminal_validation_hash,
            "completion_receipt_hash": self.completion_receipt_hash,
            "terminal_mask_hash": self.terminal_mask_hash,
            "acquisition_receipt_hashes": self.acquisition_receipt_hashes,
            "attempt_receipt_hashes": self.attempt_receipt_hashes,
            "transition_receipt_hashes": self.transition_receipt_hashes,
            "decision_receipt_hashes": self.decision_receipt_hashes,
            "completed_proposal_closure_hashes": (
                self.completed_proposal_closure_hashes
            ),
            "final_checkpoint_payload_sha256": self.final_checkpoint_payload_sha256,
            "performance_receipt_hash": self.performance_receipt_hash,
            "resumed_from_checkpoint_payload_sha256": (
                self.resumed_from_checkpoint_payload_sha256
            ),
            "claim_execution_certified": False,
            "selector_promoted": False,
            "scientific_superiority_claim": False,
        }


@dataclass(frozen=True, slots=True)
class V1ProductionOutputReceipt:
    output_directory: str
    production_binding_path: str
    production_binding_sha256: str
    index_path: str
    index_sha256: str
    checkpoint_path: str
    checkpoint_payload_sha256: str
    run_index_hash: str
    terminal_extraction_closure_path: str | None = None
    terminal_extraction_closure_sha256: str | None = None
    circuit_export_path: str | None = None
    circuit_export_sha256: str | None = None
    schema: str = _OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "output_directory",
            "production_binding_path",
            "production_binding_sha256",
            "index_path",
            "index_sha256",
            "checkpoint_path",
            "checkpoint_payload_sha256",
            "run_index_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "production_binding_sha256",
            "index_sha256",
            "checkpoint_payload_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} is not a lowercase SHA-256")
        if (
            (self.terminal_extraction_closure_path is None)
            != (self.terminal_extraction_closure_sha256 is None)
        ):
            raise ValueError("terminal extraction closure path/SHA presence disagrees")
        if self.terminal_extraction_closure_path is not None:
            object.__setattr__(
                self,
                "terminal_extraction_closure_path",
                _nonempty(
                    self.terminal_extraction_closure_path,
                    "terminal_extraction_closure_path",
                ),
            )
            digest = _nonempty(
                self.terminal_extraction_closure_sha256,
                "terminal_extraction_closure_sha256",
            )
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(
                    "terminal_extraction_closure_sha256 is not a lowercase SHA-256"
                )
            object.__setattr__(
                self,
                "terminal_extraction_closure_sha256",
                digest,
            )
        if self.schema != _OUTPUT_SCHEMA:
            raise ValueError("unsupported modern-v1 output-receipt schema")
        if (self.circuit_export_path is None) != (self.circuit_export_sha256 is None):
            raise ValueError('circuit export path/SHA presence disagrees')
        if self.circuit_export_path is not None:
            _nonempty(self.circuit_export_path, 'circuit_export_path')
            digest = self.circuit_export_sha256
            if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('circuit_export_sha256 is not a lowercase SHA-256')

    @property
    def receipt_hash(self) -> str:
        payload = {f.name:getattr(self, f.name) for f in fields(self)}
        if self.circuit_export_path is None:
            # Preserve historical non-operator output receipt identities.
            payload.pop('circuit_export_path')
            payload.pop('circuit_export_sha256')
        return content_hash(payload, prefix="partition-free-v1-production-output-")


@dataclass(frozen=True, slots=True)
class V1ProductionRunArtifacts:
    state: V1RuntimeState
    new_decisions: tuple[ProductionDecision, ...]
    terminal_extraction_closure: V1TerminalExtractionClosure
    final_checkpoint: V1RuntimeCheckpointReceipt
    run_index: V1ProductionRunIndex
    output_receipt: V1ProductionOutputReceipt
    resumed_checkpoint: V1RuntimeCheckpointReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not V1RuntimeState or self.state.completion is None:
            raise TypeError("production artifacts require a completed exact runtime state")
        if type(self.terminal_extraction_closure) is not V1TerminalExtractionClosure:
            raise TypeError(
                "production artifacts require an exact terminal extraction closure"
            )
        if type(self.final_checkpoint) is not V1RuntimeCheckpointReceipt:
            raise TypeError("production artifacts require an exact final checkpoint receipt")
        if type(self.run_index) is not V1ProductionRunIndex:
            raise TypeError("production artifacts require an exact run index")
        if type(self.output_receipt) is not V1ProductionOutputReceipt:
            raise TypeError("production artifacts require an exact output receipt")
        if self.resumed_checkpoint is not None and type(
            self.resumed_checkpoint
        ) is not V1RuntimeCheckpointReceipt:
            raise TypeError("resumed checkpoint has the wrong type")
        if any(
            type(item)
            not in (
                V1StagedAlphaDecision,
                V1PairedBCEDecision,
                V1PairedBCECalibrationHorizonSkipDecision,
            )
            for item in self.new_decisions
        ):
            raise TypeError(
                "production decisions must be exact staged-alpha, paired-BCE, "
                "or calibration-horizon skip decisions"
            )
        completion = self.state.completion
        freeze = self.state.terminal_freeze
        legacy_validation = self.state.terminal_validation
        family_validation = self.state.terminal_validation_family
        comparison_validation = self.state.terminal_comparison_family
        available_validations = tuple(
            value
            for value in (
                legacy_validation,
                family_validation,
                comparison_validation,
            )
            if value is not None
        )
        if len(available_validations) != 1:
            raise ValueError(
                "production artifacts require exactly one terminal-validation mode"
            )
        validation = available_validations[0]
        if completion is None or freeze is None or validation is None:
            raise ValueError("production artifacts lack exact terminal runtime receipts")
        snapshot = self.state.controller.snapshot_receipt()
        needs_export = (self.run_index.proposal_protocol == 'operator_binary_d_c_v1'
            and self.state.controller.accepted_step_count > 0)
        if needs_export != (self.output_receipt.circuit_export_path is not None):
            raise ValueError('production output circuit export differs from its accepted operator run')
        if needs_export:
            from .modern_v1_operator_export import read_operator_circuit_export
            artifact = read_operator_circuit_export(self.output_receipt.circuit_export_path,
                expected_sha256=self.output_receipt.circuit_export_sha256)
            payload = json.loads(artifact.raw)
            if (artifact.mask != self.state.controller.current_mask
                    or artifact.calibration.content_hash != freeze.extraction_calibration_hash
                    or content_hash(payload['terminal_freeze'], prefix='partition-free-v1-terminal-freeze-')
                    != freeze.receipt_hash):
                raise ValueError('production output circuit differs from the terminal freeze')
            _validate_operator_export_source(self.state, payload)
        _validate_terminal_extraction_closure(
            self.state.controller,
            self.terminal_extraction_closure,
        )
        if (
            freeze.extraction_state_hash
            != self.terminal_extraction_closure.extraction_state_hash
            or freeze.extraction_calibration_hash
            != self.terminal_extraction_closure.extraction_calibration_hash
        ):
            raise ValueError(
                "production terminal freeze differs from its extraction closure"
            )
        binding_path = Path(self.output_receipt.production_binding_path)
        if (
            not binding_path.is_file()
            or _sha256_file(binding_path)
            != self.output_receipt.production_binding_sha256
        ):
            raise ValueError(
                "production output receipt resume binding is unavailable or changed"
            )
        resume_binding = _read_production_resume_binding(binding_path)
        expected_binding = {
            "campaign_contract_hash": (
                self.state.controller.binding.contract.campaign_contract_hash
            ),
            "campaign_binding_hash": self.state.controller.binding.binding_hash,
            "controller_run_id": self.state.controller.controller_run_id,
        }
        if any(
            resume_binding[name] != expected
            for name, expected in expected_binding.items()
        ):
            raise ValueError(
                "production resume binding differs from the completed runtime"
            )
        _validate_completed_proposal_closure_ledger(
            self.state,
            proposal_protocol=resume_binding["proposal_protocol"],
            proposal_backend_identity_hash=(
                resume_binding["proposal_backend_identity_hash"]
            ),
        )
        expected_index = {
            "campaign_contract_hash": self.state.controller.binding.contract.campaign_contract_hash,
            "campaign_binding_hash": self.state.controller.binding.binding_hash,
            "controller_run_id": self.state.controller.controller_run_id,
            "proposal_protocol": resume_binding["proposal_protocol"],
            "proposal_backend_identity_hash": (
                resume_binding["proposal_backend_identity_hash"]
            ),
            "final_runtime_state_hash": self.state.runtime_state_hash,
            "final_controller_snapshot_hash": snapshot.receipt_hash,
            "terminal_extraction_closure_hash": (
                self.terminal_extraction_closure.closure_hash
            ),
            "terminal_freeze_hash": freeze.receipt_hash,
            "terminal_validation_hash": validation.receipt_hash,
            "completion_receipt_hash": completion.receipt_hash,
            "terminal_mask_hash": self.state.controller.current_mask.mask_hash,
            "acquisition_receipt_hashes": self.state.controller.acquisition_receipt_hashes,
            "attempt_receipt_hashes": self.state.controller.attempt_receipt_hashes,
            "transition_receipt_hashes": self.state.controller.transition_receipt_hashes,
            "decision_receipt_hashes": tuple(
                item.decision_receipt_hash for item in self.state.controller.attempt_receipts
            ),
            "completed_proposal_closure_hashes": tuple(
                item.receipt_hash
                for item in self.state.completed_proposal_closures
            ),
            "final_checkpoint_payload_sha256": self.final_checkpoint.payload_sha256,
            "performance_receipt_hash": (
                None if self.state.performance is None else self.state.performance.receipt_hash
            ),
            "resumed_from_checkpoint_payload_sha256": (
                None
                if self.resumed_checkpoint is None
                else self.resumed_checkpoint.payload_sha256
            ),
        }
        if any(
            getattr(self.run_index, name) != expected
            for name, expected in expected_index.items()
        ):
            raise ValueError("production run index differs from the exact completed runtime")
        retained_horizon_progress = (
            self.state.proposal_progress is not None
            and getattr(
                self.state.controller,
                "fqr_no_candidate_horizon_witness",
                None,
            )
            is not None
            and self.state.pending_proposal_acquisition is None
        )
        if (
            self.final_checkpoint.runtime_state_hash != self.state.runtime_state_hash
            or self.final_checkpoint.controller_snapshot_hash != snapshot.receipt_hash
            or self.final_checkpoint.pending_look_hash is not None
            or self.final_checkpoint.pending_paired_bce_chunk_hash is not None
            or (
                self.state.proposal_progress is not None
                and not retained_horizon_progress
            )
            or self.state.pending_proposal_acquisition is not None
            or self.state.pending_paired_bce_chunk is not None
        ):
            raise ValueError("production final checkpoint differs from the completed runtime")
        all_decision_hashes = tuple(
            item.decision_receipt_hash for item in self.state.controller.attempt_receipts
        )
        new_decision_hashes = tuple(
            _production_decision_hash(item) for item in self.new_decisions
        )
        if new_decision_hashes != all_decision_hashes[len(all_decision_hashes) - len(new_decision_hashes) :]:
            raise ValueError("production new decisions are not the exact completed-history suffix")
        if (
            self.output_receipt.run_index_hash != self.run_index.index_hash
            or self.output_receipt.checkpoint_payload_sha256
            != self.final_checkpoint.payload_sha256
            or Path(self.output_receipt.checkpoint_path).resolve()
            != Path(self.final_checkpoint.checkpoint_path).resolve()
        ):
            raise ValueError("production output receipt differs from the run index/checkpoint")
        closure_path = self.output_receipt.terminal_extraction_closure_path
        closure_sha256 = self.output_receipt.terminal_extraction_closure_sha256
        if closure_path is not None:
            path = Path(closure_path)
            if not path.is_file() or _sha256_file(path) != closure_sha256:
                raise ValueError(
                    "production output receipt terminal closure file is unavailable or changed"
                )


ProposalQualificationStatus = Literal[
    "candidate_family_frozen",
    "no_candidate_fqr_horizon_proved",
    "proposal_order_indeterminate",
    "resource_cap_unresolved",
]


@dataclass(frozen=True, slots=True)
class V1ProductionProposalQualificationReceipt:
    """Train-only stop receipt for one production proposal generation.

    This receipt deliberately stops *before* pure candidate attachment.  It
    can qualify D/KKT/C acquisition, decision stability, checkpoint replay,
    and the proposal-freeze decision without opening K, O, a mask commit, or
    terminal validation.  It is therefore useful for source-sealed H200
    qualification without turning a canary into a scientific outcome.
    """

    status: ProposalQualificationStatus
    production_manifest_hash: str
    production_bundle_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    final_runtime_state_hash: str
    final_controller_state_hash: str
    final_controller_snapshot_hash: str
    current_mask_hash: str
    route_axis_hash: str
    proposal_progress_hash: str
    proposal_frontier_hash: str
    proposal_joint_state_hash: str
    current_sufficient_state_hash: str
    current_sufficient_state_descriptor_hash: str
    authorized_freeze_decision_hash: str | None
    terminal_refusal_receipt_hash: str | None
    update_receipt_hashes: tuple[str, ...]
    freeze_decision_receipt_hashes: tuple[str, ...]
    microtransition_count: int
    checkpoint_payload_sha256: str
    fqr_no_candidate_horizon_witness_hash: str | None = None
    train_optimization_data_only: bool = True
    candidate_attached: bool = False
    calibration_oracle_invoked: bool = False
    bce_oracle_invoked: bool = False
    mask_commit_authority: bool = False
    terminal_outcomes_opened: bool = False
    selector_promotion_authority: bool = False
    production_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = _PROPOSAL_QUALIFICATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in {
            "candidate_family_frozen",
            "no_candidate_fqr_horizon_proved",
            "proposal_order_indeterminate",
            "resource_cap_unresolved",
        }:
            raise ValueError("unsupported proposal qualification status")
        for name in (
            "production_manifest_hash",
            "production_bundle_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "proposal_backend_identity_hash",
            "final_runtime_state_hash",
            "final_controller_state_hash",
            "final_controller_snapshot_hash",
            "current_mask_hash",
            "route_axis_hash",
            "proposal_progress_hash",
            "proposal_frontier_hash",
            "proposal_joint_state_hash",
            "current_sufficient_state_hash",
            "current_sufficient_state_descriptor_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        checkpoint = _nonempty(
            self.checkpoint_payload_sha256,
            "checkpoint_payload_sha256",
        )
        if len(checkpoint) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint
        ):
            raise ValueError("checkpoint_payload_sha256 is not lowercase SHA-256")
        object.__setattr__(self, "checkpoint_payload_sha256", checkpoint)
        transitions = _positive(self.microtransition_count, "microtransition_count")
        object.__setattr__(self, "microtransition_count", transitions)
        updates = tuple(
            _nonempty(value, "update receipt hash")
            for value in self.update_receipt_hashes
        )
        freezes = tuple(
            _nonempty(value, "freeze decision receipt hash")
            for value in self.freeze_decision_receipt_hashes
        )
        if len(set(updates)) != len(updates) or len(set(freezes)) != len(freezes):
            raise ValueError("proposal qualification receipt ledgers contain duplicates")
        object.__setattr__(self, "update_receipt_hashes", updates)
        object.__setattr__(self, "freeze_decision_receipt_hashes", freezes)
        authorized = self.authorized_freeze_decision_hash
        refusal = self.terminal_refusal_receipt_hash
        horizon_witness = self.fqr_no_candidate_horizon_witness_hash
        if authorized is not None:
            authorized = _nonempty(authorized, "authorized freeze decision hash")
        if refusal is not None:
            refusal = _nonempty(refusal, "terminal refusal receipt hash")
        if horizon_witness is not None:
            horizon_witness = _nonempty(
                horizon_witness,
                "F/Q/R no-candidate horizon witness hash",
            )
        if self.status == "candidate_family_frozen":
            if (
                authorized is None
                or refusal is not None
                or horizon_witness is not None
            ):
                raise ValueError(
                    "frozen proposal qualification requires only freeze authorization"
                )
        elif self.status == "no_candidate_fqr_horizon_proved":
            if (
                authorized is not None
                or refusal is not None
                or not freezes
                or horizon_witness is None
            ):
                raise ValueError(
                    "F/Q/R horizon qualification requires its proof-bound "
                    "freeze-refusal ledger and exact controller witness"
                )
        elif self.status == "proposal_order_indeterminate":
            if (
                authorized is not None
                or refusal is None
                or horizon_witness is not None
            ):
                raise ValueError(
                    "indeterminate proposal qualification requires only refusal"
                )
        elif (
            authorized is not None
            or refusal is not None
            or horizon_witness is not None
        ):
            raise ValueError(
                "resource-capped proposal qualification cannot claim a decision"
            )
        object.__setattr__(self, "authorized_freeze_decision_hash", authorized)
        object.__setattr__(self, "terminal_refusal_receipt_hash", refusal)
        object.__setattr__(
            self,
            "fqr_no_candidate_horizon_witness_hash",
            horizon_witness,
        )
        true_flags = (self.train_optimization_data_only,)
        false_flags = (
            self.candidate_attached,
            self.calibration_oracle_invoked,
            self.bce_oracle_invoked,
            self.mask_commit_authority,
            self.terminal_outcomes_opened,
            self.selector_promotion_authority,
            self.production_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in (*true_flags, *false_flags)):
            raise TypeError("proposal qualification authority flags must be booleans")
        if not all(true_flags) or any(false_flags):
            raise PermissionError("proposal qualification crossed its authority ceiling")
        if self.schema != _PROPOSAL_QUALIFICATION_RECEIPT_SCHEMA:
            raise ValueError("unsupported proposal qualification receipt schema")

    @property
    def receipt_hash(self) -> str:
        # Keep the historical v1 byte identity for all pre-horizon receipts.
        # The optional witness enters the projection iff it carries authority.
        projection = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "fqr_no_candidate_horizon_witness_hash"
        }
        if self.fqr_no_candidate_horizon_witness_hash is not None:
            projection["fqr_no_candidate_horizon_witness_hash"] = (
                self.fqr_no_candidate_horizon_witness_hash
            )
        return content_hash(
            projection,
            prefix="partition-free-v1-production-proposal-qualification-",
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "production_manifest_hash": self.production_manifest_hash,
            "production_bundle_hash": self.production_bundle_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "proposal_backend_identity_hash": self.proposal_backend_identity_hash,
            "final_runtime_state_hash": self.final_runtime_state_hash,
            "final_controller_state_hash": self.final_controller_state_hash,
            "final_controller_snapshot_hash": self.final_controller_snapshot_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "proposal_progress_hash": self.proposal_progress_hash,
            "proposal_frontier_hash": self.proposal_frontier_hash,
            "proposal_joint_state_hash": self.proposal_joint_state_hash,
            "current_sufficient_state_hash": self.current_sufficient_state_hash,
            "current_sufficient_state_descriptor_hash": (
                self.current_sufficient_state_descriptor_hash
            ),
            "authorized_freeze_decision_hash": (
                self.authorized_freeze_decision_hash
            ),
            "terminal_refusal_receipt_hash": self.terminal_refusal_receipt_hash,
            "fqr_no_candidate_horizon_witness_hash": (
                self.fqr_no_candidate_horizon_witness_hash
            ),
            "update_receipt_hashes": list(self.update_receipt_hashes),
            "freeze_decision_receipt_hashes": list(
                self.freeze_decision_receipt_hashes
            ),
            "microtransition_count": self.microtransition_count,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "train_optimization_data_only": self.train_optimization_data_only,
            "candidate_attached": self.candidate_attached,
            "calibration_oracle_invoked": self.calibration_oracle_invoked,
            "bce_oracle_invoked": self.bce_oracle_invoked,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_outcomes_opened": self.terminal_outcomes_opened,
            "selector_promotion_authority": self.selector_promotion_authority,
            "production_authority": self.production_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class V1ProductionProposalQualificationArtifacts:
    state: V1RuntimeState
    receipt: V1ProductionProposalQualificationReceipt
    final_checkpoint: V1RuntimeCheckpointReceipt
    resumed_checkpoint: V1RuntimeCheckpointReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not V1RuntimeState:
            raise TypeError("proposal qualification requires an exact runtime state")
        if type(self.receipt) is not V1ProductionProposalQualificationReceipt:
            raise TypeError("proposal qualification requires an exact receipt")
        if type(self.final_checkpoint) is not V1RuntimeCheckpointReceipt:
            raise TypeError("proposal qualification requires an exact checkpoint")
        if self.resumed_checkpoint is not None and type(
            self.resumed_checkpoint
        ) is not V1RuntimeCheckpointReceipt:
            raise TypeError("proposal qualification resume receipt has another type")
        progress = self.state.proposal_progress
        controller = self.state.controller
        if progress is None:
            raise ValueError("proposal qualification lost its proposal frontier")
        snapshot = controller.snapshot_receipt()
        horizon_witness = getattr(
            controller,
            "fqr_no_candidate_horizon_witness",
            None,
        )
        expected = {
            "final_runtime_state_hash": self.state.runtime_state_hash,
            "final_controller_state_hash": controller.state_hash,
            "final_controller_snapshot_hash": snapshot.receipt_hash,
            "current_mask_hash": controller.current_mask.mask_hash,
            "route_axis_hash": controller.current_mask.route_axis_hash,
            "proposal_progress_hash": progress.progress_hash,
            "proposal_frontier_hash": progress.frontier_hash,
            "proposal_joint_state_hash": progress.joint_state.content_hash,
            "current_sufficient_state_hash": (
                progress.current_sufficient_state.sufficient_state_hash
            ),
            "current_sufficient_state_descriptor_hash": (
                progress.current_sufficient_state.descriptor_hash
            ),
            "fqr_no_candidate_horizon_witness_hash": (
                None if horizon_witness is None else horizon_witness.witness_hash
            ),
            "checkpoint_payload_sha256": self.final_checkpoint.payload_sha256,
        }
        if any(
            getattr(self.receipt, name) != value
            for name, value in expected.items()
        ):
            raise ValueError(
                "proposal qualification artifacts differ from their runtime state"
            )
        if (
            self.final_checkpoint.runtime_state_hash != self.state.runtime_state_hash
            or self.final_checkpoint.controller_snapshot_hash != snapshot.receipt_hash
            or controller.acquisition is not None
            or controller.attempt_receipts
            or controller.transition_receipts
            or self.state.pending_look is not None
            or self.state.pending_paired_bce_chunk is not None
            or self.state.terminal_freeze is not None
            or self.state.terminal_validation is not None
            or self.state.completion is not None
            or self.state.performance is not None
            or progress.calibration_authority is not False
            or progress.validation_authority is not False
            or progress.mask_commit_authority is not False
        ):
            raise PermissionError(
                "proposal qualification artifacts crossed their train-only boundary"
            )


TrainOnlyDecisionQualificationStatus = Literal[
    "decision_qualified",
    "proposal_order_indeterminate",
    "scientific_endpoint_without_decision",
    "resource_cap_unresolved",
]


@dataclass(frozen=True, slots=True)
class V1TrainOnlyDecisionQualificationBinding:
    """Durable origin coordinate for one restart-safe decision qualification.

    The target is one *additional* candidate-attempt receipt relative to this
    origin.  Persisting the origin before any K/O work prevents a restart from
    silently treating the already-observed decision as the beginning of a new
    qualification.
    """

    production_manifest_hash: str
    production_bundle_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    origin_runtime_state_hash: str
    origin_controller_snapshot_hash: str
    origin_mask_hash: str
    origin_candidate_attempt_count: int
    origin_accepted_step_count: int
    target_candidate_attempt_count: int
    origin_checkpoint_payload_sha256: str
    schema: str = _TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "production_manifest_hash",
            "production_bundle_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "origin_runtime_state_hash",
            "origin_controller_snapshot_hash",
            "origin_mask_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        attempts = _nonnegative_int(
            self.origin_candidate_attempt_count,
            "origin_candidate_attempt_count",
        )
        accepted = _nonnegative_int(
            self.origin_accepted_step_count,
            "origin_accepted_step_count",
        )
        target = _positive(
            self.target_candidate_attempt_count,
            "target_candidate_attempt_count",
        )
        if target != attempts + 1 or accepted > attempts:
            raise ValueError(
                "train-only qualification target must be exactly one attempt "
                "beyond a legal origin"
            )
        digest = _nonempty(
            self.origin_checkpoint_payload_sha256,
            "origin_checkpoint_payload_sha256",
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "origin_checkpoint_payload_sha256 is not lowercase SHA-256"
            )
        if self.schema != _TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_SCHEMA:
            raise ValueError("unsupported train-only decision binding schema")
        object.__setattr__(self, "origin_candidate_attempt_count", attempts)
        object.__setattr__(self, "origin_accepted_step_count", accepted)
        object.__setattr__(self, "target_candidate_attempt_count", target)
        object.__setattr__(self, "origin_checkpoint_payload_sha256", digest)

    @property
    def binding_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-train-only-decision-binding-",
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self)
        } | {"binding_hash": self.binding_hash}


@dataclass(frozen=True, slots=True)
class V1TrainOnlyDecisionQualificationReceipt:
    """Validation-independent receipt for one complete K/O decision boundary."""

    status: TrainOnlyDecisionQualificationStatus
    qualification_binding_hash: str
    production_manifest_hash: str
    production_bundle_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    origin_runtime_state_hash: str
    final_runtime_state_hash: str
    final_controller_state_hash: str
    final_controller_snapshot_hash: str
    origin_mask_hash: str
    final_mask_hash: str
    origin_candidate_attempt_count: int
    final_candidate_attempt_count: int
    origin_accepted_step_count: int
    final_accepted_step_count: int
    decision_type: str | None
    decision_action: str | None
    decision_receipt_hash: str | None
    final_controller_phase: str
    fresh_chart_progress_hash: str | None
    checkpoint_payload_sha256: str
    train_optimization_data_only: bool = True
    candidate_calibration_invoked: bool = False
    paired_bce_oracle_invoked: bool = False
    mask_commit_occurred: bool = False
    fresh_chart_opened: bool = False
    terminal_extraction_invoked: bool = False
    terminal_validation_invoked: bool = False
    validation_population_opened: bool = False
    production_completion_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = _TRAIN_ONLY_DECISION_QUALIFICATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in {
            "decision_qualified",
            "proposal_order_indeterminate",
            "scientific_endpoint_without_decision",
            "resource_cap_unresolved",
        }:
            raise ValueError("unsupported train-only decision qualification status")
        for name in (
            "qualification_binding_hash",
            "production_manifest_hash",
            "production_bundle_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "origin_runtime_state_hash",
            "final_runtime_state_hash",
            "final_controller_state_hash",
            "final_controller_snapshot_hash",
            "origin_mask_hash",
            "final_mask_hash",
            "final_controller_phase",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        origin_attempts = _nonnegative_int(
            self.origin_candidate_attempt_count,
            "origin_candidate_attempt_count",
        )
        final_attempts = _nonnegative_int(
            self.final_candidate_attempt_count,
            "final_candidate_attempt_count",
        )
        origin_accepted = _nonnegative_int(
            self.origin_accepted_step_count,
            "origin_accepted_step_count",
        )
        final_accepted = _nonnegative_int(
            self.final_accepted_step_count,
            "final_accepted_step_count",
        )
        if origin_accepted > origin_attempts or final_accepted > final_attempts:
            raise ValueError("accepted-step counts cannot exceed attempt counts")
        decision_fields = (
            self.decision_type,
            self.decision_action,
            self.decision_receipt_hash,
        )
        has_decision = self.status == "decision_qualified"
        if has_decision:
            if any(value is None for value in decision_fields):
                raise ValueError("qualified decision receipt lacks its decision identity")
            if final_attempts != origin_attempts + 1:
                raise ValueError("qualified seam must consume exactly one candidate attempt")
            for name in (
                "decision_type",
                "decision_action",
                "decision_receipt_hash",
            ):
                object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        elif any(value is not None for value in decision_fields):
            raise ValueError("unresolved qualification cannot carry a decision identity")
        if self.mask_commit_occurred:
            if (
                not has_decision
                or self.decision_action != "accept_candidate"
                or final_accepted != origin_accepted + 1
                or self.final_mask_hash == self.origin_mask_hash
            ):
                raise ValueError("mask-commit flag disagrees with the admitted decision")
        elif (
            final_accepted != origin_accepted
            or self.final_mask_hash != self.origin_mask_hash
        ):
            raise ValueError("noncommit qualification changed the accepted mask lineage")
        if self.fresh_chart_opened != (self.fresh_chart_progress_hash is not None):
            raise ValueError("fresh-chart flag/hash presence disagrees")
        if self.fresh_chart_opened and not self.mask_commit_occurred:
            raise ValueError("only an accepted nonterminal decision opens a fresh chart")
        if self.paired_bce_oracle_invoked and not self.candidate_calibration_invoked:
            raise ValueError("paired O cannot precede candidate calibration")
        digest = _nonempty(
            self.checkpoint_payload_sha256,
            "checkpoint_payload_sha256",
        )
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("checkpoint_payload_sha256 is not lowercase SHA-256")
        true_flags = (self.train_optimization_data_only,)
        false_flags = (
            self.terminal_extraction_invoked,
            self.terminal_validation_invoked,
            self.validation_population_opened,
            self.production_completion_authority,
            self.pareto_claim_authority,
        )
        all_flags = (
            *true_flags,
            self.candidate_calibration_invoked,
            self.paired_bce_oracle_invoked,
            self.mask_commit_occurred,
            self.fresh_chart_opened,
            *false_flags,
        )
        if any(type(value) is not bool for value in all_flags):
            raise TypeError("train-only decision authority flags must be booleans")
        if not all(true_flags) or any(false_flags):
            raise PermissionError("train-only decision receipt crossed validation authority")
        if self.schema != _TRAIN_ONLY_DECISION_QUALIFICATION_RECEIPT_SCHEMA:
            raise ValueError("unsupported train-only decision receipt schema")
        object.__setattr__(self, "origin_candidate_attempt_count", origin_attempts)
        object.__setattr__(self, "final_candidate_attempt_count", final_attempts)
        object.__setattr__(self, "origin_accepted_step_count", origin_accepted)
        object.__setattr__(self, "final_accepted_step_count", final_accepted)
        object.__setattr__(self, "checkpoint_payload_sha256", digest)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-train-only-decision-qualification-",
        )

    def to_json_payload(self) -> dict[str, Any]:
        return {
            item.name: getattr(self, item.name) for item in fields(self)
        } | {"receipt_hash": self.receipt_hash}


@dataclass(frozen=True, slots=True)
class V1TrainOnlyDecisionQualificationArtifacts:
    state: V1RuntimeState
    binding: V1TrainOnlyDecisionQualificationBinding
    receipt: V1TrainOnlyDecisionQualificationReceipt
    final_checkpoint: V1RuntimeCheckpointReceipt
    decision: ProductionDecision | None = None
    resumed_checkpoint: V1RuntimeCheckpointReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.state) is not V1RuntimeState:
            raise TypeError("train-only decision artifacts require an exact runtime state")
        if type(self.binding) is not V1TrainOnlyDecisionQualificationBinding:
            raise TypeError("train-only decision artifacts require an exact binding")
        if type(self.receipt) is not V1TrainOnlyDecisionQualificationReceipt:
            raise TypeError("train-only decision artifacts require an exact receipt")
        if type(self.final_checkpoint) is not V1RuntimeCheckpointReceipt:
            raise TypeError("train-only decision artifacts require an exact checkpoint")
        if self.resumed_checkpoint is not None and type(
            self.resumed_checkpoint
        ) is not V1RuntimeCheckpointReceipt:
            raise TypeError("train-only decision resume receipt has another type")
        if self.decision is not None and type(self.decision) not in (
            V1PairedBCEDecision,
            V1PairedBCECalibrationHorizonSkipDecision,
        ):
            raise TypeError("train-only qualification accepts only paired-BCE decisions")
        controller = self.state.controller
        snapshot = controller.snapshot_receipt()
        progress = self.state.proposal_progress
        expected = {
            "qualification_binding_hash": self.binding.binding_hash,
            "final_runtime_state_hash": self.state.runtime_state_hash,
            "final_controller_state_hash": controller.state_hash,
            "final_controller_snapshot_hash": snapshot.receipt_hash,
            "final_mask_hash": controller.current_mask.mask_hash,
            "final_candidate_attempt_count": controller.candidate_attempt_count,
            "final_accepted_step_count": controller.accepted_step_count,
            "final_controller_phase": controller.phase,
            "fresh_chart_progress_hash": (
                None if not self.receipt.fresh_chart_opened else progress.progress_hash
            ),
            "checkpoint_payload_sha256": self.final_checkpoint.payload_sha256,
        }
        if any(
            getattr(self.receipt, name) != value
            for name, value in expected.items()
        ):
            raise ValueError("train-only artifacts differ from their runtime state")
        if (
            (
                self.receipt.status == "decision_qualified"
                and (
                    self.state.pending_look is not None
                    or self.state.pending_paired_bce_chunk is not None
                    or self.state.pending_proposal_acquisition is not None
                )
            )
            or self.state.terminal_freeze is not None
            or self.state.terminal_validation is not None
            or self.state.completion is not None
            or self.state.performance is not None
            or self.final_checkpoint.runtime_state_hash != self.state.runtime_state_hash
            or self.final_checkpoint.controller_snapshot_hash != snapshot.receipt_hash
        ):
            raise PermissionError("train-only artifacts crossed their sealed stop boundary")
        if self.receipt.fresh_chart_opened:
            if (
                controller.phase != "needs_acquisition"
                or progress is None
                or progress.update_receipts
                or progress.outcome_receipts
                or progress.freeze_decision_receipts
            ):
                raise ValueError("accepted qualification did not stop at a fresh chart")
        if self.receipt.status == "decision_qualified":
            if not controller.attempt_receipts:
                raise ValueError("qualified artifacts lack their terminal attempt receipt")
            terminal_attempt = controller.attempt_receipts[-1]
            if (
                terminal_attempt.decision_receipt_hash
                != self.receipt.decision_receipt_hash
                or terminal_attempt.action != self.receipt.decision_action
            ):
                raise ValueError("qualified artifacts differ from their terminal attempt")
        elif self.decision is not None:
            raise ValueError("unresolved train-only artifacts cannot expose a decision")
        if self.decision is not None:
            if (
                _production_decision_hash(self.decision)
                != self.receipt.decision_receipt_hash
                or type(self.decision).__name__ != self.receipt.decision_type
            ):
                raise ValueError("train-only receipt names another decision")


def _train_only_decision_binding_from_payload(
    payload: Mapping[str, Any],
) -> V1TrainOnlyDecisionQualificationBinding:
    field_names = {item.name for item in fields(V1TrainOnlyDecisionQualificationBinding)}
    if set(payload) != field_names | {"binding_hash"}:
        raise ValueError("train-only decision binding has unexpected coordinates")
    binding = V1TrainOnlyDecisionQualificationBinding(
        **{name: payload[name] for name in field_names}
    )
    if payload["binding_hash"] != binding.binding_hash:
        raise ValueError("train-only decision binding failed its content hash")
    return binding


def _read_train_only_decision_binding(
    path: Path,
) -> V1TrainOnlyDecisionQualificationBinding:
    _require_nonsymlink_regular_file(
        path,
        label="train-only decision qualification binding",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("train-only decision binding is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("train-only decision binding must contain a mapping")
    return _train_only_decision_binding_from_payload(payload)


def _validate_train_only_decision_binding(
    binding: V1TrainOnlyDecisionQualificationBinding,
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
) -> None:
    if type(binding) is not V1TrainOnlyDecisionQualificationBinding:
        raise TypeError("train-only decision qualification requires an exact binding")
    controller = state.controller
    expected = {
        "production_manifest_hash": bundle.manifest.manifest_hash,
        "production_bundle_hash": bundle.bundle_hash,
        "campaign_contract_hash": bundle.binding.contract.campaign_contract_hash,
        "campaign_binding_hash": bundle.binding.binding_hash,
        "controller_run_id": bundle.manifest.controller_run_id,
    }
    if any(getattr(binding, name) != value for name, value in expected.items()):
        raise ValueError("train-only decision binding names another production bundle")
    attempts = controller.candidate_attempt_count
    accepted = controller.accepted_step_count
    if (
        attempts < binding.origin_candidate_attempt_count
        or attempts > binding.target_candidate_attempt_count
        or accepted < binding.origin_accepted_step_count
        or accepted > binding.origin_accepted_step_count + 1
    ):
        raise ValueError("train-only decision checkpoint crossed its one-decision horizon")
    if attempts == binding.origin_candidate_attempt_count and (
        accepted != binding.origin_accepted_step_count
        or controller.current_mask.mask_hash != binding.origin_mask_hash
    ):
        raise ValueError("train-only decision origin mask lineage changed before decision")
    if (
        state.terminal_freeze is not None
        or state.terminal_validation is not None
        or state.completion is not None
        or state.performance is not None
    ):
        raise PermissionError("train-only decision resume crossed terminal authority")


def _new_train_only_decision_binding(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    checkpoint: V1RuntimeCheckpointReceipt,
) -> V1TrainOnlyDecisionQualificationBinding:
    controller = state.controller
    return V1TrainOnlyDecisionQualificationBinding(
        production_manifest_hash=bundle.manifest.manifest_hash,
        production_bundle_hash=bundle.bundle_hash,
        campaign_contract_hash=bundle.binding.contract.campaign_contract_hash,
        campaign_binding_hash=bundle.binding.binding_hash,
        controller_run_id=bundle.manifest.controller_run_id,
        origin_runtime_state_hash=state.runtime_state_hash,
        origin_controller_snapshot_hash=(
            controller.snapshot_receipt().receipt_hash
        ),
        origin_mask_hash=controller.current_mask.mask_hash,
        origin_candidate_attempt_count=controller.candidate_attempt_count,
        origin_accepted_step_count=controller.accepted_step_count,
        target_candidate_attempt_count=controller.candidate_attempt_count + 1,
        origin_checkpoint_payload_sha256=checkpoint.payload_sha256,
    )


def _train_only_decision_action(decision: ProductionDecision) -> str:
    if type(decision) is V1PairedBCEDecision:
        return decision.status
    if type(decision) is V1PairedBCECalibrationHorizonSkipDecision:
        return "skip_candidate_calibration_refusal"
    raise TypeError("train-only qualification accepts only paired-BCE decisions")


def _production_resume_binding_payload(
    bundle: V1ProductionRuntimeBundle,
) -> dict[str, Any]:
    backend_identity = (
        None
        if bundle.proposal_backend_seal is None
        else bundle.proposal_backend_seal.identity_hash
    )
    coordinates = {
        "schema": _PRODUCTION_RESUME_BINDING_SCHEMA,
        "resume_identity_hash": bundle.resume_identity_hash,
        "proposal_protocol": bundle.manifest.proposal_protocol,
        "proposal_backend_identity_hash": backend_identity,
        "campaign_contract_hash": bundle.binding.contract.campaign_contract_hash,
        "campaign_binding_hash": bundle.binding.binding_hash,
        "controller_run_id": bundle.manifest.controller_run_id,
        "runtime_logical_config_hash": bundle.runtime_config.logical_identity_hash,
        "entropy_tape_hash": bundle.entropy_tape.tape_hash,
    }
    return {
        **coordinates,
        "binding_receipt_hash": content_hash(
            coordinates,
            prefix="partition-free-v1-production-resume-binding-",
        ),
    }


def _read_production_resume_binding(path: Path) -> dict[str, Any]:
    _require_nonsymlink_regular_file(path, label="production resume binding")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("production resume binding is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("production resume binding must contain a mapping")
    expected_keys = {
        "schema",
        "resume_identity_hash",
        "proposal_protocol",
        "proposal_backend_identity_hash",
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "runtime_logical_config_hash",
        "entropy_tape_hash",
        "binding_receipt_hash",
    }
    if set(payload) != expected_keys:
        raise ValueError("production resume binding has unexpected coordinates")
    coordinates = {
        name: payload[name] for name in expected_keys if name != "binding_receipt_hash"
    }
    if payload["schema"] != _PRODUCTION_RESUME_BINDING_SCHEMA or payload[
        "binding_receipt_hash"
    ] != content_hash(
        coordinates,
        prefix="partition-free-v1-production-resume-binding-",
    ):
        raise ValueError("production resume binding failed its receipt hash")
    return dict(payload)


def _validate_production_resume_binding(
    path: Path,
    *,
    bundle: V1ProductionRuntimeBundle,
) -> None:
    payload = _read_production_resume_binding(path)
    expected = _production_resume_binding_payload(bundle)
    if payload != expected:
        raise ValueError(
            "production resume identity changed across checkpoint restart"
        )


def _publish_or_validate_production_resume_binding(
    path: Path,
    *,
    bundle: V1ProductionRuntimeBundle,
) -> None:
    if _path_lexists(path):
        _validate_production_resume_binding(path, bundle=bundle)
        return
    _atomic_write_json(path, _production_resume_binding_payload(bundle))


def _initial_runtime(bundle: V1ProductionRuntimeBundle) -> V1RuntimeState:
    controller = initialize_v1_controller(
        binding=bundle.binding,
        controller_run_id=bundle.manifest.controller_run_id,
        initial_mask=bundle.initial_mask,
        pareto_milestone_registration=bundle.pareto_milestone_registration,
    )
    return initialize_v1_runtime(
        config=bundle.runtime_config,
        controller=controller,
        entropy_tape=bundle.entropy_tape,
    )


def _validate_completed_proposal_closure_ledger(
    state: V1RuntimeState,
    *,
    proposal_protocol: ProposalProtocolKind,
    proposal_backend_identity_hash: str | None,
) -> None:
    """Bind every permanent staged closure to its production protocol/backend."""

    closures = state.completed_proposal_closures
    if any(
        type(item) is not V1RuntimeCompletedProposalClosureReceipt
        for item in closures
    ):
        raise TypeError("production proposal closure ledger contains a nonreceipt")
    if proposal_protocol in {"legacy_synthetic_one_shot_v1","operator_binary_d_c_v1"}:
        if proposal_backend_identity_hash is not None:
            raise ValueError("legacy proposal protocol names a staged backend identity")
        if state.staged_proposal_backend_identity_hash is not None:
            raise ValueError("legacy proposal runtime retains a staged backend anchor")
        if closures:
            raise ValueError("legacy proposal protocol contains staged proposal closures")
        return
    if proposal_protocol != "staged_train_only_proposal_v1":
        raise ValueError("production proposal protocol is unsupported")
    backend_identity = _nonempty(
        proposal_backend_identity_hash,
        "proposal_backend_identity_hash",
    )
    runtime_backend_identity = state.staged_proposal_backend_identity_hash
    if (
        runtime_backend_identity is not None
        and runtime_backend_identity != backend_identity
    ):
        raise ValueError("staged proposal runtime names another permanent backend anchor")
    acquisition_hashes = state.controller.acquisition_receipt_hashes
    if len(closures) != len(acquisition_hashes):
        raise ValueError(
            "staged proposal closure ledger and acquisition lineage disagree"
        )
    contract_hash = state.controller.binding.contract.campaign_contract_hash
    binding_hash = state.controller.binding.binding_hash
    controller_run_id = state.controller.controller_run_id
    for generation, (closure, acquisition_hash) in enumerate(
        zip(closures, acquisition_hashes, strict=True)
    ):
        closure.__post_init__()
        if (
            closure.proposal_backend_identity_hash != backend_identity
            or closure.acquisition_generation_index != generation
            or closure.acquisition_receipt_hash != acquisition_hash
            or closure.campaign_contract_hash != contract_hash
            or closure.campaign_binding_hash != binding_hash
            or closure.controller_run_id != controller_run_id
        ):
            raise ValueError(
                "staged proposal closure differs from its backend/acquisition generation"
            )


def _validate_resumed_state(
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
) -> None:
    if state.config.logical_identity_hash != bundle.runtime_config.logical_identity_hash:
        raise ValueError("resumed runtime names another logical execution config")
    if state.controller.binding.binding_hash != bundle.binding.binding_hash:
        raise ValueError("resumed runtime names another campaign binding")
    if state.controller.controller_run_id != bundle.manifest.controller_run_id:
        raise ValueError("resumed runtime names another controller run")
    if state.controller.initial_mask.mask_hash != bundle.initial_mask.mask_hash:
        raise ValueError("resumed runtime names another initial mask")
    if state.entropy_tape.tape_hash != bundle.entropy_tape.tape_hash:
        raise ValueError("resumed runtime names another entropy tape")
    expected_milestone = bundle.pareto_milestone_registration
    observed_milestone = state.controller.pareto_milestone_registration
    if (expected_milestone is None) != (observed_milestone is None) or (
        expected_milestone is not None
        and observed_milestone is not None
        and expected_milestone.registration_hash
        != observed_milestone.registration_hash
    ):
        raise ValueError("resumed runtime names another Pareto milestone registration")
    if (
        bundle.performance_receipt_factory is not None
        and state.config.physical_execution_hash
        != bundle.runtime_config.physical_execution_hash
    ):
        raise ValueError(
            "performance-bearing resume cannot change its physical packing identity"
        )
    if state.completion is not None and state.completion.source_seal_hash != bundle.runtime_config.source_seal_hash:
        raise ValueError("resumed completion names another source seal")
    legacy_family_values = (
        state.terminal_validation_family_freeze,
        state.terminal_validation_family_request,
        state.terminal_validation_family_observation,
        state.terminal_validation_family,
    )
    comparison_family_values = (
        state.terminal_comparison_family_freeze,
        state.terminal_comparison_family_request,
        state.terminal_comparison_family_observation,
        state.terminal_comparison_family,
    )
    has_legacy_family_state = any(
        value is not None for value in legacy_family_values
    )
    has_comparison_family_state = any(
        value is not None for value in comparison_family_values
    )
    if bundle.terminal_validation_family_design is None:
        if has_legacy_family_state or has_comparison_family_state:
            raise PermissionError(
                "legacy production bundle cannot resume terminal-family authority"
            )
    elif bundle.has_legacy_terminal_validation_family_protocol:
        if state.terminal_validation is not None:
            raise PermissionError(
                "terminal-family production cannot resume legacy terminal authority"
            )
        if has_comparison_family_state:
            raise PermissionError(
                "legacy terminal-family production cannot resume v2 comparison authority"
            )
        family_freeze = state.terminal_validation_family_freeze
        design = bundle.terminal_validation_family_design
        assert type(design) is V1ProductionTerminalValidationFamilyDesign
        if family_freeze is not None and (
            family_freeze.terminal_evaluator_identity_hash
            != design.terminal_evaluator_identity_hash
            or family_freeze.terminal_validation_population_hash
            != design.terminal_validation_population_hash
            or family_freeze.terminal_validation_context_ids_hash
            != design.terminal_validation_context_ids_hash
            or family_freeze.terminal_validation_context_count
            != design.terminal_validation_context_count
            or family_freeze.terminal_metric_semantics_hash
            != design.terminal_metric_semantics_hash
            or family_freeze.dense_reference_hash != design.dense_reference_hash
        ):
            raise ValueError(
                "resumed terminal-family state differs from its production design"
            )
    else:
        if state.terminal_validation is not None or has_legacy_family_state:
            raise PermissionError(
                "terminal-comparison v2 cannot resume a legacy terminal branch"
            )
        family_freeze = state.terminal_comparison_family_freeze
        design = bundle.terminal_validation_family_design
        assert type(design) is V1ProductionTerminalComparisonFamilyDesignV2
        if family_freeze is not None and (
            family_freeze.terminal_evaluator_identity_hash
            != design.terminal_evaluator_identity_hash
            or family_freeze.terminal_validation_population_hash
            != design.terminal_validation_population_hash
            or family_freeze.terminal_validation_context_ids_hash
            != design.terminal_validation_context_ids_hash
            or family_freeze.terminal_validation_context_count
            != design.terminal_validation_context_count
            or family_freeze.terminal_metric_semantics_hash
            != design.terminal_metric_semantics_hash
            or family_freeze.pending_dense_reference_design_hash
            != design.pending_dense_reference_design_hash
            or family_freeze.dense_mask_hash != design.dense_mask_hash
            or family_freeze.current_v0_mask_hash
            != design.current_v0_mask_hash
            or family_freeze.current_v0_mask_artifact_sha256
            != design.current_v0_mask_artifact_sha256
            or family_freeze.current_v0_mask_artifact_semantic_hash
            != design.current_v0_mask_artifact_semantic_hash
            or family_freeze.current_v0_mask_artifact_byte_count
            != design.current_v0_mask_artifact_byte_count
            or family_freeze.dense_current_v0_calibration_family_freeze_hash
            != design.dense_current_v0_calibration_family_freeze_hash
        ):
            raise ValueError(
                "resumed terminal-comparison state differs from its production design"
            )
    if (
        state.proposal_progress is not None
        or state.pending_proposal_acquisition is not None
    ) and not bundle.has_staged_proposal_protocol:
        raise ValueError("resumed proposal acquisition requires staged proposal hooks")
    active = state.controller.active_candidate
    has_paired_candidate = bool(
        active is not None
        and (
            active.paired_bce_registration is not None
            or active.paired_bce_state is not None
        )
    )
    if (
        state.pending_paired_bce_chunk is not None
        or has_paired_candidate
    ) and not bundle.has_paired_bce_oracle_protocol:
        raise ValueError("resumed paired-BCE oracle state requires paired-BCE hooks")
    if (
        bundle.manifest.backend_kind == "alexnet_production"
        and has_paired_candidate
        and (
            active is None
            or active.paired_bce_registration is None
            or not active.paired_bce_registration.has_dynamic_full_population_authority
        )
    ):
        raise PermissionError(
            "AlexNet resume refuses static-manifest paired-BCE O authority"
        )
    _validate_completed_proposal_closure_ledger(
        state,
        proposal_protocol=bundle.manifest.proposal_protocol,
        proposal_backend_identity_hash=(
            None
            if bundle.proposal_backend_seal is None
            else bundle.proposal_backend_seal.identity_hash
        ),
    )
    if bundle.has_staged_proposal_protocol:
        backend = bundle.proposal_backend_seal
        if backend is None:  # pragma: no cover - bundle invariant
            raise RuntimeError("staged production bundle lost its proposal backend seal")
        progress = state.proposal_progress
        if progress is not None:
            if progress.proposal_backend_identity_hash != backend.identity_hash:
                raise ValueError("resumed proposal state names another backend seal")
            for descriptor in (
                progress.initial_sufficient_state,
                progress.current_sufficient_state,
            ):
                if (
                    descriptor.representation_hash
                    != backend.sufficient_state_representation_hash
                ):
                    raise ValueError(
                        "resumed proposal state has another sufficient-state ABI"
                    )


def _restore_resumed_provider_process_state(
    bundle: V1ProductionRuntimeBundle,
    controller: V1ControllerState,
    *, runtime_state: V1RuntimeState | None = None,
) -> None:
    """Restore sealed evaluator-local mask state before resumed runtime actions."""

    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    observed = _sealed_evaluator_state_restoration_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
    )
    observed_spec = None if observed is None else observed[1]
    if observed_spec != bundle.provider_seal.evaluator_state_restoration_hook_spec:
        raise ValueError("paired-oracle evaluator restoration-hook identity changed")
    if observed is None:
        return
    operator=_bound_operator_factory(bundle)
    if operator is not None:
        if runtime_state is None or runtime_state.controller is not controller:
            raise ValueError('operator provider restoration requires its complete verified runtime')
        result=operator.adopt_restored_runtime(runtime_state)
    else:
        result = observed[0](controller)
    if result is not None:
        raise ValueError("paired-oracle evaluator restoration hook must return None")


def _retire_terminal_accepted_generation_state(
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    decision: ProductionDecision | None,
) -> bool:
    """Retire evaluator-local K/O state after the accepted stop is durable."""

    if (
        type(decision) is not V1PairedBCEDecision
        or decision.status != "accept_candidate"
        or state.controller.phase != "stopped"
    ):
        return False
    expected_spec = (
        bundle.provider_seal.evaluator_retire_committed_generation_state_hook_spec
    )
    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    observed = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
        method_name=_EVALUATOR_RETIRE_COMMITTED_GENERATION_STATE_METHOD,
        label="committed-generation-retirement",
    )
    observed_spec = None if observed is None else observed[1]
    if observed_spec != expected_spec:
        raise ValueError(
            "paired-oracle evaluator committed-generation retirement-hook "
            "identity changed"
        )
    if observed is None:
        if bundle.manifest.backend_kind == "alexnet_production":
            raise PermissionError(
                "AlexNet terminal paired accept requires its sealed "
                "committed-generation retirement hook"
            )
        return False
    operator=_bound_operator_factory(bundle)
    if operator is not None:
        from .modern_v1_operator_lifecycle import synchronize_operator_committed_mask
        synchronize_operator_committed_mask(controller=state.controller,native_evaluator=operator.native_evaluator)
        return True
    result = observed[0](state.controller)
    if result is not None:
        raise ValueError(
            "paired-oracle committed-generation retirement hook must return None"
        )
    return True


def _restore_resumed_proposal_backend_state(
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
) -> None:
    """Validate/reopen sealed backend-owned proposal state before any executor."""

    if not bundle.has_staged_proposal_protocol:
        return
    backend = bundle.proposal_backend_seal
    if backend is None:  # pragma: no cover - bundle invariant
        raise RuntimeError("staged production bundle lost its proposal backend seal")
    progress = state.proposal_progress
    if progress is not None:
        if progress.proposal_backend_identity_hash != backend.identity_hash:
            raise ValueError("resumed proposal state names another backend seal")
        for descriptor in (
            progress.initial_sufficient_state,
            progress.current_sufficient_state,
        ):
            if (
                descriptor.representation_hash
                != backend.sufficient_state_representation_hash
            ):
                raise ValueError("resumed proposal state has another sufficient-state ABI")
    validator = bundle.proposal_resume_validator
    if validator is None:
        return
    result = validator(state)
    if result is not None:
        raise ValueError("proposal resume validator must return None")


def _checkpoint_progress(
    path: Path,
    state: V1RuntimeState,
) -> tuple[V1RuntimeState, V1RuntimeCheckpointReceipt]:
    if state.controller.pending_pareto_milestone is None:
        return state, write_v1_runtime_checkpoint(path, state)
    return checkpoint_and_seal_v1_runtime_pareto_milestone(path, state)


def _pareto_milestone_archive_relative_path(
    checkpoint_path: Path,
    milestone: V1ParetoMilestoneReceipt,
) -> Path:
    return Path(f"{checkpoint_path.name}.pareto_milestones") / (
        f"{milestone.pending.pending_hash}.mv1ckpt"
    )


def _read_exact_pareto_milestone_archive(
    *,
    checkpoint_path: Path,
    state: V1RuntimeState,
) -> tuple[Path, V1RuntimeState, V1RuntimeCheckpointReceipt]:
    """Restore the exact full milestone mask from its immutable checkpoint."""

    milestone = state.controller.pareto_milestone_receipt
    if type(milestone) is not V1ParetoMilestoneReceipt:
        raise PermissionError(
            "terminal-family validation requires a sealed Pareto milestone"
        )
    relative = _pareto_milestone_archive_relative_path(checkpoint_path, milestone)
    family_freeze = state.terminal_validation_family_freeze
    if (
        family_freeze is not None
        and family_freeze.pareto_milestone_archive_relative_path
        != relative.as_posix()
    ):
        raise ValueError(
            "terminal-family freeze names a noncanonical checkpoint archive"
        )
    archive_path = checkpoint_path.parent / relative
    _require_nonsymlink_regular_file(
        archive_path,
        label="Pareto milestone checkpoint archive",
    )
    binding = milestone.checkpoint_binding
    if _sha256_file(archive_path) != binding.checkpoint_artifact_hash:
        raise ValueError("Pareto milestone archive bytes changed after sealing")
    archived, archive_receipt = read_v1_runtime_checkpoint(
        archive_path,
        trusted_local=True,
    )
    pending = archived.controller.pending_pareto_milestone
    snapshot = archived.controller.snapshot_receipt()
    if (
        pending is None
        or pending.pending_hash != milestone.pending.pending_hash
        or archived.controller.pareto_milestone_receipt is not None
        or archived.controller.current_mask.mask_hash
        != milestone.pending.milestone_mask_hash
        or archived.controller.current_mask.retained_count
        != milestone.pending.comparator_retained_route_count
        or archived.controller.accepted_step_count
        != milestone.pending.accepted_step_count
        or archived.controller.state_hash != binding.controller_state_hash
        or snapshot.receipt_hash != binding.controller_snapshot_receipt_hash
        or archive_receipt.schema != binding.checkpoint_codec_schema
        or any(
            value is not None
            for value in (
                archived.terminal_freeze,
                archived.terminal_validation,
                archived.terminal_validation_family_freeze,
                archived.terminal_validation_family_request,
                archived.terminal_validation_family_observation,
                archived.terminal_validation_family,
                archived.terminal_comparison_family_freeze,
                archived.terminal_comparison_family_request,
                archived.terminal_comparison_family_observation,
                archived.terminal_comparison_family,
                archived.completion,
            )
        )
    ):
        raise ValueError(
            "Pareto milestone archive does not restore its exact pending state"
        )
    return relative, archived, archive_receipt


def _materialize_resumed_pareto_milestone_archive(
    *,
    state: V1RuntimeState,
    source_checkpoint: Path,
    destination_checkpoint: Path,
) -> None:
    milestone = state.controller.pareto_milestone_receipt
    if milestone is None:
        return
    if type(milestone) is not V1ParetoMilestoneReceipt:
        raise TypeError("resumed Pareto milestone receipt has the wrong type")
    destination_relative = _pareto_milestone_archive_relative_path(
        destination_checkpoint,
        milestone,
    )
    family_freeze = state.terminal_validation_family_freeze
    source_relative = (
        _pareto_milestone_archive_relative_path(source_checkpoint, milestone)
        if family_freeze is None
        else Path(family_freeze.pareto_milestone_archive_relative_path)
    )
    if (
        family_freeze is not None
        and source_relative.as_posix() != destination_relative.as_posix()
    ):
        raise ValueError(
            "frozen terminal-family archive cannot be renamed across resume"
        )
    source = source_checkpoint.parent / source_relative
    destination = destination_checkpoint.parent / destination_relative
    _atomic_materialize_exact_file(
        source,
        destination,
        label="Pareto milestone checkpoint archive",
    )
    _read_exact_pareto_milestone_archive(
        checkpoint_path=destination_checkpoint,
        state=state,
    )


def _materialize_resumed_terminal_family_matrix(
    *,
    state: V1RuntimeState,
    source_checkpoint: Path,
    destination_checkpoint: Path,
) -> None:
    """Copy sidecar bytes without reading outcomes before request durability."""

    request = state.terminal_validation_family_request
    if request is None:
        return
    relative = Path(request.observation_artifact_relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0]
        != f"{destination_checkpoint.name}.terminal_validation"
    ):
        raise ValueError(
            "frozen terminal-family matrix cannot be renamed across resume"
        )
    source = source_checkpoint.parent / relative
    destination = destination_checkpoint.parent / relative
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return
    if _path_lexists(source):
        _atomic_materialize_exact_file(
            source,
            destination,
            label="terminal validation-family matrix",
        )
    elif state.terminal_validation_family_observation is not None:
        raise ValueError(
            "observed terminal-family resume lacks its immutable matrix sidecar"
        )


def _materialize_resumed_terminal_comparison_family_matrix_v2(
    *,
    state: V1RuntimeState,
    source_checkpoint: Path,
    destination_checkpoint: Path,
) -> None:
    """Copy opaque MV1TERM2 bytes while preserving the frozen request path."""

    request = state.terminal_comparison_family_request
    if request is None:
        return
    relative = Path(request.observation_artifact_relative_path)
    if (
        len(relative.parts) != 2
        or relative.parts[0]
        != f"{destination_checkpoint.name}.terminal_comparison"
        or relative.suffix != ".mv1term2"
    ):
        raise ValueError(
            "frozen terminal-comparison matrix cannot be renamed across resume"
        )
    source = source_checkpoint.parent / relative
    destination = destination_checkpoint.parent / relative
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return
    if _path_lexists(source):
        _atomic_materialize_exact_file(
            source,
            destination,
            label="terminal comparison-family matrix v2",
        )
    elif state.terminal_comparison_family_observation is not None:
        raise ValueError(
            "observed terminal-comparison resume lacks its immutable matrix sidecar"
        )


def _require_accepted_calibration_freeze(
    controller: V1ControllerState,
    *,
    accepted_step_count: int,
    expected_mask_hash: str,
) -> V1CandidateCalibrationFreeze:
    ledger = controller.accepted_oracle_trajectory
    if (
        ledger is None
        or accepted_step_count <= 0
        or len(ledger.entries) < accepted_step_count
    ):
        raise PermissionError(
            "terminal-family mask lacks accepted paired-BCE calibration lineage"
        )
    entry = ledger.entries[accepted_step_count - 1]
    if (
        entry.accepted_step_index != accepted_step_count - 1
        or entry.candidate_mask_hash != expected_mask_hash
    ):
        raise ValueError(
            "terminal-family calibration entry names another accepted mask"
        )
    return entry.require_terminal_milestone_calibration_freeze()


@dataclass(frozen=True, slots=True)
class _V1ProductionTerminalFamilyMaterial:
    family_freeze: V1TerminalValidationFamilyFreezeReceipt
    unique_lane_masks: tuple[HardMask, ...]
    unique_lane_calibrations: tuple[V1FrozenAffineCalibration, ...]
    archive_checkpoint_receipt_hash: str
    archive_runtime_state_hash: str


def _derive_terminal_validation_family_material(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    checkpoint_path: Path,
    terminal_extraction_closure: V1TerminalExtractionClosure,
) -> _V1ProductionTerminalFamilyMaterial:
    design = bundle.terminal_validation_family_design
    terminal_freeze = state.terminal_freeze
    if design is None or terminal_freeze is None:
        raise PermissionError(
            "terminal-family material requires its design and terminal freeze"
        )
    if (
        state.terminal_validation is not None
        or design.terminal_evaluator_identity_hash
        != bundle.terminal_validator.identity_hash
        or design.terminal_validation_population_hash
        != terminal_freeze.terminal_validation_population_hash
        or design.dense_reference_hash != terminal_freeze.dense_reference_hash
        or terminal_freeze.terminal_mask_hash
        != state.controller.current_mask.mask_hash
    ):
        raise ValueError(
            "terminal-family material differs from its frozen production coordinates"
        )
    relative, archived, archive_receipt = _read_exact_pareto_milestone_archive(
        checkpoint_path=checkpoint_path,
        state=state,
    )
    milestone = state.controller.pareto_milestone_receipt
    if milestone is None:  # pragma: no cover - archive helper invariant.
        raise RuntimeError("terminal-family material lost its milestone receipt")
    milestone_calibration = _require_accepted_calibration_freeze(
        archived.controller,
        accepted_step_count=milestone.pending.accepted_step_count,
        expected_mask_hash=milestone.pending.milestone_mask_hash,
    )
    final_calibration = _require_accepted_calibration_freeze(
        state.controller,
        accepted_step_count=state.controller.accepted_step_count,
        expected_mask_hash=state.controller.current_mask.mask_hash,
    )
    if (
        milestone_calibration.dense_referent_mask_hash
        != bundle.initial_mask.mask_hash
        or final_calibration.dense_referent_mask_hash
        != bundle.initial_mask.mask_hash
        or terminal_extraction_closure.extraction_calibration_hash
        != final_calibration.candidate_calibration.calibration_hash
        or terminal_freeze.extraction_calibration_hash
        != final_calibration.candidate_calibration.calibration_hash
    ):
        raise ValueError(
            "terminal extraction and accepted train-only calibration differ"
        )

    role_masks = (
        bundle.initial_mask,
        archived.controller.current_mask,
        state.controller.current_mask,
    )
    role_calibrations = (
        milestone_calibration.dense_calibration,
        milestone_calibration.candidate_calibration,
        final_calibration.candidate_calibration,
    )
    unique_masks: list[HardMask] = []
    unique_calibrations: list[V1FrozenAffineCalibration] = []
    role_to_unique_lane: list[int] = []
    for mask, calibration in zip(role_masks, role_calibrations, strict=True):
        matching = next(
            (
                index
                for index, existing in enumerate(unique_masks)
                if existing.mask_hash == mask.mask_hash
            ),
            None,
        )
        if matching is None:
            unique_masks.append(mask)
            unique_calibrations.append(calibration)
            role_to_unique_lane.append(len(unique_masks) - 1)
        else:
            if (
                unique_calibrations[matching].frozen_calibration_hash
                != calibration.frozen_calibration_hash
            ):
                raise ValueError(
                    "one terminal-family mask has two train-only calibrations"
                )
            role_to_unique_lane.append(matching)

    family_freeze = V1TerminalValidationFamilyFreezeReceipt(
        terminal_freeze_hash=terminal_freeze.receipt_hash,
        terminal_validation_population_hash=(
            design.terminal_validation_population_hash
        ),
        terminal_validation_context_ids_hash=(
            design.terminal_validation_context_ids_hash
        ),
        terminal_validation_context_count=(
            design.terminal_validation_context_count
        ),
        terminal_metric_semantics_hash=design.terminal_metric_semantics_hash,
        terminal_evaluator_identity_hash=design.terminal_evaluator_identity_hash,
        dense_reference_hash=design.dense_reference_hash,
        dense_mask_hash=bundle.initial_mask.mask_hash,
        route_axis_hash=bundle.initial_mask.route_axis_hash,
        route_count=bundle.initial_mask.route_count,
        pareto_milestone_receipt=milestone,
        pareto_milestone_archive_relative_path=relative.as_posix(),
        pareto_milestone_archive_sha256=(
            milestone.checkpoint_binding.checkpoint_artifact_hash
        ),
        final_mask_hash=state.controller.current_mask.mask_hash,
        role_mask_hashes=tuple(mask.mask_hash for mask in role_masks),
        role_retained_route_counts=tuple(
            mask.retained_count for mask in role_masks
        ),
        role_to_unique_lane=tuple(role_to_unique_lane),
        unique_lane_mask_hashes=tuple(mask.mask_hash for mask in unique_masks),
        unique_lane_calibration_hashes=tuple(
            calibration.calibration_hash for calibration in unique_calibrations
        ),
        unique_lane_calibration_freeze_hashes=tuple(
            calibration.frozen_calibration_hash
            for calibration in unique_calibrations
        ),
        unique_lane_calibration_freeze_payloads=tuple(
            calibration.to_payload() for calibration in unique_calibrations
        ),
        freeze_sequence=terminal_freeze.freeze_sequence + 1,
    )
    return _V1ProductionTerminalFamilyMaterial(
        family_freeze=family_freeze,
        unique_lane_masks=tuple(unique_masks),
        unique_lane_calibrations=tuple(unique_calibrations),
        # ``V1RuntimeCheckpointReceipt.receipt_hash`` historically includes
        # the absolute checkpoint path.  The milestone binding remains the
        # source-owned identity across relocation; exact archive bytes and the
        # restored controller/state coordinates above prove the local replica.
        archive_checkpoint_receipt_hash=(
            milestone.checkpoint_binding.checkpoint_manifest_hash
        ),
        archive_runtime_state_hash=archived.runtime_state_hash,
    )


def _terminal_family_execution(
    *,
    request: V1TerminalValidationFamilyRequestReceipt,
    material: _V1ProductionTerminalFamilyMaterial,
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt,
) -> V1ProductionTerminalValidationFamilyExecution:
    if request.family_freeze.receipt_hash != material.family_freeze.receipt_hash:
        raise ValueError("terminal-family request and derived material differ")
    return V1ProductionTerminalValidationFamilyExecution(
        request=request,
        unique_lane_masks=material.unique_lane_masks,
        unique_lane_calibrations=material.unique_lane_calibrations,
        request_checkpoint_receipt=request_checkpoint_receipt,
        pareto_milestone_archive_checkpoint_receipt_hash=(
            material.archive_checkpoint_receipt_hash
        ),
        pareto_milestone_archive_runtime_state_hash=(
            material.archive_runtime_state_hash
        ),
    )


@dataclass(frozen=True, slots=True)
class _V1ProductionTerminalComparisonFamilyMaterialV2:
    family_freeze: V1TerminalComparisonFamilyFreezeReceiptV2
    unique_lane_masks: tuple[HardMask, ...]
    unique_lane_calibrations: tuple[V1FrozenAffineCalibration, ...]


def _derive_terminal_comparison_family_material_v2(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    terminal_extraction_closure: V1TerminalExtractionClosure,
) -> _V1ProductionTerminalComparisonFamilyMaterialV2:
    """Freeze exact v2 lanes without opening or importing validation outcomes."""

    design = bundle.terminal_validation_family_design
    terminal_freeze = state.terminal_freeze
    current_v0_mask = bundle.terminal_comparison_current_v0_mask
    current_v0_freeze = (
        bundle.terminal_comparison_current_v0_calibration_freeze
    )
    if (
        type(design) is not V1ProductionTerminalComparisonFamilyDesignV2
        or terminal_freeze is None
        or type(current_v0_mask) is not HardMask
        or type(current_v0_freeze) is not V1CandidateCalibrationFreeze
    ):
        raise PermissionError(
            "terminal-comparison material requires its v2 design, terminal freeze, "
            "and exact current-v0 train-only owners"
        )
    if (
        state.terminal_validation is not None
        or state.terminal_validation_family is not None
        or design.terminal_evaluator_identity_hash
        != bundle.terminal_validator.identity_hash
        or design.terminal_validation_population_hash
        != terminal_freeze.terminal_validation_population_hash
        or design.pending_dense_reference_design_hash
        != terminal_freeze.dense_reference_hash
        or terminal_freeze.terminal_mask_hash
        != state.controller.current_mask.mask_hash
    ):
        raise ValueError(
            "terminal-comparison material differs from its frozen production coordinates"
        )
    current_v0_freeze.__post_init__()
    dense_calibration = current_v0_freeze.dense_calibration
    current_v0_calibration = current_v0_freeze.candidate_calibration
    if state.controller.accepted_step_count > 0:
        final_freeze = _require_accepted_calibration_freeze(
            state.controller,
            accepted_step_count=state.controller.accepted_step_count,
            expected_mask_hash=state.controller.current_mask.mask_hash,
        )
        final_calibration = final_freeze.candidate_calibration
    elif state.controller.current_mask.mask_hash == bundle.initial_mask.mask_hash:
        final_calibration = dense_calibration
    else:
        raise PermissionError(
            "non-dense terminal-v1 mask lacks accepted train-only calibration lineage"
        )
    if (
        current_v0_freeze.freeze_hash
        != design.dense_current_v0_calibration_family_freeze_hash
        or current_v0_freeze.dense_referent_mask_hash
        != bundle.initial_mask.mask_hash
        or current_v0_freeze.candidate_mask_hash != current_v0_mask.mask_hash
        or dense_calibration.frozen_calibration_hash
        != design.dense_calibration_freeze_hash
        or current_v0_calibration.frozen_calibration_hash
        != design.current_v0_calibration_freeze_hash
        or terminal_extraction_closure.extraction_calibration_hash
        != final_calibration.calibration_hash
        or terminal_freeze.extraction_calibration_hash
        != final_calibration.calibration_hash
        or any(
            calibration.validation_independent is not True
            or calibration.terminal_rows_used_for_fit is not False
            or calibration.terminal_outcomes_opened is not False
            for calibration in (
                dense_calibration,
                current_v0_calibration,
                final_calibration,
            )
        )
    ):
        raise ValueError(
            "terminal-comparison affine freezes differ from their validation-independent owners"
        )

    role_masks = (
        bundle.initial_mask,
        current_v0_mask,
        state.controller.current_mask,
    )
    role_calibrations = (
        dense_calibration,
        current_v0_calibration,
        final_calibration,
    )
    unique_masks: list[HardMask] = []
    unique_calibrations: list[V1FrozenAffineCalibration] = []
    role_to_unique_lane: list[int] = []
    for mask, calibration in zip(role_masks, role_calibrations, strict=True):
        matching = next(
            (
                index
                for index, existing in enumerate(unique_masks)
                if existing.mask_hash == mask.mask_hash
            ),
            None,
        )
        if matching is None:
            unique_masks.append(mask)
            unique_calibrations.append(calibration)
            role_to_unique_lane.append(len(unique_masks) - 1)
        else:
            if (
                unique_masks[matching].packed_bits != mask.packed_bits
                or unique_calibrations[matching].frozen_calibration_hash
                != calibration.frozen_calibration_hash
            ):
                raise ValueError(
                    "one terminal-comparison mask has two byte identities or calibrations"
                )
            role_to_unique_lane.append(matching)

    family_freeze = V1TerminalComparisonFamilyFreezeReceiptV2(
        terminal_freeze_hash=terminal_freeze.receipt_hash,
        terminal_validation_population_hash=(
            design.terminal_validation_population_hash
        ),
        terminal_validation_context_ids_hash=(
            design.terminal_validation_context_ids_hash
        ),
        terminal_validation_context_count=(
            design.terminal_validation_context_count
        ),
        terminal_metric_semantics_hash=design.terminal_metric_semantics_hash,
        terminal_evaluator_identity_hash=design.terminal_evaluator_identity_hash,
        pending_dense_reference_design_hash=(
            design.pending_dense_reference_design_hash
        ),
        dense_mask_hash=bundle.initial_mask.mask_hash,
        current_v0_mask_hash=current_v0_mask.mask_hash,
        current_v0_mask_artifact_sha256=(
            design.current_v0_mask_artifact_sha256
        ),
        current_v0_mask_artifact_semantic_hash=(
            design.current_v0_mask_artifact_semantic_hash
        ),
        current_v0_mask_artifact_byte_count=(
            design.current_v0_mask_artifact_byte_count
        ),
        dense_current_v0_calibration_family_freeze_hash=(
            design.dense_current_v0_calibration_family_freeze_hash
        ),
        final_v1_mask_hash=state.controller.current_mask.mask_hash,
        route_axis_hash=bundle.initial_mask.route_axis_hash,
        route_count=bundle.initial_mask.route_count,
        role_mask_hashes=tuple(mask.mask_hash for mask in role_masks),
        role_retained_route_counts=tuple(
            mask.retained_count for mask in role_masks
        ),
        role_to_unique_lane=tuple(role_to_unique_lane),
        unique_lane_mask_hashes=tuple(mask.mask_hash for mask in unique_masks),
        unique_lane_calibration_hashes=tuple(
            calibration.calibration_hash for calibration in unique_calibrations
        ),
        unique_lane_calibration_freeze_hashes=tuple(
            calibration.frozen_calibration_hash
            for calibration in unique_calibrations
        ),
        unique_lane_calibration_freeze_payloads=tuple(
            calibration.to_payload() for calibration in unique_calibrations
        ),
        freeze_sequence=terminal_freeze.freeze_sequence + 1,
    )
    return _V1ProductionTerminalComparisonFamilyMaterialV2(
        family_freeze=family_freeze,
        unique_lane_masks=tuple(unique_masks),
        unique_lane_calibrations=tuple(unique_calibrations),
    )


def _terminal_comparison_family_execution_v2(
    *,
    request: V1TerminalComparisonFamilyRequestReceiptV2,
    material: _V1ProductionTerminalComparisonFamilyMaterialV2,
    request_checkpoint_receipt: V1RuntimeCheckpointReceipt,
) -> V1ProductionTerminalComparisonFamilyExecutionV2:
    if request.family_freeze.receipt_hash != material.family_freeze.receipt_hash:
        raise ValueError("terminal-comparison request and derived material differ")
    freeze = material.family_freeze
    return V1ProductionTerminalComparisonFamilyExecutionV2(
        request=request,
        unique_lane_masks=material.unique_lane_masks,
        unique_lane_calibrations=material.unique_lane_calibrations,
        request_checkpoint_receipt=request_checkpoint_receipt,
        current_v0_mask_artifact_sha256=(
            freeze.current_v0_mask_artifact_sha256
        ),
        current_v0_mask_artifact_semantic_hash=(
            freeze.current_v0_mask_artifact_semantic_hash
        ),
        current_v0_mask_artifact_byte_count=(
            freeze.current_v0_mask_artifact_byte_count
        ),
    )


def _production_decision_hash(decision: ProductionDecision) -> str:
    if type(decision) is V1StagedAlphaDecision:
        return decision.decision_receipt_hash
    if type(decision) is V1PairedBCEDecision:
        return decision.decision_hash
    if type(decision) is V1PairedBCECalibrationHorizonSkipDecision:
        return decision.decision_hash
    raise TypeError("production decision is outside the sealed evidence union")


def _paired_bce_metadata(bundle: V1ProductionRuntimeBundle) -> Mapping[str, Any] | None:
    raw = bundle.manifest.metadata.get("paired_bce_oracle")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("paired-BCE production metadata must be a mapping")
    if raw.get("schema") != _PAIRED_BCE_PRODUCTION_ORACLE_SCHEMA:
        raise ValueError("paired-BCE production metadata has an unsupported schema")
    return raw


def _named_axis_map(value: object, *, label: str) -> dict[str, tuple[str, ...]]:
    if isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif isinstance(value, (tuple, list)):
        raw_items = tuple(value)
    else:
        raise TypeError(f"{label} must be a mapping or sequence of named axes")
    result: dict[str, tuple[str, ...]] = {}
    for item in raw_items:
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or type(item[0]) is not str
        ):
            raise TypeError(f"{label} entries must be (name, context_ids)")
        name = item[0]
        if name in result:
            raise ValueError(f"{label} repeats stratum {name!r}")
        contexts = item[1]
        if isinstance(contexts, (str, bytes)) or not isinstance(contexts, (tuple, list)):
            raise TypeError(f"{label}.{name} must be a sequence of context ids")
        axis = tuple(_nonempty(str(context), f"{label}.{name}") for context in contexts)
        if not axis or len(set(axis)) != len(axis):
            raise ValueError(f"{label}.{name} must be finite, nonempty, and unique")
        result[name] = axis
    if set(result) != set(PAIRED_BCE_STRATA):
        raise ValueError(f"{label} must contain exactly target and complement")
    return result


def _metadata_identity(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default_payload: Mapping[str, Any],
    prefix: str,
) -> str:
    value = metadata.get(key)
    if value is not None:
        return _nonempty(value, key)
    return content_hash(default_payload, prefix=prefix)


def _metadata_positive_int(
    metadata: Mapping[str, Any],
    key: str,
    default: int,
) -> int:
    value = metadata.get(key, default)
    return _positive(value, key)


def _metadata_finite_float(
    metadata: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = float(metadata.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _paired_bce_sealed_orders(metadata: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    raw_orders = metadata.get(
        "sealed_order_context_ids_by_stratum",
        metadata.get("oracle_context_ids_by_stratum"),
    )
    if raw_orders is None:
        raise ValueError("paired-BCE production metadata lacks sealed oracle orders")
    return _named_axis_map(raw_orders, label="paired-BCE sealed oracle orders")


@dataclass(frozen=True, slots=True)
class _ProductionPairedBCEAttemptOrders:
    orders: Mapping[str, tuple[str, ...]]
    receipt_hashes: Mapping[str, str]
    order_entropy_commitment_hash: str
    source_kind: Literal[
        "dynamic_oracle_frame",
        "attempt_specific_receipts",
        "static_compatibility",
    ]

    def __post_init__(self) -> None:
        normalized_orders = {
            name: tuple(_nonempty(value, f"paired-BCE {name} order context") for value in values)
            for name, values in self.orders.items()
        }
        normalized_receipts = {
            name: _nonempty(value, f"paired-BCE {name} order receipt hash")
            for name, value in self.receipt_hashes.items()
        }
        if (
            set(normalized_orders) != set(PAIRED_BCE_STRATA)
            or set(normalized_receipts) != set(PAIRED_BCE_STRATA)
            or any(
                not values or len(values) != len(set(values))
                for values in normalized_orders.values()
            )
        ):
            raise ValueError("paired-BCE attempt orders must cover both unique strata")
        if self.source_kind not in {
            "dynamic_oracle_frame",
            "attempt_specific_receipts",
            "static_compatibility",
        }:
            raise ValueError("paired-BCE attempt order source is unsupported")
        object.__setattr__(self, "orders", MappingProxyType(normalized_orders))
        object.__setattr__(
            self,
            "receipt_hashes",
            MappingProxyType(normalized_receipts),
        )
        object.__setattr__(
            self,
            "order_entropy_commitment_hash",
            _nonempty(
                self.order_entropy_commitment_hash,
                "paired-BCE order entropy commitment hash",
            ),
        )


def _production_active_dynamic_oracle_frame(
    bundle: V1ProductionRuntimeBundle,
    *,
    required: bool,
) -> object | None:
    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    observed = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
        method_name=_EVALUATOR_ACTIVE_DYNAMIC_ORACLE_FRAME_METHOD,
        label="active-dynamic-oracle-frame",
    )
    observed_spec = None if observed is None else observed[1]
    if observed_spec != bundle.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec:
        raise ValueError("paired-BCE dynamic O-frame hook identity changed")
    if observed is None:
        if required:
            raise PermissionError(
                "production paired-BCE requires a sealed generation-local O-frame hook"
            )
        return None
    from .modern_v1_alexnet_fq_registration import (  # noqa: PLC0415
        V1AlexNetFQOracleSamplingFrameReceipt,
    )

    frame = observed[0]()
    if type(frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
        raise TypeError("paired-BCE dynamic O-frame hook returned the wrong type")
    frame.assert_integrity()
    return frame


def _dynamic_attempt_orders(
    frame: object,
    dynamic: V1PairedBCEDynamicOracleReceipt,
) -> _ProductionPairedBCEAttemptOrders:
    from .modern_v1_alexnet_fq_registration import (  # noqa: PLC0415
        V1AlexNetFQOracleSamplingFrameReceipt,
    )

    if type(frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
        raise TypeError("dynamic attempt orders require an exact O frame")
    frame.assert_integrity(dynamic.oracle_sampling_frame_receipt_hash)
    orders = dict(frame.oracle_context_ids_by_stratum)
    receipts = (
        frame.target_assignment_receipt,
        frame.complement_assignment_receipt,
    )
    result = _ProductionPairedBCEAttemptOrders(
        orders={name: tuple(orders[name]) for name in PAIRED_BCE_STRATA},
        receipt_hashes={
            name: receipts[index].content_hash
            for index, name in enumerate(PAIRED_BCE_STRATA)
        },
        order_entropy_commitment_hash=(
            dynamic.oracle_order_entropy_commitment_hash
        ),
        source_kind="dynamic_oracle_frame",
    )
    if (
        tuple(
            paired_bce_sealed_order_hash(
                stratum_id=name,
                context_ids_in_sealed_order=result.orders[name],
            )
            for name in PAIRED_BCE_STRATA
        )
        != dynamic.oracle_sealed_order_hashes
        or tuple(result.receipt_hashes[name] for name in PAIRED_BCE_STRATA)
        != dynamic.oracle_permutation_receipt_hashes
    ):
        raise ValueError("rehydrated dynamic O orders differ from registration")
    return result


@dataclass(frozen=True, slots=True)
class _ProductionPairedBCEAttemptAlphaCoordinate:
    oracle_attempt_allocations: tuple[float, ...]
    candidate_attempt_alpha_axis_hash: str
    master_decision_alpha_registration_hash: str | None
    candidate_attempt_index: int
    maximum_candidate_attempts: int
    oracle_familywise_alpha: float
    alpha_committed_before_attempt: float
    attempt_alpha_allocation: float
    source_kind: Literal[
        "master_decision_alpha_registration",
        "canonical_oracle_attempt_alpha_axis",
        "static_compatibility_uniform_axis",
    ]

    def __post_init__(self) -> None:
        maximum = _positive(
            self.maximum_candidate_attempts,
            "paired-BCE maximum candidate attempts",
        )
        index = self.candidate_attempt_index
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("paired-BCE candidate attempt index must be nonnegative")
        if index >= maximum:
            raise ValueError("paired-BCE alpha coordinate leaves its axis")
        axis = tuple(
            _production_paired_bce_unit_alpha(
                value,
                f"paired-BCE oracle_attempt_allocations[{axis_index}]",
            )
            for axis_index, value in enumerate(self.oracle_attempt_allocations)
        )
        if len(axis) != maximum:
            raise ValueError(
                "paired-BCE oracle_attempt_allocations length must equal the "
                "candidate-attempt horizon"
            )
        familywise = _production_paired_bce_unit_alpha(
            self.oracle_familywise_alpha,
            "paired-BCE oracle familywise alpha",
        )
        observed = math.fsum(axis)
        tolerance = 64.0 * math.ulp(max(1.0, abs(familywise), abs(observed)))
        if abs(observed - familywise) > tolerance:
            raise ValueError(
                "paired-BCE oracle_attempt_allocations must fsum to the "
                "familywise alpha reserve"
            )
        committed = float(self.alpha_committed_before_attempt)
        expected_committed = math.fsum(axis[:index])
        if (
            not math.isfinite(committed)
            or abs(committed - expected_committed)
            > 64.0 * math.ulp(
                max(1.0, abs(committed), abs(expected_committed))
            )
        ):
            raise ValueError(
                "paired-BCE alpha committed before attempt must be derived "
                "from the allocation axis"
            )
        allocation = _production_paired_bce_unit_alpha(
            self.attempt_alpha_allocation,
            "paired-BCE attempt alpha allocation",
        )
        expected_allocation = axis[index]
        if (
            abs(allocation - expected_allocation)
            > 64.0 * math.ulp(max(1.0, abs(allocation), abs(expected_allocation)))
        ):
            raise ValueError(
                "paired-BCE attempt alpha allocation must be selected from "
                "the allocation axis"
            )
        if self.source_kind not in {
            "master_decision_alpha_registration",
            "canonical_oracle_attempt_alpha_axis",
            "static_compatibility_uniform_axis",
        }:
            raise ValueError("paired-BCE alpha axis source is unsupported")
        master_hash = (
            None
            if self.master_decision_alpha_registration_hash is None
            else _nonempty(
                self.master_decision_alpha_registration_hash,
                "master_decision_alpha_registration_hash",
            )
        )
        if self.source_kind != "static_compatibility_uniform_axis" and master_hash is None:
            raise PermissionError(
                "canonical paired-BCE alpha coordinates must bind the master "
                "decision-alpha registration hash"
            )
        object.__setattr__(self, "oracle_attempt_allocations", axis)
        object.__setattr__(
            self,
            "candidate_attempt_alpha_axis_hash",
            _nonempty(
                self.candidate_attempt_alpha_axis_hash,
                "paired-BCE candidate attempt alpha axis hash",
            ),
        )
        object.__setattr__(
            self,
            "master_decision_alpha_registration_hash",
            master_hash,
        )
        object.__setattr__(self, "candidate_attempt_index", int(index))
        object.__setattr__(self, "maximum_candidate_attempts", maximum)
        object.__setattr__(self, "oracle_familywise_alpha", familywise)
        object.__setattr__(self, "alpha_committed_before_attempt", expected_committed)
        object.__setattr__(self, "attempt_alpha_allocation", expected_allocation)


def _production_paired_bce_unit_alpha(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a numeric alpha")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result >= 1.0:
        raise ValueError(f"{label} must lie in (0, 1)")
    return result


def _alpha_close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 64.0 * math.ulp(
        max(1.0, abs(float(left)), abs(float(right)))
    )


def _alpha_axis_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        _alpha_close(a, b) for a, b in zip(left, right, strict=True)
    )


def _production_paired_bce_alpha_axis_from_metadata(value: object) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise TypeError("paired-BCE oracle_attempt_allocations must be a sequence")
    return tuple(
        _production_paired_bce_unit_alpha(
            item,
            f"paired-BCE oracle_attempt_allocations[{index}]",
        )
        for index, item in enumerate(value)
    )


def _metadata_strict_bool(metadata: Mapping[str, Any], key: str) -> bool:
    value = metadata.get(key)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a strict boolean")
    return value


def _production_paired_bce_master_alpha_registration(
    metadata: Mapping[str, Any],
) -> V1MasterDecisionAlphaRegistration | None:
    raw = metadata.get(
        "master_decision_alpha_registration",
        metadata.get("master_decision_alpha_registration_payload"),
    )
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(
            "master_decision_alpha_registration must be a frozen payload mapping"
        )
    expected_hash = _nonempty(
        metadata.get("master_decision_alpha_registration_hash"),
        "master_decision_alpha_registration_hash",
    )
    master = V1MasterDecisionAlphaRegistration(**dict(raw))  # type: ignore[arg-type]
    if master.registration_hash != expected_hash:
        raise ValueError(
            "master_decision_alpha_registration_hash does not match its payload"
        )
    return master


def _production_paired_bce_canonical_oracle_attempt_alpha_axis_hash(
    *,
    campaign_contract_hash: str,
    controller_run_hash: str,
    preoutcome_registration_receipt_hash: str,
    maximum_oracle_attempts: int,
    oracle_familywise_alpha: float,
    oracle_attempt_allocations: tuple[float, ...],
    unused_alpha_nonrecycling: bool,
    oracle_reserve_nontransferable: bool,
) -> str:
    return content_hash(
        {
            "schema": ORACLE_ATTEMPT_ALPHA_AXIS_SCHEMA,
            "campaign_contract_hash": _nonempty(
                campaign_contract_hash,
                "paired-BCE alpha-axis campaign contract hash",
            ),
            "controller_run_hash": _nonempty(
                controller_run_hash,
                "paired-BCE alpha-axis controller run hash",
            ),
            "preoutcome_registration_receipt_hash": _nonempty(
                preoutcome_registration_receipt_hash,
                "paired-BCE alpha-axis preoutcome registration receipt hash",
            ),
            "maximum_oracle_attempts": _positive(
                maximum_oracle_attempts,
                "paired-BCE alpha-axis oracle-attempt horizon",
            ),
            "oracle_familywise_alpha": _production_paired_bce_unit_alpha(
                oracle_familywise_alpha,
                "paired-BCE alpha-axis oracle familywise alpha",
            ),
            "oracle_attempt_allocations": tuple(
                _production_paired_bce_unit_alpha(
                    value,
                    "paired-BCE alpha-axis oracle attempt allocation",
                )
                for value in oracle_attempt_allocations
            ),
            "allocation_axis": "global_candidate_attempt_index",
            "unused_alpha_nonrecycling": bool(unused_alpha_nonrecycling),
            "oracle_reserve_nontransferable": bool(oracle_reserve_nontransferable),
        },
        prefix="partition-free-v1-oracle-attempt-alpha-axis-",
    )


def _production_paired_bce_attempt_alpha_coordinate(
    bundle: V1ProductionRuntimeBundle,
    *,
    metadata: Mapping[str, Any],
    controller: V1ControllerState,
) -> _ProductionPairedBCEAttemptAlphaCoordinate:
    contract = controller.binding.contract
    maximum_attempts = contract.maximum_candidate_count
    attempt_index = controller.candidate_attempt_count
    familywise = _production_paired_bce_unit_alpha(
        metadata.get("oracle_familywise_alpha", contract.campaign_wide_alpha),
        "paired-BCE oracle familywise alpha",
    )
    if "maximum_candidate_attempts" in metadata and (
        _positive(metadata["maximum_candidate_attempts"], "maximum_candidate_attempts")
        != maximum_attempts
    ):
        raise ValueError(
            "paired-BCE oracle_attempt_allocations horizon disagrees with the "
            "campaign contract"
        )
    if "oracle_attempt_allocation_horizon" in metadata and (
        _positive(
            metadata["oracle_attempt_allocation_horizon"],
            "oracle_attempt_allocation_horizon",
        )
        != maximum_attempts
    ):
        raise ValueError(
            "paired-BCE oracle_attempt_allocations horizon disagrees with the "
            "campaign contract"
        )

    master = _production_paired_bce_master_alpha_registration(metadata)
    if master is not None:
        if master.campaign_contract_hash != contract.campaign_contract_hash:
            raise ValueError(
                "master decision-alpha registration names another campaign"
            )
        if master.controller_run_hash != controller.controller_run_id:
            raise ValueError(
                "master decision-alpha registration names another controller run"
            )
        if master.maximum_oracle_attempts != maximum_attempts:
            raise ValueError(
                "master decision-alpha registration horizon disagrees with "
                "the campaign contract"
            )
        if not _alpha_close(master.oracle_familywise_alpha, familywise):
            raise ValueError(
                "master decision-alpha registration oracle familywise alpha "
                "differs from production metadata"
            )
        axis = tuple(master.oracle_attempt_allocations)
        if "oracle_attempt_allocations" in metadata:
            declared_axis = _production_paired_bce_alpha_axis_from_metadata(
                metadata["oracle_attempt_allocations"]
            )
            if not _alpha_axis_close(declared_axis, axis):
                raise ValueError(
                    "production oracle_attempt_allocations differ from the "
                    "master decision-alpha registration"
                )
        axis_hash = master.oracle_attempt_alpha_axis_hash
        declared_axis_hash = _nonempty(
            metadata.get("candidate_attempt_alpha_axis_hash"),
            "candidate_attempt_alpha_axis_hash",
        )
        if declared_axis_hash != axis_hash:
            raise ValueError(
                "candidate_attempt_alpha_axis_hash differs from the master "
                "decision-alpha registration"
            )
        source_kind = "master_decision_alpha_registration"
        master_hash = master.registration_hash
    else:
        raw_axis = metadata.get("oracle_attempt_allocations")
        if raw_axis is None:
            if not bool(metadata.get("allow_static_alpha_allocation_compatibility", False)):
                raise PermissionError(
                    "paired-BCE production requires preregistered "
                    "oracle_attempt_allocations"
                )
            if bundle.manifest.backend_kind == "alexnet_production":
                raise PermissionError(
                    "AlexNet production requires preregistered "
                    "oracle_attempt_allocations"
                )
            allocation = _production_paired_bce_unit_alpha(
                metadata.get(
                    "attempt_alpha_allocation",
                    familywise / float(maximum_attempts),
                ),
                "paired-BCE static compatibility attempt alpha allocation",
            )
            axis = tuple(allocation for _ in range(maximum_attempts))
            source_kind = "static_compatibility_uniform_axis"
            master_hash = None
            expected_axis_hash = content_hash(
                {
                    "schema": (
                        "partition_free_v1_static_compatibility_oracle_attempt_"
                        "alpha_axis_v1"
                    ),
                    "campaign_contract_hash": contract.campaign_contract_hash,
                    "controller_run_id": controller.controller_run_id,
                    "maximum_candidate_attempts": maximum_attempts,
                    "oracle_familywise_alpha": familywise,
                    "attempt_alpha_allocation": allocation,
                    "static_compatibility_only": True,
                },
                prefix=(
                    "partition-free-v1-static-compat-oracle-attempt-alpha-axis-"
                ),
            )
            declared_axis_hash = metadata.get("candidate_attempt_alpha_axis_hash")
            if declared_axis_hash is None:
                declared_axis_hash = expected_axis_hash
            else:
                declared_axis_hash = _nonempty(
                    declared_axis_hash,
                    "candidate_attempt_alpha_axis_hash",
                )
                if declared_axis_hash != expected_axis_hash:
                    raise ValueError(
                        "static compatibility candidate_attempt_alpha_axis_hash "
                        "does not bind its allocation axis"
                    )
        else:
            if "attempt_alpha_allocation" in metadata:
                raise ValueError(
                    "paired-BCE canonical alpha path cannot also declare a static "
                    "attempt_alpha_allocation"
                )
            if "alpha_committed_before_attempt" in metadata:
                raise ValueError(
                    "paired-BCE canonical alpha path derives committed alpha from "
                    "oracle_attempt_allocations"
                )
            axis = _production_paired_bce_alpha_axis_from_metadata(raw_axis)
            source_kind = "canonical_oracle_attempt_alpha_axis"
            master_hash = _nonempty(
                metadata.get("master_decision_alpha_registration_hash"),
                "master_decision_alpha_registration_hash",
            )
            preoutcome = _nonempty(
                metadata.get(
                    "oracle_attempt_alpha_preoutcome_registration_receipt_hash"
                ),
                "oracle_attempt_alpha_preoutcome_registration_receipt_hash",
            )
            unused_alpha_nonrecycling = _metadata_strict_bool(
                metadata,
                "unused_alpha_nonrecycling",
            )
            oracle_reserve_nontransferable = _metadata_strict_bool(
                metadata,
                "oracle_reserve_nontransferable",
            )
            expected_axis_hash = (
                _production_paired_bce_canonical_oracle_attempt_alpha_axis_hash(
                    campaign_contract_hash=contract.campaign_contract_hash,
                    controller_run_hash=controller.controller_run_id,
                    preoutcome_registration_receipt_hash=preoutcome,
                    maximum_oracle_attempts=maximum_attempts,
                    oracle_familywise_alpha=familywise,
                    oracle_attempt_allocations=axis,
                    unused_alpha_nonrecycling=unused_alpha_nonrecycling,
                    oracle_reserve_nontransferable=(
                        oracle_reserve_nontransferable
                    ),
                )
            )
            declared_axis_hash = _nonempty(
                metadata.get("candidate_attempt_alpha_axis_hash"),
                "candidate_attempt_alpha_axis_hash",
            )
            if declared_axis_hash != expected_axis_hash:
                raise ValueError(
                    "candidate_attempt_alpha_axis_hash does not match the "
                    "canonical oracle-attempt alpha axis"
                )

    return _ProductionPairedBCEAttemptAlphaCoordinate(
        oracle_attempt_allocations=axis,
        candidate_attempt_alpha_axis_hash=declared_axis_hash,
        master_decision_alpha_registration_hash=master_hash,
        candidate_attempt_index=attempt_index,
        maximum_candidate_attempts=maximum_attempts,
        oracle_familywise_alpha=familywise,
        alpha_committed_before_attempt=math.fsum(axis[:attempt_index]),
        attempt_alpha_allocation=axis[attempt_index],
        source_kind=source_kind,
    )


def _production_paired_bce_order_owner_candidates(
    bundle: V1ProductionRuntimeBundle,
) -> tuple[object, ...]:
    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    owners: list[object] = [evaluator]
    for seal in (
        bundle.proposal_initial_state_factory,
        bundle.proposal_forecast_factory,
        bundle.proposal_request_plan_factory,
        bundle.proposal_acquisition_executor,
        bundle.proposal_freeze_evaluator,
        bundle.proposal_acquisition_finalizer,
        bundle.proposal_resume_validator,
    ):
        if seal is None:
            continue
        owner = getattr(seal.callable_value, "__self__", None)
        if owner is not None and all(owner is not existing for existing in owners):
            owners.append(owner)
    return tuple(owners)


def _call_oracle_attempt_order_receipts(
    method: Callable[..., Any],
    *,
    candidate_attempt_index: int,
    oracle_axes: Mapping[str, tuple[str, ...]],
) -> Any:
    try:
        return method(
            candidate_attempt_index=candidate_attempt_index,
            oracle_context_ids_by_stratum=tuple(
                (name, oracle_axes[name]) for name in PAIRED_BCE_STRATA
            ),
        )
    except TypeError as first_error:
        try:
            return method(candidate_attempt_index=candidate_attempt_index)
        except TypeError:
            try:
                return method(candidate_attempt_index)
            except TypeError as final_error:
                raise TypeError(
                    "paired-BCE attempt order API must accept "
                    "candidate_attempt_index, optionally closing over oracle folds"
                ) from final_error


def _parse_oracle_attempt_order_receipts(
    raw: Any,
    *,
    oracle_axes: Mapping[str, tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    if isinstance(raw, Mapping):
        by_name = dict(raw)
    else:
        rows = tuple(raw) if isinstance(raw, (tuple, list)) else ()
        if len(rows) != 2:
            raise TypeError(
                "paired-BCE attempt order API must return two stratum receipts"
            )
        if all(
            isinstance(item, (tuple, list))
            and len(item) == 2
            and type(item[0]) is str
            for item in rows
        ):
            by_name = {str(name): receipt for name, receipt in rows}
        else:
            by_name = dict(zip(PAIRED_BCE_STRATA, rows, strict=True))
    if set(by_name) != set(PAIRED_BCE_STRATA):
        raise ValueError("paired-BCE attempt order receipts must name target/complement")
    orders: dict[str, tuple[str, ...]] = {}
    receipt_hashes: dict[str, str] = {}
    for name in PAIRED_BCE_STRATA:
        receipt = by_name[name]
        verifier = getattr(receipt, "verify", None)
        if callable(verifier):
            result = verifier()
            if result is not None:
                raise ValueError("paired-BCE order receipt verifier must return None")
        order = tuple(
            _nonempty(value, f"paired-BCE {name} generated order context")
            for value in getattr(receipt, "generated_order", ())
        )
        if set(order) != set(oracle_axes[name]) or len(order) != len(oracle_axes[name]):
            raise ValueError(
                "paired-BCE attempt order receipt does not cover its oracle stratum"
            )
        canonical = getattr(receipt, "canonical_population", tuple(sorted(order)))
        if set(tuple(canonical)) != set(oracle_axes[name]):
            raise ValueError(
                "paired-BCE attempt order receipt canonical population is stale"
            )
        receipt_hash = _nonempty(
            getattr(receipt, "content_hash", None),
            f"paired-BCE {name} order receipt hash",
        )
        orders[name] = order
        receipt_hashes[name] = receipt_hash
    return orders, receipt_hashes


def _production_paired_bce_attempt_orders(
    bundle: V1ProductionRuntimeBundle,
    *,
    metadata: Mapping[str, Any],
    candidate_attempt_index: int,
    oracle_axes: Mapping[str, tuple[str, ...]],
) -> _ProductionPairedBCEAttemptOrders:
    def _from_method(
        *,
        owner: object,
        method: Callable[..., Any],
        sealed_hook_spec: str | None,
    ) -> _ProductionPairedBCEAttemptOrders:
        raw = _call_oracle_attempt_order_receipts(
            method,
            candidate_attempt_index=candidate_attempt_index,
            oracle_axes=oracle_axes,
        )
        orders, receipt_hashes = _parse_oracle_attempt_order_receipts(
            raw,
            oracle_axes=oracle_axes,
        )
        root_commitment = _metadata_identity(
            metadata,
            "oracle_order_entropy_root_commitment_hash",
            default_payload={
                "schema": (
                    "partition_free_v1_production_paired_bce_attempt_order_root_v1"
                ),
                "owner_class": type(owner).__qualname__,
                "owner_module": type(owner).__module__,
                "provider_hash": bundle.provider_seal.provider_hash,
                "oracle_attempt_order_receipts_hook_spec": sealed_hook_spec,
            },
            prefix="partition-free-v1-production-paired-bce-order-root-",
        )
        return _ProductionPairedBCEAttemptOrders(
            orders=orders,
            receipt_hashes=receipt_hashes,
            order_entropy_commitment_hash=content_hash(
                {
                    "schema": (
                        "partition_free_v1_production_paired_bce_attempt_order_"
                        "commitment_v1"
                    ),
                    "oracle_order_entropy_root_commitment_hash": root_commitment,
                    "candidate_attempt_index": candidate_attempt_index,
                    "receipt_hashes": tuple(
                        (name, receipt_hashes[name]) for name in PAIRED_BCE_STRATA
                    ),
                    "attempt_domain_separated": True,
                    "oracle_attempt_order_receipts_hook_spec": sealed_hook_spec,
                },
                prefix="partition-free-v1-production-paired-bce-attempt-order-",
            ),
            source_kind="attempt_specific_receipts",
        )

    if bundle.manifest.backend_kind == "alexnet_production":
        expected_hook = (
            bundle.provider_seal.evaluator_oracle_attempt_order_receipts_hook_spec
        )
        if expected_hook is None:
            raise PermissionError(
                "AlexNet production requires sealed evaluator oracle-attempt "
                "order receipts"
            )
        evaluator = _active_paired_oracle_evaluator(bundle.provider)
        sealed_hook = _sealed_evaluator_hook(
            evaluator=evaluator,
            evaluator_seal=bundle.provider_seal.evaluator_seal,
            method_name=_EVALUATOR_ORACLE_ATTEMPT_ORDER_RECEIPTS_METHOD,
            label="oracle-attempt-order-receipts",
        )
        if sealed_hook is None or sealed_hook[1] != expected_hook:
            raise PermissionError(
                "AlexNet production oracle-attempt order hook is not the "
                "sealed evaluator hook"
            )
        return _from_method(
            owner=evaluator,
            method=sealed_hook[0],
            sealed_hook_spec=expected_hook,
        )

    for owner in _production_paired_bce_order_owner_candidates(bundle):
        method = getattr(owner, _EVALUATOR_ORACLE_ATTEMPT_ORDER_RECEIPTS_METHOD, None)
        if not callable(method):
            continue
        sealed_hook_spec = (
            bundle.provider_seal.evaluator_oracle_attempt_order_receipts_hook_spec
            if owner is _active_paired_oracle_evaluator(bundle.provider)
            else None
        )
        return _from_method(
            owner=owner,
            method=method,
            sealed_hook_spec=sealed_hook_spec,
        )

    if bool(metadata.get("allow_static_sealed_order_compatibility", False)):
        if bundle.manifest.backend_kind == "alexnet_production":
            raise PermissionError(
                "AlexNet production requires attempt-specific paired-BCE oracle orders"
            )
        orders = _paired_bce_sealed_orders(metadata)
        if any(set(orders[name]) != set(oracle_axes[name]) for name in PAIRED_BCE_STRATA):
            raise ValueError("paired-BCE static orders must permute their oracle axes")
        receipt_hashes = {
            name: _metadata_identity(
                metadata,
                f"{name}_sealed_permutation_receipt_hash",
                default_payload={
                    "schema": (
                        "partition_free_v1_production_static_compat_paired_bce_"
                        "order_v1"
                    ),
                    "stratum_id": name,
                    "sealed_order": orders[name],
                    "candidate_attempt_index": candidate_attempt_index,
                    "static_compatibility_only": True,
                },
                prefix="partition-free-v1-production-paired-bce-static-order-",
            )
            for name in PAIRED_BCE_STRATA
        }
        return _ProductionPairedBCEAttemptOrders(
            orders=orders,
            receipt_hashes=receipt_hashes,
            order_entropy_commitment_hash=_metadata_identity(
                metadata,
                "oracle_order_entropy_commitment_hash",
                default_payload={
                    "schema": (
                        "partition_free_v1_production_static_compat_paired_bce_"
                        "order_commitment_v1"
                    ),
                    "sealed_orders": tuple(
                        (name, orders[name]) for name in PAIRED_BCE_STRATA
                    ),
                    "static_compatibility_only": True,
                },
                prefix="partition-free-v1-production-paired-bce-static-order-",
            ),
            source_kind="static_compatibility",
        )
    raise PermissionError(
        "paired-BCE production requires attempt-specific oracle order receipts; "
        "static manifest orders are compatibility-only"
    )


def _production_paired_bce_calibration_freeze_factory(
    bundle: V1ProductionRuntimeBundle,
) -> Callable[[V1RuntimeState, Any], Any]:
    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    freeze_hook = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
        method_name=_EVALUATOR_CANDIDATE_CALIBRATION_FREEZE_METHOD,
        label="candidate-calibration-freeze",
    )
    verify_hook = _sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
        method_name=_EVALUATOR_CANDIDATE_CALIBRATION_VERIFY_METHOD,
        label="candidate-calibration-verify",
    )
    if (
        freeze_hook is None
        or verify_hook is None
        or freeze_hook[1]
        != bundle.provider_seal.evaluator_candidate_calibration_freeze_hook_spec
        or verify_hook[1]
        != bundle.provider_seal.evaluator_candidate_calibration_verify_hook_spec
    ):
        raise PermissionError(
            "paired-BCE production requires sealed evaluator calibration hooks"
        )

    def factory(state: V1RuntimeState, _threshold: Any) -> Any:
        freeze = freeze_hook[0](state.controller)
        verifier_result = verify_hook[0](freeze)
        if verifier_result is not None:
            raise ValueError("candidate calibration verifier must return None")
        return freeze

    return factory


def _production_paired_bce_calibration_horizon_skip_factory(
    bundle: V1ProductionRuntimeBundle,
) -> Callable[..., V1PairedBCECalibrationHorizonSkipDecision]:
    """Bind a validated K-horizon refusal to the sealed alpha axis."""

    metadata = _paired_bce_metadata(bundle)
    if metadata is None:  # pragma: no cover - paired production gate.
        raise RuntimeError("paired-BCE production metadata is absent")

    def factory(
        *,
        state: V1RuntimeState,
        threshold: Any,
        refusal_receipt: Mapping[str, Any],
    ) -> V1PairedBCECalibrationHorizonSkipDecision:
        controller = state.controller
        acquisition = controller.acquisition
        if controller.phase != "queue_ready" or acquisition is None:
            raise PermissionError(
                "calibration-horizon skip factory requires a ready queue"
            )
        if controller.queue_cursor >= len(acquisition.queue):
            raise ValueError("calibration-horizon skip queue is exhausted")
        proposal = acquisition.queue[controller.queue_cursor]
        alpha = _production_paired_bce_attempt_alpha_coordinate(
            bundle,
            metadata=metadata,
            controller=controller,
        )
        coordinate_hash = paired_bce_attempt_alpha_coordinate_hash(
            candidate_attempt_alpha_axis_hash=(
                alpha.candidate_attempt_alpha_axis_hash
            ),
            candidate_attempt_index=controller.candidate_attempt_count,
            maximum_candidate_attempts=(
                controller.binding.contract.maximum_candidate_count
            ),
            attempt_alpha_allocation=alpha.attempt_alpha_allocation,
        )
        return V1PairedBCECalibrationHorizonSkipDecision(
            campaign_contract_hash=(
                controller.binding.contract.campaign_contract_hash
            ),
            controller_run_id=controller.controller_run_id,
            acquisition_receipt_hash=acquisition.receipt.receipt_hash,
            acquisition_generation_index=acquisition.generation_index,
            accepted_step_index=controller.accepted_step_count,
            queue_index=controller.queue_cursor,
            candidate_id=proposal.candidate_id,
            parent_mask_hash=controller.current_mask.mask_hash,
            candidate_mask_hash=proposal.resulting_mask.mask_hash,
            candidate_threshold_hash=threshold.threshold_hash,
            maximum_mean_damage=threshold.maximum_mean_damage,
            candidate_attempt_alpha_axis_hash=(
                alpha.candidate_attempt_alpha_axis_hash
            ),
            candidate_attempt_alpha_coordinate_hash=coordinate_hash,
            candidate_attempt_index=controller.candidate_attempt_count,
            maximum_candidate_attempts=(
                controller.binding.contract.maximum_candidate_count
            ),
            oracle_familywise_alpha=alpha.oracle_familywise_alpha,
            alpha_committed_before_attempt=alpha.alpha_committed_before_attempt,
            attempt_alpha_allocation=alpha.attempt_alpha_allocation,
            master_decision_alpha_registration_hash=(
                alpha.master_decision_alpha_registration_hash
            ),
            causal_horizon_refusal_receipt=refusal_receipt,
        )

    return factory


def _production_paired_bce_registration_factory(
    bundle: V1ProductionRuntimeBundle,
) -> Callable[[V1ControllerState, Any, Any], V1PairedBCEAcquisitionRegistration]:
    metadata = _paired_bce_metadata(bundle)
    if metadata is None:  # pragma: no cover - caller gates this path.
        raise RuntimeError("paired-BCE production metadata is absent")
    dynamic_enabled = (
        bundle.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec
        is not None
    )
    if bundle.manifest.backend_kind == "alexnet_production" and not dynamic_enabled:
        raise PermissionError(
            "AlexNet paired-BCE production requires generation-local dynamic O authority"
        )
    static_oracle_axes: dict[str, tuple[str, ...]] | None = None
    role_separation = None
    if not dynamic_enabled:
        static_oracle_axes = _named_axis_map(
            metadata["oracle_context_ids_by_stratum"],
            label="paired-BCE oracle context axes",
        )
        role_separation = register_paired_bce_oracle_role_separation(
            discovery_fold_identity_hash=_metadata_identity(
                metadata,
                "discovery_fold_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_discovery_fold_v1",
                    "context_ids": tuple(metadata["discovery_context_ids"]),
                },
                prefix="partition-free-v1-production-discovery-fold-",
            ),
            fq_certification_fold_identity_hash=_metadata_identity(
                metadata,
                "fq_certification_fold_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_fq_certification_fold_v1",
                    "context_ids": tuple(metadata["fq_certification_context_ids"]),
                },
                prefix="partition-free-v1-production-fq-certification-fold-",
            ),
            calibration_fold_identity_hash=_metadata_identity(
                metadata,
                "calibration_fold_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_calibration_fold_v1",
                    "context_ids": tuple(metadata["calibration_context_ids"]),
                },
                prefix="partition-free-v1-production-calibration-fold-",
            ),
            oracle_fold_identity_hash=_metadata_identity(
                metadata,
                "oracle_fold_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_paired_bce_oracle_fold_v1",
                    "oracle_context_ids_by_stratum": tuple(
                        (name, static_oracle_axes[name])
                        for name in PAIRED_BCE_STRATA
                    ),
                },
                prefix="partition-free-v1-production-paired-bce-oracle-fold-",
            ),
            discovery_context_ids=tuple(metadata["discovery_context_ids"]),
            fq_certification_context_ids=tuple(
                metadata["fq_certification_context_ids"]
            ),
            calibration_context_ids=tuple(metadata["calibration_context_ids"]),
            oracle_target_context_ids=static_oracle_axes["target"],
            oracle_complement_context_ids=static_oracle_axes["complement"],
            disjointness_verifier_identity_hash=_metadata_identity(
                metadata,
                "disjointness_verifier_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_role_disjointness_v1",
                    "discovery_context_ids": tuple(metadata["discovery_context_ids"]),
                    "fq_certification_context_ids": tuple(
                        metadata["fq_certification_context_ids"]
                    ),
                    "calibration_context_ids": tuple(
                        metadata["calibration_context_ids"]
                    ),
                    "oracle_context_ids_by_stratum": tuple(
                        (name, static_oracle_axes[name])
                        for name in PAIRED_BCE_STRATA
                    ),
                },
                prefix="partition-free-v1-production-role-disjointness-",
            ),
        )
    evaluator = _active_paired_oracle_evaluator(bundle.provider)

    def factory(
        controller: V1ControllerState,
        threshold: Any,
        freeze: Any,
    ) -> V1PairedBCEAcquisitionRegistration:
        support = None
        recover = getattr(evaluator, "recover_paired_bce_support_bound_certificate", None)
        if callable(recover):
            support = recover(
                freeze,
                required=bool(metadata.get("require_finite_partial_bounds", False)),
            )
        loss_kind = _nonempty(
            metadata.get(
                "paired_loss_kind",
                ANALYTIC_FIXED_AFFINE_PAIRED_BCE_KIND
                if support is not None
                else CENSUS_ONLY_PAIRED_BCE_KIND,
            ),
            "paired_loss_kind",
        )
        if support is None and loss_kind != CENSUS_ONLY_PAIRED_BCE_KIND:
            raise PermissionError(
                "finite-bound paired-BCE production requires evaluator support bounds"
            )
        bound_by_name: dict[str, tuple[str, float | None, float | None]] = {}
        if support is not None:
            bound_by_name["target"] = (
                support.target_stratum_bound.bound_certificate_hash,
                support.target_stratum_bound.paired_loss_lower_bound,
                support.target_stratum_bound.paired_loss_upper_bound,
            )
            bound_by_name["complement"] = (
                support.complement_stratum_bound.bound_certificate_hash,
                support.complement_stratum_bound.paired_loss_lower_bound,
                support.complement_stratum_bound.paired_loss_upper_bound,
            )
        contract = controller.binding.contract
        dense_referent_mask_hash = v1_dense_referent_mask_hash(contract)
        maximum_attempts = contract.maximum_candidate_count
        attempt_index = controller.candidate_attempt_count
        alpha_coordinate = _production_paired_bce_attempt_alpha_coordinate(
            bundle,
            metadata=metadata,
            controller=controller,
        )
        attempt_coordinate_hash = paired_bce_attempt_alpha_coordinate_hash(
            candidate_attempt_alpha_axis_hash=(
                alpha_coordinate.candidate_attempt_alpha_axis_hash
            ),
            candidate_attempt_index=attempt_index,
            maximum_candidate_attempts=maximum_attempts,
            attempt_alpha_allocation=alpha_coordinate.attempt_alpha_allocation,
        )
        loss_ceiling = float(threshold.maximum_mean_damage)
        loss_ceiling_hash = paired_bce_loss_ceiling_identity_hash(
            dense_reference_hash=dense_referent_mask_hash,
            candidate_calibration_freeze_hash=freeze.freeze_hash,
            paired_loss_kind=loss_kind,
            loss_ceiling=loss_ceiling,
        )
        dynamic_receipt: V1PairedBCEDynamicOracleReceipt | None = None
        if dynamic_enabled:
            frame = _production_active_dynamic_oracle_frame(bundle, required=True)
            if frame is None:  # pragma: no cover - required above.
                raise RuntimeError("dynamic O frame disappeared")
            frame_hash_before_k = frame.receipt_hash
            calibration_freeze_hash_before_k = freeze.freeze_hash
            if (
                frame.generation_index != controller.acquisition_generation_index
                or frame.candidate_attempt_index != attempt_index
                or frame.candidate_mask_hash != freeze.candidate_mask_hash
                or frame.dense_referent_mask_hash != dense_referent_mask_hash
                or frame.dense_calibration_hash
                != freeze.dense_calibration.frozen_calibration_hash
                or frame.candidate_calibration_hash
                != freeze.candidate_calibration.frozen_calibration_hash
            ):
                raise ValueError(
                    "generation-local O frame has stale generation/attempt/"
                    "candidate/affine coordinates"
                )
            k_axes = dict(frame.calibration_prefix_context_ids_by_stratum)
            k_contexts = tuple(
                context
                for name in PAIRED_BCE_STRATA
                for context in k_axes[name]
            )
            if k_contexts != tuple(freeze.calibration_context_ids):
                raise ValueError(
                    "generation-local O frame K prefix differs from the frozen "
                    "calibration contexts"
                )
            if bundle.manifest.backend_kind == "alexnet_production":
                causal_hashes = {
                    "causal_generation_frame_receipt_hash": (
                        frame.generation_frame_hash
                    ),
                    "calibration_prefix_freeze_hash": (
                        frame.calibration_prefix_freeze_hash
                    ),
                    "oracle_sampling_frame_receipt_hash": frame.receipt_hash,
                }
                if any(
                    freeze.metadata.get(name) != value
                    for name, value in causal_hashes.items()
                ):
                    raise ValueError(
                        "generation-local O frame differs from the frozen causal "
                        "K/O graph receipts"
                    )
            exact_k_hook = _expected_sealed_evaluator_hook(
                evaluator=evaluator,
                evaluator_seal=bundle.provider_seal.evaluator_seal,
                method_name="evaluate_paired_bce_calibration_prefix_rows",
                label="exact-k-paired-bce-calibration-prefix",
                expected_spec=bundle.provider_seal.evaluator_exact_k_paired_bce_hook_spec,
                require_bound_to_evaluator=True,
            )
            if exact_k_hook is not None:
                raw_k = exact_k_hook[0](
                    controller=controller,
                    threshold=threshold,
                    freeze=freeze,
                    frame=frame,
                    dense_reference_hash=dense_referent_mask_hash,
                    candidate_mask_hash=freeze.candidate_mask_hash,
                    k_context_ids_by_stratum=tuple(
                        (name, tuple(k_axes[name])) for name in PAIRED_BCE_STRATA
                    ),
                    loss_kind=loss_kind,
                    loss_ceiling=loss_ceiling,
                )
                if not isinstance(raw_k, tuple) or len(raw_k) != 2:
                    raise TypeError(
                        "exact-K paired-BCE evaluator must return (rows_by_stratum, receipt_hash)"
                    )
                raw_rows_by_stratum, k_evaluator_receipt_hash = raw_k
                k_evaluator_receipt_hash = _nonempty(
                    k_evaluator_receipt_hash,
                    "exact-K paired-BCE evaluator receipt hash",
                )
                source_rows = tuple(raw_rows_by_stratum)
            else:
                k_evaluator_receipt_hash = content_hash(
                    {
                        "schema": "partition_free_v1_production_exact_k_evaluator_receipt_v1",
                        "oracle_sampling_frame_receipt_hash": frame.receipt_hash,
                        "calibration_prefix_freeze_hash": (
                            frame.calibration_prefix_freeze_hash
                        ),
                        "evaluator_identity_hash": (
                            bundle.provider_seal.evaluator_seal.identity_hash
                        ),
                        "provider_hash": bundle.provider_seal.provider_hash,
                        "dense_reference_hash": dense_referent_mask_hash,
                        "candidate_mask_hash": freeze.candidate_mask_hash,
                        "dense_affine_calibration_hash": frame.dense_calibration_hash,
                        "candidate_affine_calibration_hash": (
                            frame.candidate_calibration_hash
                        ),
                        "calibration_prefix_stratum_counts": tuple(
                            len(k_axes[name]) for name in PAIRED_BCE_STRATA
                        ),
                    },
                    prefix="partition-free-v1-production-exact-k-evaluator-",
                )
                # Preserve per-stratum validation before opening the next
                # legacy stratum's rows.
                source_rows = (
                    (
                        name,
                        evaluator(
                            dense_referent_mask_hash,
                            freeze.candidate_mask_hash,
                            tuple(k_axes[name]),
                        ),
                    )
                    for name in PAIRED_BCE_STRATA
                )
            k_rows: list[tuple[str, tuple[float, ...]]] = []
            for name, raw in source_rows:
                if name not in PAIRED_BCE_STRATA:
                    raise ValueError("exact-K evaluator returned an unknown paired-BCE stratum")
                contexts = tuple(k_axes[name])
                values = np.asarray(raw, dtype=np.float64).reshape(-1)
                if values.size != len(contexts) or not bool(np.isfinite(values).all()):
                    raise ValueError("exact-K evaluator returned invalid paired-loss rows")
                k_rows.append((name, tuple(float(value) for value in values)))
            if tuple(name for name, _rows in k_rows) != PAIRED_BCE_STRATA:
                raise ValueError("exact-K evaluator returned rows in another stratum order")
            live_frame_after_k = _production_active_dynamic_oracle_frame(bundle, required=True)
            if live_frame_after_k is None:  # pragma: no cover - required above.
                raise RuntimeError("dynamic O frame disappeared")
            live_frame_after_k.assert_integrity(frame_hash_before_k)
            if freeze.freeze_hash != calibration_freeze_hash_before_k:
                raise ValueError("candidate calibration freeze changed during exact-K evaluation")
            dynamic_receipt = register_paired_bce_dynamic_oracle(
                oracle_sampling_frame=frame,
                candidate_calibration_freeze_hash=freeze.freeze_hash,
                loss_ceiling_identity_hash=loss_ceiling_hash,
                candidate_attempt_alpha_coordinate_hash=attempt_coordinate_hash,
                calibration_paired_loss_rows_by_stratum=k_rows,
                calibration_paired_loss_evaluator_receipt_hash=(
                    k_evaluator_receipt_hash
                ),
            )
            attempt_orders = _dynamic_attempt_orders(frame, dynamic_receipt)
            oracle_axes = {
                name: tuple(attempt_orders.orders[name])
                for name in PAIRED_BCE_STRATA
            }
        else:
            assert static_oracle_axes is not None
            oracle_axes = static_oracle_axes
            attempt_orders = _production_paired_bce_attempt_orders(
                bundle,
                metadata=metadata,
                candidate_attempt_index=attempt_index,
                oracle_axes=oracle_axes,
            )
        strata = []
        for index, name in enumerate(PAIRED_BCE_STRATA):
            authority_hash = (
                dynamic_receipt.receipt_hash
                if dynamic_receipt is not None
                else role_separation.receipt_hash  # type: ignore[union-attr]
            )
            fallback_bound = content_hash(
                {
                    "schema": "partition_free_v1_production_no_finite_paired_bce_bound_v1",
                    "stratum_id": name,
                    "loss_kind": loss_kind,
                    "oracle_authority_receipt_hash": authority_hash,
                },
                prefix="partition-free-v1-production-paired-bce-no-bound-",
            )
            bound_hash, lower, upper = bound_by_name.get(
                name,
                (fallback_bound, None, None),
            )
            population_hash = (
                dynamic_receipt.oracle_population_identity_hashes[index]
                if dynamic_receipt is not None
                else (
                    role_separation.oracle_target_population_identity_hash
                    if name == "target"
                    else role_separation.oracle_complement_population_identity_hash
                )  # type: ignore[union-attr]
            )
            strata.append(
                V1PairedBCEStratumRegistration(
                    stratum_id=name,
                    stratum_weight=0.5,
                    population_size=len(oracle_axes[name]),
                    population_identity_hash=population_hash,
                    sealed_order_hash=paired_bce_sealed_order_hash(
                        stratum_id=name,
                        context_ids_in_sealed_order=attempt_orders.orders[name],
                    ),
                    sealed_permutation_receipt_hash=(
                        attempt_orders.receipt_hashes[name]
                    ),
                    bound_certificate_hash=bound_hash,
                    paired_loss_lower_bound=lower,
                    paired_loss_upper_bound=upper,
                )
            )
        return build_paired_bce_acquisition_registration(
            initial_look_count=_metadata_positive_int(
                metadata,
                "initial_look_count",
                1,
            ),
            look_growth_factor=_metadata_positive_int(
                metadata,
                "look_growth_factor",
                2,
            ),
            registration_id=content_hash(
                {
                    "schema": "partition_free_v1_production_paired_bce_registration_id_v1",
                    "controller_state_hash": controller.state_hash,
                    "candidate_attempt_index": attempt_index,
                    "candidate_calibration_freeze_hash": freeze.freeze_hash,
                    "master_decision_alpha_registration_hash": (
                        alpha_coordinate.master_decision_alpha_registration_hash
                    ),
                },
                prefix="partition-free-v1-production-paired-bce-registration-",
            ),
            campaign_contract_hash=contract.campaign_contract_hash,
            candidate_mask_hash=freeze.candidate_mask_hash,
            dense_reference_hash=dense_referent_mask_hash,
            candidate_calibration_freeze_hash=freeze.freeze_hash,
            paired_loss_kind=loss_kind,
            loss_ceiling=loss_ceiling,
            loss_ceiling_identity_hash=loss_ceiling_hash,
            source_population_identity_hash=(
                dynamic_receipt.source_population_identity_hash
                if dynamic_receipt is not None
                else _metadata_identity(
                    metadata,
                    "source_population_identity_hash",
                    default_payload={
                        "schema": "partition_free_v1_production_paired_bce_source_population_v1",
                        "oracle_context_ids_by_stratum": tuple(
                            (name, oracle_axes[name]) for name in PAIRED_BCE_STRATA
                        ),
                    },
                    prefix="partition-free-v1-production-paired-bce-source-population-",
                )
            ),
            oracle_fold_identity_hash=(
                dynamic_receipt.oracle_sampling_frame_receipt_hash
                if dynamic_receipt is not None
                else role_separation.oracle_fold_identity_hash  # type: ignore[union-attr]
            ),
            oracle_role_separation_receipt=role_separation,
            strata=tuple(strata),
            oracle_order_entropy_commitment_hash=(
                attempt_orders.order_entropy_commitment_hash
            ),
            sampling_law_identity_hash=_metadata_identity(
                metadata,
                "sampling_law_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_paired_bce_sampling_law_v1",
                    "campaign_sampling_law_hash": contract.sampling_law_hash,
                },
                prefix="partition-free-v1-production-paired-bce-sampling-law-",
            ),
            sealed_order_slice_verifier_identity_hash=_metadata_identity(
                metadata,
                "sealed_order_slice_verifier_identity_hash",
                default_payload={
                    "schema": "partition_free_v1_production_paired_bce_slice_verifier_v1",
                    "provider_hash": bundle.provider_seal.provider_hash,
                },
                prefix="partition-free-v1-production-paired-bce-slice-verifier-",
            ),
            candidate_attempt_alpha_axis_hash=(
                alpha_coordinate.candidate_attempt_alpha_axis_hash
            ),
            candidate_attempt_alpha_coordinate_hash=attempt_coordinate_hash,
            candidate_attempt_index=attempt_index,
            maximum_candidate_attempts=maximum_attempts,
            oracle_familywise_alpha=alpha_coordinate.oracle_familywise_alpha,
            alpha_committed_before_attempt=(
                alpha_coordinate.alpha_committed_before_attempt
            ),
            attempt_alpha_allocation=alpha_coordinate.attempt_alpha_allocation,
            dynamic_oracle_receipt=dynamic_receipt,
        )

    return factory


def _production_paired_bce_chunk_factory(
    bundle: V1ProductionRuntimeBundle,
    *,
    order_epoch_cache: _ProductionPairedBCEOrderEpochCache | None = None,
) -> Callable[
    [V1ControllerState, Any, V1PairedBCEAcquisitionRegistration, V1PairedBCEAcquisitionState],
    V1RuntimePendingPairedBCEChunk,
]:
    metadata = _paired_bce_metadata(bundle)
    if metadata is None:  # pragma: no cover - caller gates this path.
        raise RuntimeError("paired-BCE production metadata is absent")
    static_oracle_axes = (
        None
        if bundle.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec
        is not None
        else _named_axis_map(
            metadata["oracle_context_ids_by_stratum"],
            label="paired-BCE oracle context axes",
        )
    )
    maximum_rows = _metadata_positive_int(
        metadata,
        "maximum_chunk_rows",
        max(1, bundle.runtime_config.physical_batch_size),
    )
    row_provider_identity_hash = _metadata_identity(
        metadata,
        "row_provider_identity_hash",
        default_payload={
            "schema": "partition_free_v1_production_paired_bce_row_provider_v1",
            "provider_hash": bundle.provider_seal.provider_hash,
        },
        prefix="partition-free-v1-production-paired-bce-row-provider-",
    )
    loss_evaluator_identity_hash = _metadata_identity(
        metadata,
        "paired_loss_evaluator_identity_hash",
        default_payload={
            "schema": "partition_free_v1_production_paired_bce_loss_evaluator_v1",
            "evaluator_hash": bundle.provider_seal.evaluator_seal.identity_hash,
        },
        prefix="partition-free-v1-production-paired-bce-loss-evaluator-",
    )

    def factory(
        *,
        controller: V1ControllerState,
        active_candidate: Any,
        registration: V1PairedBCEAcquisitionRegistration,
        paired_bce_state: V1PairedBCEAcquisitionState,
    ) -> V1RuntimePendingPairedBCEChunk:
        stratum_id, start, stop = next_paired_bce_chunk_coordinates(
            registration=registration,
            state=paired_bce_state,
            maximum_rows=maximum_rows,
        )
        dynamic = registration.dynamic_oracle_receipt
        order_epoch: _ProductionPairedBCEOrderEpoch | None = None
        verified_order: V1PairedBCEVerifiedSealedOrder | None = None
        if dynamic is not None:
            frame = _production_active_dynamic_oracle_frame(bundle, required=True)
            if frame is None:  # pragma: no cover - required above.
                raise RuntimeError("dynamic O frame disappeared")
            if order_epoch_cache is None:
                attempt_orders = _dynamic_attempt_orders(frame, dynamic)
            else:
                order_epoch = order_epoch_cache.bind(
                    bundle=bundle, registration=registration, frame=frame,
                )
                verified_order = order_epoch.verified_orders[
                    PAIRED_BCE_STRATA.index(stratum_id)
                ]
        else:
            if static_oracle_axes is None:
                raise PermissionError(
                    "paired-BCE registration lacks the active dynamic O receipt"
                )
            attempt_orders = _production_paired_bce_attempt_orders(
                bundle,
                metadata=metadata,
                candidate_attempt_index=registration.candidate_attempt_index,
                oracle_axes=static_oracle_axes,
            )
        if verified_order is None:
            sealed_order = attempt_orders.orders[stratum_id]
            stratum = registration.stratum(stratum_id)
            if (
                paired_bce_sealed_order_hash(
                    stratum_id=stratum_id,
                    context_ids_in_sealed_order=sealed_order,
                )
                != stratum.sealed_order_hash
                or attempt_orders.receipt_hashes[stratum_id]
                != stratum.sealed_permutation_receipt_hash
            ):
                raise ValueError("paired-BCE attempt order replay differs from registration")
        else:
            order_epoch_cache.assert_live(order_epoch, frame=frame)
            sealed_order = verified_order.sealed_context_order
        request = make_next_paired_bce_chunk_request(
            registration=registration,
            state=paired_bce_state,
            maximum_rows=maximum_rows,
            sealed_context_order=sealed_order if verified_order is None else None,
            verified_sealed_order=verified_order,
            context_ids_in_registered_slice=sealed_order[start:stop],
            row_provider_identity_hash=row_provider_identity_hash,
            paired_loss_evaluator_identity_hash=loss_evaluator_identity_hash,
        )
        pending = V1RuntimePendingPairedBCEChunk(
            controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
            active_candidate_hash=active_candidate.active_hash,
            registration_hash=registration.registration_hash,
            prior_paired_bce_state_hash=paired_bce_state.state_hash,
            request=request,
            context_ids_in_registered_slice=sealed_order[start:stop],
            dynamic_oracle_receipt_hash=(
                registration.dynamic_oracle_receipt_hash
            ),
        )

        if order_epoch is not None:
            # Snapshot/active-hash accessors above belong to the controller
            # boundary. Reopen the actual frame before publishing a request.
            live_frame = _production_active_dynamic_oracle_frame(bundle, required=True)
            order_epoch_cache.assert_live(order_epoch, frame=live_frame)
        return pending

    return factory


def _paired_bce_receipt_payload(receipt: V1PairedBCEChunkReceipt) -> dict[str, Any]:
    return {
        "request_hash": receipt.request_hash,
        "registration_hash": receipt.registration_hash,
        "prior_state_hash": receipt.prior_state_hash,
        "candidate_attempt_alpha_coordinate_hash": (
            receipt.candidate_attempt_alpha_coordinate_hash
        ),
        "candidate_attempt_index": receipt.candidate_attempt_index,
        "request_index": receipt.request_index,
        "stratum_id": receipt.stratum_id,
        "prefix_start": receipt.prefix_start,
        "prefix_stop": receipt.prefix_stop,
        "context_slice_hash": receipt.context_slice_hash,
        "sealed_slice_registration_hash": receipt.sealed_slice_registration_hash,
        "paired_row_payload_hash": receipt.paired_row_payload_hash,
        "exact_scaled_loss_sum": receipt.exact_scaled_loss_sum,
        "observed_minimum": receipt.observed_minimum,
        "observed_maximum": receipt.observed_maximum,
        "paired_loss_evaluator_receipt_hash": (
            receipt.paired_loss_evaluator_receipt_hash
        ),
        "dynamic_oracle_receipt_hash": receipt.dynamic_oracle_receipt_hash,
        "all_rows_finite": receipt.all_rows_finite,
        "finite_bound_checked_when_available": (
            receipt.finite_bound_checked_when_available
        ),
        "full_chunk_completed": receipt.full_chunk_completed,
        "schema": receipt.schema,
    }


def _paired_bce_chunk_artifact_body(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    pending: V1RuntimePendingPairedBCEChunk,
    receipt: V1PairedBCEChunkReceipt,
    contexts: tuple[str, ...],
    rows: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "schema": _PAIRED_BCE_ROW_ARTIFACT_SCHEMA,
        "registration_hash": registration.registration_hash,
        "state_hash_before_chunk": state.state_hash,
        "request_hash": pending.request.request_hash,
        "pending_hash": pending.pending_hash,
        "receipt_hash": receipt.receipt_hash,
        "candidate_attempt_index": registration.candidate_attempt_index,
        "dynamic_oracle_receipt_hash": registration.dynamic_oracle_receipt_hash,
        "request_index": pending.request.request_index,
        "stratum_id": pending.request.stratum_id,
        "context_ids_in_registered_slice": contexts,
        "paired_loss_rows_hex": tuple(float(value).hex() for value in rows),
        "receipt": _paired_bce_receipt_payload(receipt),
        "row_payload_externalized": True,
        "checkpoint_rows_inflated": False,
        "append_only_content_addressed": True,
    }


def _paired_bce_chunk_artifact_hash(body: Mapping[str, Any]) -> str:
    return content_hash(
        body,
        prefix="partition-free-v1-production-paired-bce-row-artifact-",
    )


def _read_paired_bce_chunk_artifact(path: Path) -> dict[str, Any]:
    _require_nonsymlink_regular_file(path, label="paired-BCE row artifact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("paired-BCE row artifact is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("paired-BCE row artifact must be a mapping")
    artifact_hash = _nonempty(
        payload.get("artifact_hash"),
        "paired-BCE row artifact hash",
    )
    body = {key: value for key, value in payload.items() if key != "artifact_hash"}
    if _paired_bce_chunk_artifact_hash(body) != artifact_hash:
        raise ValueError("paired-BCE row artifact failed its content hash")
    if payload.get("schema") != _PAIRED_BCE_ROW_ARTIFACT_SCHEMA:
        raise ValueError("paired-BCE row artifact has an unsupported schema")
    return dict(payload)


def _validate_paired_bce_chunk_artifact_location(
    path: Path,
    *,
    registration_directory_name: str,
) -> dict[str, Any]:
    payload = _read_paired_bce_chunk_artifact(path)
    expected_fields = {
        "artifact_hash",
        "schema",
        "registration_hash",
        "state_hash_before_chunk",
        "request_hash",
        "pending_hash",
        "receipt_hash",
        "candidate_attempt_index",
        "dynamic_oracle_receipt_hash",
        "request_index",
        "stratum_id",
        "context_ids_in_registered_slice",
        "paired_loss_rows_hex",
        "receipt",
        "row_payload_externalized",
        "checkpoint_rows_inflated",
        "append_only_content_addressed",
    }
    if set(payload) != expected_fields:
        raise ValueError("paired-BCE row artifact has unexpected coordinates")
    registration_hash = _nonempty(
        payload["registration_hash"],
        "paired-BCE row artifact registration_hash",
    )
    if registration_hash != registration_directory_name:
        raise ValueError("paired-BCE row artifact is filed under another registration")
    request_index = payload["request_index"]
    if (
        isinstance(request_index, bool)
        or not isinstance(request_index, int)
        or request_index < 0
    ):
        raise ValueError("paired-BCE row artifact request_index must be nonnegative")
    receipt_hash = _nonempty(
        payload["receipt_hash"],
        "paired-BCE row artifact receipt_hash",
    )
    artifact_hash = _nonempty(
        payload["artifact_hash"],
        "paired-BCE row artifact artifact_hash",
    )
    expected_name = f"{request_index:08d}_{receipt_hash}_{artifact_hash}.json"
    if path.name != expected_name:
        raise ValueError("paired-BCE row artifact has a noncanonical filename")
    if (
        payload["row_payload_externalized"] is not True
        or payload["checkpoint_rows_inflated"] is not False
        or payload["append_only_content_addressed"] is not True
    ):
        raise ValueError("paired-BCE row artifact changed its storage semantics")
    raw_receipt = payload["receipt"]
    if not isinstance(raw_receipt, Mapping):
        raise TypeError("paired-BCE row artifact receipt must be a mapping")
    receipt = V1PairedBCEChunkReceipt(**dict(raw_receipt))
    receipt.assert_integrity(receipt_hash)
    if (
        receipt.registration_hash != registration_hash
        or receipt.request_index != request_index
        or receipt.dynamic_oracle_receipt_hash
        != payload["dynamic_oracle_receipt_hash"]
    ):
        raise ValueError("paired-BCE row artifact disagrees with its receipt")
    raw_contexts = payload["context_ids_in_registered_slice"]
    raw_rows = payload["paired_loss_rows_hex"]
    if (
        isinstance(raw_contexts, (str, bytes))
        or not isinstance(raw_contexts, (tuple, list))
        or isinstance(raw_rows, (str, bytes))
        or not isinstance(raw_rows, (tuple, list))
    ):
        raise TypeError("paired-BCE row artifact rows must be sequences")
    contexts = tuple(
        _nonempty(value, "paired-BCE row artifact context") for value in raw_contexts
    )
    try:
        rows = tuple(float.fromhex(value) for value in raw_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired-BCE row artifact has an invalid encoded row") from exc
    if (
        len(contexts) != len(rows)
        or len(rows) != receipt.row_count
        or not all(math.isfinite(value) for value in rows)
    ):
        raise ValueError("paired-BCE row artifact row payload disagrees with receipt")
    return payload


def _paired_bce_row_store_inventory(root: Path) -> dict[Path, str]:
    """Validate and inventory one append-only external row store.

    Relative paths are authority-bearing: registration directory, append index,
    receipt hash, and artifact hash must all agree.  Unexpected nesting, links,
    temporary files, duplicate append indices, or malformed crash residue refuse.
    """

    if not _path_lexists(root):
        return {}
    _require_nonsymlink_directory(root, label="paired-BCE external row store")
    inventory: dict[Path, str] = {}
    for directory in sorted(root.iterdir(), key=lambda item: item.name):
        _require_nonsymlink_directory(
            directory,
            label="paired-BCE registration directory",
        )
        registration_hash = _nonempty(
            directory.name,
            "paired-BCE registration directory name",
        )
        request_indices: set[int] = set()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if (
                path.name.startswith(".")
                and ".resume-" in path.name
                and path.name.endswith(".tmp")
            ):
                _require_nonsymlink_regular_file(
                    path,
                    label="paired-BCE resume import temporary",
                )
                continue
            payload = _validate_paired_bce_chunk_artifact_location(
                path,
                registration_directory_name=registration_hash,
            )
            request_index = int(payload["request_index"])
            if request_index in request_indices:
                raise ValueError(
                    "paired-BCE external row store repeats an append index"
                )
            request_indices.add(request_index)
            relative = Path(registration_hash) / path.name
            inventory[relative] = _sha256_file(path)
    return inventory


def _materialize_paired_bce_row_store(
    *,
    source_root: Path,
    destination_root: Path,
) -> None:
    """Make the destination self-contained for every source-owned row.

    A partially imported destination is a valid retry prefix.  Same-name
    content drift and any noncanonical artifact refuse; valid destination rows
    absent from the source remain inert unless the active compact-prefix replay
    validator names them.  The importer never deletes or overwrites evidence.
    """

    source_inventory = _paired_bce_row_store_inventory(source_root)
    if source_root.resolve(strict=False) == destination_root.resolve(strict=False):
        return
    destination_inventory = _paired_bce_row_store_inventory(destination_root)
    for relative, destination_sha in destination_inventory.items():
        if (
            relative in source_inventory
            and source_inventory[relative] != destination_sha
        ):
            raise ValueError(
                "destination paired-BCE row artifact conflicts with resume source"
            )
    for relative in sorted(source_inventory, key=lambda item: item.as_posix()):
        if relative in destination_inventory:
            continue
        _atomic_materialize_exact_file(
            source_root / relative,
            destination_root / relative,
            label="paired-BCE row artifact",
        )
    closed_inventory = _paired_bce_row_store_inventory(destination_root)
    for relative, source_sha in source_inventory.items():
        if closed_inventory.get(relative) != source_sha:
            raise ValueError("paired-BCE row-store import did not close exactly")


def _store_paired_bce_chunk_artifact(
    root: Path,
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    pending: V1RuntimePendingPairedBCEChunk,
    receipt: V1PairedBCEChunkReceipt,
    contexts: tuple[str, ...],
    rows: tuple[float, ...],
) -> Path:
    body = _paired_bce_chunk_artifact_body(
        registration=registration,
        state=state,
        pending=pending,
        receipt=receipt,
        contexts=contexts,
        rows=rows,
    )
    artifact_hash = _paired_bce_chunk_artifact_hash(body)
    payload = {"artifact_hash": artifact_hash, **body}
    directory = root / registration.registration_hash
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{pending.request.request_index:08d}_"
    existing = tuple(sorted(directory.glob(prefix + "*.json")))
    target = directory / (
        f"{pending.request.request_index:08d}_"
        f"{receipt.receipt_hash}_{artifact_hash}.json"
    )
    for path in existing:
        observed = _read_paired_bce_chunk_artifact(path)
        if path == target and observed.get("artifact_hash") == artifact_hash:
            return path
        raise ValueError(
            "paired-BCE external row store already contains another artifact "
            "for this append index"
        )
    _atomic_write_json(target, payload)
    return target


def _production_paired_bce_betting_replay_factory(
    bundle: V1ProductionRuntimeBundle,
    row_store_root: Path,
) -> Callable[
    [V1PairedBCEAcquisitionRegistration, V1PairedBCEAcquisitionState],
    V1PairedBCEBettingReplay | None,
]:
    metadata = _paired_bce_metadata(bundle)
    if metadata is None:  # pragma: no cover - caller gates this path.
        raise RuntimeError("paired-BCE production metadata is absent")
    static_oracle_axes = (
        None
        if bundle.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec
        is not None
        else _named_axis_map(
            metadata["oracle_context_ids_by_stratum"],
            label="paired-BCE oracle context axes",
        )
    )

    def factory(
        *,
        registration: V1PairedBCEAcquisitionRegistration,
        paired_bce_state: V1PairedBCEAcquisitionState,
    ) -> V1PairedBCEBettingReplay | None:
        if paired_bce_state.chunk_count == 0:
            return None
        dynamic = registration.dynamic_oracle_receipt
        if dynamic is not None:
            frame = _production_active_dynamic_oracle_frame(bundle, required=True)
            if frame is None:  # pragma: no cover - required above.
                raise RuntimeError("dynamic O frame disappeared")
            attempt_orders = _dynamic_attempt_orders(frame, dynamic)
        else:
            if static_oracle_axes is None:
                raise PermissionError(
                    "paired-BCE replay lacks its registered dynamic O frame"
                )
            attempt_orders = _production_paired_bce_attempt_orders(
                bundle,
                metadata=metadata,
                candidate_attempt_index=registration.candidate_attempt_index,
                oracle_axes=static_oracle_axes,
            )
        for name in PAIRED_BCE_STRATA:
            stratum = registration.stratum(name)
            if (
                paired_bce_sealed_order_hash(
                    stratum_id=name,
                    context_ids_in_sealed_order=attempt_orders.orders[name],
                )
                != stratum.sealed_order_hash
                or attempt_orders.receipt_hashes[name]
                != stratum.sealed_permutation_receipt_hash
            ):
                raise ValueError(
                    "paired-BCE replay order receipts differ from registration"
                )
        directory = row_store_root / registration.registration_hash
        if not directory.is_dir():
            raise ValueError("paired-BCE external row store is missing its registration")
        artifacts: dict[int, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            payload = _read_paired_bce_chunk_artifact(path)
            if payload["registration_hash"] != registration.registration_hash:
                raise ValueError(
                    "paired-BCE row artifact is filed under another registration"
                )
            if payload.get("dynamic_oracle_receipt_hash") != (
                registration.dynamic_oracle_receipt_hash
            ):
                raise ValueError("paired-BCE row artifact names a stale dynamic O frame")
            index = int(payload["request_index"])
            if index >= paired_bce_state.chunk_count:
                continue
            if index in artifacts:
                raise ValueError(
                    "paired-BCE row artifact store repeats an append index"
                )
            artifacts[index] = payload
        expected_indices = tuple(range(paired_bce_state.chunk_count))
        if tuple(sorted(artifacts)) != expected_indices:
            raise ValueError(
                "paired-BCE row artifact store is missing a compact receipt"
            )
        receipts: list[V1PairedBCEChunkReceipt] = []
        contexts_by_stratum: dict[str, list[str]] = {
            name: [] for name in PAIRED_BCE_STRATA
        }
        rows_by_stratum: dict[str, list[float]] = {
            name: [] for name in PAIRED_BCE_STRATA
        }
        for index in expected_indices:
            payload = artifacts[index]
            receipt = V1PairedBCEChunkReceipt(**dict(payload["receipt"]))
            receipt.assert_integrity(_nonempty(payload["receipt_hash"], "receipt_hash"))
            contexts = tuple(
                _nonempty(value, "paired-BCE replay context")
                for value in payload["context_ids_in_registered_slice"]
            )
            rows = tuple(
                float.fromhex(value) for value in payload["paired_loss_rows_hex"]
            )
            if len(contexts) != len(rows) or len(contexts) != receipt.row_count:
                raise ValueError(
                    "paired-BCE row artifact contexts/rows differ from receipt"
                )
            receipts.append(receipt)
            contexts_by_stratum[receipt.stratum_id].extend(contexts)
            rows_by_stratum[receipt.stratum_id].extend(rows)
        replay = V1PairedBCEBettingReplay(
            sealed_context_orders=tuple(
                attempt_orders.orders[name] for name in PAIRED_BCE_STRATA
            ),
            observed_context_ids_by_stratum=tuple(
                tuple(contexts_by_stratum[name]) for name in PAIRED_BCE_STRATA
            ),
            observed_paired_loss_rows_by_stratum=tuple(
                tuple(rows_by_stratum[name]) for name in PAIRED_BCE_STRATA
            ),
            chunk_receipts=tuple(receipts),
            dynamic_oracle_receipt_hash=registration.dynamic_oracle_receipt_hash,
        )
        replay.assert_matches(
            registration=registration,
            state=paired_bce_state,
        )
        return replay

    return factory


def _validate_paired_bce_external_replay_if_needed(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    row_store_root: Path,
) -> None:
    active = state.controller.active_candidate
    if (
        active is None
        or active.paired_bce_registration is None
        or active.paired_bce_state is None
        or active.paired_bce_state.chunk_count == 0
    ):
        return
    factory = _production_paired_bce_betting_replay_factory(bundle, row_store_root)
    replay = factory(
        registration=active.paired_bce_registration,
        paired_bce_state=active.paired_bce_state,
    )
    if replay is None:
        raise ValueError("paired-BCE external replay unexpectedly missing")


@contextlib.contextmanager
def _o_stage_timer(owner: object, key: str, *, generation: int | None, attempt: int | None):
    """Telemetry-only wall-clock accumulation into ``owner.timing`` (if any)."""

    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        timing = getattr(owner, "timing", None)
        if timing is None:
            # Telemetry map is created on first use so every backend owner
            # carries the same stage keys the heartbeats snapshot.
            try:
                timing = {}
                setattr(owner, "timing", timing)
            except (AttributeError, TypeError):
                timing = None
        if isinstance(timing, dict):
            timing[f"{key}_seconds"] = timing.get(f"{key}_seconds", 0.0) + elapsed
            timing[f"{key}_count"] = timing.get(f"{key}_count", 0.0) + 1.0
            if generation is not None:
                suffix = f"generation_{int(generation)}"
                if attempt is not None:
                    suffix += f"_attempt_{int(attempt)}"
                timing[f"{key}_{suffix}_seconds"] = (
                    timing.get(f"{key}_{suffix}_seconds", 0.0) + elapsed
                )
                timing[f"{key}_{suffix}_count"] = (
                    timing.get(f"{key}_{suffix}_count", 0.0) + 1.0
                )


def _issue_attempt_structural_disclosure(
    bundle: V1ProductionRuntimeBundle,
    seal: V1ProductionCallableSeal,
    **kwargs: Any,
) -> None:
    """Invoke the attempt structural-disclosure hook; never raise into the oracle path.

    The hook owns its own durability and typed refusals.  This last-resort guard
    only counts a hook failure in the provider's timing telemetry and records
    the message on the provider, so a broken disclosure can never abort or alter
    a paired-oracle decision, and never disappears silently either.
    """

    with _o_stage_timer(
        bundle.provider,
        "stage_O_structural_disclosure",
        generation=kwargs.get("generation_index"),
        attempt=kwargs.get("candidate_attempt_index"),
    ):
        try:
            seal.callable_value(**kwargs)
        except Exception as error:  # noqa: BLE001 - disclosure must not abort the oracle
            timing = getattr(bundle.provider, "timing", None)
            if isinstance(timing, dict):
                timing["stage_O_structural_disclosure_failures"] = (
                    timing.get("stage_O_structural_disclosure_failures", 0.0) + 1.0
                )
            try:
                setattr(
                    bundle.provider,
                    "structural_disclosure_last_error",
                    f"{type(error).__name__}: {error}"[:2000],
                )
            except (AttributeError, TypeError):
                pass



def _legacy_paired_bce_evaluator_receipt_hash(
    *,
    bundle: V1ProductionRuntimeBundle,
    parent_mask_hash: str,
    dense_reference_hash: str,
    candidate_mask_hash: str,
    pending: V1RuntimePendingPairedBCEChunk,
    context_count: int,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_production_paired_bce_evaluator_receipt_v1",
            "evaluator_identity_hash": (
                bundle.provider_seal.evaluator_seal.identity_hash
            ),
            "provider_hash": bundle.provider_seal.provider_hash,
            "proposal_parent_mask_hash": parent_mask_hash,
            "dense_reference_hash": dense_reference_hash,
            "candidate_mask_hash": candidate_mask_hash,
            "request_hash": pending.request.request_hash,
            "context_slice_hash": pending.request.context_slice_hash,
            "context_count": context_count,
            "row_payload_externalized": True,
        },
        prefix="partition-free-v1-production-paired-bce-evaluator-receipt-",
    )


def _evaluate_paired_bce_rows_for_pending(
    *,
    bundle: V1ProductionRuntimeBundle,
    evaluator: Callable[..., Any],
    registration: V1PairedBCEAcquisitionRegistration,
    paired_state: V1PairedBCEAcquisitionState,
    pending: V1RuntimePendingPairedBCEChunk,
    dense_reference_hash: str,
    candidate_mask_hash: str,
    parent_mask_hash: str,
) -> tuple[tuple[float, ...], str]:
    """Evaluate one prevalidated paired-BCE O request.

    Request-aware evaluators can bind rows to the old request. Legacy
    evaluators keep the exact existing receipt payload. This helper changes
    row transport only; old reduction, commit, bounds, alpha, and K/O
    accounting remain unchanged.
    """

    contexts = tuple(pending.context_ids_in_registered_slice)
    sealed_request_hook = _expected_sealed_evaluator_hook(
        evaluator=evaluator,
        evaluator_seal=bundle.provider_seal.evaluator_seal,
        method_name="evaluate_paired_bce_chunk_request",
        label="request-aware-paired-bce-chunk",
        expected_spec=bundle.provider_seal.evaluator_request_aware_paired_bce_hook_spec,
        require_bound_to_evaluator=True,
    )
    if sealed_request_hook is not None:
        hook = sealed_request_hook[0]
        result = hook(
            registration=registration,
            state=paired_state,
            pending=pending,
            dense_reference_hash=dense_reference_hash,
            candidate_mask_hash=candidate_mask_hash,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "request-aware paired-BCE evaluator must return (rows, receipt_hash)"
            )
        raw_rows, evaluator_receipt_hash = result
        values = np.asarray(raw_rows, dtype=np.float64).reshape(-1)
        if values.size != len(contexts) or not bool(np.isfinite(values).all()):
            raise ValueError(
                "paired-BCE evaluator returned nonfinite rows or the wrong row count"
            )
        return tuple(float(value) for value in values), _nonempty(
            evaluator_receipt_hash,
            "paired-BCE request-aware evaluator receipt hash",
        )

    raw_rows = evaluator(dense_reference_hash, candidate_mask_hash, contexts)
    values = np.asarray(raw_rows, dtype=np.float64).reshape(-1)
    if values.size != len(contexts) or not bool(np.isfinite(values).all()):
        raise ValueError(
            "paired-BCE evaluator returned nonfinite rows or the wrong row count"
        )
    rows = tuple(float(value) for value in values)
    evaluator_receipt_hash = _legacy_paired_bce_evaluator_receipt_hash(
        bundle=bundle,
        parent_mask_hash=parent_mask_hash,
        dense_reference_hash=dense_reference_hash,
        candidate_mask_hash=candidate_mask_hash,
        pending=pending,
        context_count=len(contexts),
    )
    return rows, evaluator_receipt_hash


def _validate_bound_dynamic_frame_live(
    *,
    bundle: V1ProductionRuntimeBundle,
    dynamic: V1PairedBCEDynamicOracleReceipt | None,
    bound_dynamic_frame_hash: str | None,
) -> object | None:
    if dynamic is None:
        return
    live_frame = _production_active_dynamic_oracle_frame(bundle, required=True)
    if live_frame is None:  # pragma: no cover - required above.
        raise RuntimeError("dynamic O frame disappeared")
    live_frame.assert_integrity(dynamic.oracle_sampling_frame_receipt_hash)
    if live_frame.receipt_hash != bound_dynamic_frame_hash:
        raise ValueError("paired-BCE dynamic O frame changed after executor binding")
    return live_frame


@dataclass(frozen=True, slots=True)
class _ProductionPairedBCEOrderEpoch:
    """Process-local verification custody, never a receipt or checkpoint field."""

    bundle: V1ProductionRuntimeBundle = field(repr=False)
    provider: object = field(repr=False)
    provider_seal: object = field(repr=False)
    evaluator: object = field(repr=False)
    registration: V1PairedBCEAcquisitionRegistration = field(repr=False)
    registration_hash: str
    dynamic: V1PairedBCEDynamicOracleReceipt = field(repr=False)
    frame: object = field(repr=False)
    frame_hash: str
    frame_fields: tuple[tuple[str, object], ...] = field(repr=False)
    permutation_owners: tuple[object, object] = field(repr=False)
    verified_orders: tuple[V1PairedBCEVerifiedSealedOrder, ...] = field(repr=False)
    handle_fields: tuple[tuple[object, ...], ...] = field(repr=False)

    def assert_unchanged(self, *, bundle, registration, frame) -> None:
        from mechanistic_systems.experimental_design.finite_population import (
            verify_sealed_uniform_permutation_custody,
        )

        if (
            bundle is not self.bundle
            or bundle.provider is not self.provider
            or bundle.provider_seal is not self.provider_seal
            or _active_paired_oracle_evaluator(bundle.provider) is not self.evaluator
            or registration is not self.registration
            or registration.dynamic_oracle_receipt is not self.dynamic
            or frame is not self.frame
        ):
            raise ValueError("paired-BCE order epoch immutable owner changed")
        for name, original in self.frame_fields:
            observed = getattr(frame, name)
            if type(observed) is not type(original) or (
                observed is not original if type(original) is tuple
                else observed != original
            ):
                raise ValueError("paired-BCE order epoch frame custody changed")
        registration.assert_integrity(self.registration_hash)
        if self.dynamic.oracle_sampling_frame_receipt_hash != self.frame_hash:
            raise ValueError("paired-BCE order epoch frame identity changed")
        owners = (frame.target_assignment_receipt, frame.complement_assignment_receipt)
        for index, owner in enumerate(owners):
            if owner is not self.permutation_owners[index]:
                raise ValueError("paired-BCE order epoch permutation owner changed")
            if (
                verify_sealed_uniform_permutation_custody(owner)
                != self.dynamic.oracle_permutation_receipt_hashes[index]
            ):
                raise ValueError("paired-BCE order epoch permutation identity changed")
        for handle, expected in zip(self.verified_orders, self.handle_fields, strict=True):
            for item, original in zip(fields(handle), expected, strict=True):
                observed = getattr(handle, item.name)
                # The sole population-length value is an immutable tuple. Never
                # compare its contents, hash it, or enumerate it on a cache hit.
                if type(observed) is not type(original) or (
                    observed is not original if type(original) is tuple
                    else observed != original
                ):
                    raise ValueError("paired-BCE order epoch verified handle changed")


class _ProductionPairedBCEOrderEpochCache:
    """One dynamic O epoch per orchestration invocation; no restored authority."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: _ProductionPairedBCEOrderEpoch | None = None
        self._closed = False

    @contextlib.contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("paired-BCE order epoch cache is busy")
        try:
            if self._closed:
                raise RuntimeError("paired-BCE order epoch cache is closed")
            yield
        finally:
            self._lock.release()

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("paired-BCE order epoch cache is busy")
        try:
            self._entry = None
            self._closed = True
        finally:
            self._lock.release()

    def bind(self, *, bundle, registration, frame) -> _ProductionPairedBCEOrderEpoch:
        from .modern_v1_alexnet_fq_registration import (
            V1AlexNetFQOracleSamplingFrameReceipt,
        )
        from mechanistic_systems.experimental_design.finite_population import (
            verify_sealed_uniform_permutation_custody,
        )

        if type(registration) is not V1PairedBCEAcquisitionRegistration:
            raise TypeError("order epoch requires an exact paired-BCE registration")
        if type(frame) is not V1AlexNetFQOracleSamplingFrameReceipt:
            raise TypeError("order epoch requires an exact dynamic O frame")
        with self._exclusive():
            registration_hash = registration.assert_integrity()
            dynamic = registration.dynamic_oracle_receipt
            if type(dynamic) is not V1PairedBCEDynamicOracleReceipt:
                raise TypeError("order epoch requires an exact dynamic O receipt")
            # Full frame validation also remains in the sealed active-frame
            # getter. This cache does not replace the frame owner's verifier.
            frame_hash = frame.assert_integrity(dynamic.oracle_sampling_frame_receipt_hash)
            if self._entry is not None:
                self._entry.assert_unchanged(
                    bundle=self._entry.bundle,
                    registration=self._entry.registration,
                    frame=self._entry.frame,
                )
                if self._entry.registration_hash == registration_hash:
                    self._entry.assert_unchanged(
                        bundle=bundle, registration=registration, frame=frame,
                    )
                    return self._entry
                # A new registration retires old executor tokens before cold
                # verification; failed admission cannot resurrect their epoch.
                self._entry = None
            owners = (frame.target_assignment_receipt, frame.complement_assignment_receipt)
            for index, owner in enumerate(owners):
                if (
                    verify_sealed_uniform_permutation_custody(owner)
                    != dynamic.oracle_permutation_receipt_hashes[index]
                ):
                    raise ValueError("paired-BCE order epoch permutation identity changed")
            attempt_orders = _dynamic_attempt_orders(frame, dynamic)
            handles = tuple(
                verify_paired_bce_sealed_order(
                    registration=registration,
                    stratum_id=name,
                    sealed_context_order=attempt_orders.orders[name],
                )
                for name in PAIRED_BCE_STRATA
            )
            snapshots = tuple(
                tuple(getattr(handle, item.name) for item in fields(handle))
                for handle in handles
            )
            frame_fields = tuple(
                (item.name, getattr(frame, item.name))
                for item in fields(frame)
                if item.name not in {
                    "target_assignment_receipt", "complement_assignment_receipt",
                }
            )

            def require_immutable(value: object) -> None:
                if type(value) is tuple:
                    for item in value:
                        require_immutable(item)
                elif value is not None and type(value) not in (str, int, bool):
                    raise TypeError("order epoch requires immutable frame fields")

            # Cold only: tuple identity is a valid future guard only after
            # every nested value is known to have immutable exact types.
            for _, value in frame_fields:
                require_immutable(value)
            entry = _ProductionPairedBCEOrderEpoch(
                bundle=bundle,
                provider=bundle.provider,
                provider_seal=bundle.provider_seal,
                evaluator=_active_paired_oracle_evaluator(bundle.provider),
                registration=registration,
                registration_hash=registration_hash,
                dynamic=dynamic,
                frame=frame,
                frame_hash=frame_hash,
                frame_fields=frame_fields,
                permutation_owners=owners,
                verified_orders=handles,
                handle_fields=snapshots,
            )
            # Reopen source joins after the cold verification and before
            # publication. No callback runs under cache custody.
            frame.assert_integrity(frame_hash)
            entry.assert_unchanged(bundle=bundle, registration=registration, frame=frame)
            self._entry = entry
            return entry

    def assert_live(self, entry: _ProductionPairedBCEOrderEpoch, *, frame: object) -> None:
        with self._exclusive():
            if entry is not self._entry:
                raise ValueError("paired-BCE order executor belongs to a retired epoch")
            entry.assert_unchanged(
                bundle=entry.bundle, registration=entry.registration, frame=frame,
            )


def _production_paired_bce_chunk_executor(
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    row_store_root: Path,
    *,
    order_epoch_cache: _ProductionPairedBCEOrderEpochCache | None = None,
) -> Callable[[V1RuntimePendingPairedBCEChunk], V1RuntimePairedBCEChunkOutcome]:
    if bundle.provider_seal.evaluator_kind != "parent_aware":
        raise PermissionError(
            "paired-BCE chunk execution requires a parent-aware direct oracle"
        )
    evaluator = _active_paired_oracle_evaluator(bundle.provider)
    active = state.controller.active_candidate
    if (
        state.controller.phase != "candidate_active"
        or active is None
        or active.paired_bce_registration is None
        or active.paired_bce_state is None
    ):
        raise PermissionError("paired-BCE chunk execution requires a live registered candidate")
    registration = active.paired_bce_registration
    paired_state = active.paired_bce_state
    parent_mask_hash = active.proposal.parent_mask_hash
    dense_reference_hash = registration.dense_reference_hash
    candidate_mask_hash = active.proposal.resulting_mask.mask_hash
    if parent_mask_hash != state.controller.current_mask.mask_hash:
        raise ValueError("paired-BCE active candidate parent differs from controller mask")
    metadata = _paired_bce_metadata(bundle)
    if metadata is None:
        raise RuntimeError("paired-BCE production metadata is absent")
    expected_row_provider_identity_hash = _metadata_identity(
        metadata,
        "row_provider_identity_hash",
        default_payload={
            "schema": "partition_free_v1_production_paired_bce_row_provider_v1",
            "provider_hash": bundle.provider_seal.provider_hash,
        },
        prefix="partition-free-v1-production-paired-bce-row-provider-",
    )
    expected_loss_evaluator_identity_hash = _metadata_identity(
        metadata,
        "paired_loss_evaluator_identity_hash",
        default_payload={
            "schema": "partition_free_v1_production_paired_bce_loss_evaluator_v1",
            "evaluator_hash": bundle.provider_seal.evaluator_seal.identity_hash,
        },
        prefix="partition-free-v1-production-paired-bce-loss-evaluator-",
    )
    dynamic = registration.dynamic_oracle_receipt
    bound_dynamic_frame_hash: str | None = None
    order_epoch: _ProductionPairedBCEOrderEpoch | None = None
    if dynamic is not None:
        frame = _production_active_dynamic_oracle_frame(bundle, required=True)
        if frame is None:  # pragma: no cover - required above.
            raise RuntimeError("dynamic O frame disappeared")
        frame.assert_integrity(dynamic.oracle_sampling_frame_receipt_hash)
        bound_dynamic_frame_hash = frame.receipt_hash
        if order_epoch_cache is None:
            attempt_orders = _dynamic_attempt_orders(frame, dynamic)
        else:
            order_epoch = order_epoch_cache.bind(
                bundle=bundle, registration=registration, frame=frame,
            )
    else:
        if bundle.provider_seal.evaluator_active_dynamic_oracle_frame_hook_spec is not None:
            raise PermissionError("paired-BCE executor lacks its registered dynamic O frame")
        static_oracle_axes = _named_axis_map(
            metadata["oracle_context_ids_by_stratum"],
            label="paired-BCE oracle context axes",
        )
        attempt_orders = _production_paired_bce_attempt_orders(
            bundle,
            metadata=metadata,
            candidate_attempt_index=registration.candidate_attempt_index,
            oracle_axes=static_oracle_axes,
        )
    if order_epoch is None:
        verified_attempt_orders = MappingProxyType({
            name: verify_paired_bce_sealed_order(
                registration=registration,
                stratum_id=name,
                sealed_context_order=attempt_orders.orders[name],
            )
            for name in PAIRED_BCE_STRATA
        })
    else:
        verified_attempt_orders = MappingProxyType(dict(
            zip(PAIRED_BCE_STRATA, order_epoch.verified_orders, strict=True)
        ))
    disclosure_seal = bundle.attempt_structural_disclosure
    disclosure_state = {"issued": False}
    disclosure_dynamic = registration.dynamic_oracle_receipt
    disclosure_generation = getattr(disclosure_dynamic, "generation_index", None)
    disclosure_attempt = getattr(disclosure_dynamic, "candidate_attempt_index", None)

    def executor(pending: V1RuntimePendingPairedBCEChunk) -> V1RuntimePairedBCEChunkOutcome:
        if type(pending) is not V1RuntimePendingPairedBCEChunk:
            raise TypeError("paired-BCE executor requires the canonical pending chunk")
        request = pending.request
        if type(request) is not V1PairedBCEChunkRequest:
            raise TypeError("paired-BCE pending chunk carries a noncanonical request")
        contexts = tuple(pending.context_ids_in_registered_slice)
        if (
            pending.registration_hash != registration.registration_hash
            or pending.prior_paired_bce_state_hash != paired_state.state_hash
            or pending.dynamic_oracle_receipt_hash
            != registration.dynamic_oracle_receipt_hash
        ):
            raise ValueError("paired-BCE pending chunk is stale for the live state")
        if (
            pending.request.row_provider_identity_hash
            != expected_row_provider_identity_hash
            or pending.request.paired_loss_evaluator_identity_hash
            != expected_loss_evaluator_identity_hash
        ):
            raise ValueError("paired-BCE pending chunk request names another production evaluator")
        live_frame = _validate_bound_dynamic_frame_live(
            bundle=bundle,
            dynamic=dynamic,
            bound_dynamic_frame_hash=bound_dynamic_frame_hash,
        )
        if order_epoch is not None:
            order_epoch_cache.assert_live(order_epoch, frame=live_frame)
        if request.stratum_id not in PAIRED_BCE_STRATA:
            raise ValueError("paired-BCE request stratum is invalid")
        verified_order = verified_attempt_orders[request.stratum_id]
        validated_contexts = validate_paired_bce_chunk_request_preoutcome(
            registration=registration,
            state=paired_state,
            request=request,
            context_ids_in_registered_slice=contexts,
            verified_sealed_order=verified_order,
        )
        if validated_contexts != contexts:
            raise ValueError("paired-BCE pending chunk context axis changed during validation")
        if disclosure_seal is not None and not disclosure_state["issued"]:
            # Once per executor on its first valid chunk; the orchestration
            # loops currently construct one executor per physical chunk.
            # exact Theorem-D bookkeeping of the action as a sibling receipt.
            # Disclosure only; it cannot alter rows, receipts, or the decision.
            disclosure_state["issued"] = True
            _issue_attempt_structural_disclosure(
                bundle,
                disclosure_seal,
                parent_mask_hash=parent_mask_hash,
                candidate_mask_hash=candidate_mask_hash,
                dense_reference_hash=dense_reference_hash,
                contexts=contexts,
                generation_index=disclosure_generation,
                candidate_attempt_index=disclosure_attempt,
            )
            live_frame = _validate_bound_dynamic_frame_live(
                bundle=bundle,
                dynamic=dynamic,
                bound_dynamic_frame_hash=bound_dynamic_frame_hash,
            )
        if order_epoch is not None:
            order_epoch_cache.assert_live(order_epoch, frame=live_frame)
        rows, evaluator_receipt_hash = _evaluate_paired_bce_rows_for_pending(
            bundle=bundle,
            evaluator=evaluator,
            registration=registration,
            paired_state=paired_state,
            pending=pending,
            dense_reference_hash=dense_reference_hash,
            candidate_mask_hash=candidate_mask_hash,
            parent_mask_hash=parent_mask_hash,
        )
        receipt = reduce_paired_bce_chunk(
            registration=registration,
            state=paired_state,
            request=pending.request,
            context_ids_in_registered_slice=contexts,
            paired_loss_rows=rows,
            paired_loss_evaluator_receipt_hash=evaluator_receipt_hash,
        )
        _store_paired_bce_chunk_artifact(
            row_store_root,
            registration=registration,
            state=paired_state,
            pending=pending,
            receipt=receipt,
            contexts=contexts,
            rows=rows,
        )
        committed = commit_paired_bce_chunk(
            registration=registration,
            state=paired_state,
            request=pending.request,
            receipt=receipt,
        )
        return V1RuntimePairedBCEChunkOutcome(
            pending_hash=pending.pending_hash,
            request_hash=pending.request.request_hash,
            receipt=receipt,
            committed_state=committed,
        )

    dynamic_receipt = registration.dynamic_oracle_receipt
    o_generation = getattr(dynamic_receipt, "generation_index", None)
    o_attempt = getattr(dynamic_receipt, "candidate_attempt_index", None)

    def timed_executor(
        pending: V1RuntimePendingPairedBCEChunk,
    ) -> V1RuntimePairedBCEChunkOutcome:
        with _o_stage_timer(
            bundle.provider,
            "stage_O_chunk",
            generation=o_generation,
            attempt=o_attempt,
        ):
            return executor(pending)

    return timed_executor


def _advance_v1_production_staged_proposal_once(
    state: V1RuntimeState,
    *,
    bundle: V1ProductionRuntimeBundle,
) -> V1RuntimeState:
    """Advance exactly one causal proposal-acquisition micro-transition."""

    if not bundle.has_staged_proposal_protocol:
        raise PermissionError("production bundle has no staged proposal protocol")
    backend = bundle.proposal_backend_seal
    initial_factory = bundle.proposal_initial_state_factory
    forecast_factory = bundle.proposal_forecast_factory
    request_plan_factory = bundle.proposal_request_plan_factory
    executor = bundle.proposal_acquisition_executor
    freeze_evaluator = bundle.proposal_freeze_evaluator
    finalizer = bundle.proposal_acquisition_finalizer
    if (
        backend is None
        or initial_factory is None
        or forecast_factory is None
        or request_plan_factory is None
        or executor is None
        or freeze_evaluator is None
        or finalizer is None
    ):  # pragma: no cover - bundle invariant
        raise RuntimeError("staged proposal protocol is incomplete")
    if v1_runtime_proposal_order_indeterminate(state):
        require_v1_runtime_proposal_order_resolved(state)
    if state.proposal_progress is None:
        # Alpha-bound score inference needs the closed generation ledger and
        # therefore consumes the exact runtime state.  Legacy backends retain
        # the controller-only ABI.  The sealed field makes this an explicit
        # compatibility branch rather than exception-driven introspection.
        initial = (
            initial_factory(state)
            if backend.proposal_order_alpha_schedule_hash is not None
            else initial_factory(state.controller)
        )
        if type(initial) is not V1RuntimeProposalSufficientStateDescriptor:
            raise TypeError(
                "proposal initial-state factory returned another descriptor type"
            )
        if (
            initial.representation_hash
            != backend.sufficient_state_representation_hash
        ):
            raise ValueError(
                "proposal initial-state factory returned another sufficient-state ABI"
            )
        return initialize_v1_runtime_proposal_acquisition(
            state,
            action_regret_metric_hash=backend.action_regret_metric_hash,
            proposal_backend_identity_hash=backend.identity_hash,
            phase_evidence_state_hash=backend.initial_phase_evidence_state_hash,
            freeze_criterion_hash=backend.freeze_criterion_hash,
            initial_sufficient_state=initial,
        )
    progress = state.proposal_progress
    if progress.proposal_backend_identity_hash != backend.identity_hash:
        raise ValueError("live proposal state names another backend seal")
    latest_decision = (
        None
        if not progress.freeze_decision_receipts
        else progress.freeze_decision_receipts[-1]
    )
    latest_decision_at_frontier = (
        latest_decision is not None
        and latest_decision.state_hash == progress.joint_state.content_hash
    )
    stable_negative_fqr = (
        progress.current_sufficient_state.storage_descriptor.get(
            "proposal_inference_stage"
        )
        == "fqr_C_stable_no_candidate"
    )
    if stable_negative_fqr and not latest_decision_at_frontier:
        frozen, _decision = decide_v1_runtime_proposal_freeze(
            state,
            evaluator=freeze_evaluator,
        )
        return frozen
    if stable_negative_fqr:
        if latest_decision.action != "refuse_candidate_freeze":
            raise PermissionError(
                "stable-negative F/Q/R frontier has an authoritative candidate "
                "freeze instead of its proof-bound refusal"
            )
        evaluator_owner = getattr(freeze_evaluator.callable_value, "__self__", None)
        proof_builder = getattr(
            evaluator_owner,
            "fqr_no_candidate_horizon_proof",
            None,
        )
        if not callable(proof_builder):
            raise PermissionError(
                "sealed F/Q/R freeze evaluator lacks its exact horizon-proof "
                "reconstruction method"
            )
        return finalize_v1_runtime_fqr_no_candidate_horizon(
            state,
            proof_builder=proof_builder,
        )
    if progress.authorized_freeze_decision is not None:
        return finalize_and_attach_v1_runtime_proposal_acquisition(
            state,
            finalizer=finalizer,
        )
    if state.pending_proposal_acquisition is not None:
        updated = execute_v1_runtime_pending_proposal_acquisition(
            state,
            executor=executor,
        )
        updated_progress = updated.proposal_progress
        if (
            updated_progress is None  # pragma: no cover - runtime invariant
            or updated_progress.current_sufficient_state.representation_hash
            != backend.sufficient_state_representation_hash
        ):
            raise ValueError(
                "proposal executor returned another sufficient-state ABI"
            )
        return updated
    if progress.update_receipts and not latest_decision_at_frontier:
        frozen, _decision = decide_v1_runtime_proposal_freeze(
            state,
            evaluator=freeze_evaluator,
        )
        return frozen
    prepared, receipt = prepare_v1_runtime_proposal_acquisition(
        state,
        forecast_factory=forecast_factory,
        request_plan_factory=request_plan_factory,
    )
    if receipt.action == "acquire":
        return prepared
    if receipt.action != "refuse_no_legal_action":
        raise RuntimeError("staged proposal scheduler returned another terminal action")
    # The dispatcher must durably checkpoint this exact refusal before raising
    # the typed indeterminacy result.  Returning it is therefore deliberate.
    return prepared


def _proposal_qualification_microtransition_count(state: V1RuntimeState) -> int:
    """Reconstruct the exact staged-proposal transition count from state.

    The count is state-derived so a crash/restart produces the same terminal
    qualification receipt as an uninterrupted run.  One transition opens the
    proposal generation; each completed request contributes prepare+execute;
    a pending request contributes its durable prepare; every freeze decision
    and a terminal scheduler refusal contribute their own transition.
    """

    progress = state.proposal_progress
    if progress is None:
        return 0
    return (
        1
        + 2 * len(progress.outcome_receipts)
        + len(progress.freeze_decision_receipts)
        + int(state.pending_proposal_acquisition is not None)
        + int(progress.terminal_refusal_receipt is not None)
    )


def _proposal_qualification_status(
    state: V1RuntimeState,
) -> ProposalQualificationStatus | None:
    progress = state.proposal_progress
    if progress is None:
        return None
    if progress.authorized_freeze_decision is not None:
        return "candidate_family_frozen"
    if getattr(
        state.controller,
        "fqr_no_candidate_horizon_witness",
        None,
    ) is not None:
        return "no_candidate_fqr_horizon_proved"
    if progress.proposal_order_indeterminate:
        return "proposal_order_indeterminate"
    return None


def run_v1_production_proposal_qualification(
    bundle: V1ProductionRuntimeBundle,
    *,
    output_directory: str | os.PathLike[str],
    resume_checkpoint: str | os.PathLike[str] | None = None,
    trusted_local_resume: bool = False,
    maximum_microtransitions: int = 100_000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> V1ProductionProposalQualificationArtifacts:
    """Run exactly one train-only proposal generation and stop before K/O.

    This is the production-shaped qualification seam for D geometry, adaptive
    rank replay, KKT composition, directional C certification, and proposal
    freeze.  It writes the ordinary restart-safe runtime checkpoint after each
    micro-transition, but never calls the pure finalizer that would attach a
    candidate queue.  Consequently calibration, the paired BCE oracle, mask
    commit, terminal validation, and Pareto authority remain structurally
    unreachable.
    """

    if type(bundle) is not V1ProductionRuntimeBundle:
        raise TypeError("proposal qualification requires an exact production bundle")
    if not bundle.has_staged_proposal_protocol:
        raise PermissionError("proposal qualification requires staged proposal owners")
    bound = _positive(maximum_microtransitions, "maximum_microtransitions")
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / _PROPOSAL_QUALIFICATION_CHECKPOINT_FILENAME
    receipt_path = root / _PROPOSAL_QUALIFICATION_RECEIPT_FILENAME
    binding_path = root / _PRODUCTION_RESUME_BINDING_FILENAME
    if receipt_path.exists():
        raise FileExistsError(
            "proposal qualification output already contains a terminal receipt"
        )

    state: V1RuntimeState
    resumed_receipt: V1RuntimeCheckpointReceipt | None = None

    def emit(phase: str, **details: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "phase": phase,
                "controller_phase": state.controller.phase,
                "proposal_microtransition_count": (
                    _proposal_qualification_microtransition_count(state)
                ),
                **details,
            }
        )

    if resume_checkpoint is None:
        if checkpoint_path.exists():
            raise FileExistsError(
                "proposal qualification checkpoint already exists; use explicit resume"
            )
        _publish_or_validate_production_resume_binding(binding_path, bundle=bundle)
        state = _initial_runtime(bundle)
        state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
    else:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        source_binding = resume_path.parent / _PRODUCTION_RESUME_BINDING_FILENAME
        if not source_binding.is_file():
            raise ValueError(
                "staged proposal qualification resume lacks its production binding"
            )
        _validate_production_resume_binding(source_binding, bundle=bundle)
        _publish_or_validate_production_resume_binding(binding_path, bundle=bundle)
        state, resumed_receipt = read_v1_runtime_checkpoint(
            resume_path,
            trusted_local=trusted_local_resume,
        )
        _validate_resumed_state(bundle, state)
        _restore_resumed_proposal_backend_state(bundle, state)
        _restore_resumed_provider_process_state(bundle, state.controller)
        state = replace(state, config=bundle.runtime_config)
        state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)

    emit("proposal_qualification_start")
    while _proposal_qualification_status(state) is None:
        if _proposal_qualification_microtransition_count(state) >= bound:
            break
        state = _advance_v1_production_staged_proposal_once(state, bundle=bundle)
        state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
        emit("proposal_qualification_transition_checkpointed")

    status = _proposal_qualification_status(state)
    if status is None:
        status = "resource_cap_unresolved"
    progress = state.proposal_progress
    backend = bundle.proposal_backend_seal
    if progress is None or backend is None:
        raise RuntimeError("proposal qualification did not retain staged owners")
    authorized = progress.authorized_freeze_decision
    refusal = progress.terminal_refusal_receipt
    horizon_witness = getattr(
        state.controller,
        "fqr_no_candidate_horizon_witness",
        None,
    )
    receipt = V1ProductionProposalQualificationReceipt(
        status=status,
        production_manifest_hash=bundle.manifest.manifest_hash,
        production_bundle_hash=bundle.bundle_hash,
        campaign_contract_hash=bundle.binding.contract.campaign_contract_hash,
        campaign_binding_hash=bundle.binding.binding_hash,
        controller_run_id=bundle.manifest.controller_run_id,
        proposal_backend_identity_hash=backend.identity_hash,
        final_runtime_state_hash=state.runtime_state_hash,
        final_controller_state_hash=state.controller.state_hash,
        final_controller_snapshot_hash=(
            state.controller.snapshot_receipt().receipt_hash
        ),
        current_mask_hash=state.controller.current_mask.mask_hash,
        route_axis_hash=state.controller.current_mask.route_axis_hash,
        proposal_progress_hash=progress.progress_hash,
        proposal_frontier_hash=progress.frontier_hash,
        proposal_joint_state_hash=progress.joint_state.content_hash,
        current_sufficient_state_hash=(
            progress.current_sufficient_state.sufficient_state_hash
        ),
        current_sufficient_state_descriptor_hash=(
            progress.current_sufficient_state.descriptor_hash
        ),
        authorized_freeze_decision_hash=(
            None if authorized is None else authorized.receipt_hash
        ),
        terminal_refusal_receipt_hash=(
            None if refusal is None else refusal.receipt_hash
        ),
        fqr_no_candidate_horizon_witness_hash=(
            None if horizon_witness is None else horizon_witness.witness_hash
        ),
        update_receipt_hashes=tuple(
            item.receipt_hash for item in progress.update_receipts
        ),
        freeze_decision_receipt_hashes=tuple(
            item.receipt_hash for item in progress.freeze_decision_receipts
        ),
        microtransition_count=_proposal_qualification_microtransition_count(state),
        checkpoint_payload_sha256=final_checkpoint.payload_sha256,
    )
    artifacts = V1ProductionProposalQualificationArtifacts(
        state=state,
        receipt=receipt,
        final_checkpoint=final_checkpoint,
        resumed_checkpoint=resumed_receipt,
    )
    _atomic_write_json(receipt_path, receipt.to_json_payload())
    emit(
        "proposal_qualification_complete",
        status=receipt.status,
        receipt_hash=receipt.receipt_hash,
    )
    return artifacts


def run_v1_production_train_only_decision_qualification(
    bundle: V1ProductionRuntimeBundle,
    *,
    output_directory: str | os.PathLike[str],
    resume_checkpoint: str | os.PathLike[str] | None = None,
    trusted_local_resume: bool = False,
    maximum_microtransitions: int = 100_000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> V1TrainOnlyDecisionQualificationArtifacts:
    """Qualify exactly one C→K→O→decision seam without opening validation.

    The durable origin binding fixes the target to one additional candidate
    attempt across crash/restart.  An accepted nonterminal decision also opens
    exactly the next fresh chart, which retires the committed generation's
    process-local D/C/K/O state and invalidates its queue before this function
    returns.  Terminal extraction, terminal validation, performance closure,
    and Pareto adjudication are intentionally absent from this call graph.
    """

    if type(bundle) is not V1ProductionRuntimeBundle:
        raise TypeError(
            "train-only decision qualification requires an exact production bundle"
        )
    if not bundle.has_staged_proposal_protocol:
        raise PermissionError(
            "train-only decision qualification requires staged proposal owners"
        )
    if not bundle.has_paired_bce_oracle_protocol:
        raise PermissionError(
            "train-only decision qualification requires paired-BCE K/O owners"
        )
    transition_bound = _positive(
        maximum_microtransitions,
        "maximum_microtransitions",
    )
    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        root / _TRAIN_ONLY_DECISION_QUALIFICATION_CHECKPOINT_FILENAME
    )
    receipt_path = root / _TRAIN_ONLY_DECISION_QUALIFICATION_RECEIPT_FILENAME
    binding_path = root / _TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_FILENAME
    production_binding_path = root / _PRODUCTION_RESUME_BINDING_FILENAME
    row_store_root = root / "paired_bce_external_rows"
    if _path_lexists(receipt_path):
        raise FileExistsError(
            "train-only decision output already contains a terminal receipt"
        )

    state: V1RuntimeState
    final_checkpoint: V1RuntimeCheckpointReceipt
    resumed_receipt: V1RuntimeCheckpointReceipt | None = None

    if resume_checkpoint is None:
        if any(
            _path_lexists(path)
            for path in (
                checkpoint_path,
                binding_path,
                production_binding_path,
                row_store_root,
            )
        ):
            raise FileExistsError(
                "train-only decision output already contains run state; "
                "use explicit trusted resume"
            )
        _publish_or_validate_production_resume_binding(
            production_binding_path,
            bundle=bundle,
        )
        state = _initial_runtime(bundle)
        state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
        qualification_binding = _new_train_only_decision_binding(
            bundle=bundle,
            state=state,
            checkpoint=final_checkpoint,
        )
        _atomic_write_json(binding_path, qualification_binding.to_json_payload())
    else:
        raw_resume_path = Path(resume_checkpoint).expanduser()
        _require_nonsymlink_regular_file(
            raw_resume_path,
            label="train-only decision resume checkpoint",
        )
        resume_path = raw_resume_path.resolve()
        source_root = resume_path.parent
        destination_checkpoint_exists = _path_lexists(checkpoint_path)
        destination_is_resume_checkpoint = bool(
            destination_checkpoint_exists
            and checkpoint_path.resolve() == resume_path
        )
        if destination_checkpoint_exists and not destination_is_resume_checkpoint:
            raise FileExistsError(
                "train-only decision destination contains another checkpoint"
            )
        source_production_binding = (
            source_root / _PRODUCTION_RESUME_BINDING_FILENAME
        )
        if not _path_lexists(source_production_binding):
            raise ValueError(
                "train-only staged resume lacks its sealed production binding"
            )
        _validate_production_resume_binding(source_production_binding, bundle=bundle)
        if _path_lexists(production_binding_path):
            _validate_production_resume_binding(
                production_binding_path,
                bundle=bundle,
            )
        state, resumed_receipt = read_v1_runtime_checkpoint(
            resume_path,
            trusted_local=trusted_local_resume,
        )
        _validate_resumed_state(bundle, state)
        source_qualification_binding = (
            source_root / _TRAIN_ONLY_DECISION_QUALIFICATION_BINDING_FILENAME
        )
        if _path_lexists(source_qualification_binding):
            qualification_binding = _read_train_only_decision_binding(
                source_qualification_binding
            )
        else:
            qualification_binding = _new_train_only_decision_binding(
                bundle=bundle,
                state=state,
                checkpoint=resumed_receipt,
            )
        _validate_train_only_decision_binding(
            qualification_binding,
            bundle=bundle,
            state=state,
        )
        if _path_lexists(binding_path):
            installed = _read_train_only_decision_binding(binding_path)
            if installed.binding_hash != qualification_binding.binding_hash:
                raise ValueError(
                    "train-only decision destination names another origin binding"
                )
        _restore_resumed_proposal_backend_state(bundle, state)
        _restore_resumed_provider_process_state(bundle, state.controller)
        state = replace(state, config=bundle.runtime_config)

        source_row_store = source_root / "paired_bce_external_rows"
        _validate_paired_bce_external_replay_if_needed(
            bundle=bundle,
            state=state,
            row_store_root=source_row_store,
        )
        _materialize_paired_bce_row_store(
            source_root=source_row_store,
            destination_root=row_store_root,
        )
        _validate_paired_bce_external_replay_if_needed(
            bundle=bundle,
            state=state,
            row_store_root=row_store_root,
        )
        _publish_or_validate_production_resume_binding(
            production_binding_path,
            bundle=bundle,
        )
        if not _path_lexists(binding_path):
            _atomic_write_json(
                binding_path,
                qualification_binding.to_json_payload(),
            )
        state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)

    _validate_train_only_decision_binding(
        qualification_binding,
        bundle=bundle,
        state=state,
    )

    paired_calibration_freeze_factory = (
        _production_paired_bce_calibration_freeze_factory(bundle)
    )
    paired_calibration_horizon_skip_factory = (
        _production_paired_bce_calibration_horizon_skip_factory(bundle)
    )
    paired_registration_factory = _production_paired_bce_registration_factory(bundle)
    order_epoch_cache = _ProductionPairedBCEOrderEpochCache()
    try:
        paired_chunk_factory = _production_paired_bce_chunk_factory(
            bundle, order_epoch_cache=order_epoch_cache,
        )
        paired_betting_replay_factory = _production_paired_bce_betting_replay_factory(
            bundle,
            row_store_root,
        )
        _validate_paired_bce_external_replay_if_needed(
            bundle=bundle,
            state=state,
            row_store_root=row_store_root,
        )

        decision: ProductionDecision | None = None
        transitions = 0
        status: TrainOnlyDecisionQualificationStatus | None = None

        def emit(phase: str, **details: Any) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "phase": phase,
                    "controller_phase": state.controller.phase,
                    "candidate_attempt_count": state.controller.candidate_attempt_count,
                    "accepted_step_count": state.controller.accepted_step_count,
                    "qualification_binding_hash": qualification_binding.binding_hash,
                    "target_candidate_attempt_count": (
                        qualification_binding.target_candidate_attempt_count
                    ),
                    "microtransitions_this_invocation": transitions,
                    **details,
                }
            )

        emit("train_only_decision_qualification_start")
        while (
            state.controller.candidate_attempt_count
            < qualification_binding.target_candidate_attempt_count
        ):
            if v1_runtime_proposal_order_indeterminate(state):
                state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
                status = "proposal_order_indeterminate"
                emit("train_only_proposal_order_indeterminate")
                break
            if state.controller.phase == "stopped":
                status = "scientific_endpoint_without_decision"
                break
            if transitions >= transition_bound:
                status = "resource_cap_unresolved"
                break

            step_decision: ProductionDecision | None = None
            if state.controller.phase == "candidate_active":
                active = state.controller.active_candidate
                if active is None or active.paired_bce_registration is None:
                    raise PermissionError(
                        "train-only candidate lacks paired-BCE registration"
                    )
                if state.pending_paired_bce_chunk is None:
                    state = prepare_v1_runtime_paired_bce_chunk(
                        state,
                        chunk_factory=paired_chunk_factory,
                    )
                    state, final_checkpoint = _checkpoint_progress(
                        checkpoint_path,
                        state,
                    )
                    emit("train_only_paired_bce_chunk_prepared")
                state = execute_v1_runtime_pending_paired_bce_chunk(
                    state,
                    executor=_production_paired_bce_chunk_executor(
                        bundle,
                        state,
                        row_store_root,
                        order_epoch_cache=order_epoch_cache,
                    ),
                )
                state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
                emit("train_only_paired_bce_chunk_executed")
                state, paired_decision = decide_v1_runtime_paired_bce_oracle(
                    state,
                    betting_replay_factory=paired_betting_replay_factory,
                )
                if paired_decision.status != "acquire_more":
                    step_decision = paired_decision
            elif state.controller.phase == "queue_ready":
                state, calibration_skip = (
                    advance_v1_runtime_paired_bce_candidate_freeze(
                        state,
                        threshold_factory=bundle.threshold_factory,
                        sampling_law_factory=bundle.sampling_law_factory,
                        calibration_freeze_factory=(
                            paired_calibration_freeze_factory
                        ),
                        paired_bce_registration_factory=(
                            paired_registration_factory
                        ),
                        calibration_horizon_skip_factory=(
                            paired_calibration_horizon_skip_factory
                        ),
                    )
                )
                step_decision = calibration_skip
            elif (
                state.controller.phase == "needs_acquisition"
                or state.proposal_progress is not None
                or state.pending_proposal_acquisition is not None
            ):
                state = _advance_v1_production_staged_proposal_once(
                    state,
                    bundle=bundle,
                )
            else:  # pragma: no cover - exact controller phase union.
                raise RuntimeError(
                    "train-only decision qualification reached an unsupported phase"
                )

            transitions += 1
            if step_decision is not None:
                decision = step_decision
            state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
            emit(
                "train_only_decision_transition_checkpointed",
                decision_action=(
                    None
                    if step_decision is None
                    else _train_only_decision_action(step_decision)
                ),
            )

    finally:
        order_epoch_cache.close()

    if (
        state.controller.candidate_attempt_count
        == qualification_binding.target_candidate_attempt_count
    ):
        status = "decision_qualified"
    elif status is None:
        status = "resource_cap_unresolved"

    # A committed nonterminal child must retire its frozen generation and open
    # a fresh chart before the qualification can return.  This calls the same
    # sealed backend initial-state joint as full production, but stops before
    # the first D request is prepared.
    terminal_attempt = (
        None
        if status != "decision_qualified"
        else state.controller.attempt_receipts[-1]
    )
    if terminal_attempt is not None and terminal_attempt.action == "accept_candidate":
        if state.controller.phase == "needs_acquisition":
            if state.proposal_progress is None:
                state = _advance_v1_production_staged_proposal_once(
                    state,
                    bundle=bundle,
                )
                state, final_checkpoint = _checkpoint_progress(
                    checkpoint_path,
                    state,
                )
                emit("train_only_fresh_chart_opened")
        elif state.controller.phase == "stopped":
            _retire_terminal_accepted_generation_state(bundle, state, decision)
        else:  # pragma: no cover - accepted controller invariant.
            raise RuntimeError("accepted train-only decision retained a stale queue")

    state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
    _validate_train_only_decision_binding(
        qualification_binding,
        bundle=bundle,
        state=state,
    )
    controller = state.controller
    progress = state.proposal_progress
    fresh_chart_opened = bool(
        terminal_attempt is not None
        and terminal_attempt.action == "accept_candidate"
        and controller.phase == "needs_acquisition"
        and progress is not None
    )
    if status == "decision_qualified":
        if terminal_attempt is None:  # pragma: no cover - state-derived above.
            raise RuntimeError("qualified train-only seam lost its attempt")
        decision_type = (
            type(decision).__name__
            if decision is not None
            else (
                "V1PairedBCECalibrationHorizonSkipDecision"
                if terminal_attempt.action
                == "skip_candidate_calibration_refusal"
                else "V1PairedBCEDecision"
            )
        )
        decision_action = terminal_attempt.action
        decision_receipt_hash = terminal_attempt.decision_receipt_hash
    else:
        decision_type = None
        decision_action = None
        decision_receipt_hash = None
    live_active = controller.active_candidate
    candidate_calibration_invoked = bool(
        status == "decision_qualified" or live_active is not None
    )
    paired_bce_oracle_invoked = bool(
        (
            status == "decision_qualified"
            and terminal_attempt is not None
            and terminal_attempt.action
            != "skip_candidate_calibration_refusal"
        )
        or (
            live_active is not None
            and live_active.paired_bce_state is not None
            and live_active.paired_bce_state.chunk_count > 0
        )
    )

    receipt = V1TrainOnlyDecisionQualificationReceipt(
        status=status,
        qualification_binding_hash=qualification_binding.binding_hash,
        production_manifest_hash=bundle.manifest.manifest_hash,
        production_bundle_hash=bundle.bundle_hash,
        campaign_contract_hash=bundle.binding.contract.campaign_contract_hash,
        campaign_binding_hash=bundle.binding.binding_hash,
        controller_run_id=bundle.manifest.controller_run_id,
        origin_runtime_state_hash=qualification_binding.origin_runtime_state_hash,
        final_runtime_state_hash=state.runtime_state_hash,
        final_controller_state_hash=controller.state_hash,
        final_controller_snapshot_hash=controller.snapshot_receipt().receipt_hash,
        origin_mask_hash=qualification_binding.origin_mask_hash,
        final_mask_hash=controller.current_mask.mask_hash,
        origin_candidate_attempt_count=(
            qualification_binding.origin_candidate_attempt_count
        ),
        final_candidate_attempt_count=controller.candidate_attempt_count,
        origin_accepted_step_count=(
            qualification_binding.origin_accepted_step_count
        ),
        final_accepted_step_count=controller.accepted_step_count,
        decision_type=decision_type,
        decision_action=decision_action,
        decision_receipt_hash=decision_receipt_hash,
        final_controller_phase=controller.phase,
        fresh_chart_progress_hash=(
            None if not fresh_chart_opened else progress.progress_hash
        ),
        checkpoint_payload_sha256=final_checkpoint.payload_sha256,
        candidate_calibration_invoked=candidate_calibration_invoked,
        paired_bce_oracle_invoked=paired_bce_oracle_invoked,
        mask_commit_occurred=(
            status == "decision_qualified"
            and terminal_attempt is not None
            and terminal_attempt.action == "accept_candidate"
        ),
        fresh_chart_opened=fresh_chart_opened,
    )
    artifacts = V1TrainOnlyDecisionQualificationArtifacts(
        state=state,
        binding=qualification_binding,
        receipt=receipt,
        final_checkpoint=final_checkpoint,
        decision=decision,
        resumed_checkpoint=resumed_receipt,
    )
    _atomic_write_json(receipt_path, receipt.to_json_payload())
    emit(
        "train_only_decision_qualification_complete",
        status=receipt.status,
        receipt_hash=receipt.receipt_hash,
    )
    return artifacts


def _legacy_terminal_extraction_closure(
    bundle: V1ProductionRuntimeBundle,
    controller: V1ControllerState,
) -> V1TerminalExtractionClosure:
    state_hash = bundle.manifest.extraction_state_hash
    calibration_hash = bundle.manifest.extraction_calibration_hash
    if state_hash is None or calibration_hash is None:  # pragma: no cover - manifest invariant
        raise RuntimeError("legacy terminal extraction hashes are unavailable")
    snapshot = controller.snapshot_receipt()
    closure = V1TerminalExtractionClosure(
        campaign_contract_hash=controller.binding.contract.campaign_contract_hash,
        campaign_binding_hash=controller.binding.binding_hash,
        controller_run_id=controller.controller_run_id,
        controller_snapshot_hash=snapshot.receipt_hash,
        terminal_mask_hash=controller.current_mask.mask_hash,
        acquisition_receipt_hashes=controller.acquisition_receipt_hashes,
        extraction_state_hash=state_hash,
        extraction_calibration_hash=calibration_hash,
    )
    _validate_terminal_extraction_closure(controller, closure)
    return closure


def _terminal_extraction_closure_from_json(
    payload: Mapping[str, Any],
) -> V1TerminalExtractionClosure:
    expected_keys = {
        "schema",
        "closure_hash",
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "controller_snapshot_hash",
        "terminal_mask_hash",
        "acquisition_receipt_hashes",
        "extraction_state_hash",
        "extraction_calibration_hash",
        "validation_independent",
        "terminal_rows_visible",
    }
    if set(payload) != expected_keys:
        raise ValueError("terminal extraction closure file has unexpected coordinates")
    lineage = payload["acquisition_receipt_hashes"]
    if isinstance(lineage, (str, bytes)) or not isinstance(lineage, (tuple, list)):
        raise TypeError("terminal extraction closure lineage must be a sequence")
    closure = V1TerminalExtractionClosure(
        campaign_contract_hash=payload["campaign_contract_hash"],
        campaign_binding_hash=payload["campaign_binding_hash"],
        controller_run_id=payload["controller_run_id"],
        controller_snapshot_hash=payload["controller_snapshot_hash"],
        terminal_mask_hash=payload["terminal_mask_hash"],
        acquisition_receipt_hashes=tuple(lineage),
        extraction_state_hash=payload["extraction_state_hash"],
        extraction_calibration_hash=payload["extraction_calibration_hash"],
        validation_independent=payload["validation_independent"],
        terminal_rows_visible=payload["terminal_rows_visible"],
        schema=payload["schema"],
    )
    if payload["closure_hash"] != closure.closure_hash:
        raise ValueError("terminal extraction closure file failed its content hash")
    return closure


def _terminal_extraction_closure_file_payload(
    *,
    finalizer_hash: str,
    closure: V1TerminalExtractionClosure,
) -> dict[str, Any]:
    return {
        "schema": _TERMINAL_EXTRACTION_CLOSURE_FILE_SCHEMA,
        "terminal_extraction_finalizer_hash": finalizer_hash,
        "closure_hash": closure.closure_hash,
        "closure": closure.to_json_payload(),
    }


def _read_terminal_extraction_closure_file(
    path: Path,
    *,
    finalizer_hash: str,
) -> V1TerminalExtractionClosure:
    _require_nonsymlink_regular_file(
        path,
        label="terminal extraction closure file",
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("terminal extraction closure file is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("terminal extraction closure file must contain a mapping")
    if set(payload) != {
        "schema",
        "terminal_extraction_finalizer_hash",
        "closure_hash",
        "closure",
    }:
        raise ValueError("terminal extraction closure file has unexpected envelope fields")
    if payload["schema"] != _TERMINAL_EXTRACTION_CLOSURE_FILE_SCHEMA:
        raise ValueError("terminal extraction closure file has another schema")
    if payload["terminal_extraction_finalizer_hash"] != finalizer_hash:
        raise ValueError("terminal extraction closure names another sealed finalizer")
    raw_closure = payload["closure"]
    if not isinstance(raw_closure, Mapping):
        raise TypeError("terminal extraction closure payload must be a mapping")
    closure = _terminal_extraction_closure_from_json(raw_closure)
    if payload["closure_hash"] != closure.closure_hash:
        raise ValueError("terminal extraction closure envelope failed its content hash")
    return closure


def _assert_closure_matches_terminal_freeze(
    state: V1RuntimeState,
    closure: V1TerminalExtractionClosure,
) -> None:
    freeze = state.terminal_freeze
    if freeze is None:
        return
    expected = {
        "controller_snapshot_hash": closure.controller_snapshot_hash,
        "terminal_mask_hash": closure.terminal_mask_hash,
        "acquisition_receipt_hashes": closure.acquisition_receipt_hashes,
        "extraction_state_hash": closure.extraction_state_hash,
        "extraction_calibration_hash": closure.extraction_calibration_hash,
    }
    for name, value in expected.items():
        if getattr(freeze, name) != value:
            raise ValueError(f"resumed terminal freeze disagrees with closure on {name}")


def _materialize_resumed_terminal_extraction_closure(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    source: Path,
    destination: Path,
) -> None:
    """Validate/import a finalizer-backed closure before checkpoint publication."""

    finalizer = bundle.terminal_extraction_finalizer
    if finalizer is None:
        return
    finalizer_hash = finalizer.identity_hash
    source_exists = _path_lexists(source)
    destination_exists = _path_lexists(destination)
    source_closure: V1TerminalExtractionClosure | None = None
    destination_closure: V1TerminalExtractionClosure | None = None
    if source_exists:
        source_closure = _read_terminal_extraction_closure_file(
            source,
            finalizer_hash=finalizer_hash,
        )
        _validate_terminal_extraction_closure(state.controller, source_closure)
        _assert_closure_matches_terminal_freeze(state, source_closure)
    if destination_exists:
        destination_closure = _read_terminal_extraction_closure_file(
            destination,
            finalizer_hash=finalizer_hash,
        )
        _validate_terminal_extraction_closure(state.controller, destination_closure)
        _assert_closure_matches_terminal_freeze(state, destination_closure)
    if (
        source_closure is not None
        and destination_closure is not None
        and source_closure.closure_hash != destination_closure.closure_hash
    ):
        raise ValueError("resume paths contain different terminal extraction closures")
    if source_closure is not None:
        _atomic_materialize_exact_file(
            source,
            destination,
            label="terminal extraction closure",
        )
        destination_closure = source_closure
    if state.terminal_freeze is not None and destination_closure is None:
        raise ValueError(
            "finalizer-backed terminal freeze lacks its durable extraction closure"
        )


def _resolve_terminal_extraction_closure(
    *,
    bundle: V1ProductionRuntimeBundle,
    state: V1RuntimeState,
    destination: Path,
    resume_checkpoint: str | os.PathLike[str] | None,
) -> V1TerminalExtractionClosure:
    controller = state.controller
    if bundle.terminal_extraction_finalizer is None:
        legacy_closure = _legacy_terminal_extraction_closure(bundle, controller)
        _assert_closure_matches_terminal_freeze(state, legacy_closure)
        return legacy_closure

    finalizer = bundle.terminal_extraction_finalizer
    finalizer_hash = finalizer.identity_hash
    source = (
        None
        if resume_checkpoint is None
        else Path(resume_checkpoint).resolve().parent
        / _TERMINAL_EXTRACTION_CLOSURE_FILENAME
    )
    candidate_paths: list[Path] = []
    for candidate in (destination, source):
        if candidate is not None and candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in candidate_paths:
                candidate_paths.append(resolved)

    closure: V1TerminalExtractionClosure | None = None
    loaded_durable_closure = False
    for path in candidate_paths:
        observed = _read_terminal_extraction_closure_file(
            path,
            finalizer_hash=finalizer_hash,
        )
        _validate_terminal_extraction_closure(controller, observed)
        if closure is not None and observed.closure_hash != closure.closure_hash:
            raise ValueError("resume paths contain different terminal extraction closures")
        closure = observed
        loaded_durable_closure = True

    if closure is None:
        if state.terminal_freeze is not None:
            raise ValueError(
                "finalizer-backed terminal freeze lacks its durable extraction closure"
            )
        produced = finalizer(controller)
        _validate_terminal_extraction_closure(controller, produced)
        closure = produced
        _atomic_write_json(
            destination,
            _terminal_extraction_closure_file_payload(
                finalizer_hash=finalizer_hash,
                closure=closure,
            ),
        )
    else:
        if loaded_durable_closure and state.terminal_validation is None:
            reinstalled = finalizer(controller)
            _validate_terminal_extraction_closure(controller, reinstalled)
            if reinstalled.closure_hash != closure.closure_hash:
                raise ValueError(
                    "reinstalled terminal extraction closure differs from durable closure"
                )
        if not destination.is_file():
            _atomic_write_json(
                destination,
                _terminal_extraction_closure_file_payload(
                    finalizer_hash=finalizer_hash,
                    closure=closure,
                ),
            )

    _assert_closure_matches_terminal_freeze(state, closure)
    return closure


def _validate_operator_export_source(state, payload):
    from .modern_v1_operator_run_binding import validate_config_binding
    binding = validate_config_binding(state.config.metadata, state.config.source_seal_hash)
    if binding is None or content_hash(payload['source']) != content_hash(binding['source']):
        raise ValueError('production output circuit source differs from the runtime binding')


def _bound_operator_factory(bundle):
    """Verify the nominal actual producer, provider and persistent run binding."""
    if bundle.manifest.proposal_protocol!='operator_binary_d_c_v1':return None
    from .modern_v1_operator_factory import OperatorAcquisitionFactory
    from .modern_v1_operator_run_binding import validate_bound_operator_sources
    seal=bundle.acquisition_factory
    if type(seal) is not V1ProductionCallableSeal or type(seal.callable_value) is not OperatorAcquisitionFactory:
        raise TypeError('operator production protocol requires the exact actual acquisition factory')
    factory=seal.callable_value
    if (seal.configuration_hash!=factory._coordinates()
            or _active_paired_oracle_evaluator(bundle.provider) is not factory.native_evaluator):
        raise ValueError('operator production producer/provider configuration differs')
    validate_bound_operator_sources(bundle.runtime_config,factory._model.sources,factory=factory)
    return factory


def run_v1_production_bundle(
    bundle: V1ProductionRuntimeBundle,
    *,
    output_directory: str | os.PathLike[str],
    resume_checkpoint: str | os.PathLike[str] | None = None,
    trusted_local_resume: bool = False,
    maximum_transitions: int = 100000,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> V1ProductionRunArtifacts:
    """Execute one admitted bundle with durable pre-outcome sampling checkpoints.

    The checkpoint is rewritten after every logical transition.  In particular,
    a system-entropy sampling plan is on disk before the paired oracle is called.
    A resumed terminal state is idempotent and never triggers a second reveal.
    """

    if type(bundle) is not V1ProductionRuntimeBundle:
        raise TypeError("production execution requires an exact runtime bundle")
    operator_factory=_bound_operator_factory(bundle)
    from .modern_v1_operator_run_binding import KEY
    if KEY in bundle.runtime_config.metadata and operator_factory is None:
        raise ValueError('bound operator runtime requires the actual operator production protocol')
    if operator_factory is not None and not bundle.has_paired_bce_oracle_protocol:
        raise ValueError('operator production requires the retained paired-BCE oracle protocol')
    transition_bound = _positive(maximum_transitions, "maximum_transitions")
    state: V1RuntimeState | None = None
    transitions: int | None = None

    def emit_progress(phase: str, **details: Any) -> None:
        if progress_callback is None:
            return
        if type(phase) is not str or not phase:
            raise ValueError("production progress phase must be a non-empty string")
        progress_callback(
            {
                "phase": phase,
                "controller_phase": None if state is None else state.controller.phase,
                "transitions": transitions,
                **details,
            }
        )

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "modern_v1_runtime.ckpt"
    index_path = root / "modern_v1_run_index.json"
    paired_bce_row_store_root = root / "paired_bce_external_rows"
    production_binding_path = root / _PRODUCTION_RESUME_BINDING_FILENAME
    terminal_extraction_closure_path = (
        root / _TERMINAL_EXTRACTION_CLOSURE_FILENAME
    )
    circuit_export_path = root / 'modern_v1_circuit.json'
    circuit_export = None
    terminal_family_matrix_root = root / (
        f"{checkpoint_path.name}.terminal_validation"
    )
    terminal_comparison_matrix_root = root / (
        f"{checkpoint_path.name}.terminal_comparison"
    )
    pareto_milestone_archive_root = root / (
        f"{checkpoint_path.name}.pareto_milestones"
    )
    paired_bce_enabled = bundle.has_paired_bce_oracle_protocol
    resumed_receipt: V1RuntimeCheckpointReceipt | None = None
    terminal_family_request_checkpoint: V1RuntimeCheckpointReceipt | None = None
    emit_progress(
        "production_start",
        output_directory=str(root),
        resume_checkpoint=None if resume_checkpoint is None else str(resume_checkpoint),
        maximum_transitions=transition_bound,
    )
    if resume_checkpoint is None:
        if (
            _path_lexists(index_path)
            or _path_lexists(checkpoint_path)
            or _path_lexists(production_binding_path)
            or _path_lexists(terminal_extraction_closure_path)
            or (operator_factory is not None and _path_lexists(circuit_export_path))
            or _path_lexists(paired_bce_row_store_root)
            or (
                bundle.has_terminal_validation_family_protocol
                and (
                    _path_lexists(terminal_family_matrix_root)
                    or _path_lexists(terminal_comparison_matrix_root)
                    or (
                        bundle.has_legacy_terminal_validation_family_protocol
                        and _path_lexists(pareto_milestone_archive_root)
                    )
                )
            )
        ):
            raise FileExistsError(
                "production output already contains run state; use explicit trusted resume"
            )
        _publish_or_validate_production_resume_binding(
            production_binding_path,
            bundle=bundle,
        )
        emit_progress("initial_runtime_start")
        state = _initial_runtime(bundle)
        state, _ = _checkpoint_progress(checkpoint_path, state)
        emit_progress("initial_runtime_checkpoint_done")
    else:
        raw_resume_path = Path(resume_checkpoint).expanduser()
        _require_nonsymlink_regular_file(
            raw_resume_path,
            label="production resume checkpoint",
        )
        resume_path = raw_resume_path.resolve()
        source_root = resume_path.parent
        destination_checkpoint_exists = _path_lexists(checkpoint_path)
        destination_is_resume_checkpoint = bool(
            destination_checkpoint_exists
            and checkpoint_path.resolve() == resume_path
        )
        if destination_checkpoint_exists and not destination_is_resume_checkpoint:
            raise FileExistsError(
                "production resume destination already contains another checkpoint"
            )
        if _path_lexists(index_path) and not destination_is_resume_checkpoint:
            raise FileExistsError(
                "production resume destination already contains a completed run index"
            )

        source_binding_path = source_root / _PRODUCTION_RESUME_BINDING_FILENAME
        if _path_lexists(source_binding_path):
            _validate_production_resume_binding(source_binding_path, bundle=bundle)
        elif bundle.has_staged_proposal_protocol or operator_factory is not None:
            raise ValueError(
                "production resume lacks its sealed production binding"
            )
        if _path_lexists(production_binding_path):
            _validate_production_resume_binding(
                production_binding_path,
                bundle=bundle,
            )
        state, resumed_receipt = read_v1_runtime_checkpoint(
            resume_path,
            trusted_local=trusted_local_resume,
            operator_sources=None if operator_factory is None else operator_factory._model.sources,
            restore_operator_mask=operator_factory is not None,
        )
        emit_progress(
            "resume_checkpoint_read",
            checkpoint_path=str(resume_path),
            checkpoint_receipt_hash=resumed_receipt.receipt_hash,
        )
        _validate_resumed_state(bundle, state)
        if bundle.has_terminal_validation_family_protocol:
            emit_progress("resume_terminal_family_dependencies_start")
        if bundle.has_legacy_terminal_validation_family_protocol:
            _materialize_resumed_pareto_milestone_archive(
                state=state,
                source_checkpoint=resume_path,
                destination_checkpoint=checkpoint_path,
            )
        _restore_resumed_proposal_backend_state(bundle, state)
        _restore_resumed_provider_process_state(bundle, state.controller,runtime_state=state)
        # The validator above permits changed physical packing only for a
        # bundle without performance accounting.  Performance-tracked bundles
        # require the same physical config across restart so one elapsed-time
        # receipt cannot silently mix execution regimes.  Logical config and
        # the sealed entropy tape are always exact.
        state = replace(state, config=bundle.runtime_config)

        if paired_bce_enabled:
            source_row_store_root = source_root / "paired_bce_external_rows"
            emit_progress("resume_paired_bce_source_validation_start")
            _validate_paired_bce_external_replay_if_needed(
                bundle=bundle,
                state=state,
                row_store_root=source_row_store_root,
            )
            emit_progress("resume_paired_bce_source_validation_done")
            emit_progress("resume_paired_bce_materialization_start")
            _materialize_paired_bce_row_store(
                source_root=source_row_store_root,
                destination_root=paired_bce_row_store_root,
            )
            _validate_paired_bce_external_replay_if_needed(
                bundle=bundle,
                state=state,
                row_store_root=paired_bce_row_store_root,
            )
            emit_progress("resume_paired_bce_materialization_done")

        emit_progress("resume_terminal_closure_materialization_start")
        _materialize_resumed_terminal_extraction_closure(
            bundle=bundle,
            state=state,
            source=source_root / _TERMINAL_EXTRACTION_CLOSURE_FILENAME,
            destination=terminal_extraction_closure_path,
        )
        emit_progress("resume_terminal_closure_materialization_done")
        _publish_or_validate_production_resume_binding(
            production_binding_path,
            bundle=bundle,
        )
        legacy_family_request_without_observation = bool(
            state.terminal_validation_family_request is not None
            and state.terminal_validation_family_observation is None
        )
        comparison_family_request_without_observation = bool(
            state.terminal_comparison_family_request is not None
            and state.terminal_comparison_family_observation is None
        )
        if (
            legacy_family_request_without_observation
            or comparison_family_request_without_observation
        ):
            if destination_is_resume_checkpoint:
                # The restored receipt is the exact already-durable request
                # checkpoint.  Rewriting it after a crash-published matrix
                # would violate the runtime's outcome-absence checkpoint gate.
                terminal_family_request_checkpoint = resumed_receipt
            else:
                # Relocation publishes the request while the destination
                # sidecar is still absent, then imports opaque sidecar bytes.
                state, terminal_family_request_checkpoint = _checkpoint_progress(
                    checkpoint_path,
                    state,
                )
            if legacy_family_request_without_observation:
                _materialize_resumed_terminal_family_matrix(
                    state=state,
                    source_checkpoint=resume_path,
                    destination_checkpoint=checkpoint_path,
                )
            else:
                _materialize_resumed_terminal_comparison_family_matrix_v2(
                    state=state,
                    source_checkpoint=resume_path,
                    destination_checkpoint=checkpoint_path,
                )
        else:
            if bundle.has_legacy_terminal_validation_family_protocol:
                _materialize_resumed_terminal_family_matrix(
                    state=state,
                    source_checkpoint=resume_path,
                    destination_checkpoint=checkpoint_path,
                )
            elif bundle.has_terminal_comparison_family_protocol_v2:
                _materialize_resumed_terminal_comparison_family_matrix_v2(
                    state=state,
                    source_checkpoint=resume_path,
                    destination_checkpoint=checkpoint_path,
                )
            state, _ = _checkpoint_progress(checkpoint_path, state)
        if bundle.has_terminal_validation_family_protocol:
            emit_progress("resume_terminal_family_dependencies_done")
        emit_progress("resume_checkpoint_republished")

    paired_calibration_freeze_factory = (
        _production_paired_bce_calibration_freeze_factory(bundle)
        if paired_bce_enabled
        else None
    )
    paired_calibration_horizon_skip_factory = (
        _production_paired_bce_calibration_horizon_skip_factory(bundle)
        if paired_bce_enabled
        else None
    )
    paired_registration_factory = (
        _production_paired_bce_registration_factory(bundle)
        if paired_bce_enabled
        else None
    )
    order_epoch_cache = _ProductionPairedBCEOrderEpochCache()
    try:
        paired_chunk_factory = (
            _production_paired_bce_chunk_factory(
                bundle, order_epoch_cache=order_epoch_cache,
            )
            if paired_bce_enabled
            else None
        )
        paired_betting_replay_factory = (
            _production_paired_bce_betting_replay_factory(
                bundle,
                paired_bce_row_store_root,
            )
            if paired_bce_enabled
            else None
        )
        if paired_bce_enabled:
            emit_progress("paired_bce_external_replay_validation_start")
            _validate_paired_bce_external_replay_if_needed(
                bundle=bundle,
                state=state,
                row_store_root=paired_bce_row_store_root,
            )
            emit_progress("paired_bce_external_replay_validation_done")

        new_decisions: list[ProductionDecision] = []
        transitions = 0
        terminal_accept_retirement_invoked = False
        while state.controller.phase != "stopped":
            emit_progress(
                "transition_start",
                controller_attempt_count=len(state.controller.attempt_receipts),
                controller_transition_count=len(state.controller.transition_receipts),
                proposal_progress_present=state.proposal_progress is not None,
                pending_proposal_acquisition=state.pending_proposal_acquisition is not None,
                pending_paired_bce_chunk=state.pending_paired_bce_chunk is not None,
            )
            if v1_runtime_proposal_order_indeterminate(state):
                # This may be a resumed refusal.  Re-publication to the selected
                # output is complete before the typed terminal result is exposed.
                state, _ = _checkpoint_progress(checkpoint_path, state)
                emit_progress("proposal_order_indeterminate_checkpoint_done")
                require_v1_runtime_proposal_order_resolved(state)
            if transitions >= transition_bound:
                raise RuntimeError("modern-v1 production runtime exceeded its orchestration bound")
            if state.controller.phase == "candidate_active":
                active = state.controller.active_candidate
                if (
                    active is not None
                    and active.paired_bce_registration is not None
                ):
                    if not paired_bce_enabled or paired_chunk_factory is None:
                        raise PermissionError(
                            "active paired-BCE candidate lacks production paired-BCE hooks"
                        )
                    if state.pending_paired_bce_chunk is None:
                        emit_progress("paired_bce_chunk_prepare_start")
                        state = prepare_v1_runtime_paired_bce_chunk(
                            state,
                            chunk_factory=paired_chunk_factory,
                        )
                        state, _ = _checkpoint_progress(checkpoint_path, state)
                        emit_progress("paired_bce_chunk_prepare_checkpoint_done")
                    emit_progress("paired_bce_chunk_execute_start")
                    state = execute_v1_runtime_pending_paired_bce_chunk(
                        state,
                        executor=_production_paired_bce_chunk_executor(
                            bundle,
                            state,
                            paired_bce_row_store_root,
                            order_epoch_cache=order_epoch_cache,
                        ),
                    )
                    state, _ = _checkpoint_progress(checkpoint_path, state)
                    emit_progress("paired_bce_chunk_execute_checkpoint_done")
                    emit_progress("paired_bce_oracle_decide_start")
                    state, paired_decision = decide_v1_runtime_paired_bce_oracle(
                        state,
                        betting_replay_factory=paired_betting_replay_factory,
                    )
                    emit_progress(
                        "paired_bce_oracle_decide_done",
                        paired_bce_status=paired_decision.status,
                    )
                    decision = (
                        None
                        if paired_decision.status == "acquire_more"
                        else paired_decision
                    )
                elif paired_bce_enabled:
                    raise PermissionError(
                        "paired-BCE production refuses legacy staged-alpha candidate oracles"
                    )
                else:
                    if state.pending_look is None:
                        emit_progress("legacy_look_prepare_start")
                        state = prepare_v1_runtime_look(state)
                        # Required durability joint: entropy draw exists on disk pre-outcome.
                        state, _ = _checkpoint_progress(checkpoint_path, state)
                        emit_progress("legacy_look_prepare_checkpoint_done")
                    emit_progress("legacy_look_execute_start")
                    state, decision = execute_v1_runtime_pending_look(
                        state,
                        provider=bundle.provider,
                    )
            elif paired_bce_enabled and state.controller.phase == "queue_ready":
                if (
                    paired_calibration_freeze_factory is None
                    or paired_registration_factory is None
                ):  # pragma: no cover - paired_bce_enabled construction invariant.
                    raise RuntimeError("paired-BCE production factories were not bound")
                state, calibration_skip_decision = (
                    advance_v1_runtime_paired_bce_candidate_freeze(
                        state,
                        threshold_factory=bundle.threshold_factory,
                        sampling_law_factory=bundle.sampling_law_factory,
                        calibration_freeze_factory=(
                            paired_calibration_freeze_factory
                        ),
                        paired_bce_registration_factory=(
                            paired_registration_factory
                        ),
                        calibration_horizon_skip_factory=(
                            paired_calibration_horizon_skip_factory
                        ),
                    )
                )
                if calibration_skip_decision is None:
                    emit_progress("paired_bce_candidate_freeze_done")
                else:
                    emit_progress(
                        "paired_bce_candidate_calibration_horizon_skip_done",
                        paired_bce_status=calibration_skip_decision.status,
                        candidate_attempt_index=(
                            calibration_skip_decision.candidate_attempt_index
                        ),
                        causal_horizon_refusal_receipt_hash=(
                            calibration_skip_decision.refusal_receipt_hash
                        ),
                        attempt_alpha_retired=True,
                        unused_alpha_recycled=False,
                    )
                decision = calibration_skip_decision
            elif bundle.has_staged_proposal_protocol and (
                state.controller.phase == "needs_acquisition"
                or state.proposal_progress is not None
                or state.pending_proposal_acquisition is not None
            ):
                emit_progress("staged_proposal_step_start")
                state = _advance_v1_production_staged_proposal_once(
                    state,
                    bundle=bundle,
                )
                emit_progress(
                    "staged_proposal_step_done",
                    controller_attempt_count=len(state.controller.attempt_receipts),
                    controller_transition_count=len(state.controller.transition_receipts),
                    proposal_progress_present=state.proposal_progress is not None,
                    pending_proposal_acquisition=state.pending_proposal_acquisition is not None,
                )
                decision = None
            elif bundle.has_staged_proposal_protocol:
                if paired_bce_enabled:
                    raise RuntimeError(
                        "paired-BCE staged-proposal production reached legacy nonproposal oracle"
                    )
                state, decision = advance_v1_runtime_nonproposal_once(
                    state,
                    threshold_factory=bundle.threshold_factory,
                    sampling_law_factory=bundle.sampling_law_factory,
                    provider=bundle.provider,
                )
                emit_progress("staged_nonproposal_step_done")
            else:
                acquisition_factory = bundle.acquisition_factory
                if acquisition_factory is None:  # pragma: no cover - bundle invariant
                    raise RuntimeError("legacy production bundle lost its acquisition factory")
                if paired_bce_enabled and state.controller.phase != "needs_acquisition":
                    raise RuntimeError(
                        "paired-BCE production reached an unexpected non-acquisition phase"
                    )
                state, decision = advance_v1_runtime_once(
                    state,
                    acquisition_factory=operator_factory if operator_factory is not None else acquisition_factory,
                    threshold_factory=bundle.threshold_factory,
                    sampling_law_factory=bundle.sampling_law_factory,
                    provider=bundle.provider,
                )
                emit_progress("operator_controller_step_done" if operator_factory is not None else "legacy_controller_step_done")
            transitions += 1
            if decision is not None:
                new_decisions.append(decision)
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress(
                "transition_checkpoint_done",
                controller_attempt_count=len(state.controller.attempt_receipts),
                controller_transition_count=len(state.controller.transition_receipts),
                accepted_decision_count=len(new_decisions),
            )
            if _retire_terminal_accepted_generation_state(bundle, state, decision):
                if terminal_accept_retirement_invoked:  # pragma: no cover - loop invariant
                    raise RuntimeError(
                        "terminal accepted generation retirement was invoked twice"
                    )
                terminal_accept_retirement_invoked = True
                emit_progress(
                    "terminal_accepted_generation_retirement_done",
                    controller_attempt_count=len(state.controller.attempt_receipts),
                    controller_transition_count=len(state.controller.transition_receipts),
                )
            if v1_runtime_proposal_order_indeterminate(state):
                require_v1_runtime_proposal_order_resolved(state)

    finally:
        order_epoch_cache.close()

    # Terminal calibration/extraction can itself observe mask-specific
    # information.  Keep that surface closed unless the controller reached a
    # registered scientific endpoint; finite compute/statistical caps remain
    # resumable, validation-independent indeterminate checkpoints.
    require_v1_runtime_scientific_endpoint(state)
    emit_progress("terminal_extraction_closure_start")
    terminal_extraction_closure = _resolve_terminal_extraction_closure(
        bundle=bundle,
        state=state,
        destination=terminal_extraction_closure_path,
        resume_checkpoint=resume_checkpoint,
    )
    emit_progress("terminal_extraction_closure_done")
    if state.terminal_freeze is None:
        emit_progress("terminal_freeze_start")
        state = freeze_v1_terminal(
            state,
            extraction_state_hash=(
                terminal_extraction_closure.extraction_state_hash
            ),
            extraction_calibration_hash=(
                terminal_extraction_closure.extraction_calibration_hash
            ),
        )
        state, _ = _checkpoint_progress(checkpoint_path, state)
        emit_progress("terminal_freeze_checkpoint_done")
    if operator_factory is not None:
        from .modern_v1_operator_export import publish_operator_circuit_for_run
        emit_progress('operator_circuit_export_start')
        circuit_export = publish_operator_circuit_for_run(state=state, factory=operator_factory,
            destination=circuit_export_path,
            source=None if resume_checkpoint is None else Path(resume_checkpoint).expanduser().resolve().parent / circuit_export_path.name)
        emit_progress('operator_circuit_export_done',
            circuit_export_sha256=None if circuit_export is None else circuit_export.sha256)
    if bundle.terminal_validation_family_design is None:
        if state.terminal_validation_family_freeze is not None:
            raise PermissionError(
                "legacy terminal mode cannot enter terminal-family validation"
            )
        if state.terminal_validation is None:
            emit_progress("terminal_validation_start")
            state = reveal_v1_terminal_once(
                state,
                validator=bundle.terminal_validator,
            )
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_validation_checkpoint_done")
    elif bundle.has_legacy_terminal_validation_family_protocol:
        if state.terminal_validation is not None:
            raise PermissionError(
                "terminal-family mode cannot enter legacy terminal validation"
            )
        material = _derive_terminal_validation_family_material(
            bundle=bundle,
            state=state,
            checkpoint_path=checkpoint_path,
            terminal_extraction_closure=terminal_extraction_closure,
        )
        if state.terminal_validation_family_freeze is None:
            emit_progress("terminal_validation_family_freeze_start")
            state = freeze_v1_terminal_validation_family(
                state,
                family_freeze=material.family_freeze,
            )
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_validation_family_freeze_checkpoint_done")
        elif (
            state.terminal_validation_family_freeze.receipt_hash
            != material.family_freeze.receipt_hash
        ):
            raise ValueError(
                "resumed terminal-family freeze differs from exact train-only material"
            )
        if state.terminal_validation_family_request is None:
            emit_progress("terminal_validation_family_request_start")
            state = prepare_v1_terminal_validation_family_request(
                state,
                observation_artifact_relative_path=(
                    f"{checkpoint_path.name}.terminal_validation/"
                    f"{_TERMINAL_VALIDATION_FAMILY_MATRIX_FILENAME}"
                ),
            )
            state, request_checkpoint = _checkpoint_progress(
                checkpoint_path,
                state,
            )
            emit_progress("terminal_validation_family_request_checkpoint_done")
        else:
            request_checkpoint = terminal_family_request_checkpoint
        if state.terminal_validation_family_observation is None:
            if request_checkpoint is None:
                raise PermissionError(
                    "terminal-family reveal lacks the exact durable request "
                    "checkpoint receipt"
                )
            request = state.terminal_validation_family_request
            if request is None:  # pragma: no cover - construction invariant.
                raise RuntimeError("terminal-family request disappeared before reveal")
            execution = _terminal_family_execution(
                request=request,
                material=material,
                request_checkpoint_receipt=request_checkpoint,
            )
            emit_progress("terminal_validation_family_reveal_start")
            state = reveal_v1_terminal_validation_family_once(
                state,
                validator=_BoundProductionTerminalFamilyValidator(
                    validator=bundle.terminal_validator,
                    execution=execution,
                ),
                artifact_root=root,
                request_checkpoint_receipt=request_checkpoint,
            )
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_validation_family_observation_checkpoint_done")
        if state.terminal_validation_family is None:
            state = complete_v1_terminal_validation_family(state)
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_validation_family_completion_checkpoint_done")
    else:
        if not bundle.has_terminal_comparison_family_protocol_v2:
            raise RuntimeError("production terminal-family mode is unsupported")
        if (
            state.terminal_validation is not None
            or state.terminal_validation_family is not None
        ):
            raise PermissionError(
                "terminal-comparison v2 cannot enter a legacy terminal branch"
            )
        material_v2 = _derive_terminal_comparison_family_material_v2(
            bundle=bundle,
            state=state,
            terminal_extraction_closure=terminal_extraction_closure,
        )
        if state.terminal_comparison_family_freeze is None:
            emit_progress("terminal_comparison_family_freeze_start")
            state = freeze_v1_terminal_comparison_family_v2(
                state,
                family_freeze=material_v2.family_freeze,
            )
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_comparison_family_freeze_checkpoint_done")
        elif (
            state.terminal_comparison_family_freeze.receipt_hash
            != material_v2.family_freeze.receipt_hash
        ):
            raise ValueError(
                "resumed terminal-comparison freeze differs from exact train-only material"
            )
        if state.terminal_comparison_family_request is None:
            emit_progress("terminal_comparison_family_request_start")
            state = prepare_v1_terminal_comparison_family_request_v2(
                state,
                observation_artifact_relative_path=(
                    f"{checkpoint_path.name}.terminal_comparison/"
                    f"{_TERMINAL_COMPARISON_FAMILY_MATRIX_FILENAME_V2}"
                ),
                observation_artifact_schema=(
                    V1_TERMINAL_COMPARISON_FAMILY_MATRIX_ARTIFACT_SCHEMA_V2
                ),
            )
            state, request_checkpoint = _checkpoint_progress(
                checkpoint_path,
                state,
            )
            emit_progress("terminal_comparison_family_request_checkpoint_done")
        else:
            request_checkpoint = terminal_family_request_checkpoint
        if state.terminal_comparison_family_observation is None:
            if request_checkpoint is None:
                raise PermissionError(
                    "terminal-comparison reveal lacks the exact durable request "
                    "checkpoint receipt"
                )
            request_v2 = state.terminal_comparison_family_request
            if request_v2 is None:  # pragma: no cover - construction invariant.
                raise RuntimeError(
                    "terminal-comparison request disappeared before reveal"
                )
            execution_v2 = _terminal_comparison_family_execution_v2(
                request=request_v2,
                material=material_v2,
                request_checkpoint_receipt=request_checkpoint,
            )
            emit_progress("terminal_comparison_family_reveal_start")
            state = reveal_v1_terminal_comparison_family_once_v2(
                state,
                validator=_BoundProductionTerminalComparisonFamilyValidatorV2(
                    validator=bundle.terminal_validator,
                    execution=execution_v2,
                ),
                artifact_root=root,
                request_checkpoint_receipt=request_checkpoint,
            )
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_comparison_family_observation_checkpoint_done")
        if state.terminal_comparison_family is None:
            state = complete_v1_terminal_comparison_family_v2(state)
            state, _ = _checkpoint_progress(checkpoint_path, state)
            emit_progress("terminal_comparison_family_completion_checkpoint_done")
    if bundle.performance_receipt_factory is not None and state.performance is None:
        emit_progress("performance_receipt_start")
        performance = bundle.performance_receipt_factory(state)
        state = attach_v1_performance_receipt(state, performance)
        state, _ = _checkpoint_progress(checkpoint_path, state)
        emit_progress("performance_receipt_checkpoint_done")
    state = complete_v1_runtime(state)
    state, final_checkpoint = _checkpoint_progress(checkpoint_path, state)
    emit_progress(
        "production_complete_checkpoint_done",
        final_checkpoint_receipt_hash=final_checkpoint.receipt_hash,
    )

    completion = state.completion
    terminal_freeze = state.terminal_freeze
    terminal_validation = (
        state.terminal_validation
        if state.terminal_validation is not None
        else (
            state.terminal_validation_family
            if state.terminal_validation_family is not None
            else state.terminal_comparison_family
        )
    )
    if completion is None or terminal_freeze is None or terminal_validation is None:
        raise RuntimeError("production runtime failed to materialize terminal artifacts")
    _validate_completed_proposal_closure_ledger(
        state,
        proposal_protocol=bundle.manifest.proposal_protocol,
        proposal_backend_identity_hash=(
            None
            if bundle.proposal_backend_seal is None
            else bundle.proposal_backend_seal.identity_hash
        ),
    )
    run_index = V1ProductionRunIndex(
        production_manifest_hash=bundle.manifest.manifest_hash,
        production_bundle_hash=bundle.bundle_hash,
        campaign_contract_hash=bundle.binding.contract.campaign_contract_hash,
        campaign_binding_hash=bundle.binding.binding_hash,
        controller_run_id=state.controller.controller_run_id,
        proposal_protocol=bundle.manifest.proposal_protocol,
        proposal_backend_identity_hash=(
            None
            if bundle.proposal_backend_seal is None
            else bundle.proposal_backend_seal.identity_hash
        ),
        final_runtime_state_hash=state.runtime_state_hash,
        final_controller_snapshot_hash=state.controller.snapshot_receipt().receipt_hash,
        terminal_extraction_closure_hash=(
            terminal_extraction_closure.closure_hash
        ),
        terminal_freeze_hash=terminal_freeze.receipt_hash,
        terminal_validation_hash=terminal_validation.receipt_hash,
        completion_receipt_hash=completion.receipt_hash,
        terminal_mask_hash=state.controller.current_mask.mask_hash,
        acquisition_receipt_hashes=state.controller.acquisition_receipt_hashes,
        attempt_receipt_hashes=state.controller.attempt_receipt_hashes,
        transition_receipt_hashes=state.controller.transition_receipt_hashes,
        decision_receipt_hashes=tuple(
            receipt.decision_receipt_hash for receipt in state.controller.attempt_receipts
        ),
        completed_proposal_closure_hashes=tuple(
            receipt.receipt_hash
            for receipt in state.completed_proposal_closures
        ),
        final_checkpoint_payload_sha256=final_checkpoint.payload_sha256,
        performance_receipt_hash=(
            None if state.performance is None else state.performance.receipt_hash
        ),
        resumed_from_checkpoint_payload_sha256=(
            None if resumed_receipt is None else resumed_receipt.payload_sha256
        ),
    )
    _atomic_write_json(index_path, run_index.to_json_payload())
    output_receipt = V1ProductionOutputReceipt(
        output_directory=str(root),
        production_binding_path=str(production_binding_path),
        production_binding_sha256=_sha256_file(production_binding_path),
        index_path=str(index_path),
        index_sha256=_sha256_file(index_path),
        checkpoint_path=str(checkpoint_path),
        checkpoint_payload_sha256=final_checkpoint.payload_sha256,
        circuit_export_path=None if circuit_export is None else str(circuit_export_path),
        circuit_export_sha256=None if circuit_export is None else circuit_export.sha256,
        run_index_hash=run_index.index_hash,
        terminal_extraction_closure_path=(
            None
            if bundle.terminal_extraction_finalizer is None
            else str(terminal_extraction_closure_path)
        ),
        terminal_extraction_closure_sha256=(
            None
            if bundle.terminal_extraction_finalizer is None
            else _sha256_file(terminal_extraction_closure_path)
        ),
    )
    return V1ProductionRunArtifacts(
        state=state,
        new_decisions=tuple(new_decisions),
        terminal_extraction_closure=terminal_extraction_closure,
        final_checkpoint=final_checkpoint,
        run_index=run_index,
        output_receipt=output_receipt,
        resumed_checkpoint=resumed_receipt,
    )


__all__ = [
    "BackendKind",
    "PerformanceReceiptFactory",
    "ProposalProtocolKind",
    "ProductionCallableRole",
    "TrainOnlyDecisionQualificationStatus",
    "TerminalExtractionFinalizer",
    "V1ProductionCallableSeal",
    "V1ProductionPairedOracleSeal",
    "V1ProductionProposalQualificationArtifacts",
    "V1ProductionProposalQualificationReceipt",
    "V1ProductionProposalBackendSeal",
    "V1ProductionOutputReceipt",
    "V1ProductionRunArtifacts",
    "V1ProductionRunIndex",
    "V1ProductionRuntimeBundle",
    "V1ProductionRuntimeManifest",
    "V1ProductionTerminalValidationFamilyDesign",
    "V1ProductionTerminalValidationFamilyExecution",
    "V1ProductionTerminalComparisonFamilyDesignV2",
    "V1ProductionTerminalComparisonFamilyExecutionV2",
    "V1TrainOnlyDecisionQualificationArtifacts",
    "V1TrainOnlyDecisionQualificationBinding",
    "V1TrainOnlyDecisionQualificationReceipt",
    "V1TerminalExtractionClosure",
    "run_v1_production_bundle",
    "run_v1_production_proposal_qualification",
    "run_v1_production_train_only_decision_qualification",
    "seal_v1_paired_oracle_provider",
    "seal_v1_production_callable",
    "seal_v1_production_proposal_backend",
]
