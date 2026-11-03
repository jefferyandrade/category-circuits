"""Fail-closed framed inspection for historical and future modern-v1 checkpoints.

This module is deliberately separate from the live runtime loader.  Its first
job is byte integrity: one descriptor is opened for each artifact, framing and
strict JSON are checked, the exact payload is hashed through that descriptor,
and every referenced sidecar is verified before *any* pickle opcode is
interpreted.  Its second job is inspect-only historical decoding.  Repository
classes are replaced by inert carriers, so old slotted state is never assigned
to whichever live class happens to have the same import path today.

SHA-256 supplies integrity and content identity only.  It does not make pickle
safe.  Even the inert compatibility decoder remains behind an explicit
``trusted_local=True`` boundary and is not a security sandbox.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import stat
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence, cast

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash


_LEGACY_MAGIC: Final = b"MV1CKPT1"
_COMPACT_MAGIC: Final = b"MV1CKPT2"
_PROPOSAL_MAGIC: Final = b"MV1CKPT3"
_ACQUISITION_MAGIC: Final = b"MV1ACQ1"
_PROPOSAL_STATE_MAGIC: Final = b"MV1PROP1"
_TERMINAL_FAMILY_MATRIX_MAGIC: Final = b"MV1TERM1"
_TERMINAL_COMPARISON_MATRIX_MAGIC_V2: Final = b"MV1TERM2"

_LEGACY_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v1"
_COMPACT_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v2"
_PROPOSAL_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_v3"
_ACQUISITION_SCHEMA: Final = "partition_free_v1_runtime_acquisition_sidecar_v1"
_PROPOSAL_STATE_SCHEMA: Final = "partition_free_v1_runtime_proposal_state_sidecar_v2"
_PROPOSAL_STATE_PAYLOAD_SCHEMA: Final = (
    "partition_free_v1_runtime_proposal_state_sidecar_payload_v2"
)
_LABEL_NATIVE_PLAN_CHECKPOINT_PAYLOAD_SCHEMA: Final = (
    "partition_free_v1_label_native_runtime_plan_checkpoint_payload_v1"
)
_LABEL_NATIVE_PLAN_SCHEMA: Final = (
    "partition_free_v1_label_native_pre_O_runtime_plan_v1"
)
_LABEL_NATIVE_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_label_native_pre_O_runtime_descriptor_v1"
)
_LABEL_NATIVE_DESCRIPTOR_PREFIX: Final = (
    "modern-v1-label-native-runtime-descriptor-"
)
_LABEL_NATIVE_PLAN_PREFIX: Final = "modern-v1-label-native-runtime-plan-"
_MAX_LABEL_NATIVE_CHECKPOINT_DESCRIPTORS: Final = 8
_LABEL_NATIVE_CHECKPOINT_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "plan_schema",
        "descriptor_schema",
        "plan_fields",
        "descriptor_records",
        "plan_hash",
    }
)
_LABEL_NATIVE_DESCRIPTOR_RECORD_FIELDS: Final = frozenset(
    {"fields", "descriptor_hash"}
)
_LABEL_NATIVE_DESCRIPTOR_INIT_FIELDS: Final = (
    "queue_index",
    "coordinate",
    "certificate",
    "crosswalk",
    "source_action_receipt",
    "compact_provider_endpoint_label",
    "source_endpoint_label",
    "physical_duplicate_coordinate_hashes",
    "physical_duplicate_certificate_hashes",
    "provider_receipt_hash",
    "provider_design_hash",
    "provider_policy_hash",
    "provider_forecast_hash",
    "provider_backend_kind",
    "provider_device_identity",
    "prefix_rank_receipt_hash",
    "endpoint_query_hash",
    "resulting_label_commitment",
    "qualified_lane_registration_hash",
    "action_axis_registration_hash",
    "source_shell_ladder_receipt_hash",
    "eligible_subset_registration_hash",
    "branch_C_design_hash",
    "bridge_branch_receipt_hash",
    "fallback_authorization_hash",
    "six_role_C_design_hash",
    "six_role_C_progress_receipt_hash",
    "six_role_certification_receipt_hash",
    "six_role_decision_hash",
    "six_role_decision_design_hash",
    "bridge_family_bundle_hash",
    "campaign_identity_hash",
    "campaign_contract_hash",
    "campaign_registration_hash",
    "live_chart_identity_hash",
    "live_current_mask_hash",
    "live_route_axis_hash",
    "live_route_count",
    "live_proposal_generation",
    "native_qualified_subset_execution_replayed",
    "exact_source_replay_completed",
    "provider_or_handle_retained",
    "mask_or_route_payload_retained",
    "calibration_K_open_authority",
    "paired_oracle_O_open_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
    "execution_authority",
    "authority_scope",
)
_LABEL_NATIVE_PLAN_INIT_FIELDS: Final = (
    "campaign_identity_hash",
    "campaign_contract_hash",
    "campaign_registration_hash",
    "live_chart_identity_hash",
    "live_current_mask_hash",
    "live_route_axis_hash",
    "live_route_count",
    "live_proposal_generation",
    "six_role_C_design_hash",
    "six_role_C_progress_receipt_hash",
    "six_role_certification_receipt_hash",
    "six_role_decision_hash",
    "six_role_decision_design_hash",
    "bridge_family_bundle_hash",
    "fallback_authorization_hash",
    "selected_coordinate_hash",
    "ordered_descriptor_hashes",
    "ordered_coordinate_hashes",
    "ordered_physical_candidate_identity_hashes",
    "blocked_unresolved_coordinate_hash",
    "fallback_exhaustion_status",
    "authorized_descriptor_count",
    "resolved_prefix_frozen_before_K_O",
    "native_qualified_subset_execution_replayed",
    "source_authorized_duplicate_quotient_only",
    "stops_before_first_unresolved_decision_changing_coordinate",
    "no_provider_handle_mask_or_route_payload_retained",
    "paired_bce_oracle_still_required",
    "candidate_selection_authority",
    "calibration_K_open_authority",
    "paired_oracle_O_open_authority",
    "irreversible_decision_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
    "execution_authority",
    "authority_scope",
)
_JOURNAL_V1_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v1"
_JOURNAL_V2_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v2"
_JOURNAL_V3_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v3"
_JOURNAL_V4_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v4"
_JOURNAL_V5_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v5"
_JOURNAL_V6_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v6"
_JOURNAL_V7_SCHEMA: Final = "partition_free_v1_runtime_checkpoint_journal_v7"
_TERMINAL_FAMILY_MATRIX_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_matrix_v1"
)
_TERMINAL_FAMILY_ARTIFACT_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_artifact_reference_v1"
)
_TERMINAL_FAMILY_FREEZE_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_freeze_receipt_v1"
)
_TERMINAL_FAMILY_REQUEST_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_request_receipt_v1"
)
_TERMINAL_FAMILY_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_observation_receipt_v1"
)
_TERMINAL_FAMILY_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_receipt_v1"
)
_TERMINAL_COMPARISON_MATRIX_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_matrix_v2"
)
_TERMINAL_COMPARISON_ARTIFACT_REFERENCE_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_artifact_reference_v2"
)
_TERMINAL_COMPARISON_FREEZE_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_freeze_receipt_v2"
)
_TERMINAL_COMPARISON_REQUEST_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_request_receipt_v2"
)
_TERMINAL_COMPARISON_OBSERVATION_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_observation_receipt_v2"
)
_TERMINAL_COMPARISON_SCHEMA_V2: Final = (
    "partition_free_v1_terminal_comparison_family_receipt_v2"
)
_TERMINAL_FAMILY_VALUES_PREFIX: Final = (
    "partition-free-v1-terminal-validation-family-values-"
)
_TERMINAL_FAMILY_LANE_VALUES_PREFIX: Final = (
    "partition-free-v1-terminal-validation-family-lane-values-"
)
_TERMINAL_COMPARISON_VALUES_PREFIX_V2: Final = (
    "partition-free-v1-terminal-comparison-family-values-v2-"
)
_TERMINAL_COMPARISON_LANE_VALUES_PREFIX_V2: Final = (
    "partition-free-v1-terminal-comparison-family-lane-values-v2-"
)
_TERMINAL_COMPARISON_PAIRED_VALUES_PREFIX_V2: Final = (
    "partition-free-v1-terminal-comparison-family-paired-values-v2-"
)
_RUNTIME_STATE_SCHEMA: Final = "partition_free_v1_runtime_state_v2"
_SECURITY_BOUNDARY: Final = "trusted_local_only_do_not_unpickle_untrusted_bytes"
_FQR_HORIZON_STOP_REASON: Final = (
    "all_registered_frozen_current_generation_fqr_endpoints_"
    "certified_infeasible_at_stable_C_horizon"
)

_MAX_HEADER_BYTES: Final = 1 << 20
_MAX_CHECKPOINT_PAYLOAD_BYTES: Final = 16 << 30
_MAX_SIDECAR_PAYLOAD_BYTES: Final = 1 << 40
_HASH_CHUNK_BYTES: Final = 16 << 20
_LOWER_SHA256 = frozenset("0123456789abcdef")

_LEGACY_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "payload_sha256",
        "payload_size_bytes",
        "runtime_state_hash",
        "controller_snapshot_hash",
        "pending_look_hash",
        "security_boundary",
    }
)
_COMPACT_HEADER_FIELDS: Final = frozenset(
    set(_LEGACY_HEADER_FIELDS) | {"payload_kind", "acquisition_sidecar"}
)
_CHECKPOINT_V3_HEADER_FIELDS: Final = frozenset(
    set(_COMPACT_HEADER_FIELDS)
    | {
        "proposal_state_sidecar",
        "completed_proposal_closure_hashes",
        "staged_proposal_backend_identity_hash",
    }
)
_CHECKPOINT_V3_PAIRED_HEADER_FIELDS: Final = frozenset(
    set(_CHECKPOINT_V3_HEADER_FIELDS) | {"pending_paired_bce_chunk_hash"}
)
_CHECKPOINT_V3_FAMILY_HEADER_FIELDS: Final = frozenset(
    set(_CHECKPOINT_V3_HEADER_FIELDS)
    | {
        "terminal_validation_family_freeze_hash",
        "terminal_validation_family_request_hash",
        "terminal_validation_family_artifact",
    }
)
_CHECKPOINT_V3_PAIRED_FAMILY_HEADER_FIELDS: Final = frozenset(
    set(_CHECKPOINT_V3_PAIRED_HEADER_FIELDS)
    | {
        "terminal_validation_family_freeze_hash",
        "terminal_validation_family_request_hash",
        "terminal_validation_family_artifact",
    }
)
_CHECKPOINT_V3_COMPARISON_HEADER_FIELDS: Final = frozenset(
    set(_CHECKPOINT_V3_HEADER_FIELDS)
    | {
        "terminal_comparison_family_freeze_hash",
        "terminal_comparison_family_request_hash",
        "terminal_comparison_family_artifact",
    }
)
_CHECKPOINT_V3_PAIRED_COMPARISON_HEADER_FIELDS: Final = frozenset(
    set(_CHECKPOINT_V3_PAIRED_HEADER_FIELDS)
    | {
        "terminal_comparison_family_freeze_hash",
        "terminal_comparison_family_request_hash",
        "terminal_comparison_family_artifact",
    }
)
_TERMINAL_FAMILY_ARTIFACT_REFERENCE_FIELDS: Final = frozenset(
    {
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
)
_TERMINAL_FAMILY_MATRIX_HEADER_FIELDS: Final = frozenset(
    {
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
)
_TERMINAL_COMPARISON_ARTIFACT_REFERENCE_FIELDS_V2: Final = frozenset(
    {
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
)
_TERMINAL_COMPARISON_MATRIX_HEADER_FIELDS_V2: Final = frozenset(
    (set(_TERMINAL_COMPARISON_ARTIFACT_REFERENCE_FIELDS_V2)
     - {"relative_path", "artifact_schema", "artifact_sha256", "artifact_size_bytes", "terminal_observation_hash"})
    | {"payload_kind", "payload_sha256", "payload_size_bytes", "security_boundary"}
)
_ACQUISITION_REFERENCE_FIELDS: Final = frozenset(
    {
        "schema",
        "relative_path",
        "payload_sha256",
        "payload_size_bytes",
        "framed_size_bytes",
        "acquisition_hash",
        "acquisition_receipt_hash",
        "campaign_binding_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
    }
)
_ACQUISITION_HEADER_FIELDS: Final = frozenset(
    (set(_ACQUISITION_REFERENCE_FIELDS) - {"relative_path", "framed_size_bytes"})
    | {"security_boundary"}
)
_PROPOSAL_REFERENCE_FIELDS_IN_ORDER: Final = (
    "schema",
    "relative_path",
    "payload_sha256",
    "payload_size_bytes",
    "framed_size_bytes",
    "codec_id",
    "representation_hash",
    "sufficient_state_hash",
    "storage_descriptor_hash",
    "proposal_progress_hash",
    "campaign_contract_hash",
    "campaign_binding_hash",
    "controller_run_id",
    "proposal_backend_identity_hash",
    "controller_state_hash",
    "controller_snapshot_hash",
    "current_mask_hash",
    "route_axis_hash",
    "acquisition_generation",
    "mask_lineage_hash",
    "transition_lineage_hash",
    "joint_state_hash",
    "phase_evidence_state_hash",
    "train_optimization_data_only",
    "validation_outcomes_observed",
    "candidate_or_mask_authority",
    "receipt_hash",
)
_PROPOSAL_REFERENCE_FIELDS: Final = frozenset(_PROPOSAL_REFERENCE_FIELDS_IN_ORDER)
_PROPOSAL_SIDECAR_HEADER_FIELDS: Final = frozenset(
    (set(_PROPOSAL_REFERENCE_FIELDS) - {"relative_path", "framed_size_bytes"})
    | {"security_boundary"}
)


class V1CheckpointRefusalCode(str, Enum):
    """Stable machine-facing reasons for inspect-only checkpoint refusal."""

    TRUSTED_LOCAL_REQUIRED = "trusted_local_required"
    PATH_SYMLINK_OR_TRAVERSAL = "path_symlink_or_traversal"
    PATH_NOT_REGULAR = "path_not_regular"
    CHECKPOINT_TOO_LARGE = "checkpoint_too_large"
    INVALID_MAGIC = "invalid_magic"
    INVALID_FRAME = "invalid_frame"
    INVALID_JSON = "invalid_json"
    INVALID_HEADER_SHAPE = "invalid_header_shape"
    PAYLOAD_DIGEST_MISMATCH = "payload_digest_mismatch"
    ARTIFACT_CHANGED_DURING_INSPECTION = "artifact_changed_during_inspection"
    SIDECAR_PATH_INVALID = "sidecar_path_invalid"
    SIDECAR_MISSING = "sidecar_missing"
    SIDECAR_REFERENCE_MISMATCH = "sidecar_reference_mismatch"
    TRUSTED_DECODE_FAILED = "trusted_decode_failed"
    RAW_ROOT_INVALID = "raw_root_invalid"
    RAW_STATE_INVALID = "raw_state_invalid"
    COHORT_INCONSISTENT = "cohort_inconsistent"
    HISTORICAL_IDENTITY_MISMATCH = "historical_identity_mismatch"
    MIGRATION_DEFERRED = "migration_deferred"


class V1CheckpointInspectionRefusal(ValueError):
    """Typed fail-closed result for an inspection that grants no authority."""

    def __init__(
        self,
        code: V1CheckpointRefusalCode,
        message: str,
        *,
        phase: str,
        artifact: str | os.PathLike[str] | None = None,
    ) -> None:
        self.code = code
        self.phase = phase
        self.artifact = None if artifact is None else str(artifact)
        suffix = "" if self.artifact is None else f" [{self.artifact}]"
        super().__init__(f"{code.value}: {message}{suffix}")


def _refuse(
    code: V1CheckpointRefusalCode,
    message: str,
    *,
    phase: str,
    artifact: str | os.PathLike[str] | None = None,
) -> NoReturn:
    raise V1CheckpointInspectionRefusal(
        code,
        message,
        phase=phase,
        artifact=artifact,
    )


def _exact_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            f"{label} must be an exact nonempty string",
            phase="header_validation",
        )
    return cast(str, value)


def _exact_nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            f"{label} must be an exact nonnegative integer",
            phase="header_validation",
        )
    return cast(int, value)


def _exact_positive_integer(value: object, label: str, *, maximum: int) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            f"{label} must be an exact positive integer no greater than {maximum}",
            phase="header_validation",
        )
    return cast(int, value)


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_SHA256 for character in value)
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            f"{label} must be one lowercase SHA-256 digest",
            phase="header_validation",
        )
    return cast(str, value)


def _identity(value: object, label: str) -> str:
    return _exact_nonempty_string(value, label)


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_float(value: str) -> float:
    observed = float(value)
    if not math.isfinite(observed):
        raise ValueError(f"non-finite JSON number: {value}")
    return observed


def _strict_json_object(
    payload: bytes,
    *,
    allow_space_padding: bool,
    artifact: str,
) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        decoder = json.JSONDecoder(
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
        )
        value, end = decoder.raw_decode(text)
        remainder = text[end:]
        if remainder and (
            not allow_space_padding or any(character != " " for character in remainder)
        ):
            raise ValueError("unexpected bytes after the JSON object")
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        _refuse(
            V1CheckpointRefusalCode.INVALID_JSON,
            "frame header is not strict finite duplicate-free JSON",
            phase="strict_json",
            artifact=artifact,
        )
        raise AssertionError("unreachable") from exc
    if type(value) is not dict:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "frame header must be an object",
            phase="header_validation",
            artifact=artifact,
        )
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
    artifact: str,
) -> None:
    observed = frozenset(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            f"{label} has missing={missing!r}, unknown={unknown!r}",
            phase="header_validation",
            artifact=artifact,
        )


def _nullable_identity(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _identity(value, label)


def _validate_checkpoint_header(
    header: dict[str, Any],
    *,
    magic: bytes,
    artifact: str,
) -> None:
    observed_header_fields = frozenset(header)
    if magic == _LEGACY_MAGIC:
        expected_fields = _LEGACY_HEADER_FIELDS
        expected_schema = _LEGACY_SCHEMA
    elif magic == _COMPACT_MAGIC:
        expected_fields = _COMPACT_HEADER_FIELDS
        expected_schema = _COMPACT_SCHEMA
    elif magic == _PROPOSAL_MAGIC:
        if observed_header_fields == _CHECKPOINT_V3_HEADER_FIELDS:
            expected_fields = _CHECKPOINT_V3_HEADER_FIELDS
        elif observed_header_fields == _CHECKPOINT_V3_PAIRED_HEADER_FIELDS:
            expected_fields = _CHECKPOINT_V3_PAIRED_HEADER_FIELDS
        elif observed_header_fields == _CHECKPOINT_V3_FAMILY_HEADER_FIELDS:
            expected_fields = _CHECKPOINT_V3_FAMILY_HEADER_FIELDS
        elif observed_header_fields == _CHECKPOINT_V3_PAIRED_FAMILY_HEADER_FIELDS:
            expected_fields = _CHECKPOINT_V3_PAIRED_FAMILY_HEADER_FIELDS
        elif observed_header_fields == _CHECKPOINT_V3_COMPARISON_HEADER_FIELDS:
            expected_fields = _CHECKPOINT_V3_COMPARISON_HEADER_FIELDS
        elif (
            observed_header_fields
            == _CHECKPOINT_V3_PAIRED_COMPARISON_HEADER_FIELDS
        ):
            expected_fields = _CHECKPOINT_V3_PAIRED_COMPARISON_HEADER_FIELDS
        else:
            _exact_fields(
                header,
                _CHECKPOINT_V3_PAIRED_COMPARISON_HEADER_FIELDS,
                label="checkpoint header",
                artifact=artifact,
            )
            raise AssertionError("unreachable")
        expected_schema = _PROPOSAL_SCHEMA
    else:  # pragma: no cover - the caller resolves magic first.
        raise AssertionError("unknown checkpoint magic")
    _exact_fields(
        header,
        expected_fields,
        label="checkpoint header",
        artifact=artifact,
    )
    if header["schema"] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "checkpoint magic and schema disagree",
            phase="header_validation",
            artifact=artifact,
        )
    _sha256(header["payload_sha256"], "payload_sha256")
    _exact_positive_integer(
        header["payload_size_bytes"],
        "payload_size_bytes",
        maximum=_MAX_CHECKPOINT_PAYLOAD_BYTES,
    )
    _identity(header["runtime_state_hash"], "runtime_state_hash")
    _identity(header["controller_snapshot_hash"], "controller_snapshot_hash")
    _nullable_identity(header["pending_look_hash"], "pending_look_hash")
    if header["security_boundary"] != _SECURITY_BOUNDARY:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "checkpoint security boundary is absent or changed",
            phase="header_validation",
            artifact=artifact,
        )
    if magic != _LEGACY_MAGIC:
        if header["payload_kind"] != "compact_runtime_journal":
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "checkpoint payload kind is unsupported",
                phase="header_validation",
                artifact=artifact,
            )
        acquisition = header["acquisition_sidecar"]
        if acquisition is not None:
            _validate_acquisition_reference(acquisition, artifact=artifact)
    if magic == _PROPOSAL_MAGIC:
        proposal = header["proposal_state_sidecar"]
        if proposal is not None:
            _validate_proposal_reference(proposal, artifact=artifact)
        completed_hashes = header["completed_proposal_closure_hashes"]
        staged_backend = _nullable_identity(
            header["staged_proposal_backend_identity_hash"],
            "staged_proposal_backend_identity_hash",
        )
        pending_paired_hash = None
        if "pending_paired_bce_chunk_hash" in header:
            pending_paired_hash = _nullable_identity(
                header["pending_paired_bce_chunk_hash"],
                "pending_paired_bce_chunk_hash",
            )
        family_freeze_hash = None
        family_request_hash = None
        family_artifact = None
        if "terminal_validation_family_freeze_hash" in header:
            family_freeze_hash = _identity(
                header["terminal_validation_family_freeze_hash"],
                "terminal_validation_family_freeze_hash",
            )
            family_request_hash = _nullable_identity(
                header["terminal_validation_family_request_hash"],
                "terminal_validation_family_request_hash",
            )
            family_artifact = header["terminal_validation_family_artifact"]
            if family_artifact is not None:
                _validate_terminal_family_artifact_reference(
                    family_artifact,
                    artifact=artifact,
                )
                if family_request_hash is None:
                    _refuse(
                        V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                        "terminal-family artifact lacks its request identity",
                        phase="header_validation",
                        artifact=artifact,
                    )
        comparison_freeze_hash = None
        comparison_request_hash = None
        comparison_artifact = None
        if "terminal_comparison_family_freeze_hash" in header:
            comparison_freeze_hash = _identity(
                header["terminal_comparison_family_freeze_hash"],
                "terminal_comparison_family_freeze_hash",
            )
            comparison_request_hash = _nullable_identity(
                header["terminal_comparison_family_request_hash"],
                "terminal_comparison_family_request_hash",
            )
            comparison_artifact = header["terminal_comparison_family_artifact"]
            if comparison_artifact is not None:
                _validate_terminal_comparison_artifact_reference_v2(
                    comparison_artifact,
                    artifact=artifact,
                )
                if comparison_request_hash is None:
                    _refuse(
                        V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                        "terminal-comparison artifact lacks its request identity",
                        phase="header_validation",
                        artifact=artifact,
                    )
        if family_freeze_hash is not None and comparison_freeze_hash is not None:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "checkpoint mixes terminal family v1 and comparison v2",
                phase="header_validation",
                artifact=artifact,
            )
        if (
            proposal is not None
            and (
                staged_backend is None
                or proposal["proposal_backend_identity_hash"] != staged_backend
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "proposal sidecar changed the staged backend anchor",
                phase="header_validation",
                artifact=artifact,
            )
        if type(completed_hashes) is not list:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "completed proposal closure hashes must be an array",
                phase="header_validation",
                artifact=artifact,
            )
        for value in completed_hashes:
            _identity(value, "completed proposal closure hash")
        if len(set(completed_hashes)) != len(completed_hashes):
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "completed proposal closure hashes contain duplicates",
                phase="header_validation",
                artifact=artifact,
            )
        has_staged_state = bool(proposal is not None or completed_hashes)
        if has_staged_state is not (staged_backend is not None):
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "staged proposal state and backend anchor disagree",
                phase="header_validation",
                artifact=artifact,
            )
        if (
            proposal is None
            and not completed_hashes
            and pending_paired_hash is None
            and family_freeze_hash is None
            and comparison_freeze_hash is None
        ):
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "checkpoint-v3 lacks staged, paired, or terminal-family state",
                phase="header_validation",
                artifact=artifact,
            )


def _validate_acquisition_reference(value: object, *, artifact: str) -> dict[str, Any]:
    if type(value) is not dict:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "acquisition sidecar reference must be an object",
            phase="header_validation",
            artifact=artifact,
        )
    reference = cast(dict[str, Any], value)
    _exact_fields(
        reference,
        _ACQUISITION_REFERENCE_FIELDS,
        label="acquisition sidecar reference",
        artifact=artifact,
    )
    if reference["schema"] != _ACQUISITION_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "acquisition sidecar schema is unsupported",
            phase="header_validation",
            artifact=artifact,
        )
    _identity(reference["relative_path"], "acquisition relative_path")
    _sha256(reference["payload_sha256"], "acquisition payload_sha256")
    payload_size = _exact_positive_integer(
        reference["payload_size_bytes"],
        "acquisition payload_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES,
    )
    framed_size = _exact_positive_integer(
        reference["framed_size_bytes"],
        "acquisition framed_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES + _MAX_HEADER_BYTES + 16,
    )
    if framed_size != len(_ACQUISITION_MAGIC) + 8 + 4096 + payload_size:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "acquisition framed size is not its exact fixed-header total",
            phase="header_validation",
            artifact=artifact,
        )
    for name in (
        "acquisition_hash",
        "acquisition_receipt_hash",
        "campaign_binding_hash",
        "current_mask_hash",
        "route_axis_hash",
    ):
        _identity(reference[name], name)
    _exact_nonnegative_integer(reference["generation_index"], "generation_index")
    return reference


def _validate_proposal_reference(value: object, *, artifact: str) -> dict[str, Any]:
    if type(value) is not dict:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "proposal-state sidecar reference must be an object",
            phase="header_validation",
            artifact=artifact,
        )
    reference = cast(dict[str, Any], value)
    _exact_fields(
        reference,
        _PROPOSAL_REFERENCE_FIELDS,
        label="proposal-state sidecar reference",
        artifact=artifact,
    )
    if reference["schema"] != _PROPOSAL_STATE_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "proposal-state sidecar schema is unsupported",
            phase="header_validation",
            artifact=artifact,
        )
    _identity(reference["relative_path"], "proposal relative_path")
    _sha256(reference["payload_sha256"], "proposal payload_sha256")
    payload_size = _exact_positive_integer(
        reference["payload_size_bytes"],
        "proposal payload_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES,
    )
    framed_size = _exact_positive_integer(
        reference["framed_size_bytes"],
        "proposal framed_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES + _MAX_HEADER_BYTES + 16,
    )
    if framed_size != len(_PROPOSAL_STATE_MAGIC) + 8 + 4096 + payload_size:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "proposal framed size is not its exact fixed-header total",
            phase="header_validation",
            artifact=artifact,
        )
    for name in (
        "codec_id",
        "representation_hash",
        "sufficient_state_hash",
        "storage_descriptor_hash",
        "proposal_progress_hash",
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "controller_state_hash",
        "controller_snapshot_hash",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "transition_lineage_hash",
        "joint_state_hash",
        "phase_evidence_state_hash",
    ):
        _identity(reference[name], name)
    storage_descriptor = cast(str, reference["storage_descriptor_hash"])
    if (
        Path(storage_descriptor).is_absolute()
        or PureWindowsPath(storage_descriptor).is_absolute()
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "storage_descriptor_hash must be a logical identity, not an absolute path",
            phase="header_validation",
            artifact=artifact,
        )
    _exact_nonnegative_integer(
        reference["acquisition_generation"],
        "acquisition_generation",
    )
    for name, required in (
        ("train_optimization_data_only", True),
        ("validation_outcomes_observed", False),
        ("candidate_or_mask_authority", False),
    ):
        if type(reference[name]) is not bool or reference[name] is not required:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                f"proposal authority firewall field {name} is invalid",
                phase="header_validation",
                artifact=artifact,
            )
    _identity(reference["receipt_hash"], "receipt_hash")
    logical = {
        name: reference[name] for name in _PROPOSAL_REFERENCE_FIELDS_IN_ORDER[:-1]
    }
    expected_hash = content_hash(
        logical,
        prefix="partition-free-v1-runtime-proposal-state-sidecar-",
    )
    if reference["receipt_hash"] != expected_hash:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "proposal-state sidecar receipt hash does not replay",
            phase="header_validation",
            artifact=artifact,
        )
    return reference


def _validate_terminal_family_lane_coordinates(
    value: Mapping[str, Any],
    *,
    artifact: str,
) -> None:
    shape = value["joint_validation_shape"]
    lane_hashes = value["unique_lane_validation_values_hashes"]
    lane_counts = value["unique_lane_validation_value_counts"]
    lane_metrics = value["unique_lane_metrics"]
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(
            type(item) is not int or item <= 0
            for item in shape
        )
        or type(lane_hashes) is not list
        or type(lane_counts) is not list
        or type(lane_metrics) is not list
        or not (
            len(lane_hashes) == len(lane_counts) == len(lane_metrics) == shape[0]
        )
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-family lane coordinates differ from their matrix shape",
            phase="header_validation",
            artifact=artifact,
        )
    for lane_hash in lane_hashes:
        _identity(lane_hash, "unique_lane_validation_values_hash")
    for count in lane_counts:
        if type(count) is not int or count != shape[1]:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-family lane count differs from the common panel",
                phase="header_validation",
                artifact=artifact,
            )
    for metrics in lane_metrics:
        if type(metrics) is not dict or set(metrics) != {
            "mean",
            "maximum",
            "root_mean_square",
        }:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-family lane metrics have unsupported fields",
                phase="header_validation",
                artifact=artifact,
            )
        for metric in metrics.values():
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
                or float(metric) < 0.0
            ):
                _refuse(
                    V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                    "terminal-family lane metric is not finite and nonnegative",
                    phase="header_validation",
                    artifact=artifact,
                )


def _validate_terminal_family_artifact_reference(
    value: object,
    *,
    artifact: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-family artifact reference must be an object",
            phase="header_validation",
            artifact=artifact,
        )
    reference = cast(dict[str, Any], value)
    _exact_fields(
        reference,
        _TERMINAL_FAMILY_ARTIFACT_REFERENCE_FIELDS,
        label="terminal-family artifact reference",
        artifact=artifact,
    )
    if (
        reference["schema"] != _TERMINAL_FAMILY_ARTIFACT_REFERENCE_SCHEMA
        or reference["artifact_schema"] != _TERMINAL_FAMILY_MATRIX_SCHEMA
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-family artifact reference schema is unsupported",
            phase="header_validation",
            artifact=artifact,
        )
    _identity(reference["relative_path"], "terminal-family relative_path")
    _sha256(reference["artifact_sha256"], "terminal-family artifact_sha256")
    _exact_positive_integer(
        reference["artifact_size_bytes"],
        "terminal-family artifact_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES + _MAX_HEADER_BYTES + 16,
    )
    for name in (
        "terminal_request_hash",
        "family_freeze_hash",
        "joint_validation_values_hash",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    ):
        _identity(reference[name], name)
    _validate_terminal_family_lane_coordinates(reference, artifact=artifact)
    return reference


def _validate_terminal_comparison_role_map_v2(
    value: Mapping[str, Any],
    *,
    artifact: str,
) -> tuple[int, int, int]:
    roles = value["roles"]
    role_map = value["role_to_unique_lane"]
    shape = value["joint_validation_shape"]
    if (
        type(roles) is not list
        or tuple(roles)
        != ("dense_reference", "current_v0_comparator", "terminal_v1")
        or type(role_map) is not list
        or len(role_map) != 3
        or any(
            type(lane) is not int or lane < 0 or lane >= shape[0]
            for lane in role_map
        )
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-comparison role map differs",
            phase="header_validation",
            artifact=artifact,
        )
    return cast(tuple[int, int, int], tuple(role_map))


def _validate_terminal_comparison_pairs_v2(
    value: Mapping[str, Any],
    *,
    artifact: str,
) -> None:
    expected = {
        "current_v0_comparator_minus_dense_reference",
        "terminal_v1_minus_dense_reference",
        "terminal_v1_minus_current_v0_comparator",
    }
    deltas = value["paired_mean_deltas"]
    hashes = value["paired_validation_difference_hashes"]
    if (
        type(deltas) is not dict
        or type(hashes) is not dict
        or set(deltas) != expected
        or set(hashes) != expected
        or value["paired_delta_derivation"]
        != "mean_of_pointwise_common_panel_row_differences_float64_v1"
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-comparison paired coordinates differ",
            phase="header_validation",
            artifact=artifact,
        )
    for delta in deltas.values():
        if (
            isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
        ):
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-comparison paired delta is nonfinite",
                phase="header_validation",
                artifact=artifact,
            )
    for pair_hash in hashes.values():
        _identity(pair_hash, "paired_validation_difference_hash")


def _validate_terminal_comparison_artifact_reference_v2(
    value: object,
    *,
    artifact: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-comparison artifact reference must be an object",
            phase="header_validation",
            artifact=artifact,
        )
    reference = cast(dict[str, Any], value)
    _exact_fields(
        reference,
        _TERMINAL_COMPARISON_ARTIFACT_REFERENCE_FIELDS_V2,
        label="terminal-comparison artifact reference",
        artifact=artifact,
    )
    if (
        reference["schema"]
        != _TERMINAL_COMPARISON_ARTIFACT_REFERENCE_SCHEMA_V2
        or reference["artifact_schema"]
        != _TERMINAL_COMPARISON_MATRIX_SCHEMA_V2
    ):
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-comparison artifact reference schema is unsupported",
            phase="header_validation",
            artifact=artifact,
        )
    _identity(reference["relative_path"], "terminal-comparison relative_path")
    _sha256(reference["artifact_sha256"], "terminal-comparison artifact_sha256")
    _exact_positive_integer(
        reference["artifact_size_bytes"],
        "terminal-comparison artifact_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES + _MAX_HEADER_BYTES + 16,
    )
    for name in (
        "terminal_request_hash",
        "family_freeze_hash",
        "joint_validation_values_hash",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_semantic_hash",
        "terminal_observation_hash",
    ):
        _identity(reference[name], name)
    _validate_terminal_family_lane_coordinates(reference, artifact=artifact)
    _validate_terminal_comparison_role_map_v2(reference, artifact=artifact)
    _validate_terminal_comparison_pairs_v2(reference, artifact=artifact)
    semantic_hash = content_hash(
        {
            "terminal_request_hash": reference["terminal_request_hash"],
            "joint_validation_values_hash": reference[
                "joint_validation_values_hash"
            ],
            "joint_validation_shape": tuple(reference["joint_validation_shape"]),
            "unique_lane_validation_values_hashes": tuple(
                reference["unique_lane_validation_values_hashes"]
            ),
            "unique_lane_validation_value_counts": tuple(
                reference["unique_lane_validation_value_counts"]
            ),
            "unique_lane_metrics": tuple(reference["unique_lane_metrics"]),
            "paired_mean_deltas": reference["paired_mean_deltas"],
            "paired_validation_difference_hashes": reference[
                "paired_validation_difference_hashes"
            ],
            "paired_delta_derivation": reference["paired_delta_derivation"],
            "common_panel_evaluator_receipt_hash": reference[
                "common_panel_evaluator_receipt_hash"
            ],
        },
        prefix=(
            "partition-free-v1-terminal-comparison-family-"
            "observation-artifact-v2-"
        ),
    )
    if reference["observation_artifact_semantic_hash"] != semantic_hash:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "terminal-comparison artifact semantic hash differs",
            phase="header_validation",
            artifact=artifact,
        )
    return reference


def _validate_sidecar_header(
    header: dict[str, Any],
    *,
    magic: bytes,
    artifact: str,
) -> None:
    if magic == _ACQUISITION_MAGIC:
        expected = _ACQUISITION_HEADER_FIELDS
        schema = _ACQUISITION_SCHEMA
    elif magic == _PROPOSAL_STATE_MAGIC:
        expected = _PROPOSAL_SIDECAR_HEADER_FIELDS
        schema = _PROPOSAL_STATE_SCHEMA
    elif magic == _TERMINAL_FAMILY_MATRIX_MAGIC:
        expected = _TERMINAL_FAMILY_MATRIX_HEADER_FIELDS
        schema = _TERMINAL_FAMILY_MATRIX_SCHEMA
    elif magic == _TERMINAL_COMPARISON_MATRIX_MAGIC_V2:
        expected = _TERMINAL_COMPARISON_MATRIX_HEADER_FIELDS_V2
        schema = _TERMINAL_COMPARISON_MATRIX_SCHEMA_V2
    else:  # pragma: no cover
        raise AssertionError("unknown sidecar magic")
    _exact_fields(header, expected, label="sidecar header", artifact=artifact)
    if header["schema"] != schema or header["security_boundary"] != _SECURITY_BOUNDARY:
        _refuse(
            V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
            "sidecar schema or security boundary is invalid",
            phase="header_validation",
            artifact=artifact,
        )
    _sha256(header["payload_sha256"], "sidecar payload_sha256")
    _exact_positive_integer(
        header["payload_size_bytes"],
        "sidecar payload_size_bytes",
        maximum=_MAX_SIDECAR_PAYLOAD_BYTES,
    )
    if magic == _ACQUISITION_MAGIC:
        for name in (
            "acquisition_hash",
            "acquisition_receipt_hash",
            "campaign_binding_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            _identity(header[name], name)
        _exact_nonnegative_integer(header["generation_index"], "generation_index")
    elif magic == _PROPOSAL_STATE_MAGIC:
        # The physical header omits only the path and framed-size coordinates.
        for name in _PROPOSAL_REFERENCE_FIELDS_IN_ORDER:
            if name in {"relative_path", "framed_size_bytes"}:
                continue
            if name == "acquisition_generation":
                _exact_nonnegative_integer(header[name], name)
            elif name in {
                "train_optimization_data_only",
                "validation_outcomes_observed",
                "candidate_or_mask_authority",
            }:
                required = name == "train_optimization_data_only"
                if type(header[name]) is not bool or header[name] is not required:
                    _refuse(
                        V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                        f"proposal authority firewall field {name} is invalid",
                        phase="header_validation",
                        artifact=artifact,
                    )
            elif name == "payload_sha256":
                _sha256(header[name], name)
            elif name == "payload_size_bytes":
                continue
            else:
                _identity(header[name], name)
    elif magic == _TERMINAL_FAMILY_MATRIX_MAGIC:
        if header["payload_kind"] != "raw_little_endian_float64_c_matrix":
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-family matrix payload kind is unsupported",
                phase="header_validation",
                artifact=artifact,
            )
        for name in (
            "terminal_request_hash",
            "family_freeze_hash",
            "joint_validation_values_hash",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
        ):
            _identity(header[name], name)
        _validate_terminal_family_lane_coordinates(header, artifact=artifact)
        shape = header["joint_validation_shape"]
        if header["payload_size_bytes"] != shape[0] * shape[1] * 8:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-family matrix byte size differs from its shape",
                phase="header_validation",
                artifact=artifact,
            )
    else:
        if header["payload_kind"] != "raw_little_endian_float64_c_matrix":
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-comparison matrix payload kind is unsupported",
                phase="header_validation",
                artifact=artifact,
            )
        for name in (
            "terminal_request_hash",
            "family_freeze_hash",
            "joint_validation_values_hash",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
        ):
            _identity(header[name], name)
        _validate_terminal_family_lane_coordinates(header, artifact=artifact)
        _validate_terminal_comparison_role_map_v2(header, artifact=artifact)
        _validate_terminal_comparison_pairs_v2(header, artifact=artifact)
        shape = header["joint_validation_shape"]
        if header["payload_size_bytes"] != shape[0] * shape[1] * 8:
            _refuse(
                V1CheckpointRefusalCode.INVALID_HEADER_SHAPE,
                "terminal-comparison matrix byte size differs from its shape",
                phase="header_validation",
                artifact=artifact,
            )


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalStateSidecarReceipt:
    """Strict logical v3 reference; it never grants execution authority."""

    schema: str
    relative_path: str
    payload_sha256: str
    payload_size_bytes: int
    framed_size_bytes: int
    codec_id: str
    representation_hash: str
    sufficient_state_hash: str
    storage_descriptor_hash: str
    proposal_progress_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    controller_state_hash: str
    controller_snapshot_hash: str
    current_mask_hash: str
    route_axis_hash: str
    acquisition_generation: int
    mask_lineage_hash: str
    transition_lineage_hash: str
    joint_state_hash: str
    phase_evidence_state_hash: str
    train_optimization_data_only: bool
    validation_outcomes_observed: bool
    candidate_or_mask_authority: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != _PROPOSAL_STATE_SCHEMA:
            raise ValueError("unsupported proposal-state sidecar schema")
        relative_text = self.relative_path
        if type(relative_text) is not str:
            raise ValueError("proposal-state sidecar path must be an exact string")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or "\x00" in relative_text
            or len(relative.parts) != 2
            or not relative.parts[0].endswith(".proposal_states")
            or relative.parts[0] == ".proposal_states"
            or not relative.parts[1].endswith(".mv1prop")
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("proposal-state sidecar path is not a safe relative path")
        if (
            type(self.payload_sha256) is not str
            or len(self.payload_sha256) != 64
            or any(character not in _LOWER_SHA256 for character in self.payload_sha256)
        ):
            raise ValueError("payload_sha256 must be one lowercase SHA-256 digest")
        if (
            type(self.payload_size_bytes) is not int
            or self.payload_size_bytes <= 0
            or self.payload_size_bytes > _MAX_SIDECAR_PAYLOAD_BYTES
            or type(self.framed_size_bytes) is not int
            or self.framed_size_bytes
            != len(_PROPOSAL_STATE_MAGIC) + 8 + 4096 + self.payload_size_bytes
            or self.framed_size_bytes
            > _MAX_SIDECAR_PAYLOAD_BYTES + _MAX_HEADER_BYTES + 16
        ):
            raise ValueError("proposal-state sidecar sizes are outside exact bounds")
        identity_fields = _PROPOSAL_REFERENCE_FIELDS_IN_ORDER[5:23]
        for name in identity_fields:
            if name == "acquisition_generation":
                continue
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be an exact nonempty string")
        if (
            Path(self.storage_descriptor_hash).is_absolute()
            or PureWindowsPath(self.storage_descriptor_hash).is_absolute()
        ):
            raise ValueError("storage_descriptor_hash must be a logical identity")
        if (
            type(self.acquisition_generation) is not int
            or self.acquisition_generation < 0
        ):
            raise ValueError(
                "acquisition_generation must be an exact nonnegative integer"
            )
        if (
            type(self.train_optimization_data_only) is not bool
            or self.train_optimization_data_only is not True
            or type(self.validation_outcomes_observed) is not bool
            or self.validation_outcomes_observed is not False
            or type(self.candidate_or_mask_authority) is not bool
            or self.candidate_or_mask_authority is not False
        ):
            raise ValueError("proposal-state sidecar crossed its authority firewall")
        if type(self.receipt_hash) is not str or not self.receipt_hash:
            raise ValueError("receipt_hash must be an exact nonempty string")
        logical = {
            name: getattr(self, name)
            for name in _PROPOSAL_REFERENCE_FIELDS_IN_ORDER[:-1]
        }
        expected = content_hash(
            logical,
            prefix="partition-free-v1-runtime-proposal-state-sidecar-",
        )
        if self.receipt_hash != expected:
            raise ValueError("proposal-state sidecar receipt hash does not replay")


@dataclass(frozen=True, slots=True)
class V1RuntimeProposalStateSidecarPayload:
    """Low-level v3 DTO; runtime validation remains a later integration joint."""

    proposal_progress_payload: object
    pending_proposal_acquisition_payload: object | None
    proposal_progress_hash: str
    pending_proposal_acquisition_hash: str | None
    proposal_backend_identity_hash: str
    representation_hash: str
    schema: str = _PROPOSAL_STATE_PAYLOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.proposal_progress_payload is None:
            raise ValueError("proposal-state sidecar requires cumulative progress")
        for name in (
            "proposal_progress_hash",
            "proposal_backend_identity_hash",
            "representation_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be an exact nonempty string")
        pending_hash = self.pending_proposal_acquisition_hash
        if (self.pending_proposal_acquisition_payload is None) != (
            pending_hash is None
        ):
            raise ValueError("pending proposal payload/hash nullability differs")
        if pending_hash is not None and (
            type(pending_hash) is not str or not pending_hash
        ):
            raise ValueError(
                "pending_proposal_acquisition_hash must be an exact nonempty string"
            )
        if (
            type(self.schema) is not str
            or self.schema != _PROPOSAL_STATE_PAYLOAD_SCHEMA
        ):
            raise ValueError("unsupported proposal-state sidecar payload schema")


@dataclass(frozen=True, slots=True)
class V1FramedArtifactInspectionReceipt:
    """One same-descriptor framing and digest receipt; never a safety claim."""

    artifact_path: str
    magic: str
    schema: str
    file_size_bytes: int
    header_size_bytes: int
    payload_offset_bytes: int
    payload_size_bytes: int
    header_sha256: str
    payload_sha256: str
    file_sha256: str
    regular_file_verified: bool
    no_final_component_symlink: bool
    framing_verified: bool
    payload_integrity_verified: bool
    pickle_safety_certified: bool = False
    schema_version: str = "partition_free_v1_framed_artifact_inspection_receipt_v1"

    def __post_init__(self) -> None:
        if self.schema_version != (
            "partition_free_v1_framed_artifact_inspection_receipt_v1"
        ):
            raise ValueError("unsupported framed inspection receipt schema")
        if any(
            type(value) is not bool or value is not True
            for value in (
                self.regular_file_verified,
                self.no_final_component_symlink,
                self.framing_verified,
                self.payload_integrity_verified,
            )
        ):
            raise ValueError("successful framed receipt lacks an integrity predicate")
        if (
            type(self.pickle_safety_certified) is not bool
            or self.pickle_safety_certified is not False
        ):
            raise ValueError("SHA-256 framing cannot certify pickle safety")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-framed-inspection-")


@dataclass(frozen=True, slots=True)
class V1CheckpointInspectionEvidenceReceipt:
    checkpoint_frame: V1FramedArtifactInspectionReceipt
    sidecar_frames: tuple[V1FramedArtifactInspectionReceipt, ...]
    all_referenced_sidecars_verified_before_decode: bool
    all_decoded_sidecar_payload_identities_replayed: bool
    historical_acquisition_layout_source_certified: bool
    proposal_runtime_contract_validation_performed: bool
    trusted_local_opt_in: bool
    raw_compatibility_decode_performed: bool
    live_runtime_class_assignment_performed: bool
    embedded_runtime_identity_replayed: bool
    embedded_controller_identity_replayed: bool
    embedded_controller_snapshot_replayed: bool
    terminal_validation_execution_performed: bool
    mask_mutation_authority: bool
    schema: str = "partition_free_v1_checkpoint_inspection_evidence_v1"

    def __post_init__(self) -> None:
        if self.schema != "partition_free_v1_checkpoint_inspection_evidence_v1":
            raise ValueError("unsupported checkpoint inspection evidence schema")
        required_true = (
            self.all_referenced_sidecars_verified_before_decode,
            self.all_decoded_sidecar_payload_identities_replayed,
            self.trusted_local_opt_in,
            self.raw_compatibility_decode_performed,
            self.embedded_runtime_identity_replayed,
            self.embedded_controller_identity_replayed,
            self.embedded_controller_snapshot_replayed,
        )
        required_false = (
            self.proposal_runtime_contract_validation_performed,
            self.live_runtime_class_assignment_performed,
            self.terminal_validation_execution_performed,
            self.mask_mutation_authority,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            raise ValueError("checkpoint inspection evidence is incomplete")
        if any(
            type(value) is not bool or value is not False for value in required_false
        ):
            raise ValueError("checkpoint inspection crossed its authority boundary")
        if type(self.historical_acquisition_layout_source_certified) is not bool:
            raise ValueError("historical acquisition evidence flag must be exact bool")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-checkpoint-inspection-")


@dataclass(frozen=True, slots=True)
class V1CheckpointMigrationEvidenceReceipt:
    source_cohort: str
    source_checkpoint_file_sha256: str
    source_payload_sha256: str
    source_runtime_state_hash: str
    historical_projection_validated: bool
    executable_source_state_materialized: bool
    migration_bytes_published: bool
    migration_authority: bool
    disposition: str
    schema: str = "partition_free_v1_checkpoint_migration_evidence_v1"

    def __post_init__(self) -> None:
        if self.schema != "partition_free_v1_checkpoint_migration_evidence_v1":
            raise ValueError("unsupported checkpoint migration evidence schema")
        if (
            type(self.historical_projection_validated) is not bool
            or self.historical_projection_validated is not True
            or type(self.executable_source_state_materialized) is not bool
            or self.executable_source_state_materialized is not False
            or type(self.migration_bytes_published) is not bool
            or self.migration_bytes_published is not False
            or type(self.migration_authority) is not bool
            or self.migration_authority is not False
        ):
            raise ValueError("inspection-only migration evidence crossed its boundary")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-checkpoint-migration-")


@dataclass(frozen=True, slots=True)
class V1HistoricalRuntimeCheckpointView:
    """Immutable inspect-only projection; it cannot be executed or resumed."""

    checkpoint_path: str
    cohort: str
    checkpoint_magic: str
    checkpoint_schema: str
    runtime_schema: str
    runtime_state_hash: str
    controller_state_hash: str
    controller_snapshot_hash: str
    pending_look_hash: str | None
    observed_type_arities: Mapping[str, tuple[int, ...]]
    observed_schema_signals: Mapping[str, tuple[str, ...]]
    acquisition_sidecar_reference: Mapping[str, Any] | None
    proposal_state_sidecar_reference: Mapping[str, Any] | None
    inspection_evidence: V1CheckpointInspectionEvidenceReceipt
    migration_evidence: V1CheckpointMigrationEvidenceReceipt
    execution_authority: bool = False
    resume_authority: bool = False
    mask_mutation_authority: bool = False
    terminal_validation_authority: bool = False
    schema: str = "partition_free_v1_historical_runtime_checkpoint_view_v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_type_arities",
            MappingProxyType(dict(self.observed_type_arities)),
        )
        object.__setattr__(
            self,
            "observed_schema_signals",
            MappingProxyType(dict(self.observed_schema_signals)),
        )
        if self.acquisition_sidecar_reference is not None:
            object.__setattr__(
                self,
                "acquisition_sidecar_reference",
                MappingProxyType(dict(self.acquisition_sidecar_reference)),
            )
        if self.proposal_state_sidecar_reference is not None:
            object.__setattr__(
                self,
                "proposal_state_sidecar_reference",
                MappingProxyType(dict(self.proposal_state_sidecar_reference)),
            )
        if self.schema != "partition_free_v1_historical_runtime_checkpoint_view_v1":
            raise ValueError("unsupported historical checkpoint view schema")
        if any(
            type(value) is not bool or value is not False
            for value in (
                self.execution_authority,
                self.resume_authority,
                self.mask_mutation_authority,
                self.terminal_validation_authority,
            )
        ):
            raise ValueError(
                "historical checkpoint view crossed its authority firewall"
            )

    @property
    def view_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-historical-checkpoint-view-"
        )


@dataclass(frozen=True, slots=True)
class _VerifiedFrame:
    fd: int = field(repr=False, compare=False)
    artifact_path: str
    magic: bytes
    header: dict[str, Any]
    payload_offset: int
    payload_size: int
    stat_signature: tuple[int, int, int, int, int]
    receipt: V1FramedArtifactInspectionReceipt


def _open_parent_without_symlinks(path: Path) -> int:
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "pread")
    ):
        _refuse(
            V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
            "platform lacks required descriptor-relative no-symlink primitives",
            phase="path_open",
            artifact=path,
        )
    absolute = path.absolute()
    components = absolute.parts
    if not components or components[0] != os.sep:
        _refuse(
            V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
            "checkpoint path cannot be normalized to an absolute path",
            phase="path_open",
            artifact=path,
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    nofollow = os.O_NOFOLLOW
    descriptor = os.open(os.sep, flags)
    try:
        for component in components[1:]:
            if component in {"", ".", ".."}:
                _refuse(
                    V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
                    "path contains a noncanonical component",
                    phase="path_open",
                    artifact=path,
                )
            next_descriptor = os.open(
                component,
                flags | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except V1CheckpointInspectionRefusal:
        os.close(descriptor)
        raise
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        _refuse(
            V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
            "path parent is absent, traversing, or symlinked",
            phase="path_open",
            artifact=path,
        )
        raise AssertionError("unreachable") from exc


def _open_regular_at(
    parent_fd: int,
    name: str,
    *,
    artifact: str,
    missing_code: V1CheckpointRefusalCode,
) -> int:
    if not name or name in {".", ".."} or os.sep in name or "\x00" in name:
        _refuse(
            V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
            "artifact filename is not one safe path component",
            phase="path_open",
            artifact=artifact,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        _refuse(
            missing_code,
            "artifact does not exist",
            phase="path_open",
            artifact=artifact,
        )
        raise AssertionError("unreachable") from exc
    except (OSError, ValueError) as exc:
        _refuse(
            V1CheckpointRefusalCode.PATH_SYMLINK_OR_TRAVERSAL,
            "artifact is symlinked or cannot be opened safely",
            phase="path_open",
            artifact=artifact,
        )
        raise AssertionError("unreachable") from exc
    try:
        observed = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        _refuse(
            V1CheckpointRefusalCode.PATH_NOT_REGULAR,
            "artifact is not a regular file",
            phase="path_open",
            artifact=artifact,
        )
    return descriptor


def _pread_exact(fd: int, count: int, offset: int, *, artifact: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            _refuse(
                V1CheckpointRefusalCode.INVALID_FRAME,
                "artifact ended before its declared framing boundary",
                phase="framing",
                artifact=artifact,
            )
        chunks.append(chunk)
        cursor += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _assert_frame_unchanged(frame: _VerifiedFrame) -> None:
    """Recheck a verified descriptor after trusted pickle interpretation."""

    try:
        observed = os.fstat(frame.fd)
    except OSError as exc:
        _refuse(
            V1CheckpointRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "verified artifact descriptor became unavailable during decode",
            phase="post_decode_integrity",
            artifact=frame.artifact_path,
        )
        raise AssertionError("unreachable") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or _stat_signature(observed) != frame.stat_signature
    ):
        _refuse(
            V1CheckpointRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "artifact metadata changed after its integrity barrier",
            phase="post_decode_integrity",
            artifact=frame.artifact_path,
        )


def _assert_frames_unchanged(frames: Sequence[_VerifiedFrame]) -> None:
    for frame in frames:
        _assert_frame_unchanged(frame)


def _verify_frame(
    fd: int,
    *,
    artifact: str,
    allowed_magics: frozenset[bytes],
    sidecar: bool,
) -> _VerifiedFrame:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        _refuse(
            V1CheckpointRefusalCode.PATH_NOT_REGULAR,
            "artifact is not a regular file",
            phase="framing",
            artifact=artifact,
        )
    minimum_magic_size = min(len(value) for value in allowed_magics)
    if before.st_size < minimum_magic_size + 9:
        _refuse(
            V1CheckpointRefusalCode.INVALID_FRAME,
            "artifact is shorter than the minimum frame",
            phase="framing",
            artifact=artifact,
        )
    magic_matches = tuple(
        value for value in allowed_magics if os.pread(fd, len(value), 0) == value
    )
    if len(magic_matches) != 1:
        _refuse(
            V1CheckpointRefusalCode.INVALID_MAGIC,
            "artifact magic is unsupported",
            phase="framing",
            artifact=artifact,
        )
    magic = magic_matches[0]
    size_bytes = _pread_exact(fd, 8, len(magic), artifact=artifact)
    prefix = magic + size_bytes
    header_size = struct.unpack(">Q", size_bytes)[0]
    if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
        _refuse(
            V1CheckpointRefusalCode.INVALID_FRAME,
            "frame header length is outside its exact bound",
            phase="framing",
            artifact=artifact,
        )
    if magic in {_ACQUISITION_MAGIC, _PROPOSAL_STATE_MAGIC} and header_size != 4096:
        _refuse(
            V1CheckpointRefusalCode.INVALID_FRAME,
            "sidecar header is not its exact 4096-byte frame",
            phase="framing",
            artifact=artifact,
        )
    frame_prefix_size = len(magic) + 8
    header_bytes = _pread_exact(fd, header_size, frame_prefix_size, artifact=artifact)
    header = _strict_json_object(
        header_bytes,
        allow_space_padding=sidecar,
        artifact=artifact,
    )
    if sidecar:
        _validate_sidecar_header(header, magic=magic, artifact=artifact)
        maximum = _MAX_SIDECAR_PAYLOAD_BYTES
    else:
        _validate_checkpoint_header(header, magic=magic, artifact=artifact)
        maximum = _MAX_CHECKPOINT_PAYLOAD_BYTES
    payload_size = _exact_positive_integer(
        header["payload_size_bytes"],
        "payload_size_bytes",
        maximum=maximum,
    )
    payload_offset = frame_prefix_size + header_size
    expected_file_size = payload_offset + payload_size
    if before.st_size != expected_file_size:
        _refuse(
            V1CheckpointRefusalCode.INVALID_FRAME,
            "frame file size differs from its exact declared total",
            phase="framing",
            artifact=artifact,
        )
    payload_digest = hashlib.sha256()
    file_digest = hashlib.sha256(prefix + header_bytes)
    cursor = payload_offset
    remaining = payload_size
    while remaining:
        count = min(remaining, _HASH_CHUNK_BYTES)
        chunk = os.pread(fd, count, cursor)
        if len(chunk) != count:
            _refuse(
                V1CheckpointRefusalCode.INVALID_FRAME,
                "payload ended during same-descriptor hashing",
                phase="payload_hash",
                artifact=artifact,
            )
        payload_digest.update(chunk)
        file_digest.update(chunk)
        cursor += count
        remaining -= count
    observed_payload = payload_digest.hexdigest()
    if observed_payload != header["payload_sha256"]:
        _refuse(
            V1CheckpointRefusalCode.PAYLOAD_DIGEST_MISMATCH,
            "payload SHA-256 differs from the strict frame header",
            phase="payload_hash",
            artifact=artifact,
        )
    after = os.fstat(fd)
    if _stat_signature(after) != _stat_signature(before):
        _refuse(
            V1CheckpointRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "artifact metadata changed during inspection",
            phase="payload_hash",
            artifact=artifact,
        )
    receipt = V1FramedArtifactInspectionReceipt(
        artifact_path=artifact,
        magic=magic.decode("ascii"),
        schema=str(header["schema"]),
        file_size_bytes=int(before.st_size),
        header_size_bytes=header_size,
        payload_offset_bytes=payload_offset,
        payload_size_bytes=payload_size,
        header_sha256=hashlib.sha256(header_bytes).hexdigest(),
        payload_sha256=observed_payload,
        file_sha256=file_digest.hexdigest(),
        regular_file_verified=True,
        no_final_component_symlink=True,
        framing_verified=True,
        payload_integrity_verified=True,
    )
    return _VerifiedFrame(
        fd=fd,
        artifact_path=artifact,
        magic=magic,
        header=header,
        payload_offset=payload_offset,
        payload_size=payload_size,
        stat_signature=_stat_signature(after),
        receipt=receipt,
    )


def _validate_terminal_family_matrix_payload(
    frame: _VerifiedFrame,
    *,
    reference: Mapping[str, Any],
) -> None:
    """Replay exact matrix semantics before any pickle interpreter exists."""

    payload = _pread_exact(
        frame.fd,
        frame.payload_size,
        frame.payload_offset,
        artifact=frame.artifact_path,
    )
    shape = tuple(frame.header["joint_validation_shape"])
    matrix = np.frombuffer(payload, dtype="<f8").reshape(shape)
    if not np.isfinite(matrix).all() or bool(np.any(matrix < 0.0)):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "terminal-family matrix contains nonfinite or negative values",
            phase="sidecar_binding",
            artifact=frame.artifact_path,
        )
    joint_hash = content_hash(
        {"shape": shape, "payload": payload},
        prefix=_TERMINAL_FAMILY_VALUES_PREFIX,
    )
    lane_hashes = tuple(
        content_hash(
            {"shape": row.shape, "payload": row.tobytes(order="C")},
            prefix=_TERMINAL_FAMILY_LANE_VALUES_PREFIX,
        )
        for row in matrix
    )
    counts = (shape[1],) * shape[0]
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
    semantic_hash = content_hash(
        {
            "terminal_request_hash": frame.header["terminal_request_hash"],
            "joint_validation_values_hash": joint_hash,
            "joint_validation_shape": shape,
            "unique_lane_validation_values_hashes": lane_hashes,
            "unique_lane_validation_value_counts": counts,
            "unique_lane_metrics": metrics,
            "common_panel_evaluator_receipt_hash": frame.header[
                "common_panel_evaluator_receipt_hash"
            ],
        },
        prefix=(
            "partition-free-v1-terminal-validation-family-observation-artifact-"
        ),
    )
    if (
        frame.header["joint_validation_values_hash"] != joint_hash
        or tuple(frame.header["unique_lane_validation_values_hashes"])
        != lane_hashes
        or tuple(frame.header["unique_lane_validation_value_counts"]) != counts
        or tuple(frame.header["unique_lane_metrics"]) != metrics
        or frame.header["observation_artifact_semantic_hash"] != semantic_hash
        or reference["terminal_request_hash"]
        != frame.header["terminal_request_hash"]
        or reference["family_freeze_hash"] != frame.header["family_freeze_hash"]
    ):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "terminal-family matrix semantic replay differs from its reference",
            phase="sidecar_binding",
            artifact=frame.artifact_path,
        )


def _validate_terminal_comparison_matrix_payload_v2(
    frame: _VerifiedFrame,
    *,
    reference: Mapping[str, Any],
) -> None:
    """Replay MV1TERM2 rows, paired deltas, and semantic identity inertly."""

    payload = _pread_exact(
        frame.fd,
        frame.payload_size,
        frame.payload_offset,
        artifact=frame.artifact_path,
    )
    shape = tuple(frame.header["joint_validation_shape"])
    matrix = np.frombuffer(payload, dtype="<f8").reshape(shape)
    if not np.isfinite(matrix).all() or bool(np.any(matrix < 0.0)):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "terminal-comparison matrix contains nonfinite or negative values",
            phase="sidecar_binding",
            artifact=frame.artifact_path,
        )
    joint_hash = content_hash(
        {"shape": shape, "payload": payload},
        prefix=_TERMINAL_COMPARISON_VALUES_PREFIX_V2,
    )
    lane_hashes = tuple(
        content_hash(
            {"shape": row.shape, "payload": row.tobytes(order="C")},
            prefix=_TERMINAL_COMPARISON_LANE_VALUES_PREFIX_V2,
        )
        for row in matrix
    )
    counts = (shape[1],) * shape[0]
    metrics = tuple(
        {
            "mean": math.fsum(float(value) for value in row) / len(row),
            "maximum": float(np.max(row)),
            "root_mean_square": math.sqrt(
                math.fsum(float(value) * float(value) for value in row)
                / len(row)
            ),
        }
        for row in matrix
    )
    dense_lane, current_v0_lane, terminal_v1_lane = (
        _validate_terminal_comparison_role_map_v2(
            frame.header,
            artifact=frame.artifact_path,
        )
    )
    pair_rows = (
        (
            "current_v0_comparator_minus_dense_reference",
            matrix[current_v0_lane],
            matrix[dense_lane],
        ),
        (
            "terminal_v1_minus_dense_reference",
            matrix[terminal_v1_lane],
            matrix[dense_lane],
        ),
        (
            "terminal_v1_minus_current_v0_comparator",
            matrix[terminal_v1_lane],
            matrix[current_v0_lane],
        ),
    )
    deltas: dict[str, float] = {}
    pair_hashes: dict[str, str] = {}
    for name, left, right in pair_rows:
        differences = tuple(
            float(left_value) - float(right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
        deltas[name] = math.fsum(differences) / len(differences)
        pair_hashes[name] = content_hash(
            {
                "shape": (len(differences),),
                "payload": np.asarray(differences, dtype="<f8").tobytes(order="C"),
            },
            prefix=_TERMINAL_COMPARISON_PAIRED_VALUES_PREFIX_V2,
        )
    semantic_hash = content_hash(
        {
            "terminal_request_hash": frame.header["terminal_request_hash"],
            "joint_validation_values_hash": joint_hash,
            "joint_validation_shape": shape,
            "unique_lane_validation_values_hashes": lane_hashes,
            "unique_lane_validation_value_counts": counts,
            "unique_lane_metrics": metrics,
            "paired_mean_deltas": deltas,
            "paired_validation_difference_hashes": pair_hashes,
            "paired_delta_derivation": frame.header["paired_delta_derivation"],
            "common_panel_evaluator_receipt_hash": frame.header[
                "common_panel_evaluator_receipt_hash"
            ],
        },
        prefix=(
            "partition-free-v1-terminal-comparison-family-"
            "observation-artifact-v2-"
        ),
    )
    if (
        frame.header["joint_validation_values_hash"] != joint_hash
        or tuple(frame.header["unique_lane_validation_values_hashes"])
        != lane_hashes
        or tuple(frame.header["unique_lane_validation_value_counts"]) != counts
        or tuple(frame.header["unique_lane_metrics"]) != metrics
        or frame.header["paired_mean_deltas"] != deltas
        or frame.header["paired_validation_difference_hashes"] != pair_hashes
        or frame.header["observation_artifact_semantic_hash"] != semantic_hash
        or reference["terminal_request_hash"]
        != frame.header["terminal_request_hash"]
        or reference["family_freeze_hash"] != frame.header["family_freeze_hash"]
    ):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "terminal-comparison matrix replay differs from its reference",
            phase="sidecar_binding",
            artifact=frame.artifact_path,
        )
def _safe_sidecar_parts(
    relative_path: object,
    *,
    checkpoint_name: str,
    kind: str,
    artifact: str,
) -> tuple[str, str]:
    if type(relative_path) is not str:
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_PATH_INVALID,
            "sidecar path is not a string",
            phase="sidecar_resolution",
            artifact=artifact,
        )
    relative_path_string = cast(str, relative_path)
    relative = Path(relative_path_string)
    expected_directory = (
        checkpoint_name + ".acquisitions"
        if kind == "acquisition"
        else (
            checkpoint_name + ".proposal_states"
            if kind == "proposal"
            else (
                checkpoint_name + ".terminal_comparison"
                if kind == "terminal_comparison"
                else checkpoint_name + ".terminal_validation"
            )
        )
    )
    expected_suffix = (
        ".mv1acq"
        if kind == "acquisition"
        else (
            ".mv1prop"
            if kind == "proposal"
            else (".mv1term2" if kind == "terminal_comparison" else ".mv1term")
        )
    )
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_path_string
        or "\x00" in relative_path_string
        or len(relative.parts) != 2
        or relative.parts[0] != expected_directory
        or relative.parts[1] in {"", ".", ".."}
        or not relative.parts[1].endswith(expected_suffix)
    ):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_PATH_INVALID,
            "sidecar path leaves its exact checkpoint-owned directory",
            phase="sidecar_resolution",
            artifact=artifact,
        )
    return expected_directory, relative.parts[1]


def _open_and_verify_sidecar(
    *,
    checkpoint_parent_fd: int,
    checkpoint_parent_path: Path,
    checkpoint_name: str,
    reference: dict[str, Any],
    kind: str,
) -> tuple[_VerifiedFrame, int]:
    relative = str(reference["relative_path"])
    artifact_path = str(checkpoint_parent_path / relative)
    directory_name, filename = _safe_sidecar_parts(
        relative,
        checkpoint_name=checkpoint_name,
        kind=kind,
        artifact=artifact_path,
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(
            directory_name,
            directory_flags,
            dir_fd=checkpoint_parent_fd,
        )
    except FileNotFoundError as exc:
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_MISSING,
            "referenced sidecar directory is absent",
            phase="sidecar_resolution",
            artifact=artifact_path,
        )
        raise AssertionError("unreachable") from exc
    except (OSError, ValueError) as exc:
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_PATH_INVALID,
            "referenced sidecar directory is symlinked or invalid",
            phase="sidecar_resolution",
            artifact=artifact_path,
        )
        raise AssertionError("unreachable") from exc
    try:
        sidecar_fd = _open_regular_at(
            directory_fd,
            filename,
            artifact=artifact_path,
            missing_code=V1CheckpointRefusalCode.SIDECAR_MISSING,
        )
    except BaseException:
        os.close(directory_fd)
        raise
    os.close(directory_fd)
    expected_magic = (
        frozenset({_ACQUISITION_MAGIC})
        if kind == "acquisition"
        else (
            frozenset({_PROPOSAL_STATE_MAGIC})
            if kind == "proposal"
            else (
                frozenset({_TERMINAL_COMPARISON_MATRIX_MAGIC_V2})
                if kind == "terminal_comparison"
                else frozenset({_TERMINAL_FAMILY_MATRIX_MAGIC})
            )
        )
    )
    try:
        frame = _verify_frame(
            sidecar_fd,
            artifact=artifact_path,
            allowed_magics=expected_magic,
            sidecar=True,
        )
        reference_size = (
            reference["artifact_size_bytes"]
            if kind in {"terminal_family", "terminal_comparison"}
            else reference["framed_size_bytes"]
        )
        if frame.receipt.file_size_bytes != reference_size:
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "sidecar framed size differs from the checkpoint reference",
                phase="sidecar_binding",
                artifact=artifact_path,
            )
        if kind == "terminal_family":
            observed = {
                "schema": _TERMINAL_FAMILY_ARTIFACT_REFERENCE_SCHEMA,
                "relative_path": reference["relative_path"],
                "artifact_schema": frame.header["schema"],
                "artifact_sha256": frame.receipt.file_sha256,
                "artifact_size_bytes": frame.receipt.file_size_bytes,
                "terminal_request_hash": frame.header["terminal_request_hash"],
                "family_freeze_hash": frame.header["family_freeze_hash"],
                "joint_validation_values_hash": frame.header[
                    "joint_validation_values_hash"
                ],
                "joint_validation_shape": frame.header["joint_validation_shape"],
                "unique_lane_validation_values_hashes": frame.header[
                    "unique_lane_validation_values_hashes"
                ],
                "unique_lane_validation_value_counts": frame.header[
                    "unique_lane_validation_value_counts"
                ],
                "unique_lane_metrics": frame.header["unique_lane_metrics"],
                "common_panel_evaluator_receipt_hash": frame.header[
                    "common_panel_evaluator_receipt_hash"
                ],
                "observation_artifact_semantic_hash": frame.header[
                    "observation_artifact_semantic_hash"
                ],
                "terminal_observation_hash": reference["terminal_observation_hash"],
            }
        elif kind == "terminal_comparison":
            observed = {
                "schema": _TERMINAL_COMPARISON_ARTIFACT_REFERENCE_SCHEMA_V2,
                "relative_path": reference["relative_path"],
                "artifact_schema": frame.header["schema"],
                "artifact_sha256": frame.receipt.file_sha256,
                "artifact_size_bytes": frame.receipt.file_size_bytes,
                "terminal_request_hash": frame.header["terminal_request_hash"],
                "family_freeze_hash": frame.header["family_freeze_hash"],
                "roles": frame.header["roles"],
                "role_to_unique_lane": frame.header["role_to_unique_lane"],
                "joint_validation_values_hash": frame.header[
                    "joint_validation_values_hash"
                ],
                "joint_validation_shape": frame.header["joint_validation_shape"],
                "unique_lane_validation_values_hashes": frame.header[
                    "unique_lane_validation_values_hashes"
                ],
                "unique_lane_validation_value_counts": frame.header[
                    "unique_lane_validation_value_counts"
                ],
                "unique_lane_metrics": frame.header["unique_lane_metrics"],
                "paired_mean_deltas": frame.header["paired_mean_deltas"],
                "paired_validation_difference_hashes": frame.header[
                    "paired_validation_difference_hashes"
                ],
                "paired_delta_derivation": frame.header[
                    "paired_delta_derivation"
                ],
                "common_panel_evaluator_receipt_hash": frame.header[
                    "common_panel_evaluator_receipt_hash"
                ],
                "observation_artifact_semantic_hash": frame.header[
                    "observation_artifact_semantic_hash"
                ],
                "terminal_observation_hash": reference["terminal_observation_hash"],
            }
        else:
            observed = {
                key: value
                for key, value in frame.header.items()
                if key != "security_boundary"
            }
            observed["relative_path"] = reference["relative_path"]
            observed["framed_size_bytes"] = frame.receipt.file_size_bytes
        if observed != reference:
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "sidecar header differs from the exact checkpoint reference",
                phase="sidecar_binding",
                artifact=artifact_path,
            )
        return frame, sidecar_fd
    except BaseException:
        os.close(sidecar_fd)
        raise


class _PreadPayloadReader(io.RawIOBase):
    """Bounded seekable view over one already-verified open descriptor."""

    def __init__(self, frame: _VerifiedFrame) -> None:
        super().__init__()
        self._fd = frame.fd
        self._offset = frame.payload_offset
        self._size = frame.payload_size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError("unsupported seek origin")
        if position < 0 or position > self._size:
            raise ValueError("payload seek leaves verified frame")
        self._position = position
        return position

    def readinto(self, buffer: Any) -> int:
        if self._position >= self._size:
            return 0
        view = memoryview(buffer).cast("B")
        try:
            count = min(view.nbytes, self._size - self._position)
            payload = os.pread(self._fd, count, self._offset + self._position)
            if len(payload) != count:
                raise OSError("verified payload changed before trusted decode")
            view[:count] = payload
            self._position += count
            return count
        finally:
            view.release()


class _HistoricalCarrier:
    """Inert target for a historical repository global during trusted decode."""

    __slots__ = ("constructor_args", "constructor_kwargs", "state")
    constructor_args: tuple[Any, ...]
    constructor_kwargs: dict[str, Any]
    state: Any
    historical_module: str = ""
    historical_name: str = ""

    def __new__(cls, *args: Any, **kwargs: Any) -> _HistoricalCarrier:
        instance = super().__new__(cls)
        instance.constructor_args = tuple(args)
        instance.constructor_kwargs = dict(kwargs)
        instance.state = None
        return instance

    def __setstate__(self, state: Any) -> None:
        self.state = state


_CARRIER_TYPES: dict[tuple[str, str], type[_HistoricalCarrier]] = {}


def _carrier_type(module: str, name: str) -> type[_HistoricalCarrier]:
    key = (module, name)
    observed = _CARRIER_TYPES.get(key)
    if observed is not None:
        return observed
    safe_name = "_Historical_" + "".join(
        character if character.isalnum() else "_" for character in name
    )
    created = type(
        safe_name,
        (_HistoricalCarrier,),
        {
            "__slots__": (),
            "historical_module": module,
            "historical_name": name,
        },
    )
    _CARRIER_TYPES[key] = created
    return created


class _RawCompatibilityUnpickler(pickle.Unpickler):
    """Decode trusted bytes while replacing every repository class by a carrier."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "mechanistic_systems" or module.startswith("mechanistic_systems."):
            return _carrier_type(module, name)
        # Nonrepository globals are still resolved by pickle.  This is why the
        # API requires trusted-local opt-in and explicitly makes no pickle
        # safety claim.  NumPy reconstruction and cloudpickle's mappingproxy
        # helper need these stable external globals for historical payloads.
        return super().find_class(module, name)


def _raw_decode(frame: _VerifiedFrame) -> Any:
    try:
        raw = _PreadPayloadReader(frame)
        buffered = io.BufferedReader(raw, buffer_size=1 << 20)
        try:
            value = _RawCompatibilityUnpickler(buffered).load()
            if buffered.read(1):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "trusted pickle payload contains trailing serialized bytes",
                    phase="trusted_raw_decode",
                    artifact=frame.artifact_path,
                )
            return value
        finally:
            buffered.close()
    except V1CheckpointInspectionRefusal:
        raise
    except BaseException as exc:
        _refuse(
            V1CheckpointRefusalCode.TRUSTED_DECODE_FAILED,
            f"trusted compatibility decode failed: {type(exc).__name__}",
            phase="trusted_raw_decode",
            artifact=frame.artifact_path,
        )
        raise AssertionError("unreachable") from exc


# Historical slotted field maps are frozen from the sealed v15/v18/v56 source
# generations.  Cohort-changing types receive explicit maps below; common
# types retain the stable pre-v29 map.  These names are data coordinates only:
# no current class is instantiated or assigned.
_COMMON_RAW_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "MeasurementProvenance": (
        "producer_id",
        "producer_version",
        "code_hash",
        "parent_request_ids",
        "notes",
    ),
    "HardMask": ("route_axis_hash", "route_count", "packed_bits"),
    "V1OptionACampaignBinding": ("contract", "coordinates", "registration", "schema"),
    "V1OwnerRegistration": (
        "registration_id",
        "referent_id",
        "route_axis_hash",
        "context_population_hash",
        "calibration_identity",
        "certificate_family",
        "cross_query_shrinkage",
        "a1_coordinates_hash",
        "a1_proof_hash",
        "a2_coordinates_hash",
        "owner_signature",
        "owner_signature_date",
        "owner_signature_verification_hash",
        "campaign_contract_hash",
        "metadata",
        "schema",
    ),
    "V1RuntimeExecutionConfig": (
        "executor_identity_hash",
        "source_seal_hash",
        "environment_hash",
        "provider_provenance",
        "physical_batch_size",
        "prefetch_depth",
        "metadata",
        "schema",
    ),
    "V1RuntimePendingLook": (
        "controller_snapshot_hash",
        "active_candidate_hash",
        "look_index",
        "sampling_plan",
        "request",
        "evidence_packet",
        "schema",
    ),
    "V1RuntimePendingPairedBCEChunk": (
        "controller_snapshot_hash",
        "active_candidate_hash",
        "registration_hash",
        "prior_paired_bce_state_hash",
        "request",
        "context_ids_in_registered_slice",
        "dynamic_oracle_receipt_hash",
        "betting_replay_reference_hash",
        "row_payload_externalized",
        "outcome_observed",
        "candidate_or_mask_authority",
        "schema",
    ),
    "V1RuntimeState": (
        "config",
        "controller",
        "entropy_tape",
        "pending_look",
        "terminal_freeze",
        "terminal_validation",
        "completion",
        "performance",
        "schema",
    ),
    "_V1ControllerCheckpointProjection": (
        "binding",
        "controller_run_id",
        "initial_mask",
        "current_mask",
        "phase",
        "queue_cursor",
        "active_candidate",
        "accepted_trajectory",
        "attempt_receipts",
        "transition_receipts",
        "acquisition_receipt_hashes",
        "globally_used_draw_ids",
        "stop_reason",
        "logical_sequence",
        "controller_schema",
        "acquisition_sidecar",
        "controller_state_hash",
        "controller_snapshot_hash",
        "accepted_oracle_trajectory",
        "pareto_milestone_registration",
        "live_exact_prefix_rung_registration",
        "pending_pareto_milestone",
        "pareto_milestone_receipt",
        "fqr_no_candidate_horizon_witness",
        "label_native_runtime_plan_checkpoint_payload",
        "label_native_runtime_plan_hash",
        "label_native_runtime_plan_hashes",
        "label_native_rejected_plan_indices",
        "label_native_selected_acquisition_receipt_hash",
        "schema",
    ),
    "_V1RuntimeCheckpointJournal": (
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
        "schema",
    ),
    "V1RuntimeAcquisitionSidecarReceipt": (
        "relative_path",
        "payload_sha256",
        "payload_size_bytes",
        "framed_size_bytes",
        "acquisition_hash",
        "acquisition_receipt_hash",
        "campaign_binding_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "schema",
    ),
    "V1RuntimeProposalStateSidecarReceipt": _PROPOSAL_REFERENCE_FIELDS_IN_ORDER,
    "V1RuntimeProposalStateSidecarPayload": (
        "proposal_progress_payload",
        "pending_proposal_acquisition_payload",
        "proposal_progress_hash",
        "pending_proposal_acquisition_hash",
        "proposal_backend_identity_hash",
        "representation_hash",
        "schema",
    ),
    "V1ProposalFreezeDecisionReceipt": (
        "state_hash",
        "phase_evidence_state_hash",
        "logical_sequence",
        "sufficient_state_hash",
        "verified_receipt_graph_hash",
        "freeze_criterion_hash",
        "criterion_evaluator_identity_hash",
        "criterion_verification_receipt_hash",
        "evidence_hash",
        "evidence_evaluation_hash",
        "action",
        "candidate_identity_hash",
        "refusal_reason",
        "phase",
        "scheduler_selection_authorizes_freeze",
        "candidate_freeze_authority",
        "mask_commit_authority",
        "validation_outcomes_observed",
        "train_optimization_data_only",
        "schema",
    ),
    "V1ControllerSnapshotReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "phase",
        "current_mask_hash",
        "acquisition_generation_index",
        "queue_cursor",
        "accepted_step_count",
        "candidate_attempt_count",
        "cumulative_removed_count",
        "active_candidate_id",
        "active_step_freeze_hash",
        "active_ledger_hash",
        "accepted_trajectory_hash",
        "acquisition_receipt_hash",
        "acquisition_receipt_hashes",
        "attempt_receipt_hashes",
        "transition_receipt_hashes",
        "globally_used_draw_ids_hash",
        "stop_reason",
        "logical_sequence",
        "fqr_no_candidate_horizon_witness_hash",
        "label_native_runtime_plan_hash",
        "label_native_runtime_plan_hashes",
        "label_native_rejected_plan_indices",
        "label_native_selected_acquisition_receipt_hash",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1SignedWorkSixRoleEndpointCoordinate": (
        "pre_C_priority",
        "branch",
        "endpoint_index",
        "endpoint_label",
        "action_hash",
        "canonical_candidate_identity_hash",
        "cumulative_removed_count",
        "crosswalk_receipt_hash",
        "hard_replay_endpoint_receipt_hash",
    ),
    "V1SignedWorkSixRoleEndpointCertificate": (
        "coordinate",
        "f_native_interval",
        "q_j_interval",
        "r_half_interval",
        "r_one_interval",
        "d_half_interval",
        "d_one_interval",
        "chart_half_interval",
        "chart_one_interval",
        "tau_l",
        "direct_D_status",
        "chart_transport_status",
        "admission_basis",
        "signed_F_used_as_feasibility_veto",
        "schema",
    ),
    "V1SignedWorkLabelNativeCActionCrosswalkReceipt": (
        "branch",
        "compact_C_endpoint_index",
        "source_endpoint_label",
        "source_action_receipt",
        "endpoint_decision_receipt_hash",
        "hard_fq_endpoint_receipt_hash",
        "resource_nominal",
        "resource_lower",
        "resource_upper",
        "signed_f_nominal",
        "signed_f_lower",
        "signed_f_upper",
        "route_axis_hash",
        "route_count",
        "parent_mask_hash",
        "canonical_physical_identity_hash",
        "canonical_physical_occurrence",
        "first_occurrence_branch",
        "first_occurrence_source_endpoint_label",
        "exact_duplicate_of_earlier_occurrence",
        "hard_fq_status",
        "physical_identity_scheme",
        "endpoint_hard_mask_materialized",
        "packed_mask_payload_materialized",
        "route_scale_D2H_bytes",
        "C_outcomes_seen",
        "C_observation_count",
        "authority_scope",
        "schema",
    ),
    "V1DeviceShellFrozenActionReceipt": (
        "endpoint_label",
        "shell_count",
        "cumulative_removed_count",
        "removed_route_index_hash",
        "resulting_mask_packed_sha256",
        "resulting_mask_hash",
        "packed_action_hash",
        "schema",
    ),
    "V1FQRNoCandidateHorizonProofReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "chart_identity_hash",
        "fqr_graph_hash",
        "fqr_policy_hash",
        "fqr_branch_family_design_hash",
        "prefrozen_branch_family_hash",
        "complete_train_population_hash",
        "causal_generation_frame_hash",
        "causal_discovery_stream_hash",
        "causal_discovery_prefix_freeze_hash",
        "causal_certification_stream_hash",
        "sampled_d_sufficient_state_hash",
        "fqr_acquisition_design_hash",
        "fqr_progress_receipt_hash",
        "fqr_readiness_receipt_hash",
        "fqr_accumulator_snapshot_hash",
        "fqr_transaction_chain_hash",
        "proposal_sufficient_state_hash",
        "proposal_sufficient_state_descriptor_hash",
        "fqr_c_state_payload_hash",
        "native_execution_binding_chain_hash",
        "registered_branch_axis",
        "branch_negative_evidence_hashes",
        "shared_observed_counts_by_stratum",
        "readiness_status",
        "family_refusal_code",
        "minimum_nonempty_action_count",
        "all_registered_branches_decision_stable",
        "all_frozen_endpoints_lower_refuted",
        "minimum_nonempty_action_registered_in_every_branch",
        "adaptive_optional_stopping_valid",
        "candidate_queue_materialized",
        "calibration_K_opened",
        "paired_oracle_O_opened",
        "mask_mutated",
        "validation_outcomes_observed",
        "candidate_freeze_authority",
        "calibration_or_oracle_authority",
        "mask_commit_authority",
        "terminal_validation_authority",
        "pareto_claim_authority",
        "schema",
    ),
    "V1ControllerProposalHorizonWitness": (
        "proof",
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "pre_stop_controller_state_hash",
        "pre_stop_controller_snapshot_hash",
        "pre_stop_logical_sequence",
        "terminal_proposal_progress_hash",
        "terminal_proposal_frontier_hash",
        "proposal_joint_state_hash",
        "proposal_phase_evidence_state_hash",
        "proposal_sufficient_state_hash",
        "proposal_sufficient_state_descriptor_hash",
        "verified_receipt_graph_hash",
        "freeze_evaluation_receipt_hash",
        "freeze_evidence_receipt_hash",
        "freeze_decision_receipt_hash",
        "post_stop_logical_sequence",
        "stop_reason",
        "proposal_progress_retention_required",
        "candidate_queue_materialized",
        "calibration_K_opened",
        "paired_oracle_O_opened",
        "mask_mutated",
        "validation_outcomes_observed",
        "mask_commit_authority",
        "terminal_validation_authority",
        "pareto_claim_authority",
        "schema",
    ),
    "V1RuntimeCompletedProposalClosureReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "acquisition_generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "pre_attach_controller_state_hash",
        "pre_attach_controller_snapshot",
        "proposal_progress_hash",
        "proposal_frontier_hash",
        "proposal_joint_state_hash",
        "proposal_phase_evidence_state_hash",
        "proposal_sufficient_state_hash",
        "proposal_logical_sequence",
        "verified_receipt_graph_hash",
        "freeze_criterion_hash",
        "freeze_evidence_hash",
        "freeze_evidence_evaluation_hash",
        "criterion_verification_receipt_hash",
        "freeze_decision",
        "acquisition_hash",
        "acquisition_receipt_hash",
        "ordered_candidate_ids",
        "candidate_identity_hash",
        "post_attach_controller_state_hash",
        "post_attach_controller_snapshot",
        "proposal_order_alpha_schedule_hash",
        "proposal_order_alpha_ledger_before_hash",
        "proposal_order_design_hash",
        "proposal_order_base_serfling_design_hash",
        "proposal_order_alpha_ledger_after_hash",
        "proposal_order_alpha_ledger_after_replay",
        "proposal_order_envelope_certificate_hash",
        "proposal_order_action_family_registry_hash",
        "sampled_v7_D_alpha_family_plan_hash",
        "sampled_v7_D_alpha_family_registry_receipt_hash",
        "sampled_v7_D_alpha_family_registry_replay",
        "sampled_v7_D_alpha_head_CAS_receipt_hash",
        "sampled_v7_D_alpha_published_ledger_head_hash",
        "sampled_v7_D_alpha_durable_controller_head_closed",
        "validation_outcomes_observed",
        "train_optimization_data_only",
        "mask_commit_authority",
        "schema",
    ),
    "V1ControllerState": (
        "binding",
        "controller_run_id",
        "initial_mask",
        "current_mask",
        "phase",
        "acquisition",
        "queue_cursor",
        "label_native_runtime_plan",
        "label_native_runtime_plan_hashes",
        "label_native_rejected_plan_indices",
        "active_candidate",
        "accepted_trajectory",
        "attempt_receipts",
        "transition_receipts",
        "acquisition_receipt_hashes",
        "globally_used_draw_ids",
        "pareto_milestone_registration",
        "live_exact_prefix_rung_registration",
        "pending_pareto_milestone",
        "pareto_milestone_receipt",
        "fqr_no_candidate_horizon_witness",
        "stop_reason",
        "logical_sequence",
        "schema",
    ),
    "V1ParetoMilestoneRegistration": (
        "campaign_contract_hash",
        "route_axis_hash",
        "route_count",
        "comparator_retained_route_count",
        "comparator_registration_artifact_hash",
        "validation_independent",
        "comparator_mask_supplied",
        "comparator_order_supplied",
        "comparator_trajectory_supplied",
        "schema",
    ),
    "V1ExactPrefixRungRegistration": (
        "campaign_contract_hash",
        "milestone_registration_hash",
        "acquisition_receipt_hash",
        "generation_index",
        "parent_mask_hash",
        "parent_retained_route_count",
        "comparator_retained_route_count",
        "required_removed_route_count",
        "queue_index",
        "candidate_id",
        "candidate_mask_hash",
        "removed_route_index_hash",
        "v1_order_receipt_hash",
        "validation_independent",
        "exact_v1_order_prefix",
        "comparator_mask_used",
        "comparator_order_used",
        "comparator_trajectory_used",
        "oracle_outcomes_opened",
        "schema",
    ),
    "V1ParetoMilestonePending": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "milestone_registration_hash",
        "comparator_retained_route_count",
        "milestone_mask_hash",
        "mask_lineage_hashes",
        "transition_receipt_hashes",
        "accepted_trajectory_hash",
        "accepted_step_count",
        "candidate_attempt_count",
        "milestone_transition_receipt_hash",
        "logical_sequence",
        "validation_independent",
        "terminal_validation_opened",
        "schema",
    ),
    "V1ParetoMilestoneCheckpointBinding": (
        "pending_milestone_hash",
        "controller_state_hash",
        "controller_snapshot_receipt_hash",
        "checkpoint_artifact_hash",
        "checkpoint_manifest_hash",
        "checkpoint_codec_schema",
        "atomic_checkpoint_write_completed",
        "restore_identity_verified",
        "schema",
    ),
    "V1ParetoMilestoneReceipt": (
        "pending",
        "checkpoint_binding",
        "continue_after_milestone",
        "validation_independent",
        "terminal_validation_opened",
        "comparator_mask_used",
        "comparator_order_used",
        "comparator_trajectory_used",
        "schema",
    ),
    "V1ActiveCandidate": (
        "acquisition_receipt_hash",
        "queue_index",
        "proposal",
        "threshold",
        "sampling_law",
        "step_freeze",
        "ledger",
        "paired_bce_registration",
        "paired_bce_state",
        "schema",
    ),
    "V1FrozenAffineCalibration": (
        "mask_hash",
        "query_id",
        "query_hash",
        "target_class_id",
        "scale",
        "offset",
        "calibration_hash",
        "calibration_state_id",
        "calibration_source_hash",
        "calibration_context_axis_hash",
        "source_label_map_hash",
        "model_state_hash",
        "calibration_schema",
        "fit_receipt",
        "fit_receipt_hash",
        "validation_independent",
        "terminal_rows_used_for_fit",
        "terminal_outcomes_opened",
        "schema",
    ),
    "V1CandidateCalibrationFreeze": (
        "campaign_contract_hash",
        "acquisition_receipt_hash",
        "queue_index",
        "candidate_id",
        "proposal_parent_mask_hash",
        "dense_referent_mask_hash",
        "candidate_mask_hash",
        "calibration_policy_id",
        "dense_calibration",
        "candidate_calibration",
        "calibration_recipe",
        "calibration_recipe_hash",
        "calibration_context_ids",
        "calibration_context_ids_hash",
        "calibration_context_selection_hash",
        "calibration_context_selection_policy",
        "calibration_context_selection_policy_hash",
        "source_receipt_hash",
        "metadata",
        "validation_independent",
        "terminal_rows_used_for_fit",
        "terminal_outcomes_opened",
        "schema",
    ),
    "V1AcceptedOracleTrajectoryEntry": (
        "protocol",
        "accepted_step_index",
        "candidate_attempt_index",
        "acquisition_receipt_hash",
        "candidate_id",
        "parent_mask_hash",
        "candidate_mask_hash",
        "step_freeze_hash",
        "evidence_ledger_hash",
        "decision_receipt_hash",
        "metadata",
        "schema",
        "candidate_calibration_freeze",
    ),
    "V1AcceptedOracleTrajectoryLedger": ("entries", "schema"),
    "V1PairedBCEOracleRoleSeparationReceipt": (
        "discovery_fold_identity_hash",
        "fq_certification_fold_identity_hash",
        "calibration_fold_identity_hash",
        "oracle_fold_identity_hash",
        "discovery_context_set_hash",
        "fq_certification_context_set_hash",
        "calibration_context_set_hash",
        "oracle_target_population_identity_hash",
        "oracle_complement_population_identity_hash",
        "discovery_context_count",
        "fq_certification_context_count",
        "calibration_context_count",
        "oracle_target_context_count",
        "oracle_complement_context_count",
        "disjointness_verifier_identity_hash",
        "oracle_disjoint_from_discovery",
        "oracle_disjoint_from_fq_certification",
        "oracle_disjoint_from_calibration",
        "oracle_strata_disjoint",
        "oracle_untouched_before_candidate_freeze",
        "candidate_calibration_excludes_oracle",
        "schema",
    ),
    "V1PairedBCEStratumRegistration": (
        "stratum_id",
        "stratum_weight",
        "population_size",
        "population_identity_hash",
        "sealed_order_hash",
        "sealed_permutation_receipt_hash",
        "bound_certificate_hash",
        "paired_loss_lower_bound",
        "paired_loss_upper_bound",
        "bound_frozen_before_rows",
        "without_replacement",
        "schema",
    ),
    "V1PairedBCEDynamicOracleReceipt": (
        "generation_frame_hash",
        "oracle_sampling_frame_receipt_hash",
        "generation_index",
        "candidate_attempt_index",
        "dense_reference_hash",
        "candidate_mask_hash",
        "candidate_calibration_freeze_hash",
        "dense_affine_calibration_hash",
        "candidate_affine_calibration_hash",
        "calibration_prefix_freeze_hash",
        "loss_ceiling_identity_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "full_population_identity_hashes",
        "full_population_stratum_counts",
        "calibration_prefix_membership_hashes",
        "calibration_prefix_stratum_counts",
        "exact_calibration_scaled_loss_sums",
        "calibration_paired_row_payload_hashes",
        "calibration_paired_loss_evaluator_receipt_hash",
        "oracle_population_identity_hashes",
        "oracle_population_stratum_counts",
        "oracle_sealed_order_hashes",
        "oracle_permutation_receipt_hashes",
        "calibration_prefix_frozen_before_oracle_open",
        "oracle_is_exact_full_population_minus_calibration_prefix",
        "candidate_and_calibrations_immutable_after_oracle_open",
        "replacement_requires_new_attempt_alpha_coordinate",
        "terminal_residual_census_plus_exact_k_is_full_population_census",
        "schema",
    ),
    "V1PairedBCEAcquisitionRegistration": (
        "registration_id",
        "campaign_contract_hash",
        "candidate_mask_hash",
        "dense_reference_hash",
        "candidate_calibration_freeze_hash",
        "paired_loss_kind",
        "loss_ceiling",
        "loss_ceiling_identity_hash",
        "source_population_identity_hash",
        "oracle_fold_identity_hash",
        "oracle_role_separation_receipt",
        "strata",
        "oracle_order_entropy_commitment_hash",
        "sampling_law_identity_hash",
        "sealed_order_slice_verifier_identity_hash",
        "candidate_attempt_alpha_axis_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "maximum_candidate_attempts",
        "oracle_familywise_alpha",
        "alpha_committed_before_attempt",
        "attempt_alpha_allocation",
        "registered_look_stratum_counts",
        "alpha_by_look",
        "dynamic_oracle_receipt",
        "stratum_populations_disjoint",
        "identities_frozen_before_rows",
        "alpha_allocation_nonrecycling",
        "alpha_nonrecycling_policy",
        "partial_look_alpha_policy",
        "sampling_law_kind",
        "predictable_stratum_selection_policy",
        "group_sequential_fallback_enabled",
        "partial_evidence_method",
        "partial_evidence_ceiling",
        "authority_scope",
        "schema",
    ),
    "V1PairedBCEAcquisitionState": (
        "registration_hash",
        "candidate_attempt_index",
        "stratum_counts",
        "exact_scaled_loss_sums",
        "observed_minima",
        "observed_maxima",
        "completed_look_count",
        "completed_chunk_count",
        "last_chunk_receipt_hash",
        "receipt_chain_hash",
        "dynamic_oracle_receipt_hash",
        "schema",
    ),
    "V1PairedBCEChunkRequest": (
        "registration_hash",
        "prior_state_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "request_index",
        "stratum_id",
        "sealed_order_hash",
        "prefix_start",
        "prefix_stop",
        "context_slice_hash",
        "sealed_slice_registration_hash",
        "target_registered_look_index",
        "target_registered_look_stratum_counts",
        "row_provider_identity_hash",
        "paired_loss_evaluator_identity_hash",
        "dynamic_oracle_receipt_hash",
        "pending_before_outcomes",
        "outcome_fields_absent",
        "schema",
    ),
    "V1PairedBCEChunkReceipt": (
        "request_hash",
        "registration_hash",
        "prior_state_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "request_index",
        "stratum_id",
        "prefix_start",
        "prefix_stop",
        "context_slice_hash",
        "sealed_slice_registration_hash",
        "paired_row_payload_hash",
        "exact_scaled_loss_sum",
        "observed_minimum",
        "observed_maximum",
        "paired_loss_evaluator_receipt_hash",
        "dynamic_oracle_receipt_hash",
        "all_rows_finite",
        "finite_bound_checked_when_available",
        "full_chunk_completed",
        "schema",
    ),
    "V1PairedBCEDecision": (
        "registration_hash",
        "state_hash",
        "candidate_attempt_alpha_axis_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "status",
        "reason",
        "stratum_counts",
        "population_stratum_counts",
        "stratum_point_means",
        "stratum_confidence_intervals",
        "weighted_point_mean",
        "weighted_confidence_lower",
        "weighted_confidence_upper",
        "loss_ceiling",
        "evaluated_registered_look_index",
        "look_alpha",
        "cumulative_registered_alpha_through_look",
        "full_census",
        "exact_finite_set_decision",
        "partial_confidence_claim_bearing",
        "calibration_prefix_stratum_counts",
        "residual_oracle_stratum_counts",
        "residual_oracle_population_stratum_counts",
        "dynamic_oracle_receipt_hash",
        "mask_commit_authority",
        "resource_exhausted",
        "exact_census_method",
        "partial_evidence_method",
        "partial_evidence_ceiling",
        "alpha_nonrecycling_policy",
        "authority_scope",
        "schema",
    ),
    "V1CandidateDamageThreshold": (
        "campaign_contract_hash",
        "acquisition_receipt_hash",
        "candidate_id",
        "parent_mask_hash",
        "candidate_mask_hash",
        "maximum_mean_damage",
        "damage_target_derivation_hash",
        "threshold_source_receipt_hash",
        "metadata",
        "schema",
    ),
    "V1CertificateFreeze": (
        "registration_hash",
        "parent_mask_hash",
        "candidate_mask_hash",
        "candidate_definition_hash",
        "null_id",
        "loss_id",
        "sampling_law_hash",
        "importance_weight_hash",
        "certificate_factor_hash",
        "pre_draw_filtration_hash",
        "allowed_roles",
        "metadata",
        "schema",
    ),
    "V1EvidencePacket": (
        "packet_id",
        "role",
        "draw_ids",
        "source_context_ids",
        "registration_hash",
        "parent_mask_hash",
        "candidate_mask_hash",
        "candidate_definition_hash",
        "null_id",
        "loss_id",
        "sampling_law_hash",
        "importance_weight_hash",
        "certificate_factor_hash",
        "pre_draw_filtration_hash",
        "outcome_observed",
        "metadata",
        "schema",
    ),
    "V1CertificateLedger": ("freeze", "admitted_packets", "metadata", "schema"),
    "V1StagedAlphaEntropyTape": (
        "campaign_contract_hash",
        "controller_run_id",
        "executor_identity_hash",
        "maximum_candidate_count",
        "draws_per_candidate",
        "entropy_hex",
        "entropy_source_id",
        "schema",
    ),
    "V1StagedAlphaSamplingPlan": (
        "population_context_ids",
        "sampling_masses_u64",
        "entropy_hex",
        "entropy_source_receipt_hash",
        "entropy_source_id",
        "birth_binding",
        "sampling_law_hash",
        "draw_ids",
        "source_context_ids",
        "algorithm_id",
        "schema",
    ),
    "V1StagedAlphaPacketEnvelope": (
        "step_freeze_hash",
        "evidence_packet_hash",
        "request_id",
        "outcome_integrity_hash",
        "look_index",
        "draw_ids",
        "source_context_ids",
        "weighted_damage_payload",
        "target_probability_hash",
        "sampling_probability_hash",
        "importance_weight_hash",
        "sampling_plan_hash",
        "provider_contract_hash",
        "envelope_receipt_hash",
        "schema",
    ),
    "V1StagedAlphaLedger": ("step_freeze", "packets", "schema"),
    "V1StagedAlphaDecision": (
        "ledger_hash",
        "coordinates_hash",
        "campaign_contract_hash",
        "step_index",
        "look_index",
        "count",
        "look_alpha",
        "sample_mean_damage",
        "lower_mean_damage",
        "upper_mean_damage",
        "maximum_mean_damage",
        "action",
        "reason",
        "decision_receipt_hash",
        "proof_artifact_hash",
        "owner_signature_verified",
        "claim_execution_certified",
        "schema",
    ),
    "V1StagedAlphaTrajectoryLedger": ("step_ledgers", "step_decisions", "schema"),
    "V1AcquisitionGenerationReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "registration_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "acquisition_hash",
        "geometry_state_hash",
        "fd_adapter_result_hash",
        "correction_result_hash",
        "chart_hash",
        "numeric_state_hash",
        "finite_domain_hash",
        "constraints_hash",
        "composition_result_hash",
        "ordered_candidate_ids",
        "source_measurement_hashes",
        "refresh_reason",
        "logical_sequence",
        "proposal_only",
        "claim_execution_certified",
        "mask_commit_authority",
        "metadata",
        "schema",
    ),
    "V1CandidateAttemptReceipt": (
        "campaign_contract_hash",
        "controller_run_id",
        "candidate_attempt_index",
        "accepted_step_index",
        "acquisition_generation_index",
        "acquisition_receipt_hash",
        "queue_index",
        "candidate_id",
        "parent_mask_hash",
        "candidate_mask_hash",
        "step_freeze_hash",
        "ledger_hash",
        "decision_receipt_hash",
        "action",
        "mask_mutated",
        "resulting_current_mask_hash",
        "draw_ids",
        "source_context_ids",
        "reason",
        "logical_sequence",
        "claim_execution_certified",
        "mask_commit_authority",
        "metadata",
        "schema",
    ),
    "V1MaskTransitionReceipt": (
        "campaign_contract_hash",
        "controller_run_id",
        "transition_index",
        "accepted_step_index",
        "candidate_attempt_index",
        "candidate_id",
        "parent_mask_hash",
        "child_mask_hash",
        "removed_route_indices",
        "accepted_decision_receipt_hash",
        "accepted_ledger_hash",
        "acquisition_receipt_hash",
        "invalidated_chart_hash",
        "invalidated_composition_hash",
        "commit_capability_consumption_hash",
        "logical_sequence",
        "single_use_consumed",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalFreezeReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "controller_snapshot_hash",
        "terminal_mask_hash",
        "accepted_trajectory_hash",
        "extraction_state_hash",
        "extraction_calibration_hash",
        "dense_reference_hash",
        "terminal_validation_population_hash",
        "acquisition_receipt_hashes",
        "attempt_receipt_hashes",
        "transition_receipt_hashes",
        "stop_reason",
        "freeze_sequence",
        "terminal_rows_visible",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalValidationReceipt": (
        "terminal_freeze_hash",
        "terminal_request_hash",
        "terminal_outcome_hash",
        "validation_values_hash",
        "validation_value_count",
        "calibration_identity",
        "terminal_mask_hash",
        "reveal_sequence",
        "metrics",
        "single_reveal",
        "fed_back_into_selection",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalValidationFamilyFreezeReceipt": (
        "terminal_freeze_hash",
        "terminal_validation_population_hash",
        "terminal_validation_context_ids_hash",
        "terminal_validation_context_count",
        "terminal_metric_semantics_hash",
        "terminal_evaluator_identity_hash",
        "dense_reference_hash",
        "dense_mask_hash",
        "route_axis_hash",
        "route_count",
        "pareto_milestone_receipt",
        "pareto_milestone_archive_relative_path",
        "pareto_milestone_archive_sha256",
        "final_mask_hash",
        "role_mask_hashes",
        "role_retained_route_counts",
        "role_to_unique_lane",
        "unique_lane_mask_hashes",
        "unique_lane_calibration_hashes",
        "unique_lane_calibration_freeze_hashes",
        "unique_lane_calibration_freeze_payloads",
        "freeze_sequence",
        "roles",
        "validation_independent",
        "terminal_rows_visible",
        "terminal_outcomes_used_for_selection",
        "matched_density_comparison_registered",
        "schema",
    ),
    "V1TerminalValidationFamilyRequestReceipt": (
        "family_freeze",
        "observation_artifact_relative_path",
        "observation_artifact_schema",
        "request_sequence",
        "evaluation_mode",
        "validation_independent",
        "outcome_fields_absent",
        "terminal_rows_visible",
        "checkpoint_before_outcomes_required",
        "schema",
    ),
    "V1TerminalValidationFamilyObservationReceipt": (
        "request",
        "joint_validation_values_hash",
        "joint_validation_shape",
        "unique_lane_validation_values_hashes",
        "unique_lane_validation_value_counts",
        "unique_lane_metrics",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_relative_path",
        "observation_artifact_sha256",
        "observation_artifact_size_bytes",
        "observation_artifact_semantic_hash",
        "observation_sequence",
        "artifact_durable_before_receipt",
        "artifact_replay_verified",
        "single_common_panel_evaluation",
        "raw_validation_rows_persisted",
        "fed_back_into_selection",
        "terminal_outcomes_used_for_selection",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalValidationFamilyReceipt": (
        "observation",
        "reveal_sequence",
        "single_reveal",
        "matched_density_comparison_authority",
        "fed_back_into_selection",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalComparisonFamilyFreezeReceiptV2": (
        "terminal_freeze_hash",
        "terminal_validation_population_hash",
        "terminal_validation_context_ids_hash",
        "terminal_validation_context_count",
        "terminal_metric_semantics_hash",
        "terminal_evaluator_identity_hash",
        "pending_dense_reference_design_hash",
        "dense_mask_hash",
        "current_v0_mask_hash",
        "current_v0_mask_artifact_sha256",
        "current_v0_mask_artifact_semantic_hash",
        "current_v0_mask_artifact_byte_count",
        "dense_current_v0_calibration_family_freeze_hash",
        "final_v1_mask_hash",
        "route_axis_hash",
        "route_count",
        "role_mask_hashes",
        "role_retained_route_counts",
        "role_to_unique_lane",
        "unique_lane_mask_hashes",
        "unique_lane_calibration_hashes",
        "unique_lane_calibration_freeze_hashes",
        "unique_lane_calibration_freeze_payloads",
        "freeze_sequence",
        "roles",
        "validation_independent",
        "outcome_fields_absent",
        "terminal_rows_visible",
        "terminal_outcomes_opened",
        "terminal_outcomes_used_for_selection",
        "current_v0_comparison_only",
        "current_v0_candidate_or_commit_authority",
        "schema",
    ),
    "V1TerminalComparisonFamilyRequestReceiptV2": (
        "family_freeze",
        "observation_artifact_relative_path",
        "observation_artifact_schema",
        "request_sequence",
        "evaluation_mode",
        "validation_independent",
        "outcome_fields_absent",
        "terminal_rows_visible",
        "terminal_outcomes_opened",
        "checkpoint_before_outcomes_required",
        "schema",
    ),
    "V1TerminalComparisonFamilyObservationReceiptV2": (
        "request",
        "joint_validation_values_hash",
        "joint_validation_shape",
        "unique_lane_validation_values_hashes",
        "unique_lane_validation_value_counts",
        "unique_lane_metrics",
        "paired_mean_deltas",
        "paired_validation_difference_hashes",
        "common_panel_evaluator_receipt_hash",
        "observation_artifact_relative_path",
        "observation_artifact_sha256",
        "observation_artifact_size_bytes",
        "observation_artifact_semantic_hash",
        "observation_sequence",
        "artifact_durable_before_receipt",
        "artifact_replay_verified",
        "single_common_panel_evaluation",
        "raw_validation_rows_persisted",
        "paired_delta_derivation",
        "fed_back_into_selection",
        "terminal_outcomes_used_for_selection",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1TerminalComparisonFamilyReceiptV2": (
        "observation",
        "reveal_sequence",
        "single_reveal",
        "terminal_comparison_authority",
        "dense_reference_resolved",
        "fed_back_into_selection",
        "claim_execution_certified",
        "mask_commit_authority",
        "schema",
    ),
    "V1PerformanceReceipt": (
        "controller_run_id",
        "source_environment_hash",
        "cold_staging_seconds",
        "cold_compile_seconds",
        "warm_acquisition_seconds",
        "warm_decision_seconds",
        "warm_terminal_seconds",
        "peak_device_memory_bytes",
        "peak_host_rss_bytes",
        "host_transfer_bytes",
        "gpu_utilization_fraction",
        "fallback_used",
        "fallback_reason",
        "structural_lowering_receipt_hash",
        "claim_execution_certified",
        "schema",
    ),
    "V1RunCompletionReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "final_controller_snapshot_hash",
        "terminal_freeze_hash",
        "terminal_validation_hash",
        "terminal_mask_hash",
        "accepted_step_count",
        "candidate_attempt_count",
        "performance_receipt_hash",
        "source_seal_hash",
        "environment_hash",
        "completion_sequence",
        "workflow_identity",
        "genuine_partition_free_controller",
        "direct_rule_over_v0_adapter",
        "retired_trust_walk",
        "claim_execution_certified",
        "selector_promoted",
        "schema",
    ),
    # Top-level acquisition carriers are recognized even when inspect-only
    # validation intentionally uses their sealed receipt rather than executing
    # their live numerical objects.
    "V1FDStreamBatch": (
        "batch_id",
        "draw_ids",
        "source_context_ids",
        "rows_payload",
        "row_shape",
        "weights_payload",
        "source_measurement_hash",
        "request_hash",
        "provider_hash",
        "calibration_identity",
        "current_mask_hash",
        "route_axis_hash",
        "metadata",
        "schema",
    ),
    "V1FDProducerAdmissionReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "current_mask_hash",
        "route_axis_hash",
        "backend",
        "source_state_hash",
        "source_ledger_hash",
        "accumulator_derivation_receipt_hash",
        "adapter_result_hash",
        "derivation_proof_mode",
        "proof_streaming_window_contexts",
        "proof_peak_host_row_bytes",
        "proof_total_host_transfer_bytes",
        "full_tensor_host_snapshot",
        "proposal_only",
        "claim_execution_certified",
        "schema",
    ),
    "V1FDStreamAcquisition": (
        "campaign_contract_hash",
        "current_mask_hash",
        "route_axis_hash",
        "backend",
        "batches",
        "geometry_state",
        "source_ledger",
        "adapter_result",
        "producer_admission_receipt",
        "schema",
    ),
    "V1CorrectionAcquisition": (
        "campaign_contract_hash",
        "current_mask_hash",
        "route_axis_hash",
        "feature_map",
        "tangent_prices",
        "atom_costs",
        "design",
        "specification",
        "outcomes",
        "result",
        "schema",
    ),
    "V1AnalyticExactGateCorrectionAcquisition": (
        "campaign_contract_hash",
        "current_mask_hash",
        "route_axis_hash",
        "context_mixture_hash",
        "tangent_prices",
        "atom_costs",
        "result",
        "source_measurement_hashes",
        "outcomes",
        "metadata",
        "schema",
    ),
    "V1RelinearizedAcquisition": (
        "binding",
        "generation_index",
        "current_mask",
        "committed_mask_lineage_hashes",
        "committed_mask_transition_receipt_hashes",
        "fd",
        "correction",
        "chart",
        "numeric_state",
        "domain",
        "constraints",
        "composition",
        "receipt",
        "metadata",
        "schema",
    ),
    # Compact selected-shell F/Q carriers have a distinct, frozen nominal
    # graph.  These are exact landed slot orders, including the two private
    # integrity caches.  They are not compatible extensions of the legacy
    # exact-stream acquisition carriers above.
    "V1FQActionNumericState": (
        "endpoint_index",
        "action_id",
        "action_hash",
        "parent_mask",
        "packed_action",
        "route_axis_hash",
        "route_count",
        "parent_mask_hash",
        "resulting_mask_hash",
        "packed_action_hash",
        "removed_count",
        "removed_route_index_hash",
        "resource_gain",
        "atom_cost_source_hash",
        "resource_gain_receipt_hash",
        "f_lower",
        "f_upper",
        "q_lower",
        "q_upper",
        "sqrt_q_lower",
        "sqrt_q_upper",
        "f_units",
        "q_units",
        "sqrt_q_units",
        "fq_interval_schema",
        "fq_interval_hash",
        "fq_interval_full_census",
        "fq_interval_population_census_complete",
        "fq_interval_lower_confidence_authority",
        "fq_interval_upper_confidence_authority",
        "fq_interval_observation_semantics",
        "f_upper_budget",
        "q_upper_budget",
        "f_upper_budget_ratio",
        "q_upper_budget_ratio",
        "sqrt_q_upper_budget_ratio",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "decisive_fq_commit_hash",
        "fq_queue_authorization_hash",
        "sampled_diagnostic_lineage_hash",
        "generation_index",
        "independently_certified_fq_interval",
        "exact_scalar_resource_gain_from_bound_source",
        "proposal_only",
        "fq_monotonicity_assumed",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
        "_state_hash",
    ),
    "V1FQCandidateProposal": (
        "numeric_state",
        "domain_hash",
        "constraints_hash",
        "evaluation_receipt_hash",
        "candidate_id",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "fq_monotonicity_assumed",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
    ),
    "V1FQCandidateCompositionResult": (
        "action_numeric_states",
        "proposals",
        "domain_hash",
        "constraints_hash",
        "fq_queue_authorization_hash",
        "decisive_fq_commit_hash",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "sampled_diagnostic_lineage_hash",
        "atom_cost_source_hash",
        "parent_mask_hash",
        "route_axis_hash",
        "route_count",
        "queue_endpoint_indices",
        "queue_action_hashes",
        "queue_resulting_mask_hashes",
        "queue_resource_gain_receipt_hashes",
        "validated_action_count",
        "proposal_count",
        "composer_id",
        "selection_receipt_hash",
        "authorized_finite_queue_materialized_exactly",
        "arbitrary_certified_endpoint_subset_preserved",
        "resource_descending_order_verified",
        "filtering_performed",
        "reordering_performed",
        "fq_monotonicity_assumed",
        "claims_global_powerset_optimum",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
        "_result_hash",
    ),
    "V1FQAcquisitionGenerationReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "registration_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "acquisition_hash",
        "invalidation_chart_hash",
        "composition_result_hash",
        "compact_domain_hash",
        "fq_constraints_hash",
        "fq_queue_authorization_hash",
        "decisive_fq_commit_hash",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "sampled_diagnostic_lineage_hash",
        "atom_cost_source_hash",
        "ordered_candidate_ids",
        "ordered_endpoint_indices",
        "ordered_action_hashes",
        "refresh_reason",
        "logical_sequence",
        "proposal_only",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "claim_execution_certified",
        "metadata",
        "schema",
    ),
    "V1FQAuthorizedAcquisition": (
        "binding",
        "generation_index",
        "current_mask",
        "committed_mask_lineage_hashes",
        "committed_mask_transition_receipt_hashes",
        "composition",
        "receipt",
        "metadata",
        "schema",
    ),
    # Dedicated compact F/Q/R controller carriers are a sibling graph, not
    # an additive extension of the legacy F/Q carriers above.  Freeze their
    # exact landed slot orders so inspect-only decoding cannot silently assign
    # a newer layout to historical bytes.
    "V1CompactFQRBranchFamilySourceAlias": (
        "branch",
        "endpoint_index",
        "frozen_action_hash",
        "packed_action_hash",
        "registered_resource",
        "branch_binding_hash",
        "branch_evidence_hash",
        "resource_registration_hash",
        "certification_receipt_hash",
        "decision_hash",
        "proposal_authorization_hash",
        "endpoint_certificate_hash",
        "f_lower",
        "f_upper",
        "q_lower",
        "q_upper",
        "r_half_lower",
        "r_half_upper",
        "r_one_lower",
        "r_one_upper",
        "half_response_lower",
        "half_response_upper",
        "one_response_lower",
        "one_response_upper",
        "independently_upper_certified",
        "schema",
        "_alias_hash",
    ),
    "V1FQRActionNumericState": (
        "queue_index",
        "representative_branch",
        "representative_endpoint_index",
        "representative_frozen_action_hash",
        "source_aliases",
        "source_aliases_hash",
        "parent_mask",
        "packed_action",
        "parent_mask_hash",
        "resulting_mask_hash",
        "packed_action_hash",
        "route_axis_hash",
        "route_count",
        "removed_count",
        "removed_route_index_hash",
        "registered_resource",
        "resource_units",
        "resource_source_hash",
        "resource_receipt_hash",
        "f_lower",
        "f_upper",
        "q_lower",
        "q_upper",
        "r_half_lower",
        "r_half_upper",
        "r_one_lower",
        "r_one_upper",
        "half_response_lower",
        "half_response_upper",
        "one_response_lower",
        "one_response_upper",
        "fqr_interval_hash",
        "branch_family_authorization_hash",
        "branch_family_entry_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
        "source_branch_evidence_hashes",
        "generation_index",
        "independently_certified_fqr_bounds",
        "complete_queue_frozen_before_K_O",
        "proposal_only",
        "paired_bce_oracle_required",
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "candidate_freeze_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
        "_state_hash",
    ),
    "V1FQRCandidateProposal": (
        "numeric_state",
        "domain_hash",
        "certification_scope_hash",
        "evaluation_receipt_hash",
        "candidate_id",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "paired_bce_oracle_required",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
    ),
    "V1FQRCandidateCompositionResult": (
        "action_numeric_states",
        "proposals",
        "domain_hash",
        "certification_scope_hash",
        "branch_family_authorization_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
        "source_branch_evidence_hashes",
        "parent_mask_hash",
        "route_axis_hash",
        "route_count",
        "queue_entry_hashes",
        "queue_branch_endpoint_coordinates",
        "queue_packed_action_hashes",
        "queue_resulting_mask_hashes",
        "queue_fqr_interval_hashes",
        "queue_resource_receipt_hashes",
        "validated_action_count",
        "proposal_count",
        "composer_id",
        "selection_receipt_hash",
        "complete_authorized_queue_materialized_exactly",
        "duplicate_physical_actions_quotiented_upstream",
        "resource_descending_order_verified",
        "filtering_performed",
        "reordering_performed",
        "fqr_monotonicity_assumed",
        "claims_hard_global_optimum",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "paired_bce_oracle_required",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
        "schema",
        "_result_hash",
    ),
    "V1FQRAcquisitionGenerationReceipt": (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "registration_hash",
        "generation_index",
        "current_mask_hash",
        "route_axis_hash",
        "acquisition_hash",
        "invalidation_chart_hash",
        "composition_result_hash",
        "compact_domain_hash",
        "certification_scope_hash",
        "branch_family_authorization_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
        "source_branch_evidence_hashes",
        "ordered_candidate_ids",
        "ordered_entry_hashes",
        "ordered_branch_endpoint_coordinates",
        "ordered_packed_action_hashes",
        "ordered_fqr_interval_hashes",
        "refresh_reason",
        "logical_sequence",
        "proposal_only",
        "complete_queue_frozen_before_K_O",
        "paired_bce_oracle_required",
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "candidate_freeze_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "claim_execution_certified",
        "metadata",
        "schema",
    ),
    "V1FQRAuthorizedAcquisition": (
        "binding",
        "generation_index",
        "current_mask",
        "committed_mask_lineage_hashes",
        "committed_mask_transition_receipt_hashes",
        "composition",
        "receipt",
        "metadata",
        "schema",
    ),
    # Exact current acquisition-graph maps, frozen from landed v29+ source.
    # An unmatched historical payload fails closed; absent old sidecar bytes
    # are not grounds for inventing a field order.
    "GlobalTrustGeometryDerivationReceipt": (
        "producer_algorithm_id",
        "producer_implementation",
        "trust_spec_hash",
        "ordered_route_axis_hash",
        "context_axis_hash",
        "stream_entry_count",
        "effective_context_count",
        "ordered_source_rows_hash",
        "stream_weight_hash",
        "mean_component_hash",
        "centered_factor_hash",
        "exact_residual_diagonal_hash",
        "total_weight",
        "cumulative_fd_shrink",
        "normalized_stream_squared_frobenius",
        "sketch_rank",
        "measured_mask_hash",
        "calibration_state_id",
        "source_measurement_hashes",
        "schema",
    ),
    "GlobalTrustGeometrySpec": (
        "trust_id",
        "initial_rank",
        "maximum_rank",
        "reuse_threshold",
        "refresh_threshold",
        "residual_policy",
        "metadata",
        "schema",
    ),
    "GlobalTrustGeometryState": (
        "spec",
        "route_ids",
        "mean_component",
        "centered_factor",
        "residual_spectral_bound",
        "total_weight",
        "context_count",
        "context_axis_hash",
        "measured_mask_hash",
        "calibration_state_id",
        "source_measurement_hashes",
        "residual_diagonal_bound",
        "derivation_receipt",
        "metadata",
        "schema",
        "_content_hash",
    ),
    "MeasurementOutcome": (
        "request_id",
        "state_hash",
        "context_ids",
        "candidate_ids",
        "values",
        "calibration_state_id",
        "realized_cost",
        "cache_artifacts",
        "integrity_hash",
        "metadata",
    ),
    "ContextSelector": (
        "selector_id",
        "requested_count",
        "strata",
        "context_ids",
        "random_seed",
        "minimum_per_stratum",
        "parameters",
    ),
    "CostEstimate": (
        "wall_seconds",
        "device_seconds",
        "forward_passes",
        "backward_passes",
        "calibration_fits",
        "transferred_bytes",
        "peak_memory_bytes",
        "uncertainty_fraction",
    ),
    "MeasurementRequest": (
        "request_id",
        "kind",
        "state_hash",
        "version",
        "context_selector",
        "candidate_ids",
        "expected_cost",
        "expected_decision_value",
        "reuse_keys",
        "provenance",
        "payload",
    ),
    "RealizedCost": (
        "wall_seconds",
        "device_seconds",
        "forward_passes",
        "backward_passes",
        "calibration_fits",
        "transferred_bytes",
        "peak_memory_bytes",
        "packing_degree",
        "cache_hit",
    ),
    "RemovedRouteIndexSet": (
        "route_count",
        "count",
        "dtype_code",
        "encoded",
    ),
    "V1ActionFeatureMap": (
        "route_ids",
        "feature_ids",
        "atom_feature_rows",
        "construction_receipt_hash",
        "metadata",
        "schema",
    ),
    "V1AtomCostState": (
        "route_ids",
        "costs",
        "cost_functional_id",
        "source_receipt_hash",
        "schema",
    ),
    "V1CandidateCompositionConstraints": (
        "maximum_removed_count",
        "minimum_resource_gain",
        "maximum_resource_gain",
        "maximum_proposal_damage",
        "maximum_fd_geometry_upper",
        "diagonal_hard_bound_mode",
        "proposal_damage_cap_source_hash",
        "resource_cap_source_hash",
        "geometry_cap_source_hash",
        "constraints_source_receipt_hash",
        "schema",
    ),
    "V1CandidateCompositionResult": (
        "numeric_state",
        "domain",
        "constraints",
        "ranked_proposals",
        "evaluated_action_count",
        "eligible_action_count",
        "composer_id",
        "selection_receipt_hash",
        "declared_domain_exhausted",
        "exact_within_declared_domain",
        "claims_global_powerset_optimum",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "claim_execution_certified",
        "schema",
    ),
    "V1CandidateNumericState": (
        "route_ids",
        "live_route_ids",
        "parent_mask",
        "effective_factor",
        "residual_diagonal",
        "corrected_prices",
        "atom_costs",
        "fd_shrink",
        "price_orientation_id",
        "surrogate_equation_id",
        "fd_spectral_upper_equation_id",
        "fresh_relinearization_required",
        "chart_hash",
        "registration_hash",
        "source_geometry_state_hash",
        "fd_adapter_result_hash",
        "fd_lineage_hash",
        "fd_verification_receipt_hash",
        "correction_lineage_hash",
        "correction_result_receipt_hash",
        "correction_verification_receipt_hash",
        "atom_cost_state_hash",
        "route_axis_hash",
        "live_route_axis_hash",
        "parent_mask_hash",
        "effective_factor_hash",
        "source_exact_residual_diagonal_hash",
        "registered_residual_diagonal_hash",
        "corrected_price_hash",
        "atom_cost_hash",
        "fd_shrink_receipt_hash",
        "residual_diagonal_status",
        "residual_diagonal_receipt_hash",
        "residual_bound_proof_hash",
        "proved_action_class",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "schema",
    ),
    "V1CandidateProposal": (
        "candidate_id",
        "numeric_state_hash",
        "domain_hash",
        "constraints_hash",
        "parent_mask_hash",
        "route_axis_hash",
        "removed_routes",
        "removed_route_ids",
        "resulting_mask",
        "resource_gain",
        "corrected_linear_damage",
        "fd_quadratic",
        "residual_diagonal_damage",
        "predicted_surrogate_damage",
        "proposal_damage_constraint_value",
        "fd_spectral_upper",
        "diagonal_hard_bound_used",
        "surrogate_equation_id",
        "fd_spectral_upper_equation_id",
        "evaluation_receipt_hash",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "claim_execution_certified",
        "schema",
    ),
    "V1ChartThresholds": (
        "action_leverage_bound",
        "metric_path_length_bound",
        "layer_local_leverage_bound",
        "residual_bound",
        "fd_shrink_bound",
        "metadata",
        "schema",
    ),
    "V1FDAdapterResult": ("lineage", "verification_receipt", "schema"),
    "V1FDLineageVerificationReceipt": (
        "lineage_hash",
        "source_state_hash",
        "source_ledger_hash",
        "accumulator_derivation_receipt_hash",
        "ordered_source_rows_hash",
        "stream_weight_hash",
        "factor_derivation_receipt_hash",
        "fd_shrink_derivation_receipt_hash",
        "exact_norm_derivation_receipt_hash",
        "verified_algebra_hash",
        "proposal_only",
        "claim_execution_certified",
        "schema",
    ),
    "V1FDSourceLedger": (
        "draw_ids",
        "source_context_ids",
        "source_row_hashes",
        "source_measurement_hashes",
        "pre_outcome_stream_weight_hash",
        "expected_ordered_route_axis_hash",
        "expected_measured_mask_hash",
        "expected_calibration_state_id",
        "proposal_only",
        "claim_execution_certified",
        "schema",
    ),
    "V1FiniteActionDefinition": (
        "action_id",
        "measurement_request_id",
        "candidate_mask_hash",
        "action_indices",
        "coefficients",
        "weight",
        "schema",
    ),
    "V1FiniteInterventionDesign": (
        "base_mask_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "risk_functional",
        "estimand_contrast_hash",
        "sampling_law_hash",
        "pre_outcome_lock_hash",
        "actions",
        "metadata",
        "schema",
    ),
    "V1FiniteRemovalDomain": (
        "route_axis_hash",
        "route_count",
        "removal_sets",
        "declaration_receipt_hash",
        "action_class_proof",
        "schema",
    ),
    "V1FrequentDirectionsLineage": (
        "algorithm_id",
        "implementation",
        "route_axis_hash",
        "measured_mask_hash",
        "draw_ids",
        "source_context_ids",
        "source_row_hashes",
        "stream_weight_hash",
        "accumulator_context_axis_hash",
        "factor_hash",
        "factor_receipt_hash",
        "fd_shrink_receipt_hash",
        "exact_streamed_norm_receipt_hash",
        "fd_rank",
        "factor_row_count",
        "route_count",
        "fd_shrink",
        "exact_streamed_norm",
        "canonical_equivalence_receipt_hash",
        "retained_capacity_rank",
        "retained_capacity_factor_hash",
        "full_replay_source_hash",
        "metadata",
        "schema",
    ),
    "V1InterventionalCorrectionLineage": (
        "correction_kind",
        "base_mask_hash",
        "route_axis_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "feature_map_hash",
        "estimand_contrast_hash",
        "observation_model_hash",
        "interventional_support_hash",
        "design_matrix_hash",
        "identification_receipt_hash",
        "corrected_price_hash",
        "corrected_price_receipt_hash",
        "atom_cost_hash",
        "atom_cost_receipt_hash",
        "measured_action_hashes",
        "intervention_draw_ids",
        "identified_design_rank",
        "required_design_rank",
        "metadata",
        "schema",
    ),
    "V1InterventionalCorrectionResult": (
        "lineage",
        "corrected_prices",
        "correction_coefficients",
        "design_matrix",
        "observed_action_residuals",
        "fitted_action_residuals",
        "result_receipt_hash",
        "schema",
    ),
    "V1InterventionalCorrectionSpec": (
        "feature_map_hash",
        "tangent_price_state_hash",
        "atom_cost_state_hash",
        "intervention_design_hash",
        "required_design_rank",
        "maximum_condition_number",
        "observation_model_id",
        "schema",
    ),
    "V1InterventionalCorrectionVerificationReceipt": (
        "lineage_hash",
        "adapter_result_receipt_hash",
        "corrected_price_values_hash",
        "design_matrix_values_hash",
        "correction_coefficients_hash",
        "observed_action_residuals_hash",
        "fitted_action_residuals_hash",
        "verified_algebra_hash",
        "proposal_only",
        "claim_execution_certified",
        "schema",
    ),
    "V1ProposalChart": (
        "build_contract",
        "fd_lineage",
        "correction_lineage",
        "fd_verification_receipt",
        "correction_verification_receipt",
        "metadata",
        "schema",
    ),
    "V1ProposalChartBuildContract": (
        "registration_hash",
        "base_mask_hash",
        "current_mask_hash",
        "cumulative_removal_hash",
        "committed_mask_lineage_hashes",
        "committed_mask_transition_receipt_hashes",
        "route_axis_hash",
        "live_route_axis_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "thresholds",
        "residual_diagonal_status",
        "residual_diagonal_receipt_hash",
        "residual_bound_proof_hash",
        "mixed_difference_envelope",
        "metadata",
        "schema",
    ),
    "V1QuadraticResponseRiskFunctional": (
        "response_dimension",
        "normalization_metric",
        "construction_receipt_hash",
        "aggregation_id",
        "schema",
    ),
    "V1TangentPriceState": (
        "route_ids",
        "prices",
        "base_mask_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "source_receipt_hash",
        "price_orientation_id",
        "schema",
    ),
    "V1RouteIdAxisIdentity": ("count", "ordered_hash", "schema"),
    "V1FiniteActionClassProof": (
        "route_axis_hash",
        "route_count",
        "base_mask_hash",
        "current_mask_hash",
        "ordered_removal_sets",
        "proof_artifact_hash",
        "construction_receipt_hash",
        "schema",
    ),
    "V1PackedRemovalAction": (
        "resulting_mask",
        "removed_count",
        "removed_route_index_hash",
    ),
    "V1MixedDifferenceEnvelope": (
        "proof_basis",
        "proof_artifact_hash",
        "action_class_proof",
        "base_mask_hash",
        "route_axis_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "threshold_hash",
        "construction_receipt_hash",
        "metadata",
        "schema",
    ),
    "V1AnalyticExactGateCorrectionLineage": (
        "base_mask_hash",
        "route_axis_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
        "source_attribution_hash",
        "streaming_receipt_hash",
        "tangent_price_state_hash",
        "corrected_price_hash",
        "corrected_price_receipt_hash",
        "absolute_salience_hash",
        "atom_cost_hash",
        "atom_cost_receipt_hash",
        "price_orientation_id",
        "metadata",
        "correction_kind",
        "raw_target_logit_unit_cotangent",
        "empirical_child_parent_mask_evaluation_count",
        "schema",
    ),
    "V1AnalyticExactGateCorrectionVerificationReceipt": (
        "lineage_hash",
        "result_receipt_hash",
        "corrected_price_values_hash",
        "absolute_salience_values_hash",
        "source_attribution_hash",
        "streaming_receipt_hash",
        "verified_algebra_hash",
        "proposal_only",
        "claim_execution_certified",
        "schema",
    ),
    "V1AnalyticExactGateCorrectionResult": (
        "lineage",
        "corrected_prices",
        "absolute_salience",
        "result_receipt_hash",
        "schema",
    ),
    "ConstructiveOrdinalRouteAxis": (
        "_count",
        "_ordered_digest",
        "_canonical_json_digest",
    ),
}

# Compact accepted transitions introduced this identity after the frozen
# common historical layout.  Keep its exact three-slot ABI separate so it
# cannot silently become part of an older cohort's generic field map.
_REMOVED_ROUTE_SET_IDENTITY_RAW_FIELDS: Final = (
    "route_count",
    "count",
    "index_hash",
)

# The mixed positive/stable-negative F/Q/R graph is deliberately represented
# by new nested carrier names.  Its additional provenance coordinate is the
# final slotted value in each subclass; retaining separate exact maps here is
# what prevents a v2 pickle from being decoded with a v1 positional layout (or
# vice versa).  Do not fold these into the frozen v1 maps above.
_FQR_V2_RAW_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "V1FQRActionNumericStateV2": (
        *_COMMON_RAW_FIELDS["V1FQRActionNumericState"],
        "source_branch_negative_evidence_hashes",
    ),
    "V1FQRCandidateCompositionResultV2": (
        *_COMMON_RAW_FIELDS["V1FQRCandidateCompositionResult"],
        "source_branch_negative_evidence_hashes",
    ),
    "V1FQRAcquisitionGenerationReceiptV2": (
        *_COMMON_RAW_FIELDS["V1FQRAcquisitionGenerationReceipt"],
        "source_branch_negative_evidence_hashes",
    ),
}
_RAW_ALLOWED_CACHED_FIELDS: Final[dict[str, frozenset[str]]] = {
    # ``HardMask`` historically used cached_property without slots, so trusted
    # checkpoints may carry these derived cache entries in addition to its
    # three dataclass fields.  They are replay-checked before being ignored by
    # canonical dataclass hashing.
    "HardMask": frozenset({"mask_hash", "retained_count"}),
    "RemovedRouteIndexSet": frozenset({"index_hash"}),
}
_PAIRED_BCE_HASHED_REBUILD_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_paired_bce_acquisition"
)
_PAIRED_BCE_HASHED_REBUILD_TARGETS: Final = frozenset(
    {
        "V1PairedBCEOracleRoleSeparationReceipt",
        "V1PairedBCEDynamicOracleReceipt",
        "V1PairedBCEStratumRegistration",
        "V1PairedBCEAcquisitionRegistration",
        "V1PairedBCEAcquisitionState",
        "V1PairedBCEChunkRequest",
        "V1PairedBCEChunkReceipt",
        "V1PairedBCEDecision",
    }
)
_PAIRED_BCE_HASH_CACHE_FIELDS: Final = {
    "V1PairedBCEOracleRoleSeparationReceipt": "_receipt_hash_cache",
    "V1PairedBCEDynamicOracleReceipt": "_receipt_hash_cache",
    "V1PairedBCEStratumRegistration": "_content_hash_cache",
    "V1PairedBCEAcquisitionRegistration": "_registration_hash_cache",
    "V1PairedBCEAcquisitionState": "_state_hash_cache",
    "V1PairedBCEChunkRequest": "_request_hash_cache",
    "V1PairedBCEChunkReceipt": "_receipt_hash_cache",
    "V1PairedBCEDecision": "_decision_hash_cache",
}
_PAIRED_BCE_DYNAMIC_REGISTRATION_ADDITIONS: Final = frozenset(
    {
        "dynamic_oracle_receipt",
    }
)
_PAIRED_BCE_DYNAMIC_RECEIPT_HASH_ADDITIONS: Final = frozenset(
    {
        "dynamic_oracle_receipt_hash",
    }
)
_PAIRED_BCE_DYNAMIC_DECISION_ADDITIONS: Final = frozenset(
    {
        "calibration_prefix_stratum_counts",
        "residual_oracle_stratum_counts",
        "residual_oracle_population_stratum_counts",
        "dynamic_oracle_receipt_hash",
    }
)


def _paired_bce_hashed_rebuild_field_variants(
    target_name: str,
) -> tuple[tuple[str, ...], ...]:
    """Return exact constructor-payload field variants for hashed replay.

    The paired-BCE dynamic full-population oracle contract was an additive
    growth over the static-manifest registration objects.  Current checkpoints
    must carry the full field set, while historical hashed payloads are still
    allowed to replay if and only if they match the prior exact constructor
    shape and their stored content hash.
    """

    current = _COMMON_RAW_FIELDS[target_name]
    variants: list[tuple[str, ...]] = [current]
    if target_name == "V1PairedBCEAcquisitionRegistration":
        variants.append(
            tuple(
                field_name
                for field_name in current
                if field_name not in _PAIRED_BCE_DYNAMIC_REGISTRATION_ADDITIONS
            )
        )
    elif target_name in {
        "V1PairedBCEAcquisitionState",
        "V1PairedBCEChunkRequest",
        "V1PairedBCEChunkReceipt",
    }:
        variants.append(
            tuple(
                field_name
                for field_name in current
                if field_name not in _PAIRED_BCE_DYNAMIC_RECEIPT_HASH_ADDITIONS
            )
        )
    elif target_name == "V1PairedBCEDecision":
        variants.append(
            tuple(
                field_name
                for field_name in current
                if field_name not in _PAIRED_BCE_DYNAMIC_DECISION_ADDITIONS
            )
        )
    return tuple(dict.fromkeys(variants))

_CONTRACT_FIELDS_58: Final = (
    "campaign_id",
    "registration_id",
    "model_seal_hash",
    "query_id",
    "referent_id",
    "route_axis_hash",
    "context_population_hash",
    "context_population_source_manifest_hash",
    "target_population_hash",
    "dense_reference_hash",
    "calibration_identity",
    "terminal_validation_population_hash",
    "per_step_alpha",
    "cumulative_draw_counts",
    "maximum_importance_weight",
    "maximum_step_count",
    "maximum_candidate_count",
    "maximum_removed_count",
    "look_schedule_policy_hash",
    "damage_target_derivation_hash",
    "sampling_law_hash",
    "support_construction_hash",
    "importance_weight_bound_proof_hash",
    "entropy_source_policy_hash",
    "entropy_replay_policy_hash",
    "candidate_domain_policy_hash",
    "resource_cost_policy_hash",
    "queue_exhaustion_policy_hash",
    "terminal_calibration_policy_hash",
    "terminal_validation_policy_hash",
    "student_t_evidence_disclosure_hash",
    "proof_artifact_hash",
    "owner_decision_record_hash",
    "config_source_receipt_hash",
    "design_identity",
    "workflow_identity",
    "workflow_alias",
    "certificate_family",
    "cross_query_shrinkage",
    "step_spending_id",
    "look_spending_id",
    "interval_bound_id",
    "sampling_algorithm_id",
    "proof_id",
    "direct_increment_target_id",
    "increment_orientation_id",
    "exhaustion_action_id",
    "fd_rank",
    "refresh_policy_id",
    "d_max_disposition_id",
    "cross_step_spending_id",
    "cross_query_borrowing_id",
    "trajectory_confidence_claim_id",
    "terminal_reveal_policy_id",
    "proposal_authority_id",
    "student_t_evidence_ceiling_id",
    "metadata",
    "schema",
)
_CONTRACT_FIELDS_59: Final = (
    *_CONTRACT_FIELDS_58[:12],
    "campaign_wide_alpha",
    *_CONTRACT_FIELDS_58[12:],
)
_COORDINATE_FIELDS_20: Final = (
    "per_step_alpha",
    "maximum_looks_per_step",
    "maximum_importance_weight",
    "campaign_contract_hash",
    "registered_cumulative_draw_counts",
    "look_schedule_policy_hash",
    "proof_artifact_hash",
    "owner_decision_record_hash",
    "importance_weight_bound_proof_hash",
    "entropy_source_policy_hash",
    "deferred_family",
    "cross_query_shrinkage",
    "step_spending_id",
    "look_spending_id",
    "bound_id",
    "sampling_id",
    "proof_id",
    "exhaustion_action_id",
    "metadata",
    "schema",
)
_COORDINATE_FIELDS_22: Final = (
    "campaign_wide_alpha",
    "per_step_alpha",
    "maximum_candidate_count",
    *_COORDINATE_FIELDS_20[1:],
)
_STEP_FIELDS_9: Final = (
    "registration_hash",
    "campaign_contract_hash",
    "certificate_freeze",
    "coordinates_hash",
    "step_index",
    "maximum_mean_damage",
    "cumulative_draw_counts",
    "schedule_construction_hash",
    "schema",
)
_STEP_FIELDS_10: Final = (
    *_STEP_FIELDS_9[:5],
    "candidate_attempt_index",
    *_STEP_FIELDS_9[5:],
)
_SAMPLING_FIELDS_11: Final = (
    "campaign_contract_hash",
    "population_context_ids",
    "target_population_probabilities",
    "sampling_population_probabilities",
    "declared_target_population_hash",
    "sampling_law_hash",
    "support_construction_hash",
    "source_receipt_hash",
    "maximum_importance_weight",
    "metadata",
    "schema",
)
_SAMPLING_FIELDS_12: Final = (
    *_SAMPLING_FIELDS_11[:4],
    "sampling_masses_u64",
    *_SAMPLING_FIELDS_11[4:],
)
_JOURNAL_V2_FIELDS: Final = (
    *_COMMON_RAW_FIELDS["_V1RuntimeCheckpointJournal"][:-1],
    "proposal_state_sidecar",
    "schema",
)
_JOURNAL_V3_FIELDS: Final = (
    *_COMMON_RAW_FIELDS["_V1RuntimeCheckpointJournal"][:-1],
    "proposal_state_sidecar",
    "completed_proposal_freeze_decisions",
    "schema",
)
_JOURNAL_V4_FIELDS: Final = (
    *_COMMON_RAW_FIELDS["_V1RuntimeCheckpointJournal"][:-1],
    "proposal_state_sidecar",
    "staged_proposal_backend_identity_hash",
    "completed_proposal_closures",
    "schema",
)
_JOURNAL_V5_FIELDS: Final = (
    *_COMMON_RAW_FIELDS["_V1RuntimeCheckpointJournal"][:-1],
    "proposal_state_sidecar",
    "staged_proposal_backend_identity_hash",
    "completed_proposal_closures",
    "pending_paired_bce_chunk",
    "schema",
)
_JOURNAL_V6_FIELDS: Final = (
    *_COMMON_RAW_FIELDS["_V1RuntimeCheckpointJournal"][:-1],
    "proposal_state_sidecar",
    "staged_proposal_backend_identity_hash",
    "completed_proposal_closures",
    "pending_paired_bce_chunk",
    "terminal_validation_family_freeze",
    "terminal_validation_family_request",
    "terminal_validation_family_observation",
    "terminal_validation_family",
    "schema",
)
_JOURNAL_V7_FIELDS: Final = (
    *_JOURNAL_V6_FIELDS[:-1],
    "terminal_comparison_family_freeze",
    "terminal_comparison_family_request",
    "terminal_comparison_family_observation",
    "terminal_comparison_family",
    "schema",
)


@dataclass(frozen=True, slots=True)
class _CohortSpec:
    label: str
    contract_fields: tuple[str, ...]
    contract_schema: str
    coordinate_fields: tuple[str, ...]
    coordinate_schema: str
    step_fields: tuple[str, ...]
    step_schema: str
    sampling_fields: tuple[str, ...]
    sampling_schema: str


_PRE16 = _CohortSpec(
    "sealed_v2_v15_mv1ckpt1",
    _CONTRACT_FIELDS_58,
    "partition_free_v1_option_a_campaign_contract_v1",
    _COORDINATE_FIELDS_20,
    "partition_free_v1_staged_alpha_coordinates_v3",
    _STEP_FIELDS_9,
    "partition_free_v1_staged_alpha_step_freeze_v2",
    _SAMPLING_FIELDS_11,
    "partition_free_v1_step_sampling_law_v1",
)
_V16_TO_V28 = _CohortSpec(
    "sealed_v16_v28_mv1ckpt2",
    _CONTRACT_FIELDS_58,
    "partition_free_v1_option_a_campaign_contract_v1",
    _COORDINATE_FIELDS_20,
    "partition_free_v1_staged_alpha_coordinates_v3",
    _STEP_FIELDS_9,
    "partition_free_v1_staged_alpha_step_freeze_v2",
    _SAMPLING_FIELDS_12,
    "partition_free_v1_step_sampling_law_v2",
)
_V29_PLUS = _CohortSpec(
    "v29_plus_typed_historical_or_current",
    _CONTRACT_FIELDS_59,
    "partition_free_v1_option_a_campaign_contract_v2",
    _COORDINATE_FIELDS_22,
    "partition_free_v1_staged_alpha_coordinates_v4",
    _STEP_FIELDS_10,
    "partition_free_v1_staged_alpha_step_freeze_v3",
    _SAMPLING_FIELDS_12,
    "partition_free_v1_step_sampling_law_v2",
)


def _carrier_state_sequence(
    value: _HistoricalCarrier, *, label: str
) -> tuple[Any, ...]:
    if value.constructor_args or value.constructor_kwargs:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} used unexpected constructor state",
            phase="raw_state_mapping",
        )
    state_value = value.state
    if type(state_value) is dict:
        # A few non-slotted low-layer contracts (notably HardMask and
        # MeasurementProvenance) use ordinary named state.  Exact key checking
        # occurs in ``_raw_mapping`` once the frozen field map is known.
        return tuple(state_value.values())
    if type(state_value) not in {list, tuple}:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} is not positional slotted state",
            phase="raw_state_mapping",
        )
    return tuple(state_value)


def _raw_instances(root: Any) -> tuple[_HistoricalCarrier, ...]:
    found: list[_HistoricalCarrier] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, _HistoricalCarrier):
            found.append(value)
            visit(value.state)
            visit(value.constructor_args)
            visit(value.constructor_kwargs)
        elif isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    visit(root)
    return tuple(found)


def _is_paired_bce_hashed_rebuild(value: Any) -> bool:
    return bool(
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _PAIRED_BCE_HASHED_REBUILD_MODULE
        and value.historical_name == "_rebuild_hashed"
    )


def _paired_bce_rebuild_target(value: _HistoricalCarrier) -> tuple[str, str]:
    if not _is_paired_bce_hashed_rebuild(value):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "state is not a paired-BCE hashed rebuild carrier",
            phase="raw_state_mapping",
        )
    if value.state is not None or value.constructor_kwargs:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild used unexpected state or kwargs",
            phase="raw_state_mapping",
        )
    args = value.constructor_args
    if len(args) != 3:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild has invalid arity",
            phase="raw_state_mapping",
        )
    value_type = args[0]
    target_module = getattr(value_type, "historical_module", None)
    target_name = getattr(value_type, "historical_name", None)
    if (
        type(target_module) is not str
        or target_module != _PAIRED_BCE_HASHED_REBUILD_MODULE
        or type(target_name) is not str
        or target_name not in _PAIRED_BCE_HASHED_REBUILD_TARGETS
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild targets an unsupported type",
            phase="raw_state_mapping",
        )
    return target_module, target_name


def _paired_bce_rebuild_expected_hash(value: Any, *, expected_name: str | None = None) -> str:
    if not isinstance(value, _HistoricalCarrier) or not _is_paired_bce_hashed_rebuild(value):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE nested hash-bearing value is not a hashed rebuild",
            phase="raw_state_mapping",
        )
    _module, target_name = _paired_bce_rebuild_target(value)
    if expected_name is not None and target_name != expected_name:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild target type is stale",
            phase="raw_state_mapping",
        )
    expected_hash = value.constructor_args[2]
    if type(expected_hash) is not str or not expected_hash:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild expected hash is invalid",
            phase="raw_state_mapping",
        )
    return expected_hash


def _paired_bce_rebuild_hash_payload(
    *,
    target_name: str,
    payload: Mapping[str, Any],
    cohort: _CohortSpec,
) -> dict[str, Any]:
    if target_name != "V1PairedBCEAcquisitionRegistration":
        return {key: _raw_canonical(item, cohort=cohort) for key, item in payload.items()}
    strata = payload["strata"]
    if type(strata) not in {tuple, list} or len(strata) != 2:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE registration rebuild has invalid strata",
            phase="raw_state_mapping",
        )
    result = {key: _raw_canonical(item, cohort=cohort) for key, item in payload.items()}
    result["stratum_registration_hashes"] = tuple(
        _paired_bce_rebuild_expected_hash(
            item,
            expected_name="V1PairedBCEStratumRegistration",
        )
        for item in strata
    )
    if "oracle_role_separation_receipt" in payload:
        separation = payload["oracle_role_separation_receipt"]
        result["oracle_role_separation_receipt_hash"] = (
            None
            if separation is None
            else _paired_bce_rebuild_expected_hash(
                separation,
                expected_name="V1PairedBCEOracleRoleSeparationReceipt",
            )
        )
    if "dynamic_oracle_receipt" in payload:
        dynamic = payload["dynamic_oracle_receipt"]
        result["dynamic_oracle_receipt_hash"] = (
            None
            if dynamic is None
            else _paired_bce_rebuild_expected_hash(
                dynamic,
                expected_name="V1PairedBCEDynamicOracleReceipt",
            )
        )
    result.pop("strata")
    result.pop("oracle_role_separation_receipt", None)
    result.pop("dynamic_oracle_receipt", None)
    return result


def _paired_bce_rebuild_full_canonical(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    _module, target_name = _paired_bce_rebuild_target(value)
    payload = _paired_bce_rebuild_payload(value, cohort=cohort)
    result = {key: _raw_canonical(item, cohort=cohort) for key, item in payload.items()}
    result[_PAIRED_BCE_HASH_CACHE_FIELDS[target_name]] = (
        _paired_bce_rebuild_expected_hash(value)
    )
    return result


def _paired_bce_rebuild_payload(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    _module, target_name = _paired_bce_rebuild_target(value)
    payload = value.constructor_args[1]
    expected_hash = value.constructor_args[2]
    if type(payload) is not dict or type(expected_hash) is not str or not expected_hash:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild payload/hash is invalid",
            phase="raw_state_mapping",
        )
    observed_fields = frozenset(payload)
    expected_field_variants = tuple(
        frozenset(fields)
        for fields in _paired_bce_hashed_rebuild_field_variants(target_name)
    )
    if observed_fields not in expected_field_variants:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "paired-BCE hashed rebuild payload has unknown or missing fields",
            phase="raw_state_mapping",
        )
    hash_payload = _paired_bce_rebuild_hash_payload(
        target_name=target_name,
        payload=payload,
        cohort=cohort,
    )
    if content_hash(hash_payload) != expected_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "paired-BCE hashed rebuild payload does not replay its expected hash",
            phase="historical_projection",
        )
    return dict(payload)


def _target_fields(spec: _CohortSpec, name: str) -> tuple[str, ...] | None:
    if name == "V1OptionACampaignContract":
        return spec.contract_fields
    if name == "V1StagedAlphaCoordinates":
        return spec.coordinate_fields
    if name == "V1StagedAlphaStepFreeze":
        return spec.step_fields
    if name == "V1StepSamplingLaw":
        return spec.sampling_fields
    return None


_LABEL_NATIVE_CHECKPOINT_CARRIER_MODULES: Final = MappingProxyType(
    {
        "V1SignedWorkSixRoleEndpointCoordinate": (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_six_role_certification"
        ),
        "V1SignedWorkSixRoleEndpointCertificate": (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_six_role_certification"
        ),
        "V1SignedWorkLabelNativeCActionCrosswalkReceipt": (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_label_native_c_bridge"
        ),
        "V1DeviceShellFrozenActionReceipt": (
            "mechanistic_systems.circuits.irg.modern_v1_device_shell_ladder"
        ),
    }
)
_LABEL_NATIVE_CHECKPOINT_ENUM_VALUES: Final = MappingProxyType(
    {
        (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_hard_fq_replay_cuda",
            "V1SignedWorkHardFQEndpointStatus",
        ): frozenset(
            {
                "q_safe_feasible",
                "q_robust_infeasible",
                "q_envelope_ambiguous",
            }
        ),
        (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_six_role_certification",
            "V1SignedWorkChartTransportStatus",
        ): frozenset(
            {"chart_certified", "chart_refuted", "chart_unresolved"}
        ),
        (
            "mechanistic_systems.circuits.irg."
            "modern_v1_signed_work_six_role_certification",
            "V1SignedWorkIntervalStatus",
        ): frozenset({"certified", "refuted", "unresolved"}),
    }
)


def _is_label_native_checkpoint_enum(value: object) -> bool:
    return bool(
        isinstance(value, _HistoricalCarrier)
        and (value.historical_module, value.historical_name)
        in _LABEL_NATIVE_CHECKPOINT_ENUM_VALUES
    )


def _raw_label_native_checkpoint_enum(value: object) -> str:
    if not _is_label_native_checkpoint_enum(value):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native checkpoint enum carrier has the wrong type",
            phase="raw_state_validation",
        )
    carrier = cast(_HistoricalCarrier, value)
    allowed = _LABEL_NATIVE_CHECKPOINT_ENUM_VALUES[
        (carrier.historical_module, carrier.historical_name)
    ]
    if (
        carrier.state is not None
        or carrier.constructor_kwargs
        or len(carrier.constructor_args) != 1
        or type(carrier.constructor_args[0]) is not str
        or carrier.constructor_args[0] not in allowed
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native checkpoint enum constructor state is invalid",
            phase="raw_state_validation",
        )
    return cast(str, carrier.constructor_args[0])


def _field_map_for(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> tuple[str, ...]:
    name = value.historical_name
    if _is_label_native_checkpoint_enum(value):
        _raw_label_native_checkpoint_enum(value)
        return ()
    if _is_fqr_branch_carrier(value):
        _raw_fqr_branch_value(value)
        return ()
    if name in _FQR_V2_RAW_FIELDS:
        if value.historical_module != _FQR_ACQUISITION_MODULE:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"mixed F/Q/R carrier {name} arrived from an unsupported module",
                phase="raw_state_mapping",
            )
        return _FQR_V2_RAW_FIELDS[name]
    expected_label_native_module = (
        _LABEL_NATIVE_CHECKPOINT_CARRIER_MODULES.get(name)
    )
    if expected_label_native_module is not None:
        if value.historical_module != expected_label_native_module:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"label-native checkpoint carrier {name} came from another module",
                phase="raw_state_mapping",
            )
        return _COMMON_RAW_FIELDS[name]
    if name == "V1RemovedRouteSetIdentity":
        if value.historical_module != (
            "mechanistic_systems.circuits.irg.modern_v1_candidate_composition"
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "compact removed-route identity arrived from an unsupported module",
                phase="raw_state_mapping",
            )
        return _REMOVED_ROUTE_SET_IDENTITY_RAW_FIELDS
    if _is_paired_bce_hashed_rebuild(value):
        _module, target_name = _paired_bce_rebuild_target(value)
        return _COMMON_RAW_FIELDS[target_name]
    if name in {"_V1ControllerCheckpointProjection", "V1ControllerState"}:
        current = _COMMON_RAW_FIELDS[name]
        milestone_fields = {
            "pareto_milestone_registration",
            "live_exact_prefix_rung_registration",
            "pending_pareto_milestone",
            "pareto_milestone_receipt",
        }
        oracle_fields = {"accepted_oracle_trajectory"}
        horizon_fields = {"fqr_no_candidate_horizon_witness"}
        label_native_fields = (
            {
                "label_native_runtime_plan_checkpoint_payload",
                "label_native_runtime_plan_hash",
                "label_native_runtime_plan_hashes",
                "label_native_rejected_plan_indices",
                "label_native_selected_acquisition_receipt_hash",
            }
            if name == "_V1ControllerCheckpointProjection"
            else {
                "label_native_runtime_plan",
                "label_native_runtime_plan_hashes",
                "label_native_rejected_plan_indices",
            }
        )
        optional_groups = (
            milestone_fields,
            oracle_fields,
            horizon_fields,
            label_native_fields,
        )
        variants = tuple(
            tuple(
                field_name
                for field_name in current
                if not any(
                    bit & (1 << group_index)
                    and field_name in group
                    for group_index, group in enumerate(optional_groups)
                )
            )
            for bit in range(1 << len(optional_groups))
        )
        if type(value.state) is dict:
            observed = set(value.state)
            for fields in variants:
                required = set(fields)
                removed = set(current) - required
                if required.issubset(observed) and not (observed & removed):
                    return fields
        else:
            arity = len(_carrier_state_sequence(value, label=name))
            for fields in variants:
                if arity == len(fields):
                    return fields
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{name} has unsupported controller projection arity",
            phase="raw_state_mapping",
        )
    if name == "V1ControllerSnapshotReceipt":
        current = _COMMON_RAW_FIELDS[name]
        horizon_fields = {"fqr_no_candidate_horizon_witness_hash"}
        label_native_fields = {
            "label_native_runtime_plan_hash",
            "label_native_runtime_plan_hashes",
            "label_native_rejected_plan_indices",
            "label_native_selected_acquisition_receipt_hash",
        }
        optional_groups = (horizon_fields, label_native_fields)
        variants = tuple(
            tuple(
                field_name
                for field_name in current
                if not any(
                    bit & (1 << group_index) and field_name in group
                    for group_index, group in enumerate(optional_groups)
                )
            )
            for bit in range(1 << len(optional_groups))
        )
        if type(value.state) is dict:
            observed = set(value.state)
            for fields in variants:
                required = set(fields)
                removed = set(current) - required
                if required.issubset(observed) and not (observed & removed):
                    return fields
        else:
            arity = len(_carrier_state_sequence(value, label=name))
            for fields in variants:
                if arity == len(fields):
                    return fields
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "V1ControllerSnapshotReceipt has unsupported additive schema arity",
            phase="raw_state_mapping",
        )
    if name == "V1ActiveCandidate":
        current = _COMMON_RAW_FIELDS[name]
        paired_fields = {"paired_bce_registration", "paired_bce_state"}
        legacy = tuple(
            field_name for field_name in current if field_name not in paired_fields
        )
        if type(value.state) is dict:
            observed = set(value.state)
            if set(current).issubset(observed):
                return current
            if set(legacy).issubset(observed) and not (observed & paired_fields):
                return legacy
        else:
            arity = len(_carrier_state_sequence(value, label=name))
            if arity == len(current):
                return current
            if arity == len(legacy):
                return legacy
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "V1ActiveCandidate has unsupported paired-BCE arity",
            phase="raw_state_mapping",
        )
    if name == "V1AcceptedOracleTrajectoryEntry":
        current = _COMMON_RAW_FIELDS[name]
        legacy = tuple(
            field_name
            for field_name in current
            if field_name != "candidate_calibration_freeze"
        )
        if type(value.state) is dict:
            observed = set(value.state)
            if set(current).issubset(observed):
                return current
            if set(legacy).issubset(observed) and (
                "candidate_calibration_freeze" not in observed
            ):
                return legacy
        else:
            arity = len(_carrier_state_sequence(value, label=name))
            if arity == len(current):
                return current
            if arity == len(legacy):
                return legacy
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "V1AcceptedOracleTrajectoryEntry has unsupported calibration-freeze arity",
            phase="raw_state_mapping",
        )
    if name == "V1RuntimeCompletedProposalClosureReceipt":
        current = _COMMON_RAW_FIELDS[name]
        alpha_family_fields = {
            "sampled_v7_D_alpha_family_plan_hash",
            "sampled_v7_D_alpha_family_registry_receipt_hash",
            "sampled_v7_D_alpha_family_registry_replay",
            "sampled_v7_D_alpha_head_CAS_receipt_hash",
            "sampled_v7_D_alpha_published_ledger_head_hash",
            "sampled_v7_D_alpha_durable_controller_head_closed",
        }
        alpha_v3 = tuple(
            field_name
            for field_name in current
            if field_name not in alpha_family_fields
        )
        alpha_v2 = tuple(
            field_name
            for field_name in alpha_v3
            if field_name != "proposal_order_base_serfling_design_hash"
        )
        legacy = tuple(
            field_name
            for field_name in alpha_v3
            if field_name
            not in {
                "proposal_order_alpha_schedule_hash",
                "proposal_order_alpha_ledger_before_hash",
                "proposal_order_design_hash",
                "proposal_order_base_serfling_design_hash",
                "proposal_order_alpha_ledger_after_hash",
                "proposal_order_alpha_ledger_after_replay",
                "proposal_order_envelope_certificate_hash",
                "proposal_order_action_family_registry_hash",
            }
        )
        if type(value.state) is dict:
            observed = set(value.state)
            if set(current).issubset(observed):
                return current
            if set(alpha_v3).issubset(observed):
                return alpha_v3
            if set(alpha_v2).issubset(observed):
                return alpha_v2
            return legacy
        arity = len(_carrier_state_sequence(value, label=name))
        if arity == len(current):
            return current
        if arity == len(alpha_v3):
            return alpha_v3
        if arity == len(alpha_v2):
            return alpha_v2
        if arity == len(legacy):
            return legacy
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{name} has unsupported alpha-coordinate arity {arity}",
            phase="raw_state_mapping",
        )
    targeted = _target_fields(cohort, name)
    if targeted is not None:
        return targeted
    if name == "_V1RuntimeCheckpointJournal":
        state = _carrier_state_sequence(value, label=name)
        if len(state) == 23:
            if state[-1] != _JOURNAL_V7_SCHEMA:
                _refuse(
                    V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                    "23-slot checkpoint journal has an unsupported schema",
                    phase="cohort_classification",
                )
            return _JOURNAL_V7_FIELDS
        if len(state) == 19:
            if state[-1] != _JOURNAL_V6_SCHEMA:
                _refuse(
                    V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                    "19-slot checkpoint journal has an unsupported schema",
                    phase="cohort_classification",
                )
            return _JOURNAL_V6_FIELDS
        if len(state) == 15:
            if state[-1] != _JOURNAL_V5_SCHEMA:
                _refuse(
                    V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                    "15-slot checkpoint journal has an unsupported schema",
                    phase="cohort_classification",
                )
            return _JOURNAL_V5_FIELDS
        if len(state) == 14:
            if state[-1] != _JOURNAL_V4_SCHEMA:
                _refuse(
                    V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                    "14-slot checkpoint journal has an unsupported schema",
                    phase="cohort_classification",
                )
            return _JOURNAL_V4_FIELDS
        if len(state) == 13:
            return (
                _JOURNAL_V3_FIELDS
            )
        if len(state) == 12:
            return _JOURNAL_V2_FIELDS
        return _COMMON_RAW_FIELDS[name]
    try:
        return _COMMON_RAW_FIELDS[name]
    except KeyError:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"historical field map is unavailable for {value.historical_module}.{name}",
            phase="raw_state_mapping",
        )
        raise AssertionError("unreachable")


def _raw_mapping(value: _HistoricalCarrier, *, cohort: _CohortSpec) -> dict[str, Any]:
    if _is_label_native_checkpoint_enum(value):
        _raw_label_native_checkpoint_enum(value)
        return {}
    if _is_fqr_branch_carrier(value):
        _raw_fqr_branch_value(value)
        return {}
    if _is_paired_bce_hashed_rebuild(value):
        return _paired_bce_rebuild_payload(value, cohort=cohort)
    fields = _field_map_for(value, cohort=cohort)
    if value.historical_name == "ConstructiveOrdinalRouteAxis":
        state = value.state
        if (
            type(state) not in {tuple, list}
            or len(state) != 2
            or state[0] is not None
            or type(state[1]) is not dict
            or frozenset(state[1]) != frozenset(fields)
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "constructive route axis has unsupported slotted state",
                phase="raw_state_mapping",
            )
        return {name: state[1][name] for name in fields}
    if type(value.state) is dict:
        observed = frozenset(value.state)
        expected = frozenset(fields)
        allowed_cached = _RAW_ALLOWED_CACHED_FIELDS.get(
            value.historical_name, frozenset()
        )
        if not expected.issubset(observed) or not (observed - expected).issubset(
            allowed_cached
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"{value.historical_name} named state has unknown or missing fields",
                phase="raw_state_mapping",
            )
        return {name: value.state[name] for name in fields}
    state = _carrier_state_sequence(value, label=value.historical_name)
    if len(state) != len(fields):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{value.historical_name} arity {len(state)} != frozen arity {len(fields)}",
            phase="raw_state_mapping",
        )
    return dict(zip(fields, state, strict=True))


def _raw_canonical(value: Any, *, cohort: _CohortSpec) -> Any:
    if isinstance(value, _HistoricalCarrier):
        if _is_label_native_checkpoint_enum(value):
            return _raw_label_native_checkpoint_enum(value)
        if _is_fqr_branch_carrier(value):
            return _raw_fqr_branch_value(value)
        if value.historical_name == "V1RemovedRouteSetIdentity":
            return _raw_removed_route_set_identity(value, cohort=cohort)
        if _is_paired_bce_hashed_rebuild(value):
            return _paired_bce_rebuild_full_canonical(value, cohort=cohort)
        if value.historical_name == "ConstructiveOrdinalRouteAxis":
            mapping = _raw_mapping(value, cohort=cohort)
            count = mapping["_count"]
            digest = mapping["_ordered_digest"]
            canonical_digest = mapping["_canonical_json_digest"]
            if type(count) is not int or count <= 0:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "constructive route axis count is invalid",
                    phase="raw_state_validation",
                )
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in _LOWER_SHA256 for character in digest)
                or (
                    canonical_digest is not None
                    and (
                        type(canonical_digest) is not str
                        or len(canonical_digest) != 64
                        or any(
                            character not in _LOWER_SHA256
                            for character in canonical_digest
                        )
                    )
                )
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "constructive route axis lacks its exact cached digest",
                    phase="raw_state_validation",
                )
            return {
                "schema": "constructive_ordinal_route_axis_ref_v1",
                "count": count,
                "ordered_digest": digest,
            }
        return {
            key: _raw_canonical(item, cohort=cohort)
            for key, item in _raw_mapping(value, cohort=cohort).items()
        }
    if isinstance(value, Mapping):
        return {key: _raw_canonical(item, cohort=cohort) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_raw_canonical(item, cohort=cohort) for item in value)
    if isinstance(value, list):
        return [_raw_canonical(item, cohort=cohort) for item in value]
    if isinstance(value, set):
        return {_raw_canonical(item, cohort=cohort) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_raw_canonical(item, cohort=cohort) for item in value)
    return value


def _raw_content_hash(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
    prefix: str,
) -> str:
    return content_hash(_raw_canonical(value, cohort=cohort), prefix=prefix)


def _observed_signals(
    instances: Sequence[_HistoricalCarrier],
    *,
    cohort: _CohortSpec,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[str, ...]]]:
    arities: dict[str, list[int]] = {}
    schemas: dict[str, list[str]] = {}
    for value in instances:
        fields = _field_map_for(value, cohort=cohort)
        mapping = _raw_mapping(value, cohort=cohort)
        arities.setdefault(value.historical_name, []).append(len(fields))
        if "schema" in fields:
            schema = mapping["schema"]
            if type(schema) is not str or not schema:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    f"{value.historical_name} schema is not an exact nonempty string",
                    phase="raw_state_validation",
                )
            schemas.setdefault(value.historical_name, []).append(schema)
    return (
        {name: tuple(sorted(set(values))) for name, values in arities.items()},
        {name: tuple(sorted(set(values))) for name, values in schemas.items()},
    )


def _classify_cohort(
    *,
    magic: bytes,
    instances: Sequence[_HistoricalCarrier],
) -> _CohortSpec:
    by_name: dict[str, list[_HistoricalCarrier]] = {}
    for value in instances:
        by_name.setdefault(value.historical_name, []).append(value)
    contracts = by_name.get("V1OptionACampaignContract", [])
    coordinates = by_name.get("V1StagedAlphaCoordinates", [])
    if not contracts or not coordinates:
        _refuse(
            V1CheckpointRefusalCode.COHORT_INCONSISTENT,
            "checkpoint lacks campaign-contract or coordinate cohort signals",
            phase="cohort_classification",
        )
    if magic == _LEGACY_MAGIC:
        selected = _PRE16
    else:
        contract_arities = {
            len(_carrier_state_sequence(item, label=item.historical_name))
            for item in contracts
        }
        coordinate_arities = {
            len(_carrier_state_sequence(item, label=item.historical_name))
            for item in coordinates
        }
        if contract_arities == {59} or coordinate_arities == {22}:
            selected = _V29_PLUS
        elif contract_arities == {58} and coordinate_arities == {20}:
            selected = _V16_TO_V28
        else:
            _refuse(
                V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                "contract and coordinate arities do not identify one cohort",
                phase="cohort_classification",
            )
    if magic == _PROPOSAL_MAGIC and selected is not _V29_PLUS:
        _refuse(
            V1CheckpointRefusalCode.COHORT_INCONSISTENT,
            "checkpoint-v3 cannot contain a pre-v29 raw graph",
            phase="cohort_classification",
        )
    expectations = {
        "V1OptionACampaignContract": (
            selected.contract_fields,
            selected.contract_schema,
        ),
        "V1StagedAlphaCoordinates": (
            selected.coordinate_fields,
            selected.coordinate_schema,
        ),
        "V1StagedAlphaStepFreeze": (selected.step_fields, selected.step_schema),
        "V1StepSamplingLaw": (selected.sampling_fields, selected.sampling_schema),
    }
    for name, (fields, schema) in expectations.items():
        for item in by_name.get(name, []):
            state = _carrier_state_sequence(item, label=name)
            if len(state) != len(fields) or not state or state[-1] != schema:
                _refuse(
                    V1CheckpointRefusalCode.COHORT_INCONSISTENT,
                    f"{name} arity/schema conflicts with {selected.label}",
                    phase="cohort_classification",
                )
    return selected


def _expect_carrier(value: Any, name: str) -> _HistoricalCarrier:
    if not isinstance(value, _HistoricalCarrier):
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            f"expected inert historical carrier {name}",
            phase="raw_root_validation",
        )
    if value.historical_name == name:
        return value
    if _is_paired_bce_hashed_rebuild(value):
        _module, target_name = _paired_bce_rebuild_target(value)
        if target_name == name:
            return value
    if value.historical_name != name:
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            f"expected inert historical carrier {name}",
            phase="raw_root_validation",
        )
    return value


def _expect_exact_carrier(
    value: Any,
    *,
    module: str,
    name: str,
) -> _HistoricalCarrier:
    carrier = _expect_carrier(value, name)
    effective_module = carrier.historical_module
    if _is_paired_bce_hashed_rebuild(carrier):
        effective_module, _target_name = _paired_bce_rebuild_target(carrier)
    if effective_module != module:
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            f"{name} arrived from unsupported module {effective_module!r}",
            phase="raw_root_validation",
        )
    return carrier


def _raw_nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be an exact nonempty string",
            phase="raw_state_validation",
        )
    return cast(str, value)


def _raw_nonnegative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be an exact nonnegative integer",
            phase="raw_state_validation",
        )
    return cast(int, value)


def _validate_proposal_state_payload(
    value: Any,
    *,
    reference: Mapping[str, Any],
) -> tuple[str, str | None]:
    payload = _expect_exact_carrier(
        value,
        module=__name__,
        name="V1RuntimeProposalStateSidecarPayload",
    )
    mapping = _raw_mapping(payload, cohort=_V29_PLUS)
    if mapping["schema"] != _PROPOSAL_STATE_PAYLOAD_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "proposal-state payload schema is unsupported",
            phase="raw_state_validation",
        )
    if mapping["proposal_progress_payload"] is None:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "proposal-state payload lacks cumulative progress",
            phase="raw_state_validation",
        )
    progress_hash = _raw_nonempty_string(
        mapping["proposal_progress_hash"],
        "proposal_progress_hash",
    )
    representation_hash = _raw_nonempty_string(
        mapping["representation_hash"],
        "representation_hash",
    )
    proposal_backend_identity_hash = _raw_nonempty_string(
        mapping["proposal_backend_identity_hash"],
        "proposal_backend_identity_hash",
    )
    pending_payload = mapping["pending_proposal_acquisition_payload"]
    pending_hash = mapping["pending_proposal_acquisition_hash"]
    if (pending_payload is None) != (pending_hash is None):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "pending proposal payload/hash nullability differs",
            phase="raw_state_validation",
        )
    if pending_hash is not None:
        _raw_nonempty_string(pending_hash, "pending_proposal_acquisition_hash")
    if (
        progress_hash != reference["proposal_progress_hash"]
        or representation_hash != reference["representation_hash"]
        or proposal_backend_identity_hash
        != reference["proposal_backend_identity_hash"]
    ):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "proposal-state payload identities differ from the framed receipt",
            phase="raw_state_validation",
        )
    return progress_hash, cast(str | None, pending_hash)


def _carrier_hash(
    value: Any,
    *,
    cohort: _CohortSpec,
    prefix: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, _HistoricalCarrier):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "hash-bearing historical state is not an inert carrier",
            phase="historical_projection",
        )
    return _raw_content_hash(value, cohort=cohort, prefix=prefix)


def _mask_hash(value: Any, *, cohort: _CohortSpec) -> str:
    carrier = _expect_carrier(value, "HardMask")
    observed = _raw_content_hash(carrier, cohort=cohort, prefix="irg-mask-")
    if type(carrier.state) is dict:
        cached_hash = carrier.state.get("mask_hash")
        if cached_hash is not None and cached_hash != observed:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "HardMask cached hash does not replay",
                phase="historical_projection",
            )
        cached_count = carrier.state.get("retained_count")
        packed = carrier.state.get("packed_bits")
        if cached_count is not None:
            if type(cached_count) is not int or type(packed) is not bytes:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "HardMask cached retained count has invalid types",
                    phase="historical_projection",
                )
            packed_bytes = cast(bytes, packed)
            if cached_count != sum(byte.bit_count() for byte in packed_bytes):
                _refuse(
                    V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                    "HardMask cached retained count does not replay",
                    phase="historical_projection",
                )
    return observed


def _binding_hash(value: Any, *, cohort: _CohortSpec) -> str:
    carrier = _expect_carrier(value, "V1OptionACampaignBinding")
    return _raw_content_hash(
        carrier,
        cohort=cohort,
        prefix="partition-free-v1-option-a-campaign-binding-",
    )


def _composition_result_hash(
    value: Any,
    *,
    cohort: _CohortSpec,
) -> str:
    result = _expect_carrier(value, "V1CandidateCompositionResult")
    mapping = _raw_mapping(result, cohort=cohort)
    proposals = mapping["ranked_proposals"]
    if type(proposals) not in {tuple, list}:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "candidate composition proposals are not positional",
            phase="raw_state_validation",
        )
    candidate_ids = tuple(
        _raw_nonempty_string(
            _raw_mapping(
                _expect_carrier(item, "V1CandidateProposal"),
                cohort=cohort,
            )["candidate_id"],
            "candidate_id",
        )
        for item in proposals
    )
    projection = {
        "schema": mapping["schema"],
        "numeric_state_hash": _raw_content_hash(
            _expect_carrier(mapping["numeric_state"], "V1CandidateNumericState"),
            cohort=cohort,
            prefix="partition-free-v1-candidate-numeric-state-",
        ),
        "domain_hash": _raw_content_hash(
            _expect_carrier(mapping["domain"], "V1FiniteRemovalDomain"),
            cohort=cohort,
            prefix="partition-free-v1-removal-domain-",
        ),
        "constraints_hash": _raw_content_hash(
            _expect_carrier(
                mapping["constraints"],
                "V1CandidateCompositionConstraints",
            ),
            cohort=cohort,
            prefix="partition-free-v1-candidate-constraints-",
        ),
        "ranked_candidate_ids": candidate_ids,
        "evaluated_action_count": mapping["evaluated_action_count"],
        "eligible_action_count": mapping["eligible_action_count"],
        "composer_id": mapping["composer_id"],
        "selection_receipt_hash": mapping["selection_receipt_hash"],
        "declared_domain_exhausted": mapping["declared_domain_exhausted"],
        "exact_within_declared_domain": mapping["exact_within_declared_domain"],
        "claims_global_powerset_optimum": mapping["claims_global_powerset_optimum"],
        "proposal_only": mapping["proposal_only"],
        "proposal_witness_only": mapping["proposal_witness_only"],
        "fresh_relinearization_required": mapping["fresh_relinearization_required"],
        "claim_execution_certified": mapping["claim_execution_certified"],
    }
    return content_hash(
        projection,
        prefix="partition-free-v1-candidate-composition-",
    )


def _relinearized_acquisition_hash(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> str:
    mapping = _raw_mapping(value, cohort=cohort)
    correction = mapping["correction"]
    if not isinstance(correction, _HistoricalCarrier):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "acquisition correction is not an inert carrier",
            phase="raw_state_validation",
        )
    if correction.historical_name == "V1CorrectionAcquisition":
        correction_prefix = "partition-free-v1-correction-acquisition-"
    elif correction.historical_name == "V1AnalyticExactGateCorrectionAcquisition":
        correction_prefix = "partition-free-v1-analytic-exact-gate-acquisition-"
    else:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "acquisition correction type is unsupported",
            phase="raw_state_validation",
        )
    projection = {
        "schema": mapping["schema"],
        "campaign_binding_hash": _binding_hash(mapping["binding"], cohort=cohort),
        "generation_index": mapping["generation_index"],
        "current_mask_hash": _mask_hash(mapping["current_mask"], cohort=cohort),
        "committed_mask_lineage_hashes": mapping["committed_mask_lineage_hashes"],
        "committed_mask_transition_receipt_hashes": (
            mapping["committed_mask_transition_receipt_hashes"]
        ),
        "fd_acquisition_hash": _raw_content_hash(
            _expect_carrier(mapping["fd"], "V1FDStreamAcquisition"),
            cohort=cohort,
            prefix="partition-free-v1-fd-stream-acquisition-",
        ),
        "correction_acquisition_hash": _raw_content_hash(
            correction,
            cohort=cohort,
            prefix=correction_prefix,
        ),
        "chart_hash": _raw_content_hash(
            _expect_carrier(mapping["chart"], "V1ProposalChart"),
            cohort=cohort,
            prefix="partition-free-v1-chart-",
        ),
        "numeric_state_hash": _raw_content_hash(
            _expect_carrier(mapping["numeric_state"], "V1CandidateNumericState"),
            cohort=cohort,
            prefix="partition-free-v1-candidate-numeric-state-",
        ),
        "domain_hash": _raw_content_hash(
            _expect_carrier(mapping["domain"], "V1FiniteRemovalDomain"),
            cohort=cohort,
            prefix="partition-free-v1-removal-domain-",
        ),
        "constraints_hash": _raw_content_hash(
            _expect_carrier(
                mapping["constraints"],
                "V1CandidateCompositionConstraints",
            ),
            cohort=cohort,
            prefix="partition-free-v1-candidate-constraints-",
        ),
        "composition_result_hash": _composition_result_hash(
            mapping["composition"],
            cohort=cohort,
        ),
        "metadata": _raw_canonical(mapping["metadata"], cohort=cohort),
    }
    return content_hash(
        projection,
        prefix="partition-free-v1-relinearized-acquisition-",
    )


_FQ_CANDIDATE_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_candidate_composition"
)
_FQ_ACQUISITION_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_acquisition"
)
_FQR_ACQUISITION_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_fqr_runtime_acquisition"
)
_FQR_COMPACT_BRANCH_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_fqr_compact_branch_family"
)
_FQR_KKT_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_relaxed_kkt"
)
_FQR_HORIZON_MODULE: Final = (
    "mechanistic_systems.circuits.irg.modern_v1_fqr_horizon"
)
_HARD_MASK_MODULE: Final = "mechanistic_systems.contracts.hard_mask_actions"
_FQ_ACTION_STATE_SCHEMA: Final = "partition_free_v1_fq_action_numeric_state_v1"
_FQ_PROPOSAL_SCHEMA: Final = "partition_free_v1_fq_candidate_proposal_v1"
_FQ_COMPOSITION_SCHEMA: Final = (
    "partition_free_v1_fq_candidate_composition_result_v1"
)
_FQ_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fq_acquisition_generation_receipt_v1"
)
_FQ_ACQUISITION_SCHEMA: Final = "partition_free_v1_fq_authorized_acquisition_v1"
_FQ_COMPACT_DOMAIN_SCHEMA: Final = "partition_free_v1_fq_compact_domain_identity_v1"
_FQ_CONSTRAINTS_SCHEMA: Final = "partition_free_v1_fq_constraints_identity_v1"
_FQ_RESOURCE_GAIN_SCHEMA: Final = (
    "partition_free_v1_fq_exact_scalar_resource_gain_receipt_v1"
)
_FQ_PACKED_LADDER_ACTION_SCHEMA: Final = "modern_v1_frozen_ladder_action_v2"
_FQ_POINT_INTERVAL_SCHEMA: Final = "modern_v1_frozen_ladder_endpoint_interval_v1"
_FQ_ENCLOSED_INTERVAL_SCHEMA: Final = (
    "modern_v1_frozen_ladder_endpoint_interval_v2"
)
_FQ_POINT_OBSERVATION_SEMANTICS: Final = (
    "registered_signed_projection_point_observation_v1"
)
_FQ_ENCLOSED_OBSERVATION_SEMANTICS: Final = (
    "exact_real_projection_functional_upper_envelope_observation_v1"
)
_FQ_COMPOSER_ID: Final = "fq_authorized_resource_descending_compact_v1"
_FQ_F_UNITS: Final = "absolute_raw_target_logit"
_FQ_Q_UNITS: Final = "squared_raw_target_logit"
_FQ_SQRT_Q_UNITS: Final = "absolute_raw_target_logit"
_FQ_DELTA_DIGEST_BUFFER_BYTES: Final = 1 << 20

_FQR_ACTION_STATE_SCHEMA: Final = "partition_free_v1_fqr_action_numeric_state_v1"
_FQR_ACTION_STATE_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_action_numeric_state_v2"
)
_FQR_PROPOSAL_SCHEMA: Final = "partition_free_v1_fqr_candidate_proposal_v1"
_FQR_PROPOSAL_V2_SCHEMA: Final = "partition_free_v1_fqr_candidate_proposal_v2"
_FQR_COMPOSITION_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_composition_result_v1"
)
_FQR_COMPOSITION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_composition_result_v2"
)
_FQR_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fqr_acquisition_generation_receipt_v1"
)
_FQR_RECEIPT_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_acquisition_generation_receipt_v2"
)
_FQR_ACQUISITION_SCHEMA: Final = "partition_free_v1_fqr_authorized_acquisition_v1"
_FQR_ACQUISITION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_authorized_acquisition_v2"
)
_FQR_SOURCE_ALIAS_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_family_source_alias_v1"
)
_FQR_COMPOSER_ID: Final = (
    "canonical_compact_fqr_cross_branch_resource_descending_queue_v1"
)
_FQR_BRANCH_VALUES: Final = frozenset({"spectral", "diagonal_l1"})


def _is_fqr_branch_carrier(value: object) -> bool:
    return bool(
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _FQR_KKT_MODULE
        and value.historical_name == "V1RelaxedKKTBranch"
    )


def _raw_fqr_branch_value(value: object) -> str:
    if not _is_fqr_branch_carrier(value):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R branch is not the exact inert relaxed-KKT enum carrier",
            phase="raw_state_validation",
        )
    carrier = cast(_HistoricalCarrier, value)
    if (
        carrier.state is not None
        or carrier.constructor_kwargs
        or len(carrier.constructor_args) != 1
        or type(carrier.constructor_args[0]) is not str
        or carrier.constructor_args[0] not in _FQR_BRANCH_VALUES
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R relaxed-KKT branch constructor state is invalid",
            phase="raw_state_validation",
        )
    return cast(str, carrier.constructor_args[0])


def _raw_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be an exact positive integer",
            phase="raw_state_validation",
        )
    return cast(int, value)


def _raw_finite_nonnegative(value: object, label: str) -> float:
    if type(value) not in {int, float} or type(value) is bool:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be a finite nonnegative scalar",
            phase="raw_state_validation",
        )
    observed = float(cast(int | float, value))
    if not math.isfinite(observed) or observed < 0.0:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be a finite nonnegative scalar",
            phase="raw_state_validation",
        )
    return observed


def _raw_exact_bool(value: object, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be exactly {expected!r}",
            phase="raw_state_validation",
        )


def _raw_tuple(value: object, label: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be exact positional tuple state",
            phase="raw_state_validation",
        )
    return cast(tuple[Any, ...], value)


def _raw_fq_hard_mask(
    value: object,
    *,
    cohort: _CohortSpec,
    label: str,
) -> dict[str, Any]:
    mask = _expect_exact_carrier(
        value,
        module=_HARD_MASK_MODULE,
        name="HardMask",
    )
    mapping = _raw_mapping(mask, cohort=cohort)
    route_axis_hash = _raw_nonempty_string(
        mapping["route_axis_hash"],
        f"{label} route_axis_hash",
    )
    route_count = _raw_positive_integer(
        mapping["route_count"],
        f"{label} route_count",
    )
    packed_bits = mapping["packed_bits"]
    if type(packed_bits) is not bytes or len(packed_bits) != (route_count + 7) // 8:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} packed byte length differs from its route count",
            phase="raw_state_validation",
        )
    padding = len(packed_bits) * 8 - route_count
    if padding and packed_bits[-1] & ((1 << padding) - 1):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} has noncanonical nonzero padding bits",
            phase="raw_state_validation",
        )
    return {
        "carrier": mask,
        "route_axis_hash": route_axis_hash,
        "route_count": route_count,
        "packed_bits": packed_bits,
        "mask_hash": _mask_hash(mask, cohort=cohort),
    }


def _raw_fq_removed_delta_identity(
    *,
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> tuple[int, str]:
    if (
        parent["route_axis_hash"] != child["route_axis_hash"]
        or parent["route_count"] != child["route_count"]
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q parent and child masks lie on different route axes",
            phase="raw_state_validation",
        )
    route_count = cast(int, parent["route_count"])
    parent_bits = cast(bytes, parent["packed_bits"])
    child_bits = cast(bytes, child["packed_bits"])
    itemsize = 4 if route_count <= (1 << 32) - 1 else 8
    dtype_code = "u32" if itemsize == 4 else "u64"
    digest = hashlib.sha256()
    buffer = bytearray()
    removed_count = 0
    for byte_index, (parent_byte, child_byte) in enumerate(
        zip(parent_bits, child_bits, strict=True)
    ):
        if child_byte & ((~parent_byte) & 0xFF):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "compact F/Q child mask reactivates a parent-pruned route",
                phase="raw_state_validation",
            )
        removed_byte = parent_byte & ((~child_byte) & 0xFF)
        if not removed_byte:
            continue
        for bit_offset in range(8):
            if not removed_byte & (0x80 >> bit_offset):
                continue
            route_index = byte_index * 8 + bit_offset
            if route_index >= route_count:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "compact F/Q delta reaches a padding coordinate",
                    phase="raw_state_validation",
                )
            buffer.extend(route_index.to_bytes(itemsize, "little", signed=False))
            removed_count += 1
            if len(buffer) >= _FQ_DELTA_DIGEST_BUFFER_BYTES:
                digest.update(buffer)
                buffer.clear()
    if buffer:
        digest.update(buffer)
    removed_hash = content_hash(
        {
            "route_count": route_count,
            "count": removed_count,
            "dtype_code": dtype_code,
            "encoded": {
                "bytes_sha256": digest.hexdigest(),
                "length": removed_count * itemsize,
            },
        },
        prefix="irg-removed-routes-",
    )
    return removed_count, removed_hash


def _raw_fq_packed_action(
    value: object,
    *,
    cohort: _CohortSpec,
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    action = _expect_exact_carrier(
        value,
        module=_FQ_CANDIDATE_MODULE,
        name="V1PackedRemovalAction",
    )
    mapping = _raw_mapping(action, cohort=cohort)
    child = _raw_fq_hard_mask(
        mapping["resulting_mask"],
        cohort=cohort,
        label="compact F/Q resulting mask",
    )
    removed_count, removed_hash = _raw_fq_removed_delta_identity(
        parent=parent,
        child=child,
    )
    if (
        _raw_positive_integer(mapping["removed_count"], "packed removed_count")
        != removed_count
        or _raw_nonempty_string(
            mapping["removed_route_index_hash"],
            "packed removed_route_index_hash",
        )
        != removed_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q packed action differs from its parent-to-child delta",
            phase="raw_state_validation",
        )
    return {
        "carrier": action,
        "mapping": mapping,
        "child": child,
        "removed_count": removed_count,
        "removed_route_index_hash": removed_hash,
        "packed_action_hash": _raw_content_hash(
            action,
            cohort=cohort,
            prefix="partition-free-v1-packed-removal-action-",
        ),
    }


def _raw_fq_action_state(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    state = _expect_exact_carrier(
        value,
        module=_FQ_CANDIDATE_MODULE,
        name="V1FQActionNumericState",
    )
    mapping = _raw_mapping(state, cohort=cohort)
    if mapping["schema"] != _FQ_ACTION_STATE_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q action-state schema is unsupported",
            phase="raw_state_validation",
        )
    endpoint = _raw_positive_integer(mapping["endpoint_index"], "endpoint_index")
    generation = _raw_nonnegative_integer(
        mapping["generation_index"],
        "generation_index",
    )
    parent = _raw_fq_hard_mask(
        mapping["parent_mask"],
        cohort=cohort,
        label="compact F/Q parent mask",
    )
    packed = _raw_fq_packed_action(
        mapping["packed_action"],
        cohort=cohort,
        parent=parent,
    )
    child = cast(Mapping[str, Any], packed["child"])
    route_count = _raw_positive_integer(mapping["route_count"], "route_count")
    for name in (
        "action_id",
        "action_hash",
        "route_axis_hash",
        "parent_mask_hash",
        "resulting_mask_hash",
        "packed_action_hash",
        "removed_route_index_hash",
        "atom_cost_source_hash",
        "resource_gain_receipt_hash",
        "fq_interval_schema",
        "fq_interval_hash",
        "fq_interval_observation_semantics",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "decisive_fq_commit_hash",
        "fq_queue_authorization_hash",
        "sampled_diagnostic_lineage_hash",
    ):
        _raw_nonempty_string(mapping[name], name)
    removed_count = _raw_positive_integer(mapping["removed_count"], "removed_count")
    if (
        route_count != parent["route_count"]
        or mapping["route_axis_hash"] != parent["route_axis_hash"]
        or mapping["parent_mask_hash"] != parent["mask_hash"]
        or mapping["resulting_mask_hash"] != child["mask_hash"]
        or mapping["packed_action_hash"] != packed["packed_action_hash"]
        or removed_count != packed["removed_count"]
        or mapping["removed_route_index_hash"]
        != packed["removed_route_index_hash"]
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q action identity differs from its packed masks",
            phase="raw_state_validation",
        )
    frozen_action_hash = content_hash(
        {
            "schema": _FQ_PACKED_LADDER_ACTION_SCHEMA,
            "endpoint_index": endpoint,
            "action_id": mapping["action_id"],
            "packed_action_hash": packed["packed_action_hash"],
            "resulting_mask_hash": child["mask_hash"],
            "route_axis_hash": child["route_axis_hash"],
            "route_count": child["route_count"],
            "removed_count": removed_count,
            "removed_route_index_hash": mapping["removed_route_index_hash"],
        }
    )
    if mapping["action_hash"] != frozen_action_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q frozen action hash does not replay its packed identity",
            phase="raw_state_validation",
        )

    numeric = {
        name: _raw_finite_nonnegative(mapping[name], name)
        for name in (
            "resource_gain",
            "f_lower",
            "f_upper",
            "q_lower",
            "q_upper",
            "sqrt_q_lower",
            "sqrt_q_upper",
            "f_upper_budget",
            "q_upper_budget",
        )
    }
    if (
        numeric["f_lower"] > numeric["f_upper"]
        or numeric["q_lower"] > numeric["q_upper"]
        or not math.isclose(
            numeric["sqrt_q_lower"],
            math.sqrt(numeric["q_lower"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            numeric["sqrt_q_upper"],
            math.sqrt(numeric["q_upper"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or numeric["f_upper"] > numeric["f_upper_budget"]
        or numeric["q_upper"] > numeric["q_upper_budget"]
        or (
            mapping["f_units"],
            mapping["q_units"],
            mapping["sqrt_q_units"],
        )
        != (_FQ_F_UNITS, _FQ_Q_UNITS, _FQ_SQRT_Q_UNITS)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q intervals, budgets, or physical units are inconsistent",
            phase="raw_state_validation",
        )
    expected_ratios: tuple[tuple[str, float | None], ...] = (
        (
            "f_upper_budget_ratio",
            None
            if numeric["f_upper_budget"] == 0.0
            else numeric["f_upper"] / numeric["f_upper_budget"],
        ),
        (
            "q_upper_budget_ratio",
            None
            if numeric["q_upper_budget"] == 0.0
            else numeric["q_upper"] / numeric["q_upper_budget"],
        ),
        (
            "sqrt_q_upper_budget_ratio",
            None
            if numeric["q_upper_budget"] == 0.0
            else numeric["sqrt_q_upper"] / math.sqrt(numeric["q_upper_budget"]),
        ),
    )
    for name, expected in expected_ratios:
        observed = mapping[name]
        if expected is None:
            valid = observed is None
        else:
            valid = (
                type(observed) in {int, float}
                and type(observed) is not bool
                and math.isfinite(float(cast(int | float, observed)))
                and math.isclose(
                    float(cast(int | float, observed)),
                    expected,
                    rel_tol=1e-15,
                    abs_tol=0.0,
                )
            )
        if not valid:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"compact F/Q ratio {name} does not replay",
                phase="raw_state_validation",
            )
    for name in (
        "fq_interval_full_census",
        "fq_interval_population_census_complete",
        "fq_interval_lower_confidence_authority",
        "fq_interval_upper_confidence_authority",
    ):
        if type(mapping[name]) is not bool:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"compact F/Q interval field {name} is not boolean",
                phase="raw_state_validation",
            )
    _raw_exact_bool(
        mapping["fq_interval_upper_confidence_authority"],
        True,
        "fq_interval_upper_confidence_authority",
    )
    interval_schema = mapping["fq_interval_schema"]
    interval_payload = {
        "schema": interval_schema,
        "endpoint_index": endpoint,
        "action_hash": frozen_action_hash,
        "f_lower": mapping["f_lower"],
        "f_upper": mapping["f_upper"],
        "q_lower": mapping["q_lower"],
        "q_upper": mapping["q_upper"],
        "sqrt_q_lower": mapping["sqrt_q_lower"],
        "sqrt_q_upper": mapping["sqrt_q_upper"],
        "empty": False,
        "full_census": mapping["fq_interval_full_census"],
        "f_units": mapping["f_units"],
        "q_units": mapping["q_units"],
        "sqrt_q_units": mapping["sqrt_q_units"],
    }
    if interval_schema == _FQ_POINT_INTERVAL_SCHEMA:
        if (
            mapping["fq_interval_full_census"]
            != mapping["fq_interval_population_census_complete"]
            or mapping["fq_interval_lower_confidence_authority"] is not True
            or mapping["fq_interval_upper_confidence_authority"] is not True
            or mapping["fq_interval_observation_semantics"]
            != _FQ_POINT_OBSERVATION_SEMANTICS
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "compact F/Q point-interval authority metadata is inconsistent",
                phase="raw_state_validation",
            )
    elif interval_schema == _FQ_ENCLOSED_INTERVAL_SCHEMA:
        if (
            mapping["fq_interval_full_census"] is not False
            or mapping["fq_interval_lower_confidence_authority"] is not False
            or mapping["fq_interval_upper_confidence_authority"] is not True
            or mapping["fq_interval_observation_semantics"]
            != _FQ_ENCLOSED_OBSERVATION_SEMANTICS
            or any(
                numeric[name] != 0.0
                for name in ("f_lower", "q_lower", "sqrt_q_lower")
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "compact F/Q enclosed-interval authority metadata is inconsistent",
                phase="raw_state_validation",
            )
        interval_payload.update(
            {
                "population_census_complete": mapping[
                    "fq_interval_population_census_complete"
                ],
                "lower_confidence_authority": mapping[
                    "fq_interval_lower_confidence_authority"
                ],
                "upper_confidence_authority": mapping[
                    "fq_interval_upper_confidence_authority"
                ],
                "observation_semantics": mapping[
                    "fq_interval_observation_semantics"
                ],
            }
        )
    else:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q endpoint-interval schema is unsupported",
            phase="raw_state_validation",
        )
    if mapping["fq_interval_hash"] != content_hash(interval_payload):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q endpoint-interval hash does not replay",
            phase="raw_state_validation",
        )
    for name in (
        "independently_certified_fq_interval",
        "exact_scalar_resource_gain_from_bound_source",
        "proposal_only",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "fq_monotonicity_assumed",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)

    resource_receipt = content_hash(
        {
            "schema": _FQ_RESOURCE_GAIN_SCHEMA,
            "endpoint_index": endpoint,
            "action_hash": mapping["action_hash"],
            "atom_cost_source_hash": mapping["atom_cost_source_hash"],
            "parent_mask_hash": parent["mask_hash"],
            "resulting_mask_hash": child["mask_hash"],
            "route_axis_hash": parent["route_axis_hash"],
            "route_count": route_count,
            "removed_count": removed_count,
            "removed_route_index_hash": mapping["removed_route_index_hash"],
            "exact_scalar_resource_gain": mapping["resource_gain"],
            "full_route_cost_array_retained": False,
        },
        prefix="partition-free-v1-fq-resource-gain-",
    )
    if mapping["resource_gain_receipt_hash"] != resource_receipt:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q resource-gain receipt does not replay",
            phase="raw_state_validation",
        )
    payload_fields = (
        "schema",
        "endpoint_index",
        "action_id",
        "action_hash",
        "route_axis_hash",
        "route_count",
        "parent_mask_hash",
        "resulting_mask_hash",
        "packed_action_hash",
        "removed_count",
        "removed_route_index_hash",
        "resource_gain",
        "atom_cost_source_hash",
        "resource_gain_receipt_hash",
        "f_lower",
        "f_upper",
        "q_lower",
        "q_upper",
        "sqrt_q_lower",
        "sqrt_q_upper",
        "f_units",
        "q_units",
        "sqrt_q_units",
        "fq_interval_schema",
        "fq_interval_hash",
        "fq_interval_full_census",
        "fq_interval_population_census_complete",
        "fq_interval_lower_confidence_authority",
        "fq_interval_upper_confidence_authority",
        "fq_interval_observation_semantics",
        "f_upper_budget",
        "q_upper_budget",
        "f_upper_budget_ratio",
        "q_upper_budget_ratio",
        "sqrt_q_upper_budget_ratio",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "decisive_fq_commit_hash",
        "fq_queue_authorization_hash",
        "sampled_diagnostic_lineage_hash",
        "generation_index",
        "independently_certified_fq_interval",
        "exact_scalar_resource_gain_from_bound_source",
        "proposal_only",
        "fq_monotonicity_assumed",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    )
    state_hash = content_hash(
        {
            name: _raw_canonical(mapping[name], cohort=cohort)
            for name in payload_fields
        },
        prefix="partition-free-v1-fq-action-numeric-state-",
    )
    cached_hash = mapping["_state_hash"]
    if cached_hash is not None and cached_hash != state_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q action-state cache does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": state,
        "mapping": mapping,
        "state_hash": state_hash,
        "parent": parent,
        "packed": packed,
        "child": child,
        "endpoint_index": endpoint,
        "generation_index": generation,
        "resource_gain": numeric["resource_gain"],
    }


def _raw_fq_candidate_proposal(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    proposal = _expect_exact_carrier(
        value,
        module=_FQ_CANDIDATE_MODULE,
        name="V1FQCandidateProposal",
    )
    mapping = _raw_mapping(proposal, cohort=cohort)
    if mapping["schema"] != _FQ_PROPOSAL_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q proposal schema is unsupported",
            phase="raw_state_validation",
        )
    state = _raw_fq_action_state(mapping["numeric_state"], cohort=cohort)
    state_mapping = cast(Mapping[str, Any], state["mapping"])
    for name in (
        "domain_hash",
        "constraints_hash",
        "evaluation_receipt_hash",
        "candidate_id",
    ):
        _raw_nonempty_string(mapping[name], name)
    for name in (
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "fq_monotonicity_assumed",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)
    evaluation_payload = {
        "schema": "partition_free_v1_fq_candidate_evaluation_receipt_v1",
        "numeric_state_hash": state["state_hash"],
        "domain_hash": mapping["domain_hash"],
        "constraints_hash": mapping["constraints_hash"],
        "endpoint_index": state_mapping["endpoint_index"],
        "action_hash": state_mapping["action_hash"],
        "parent_mask_hash": state_mapping["parent_mask_hash"],
        "resulting_mask_hash": state_mapping["resulting_mask_hash"],
        "removed_count": state_mapping["removed_count"],
        "removed_route_index_hash": state_mapping["removed_route_index_hash"],
        "resource_gain": state_mapping["resource_gain"],
        "resource_gain_receipt_hash": state_mapping["resource_gain_receipt_hash"],
        "fq_interval_hash": state_mapping["fq_interval_hash"],
        "f_interval": (state_mapping["f_lower"], state_mapping["f_upper"]),
        "q_interval": (state_mapping["q_lower"], state_mapping["q_upper"]),
        "sqrt_q_interval": (
            state_mapping["sqrt_q_lower"],
            state_mapping["sqrt_q_upper"],
        ),
        "f_units": state_mapping["f_units"],
        "q_units": state_mapping["q_units"],
        "sqrt_q_units": state_mapping["sqrt_q_units"],
        "f_upper_budget": state_mapping["f_upper_budget"],
        "q_upper_budget": state_mapping["q_upper_budget"],
        "f_upper_budget_ratio": state_mapping["f_upper_budget_ratio"],
        "q_upper_budget_ratio": state_mapping["q_upper_budget_ratio"],
        "sqrt_q_upper_budget_ratio": state_mapping[
            "sqrt_q_upper_budget_ratio"
        ],
        "fq_budget_hash": state_mapping["fq_budget_hash"],
        "decisive_fq_commit_hash": state_mapping["decisive_fq_commit_hash"],
        "fq_queue_authorization_hash": state_mapping[
            "fq_queue_authorization_hash"
        ],
        "sampled_diagnostic_lineage_hash": state_mapping[
            "sampled_diagnostic_lineage_hash"
        ],
        "independently_certified_fq_interval": True,
        "exact_registered_surrogate_evaluation": False,
        "exact_intervention_damage_evaluated": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    evaluation_hash = content_hash(
        evaluation_payload,
        prefix="partition-free-v1-fq-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": evaluation_hash,
            "resulting_mask_hash": state_mapping["resulting_mask_hash"],
            "removed_route_index_hash": state_mapping["removed_route_index_hash"],
            "fq_interval_hash": state_mapping["fq_interval_hash"],
        },
        prefix="partition-free-v1-fq-candidate-",
    )
    if (
        mapping["evaluation_receipt_hash"] != evaluation_hash
        or mapping["candidate_id"] != candidate_id
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q proposal evaluation or candidate hash does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": proposal,
        "mapping": mapping,
        "state": state,
        "candidate_id": candidate_id,
    }


def _raw_fq_domain_hash(states: Sequence[Mapping[str, Any]]) -> str:
    first = states[0]
    first_mapping = cast(Mapping[str, Any], first["mapping"])
    return content_hash(
        {
            "schema": _FQ_COMPACT_DOMAIN_SCHEMA,
            "fq_queue_authorization_hash": first_mapping[
                "fq_queue_authorization_hash"
            ],
            "decisive_fq_commit_hash": first_mapping["decisive_fq_commit_hash"],
            "parent_mask_hash": first_mapping["parent_mask_hash"],
            "route_axis_hash": first_mapping["route_axis_hash"],
            "route_count": first_mapping["route_count"],
            "queue_endpoint_indices": tuple(
                cast(Mapping[str, Any], item["mapping"])["endpoint_index"]
                for item in states
            ),
            "queue_action_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])["action_hash"]
                for item in states
            ),
            "packed_action_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])["packed_action_hash"]
                for item in states
            ),
            "resulting_mask_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])["resulting_mask_hash"]
                for item in states
            ),
            "removed_counts": tuple(
                cast(Mapping[str, Any], item["mapping"])["removed_count"]
                for item in states
            ),
            "removed_route_index_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])[
                    "removed_route_index_hash"
                ]
                for item in states
            ),
            "arbitrary_certified_endpoint_subset_preserved": True,
            "fq_monotonicity_assumed": False,
            "filtering_performed": False,
            "reordering_performed": False,
            "claims_global_powerset_domain": False,
        },
        prefix="partition-free-v1-fq-compact-domain-",
    )


def _raw_fq_constraints_hash(state: Mapping[str, Any]) -> str:
    mapping = cast(Mapping[str, Any], state["mapping"])
    return content_hash(
        {
            "schema": _FQ_CONSTRAINTS_SCHEMA,
            "fq_budget_hash": mapping["fq_budget_hash"],
            "f_upper_budget": mapping["f_upper_budget"],
            "q_upper_budget": mapping["q_upper_budget"],
            "f_units": mapping["f_units"],
            "q_units": mapping["q_units"],
            "fq_queue_authorization_hash": mapping[
                "fq_queue_authorization_hash"
            ],
            "decisive_fq_commit_hash": mapping["decisive_fq_commit_hash"],
            "independently_certified_endpoint_intervals_required": True,
            "legacy_surrogate_constraints_present": False,
        },
        prefix="partition-free-v1-fq-constraints-",
    )


def _raw_fq_composition(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    result = _expect_exact_carrier(
        value,
        module=_FQ_CANDIDATE_MODULE,
        name="V1FQCandidateCompositionResult",
    )
    mapping = _raw_mapping(result, cohort=cohort)
    if mapping["schema"] != _FQ_COMPOSITION_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q composition schema is unsupported",
            phase="raw_state_validation",
        )
    raw_states = _raw_tuple(mapping["action_numeric_states"], "F/Q action states")
    raw_proposals = _raw_tuple(mapping["proposals"], "F/Q proposals")
    if not raw_states or len(raw_states) != len(raw_proposals):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q composition has an incomplete proposal axis",
            phase="raw_state_validation",
        )
    states = tuple(_raw_fq_action_state(item, cohort=cohort) for item in raw_states)
    proposals = tuple(
        _raw_fq_candidate_proposal(item, cohort=cohort) for item in raw_proposals
    )
    for raw_state, proposal in zip(raw_states, proposals, strict=True):
        proposal_mapping = cast(Mapping[str, Any], proposal["mapping"])
        if (
            proposal_mapping["numeric_state"] is not raw_state
            or proposal_mapping["domain_hash"] != mapping["domain_hash"]
            or proposal_mapping["constraints_hash"] != mapping["constraints_hash"]
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "compact F/Q proposal recontextualized its numeric state",
                phase="raw_state_validation",
            )
    first_mapping = cast(Mapping[str, Any], states[0]["mapping"])
    shared_names = (
        "fq_queue_authorization_hash",
        "decisive_fq_commit_hash",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "sampled_diagnostic_lineage_hash",
        "atom_cost_source_hash",
        "parent_mask_hash",
        "route_axis_hash",
        "route_count",
    )
    if any(mapping[name] != first_mapping[name] for name in shared_names) or any(
        cast(Mapping[str, Any], item["mapping"])[name] != first_mapping[name]
        for item in states
        for name in shared_names
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q composition mixes source coordinates",
            phase="raw_state_validation",
        )
    endpoint_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["endpoint_index"] for item in states
    )
    action_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["action_hash"] for item in states
    )
    mask_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["resulting_mask_hash"]
        for item in states
    )
    resource_receipts = tuple(
        cast(Mapping[str, Any], item["mapping"])["resource_gain_receipt_hash"]
        for item in states
    )
    if (
        mapping["queue_endpoint_indices"] != endpoint_axis
        or mapping["queue_action_hashes"] != action_axis
        or mapping["queue_resulting_mask_hashes"] != mask_axis
        or mapping["queue_resource_gain_receipt_hashes"] != resource_receipts
        or len(set(endpoint_axis)) != len(endpoint_axis)
        or len(set(action_axis)) != len(action_axis)
        or len(set(mask_axis)) != len(mask_axis)
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q composition changed its authorized queue axes",
            phase="raw_state_validation",
        )
    if any(
        cast(float, right["resource_gain"]) > cast(float, left["resource_gain"])
        for left, right in zip(states, states[1:], strict=False)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q queue is not resource descending",
            phase="raw_state_validation",
        )
    domain_hash = _raw_fq_domain_hash(states)
    constraints_hash = _raw_fq_constraints_hash(states[0])
    if (
        mapping["domain_hash"] != domain_hash
        or mapping["constraints_hash"] != constraints_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q domain or constraints hash does not replay",
            phase="raw_state_validation",
        )
    validated_count = _raw_positive_integer(
        mapping["validated_action_count"],
        "validated_action_count",
    )
    proposal_count = _raw_positive_integer(mapping["proposal_count"], "proposal_count")
    if (
        validated_count != len(states)
        or proposal_count != len(proposals)
        or mapping["composer_id"] != _FQ_COMPOSER_ID
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q composition count or composer changed",
            phase="raw_state_validation",
        )
    for name in (
        "authorized_finite_queue_materialized_exactly",
        "arbitrary_certified_endpoint_subset_preserved",
        "resource_descending_order_verified",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "filtering_performed",
        "reordering_performed",
        "fq_monotonicity_assumed",
        "claims_global_powerset_optimum",
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)
    candidate_ids = tuple(cast(str, item["candidate_id"]) for item in proposals)
    selection_hash = content_hash(
        {
            "schema": "partition_free_v1_fq_candidate_selection_receipt_v1",
            "composer_id": mapping["composer_id"],
            "domain_hash": domain_hash,
            "constraints_hash": constraints_hash,
            "fq_queue_authorization_hash": mapping[
                "fq_queue_authorization_hash"
            ],
            "decisive_fq_commit_hash": mapping["decisive_fq_commit_hash"],
            "fq_budget_hash": mapping["fq_budget_hash"],
            "fq_certification_receipt_hash": mapping[
                "fq_certification_receipt_hash"
            ],
            "sampled_diagnostic_lineage_hash": mapping[
                "sampled_diagnostic_lineage_hash"
            ],
            "atom_cost_source_hash": mapping["atom_cost_source_hash"],
            "queue_endpoint_indices": endpoint_axis,
            "queue_action_hashes": action_axis,
            "queue_candidate_ids": candidate_ids,
            "validated_authorized_action_count": validated_count,
            "proposal_count": proposal_count,
            "authorized_finite_queue_materialized_exactly": True,
            "arbitrary_certified_endpoint_subset_preserved": True,
            "resource_descending_order_verified": True,
            "filtering_performed": False,
            "reordering_performed": False,
            "fq_monotonicity_assumed": False,
            "claims_global_powerset_optimum": False,
            "exact_registered_surrogate_evaluation": False,
            "exact_intervention_damage_evaluated": False,
            "proposal_only": True,
            "proposal_witness_only": True,
            "fresh_relinearization_required": True,
        },
        prefix="partition-free-v1-fq-candidate-selection-",
    )
    if mapping["selection_receipt_hash"] != selection_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q selection receipt does not replay",
            phase="raw_state_validation",
        )
    result_payload = {
        "schema": mapping["schema"],
        "action_numeric_state_hashes": tuple(item["state_hash"] for item in states),
        "candidate_ids": candidate_ids,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "fq_queue_authorization_hash": mapping["fq_queue_authorization_hash"],
        "decisive_fq_commit_hash": mapping["decisive_fq_commit_hash"],
        "fq_budget_hash": mapping["fq_budget_hash"],
        "fq_certification_receipt_hash": mapping[
            "fq_certification_receipt_hash"
        ],
        "sampled_diagnostic_lineage_hash": mapping[
            "sampled_diagnostic_lineage_hash"
        ],
        "atom_cost_source_hash": mapping["atom_cost_source_hash"],
        "parent_mask_hash": mapping["parent_mask_hash"],
        "route_axis_hash": mapping["route_axis_hash"],
        "route_count": mapping["route_count"],
        "queue_endpoint_indices": endpoint_axis,
        "queue_action_hashes": action_axis,
        "queue_resulting_mask_hashes": mask_axis,
        "queue_resource_gain_receipt_hashes": resource_receipts,
        "validated_action_count": validated_count,
        "proposal_count": proposal_count,
        "surrogate_evaluated_action_count": 0,
        "composer_id": mapping["composer_id"],
        "selection_receipt_hash": selection_hash,
        "authorized_finite_queue_materialized_exactly": mapping[
            "authorized_finite_queue_materialized_exactly"
        ],
        "arbitrary_certified_endpoint_subset_preserved": mapping[
            "arbitrary_certified_endpoint_subset_preserved"
        ],
        "resource_descending_order_verified": mapping[
            "resource_descending_order_verified"
        ],
        "filtering_performed": mapping["filtering_performed"],
        "reordering_performed": mapping["reordering_performed"],
        "fq_monotonicity_assumed": mapping["fq_monotonicity_assumed"],
        "claims_global_powerset_optimum": mapping["claims_global_powerset_optimum"],
        "exact_registered_surrogate_evaluation": mapping[
            "exact_registered_surrogate_evaluation"
        ],
        "exact_intervention_damage_evaluated": mapping[
            "exact_intervention_damage_evaluated"
        ],
        "proposal_only": mapping["proposal_only"],
        "proposal_witness_only": mapping["proposal_witness_only"],
        "fresh_relinearization_required": mapping[
            "fresh_relinearization_required"
        ],
        "candidate_freeze_authority": mapping["candidate_freeze_authority"],
        "bce_oracle_authority": mapping["bce_oracle_authority"],
        "mask_commit_authority": mapping["mask_commit_authority"],
        "terminal_authority": mapping["terminal_authority"],
        "pareto_authority": mapping["pareto_authority"],
        "execution_authority": mapping["execution_authority"],
    }
    result_hash = content_hash(
        result_payload,
        prefix="partition-free-v1-fq-candidate-composition-",
    )
    cached_hash = mapping["_result_hash"]
    if cached_hash is not None and cached_hash != result_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q composition cache does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": result,
        "mapping": mapping,
        "states": states,
        "proposals": proposals,
        "result_hash": result_hash,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "candidate_ids": candidate_ids,
    }


def _raw_fq_acquisition_receipt(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    receipt = _expect_exact_carrier(
        value,
        module=_FQ_ACQUISITION_MODULE,
        name="V1FQAcquisitionGenerationReceipt",
    )
    mapping = _raw_mapping(receipt, cohort=cohort)
    if mapping["schema"] != _FQ_RECEIPT_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q acquisition-receipt schema is unsupported",
            phase="raw_state_validation",
        )
    for name in (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "registration_hash",
        "current_mask_hash",
        "route_axis_hash",
        "acquisition_hash",
        "invalidation_chart_hash",
        "composition_result_hash",
        "compact_domain_hash",
        "fq_constraints_hash",
        "fq_queue_authorization_hash",
        "decisive_fq_commit_hash",
        "fq_budget_hash",
        "fq_certification_receipt_hash",
        "sampled_diagnostic_lineage_hash",
        "atom_cost_source_hash",
        "refresh_reason",
    ):
        _raw_nonempty_string(mapping[name], name)
    generation = _raw_nonnegative_integer(mapping["generation_index"], "generation_index")
    logical_sequence = _raw_nonnegative_integer(
        mapping["logical_sequence"],
        "logical_sequence",
    )
    candidate_ids = _raw_tuple(mapping["ordered_candidate_ids"], "ordered candidate IDs")
    endpoints = _raw_tuple(mapping["ordered_endpoint_indices"], "ordered endpoints")
    action_hashes = _raw_tuple(mapping["ordered_action_hashes"], "ordered action hashes")
    if (
        not candidate_ids
        or len(candidate_ids) != len(endpoints)
        or len(candidate_ids) != len(action_hashes)
        or any(type(item) is not str or not item for item in candidate_ids)
        or any(type(item) is not str or not item for item in action_hashes)
        or any(type(item) is not int or item <= 0 for item in endpoints)
        or len(set(candidate_ids)) != len(candidate_ids)
        or len(set(action_hashes)) != len(action_hashes)
        or len(set(endpoints)) != len(endpoints)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q acquisition-receipt queue axes do not align",
            phase="raw_state_validation",
        )
    _raw_exact_bool(mapping["proposal_only"], True, "proposal_only")
    for name in (
        "exact_registered_surrogate_evaluation",
        "exact_intervention_damage_evaluated",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "claim_execution_certified",
    ):
        _raw_exact_bool(mapping[name], False, name)
    return {
        "carrier": receipt,
        "mapping": mapping,
        "generation_index": generation,
        "logical_sequence": logical_sequence,
        "candidate_ids": candidate_ids,
        "endpoint_indices": endpoints,
        "action_hashes": action_hashes,
        "receipt_hash": _raw_content_hash(
            receipt,
            cohort=cohort,
            prefix="partition-free-v1-fq-acquisition-generation-",
        ),
    }


def _validate_fq_acquisition_sidecar_payload(
    value: object,
    *,
    cohort: _CohortSpec,
    reference: Mapping[str, Any],
) -> None:
    acquisition = _expect_exact_carrier(
        value,
        module=_FQ_ACQUISITION_MODULE,
        name="V1FQAuthorizedAcquisition",
    )
    for carrier in _raw_instances(acquisition):
        _raw_mapping(carrier, cohort=cohort)
    mapping = _raw_mapping(acquisition, cohort=cohort)
    if mapping["schema"] != _FQ_ACQUISITION_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q acquisition schema is unsupported",
            phase="raw_state_validation",
        )
    generation = _raw_nonnegative_integer(mapping["generation_index"], "generation_index")
    binding = _expect_exact_carrier(
        mapping["binding"],
        module="mechanistic_systems.circuits.irg.modern_v1_campaign_contract",
        name="V1OptionACampaignBinding",
    )
    binding_mapping = _raw_mapping(binding, cohort=cohort)
    binding_hash = _binding_hash(binding, cohort=cohort)
    contract = _expect_exact_carrier(
        binding_mapping["contract"],
        module="mechanistic_systems.circuits.irg.modern_v1_campaign_contract",
        name="V1OptionACampaignContract",
    )
    contract_mapping = _raw_mapping(contract, cohort=cohort)
    contract_hash = _raw_content_hash(
        contract,
        cohort=cohort,
        prefix="partition-free-v1-option-a-campaign-",
    )
    registration = _expect_exact_carrier(
        binding_mapping["registration"],
        module="mechanistic_systems.circuits.irg.modern_v1_core",
        name="V1OwnerRegistration",
    )
    registration_hash = _raw_content_hash(
        registration,
        cohort=cohort,
        prefix="partition-free-v1-registration-",
    )
    current_mask = _raw_fq_hard_mask(
        mapping["current_mask"],
        cohort=cohort,
        label="compact F/Q current mask",
    )
    composition = _raw_fq_composition(mapping["composition"], cohort=cohort)
    composition_mapping = cast(Mapping[str, Any], composition["mapping"])
    states = cast(tuple[Mapping[str, Any], ...], composition["states"])
    first_state_mapping = cast(Mapping[str, Any], states[0]["mapping"])
    lineage = _raw_tuple(
        mapping["committed_mask_lineage_hashes"],
        "committed mask lineage",
    )
    transitions = _raw_tuple(
        mapping["committed_mask_transition_receipt_hashes"],
        "committed transition lineage",
    )
    if (
        not lineage
        or any(type(item) is not str or not item for item in lineage)
        or any(type(item) is not str or not item for item in transitions)
        or lineage[-1] != current_mask["mask_hash"]
        or len(transitions) != len(lineage) - 1
        or len(set(lineage)) != len(lineage)
        or len(set(transitions)) != len(transitions)
        or contract_mapping["route_axis_hash"] != current_mask["route_axis_hash"]
        or composition_mapping["route_axis_hash"] != current_mask["route_axis_hash"]
        or composition_mapping["route_count"] != current_mask["route_count"]
        or composition_mapping["parent_mask_hash"] != current_mask["mask_hash"]
        or first_state_mapping["generation_index"] != generation
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q acquisition differs from its current generation or ancestry",
            phase="raw_state_validation",
        )
    invalidation_hash = content_hash(
        {
            "schema": "partition_free_v1_fq_invalidation_chart_v1",
            "generation_index": generation,
            "current_mask_hash": current_mask["mask_hash"],
            "route_axis_hash": current_mask["route_axis_hash"],
            "sampled_diagnostic_lineage_hash": composition_mapping[
                "sampled_diagnostic_lineage_hash"
            ],
            "fq_queue_authorization_hash": composition_mapping[
                "fq_queue_authorization_hash"
            ],
            "decisive_fq_commit_hash": composition_mapping[
                "decisive_fq_commit_hash"
            ],
            "fq_budget_hash": composition_mapping["fq_budget_hash"],
            "atom_cost_source_hash": composition_mapping["atom_cost_source_hash"],
        },
        prefix="partition-free-v1-fq-invalidation-chart-",
    )
    acquisition_hash = content_hash(
        {
            "schema": mapping["schema"],
            "campaign_binding_hash": binding_hash,
            "generation_index": generation,
            "current_mask_hash": current_mask["mask_hash"],
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "invalidation_chart_hash": invalidation_hash,
            "composition_result_hash": composition["result_hash"],
            "metadata": _raw_canonical(mapping["metadata"], cohort=cohort),
        },
        prefix="partition-free-v1-fq-authorized-acquisition-",
    )
    receipt = _raw_fq_acquisition_receipt(mapping["receipt"], cohort=cohort)
    receipt_mapping = cast(Mapping[str, Any], receipt["mapping"])
    receipt_expected = {
        "campaign_contract_hash": contract_hash,
        "campaign_binding_hash": binding_hash,
        "registration_hash": registration_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask["mask_hash"],
        "route_axis_hash": current_mask["route_axis_hash"],
        "acquisition_hash": acquisition_hash,
        "invalidation_chart_hash": invalidation_hash,
        "composition_result_hash": composition["result_hash"],
        "compact_domain_hash": composition["domain_hash"],
        "fq_constraints_hash": composition["constraints_hash"],
        "fq_queue_authorization_hash": composition_mapping[
            "fq_queue_authorization_hash"
        ],
        "decisive_fq_commit_hash": composition_mapping["decisive_fq_commit_hash"],
        "fq_budget_hash": composition_mapping["fq_budget_hash"],
        "fq_certification_receipt_hash": composition_mapping[
            "fq_certification_receipt_hash"
        ],
        "sampled_diagnostic_lineage_hash": composition_mapping[
            "sampled_diagnostic_lineage_hash"
        ],
        "atom_cost_source_hash": composition_mapping["atom_cost_source_hash"],
        "ordered_candidate_ids": composition["candidate_ids"],
        "ordered_endpoint_indices": composition_mapping["queue_endpoint_indices"],
        "ordered_action_hashes": composition_mapping["queue_action_hashes"],
    }
    if any(receipt_mapping[name] != expected for name, expected in receipt_expected.items()):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q acquisition receipt does not replay its source graph",
            phase="raw_state_validation",
        )
    reference_expected = {
        "acquisition_hash": acquisition_hash,
        "acquisition_receipt_hash": receipt["receipt_hash"],
        "campaign_binding_hash": binding_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask["mask_hash"],
        "route_axis_hash": current_mask["route_axis_hash"],
    }
    if any(reference[name] != expected for name, expected in reference_expected.items()):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "compact F/Q acquisition identities differ from the framed receipt",
            phase="raw_state_validation",
        )


_FQR_INTERVAL_NAMES: Final = (
    "f_lower",
    "f_upper",
    "q_lower",
    "q_upper",
    "r_half_lower",
    "r_half_upper",
    "r_one_lower",
    "r_one_upper",
    "half_response_lower",
    "half_response_upper",
    "one_response_lower",
    "one_response_upper",
)
_FQR_INTERVAL_PAIRS: Final = tuple(
    zip(_FQR_INTERVAL_NAMES[::2], _FQR_INTERVAL_NAMES[1::2], strict=True)
)


def _raw_fqr_source_evidence(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    rows = _raw_tuple(value, label)
    normalized: list[tuple[str, str]] = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"{label} rows must be exact two-field tuples",
                phase="raw_state_validation",
            )
        branch, evidence_hash = row
        if type(branch) is not str or branch not in _FQR_BRANCH_VALUES:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"{label} names an unsupported KKT branch",
                phase="raw_state_validation",
            )
        normalized.append(
            (branch, _raw_nonempty_string(evidence_hash, f"{label} evidence hash"))
        )
    result = tuple(normalized)
    if (
        not result
        or len({branch for branch, _digest in result}) != len(result)
        or result != tuple(sorted(result))
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} is empty, duplicated, or not in registered branch order",
            phase="raw_state_validation",
        )
    return result


def _expect_versioned_fqr_carrier(
    value: object,
    *,
    v1_name: str,
    v2_name: str,
) -> tuple[_HistoricalCarrier, bool]:
    """Admit only the exact v1/v2 nominal pair for one nested F/Q/R role."""

    if not isinstance(value, _HistoricalCarrier):
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            f"expected inert historical carrier {v1_name} or {v2_name}",
            phase="raw_root_validation",
        )
    carrier = cast(_HistoricalCarrier, value)
    if (
        carrier.historical_module != _FQR_ACQUISITION_MODULE
        or carrier.historical_name not in {v1_name, v2_name}
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            f"expected exact F/Q/R carrier union {v1_name}/{v2_name}",
            phase="raw_root_validation",
        )
    return carrier, carrier.historical_name == v2_name


def _raw_fqr_branch_provenance(
    mapping: Mapping[str, Any],
    *,
    is_v2: bool,
    label: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Validate the positive axis and the disjoint stable-negative v2 axis."""

    positive = _raw_fqr_source_evidence(
        mapping["source_branch_evidence_hashes"],
        label=f"{label} positive branch evidence",
    )
    if not is_v2:
        return positive, ()
    negative = _raw_fqr_source_evidence(
        mapping["source_branch_negative_evidence_hashes"],
        label=f"{label} stable-negative branch evidence",
    )
    positive_branches = {branch for branch, _digest in positive}
    negative_branches = {branch for branch, _digest in negative}
    if (
        len(positive) != 1
        or len(negative) != 1
        or positive_branches & negative_branches
        or positive_branches | negative_branches != _FQR_BRANCH_VALUES
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} v2 provenance is not one positive plus one distinct stable-negative branch",
            phase="raw_state_validation",
        )
    return positive, negative


def _raw_fqr_branch_endpoint_axis(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, int], ...]:
    rows = _raw_tuple(value, label)
    normalized: list[tuple[str, int]] = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"{label} rows must be exact two-field tuples",
                phase="raw_state_validation",
            )
        branch, endpoint = row
        if type(branch) is not str or branch not in _FQR_BRANCH_VALUES:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"{label} names an unsupported KKT branch",
                phase="raw_state_validation",
            )
        normalized.append(
            (branch, _raw_positive_integer(endpoint, f"{label} endpoint"))
        )
    return tuple(normalized)


def _raw_fqr_alias(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    alias = _expect_exact_carrier(
        value,
        module=_FQR_COMPACT_BRANCH_MODULE,
        name="V1CompactFQRBranchFamilySourceAlias",
    )
    mapping = _raw_mapping(alias, cohort=cohort)
    if mapping["schema"] != _FQR_SOURCE_ALIAS_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R source-alias schema is unsupported",
            phase="raw_state_validation",
        )
    branch = _raw_fqr_branch_value(mapping["branch"])
    endpoint = _raw_positive_integer(mapping["endpoint_index"], "alias endpoint")
    for name in (
        "frozen_action_hash",
        "packed_action_hash",
        "branch_binding_hash",
        "branch_evidence_hash",
        "resource_registration_hash",
        "certification_receipt_hash",
        "decision_hash",
        "proposal_authorization_hash",
        "endpoint_certificate_hash",
    ):
        _raw_nonempty_string(mapping[name], f"alias {name}")
    resource = _raw_finite_nonnegative(
        mapping["registered_resource"],
        "alias registered_resource",
    )
    if resource <= 0.0:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "alias registered_resource must be positive",
            phase="raw_state_validation",
        )
    intervals = {
        name: _raw_finite_nonnegative(mapping[name], f"alias {name}")
        for name in _FQR_INTERVAL_NAMES
    }
    if any(intervals[lower] > intervals[upper] for lower, upper in _FQR_INTERVAL_PAIRS):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R source alias contains an inverted interval",
            phase="raw_state_validation",
        )
    _raw_exact_bool(
        mapping["independently_upper_certified"],
        True,
        "alias independently_upper_certified",
    )
    payload = {
        "schema": mapping["schema"],
        "branch": branch,
        "endpoint_index": endpoint,
        "frozen_action_hash": mapping["frozen_action_hash"],
        "packed_action_hash": mapping["packed_action_hash"],
        "registered_resource": mapping["registered_resource"],
        "branch_binding_hash": mapping["branch_binding_hash"],
        "branch_evidence_hash": mapping["branch_evidence_hash"],
        "resource_registration_hash": mapping["resource_registration_hash"],
        "certification_receipt_hash": mapping["certification_receipt_hash"],
        "decision_hash": mapping["decision_hash"],
        "proposal_authorization_hash": mapping["proposal_authorization_hash"],
        "endpoint_certificate_hash": mapping["endpoint_certificate_hash"],
        **{name: mapping[name] for name in _FQR_INTERVAL_NAMES},
        "independently_upper_certified": True,
    }
    alias_hash = content_hash(
        payload,
        prefix="modern-v1-compact-fqr-branch-source-alias-",
    )
    cached = mapping["_alias_hash"]
    if cached is not None and cached != alias_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R source-alias cache does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": alias,
        "mapping": mapping,
        "branch": branch,
        "endpoint_index": endpoint,
        "registered_resource": resource,
        "intervals": intervals,
        "alias_hash": alias_hash,
    }


def _raw_fqr_intersection(
    aliases: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for lower, upper in _FQR_INTERVAL_PAIRS:
        observed_lower = max(
            cast(Mapping[str, float], alias["intervals"])[lower]
            for alias in aliases
        )
        observed_upper = min(
            cast(Mapping[str, float], alias["intervals"])[upper]
            for alias in aliases
        )
        if observed_lower > observed_upper:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "compact F/Q/R aliases have an empty simultaneous interval intersection",
                phase="raw_state_validation",
            )
        result[lower] = observed_lower
        result[upper] = observed_upper
    return result


def _raw_fqr_action_state(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    state, is_v2 = _expect_versioned_fqr_carrier(
        value,
        v1_name="V1FQRActionNumericState",
        v2_name="V1FQRActionNumericStateV2",
    )
    mapping = _raw_mapping(state, cohort=cohort)
    expected_schema = (
        _FQR_ACTION_STATE_V2_SCHEMA if is_v2 else _FQR_ACTION_STATE_SCHEMA
    )
    if mapping["schema"] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R action-state schema is unsupported",
            phase="raw_state_validation",
        )
    queue_index = _raw_nonnegative_integer(mapping["queue_index"], "queue_index")
    endpoint = _raw_positive_integer(
        mapping["representative_endpoint_index"],
        "representative_endpoint_index",
    )
    generation = _raw_nonnegative_integer(
        mapping["generation_index"],
        "generation_index",
    )
    branch = mapping["representative_branch"]
    if type(branch) is not str or branch not in _FQR_BRANCH_VALUES:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R state representative branch is unsupported",
            phase="raw_state_validation",
        )
    for name in (
        "representative_frozen_action_hash",
        "source_aliases_hash",
        "parent_mask_hash",
        "resulting_mask_hash",
        "packed_action_hash",
        "route_axis_hash",
        "removed_route_index_hash",
        "resource_units",
        "resource_source_hash",
        "resource_receipt_hash",
        "fqr_interval_hash",
        "branch_family_authorization_hash",
        "branch_family_entry_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
    ):
        _raw_nonempty_string(mapping[name], name)
    parent = _raw_fq_hard_mask(
        mapping["parent_mask"],
        cohort=cohort,
        label="compact F/Q/R parent mask",
    )
    packed = _raw_fq_packed_action(
        mapping["packed_action"],
        cohort=cohort,
        parent=parent,
    )
    child = cast(Mapping[str, Any], packed["child"])
    route_count = _raw_positive_integer(mapping["route_count"], "route_count")
    removed_count = _raw_positive_integer(mapping["removed_count"], "removed_count")
    if (
        route_count != parent["route_count"]
        or mapping["route_axis_hash"] != parent["route_axis_hash"]
        or mapping["parent_mask_hash"] != parent["mask_hash"]
        or mapping["resulting_mask_hash"] != child["mask_hash"]
        or mapping["packed_action_hash"] != packed["packed_action_hash"]
        or removed_count != packed["removed_count"]
        or mapping["removed_route_index_hash"]
        != packed["removed_route_index_hash"]
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R action identity differs from its exact packed masks",
            phase="raw_state_validation",
        )

    raw_aliases = _raw_tuple(mapping["source_aliases"], "F/Q/R source aliases")
    aliases = tuple(_raw_fqr_alias(item, cohort=cohort) for item in raw_aliases)
    if not aliases:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R state has no independently certified source alias",
            phase="raw_state_validation",
        )
    alias_hashes = tuple(cast(str, item["alias_hash"]) for item in aliases)
    if (
        len(set(alias_hashes)) != len(alias_hashes)
        or mapping["source_aliases_hash"]
        != content_hash(
            alias_hashes,
            prefix="partition-free-v1-fqr-source-aliases-",
        )
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R state source-alias identity does not replay",
            phase="raw_state_validation",
        )
    alias_keys = tuple(
        (
            -cast(float, item["registered_resource"]),
            cast(str, item["branch"]),
            cast(Mapping[str, Any], item["mapping"])["packed_action_hash"],
            cast(int, item["endpoint_index"]),
        )
        for item in aliases
    )
    if alias_keys != tuple(sorted(alias_keys)):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R state source aliases are not in registered order",
            phase="raw_state_validation",
        )
    if len(
        {
            (cast(str, item["branch"]), cast(int, item["endpoint_index"]))
            for item in aliases
        }
    ) != len(aliases):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R state repeats a branch/endpoint source coordinate",
            phase="raw_state_validation",
        )
    representative = aliases[0]
    representative_mapping = cast(Mapping[str, Any], representative["mapping"])
    resource = _raw_finite_nonnegative(
        mapping["registered_resource"],
        "registered_resource",
    )
    if resource <= 0.0:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R registered_resource must be positive",
            phase="raw_state_validation",
        )
    intervals = {
        name: _raw_finite_nonnegative(mapping[name], name)
        for name in _FQR_INTERVAL_NAMES
    }
    intersection = _raw_fqr_intersection(aliases)
    if (
        any(
            cast(Mapping[str, Any], item["mapping"])["packed_action_hash"]
            != mapping["packed_action_hash"]
            for item in aliases
        )
        or any(cast(float, item["registered_resource"]) != resource for item in aliases)
        or representative["branch"] != branch
        or representative["endpoint_index"] != endpoint
        or representative_mapping["frozen_action_hash"]
        != mapping["representative_frozen_action_hash"]
        or any(intervals[name] != intersection[name] for name in _FQR_INTERVAL_NAMES)
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R state is not the exact physical-alias quotient",
            phase="raw_state_validation",
        )
    source_evidence, negative_evidence = _raw_fqr_branch_provenance(
        mapping,
        is_v2=is_v2,
        label="F/Q/R action-state",
    )
    evidence_by_branch: dict[str, str] = {}
    for item in aliases:
        item_mapping = cast(Mapping[str, Any], item["mapping"])
        item_branch = cast(str, item["branch"])
        item_hash = cast(str, item_mapping["branch_evidence_hash"])
        existing = evidence_by_branch.get(item_branch)
        if existing is not None and existing != item_hash:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "one F/Q/R branch has conflicting compact evidence hashes",
                phase="raw_state_validation",
            )
        evidence_by_branch[item_branch] = item_hash
    registered_evidence = dict(source_evidence)
    if any(
        registered_evidence.get(branch_name) != evidence_hash
        for branch_name, evidence_hash in evidence_by_branch.items()
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "F/Q/R state source evidence does not contain its compact aliases",
            phase="raw_state_validation",
        )

    interval_payload = {
        "schema": "partition_free_v1_fqr_certified_interval_v1",
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "branch_family_entry_hash": mapping["branch_family_entry_hash"],
        "packed_action_hash": mapping["packed_action_hash"],
        "representative_branch": branch,
        "representative_endpoint_index": endpoint,
        **{name: mapping[name] for name in _FQR_INTERVAL_NAMES},
        "independently_certified_fqr_bounds": True,
    }
    interval_hash = content_hash(
        interval_payload,
        prefix="partition-free-v1-fqr-certified-interval-",
    )
    resource_payload = {
        "schema": "partition_free_v1_fqr_resource_receipt_v1",
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "branch_family_entry_hash": mapping["branch_family_entry_hash"],
        "packed_action_hash": mapping["packed_action_hash"],
        "parent_mask_hash": mapping["parent_mask_hash"],
        "resulting_mask_hash": mapping["resulting_mask_hash"],
        "registered_resource": mapping["registered_resource"],
        "resource_units": mapping["resource_units"],
        "resource_source_hash": mapping["resource_source_hash"],
    }
    resource_hash = content_hash(
        resource_payload,
        prefix="partition-free-v1-fqr-resource-",
    )
    entry_payload = {
        "schema": "partition_free_v1_compact_fqr_branch_family_queue_entry_v1",
        "representative_branch": branch,
        "representative_endpoint_index": endpoint,
        "representative_frozen_action_hash": mapping[
            "representative_frozen_action_hash"
        ],
        "packed_action_hash": mapping["packed_action_hash"],
        "parent_mask_hash": mapping["parent_mask_hash"],
        "resulting_mask_hash": mapping["resulting_mask_hash"],
        "route_axis_hash": mapping["route_axis_hash"],
        "route_count": route_count,
        "registered_resource": mapping["registered_resource"],
        "resource_units": mapping["resource_units"],
        "resource_source_hash": mapping["resource_source_hash"],
        **{name: mapping[name] for name in _FQR_INTERVAL_NAMES},
        "proposal_queue_entry_only": True,
        "packed_action_content_hash": mapping["packed_action_hash"],
        "source_alias_hashes": alias_hashes,
    }
    entry_hash = content_hash(
        entry_payload,
        prefix="modern-v1-compact-fqr-branch-queue-entry-",
    )
    if (
        mapping["fqr_interval_hash"] != interval_hash
        or mapping["resource_receipt_hash"] != resource_hash
        or mapping["branch_family_entry_hash"] != entry_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "F/Q/R interval, resource, or compact entry hash does not replay",
            phase="raw_state_validation",
        )
    for name in (
        "independently_certified_fqr_bounds",
        "complete_queue_frozen_before_K_O",
        "proposal_only",
        "paired_bce_oracle_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "candidate_freeze_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)
    state_payload = {
        "schema": mapping["schema"],
        "queue_index": queue_index,
        "representative_branch": branch,
        "representative_endpoint_index": endpoint,
        "representative_frozen_action_hash": mapping[
            "representative_frozen_action_hash"
        ],
        "source_alias_hashes": alias_hashes,
        "source_aliases_hash": mapping["source_aliases_hash"],
        "parent_mask_hash": mapping["parent_mask_hash"],
        "resulting_mask_hash": mapping["resulting_mask_hash"],
        "packed_action_hash": mapping["packed_action_hash"],
        "route_axis_hash": mapping["route_axis_hash"],
        "route_count": route_count,
        "removed_count": removed_count,
        "removed_route_index_hash": mapping["removed_route_index_hash"],
        "registered_resource": mapping["registered_resource"],
        "resource_units": mapping["resource_units"],
        "resource_source_hash": mapping["resource_source_hash"],
        "resource_receipt_hash": mapping["resource_receipt_hash"],
        "fqr_interval_hash": mapping["fqr_interval_hash"],
        "fqr_intervals": interval_payload,
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "branch_family_entry_hash": mapping["branch_family_entry_hash"],
        "prefrozen_family_hash": mapping["prefrozen_family_hash"],
        "joint_fqr_family_registration_hash": mapping[
            "joint_fqr_family_registration_hash"
        ],
        "source_branch_evidence_hashes": source_evidence,
        "generation_index": generation,
        "independently_certified_fqr_bounds": True,
        "complete_queue_frozen_before_K_O": True,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "calibration_K_open_authority": False,
        "paired_oracle_O_open_authority": False,
        "candidate_freeze_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    if is_v2:
        state_payload["source_branch_negative_evidence_hashes"] = (
            negative_evidence
        )
    state_hash = content_hash(
        state_payload,
        prefix="partition-free-v1-fqr-action-state-",
    )
    cached = mapping["_state_hash"]
    if cached is not None and cached != state_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R action-state cache does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": state,
        "mapping": mapping,
        "state_hash": state_hash,
        "parent": parent,
        "packed": packed,
        "child": child,
        "queue_index": queue_index,
        "endpoint_index": endpoint,
        "branch": branch,
        "generation_index": generation,
        "registered_resource": resource,
        "source_evidence": source_evidence,
        "negative_evidence": negative_evidence,
        "is_v2": is_v2,
        "aliases": aliases,
    }


def _raw_fqr_candidate_proposal(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    proposal = _expect_exact_carrier(
        value,
        module=_FQR_ACQUISITION_MODULE,
        name="V1FQRCandidateProposal",
    )
    mapping = _raw_mapping(proposal, cohort=cohort)
    state = _raw_fqr_action_state(mapping["numeric_state"], cohort=cohort)
    expected_schema = (
        _FQR_PROPOSAL_V2_SCHEMA
        if cast(bool, state["is_v2"])
        else _FQR_PROPOSAL_SCHEMA
    )
    if mapping["schema"] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R candidate-proposal schema is unsupported",
            phase="raw_state_validation",
        )
    state_mapping = cast(Mapping[str, Any], state["mapping"])
    for name in (
        "domain_hash",
        "certification_scope_hash",
        "evaluation_receipt_hash",
        "candidate_id",
    ):
        _raw_nonempty_string(mapping[name], name)
    for name in (
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "paired_bce_oracle_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)
    evaluation_payload = {
        "schema": (
            "partition_free_v1_fqr_candidate_evaluation_v2"
            if cast(bool, state["is_v2"])
            else "partition_free_v1_fqr_candidate_evaluation_v1"
        ),
        "numeric_state_hash": state["state_hash"],
        "domain_hash": mapping["domain_hash"],
        "certification_scope_hash": mapping["certification_scope_hash"],
        "queue_index": state["queue_index"],
        "representative_branch": state["branch"],
        "representative_endpoint_index": state["endpoint_index"],
        "packed_action_hash": state_mapping["packed_action_hash"],
        "parent_mask_hash": state_mapping["parent_mask_hash"],
        "resulting_mask_hash": state_mapping["resulting_mask_hash"],
        "removed_count": state_mapping["removed_count"],
        "removed_route_index_hash": state_mapping["removed_route_index_hash"],
        "registered_resource": state_mapping["registered_resource"],
        "resource_receipt_hash": state_mapping["resource_receipt_hash"],
        "fqr_interval_hash": state_mapping["fqr_interval_hash"],
        "branch_family_authorization_hash": state_mapping[
            "branch_family_authorization_hash"
        ],
        "branch_family_entry_hash": state_mapping["branch_family_entry_hash"],
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    evaluation_hash = content_hash(
        evaluation_payload,
        prefix="partition-free-v1-fqr-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": evaluation_hash,
            "resulting_mask_hash": state_mapping["resulting_mask_hash"],
            "removed_route_index_hash": state_mapping[
                "removed_route_index_hash"
            ],
            "fqr_interval_hash": state_mapping["fqr_interval_hash"],
            "branch_family_entry_hash": state_mapping[
                "branch_family_entry_hash"
            ],
        },
        prefix="partition-free-v1-fqr-candidate-",
    )
    if (
        mapping["evaluation_receipt_hash"] != evaluation_hash
        or mapping["candidate_id"] != candidate_id
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R proposal evaluation or candidate hash does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": proposal,
        "mapping": mapping,
        "state": state,
        "candidate_id": candidate_id,
    }


def _raw_fqr_domain_hash(states: Sequence[Mapping[str, Any]]) -> str:
    first_mapping = cast(Mapping[str, Any], states[0]["mapping"])
    return content_hash(
        {
            "schema": (
                "partition_free_v1_fqr_compact_physical_domain_v2"
                if cast(bool, states[0]["is_v2"])
                else "partition_free_v1_fqr_compact_physical_domain_v1"
            ),
            "branch_family_authorization_hash": first_mapping[
                "branch_family_authorization_hash"
            ],
            "parent_mask_hash": first_mapping["parent_mask_hash"],
            "route_axis_hash": first_mapping["route_axis_hash"],
            "route_count": first_mapping["route_count"],
            "queue_entry_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])[
                    "branch_family_entry_hash"
                ]
                for item in states
            ),
            "queue_packed_action_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])["packed_action_hash"]
                for item in states
            ),
            "queue_resulting_mask_hashes": tuple(
                cast(Mapping[str, Any], item["mapping"])["resulting_mask_hash"]
                for item in states
            ),
            "duplicate_physical_actions_quotiented_upstream": True,
            "filtering_performed": False,
            "reordering_performed": False,
        },
        prefix="partition-free-v1-fqr-compact-domain-",
    )


def _raw_fqr_certification_scope_hash(state: Mapping[str, Any]) -> str:
    mapping = cast(Mapping[str, Any], state["mapping"])
    is_v2 = cast(bool, state["is_v2"])
    payload: dict[str, Any] = {
        "schema": (
            "partition_free_v1_fqr_certification_scope_v2"
            if is_v2
            else "partition_free_v1_fqr_certification_scope_v1"
        ),
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "prefrozen_family_hash": mapping["prefrozen_family_hash"],
        "joint_fqr_family_registration_hash": mapping[
            "joint_fqr_family_registration_hash"
        ],
        "source_branch_evidence_hashes": state["source_evidence"],
        "independently_certified_fqr_bounds": True,
        "paired_bce_oracle_required": True,
    }
    if is_v2:
        payload["source_branch_negative_evidence_hashes"] = state[
            "negative_evidence"
        ]
    return content_hash(
        payload,
        prefix="partition-free-v1-fqr-certification-scope-",
    )


def _raw_fqr_composition(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    result, is_v2 = _expect_versioned_fqr_carrier(
        value,
        v1_name="V1FQRCandidateCompositionResult",
        v2_name="V1FQRCandidateCompositionResultV2",
    )
    mapping = _raw_mapping(result, cohort=cohort)
    expected_schema = (
        _FQR_COMPOSITION_V2_SCHEMA if is_v2 else _FQR_COMPOSITION_SCHEMA
    )
    if mapping["schema"] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R composition schema is unsupported",
            phase="raw_state_validation",
        )
    raw_states = _raw_tuple(mapping["action_numeric_states"], "F/Q/R action states")
    raw_proposals = _raw_tuple(mapping["proposals"], "F/Q/R proposals")
    if not raw_states or len(raw_states) != len(raw_proposals):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R composition has an incomplete proposal axis",
            phase="raw_state_validation",
        )
    states = tuple(_raw_fqr_action_state(item, cohort=cohort) for item in raw_states)
    proposals = tuple(
        _raw_fqr_candidate_proposal(item, cohort=cohort) for item in raw_proposals
    )
    if any(cast(bool, item["is_v2"]) is not is_v2 for item in states):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R composition mixes v1 and v2 nested state carriers",
            phase="raw_state_validation",
        )
    for raw_state, proposal in zip(raw_states, proposals, strict=True):
        proposal_mapping = cast(Mapping[str, Any], proposal["mapping"])
        if (
            proposal_mapping["numeric_state"] is not raw_state
            or proposal_mapping["domain_hash"] != mapping["domain_hash"]
            or proposal_mapping["certification_scope_hash"]
            != mapping["certification_scope_hash"]
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "compact F/Q/R proposal recontextualized its numeric state",
                phase="raw_state_validation",
            )
    first_mapping = cast(Mapping[str, Any], states[0]["mapping"])
    source_evidence = cast(tuple[tuple[str, str], ...], states[0]["source_evidence"])
    negative_evidence = cast(
        tuple[tuple[str, str], ...],
        states[0]["negative_evidence"],
    )
    shared_names = (
        "branch_family_authorization_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
        "parent_mask_hash",
        "route_axis_hash",
        "route_count",
    )
    if any(mapping[name] != first_mapping[name] for name in shared_names) or any(
        cast(Mapping[str, Any], item["mapping"])[name] != first_mapping[name]
        for item in states
        for name in shared_names
    ) or any(
        item["source_evidence"] != source_evidence
        or item["negative_evidence"] != negative_evidence
        for item in states
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R composition mixes source coordinates",
            phase="raw_state_validation",
        )
    composition_evidence, composition_negative_evidence = (
        _raw_fqr_branch_provenance(
            mapping,
            is_v2=is_v2,
            label="F/Q/R composition",
        )
    )
    if (
        composition_evidence != source_evidence
        or composition_negative_evidence != negative_evidence
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R composition changed its source evidence axis",
            phase="raw_state_validation",
        )

    entry_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["branch_family_entry_hash"]
        for item in states
    )
    coordinate_axis = tuple(
        (cast(str, item["branch"]), cast(int, item["endpoint_index"]))
        for item in states
    )
    packed_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["packed_action_hash"]
        for item in states
    )
    mask_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["resulting_mask_hash"]
        for item in states
    )
    interval_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["fqr_interval_hash"]
        for item in states
    )
    resource_axis = tuple(
        cast(Mapping[str, Any], item["mapping"])["resource_receipt_hash"]
        for item in states
    )
    if (
        mapping["queue_entry_hashes"] != entry_axis
        or _raw_fqr_branch_endpoint_axis(
            mapping["queue_branch_endpoint_coordinates"],
            label="composition branch/endpoint axis",
        )
        != coordinate_axis
        or mapping["queue_packed_action_hashes"] != packed_axis
        or mapping["queue_resulting_mask_hashes"] != mask_axis
        or mapping["queue_fqr_interval_hashes"] != interval_axis
        or mapping["queue_resource_receipt_hashes"] != resource_axis
        or len(set(entry_axis)) != len(entry_axis)
        or len(set(packed_axis)) != len(packed_axis)
        or len(set(mask_axis)) != len(mask_axis)
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R composition changed its physical queue axes",
            phase="raw_state_validation",
        )
    queue_indices = tuple(cast(int, item["queue_index"]) for item in states)
    if queue_indices != tuple(range(len(states))):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R queue indices are not contiguous from zero",
            phase="raw_state_validation",
        )
    order_keys = tuple(
        (
            -cast(float, item["registered_resource"]),
            cast(str, item["branch"]),
            cast(Mapping[str, Any], item["mapping"])["packed_action_hash"],
            cast(int, item["endpoint_index"]),
        )
        for item in states
    )
    if order_keys != tuple(sorted(order_keys)):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R queue is not in its predeclared deterministic order",
            phase="raw_state_validation",
        )
    domain_hash = _raw_fqr_domain_hash(states)
    scope_hash = _raw_fqr_certification_scope_hash(states[0])
    if (
        mapping["domain_hash"] != domain_hash
        or mapping["certification_scope_hash"] != scope_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R domain or certification-scope hash does not replay",
            phase="raw_state_validation",
        )
    validated_count = _raw_positive_integer(
        mapping["validated_action_count"],
        "validated_action_count",
    )
    proposal_count = _raw_positive_integer(mapping["proposal_count"], "proposal_count")
    if (
        validated_count != len(states)
        or proposal_count != len(proposals)
        or mapping["composer_id"] != _FQR_COMPOSER_ID
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R composition count or composer changed",
            phase="raw_state_validation",
        )
    for name in (
        "complete_authorized_queue_materialized_exactly",
        "duplicate_physical_actions_quotiented_upstream",
        "resource_descending_order_verified",
        "proposal_only",
        "proposal_witness_only",
        "fresh_relinearization_required",
        "paired_bce_oracle_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "filtering_performed",
        "reordering_performed",
        "fqr_monotonicity_assumed",
        "claims_hard_global_optimum",
        "candidate_freeze_authority",
        "bce_oracle_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "execution_authority",
    ):
        _raw_exact_bool(mapping[name], False, name)
    candidate_ids = tuple(cast(str, item["candidate_id"]) for item in proposals)
    selection_payload = {
        "schema": (
            "partition_free_v1_fqr_candidate_selection_v2"
            if is_v2
            else "partition_free_v1_fqr_candidate_selection_v1"
        ),
        "composer_id": mapping["composer_id"],
        "domain_hash": domain_hash,
        "certification_scope_hash": scope_hash,
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "queue_entry_hashes": entry_axis,
        "queue_candidate_ids": candidate_ids,
        "queue_branch_endpoint_coordinates": coordinate_axis,
        "queue_packed_action_hashes": packed_axis,
        "validated_action_count": validated_count,
        "proposal_count": proposal_count,
        "complete_authorized_queue_materialized_exactly": True,
        "duplicate_physical_actions_quotiented_upstream": True,
        "resource_descending_order_verified": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "fqr_monotonicity_assumed": False,
        "claims_hard_global_optimum": False,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
    }
    selection_hash = content_hash(
        selection_payload,
        prefix="partition-free-v1-fqr-candidate-selection-",
    )
    if mapping["selection_receipt_hash"] != selection_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R selection receipt does not replay",
            phase="raw_state_validation",
        )
    result_payload = {
        "schema": mapping["schema"],
        "action_numeric_state_hashes": tuple(item["state_hash"] for item in states),
        "candidate_ids": candidate_ids,
        "domain_hash": domain_hash,
        "certification_scope_hash": scope_hash,
        "branch_family_authorization_hash": mapping[
            "branch_family_authorization_hash"
        ],
        "prefrozen_family_hash": mapping["prefrozen_family_hash"],
        "joint_fqr_family_registration_hash": mapping[
            "joint_fqr_family_registration_hash"
        ],
        "source_branch_evidence_hashes": source_evidence,
        "parent_mask_hash": mapping["parent_mask_hash"],
        "route_axis_hash": mapping["route_axis_hash"],
        "route_count": mapping["route_count"],
        "queue_entry_hashes": entry_axis,
        "queue_branch_endpoint_coordinates": coordinate_axis,
        "queue_packed_action_hashes": packed_axis,
        "queue_resulting_mask_hashes": mask_axis,
        "queue_fqr_interval_hashes": interval_axis,
        "queue_resource_receipt_hashes": resource_axis,
        "validated_action_count": validated_count,
        "proposal_count": proposal_count,
        "composer_id": mapping["composer_id"],
        "selection_receipt_hash": selection_hash,
        "complete_authorized_queue_materialized_exactly": True,
        "duplicate_physical_actions_quotiented_upstream": True,
        "resource_descending_order_verified": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "fqr_monotonicity_assumed": False,
        "claims_hard_global_optimum": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    if is_v2:
        result_payload["source_branch_negative_evidence_hashes"] = (
            negative_evidence
        )
    result_hash = content_hash(
        result_payload,
        prefix="partition-free-v1-fqr-composition-",
    )
    cached = mapping["_result_hash"]
    if cached is not None and cached != result_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R composition cache does not replay",
            phase="raw_state_validation",
        )
    return {
        "carrier": result,
        "mapping": mapping,
        "states": states,
        "proposals": proposals,
        "result_hash": result_hash,
        "domain_hash": domain_hash,
        "certification_scope_hash": scope_hash,
        "candidate_ids": candidate_ids,
        "entry_hashes": entry_axis,
        "branch_endpoint_coordinates": coordinate_axis,
        "packed_action_hashes": packed_axis,
        "fqr_interval_hashes": interval_axis,
        "source_evidence": source_evidence,
        "negative_evidence": negative_evidence,
        "is_v2": is_v2,
    }


def _raw_fqr_acquisition_receipt(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    receipt, is_v2 = _expect_versioned_fqr_carrier(
        value,
        v1_name="V1FQRAcquisitionGenerationReceipt",
        v2_name="V1FQRAcquisitionGenerationReceiptV2",
    )
    mapping = _raw_mapping(receipt, cohort=cohort)
    expected_schema = _FQR_RECEIPT_V2_SCHEMA if is_v2 else _FQR_RECEIPT_SCHEMA
    if mapping["schema"] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition-receipt schema is unsupported",
            phase="raw_state_validation",
        )
    for name in (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "registration_hash",
        "current_mask_hash",
        "route_axis_hash",
        "acquisition_hash",
        "invalidation_chart_hash",
        "composition_result_hash",
        "compact_domain_hash",
        "certification_scope_hash",
        "branch_family_authorization_hash",
        "prefrozen_family_hash",
        "joint_fqr_family_registration_hash",
        "refresh_reason",
    ):
        _raw_nonempty_string(mapping[name], name)
    generation = _raw_nonnegative_integer(mapping["generation_index"], "generation_index")
    logical_sequence = _raw_nonnegative_integer(
        mapping["logical_sequence"],
        "logical_sequence",
    )
    source_evidence, negative_evidence = _raw_fqr_branch_provenance(
        mapping,
        is_v2=is_v2,
        label="F/Q/R acquisition receipt",
    )
    candidate_ids = _raw_tuple(mapping["ordered_candidate_ids"], "ordered candidate IDs")
    entry_hashes = _raw_tuple(mapping["ordered_entry_hashes"], "ordered entry hashes")
    coordinates = _raw_fqr_branch_endpoint_axis(
        mapping["ordered_branch_endpoint_coordinates"],
        label="ordered branch/endpoint coordinates",
    )
    packed_hashes = _raw_tuple(
        mapping["ordered_packed_action_hashes"],
        "ordered packed action hashes",
    )
    interval_hashes = _raw_tuple(
        mapping["ordered_fqr_interval_hashes"],
        "ordered F/Q/R interval hashes",
    )
    size = len(candidate_ids)
    if (
        size == 0
        or any(
            len(axis) != size
            for axis in (entry_hashes, coordinates, packed_hashes, interval_hashes)
        )
        or any(type(item) is not str or not item for item in candidate_ids)
        or any(type(item) is not str or not item for item in entry_hashes)
        or any(type(item) is not str or not item for item in packed_hashes)
        or any(type(item) is not str or not item for item in interval_hashes)
        or len(set(candidate_ids)) != size
        or len(set(entry_hashes)) != size
        or len(set(packed_hashes)) != size
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition-receipt queue axes do not align",
            phase="raw_state_validation",
        )
    for name in (
        "proposal_only",
        "complete_queue_frozen_before_K_O",
        "paired_bce_oracle_required",
    ):
        _raw_exact_bool(mapping[name], True, name)
    for name in (
        "calibration_K_open_authority",
        "paired_oracle_O_open_authority",
        "candidate_freeze_authority",
        "irreversible_decision_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
        "claim_execution_certified",
    ):
        _raw_exact_bool(mapping[name], False, name)
    return {
        "carrier": receipt,
        "mapping": mapping,
        "generation_index": generation,
        "logical_sequence": logical_sequence,
        "source_evidence": source_evidence,
        "negative_evidence": negative_evidence,
        "is_v2": is_v2,
        "candidate_ids": candidate_ids,
        "entry_hashes": entry_hashes,
        "coordinates": coordinates,
        "packed_hashes": packed_hashes,
        "interval_hashes": interval_hashes,
        "receipt_hash": _raw_content_hash(
            receipt,
            cohort=cohort,
            prefix="partition-free-v1-fqr-acquisition-generation-",
        ),
    }


def _validate_fqr_acquisition_sidecar_payload(
    value: object,
    *,
    cohort: _CohortSpec,
    reference: Mapping[str, Any],
) -> None:
    acquisition = _expect_exact_carrier(
        value,
        module=_FQR_ACQUISITION_MODULE,
        name="V1FQRAuthorizedAcquisition",
    )
    # Every repository carrier reachable from the sidecar must have either an
    # exact frozen slot map or the exact constructor-only relaxed-KKT enum
    # representation.  No live class receives historical slotted state.
    for carrier in _raw_instances(acquisition):
        _raw_mapping(carrier, cohort=cohort)
    mapping = _raw_mapping(acquisition, cohort=cohort)
    generation = _raw_nonnegative_integer(mapping["generation_index"], "generation_index")
    binding = _expect_exact_carrier(
        mapping["binding"],
        module="mechanistic_systems.circuits.irg.modern_v1_campaign_contract",
        name="V1OptionACampaignBinding",
    )
    binding_mapping = _raw_mapping(binding, cohort=cohort)
    binding_hash = _binding_hash(binding, cohort=cohort)
    contract = _expect_exact_carrier(
        binding_mapping["contract"],
        module="mechanistic_systems.circuits.irg.modern_v1_campaign_contract",
        name="V1OptionACampaignContract",
    )
    contract_mapping = _raw_mapping(contract, cohort=cohort)
    contract_hash = _raw_content_hash(
        contract,
        cohort=cohort,
        prefix="partition-free-v1-option-a-campaign-",
    )
    registration = _expect_exact_carrier(
        binding_mapping["registration"],
        module="mechanistic_systems.circuits.irg.modern_v1_core",
        name="V1OwnerRegistration",
    )
    registration_hash = _raw_content_hash(
        registration,
        cohort=cohort,
        prefix="partition-free-v1-registration-",
    )
    current_mask = _raw_fq_hard_mask(
        mapping["current_mask"],
        cohort=cohort,
        label="compact F/Q/R current mask",
    )
    composition = _raw_fqr_composition(mapping["composition"], cohort=cohort)
    is_v2 = cast(bool, composition["is_v2"])
    expected_acquisition_schema = (
        _FQR_ACQUISITION_V2_SCHEMA if is_v2 else _FQR_ACQUISITION_SCHEMA
    )
    if mapping["schema"] != expected_acquisition_schema:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition schema is unsupported",
            phase="raw_state_validation",
        )
    composition_mapping = cast(Mapping[str, Any], composition["mapping"])
    states = cast(tuple[Mapping[str, Any], ...], composition["states"])
    first_state_mapping = cast(Mapping[str, Any], states[0]["mapping"])
    lineage = _raw_tuple(
        mapping["committed_mask_lineage_hashes"],
        "committed mask lineage",
    )
    transitions = _raw_tuple(
        mapping["committed_mask_transition_receipt_hashes"],
        "committed transition lineage",
    )
    if (
        not lineage
        or any(type(item) is not str or not item for item in lineage)
        or any(type(item) is not str or not item for item in transitions)
        or lineage[-1] != current_mask["mask_hash"]
        or len(transitions) != len(lineage) - 1
        or len(set(lineage)) != len(lineage)
        or len(set(transitions)) != len(transitions)
        or contract_mapping["route_axis_hash"] != current_mask["route_axis_hash"]
        or composition_mapping["route_axis_hash"] != current_mask["route_axis_hash"]
        or composition_mapping["route_count"] != current_mask["route_count"]
        or composition_mapping["parent_mask_hash"] != current_mask["mask_hash"]
        or first_state_mapping["generation_index"] != generation
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R acquisition differs from its current generation or ancestry",
            phase="raw_state_validation",
        )
    invalidation_payload: dict[str, Any] = {
        "schema": (
            "partition_free_v1_fqr_invalidation_chart_v2"
            if is_v2
            else "partition_free_v1_fqr_invalidation_chart_v1"
        ),
        "generation_index": generation,
        "current_mask_hash": current_mask["mask_hash"],
        "route_axis_hash": current_mask["route_axis_hash"],
        "branch_family_authorization_hash": composition_mapping[
            "branch_family_authorization_hash"
        ],
        "prefrozen_family_hash": composition_mapping[
            "prefrozen_family_hash"
        ],
        "joint_fqr_family_registration_hash": composition_mapping[
            "joint_fqr_family_registration_hash"
        ],
        "source_branch_evidence_hashes": composition["source_evidence"],
    }
    if is_v2:
        invalidation_payload["source_branch_negative_evidence_hashes"] = (
            composition["negative_evidence"]
        )
    invalidation_hash = content_hash(
        invalidation_payload,
        prefix="partition-free-v1-fqr-invalidation-chart-",
    )
    metadata = mapping["metadata"]
    if not isinstance(metadata, Mapping):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition metadata is not a mapping",
            phase="raw_state_validation",
        )
    required_metadata = {
        "refresh_reason": None,
        "logical_sequence": None,
        "generation_index": generation,
        "authority_scope": (
            "canonical_prefrozen_cross_branch_fqr_proposal_queue_only_"
            "paired_bce_required_no_calibration_oracle_commit_terminal_or_pareto_"
            "authority_v1"
        ),
        "candidate_composition_mode": (
            "canonical_fqr_cross_branch_resource_descending_compact"
        ),
        "complete_queue_frozen_before_K_O": True,
        "full_compact_branch_authorization_embedded": False,
        "raw_C_records_embedded": False,
        "record_retaining_receipts_embedded": False,
        "full_route_numeric_state_embedded": False,
        "fqr_monotonicity_assumed": False,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "calibration_K_open_authority": False,
        "paired_oracle_O_open_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
    }
    if any(name not in metadata for name in required_metadata):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition metadata lacks its authority firewall",
            phase="raw_state_validation",
        )
    if (
        type(metadata["refresh_reason"]) is not str
        or not metadata["refresh_reason"]
        or type(metadata["logical_sequence"]) is not int
        or metadata["logical_sequence"] < 0
        or any(
            metadata[name] != expected
            for name, expected in required_metadata.items()
            if expected is not None
        )
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition metadata crossed its authority firewall",
            phase="raw_state_validation",
        )
    acquisition_hash = content_hash(
        {
            "schema": mapping["schema"],
            "campaign_binding_hash": binding_hash,
            "generation_index": generation,
            "current_mask_hash": current_mask["mask_hash"],
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "invalidation_chart_hash": invalidation_hash,
            "composition_result_hash": composition["result_hash"],
            "metadata": _raw_canonical(metadata, cohort=cohort),
        },
        prefix="partition-free-v1-fqr-authorized-acquisition-",
    )
    receipt = _raw_fqr_acquisition_receipt(mapping["receipt"], cohort=cohort)
    if cast(bool, receipt["is_v2"]) is not is_v2:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "compact F/Q/R acquisition mixes v1 and v2 receipt provenance",
            phase="raw_state_validation",
        )
    receipt_mapping = cast(Mapping[str, Any], receipt["mapping"])
    receipt_expected = {
        "campaign_contract_hash": contract_hash,
        "campaign_binding_hash": binding_hash,
        "registration_hash": registration_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask["mask_hash"],
        "route_axis_hash": current_mask["route_axis_hash"],
        "acquisition_hash": acquisition_hash,
        "invalidation_chart_hash": invalidation_hash,
        "composition_result_hash": composition["result_hash"],
        "compact_domain_hash": composition["domain_hash"],
        "certification_scope_hash": composition["certification_scope_hash"],
        "branch_family_authorization_hash": composition_mapping[
            "branch_family_authorization_hash"
        ],
        "prefrozen_family_hash": composition_mapping["prefrozen_family_hash"],
        "joint_fqr_family_registration_hash": composition_mapping[
            "joint_fqr_family_registration_hash"
        ],
        "source_branch_evidence_hashes": composition["source_evidence"],
        "ordered_candidate_ids": composition["candidate_ids"],
        "ordered_entry_hashes": composition["entry_hashes"],
        "ordered_branch_endpoint_coordinates": composition[
            "branch_endpoint_coordinates"
        ],
        "ordered_packed_action_hashes": composition["packed_action_hashes"],
        "ordered_fqr_interval_hashes": composition["fqr_interval_hashes"],
        "refresh_reason": metadata["refresh_reason"],
        "logical_sequence": metadata["logical_sequence"],
    }
    if is_v2:
        receipt_expected["source_branch_negative_evidence_hashes"] = composition[
            "negative_evidence"
        ]
    if any(receipt_mapping[name] != expected for name, expected in receipt_expected.items()):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "compact F/Q/R acquisition receipt does not replay its source graph",
            phase="raw_state_validation",
        )
    reference_expected = {
        "acquisition_hash": acquisition_hash,
        "acquisition_receipt_hash": receipt["receipt_hash"],
        "campaign_binding_hash": binding_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask["mask_hash"],
        "route_axis_hash": current_mask["route_axis_hash"],
    }
    if any(reference[name] != expected for name, expected in reference_expected.items()):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "compact F/Q/R acquisition identities differ from the framed receipt",
            phase="raw_state_validation",
        )


def _validate_acquisition_sidecar_payload(
    value: Any,
    *,
    cohort: _CohortSpec,
    reference: Mapping[str, Any],
) -> None:
    if type(value) is bytes:
        from .modern_v1_operator_checkpoint_wire import inspect_wire_payload
        try:
            wire=inspect_wire_payload(value)
            receipt=wire['receipt'];parent=wire['parent_mask']
            expected=dict(acquisition_hash=wire['acquisition_hash'],
                acquisition_receipt_hash=wire['acquisition_hash'],
                campaign_binding_hash=receipt['campaign_binding_hash'],
                generation_index=receipt['generation_index'],current_mask_hash=parent['mask_hash'],
                route_axis_hash=parent['route_axis_hash'])
            if any(reference[name]!=v for name,v in expected.items()):
                raise ValueError('operator acquisition differs from journal sidecar reference')
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,str(error),phase='raw_state_validation')
        return
    if (
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _FQ_ACQUISITION_MODULE
        and value.historical_name == "V1FQAuthorizedAcquisition"
    ):
        _validate_fq_acquisition_sidecar_payload(
            value,
            cohort=cohort,
            reference=reference,
        )
        return
    if (
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _FQR_ACQUISITION_MODULE
        and value.historical_name == "V1FQRAuthorizedAcquisition"
    ):
        _validate_fqr_acquisition_sidecar_payload(
            value,
            cohort=cohort,
            reference=reference,
        )
        return
    acquisition = _expect_exact_carrier(
        value,
        module="mechanistic_systems.circuits.irg.modern_v1_acquisition",
        name="V1RelinearizedAcquisition",
    )
    # Force exact field maps for every reachable repository carrier.  This is
    # the typed refusal boundary for unproven historical acquisition layouts.
    for carrier in _raw_instances(acquisition):
        _raw_mapping(carrier, cohort=cohort)
    mapping = _raw_mapping(acquisition, cohort=cohort)
    if mapping["schema"] != "partition_free_v1_relinearized_acquisition_v1":
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "relinearized acquisition schema is unsupported",
            phase="raw_state_validation",
        )
    generation = _raw_nonnegative_integer(
        mapping["generation_index"],
        "generation_index",
    )
    binding_hash = _binding_hash(mapping["binding"], cohort=cohort)
    mask = _expect_carrier(mapping["current_mask"], "HardMask")
    mask_mapping = _raw_mapping(mask, cohort=cohort)
    current_mask_hash = _mask_hash(mask, cohort=cohort)
    route_axis_hash = _raw_nonempty_string(
        mask_mapping["route_axis_hash"],
        "route_axis_hash",
    )
    receipt = _expect_exact_carrier(
        mapping["receipt"],
        module="mechanistic_systems.circuits.irg.modern_v1_receipts",
        name="V1AcquisitionGenerationReceipt",
    )
    receipt_mapping = _raw_mapping(receipt, cohort=cohort)
    if (
        receipt_mapping["schema"]
        != "partition_free_v1_acquisition_generation_receipt_v2"
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "acquisition-generation receipt schema is unsupported",
            phase="raw_state_validation",
        )
    for name, required in (
        ("proposal_only", True),
        ("claim_execution_certified", False),
        ("mask_commit_authority", False),
    ):
        if (
            type(receipt_mapping[name]) is not bool
            or receipt_mapping[name] is not required
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                f"acquisition receipt authority field {name} is invalid",
                phase="raw_state_validation",
            )
    acquisition_hash = _relinearized_acquisition_hash(acquisition, cohort=cohort)
    receipt_hash = _raw_content_hash(
        receipt,
        cohort=cohort,
        prefix="partition-free-v1-acquisition-generation-",
    )
    expected = {
        "acquisition_hash": acquisition_hash,
        "acquisition_receipt_hash": receipt_hash,
        "campaign_binding_hash": binding_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask_hash,
        "route_axis_hash": route_axis_hash,
    }
    if any(
        reference[name] != expected_value for name, expected_value in expected.items()
    ):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "acquisition payload identities differ from the framed receipt",
            phase="raw_state_validation",
        )
    receipt_expected = {
        "acquisition_hash": acquisition_hash,
        "campaign_binding_hash": binding_hash,
        "generation_index": generation,
        "current_mask_hash": current_mask_hash,
        "route_axis_hash": route_axis_hash,
    }
    if any(
        receipt_mapping[name] != expected_value
        for name, expected_value in receipt_expected.items()
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "acquisition receipt does not replay its payload identities",
            phase="raw_state_validation",
        )


def _logical_config_hash(value: Any, *, cohort: _CohortSpec) -> str:
    carrier = _expect_carrier(value, "V1RuntimeExecutionConfig")
    mapping = _raw_mapping(carrier, cohort=cohort)
    from .modern_v1_operator_run_binding import validate_config_binding
    try:
        validate_config_binding(_raw_canonical(mapping['metadata'],cohort=cohort),mapping['source_seal_hash'])
    except (ValueError,TypeError,KeyError) as error:
        _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,str(error),phase='raw_state_validation')
    return content_hash(
        {
            "schema": mapping["schema"],
            "executor_identity_hash": mapping["executor_identity_hash"],
            "source_seal_hash": mapping["source_seal_hash"],
            "environment_hash": mapping["environment_hash"],
            "provider_provenance": _raw_canonical(
                mapping["provider_provenance"], cohort=cohort
            ),
        },
        prefix="partition-free-v1-runtime-logical-config-",
    )


def _acquisition_hash_from_state(value: Any, *, cohort: _CohortSpec) -> str | None:
    if value is None:
        return None
    acquisition = _expect_carrier(value, "V1RelinearizedAcquisition")
    mapping = _raw_mapping(acquisition, cohort=cohort)
    receipt = _expect_carrier(mapping["receipt"], "V1AcquisitionGenerationReceipt")
    receipt_mapping = _raw_mapping(receipt, cohort=cohort)
    result = receipt_mapping["acquisition_hash"]
    if type(result) is not str or not result:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "acquisition receipt lacks its sealed acquisition hash",
            phase="historical_projection",
        )
    return result


def _active_hash(value: Any, *, cohort: _CohortSpec) -> str | None:
    if value is None:
        return None
    if type(value) is dict:
        active=_operator_active_mapping(value)
        return _raw_content_hash(active,cohort=cohort,prefix='partition-free-v1-active-candidate-')
    carrier = _expect_carrier(value, "V1ActiveCandidate")
    return _raw_content_hash(
        carrier,
        cohort=cohort,
        prefix="partition-free-v1-active-candidate-",
    )


def _operator_active_mapping(value):
    from .modern_v1_operator_checkpoint_wire import validate_active_projection
    try:
        return validate_active_projection(value)
    except (ValueError,TypeError,KeyError,OverflowError) as error:
        _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,str(error),phase='raw_state_validation')


def _trajectory_hash(value: Any, *, cohort: _CohortSpec) -> str:
    carrier = _expect_carrier(value, "V1StagedAlphaTrajectoryLedger")
    return _raw_content_hash(
        carrier,
        cohort=cohort,
        prefix="partition-free-v1-staged-alpha-trajectory-",
    )


def _accepted_oracle_trajectory_hash(value: Any, *, cohort: _CohortSpec) -> str | None:
    if value is None:
        return None
    carrier = _expect_carrier(value, "V1AcceptedOracleTrajectoryLedger")
    return _raw_content_hash(
        carrier,
        cohort=cohort,
        prefix="partition-free-v1-accepted-oracle-trajectory-",
    )


def _attempt_hashes(values: Any, *, cohort: _CohortSpec) -> tuple[str, ...]:
    if type(values) not in {tuple, list}:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "attempt receipts are not a positional sequence",
            phase="historical_projection",
        )
    result: list[str] = []
    for value in values:
        carrier = _expect_carrier(value, "V1CandidateAttemptReceipt")
        mapping = _raw_mapping(carrier, cohort=cohort)
        _validate_raw_candidate_attempt(mapping, cohort=cohort)
        result.append(
            _raw_content_hash(
                carrier,
                cohort=cohort,
                prefix="partition-free-v1-candidate-attempt-",
            )
        )
    return tuple(result)


def _raw_nonnegative_pair(value: object, *, label: str) -> tuple[int, int]:
    if type(value) not in {tuple, list} or len(value) != 2 or any(
        type(item) is not int or item < 0 for item in value
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"{label} must be two nonnegative integers",
            phase="historical_projection",
        )
    return int(value[0]), int(value[1])  # type: ignore[index]


def _validate_raw_calibration_horizon_refusal(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "calibration-horizon skip lacks its nested refusal",
            phase="historical_projection",
        )
    refusal = dict(value)
    expected = {
        "schema",
        "status",
        "candidate_mask_hash",
        "candidate_attempt_index",
        "candidate_freeze_hash",
        "calibration_stream_hash",
        "full_population_stratum_counts",
        "calibration_sampling_capacity_stratum_counts",
        "registered_look_cumulative_stratum_counts",
        "last_attempt_cumulative_stratum_counts",
        "last_fit_failure_receipt",
        "last_fit_failure_receipt_hash",
        "oracle_stream_opened",
        "oracle_population_reserved_nonempty",
        "terminal_outcomes_opened",
        "refusal_receipt_hash",
    }
    full = _raw_nonnegative_pair(
        refusal.get("full_population_stratum_counts"),
        label="calibration refusal full population",
    )
    capacity = _raw_nonnegative_pair(
        refusal.get("calibration_sampling_capacity_stratum_counts"),
        label="calibration refusal capacity",
    )
    raw_schedule = refusal.get("registered_look_cumulative_stratum_counts")
    schedule = (
        tuple(
            _raw_nonnegative_pair(row, label="calibration refusal K look")
            for row in raw_schedule
        )
        if type(raw_schedule) in {tuple, list}
        else ()
    )
    terminal = _raw_nonnegative_pair(
        refusal.get("last_attempt_cumulative_stratum_counts"),
        label="calibration refusal terminal K look",
    )
    failure = refusal.get("last_fit_failure_receipt")
    if not isinstance(failure, Mapping):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "calibration-horizon refusal lacks its fit-failure receipt",
            phase="historical_projection",
        )
    failure_payload = dict(failure)
    failure_hash = failure_payload.pop("failure_receipt_hash", None)
    declared_refusal_hash = refusal.pop("refusal_receipt_hash", None)
    previous = (0, 0)
    schedule_valid = bool(schedule)
    for row in schedule:
        schedule_valid = schedule_valid and (
            all(
                0 < row[index] <= capacity[index]
                and previous[index] <= row[index]
                for index in (0, 1)
            )
            and any(row[index] > previous[index] for index in (0, 1))
        )
        previous = row
    if (
        set(value) != expected
        or refusal.get("schema")
        != "partition_free_v1_alexnet_causal_calibration_horizon_refusal_v1"
        or refusal.get("status")
        != "failed_closed_calibration_convergence_by_n_minus_1"
        or type(refusal.get("candidate_attempt_index")) is not int
        or refusal["candidate_attempt_index"] < 0
        or any(
            type(refusal.get(name)) is not str or not refusal.get(name)
            for name in (
                "candidate_mask_hash",
                "candidate_freeze_hash",
                "calibration_stream_hash",
                "last_fit_failure_receipt_hash",
            )
        )
        or any(item <= 1 for item in full)
        or capacity != (full[0] - 1, full[1] - 1)
        or not schedule_valid
        or terminal != schedule[-1]
        or terminal != capacity
        or failure_hash != refusal.get("last_fit_failure_receipt_hash")
        or failure_hash
        != content_hash(
            failure_payload,
            prefix="partition-free-v1-alexnet-mask-calibration-failure-",
        )
        or refusal.get("oracle_stream_opened") is not False
        or refusal.get("oracle_population_reserved_nonempty") is not True
        or refusal.get("terminal_outcomes_opened") is not False
        or type(declared_refusal_hash) is not str
        or declared_refusal_hash
        != content_hash(
            refusal,
            prefix=(
                "partition-free-v1-alexnet-causal-calibration-horizon-refusal-"
            ),
        )
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "calibration-horizon refusal does not replay its N-1 identity",
            phase="historical_projection",
        )
    refusal["refusal_receipt_hash"] = declared_refusal_hash
    return refusal


def _validate_raw_candidate_attempt(
    mapping: Mapping[str, Any],
    *,
    cohort: _CohortSpec,
) -> None:
    action = mapping.get("action")
    schema = mapping.get("schema")
    if action != "skip_candidate_calibration_refusal":
        if (
            schema != "partition_free_v1_candidate_attempt_receipt_v1"
            or type(mapping.get("step_freeze_hash")) is not str
            or not mapping.get("step_freeze_hash")
            or not tuple(mapping.get("draw_ids", ()))
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "ordinary candidate attempt no longer has its v1 lineage",
                phase="historical_projection",
            )
        return
    if (
        schema != "partition_free_v1_candidate_attempt_receipt_v2"
        or mapping.get("step_freeze_hash") is not None
        or tuple(mapping.get("draw_ids", ()))
        or tuple(mapping.get("source_context_ids", ()))
        or mapping.get("mask_mutated") is not False
        or mapping.get("resulting_current_mask_hash")
        != mapping.get("parent_mask_hash")
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "calibration-horizon attempt crossed its v2 no-oracle firewall",
            phase="historical_projection",
        )
    metadata_raw = _raw_canonical(mapping.get("metadata"), cohort=cohort)
    if not isinstance(metadata_raw, Mapping):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "calibration-horizon attempt lacks frozen metadata",
            phase="historical_projection",
        )
    metadata = dict(metadata_raw)
    decision_raw = metadata.get("calibration_horizon_skip_decision")
    if not isinstance(decision_raw, Mapping):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "calibration-horizon attempt lacks its full decision payload",
            phase="historical_projection",
        )
    decision = dict(decision_raw)
    decision_fields = {
        "campaign_contract_hash",
        "controller_run_id",
        "acquisition_receipt_hash",
        "acquisition_generation_index",
        "accepted_step_index",
        "queue_index",
        "candidate_id",
        "parent_mask_hash",
        "candidate_mask_hash",
        "candidate_threshold_hash",
        "maximum_mean_damage",
        "candidate_attempt_alpha_axis_hash",
        "candidate_attempt_alpha_coordinate_hash",
        "candidate_attempt_index",
        "maximum_candidate_attempts",
        "oracle_familywise_alpha",
        "alpha_committed_before_attempt",
        "attempt_alpha_allocation",
        "master_decision_alpha_registration_hash",
        "causal_horizon_refusal_receipt",
        "status",
        "reason",
        "attempt_alpha_retired",
        "unused_alpha_recycled",
        "sampling_law_opened",
        "oracle_registration_opened",
        "oracle_stream_opened",
        "oracle_rows_observed",
        "mask_mutated",
        "mask_commit_authority",
        "alpha_nonrecycling_policy",
        "authority_scope",
        "schema",
    }
    refusal = _validate_raw_calibration_horizon_refusal(
        decision.get("causal_horizon_refusal_receipt")
    )
    separate_refusal = metadata.get("causal_horizon_refusal_receipt")
    maximum = decision.get("maximum_candidate_attempts")
    attempt_index = decision.get("candidate_attempt_index")
    allocation = decision.get("attempt_alpha_allocation")
    axis_hash = decision.get("candidate_attempt_alpha_axis_hash")
    coordinate_hash = decision.get("candidate_attempt_alpha_coordinate_hash")
    coordinate_payload = {
        "schema": "modern_v1_paired_bce_attempt_alpha_coordinate_v1",
        "candidate_attempt_alpha_axis_hash": axis_hash,
        "candidate_attempt_index": attempt_index,
        "maximum_candidate_attempts": maximum,
        "attempt_alpha_allocation": allocation,
        "alpha_nonrecycling_policy": (
            "global_candidate_attempt_alpha_axis_fixed_preoutcome_"
            "nontransferable_nonrecycling_v1"
        ),
    }
    decision_hash = content_hash(decision)
    ledger_hash = content_hash(
        {
            "schema": (
                "partition_free_v1_paired_bce_calibration_horizon_"
                "refusal_ledger_v1"
            ),
            "causal_horizon_refusal_receipt_hash": refusal.get(
                "refusal_receipt_hash"
            ),
            "candidate_attempt_alpha_axis_hash": axis_hash,
            "candidate_attempt_alpha_coordinate_hash": coordinate_hash,
            "candidate_attempt_index": attempt_index,
            "attempt_alpha_retired": True,
            "unused_alpha_recycled": False,
        },
        prefix="partition-free-v1-calibration-horizon-refusal-ledger-",
    )
    exact = {
        "campaign_contract_hash": mapping.get("campaign_contract_hash"),
        "controller_run_id": mapping.get("controller_run_id"),
        "acquisition_receipt_hash": mapping.get("acquisition_receipt_hash"),
        "acquisition_generation_index": mapping.get(
            "acquisition_generation_index"
        ),
        "accepted_step_index": mapping.get("accepted_step_index"),
        "queue_index": mapping.get("queue_index"),
        "candidate_id": mapping.get("candidate_id"),
        "parent_mask_hash": mapping.get("parent_mask_hash"),
        "candidate_mask_hash": mapping.get("candidate_mask_hash"),
        "candidate_attempt_index": mapping.get("candidate_attempt_index"),
        "reason": mapping.get("reason"),
    }
    alpha_values = (
        decision.get("oracle_familywise_alpha"),
        decision.get("alpha_committed_before_attempt"),
        allocation,
    )
    master_hash = decision.get("master_decision_alpha_registration_hash")
    maximum_mean_damage = decision.get("maximum_mean_damage")
    if (
        set(decision) != decision_fields
        or any(decision.get(name) != value for name, value in exact.items())
        or decision.get("schema")
        != "partition_free_v1_paired_bce_calibration_horizon_skip_decision_v1"
        or decision.get("status") != "skip_candidate_calibration_refusal"
        or decision.get("reason")
        != "causal_candidate_calibration_failed_by_registered_n_minus_1_horizon"
        or decision.get("alpha_nonrecycling_policy")
        != (
            "global_candidate_attempt_alpha_axis_fixed_preoutcome_"
            "nontransferable_nonrecycling_v1"
        )
        or decision.get("authority_scope")
        != "statistical_candidate_attempt_evidence_only_no_mask_commit_v1"
        or type(maximum) is not int
        or type(attempt_index) is not int
        or not 0 <= attempt_index < maximum
        or type(axis_hash) is not str
        or not axis_hash
        or (
            master_hash is not None
            and (type(master_hash) is not str or not master_hash)
        )
        or type(decision.get("candidate_threshold_hash")) is not str
        or not decision.get("candidate_threshold_hash")
        or type(maximum_mean_damage) not in {int, float}
        or not math.isfinite(float(maximum_mean_damage))
        or float(maximum_mean_damage) < 0.0
        or any(type(item) not in {int, float} for item in alpha_values)
        or any(not math.isfinite(float(item)) for item in alpha_values)
        or not 0.0 < float(alpha_values[0]) < 1.0
        or float(alpha_values[1]) < 0.0
        or not 0.0 < float(alpha_values[2]) < 1.0
        or float(alpha_values[1]) + float(alpha_values[2])
        > float(alpha_values[0]) + 64.0 * math.ulp(1.0)
        or coordinate_hash != content_hash(coordinate_payload)
        or refusal.get("candidate_mask_hash") != mapping.get("candidate_mask_hash")
        or refusal.get("candidate_attempt_index")
        != mapping.get("candidate_attempt_index")
        or separate_refusal != decision.get("causal_horizon_refusal_receipt")
        or metadata.get("calibration_horizon_skip_decision_hash") != decision_hash
        or metadata.get("causal_horizon_refusal_receipt_hash")
        != refusal.get("refusal_receipt_hash")
        or metadata.get("candidate_threshold_hash")
        != decision.get("candidate_threshold_hash")
        or metadata.get("maximum_mean_damage") != maximum_mean_damage
        or metadata.get("candidate_attempt_alpha_axis_hash") != axis_hash
        or metadata.get("candidate_attempt_alpha_coordinate_hash")
        != coordinate_hash
        or metadata.get("candidate_attempt_alpha_index") != attempt_index
        or metadata.get("maximum_candidate_attempts") != maximum
        or metadata.get("oracle_familywise_alpha") != alpha_values[0]
        or metadata.get("alpha_committed_before_attempt") != alpha_values[1]
        or metadata.get("attempt_alpha_allocation") != allocation
        or metadata.get("master_decision_alpha_registration_hash") != master_hash
        or metadata.get("oracle_protocol")
        != "paired_bce_calibration_horizon_skip"
        or metadata.get("attempt_alpha_retired") is not True
        or metadata.get("unused_alpha_recycled") is not False
        or metadata.get("sampling_law_opened") is not False
        or metadata.get("oracle_registration_opened") is not False
        or metadata.get("oracle_stream_opened") is not False
        or metadata.get("oracle_rows_observed") is not False
        or decision.get("attempt_alpha_retired") is not True
        or decision.get("unused_alpha_recycled") is not False
        or decision.get("sampling_law_opened") is not False
        or decision.get("oracle_registration_opened") is not False
        or decision.get("oracle_stream_opened") is not False
        or decision.get("oracle_rows_observed") is not False
        or decision.get("mask_mutated") is not False
        or decision.get("mask_commit_authority") is not False
        or mapping.get("decision_receipt_hash") != decision_hash
        or mapping.get("ledger_hash") != ledger_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "calibration-horizon attempt nested decision/alpha does not replay",
            phase="historical_projection",
        )


def _raw_fqr_no_candidate_horizon_witness(
    value: object,
    *,
    cohort: _CohortSpec,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Validate and replay the compact stable-negative F/Q/R horizon graph."""

    witness_carrier = _expect_exact_carrier(
        value,
        module=_FQR_HORIZON_MODULE,
        name="V1ControllerProposalHorizonWitness",
    )
    witness = _raw_mapping(witness_carrier, cohort=cohort)
    proof_carrier = _expect_exact_carrier(
        witness["proof"],
        module=_FQR_HORIZON_MODULE,
        name="V1FQRNoCandidateHorizonProofReceipt",
    )
    proof = _raw_mapping(proof_carrier, cohort=cohort)

    proof_identity_fields = (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "chart_identity_hash",
        "fqr_graph_hash",
        "fqr_policy_hash",
        "fqr_branch_family_design_hash",
        "prefrozen_branch_family_hash",
        "complete_train_population_hash",
        "causal_generation_frame_hash",
        "causal_discovery_stream_hash",
        "causal_discovery_prefix_freeze_hash",
        "causal_certification_stream_hash",
        "sampled_d_sufficient_state_hash",
        "fqr_acquisition_design_hash",
        "fqr_progress_receipt_hash",
        "fqr_readiness_receipt_hash",
        "fqr_accumulator_snapshot_hash",
        "fqr_transaction_chain_hash",
        "proposal_sufficient_state_hash",
        "proposal_sufficient_state_descriptor_hash",
        "fqr_c_state_payload_hash",
        "native_execution_binding_chain_hash",
    )
    for name in proof_identity_fields:
        _raw_nonempty_string(proof[name], f"F/Q/R horizon proof {name}")
    generation = _raw_nonnegative_integer(
        proof["generation_index"],
        "F/Q/R horizon proof generation",
    )
    if proof["minimum_nonempty_action_count"] != 1:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon proof lacks the canonical one-route action",
            phase="raw_state_validation",
        )
    branch_axis = _raw_tuple(
        proof["registered_branch_axis"],
        "F/Q/R horizon registered branch axis",
    )
    if branch_axis not in {
        ("spectral",),
        ("diagonal_l1",),
        ("spectral", "diagonal_l1"),
    }:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon registered branch axis is not canonical",
            phase="raw_state_validation",
        )
    negative_axis = _raw_tuple(
        proof["branch_negative_evidence_hashes"],
        "F/Q/R horizon negative branch evidence",
    )
    if (
        len(negative_axis) != len(branch_axis)
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not str
            or not item[1]
            for item in negative_axis
        )
        or tuple(item[0] for item in negative_axis) != branch_axis
        or len({item[1] for item in negative_axis}) != len(negative_axis)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon negative evidence does not cover its exact branch axis",
            phase="raw_state_validation",
        )
    observed_counts = _raw_tuple(
        proof["shared_observed_counts_by_stratum"],
        "F/Q/R horizon observed counts",
    )
    if (
        not observed_counts
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or type(item[1]) is not int
            or type(item[1]) is bool
            or item[1] < 0
            for item in observed_counts
        )
        or len({item[0] for item in observed_counts}) != len(observed_counts)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon observed-count axis is invalid",
            phase="raw_state_validation",
        )
    for name in (
        "all_registered_branches_decision_stable",
        "all_frozen_endpoints_lower_refuted",
        "minimum_nonempty_action_registered_in_every_branch",
        "adaptive_optional_stopping_valid",
    ):
        _raw_exact_bool(proof[name], True, f"F/Q/R horizon proof {name}")
    for name in (
        "candidate_queue_materialized",
        "calibration_K_opened",
        "paired_oracle_O_opened",
        "mask_mutated",
        "validation_outcomes_observed",
        "candidate_freeze_authority",
        "calibration_or_oracle_authority",
        "mask_commit_authority",
        "terminal_validation_authority",
        "pareto_claim_authority",
    ):
        _raw_exact_bool(proof[name], False, f"F/Q/R horizon proof {name}")
    if (
        proof["readiness_status"] != "ready_for_cross_branch_fqr_screening"
        or proof["family_refusal_code"] != "no_certified_endpoint"
        or proof["schema"]
        != "partition_free_v1_fqr_no_candidate_horizon_proof_v1"
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon proof status or schema is unsupported",
            phase="raw_state_validation",
        )

    witness_identity_fields = (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "pre_stop_controller_state_hash",
        "pre_stop_controller_snapshot_hash",
        "terminal_proposal_progress_hash",
        "terminal_proposal_frontier_hash",
        "proposal_joint_state_hash",
        "proposal_phase_evidence_state_hash",
        "proposal_sufficient_state_hash",
        "proposal_sufficient_state_descriptor_hash",
        "verified_receipt_graph_hash",
        "freeze_evaluation_receipt_hash",
        "freeze_evidence_receipt_hash",
        "freeze_decision_receipt_hash",
    )
    for name in witness_identity_fields:
        _raw_nonempty_string(witness[name], f"F/Q/R horizon witness {name}")
    witness_generation = _raw_nonnegative_integer(
        witness["generation_index"],
        "F/Q/R horizon witness generation",
    )
    pre_sequence = _raw_nonnegative_integer(
        witness["pre_stop_logical_sequence"],
        "F/Q/R horizon witness pre-stop sequence",
    )
    post_sequence = _raw_nonnegative_integer(
        witness["post_stop_logical_sequence"],
        "F/Q/R horizon witness post-stop sequence",
    )
    proof_alignment_fields = (
        "campaign_contract_hash",
        "campaign_binding_hash",
        "controller_run_id",
        "proposal_backend_identity_hash",
        "current_mask_hash",
        "route_axis_hash",
        "mask_lineage_hash",
        "proposal_sufficient_state_hash",
        "proposal_sufficient_state_descriptor_hash",
    )
    if (
        witness_generation != generation
        or post_sequence != pre_sequence + 1
        or any(witness[name] != proof[name] for name in proof_alignment_fields)
        or witness["stop_reason"] != _FQR_HORIZON_STOP_REASON
        or witness["schema"]
        != "partition_free_v1_controller_proposal_horizon_witness_v1"
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "F/Q/R horizon witness does not replay its proof coordinates",
            phase="raw_state_validation",
        )
    _raw_exact_bool(
        witness["proposal_progress_retention_required"],
        True,
        "F/Q/R horizon proposal-progress retention",
    )
    for name in (
        "candidate_queue_materialized",
        "calibration_K_opened",
        "paired_oracle_O_opened",
        "mask_mutated",
        "validation_outcomes_observed",
        "mask_commit_authority",
        "terminal_validation_authority",
        "pareto_claim_authority",
    ):
        _raw_exact_bool(witness[name], False, f"F/Q/R horizon witness {name}")

    proof_hash = _raw_content_hash(
        proof_carrier,
        cohort=cohort,
        prefix="partition-free-v1-fqr-no-candidate-horizon-proof-",
    )
    witness_hash = _raw_content_hash(
        witness_carrier,
        cohort=cohort,
        prefix="partition-free-v1-controller-proposal-horizon-witness-",
    )
    return witness, proof, witness_hash, proof_hash


def _transition_hashes(values: Any, *, cohort: _CohortSpec) -> tuple[str, ...]:
    if type(values) not in {tuple, list}:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "transition receipts are not a positional sequence",
            phase="historical_projection",
        )
    return tuple(
        _raw_content_hash(
            _expect_carrier(value, "V1MaskTransitionReceipt"),
            cohort=cohort,
            prefix="partition-free-v1-mask-transition-",
        )
        for value in values
    )


def _raw_label_native_runtime_plan_checkpoint_hash(
    value: object,
    *,
    cohort: _CohortSpec,
) -> str:
    if not isinstance(value, Mapping) or (
        set(value) != _LABEL_NATIVE_CHECKPOINT_PAYLOAD_FIELDS
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native runtime plan checkpoint payload has stale fields",
            phase="historical_projection",
        )
    payload = cast(Mapping[str, Any], value)
    if (
        payload["schema"] != _LABEL_NATIVE_PLAN_CHECKPOINT_PAYLOAD_SCHEMA
        or payload["plan_schema"] != _LABEL_NATIVE_PLAN_SCHEMA
        or payload["descriptor_schema"] != _LABEL_NATIVE_DESCRIPTOR_SCHEMA
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native runtime plan checkpoint payload schema is unsupported",
            phase="historical_projection",
        )
    raw_records = payload["descriptor_records"]
    if (
        type(raw_records) is not tuple
        or not raw_records
        or len(raw_records) > _MAX_LABEL_NATIVE_CHECKPOINT_DESCRIPTORS
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native runtime plan checkpoint descriptor axis is unbounded",
            phase="historical_projection",
        )
    descriptor_hashes: list[str] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or (
            set(raw_record) != _LABEL_NATIVE_DESCRIPTOR_RECORD_FIELDS
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "label-native checkpoint descriptor record has stale fields",
                phase="historical_projection",
            )
        record = cast(Mapping[str, Any], raw_record)
        raw_fields = record["fields"]
        if not isinstance(raw_fields, Mapping) or (
            set(raw_fields) != set(_LABEL_NATIVE_DESCRIPTOR_INIT_FIELDS)
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "label-native checkpoint descriptor has stale init fields",
                phase="historical_projection",
            )
        fields = cast(Mapping[str, Any], raw_fields)
        descriptor_projection = {
            name: _raw_canonical(fields[name], cohort=cohort)
            for name in _LABEL_NATIVE_DESCRIPTOR_INIT_FIELDS
        }
        descriptor_projection["schema"] = _LABEL_NATIVE_DESCRIPTOR_SCHEMA
        descriptor_hash = content_hash(
            descriptor_projection,
            prefix=_LABEL_NATIVE_DESCRIPTOR_PREFIX,
        )
        if record["descriptor_hash"] != descriptor_hash:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "label-native checkpoint descriptor does not replay its hash",
                phase="historical_projection",
            )
        descriptor_hashes.append(descriptor_hash)
    raw_plan_fields = payload["plan_fields"]
    if not isinstance(raw_plan_fields, Mapping) or (
        set(raw_plan_fields) != set(_LABEL_NATIVE_PLAN_INIT_FIELDS)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "label-native checkpoint plan has stale init fields",
            phase="historical_projection",
        )
    plan_fields = cast(Mapping[str, Any], raw_plan_fields)
    if (
        plan_fields["authorized_descriptor_count"] != len(descriptor_hashes)
        or tuple(plan_fields["ordered_descriptor_hashes"])
        != tuple(descriptor_hashes)
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "label-native checkpoint plan descriptor ledger is stale",
            phase="historical_projection",
        )
    plan_projection = {
        name: _raw_canonical(plan_fields[name], cohort=cohort)
        for name in _LABEL_NATIVE_PLAN_INIT_FIELDS
    }
    plan_projection.update(
        {
            "descriptors": tuple(descriptor_hashes),
            "schema": _LABEL_NATIVE_PLAN_SCHEMA,
        }
    )
    plan_hash = content_hash(
        plan_projection,
        prefix=_LABEL_NATIVE_PLAN_PREFIX,
    )
    if payload["plan_hash"] != plan_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "label-native checkpoint plan does not replay its hash",
            phase="historical_projection",
        )
    return plan_hash


def _controller_projection(
    controller: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    mapping = _raw_mapping(controller, cohort=cohort)
    is_compact = controller.historical_name == "_V1ControllerCheckpointProjection"
    if controller.historical_name not in {
        "_V1ControllerCheckpointProjection",
        "V1ControllerState",
    }:
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            "runtime root contains an unsupported controller carrier",
            phase="historical_projection",
        )
    binding = mapping["binding"]
    acquisition_hash: str | None
    acquisition_receipt_hash: str | None = None
    if is_compact:
        sidecar = mapping["acquisition_sidecar"]
        if sidecar is None:
            acquisition_hash = None
        else:
            sidecar_mapping = _raw_mapping(
                _expect_carrier(sidecar, "V1RuntimeAcquisitionSidecarReceipt"),
                cohort=cohort,
            )
            acquisition_hash = sidecar_mapping["acquisition_hash"]
            acquisition_receipt_hash = sidecar_mapping[
                "acquisition_receipt_hash"
            ]
    else:
        acquisition_hash = _acquisition_hash_from_state(
            mapping["acquisition"], cohort=cohort
        )
    raw_plan_history = mapping.get("label_native_runtime_plan_hashes", ())
    plan_history = tuple(
        _raw_nonempty_string(
            value,
            "controller label-native runtime plan history entry",
        )
        for value in _raw_tuple(
            raw_plan_history,
            "controller label-native runtime plan history",
        )
    )
    if len(set(plan_history)) != len(plan_history):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "controller label-native runtime plan history is reused",
            phase="historical_projection",
        )
    rejected_plan_indices = tuple(
        _raw_nonnegative_integer(
            value,
            "controller label-native rejected plan index",
        )
        for value in _raw_tuple(
            mapping.get("label_native_rejected_plan_indices", ()),
            "controller label-native rejected plan indices",
        )
    )
    if is_compact:
        plan_checkpoint_payload = mapping.get(
            "label_native_runtime_plan_checkpoint_payload"
        )
        plan_hash = mapping.get("label_native_runtime_plan_hash")
        selected_receipt_hash = mapping.get(
            "label_native_selected_acquisition_receipt_hash"
        )
    else:
        plan_checkpoint_payload = None
        runtime_plan = mapping.get("label_native_runtime_plan")
        if runtime_plan is not None:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "live label-native runtime plans are deliberately nonserializable",
                phase="historical_projection",
            )
        plan_hash = None
        selected_receipt_hash = None
    if plan_hash is not None:
        plan_hash = _raw_nonempty_string(
            plan_hash,
            "controller current label-native runtime plan hash",
        )
    if plan_checkpoint_payload is not None:
        payload_plan_hash = _raw_label_native_runtime_plan_checkpoint_hash(
            plan_checkpoint_payload,
            cohort=cohort,
        )
        if plan_hash is None or payload_plan_hash != plan_hash:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller label-native plan payload differs from its hash",
                phase="historical_projection",
            )
    elif plan_hash is not None:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "controller live label-native plan lacks its compact payload",
            phase="historical_projection",
        )
    if selected_receipt_hash is not None:
        selected_receipt_hash = _raw_nonempty_string(
            selected_receipt_hash,
            "controller selected label-native acquisition receipt hash",
        )
    has_label_native = bool(
        plan_checkpoint_payload is not None
        or plan_hash is not None
        or plan_history
        or rejected_plan_indices
        or selected_receipt_hash is not None
    )
    if has_label_native:
        if not plan_history:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "controller label-native state lacks immutable plan history",
                phase="historical_projection",
            )
        if plan_hash is None:
            if rejected_plan_indices or selected_receipt_hash is not None:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "retired controller label-native plan retained live coordinates",
                    phase="historical_projection",
                )
        else:
            queue_cursor = _raw_nonnegative_integer(
                mapping["queue_cursor"],
                "controller queue cursor",
            )
            if (
                plan_history[-1] != plan_hash
                or rejected_plan_indices != tuple(range(queue_cursor))
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller label-native cursor/rejections differ from its plan",
                    phase="historical_projection",
                )
            if mapping["phase"] in {"queue_ready", "candidate_active"}:
                if (
                    selected_receipt_hash is None
                    or acquisition_receipt_hash is None
                    or selected_receipt_hash != acquisition_receipt_hash
                ):
                    _refuse(
                        V1CheckpointRefusalCode.RAW_STATE_INVALID,
                        "controller label-native queue lacks its selected endpoint",
                        phase="historical_projection",
                    )
            elif selected_receipt_hash is not None:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller label-native nonqueue phase retained an endpoint",
                    phase="historical_projection",
                )
        if selected_receipt_hash is not None and (
            selected_receipt_hash
            not in tuple(mapping["acquisition_receipt_hashes"])
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "controller label-native endpoint is absent from acquisition history",
                phase="historical_projection",
            )
    attempt_hashes = _attempt_hashes(mapping["attempt_receipts"], cohort=cohort)
    transition_hashes = _transition_hashes(
        mapping["transition_receipts"], cohort=cohort
    )
    accepted_oracle_hash = _accepted_oracle_trajectory_hash(
        mapping.get("accepted_oracle_trajectory"),
        cohort=cohort,
    )
    horizon_value = mapping.get("fqr_no_candidate_horizon_witness")
    if (horizon_value is not None) != (
        mapping["stop_reason"] == _FQR_HORIZON_STOP_REASON
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "controller F/Q/R horizon reason and witness disagree",
            phase="historical_projection",
        )
    horizon_witness: dict[str, Any] | None = None
    horizon_proof: dict[str, Any] | None = None
    horizon_witness_hash: str | None = None
    if horizon_value is not None:
        (
            horizon_witness,
            horizon_proof,
            horizon_witness_hash,
            _horizon_proof_hash,
        ) = _raw_fqr_no_candidate_horizon_witness(
            horizon_value,
            cohort=cohort,
        )
    projection = {
        "schema": mapping["controller_schema"] if is_compact else mapping["schema"],
        "campaign_binding_hash": _binding_hash(binding, cohort=cohort),
        "controller_run_id": mapping["controller_run_id"],
        "initial_mask_hash": _mask_hash(mapping["initial_mask"], cohort=cohort),
        "current_mask_hash": _mask_hash(mapping["current_mask"], cohort=cohort),
        "phase": mapping["phase"],
        "acquisition_hash": acquisition_hash,
        "queue_cursor": mapping["queue_cursor"],
        "active_candidate_hash": _active_hash(
            mapping["active_candidate"], cohort=cohort
        ),
        "accepted_trajectory_hash": (
            accepted_oracle_hash
            if accepted_oracle_hash is not None
            else _trajectory_hash(mapping["accepted_trajectory"], cohort=cohort)
        ),
        "attempt_receipt_hashes": attempt_hashes,
        "transition_receipt_hashes": transition_hashes,
        "acquisition_receipt_hashes": mapping["acquisition_receipt_hashes"],
        "globally_used_draw_ids": mapping["globally_used_draw_ids"],
        "stop_reason": mapping["stop_reason"],
        "logical_sequence": mapping["logical_sequence"],
    }
    if plan_history:
        projection.update(
            {
                "label_native_runtime_plan_hash": plan_hash,
                "label_native_runtime_plan_hashes": plan_history,
                "label_native_rejected_plan_indices": rejected_plan_indices,
                "label_native_selected_acquisition_receipt_hash": (
                    selected_receipt_hash
                ),
            }
        )
    milestone_registration = mapping.get("pareto_milestone_registration")
    live_rung = mapping.get("live_exact_prefix_rung_registration")
    pending_milestone = mapping.get("pending_pareto_milestone")
    milestone_receipt = mapping.get("pareto_milestone_receipt")
    if milestone_registration is None:
        if any(
            value is not None
            for value in (live_rung, pending_milestone, milestone_receipt)
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "controller milestone state lacks its scalar registration",
                phase="historical_projection",
            )
    else:
        projection.update(
            {
                "pareto_milestone_registration_hash": _raw_content_hash(
                    _expect_carrier(
                        milestone_registration,
                        "V1ParetoMilestoneRegistration",
                    ),
                    cohort=cohort,
                    prefix="partition-free-v1-pareto-milestone-registration-",
                ),
                "live_exact_prefix_rung_registration_hash": (
                    None
                    if live_rung is None
                    else _raw_content_hash(
                        _expect_carrier(live_rung, "V1ExactPrefixRungRegistration"),
                        cohort=cohort,
                        prefix="partition-free-v1-exact-prefix-rung-registration-",
                    )
                ),
                "pending_pareto_milestone_hash": (
                    None
                    if pending_milestone is None
                    else _raw_content_hash(
                        _expect_carrier(
                            pending_milestone,
                            "V1ParetoMilestonePending",
                        ),
                        cohort=cohort,
                        prefix="partition-free-v1-pareto-milestone-pending-",
                    )
                ),
                "pareto_milestone_receipt_hash": (
                    None
                    if milestone_receipt is None
                    else _raw_content_hash(
                        _expect_carrier(
                            milestone_receipt,
                            "V1ParetoMilestoneReceipt",
                        ),
                        cohort=cohort,
                        prefix="partition-free-v1-pareto-milestone-receipt-",
                    )
                ),
            }
        )
    if horizon_witness_hash is not None:
        projection["fqr_no_candidate_horizon_witness_hash"] = (
            horizon_witness_hash
        )
    if horizon_witness is not None and horizon_proof is not None:
        current_mask = _raw_mapping(
            _expect_carrier(mapping["current_mask"], "HardMask"),
            cohort=cohort,
        )
        transition_children = tuple(
            _raw_mapping(
                _expect_carrier(item, "V1MaskTransitionReceipt"),
                cohort=cohort,
            )["child_mask_hash"]
            for item in mapping["transition_receipts"]
        )
        lineage_hash = content_hash(
            {
                "schema": "partition_free_v1_runtime_proposal_mask_lineage_v1",
                "mask_lineage_hashes": (
                    projection["initial_mask_hash"],
                    *transition_children,
                ),
                "transition_receipt_hashes": transition_hashes,
            },
            prefix="partition-free-v1-runtime-proposal-mask-lineage-",
        )
        binding_mapping = _raw_mapping(
            _expect_carrier(binding, "V1OptionACampaignBinding"),
            cohort=cohort,
        )
        campaign_contract_hash = _raw_content_hash(
            _expect_carrier(
                binding_mapping["contract"],
                "V1OptionACampaignContract",
            ),
            cohort=cohort,
            prefix="partition-free-v1-option-a-campaign-",
        )
        if (
            projection["phase"] != "stopped"
            or projection["acquisition_hash"] is not None
            or projection["active_candidate_hash"] is not None
            or horizon_witness["campaign_contract_hash"]
            != campaign_contract_hash
            or horizon_witness["campaign_binding_hash"]
            != projection["campaign_binding_hash"]
            or horizon_witness["controller_run_id"]
            != projection["controller_run_id"]
            or horizon_witness["generation_index"] != len(transition_hashes)
            or horizon_witness["current_mask_hash"]
            != projection["current_mask_hash"]
            or horizon_witness["route_axis_hash"]
            != current_mask["route_axis_hash"]
            or horizon_witness["mask_lineage_hash"] != lineage_hash
            or horizon_witness["post_stop_logical_sequence"]
            != projection["logical_sequence"]
            or horizon_proof["campaign_contract_hash"]
            != campaign_contract_hash
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller F/Q/R horizon witness names another live lineage",
                phase="historical_projection",
            )
        pre_stop_projection = dict(projection)
        pre_stop_projection.pop("fqr_no_candidate_horizon_witness_hash")
        pre_stop_projection.update(
            {
                "phase": "needs_acquisition",
                "stop_reason": None,
                "logical_sequence": projection["logical_sequence"] - 1,
            }
        )
        if (
            projection["logical_sequence"] <= 0
            or horizon_witness["pre_stop_logical_sequence"]
            != pre_stop_projection["logical_sequence"]
            or horizon_witness["pre_stop_controller_state_hash"]
            != content_hash(
                pre_stop_projection,
                prefix="partition-free-v1-controller-state-",
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller F/Q/R horizon witness does not replay its pre-stop state",
                phase="historical_projection",
            )
    controller_hash = content_hash(
        projection,
        prefix="partition-free-v1-controller-state-",
    )
    stored_hash = mapping.get("controller_state_hash") if is_compact else None
    if stored_hash is not None and stored_hash != controller_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "controller projection does not replay its stored state hash",
            phase="historical_projection",
        )
    summary = {
        "binding": binding,
        "controller_run_id": mapping["controller_run_id"],
        "phase": mapping["phase"],
        "current_mask_hash": projection["current_mask_hash"],
        "queue_cursor": mapping["queue_cursor"],
        "active_candidate": mapping["active_candidate"],
        "accepted_trajectory_hash": projection["accepted_trajectory_hash"],
        "acquisition_receipt_hash": (
            None
            if mapping["phase"] not in {"queue_ready", "candidate_active"}
            else (
                None
                if not mapping["acquisition_receipt_hashes"]
                else mapping["acquisition_receipt_hashes"][-1]
            )
        ),
        "acquisition_receipt_hashes": mapping["acquisition_receipt_hashes"],
        "attempt_receipt_hashes": attempt_hashes,
        "transition_receipt_hashes": transition_hashes,
        "globally_used_draw_ids": mapping["globally_used_draw_ids"],
        "stop_reason": mapping["stop_reason"],
        "logical_sequence": mapping["logical_sequence"],
        "transition_receipts": mapping["transition_receipts"],
    }
    if plan_history:
        summary.update(
            {
                "label_native_runtime_plan_hash": plan_hash,
                "label_native_runtime_plan_hashes": plan_history,
                "label_native_rejected_plan_indices": rejected_plan_indices,
                "label_native_selected_acquisition_receipt_hash": (
                    selected_receipt_hash
                ),
            }
        )
    if horizon_witness_hash is not None:
        summary["fqr_no_candidate_horizon_witness_hash"] = (
            horizon_witness_hash
        )
        assert horizon_witness is not None
        pre_stop_summary = dict(summary)
        pre_stop_summary.pop("fqr_no_candidate_horizon_witness_hash")
        pre_stop_summary.update(
            {
                "phase": "needs_acquisition",
                "stop_reason": None,
                "logical_sequence": summary["logical_sequence"] - 1,
            }
        )
        if horizon_witness["pre_stop_controller_snapshot_hash"] != (
            _controller_snapshot_hash(pre_stop_summary, cohort=cohort)
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller F/Q/R horizon witness does not replay its pre-stop snapshot",
                phase="historical_projection",
            )
    return projection, summary, is_compact


def _raw_removed_route_set_identity(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, object]:
    carrier = _expect_exact_carrier(
        value,
        module=_FQ_CANDIDATE_MODULE,
        name="V1RemovedRouteSetIdentity",
    )
    mapping = _raw_mapping(carrier, cohort=cohort)
    route_count = _raw_positive_integer(
        mapping["route_count"],
        "removed-route identity route_count",
    )
    count = _raw_positive_integer(
        mapping["count"],
        "removed-route identity count",
    )
    index_hash = _raw_nonempty_string(
        mapping["index_hash"],
        "removed-route identity index_hash",
    )
    if count > route_count:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "removed-route identity exceeds its physical route axis",
            phase="raw_state_validation",
        )
    return {
        "route_count": route_count,
        "count": count,
        "index_hash": index_hash,
    }


def _removed_count(value: _HistoricalCarrier, *, cohort: _CohortSpec) -> int:
    mapping = _raw_mapping(value, cohort=cohort)
    removed = mapping["removed_route_indices"]
    if type(removed) is tuple:
        return len(removed)
    if isinstance(removed, _HistoricalCarrier):
        return int(
            _raw_removed_route_set_identity(removed, cohort=cohort)["count"]
        )
    _refuse(
        V1CheckpointRefusalCode.RAW_STATE_INVALID,
        "transition receipt has no exact removed-count coordinate",
        phase="historical_projection",
    )
    raise AssertionError("unreachable")


def _raw_active_candidate_proposal(
    value: object,
    *,
    cohort: _CohortSpec,
) -> dict[str, Any]:
    if type(value) is dict:
        from .modern_v1_operator_checkpoint_wire import validate_candidate_definition
        try:
            return validate_candidate_definition(value)
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,str(error),phase='raw_state_validation')
    is_legacy_or_fq = bool(
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _FQ_CANDIDATE_MODULE
        and value.historical_name
        in {"V1CandidateProposal", "V1FQCandidateProposal"}
    )
    is_fqr = bool(
        isinstance(value, _HistoricalCarrier)
        and value.historical_module == _FQR_ACQUISITION_MODULE
        and value.historical_name == "V1FQRCandidateProposal"
    )
    if not is_legacy_or_fq and not is_fqr:
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            "active candidate proposal is outside the closed supported nominal union",
            phase="raw_root_validation",
        )
    if is_fqr:
        replay = _raw_fqr_candidate_proposal(value, cohort=cohort)
        return cast(dict[str, Any], replay["mapping"])
    assert isinstance(value, _HistoricalCarrier)
    if value.historical_name == "V1FQCandidateProposal":
        replay = _raw_fq_candidate_proposal(value, cohort=cohort)
        return cast(dict[str, Any], replay["mapping"])
    return _raw_mapping(value, cohort=cohort)


def _active_snapshot_fields(value: Any, *, cohort: _CohortSpec) -> tuple[Any, Any, Any]:
    if value is None:
        return None, None, None
    active = (_operator_active_mapping(value) if type(value) is dict else
              _raw_mapping(_expect_carrier(value, "V1ActiveCandidate"), cohort=cohort))
    proposal = _raw_active_candidate_proposal(
        active["proposal"],
        cohort=cohort,
    )
    _raw_mapping(
        _expect_carrier(active["step_freeze"], "V1StagedAlphaStepFreeze"),
        cohort=cohort,
    )
    paired_state = active.get("paired_bce_state")
    ledger_hash = (
        _paired_bce_rebuild_expected_hash(
            _expect_carrier(paired_state, "V1PairedBCEAcquisitionState"),
            expected_name="V1PairedBCEAcquisitionState",
        )
        if paired_state is not None
        else _raw_content_hash(
            _expect_carrier(active["ledger"], "V1StagedAlphaLedger"),
            cohort=cohort,
            prefix="partition-free-v1-staged-alpha-ledger-",
        )
    )
    return (
        proposal["candidate_id"],
        content_hash(
            _raw_canonical(active["step_freeze"], cohort=cohort),
            prefix="partition-free-v1-staged-alpha-step-freeze-",
        ),
        ledger_hash,
    )


def _controller_snapshot_hash(
    summary: Mapping[str, Any],
    *,
    cohort: _CohortSpec,
) -> str:
    binding = _raw_mapping(
        _expect_carrier(summary["binding"], "V1OptionACampaignBinding"),
        cohort=cohort,
    )
    contract = _raw_mapping(
        _expect_carrier(binding["contract"], "V1OptionACampaignContract"),
        cohort=cohort,
    )
    active_id, active_step, active_ledger = _active_snapshot_fields(
        summary["active_candidate"], cohort=cohort
    )
    transitions = summary["transition_receipts"]
    horizon_witness_hash = summary.get(
        "fqr_no_candidate_horizon_witness_hash"
    )
    label_native_plan_history = tuple(
        summary.get("label_native_runtime_plan_hashes", ())
    )
    snapshot = {
        "campaign_contract_hash": content_hash(
            _raw_canonical(binding["contract"], cohort=cohort),
            prefix="partition-free-v1-option-a-campaign-",
        ),
        "campaign_binding_hash": _binding_hash(summary["binding"], cohort=cohort),
        "controller_run_id": summary["controller_run_id"],
        "phase": summary["phase"],
        "current_mask_hash": summary["current_mask_hash"],
        "acquisition_generation_index": len(transitions),
        "queue_cursor": summary["queue_cursor"],
        "accepted_step_count": len(transitions),
        "candidate_attempt_count": len(summary["attempt_receipt_hashes"]),
        "cumulative_removed_count": sum(
            _removed_count(
                _expect_carrier(item, "V1MaskTransitionReceipt"), cohort=cohort
            )
            for item in transitions
        ),
        "active_candidate_id": active_id,
        "active_step_freeze_hash": active_step,
        "active_ledger_hash": active_ledger,
        "accepted_trajectory_hash": summary["accepted_trajectory_hash"],
        "acquisition_receipt_hash": summary["acquisition_receipt_hash"],
        "acquisition_receipt_hashes": summary["acquisition_receipt_hashes"],
        "attempt_receipt_hashes": summary["attempt_receipt_hashes"],
        "transition_receipt_hashes": summary["transition_receipt_hashes"],
        "globally_used_draw_ids_hash": content_hash(
            summary["globally_used_draw_ids"],
            prefix="partition-free-v1-globally-used-draw-ids-",
        ),
        "stop_reason": summary["stop_reason"],
        "logical_sequence": summary["logical_sequence"],
        "claim_execution_certified": False,
        "mask_commit_authority": False,
        "schema": (
            (
                "partition_free_v1_controller_snapshot_receipt_v4"
                if horizon_witness_hash is None
                else "partition_free_v1_controller_snapshot_receipt_v5"
            )
            if label_native_plan_history
            else (
                "partition_free_v1_controller_snapshot_receipt_v2"
                if horizon_witness_hash is None
                else "partition_free_v1_controller_snapshot_receipt_v3"
            )
        ),
    }
    if horizon_witness_hash is not None:
        snapshot["fqr_no_candidate_horizon_witness_hash"] = (
            horizon_witness_hash
        )
    if label_native_plan_history:
        snapshot.update(
            {
                "label_native_runtime_plan_hash": summary.get(
                    "label_native_runtime_plan_hash"
                ),
                "label_native_runtime_plan_hashes": (
                    label_native_plan_history
                ),
                "label_native_rejected_plan_indices": tuple(
                    summary.get("label_native_rejected_plan_indices", ())
                ),
                "label_native_selected_acquisition_receipt_hash": summary.get(
                    "label_native_selected_acquisition_receipt_hash"
                ),
            }
        )
    # Touch the contract mapping so malformed cohort maps cannot hide behind a
    # precomputed campaign hash stored elsewhere in the graph.
    if contract.get("schema") not in {
        _PRE16.contract_schema,
        _V16_TO_V28.contract_schema,
        _V29_PLUS.contract_schema,
    }:
        _refuse(
            V1CheckpointRefusalCode.COHORT_INCONSISTENT,
            "controller snapshot contract schema is unsupported",
            phase="historical_projection",
        )
    return content_hash(snapshot, prefix="partition-free-v1-controller-snapshot-")


def _raw_controller_snapshot_receipt_hash(
    value: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
) -> str:
    """Replay the receipt's conditional v2--v5 snapshot hash projection."""

    mapping = _raw_mapping(value, cohort=cohort)
    payload = dict(mapping)
    witness_hash = payload.get("fqr_no_candidate_horizon_witness_hash")
    if witness_hash is None:
        payload.pop("fqr_no_candidate_horizon_witness_hash", None)
    else:
        _raw_nonempty_string(
            witness_hash,
            "controller snapshot F/Q/R horizon witness hash",
        )

    label_native_field_names = (
        "label_native_runtime_plan_hash",
        "label_native_runtime_plan_hashes",
        "label_native_rejected_plan_indices",
        "label_native_selected_acquisition_receipt_hash",
    )
    plan_hash = payload.get("label_native_runtime_plan_hash")
    raw_plan_history = payload.get("label_native_runtime_plan_hashes", ())
    raw_rejected_indices = payload.get("label_native_rejected_plan_indices", ())
    selected_receipt_hash = payload.get(
        "label_native_selected_acquisition_receipt_hash"
    )
    plan_history = _raw_tuple(
        raw_plan_history,
        "controller snapshot label-native plan history",
    )
    rejected_indices = _raw_tuple(
        raw_rejected_indices,
        "controller snapshot label-native rejected indices",
    )
    has_label_native = bool(
        plan_hash is not None
        or plan_history
        or rejected_indices
        or selected_receipt_hash is not None
    )

    if has_label_native:
        if any(name not in payload for name in label_native_field_names):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "controller snapshot has partial label-native plan state",
                phase="historical_projection",
            )
        normalized_plan_history = tuple(
            _raw_nonempty_string(
                item,
                "controller snapshot label-native plan history entry",
            )
            for item in plan_history
        )
        if (
            not normalized_plan_history
            or len(set(normalized_plan_history)) != len(normalized_plan_history)
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "controller snapshot label-native plan history is empty or reused",
                phase="historical_projection",
            )
        normalized_rejected_indices = tuple(
            _raw_nonnegative_integer(
                item,
                "controller snapshot label-native rejected plan index",
            )
            for item in rejected_indices
        )
        if plan_hash is None:
            if normalized_rejected_indices or selected_receipt_hash is not None:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller snapshot retained live label-native coordinates without a plan",
                    phase="historical_projection",
                )
        else:
            normalized_plan_hash = _raw_nonempty_string(
                plan_hash,
                "controller snapshot current label-native plan hash",
            )
            queue_cursor = _raw_nonnegative_integer(
                payload.get("queue_cursor"),
                "controller snapshot queue cursor",
            )
            if (
                normalized_plan_history[-1] != normalized_plan_hash
                or normalized_rejected_indices != tuple(range(queue_cursor))
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller snapshot label-native cursor/rejection ledger is inconsistent",
                    phase="historical_projection",
                )
            queue_bearing = payload.get("phase") in {
                "queue_ready",
                "candidate_active",
            }
            if queue_bearing:
                normalized_selected_receipt_hash = _raw_nonempty_string(
                    selected_receipt_hash,
                    "controller snapshot selected label-native acquisition receipt hash",
                )
                if (
                    normalized_selected_receipt_hash
                    != payload.get("acquisition_receipt_hash")
                ):
                    _refuse(
                        V1CheckpointRefusalCode.RAW_STATE_INVALID,
                        "controller snapshot selected label-native receipt is stale",
                        phase="historical_projection",
                    )
            elif selected_receipt_hash is not None:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller snapshot nonqueue phase retained a selected endpoint",
                    phase="historical_projection",
                )
        if selected_receipt_hash is not None:
            normalized_selected_receipt_hash = _raw_nonempty_string(
                selected_receipt_hash,
                "controller snapshot selected label-native acquisition receipt hash",
            )
            acquisition_history = _raw_tuple(
                payload.get("acquisition_receipt_hashes"),
                "controller snapshot acquisition receipt history",
            )
            if normalized_selected_receipt_hash not in acquisition_history:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "controller snapshot selected endpoint is absent from acquisition history",
                    phase="historical_projection",
                )
        expected_schema = (
            "partition_free_v1_controller_snapshot_receipt_v5"
            if witness_hash is not None
            else "partition_free_v1_controller_snapshot_receipt_v4"
        )
    else:
        for name in label_native_field_names:
            payload.pop(name, None)
        expected_schema = (
            "partition_free_v1_controller_snapshot_receipt_v3"
            if witness_hash is not None
            else "partition_free_v1_controller_snapshot_receipt_v2"
        )

    if (
        payload.get("schema") != expected_schema
        or (witness_hash is not None)
        != (payload.get("stop_reason") == _FQR_HORIZON_STOP_REASON)
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "controller snapshot additive schema projection is inconsistent",
            phase="historical_projection",
        )
    return content_hash(
        payload,
        prefix="partition-free-v1-controller-snapshot-",
    )


def _runtime_projection(
    root: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
    header: Mapping[str, Any],
    proposal_state_hashes: tuple[str, str | None] | None = None,
) -> tuple[str, str, str, str | None]:
    mapping = _raw_mapping(root, cohort=cohort)
    if root.historical_name == "_V1RuntimeCheckpointJournal":
        runtime_schema = mapping["runtime_schema"]
        embedded_runtime_hash = mapping["runtime_state_hash"]
    elif root.historical_name == "V1RuntimeState":
        runtime_schema = mapping["schema"]
        embedded_runtime_hash = None
    else:
        _refuse(
            V1CheckpointRefusalCode.RAW_ROOT_INVALID,
            "checkpoint payload is neither a historical runtime state nor journal",
            phase="raw_root_validation",
        )
    if runtime_schema != _RUNTIME_STATE_SCHEMA:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "historical runtime schema is unsupported",
            phase="historical_projection",
        )
    controller = mapping["controller"]
    if not isinstance(controller, _HistoricalCarrier):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "runtime controller is not an inert historical carrier",
            phase="historical_projection",
        )
    controller_projection, controller_summary, compact = _controller_projection(
        controller,
        cohort=cohort,
    )
    from .modern_v1_operator_run_binding import validate_history_binding
    config_mapping=_raw_mapping(mapping['config'],cohort=cohort)
    try:
        validate_history_binding(_raw_canonical(config_mapping['metadata'],cohort=cohort),
            config_mapping['source_seal_hash'],controller_projection['acquisition_receipt_hashes'])
    except (ValueError,TypeError,KeyError) as error:
        _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,str(error),phase='raw_state_validation')
    controller_hash = content_hash(
        controller_projection,
        prefix="partition-free-v1-controller-state-",
    )
    pending_hash = _carrier_hash(
        mapping["pending_look"],
        cohort=cohort,
        prefix="partition-free-v1-runtime-pending-look-",
    )
    projection = {
        "schema": runtime_schema,
        "logical_config_hash": _logical_config_hash(mapping["config"], cohort=cohort),
        "entropy_tape_hash": _carrier_hash(
            mapping["entropy_tape"],
            cohort=cohort,
            prefix="partition-free-v1-staged-alpha-entropy-tape-",
        ),
        "controller_state_hash": controller_hash,
        "pending_hash": pending_hash,
        "terminal_freeze_hash": _carrier_hash(
            mapping["terminal_freeze"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-freeze-",
        ),
        "terminal_validation_hash": _carrier_hash(
            mapping["terminal_validation"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-",
        ),
        "completion_hash": _carrier_hash(
            mapping["completion"],
            cohort=cohort,
            prefix="partition-free-v1-run-completion-",
        ),
        "performance_hash": _carrier_hash(
            mapping["performance"],
            cohort=cohort,
            prefix="partition-free-v1-performance-",
        ),
    }
    completed_decision_hashes: tuple[str, ...] | None = None
    if (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "completed_proposal_freeze_decisions" in mapping
    ):
        raw_decisions = mapping["completed_proposal_freeze_decisions"]
        if type(raw_decisions) not in {tuple, list}:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal freeze ledger is not positional",
                phase="historical_projection",
            )
        completed_decision_hashes = tuple(
            cast(
                str,
                _carrier_hash(
                    _expect_exact_carrier(
                        item,
                        module=(
                            "mechanistic_systems.circuits.irg."
                            "modern_v1_joint_acquisition"
                        ),
                        name="V1ProposalFreezeDecisionReceipt",
                    ),
                    cohort=cohort,
                    prefix="partition-free-v1-proposal-freeze-decision-",
                ),
            )
            for item in raw_decisions
        )
        if len(set(completed_decision_hashes)) != len(completed_decision_hashes):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal freeze ledger contains duplicates",
                phase="historical_projection",
            )
    completed_closure_hashes: tuple[str, ...] | None = None
    if (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "completed_proposal_closures" in mapping
    ):
        raw_closures = mapping["completed_proposal_closures"]
        if type(raw_closures) not in {tuple, list}:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure ledger is not positional",
                phase="historical_projection",
            )
        completed_closure_hashes = tuple(
            cast(
                str,
                _carrier_hash(
                    _expect_exact_carrier(
                        item,
                        module="mechanistic_systems.circuits.irg.modern_v1_runtime",
                        name="V1RuntimeCompletedProposalClosureReceipt",
                    ),
                    cohort=cohort,
                    prefix="partition-free-v1-runtime-completed-proposal-closure-",
                ),
            )
            for item in raw_closures
        )
        if len(set(completed_closure_hashes)) != len(completed_closure_hashes):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure ledger contains duplicates",
                phase="historical_projection",
            )
    if proposal_state_hashes is not None:
        projection.update(
            {
                "proposal_progress_hash": proposal_state_hashes[0],
                "pending_proposal_acquisition_hash": proposal_state_hashes[1],
            }
        )
    if completed_decision_hashes is not None:
        projection.setdefault("proposal_progress_hash", None)
        projection.setdefault("pending_proposal_acquisition_hash", None)
        projection["completed_proposal_freeze_decision_hashes"] = (
            completed_decision_hashes
        )
    if completed_closure_hashes is not None and (
        proposal_state_hashes is not None or bool(completed_closure_hashes)
    ):
        projection.setdefault("proposal_progress_hash", None)
        projection.setdefault("pending_proposal_acquisition_hash", None)
        projection["completed_proposal_closure_hashes"] = completed_closure_hashes
        projection["staged_proposal_backend_identity_hash"] = mapping[
            "staged_proposal_backend_identity_hash"
        ]
    elif (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "staged_proposal_backend_identity_hash" in mapping
        and mapping["staged_proposal_backend_identity_hash"] is not None
    ):
        projection["staged_proposal_backend_identity_hash"] = mapping[
            "staged_proposal_backend_identity_hash"
        ]
    pending_paired_hash = None
    if (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "pending_paired_bce_chunk" in mapping
    ):
        pending_paired_hash = _carrier_hash(
            mapping["pending_paired_bce_chunk"],
            cohort=cohort,
            prefix="partition-free-v1-runtime-pending-paired-bce-chunk-",
        )
        if pending_paired_hash is not None:
            projection["pending_paired_bce_chunk_hash"] = pending_paired_hash
    if (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "terminal_validation_family_freeze" in mapping
        and mapping["terminal_validation_family_freeze"] is not None
    ):
        family_freeze_hash = _carrier_hash(
            mapping["terminal_validation_family_freeze"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-family-freeze-",
        )
        family_request_hash = _carrier_hash(
            mapping["terminal_validation_family_request"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-family-request-",
        )
        family_observation_hash = _carrier_hash(
            mapping["terminal_validation_family_observation"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-family-outcome-",
        )
        family_hash = _carrier_hash(
            mapping["terminal_validation_family"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-family-",
        )
        if family_freeze_hash is None:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "checkpoint journal-v6 lacks terminal-family freeze state",
                phase="historical_projection",
            )
        projection.update(
            {
                "terminal_validation_family_freeze_hash": family_freeze_hash,
                "terminal_validation_family_request_hash": family_request_hash,
                "terminal_validation_family_observation_hash": family_observation_hash,
                "terminal_validation_family_hash": family_hash,
            }
        )
    if (
        root.historical_name == "_V1RuntimeCheckpointJournal"
        and "terminal_comparison_family_freeze" in mapping
    ):
        comparison_freeze_hash = _carrier_hash(
            mapping["terminal_comparison_family_freeze"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-freeze-v2-",
        )
        comparison_request_hash = _carrier_hash(
            mapping["terminal_comparison_family_request"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-request-v2-",
        )
        comparison_observation_hash = _carrier_hash(
            mapping["terminal_comparison_family_observation"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-outcome-v2-",
        )
        comparison_hash = _carrier_hash(
            mapping["terminal_comparison_family"],
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-v2-",
        )
        if comparison_freeze_hash is None:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "checkpoint journal-v7 lacks comparison-family freeze state",
                phase="historical_projection",
            )
        projection.update(
            {
                "terminal_comparison_family_freeze_hash": comparison_freeze_hash,
                "terminal_comparison_family_request_hash": comparison_request_hash,
                "terminal_comparison_family_observation_hash": comparison_observation_hash,
                "terminal_comparison_family_hash": comparison_hash,
            }
        )
    runtime_hash = content_hash(
        projection,
        prefix="partition-free-v1-runtime-state-",
    )
    if embedded_runtime_hash is not None and embedded_runtime_hash != runtime_hash:
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "runtime projection does not replay its embedded state hash",
            phase="historical_projection",
        )
    snapshot_hash = _controller_snapshot_hash(controller_summary, cohort=cohort)
    if compact:
        controller_mapping = _raw_mapping(controller, cohort=cohort)
        if controller_mapping["controller_snapshot_hash"] != snapshot_hash:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller snapshot does not replay its journal identity",
                phase="historical_projection",
            )
    if (
        header["runtime_state_hash"] != runtime_hash
        or header["controller_snapshot_hash"] != snapshot_hash
        or header["pending_look_hash"] != pending_hash
        or header.get("pending_paired_bce_chunk_hash") != pending_paired_hash
    ):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "strict header differs from the replayed historical projection",
            phase="historical_projection",
        )
    return runtime_schema, runtime_hash, controller_hash, pending_hash


def _validate_completed_proposal_closures(
    values: object,
    *,
    cohort: _CohortSpec,
    controller: _HistoricalCarrier,
) -> tuple[tuple[str, ...], str | None]:
    if type(values) not in {tuple, list}:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "completed proposal closure ledger is not positional",
            phase="raw_state_validation",
        )
    controller_projection, summary, _compact = _controller_projection(
        controller,
        cohort=cohort,
    )
    controller_mapping = _raw_mapping(controller, cohort=cohort)
    binding_mapping = _raw_mapping(
        _expect_carrier(summary["binding"], "V1OptionACampaignBinding"),
        cohort=cohort,
    )
    campaign_contract_hash = content_hash(
        _raw_canonical(binding_mapping["contract"], cohort=cohort),
        prefix="partition-free-v1-option-a-campaign-",
    )
    campaign_binding_hash = controller_projection["campaign_binding_hash"]
    controller_run_id = controller_projection["controller_run_id"]
    current_mask_mapping = _raw_mapping(
        _expect_carrier(controller_mapping["current_mask"], "HardMask"),
        cohort=cohort,
    )
    route_axis_hash = current_mask_mapping["route_axis_hash"]
    transition_carriers = tuple(summary["transition_receipts"])
    transition_hashes = tuple(summary["transition_receipt_hashes"])
    lineage = [controller_projection["initial_mask_hash"]]
    expected_parent = lineage[0]
    for carrier in transition_carriers:
        transition = _raw_mapping(
            _expect_carrier(carrier, "V1MaskTransitionReceipt"),
            cohort=cohort,
        )
        if transition["parent_mask_hash"] != expected_parent:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "controller mask transition lineage is discontinuous",
                phase="historical_projection",
            )
        expected_parent = _raw_nonempty_string(
            transition["child_mask_hash"],
            "transition child_mask_hash",
        )
        lineage.append(expected_parent)
    acquisition_history = tuple(summary["acquisition_receipt_hashes"])
    all_attempt_carriers = tuple(controller_mapping["attempt_receipts"])
    all_attempt_hashes = tuple(summary["attempt_receipt_hashes"])
    closure_hashes: list[str] = []
    backend_identity: str | None = None
    prior_alpha_after_hash: str | None = None
    for generation, value in enumerate(cast(Sequence[object], values)):
        closure = _expect_exact_carrier(
            value,
            module="mechanistic_systems.circuits.irg.modern_v1_runtime",
            name="V1RuntimeCompletedProposalClosureReceipt",
        )
        mapping = _raw_mapping(closure, cohort=cohort)
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
            _raw_nonempty_string(mapping[name], name)
        alpha_names = (
            "proposal_order_alpha_schedule_hash",
            "proposal_order_alpha_ledger_before_hash",
            "proposal_order_design_hash",
            "proposal_order_alpha_ledger_after_hash",
            "proposal_order_alpha_ledger_after_replay",
            "proposal_order_envelope_certificate_hash",
            "proposal_order_action_family_registry_hash",
        )
        alpha_values = tuple(mapping.get(name) for name in alpha_names)
        alpha_bound = all(value is not None for value in alpha_values)
        base_design_hash = mapping.get(
            "proposal_order_base_serfling_design_hash"
        )
        alpha_family_names = (
            "sampled_v7_D_alpha_family_plan_hash",
            "sampled_v7_D_alpha_family_registry_receipt_hash",
            "sampled_v7_D_alpha_family_registry_replay",
            "sampled_v7_D_alpha_head_CAS_receipt_hash",
            "sampled_v7_D_alpha_published_ledger_head_hash",
            "sampled_v7_D_alpha_durable_controller_head_closed",
        )
        alpha_family_values = tuple(mapping.get(name) for name in alpha_family_names)
        alpha_family_bound = all(
            value is not None for value in alpha_family_values
        )
        if any(value is not None for value in alpha_values) and not alpha_bound:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure alpha coordinates are partial",
                phase="raw_state_validation",
            )
        if (
            any(value is not None for value in alpha_family_values)
            and not alpha_family_bound
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure A/F/J alpha coordinates are partial",
                phase="raw_state_validation",
            )
        if alpha_family_bound and (not alpha_bound or base_design_hash is None):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure A/F/J alpha family lacks its base",
                phase="raw_state_validation",
            )
        expected_closure_schema = (
            "partition_free_v1_runtime_completed_proposal_closure_receipt_v4"
            if alpha_family_bound
            else "partition_free_v1_runtime_completed_proposal_closure_receipt_v3"
            if alpha_bound and base_design_hash is not None
            else "partition_free_v1_runtime_completed_proposal_closure_receipt_v2"
            if alpha_bound
            else "partition_free_v1_runtime_completed_proposal_closure_receipt_v1"
        )
        if mapping["schema"] != expected_closure_schema:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure schema is unsupported",
                phase="raw_state_validation",
            )
        if alpha_bound:
            for name in alpha_names:
                if name != "proposal_order_alpha_ledger_after_replay":
                    _raw_nonempty_string(mapping[name], name)
            replay = mapping["proposal_order_alpha_ledger_after_replay"]
            if base_design_hash is not None:
                _raw_nonempty_string(
                    base_design_hash,
                    "proposal_order_base_serfling_design_hash",
                )
            if not isinstance(replay, Mapping) or set(replay) != {
                "schema",
                "schedule_payload",
                "schedule_hash",
                "ledger_payload",
                "ledger_hash",
            }:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal closure alpha replay envelope is invalid",
                    phase="raw_state_validation",
                )
            replay = cast(Mapping[str, Any], replay)
            schedule_payload = replay["schedule_payload"]
            ledger_payload = replay["ledger_payload"]
            if (
                replay["schema"]
                != "partition_free_v1_proposal_order_trajectory_alpha_replay_v1"
                or not isinstance(schedule_payload, Mapping)
                or not isinstance(ledger_payload, Mapping)
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal closure alpha replay payload is invalid",
                    phase="raw_state_validation",
                )
            schedule_payload = cast(Mapping[str, Any], schedule_payload)
            ledger_payload = cast(Mapping[str, Any], ledger_payload)
            schedule_hash = content_hash(
                _raw_canonical(schedule_payload, cohort=cohort),
                prefix="partition-free-v1-proposal-order-alpha-schedule-",
            )
            ledger_hash = content_hash(
                {
                    "schedule_hash": schedule_hash,
                    **_raw_canonical(ledger_payload, cohort=cohort),
                },
                prefix="partition-free-v1-proposal-order-alpha-ledger-",
            )
            registered_designs = tuple(
                ledger_payload.get("registered_design_hashes", ())
            )
            before_hash = mapping["proposal_order_alpha_ledger_before_hash"]
            if generation == 0:
                genesis_hash = content_hash(
                    {
                        "schema": (
                            "partition_free_v1_proposal_order_trajectory_alpha_ledger_v1"
                        ),
                        "schedule_hash": schedule_hash,
                        "proposal_generation": 0,
                        "registered_design_hashes": (),
                        "committed_generation_allocations": (),
                        "prior_state_hash": None,
                        "cumulative_committed_alpha": 0.0,
                        "contains_outcomes": False,
                        "f_q_authority": False,
                        "bce_oracle_authority": False,
                        "mask_commit_authority": False,
                        "alpha_policy_kind": (
                            "fixed_nonrecycling_trajectory_bonferroni_fwer_v1"
                        ),
                        "authority_scope": (
                            "trajectory_fwer_budget_for_sampled_corrected_"
                            "attribution_order_only_v1"
                        ),
                    },
                    prefix="partition-free-v1-proposal-order-alpha-ledger-",
                )
                continuity_ok = before_hash == genesis_hash
            else:
                continuity_ok = before_hash == prior_alpha_after_hash
            if (
                schedule_hash != replay["schedule_hash"]
                or schedule_hash != mapping["proposal_order_alpha_schedule_hash"]
                or ledger_hash != replay["ledger_hash"]
                or ledger_hash != mapping["proposal_order_alpha_ledger_after_hash"]
                or schedule_payload.get("campaign_contract_hash")
                != campaign_contract_hash
                or schedule_payload.get("controller_run_hash") != controller_run_id
                or ledger_payload.get("proposal_generation") != generation + 1
                or ledger_payload.get("prior_state_hash") != before_hash
                or not registered_designs
                or registered_designs[-1]
                != (
                    base_design_hash
                    if base_design_hash is not None
                    else mapping["proposal_order_design_hash"]
                )
                or not continuity_ok
            ):
                _refuse(
                    V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                    "completed proposal closure alpha replay forked or changed",
                    phase="historical_projection",
                )
            prior_alpha_after_hash = cast(
                str, mapping["proposal_order_alpha_ledger_after_hash"]
            )
        if alpha_family_bound:
            for name in alpha_family_names:
                if name not in {
                    "sampled_v7_D_alpha_family_registry_replay",
                    "sampled_v7_D_alpha_durable_controller_head_closed",
                }:
                    _raw_nonempty_string(mapping[name], name)
            if mapping["sampled_v7_D_alpha_durable_controller_head_closed"] is not True:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal closure A/F/J durable head is not closed",
                    phase="raw_state_validation",
                )
            registry_replay = mapping[
                "sampled_v7_D_alpha_family_registry_replay"
            ]
            if not isinstance(registry_replay, Mapping) or set(registry_replay) != {
                "schema",
                "receipt_payload",
                "receipt_hash",
            }:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal closure A/F/J registry replay is invalid",
                    phase="raw_state_validation",
                )
            registry_replay = cast(Mapping[str, Any], registry_replay)
            registry_payload = registry_replay["receipt_payload"]
            if (
                registry_replay["schema"]
                != "partition_free_v1_sampled_v7_D_alpha_family_registry_replay_v2"
                or not isinstance(registry_payload, Mapping)
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal closure A/F/J registry schema is invalid",
                    phase="raw_state_validation",
                )
            registry_payload = cast(Mapping[str, Any], registry_payload)
            registry_hash = content_hash(
                _raw_canonical(registry_payload, cohort=cohort),
                prefix="partition-free-v1-sampled-v7-D-alpha-registry-",
            )
            if (
                registry_hash != registry_replay["receipt_hash"]
                or registry_hash
                != mapping["sampled_v7_D_alpha_family_registry_receipt_hash"]
                or registry_payload.get("alpha_family_plan_hash")
                != mapping["sampled_v7_D_alpha_family_plan_hash"]
                or registry_payload.get("campaign_contract_hash")
                != campaign_contract_hash
                or registry_payload.get("controller_run_hash") != controller_run_id
                or registry_payload.get("proposal_generation") != generation
                or registry_payload.get("ledger_before_hash")
                != mapping["proposal_order_alpha_ledger_before_hash"]
                or registry_payload.get("ledger_after_hash")
                != mapping["proposal_order_alpha_ledger_after_hash"]
                or registry_payload.get("charged_base_design_hash")
                != base_design_hash
                or registry_payload.get("hybrid_wrapper_design_hash")
                != mapping["proposal_order_design_hash"]
                or mapping["sampled_v7_D_alpha_published_ledger_head_hash"]
                != mapping["proposal_order_alpha_ledger_after_hash"]
            ):
                _refuse(
                    V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                    "completed proposal closure A/F/J registry replay forked",
                    phase="historical_projection",
                )
        elif prior_alpha_after_hash is not None:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closures mix alpha-bound and legacy chains",
                phase="historical_projection",
            )
        if (
            mapping["validation_outcomes_observed"] is not False
            or mapping["train_optimization_data_only"] is not True
            or mapping["mask_commit_authority"] is not False
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure crossed its authority firewall",
                phase="raw_state_validation",
            )
        observed_generation = _raw_nonnegative_integer(
            mapping["acquisition_generation_index"],
            "closure acquisition_generation_index",
        )
        proposal_logical_sequence = _raw_nonnegative_integer(
            mapping["proposal_logical_sequence"],
            "closure proposal_logical_sequence",
        )
        prefix_lineage = tuple(lineage[: generation + 1])
        prefix_transitions = transition_hashes[:generation]
        expected_lineage_hash = content_hash(
            {
                "schema": "partition_free_v1_runtime_proposal_mask_lineage_v1",
                "mask_lineage_hashes": prefix_lineage,
                "transition_receipt_hashes": prefix_transitions,
            },
            prefix="partition-free-v1-runtime-proposal-mask-lineage-",
        )
        raw_candidate_ids = mapping["ordered_candidate_ids"]
        if type(raw_candidate_ids) not in {tuple, list}:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure candidate IDs are not positional",
                phase="raw_state_validation",
            )
        ordered_candidate_ids = tuple(
            _raw_nonempty_string(value, "closure ordered_candidate_id")
            for value in cast(Sequence[object], raw_candidate_ids)
        )
        if len(set(ordered_candidate_ids)) != len(ordered_candidate_ids):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal closure candidate IDs are not unique",
                phase="raw_state_validation",
            )
        expected_candidate_identity = content_hash(
            {
                "schema": "partition_free_v1_runtime_authorized_candidate_queue_v1",
                "campaign_binding_hash": mapping["campaign_binding_hash"],
                "generation_index": generation,
                "current_mask_hash": mapping["current_mask_hash"],
                "committed_mask_lineage_hashes": prefix_lineage,
                "committed_mask_transition_receipt_hashes": prefix_transitions,
                "acquisition_hash": mapping["acquisition_hash"],
                "acquisition_receipt_hash": mapping["acquisition_receipt_hash"],
                "ordered_candidate_ids": ordered_candidate_ids,
            },
            prefix="partition-free-v1-runtime-authorized-candidate-queue-",
        )
        if (
            observed_generation != generation
            or generation >= len(lineage)
            or generation >= len(acquisition_history)
            or mapping["campaign_contract_hash"] != campaign_contract_hash
            or mapping["campaign_binding_hash"] != campaign_binding_hash
            or mapping["controller_run_id"] != controller_run_id
            or mapping["current_mask_hash"] != lineage[generation]
            or mapping["route_axis_hash"] != route_axis_hash
            or mapping["mask_lineage_hash"] != expected_lineage_hash
            or mapping["candidate_identity_hash"] != expected_candidate_identity
            or mapping["acquisition_receipt_hash"]
            != acquisition_history[generation]
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure has stale positional lineage",
                phase="historical_projection",
            )
        observed_backend = cast(str, mapping["proposal_backend_identity_hash"])
        if backend_identity is None:
            backend_identity = observed_backend
        elif backend_identity != observed_backend:
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closures mix backend identities",
                phase="historical_projection",
            )
        decision = _expect_exact_carrier(
            mapping["freeze_decision"],
            module="mechanistic_systems.circuits.irg.modern_v1_joint_acquisition",
            name="V1ProposalFreezeDecisionReceipt",
        )
        decision_mapping = _raw_mapping(decision, cohort=cohort)
        if (
            decision_mapping["action"] != "authorize_candidate_freeze"
            or decision_mapping["candidate_identity_hash"]
            != mapping["candidate_identity_hash"]
            or decision_mapping["verified_receipt_graph_hash"]
            != mapping["verified_receipt_graph_hash"]
            or decision_mapping["state_hash"]
            != mapping["proposal_joint_state_hash"]
            or decision_mapping["phase_evidence_state_hash"]
            != mapping["proposal_phase_evidence_state_hash"]
            or decision_mapping["sufficient_state_hash"]
            != mapping["proposal_sufficient_state_hash"]
            or decision_mapping["logical_sequence"]
            != proposal_logical_sequence
            or decision_mapping["freeze_criterion_hash"]
            != mapping["freeze_criterion_hash"]
            or decision_mapping["evidence_hash"]
            != mapping["freeze_evidence_hash"]
            or decision_mapping["evidence_evaluation_hash"]
            != mapping["freeze_evidence_evaluation_hash"]
            or decision_mapping["criterion_verification_receipt_hash"]
            != mapping["criterion_verification_receipt_hash"]
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure has stale freeze authorization",
                phase="historical_projection",
            )
        pre_carrier = _expect_exact_carrier(
            mapping["pre_attach_controller_snapshot"],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1ControllerSnapshotReceipt",
        )
        pre = _raw_mapping(pre_carrier, cohort=cohort)
        post_carrier = _expect_exact_carrier(
            mapping["post_attach_controller_snapshot"],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1ControllerSnapshotReceipt",
        )
        post = _raw_mapping(post_carrier, cohort=cohort)
        for snapshot_carrier, snapshot in (
            (pre_carrier, pre),
            (post_carrier, post),
        ):
            label_native_snapshot = bool(
                tuple(snapshot.get("label_native_runtime_plan_hashes", ()))
            )
            if (
                snapshot["campaign_contract_hash"] != campaign_contract_hash
                or snapshot["campaign_binding_hash"] != campaign_binding_hash
                or snapshot["controller_run_id"] != controller_run_id
                or snapshot["current_mask_hash"] != lineage[generation]
                or snapshot["acquisition_generation_index"] != generation
                or snapshot["accepted_step_count"] != generation
                or snapshot["claim_execution_certified"] is not False
                or snapshot["mask_commit_authority"] is not False
                or snapshot["schema"]
                != (
                    "partition_free_v1_controller_snapshot_receipt_v4"
                    if label_native_snapshot
                    else "partition_free_v1_controller_snapshot_receipt_v2"
                )
            ):
                _refuse(
                    V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                    "completed proposal closure snapshot has stale run lineage",
                    phase="historical_projection",
                )
            _raw_controller_snapshot_receipt_hash(
                snapshot_carrier,
                cohort=cohort,
            )
        pre_plan_history = tuple(
            pre.get("label_native_runtime_plan_hashes", ())
        )
        post_plan_history = tuple(
            post.get("label_native_runtime_plan_hashes", ())
        )
        if (
            post_plan_history != pre_plan_history
            or post.get("label_native_runtime_plan_hash")
            != pre.get("label_native_runtime_plan_hash")
            or tuple(post.get("label_native_rejected_plan_indices", ()))
            != tuple(pre.get("label_native_rejected_plan_indices", ()))
            or pre.get("label_native_selected_acquisition_receipt_hash")
            is not None
            or post.get("label_native_selected_acquisition_receipt_hash")
            != (
                mapping["acquisition_receipt_hash"]
                if post.get("label_native_runtime_plan_hash") is not None
                and post["phase"] == "queue_ready"
                else None
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure changed its label-native plan frame",
                phase="historical_projection",
            )
        if (
            pre["phase"] != "needs_acquisition"
            or pre["queue_cursor"] != 0
            or tuple(pre["acquisition_receipt_hashes"])
            != acquisition_history[:generation]
            or tuple(pre["transition_receipt_hashes"])
            != transition_hashes[:generation]
            or post["phase"] not in {"queue_ready", "stopped"}
            or post["queue_cursor"] != 0
            or post["logical_sequence"] != pre["logical_sequence"] + 1
            or tuple(post["acquisition_receipt_hashes"])
            != acquisition_history[: generation + 1]
            or tuple(post["transition_receipt_hashes"])
            != transition_hashes[:generation]
            or tuple(post["attempt_receipt_hashes"])
            != tuple(pre["attempt_receipt_hashes"])
            or post["accepted_trajectory_hash"] != pre["accepted_trajectory_hash"]
            or post["candidate_attempt_count"] != pre["candidate_attempt_count"]
            or post["cumulative_removed_count"] != pre["cumulative_removed_count"]
            or post["globally_used_draw_ids_hash"]
            != pre["globally_used_draw_ids_hash"]
            or (
                post["phase"] == "queue_ready"
                and post["acquisition_receipt_hash"]
                != mapping["acquisition_receipt_hash"]
            )
            or (
                post["phase"] == "stopped"
                and post["stop_reason"] != "global_queue_exhausted"
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure does not replay controller attach",
                phase="historical_projection",
            )
        attempt_count = len(tuple(pre["attempt_receipt_hashes"]))
        attempt_hash_prefix = all_attempt_hashes[:attempt_count]
        prefix_draw_ids = tuple(
            draw_id
            for attempt_carrier in all_attempt_carriers[:attempt_count]
            for draw_id in tuple(
                _raw_mapping(
                    _expect_carrier(
                        attempt_carrier,
                        "V1CandidateAttemptReceipt",
                    ),
                    cohort=cohort,
                )["draw_ids"]
            )
        )
        expected_draw_hash = content_hash(
            prefix_draw_ids,
            prefix="partition-free-v1-globally-used-draw-ids-",
        )
        if (
            tuple(pre["attempt_receipt_hashes"]) != attempt_hash_prefix
            or tuple(post["attempt_receipt_hashes"]) != attempt_hash_prefix
            or pre["globally_used_draw_ids_hash"] != expected_draw_hash
            or post["globally_used_draw_ids_hash"] != expected_draw_hash
            or any(
                snapshot[name] is not None
                for snapshot in (pre, post)
                for name in (
                    "active_candidate_id",
                    "active_step_freeze_hash",
                    "active_ledger_hash",
                )
            )
            or pre["acquisition_receipt_hash"] is not None
            or pre["stop_reason"] is not None
            or (
                post["phase"] == "queue_ready"
                and post["stop_reason"] is not None
            )
            or (
                post["phase"] == "stopped"
                and post["acquisition_receipt_hash"] is not None
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure controller snapshot cannot replay",
                phase="historical_projection",
            )

        def controller_state_hash(
            snapshot: Mapping[str, Any],
            *,
            acquisition_hash: str | None,
        ) -> str:
            snapshot_projection = {
                "schema": controller_projection["schema"],
                "campaign_binding_hash": snapshot["campaign_binding_hash"],
                "controller_run_id": snapshot["controller_run_id"],
                "initial_mask_hash": controller_projection["initial_mask_hash"],
                "current_mask_hash": snapshot["current_mask_hash"],
                "phase": snapshot["phase"],
                "acquisition_hash": acquisition_hash,
                "queue_cursor": snapshot["queue_cursor"],
                "active_candidate_hash": None,
                "accepted_trajectory_hash": snapshot[
                    "accepted_trajectory_hash"
                ],
                "attempt_receipt_hashes": tuple(
                    snapshot["attempt_receipt_hashes"]
                ),
                "transition_receipt_hashes": tuple(
                    snapshot["transition_receipt_hashes"]
                ),
                "acquisition_receipt_hashes": tuple(
                    snapshot["acquisition_receipt_hashes"]
                ),
                "globally_used_draw_ids": prefix_draw_ids,
                "stop_reason": snapshot["stop_reason"],
                "logical_sequence": snapshot["logical_sequence"],
            }
            if tuple(snapshot.get("label_native_runtime_plan_hashes", ())):
                snapshot_projection.update(
                    {
                        "label_native_runtime_plan_hash": snapshot.get(
                            "label_native_runtime_plan_hash"
                        ),
                        "label_native_runtime_plan_hashes": tuple(
                            snapshot["label_native_runtime_plan_hashes"]
                        ),
                        "label_native_rejected_plan_indices": tuple(
                            snapshot.get(
                                "label_native_rejected_plan_indices",
                                (),
                            )
                        ),
                        "label_native_selected_acquisition_receipt_hash": (
                            snapshot.get(
                                "label_native_selected_acquisition_receipt_hash"
                            )
                        ),
                    }
                )
            return content_hash(
                snapshot_projection,
                prefix="partition-free-v1-controller-state-",
            )

        if (
            mapping["pre_attach_controller_state_hash"]
            != controller_state_hash(pre, acquisition_hash=None)
            or mapping["post_attach_controller_state_hash"]
            != controller_state_hash(
                post,
                acquisition_hash=(
                    mapping["acquisition_hash"]
                    if post["phase"] == "queue_ready"
                    else None
                ),
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "completed proposal closure controller state identity cannot replay",
                phase="historical_projection",
            )
        if (
            controller_projection["logical_sequence"] == post["logical_sequence"]
            and (
                mapping["post_attach_controller_state_hash"]
                != content_hash(
                    controller_projection,
                    prefix="partition-free-v1-controller-state-",
                )
                or _raw_controller_snapshot_receipt_hash(
                    post_carrier,
                    cohort=cohort,
                )
                != _controller_snapshot_hash(summary, cohort=cohort)
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "latest proposal closure has stale post-attach controller identity",
                phase="historical_projection",
            )
        if (
            generation == len(cast(Sequence[object], values)) - 1
            and controller_projection["acquisition_hash"] is not None
            and mapping["acquisition_hash"]
            != controller_projection["acquisition_hash"]
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "latest proposal closure names another live acquisition",
                phase="historical_projection",
            )
        closure_hash = _raw_content_hash(
            closure,
            cohort=cohort,
            prefix="partition-free-v1-runtime-completed-proposal-closure-",
        )
        closure_hashes.append(closure_hash)
    if values and len(closure_hashes) != len(acquisition_history):
        _refuse(
            V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
            "completed proposal closure count differs from acquisition history",
            phase="historical_projection",
        )
    return tuple(closure_hashes), backend_identity


def _validate_terminal_comparison_family_v2(
    journal: Mapping[str, Any],
    *,
    cohort: _CohortSpec,
    header: Mapping[str, Any],
) -> None:
    legacy_values = (
        journal["terminal_validation_family_freeze"],
        journal["terminal_validation_family_request"],
        journal["terminal_validation_family_observation"],
        journal["terminal_validation_family"],
    )
    if any(value is not None for value in legacy_values):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "journal-v7 mixes terminal family v1 and comparison v2",
            phase="raw_state_validation",
        )
    values = (
        journal["terminal_comparison_family_freeze"],
        journal["terminal_comparison_family_request"],
        journal["terminal_comparison_family_observation"],
        journal["terminal_comparison_family"],
    )
    seen_gap = False
    for value in values:
        if value is None:
            seen_gap = True
        elif seen_gap:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "journal-v7 skips a comparison-family causal joint",
                phase="raw_state_validation",
            )
    freeze = _expect_exact_carrier(
        values[0],
        module="mechanistic_systems.circuits.irg.modern_v1_receipts",
        name="V1TerminalComparisonFamilyFreezeReceiptV2",
    )
    freeze_hash = _carrier_hash(
        freeze,
        cohort=cohort,
        prefix="partition-free-v1-terminal-comparison-family-freeze-v2-",
    )
    assert freeze_hash is not None
    freeze_mapping = _raw_mapping(freeze, cohort=cohort)
    route_count = freeze_mapping["route_count"]
    role_masks = tuple(freeze_mapping["role_mask_hashes"])
    role_counts = tuple(freeze_mapping["role_retained_route_counts"])
    role_map = tuple(freeze_mapping["role_to_unique_lane"])
    unique_masks = tuple(freeze_mapping["unique_lane_mask_hashes"])
    expected_unique: list[object] = []
    expected_map: list[int] = []
    for mask_hash in role_masks:
        if mask_hash not in expected_unique:
            expected_unique.append(mask_hash)
        expected_map.append(expected_unique.index(mask_hash))
    current_v0_sha = freeze_mapping["current_v0_mask_artifact_sha256"]
    _sha256(current_v0_sha, "current_v0_mask_artifact_sha256")
    for name in (
        "pending_dense_reference_design_hash",
        "dense_current_v0_calibration_family_freeze_hash",
    ):
        _identity(freeze_mapping[name], name)
    calibration_hashes = tuple(freeze_mapping["unique_lane_calibration_hashes"])
    calibration_freeze_hashes = tuple(
        freeze_mapping["unique_lane_calibration_freeze_hashes"]
    )
    raw_calibration_payloads = tuple(
        freeze_mapping["unique_lane_calibration_freeze_payloads"]
    )
    calibration_payloads = tuple(
        _raw_canonical(payload, cohort=cohort)
        for payload in raw_calibration_payloads
    )
    expected_calibration_fields = set(
        _COMMON_RAW_FIELDS["V1FrozenAffineCalibration"]
    )
    shared_coordinates: tuple[object, ...] | None = None
    calibrations_valid = (
        len(calibration_hashes)
        == len(calibration_freeze_hashes)
        == len(calibration_payloads)
        == len(unique_masks)
    )
    for index, payload in enumerate(calibration_payloads):
        if not isinstance(payload, Mapping) or set(payload) != expected_calibration_fields:
            calibrations_valid = False
            continue
        fit_receipt = payload["fit_receipt"]
        coordinates = (
            payload["query_id"],
            payload["query_hash"],
            payload["target_class_id"],
            payload["calibration_context_axis_hash"],
            payload["source_label_map_hash"],
            payload["model_state_hash"],
            payload["calibration_schema"],
        )
        if shared_coordinates is None:
            shared_coordinates = coordinates
        elif coordinates != shared_coordinates:
            calibrations_valid = False
        if (
            index >= len(unique_masks)
            or payload["mask_hash"] != unique_masks[index]
            or payload["calibration_hash"] != calibration_hashes[index]
            or payload["schema"]
            != "partition_free_v1_frozen_affine_calibration_v1"
            or payload["validation_independent"] is not True
            or payload["terminal_rows_used_for_fit"] is not False
            or payload["terminal_outcomes_opened"] is not False
            or not isinstance(fit_receipt, Mapping)
            or fit_receipt.get("fit_succeeded") is not True
            or fit_receipt.get("mask_hash") != unique_masks[index]
            or fit_receipt.get("validation_independent") is not True
            or fit_receipt.get("terminal_rows_used_for_fit") is not False
            or fit_receipt.get("terminal_outcomes_opened") is not False
            or payload["fit_receipt_hash"]
            != content_hash(
                fit_receipt,
                prefix="partition-free-v1-frozen-calibration-fit-receipt-",
            )
            or content_hash(
                payload,
                prefix="partition-free-v1-frozen-affine-calibration-",
            )
            != calibration_freeze_hashes[index]
        ):
            calibrations_valid = False
    if (
        freeze_hash != header.get("terminal_comparison_family_freeze_hash")
        or freeze_mapping["schema"] != _TERMINAL_COMPARISON_FREEZE_SCHEMA_V2
        or tuple(freeze_mapping["roles"])
        != ("dense_reference", "current_v0_comparator", "terminal_v1")
        or freeze_mapping["validation_independent"] is not True
        or freeze_mapping["outcome_fields_absent"] is not True
        or freeze_mapping["terminal_rows_visible"] is not False
        or freeze_mapping["terminal_outcomes_opened"] is not False
        or freeze_mapping["terminal_outcomes_used_for_selection"] is not False
        or freeze_mapping["current_v0_comparison_only"] is not True
        or freeze_mapping["current_v0_candidate_or_commit_authority"] is not False
        or type(route_count) is not int
        or route_count <= 0
        or freeze_mapping["current_v0_mask_artifact_semantic_hash"]
        != freeze_mapping["current_v0_mask_hash"]
        or freeze_mapping["current_v0_mask_hash"]
        == freeze_mapping["dense_mask_hash"]
        or freeze_mapping["current_v0_mask_artifact_byte_count"]
        != (route_count + 7) // 8
        or role_masks
        != (
            freeze_mapping["dense_mask_hash"],
            freeze_mapping["current_v0_mask_hash"],
            freeze_mapping["final_v1_mask_hash"],
        )
        or len(role_counts) != 3
        or role_counts[0] != route_count
        or any(type(count) is not int or count <= 0 or count > route_count for count in role_counts)
        or tuple(expected_unique) != unique_masks
        or tuple(expected_map) != role_map
        or not calibrations_valid
    ):
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            "terminal-comparison freeze crossed its validation-independent firewall",
            phase="raw_state_validation",
        )

    request_hash: str | None = None
    request_carrier: _HistoricalCarrier | None = None
    request_mapping: Mapping[str, Any] | None = None
    if values[1] is not None:
        request_carrier = _expect_exact_carrier(
            values[1],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1TerminalComparisonFamilyRequestReceiptV2",
        )
        request_hash = _carrier_hash(
            request_carrier,
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-request-v2-",
        )
        request_mapping = _raw_mapping(request_carrier, cohort=cohort)
        nested_freeze_hash = _carrier_hash(
            _expect_exact_carrier(
                request_mapping["family_freeze"],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalComparisonFamilyFreezeReceiptV2",
            ),
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-freeze-v2-",
        )
        if (
            nested_freeze_hash != freeze_hash
            or request_mapping["request_sequence"]
            != freeze_mapping["freeze_sequence"] + 1
            or request_mapping["observation_artifact_schema"]
            != _TERMINAL_COMPARISON_MATRIX_SCHEMA_V2
            or request_mapping["evaluation_mode"]
            != "single_common_panel_dense_current_v0_terminal_v1_unique_mask_lanes"
            or request_mapping["validation_independent"] is not True
            or request_mapping["outcome_fields_absent"] is not True
            or request_mapping["terminal_rows_visible"] is not False
            or request_mapping["terminal_outcomes_opened"] is not False
            or request_mapping["checkpoint_before_outcomes_required"] is not True
            or request_mapping["schema"] != _TERMINAL_COMPARISON_REQUEST_SCHEMA_V2
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "terminal-comparison request crossed its pre-reveal firewall",
                phase="raw_state_validation",
            )
    if request_hash != header.get("terminal_comparison_family_request_hash"):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "terminal-comparison request differs from its strict header",
            phase="historical_projection",
        )

    expected_artifact = None
    observation_hash: str | None = None
    observation_mapping: Mapping[str, Any] | None = None
    if values[2] is not None:
        assert request_carrier is not None and request_mapping is not None
        assert request_hash is not None
        observation = _expect_exact_carrier(
            values[2],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1TerminalComparisonFamilyObservationReceiptV2",
        )
        observation_hash = _carrier_hash(
            observation,
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-outcome-v2-",
        )
        observation_mapping = _raw_mapping(observation, cohort=cohort)
        nested_request_hash = _carrier_hash(
            _expect_exact_carrier(
                observation_mapping["request"],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalComparisonFamilyRequestReceiptV2",
            ),
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-request-v2-",
        )
        metrics = tuple(
            _raw_canonical(item, cohort=cohort)
            for item in observation_mapping["unique_lane_metrics"]
        )
        deltas = _raw_canonical(
            observation_mapping["paired_mean_deltas"], cohort=cohort
        )
        pair_hashes = _raw_canonical(
            observation_mapping["paired_validation_difference_hashes"],
            cohort=cohort,
        )
        semantic_hash = content_hash(
            {
                "terminal_request_hash": request_hash,
                "joint_validation_values_hash": observation_mapping[
                    "joint_validation_values_hash"
                ],
                "joint_validation_shape": tuple(
                    observation_mapping["joint_validation_shape"]
                ),
                "unique_lane_validation_values_hashes": tuple(
                    observation_mapping["unique_lane_validation_values_hashes"]
                ),
                "unique_lane_validation_value_counts": tuple(
                    observation_mapping["unique_lane_validation_value_counts"]
                ),
                "unique_lane_metrics": metrics,
                "paired_mean_deltas": deltas,
                "paired_validation_difference_hashes": pair_hashes,
                "paired_delta_derivation": observation_mapping[
                    "paired_delta_derivation"
                ],
                "common_panel_evaluator_receipt_hash": observation_mapping[
                    "common_panel_evaluator_receipt_hash"
                ],
            },
            prefix=(
                "partition-free-v1-terminal-comparison-family-"
                "observation-artifact-v2-"
            ),
        )
        if (
            nested_request_hash != request_hash
            or observation_mapping["observation_sequence"]
            != request_mapping["request_sequence"] + 1
            or observation_mapping["observation_artifact_relative_path"]
            != request_mapping["observation_artifact_relative_path"]
            or observation_mapping["observation_artifact_semantic_hash"]
            != semantic_hash
            or observation_mapping["artifact_durable_before_receipt"] is not True
            or observation_mapping["artifact_replay_verified"] is not True
            or observation_mapping["single_common_panel_evaluation"] is not True
            or observation_mapping["raw_validation_rows_persisted"] is not True
            or observation_mapping["paired_delta_derivation"]
            != "mean_of_pointwise_common_panel_row_differences_float64_v1"
            or observation_mapping["fed_back_into_selection"] is not False
            or observation_mapping["terminal_outcomes_used_for_selection"] is not False
            or observation_mapping["claim_execution_certified"] is not False
            or observation_mapping["mask_commit_authority"] is not False
            or observation_mapping["schema"]
            != _TERMINAL_COMPARISON_OBSERVATION_SCHEMA_V2
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "terminal-comparison observation crossed its replay firewall",
                phase="raw_state_validation",
            )
        expected_artifact = {
            "schema": _TERMINAL_COMPARISON_ARTIFACT_REFERENCE_SCHEMA_V2,
            "relative_path": observation_mapping[
                "observation_artifact_relative_path"
            ],
            "artifact_schema": request_mapping["observation_artifact_schema"],
            "artifact_sha256": observation_mapping["observation_artifact_sha256"],
            "artifact_size_bytes": observation_mapping[
                "observation_artifact_size_bytes"
            ],
            "terminal_request_hash": request_hash,
            "family_freeze_hash": freeze_hash,
            "roles": list(freeze_mapping["roles"]),
            "role_to_unique_lane": list(freeze_mapping["role_to_unique_lane"]),
            "joint_validation_values_hash": observation_mapping[
                "joint_validation_values_hash"
            ],
            "joint_validation_shape": list(
                observation_mapping["joint_validation_shape"]
            ),
            "unique_lane_validation_values_hashes": list(
                observation_mapping["unique_lane_validation_values_hashes"]
            ),
            "unique_lane_validation_value_counts": list(
                observation_mapping["unique_lane_validation_value_counts"]
            ),
            "unique_lane_metrics": [dict(item) for item in metrics],
            "paired_mean_deltas": dict(deltas),
            "paired_validation_difference_hashes": dict(pair_hashes),
            "paired_delta_derivation": observation_mapping[
                "paired_delta_derivation"
            ],
            "common_panel_evaluator_receipt_hash": observation_mapping[
                "common_panel_evaluator_receipt_hash"
            ],
            "observation_artifact_semantic_hash": observation_mapping[
                "observation_artifact_semantic_hash"
            ],
            "terminal_observation_hash": observation_hash,
        }
    if expected_artifact != header.get("terminal_comparison_family_artifact"):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "terminal-comparison observation differs from its matrix reference",
            phase="historical_projection",
        )
    if values[3] is not None:
        assert observation_hash is not None and observation_mapping is not None
        family = _expect_exact_carrier(
            values[3],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1TerminalComparisonFamilyReceiptV2",
        )
        family_mapping = _raw_mapping(family, cohort=cohort)
        nested_observation_hash = _carrier_hash(
            _expect_exact_carrier(
                family_mapping["observation"],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalComparisonFamilyObservationReceiptV2",
            ),
            cohort=cohort,
            prefix="partition-free-v1-terminal-comparison-family-outcome-v2-",
        )
        if (
            nested_observation_hash != observation_hash
            or family_mapping["reveal_sequence"]
            != observation_mapping["observation_sequence"]
            or family_mapping["single_reveal"] is not True
            or family_mapping["terminal_comparison_authority"] is not True
            or family_mapping["dense_reference_resolved"] is not True
            or family_mapping["fed_back_into_selection"] is not False
            or family_mapping["claim_execution_certified"] is not False
            or family_mapping["mask_commit_authority"] is not False
            or family_mapping["schema"] != _TERMINAL_COMPARISON_SCHEMA_V2
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "terminal-comparison receipt crossed its authority firewall",
                phase="raw_state_validation",
            )


def _validate_journal_sidecar_carriers(
    root: _HistoricalCarrier,
    *,
    cohort: _CohortSpec,
    checkpoint_magic: bytes,
    header: Mapping[str, Any],
) -> None:
    if root.historical_name != "_V1RuntimeCheckpointJournal":
        return
    state = _carrier_state_sequence(root, label=root.historical_name)
    if checkpoint_magic == _COMPACT_MAGIC:
        expected_arity = 11
        expected_schema = _JOURNAL_V1_SCHEMA
    elif checkpoint_magic == _PROPOSAL_MAGIC:
        if len(state) == 23 and state[-1] == _JOURNAL_V7_SCHEMA:
            expected_arity = 23
            expected_schema = _JOURNAL_V7_SCHEMA
        elif len(state) == 19 and state[-1] == _JOURNAL_V6_SCHEMA:
            expected_arity = 19
            expected_schema = _JOURNAL_V6_SCHEMA
        elif len(state) == 12 and state[-1] == _JOURNAL_V2_SCHEMA:
            expected_arity = 12
            expected_schema = _JOURNAL_V2_SCHEMA
        elif len(state) == 13 and state[-1] == _JOURNAL_V3_SCHEMA:
            # Journal-v3 is historical, decision-only staged authority.  It
            # remains inertly inspectable, but the live runtime refuses it.
            expected_arity = 13
            expected_schema = _JOURNAL_V3_SCHEMA
        elif len(state) == 15 and state[-1] == _JOURNAL_V5_SCHEMA:
            expected_arity = 15
            expected_schema = _JOURNAL_V5_SCHEMA
        else:
            expected_arity = 14
            expected_schema = _JOURNAL_V4_SCHEMA
    else:
        _refuse(
            V1CheckpointRefusalCode.COHORT_INCONSISTENT,
            "journal payload is incompatible with checkpoint magic",
            phase="cohort_classification",
        )
    if len(state) != expected_arity or state[-1] != expected_schema:
        _refuse(
            V1CheckpointRefusalCode.COHORT_INCONSISTENT,
            "journal arity/schema disagrees with checkpoint framing",
            phase="cohort_classification",
        )
    journal = _raw_mapping(root, cohort=cohort)
    controller = _raw_mapping(
        _expect_carrier(journal["controller"], "_V1ControllerCheckpointProjection"),
        cohort=cohort,
    )
    acquisition = controller["acquisition_sidecar"]
    observed_acquisition = (
        None
        if acquisition is None
        else _raw_canonical(
            _expect_exact_carrier(
                acquisition,
                module="mechanistic_systems.circuits.irg.modern_v1_runtime",
                name="V1RuntimeAcquisitionSidecarReceipt",
            ),
            cohort=cohort,
        )
    )
    if observed_acquisition != header.get("acquisition_sidecar"):
        _refuse(
            V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
            "raw journal acquisition receipt differs from its strict header",
            phase="historical_projection",
        )
    if expected_arity in {12, 13, 14, 15, 19, 23}:
        proposal = journal["proposal_state_sidecar"]
        observed_proposal = (
            None
            if proposal is None
            else _raw_canonical(
                _expect_exact_carrier(
                    proposal,
                    module=__name__,
                    name="V1RuntimeProposalStateSidecarReceipt",
                ),
                cohort=cohort,
            )
        )
        if observed_proposal != header.get("proposal_state_sidecar"):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "raw journal proposal-state receipt differs from its strict header",
                phase="historical_projection",
            )
    if expected_schema == _JOURNAL_V3_SCHEMA:
        decisions = journal["completed_proposal_freeze_decisions"]
        if type(decisions) not in {tuple, list}:
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "completed proposal freeze ledger is not positional",
                phase="raw_state_validation",
            )
        observed_decision_hashes: list[str] = []
        for item in decisions:
            decision = _expect_exact_carrier(
                item,
                module=(
                    "mechanistic_systems.circuits.irg."
                    "modern_v1_joint_acquisition"
                ),
                name="V1ProposalFreezeDecisionReceipt",
            )
            decision_mapping = _raw_mapping(decision, cohort=cohort)
            if decision_mapping["action"] != "authorize_candidate_freeze":
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "completed proposal freeze ledger contains a refusal",
                    phase="raw_state_validation",
                )
            decision_hash = _carrier_hash(
                decision,
                cohort=cohort,
                prefix="partition-free-v1-proposal-freeze-decision-",
            )
            assert decision_hash is not None
            observed_decision_hashes.append(decision_hash)
        if observed_decision_hashes != header.get(
            "completed_proposal_closure_hashes"
        ):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "historical decision-only proposal ledger is not a closure ledger",
                phase="historical_projection",
            )
    elif expected_schema in {
        _JOURNAL_V4_SCHEMA,
        _JOURNAL_V5_SCHEMA,
        _JOURNAL_V6_SCHEMA,
        _JOURNAL_V7_SCHEMA,
    }:
        observed_pending_paired_hash = None
        if expected_schema in {
            _JOURNAL_V5_SCHEMA,
            _JOURNAL_V6_SCHEMA,
            _JOURNAL_V7_SCHEMA,
        }:
            observed_pending_paired_hash = _carrier_hash(
                journal["pending_paired_bce_chunk"],
                cohort=cohort,
                prefix="partition-free-v1-runtime-pending-paired-bce-chunk-",
            )
            if (
                expected_schema == _JOURNAL_V5_SCHEMA
                and observed_pending_paired_hash is None
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "checkpoint journal-v5 lacks pending paired-BCE state",
                    phase="raw_state_validation",
                )
            if observed_pending_paired_hash != header.get(
                "pending_paired_bce_chunk_hash"
            ):
                _refuse(
                    V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                    "pending paired-BCE journal hash differs from its strict header",
                    phase="historical_projection",
                )
        journal_backend = journal["staged_proposal_backend_identity_hash"]
        if journal_backend is not None:
            journal_backend = _raw_nonempty_string(
                journal_backend,
                "staged_proposal_backend_identity_hash",
            )
        if journal_backend != header.get("staged_proposal_backend_identity_hash"):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "journal staged backend differs from its strict header",
                phase="historical_projection",
            )
        observed_closure_hashes, closure_backend = (
            _validate_completed_proposal_closures(
                journal["completed_proposal_closures"],
                cohort=cohort,
                controller=_expect_carrier(
                    journal["controller"],
                    "_V1ControllerCheckpointProjection",
                ),
            )
        )
        if list(observed_closure_hashes) != header.get(
            "completed_proposal_closure_hashes"
        ):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "completed proposal closure ledger differs from its strict header",
                phase="historical_projection",
            )
        family_freeze_present = (
            expected_schema == _JOURNAL_V6_SCHEMA
            and journal["terminal_validation_family_freeze"] is not None
        )
        comparison_freeze_present = (
            expected_schema == _JOURNAL_V7_SCHEMA
            and journal["terminal_comparison_family_freeze"] is not None
        )
        if (
            observed_proposal is None
            and not observed_closure_hashes
            and observed_pending_paired_hash is None
            and not family_freeze_present
            and not comparison_freeze_present
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "checkpoint journal lacks staged, paired, or terminal-family state",
                phase="raw_state_validation",
            )
        if (
            observed_proposal is not None
            and closure_backend is not None
            and observed_proposal["proposal_backend_identity_hash"]
            != closure_backend
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "live proposal state drifts from the completed closure backend",
                phase="historical_projection",
            )
        if (
            journal_backend is None
            and (observed_proposal is not None or closure_backend is not None)
        ) or (
            journal_backend is not None
            and (
                (
                    observed_proposal is not None
                    and observed_proposal["proposal_backend_identity_hash"]
                    != journal_backend
                )
                or (closure_backend is not None and closure_backend != journal_backend)
            )
        ):
            _refuse(
                V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                "proposal closure/live state changed the staged backend anchor",
                phase="historical_projection",
            )
    if expected_schema == _JOURNAL_V6_SCHEMA:
        family_values = (
            journal["terminal_validation_family_freeze"],
            journal["terminal_validation_family_request"],
            journal["terminal_validation_family_observation"],
            journal["terminal_validation_family"],
        )
        seen_gap = False
        for value in family_values:
            if value is None:
                seen_gap = True
            elif seen_gap:
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "journal-v6 skips a terminal-family causal joint",
                    phase="raw_state_validation",
                )
        freeze_carrier = _expect_exact_carrier(
            family_values[0],
            module="mechanistic_systems.circuits.irg.modern_v1_receipts",
            name="V1TerminalValidationFamilyFreezeReceipt",
        )
        freeze_hash = _carrier_hash(
            freeze_carrier,
            cohort=cohort,
            prefix="partition-free-v1-terminal-validation-family-freeze-",
        )
        assert freeze_hash is not None
        freeze_mapping = _raw_mapping(freeze_carrier, cohort=cohort)
        if (
            freeze_mapping["schema"] != _TERMINAL_FAMILY_FREEZE_SCHEMA
            or freeze_mapping["validation_independent"] is not True
            or freeze_mapping["terminal_rows_visible"] is not False
            or freeze_mapping["terminal_outcomes_used_for_selection"] is not False
            or freeze_mapping["matched_density_comparison_registered"] is not True
            or tuple(freeze_mapping["roles"])
            != ("dense", "pareto_milestone", "final")
        ):
            _refuse(
                V1CheckpointRefusalCode.RAW_STATE_INVALID,
                "terminal-family freeze crossed its causal firewall",
                phase="raw_state_validation",
            )
        if freeze_hash != header.get("terminal_validation_family_freeze_hash"):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "terminal-family freeze differs from its strict header",
                phase="historical_projection",
            )
        request_hash: str | None = None
        request_carrier: _HistoricalCarrier | None = None
        if family_values[1] is not None:
            request_carrier = _expect_exact_carrier(
                family_values[1],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalValidationFamilyRequestReceipt",
            )
            request_hash = _carrier_hash(
                request_carrier,
                cohort=cohort,
                prefix="partition-free-v1-terminal-validation-family-request-",
            )
            request_mapping = _raw_mapping(request_carrier, cohort=cohort)
            nested_freeze_hash = _carrier_hash(
                _expect_exact_carrier(
                    request_mapping["family_freeze"],
                    module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                    name="V1TerminalValidationFamilyFreezeReceipt",
                ),
                cohort=cohort,
                prefix="partition-free-v1-terminal-validation-family-freeze-",
            )
            if (
                nested_freeze_hash != freeze_hash
                or request_mapping["request_sequence"]
                != freeze_mapping["freeze_sequence"] + 1
                or request_mapping["evaluation_mode"]
                != "single_common_panel_unique_mask_lanes"
                or request_mapping["validation_independent"] is not True
                or request_mapping["outcome_fields_absent"] is not True
                or request_mapping["terminal_rows_visible"] is not False
                or request_mapping["checkpoint_before_outcomes_required"] is not True
                or request_mapping["schema"] != _TERMINAL_FAMILY_REQUEST_SCHEMA
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "terminal-family request crossed its pre-reveal firewall",
                    phase="raw_state_validation",
                )
        if request_hash != header.get("terminal_validation_family_request_hash"):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "terminal-family request differs from its strict header",
                phase="historical_projection",
            )
        expected_artifact = None
        observation_hash: str | None = None
        observation_carrier: _HistoricalCarrier | None = None
        if family_values[2] is not None:
            assert request_carrier is not None and request_hash is not None
            observation_carrier = _expect_exact_carrier(
                family_values[2],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalValidationFamilyObservationReceipt",
            )
            observation_hash = _carrier_hash(
                observation_carrier,
                cohort=cohort,
                prefix="partition-free-v1-terminal-validation-family-outcome-",
            )
            assert observation_hash is not None
            request_mapping = _raw_mapping(request_carrier, cohort=cohort)
            observation_mapping = _raw_mapping(observation_carrier, cohort=cohort)
            nested_request_hash = _carrier_hash(
                _expect_exact_carrier(
                    observation_mapping["request"],
                    module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                    name="V1TerminalValidationFamilyRequestReceipt",
                ),
                cohort=cohort,
                prefix="partition-free-v1-terminal-validation-family-request-",
            )
            semantic_hash = content_hash(
                {
                    "terminal_request_hash": request_hash,
                    "joint_validation_values_hash": observation_mapping[
                        "joint_validation_values_hash"
                    ],
                    "joint_validation_shape": tuple(
                        observation_mapping["joint_validation_shape"]
                    ),
                    "unique_lane_validation_values_hashes": tuple(
                        observation_mapping[
                            "unique_lane_validation_values_hashes"
                        ]
                    ),
                    "unique_lane_validation_value_counts": tuple(
                        observation_mapping[
                            "unique_lane_validation_value_counts"
                        ]
                    ),
                    "unique_lane_metrics": tuple(
                        _raw_canonical(item, cohort=cohort)
                        for item in observation_mapping["unique_lane_metrics"]
                    ),
                    "common_panel_evaluator_receipt_hash": observation_mapping[
                        "common_panel_evaluator_receipt_hash"
                    ],
                },
                prefix=(
                    "partition-free-v1-terminal-validation-family-"
                    "observation-artifact-"
                ),
            )
            if (
                nested_request_hash != request_hash
                or observation_mapping["observation_sequence"]
                != request_mapping["request_sequence"] + 1
                or observation_mapping["observation_artifact_relative_path"]
                != request_mapping["observation_artifact_relative_path"]
                or observation_mapping["observation_artifact_semantic_hash"]
                != semantic_hash
                or observation_mapping["artifact_durable_before_receipt"] is not True
                or observation_mapping["artifact_replay_verified"] is not True
                or observation_mapping["single_common_panel_evaluation"] is not True
                or observation_mapping["raw_validation_rows_persisted"] is not True
                or observation_mapping["fed_back_into_selection"] is not False
                or observation_mapping["terminal_outcomes_used_for_selection"] is not False
                or observation_mapping["claim_execution_certified"] is not False
                or observation_mapping["mask_commit_authority"] is not False
                or observation_mapping["schema"]
                != _TERMINAL_FAMILY_OBSERVATION_SCHEMA
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "terminal-family observation crossed its replay firewall",
                    phase="raw_state_validation",
                )
            expected_artifact = {
                "schema": _TERMINAL_FAMILY_ARTIFACT_REFERENCE_SCHEMA,
                "relative_path": observation_mapping[
                    "observation_artifact_relative_path"
                ],
                "artifact_schema": request_mapping["observation_artifact_schema"],
                "artifact_sha256": observation_mapping[
                    "observation_artifact_sha256"
                ],
                "artifact_size_bytes": observation_mapping[
                    "observation_artifact_size_bytes"
                ],
                "terminal_request_hash": request_hash,
                "family_freeze_hash": freeze_hash,
                "joint_validation_values_hash": observation_mapping[
                    "joint_validation_values_hash"
                ],
                "joint_validation_shape": list(
                    observation_mapping["joint_validation_shape"]
                ),
                "unique_lane_validation_values_hashes": list(
                    observation_mapping["unique_lane_validation_values_hashes"]
                ),
                "unique_lane_validation_value_counts": list(
                    observation_mapping["unique_lane_validation_value_counts"]
                ),
                "unique_lane_metrics": [
                    dict(_raw_canonical(item, cohort=cohort))
                    for item in observation_mapping["unique_lane_metrics"]
                ],
                "common_panel_evaluator_receipt_hash": observation_mapping[
                    "common_panel_evaluator_receipt_hash"
                ],
                "observation_artifact_semantic_hash": observation_mapping[
                    "observation_artifact_semantic_hash"
                ],
                "terminal_observation_hash": observation_hash,
            }
        if expected_artifact != header.get("terminal_validation_family_artifact"):
            _refuse(
                V1CheckpointRefusalCode.SIDECAR_REFERENCE_MISMATCH,
                "terminal-family observation differs from its matrix reference",
                phase="historical_projection",
            )
        if family_values[3] is not None:
            assert observation_carrier is not None
            family_carrier = _expect_exact_carrier(
                family_values[3],
                module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                name="V1TerminalValidationFamilyReceipt",
            )
            family_mapping = _raw_mapping(family_carrier, cohort=cohort)
            nested_observation_hash = _carrier_hash(
                _expect_exact_carrier(
                    family_mapping["observation"],
                    module="mechanistic_systems.circuits.irg.modern_v1_receipts",
                    name="V1TerminalValidationFamilyObservationReceipt",
                ),
                cohort=cohort,
                prefix="partition-free-v1-terminal-validation-family-outcome-",
            )
            if nested_observation_hash != observation_hash:
                _refuse(
                    V1CheckpointRefusalCode.HISTORICAL_IDENTITY_MISMATCH,
                    "terminal-family final receipt names another observation",
                    phase="historical_projection",
                )
            if (
                family_mapping["reveal_sequence"]
                != observation_mapping["observation_sequence"]
                or family_mapping["single_reveal"] is not True
                or family_mapping["matched_density_comparison_authority"] is not True
                or family_mapping["fed_back_into_selection"] is not False
                or family_mapping["claim_execution_certified"] is not False
                or family_mapping["mask_commit_authority"] is not False
                or family_mapping["schema"] != _TERMINAL_FAMILY_SCHEMA
            ):
                _refuse(
                    V1CheckpointRefusalCode.RAW_STATE_INVALID,
                    "terminal-family receipt crossed its authority firewall",
                    phase="raw_state_validation",
                )
    if expected_schema == _JOURNAL_V7_SCHEMA:
        _validate_terminal_comparison_family_v2(
            journal,
            cohort=cohort,
            header=header,
        )


def inspect_v1_runtime_checkpoint_frame(
    path: str | os.PathLike[str],
) -> V1FramedArtifactInspectionReceipt:
    """Verify only the checkpoint frame through one descriptor, without pickle.

    This is useful for quarantined or incomplete fixture diagnosis.  It does
    not verify referenced sidecars and therefore grants no resume or migration
    authority.
    """

    source = Path(path)
    parent_fd = _open_parent_without_symlinks(source.parent)
    checkpoint_fd: int | None = None
    try:
        checkpoint_fd = _open_regular_at(
            parent_fd,
            source.name,
            artifact=str(source),
            missing_code=V1CheckpointRefusalCode.PATH_NOT_REGULAR,
        )
        frame = _verify_frame(
            checkpoint_fd,
            artifact=str(source),
            allowed_magics=frozenset({_LEGACY_MAGIC, _COMPACT_MAGIC, _PROPOSAL_MAGIC}),
            sidecar=False,
        )
        return frame.receipt
    finally:
        if checkpoint_fd is not None:
            os.close(checkpoint_fd)
        os.close(parent_fd)


def inspect_v1_runtime_checkpoint(
    path: str | os.PathLike[str],
    *,
    trusted_local: bool,
) -> V1HistoricalRuntimeCheckpointView:
    """Integrity-verify, raw-map, and classify one inspect-only checkpoint.

    All main and sidecar framing/digests are complete before the compatibility
    decoder is constructed.  Historical states are never returned as live
    runtime objects, and this function performs no migration or publication.
    """

    if trusted_local is not True:
        _refuse(
            V1CheckpointRefusalCode.TRUSTED_LOCAL_REQUIRED,
            "pickle-bearing checkpoints require explicit trusted-local opt-in",
            phase="authority_boundary",
            artifact=path,
        )
    source = Path(path)
    parent_fd = _open_parent_without_symlinks(source.parent)
    checkpoint_fd: int | None = None
    sidecar_fds: list[int] = []
    try:
        checkpoint_fd = _open_regular_at(
            parent_fd,
            source.name,
            artifact=str(source),
            missing_code=V1CheckpointRefusalCode.PATH_NOT_REGULAR,
        )
        checkpoint = _verify_frame(
            checkpoint_fd,
            artifact=str(source),
            allowed_magics=frozenset({_LEGACY_MAGIC, _COMPACT_MAGIC, _PROPOSAL_MAGIC}),
            sidecar=False,
        )
        sidecars: list[_VerifiedFrame] = []
        pickle_sidecars: list[_VerifiedFrame] = []
        sidecar_kinds: list[str] = []
        acquisition_reference = checkpoint.header.get("acquisition_sidecar")
        if acquisition_reference is not None:
            acquisition_frame, acquisition_fd = _open_and_verify_sidecar(
                checkpoint_parent_fd=parent_fd,
                checkpoint_parent_path=source.parent,
                checkpoint_name=source.name,
                reference=acquisition_reference,
                kind="acquisition",
            )
            sidecar_fds.append(acquisition_fd)
            sidecars.append(acquisition_frame)
            pickle_sidecars.append(acquisition_frame)
            sidecar_kinds.append("acquisition")
        proposal_reference = checkpoint.header.get("proposal_state_sidecar")
        if proposal_reference is not None:
            proposal_frame, proposal_fd = _open_and_verify_sidecar(
                checkpoint_parent_fd=parent_fd,
                checkpoint_parent_path=source.parent,
                checkpoint_name=source.name,
                reference=proposal_reference,
                kind="proposal",
            )
            sidecar_fds.append(proposal_fd)
            sidecars.append(proposal_frame)
            pickle_sidecars.append(proposal_frame)
            sidecar_kinds.append("proposal")
        terminal_family_reference = checkpoint.header.get(
            "terminal_validation_family_artifact"
        )
        if terminal_family_reference is not None:
            terminal_frame, terminal_fd = _open_and_verify_sidecar(
                checkpoint_parent_fd=parent_fd,
                checkpoint_parent_path=source.parent,
                checkpoint_name=source.name,
                reference=terminal_family_reference,
                kind="terminal_family",
            )
            sidecar_fds.append(terminal_fd)
            sidecars.append(terminal_frame)
            _validate_terminal_family_matrix_payload(
                terminal_frame,
                reference=terminal_family_reference,
            )
        terminal_comparison_reference = checkpoint.header.get(
            "terminal_comparison_family_artifact"
        )
        if terminal_comparison_reference is not None:
            comparison_frame, comparison_fd = _open_and_verify_sidecar(
                checkpoint_parent_fd=parent_fd,
                checkpoint_parent_path=source.parent,
                checkpoint_name=source.name,
                reference=terminal_comparison_reference,
                kind="terminal_comparison",
            )
            sidecar_fds.append(comparison_fd)
            sidecars.append(comparison_frame)
            _validate_terminal_comparison_matrix_payload_v2(
                comparison_frame,
                reference=terminal_comparison_reference,
            )

        # Integrity barrier: no compatibility unpickler exists above this line.
        root = _raw_decode(checkpoint)
        all_frames = (checkpoint, *sidecars)
        _assert_frames_unchanged(all_frames)
        decoded_sidecars: list[Any] = []
        for frame in pickle_sidecars:
            decoded_sidecars.append(_raw_decode(frame))
            _assert_frames_unchanged(all_frames)
        raw_sidecars = dict(zip(sidecar_kinds, decoded_sidecars, strict=True))
        raw_proposal = raw_sidecars.get("proposal")
        proposal_state_hashes: tuple[str, str | None] | None = None
        if proposal_reference is not None:
            proposal_state_hashes = _validate_proposal_state_payload(
                raw_proposal,
                reference=proposal_reference,
            )
        root_carrier = _expect_exact_carrier(
            root,
            module="mechanistic_systems.circuits.irg.modern_v1_runtime",
            name=(
                "V1RuntimeState"
                if checkpoint.magic == _LEGACY_MAGIC
                else "_V1RuntimeCheckpointJournal"
            ),
        )
        root_instances = _raw_instances(root_carrier)
        classification_instances = _raw_instances(
            (root_carrier, tuple(raw_sidecars.values()))
        )
        cohort = _classify_cohort(
            magic=checkpoint.magic,
            instances=classification_instances,
        )
        for carrier in root_instances:
            _raw_mapping(carrier, cohort=cohort)
        signal_instances = list(root_instances)
        raw_acquisition = raw_sidecars.get("acquisition")
        if acquisition_reference is not None:
            _validate_acquisition_sidecar_payload(
                raw_acquisition,
                cohort=cohort,
                reference=acquisition_reference,
            )
            signal_instances.extend(_raw_instances(raw_acquisition))
            if type(raw_acquisition) is bytes:
                from .modern_v1_operator_checkpoint_wire import inspect_wire_payload
                wire=inspect_wire_payload(raw_acquisition)
                journal_fields=_raw_mapping(root_carrier,cohort=cohort)
                controller_fields=_raw_mapping(journal_fields['controller'],cohort=cohort)
                active=controller_fields['active_candidate']
                if active is not None:
                    fields=_operator_active_mapping(active)
                    if (fields['acquisition_receipt_hash']!=wire['acquisition_hash']
                            or content_hash(fields['proposal'])!=content_hash(wire['candidate_definition'])):
                        _refuse(V1CheckpointRefusalCode.RAW_STATE_INVALID,
                            'operator active projection differs from verified sidecar',phase='raw_state_validation')
        if raw_proposal is not None:
            signal_instances.append(
                _expect_carrier(
                    raw_proposal,
                    "V1RuntimeProposalStateSidecarPayload",
                )
            )
        _validate_journal_sidecar_carriers(
            root_carrier,
            cohort=cohort,
            checkpoint_magic=checkpoint.magic,
            header=checkpoint.header,
        )
        runtime_schema, runtime_hash, controller_hash, pending_hash = (
            _runtime_projection(
                root_carrier,
                cohort=cohort,
                header=checkpoint.header,
                proposal_state_hashes=proposal_state_hashes,
            )
        )
        arities, schemas = _observed_signals(
            tuple(signal_instances),
            cohort=cohort,
        )
        evidence = V1CheckpointInspectionEvidenceReceipt(
            checkpoint_frame=checkpoint.receipt,
            sidecar_frames=tuple(frame.receipt for frame in sidecars),
            all_referenced_sidecars_verified_before_decode=True,
            all_decoded_sidecar_payload_identities_replayed=True,
            historical_acquisition_layout_source_certified=(
                acquisition_reference is None or cohort is _V29_PLUS
            ),
            proposal_runtime_contract_validation_performed=False,
            trusted_local_opt_in=True,
            raw_compatibility_decode_performed=True,
            live_runtime_class_assignment_performed=False,
            embedded_runtime_identity_replayed=True,
            embedded_controller_identity_replayed=True,
            embedded_controller_snapshot_replayed=True,
            terminal_validation_execution_performed=False,
            mask_mutation_authority=False,
        )
        disposition = (
            "validated_pre_v29_inspection_only_no_execution_or_migration"
            if cohort is not _V29_PLUS
            else "validated_v29_plus_migration_and_publication_deferred"
        )
        migration = V1CheckpointMigrationEvidenceReceipt(
            source_cohort=cohort.label,
            source_checkpoint_file_sha256=checkpoint.receipt.file_sha256,
            source_payload_sha256=checkpoint.receipt.payload_sha256,
            source_runtime_state_hash=runtime_hash,
            historical_projection_validated=True,
            executable_source_state_materialized=False,
            migration_bytes_published=False,
            migration_authority=False,
            disposition=disposition,
        )
        return V1HistoricalRuntimeCheckpointView(
            checkpoint_path=str(source),
            cohort=cohort.label,
            checkpoint_magic=checkpoint.magic.decode("ascii"),
            checkpoint_schema=str(checkpoint.header["schema"]),
            runtime_schema=runtime_schema,
            runtime_state_hash=runtime_hash,
            controller_state_hash=controller_hash,
            controller_snapshot_hash=str(checkpoint.header["controller_snapshot_hash"]),
            pending_look_hash=pending_hash,
            observed_type_arities=arities,
            observed_schema_signals=schemas,
            acquisition_sidecar_reference=acquisition_reference,
            proposal_state_sidecar_reference=proposal_reference,
            inspection_evidence=evidence,
            migration_evidence=migration,
        )
    except V1CheckpointInspectionRefusal:
        raise
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        _refuse(
            V1CheckpointRefusalCode.RAW_STATE_INVALID,
            f"trusted raw graph validation failed: {type(exc).__name__}",
            phase="raw_state_validation",
            artifact=source,
        )
        raise AssertionError("unreachable") from exc
    finally:
        for descriptor in reversed(sidecar_fds):
            os.close(descriptor)
        if checkpoint_fd is not None:
            os.close(checkpoint_fd)
        os.close(parent_fd)


__all__ = [
    "V1CheckpointInspectionEvidenceReceipt",
    "V1CheckpointInspectionRefusal",
    "V1CheckpointMigrationEvidenceReceipt",
    "V1CheckpointRefusalCode",
    "V1FramedArtifactInspectionReceipt",
    "V1HistoricalRuntimeCheckpointView",
    "V1RuntimeProposalStateSidecarPayload",
    "V1RuntimeProposalStateSidecarReceipt",
    "inspect_v1_runtime_checkpoint",
    "inspect_v1_runtime_checkpoint_frame",
]
