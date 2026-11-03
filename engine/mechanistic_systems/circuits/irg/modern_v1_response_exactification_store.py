"""Durable selected-shell F/Q acquisition for response-exactified v1 ladders.

The response-exactification module freezes a packed candidate ladder, its
action-specific endpoint envelope, and an independent F/Q alpha design before
fresh response rows are observed.  This module owns the next boundary:

* a train-only source role and every statistical look are registered before
  outcomes;
* the existing selected-shell projection consumer produces endpoint-only
  scalar rows and a joint scorer/projection closure;
* every row is replayed through :class:`V1FrozenLadderCertifier`;
* every endpoint is classified independently at every registered look; and
* one immutable, content-addressed look commit is published with no-replace
  semantics.

The store deliberately persists only endpoint projections and their numerical
enclosures.  It never stages a host ``[context, route]`` matrix.  Its decisions
have F/Q proposal-screening authority only: they cannot open the BCE oracle,
freeze or commit a mask, inspect terminal validation, or terminate a campaign.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Final, Literal, NoReturn, cast
import uuid

from mechanistic_systems.circuits.irg.modern_v1_ladder_projection_gpu import (
    V1FrozenActionProjectionRequest,
    V1LadderProjectionAcquisition,
    V1LadderProjectionGPUConsumer,
    projection_kernel_source_hash,
)
from mechanistic_systems.circuits.irg.modern_v1_ladder_statistics import (
    V1FQProposalBudget,
    V1FrozenLadderCertificationDesign,
    V1FrozenLadderCertificationReceipt,
    V1FrozenLadderCertifier,
    V1FrozenLadderDecision,
    V1FrozenLadderEndpointInterval,
    V1FrozenLadderProjectionRecord,
    select_largest_certified_endpoint,
    verify_frozen_ladder_certification_receipt,
)
from mechanistic_systems.circuits.irg.modern_v1_response_exactification import (
    V1ResponseExactificationDecisionScopeReceipt,
    V1ResponseExactificationArtifacts,
)
from mechanistic_systems.circuits.irg.modern_v1_composition_fq import (
    V1CompositionFQArtifacts,
)
from mechanistic_systems.circuits.irg.modern_v1_composition_scope import (
    LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND,
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
    V1CompositionDecisionScopeReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
    V1AlexNetFQDiscoveryPrefixFreezeReceipt,
)
from mechanistic_systems.contracts.content_address import content_hash


FQ_CONTEXT_ROLE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_fq_context_role_registration_v2"
)
FQ_REGISTERED_LOOK_ABI_SCHEMA: Final = (
    "partition_free_v1_fq_registered_look_abi_v1"
)
FQ_ACQUISITION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_selected_shell_fq_acquisition_registration_v2"
)
FQ_COMPOSITION_ACQUISITION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_composition_selected_shell_fq_acquisition_registration_v1"
)
FQ_ENDPOINT_CLASSIFICATION_SCHEMA: Final = (
    "partition_free_v1_fq_endpoint_classification_v1"
)
FQ_REGISTERED_LOOK_ASSESSMENT_SCHEMA: Final = (
    "partition_free_v1_fq_registered_look_assessment_v3"
)
FQ_REGISTERED_LOOK_COMMIT_SCHEMA: Final = (
    "partition_free_v1_selected_shell_fq_registered_look_commit_v3"
)

FQ_STORE_AUTHORITY_SCOPE: Final = (
    "fq_proposal_screening_only_no_oracle_mask_commit_or_terminal_authority_v1"
)
FQ_STORE_MEMORY_SCOPE: Final = (
    "selected_shell_endpoint_rows_only_no_host_context_route_matrix_v1"
)
FQ_STALE_PARTIAL_POLICY: Final = (
    "ignore_uncommitted_dot_partial_files_no_resume_or_authority_v1"
)

_ROLE_PREFIX: Final = "partition-free-v1-fq-context-role-"
_LOOK_ABI_PREFIX: Final = "partition-free-v1-fq-look-abi-"
_REGISTRATION_PREFIX: Final = "partition-free-v1-fq-acquisition-registration-"
_COMPOSITION_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-composition-fq-acquisition-registration-"
)
_LINEAGE_PREFIX: Final = "partition-free-v1-fq-acquisition-lineage-"
_CLASSIFICATION_PREFIX: Final = "partition-free-v1-fq-endpoint-classification-"
_ASSESSMENT_PREFIX: Final = "partition-free-v1-fq-look-assessment-v2-"
_COMMIT_PREFIX: Final = "partition-free-v1-fq-look-commit-v3-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-fq-context-membership-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-fq-draw-membership-"
_DISCOVERY_AXIS_PREFIX: Final = "partition-free-v1-discovery-context-membership-"
_REGISTRATION_FILE: Final = "registration.json"
_LOOK_DIRECTORY: Final = "looks"
_LOOK_PATTERN: Final = re.compile(r"^look_(\d{8})\.json$")
_SELECTED_SHELL_NUMERICAL_BACKENDS: Final = {
    "torch_cpu_selected_shell_fixed_tree_v2",
    "torch_cuda_selected_shell_fixed_tree_v2",
}

ContextSourceKind = Literal[
    "sealed_untouched_train",
    "preregistered_cross_fit",
    "causal_full_population_streams",
]
EndpointClassificationStatus = Literal[
    "certified_within_budgets",
    "certified_infeasible",
    "unresolved",
    "inconsistent_confidence_sequence",
]
LookAssessmentStatus = Literal[
    "certified_largest_endpoint",
    "continue_acquisition",
    "score_order_refinement_required",
    "certified_all_actions_infeasible",
    "indeterminate_upper_only_full_population_exhaustion",
    "indeterminate_full_population_without_separation",
    "indeterminate_registered_look_exhaustion",
    "refuse_inconsistent_confidence_sequence",
]


class V1FQStoreRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_SCHEMA = "invalid_schema"
    CONTEXT_ROLE_VIOLATION = "context_role_violation"
    REGISTRATION_MISMATCH = "registration_mismatch"
    ACQUISITION_MISMATCH = "acquisition_mismatch"
    OUTCOME_DUPLICATION = "outcome_duplication"
    LOOK_GAP = "look_gap"
    FORK_OR_REPLAY = "fork_or_replay"
    TAMPERING = "tampering"
    HORIZON_EXHAUSTED = "horizon_exhausted"


class V1FQStoreRefusal(ValueError):
    """Stable fail-closed error for the selected-shell F/Q store."""

    def __init__(self, code: V1FQStoreRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQStoreRefusalCode, message: str) -> NoReturn:
    raise V1FQStoreRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not cast(str, value):
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be one nonempty string",
        )
    return cast(str, value)


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or cast(int, value) < 0:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be one exact nonnegative integer",
        )
    return cast(int, value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be finite and nonnegative",
        )
    return result


def _string_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be an ordered string sequence",
        )
    return tuple(
        _identity(item, name=f"{name} item")
        for item in cast(Sequence[object], value)
    )


def _count_axis(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be an ordered integer sequence",
        )
    return tuple(
        _nonnegative_int(item, name=f"{name} item")
        for item in cast(Sequence[object], value)
    )


def _exact_mapping(
    value: object, *, keys: set[str], name: str
) -> dict[str, object]:
    if type(value) is not dict:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be one exact JSON object",
        )
    result = cast(dict[str, object], value)
    if set(result) != keys:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} fields are not exact",
        )
    # Validators may pop receipt fields from their private working object;
    # never mutate the caller's decoded mapping while doing so.
    return dict(result)


def _axis_pairs(value: object, *, name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FQStoreRefusalCode.INVALID_SCHEMA,
            f"{name} must be an ordered pair sequence",
        )
    result: list[tuple[str, tuple[str, ...]]] = []
    for row in cast(Sequence[object], value):
        if type(row) not in {tuple, list} or len(cast(Sequence[object], row)) != 2:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                f"{name} rows must be name/axis pairs",
            )
        pair = cast(Sequence[object], row)
        result.append(
            (
                _identity(pair[0], name=f"{name} name"),
                _string_axis(pair[1], name=f"{name} axis"),
            )
        )
    return tuple(result)


SelectedShellFQArtifacts = V1ResponseExactificationArtifacts | V1CompositionFQArtifacts


def _artifact_family(artifacts: object) -> Literal[
    "response_exactification",
    "composition",
]:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return "response_exactification"
    if type(artifacts) is V1CompositionFQArtifacts:
        return "composition"
    _refuse(
        V1FQStoreRefusalCode.INVALID_TYPE,
        "F/Q acquisition requires exact response or composition F/Q artifacts",
    )


def _registration_schema_for_artifacts(artifacts: SelectedShellFQArtifacts) -> str:
    return (
        FQ_ACQUISITION_REGISTRATION_SCHEMA
        if _artifact_family(artifacts) == "response_exactification"
        else FQ_COMPOSITION_ACQUISITION_REGISTRATION_SCHEMA
    )


def _registration_prefix_for_artifacts(artifacts: SelectedShellFQArtifacts) -> str:
    return (
        _REGISTRATION_PREFIX
        if _artifact_family(artifacts) == "response_exactification"
        else _COMPOSITION_REGISTRATION_PREFIX
    )


def _artifacts_hash(artifacts: SelectedShellFQArtifacts) -> str:
    return artifacts.assert_integrity()


def _artifacts_freeze_hash(artifacts: SelectedShellFQArtifacts) -> str:
    return artifacts.freeze.assert_integrity()


def _artifacts_ladder(artifacts: SelectedShellFQArtifacts):
    return artifacts.freeze.ladder


def _artifacts_certification_design(
    artifacts: SelectedShellFQArtifacts,
) -> V1FrozenLadderCertificationDesign:
    return artifacts.certification_design


def _artifacts_decision_scope(
    artifacts: SelectedShellFQArtifacts,
) -> V1ResponseExactificationDecisionScopeReceipt | V1CompositionDecisionScopeReceipt:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.replay_receipt.decision_scope_receipt
    return artifacts.decision_scope


def _artifacts_physical_route_axis_hash(artifacts: SelectedShellFQArtifacts) -> str:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.axis_bridge.physical_route_axis_hash
    return artifacts.physical_route_axis_hash


def _artifacts_ordered_route_axis_hash(artifacts: SelectedShellFQArtifacts) -> str:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.axis_bridge.ordered_route_axis_hash
    return artifacts.ordered_route_axis_hash


def _coordinates_match_budget(
    artifacts: SelectedShellFQArtifacts,
    budget: V1FQProposalBudget,
) -> bool:
    ladder = _artifacts_ladder(artifacts)
    return all(
        getattr(ladder, name) == getattr(budget, name)
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
        )
    )


@dataclass(frozen=True, slots=True)
class V1FQContextRoleRegistration:
    """Proof identity for legacy folds or causal full-population D/C streams."""

    source_kind: ContextSourceKind
    train_dataset_identity_hash: str
    train_split_identity_hash: str
    source_role_registration_hash: str
    certification_context_population_hash: str
    certification_context_membership_hash: str
    discovery_context_membership_hash: str
    cross_fit_registration_hash: str | None = None
    cross_fit_fold_assignment_hash: str | None = None
    discovery_fold_id: str | None = None
    certification_fold_id: str | None = None
    causal_generation_frame_receipt_hash: str | None = None
    discovery_stream_receipt_hash: str | None = None
    discovery_prefix_freeze_hash: str | None = None
    certification_stream_receipt_hash: str | None = None
    certification_rows_are_train: bool = True
    discovery_rows_excluded: bool = True
    incidental_discovery_certification_overlap_permitted: bool = False
    validation_rows_seen: bool = False
    registered_before_fq_outcomes: bool = True
    cross_fit_registered_before_discovery_outcomes: bool | None = None
    schema: str = FQ_CONTEXT_ROLE_REGISTRATION_SCHEMA
    _role_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.source_kind not in {
            "sealed_untouched_train",
            "preregistered_cross_fit",
            "causal_full_population_streams",
        }:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "unsupported F/Q train context source kind",
            )
        for name in (
            "train_dataset_identity_hash",
            "train_split_identity_hash",
            "source_role_registration_hash",
            "certification_context_population_hash",
            "certification_context_membership_hash",
            "discovery_context_membership_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        optional = (
            "cross_fit_registration_hash",
            "cross_fit_fold_assignment_hash",
            "discovery_fold_id",
            "certification_fold_id",
        )
        for name in optional:
            object.__setattr__(
                self, name, _optional_identity(getattr(self, name), name=name)
            )
        cross_values = tuple(getattr(self, name) for name in optional)
        causal_names = (
            "causal_generation_frame_receipt_hash",
            "discovery_stream_receipt_hash",
            "discovery_prefix_freeze_hash",
            "certification_stream_receipt_hash",
        )
        for name in causal_names:
            object.__setattr__(
                self, name, _optional_identity(getattr(self, name), name=name)
            )
        causal_values = tuple(getattr(self, name) for name in causal_names)
        if self.source_kind == "sealed_untouched_train":
            if (
                any(value is not None for value in (*cross_values, *causal_values))
                or self.incidental_discovery_certification_overlap_permitted is not False
                or self.discovery_rows_excluded is not True
                or (
                self.cross_fit_registered_before_discovery_outcomes is not None
                )
            ):
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "an untouched train role cannot claim cross-fit metadata",
                )
        elif self.source_kind == "preregistered_cross_fit":
            if any(value is None for value in cross_values):
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "cross-fit F/Q rows require a complete pre-registered fold proof",
                )
            if self.discovery_fold_id == self.certification_fold_id:
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "discovery and certification must use distinct cross-fit folds",
                )
            if self.cross_fit_registered_before_discovery_outcomes is not True:
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "cross-fit fold assignment must predate discovery outcomes",
                )
            if (
                any(value is not None for value in causal_values)
                or self.incidental_discovery_certification_overlap_permitted is not False
                or self.discovery_rows_excluded is not True
            ):
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "legacy cross-fit rows cannot claim causal overlap authority",
                )
        else:
            if (
                any(value is not None for value in cross_values)
                or any(value is None for value in causal_values)
                or self.cross_fit_registered_before_discovery_outcomes is not None
                or self.discovery_rows_excluded is not False
                or self.incidental_discovery_certification_overlap_permitted is not True
            ):
                _refuse(
                    V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                    "causal full-frame rows require the complete D-freeze/C-stream graph",
                )
        if (
            self.certification_rows_are_train is not True
            or self.validation_rows_seen is not False
            or self.registered_before_fq_outcomes is not True
            or self.schema != FQ_CONTEXT_ROLE_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "F/Q context role crossed its train-only evidence boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_kind": self.source_kind,
            "train_dataset_identity_hash": self.train_dataset_identity_hash,
            "train_split_identity_hash": self.train_split_identity_hash,
            "source_role_registration_hash": self.source_role_registration_hash,
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "certification_context_membership_hash": (
                self.certification_context_membership_hash
            ),
            "discovery_context_membership_hash": (
                self.discovery_context_membership_hash
            ),
            "cross_fit_registration_hash": self.cross_fit_registration_hash,
            "cross_fit_fold_assignment_hash": self.cross_fit_fold_assignment_hash,
            "discovery_fold_id": self.discovery_fold_id,
            "certification_fold_id": self.certification_fold_id,
            "causal_generation_frame_receipt_hash": (
                self.causal_generation_frame_receipt_hash
            ),
            "discovery_stream_receipt_hash": self.discovery_stream_receipt_hash,
            "discovery_prefix_freeze_hash": self.discovery_prefix_freeze_hash,
            "certification_stream_receipt_hash": (
                self.certification_stream_receipt_hash
            ),
            "certification_rows_are_train": self.certification_rows_are_train,
            "discovery_rows_excluded": self.discovery_rows_excluded,
            "incidental_discovery_certification_overlap_permitted": (
                self.incidental_discovery_certification_overlap_permitted
            ),
            "validation_rows_seen": self.validation_rows_seen,
            "registered_before_fq_outcomes": self.registered_before_fq_outcomes,
            "cross_fit_registered_before_discovery_outcomes": (
                self.cross_fit_registered_before_discovery_outcomes
            ),
        }

    @property
    def role_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ROLE_PREFIX)
        if self._role_hash is not None and observed != self._role_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q context role was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "F/Q context role hash does not match",
            )
        object.__setattr__(self, "_role_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "role_hash": self.role_hash}

    @classmethod
    def from_json_dict(cls, value: object) -> "V1FQContextRoleRegistration":
        keys = {
            "schema",
            "source_kind",
            "train_dataset_identity_hash",
            "train_split_identity_hash",
            "source_role_registration_hash",
            "certification_context_population_hash",
            "certification_context_membership_hash",
            "discovery_context_membership_hash",
            "cross_fit_registration_hash",
            "cross_fit_fold_assignment_hash",
            "discovery_fold_id",
            "certification_fold_id",
            "causal_generation_frame_receipt_hash",
            "discovery_stream_receipt_hash",
            "discovery_prefix_freeze_hash",
            "certification_stream_receipt_hash",
            "certification_rows_are_train",
            "discovery_rows_excluded",
            "incidental_discovery_certification_overlap_permitted",
            "validation_rows_seen",
            "registered_before_fq_outcomes",
            "cross_fit_registered_before_discovery_outcomes",
            "role_hash",
        }
        payload = _exact_mapping(value, keys=keys, name="F/Q context role")
        expected = _identity(payload.pop("role_hash"), name="context role hash")
        result = cls(**payload)  # type: ignore[arg-type]
        result.assert_integrity(expected)
        return result


def register_v1_fq_train_context_role(
    *,
    artifacts: SelectedShellFQArtifacts,
    source_kind: ContextSourceKind,
    train_dataset_identity_hash: str,
    train_split_identity_hash: str,
    source_role_registration_hash: str,
    cross_fit_registration_hash: str | None = None,
    cross_fit_fold_assignment_hash: str | None = None,
    discovery_fold_id: str | None = None,
    certification_fold_id: str | None = None,
    causal_generation_frame: V1AlexNetFQCausalGenerationFrameReceipt | None = None,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = None,
    discovery_prefix_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt | None = None,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = None,
) -> V1FQContextRoleRegistration:
    """Register train-only context provenance against an exact F/Q artifact graph."""

    _artifact_family(artifacts)
    _artifacts_hash(artifacts)
    design = _artifacts_certification_design(artifacts)
    contexts = design.certification_context_ids_by_stratum
    discovery = _artifacts_ladder(artifacts).discovery_context_ids
    flat = tuple(context for _, axis in contexts for context in axis)
    causal_receipts = (
        causal_generation_frame,
        discovery_stream,
        discovery_prefix_freeze,
        certification_stream,
    )
    causal_kind = source_kind == "causal_full_population_streams"
    if causal_kind:
        if (
            type(causal_generation_frame)
            is not V1AlexNetFQCausalGenerationFrameReceipt
            or type(discovery_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(discovery_prefix_freeze)
            is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
            or type(certification_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        ):
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "causal F/Q role requires exact D-freeze/C-stream receipts",
            )
        assert causal_generation_frame is not None
        assert discovery_stream is not None
        assert discovery_prefix_freeze is not None
        assert certification_stream is not None
        causal_generation_frame.assert_integrity()
        discovery_stream.assert_integrity()
        discovery_prefix_freeze.assert_integrity()
        certification_stream.assert_integrity()
        d_counts = discovery_prefix_freeze.cumulative_stratum_counts
        d_prefixes = (
            discovery_stream.target_assignment_receipt.generated_order[: d_counts[0]],
            discovery_stream.complement_assignment_receipt.generated_order[
                : d_counts[1]
            ],
        )
        if (
            discovery_stream.role != "discovery"
            or certification_stream.role != "certification"
            or discovery_stream.generation_frame_hash
            != causal_generation_frame.receipt_hash
            or discovery_prefix_freeze.generation_frame_hash
            != causal_generation_frame.receipt_hash
            or discovery_prefix_freeze.discovery_stream_hash
            != discovery_stream.receipt_hash
            or certification_stream.generation_frame_hash
            != causal_generation_frame.receipt_hash
            or certification_stream.discovery_prefix_freeze_hash
            != discovery_prefix_freeze.receipt_hash
            or certification_stream.selected_shell_hash
            != discovery_prefix_freeze.selected_shell_hash
            or discovery_prefix_freeze.selected_shell_hash
            != _artifacts_ladder(artifacts).assert_integrity()
            or set(discovery) != set((*d_prefixes[0], *d_prefixes[1]))
            or contexts != certification_stream.context_ids_by_stratum
        ):
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "causal F/Q role receipt graph differs from frozen D/C rows",
            )
    elif any(value is not None for value in causal_receipts):
        _refuse(
            V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
            "legacy F/Q roles cannot carry causal overlap receipts",
        )
    elif set(flat).intersection(discovery):
        _refuse(
            V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
            "discovery rows cannot become F/Q certification rows",
        )
    return V1FQContextRoleRegistration(
        source_kind=source_kind,
        train_dataset_identity_hash=train_dataset_identity_hash,
        train_split_identity_hash=train_split_identity_hash,
        source_role_registration_hash=source_role_registration_hash,
        certification_context_population_hash=(
            design.certification_context_population_hash
        ),
        certification_context_membership_hash=content_hash(
            contexts, prefix=_CONTEXT_AXIS_PREFIX
        ),
        discovery_context_membership_hash=content_hash(
            discovery, prefix=_DISCOVERY_AXIS_PREFIX
        ),
        cross_fit_registration_hash=cross_fit_registration_hash,
        cross_fit_fold_assignment_hash=cross_fit_fold_assignment_hash,
        discovery_fold_id=discovery_fold_id,
        certification_fold_id=certification_fold_id,
        causal_generation_frame_receipt_hash=(
            causal_generation_frame.receipt_hash if causal_kind else None
        ),
        discovery_stream_receipt_hash=(
            discovery_stream.receipt_hash if causal_kind else None
        ),
        discovery_prefix_freeze_hash=(
            discovery_prefix_freeze.receipt_hash if causal_kind else None
        ),
        certification_stream_receipt_hash=(
            certification_stream.receipt_hash if causal_kind else None
        ),
        discovery_rows_excluded=not causal_kind,
        incidental_discovery_certification_overlap_permitted=causal_kind,
        cross_fit_registered_before_discovery_outcomes=(
            True if source_kind == "preregistered_cross_fit" else None
        ),
    )


@dataclass(frozen=True, slots=True)
class V1FQRegisteredLookABI:
    """One exact pre-outcome selected-shell projection ABI."""

    look_index: int
    cumulative_stratum_counts: tuple[int, ...]
    raw_logit_stream_identity_hash: str
    destination_transport_hash: str
    scorer_request_identity_hash: str
    acquisition_lineage_seed_hash: str
    registered_route_chunk_size: int
    schema: str = FQ_REGISTERED_LOOK_ABI_SCHEMA
    _look_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "look_index", _positive_int(self.look_index, name="look index"))
        counts = _count_axis(
            self.cumulative_stratum_counts, name="cumulative stratum counts"
        )
        if not counts:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "registered F/Q look requires a stratum count axis",
            )
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        for name in (
            "raw_logit_stream_identity_hash",
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_seed_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_route_chunk_size",
            _positive_int(
                self.registered_route_chunk_size,
                name="registered route chunk size",
            ),
        )
        if self.schema != FQ_REGISTERED_LOOK_ABI_SCHEMA:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "F/Q look ABI schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "look_index": self.look_index,
            "cumulative_stratum_counts": self.cumulative_stratum_counts,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "destination_transport_hash": self.destination_transport_hash,
            "scorer_request_identity_hash": self.scorer_request_identity_hash,
            "acquisition_lineage_seed_hash": self.acquisition_lineage_seed_hash,
            "registered_route_chunk_size": self.registered_route_chunk_size,
        }

    @property
    def look_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_LOOK_ABI_PREFIX)
        if self._look_hash is not None and observed != self._look_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "registered F/Q look ABI was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "registered F/Q look ABI hash does not match",
            )
        object.__setattr__(self, "_look_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "look_hash": self.look_hash}

    @classmethod
    def from_json_dict(cls, value: object) -> "V1FQRegisteredLookABI":
        keys = {
            "schema",
            "look_index",
            "cumulative_stratum_counts",
            "raw_logit_stream_identity_hash",
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_seed_hash",
            "registered_route_chunk_size",
            "look_hash",
        }
        payload = _exact_mapping(value, keys=keys, name="registered F/Q look ABI")
        expected = _identity(payload.pop("look_hash"), name="look ABI hash")
        payload["cumulative_stratum_counts"] = _count_axis(
            payload["cumulative_stratum_counts"], name="cumulative stratum counts"
        )
        result = cls(**payload)  # type: ignore[arg-type]
        result.assert_integrity(expected)
        return result


@dataclass(frozen=True, slots=True)
class V1FQAcquisitionRegistration:
    """Complete pre-outcome selected-shell acquisition and look schedule."""

    artifacts: SelectedShellFQArtifacts = field(repr=False, compare=False)
    budget: V1FQProposalBudget
    context_role: V1FQContextRoleRegistration
    expected_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    registered_looks: tuple[V1FQRegisteredLookABI, ...]
    ordered_score_route_axis_hash: str
    issued_before_any_fq_outcomes: bool = True
    every_endpoint_registered_at_every_look: bool = True
    fq_monotonicity_assumed: bool = False
    host_context_route_matrix_allowed: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQ_STORE_AUTHORITY_SCOPE
    memory_scope: str = FQ_STORE_MEMORY_SCOPE
    schema: str = FQ_ACQUISITION_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        _artifact_family(self.artifacts)
        if (
            _artifact_family(self.artifacts) == "composition"
            and self.schema == FQ_ACQUISITION_REGISTRATION_SCHEMA
        ):
            object.__setattr__(
                self,
                "schema",
                FQ_COMPOSITION_ACQUISITION_REGISTRATION_SCHEMA,
            )
        if type(self.budget) is not V1FQProposalBudget:
            _refuse(
                V1FQStoreRefusalCode.INVALID_TYPE,
                "F/Q acquisition requires the typed proposal budget",
            )
        if type(self.context_role) is not V1FQContextRoleRegistration:
            _refuse(
                V1FQStoreRefusalCode.INVALID_TYPE,
                "F/Q acquisition requires the typed context role",
            )
        _artifacts_hash(self.artifacts)
        self.budget.assert_integrity()
        self.context_role.assert_integrity()
        object.__setattr__(
            self,
            "ordered_score_route_axis_hash",
            _identity(
                self.ordered_score_route_axis_hash,
                name="ordered score-route-axis hash",
            ),
        )
        if not _coordinates_match_budget(self.artifacts, self.budget):
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "F/Q budget coordinates differ from selected-shell artifacts",
            )
        design = _artifacts_certification_design(self.artifacts)
        ladder = _artifacts_ladder(self.artifacts)
        if (
            design.ladder.route_axis_hash
            != _artifacts_physical_route_axis_hash(self.artifacts)
            or self.ordered_score_route_axis_hash
            != _artifacts_ordered_route_axis_hash(self.artifacts)
        ):
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "F/Q registration physical mask and ordered score-route axes "
                "differ from selected-shell artifacts",
            )
        if design.projection_kernel_source_hash != projection_kernel_source_hash():
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "F/Q design names another selected-shell projection source",
            )
        contexts = design.certification_context_ids_by_stratum
        flat_contexts = tuple(context for _, axis in contexts for context in axis)
        discovery_overlap = set(flat_contexts).intersection(
            ladder.discovery_context_ids
        )
        causal_full_frame = (
            self.context_role.source_kind == "causal_full_population_streams"
        )
        causal_receipt_hashes = (
            self.context_role.causal_generation_frame_receipt_hash,
            self.context_role.discovery_stream_receipt_hash,
            self.context_role.discovery_prefix_freeze_hash,
            self.context_role.certification_stream_receipt_hash,
        )
        if discovery_overlap and not causal_full_frame:
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "discovery rows cannot enter the F/Q population",
            )
        if causal_full_frame and (
            any(value is None for value in causal_receipt_hashes)
            or self.context_role.discovery_rows_excluded is not False
            or self.context_role.incidental_discovery_certification_overlap_permitted
            is not True
            or design.certification_stream_commitment_hash
            != self.context_role.certification_stream_receipt_hash
        ):
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "causal full-frame F/Q registration lost its D-freeze/C-stream proof",
            )
        expected_context_membership = content_hash(contexts, prefix=_CONTEXT_AXIS_PREFIX)
        expected_discovery_membership = content_hash(
            ladder.discovery_context_ids,
            prefix=_DISCOVERY_AXIS_PREFIX,
        )
        if (
            self.context_role.certification_context_population_hash
            != design.certification_context_population_hash
            or self.context_role.certification_context_membership_hash
            != expected_context_membership
            or self.context_role.discovery_context_membership_hash
            != expected_discovery_membership
        ):
            _refuse(
                V1FQStoreRefusalCode.CONTEXT_ROLE_VIOLATION,
                "F/Q train role differs from the frozen certification population",
            )
        draws = _axis_pairs(
            self.expected_draw_ids_by_stratum, name="F/Q draw ledger"
        )
        if tuple(name for name, _ in draws) != design.stratum_names or any(
            len(draw_axis) != len(context_axis)
            for (_, draw_axis), (_, context_axis) in zip(draws, contexts, strict=True)
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "F/Q draw ledger differs from the frozen context strata",
            )
        flat_draws = tuple(draw for _, axis in draws for draw in axis)
        if len(flat_draws) != len(set(flat_draws)) or set(flat_draws).intersection(
            ladder.discovery_draw_ids
        ):
            _refuse(
                V1FQStoreRefusalCode.OUTCOME_DUPLICATION,
                "F/Q draws must be globally fresh from discovery",
            )
        looks = tuple(self.registered_looks)
        if not looks or any(type(item) is not V1FQRegisteredLookABI for item in looks):
            _refuse(
                V1FQStoreRefusalCode.INVALID_TYPE,
                "F/Q acquisition requires typed registered looks",
            )
        populations = design.population_sizes
        previous = tuple(0 for _ in populations)
        for position, look in enumerate(looks, start=1):
            look.assert_integrity()
            counts = look.cumulative_stratum_counts
            if look.look_index != position or len(counts) != len(populations):
                _refuse(
                    V1FQStoreRefusalCode.LOOK_GAP,
                    "F/Q registered looks must be contiguous on the stratum axis",
                )
            if (
                any(left < right for left, right in zip(counts, previous, strict=True))
                or any(left > limit for left, limit in zip(counts, populations, strict=True))
                or counts == previous
            ):
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    "F/Q look counts must advance bounded sealed prefixes",
                )
            previous = counts
        if (
            self.issued_before_any_fq_outcomes is not True
            or self.every_endpoint_registered_at_every_look is not True
            or self.fq_monotonicity_assumed is not False
            or self.host_context_route_matrix_allowed is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.authority_scope != FQ_STORE_AUTHORITY_SCOPE
            or self.memory_scope != FQ_STORE_MEMORY_SCOPE
            or self.schema != _registration_schema_for_artifacts(self.artifacts)
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "F/Q acquisition registration crossed its authority boundary",
            )
        object.__setattr__(self, "expected_draw_ids_by_stratum", draws)
        object.__setattr__(self, "registered_looks", looks)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        design = _artifacts_certification_design(self.artifacts)
        bundle = self.artifacts.endpoint_envelope_bundle
        if type(self.artifacts) is V1CompositionFQArtifacts:
            return {
                "schema": self.schema,
                "composition_artifacts_hash": _artifacts_hash(self.artifacts),
                "composition_freeze_hash": _artifacts_freeze_hash(self.artifacts),
                "composition_kind": self.artifacts.composition_kind,
                "composition_scope_receipt_hash": (
                    self.artifacts.decision_scope.assert_integrity()
                ),
                "ladder_hash": self.artifacts.ladder.assert_integrity(),
                "certification_design_hash": design.assert_integrity(),
                "physical_fq_envelope_certificate_hash": (
                    self.artifacts.physical_fq_envelope_certificate.assert_integrity()
                ),
                "endpoint_envelope_bundle_hash": bundle.assert_integrity(),
                "action_drive_envelope_certificate_hash": (
                    bundle.action_drive_certificate.assert_integrity()
                ),
                "route_sum_fallback_certificate_hash": (
                    bundle.route_sum_fallback_certificate.assert_integrity()
                ),
                "fq_alpha_allocation_receipt_hash": (
                    self.artifacts.fq_alpha_allocation.assert_integrity()
                ),
                "common_panel_receipt_hash": (
                    self.artifacts.common_panel_receipt.assert_integrity()
                ),
                "trajectory_alpha_ledger_hash": design.trajectory_alpha_ledger_hash,
                "fq_budget_hash": self.budget.assert_integrity(),
                "physical_mask_axis_hash": self.physical_mask_axis_hash,
                "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
                "context_role": self.context_role.to_json_dict(),
                "expected_draw_ids_by_stratum": self.expected_draw_ids_by_stratum,
                "draw_membership_hash": content_hash(
                    self.expected_draw_ids_by_stratum,
                    prefix=_DRAW_AXIS_PREFIX,
                ),
                "registered_looks": tuple(
                    item.to_json_dict() for item in self.registered_looks
                ),
                "issued_before_any_fq_outcomes": (
                    self.issued_before_any_fq_outcomes
                ),
                "every_endpoint_registered_at_every_look": (
                    self.every_endpoint_registered_at_every_look
                ),
                "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
                "host_context_route_matrix_allowed": (
                    self.host_context_route_matrix_allowed
                ),
                "bce_oracle_authority": self.bce_oracle_authority,
                "candidate_freeze_authority": self.candidate_freeze_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
                "memory_scope": self.memory_scope,
                "stale_partial_policy": FQ_STALE_PARTIAL_POLICY,
            }
        return {
            "schema": self.schema,
            "response_artifacts_hash": self.artifacts.assert_integrity(),
            "response_freeze_hash": self.artifacts.freeze.assert_integrity(),
            "ladder_hash": self.artifacts.freeze.ladder.assert_integrity(),
            "certification_design_hash": design.assert_integrity(),
            "physical_fq_envelope_certificate_hash": (
                self.artifacts.physical_fq_envelope_certificate.assert_integrity()
            ),
            "endpoint_envelope_bundle_hash": bundle.assert_integrity(),
            "action_drive_envelope_certificate_hash": (
                bundle.action_drive_certificate.assert_integrity()
            ),
            "route_sum_fallback_certificate_hash": (
                bundle.route_sum_fallback_certificate.assert_integrity()
            ),
            "fq_alpha_allocation_receipt_hash": (
                self.artifacts.fq_alpha_allocation.assert_integrity()
            ),
            "trajectory_alpha_ledger_hash": design.trajectory_alpha_ledger_hash,
            "fq_budget_hash": self.budget.assert_integrity(),
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "context_role": self.context_role.to_json_dict(),
            "expected_draw_ids_by_stratum": self.expected_draw_ids_by_stratum,
            "draw_membership_hash": content_hash(
                self.expected_draw_ids_by_stratum, prefix=_DRAW_AXIS_PREFIX
            ),
            "registered_looks": tuple(item.to_json_dict() for item in self.registered_looks),
            "issued_before_any_fq_outcomes": self.issued_before_any_fq_outcomes,
            "every_endpoint_registered_at_every_look": (
                self.every_endpoint_registered_at_every_look
            ),
            "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
            "host_context_route_matrix_allowed": self.host_context_route_matrix_allowed,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
            "memory_scope": self.memory_scope,
            "stale_partial_policy": FQ_STALE_PARTIAL_POLICY,
        }

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    @property
    def physical_mask_axis_hash(self) -> str:
        """Physical packed-mask coordinate registered for every look."""

        return _artifacts_physical_route_axis_hash(self.artifacts)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        _artifacts_hash(self.artifacts)
        self.budget.assert_integrity()
        self.context_role.assert_integrity()
        for look in self.registered_looks:
            look.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=_registration_prefix_for_artifacts(self.artifacts),
        )
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q acquisition registration was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "F/Q acquisition registration hash does not match",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "registration_hash": self.registration_hash}

    @classmethod
    def from_json_dict(
        cls,
        value: object,
        *,
        artifacts: SelectedShellFQArtifacts,
        budget: V1FQProposalBudget,
    ) -> "V1FQAcquisitionRegistration":
        if type(value) is not dict:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "F/Q acquisition registration must be one exact JSON object",
            )
        raw = cast(dict[str, object], value)
        schema = _identity(raw.get("schema"), name="F/Q acquisition schema")
        legacy_keys = {
            "schema",
            "response_artifacts_hash",
            "response_freeze_hash",
            "ladder_hash",
            "certification_design_hash",
            "physical_fq_envelope_certificate_hash",
            "endpoint_envelope_bundle_hash",
            "action_drive_envelope_certificate_hash",
            "route_sum_fallback_certificate_hash",
            "fq_alpha_allocation_receipt_hash",
            "trajectory_alpha_ledger_hash",
            "fq_budget_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "context_role",
            "expected_draw_ids_by_stratum",
            "draw_membership_hash",
            "registered_looks",
            "issued_before_any_fq_outcomes",
            "every_endpoint_registered_at_every_look",
            "fq_monotonicity_assumed",
            "host_context_route_matrix_allowed",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "terminal_authority",
            "authority_scope",
            "memory_scope",
            "stale_partial_policy",
            "registration_hash",
        }
        composition_keys = {
            "schema",
            "composition_artifacts_hash",
            "composition_freeze_hash",
            "composition_kind",
            "composition_scope_receipt_hash",
            "ladder_hash",
            "certification_design_hash",
            "physical_fq_envelope_certificate_hash",
            "endpoint_envelope_bundle_hash",
            "action_drive_envelope_certificate_hash",
            "route_sum_fallback_certificate_hash",
            "fq_alpha_allocation_receipt_hash",
            "common_panel_receipt_hash",
            "trajectory_alpha_ledger_hash",
            "fq_budget_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "context_role",
            "expected_draw_ids_by_stratum",
            "draw_membership_hash",
            "registered_looks",
            "issued_before_any_fq_outcomes",
            "every_endpoint_registered_at_every_look",
            "fq_monotonicity_assumed",
            "host_context_route_matrix_allowed",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "terminal_authority",
            "authority_scope",
            "memory_scope",
            "stale_partial_policy",
            "registration_hash",
        }
        if schema == FQ_ACQUISITION_REGISTRATION_SCHEMA:
            if type(artifacts) is not V1ResponseExactificationArtifacts:
                _refuse(
                    V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                    "legacy F/Q registration requires response artifacts",
                )
            keys = legacy_keys
        elif schema == FQ_COMPOSITION_ACQUISITION_REGISTRATION_SCHEMA:
            if type(artifacts) is not V1CompositionFQArtifacts:
                _refuse(
                    V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                    "composition F/Q registration requires composition artifacts",
                )
            keys = composition_keys
        else:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "unsupported F/Q acquisition registration schema",
            )
        payload = _exact_mapping(value, keys=keys, name="F/Q acquisition registration")
        expected = _identity(payload["registration_hash"], name="registration hash")
        context_role = V1FQContextRoleRegistration.from_json_dict(payload["context_role"])
        if type(payload["registered_looks"]) not in {tuple, list}:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "registered looks must be a sequence")
        looks = tuple(
            V1FQRegisteredLookABI.from_json_dict(item)
            for item in cast(Sequence[object], payload["registered_looks"])
        )
        draws = _axis_pairs(payload["expected_draw_ids_by_stratum"], name="F/Q draw ledger")
        result = cls(
            artifacts=artifacts,
            budget=budget,
            context_role=context_role,
            expected_draw_ids_by_stratum=draws,
            registered_looks=looks,
            ordered_score_route_axis_hash=cast(
                str, payload["ordered_score_route_axis_hash"]
            ),
            issued_before_any_fq_outcomes=cast(bool, payload["issued_before_any_fq_outcomes"]),
            every_endpoint_registered_at_every_look=cast(bool, payload["every_endpoint_registered_at_every_look"]),
            fq_monotonicity_assumed=cast(bool, payload["fq_monotonicity_assumed"]),
            host_context_route_matrix_allowed=cast(bool, payload["host_context_route_matrix_allowed"]),
            bce_oracle_authority=cast(bool, payload["bce_oracle_authority"]),
            candidate_freeze_authority=cast(bool, payload["candidate_freeze_authority"]),
            mask_commit_authority=cast(bool, payload["mask_commit_authority"]),
            terminal_authority=cast(bool, payload["terminal_authority"]),
            authority_scope=cast(str, payload["authority_scope"]),
            memory_scope=cast(str, payload["memory_scope"]),
            schema=cast(str, payload["schema"]),
        )
        # The content hash transitively checks every explicit lineage field in
        # the persisted payload against the supplied typed artifacts/budget.
        result.assert_integrity(expected)
        if _canonical_json_bytes(result.to_json_dict()) != _canonical_json_bytes(
            payload
        ):
            _refuse(
                V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
                "persisted F/Q registration differs from the supplied artifact graph",
            )
        return result

    def projection_request(
        self, *, look_index: int, prior_commit_hash: str | None
    ) -> V1FrozenActionProjectionRequest:
        """Reconstruct the exact pre-registered request for one chain position."""

        index = _positive_int(look_index, name="look index")
        if index > len(self.registered_looks):
            _refuse(V1FQStoreRefusalCode.HORIZON_EXHAUSTED, "no registered F/Q look remains")
        if index == 1:
            if prior_commit_hash is not None:
                _refuse(V1FQStoreRefusalCode.FORK_OR_REPLAY, "the first F/Q look cannot have a prior commit")
            starts = tuple(
                0
                for _ in _artifacts_certification_design(
                    self.artifacts
                ).stratum_names
            )
        else:
            _identity(prior_commit_hash, name="prior F/Q commit hash")
            starts = self.registered_looks[index - 2].cumulative_stratum_counts
        look = self.registered_looks[index - 1]
        design = _artifacts_certification_design(self.artifacts)
        context_map = dict(design.certification_context_ids_by_stratum)
        draw_map = dict(self.expected_draw_ids_by_stratum)
        contexts: list[str] = []
        draws: list[str] = []
        strata: list[str] = []
        for stratum_position, stratum in enumerate(design.stratum_names):
            start = starts[stratum_position]
            stop = look.cumulative_stratum_counts[stratum_position]
            selected_contexts = context_map[stratum][start:stop]
            selected_draws = draw_map[stratum][start:stop]
            contexts.extend(selected_contexts)
            draws.extend(selected_draws)
            strata.extend((stratum,) * len(selected_contexts))
        lineage = content_hash(
            {
                "registration_hash": self.registration_hash,
                "look_index": index,
                "prior_commit_hash": prior_commit_hash,
                "registered_lineage_seed_hash": look.acquisition_lineage_seed_hash,
            },
            prefix=_LINEAGE_PREFIX,
        )
        return V1FrozenActionProjectionRequest(
            design=design,
            expected_context_ids=tuple(contexts),
            expected_draw_ids=tuple(draws),
            expected_stratum_ids=tuple(strata),
            sealed_order_start_indices_by_stratum=tuple(
                zip(design.stratum_names, starts, strict=True)
            ),
            raw_logit_stream_identity_hash=look.raw_logit_stream_identity_hash,
            destination_transport_hash=look.destination_transport_hash,
            scorer_request_identity_hash=look.scorer_request_identity_hash,
            acquisition_lineage_hash=lineage,
            ordered_score_route_axis_hash=self.ordered_score_route_axis_hash,
            registered_route_chunk_size=look.registered_route_chunk_size,
        )


@dataclass(frozen=True, slots=True)
class V1FQEndpointClassification:
    endpoint_index: int
    action_hash: str
    f_lower: float
    f_upper: float
    q_lower: float
    q_upper: float
    lower_confidence_authority: bool
    upper_confidence_authority: bool
    status: EndpointClassificationStatus
    schema: str = FQ_ENDPOINT_CLASSIFICATION_SCHEMA
    _classification_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_index", _positive_int(self.endpoint_index, name="endpoint index"))
        object.__setattr__(self, "action_hash", _identity(self.action_hash, name="action hash"))
        for name in ("f_lower", "f_upper", "q_lower", "q_upper"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name=name))
        if type(self.lower_confidence_authority) is not bool or type(self.upper_confidence_authority) is not bool:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "confidence authority flags must be exact booleans")
        if self.status not in {
            "certified_within_budgets",
            "certified_infeasible",
            "unresolved",
            "inconsistent_confidence_sequence",
        } or self.schema != FQ_ENDPOINT_CLASSIFICATION_SCHEMA:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "endpoint classification is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "endpoint_index": self.endpoint_index,
            "action_hash": self.action_hash,
            "f_lower": self.f_lower,
            "f_upper": self.f_upper,
            "q_lower": self.q_lower,
            "q_upper": self.q_upper,
            "lower_confidence_authority": self.lower_confidence_authority,
            "upper_confidence_authority": self.upper_confidence_authority,
            "status": self.status,
        }

    @property
    def classification_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_CLASSIFICATION_PREFIX)
        if self._classification_hash is not None and observed != self._classification_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "endpoint classification was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "endpoint classification hash does not match")
        object.__setattr__(self, "_classification_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "classification_hash": self.classification_hash}


@dataclass(frozen=True, slots=True)
class V1FQRegisteredLookAssessment:
    receipt_hash: str
    decision_hash: str
    budget_hash: str
    decision_scope_receipt_hash: str
    classifications: tuple[V1FQEndpointClassification, ...]
    score_resolved_endpoint_indices: tuple[int, ...]
    provisional_endpoint_indices: tuple[int, ...]
    decision_competitive_endpoint_indices: tuple[int, ...]
    certified_endpoint_indices: tuple[int, ...]
    certified_action_hashes: tuple[str, ...]
    score_resolved_fq_certified_endpoint_indices: tuple[int, ...]
    score_resolved_fq_certified_action_hashes: tuple[str, ...]
    selection_blocking_endpoint_indices: tuple[int, ...]
    score_scope_blocking_endpoint_indices: tuple[int, ...]
    larger_fq_noninfeasible_endpoint_indices: tuple[int, ...]
    score_order_refinement_required: bool
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    full_population_look: bool
    final_registered_look: bool
    status: LookAssessmentStatus
    composition_kind: str = LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
    composition_frozen_endpoint_indices: tuple[int, ...] = ()
    composition_provisional_endpoint_indices: tuple[int, ...] = ()
    composition_fq_certified_endpoint_indices: tuple[int, ...] = ()
    composition_fq_certified_action_hashes: tuple[str, ...] = ()
    composition_scope_blocking_endpoint_indices: tuple[int, ...] = ()
    every_endpoint_classified: bool = True
    fq_monotonicity_assumed: bool = False
    smaller_certified_endpoints_retained_for_oracle_backtracking: bool = True
    decision_scope_selection_rule_enforced: bool = True
    selected_endpoint_score_resolution_required: bool = True
    every_larger_endpoint_certified_infeasible_required: bool = True
    larger_unresolved_endpoints_block_selection: bool = True
    smaller_unresolved_endpoints_block_selection: bool = False
    smaller_score_provisional_endpoints_oracle_ineligible: bool = True
    paired_bce_oracle_required_before_mask_commit: bool = True
    validation_outcomes_observed: bool = False
    terminal_validation_access_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQ_REGISTERED_LOOK_ASSESSMENT_SCHEMA
    _assessment_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "receipt_hash",
            "decision_hash",
            "budget_hash",
            "decision_scope_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rows = tuple(self.classifications)
        if not rows or any(type(row) is not V1FQEndpointClassification for row in rows):
            _refuse(V1FQStoreRefusalCode.INVALID_TYPE, "look assessment requires typed endpoint classifications")
        for row in rows:
            row.assert_integrity()
        if tuple(row.endpoint_index for row in rows) != tuple(range(1, len(rows) + 1)):
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "every endpoint must be classified exactly once")
        endpoint_count = len(rows)
        registered_indices = tuple(range(1, endpoint_count + 1))
        kind = _identity(self.composition_kind, name="composition kind")
        if kind not in {
            LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND,
            SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
        }:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "unsupported F/Q composition scope kind",
            )
        resolved = tuple(
            _positive_int(value, name="score-resolved endpoint index")
            for value in self.score_resolved_endpoint_indices
        )
        provisional = tuple(
            _positive_int(value, name="provisional endpoint index")
            for value in self.provisional_endpoint_indices
        )
        competitive = tuple(
            _positive_int(value, name="decision-competitive endpoint index")
            for value in self.decision_competitive_endpoint_indices
        )
        for name, axis in (
            ("score-resolved endpoint axis", resolved),
            ("provisional endpoint axis", provisional),
            ("decision-competitive endpoint axis", competitive),
        ):
            if tuple(sorted(axis)) != axis or len(set(axis)) != len(axis):
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    f"{name} must be strictly increasing and unique",
                )
            if axis and axis[-1] > endpoint_count:
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    f"{name} left the frozen ladder",
                )
        default_legacy_scope = (
            kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
            and not self.composition_frozen_endpoint_indices
            and not self.composition_provisional_endpoint_indices
        )
        composition_frozen = (
            resolved
            if default_legacy_scope
            else tuple(
                _positive_int(value, name="composition-frozen endpoint index")
                for value in self.composition_frozen_endpoint_indices
            )
        )
        composition_provisional = (
            provisional
            if default_legacy_scope
            else tuple(
                _positive_int(value, name="composition-provisional endpoint index")
                for value in self.composition_provisional_endpoint_indices
            )
        )
        for name, axis in (
            ("composition-frozen endpoint axis", composition_frozen),
            ("composition-provisional endpoint axis", composition_provisional),
        ):
            if tuple(sorted(axis)) != axis or len(set(axis)) != len(axis):
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    f"{name} must be strictly increasing and unique",
                )
            if axis and axis[-1] > endpoint_count:
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    f"{name} left the frozen ladder",
                )
        if (
            tuple(sorted((*composition_frozen, *composition_provisional)))
            != registered_indices
            or set(composition_frozen).intersection(composition_provisional)
            or not set(competitive).issubset(composition_frozen)
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "composition axes do not partition the frozen ladder",
            )
        if kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND:
            if (
                tuple(sorted((*resolved, *provisional))) != registered_indices
                or set(resolved).intersection(provisional)
                or composition_frozen != resolved
                or composition_provisional != provisional
                or not set(competitive).issubset(resolved)
            ):
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    "legacy score-resolution axes do not partition the frozen ladder",
                )
        elif (
            resolved
            or self.score_resolved_fq_certified_endpoint_indices
            or self.score_resolved_fq_certified_action_hashes
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "sampled geometry assessment cannot carry score-resolved ledgers",
            )
        fq_certified = tuple(
            row for row in rows if row.status == "certified_within_budgets"
        )
        composition_fq_certified = tuple(
            row
            for row in fq_certified
            if row.endpoint_index in set(competitive)
        )
        inconsistent = any(
            row.status == "inconsistent_confidence_sequence" for row in rows
        )
        every_infeasible = all(
            row.status == "certified_infeasible" for row in rows
        )
        every_lower_authoritative = all(
            row.lower_confidence_authority for row in rows
        )
        indices = tuple(_positive_int(value, name="certified endpoint index") for value in self.certified_endpoint_indices)
        hashes = tuple(_identity(value, name="certified action hash") for value in self.certified_action_hashes)
        resolved_fq_indices = tuple(
            _positive_int(value, name="score-resolved F/Q-certified endpoint index")
            for value in self.score_resolved_fq_certified_endpoint_indices
        )
        resolved_fq_hashes = tuple(
            _identity(value, name="score-resolved F/Q-certified action hash")
            for value in self.score_resolved_fq_certified_action_hashes
        )
        if indices != tuple(row.endpoint_index for row in fq_certified) or hashes != tuple(row.action_hash for row in fq_certified):
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "certified endpoint backtracking ledger is incomplete")
        observed_composition_fq_indices = (
            tuple(row.endpoint_index for row in composition_fq_certified)
            if not self.composition_fq_certified_endpoint_indices
            else tuple(
                _positive_int(
                    value,
                    name="composition F/Q-certified endpoint index",
                )
                for value in self.composition_fq_certified_endpoint_indices
            )
        )
        observed_composition_fq_hashes = (
            tuple(row.action_hash for row in composition_fq_certified)
            if not self.composition_fq_certified_action_hashes
            else tuple(
                _identity(value, name="composition F/Q-certified action hash")
                for value in self.composition_fq_certified_action_hashes
            )
        )
        if (
            observed_composition_fq_indices
            != tuple(row.endpoint_index for row in composition_fq_certified)
            or observed_composition_fq_hashes
            != tuple(row.action_hash for row in composition_fq_certified)
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "composition-scoped F/Q-certified queue ledger is incomplete",
            )
        if (
            kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
            and (
                resolved_fq_indices != observed_composition_fq_indices
                or resolved_fq_hashes != observed_composition_fq_hashes
            )
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "score-resolved F/Q-certified queue ledger is incomplete",
            )

        score_blockers: set[int] = set()
        fq_blockers: set[int] = set()
        selected: V1FQEndpointClassification | None = None
        if fq_certified:
            candidate = fq_certified[-1]
            candidate_index = candidate.endpoint_index
            if candidate_index not in set(competitive):
                score_blockers.add(candidate_index)
            larger = rows[candidate_index:]
            for row in larger:
                if row.endpoint_index not in set(composition_frozen):
                    score_blockers.add(row.endpoint_index)
                if row.status != "certified_infeasible":
                    fq_blockers.add(row.endpoint_index)
            if all(
                row.endpoint_index in set(composition_frozen)
                and row.status == "certified_infeasible"
                for row in larger
            ) and candidate_index in set(competitive):
                selected = candidate
        if self.full_population_look and every_infeasible:
            score_blockers.update(composition_provisional)
        expected_score_blockers = tuple(sorted(score_blockers))
        expected_fq_blockers = tuple(sorted(fq_blockers))
        expected_blockers = tuple(sorted(score_blockers.union(fq_blockers)))
        observed_blockers = tuple(
            _positive_int(value, name="selection blocker endpoint index")
            for value in self.selection_blocking_endpoint_indices
        )
        observed_score_blockers = tuple(
            _positive_int(value, name="score-scope blocker endpoint index")
            for value in self.score_scope_blocking_endpoint_indices
        )
        observed_composition_blockers = (
            expected_score_blockers
            if not self.composition_scope_blocking_endpoint_indices
            else tuple(
                _positive_int(
                    value,
                    name="composition-scope blocker endpoint index",
                )
                for value in self.composition_scope_blocking_endpoint_indices
            )
        )
        observed_fq_blockers = tuple(
            _positive_int(value, name="larger F/Q-noninfeasible endpoint index")
            for value in self.larger_fq_noninfeasible_endpoint_indices
        )
        if (
            observed_blockers != expected_blockers
            or observed_score_blockers
            != (
                expected_score_blockers
                if kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
                else ()
            )
            or observed_composition_blockers != expected_score_blockers
            or observed_fq_blockers != expected_fq_blockers
        ):
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "decision-scope blocker receipt differs from endpoint evidence",
            )
        if selected is not None:
            if (
                self.status != "certified_largest_endpoint"
                or self.selected_endpoint_index != selected.endpoint_index
                or self.selected_action_hash != selected.action_hash
            ):
                _refuse(
                    V1FQStoreRefusalCode.INVALID_SCHEMA,
                    "look assessment violated the decision-scoped suffix rule",
                )
        elif self.selected_endpoint_index is not None or self.selected_action_hash is not None:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "non-selection assessment cannot name an endpoint")
        if self.status not in {
            "certified_largest_endpoint",
            "continue_acquisition",
            "score_order_refinement_required",
            "certified_all_actions_infeasible",
            "indeterminate_upper_only_full_population_exhaustion",
            "indeterminate_full_population_without_separation",
            "indeterminate_registered_look_exhaustion",
            "refuse_inconsistent_confidence_sequence",
        }:
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "look assessment status is invalid")
        if type(self.full_population_look) is not bool or type(
            self.final_registered_look
        ) is not bool:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "look exhaustion flags must be exact booleans",
            )
        expected_nonselection = (
            "refuse_inconsistent_confidence_sequence"
            if inconsistent
            else (
                "score_order_refinement_required"
                if (
                    (bool(fq_certified) and bool(observed_score_blockers))
                    or (
                        self.full_population_look
                        and every_infeasible
                        and bool(composition_provisional)
                    )
                )
                else (
                    "certified_all_actions_infeasible"
                    if self.full_population_look and every_infeasible
                    else (
                        "indeterminate_upper_only_full_population_exhaustion"
                        if self.full_population_look and not every_lower_authoritative
                        else (
                            "indeterminate_full_population_without_separation"
                            if self.full_population_look
                            else (
                                "indeterminate_registered_look_exhaustion"
                                if self.final_registered_look
                                else "continue_acquisition"
                            )
                        )
                    )
                )
            )
        )
        if selected is None and self.status != expected_nonselection:
            _refuse(
                V1FQStoreRefusalCode.INVALID_SCHEMA,
                "look assessment status differs from its endpoint evidence",
            )
        if (
            self.every_endpoint_classified is not True
            or self.score_order_refinement_required
            is not (self.status == "score_order_refinement_required")
            or self.fq_monotonicity_assumed is not False
            or self.smaller_certified_endpoints_retained_for_oracle_backtracking is not True
            or self.decision_scope_selection_rule_enforced is not True
            or self.selected_endpoint_score_resolution_required
            is not (kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND)
            or self.every_larger_endpoint_certified_infeasible_required is not True
            or self.larger_unresolved_endpoints_block_selection is not True
            or self.smaller_unresolved_endpoints_block_selection is not False
            or self.smaller_score_provisional_endpoints_oracle_ineligible
            is not (kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND)
            or self.paired_bce_oracle_required_before_mask_commit is not True
            or self.validation_outcomes_observed is not False
            or self.terminal_validation_access_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.schema != FQ_REGISTERED_LOOK_ASSESSMENT_SCHEMA
        ):
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "look assessment crossed its authority boundary")
        object.__setattr__(self, "classifications", rows)
        object.__setattr__(self, "composition_kind", kind)
        object.__setattr__(self, "score_resolved_endpoint_indices", resolved)
        object.__setattr__(self, "provisional_endpoint_indices", provisional)
        object.__setattr__(
            self, "composition_frozen_endpoint_indices", composition_frozen
        )
        object.__setattr__(
            self,
            "composition_provisional_endpoint_indices",
            composition_provisional,
        )
        object.__setattr__(self, "decision_competitive_endpoint_indices", competitive)
        object.__setattr__(self, "certified_endpoint_indices", indices)
        object.__setattr__(self, "certified_action_hashes", hashes)
        object.__setattr__(
            self,
            "score_resolved_fq_certified_endpoint_indices",
            resolved_fq_indices,
        )
        object.__setattr__(
            self,
            "score_resolved_fq_certified_action_hashes",
            resolved_fq_hashes,
        )
        object.__setattr__(
            self,
            "composition_fq_certified_endpoint_indices",
            observed_composition_fq_indices,
        )
        object.__setattr__(
            self,
            "composition_fq_certified_action_hashes",
            observed_composition_fq_hashes,
        )
        object.__setattr__(self, "selection_blocking_endpoint_indices", observed_blockers)
        object.__setattr__(
            self, "score_scope_blocking_endpoint_indices", observed_score_blockers
        )
        object.__setattr__(
            self,
            "composition_scope_blocking_endpoint_indices",
            observed_composition_blockers,
        )
        object.__setattr__(
            self,
            "larger_fq_noninfeasible_endpoint_indices",
            observed_fq_blockers,
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_hash": self.receipt_hash,
            "decision_hash": self.decision_hash,
            "budget_hash": self.budget_hash,
            "decision_scope_receipt_hash": self.decision_scope_receipt_hash,
            "classifications": tuple(row.to_json_dict() for row in self.classifications),
            "score_resolved_endpoint_indices": self.score_resolved_endpoint_indices,
            "provisional_endpoint_indices": self.provisional_endpoint_indices,
            "decision_competitive_endpoint_indices": (
                self.decision_competitive_endpoint_indices
            ),
            "certified_endpoint_indices": self.certified_endpoint_indices,
            "certified_action_hashes": self.certified_action_hashes,
            "score_resolved_fq_certified_endpoint_indices": (
                self.score_resolved_fq_certified_endpoint_indices
            ),
            "score_resolved_fq_certified_action_hashes": (
                self.score_resolved_fq_certified_action_hashes
            ),
            "selection_blocking_endpoint_indices": (
                self.selection_blocking_endpoint_indices
            ),
            "score_scope_blocking_endpoint_indices": (
                self.score_scope_blocking_endpoint_indices
            ),
            "larger_fq_noninfeasible_endpoint_indices": (
                self.larger_fq_noninfeasible_endpoint_indices
            ),
            "score_order_refinement_required": (
                self.score_order_refinement_required
            ),
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "full_population_look": self.full_population_look,
            "final_registered_look": self.final_registered_look,
            "status": self.status,
            "composition_kind": self.composition_kind,
            "composition_frozen_endpoint_indices": (
                self.composition_frozen_endpoint_indices
            ),
            "composition_provisional_endpoint_indices": (
                self.composition_provisional_endpoint_indices
            ),
            "composition_fq_certified_endpoint_indices": (
                self.composition_fq_certified_endpoint_indices
            ),
            "composition_fq_certified_action_hashes": (
                self.composition_fq_certified_action_hashes
            ),
            "composition_scope_blocking_endpoint_indices": (
                self.composition_scope_blocking_endpoint_indices
            ),
            "every_endpoint_classified": self.every_endpoint_classified,
            "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
            "smaller_certified_endpoints_retained_for_oracle_backtracking": (
                self.smaller_certified_endpoints_retained_for_oracle_backtracking
            ),
            "decision_scope_selection_rule_enforced": (
                self.decision_scope_selection_rule_enforced
            ),
            "selected_endpoint_score_resolution_required": (
                self.selected_endpoint_score_resolution_required
            ),
            "every_larger_endpoint_certified_infeasible_required": (
                self.every_larger_endpoint_certified_infeasible_required
            ),
            "larger_unresolved_endpoints_block_selection": (
                self.larger_unresolved_endpoints_block_selection
            ),
            "smaller_unresolved_endpoints_block_selection": (
                self.smaller_unresolved_endpoints_block_selection
            ),
            "smaller_score_provisional_endpoints_oracle_ineligible": (
                self.smaller_score_provisional_endpoints_oracle_ineligible
            ),
            "paired_bce_oracle_required_before_mask_commit": (
                self.paired_bce_oracle_required_before_mask_commit
            ),
            "validation_outcomes_observed": self.validation_outcomes_observed,
            "terminal_validation_access_authority": (
                self.terminal_validation_access_authority
            ),
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def assessment_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for row in self.classifications:
            row.assert_integrity()
        observed = content_hash(self._payload(), prefix=_ASSESSMENT_PREFIX)
        if self._assessment_hash is not None and observed != self._assessment_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look assessment was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look assessment hash does not match")
        object.__setattr__(self, "_assessment_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "assessment_hash": self.assessment_hash}


def _classify_endpoint(
    endpoint: V1FrozenLadderEndpointInterval,
    *,
    budget: V1FQProposalBudget,
) -> V1FQEndpointClassification:
    if endpoint.empty:
        status: EndpointClassificationStatus = "inconsistent_confidence_sequence"
    elif (
        endpoint.upper_confidence_authority
        and endpoint.f_upper <= budget.f_upper_budget
        and endpoint.q_upper <= budget.q_upper_budget
    ):
        status = "certified_within_budgets"
    elif endpoint.lower_confidence_authority and (
        endpoint.f_lower > budget.f_upper_budget
        or endpoint.q_lower > budget.q_upper_budget
    ):
        status = "certified_infeasible"
    else:
        status = "unresolved"
    return V1FQEndpointClassification(
        endpoint_index=endpoint.endpoint_index,
        action_hash=endpoint.action_hash,
        f_lower=endpoint.f_lower,
        f_upper=endpoint.f_upper,
        q_lower=endpoint.q_lower,
        q_upper=endpoint.q_upper,
        lower_confidence_authority=endpoint.lower_confidence_authority,
        upper_confidence_authority=endpoint.upper_confidence_authority,
        status=status,
    )


def _assess_look(
    *,
    receipt: V1FrozenLadderCertificationReceipt,
    decision: V1FrozenLadderDecision,
    budget: V1FQProposalBudget,
    final_registered_look: bool,
    decision_scope_receipt: V1ResponseExactificationDecisionScopeReceipt
    | V1CompositionDecisionScopeReceipt
    | None,
) -> V1FQRegisteredLookAssessment:
    rows = tuple(_classify_endpoint(item, budget=budget) for item in receipt.endpoint_intervals)
    scope = decision_scope_receipt
    if scope is None:
        # Compatibility for the public pure assessment seam.  Production
        # store commits always pass the hash-bound response-exactification
        # scope carried by their registered artifacts.
        endpoint_count = len(rows)
        scope = V1ResponseExactificationDecisionScopeReceipt(
            score_order_registration_hash=receipt.design_hash,
            score_order_look_receipt_hash=receipt.receipt_hash,
            registered_endpoint_new_removal_counts=tuple(
                range(1, endpoint_count + 1)
            ),
            score_resolved_endpoint_indices=tuple(range(1, endpoint_count + 1)),
            provisional_endpoint_indices=(),
            decision_competitive_endpoint_indices=tuple(
                range(1, endpoint_count + 1)
            ),
            registered_comparator_endpoint_index=None,
            registered_comparator_rung_new_removal_count=None,
        )
    if type(scope) not in {
        V1ResponseExactificationDecisionScopeReceipt,
        V1CompositionDecisionScopeReceipt,
    }:
        _refuse(
            V1FQStoreRefusalCode.INVALID_TYPE,
            "registered-look assessment requires the typed composition decision scope",
        )
    scope.assert_integrity()
    if len(scope.registered_endpoint_new_removal_counts) != len(rows):
        _refuse(
            V1FQStoreRefusalCode.REGISTRATION_MISMATCH,
            "composition decision scope differs from the F/Q endpoint axis",
        )
    if type(scope) is V1ResponseExactificationDecisionScopeReceipt:
        composition_kind = LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
        score_resolved_indices = scope.score_resolved_endpoint_indices
        provisional_indices = scope.provisional_endpoint_indices
        composition_frozen_indices = scope.score_resolved_endpoint_indices
        composition_provisional_indices = scope.provisional_endpoint_indices
        selected_endpoint_score_resolution_required = True
        smaller_score_provisional_endpoints_oracle_ineligible = True
    else:
        composition_kind = scope.composition_kind
        if composition_kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND:
            score_resolved_indices = scope.legacy_score_resolved_endpoint_indices
            provisional_indices = scope.composition_provisional_endpoint_indices
            selected_endpoint_score_resolution_required = True
            smaller_score_provisional_endpoints_oracle_ineligible = True
        else:
            score_resolved_indices = ()
            provisional_indices = ()
            selected_endpoint_score_resolution_required = False
            smaller_score_provisional_endpoints_oracle_ineligible = False
        composition_frozen_indices = scope.composition_frozen_endpoint_indices
        composition_provisional_indices = (
            scope.composition_provisional_endpoint_indices
        )
    certified = tuple(row for row in rows if row.status == "certified_within_budgets")
    resolved_set = set(composition_frozen_indices)
    competitive_set = set(scope.decision_competitive_endpoint_indices)
    composition_certified = tuple(
        row for row in certified if row.endpoint_index in competitive_set
    )
    full = receipt.population_census_complete
    selected = None
    score_blockers: set[int] = set()
    fq_blockers: set[int] = set()
    if certified:
        candidate = certified[-1]
        if candidate.endpoint_index not in competitive_set:
            score_blockers.add(candidate.endpoint_index)
        larger = rows[candidate.endpoint_index:]
        for row in larger:
            if row.endpoint_index not in resolved_set:
                score_blockers.add(row.endpoint_index)
            if row.status != "certified_infeasible":
                fq_blockers.add(row.endpoint_index)
        if (
            candidate.endpoint_index in competitive_set
            and all(
                row.endpoint_index in resolved_set
                and row.status == "certified_infeasible"
                for row in larger
            )
        ):
            selected = candidate
    if full and all(row.status == "certified_infeasible" for row in rows):
        score_blockers.update(composition_provisional_indices)
    if any(row.status == "inconsistent_confidence_sequence" for row in rows):
        status: LookAssessmentStatus = "refuse_inconsistent_confidence_sequence"
        selected = None
    elif selected is not None:
        status = "certified_largest_endpoint"
    elif certified and score_blockers:
        status = "score_order_refinement_required"
    elif (
        full
        and all(row.status == "certified_infeasible" for row in rows)
        and not composition_provisional_indices
    ):
        status = "certified_all_actions_infeasible"
    elif (
        full
        and all(row.status == "certified_infeasible" for row in rows)
        and composition_provisional_indices
    ):
        status = "score_order_refinement_required"
    elif full and not receipt.lower_confidence_authority:
        status = "indeterminate_upper_only_full_population_exhaustion"
    elif full:
        status = "indeterminate_full_population_without_separation"
    elif final_registered_look:
        status = "indeterminate_registered_look_exhaustion"
    else:
        status = "continue_acquisition"
    fq_selected = certified[-1] if certified else None
    if fq_selected is not None and (
        decision.selected_endpoint_index != fq_selected.endpoint_index
        or decision.selected_action_hash != fq_selected.action_hash
    ):
        _refuse(
            V1FQStoreRefusalCode.TAMPERING,
            "canonical certifier and independent endpoint classification disagree",
        )
    return V1FQRegisteredLookAssessment(
        receipt_hash=receipt.receipt_hash,
        decision_hash=decision.decision_hash,
        budget_hash=budget.budget_hash,
        decision_scope_receipt_hash=scope.receipt_hash,
        classifications=rows,
        score_resolved_endpoint_indices=score_resolved_indices,
        provisional_endpoint_indices=provisional_indices,
        decision_competitive_endpoint_indices=(
            scope.decision_competitive_endpoint_indices
        ),
        certified_endpoint_indices=tuple(row.endpoint_index for row in certified),
        certified_action_hashes=tuple(row.action_hash for row in certified),
        score_resolved_fq_certified_endpoint_indices=tuple(
            row.endpoint_index
            for row in composition_certified
            if row.endpoint_index in set(score_resolved_indices)
        ),
        score_resolved_fq_certified_action_hashes=tuple(
            row.action_hash
            for row in composition_certified
            if row.endpoint_index in set(score_resolved_indices)
        ),
        composition_kind=composition_kind,
        composition_frozen_endpoint_indices=composition_frozen_indices,
        composition_provisional_endpoint_indices=composition_provisional_indices,
        composition_fq_certified_endpoint_indices=tuple(
            row.endpoint_index for row in composition_certified
        ),
        composition_fq_certified_action_hashes=tuple(
            row.action_hash for row in composition_certified
        ),
        selection_blocking_endpoint_indices=tuple(
            sorted(score_blockers.union(fq_blockers))
        ),
        score_scope_blocking_endpoint_indices=(
            tuple(sorted(score_blockers))
            if composition_kind == LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND
            else ()
        ),
        composition_scope_blocking_endpoint_indices=tuple(sorted(score_blockers)),
        larger_fq_noninfeasible_endpoint_indices=tuple(sorted(fq_blockers)),
        score_order_refinement_required=(
            status == "score_order_refinement_required"
        ),
        selected_endpoint_index=None if selected is None else selected.endpoint_index,
        selected_action_hash=None if selected is None else selected.action_hash,
        full_population_look=full,
        final_registered_look=final_registered_look,
        status=status,
        selected_endpoint_score_resolution_required=(
            selected_endpoint_score_resolution_required
        ),
        smaller_score_provisional_endpoints_oracle_ineligible=(
            smaller_score_provisional_endpoints_oracle_ineligible
        ),
    )


def assess_v1_fq_registered_look(
    *,
    receipt: V1FrozenLadderCertificationReceipt,
    decision: V1FrozenLadderDecision,
    budget: V1FQProposalBudget,
    final_registered_look: bool,
    decision_scope_receipt: V1ResponseExactificationDecisionScopeReceipt
    | V1CompositionDecisionScopeReceipt
    | None = None,
) -> V1FQRegisteredLookAssessment:
    """Public pure assessment seam for one simultaneous F/Q receipt.

    Endpoints are classified independently; in particular, this function does
    not turn the ladder into a monotone prefix assumption.
    """

    if (
        type(receipt) is not V1FrozenLadderCertificationReceipt
        or type(decision) is not V1FrozenLadderDecision
        or type(budget) is not V1FQProposalBudget
        or type(final_registered_look) is not bool
    ):
        _refuse(
            V1FQStoreRefusalCode.INVALID_TYPE,
            "registered-look assessment requires exact certification types",
        )
    return _assess_look(
        receipt=receipt,
        decision=decision,
        budget=budget,
        final_registered_look=final_registered_look,
        decision_scope_receipt=decision_scope_receipt,
    )


def _projection_record_payload(record: V1FrozenLadderProjectionRecord) -> dict[str, object]:
    return {
        "schema": record.schema,
        "design_hash": record.design_hash,
        "ladder_hash": record.ladder_hash,
        "stratum_id": record.stratum_id,
        "context_id": record.context_id,
        "draw_id": record.draw_id,
        "projection_proof_hash": record.projection_proof_hash,
        "endpoint_action_hashes": record.endpoint_action_hashes,
        "signed_projection_by_endpoint": record.signed_projection_by_endpoint,
        "numerical_radius_by_endpoint": record.numerical_radius_by_endpoint,
        "record_hash": record.assert_integrity(),
    }


def _projection_record_from_payload(value: object) -> V1FrozenLadderProjectionRecord:
    keys = {
        "schema",
        "design_hash",
        "ladder_hash",
        "stratum_id",
        "context_id",
        "draw_id",
        "projection_proof_hash",
        "endpoint_action_hashes",
        "signed_projection_by_endpoint",
        "numerical_radius_by_endpoint",
        "record_hash",
    }
    payload = _exact_mapping(value, keys=keys, name="F/Q projection record")
    expected = _identity(payload.pop("record_hash"), name="projection record hash")
    record = V1FrozenLadderProjectionRecord(
        design_hash=cast(str, payload["design_hash"]),
        ladder_hash=cast(str, payload["ladder_hash"]),
        stratum_id=cast(str, payload["stratum_id"]),
        context_id=cast(str, payload["context_id"]),
        draw_id=cast(str, payload["draw_id"]),
        projection_proof_hash=cast(str, payload["projection_proof_hash"]),
        endpoint_action_hashes=_string_axis(payload["endpoint_action_hashes"], name="endpoint action hashes"),
        signed_projection_by_endpoint=tuple(float(item) for item in cast(Sequence[object], payload["signed_projection_by_endpoint"])),
        numerical_radius_by_endpoint=tuple(float(item) for item in cast(Sequence[object], payload["numerical_radius_by_endpoint"])),
        schema=cast(str, payload["schema"]),
    )
    try:
        record.assert_integrity(expected)
    except (TypeError, ValueError) as error:
        _refuse(V1FQStoreRefusalCode.TAMPERING, f"projection record failed integrity: {error}")
    return record


def _replay_records(
    registration: V1FQAcquisitionRegistration,
    records: Sequence[V1FrozenLadderProjectionRecord],
) -> V1FrozenLadderCertifier:
    certifier = V1FrozenLadderCertifier(
        _artifacts_certification_design(registration.artifacts)
    )
    for source in records:
        try:
            replayed = certifier.record(
                stratum_id=source.stratum_id,
                context_id=source.context_id,
                draw_id=source.draw_id,
                signed_projection_by_action=dict(
                    zip(source.endpoint_action_hashes, source.signed_projection_by_endpoint, strict=True)
                ),
                projection_proof_hash=source.projection_proof_hash,
                numerical_radius_by_action=(
                    dict(
                        zip(source.endpoint_action_hashes, source.numerical_radius_by_endpoint, strict=True)
                    )
                    if source.has_numerical_enclosure
                    else None
                ),
            )
        except (TypeError, ValueError) as error:
            _refuse(V1FQStoreRefusalCode.OUTCOME_DUPLICATION, f"F/Q scalar replay refused: {error}")
        if replayed.record_hash != source.record_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q scalar row does not replay exactly")
    return certifier


@dataclass(frozen=True, slots=True)
class V1FQRegisteredLookCommit:
    registration_hash: str
    prior_commit_hash: str | None
    look_index: int
    cumulative_stratum_counts: tuple[int, ...]
    projection_request_hash: str
    projection_csr_hash: str
    projection_receipt_hash: str
    acquisition_closure_hash: str
    scorer_receipt_hash: str
    scorer_request_identity_hash: str
    acquisition_lineage_hash: str
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    signed_projection_matrix_hash: str
    absolute_input_sum_upper_matrix_hash: str
    numerical_radius_matrix_hash: str
    projection_records: tuple[V1FrozenLadderProjectionRecord, ...] = field(repr=False)
    certification_receipt: V1FrozenLadderCertificationReceipt = field(repr=False, compare=False)
    decision: V1FrozenLadderDecision
    assessment: V1FQRegisteredLookAssessment
    endpoint_only_host_materialization: bool = True
    host_context_route_matrix_materialized: bool = False
    fq_proposal_screening_authority: bool = True
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQ_STORE_AUTHORITY_SCOPE
    memory_scope: str = FQ_STORE_MEMORY_SCOPE
    schema: str = FQ_REGISTERED_LOOK_COMMIT_SCHEMA
    _commit_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "projection_request_hash",
            "projection_csr_hash",
            "projection_receipt_hash",
            "acquisition_closure_hash",
            "scorer_receipt_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "signed_projection_matrix_hash",
            "absolute_input_sum_upper_matrix_hash",
            "numerical_radius_matrix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "prior_commit_hash", _optional_identity(self.prior_commit_hash, name="prior commit hash"))
        object.__setattr__(self, "look_index", _positive_int(self.look_index, name="look index"))
        counts = _count_axis(self.cumulative_stratum_counts, name="cumulative stratum counts")
        records = tuple(self.projection_records)
        if not records or any(type(item) is not V1FrozenLadderProjectionRecord for item in records):
            _refuse(V1FQStoreRefusalCode.INVALID_TYPE, "F/Q look commit requires typed projection rows")
        for item in records:
            item.assert_integrity()
        if type(self.certification_receipt) is not V1FrozenLadderCertificationReceipt or type(self.decision) is not V1FrozenLadderDecision or type(self.assessment) is not V1FQRegisteredLookAssessment:
            _refuse(V1FQStoreRefusalCode.INVALID_TYPE, "F/Q look commit requires typed certification artifacts")
        self.certification_receipt.assert_integrity()
        self.decision.assert_integrity()
        self.assessment.assert_integrity()
        observed_counts = tuple(
            len(contexts)
            for _stratum, contexts in (
                self.certification_receipt.observed_context_ids_by_stratum
            )
        )
        if (
            self.decision.receipt_hash != self.certification_receipt.receipt_hash
            or self.assessment.receipt_hash != self.certification_receipt.receipt_hash
            or self.assessment.decision_hash != self.decision.decision_hash
            or observed_counts != counts
        ):
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look certification graph does not close")
        if (
            self.endpoint_only_host_materialization is not True
            or self.host_context_route_matrix_materialized is not False
            or self.fq_proposal_screening_authority is not True
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.authority_scope != FQ_STORE_AUTHORITY_SCOPE
            or self.memory_scope != FQ_STORE_MEMORY_SCOPE
            or self.schema != FQ_REGISTERED_LOOK_COMMIT_SCHEMA
        ):
            _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "F/Q look commit crossed its authority boundary")
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        object.__setattr__(self, "projection_records", records)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "prior_commit_hash": self.prior_commit_hash,
            "look_index": self.look_index,
            "cumulative_stratum_counts": self.cumulative_stratum_counts,
            "projection_request_hash": self.projection_request_hash,
            "projection_csr_hash": self.projection_csr_hash,
            "projection_receipt_hash": self.projection_receipt_hash,
            "acquisition_closure_hash": self.acquisition_closure_hash,
            "scorer_receipt_hash": self.scorer_receipt_hash,
            "scorer_request_identity_hash": self.scorer_request_identity_hash,
            "acquisition_lineage_hash": self.acquisition_lineage_hash,
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "signed_projection_matrix_hash": self.signed_projection_matrix_hash,
            "absolute_input_sum_upper_matrix_hash": self.absolute_input_sum_upper_matrix_hash,
            "numerical_radius_matrix_hash": self.numerical_radius_matrix_hash,
            "projection_record_hashes": tuple(item.assert_integrity() for item in self.projection_records),
            "certification_receipt_hash": self.certification_receipt.assert_integrity(),
            "decision_hash": self.decision.assert_integrity(),
            "assessment": self.assessment.to_json_dict(),
            "endpoint_only_host_materialization": self.endpoint_only_host_materialization,
            "host_context_route_matrix_materialized": self.host_context_route_matrix_materialized,
            "fq_proposal_screening_authority": self.fq_proposal_screening_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
            "memory_scope": self.memory_scope,
        }

    @property
    def commit_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.certification_receipt.assert_integrity()
        self.decision.assert_integrity()
        self.assessment.assert_integrity()
        for item in self.projection_records:
            item.assert_integrity()
        observed = content_hash(self._payload(), prefix=_COMMIT_PREFIX)
        if self._commit_hash is not None and observed != self._commit_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look commit was tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look commit hash does not match")
        object.__setattr__(self, "_commit_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "projection_records": tuple(_projection_record_payload(item) for item in self.projection_records),
            "commit_hash": self.commit_hash,
        }


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, f"artifact is not canonical JSON: {error}")


def _decode_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _refuse(V1FQStoreRefusalCode.TAMPERING, f"JSON repeats field {key!r}")
        result[key] = value
    return result


def _read_canonical_json(path: Path) -> dict[str, object]:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        _refuse(V1FQStoreRefusalCode.LOOK_GAP, f"cannot open {path.name}: {error}")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            _refuse(
                V1FQStoreRefusalCode.TAMPERING,
                f"{path.name} is not a regular file",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_decode_pairs,
            parse_constant=lambda token: _refuse(
                V1FQStoreRefusalCode.TAMPERING,
                f"nonfinite JSON token {token!r} is forbidden",
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _refuse(V1FQStoreRefusalCode.TAMPERING, f"{path.name} JSON is invalid: {error}")
    if type(value) is not dict or raw != _canonical_json_bytes(cast(dict[str, object], value)):
        _refuse(V1FQStoreRefusalCode.TAMPERING, f"{path.name} is not canonical JSON")
    return cast(dict[str, object], value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_noreplace(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".partial-{path.name}-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _refuse(
                V1FQStoreRefusalCode.FORK_OR_REPLAY,
                f"{path.name} already exists; replay/fork publication refused",
            )
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _assessment_from_json(
    value: object,
    *,
    expected: V1FQRegisteredLookAssessment,
) -> V1FQRegisteredLookAssessment:
    if type(value) is not dict or _canonical_json_bytes(
        cast(dict[str, object], value)
    ) != _canonical_json_bytes(expected.to_json_dict()):
        _refuse(V1FQStoreRefusalCode.TAMPERING, "persisted F/Q assessment does not replay")
    return expected


def _commit_from_json(
    value: object,
    *,
    registration: V1FQAcquisitionRegistration,
    expected_request: V1FrozenActionProjectionRequest,
    prior_records: tuple[V1FrozenLadderProjectionRecord, ...],
    expected_prior_hash: str | None,
    expected_look_index: int,
) -> V1FQRegisteredLookCommit:
    keys = {
        "schema",
        "registration_hash",
        "prior_commit_hash",
        "look_index",
        "cumulative_stratum_counts",
        "projection_request_hash",
        "projection_csr_hash",
        "projection_receipt_hash",
        "acquisition_closure_hash",
        "scorer_receipt_hash",
        "scorer_request_identity_hash",
        "acquisition_lineage_hash",
        "physical_mask_axis_hash",
        "ordered_score_route_axis_hash",
        "signed_projection_matrix_hash",
        "absolute_input_sum_upper_matrix_hash",
        "numerical_radius_matrix_hash",
        "projection_record_hashes",
        "projection_records",
        "certification_receipt_hash",
        "decision_hash",
        "assessment",
        "endpoint_only_host_materialization",
        "host_context_route_matrix_materialized",
        "fq_proposal_screening_authority",
        "bce_oracle_authority",
        "candidate_freeze_authority",
        "mask_commit_authority",
        "terminal_authority",
        "authority_scope",
        "memory_scope",
        "commit_hash",
    }
    payload = _exact_mapping(value, keys=keys, name="F/Q registered-look commit")
    look_index = _positive_int(payload["look_index"], name="look index")
    expected_index = _positive_int(
        expected_look_index, name="expected look index"
    )
    cumulative_counts = _count_axis(
        payload["cumulative_stratum_counts"], name="cumulative counts"
    )
    if (
        look_index != expected_index
        or expected_index > len(registration.registered_looks)
        or cumulative_counts
        != registration.registered_looks[
            expected_index - 1
        ].cumulative_stratum_counts
    ):
        _refuse(
            V1FQStoreRefusalCode.FORK_OR_REPLAY,
            "persisted F/Q look index/counts left the registered chain position",
        )
    if type(payload["projection_records"]) not in {tuple, list}:
        _refuse(V1FQStoreRefusalCode.INVALID_SCHEMA, "projection records must be a sequence")
    new_records = tuple(
        _projection_record_from_payload(item)
        for item in cast(Sequence[object], payload["projection_records"])
    )
    record_axes = tuple(
        (record.context_id, record.draw_id, record.stratum_id)
        for record in new_records
    )
    expected_record_axes = tuple(
        zip(
            expected_request.expected_context_ids,
            expected_request.expected_draw_ids,
            expected_request.expected_stratum_ids,
            strict=True,
        )
    )
    persisted_closure_hash = _identity(
        payload["acquisition_closure_hash"],
        name="acquisition closure hash",
    )
    if (
        len(new_records) != expected_request.context_count
        or record_axes != expected_record_axes
        or any(
            record.projection_proof_hash != persisted_closure_hash
            for record in new_records
        )
    ):
        _refuse(
            V1FQStoreRefusalCode.FORK_OR_REPLAY,
            "persisted projection rows left their registered look/acquisition",
        )
    certifier = _replay_records(registration, (*prior_records, *new_records))
    receipt = certifier.receipt()
    try:
        design = _artifacts_certification_design(registration.artifacts)
        verify_frozen_ladder_certification_receipt(
            design=design,
            receipt=receipt,
            expected_receipt_hash=cast(str, payload["certification_receipt_hash"]),
        )
        decision = select_largest_certified_endpoint(
            design=design,
            receipt=receipt,
            budget=registration.budget,
        )
    except (TypeError, ValueError) as error:
        _refuse(V1FQStoreRefusalCode.TAMPERING, f"F/Q certification replay failed: {error}")
    if decision.decision_hash != payload["decision_hash"]:
        _refuse(V1FQStoreRefusalCode.TAMPERING, "persisted F/Q decision does not replay")
    assessment = _assess_look(
        receipt=receipt,
        decision=decision,
        budget=registration.budget,
        final_registered_look=look_index == len(registration.registered_looks),
        decision_scope_receipt=_artifacts_decision_scope(registration.artifacts),
    )
    _assessment_from_json(payload["assessment"], expected=assessment)
    result = V1FQRegisteredLookCommit(
        registration_hash=cast(str, payload["registration_hash"]),
        prior_commit_hash=cast(str | None, payload["prior_commit_hash"]),
        look_index=look_index,
        cumulative_stratum_counts=cumulative_counts,
        projection_request_hash=cast(str, payload["projection_request_hash"]),
        projection_csr_hash=cast(str, payload["projection_csr_hash"]),
        projection_receipt_hash=cast(str, payload["projection_receipt_hash"]),
        acquisition_closure_hash=cast(str, payload["acquisition_closure_hash"]),
        scorer_receipt_hash=cast(str, payload["scorer_receipt_hash"]),
        scorer_request_identity_hash=cast(str, payload["scorer_request_identity_hash"]),
        acquisition_lineage_hash=cast(str, payload["acquisition_lineage_hash"]),
        physical_mask_axis_hash=cast(str, payload["physical_mask_axis_hash"]),
        ordered_score_route_axis_hash=cast(
            str, payload["ordered_score_route_axis_hash"]
        ),
        signed_projection_matrix_hash=cast(str, payload["signed_projection_matrix_hash"]),
        absolute_input_sum_upper_matrix_hash=cast(str, payload["absolute_input_sum_upper_matrix_hash"]),
        numerical_radius_matrix_hash=cast(str, payload["numerical_radius_matrix_hash"]),
        projection_records=new_records,
        certification_receipt=receipt,
        decision=decision,
        assessment=assessment,
        endpoint_only_host_materialization=cast(bool, payload["endpoint_only_host_materialization"]),
        host_context_route_matrix_materialized=cast(bool, payload["host_context_route_matrix_materialized"]),
        fq_proposal_screening_authority=cast(bool, payload["fq_proposal_screening_authority"]),
        bce_oracle_authority=cast(bool, payload["bce_oracle_authority"]),
        candidate_freeze_authority=cast(bool, payload["candidate_freeze_authority"]),
        mask_commit_authority=cast(bool, payload["mask_commit_authority"]),
        terminal_authority=cast(bool, payload["terminal_authority"]),
        authority_scope=cast(str, payload["authority_scope"]),
        memory_scope=cast(str, payload["memory_scope"]),
        schema=cast(str, payload["schema"]),
    )
    if (
        result.registration_hash != registration.registration_hash
        or result.prior_commit_hash != expected_prior_hash
        or result.projection_request_hash != expected_request.request_hash
        or result.projection_csr_hash != expected_request.csr.csr_hash
        or result.scorer_request_identity_hash != expected_request.scorer_request_identity_hash
        or result.acquisition_lineage_hash != expected_request.acquisition_lineage_hash
        or result.physical_mask_axis_hash != expected_request.physical_mask_axis_hash
        or result.ordered_score_route_axis_hash
        != expected_request.ordered_score_route_axis_hash
        or tuple(item.record_hash for item in new_records)
        != tuple(_string_axis(payload["projection_record_hashes"], name="projection record hashes"))
    ):
        _refuse(V1FQStoreRefusalCode.FORK_OR_REPLAY, "F/Q look commit left its registered chain")
    result.assert_integrity(_identity(payload["commit_hash"], name="commit hash"))
    if _canonical_json_bytes(result.to_json_dict()) != _canonical_json_bytes(payload):
        _refuse(V1FQStoreRefusalCode.TAMPERING, "persisted F/Q look differs from canonical replay")
    return result


class V1ResponseExactificationStore:
    """Durable owner for one response-exactified selected-shell F/Q run."""

    def __init__(
        self,
        *,
        root: Path,
        registration: V1FQAcquisitionRegistration,
        commits: tuple[V1FQRegisteredLookCommit, ...],
    ) -> None:
        self.root = root
        self.registration = registration
        self._commits = commits
        self._records = tuple(
            record for commit in commits for record in commit.projection_records
        )

    @classmethod
    def create(
        cls,
        root: str | os.PathLike[str],
        *,
        registration: V1FQAcquisitionRegistration,
    ) -> "V1ResponseExactificationStore":
        if type(registration) is not V1FQAcquisitionRegistration:
            _refuse(V1FQStoreRefusalCode.INVALID_TYPE, "store creation requires the exact F/Q registration")
        registration.assert_integrity()
        path = Path(root)
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            _refuse(
                V1FQStoreRefusalCode.TAMPERING,
                "F/Q store root must be a real directory",
            )
        path.mkdir(parents=True, exist_ok=True)
        look_directory = path / _LOOK_DIRECTORY
        try:
            look_metadata = os.lstat(look_directory)
        except FileNotFoundError:
            look_directory.mkdir()
            look_metadata = os.lstat(look_directory)
        if stat.S_ISLNK(look_metadata.st_mode) or not stat.S_ISDIR(
            look_metadata.st_mode
        ):
            _refuse(
                V1FQStoreRefusalCode.TAMPERING,
                "F/Q look namespace must be a real directory",
            )
        allowed = {_LOOK_DIRECTORY}
        unexpected = tuple(item.name for item in path.iterdir() if item.name not in allowed and not item.name.startswith(".partial-"))
        if unexpected:
            _refuse(V1FQStoreRefusalCode.FORK_OR_REPLAY, "F/Q store root is not fresh")
        _atomic_publish_noreplace(
            path / _REGISTRATION_FILE,
            _canonical_json_bytes(registration.to_json_dict()),
        )
        _fsync_directory(path)
        return cls(root=path, registration=registration, commits=())

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        artifacts: SelectedShellFQArtifacts,
        budget: V1FQProposalBudget,
    ) -> "V1ResponseExactificationStore":
        path = Path(root)
        if path.is_symlink() or not path.is_dir():
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q store root is not a real directory")
        registration = V1FQAcquisitionRegistration.from_json_dict(
            _read_canonical_json(path / _REGISTRATION_FILE),
            artifacts=artifacts,
            budget=budget,
        )
        unexpected_root = tuple(
            item.name
            for item in path.iterdir()
            if item.name not in {_REGISTRATION_FILE, _LOOK_DIRECTORY}
            and not item.name.startswith(".partial-")
        )
        if unexpected_root:
            _refuse(
                V1FQStoreRefusalCode.FORK_OR_REPLAY,
                "unexpected file in the F/Q store root",
            )
        look_directory = path / _LOOK_DIRECTORY
        if look_directory.is_symlink() or not look_directory.is_dir():
            _refuse(V1FQStoreRefusalCode.TAMPERING, "F/Q look namespace is invalid")
        indexed: list[tuple[int, Path]] = []
        for item in look_directory.iterdir():
            if item.name.startswith(".partial-"):
                continue
            match = _LOOK_PATTERN.fullmatch(item.name)
            if match is None:
                _refuse(V1FQStoreRefusalCode.FORK_OR_REPLAY, "unexpected file in F/Q look namespace")
            indexed.append((int(match.group(1)), item))
        indexed.sort()
        if tuple(index for index, _ in indexed) != tuple(range(1, len(indexed) + 1)):
            _refuse(V1FQStoreRefusalCode.LOOK_GAP, "persisted F/Q look chain contains a gap")
        commits: list[V1FQRegisteredLookCommit] = []
        records: tuple[V1FrozenLadderProjectionRecord, ...] = ()
        prior_hash: str | None = None
        for index, item in indexed:
            request = registration.projection_request(
                look_index=index, prior_commit_hash=prior_hash
            )
            commit = _commit_from_json(
                _read_canonical_json(item),
                registration=registration,
                expected_request=request,
                prior_records=records,
                expected_prior_hash=prior_hash,
                expected_look_index=index,
            )
            commits.append(commit)
            records = (*records, *commit.projection_records)
            prior_hash = commit.commit_hash
        return cls(root=path, registration=registration, commits=tuple(commits))

    @property
    def commits(self) -> tuple[V1FQRegisteredLookCommit, ...]:
        return self._commits

    @property
    def next_look_index(self) -> int:
        return len(self._commits) + 1

    @property
    def prior_commit_hash(self) -> str | None:
        return None if not self._commits else self._commits[-1].commit_hash

    @property
    def horizon_exhausted(self) -> bool:
        return self.next_look_index > len(self.registration.registered_looks)

    def next_projection_request(self) -> V1FrozenActionProjectionRequest:
        return self.registration.projection_request(
            look_index=self.next_look_index,
            prior_commit_hash=self.prior_commit_hash,
        )

    def prepare_next_projection(
        self,
        *,
        route_ids: Sequence[str],
        device: Any = "cpu",
    ) -> tuple[V1FrozenActionProjectionRequest, V1LadderProjectionGPUConsumer]:
        request = self.next_projection_request()
        consumer = V1LadderProjectionGPUConsumer(
            request, route_ids=route_ids, device=device
        )
        consumer.bind_route_chunk_size(request.registered_route_chunk_size)
        return request, consumer

    def execute_next_look(
        self,
        *,
        route_ids: Sequence[str],
        execute: Callable[
            [V1FrozenActionProjectionRequest, V1LadderProjectionGPUConsumer],
            Mapping[str, object],
        ],
        device: Any = "cpu",
    ) -> V1FQRegisteredLookCommit:
        """Execute one provider transaction and publish only after joint closure."""

        request, consumer = self.prepare_next_projection(
            route_ids=route_ids, device=device
        )
        scorer_receipt = execute(request, consumer)
        acquisition = consumer.close_acquisition(scorer_receipt)
        return self.commit_acquisition(acquisition)

    def commit_acquisition(
        self, acquisition: V1LadderProjectionAcquisition
    ) -> V1FQRegisteredLookCommit:
        """Validate, certify, and atomically append the next registered look."""

        if type(acquisition) is not V1LadderProjectionAcquisition:
            _refuse(V1FQStoreRefusalCode.INVALID_TYPE, "F/Q commit requires the exact projection acquisition")
        request = self.next_projection_request()
        result = acquisition.result
        receipt = result.receipt
        closure = acquisition.closure
        try:
            closure.assert_integrity()
            receipt.assert_integrity()
        except (TypeError, ValueError) as error:
            _refuse(V1FQStoreRefusalCode.ACQUISITION_MISMATCH, f"projection acquisition failed integrity: {error}")
        expected_axes = (
            request.expected_context_ids,
            request.expected_draw_ids,
            request.expected_stratum_ids,
        )
        observed_axes = (receipt.context_ids, receipt.draw_ids, receipt.stratum_ids)
        endpoint_shape = (request.context_count, request.endpoint_count)
        expected_closure_axes = (
            content_hash(
                request.expected_context_ids,
                prefix="irg-device-panel-contexts-",
            ),
            content_hash(
                request.expected_draw_ids,
                prefix="partition-free-v1-projection-draw-axis-",
            ),
            content_hash(
                request.expected_stratum_ids,
                prefix="partition-free-v1-projection-stratum-axis-",
            ),
        )
        observed_closure_axes = (
            closure.context_axis_hash,
            closure.draw_axis_hash,
            closure.stratum_axis_hash,
        )
        if (
            receipt.request_hash != request.request_hash
            or closure.request_hash != request.request_hash
            or receipt.design_hash != request.design_hash
            or closure.design_hash != request.design_hash
            or receipt.ladder_hash != request.ladder_hash
            or closure.ladder_hash != request.ladder_hash
            or receipt.csr_hash != request.csr.csr_hash
            or receipt.current_mask_hash != request.current_mask_hash
            or receipt.route_count != request.route_count
            or request.csr.physical_mask_axis_hash
            != request.physical_mask_axis_hash
            or receipt.physical_mask_axis_hash
            != request.physical_mask_axis_hash
            or receipt.ordered_score_route_axis_hash
            != request.ordered_score_route_axis_hash
            or closure.physical_mask_axis_hash
            != request.physical_mask_axis_hash
            or closure.ordered_score_route_axis_hash
            != request.ordered_score_route_axis_hash
            or receipt.raw_logit_stream_source_hash
            != request.design.ladder.raw_logit_stream_source_hash
            or receipt.raw_logit_stream_identity_hash
            != request.raw_logit_stream_identity_hash
            or receipt.destination_transport_hash
            != request.destination_transport_hash
            or receipt.acquisition_lineage_hash != request.acquisition_lineage_hash
            or closure.acquisition_lineage_hash != request.acquisition_lineage_hash
            or receipt.scorer_request_identity_hash != request.scorer_request_identity_hash
            or closure.scorer_request_identity_hash != request.scorer_request_identity_hash
            or observed_axes != expected_axes
            or observed_closure_axes != expected_closure_axes
            or result.signed_projection_matrix.shape != endpoint_shape
            or result.numerical_radius_matrix.shape != endpoint_shape
            or closure.projection_receipt_hash != receipt.receipt_hash
            or closure.signed_projection_matrix_hash
            != receipt.signed_projection_matrix_hash
            or closure.absolute_input_sum_upper_matrix_hash
            != receipt.absolute_input_sum_upper_matrix_hash
            or closure.numerical_radius_matrix_hash
            != receipt.numerical_radius_matrix_hash
            or closure.numerical_lower_matrix_hash
            != receipt.numerical_lower_matrix_hash
            or closure.numerical_upper_matrix_hash
            != receipt.numerical_upper_matrix_hash
            or closure.scorer_receipt_schema
            != "partition_free_v1_raw_target_logit_stream_receipt_v1"
        ):
            _refuse(V1FQStoreRefusalCode.ACQUISITION_MISMATCH, "projection acquisition differs from the next registered look")
        expected_host_bytes = request.context_count * request.endpoint_count * 16
        packet_host_bytes = tuple(
            (packet.context_stop - packet.context_start)
            * request.endpoint_count
            * 16
            for packet in result.packet_receipts
        )
        expected_device_to_host = (
            expected_host_bytes
            if receipt.numerical_backend
            == "torch_cuda_selected_shell_fixed_tree_v2"
            else 0
        )
        if (
            receipt.projection_kernel_source_hash != projection_kernel_source_hash()
            or receipt.numerical_backend not in _SELECTED_SHELL_NUMERICAL_BACKENDS
            or any(
                packet.numerical_backend != receipt.numerical_backend
                for packet in result.packet_receipts
            )
            or any(
                packet.physical_mask_axis_hash
                != request.physical_mask_axis_hash
                or packet.ordered_score_route_axis_hash
                != request.ordered_score_route_axis_hash
                for packet in result.packet_receipts
            )
            or tuple(
                packet.projection_host_materialization_bytes
                for packet in result.packet_receipts
            )
            != packet_host_bytes
            or tuple(
                packet.device_to_host_bytes for packet in result.packet_receipts
            )
            != tuple(
                value
                if receipt.numerical_backend
                == "torch_cuda_selected_shell_fixed_tree_v2"
                else 0
                for value in packet_host_bytes
            )
            or receipt.total_projection_host_materialization_bytes
            != expected_host_bytes
            or receipt.exact_device_to_host_bytes != expected_device_to_host
            or receipt.packet_count != len(result.packet_receipts)
            or receipt.route_chunk_update_count
            != sum(len(packet.route_windows) for packet in result.packet_receipts)
            or any(
                right - left > request.registered_route_chunk_size
                for packet in result.packet_receipts
                for left, right in packet.route_windows
            )
            or receipt.estimator_rows_exactly_projected is not True
            or receipt.real_arithmetic_exactness_claimed is not False
            or receipt.numerical_enclosure_provided is not True
            or receipt.scalar_projection_derivation_authority is not True
            or receipt.fq_certification_authority is not False
            or receipt.score_order_authority is not False
            or receipt.bce_oracle_authority is not False
            or receipt.mask_commit_authority is not False
            or closure.estimator_rows_exactly_projected is not True
            or closure.numerical_enclosure_bound is not True
            or closure.certifier_record_proof_authority is not True
            or closure.fq_certification_authority is not False
            or closure.score_order_authority is not False
            or closure.bce_oracle_authority is not False
            or closure.mask_commit_authority is not False
            or receipt.endpoint_action_hashes != request.endpoint_action_hashes
        ):
            _refuse(
                V1FQStoreRefusalCode.ACQUISITION_MISMATCH,
                "projection did not preserve the endpoint-only selected-shell memory ABI",
            )
        new_records = tuple(acquisition.projection_records)
        if len(new_records) != request.context_count:
            _refuse(V1FQStoreRefusalCode.ACQUISITION_MISMATCH, "projection acquisition has a partial scalar ledger")
        for row, record in enumerate(new_records):
            if (
                record.design_hash != request.design_hash
                or record.ladder_hash != request.ladder_hash
                or record.context_id != request.expected_context_ids[row]
                or record.draw_id != request.expected_draw_ids[row]
                or record.stratum_id != request.expected_stratum_ids[row]
                or record.projection_proof_hash != closure.closure_hash
                or record.endpoint_action_hashes != request.endpoint_action_hashes
                or record.signed_projection_by_endpoint
                != tuple(float(value) for value in result.signed_projection_matrix[row])
                or record.numerical_radius_by_endpoint
                != tuple(float(value) for value in result.numerical_radius_matrix[row])
            ):
                _refuse(
                    V1FQStoreRefusalCode.ACQUISITION_MISMATCH,
                    "projection scalar ledger differs from its endpoint-only closure",
                )
        certifier = _replay_records(self.registration, (*self._records, *new_records))
        certification_receipt = certifier.receipt()
        try:
            design = _artifacts_certification_design(self.registration.artifacts)
            verify_frozen_ladder_certification_receipt(
                design=design,
                receipt=certification_receipt,
            )
            decision = select_largest_certified_endpoint(
                design=design,
                receipt=certification_receipt,
                budget=self.registration.budget,
            )
        except (TypeError, ValueError) as error:
            _refuse(V1FQStoreRefusalCode.ACQUISITION_MISMATCH, f"F/Q certification refused acquisition: {error}")
        look_index = self.next_look_index
        assessment = _assess_look(
            receipt=certification_receipt,
            decision=decision,
            budget=self.registration.budget,
            final_registered_look=look_index == len(self.registration.registered_looks),
            decision_scope_receipt=_artifacts_decision_scope(
                self.registration.artifacts
            ),
        )
        look = self.registration.registered_looks[look_index - 1]
        commit = V1FQRegisteredLookCommit(
            registration_hash=self.registration.registration_hash,
            prior_commit_hash=self.prior_commit_hash,
            look_index=look_index,
            cumulative_stratum_counts=look.cumulative_stratum_counts,
            projection_request_hash=request.request_hash,
            projection_csr_hash=request.csr.csr_hash,
            projection_receipt_hash=receipt.receipt_hash,
            acquisition_closure_hash=closure.closure_hash,
            scorer_receipt_hash=closure.scorer_receipt_hash,
            scorer_request_identity_hash=request.scorer_request_identity_hash,
            acquisition_lineage_hash=request.acquisition_lineage_hash,
            physical_mask_axis_hash=request.physical_mask_axis_hash,
            ordered_score_route_axis_hash=request.ordered_score_route_axis_hash,
            signed_projection_matrix_hash=receipt.signed_projection_matrix_hash,
            absolute_input_sum_upper_matrix_hash=receipt.absolute_input_sum_upper_matrix_hash,
            numerical_radius_matrix_hash=receipt.numerical_radius_matrix_hash,
            projection_records=new_records,
            certification_receipt=certification_receipt,
            decision=decision,
            assessment=assessment,
        )
        destination = self.root / _LOOK_DIRECTORY / f"look_{look_index:08d}.json"
        _atomic_publish_noreplace(
            destination, _canonical_json_bytes(commit.to_json_dict())
        )
        self._commits = (*self._commits, commit)
        self._records = (*self._records, *new_records)
        return commit


V1SelectedShellFQStore = V1ResponseExactificationStore


__all__ = [
    "FQ_ACQUISITION_REGISTRATION_SCHEMA",
    "FQ_COMPOSITION_ACQUISITION_REGISTRATION_SCHEMA",
    "FQ_CONTEXT_ROLE_REGISTRATION_SCHEMA",
    "FQ_REGISTERED_LOOK_ABI_SCHEMA",
    "FQ_REGISTERED_LOOK_ASSESSMENT_SCHEMA",
    "FQ_REGISTERED_LOOK_COMMIT_SCHEMA",
    "FQ_STORE_AUTHORITY_SCOPE",
    "FQ_STORE_MEMORY_SCOPE",
    "FQ_STALE_PARTIAL_POLICY",
    "V1FQAcquisitionRegistration",
    "V1FQContextRoleRegistration",
    "V1FQEndpointClassification",
    "V1FQRegisteredLookABI",
    "V1FQRegisteredLookAssessment",
    "V1FQRegisteredLookCommit",
    "V1FQStoreRefusal",
    "V1FQStoreRefusalCode",
    "V1ResponseExactificationStore",
    "V1SelectedShellFQStore",
    "assess_v1_fq_registered_look",
    "register_v1_fq_train_context_role",
]
