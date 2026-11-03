"""Content-addressed receipts for the genuine partition-free modern-v1 loop.

The receipt layer records what the acquisition, controller, runtime, terminal
firewall, and comparison harness did.  It deliberately carries no reusable
mask-commit capability and no selector or scientific-claim authority.  Live
mutation authority is private to :mod:`modern_v1_controller` and is consumed
before a transition receipt is emitted.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final, Literal

from .contracts import content_hash
from .modern_v1_candidate_composition import V1RemovedRouteSetIdentity
from .modern_v1_fqr_horizon import FQR_NO_CANDIDATE_HORIZON_STOP_REASON
from .modern_v1_pareto_milestones import V1ParetoMilestoneReceipt

_ACQUISITION_SCHEMA: Final = "partition_free_v1_acquisition_generation_receipt_v2"
_ATTEMPT_SCHEMA: Final = "partition_free_v1_candidate_attempt_receipt_v1"
_ATTEMPT_V2_SCHEMA: Final = "partition_free_v1_candidate_attempt_receipt_v2"
_TRANSITION_SCHEMA: Final = "partition_free_v1_mask_transition_receipt_v1"
_TRANSITION_COMPACT_SCHEMA: Final = "partition_free_v1_mask_transition_receipt_v2"
_SNAPSHOT_SCHEMA: Final = "partition_free_v1_controller_snapshot_receipt_v2"
_SNAPSHOT_HORIZON_SCHEMA: Final = (
    "partition_free_v1_controller_snapshot_receipt_v3"
)
_SNAPSHOT_LABEL_NATIVE_SCHEMA: Final = (
    "partition_free_v1_controller_snapshot_receipt_v4"
)
_SNAPSHOT_LABEL_NATIVE_HORIZON_SCHEMA: Final = (
    "partition_free_v1_controller_snapshot_receipt_v5"
)
_TERMINAL_FREEZE_SCHEMA: Final = "partition_free_v1_terminal_freeze_receipt_v2"
_TERMINAL_VALIDATION_SCHEMA: Final = "partition_free_v1_terminal_validation_receipt_v1"
_TERMINAL_VALIDATION_FAMILY_FREEZE_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_freeze_receipt_v1"
)
_TERMINAL_VALIDATION_FAMILY_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_receipt_v1"
)
_TERMINAL_VALIDATION_FAMILY_REQUEST_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_request_receipt_v1"
)
_TERMINAL_VALIDATION_FAMILY_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_terminal_validation_family_observation_receipt_v1"
)
_TERMINAL_COMPARISON_FAMILY_FREEZE_V2_SCHEMA: Final = (
    "partition_free_v1_terminal_comparison_family_freeze_receipt_v2"
)
_TERMINAL_COMPARISON_FAMILY_REQUEST_V2_SCHEMA: Final = (
    "partition_free_v1_terminal_comparison_family_request_receipt_v2"
)
_TERMINAL_COMPARISON_FAMILY_OBSERVATION_V2_SCHEMA: Final = (
    "partition_free_v1_terminal_comparison_family_observation_receipt_v2"
)
_TERMINAL_COMPARISON_FAMILY_V2_SCHEMA: Final = (
    "partition_free_v1_terminal_comparison_family_receipt_v2"
)
_TERMINAL_COMPARISON_FAMILY_VALUES_V2_PREFIX: Final = (
    "partition-free-v1-terminal-comparison-family-values-v2-"
)
_TERMINAL_COMPARISON_FAMILY_LANE_VALUES_V2_PREFIX: Final = (
    "partition-free-v1-terminal-comparison-family-lane-values-v2-"
)
_TERMINAL_COMPARISON_FAMILY_PAIRED_VALUES_V2_PREFIX: Final = (
    "partition-free-v1-terminal-comparison-family-paired-values-v2-"
)
_MEASURED_DENSE_REFERENCE_V2_SCHEMA: Final = (
    "partition_free_v1_measured_dense_reference_v2"
)
_PERFORMANCE_SCHEMA: Final = "partition_free_v1_performance_receipt_v1"
_COMPLETION_SCHEMA: Final = "partition_free_v1_run_completion_receipt_v1"

DecisionAction = Literal[
    "accept_candidate",
    "reject_candidate",
    "refuse_exhausted",
    "skip_candidate_calibration_refusal",
]
ControllerPhase = Literal[
    "needs_acquisition",
    "queue_ready",
    "candidate_active",
    "stopped",
]

TERMINAL_VALIDATION_FAMILY_ROLES: Final = (
    "dense",
    "pareto_milestone",
    "final",
)
TERMINAL_COMPARISON_FAMILY_ROLES_V2: Final = (
    "dense_reference",
    "current_v0_comparator",
    "terminal_v1",
)
TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2: Final = (
    "current_v0_comparator_minus_dense_reference",
    "terminal_v1_minus_dense_reference",
    "terminal_v1_minus_current_v0_comparator",
)
_FROZEN_AFFINE_CALIBRATION_FIELDS: Final = frozenset(
    {
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
    }
)


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


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
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256_hex(value: object, label: str) -> str:
    result = _nonempty(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _relative_posix_path(value: object, label: str) -> str:
    raw = _nonempty(value, label)
    path = PurePosixPath(raw)
    if (
        raw == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
        or str(path) != raw
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return raw


def _payload(instance: object) -> Mapping[str, Any]:
    fields = instance.__dataclass_fields__  # type: ignore[attr-defined]
    return MappingProxyType({name: getattr(instance, name) for name in fields})


def _strict_values(
    cls: type[Any],
    payload: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} payload must be a mapping")
    expected = set(cls.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(f"{label} payload fields differ")
    return dict(payload)


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _strings(
    values: Sequence[str],
    label: str,
    *,
    unique: bool = False,
) -> tuple[str, ...]:
    result = tuple(_nonempty(value, label) for value in values)
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return result


def _little_endian_float64_payload(values: Sequence[float]) -> bytes:
    """Encode one finite row with the terminal matrix's exact scalar ABI."""

    row = tuple(float(value) for value in values)
    if not row:
        raise ValueError("terminal comparison row must be nonempty")
    return struct.pack(f"<{len(row)}d", *row)


def _close_at_float64_roundoff(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=32.0 * math.ulp(scale),
    )


@dataclass(frozen=True, slots=True)
class V1AcquisitionGenerationReceipt:
    """One complete current-mask FD/correction/domain/order generation."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_hash: str
    geometry_state_hash: str
    fd_adapter_result_hash: str
    correction_result_hash: str
    chart_hash: str
    numeric_state_hash: str
    finite_domain_hash: str
    constraints_hash: str
    composition_result_hash: str
    ordered_candidate_ids: tuple[str, ...]
    source_measurement_hashes: tuple[str, ...]
    refresh_reason: str
    logical_sequence: int
    proposal_only: bool = True
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
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
            "refresh_reason",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, "generation_index"),
        )
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        candidates = _strings(
            self.ordered_candidate_ids,
            "ordered_candidate_ids",
            unique=True,
        )
        object.__setattr__(self, "ordered_candidate_ids", candidates)
        object.__setattr__(
            self,
            "source_measurement_hashes",
            _strings(
                self.source_measurement_hashes,
                "source_measurement_hashes",
                unique=True,
            ),
        )
        if (
            self.schema != _ACQUISITION_SCHEMA
            or self.proposal_only is not True
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("acquisition receipt crossed its proposal-only authority firewall")
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, "acquisition receipt metadata"),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-acquisition-generation-")


@dataclass(frozen=True, slots=True)
class V1CandidateAttemptReceipt:
    """Terminal record for one frozen candidate at one current mask."""

    campaign_contract_hash: str
    controller_run_id: str
    candidate_attempt_index: int
    accepted_step_index: int
    acquisition_generation_index: int
    acquisition_receipt_hash: str
    queue_index: int
    candidate_id: str
    parent_mask_hash: str
    candidate_mask_hash: str
    step_freeze_hash: str | None
    ledger_hash: str
    decision_receipt_hash: str
    action: DecisionAction
    mask_mutated: bool
    resulting_current_mask_hash: str
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    reason: str
    logical_sequence: int
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ATTEMPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_id",
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "candidate_mask_hash",
            "ledger_hash",
            "decision_receipt_hash",
            "resulting_current_mask_hash",
            "reason",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "candidate_attempt_index",
            "accepted_step_index",
            "acquisition_generation_index",
            "queue_index",
            "logical_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.action not in {
            "accept_candidate",
            "reject_candidate",
            "refuse_exhausted",
            "skip_candidate_calibration_refusal",
        }:
            raise ValueError("candidate attempt action is not terminal")
        if type(self.mask_mutated) is not bool:
            raise TypeError("mask_mutated must be bool")
        expected_mutation = self.action == "accept_candidate"
        if self.mask_mutated is not expected_mutation:
            raise ValueError("only an accepted candidate may record a mask mutation")
        expected_mask = (
            self.candidate_mask_hash if expected_mutation else self.parent_mask_hash
        )
        if self.resulting_current_mask_hash != expected_mask:
            raise ValueError("candidate-attempt resulting mask disagrees with its action")
        is_calibration_skip = (
            self.action == "skip_candidate_calibration_refusal"
        )
        if is_calibration_skip:
            if self.schema != _ATTEMPT_V2_SCHEMA or self.step_freeze_hash is not None:
                raise ValueError(
                    "calibration-refusal attempt requires v2 and no step freeze"
                )
        else:
            if self.schema != _ATTEMPT_SCHEMA:
                raise ValueError("ordinary candidate attempts must retain v1 schema")
            object.__setattr__(
                self,
                "step_freeze_hash",
                _nonempty(self.step_freeze_hash, "step_freeze_hash"),
            )
        draws = _strings(self.draw_ids, "draw_ids", unique=True)
        sources = _strings(self.source_context_ids, "source_context_ids")
        if is_calibration_skip:
            if draws or sources:
                raise ValueError(
                    "calibration-refusal attempt cannot carry oracle draw/source lineage"
                )
        elif not draws or len(draws) != len(sources):
            raise ValueError("candidate-attempt draw/source lineage must be aligned")
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        if (
            self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("candidate-attempt receipt crossed its authority firewall")
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, "candidate-attempt metadata"),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-candidate-attempt-")


@dataclass(frozen=True, slots=True)
class V1MaskTransitionReceipt:
    """Consumed single-use accepted parent-to-exact-child transition receipt."""

    campaign_contract_hash: str
    controller_run_id: str
    transition_index: int
    accepted_step_index: int
    candidate_attempt_index: int
    candidate_id: str
    parent_mask_hash: str
    child_mask_hash: str
    removed_route_indices: tuple[int, ...] | V1RemovedRouteSetIdentity
    accepted_decision_receipt_hash: str
    accepted_ledger_hash: str
    acquisition_receipt_hash: str
    invalidated_chart_hash: str
    invalidated_composition_hash: str
    commit_capability_consumption_hash: str
    logical_sequence: int
    single_use_consumed: bool = True
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_id",
            "candidate_id",
            "parent_mask_hash",
            "child_mask_hash",
            "accepted_decision_receipt_hash",
            "accepted_ledger_hash",
            "acquisition_receipt_hash",
            "invalidated_chart_hash",
            "invalidated_composition_hash",
            "commit_capability_consumption_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "transition_index",
            "accepted_step_index",
            "candidate_attempt_index",
            "logical_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if type(self.removed_route_indices) is V1RemovedRouteSetIdentity:
            object.__setattr__(self, "schema", _TRANSITION_COMPACT_SCHEMA)
            raw_indices: tuple[int, ...] | None = None
        else:
            raw_indices = tuple(self.removed_route_indices)
        if raw_indices is None:
            pass
        elif (
            not raw_indices
            or any(
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or int(index) < 0
                for index in raw_indices
            )
        ):
            raise ValueError("transition requires nonnegative removed-route indices")
        else:
            indices = tuple(sorted(int(index) for index in raw_indices))
            if len(set(indices)) != len(indices):
                raise ValueError("transition removed-route indices must be unique")
            object.__setattr__(self, "removed_route_indices", indices)
        if self.parent_mask_hash == self.child_mask_hash:
            raise ValueError("an accepted transition must change the exact hard mask")
        if (
            self.schema not in {_TRANSITION_SCHEMA, _TRANSITION_COMPACT_SCHEMA}
            or self.single_use_consumed is not True
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("transition receipt retained or widened live authority")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-mask-transition-")

    @property
    def removed_count(self) -> int:
        value = self.removed_route_indices
        return value.count if type(value) is V1RemovedRouteSetIdentity else len(value)

    @property
    def removed_route_index_hash(self) -> str | None:
        value = self.removed_route_indices
        return value.index_hash if type(value) is V1RemovedRouteSetIdentity else None

    @property
    def removed_route_indices_complete(self) -> bool:
        return type(self.removed_route_indices) is tuple


@dataclass(frozen=True, slots=True)
class V1ControllerSnapshotReceipt:
    """Compact exact-state identity used by checkpoints and replay audits."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    phase: ControllerPhase
    current_mask_hash: str
    acquisition_generation_index: int
    queue_cursor: int
    accepted_step_count: int
    candidate_attempt_count: int
    cumulative_removed_count: int
    active_candidate_id: str | None
    active_step_freeze_hash: str | None
    active_ledger_hash: str | None
    accepted_trajectory_hash: str
    acquisition_receipt_hash: str | None
    acquisition_receipt_hashes: tuple[str, ...]
    attempt_receipt_hashes: tuple[str, ...]
    transition_receipt_hashes: tuple[str, ...]
    globally_used_draw_ids_hash: str
    stop_reason: str | None
    logical_sequence: int
    fqr_no_candidate_horizon_witness_hash: str | None = None
    label_native_runtime_plan_hash: str | None = None
    label_native_runtime_plan_hashes: tuple[str, ...] = ()
    label_native_rejected_plan_indices: tuple[int, ...] = ()
    label_native_selected_acquisition_receipt_hash: str | None = None
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "current_mask_hash",
            "accepted_trajectory_hash",
            "globally_used_draw_ids_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.phase not in {
            "needs_acquisition",
            "queue_ready",
            "candidate_active",
            "stopped",
        }:
            raise ValueError("controller snapshot phase is invalid")
        for name in (
            "acquisition_generation_index",
            "queue_cursor",
            "accepted_step_count",
            "candidate_attempt_count",
            "cumulative_removed_count",
            "logical_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        for name in (
            "active_candidate_id",
            "active_step_freeze_hash",
            "active_ledger_hash",
            "acquisition_receipt_hash",
            "stop_reason",
            "fqr_no_candidate_horizon_witness_hash",
            "label_native_runtime_plan_hash",
            "label_native_selected_acquisition_receipt_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        active = (
            self.active_candidate_id,
            self.active_step_freeze_hash,
            self.active_ledger_hash,
        )
        if self.phase == "candidate_active":
            if any(value is None for value in active):
                raise ValueError("active snapshot lacks candidate decision identity")
        elif any(value is not None for value in active):
            raise ValueError("inactive snapshot retains candidate decision identity")
        if self.phase in {"queue_ready", "candidate_active"}:
            if self.acquisition_receipt_hash is None:
                raise ValueError("queue-bearing snapshot lacks an acquisition receipt")
        elif self.acquisition_receipt_hash is not None:
            raise ValueError("nonqueue snapshot retains a stale acquisition receipt")
        if (self.phase == "stopped") != (self.stop_reason is not None):
            raise ValueError("controller stop reason and phase disagree")
        has_horizon_witness = (
            self.fqr_no_candidate_horizon_witness_hash is not None
        )
        if has_horizon_witness != (
            self.stop_reason == FQR_NO_CANDIDATE_HORIZON_STOP_REASON
        ):
            raise ValueError(
                "controller snapshot F/Q/R horizon reason and witness disagree"
            )
        acquisition_history = _strings(
            self.acquisition_receipt_hashes,
            "acquisition_receipt_hashes",
            unique=True,
        )
        plan_history = _strings(
            self.label_native_runtime_plan_hashes,
            "label_native_runtime_plan_hashes",
            unique=True,
        )
        rejected_indices = tuple(
            _nonnegative_int(value, "label_native_rejected_plan_index")
            for value in self.label_native_rejected_plan_indices
        )
        has_label_native = bool(
            plan_history
            or self.label_native_runtime_plan_hash is not None
            or rejected_indices
            or self.label_native_selected_acquisition_receipt_hash is not None
        )
        if has_label_native:
            current_plan_hash = self.label_native_runtime_plan_hash
            selected_receipt_hash = (
                self.label_native_selected_acquisition_receipt_hash
            )
            if not plan_history:
                raise ValueError(
                    "label-native snapshot lacks immutable plan history"
                )
            if current_plan_hash is None:
                if rejected_indices or selected_receipt_hash is not None:
                    raise ValueError(
                        "label-native snapshot retained live coordinates without a plan"
                    )
            else:
                if (
                    plan_history[-1] != current_plan_hash
                    or rejected_indices != tuple(range(self.queue_cursor))
                ):
                    raise ValueError(
                        "label-native snapshot cursor/rejections differ from its live plan"
                    )
                if self.phase in {"queue_ready", "candidate_active"}:
                    if (
                        selected_receipt_hash is None
                        or selected_receipt_hash != self.acquisition_receipt_hash
                    ):
                        raise ValueError(
                            "label-native queue snapshot lacks its exact selected receipt"
                        )
                elif selected_receipt_hash is not None:
                    raise ValueError(
                        "label-native nonqueue snapshot retained a selected endpoint"
                    )
            if selected_receipt_hash is not None and (
                selected_receipt_hash not in acquisition_history
            ):
                raise ValueError(
                    "label-native selected receipt is absent from durable acquisition history"
                )
        elif (
            self.label_native_runtime_plan_hash is not None
            or plan_history
            or rejected_indices
            or self.label_native_selected_acquisition_receipt_hash is not None
        ):
            raise ValueError("partial label-native snapshot state is invalid")
        if self.phase in {"queue_ready", "candidate_active"}:
            if not acquisition_history or acquisition_history[-1] != self.acquisition_receipt_hash:
                raise ValueError("queue snapshot acquisition is absent from durable history")
        if not has_label_native:
            if self.phase == "needs_acquisition" and len(acquisition_history) != self.accepted_step_count:
                raise ValueError("needs-acquisition snapshot has the wrong acquisition history length")
            if self.phase in {"queue_ready", "candidate_active"} and len(acquisition_history) != self.accepted_step_count + 1:
                raise ValueError("queue snapshot has the wrong acquisition history length")
            if self.phase == "stopped" and len(acquisition_history) not in {
                self.accepted_step_count,
                self.accepted_step_count + 1,
            }:
                raise ValueError("stopped snapshot has an impossible acquisition history length")
        object.__setattr__(self, "acquisition_receipt_hashes", acquisition_history)
        object.__setattr__(self, "label_native_runtime_plan_hashes", plan_history)
        object.__setattr__(
            self,
            "label_native_rejected_plan_indices",
            rejected_indices,
        )
        object.__setattr__(
            self,
            "attempt_receipt_hashes",
            _strings(
                self.attempt_receipt_hashes,
                "attempt_receipt_hashes",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "transition_receipt_hashes",
            _strings(
                self.transition_receipt_hashes,
                "transition_receipt_hashes",
                unique=True,
            ),
        )
        if has_label_native:
            expected_schema = (
                _SNAPSHOT_LABEL_NATIVE_HORIZON_SCHEMA
                if has_horizon_witness
                else _SNAPSHOT_LABEL_NATIVE_SCHEMA
            )
        else:
            expected_schema = (
                _SNAPSHOT_HORIZON_SCHEMA
                if has_horizon_witness
                else _SNAPSHOT_SCHEMA
            )
        if (
            self.schema != expected_schema
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("controller snapshot crossed its authority firewall")

    @property
    def receipt_hash(self) -> str:
        # Preserve the exact historical v2 projection when the additive
        # horizon coordinate is absent.  Including an explicit ``None`` would
        # silently rewrite every legacy snapshot identity.
        additive_names = {
            "fqr_no_candidate_horizon_witness_hash",
            "label_native_runtime_plan_hash",
            "label_native_runtime_plan_hashes",
            "label_native_rejected_plan_indices",
            "label_native_selected_acquisition_receipt_hash",
        }
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in additive_names
        }
        if self.fqr_no_candidate_horizon_witness_hash is not None:
            payload["fqr_no_candidate_horizon_witness_hash"] = (
                self.fqr_no_candidate_horizon_witness_hash
            )
        if self.label_native_runtime_plan_hashes:
            payload.update(
                {
                    "label_native_runtime_plan_hash": (
                        self.label_native_runtime_plan_hash
                    ),
                    "label_native_runtime_plan_hashes": (
                        self.label_native_runtime_plan_hashes
                    ),
                    "label_native_rejected_plan_indices": (
                        self.label_native_rejected_plan_indices
                    ),
                    "label_native_selected_acquisition_receipt_hash": (
                        self.label_native_selected_acquisition_receipt_hash
                    ),
                }
            )
        return content_hash(
            payload,
            prefix="partition-free-v1-controller-snapshot-",
        )


@dataclass(frozen=True, slots=True)
class V1TerminalFreezeReceipt:
    """Freeze extraction and extraction-trained calibration before reveal."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    controller_snapshot_hash: str
    terminal_mask_hash: str
    accepted_trajectory_hash: str
    extraction_state_hash: str
    extraction_calibration_hash: str
    dense_reference_hash: str
    terminal_validation_population_hash: str
    acquisition_receipt_hashes: tuple[str, ...]
    attempt_receipt_hashes: tuple[str, ...]
    transition_receipt_hashes: tuple[str, ...]
    stop_reason: str
    freeze_sequence: int
    terminal_rows_visible: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
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
            "stop_reason",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "freeze_sequence",
            _nonnegative_int(self.freeze_sequence, "freeze_sequence"),
        )
        object.__setattr__(
            self,
            "acquisition_receipt_hashes",
            _strings(
                self.acquisition_receipt_hashes,
                "acquisition_receipt_hashes",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "attempt_receipt_hashes",
            _strings(
                self.attempt_receipt_hashes,
                "attempt_receipt_hashes",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "transition_receipt_hashes",
            _strings(
                self.transition_receipt_hashes,
                "transition_receipt_hashes",
                unique=True,
            ),
        )
        if (
            self.schema != _TERMINAL_FREEZE_SCHEMA
            or self.terminal_rows_visible is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("terminal freeze crossed the reveal/authority firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-terminal-freeze-")


@dataclass(frozen=True, slots=True)
class V1TerminalValidationReceipt:
    """One post-freeze terminal reveal, content-addressed and nonadaptive."""

    terminal_freeze_hash: str
    terminal_request_hash: str
    terminal_outcome_hash: str
    validation_values_hash: str
    validation_value_count: int
    calibration_identity: str
    terminal_mask_hash: str
    reveal_sequence: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    single_reveal: bool = True
    fed_back_into_selection: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_VALIDATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "terminal_freeze_hash",
            "terminal_request_hash",
            "terminal_outcome_hash",
            "validation_values_hash",
            "calibration_identity",
            "terminal_mask_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "validation_value_count",
            _positive_int(self.validation_value_count, "validation_value_count"),
        )
        object.__setattr__(
            self,
            "reveal_sequence",
            _nonnegative_int(self.reveal_sequence, "reveal_sequence"),
        )
        frozen_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            if type(key) is not str or not key:
                raise TypeError("terminal validation metric names must be nonempty strings")
            frozen_metrics[key] = _finite_nonnegative(value, f"metric {key}")
        object.__setattr__(self, "metrics", MappingProxyType(frozen_metrics))
        if (
            self.schema != _TERMINAL_VALIDATION_SCHEMA
            or self.single_reveal is not True
            or self.fed_back_into_selection is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
        ):
            raise ValueError("terminal validation crossed its single-reveal firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-terminal-validation-")


@dataclass(frozen=True, slots=True)
class V1TerminalValidationFamilyFreezeReceipt:
    """Outcome-independent dense/matched-density/final terminal family.

    The exact validation population, its order, metric semantics, evaluator,
    milestone checkpoint bytes, and validation-independent affine states are fixed
    here.  This receipt contains no terminal observation and cannot authorize
    a comparison by itself.
    """

    terminal_freeze_hash: str
    terminal_validation_population_hash: str
    terminal_validation_context_ids_hash: str
    terminal_validation_context_count: int
    terminal_metric_semantics_hash: str
    terminal_evaluator_identity_hash: str
    dense_reference_hash: str
    dense_mask_hash: str
    route_axis_hash: str
    route_count: int
    pareto_milestone_receipt: V1ParetoMilestoneReceipt
    pareto_milestone_archive_relative_path: str
    pareto_milestone_archive_sha256: str
    final_mask_hash: str
    role_mask_hashes: tuple[str, str, str]
    role_retained_route_counts: tuple[int, int, int]
    role_to_unique_lane: tuple[int, int, int]
    unique_lane_mask_hashes: tuple[str, ...]
    unique_lane_calibration_hashes: tuple[str, ...]
    unique_lane_calibration_freeze_hashes: tuple[str, ...]
    unique_lane_calibration_freeze_payloads: tuple[Mapping[str, Any], ...]
    freeze_sequence: int
    roles: tuple[str, str, str] = TERMINAL_VALIDATION_FAMILY_ROLES
    validation_independent: bool = True
    terminal_rows_visible: bool = False
    terminal_outcomes_used_for_selection: bool = False
    matched_density_comparison_registered: bool = True
    schema: str = _TERMINAL_VALIDATION_FAMILY_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "terminal_freeze_hash",
            "terminal_validation_population_hash",
            "terminal_validation_context_ids_hash",
            "terminal_metric_semantics_hash",
            "terminal_evaluator_identity_hash",
            "dense_reference_hash",
            "dense_mask_hash",
            "route_axis_hash",
            "final_mask_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.pareto_milestone_receipt) is not V1ParetoMilestoneReceipt:
            raise TypeError("terminal family requires the exact Pareto milestone receipt")
        self.pareto_milestone_receipt.__post_init__()
        route_count = _positive_int(self.route_count, "route_count")
        context_count = _positive_int(
            self.terminal_validation_context_count,
            "terminal_validation_context_count",
        )
        archive_path = _relative_posix_path(
            self.pareto_milestone_archive_relative_path,
            "pareto_milestone_archive_relative_path",
        )
        archive_sha256 = _sha256_hex(
            self.pareto_milestone_archive_sha256,
            "pareto_milestone_archive_sha256",
        )
        counts = tuple(
            _positive_int(value, "role_retained_route_count")
            for value in self.role_retained_route_counts
        )
        if len(counts) != 3 or any(value > route_count for value in counts):
            raise ValueError("terminal family retained-route counts are invalid")
        roles = tuple(self.roles)
        if roles != TERMINAL_VALIDATION_FAMILY_ROLES:
            raise ValueError("terminal family roles are not canonical")
        role_masks = _strings(self.role_mask_hashes, "role_mask_hash", unique=False)
        if len(role_masks) != 3:
            raise ValueError("terminal family must bind exactly three roles")
        lane_map = tuple(
            _nonnegative_int(value, "role_to_unique_lane")
            for value in self.role_to_unique_lane
        )
        unique_masks = _strings(
            self.unique_lane_mask_hashes,
            "unique_lane_mask_hash",
            unique=True,
        )
        if (
            len(lane_map) != 3
            or not unique_masks
            or any(value >= len(unique_masks) for value in lane_map)
        ):
            raise ValueError("terminal family role-to-lane map is invalid")
        expected_unique: list[str] = []
        expected_map: list[int] = []
        for mask_hash in role_masks:
            if mask_hash not in expected_unique:
                expected_unique.append(mask_hash)
            expected_map.append(expected_unique.index(mask_hash))
        if tuple(expected_unique) != unique_masks or tuple(expected_map) != lane_map:
            raise ValueError("terminal family lanes are not deterministic first occurrences")
        milestone = self.pareto_milestone_receipt
        milestone_mask = milestone.pending.milestone_mask_hash
        milestone_count = milestone.pending.comparator_retained_route_count
        expected_archive_name = f"{milestone.pending.pending_hash}.mv1ckpt"
        if (
            PurePosixPath(archive_path).name != expected_archive_name
            or archive_sha256
            != milestone.checkpoint_binding.checkpoint_artifact_hash
            or role_masks[0] != self.dense_mask_hash
            or role_masks[1] != milestone_mask
            or role_masks[2] != self.final_mask_hash
            or counts[0] != route_count
            or counts[1] != milestone_count
            or counts[2] > counts[1]
            or (counts[2] == counts[1]) is not (role_masks[2] == role_masks[1])
        ):
            raise ValueError("terminal family masks/counts differ from dense/milestone/final")
        calibration_hashes = _strings(
            self.unique_lane_calibration_hashes,
            "unique_lane_calibration_hash",
            unique=False,
        )
        freeze_hashes = _strings(
            self.unique_lane_calibration_freeze_hashes,
            "unique_lane_calibration_freeze_hash",
            unique=True,
        )
        raw_payloads = tuple(self.unique_lane_calibration_freeze_payloads)
        if not (
            len(calibration_hashes)
            == len(freeze_hashes)
            == len(raw_payloads)
            == len(unique_masks)
        ):
            raise ValueError("terminal family calibration lane counts differ")
        payloads: list[Mapping[str, Any]] = []
        shared_calibration_coordinates: tuple[object, ...] | None = None
        for index, raw in enumerate(raw_payloads):
            if not isinstance(raw, Mapping) or set(raw) != _FROZEN_AFFINE_CALIBRATION_FIELDS:
                raise ValueError("terminal family calibration freeze fields differ")
            payload = _freeze(raw, f"unique_lane_calibration_freeze_payload[{index}]")
            fit_receipt = payload.get("fit_receipt")
            expected_fit_receipt_hash = (
                content_hash(
                    fit_receipt,
                    prefix="partition-free-v1-frozen-calibration-fit-receipt-",
                )
                if isinstance(fit_receipt, Mapping)
                else None
            )
            target_class_id = _nonnegative_int(
                payload.get("target_class_id"),
                "calibration target_class_id",
            )
            scale = _finite(payload.get("scale"), "calibration scale")
            _finite(payload.get("offset"), "calibration offset")
            coordinates = (
                payload.get("query_id"),
                payload.get("query_hash"),
                target_class_id,
                payload.get("calibration_context_axis_hash"),
                payload.get("source_label_map_hash"),
                payload.get("model_state_hash"),
                payload.get("calibration_schema"),
            )
            if shared_calibration_coordinates is None:
                shared_calibration_coordinates = coordinates
            elif coordinates != shared_calibration_coordinates:
                raise ValueError("terminal family calibrations differ in shared design")
            if (
                payload.get("mask_hash") != unique_masks[index]
                or payload.get("calibration_hash") != calibration_hashes[index]
                or payload.get("schema")
                != "partition_free_v1_frozen_affine_calibration_v1"
                or scale <= 0.0
                or payload.get("validation_independent") is not True
                or payload.get("terminal_rows_used_for_fit") is not False
                or payload.get("terminal_outcomes_opened") is not False
                or not isinstance(fit_receipt, Mapping)
                or fit_receipt.get("fit_succeeded") is not True
                or fit_receipt.get("mask_hash") != unique_masks[index]
                or fit_receipt.get("validation_independent") is not True
                or fit_receipt.get("terminal_rows_used_for_fit") is not False
                or fit_receipt.get("terminal_outcomes_opened") is not False
                or payload.get("fit_receipt_hash") != expected_fit_receipt_hash
                or content_hash(
                    payload,
                    prefix="partition-free-v1-frozen-affine-calibration-",
                )
                != freeze_hashes[index]
            ):
                raise ValueError("terminal family calibration freeze is not exact/validation-independent")
            payloads.append(payload)
        sequence = _nonnegative_int(self.freeze_sequence, "freeze_sequence")
        if (
            self.validation_independent is not True
            or self.terminal_rows_visible is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.matched_density_comparison_registered is not True
            or self.schema != _TERMINAL_VALIDATION_FAMILY_FREEZE_SCHEMA
        ):
            raise ValueError("terminal family freeze crossed its causal firewall")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "terminal_validation_context_count", context_count)
        object.__setattr__(
            self,
            "pareto_milestone_archive_relative_path",
            archive_path,
        )
        object.__setattr__(
            self,
            "pareto_milestone_archive_sha256",
            archive_sha256,
        )
        object.__setattr__(self, "role_mask_hashes", role_masks)
        object.__setattr__(self, "role_retained_route_counts", counts)
        object.__setattr__(self, "role_to_unique_lane", lane_map)
        object.__setattr__(self, "unique_lane_mask_hashes", unique_masks)
        object.__setattr__(self, "unique_lane_calibration_hashes", calibration_hashes)
        object.__setattr__(self, "unique_lane_calibration_freeze_hashes", freeze_hashes)
        object.__setattr__(
            self,
            "unique_lane_calibration_freeze_payloads",
            tuple(payloads),
        )
        object.__setattr__(self, "freeze_sequence", sequence)
        object.__setattr__(self, "roles", TERMINAL_VALIDATION_FAMILY_ROLES)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-validation-family-freeze-",
        )

    @property
    def milestone_final_deduplicated(self) -> bool:
        return self.role_to_unique_lane[1] == self.role_to_unique_lane[2]

    def lane_for_role(self, role: str) -> int:
        try:
            return self.role_to_unique_lane[self.roles.index(role)]
        except ValueError as exc:
            raise KeyError(role) from exc

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["pareto_milestone_receipt"] = (
            self.pareto_milestone_receipt.to_payload()
        )
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalValidationFamilyFreezeReceipt:
        values = _strict_values(cls, payload, "terminal validation family freeze")
        values["pareto_milestone_receipt"] = V1ParetoMilestoneReceipt.from_payload(
            values["pareto_milestone_receipt"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1TerminalValidationFamilyRequestReceipt:
    """Checkpointable authorization request with terminal outcomes absent."""

    family_freeze: V1TerminalValidationFamilyFreezeReceipt
    observation_artifact_relative_path: str
    observation_artifact_schema: str
    request_sequence: int
    evaluation_mode: str = "single_common_panel_unique_mask_lanes"
    validation_independent: bool = True
    outcome_fields_absent: bool = True
    terminal_rows_visible: bool = False
    checkpoint_before_outcomes_required: bool = True
    schema: str = _TERMINAL_VALIDATION_FAMILY_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if type(self.family_freeze) is not V1TerminalValidationFamilyFreezeReceipt:
            raise TypeError("terminal family request requires the exact family freeze")
        self.family_freeze.__post_init__()
        artifact_path = _relative_posix_path(
            self.observation_artifact_relative_path,
            "observation_artifact_relative_path",
        )
        object.__setattr__(
            self,
            "observation_artifact_schema",
            _nonempty(self.observation_artifact_schema, "observation_artifact_schema"),
        )
        sequence = _nonnegative_int(self.request_sequence, "request_sequence")
        if sequence != self.family_freeze.freeze_sequence + 1:
            raise ValueError("terminal family request sequence does not follow its freeze")
        if (
            self.evaluation_mode != "single_common_panel_unique_mask_lanes"
            or self.validation_independent is not True
            or self.outcome_fields_absent is not True
            or self.terminal_rows_visible is not False
            or self.checkpoint_before_outcomes_required is not True
            or self.schema != _TERMINAL_VALIDATION_FAMILY_REQUEST_SCHEMA
        ):
            raise ValueError("terminal family request crossed the pre-reveal firewall")
        object.__setattr__(self, "observation_artifact_relative_path", artifact_path)
        object.__setattr__(self, "request_sequence", sequence)

    @property
    def request_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-validation-family-request-",
        )

    @property
    def receipt_hash(self) -> str:
        return self.request_hash

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["family_freeze"] = self.family_freeze.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalValidationFamilyRequestReceipt:
        values = _strict_values(cls, payload, "terminal validation family request")
        values["family_freeze"] = V1TerminalValidationFamilyFreezeReceipt.from_payload(
            values["family_freeze"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1TerminalValidationFamilyObservationReceipt:
    """Durable, replay-verified common-panel observation for one request.

    The raw lane-by-context panel is required to reach stable storage before
    this receipt is constructed.  That sidecar lets a restart reconstruct the
    exact metrics and final family receipt without evaluating validation data
    a second time.
    """

    request: V1TerminalValidationFamilyRequestReceipt
    joint_validation_values_hash: str
    joint_validation_shape: tuple[int, int]
    unique_lane_validation_values_hashes: tuple[str, ...]
    unique_lane_validation_value_counts: tuple[int, ...]
    unique_lane_metrics: tuple[Mapping[str, float], ...]
    common_panel_evaluator_receipt_hash: str
    observation_artifact_relative_path: str
    observation_artifact_sha256: str
    observation_artifact_size_bytes: int
    observation_artifact_semantic_hash: str
    observation_sequence: int
    artifact_durable_before_receipt: bool = True
    artifact_replay_verified: bool = True
    single_common_panel_evaluation: bool = True
    raw_validation_rows_persisted: bool = True
    fed_back_into_selection: bool = False
    terminal_outcomes_used_for_selection: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_VALIDATION_FAMILY_OBSERVATION_SCHEMA

    @staticmethod
    def derive_artifact_semantic_hash(
        *,
        terminal_request_hash: str,
        joint_validation_values_hash: str,
        joint_validation_shape: Sequence[int],
        unique_lane_validation_values_hashes: Sequence[str],
        unique_lane_validation_value_counts: Sequence[int],
        unique_lane_metrics: Sequence[Mapping[str, float]],
        common_panel_evaluator_receipt_hash: str,
    ) -> str:
        """Bind the replayable sidecar semantics to its exact frozen request."""

        return content_hash(
            {
                "terminal_request_hash": terminal_request_hash,
                "joint_validation_values_hash": joint_validation_values_hash,
                "joint_validation_shape": tuple(joint_validation_shape),
                "unique_lane_validation_values_hashes": tuple(
                    unique_lane_validation_values_hashes
                ),
                "unique_lane_validation_value_counts": tuple(
                    unique_lane_validation_value_counts
                ),
                "unique_lane_metrics": tuple(unique_lane_metrics),
                "common_panel_evaluator_receipt_hash": (
                    common_panel_evaluator_receipt_hash
                ),
            },
            prefix=(
                "partition-free-v1-terminal-validation-family-observation-artifact-"
            ),
        )

    def __post_init__(self) -> None:
        if type(self.request) is not V1TerminalValidationFamilyRequestReceipt:
            raise TypeError("terminal family observation requires the exact request")
        self.request.__post_init__()
        for name in (
            "joint_validation_values_hash",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        artifact_path = _relative_posix_path(
            self.observation_artifact_relative_path,
            "observation_artifact_relative_path",
        )
        artifact_sha256 = _sha256_hex(
            self.observation_artifact_sha256,
            "observation_artifact_sha256",
        )
        artifact_size = _positive_int(
            self.observation_artifact_size_bytes,
            "observation_artifact_size_bytes",
        )
        freeze = self.request.family_freeze
        lane_count = len(freeze.unique_lane_mask_hashes)
        shape = tuple(
            _positive_int(value, "joint_validation_shape")
            for value in self.joint_validation_shape
        )
        hashes = _strings(
            self.unique_lane_validation_values_hashes,
            "unique_lane_validation_values_hash",
            unique=False,
        )
        counts = tuple(
            _positive_int(value, "unique_lane_validation_value_count")
            for value in self.unique_lane_validation_value_counts
        )
        raw_metrics = tuple(self.unique_lane_metrics)
        if (
            shape != (lane_count, freeze.terminal_validation_context_count)
            or not (len(hashes) == len(counts) == len(raw_metrics) == lane_count)
            or any(
                value != freeze.terminal_validation_context_count for value in counts
            )
        ):
            raise ValueError("terminal family outcome lanes are incoherent")
        metrics: list[Mapping[str, float]] = []
        for raw in raw_metrics:
            if not isinstance(raw, Mapping) or set(raw) != {
                "mean",
                "maximum",
                "root_mean_square",
            }:
                raise ValueError("terminal family lane metrics differ")
            metrics.append(
                MappingProxyType(
                    {
                        key: _finite_nonnegative(value, f"metric {key}")
                        for key, value in raw.items()
                    }
                )
            )
        expected_artifact_semantic_hash = self.derive_artifact_semantic_hash(
            terminal_request_hash=self.request.request_hash,
            joint_validation_values_hash=self.joint_validation_values_hash,
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=hashes,
            unique_lane_validation_value_counts=counts,
            unique_lane_metrics=metrics,
            common_panel_evaluator_receipt_hash=(
                self.common_panel_evaluator_receipt_hash
            ),
        )
        sequence = _nonnegative_int(self.observation_sequence, "observation_sequence")
        if (
            sequence != self.request.request_sequence + 1
            or artifact_path != self.request.observation_artifact_relative_path
            or self.observation_artifact_semantic_hash
            != expected_artifact_semantic_hash
            or self.artifact_durable_before_receipt is not True
            or self.artifact_replay_verified is not True
            or self.single_common_panel_evaluation is not True
            or self.raw_validation_rows_persisted is not True
            or self.fed_back_into_selection is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
            or self.schema != _TERMINAL_VALIDATION_FAMILY_OBSERVATION_SCHEMA
        ):
            raise ValueError("terminal family observation crossed its replay firewall")
        object.__setattr__(self, "joint_validation_shape", shape)
        object.__setattr__(self, "unique_lane_validation_values_hashes", hashes)
        object.__setattr__(self, "unique_lane_validation_value_counts", counts)
        object.__setattr__(self, "unique_lane_metrics", tuple(metrics))
        object.__setattr__(self, "observation_artifact_relative_path", artifact_path)
        object.__setattr__(self, "observation_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "observation_artifact_size_bytes", artifact_size)
        object.__setattr__(
            self,
            "observation_sequence",
            sequence,
        )

    @property
    def terminal_outcome_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-validation-family-outcome-",
        )

    @property
    def receipt_hash(self) -> str:
        return self.terminal_outcome_hash

    def metrics_for_role(self, role: str) -> Mapping[str, float]:
        freeze = self.request.family_freeze
        try:
            lane = freeze.role_to_unique_lane[freeze.roles.index(role)]
        except ValueError as exc:
            raise KeyError(role) from exc
        return self.unique_lane_metrics[lane]

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["request"] = self.request.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalValidationFamilyObservationReceipt:
        values = _strict_values(cls, payload, "terminal validation family observation")
        values["request"] = V1TerminalValidationFamilyRequestReceipt.from_payload(
            values["request"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1TerminalValidationFamilyReceipt:
    """Matched-density comparison authority after one durable common reveal."""

    observation: V1TerminalValidationFamilyObservationReceipt
    reveal_sequence: int
    single_reveal: bool = True
    matched_density_comparison_authority: bool = True
    fed_back_into_selection: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_VALIDATION_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.observation) is not V1TerminalValidationFamilyObservationReceipt:
            raise TypeError("terminal family receipt requires the exact observation")
        self.observation.__post_init__()
        sequence = _nonnegative_int(self.reveal_sequence, "reveal_sequence")
        if (
            sequence != self.observation.observation_sequence
            or self.single_reveal is not True
            or self.matched_density_comparison_authority is not True
            or self.fed_back_into_selection is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
            or self.schema != _TERMINAL_VALIDATION_FAMILY_SCHEMA
        ):
            raise ValueError("terminal family receipt crossed its authority firewall")
        object.__setattr__(self, "reveal_sequence", sequence)

    @property
    def request(self) -> V1TerminalValidationFamilyRequestReceipt:
        return self.observation.request

    @property
    def family_freeze(self) -> V1TerminalValidationFamilyFreezeReceipt:
        return self.request.family_freeze

    @property
    def family_freeze_hash(self) -> str:
        return self.family_freeze.receipt_hash

    @property
    def terminal_freeze_hash(self) -> str:
        return self.family_freeze.terminal_freeze_hash

    @property
    def terminal_request_hash(self) -> str:
        return self.request.request_hash

    @property
    def terminal_outcome_hash(self) -> str:
        return self.observation.terminal_outcome_hash

    @property
    def terminal_validation_population_hash(self) -> str:
        return self.family_freeze.terminal_validation_population_hash

    @property
    def roles(self) -> tuple[str, str, str]:
        return self.family_freeze.roles

    @property
    def role_to_unique_lane(self) -> tuple[int, int, int]:
        return self.family_freeze.role_to_unique_lane

    @property
    def unique_lane_mask_hashes(self) -> tuple[str, ...]:
        return self.family_freeze.unique_lane_mask_hashes

    @property
    def milestone_final_deduplicated(self) -> bool:
        return self.family_freeze.milestone_final_deduplicated

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-validation-family-",
        )

    def metrics_for_role(self, role: str) -> Mapping[str, float]:
        return self.observation.metrics_for_role(role)

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["observation"] = self.observation.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalValidationFamilyReceipt:
        values = _strict_values(cls, payload, "terminal validation family receipt")
        values["observation"] = (
            V1TerminalValidationFamilyObservationReceipt.from_payload(
                values["observation"]
            )
        )
        return cls(**values)


def require_v1_matched_density_terminal_validation(
    value: object,
) -> V1TerminalValidationFamilyReceipt:
    """Refuse legacy dense/final receipts as matched-density authority."""

    if type(value) is not V1TerminalValidationFamilyReceipt:
        raise PermissionError(
            "matched-density comparison requires the exact terminal family receipt; "
            "legacy terminal validation is descriptive only"
        )
    value.__post_init__()
    return value


@dataclass(frozen=True, slots=True)
class V1TerminalComparisonFamilyFreezeReceiptV2:
    """Outcome-free dense/current-v0/terminal-v1 common-panel design.

    Unlike the legacy matched-density family, this design contains no Pareto
    milestone or checkpoint archive.  The current-v0 comparator is bound by
    both its packed artifact digest and its semantic hard-mask identity, and
    all affine states are exact validation-independent freezes.  The shared
    ``calibration_context_axis_hash`` in those freezes identifies the complete
    ImageNet-train *source axis*, not a common fitted K prefix.  Mask-specific
    K prefixes and fit outcomes remain distinct in each calibration's source
    hash and fit receipt.
    """

    terminal_freeze_hash: str
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
    dense_current_v0_calibration_family_freeze_hash: str
    final_v1_mask_hash: str
    route_axis_hash: str
    route_count: int
    role_mask_hashes: tuple[str, str, str]
    role_retained_route_counts: tuple[int, int, int]
    role_to_unique_lane: tuple[int, int, int]
    unique_lane_mask_hashes: tuple[str, ...]
    unique_lane_calibration_hashes: tuple[str, ...]
    unique_lane_calibration_freeze_hashes: tuple[str, ...]
    unique_lane_calibration_freeze_payloads: tuple[Mapping[str, Any], ...]
    freeze_sequence: int
    roles: tuple[str, str, str] = TERMINAL_COMPARISON_FAMILY_ROLES_V2
    validation_independent: bool = True
    outcome_fields_absent: bool = True
    terminal_rows_visible: bool = False
    terminal_outcomes_opened: bool = False
    terminal_outcomes_used_for_selection: bool = False
    current_v0_comparison_only: bool = True
    current_v0_candidate_or_commit_authority: bool = False
    schema: str = _TERMINAL_COMPARISON_FAMILY_FREEZE_V2_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "terminal_freeze_hash",
            "terminal_validation_population_hash",
            "terminal_validation_context_ids_hash",
            "terminal_metric_semantics_hash",
            "terminal_evaluator_identity_hash",
            "pending_dense_reference_design_hash",
            "dense_mask_hash",
            "current_v0_mask_hash",
            "current_v0_mask_artifact_semantic_hash",
            "dense_current_v0_calibration_family_freeze_hash",
            "final_v1_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        route_count = _positive_int(self.route_count, "route_count")
        context_count = _positive_int(
            self.terminal_validation_context_count,
            "terminal_validation_context_count",
        )
        current_v0_sha256 = _sha256_hex(
            self.current_v0_mask_artifact_sha256,
            "current_v0_mask_artifact_sha256",
        )
        current_v0_byte_count = _positive_int(
            self.current_v0_mask_artifact_byte_count,
            "current_v0_mask_artifact_byte_count",
        )
        expected_byte_count = (route_count + 7) // 8
        if (
            self.current_v0_mask_artifact_semantic_hash
            != self.current_v0_mask_hash
            or current_v0_byte_count != expected_byte_count
            or self.current_v0_mask_hash == self.dense_mask_hash
        ):
            raise ValueError(
                "current-v0 packed artifact differs from its semantic mask binding"
            )

        roles = tuple(self.roles)
        if roles != TERMINAL_COMPARISON_FAMILY_ROLES_V2:
            raise ValueError("terminal comparison family roles are not canonical")
        role_masks = _strings(
            self.role_mask_hashes,
            "terminal comparison role mask hash",
            unique=False,
        )
        counts = tuple(
            _positive_int(value, "terminal comparison retained-route count")
            for value in self.role_retained_route_counts
        )
        if (
            len(role_masks) != 3
            or len(counts) != 3
            or any(value > route_count for value in counts)
            or role_masks
            != (
                self.dense_mask_hash,
                self.current_v0_mask_hash,
                self.final_v1_mask_hash,
            )
            or counts[0] != route_count
            or counts[1] >= route_count
        ):
            raise ValueError(
                "terminal comparison masks/counts differ from dense/current-v0/final-v1"
            )
        for left in range(len(role_masks)):
            for right in range(left + 1, len(role_masks)):
                if (
                    role_masks[left] == role_masks[right]
                    and counts[left] != counts[right]
                ):
                    raise ValueError(
                        "deduplicated terminal comparison roles disagree on retained count"
                    )

        lane_map = tuple(
            _nonnegative_int(value, "terminal comparison role-to-lane index")
            for value in self.role_to_unique_lane
        )
        unique_masks = _strings(
            self.unique_lane_mask_hashes,
            "terminal comparison unique-lane mask hash",
            unique=True,
        )
        if (
            len(lane_map) != 3
            or not unique_masks
            or any(value >= len(unique_masks) for value in lane_map)
        ):
            raise ValueError("terminal comparison role-to-lane map is invalid")
        expected_unique: list[str] = []
        expected_map: list[int] = []
        for mask_hash in role_masks:
            if mask_hash not in expected_unique:
                expected_unique.append(mask_hash)
            expected_map.append(expected_unique.index(mask_hash))
        if tuple(expected_unique) != unique_masks or tuple(expected_map) != lane_map:
            raise ValueError(
                "terminal comparison lanes are not deterministic first occurrences"
            )

        calibration_hashes = _strings(
            self.unique_lane_calibration_hashes,
            "terminal comparison calibration hash",
            unique=True,
        )
        freeze_hashes = _strings(
            self.unique_lane_calibration_freeze_hashes,
            "terminal comparison calibration freeze hash",
            unique=True,
        )
        raw_payloads = tuple(self.unique_lane_calibration_freeze_payloads)
        if not (
            len(calibration_hashes)
            == len(freeze_hashes)
            == len(raw_payloads)
            == len(unique_masks)
        ):
            raise ValueError("terminal comparison calibration lane counts differ")
        payloads: list[Mapping[str, Any]] = []
        shared_coordinates: tuple[object, ...] | None = None
        for index, raw in enumerate(raw_payloads):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != _FROZEN_AFFINE_CALIBRATION_FIELDS
            ):
                raise ValueError(
                    "terminal comparison calibration freeze fields differ"
                )
            payload = _freeze(
                raw,
                f"terminal comparison calibration freeze payload[{index}]",
            )
            fit_receipt = payload.get("fit_receipt")
            expected_fit_receipt_hash = (
                content_hash(
                    fit_receipt,
                    prefix="partition-free-v1-frozen-calibration-fit-receipt-",
                )
                if isinstance(fit_receipt, Mapping)
                else None
            )
            target_class_id = _nonnegative_int(
                payload.get("target_class_id"),
                "terminal comparison calibration target_class_id",
            )
            scale = _finite(
                payload.get("scale"),
                "terminal comparison calibration scale",
            )
            _finite(
                payload.get("offset"),
                "terminal comparison calibration offset",
            )
            coordinates = (
                payload.get("query_id"),
                payload.get("query_hash"),
                target_class_id,
                payload.get("calibration_context_axis_hash"),
                payload.get("source_label_map_hash"),
                payload.get("model_state_hash"),
                payload.get("calibration_schema"),
            )
            if shared_coordinates is None:
                shared_coordinates = coordinates
            elif coordinates != shared_coordinates:
                raise ValueError(
                    "terminal comparison calibrations differ in shared design"
                )
            if (
                payload.get("mask_hash") != unique_masks[index]
                or payload.get("calibration_hash") != calibration_hashes[index]
                or payload.get("schema")
                != "partition_free_v1_frozen_affine_calibration_v1"
                or scale <= 0.0
                or payload.get("validation_independent") is not True
                or payload.get("terminal_rows_used_for_fit") is not False
                or payload.get("terminal_outcomes_opened") is not False
                or not isinstance(fit_receipt, Mapping)
                or fit_receipt.get("fit_succeeded") is not True
                or fit_receipt.get("mask_hash") != unique_masks[index]
                or fit_receipt.get("validation_independent") is not True
                or fit_receipt.get("terminal_rows_used_for_fit") is not False
                or fit_receipt.get("terminal_outcomes_opened") is not False
                or payload.get("fit_receipt_hash") != expected_fit_receipt_hash
                or content_hash(
                    payload,
                    prefix="partition-free-v1-frozen-affine-calibration-",
                )
                != freeze_hashes[index]
            ):
                raise ValueError(
                    "terminal comparison calibration freeze is not exact/validation-independent"
                )
            payloads.append(payload)
        sequence = _nonnegative_int(self.freeze_sequence, "freeze_sequence")
        if (
            self.validation_independent is not True
            or self.outcome_fields_absent is not True
            or self.terminal_rows_visible is not False
            or self.terminal_outcomes_opened is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.current_v0_comparison_only is not True
            or self.current_v0_candidate_or_commit_authority is not False
            or self.schema != _TERMINAL_COMPARISON_FAMILY_FREEZE_V2_SCHEMA
        ):
            raise ValueError(
                "terminal comparison freeze crossed its validation-independent firewall"
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(
            self,
            "terminal_validation_context_count",
            context_count,
        )
        object.__setattr__(
            self,
            "current_v0_mask_artifact_sha256",
            current_v0_sha256,
        )
        object.__setattr__(
            self,
            "current_v0_mask_artifact_byte_count",
            current_v0_byte_count,
        )
        object.__setattr__(self, "roles", TERMINAL_COMPARISON_FAMILY_ROLES_V2)
        object.__setattr__(self, "role_mask_hashes", role_masks)
        object.__setattr__(self, "role_retained_route_counts", counts)
        object.__setattr__(self, "role_to_unique_lane", lane_map)
        object.__setattr__(self, "unique_lane_mask_hashes", unique_masks)
        object.__setattr__(
            self,
            "unique_lane_calibration_hashes",
            calibration_hashes,
        )
        object.__setattr__(
            self,
            "unique_lane_calibration_freeze_hashes",
            freeze_hashes,
        )
        object.__setattr__(
            self,
            "unique_lane_calibration_freeze_payloads",
            tuple(payloads),
        )
        object.__setattr__(self, "freeze_sequence", sequence)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-comparison-family-freeze-v2-",
        )

    def lane_for_role(self, role: str) -> int:
        try:
            return self.role_to_unique_lane[self.roles.index(role)]
        except ValueError as exc:
            raise KeyError(role) from exc

    def to_payload(self) -> Mapping[str, Any]:
        return _payload(self)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalComparisonFamilyFreezeReceiptV2:
        return cls(
            **_strict_values(
                cls,
                payload,
                "terminal comparison family freeze v2",
            )
        )


@dataclass(frozen=True, slots=True)
class V1TerminalComparisonFamilyRequestReceiptV2:
    """Checkpointable v2 terminal request with every outcome still absent."""

    family_freeze: V1TerminalComparisonFamilyFreezeReceiptV2
    observation_artifact_relative_path: str
    observation_artifact_schema: str
    request_sequence: int
    evaluation_mode: str = (
        "single_common_panel_dense_current_v0_terminal_v1_unique_mask_lanes"
    )
    validation_independent: bool = True
    outcome_fields_absent: bool = True
    terminal_rows_visible: bool = False
    terminal_outcomes_opened: bool = False
    checkpoint_before_outcomes_required: bool = True
    schema: str = _TERMINAL_COMPARISON_FAMILY_REQUEST_V2_SCHEMA

    def __post_init__(self) -> None:
        if type(self.family_freeze) is not V1TerminalComparisonFamilyFreezeReceiptV2:
            raise TypeError(
                "terminal comparison request requires the exact v2 family freeze"
            )
        self.family_freeze.__post_init__()
        artifact_path = _relative_posix_path(
            self.observation_artifact_relative_path,
            "observation_artifact_relative_path",
        )
        artifact_schema = _nonempty(
            self.observation_artifact_schema,
            "observation_artifact_schema",
        )
        sequence = _nonnegative_int(self.request_sequence, "request_sequence")
        if sequence != self.family_freeze.freeze_sequence + 1:
            raise ValueError(
                "terminal comparison request sequence does not follow its freeze"
            )
        if (
            self.evaluation_mode
            != "single_common_panel_dense_current_v0_terminal_v1_unique_mask_lanes"
            or self.validation_independent is not True
            or self.outcome_fields_absent is not True
            or self.terminal_rows_visible is not False
            or self.terminal_outcomes_opened is not False
            or self.checkpoint_before_outcomes_required is not True
            or self.schema != _TERMINAL_COMPARISON_FAMILY_REQUEST_V2_SCHEMA
        ):
            raise ValueError(
                "terminal comparison request crossed its pre-reveal firewall"
            )
        object.__setattr__(self, "observation_artifact_relative_path", artifact_path)
        object.__setattr__(self, "observation_artifact_schema", artifact_schema)
        object.__setattr__(self, "request_sequence", sequence)

    @property
    def request_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-comparison-family-request-v2-",
        )

    @property
    def receipt_hash(self) -> str:
        return self.request_hash

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["family_freeze"] = self.family_freeze.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalComparisonFamilyRequestReceiptV2:
        values = _strict_values(
            cls,
            payload,
            "terminal comparison family request v2",
        )
        values["family_freeze"] = (
            V1TerminalComparisonFamilyFreezeReceiptV2.from_payload(
                values["family_freeze"]
            )
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1TerminalComparisonFamilyObservationReceiptV2:
    """Durable replay of one dense/current-v0/terminal-v1 matrix.

    Signed comparison deltas are means of pointwise differences on the one
    common panel.  ``from_replayed_matrix`` is the canonical constructor that
    derives every row statistic, paired statistic, and semantic hash from the
    exact persisted float64 rows.
    """

    request: V1TerminalComparisonFamilyRequestReceiptV2
    joint_validation_values_hash: str
    joint_validation_shape: tuple[int, int]
    unique_lane_validation_values_hashes: tuple[str, ...]
    unique_lane_validation_value_counts: tuple[int, ...]
    unique_lane_metrics: tuple[Mapping[str, float], ...]
    paired_mean_deltas: Mapping[str, float]
    paired_validation_difference_hashes: Mapping[str, str]
    common_panel_evaluator_receipt_hash: str
    observation_artifact_relative_path: str
    observation_artifact_sha256: str
    observation_artifact_size_bytes: int
    observation_artifact_semantic_hash: str
    observation_sequence: int
    artifact_durable_before_receipt: bool = True
    artifact_replay_verified: bool = True
    single_common_panel_evaluation: bool = True
    raw_validation_rows_persisted: bool = True
    paired_delta_derivation: str = (
        "mean_of_pointwise_common_panel_row_differences_float64_v1"
    )
    fed_back_into_selection: bool = False
    terminal_outcomes_used_for_selection: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_COMPARISON_FAMILY_OBSERVATION_V2_SCHEMA

    @staticmethod
    def derive_artifact_semantic_hash(
        *,
        terminal_request_hash: str,
        joint_validation_values_hash: str,
        joint_validation_shape: Sequence[int],
        unique_lane_validation_values_hashes: Sequence[str],
        unique_lane_validation_value_counts: Sequence[int],
        unique_lane_metrics: Sequence[Mapping[str, float]],
        paired_mean_deltas: Mapping[str, float],
        paired_validation_difference_hashes: Mapping[str, str],
        common_panel_evaluator_receipt_hash: str,
    ) -> str:
        return content_hash(
            {
                "terminal_request_hash": terminal_request_hash,
                "joint_validation_values_hash": joint_validation_values_hash,
                "joint_validation_shape": tuple(joint_validation_shape),
                "unique_lane_validation_values_hashes": tuple(
                    unique_lane_validation_values_hashes
                ),
                "unique_lane_validation_value_counts": tuple(
                    unique_lane_validation_value_counts
                ),
                "unique_lane_metrics": tuple(unique_lane_metrics),
                "paired_mean_deltas": paired_mean_deltas,
                "paired_validation_difference_hashes": (
                    paired_validation_difference_hashes
                ),
                "paired_delta_derivation": (
                    "mean_of_pointwise_common_panel_row_differences_float64_v1"
                ),
                "common_panel_evaluator_receipt_hash": (
                    common_panel_evaluator_receipt_hash
                ),
            },
            prefix=(
                "partition-free-v1-terminal-comparison-family-"
                "observation-artifact-v2-"
            ),
        )

    @classmethod
    def from_replayed_matrix(
        cls,
        *,
        request: V1TerminalComparisonFamilyRequestReceiptV2,
        validation_rows: Sequence[Sequence[float]],
        common_panel_evaluator_receipt_hash: str,
        observation_artifact_relative_path: str,
        observation_artifact_sha256: str,
        observation_artifact_size_bytes: int,
    ) -> V1TerminalComparisonFamilyObservationReceiptV2:
        """Derive an observation only from already-durable replayed rows."""

        if type(request) is not V1TerminalComparisonFamilyRequestReceiptV2:
            raise TypeError(
                "terminal comparison matrix replay requires the exact v2 request"
            )
        request.__post_init__()
        freeze = request.family_freeze
        try:
            raw_rows = tuple(validation_rows)
        except TypeError as exc:
            raise TypeError("terminal comparison matrix must be row-iterable") from exc
        if len(raw_rows) != len(freeze.unique_lane_mask_hashes):
            raise ValueError(
                "terminal comparison matrix differs from its frozen lane count"
            )
        rows: list[tuple[float, ...]] = []
        row_payloads: list[bytes] = []
        for lane_index, raw_row in enumerate(raw_rows):
            try:
                row = tuple(
                    _finite_nonnegative(
                        value,
                        f"terminal comparison lane {lane_index} value",
                    )
                    for value in raw_row
                )
            except TypeError as exc:
                raise TypeError(
                    "terminal comparison matrix rows must be iterable"
                ) from exc
            if len(row) != freeze.terminal_validation_context_count:
                raise ValueError(
                    "terminal comparison matrix differs from its frozen context count"
                )
            rows.append(row)
            row_payloads.append(_little_endian_float64_payload(row))
        shape = (len(rows), freeze.terminal_validation_context_count)
        joint_hash = content_hash(
            {"shape": shape, "payload": b"".join(row_payloads)},
            prefix=_TERMINAL_COMPARISON_FAMILY_VALUES_V2_PREFIX,
        )
        lane_hashes = tuple(
            content_hash(
                {"shape": (len(row),), "payload": payload},
                prefix=_TERMINAL_COMPARISON_FAMILY_LANE_VALUES_V2_PREFIX,
            )
            for row, payload in zip(rows, row_payloads, strict=True)
        )
        metrics = tuple(
            {
                "mean": math.fsum(row) / len(row),
                "maximum": max(row),
                "root_mean_square": math.sqrt(
                    math.fsum(value * value for value in row) / len(row)
                ),
            }
            for row in rows
        )
        for lane_metrics in metrics:
            for key, value in lane_metrics.items():
                _finite_nonnegative(value, f"terminal comparison metric {key}")

        pair_definitions = (
            (
                TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[0],
                "current_v0_comparator",
                "dense_reference",
            ),
            (
                TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[1],
                "terminal_v1",
                "dense_reference",
            ),
            (
                TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[2],
                "terminal_v1",
                "current_v0_comparator",
            ),
        )
        paired_means: dict[str, float] = {}
        paired_hashes: dict[str, str] = {}
        for pair_name, left_role, right_role in pair_definitions:
            left = rows[freeze.lane_for_role(left_role)]
            right = rows[freeze.lane_for_role(right_role)]
            differences = tuple(
                left_value - right_value
                for left_value, right_value in zip(left, right, strict=True)
            )
            paired_means[pair_name] = math.fsum(differences) / len(differences)
            paired_hashes[pair_name] = content_hash(
                {
                    "shape": (len(differences),),
                    "payload": _little_endian_float64_payload(differences),
                },
                prefix=_TERMINAL_COMPARISON_FAMILY_PAIRED_VALUES_V2_PREFIX,
            )
        evaluator_hash = _nonempty(
            common_panel_evaluator_receipt_hash,
            "common_panel_evaluator_receipt_hash",
        )
        semantic_hash = cls.derive_artifact_semantic_hash(
            terminal_request_hash=request.request_hash,
            joint_validation_values_hash=joint_hash,
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=lane_hashes,
            unique_lane_validation_value_counts=(shape[1],) * shape[0],
            unique_lane_metrics=metrics,
            paired_mean_deltas=paired_means,
            paired_validation_difference_hashes=paired_hashes,
            common_panel_evaluator_receipt_hash=evaluator_hash,
        )
        return cls(
            request=request,
            joint_validation_values_hash=joint_hash,
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=lane_hashes,
            unique_lane_validation_value_counts=(shape[1],) * shape[0],
            unique_lane_metrics=metrics,
            paired_mean_deltas=paired_means,
            paired_validation_difference_hashes=paired_hashes,
            common_panel_evaluator_receipt_hash=evaluator_hash,
            observation_artifact_relative_path=(
                observation_artifact_relative_path
            ),
            observation_artifact_sha256=observation_artifact_sha256,
            observation_artifact_size_bytes=observation_artifact_size_bytes,
            observation_artifact_semantic_hash=semantic_hash,
            observation_sequence=request.request_sequence + 1,
        )

    def __post_init__(self) -> None:
        if type(self.request) is not V1TerminalComparisonFamilyRequestReceiptV2:
            raise TypeError(
                "terminal comparison observation requires the exact v2 request"
            )
        self.request.__post_init__()
        for name in (
            "joint_validation_values_hash",
            "common_panel_evaluator_receipt_hash",
            "observation_artifact_semantic_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        artifact_path = _relative_posix_path(
            self.observation_artifact_relative_path,
            "observation_artifact_relative_path",
        )
        artifact_sha256 = _sha256_hex(
            self.observation_artifact_sha256,
            "observation_artifact_sha256",
        )
        artifact_size = _positive_int(
            self.observation_artifact_size_bytes,
            "observation_artifact_size_bytes",
        )
        freeze = self.request.family_freeze
        lane_count = len(freeze.unique_lane_mask_hashes)
        shape = tuple(
            _positive_int(value, "joint_validation_shape")
            for value in self.joint_validation_shape
        )
        hashes = _strings(
            self.unique_lane_validation_values_hashes,
            "terminal comparison lane values hash",
            unique=False,
        )
        counts = tuple(
            _positive_int(value, "terminal comparison lane value count")
            for value in self.unique_lane_validation_value_counts
        )
        raw_metrics = tuple(self.unique_lane_metrics)
        if (
            shape != (lane_count, freeze.terminal_validation_context_count)
            or not (len(hashes) == len(counts) == len(raw_metrics) == lane_count)
            or any(value != freeze.terminal_validation_context_count for value in counts)
        ):
            raise ValueError("terminal comparison outcome lanes are incoherent")
        metrics: list[Mapping[str, float]] = []
        for raw in raw_metrics:
            if not isinstance(raw, Mapping) or set(raw) != {
                "mean",
                "maximum",
                "root_mean_square",
            }:
                raise ValueError("terminal comparison lane metrics differ")
            metrics.append(
                MappingProxyType(
                    {
                        key: _finite_nonnegative(value, f"metric {key}")
                        for key, value in raw.items()
                    }
                )
            )

        raw_deltas = self.paired_mean_deltas
        raw_pair_hashes = self.paired_validation_difference_hashes
        expected_pair_keys = set(TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2)
        if (
            not isinstance(raw_deltas, Mapping)
            or not isinstance(raw_pair_hashes, Mapping)
            or set(raw_deltas) != expected_pair_keys
            or set(raw_pair_hashes) != expected_pair_keys
        ):
            raise ValueError("terminal comparison paired-delta fields differ")
        deltas = MappingProxyType(
            {
                key: _finite(raw_deltas[key], f"paired delta {key}")
                for key in TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2
            }
        )
        pair_hashes = MappingProxyType(
            {
                key: _nonempty(
                    raw_pair_hashes[key],
                    f"paired validation difference hash {key}",
                )
                for key in TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2
            }
        )
        role_means = {
            role: metrics[freeze.lane_for_role(role)]["mean"]
            for role in TERMINAL_COMPARISON_FAMILY_ROLES_V2
        }
        expected_deltas = {
            TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[0]: (
                role_means["current_v0_comparator"]
                - role_means["dense_reference"]
            ),
            TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[1]: (
                role_means["terminal_v1"] - role_means["dense_reference"]
            ),
            TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2[2]: (
                role_means["terminal_v1"]
                - role_means["current_v0_comparator"]
            ),
        }
        if any(
            not _close_at_float64_roundoff(deltas[key], expected)
            for key, expected in expected_deltas.items()
        ):
            raise ValueError(
                "terminal comparison paired means disagree with common-panel lane means"
            )
        pair_roles = (
            ("current_v0_comparator", "dense_reference"),
            ("terminal_v1", "dense_reference"),
            ("terminal_v1", "current_v0_comparator"),
        )
        zero_pair_hash = content_hash(
            {
                "shape": (freeze.terminal_validation_context_count,),
                "payload": b"\x00" * (8 * freeze.terminal_validation_context_count),
            },
            prefix=_TERMINAL_COMPARISON_FAMILY_PAIRED_VALUES_V2_PREFIX,
        )
        for key, (left_role, right_role) in zip(
            TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2,
            pair_roles,
            strict=True,
        ):
            if freeze.lane_for_role(left_role) == freeze.lane_for_role(right_role) and (
                deltas[key] != 0.0 or pair_hashes[key] != zero_pair_hash
            ):
                raise ValueError(
                    "deduplicated terminal comparison roles have a nonzero paired delta"
                )

        expected_semantic_hash = self.derive_artifact_semantic_hash(
            terminal_request_hash=self.request.request_hash,
            joint_validation_values_hash=self.joint_validation_values_hash,
            joint_validation_shape=shape,
            unique_lane_validation_values_hashes=hashes,
            unique_lane_validation_value_counts=counts,
            unique_lane_metrics=metrics,
            paired_mean_deltas=deltas,
            paired_validation_difference_hashes=pair_hashes,
            common_panel_evaluator_receipt_hash=(
                self.common_panel_evaluator_receipt_hash
            ),
        )
        sequence = _nonnegative_int(self.observation_sequence, "observation_sequence")
        if (
            sequence != self.request.request_sequence + 1
            or artifact_path != self.request.observation_artifact_relative_path
            or self.observation_artifact_semantic_hash != expected_semantic_hash
            or self.artifact_durable_before_receipt is not True
            or self.artifact_replay_verified is not True
            or self.single_common_panel_evaluation is not True
            or self.raw_validation_rows_persisted is not True
            or self.paired_delta_derivation
            != "mean_of_pointwise_common_panel_row_differences_float64_v1"
            or self.fed_back_into_selection is not False
            or self.terminal_outcomes_used_for_selection is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
            or self.schema != _TERMINAL_COMPARISON_FAMILY_OBSERVATION_V2_SCHEMA
        ):
            raise ValueError(
                "terminal comparison observation crossed its durable-replay firewall"
            )
        object.__setattr__(self, "joint_validation_shape", shape)
        object.__setattr__(self, "unique_lane_validation_values_hashes", hashes)
        object.__setattr__(self, "unique_lane_validation_value_counts", counts)
        object.__setattr__(self, "unique_lane_metrics", tuple(metrics))
        object.__setattr__(self, "paired_mean_deltas", deltas)
        object.__setattr__(
            self,
            "paired_validation_difference_hashes",
            pair_hashes,
        )
        object.__setattr__(self, "observation_artifact_relative_path", artifact_path)
        object.__setattr__(self, "observation_artifact_sha256", artifact_sha256)
        object.__setattr__(self, "observation_artifact_size_bytes", artifact_size)
        object.__setattr__(self, "observation_sequence", sequence)

    @property
    def terminal_outcome_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-comparison-family-outcome-v2-",
        )

    @property
    def receipt_hash(self) -> str:
        return self.terminal_outcome_hash

    def metrics_for_role(self, role: str) -> Mapping[str, float]:
        lane = self.request.family_freeze.lane_for_role(role)
        return self.unique_lane_metrics[lane]

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["request"] = self.request.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalComparisonFamilyObservationReceiptV2:
        values = _strict_values(
            cls,
            payload,
            "terminal comparison family observation v2",
        )
        values["request"] = V1TerminalComparisonFamilyRequestReceiptV2.from_payload(
            values["request"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1TerminalComparisonFamilyReceiptV2:
    """Post-replay authority for the one matched dense/v0/v1 matrix."""

    observation: V1TerminalComparisonFamilyObservationReceiptV2
    reveal_sequence: int
    single_reveal: bool = True
    terminal_comparison_authority: bool = True
    dense_reference_resolved: bool = True
    fed_back_into_selection: bool = False
    claim_execution_certified: bool = False
    mask_commit_authority: bool = False
    schema: str = _TERMINAL_COMPARISON_FAMILY_V2_SCHEMA

    def __post_init__(self) -> None:
        if type(self.observation) is not V1TerminalComparisonFamilyObservationReceiptV2:
            raise TypeError(
                "terminal comparison receipt requires the exact v2 observation"
            )
        self.observation.__post_init__()
        sequence = _nonnegative_int(self.reveal_sequence, "reveal_sequence")
        if (
            sequence != self.observation.observation_sequence
            or self.single_reveal is not True
            or self.terminal_comparison_authority is not True
            or self.dense_reference_resolved is not True
            or self.fed_back_into_selection is not False
            or self.claim_execution_certified is not False
            or self.mask_commit_authority is not False
            or self.schema != _TERMINAL_COMPARISON_FAMILY_V2_SCHEMA
        ):
            raise ValueError(
                "terminal comparison receipt crossed its post-replay authority firewall"
            )
        object.__setattr__(self, "reveal_sequence", sequence)

    @property
    def request(self) -> V1TerminalComparisonFamilyRequestReceiptV2:
        return self.observation.request

    @property
    def family_freeze(self) -> V1TerminalComparisonFamilyFreezeReceiptV2:
        return self.request.family_freeze

    @property
    def family_freeze_hash(self) -> str:
        return self.family_freeze.receipt_hash

    @property
    def terminal_freeze_hash(self) -> str:
        return self.family_freeze.terminal_freeze_hash

    @property
    def terminal_request_hash(self) -> str:
        return self.request.request_hash

    @property
    def terminal_outcome_hash(self) -> str:
        return self.observation.terminal_outcome_hash

    @property
    def terminal_validation_population_hash(self) -> str:
        return self.family_freeze.terminal_validation_population_hash

    @property
    def roles(self) -> tuple[str, str, str]:
        return self.family_freeze.roles

    @property
    def role_to_unique_lane(self) -> tuple[int, int, int]:
        return self.family_freeze.role_to_unique_lane

    @property
    def unique_lane_mask_hashes(self) -> tuple[str, ...]:
        return self.family_freeze.unique_lane_mask_hashes

    @property
    def measured_dense_reference_value(self) -> float:
        return self.observation.metrics_for_role("dense_reference")["mean"]

    @property
    def measured_dense_reference_hash(self) -> str:
        dense_lane = self.family_freeze.lane_for_role("dense_reference")
        return content_hash(
            {
                "schema": _MEASURED_DENSE_REFERENCE_V2_SCHEMA,
                "pending_dense_reference_design_hash": (
                    self.family_freeze.pending_dense_reference_design_hash
                ),
                "terminal_request_hash": self.terminal_request_hash,
                "terminal_outcome_hash": self.terminal_outcome_hash,
                "terminal_metric_semantics_hash": (
                    self.family_freeze.terminal_metric_semantics_hash
                ),
                "terminal_validation_population_hash": (
                    self.terminal_validation_population_hash
                ),
                "dense_mask_hash": self.family_freeze.dense_mask_hash,
                "dense_calibration_hash": (
                    self.family_freeze.unique_lane_calibration_hashes[dense_lane]
                ),
                "dense_lane_validation_values_hash": (
                    self.observation.unique_lane_validation_values_hashes[dense_lane]
                ),
                "joint_validation_values_hash": (
                    self.observation.joint_validation_values_hash
                ),
                "dense_reference_value": self.measured_dense_reference_value,
            },
            prefix="partition-free-v1-measured-dense-reference-v2-",
        )

    @property
    def paired_mean_deltas(self) -> Mapping[str, float]:
        return self.observation.paired_mean_deltas

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-terminal-comparison-family-v2-",
        )

    def metrics_for_role(self, role: str) -> Mapping[str, float]:
        return self.observation.metrics_for_role(role)

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["observation"] = self.observation.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1TerminalComparisonFamilyReceiptV2:
        values = _strict_values(
            cls,
            payload,
            "terminal comparison family receipt v2",
        )
        values["observation"] = (
            V1TerminalComparisonFamilyObservationReceiptV2.from_payload(
                values["observation"]
            )
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1PerformanceReceipt:
    """Separate logical/controller/accelerator costs without a speedup claim."""

    controller_run_id: str
    source_environment_hash: str
    cold_staging_seconds: float
    cold_compile_seconds: float
    warm_acquisition_seconds: float
    warm_decision_seconds: float
    warm_terminal_seconds: float
    peak_device_memory_bytes: int
    peak_host_rss_bytes: int
    host_transfer_bytes: int
    gpu_utilization_fraction: float | None
    fallback_used: bool
    fallback_reason: str | None
    structural_lowering_receipt_hash: str | None = None
    claim_execution_certified: bool = False
    schema: str = _PERFORMANCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "controller_run_id",
            _nonempty(self.controller_run_id, "controller_run_id"),
        )
        object.__setattr__(
            self,
            "source_environment_hash",
            _nonempty(self.source_environment_hash, "source_environment_hash"),
        )
        for name in (
            "cold_staging_seconds",
            "cold_compile_seconds",
            "warm_acquisition_seconds",
            "warm_decision_seconds",
            "warm_terminal_seconds",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in (
            "peak_device_memory_bytes",
            "peak_host_rss_bytes",
            "host_transfer_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.gpu_utilization_fraction is not None:
            utilization = float(self.gpu_utilization_fraction)
            if not math.isfinite(utilization) or not 0.0 <= utilization <= 1.0:
                raise ValueError("gpu_utilization_fraction must lie in [0,1]")
            object.__setattr__(self, "gpu_utilization_fraction", utilization)
        if type(self.fallback_used) is not bool:
            raise TypeError("fallback_used must be bool")
        if self.fallback_used:
            object.__setattr__(
                self,
                "fallback_reason",
                _nonempty(self.fallback_reason, "fallback_reason"),
            )
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason requires fallback_used=True")
        if self.structural_lowering_receipt_hash is not None:
            object.__setattr__(
                self,
                "structural_lowering_receipt_hash",
                _nonempty(
                    self.structural_lowering_receipt_hash,
                    "structural_lowering_receipt_hash",
                ),
            )
        if self.schema != _PERFORMANCE_SCHEMA or self.claim_execution_certified is not False:
            raise ValueError("performance receipt crossed its evidence ceiling")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-performance-")


@dataclass(frozen=True, slots=True)
class V1RunCompletionReceipt:
    """Top-level completed development-run identity; never selector authority."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    final_controller_snapshot_hash: str
    terminal_freeze_hash: str
    terminal_validation_hash: str
    terminal_mask_hash: str
    accepted_step_count: int
    candidate_attempt_count: int
    performance_receipt_hash: str | None
    source_seal_hash: str
    environment_hash: str
    completion_sequence: int
    workflow_identity: Literal["irg/v1"] = "irg/v1"
    genuine_partition_free_controller: bool = True
    direct_rule_over_v0_adapter: bool = False
    retired_trust_walk: bool = False
    claim_execution_certified: bool = False
    selector_promoted: bool = False
    schema: str = _COMPLETION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "final_controller_snapshot_hash",
            "terminal_freeze_hash",
            "terminal_validation_hash",
            "terminal_mask_hash",
            "source_seal_hash",
            "environment_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "accepted_step_count",
            "candidate_attempt_count",
            "completion_sequence",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        if self.performance_receipt_hash is not None:
            object.__setattr__(
                self,
                "performance_receipt_hash",
                _nonempty(self.performance_receipt_hash, "performance_receipt_hash"),
            )
        if (
            self.schema != _COMPLETION_SCHEMA
            or self.workflow_identity != "irg/v1"
            or self.genuine_partition_free_controller is not True
            or self.direct_rule_over_v0_adapter is not False
            or self.retired_trust_walk is not False
            or self.claim_execution_certified is not False
            or self.selector_promoted is not False
        ):
            raise ValueError("completion receipt mislabels or promotes the development run")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-run-completion-")


__all__ = [
    "ControllerPhase",
    "DecisionAction",
    "TERMINAL_COMPARISON_FAMILY_PAIRED_DELTAS_V2",
    "TERMINAL_COMPARISON_FAMILY_ROLES_V2",
    "TERMINAL_VALIDATION_FAMILY_ROLES",
    "V1AcquisitionGenerationReceipt",
    "V1CandidateAttemptReceipt",
    "V1ControllerSnapshotReceipt",
    "V1MaskTransitionReceipt",
    "V1PerformanceReceipt",
    "V1RunCompletionReceipt",
    "V1TerminalFreezeReceipt",
    "V1TerminalComparisonFamilyFreezeReceiptV2",
    "V1TerminalComparisonFamilyObservationReceiptV2",
    "V1TerminalComparisonFamilyReceiptV2",
    "V1TerminalComparisonFamilyRequestReceiptV2",
    "V1TerminalValidationFamilyFreezeReceipt",
    "V1TerminalValidationFamilyObservationReceipt",
    "V1TerminalValidationFamilyReceipt",
    "V1TerminalValidationFamilyRequestReceipt",
    "V1TerminalValidationReceipt",
    "require_v1_matched_density_terminal_validation",
]
