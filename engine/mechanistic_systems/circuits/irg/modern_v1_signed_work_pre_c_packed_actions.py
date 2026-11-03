"""Coordinate-aware custody for already-materialized signed-work actions.

Phase A chooses endogenous resource coordinates and supporting Q branches;
Phase B plus complete hard F/Q replay then yields coordinate/branch endpoint
aliases.  Device-shell materialization produces an ordinary
``V1PackedRemovalAction`` only for canonically Q-safe endpoints.  This module
groups those aliases by exact physical action while retaining every coordinate
provenance record.

The grouping key includes the SHA-256 of the actual packed child bytes, all
canonical action identities, and the removed count.  Matching digests are not
enough: the builder also compares the packed bytes before sharing one action
object.  Robustly infeasible and ambiguous aliases remain in the family but
can never carry a payload.  Nothing here opens C, K, O, commit, terminal, or
Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from numbers import Integral
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_c_bridge import (
    V1SignedWorkCActionMaterializationReceipt,
    V1SignedWorkCEndpointDecisionReceipt,
    V1SignedWorkCSingleBranchMaterializationBundle,
    V1SignedWorkCSingleBranchPackedAction,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointStatus,
)


SIGNED_WORK_PRE_C_PACKED_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_signed_work_coordinate_candidate_packed_crosswalk_v1"
)
SIGNED_WORK_PRE_C_CANONICAL_PACKED_ACTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_canonical_packed_physical_action_v1"
)
SIGNED_WORK_PRE_C_PACKED_ACTION_FAMILY_SCHEMA: Final = (
    "partition_free_v1_signed_work_coordinate_aware_packed_action_family_v1"
)
SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY: Final = (
    "pre_C_coordinate_alias_to_exact_packed_physical_action_identity_only_"
    "no_C_K_O_calibration_oracle_commit_terminal_or_pareto_authority_v1"
)

_CROSSWALK_PREFIX: Final = "partition-free-v1-signed-work-packed-crosswalk-"
_CANONICAL_PREFIX: Final = "partition-free-v1-signed-work-canonical-action-"
_FAMILY_PREFIX: Final = "partition-free-v1-signed-work-packed-action-family-"

V1SignedWorkPackedActionDedupIdentity = tuple[str, str, str, str, int]


class V1SignedWorkPreCPackedActionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    Q_STATUS_PAYLOAD_VIOLATION = "q_status_payload_violation"
    PACKED_BYTE_MISMATCH = "packed_byte_mismatch"
    INCOMPLETE_ALIAS_PARTITION = "incomplete_alias_partition"
    WORKSPACE_ACCOUNTING_MISMATCH = "workspace_accounting_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkPreCPackedActionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkPreCPackedActionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkPreCPackedActionRefusalCode:
            raise TypeError("packed-action refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkPreCPackedActionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkPreCPackedActionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _closed_authority(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority flags must be exact booleans")
    if any(values):
        _refuse(
            V1SignedWorkPreCPackedActionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its pre-C representation boundary",
        )


def v1_signed_work_packed_action_dedup_identity(
    action: V1PackedRemovalAction,
) -> V1SignedWorkPackedActionDedupIdentity:
    """Return the complete representation-and-action identity required by v1."""

    if type(action) is not V1PackedRemovalAction:
        raise TypeError("packed-action identity requires an exact action")
    return (
        hashlib.sha256(action.resulting_mask.packed_bits).hexdigest(),
        action.action_hash,
        action.resulting_mask.mask_hash,
        action.removed_route_index_hash,
        action.removed_count,
    )


def _sequential_host_peak(
    source_bounds: dict[str, tuple[int, int, int, int]],
) -> int:
    """Peak while each source payload remains live for later sources."""

    cumulative_retained = 0
    peak = 0
    for retained, source_peak, _cuda_peak, _d2h in source_bounds.values():
        peak = max(peak, cumulative_retained + source_peak)
        cumulative_retained += retained
    return peak


@dataclass(frozen=True, slots=True)
class V1SignedWorkCoordinateCandidateCrosswalk:
    """One coordinate/branch alias and its optional exact Q-safe payload."""

    coordinate_candidate_key_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    q_branch: V1RelaxedKKTBranch
    source_endpoint_label: int
    source_materialization_bundle_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    endpoint_decision: V1SignedWorkCEndpointDecisionReceipt
    source_packed_action: V1SignedWorkCSingleBranchPackedAction | None = field(
        repr=False,
        compare=False,
    )
    packed_action: V1PackedRemovalAction | None = field(
        repr=False,
        compare=False,
    )
    materialization_receipt: V1SignedWorkCActionMaterializationReceipt | None
    source_retained_host_packed_payload_bytes: int
    source_peak_host_workspace_upper_bound_bytes: int
    source_cuda_logical_workspace_upper_bound_bytes: int
    source_packed_payload_d2h_bytes: int
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_PACKED_CROSSWALK_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "coordinate_candidate_key_hash",
            "resource_coordinate_hash",
            "source_materialization_bundle_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "coordinate_ordinal",
            _positive_int(self.coordinate_ordinal, name="coordinate ordinal"),
        )
        object.__setattr__(
            self,
            "source_endpoint_label",
            _positive_int(self.source_endpoint_label, name="source endpoint label"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        for name in (
            "source_retained_host_packed_payload_bytes",
            "source_peak_host_workspace_upper_bound_bytes",
            "source_cuda_logical_workspace_upper_bound_bytes",
            "source_packed_payload_d2h_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if type(self.q_branch) is not V1RelaxedKKTBranch:
            raise TypeError("coordinate crosswalk requires an exact Q branch")
        if type(self.endpoint_decision) is not V1SignedWorkCEndpointDecisionReceipt:
            raise TypeError("coordinate crosswalk requires an exact endpoint decision")
        decision = self.endpoint_decision
        if (
            decision.branch is not self.q_branch
            or decision.endpoint_label != self.source_endpoint_label
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
                "coordinate alias and endpoint decision differ",
            )
        safe = (
            decision.canonical_status
            is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        )
        if safe:
            if (
                type(self.source_packed_action)
                is not V1SignedWorkCSingleBranchPackedAction
                or type(self.packed_action) is not V1PackedRemovalAction
                or type(self.materialization_receipt)
                is not V1SignedWorkCActionMaterializationReceipt
            ):
                _refuse(
                    V1SignedWorkPreCPackedActionRefusalCode.Q_STATUS_PAYLOAD_VIOLATION,
                    "a Q-safe alias must retain its exact action and materialization",
                )
            action = self.packed_action
            receipt = self.materialization_receipt
            source = self.source_packed_action
            if (
                source.endpoint_decision.receipt_hash != decision.receipt_hash
                or source.hard_fq_endpoint.endpoint_label != self.source_endpoint_label
                or source.packed_action.action_hash != action.action_hash
                or source.packed_action.resulting_mask.packed_bits
                != action.resulting_mask.packed_bits
                or source.materialization.receipt_hash != receipt.receipt_hash
                or receipt.branch is not self.q_branch
                or receipt.endpoint_label != self.source_endpoint_label
                or receipt.hard_replay_endpoint_receipt_hash
                != decision.hard_replay_endpoint_receipt_hash
                or receipt.route_axis_hash != self.route_axis_hash
                or receipt.route_count != self.route_count
                or receipt.parent_mask_hash != self.parent_mask_hash
                or receipt.packed_action_hash != action.action_hash
                or receipt.resulting_mask_hash != action.resulting_mask.mask_hash
                or receipt.removed_route_index_hash != action.removed_route_index_hash
                or receipt.resulting_mask_packed_sha256
                != hashlib.sha256(action.resulting_mask.packed_bits).hexdigest()
                or action.resulting_mask.route_axis_hash != self.route_axis_hash
                or action.resulting_mask.route_count != self.route_count
                or action.removed_count != decision.cumulative_removed_count
            ):
                _refuse(
                    V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
                    "coordinate alias payload differs from its exact source receipt",
                )
        elif (
            self.source_packed_action is not None
            or self.packed_action is not None
            or self.materialization_receipt is not None
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.Q_STATUS_PAYLOAD_VIOLATION,
                "robust-infeasible or ambiguous aliases cannot carry payload",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="coordinate candidate crosswalk",
        )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_PACKED_CROSSWALK_SCHEMA
        ):
            raise ValueError("unsupported coordinate packed-crosswalk semantics")

    @property
    def q_status(self) -> V1SignedWorkHardFQEndpointStatus:
        return self.endpoint_decision.canonical_status

    @property
    def crosswalk_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "coordinate_candidate_key_hash": self.coordinate_candidate_key_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "q_branch": self.q_branch,
                "source_endpoint_label": self.source_endpoint_label,
                "source_materialization_bundle_hash": (
                    self.source_materialization_bundle_hash
                ),
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "endpoint_decision_receipt_hash": self.endpoint_decision.receipt_hash,
                "source_packed_action_hash": (
                    None
                    if self.source_packed_action is None
                    else self.source_packed_action.action_hash
                ),
                "packed_action_hash": (
                    None
                    if self.packed_action is None
                    else self.packed_action.action_hash
                ),
                "resulting_mask_packed_sha256": (
                    None
                    if self.packed_action is None
                    else hashlib.sha256(
                        self.packed_action.resulting_mask.packed_bits
                    ).hexdigest()
                ),
                "materialization_receipt_hash": (
                    None
                    if self.materialization_receipt is None
                    else self.materialization_receipt.receipt_hash
                ),
                "source_retained_host_packed_payload_bytes": (
                    self.source_retained_host_packed_payload_bytes
                ),
                "source_peak_host_workspace_upper_bound_bytes": (
                    self.source_peak_host_workspace_upper_bound_bytes
                ),
                "source_cuda_logical_workspace_upper_bound_bytes": (
                    self.source_cuda_logical_workspace_upper_bound_bytes
                ),
                "source_packed_payload_d2h_bytes": (
                    self.source_packed_payload_d2h_bytes
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_CROSSWALK_PREFIX,
        )


def crosswalk_v1_signed_work_coordinate_candidate(
    *,
    coordinate_candidate_key_hash: str,
    resource_coordinate_hash: str,
    coordinate_ordinal: int,
    materialization: V1SignedWorkCSingleBranchMaterializationBundle,
    source_endpoint_label: int,
) -> V1SignedWorkCoordinateCandidateCrosswalk:
    """Bind one coordinate alias to an already-materialized branch endpoint."""

    if type(materialization) is not V1SignedWorkCSingleBranchMaterializationBundle:
        raise TypeError("coordinate crosswalk requires an exact materialization bundle")
    label = _positive_int(source_endpoint_label, name="source endpoint label")
    decisions = tuple(
        value
        for value in materialization.endpoint_decisions
        if value.endpoint_label == label
    )
    if len(decisions) != 1:
        raise KeyError(label)
    packed = tuple(
        value
        for value in materialization.packed_actions
        if value.hard_fq_endpoint.endpoint_label == label
    )
    if len(packed) > 1:
        _refuse(
            V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
            "single-branch materialization repeats an endpoint payload",
        )
    source = packed[0] if packed else None
    return V1SignedWorkCoordinateCandidateCrosswalk(
        coordinate_candidate_key_hash=coordinate_candidate_key_hash,
        resource_coordinate_hash=resource_coordinate_hash,
        coordinate_ordinal=coordinate_ordinal,
        q_branch=materialization.branch,
        source_endpoint_label=label,
        source_materialization_bundle_hash=materialization.bundle_hash,
        parent_mask_hash=materialization.parent_mask_hash,
        route_axis_hash=materialization.route_axis_hash,
        route_count=materialization.route_count,
        endpoint_decision=decisions[0],
        source_packed_action=source,
        packed_action=None if source is None else source.packed_action,
        materialization_receipt=(None if source is None else source.materialization),
        source_retained_host_packed_payload_bytes=(
            materialization.retained_host_packed_payload_bytes
        ),
        source_peak_host_workspace_upper_bound_bytes=(
            materialization.peak_host_workspace_upper_bound_bytes
        ),
        source_cuda_logical_workspace_upper_bound_bytes=(
            materialization.cuda_logical_workspace_upper_bound_bytes
        ),
        source_packed_payload_d2h_bytes=(materialization.packed_payload_d2h_bytes),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkCanonicalPackedPhysicalAction:
    """One physical packed action with every coordinate/branch alias retained."""

    canonical_ordinal: int
    dedup_identity: V1SignedWorkPackedActionDedupIdentity
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    aliases: tuple[V1SignedWorkCoordinateCandidateCrosswalk, ...]
    packed_payload_bytes: int
    actual_packed_byte_equality_verified: bool = True
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_CANONICAL_PACKED_ACTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "canonical_ordinal",
            _positive_int(self.canonical_ordinal, name="canonical action ordinal"),
        )
        if type(self.packed_action) is not V1PackedRemovalAction:
            raise TypeError("canonical physical action requires an exact packed action")
        identity = tuple(self.dedup_identity)
        if identity != v1_signed_work_packed_action_dedup_identity(self.packed_action):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
                "canonical physical-action dedup identity does not replay",
            )
        object.__setattr__(self, "dedup_identity", identity)
        aliases = tuple(self.aliases)
        if not aliases or any(
            type(value) is not V1SignedWorkCoordinateCandidateCrosswalk
            for value in aliases
        ):
            raise TypeError("canonical physical action requires exact aliases")
        if len({value.crosswalk_hash for value in aliases}) != len(aliases):
            raise ValueError("canonical physical action repeats an alias")
        canonical_bytes = self.packed_action.resulting_mask.packed_bits
        for alias in aliases:
            if (
                alias.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
                or type(alias.packed_action) is not V1PackedRemovalAction
                or v1_signed_work_packed_action_dedup_identity(alias.packed_action)
                != identity
                or alias.packed_action.resulting_mask.packed_bits != canonical_bytes
                or alias.packed_action is not self.packed_action
            ):
                _refuse(
                    V1SignedWorkPreCPackedActionRefusalCode.PACKED_BYTE_MISMATCH,
                    "canonical action aliases are not one exact shared packed payload",
                )
        payload = _positive_int(self.packed_payload_bytes, name="packed payload bytes")
        if payload != len(canonical_bytes):
            raise ValueError("canonical packed payload byte count does not replay")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "packed_payload_bytes", payload)
        if (
            type(self.actual_packed_byte_equality_verified) is not bool
            or not self.actual_packed_byte_equality_verified
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.PACKED_BYTE_MISMATCH,
                "canonical action did not verify actual packed-byte equality",
            )
        _closed_authority(
            (
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="canonical packed physical action",
        )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_CANONICAL_PACKED_ACTION_SCHEMA
        ):
            raise ValueError("unsupported canonical packed-action semantics")

    @property
    def canonical_action_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "canonical_ordinal": self.canonical_ordinal,
                "dedup_identity": self.dedup_identity,
                "alias_crosswalk_hashes": tuple(
                    value.crosswalk_hash for value in self.aliases
                ),
                "packed_payload_bytes": self.packed_payload_bytes,
                "actual_packed_byte_equality_verified": (
                    self.actual_packed_byte_equality_verified
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_CANONICAL_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCPackedActionFamily:
    """Complete alias partition and exact unique physical-action payload family."""

    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    candidate_crosswalks: tuple[V1SignedWorkCoordinateCandidateCrosswalk, ...]
    canonical_actions: tuple[V1SignedWorkCanonicalPackedPhysicalAction, ...]
    q_safe_crosswalk_hashes: tuple[str, ...]
    q_robust_infeasible_crosswalk_hashes: tuple[str, ...]
    q_ambiguous_crosswalk_hashes: tuple[str, ...]
    source_materialization_bundle_hashes: tuple[str, ...]
    source_retained_host_packed_payload_bytes: int
    maximum_source_peak_host_workspace_upper_bound_bytes: int
    maximum_source_cuda_logical_workspace_upper_bound_bytes: int
    source_packed_payload_d2h_bytes: int
    canonical_retained_host_packed_payload_bytes: int
    builder_additional_packed_payload_bytes: int = 0
    builder_additional_cuda_bytes: int = 0
    all_coordinate_aliases_retained: bool = True
    actual_packed_byte_equality_verified: bool = True
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_PACKED_ACTION_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        for name in ("parent_mask_hash", "route_axis_hash", "authority_scope"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", routes)
        crosswalks = tuple(self.candidate_crosswalks)
        canonical = tuple(self.canonical_actions)
        if not crosswalks or any(
            type(value) is not V1SignedWorkCoordinateCandidateCrosswalk
            for value in crosswalks
        ):
            raise TypeError("packed-action family requires exact candidate crosswalks")
        if any(
            value.parent_mask_hash != self.parent_mask_hash
            or value.route_axis_hash != self.route_axis_hash
            or value.route_count != routes
            for value in crosswalks
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
                "packed-action family crossed a parent or physical route axis",
            )
        if len({value.coordinate_candidate_key_hash for value in crosswalks}) != len(
            crosswalks
        ):
            raise ValueError("packed-action family repeats a coordinate candidate key")
        if any(
            type(value) is not V1SignedWorkCanonicalPackedPhysicalAction
            for value in canonical
        ):
            raise TypeError("packed-action family canonical actions must be exact")
        if tuple(value.canonical_ordinal for value in canonical) != tuple(
            range(1, len(canonical) + 1)
        ):
            raise ValueError("canonical physical-action ordinals are incomplete")
        safe = tuple(
            value.crosswalk_hash
            for value in crosswalks
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        )
        robust = tuple(
            value.crosswalk_hash
            for value in crosswalks
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
        )
        ambiguous = tuple(
            value.crosswalk_hash
            for value in crosswalks
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        )
        if (
            tuple(self.q_safe_crosswalk_hashes) != safe
            or tuple(self.q_robust_infeasible_crosswalk_hashes) != robust
            or tuple(self.q_ambiguous_crosswalk_hashes) != ambiguous
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "packed-action family Q-status partition does not replay",
            )
        grouped_aliases = tuple(
            alias.crosswalk_hash for action in canonical for alias in action.aliases
        )
        if sorted(grouped_aliases) != sorted(safe) or len(grouped_aliases) != len(safe):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "canonical actions lost or duplicated a Q-safe coordinate alias",
            )
        source_order: list[str] = []
        source_bounds: dict[str, tuple[int, int, int, int]] = {}
        for value in crosswalks:
            key = value.source_materialization_bundle_hash
            bounds = (
                value.source_retained_host_packed_payload_bytes,
                value.source_peak_host_workspace_upper_bound_bytes,
                value.source_cuda_logical_workspace_upper_bound_bytes,
                value.source_packed_payload_d2h_bytes,
            )
            if key in source_bounds and source_bounds[key] != bounds:
                _refuse(
                    V1SignedWorkPreCPackedActionRefusalCode.WORKSPACE_ACCOUNTING_MISMATCH,
                    "one source materialization carried conflicting workspace bounds",
                )
            if key not in source_bounds:
                source_order.append(key)
                source_bounds[key] = bounds
        expected_source_hashes = tuple(source_order)
        expected_source_retained = sum(value[0] for value in source_bounds.values())
        expected_host_peak = _sequential_host_peak(source_bounds)
        expected_cuda_peak = max(value[2] for value in source_bounds.values())
        expected_d2h = sum(value[3] for value in source_bounds.values())
        expected_canonical_retained = len(canonical) * ((routes + 7) // 8)
        byte_fields = (
            self.source_retained_host_packed_payload_bytes,
            self.maximum_source_peak_host_workspace_upper_bound_bytes,
            self.maximum_source_cuda_logical_workspace_upper_bound_bytes,
            self.source_packed_payload_d2h_bytes,
            self.canonical_retained_host_packed_payload_bytes,
            self.builder_additional_packed_payload_bytes,
            self.builder_additional_cuda_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
            for value in byte_fields
        ):
            raise TypeError(
                "packed-action family byte bounds must be nonnegative integers"
            )
        if (
            tuple(self.source_materialization_bundle_hashes) != expected_source_hashes
            or int(self.source_retained_host_packed_payload_bytes)
            != expected_source_retained
            or int(self.maximum_source_peak_host_workspace_upper_bound_bytes)
            != expected_host_peak
            or int(self.maximum_source_cuda_logical_workspace_upper_bound_bytes)
            != expected_cuda_peak
            or int(self.source_packed_payload_d2h_bytes) != expected_d2h
            or int(self.canonical_retained_host_packed_payload_bytes)
            != expected_canonical_retained
            or int(self.builder_additional_packed_payload_bytes) != 0
            or int(self.builder_additional_cuda_bytes) != 0
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.WORKSPACE_ACCOUNTING_MISMATCH,
                "packed-action family workspace bounds do not replay exactly",
            )
        object.__setattr__(self, "candidate_crosswalks", crosswalks)
        object.__setattr__(self, "canonical_actions", canonical)
        object.__setattr__(self, "q_safe_crosswalk_hashes", safe)
        object.__setattr__(self, "q_robust_infeasible_crosswalk_hashes", robust)
        object.__setattr__(self, "q_ambiguous_crosswalk_hashes", ambiguous)
        object.__setattr__(
            self,
            "source_materialization_bundle_hashes",
            expected_source_hashes,
        )
        if (
            type(self.all_coordinate_aliases_retained) is not bool
            or not self.all_coordinate_aliases_retained
            or type(self.actual_packed_byte_equality_verified) is not bool
            or not self.actual_packed_byte_equality_verified
        ):
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "packed-action family did not retain and byte-check every alias",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="coordinate-aware packed-action family",
        )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_PACKED_ACTION_FAMILY_SCHEMA
        ):
            raise ValueError("unsupported packed-action family semantics")

    @property
    def family_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "candidate_crosswalk_hashes": tuple(
                    value.crosswalk_hash for value in self.candidate_crosswalks
                ),
                "canonical_action_hashes": tuple(
                    value.canonical_action_hash for value in self.canonical_actions
                ),
                "q_safe_crosswalk_hashes": self.q_safe_crosswalk_hashes,
                "q_robust_infeasible_crosswalk_hashes": (
                    self.q_robust_infeasible_crosswalk_hashes
                ),
                "q_ambiguous_crosswalk_hashes": self.q_ambiguous_crosswalk_hashes,
                "source_materialization_bundle_hashes": (
                    self.source_materialization_bundle_hashes
                ),
                "source_retained_host_packed_payload_bytes": (
                    self.source_retained_host_packed_payload_bytes
                ),
                "maximum_source_peak_host_workspace_upper_bound_bytes": (
                    self.maximum_source_peak_host_workspace_upper_bound_bytes
                ),
                "maximum_source_cuda_logical_workspace_upper_bound_bytes": (
                    self.maximum_source_cuda_logical_workspace_upper_bound_bytes
                ),
                "source_packed_payload_d2h_bytes": (
                    self.source_packed_payload_d2h_bytes
                ),
                "canonical_retained_host_packed_payload_bytes": (
                    self.canonical_retained_host_packed_payload_bytes
                ),
                "builder_additional_packed_payload_bytes": (
                    self.builder_additional_packed_payload_bytes
                ),
                "builder_additional_cuda_bytes": self.builder_additional_cuda_bytes,
                "all_coordinate_aliases_retained": self.all_coordinate_aliases_retained,
                "actual_packed_byte_equality_verified": (
                    self.actual_packed_byte_equality_verified
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_FAMILY_PREFIX,
        )


def build_v1_signed_work_pre_c_packed_action_family(
    *,
    candidate_crosswalks: tuple[V1SignedWorkCoordinateCandidateCrosswalk, ...],
) -> V1SignedWorkPreCPackedActionFamily:
    """Deduplicate already-materialized Q-safe aliases by exact physical action."""

    crosswalks = tuple(candidate_crosswalks)
    if not crosswalks or any(
        type(value) is not V1SignedWorkCoordinateCandidateCrosswalk
        for value in crosswalks
    ):
        raise TypeError("packed-action builder requires exact nonempty crosswalks")
    parent = crosswalks[0].parent_mask_hash
    axis = crosswalks[0].route_axis_hash
    routes = crosswalks[0].route_count
    if any(
        value.parent_mask_hash != parent
        or value.route_axis_hash != axis
        or value.route_count != routes
        for value in crosswalks
    ):
        _refuse(
            V1SignedWorkPreCPackedActionRefusalCode.SOURCE_MISMATCH,
            "candidate crosswalks do not share one parent route axis",
        )
    canonical_by_key: dict[
        V1SignedWorkPackedActionDedupIdentity,
        V1PackedRemovalAction,
    ] = {}
    aliases_by_key: dict[
        V1SignedWorkPackedActionDedupIdentity,
        list[V1SignedWorkCoordinateCandidateCrosswalk],
    ] = {}
    key_order: list[V1SignedWorkPackedActionDedupIdentity] = []
    normalized: list[V1SignedWorkCoordinateCandidateCrosswalk] = []
    for crosswalk in crosswalks:
        if crosswalk.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE:
            normalized.append(crosswalk)
            continue
        action = crosswalk.packed_action
        if (
            type(action) is not V1PackedRemovalAction
        ):  # pragma: no cover - constructor closes.
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.Q_STATUS_PAYLOAD_VIOLATION,
                "Q-safe crosswalk lost its packed action",
            )
        key = v1_signed_work_packed_action_dedup_identity(action)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = action
            aliases_by_key[key] = []
            key_order.append(key)
            canonical = action
        elif canonical.resulting_mask.packed_bits != action.resulting_mask.packed_bits:
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.PACKED_BYTE_MISMATCH,
                "matching action identities carried unequal actual packed bytes",
            )
        normalized_crosswalk = replace(crosswalk, packed_action=canonical)
        normalized.append(normalized_crosswalk)
        aliases_by_key[key].append(normalized_crosswalk)
    canonical_actions = tuple(
        V1SignedWorkCanonicalPackedPhysicalAction(
            canonical_ordinal=ordinal,
            dedup_identity=key,
            packed_action=canonical_by_key[key],
            aliases=tuple(aliases_by_key[key]),
            packed_payload_bytes=len(canonical_by_key[key].resulting_mask.packed_bits),
        )
        for ordinal, key in enumerate(key_order, start=1)
    )
    normalized_tuple = tuple(normalized)
    source_order: list[str] = []
    source_bounds: dict[str, tuple[int, int, int, int]] = {}
    for value in normalized_tuple:
        key = value.source_materialization_bundle_hash
        bounds = (
            value.source_retained_host_packed_payload_bytes,
            value.source_peak_host_workspace_upper_bound_bytes,
            value.source_cuda_logical_workspace_upper_bound_bytes,
            value.source_packed_payload_d2h_bytes,
        )
        if key not in source_bounds:
            source_order.append(key)
            source_bounds[key] = bounds
        elif source_bounds[key] != bounds:
            _refuse(
                V1SignedWorkPreCPackedActionRefusalCode.WORKSPACE_ACCOUNTING_MISMATCH,
                "one source materialization carried conflicting workspace bounds",
            )
    return V1SignedWorkPreCPackedActionFamily(
        parent_mask_hash=parent,
        route_axis_hash=axis,
        route_count=routes,
        candidate_crosswalks=normalized_tuple,
        canonical_actions=canonical_actions,
        q_safe_crosswalk_hashes=tuple(
            value.crosswalk_hash
            for value in normalized_tuple
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        ),
        q_robust_infeasible_crosswalk_hashes=tuple(
            value.crosswalk_hash
            for value in normalized_tuple
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
        ),
        q_ambiguous_crosswalk_hashes=tuple(
            value.crosswalk_hash
            for value in normalized_tuple
            if value.q_status is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
        ),
        source_materialization_bundle_hashes=tuple(source_order),
        source_retained_host_packed_payload_bytes=sum(
            value[0] for value in source_bounds.values()
        ),
        maximum_source_peak_host_workspace_upper_bound_bytes=(
            _sequential_host_peak(source_bounds)
        ),
        maximum_source_cuda_logical_workspace_upper_bound_bytes=max(
            value[2] for value in source_bounds.values()
        ),
        source_packed_payload_d2h_bytes=sum(
            value[3] for value in source_bounds.values()
        ),
        canonical_retained_host_packed_payload_bytes=(
            len(canonical_actions) * ((routes + 7) // 8)
        ),
    )


__all__ = [
    "SIGNED_WORK_PRE_C_CANONICAL_PACKED_ACTION_SCHEMA",
    "SIGNED_WORK_PRE_C_PACKED_ACTION_AUTHORITY",
    "SIGNED_WORK_PRE_C_PACKED_ACTION_FAMILY_SCHEMA",
    "SIGNED_WORK_PRE_C_PACKED_CROSSWALK_SCHEMA",
    "V1SignedWorkCanonicalPackedPhysicalAction",
    "V1SignedWorkCoordinateCandidateCrosswalk",
    "V1SignedWorkPackedActionDedupIdentity",
    "V1SignedWorkPreCPackedActionFamily",
    "V1SignedWorkPreCPackedActionRefusal",
    "V1SignedWorkPreCPackedActionRefusalCode",
    "build_v1_signed_work_pre_c_packed_action_family",
    "crosswalk_v1_signed_work_coordinate_candidate",
    "v1_signed_work_packed_action_dedup_identity",
]
