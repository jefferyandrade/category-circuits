"""Exact signed-work shell custody bridge into independent C measurement.

The signed-work solver, its two deterministic hard ladders, and complete hard
F/Q replay all belong to discovery stream D.  Directional secant measurement
on stream C may begin only after those identities are frozen.  This module is
the deliberately narrow bridge between the two stages:

* it proves a branch/endpoint bijection between the shell and hard-F/Q replay;
* it refuses any unresolved canonical Q envelope;
* it materializes every, and only, canonically Q-safe endpoint from a
  ``C_DIRECTIONAL_CERTIFICATION`` label lease;
* it emits ordinary packed ``V1FrozenCandidateLadder`` values without
  reinterpreting any discovery F/Q value as a C observation; and
* it can atomically evict both branch-label vectors after every lease closes.

Production execution is CUDA-only.  The NumPy implementation is a bounded,
explicitly registered mathematical reference for focused tests.  None of the
types in this module authorizes C outcomes, F/Q/R certification, calibration,
the paired oracle, mask commit, terminality, or a Pareto claim.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_device_shell_ladder import (
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenActionReceipt,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelLease,
    V1DeviceShellRefusal,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellExecutionBundle,
    V1SignedWorkDeviceShellFamilyBundle,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQBranchReceipt,
    V1SignedWorkHardFQEndpointReceipt,
    V1SignedWorkHardFQEndpointStatus,
    V1SignedWorkHardFQFamilyBundle,
    V1SignedWorkHardFQSingleBranchBundle,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_C_BRIDGE_COORDINATES_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_coordinates_v1"
)
SIGNED_WORK_C_BRIDGE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_materialization_design_v1"
)
SIGNED_WORK_C_MATERIALIZATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_action_materialization_v1"
)
SIGNED_WORK_C_SINGLE_BRANCH_ACTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_single_branch_packed_action_v1"
)
SIGNED_WORK_C_SINGLE_BRANCH_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_single_branch_materialization_v1"
)
SIGNED_WORK_C_ENDPOINT_DECISION_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_endpoint_decision_v1"
)
SIGNED_WORK_C_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_action_crosswalk_v1"
)
SIGNED_WORK_C_BRANCH_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_branch_ladder_v1"
)
SIGNED_WORK_C_FAMILY_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_two_branch_family_v1"
)
SIGNED_WORK_C_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_family_bundle_v1"
)
SIGNED_WORK_C_AUTHORITY: Final = (
    "exact_D_shell_and_hard_FQ_to_pre_C_packed_direction_identity_only_"
    "no_D_value_recycling_no_C_outcome_FQR_K_O_commit_terminal_or_pareto_v1"
)

_COORDINATES_PREFIX: Final = "partition-free-v1-signed-work-C-coordinates-"
_DESIGN_PREFIX: Final = "partition-free-v1-signed-work-C-design-"
_MATERIALIZATION_PREFIX: Final = "partition-free-v1-signed-work-C-materialization-"
_SINGLE_BRANCH_ACTION_PREFIX: Final = (
    "partition-free-v1-signed-work-C-single-branch-action-"
)
_SINGLE_BRANCH_BUNDLE_PREFIX: Final = (
    "partition-free-v1-signed-work-C-single-branch-bundle-"
)
_DECISION_PREFIX: Final = "partition-free-v1-signed-work-C-decision-"
_CROSSWALK_PREFIX: Final = "partition-free-v1-signed-work-C-crosswalk-"
_FREEZE_PREFIX: Final = "partition-free-v1-signed-work-C-ladder-freeze-"
_BRANCH_PREFIX: Final = "partition-free-v1-signed-work-C-branch-"
_FAMILY_PREFIX: Final = "partition-free-v1-signed-work-C-family-"
_BUNDLE_PREFIX: Final = "partition-free-v1-signed-work-C-bundle-"
_CANDIDATE_PREFIX: Final = "partition-free-v1-signed-work-C-candidate-"


class V1SignedWorkCBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    ENDPOINT_BIJECTION_MISMATCH = "endpoint_bijection_mismatch"
    CANONICAL_Q_AMBIGUOUS = "canonical_q_ambiguous"
    NO_Q_SAFE_CANDIDATE = "no_q_safe_c_candidate"
    PRODUCTION_CUDA_REQUIRED = "production_cuda_required"
    CPU_REFERENCE_SCALE_EXCEEDED = "cpu_reference_scale_exceeded"
    WORKSPACE_FORECAST_EXCEEDED = "workspace_forecast_exceeded"
    LABEL_IDENTITY_MISMATCH = "label_identity_mismatch"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkCBridgeRefusal(RuntimeError):
    def __init__(self, code: V1SignedWorkCBridgeRefusalCode, message: str) -> None:
        if type(code) is not V1SignedWorkCBridgeRefusalCode:
            raise TypeError("signed-work C-bridge refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkCBridgeRefusalCode, message: str) -> NoReturn:
    raise V1SignedWorkCBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 0.85:
        raise ValueError(f"{name} must lie in (0, 0.85]")
    return result


def _direct_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            f"{name} must use the exact HardMask type",
        )
    return HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=mask.packed_bits,
    ).mask_hash


def _closed_authority_flags(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority flags must be exact booleans")
    if any(values):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed the pre-C identity boundary",
        )


def _status_value(value: object, *, name: str) -> str:
    if isinstance(value, Enum):
        result = value.value
    else:
        result = value
    return _identity(result, name=name)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCBridgeCoordinates:
    """Registered D coordinates needed to freeze an independent C ladder."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    route_axis_hash: str
    parent_mask_hash: str
    proposal_generation: int
    discovery_context_ids: tuple[str, ...]
    discovery_draw_ids: tuple[str, ...]
    registered_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    schema: str = SIGNED_WORK_C_BRIDGE_COORDINATES_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "route_axis_hash",
            "parent_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        contexts = tuple(
            _identity(value, name="discovery context id")
            for value in self.discovery_context_ids
        )
        draws = tuple(
            _identity(value, name="discovery draw id")
            for value in self.discovery_draw_ids
        )
        if (
            not contexts
            or len(contexts) != len(draws)
            or len(set(contexts)) != len(contexts)
            or len(set(draws)) != len(draws)
        ):
            raise ValueError(
                "D discovery ledgers must be aligned, nonempty, and unique"
            )
        object.__setattr__(self, "discovery_context_ids", contexts)
        object.__setattr__(self, "discovery_draw_ids", draws)
        if (
            type(self.registered_before_C_outcomes) is not bool
            or type(self.C_outcomes_seen) is not bool
            or not self.registered_before_C_outcomes
            or self.C_outcomes_seen
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "C-bridge coordinates must be registered before C outcomes",
            )
        if self.schema != SIGNED_WORK_C_BRIDGE_COORDINATES_SCHEMA:
            raise ValueError("unsupported signed-work C-coordinate schema")

    @property
    def coordinates_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATES_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCBridgeDesign:
    """Pre-C operational limits; these do not tune scientific decisions."""

    route_tile_size: int = 1 << 20
    endpoint_lane_count: int = 8
    maximum_host_packed_payload_bytes: int = 4 << 30
    maximum_cuda_logical_workspace_bytes: int = 4 << 30
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    private_reference_route_limit: int = 4096
    schema: str = SIGNED_WORK_C_BRIDGE_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        route_tile = _positive_int(self.route_tile_size, name="route tile size")
        if route_tile % 8:
            raise ValueError("C-bridge route tile must be byte aligned")
        object.__setattr__(self, "route_tile_size", route_tile)
        for name in (
            "endpoint_lane_count",
            "maximum_host_packed_payload_bytes",
            "maximum_cuda_logical_workspace_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "private_reference_route_limit",
            _nonnegative_int(
                self.private_reference_route_limit,
                name="private reference route limit",
            ),
        )
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction"),
        )
        if (
            type(self.production_cuda_required) is not bool
            or type(self.allow_private_cpu_reference) is not bool
        ):
            raise TypeError("C-bridge execution flags must be exact booleans")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production C bridge cannot enable a CPU fallback")
        if self.schema != SIGNED_WORK_C_BRIDGE_DESIGN_SCHEMA:
            raise ValueError("unsupported signed-work C-bridge design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCEndpointDecisionReceipt:
    """Canonical Q eligibility for every endpoint, including excluded rungs."""

    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    shell_action_receipt_hash: str
    hard_replay_endpoint_receipt_hash: str
    canonical_status: V1SignedWorkHardFQEndpointStatus
    origin_branch_status: str
    eligible_for_C: bool
    C_outcomes_seen: bool = False
    D_FQ_values_reused_as_C_observations: bool = False
    schema: str = SIGNED_WORK_C_ENDPOINT_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("C endpoint decision branch must be exact")
        object.__setattr__(
            self,
            "endpoint_label",
            _positive_int(self.endpoint_label, name="endpoint label"),
        )
        object.__setattr__(
            self,
            "cumulative_removed_count",
            _positive_int(
                self.cumulative_removed_count, name="cumulative removed count"
            ),
        )
        for name in (
            "shell_action_receipt_hash",
            "hard_replay_endpoint_receipt_hash",
            "origin_branch_status",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.canonical_status) is not V1SignedWorkHardFQEndpointStatus:
            raise TypeError(
                "C eligibility requires the exact canonical hard-F/Q status"
            )
        expected = (
            self.canonical_status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        )
        if type(self.eligible_for_C) is not bool or self.eligible_for_C is not expected:
            raise ValueError("C eligibility does not replay the canonical Q status")
        _closed_authority_flags(
            (
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
            ),
            owner="C endpoint decision",
        )
        if self.schema != SIGNED_WORK_C_ENDPOINT_DECISION_SCHEMA:
            raise ValueError("unsupported C endpoint-decision schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DECISION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCActionMaterializationReceipt:
    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    shell_execution_hash: str
    shell_ladder_receipt_hash: str
    label_commitment: str
    hard_replay_endpoint_receipt_hash: str
    backend_kind: str
    device_identity: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    resulting_mask_packed_sha256: str
    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    packed_payload_d2h_bytes: int
    raw_label_d2h_bytes: int = 0
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    D_FQ_values_reused_as_C_observations: bool = False
    authority_scope: str = SIGNED_WORK_C_AUTHORITY
    schema: str = SIGNED_WORK_C_MATERIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("C materialization branch must be exact")
        for name in (
            "endpoint_label",
            "cumulative_removed_count",
            "route_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "shell_execution_hash",
            "shell_ladder_receipt_hash",
            "label_commitment",
            "hard_replay_endpoint_receipt_hash",
            "backend_kind",
            "device_identity",
            "route_axis_hash",
            "parent_mask_hash",
            "resulting_mask_packed_sha256",
            "packed_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "packed_payload_d2h_bytes",
            "raw_label_d2h_bytes",
            "C_observation_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if self.backend_kind not in {"torch_cuda", "numpy_reference"}:
            raise ValueError("unsupported C materialization backend")
        expected_d2h = (
            (self.route_count + 7) // 8 if self.backend_kind == "torch_cuda" else 0
        )
        if self.packed_payload_d2h_bytes != expected_d2h:
            raise ValueError("packed C-action D2H accounting does not replay")
        if self.raw_label_d2h_bytes != 0 or self.C_observation_count != 0:
            _refuse(
                V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "C materialization may not transfer raw labels or observe C values",
            )
        _closed_authority_flags(
            (
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
            ),
            owner="C action materialization",
        )
        if (
            self.authority_scope != SIGNED_WORK_C_AUTHORITY
            or self.schema != SIGNED_WORK_C_MATERIALIZATION_SCHEMA
        ):
            raise ValueError("unsupported C materialization receipt semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_MATERIALIZATION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCSingleBranchPackedAction:
    """One exact Q-safe physical action materialized from a shell label vector.

    The object retains the ordinary packed action and both source receipts.  A
    downstream coordinate-aware bridge therefore never has to reconstruct a
    deployable action from hashes.  This type is representation custody only;
    it carries no authority to open C, K, or O.
    """

    endpoint_decision: V1SignedWorkCEndpointDecisionReceipt
    shell_action: V1DeviceShellFrozenActionReceipt
    hard_fq_endpoint: V1SignedWorkHardFQEndpointReceipt
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    materialization: V1SignedWorkCActionMaterializationReceipt
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_C_SINGLE_BRANCH_ACTION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.endpoint_decision) is not V1SignedWorkCEndpointDecisionReceipt
            or type(self.shell_action) is not V1DeviceShellFrozenActionReceipt
            or type(self.hard_fq_endpoint) is not V1SignedWorkHardFQEndpointReceipt
            or type(self.packed_action) is not V1PackedRemovalAction
            or type(self.materialization)
            is not V1SignedWorkCActionMaterializationReceipt
        ):
            raise TypeError("single-branch C action requires exact source objects")
        decision = self.endpoint_decision
        endpoint = self.hard_fq_endpoint
        shell = self.shell_action
        action = self.packed_action
        materialization = self.materialization
        packed_sha = hashlib.sha256(action.resulting_mask.packed_bits).hexdigest()
        if (
            decision.canonical_status
            is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            or not decision.eligible_for_C
            or endpoint.status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            or decision.branch is not endpoint.branch
            or materialization.branch is not endpoint.branch
            or decision.endpoint_label != endpoint.endpoint_label
            or shell.endpoint_label != endpoint.endpoint_label
            or materialization.endpoint_label != endpoint.endpoint_label
            or decision.cumulative_removed_count != endpoint.cumulative_removed_count
            or shell.cumulative_removed_count != endpoint.cumulative_removed_count
            or materialization.cumulative_removed_count
            != endpoint.cumulative_removed_count
            or decision.shell_action_receipt_hash != shell.receipt_hash
            or decision.hard_replay_endpoint_receipt_hash != endpoint.receipt_hash
            or endpoint.frozen_action_receipt_hash != shell.receipt_hash
            or materialization.hard_replay_endpoint_receipt_hash
            != endpoint.receipt_hash
            or action.action_hash != endpoint.packed_action_hash
            or action.resulting_mask.mask_hash != endpoint.resulting_mask_hash
            or action.removed_route_index_hash != endpoint.removed_route_index_hash
            or action.removed_count != endpoint.cumulative_removed_count
            or materialization.packed_action_hash != action.action_hash
            or materialization.resulting_mask_hash != action.resulting_mask.mask_hash
            or materialization.removed_route_index_hash
            != action.removed_route_index_hash
            or materialization.resulting_mask_packed_sha256 != packed_sha
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.ACTION_IDENTITY_MISMATCH,
                "single-branch packed action does not replay its Q-safe source",
            )
        _closed_authority_flags(
            (
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="single-branch packed action",
        )
        if self.schema != SIGNED_WORK_C_SINGLE_BRANCH_ACTION_SCHEMA:
            raise ValueError("unsupported single-branch packed-action schema")

    @property
    def action_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "endpoint_decision_receipt_hash": self.endpoint_decision.receipt_hash,
                "shell_action_receipt_hash": self.shell_action.receipt_hash,
                "hard_fq_endpoint_receipt_hash": self.hard_fq_endpoint.receipt_hash,
                "packed_action_hash": self.packed_action.action_hash,
                "resulting_mask_packed_sha256": hashlib.sha256(
                    self.packed_action.resulting_mask.packed_bits
                ).hexdigest(),
                "materialization_receipt_hash": self.materialization.receipt_hash,
            },
            prefix=_SINGLE_BRANCH_ACTION_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkCSingleBranchMaterializationBundle:
    """Complete Q-status partition plus the exact payload-bearing safe actions."""

    branch: V1RelaxedKKTBranch
    shell_execution_hash: str
    hard_fq_bundle_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    design_hash: str
    route_tile_size: int
    endpoint_lane_count: int
    backend_kind: str
    endpoint_decisions: tuple[V1SignedWorkCEndpointDecisionReceipt, ...]
    packed_actions: tuple[V1SignedWorkCSingleBranchPackedAction, ...]
    q_safe_endpoint_labels: tuple[int, ...]
    q_robust_infeasible_endpoint_labels: tuple[int, ...]
    q_ambiguous_endpoint_labels: tuple[int, ...]
    retained_host_packed_payload_bytes: int
    peak_host_workspace_upper_bound_bytes: int
    cuda_logical_workspace_upper_bound_bytes: int
    packed_payload_d2h_bytes: int
    raw_label_d2h_bytes: int = 0
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_C_SINGLE_BRANCH_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("single-branch C materialization branch must be exact")
        for name in (
            "shell_execution_hash",
            "hard_fq_bundle_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "design_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        object.__setattr__(
            self,
            "endpoint_lane_count",
            _positive_int(self.endpoint_lane_count, name="endpoint lane count"),
        )
        if self.backend_kind not in {"torch_cuda", "numpy_reference"}:
            raise ValueError("single-branch materialization backend is invalid")
        decisions = tuple(self.endpoint_decisions)
        actions = tuple(self.packed_actions)
        if not decisions or any(
            type(value) is not V1SignedWorkCEndpointDecisionReceipt
            or value.branch is not self.branch
            for value in decisions
        ):
            raise TypeError("single-branch C bundle requires exact endpoint decisions")
        if tuple(value.endpoint_label for value in decisions) != tuple(
            range(1, len(decisions) + 1)
        ):
            raise ValueError(
                "single-branch endpoint decisions are incomplete or reordered"
            )
        if any(
            type(value) is not V1SignedWorkCSingleBranchPackedAction
            or value.hard_fq_endpoint.branch is not self.branch
            for value in actions
        ):
            raise TypeError("single-branch C bundle requires exact packed actions")
        partitions = tuple(
            tuple(
                _positive_int(label, name="Q-status endpoint label") for label in axis
            )
            for axis in (
                self.q_safe_endpoint_labels,
                self.q_robust_infeasible_endpoint_labels,
                self.q_ambiguous_endpoint_labels,
            )
        )
        safe, robust, ambiguous = partitions
        complete = set(range(1, len(decisions) + 1))
        if (
            any(tuple(sorted(axis)) != axis for axis in partitions)
            or any(len(set(axis)) != len(axis) for axis in partitions)
            or set(safe) & set(robust)
            or set(safe) & set(ambiguous)
            or set(robust) & set(ambiguous)
            or set(safe) | set(robust) | set(ambiguous) != complete
            or tuple(
                value.endpoint_label for value in decisions if value.eligible_for_C
            )
            != safe
            or tuple(value.hard_fq_endpoint.endpoint_label for value in actions) != safe
        ):
            raise ValueError(
                "single-branch Q-status partition or payload axis is invalid"
            )
        expected_statuses = {
            V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE: safe,
            V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE: robust,
            V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS: ambiguous,
        }
        for status, labels in expected_statuses.items():
            if (
                tuple(
                    value.endpoint_label
                    for value in decisions
                    if value.canonical_status is status
                )
                != labels
            ):
                raise ValueError(
                    "single-branch Q-status receipt partition does not replay"
                )
        packed_bytes = (routes + 7) // 8
        retained = len(safe) * packed_bytes
        expected_retained, expected_peak_host, expected_cuda = _workspace_formula(
            route_count=routes,
            safe_endpoint_count=len(safe),
            route_tile_size=self.route_tile_size,
            endpoint_lane_count=self.endpoint_lane_count,
            backend_kind=self.backend_kind,
        )
        for name in (
            "retained_host_packed_payload_bytes",
            "peak_host_workspace_upper_bound_bytes",
            "cuda_logical_workspace_upper_bound_bytes",
            "packed_payload_d2h_bytes",
            "raw_label_d2h_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if (
            retained != expected_retained
            or self.retained_host_packed_payload_bytes != expected_retained
            or self.peak_host_workspace_upper_bound_bytes != expected_peak_host
            or self.cuda_logical_workspace_upper_bound_bytes != expected_cuda
            or self.raw_label_d2h_bytes != 0
            or self.packed_payload_d2h_bytes
            != (expected_retained if self.backend_kind == "torch_cuda" else 0)
            or sum(value.materialization.packed_payload_d2h_bytes for value in actions)
            != self.packed_payload_d2h_bytes
            or any(
                value.materialization.backend_kind != self.backend_kind
                for value in actions
            )
        ):
            raise ValueError(
                "single-branch materialization byte accounting does not replay"
            )
        object.__setattr__(self, "endpoint_decisions", decisions)
        object.__setattr__(self, "packed_actions", actions)
        object.__setattr__(self, "q_safe_endpoint_labels", safe)
        object.__setattr__(self, "q_robust_infeasible_endpoint_labels", robust)
        object.__setattr__(self, "q_ambiguous_endpoint_labels", ambiguous)
        _closed_authority_flags(
            (
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="single-branch C materialization bundle",
        )
        if self.schema != SIGNED_WORK_C_SINGLE_BRANCH_BUNDLE_SCHEMA:
            raise ValueError("unsupported single-branch materialization schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch,
                "shell_execution_hash": self.shell_execution_hash,
                "hard_fq_bundle_hash": self.hard_fq_bundle_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "design_hash": self.design_hash,
                "route_tile_size": self.route_tile_size,
                "endpoint_lane_count": self.endpoint_lane_count,
                "backend_kind": self.backend_kind,
                "endpoint_decision_receipt_hashes": tuple(
                    value.receipt_hash for value in self.endpoint_decisions
                ),
                "packed_action_hashes": tuple(
                    value.action_hash for value in self.packed_actions
                ),
                "q_safe_endpoint_labels": self.q_safe_endpoint_labels,
                "q_robust_infeasible_endpoint_labels": (
                    self.q_robust_infeasible_endpoint_labels
                ),
                "q_ambiguous_endpoint_labels": self.q_ambiguous_endpoint_labels,
                "retained_host_packed_payload_bytes": (
                    self.retained_host_packed_payload_bytes
                ),
                "peak_host_workspace_upper_bound_bytes": (
                    self.peak_host_workspace_upper_bound_bytes
                ),
                "cuda_logical_workspace_upper_bound_bytes": (
                    self.cuda_logical_workspace_upper_bound_bytes
                ),
                "packed_payload_d2h_bytes": self.packed_payload_d2h_bytes,
                "raw_label_d2h_bytes": self.raw_label_d2h_bytes,
            },
            prefix=_SINGLE_BRANCH_BUNDLE_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkCCrosswalkReceipt:
    """Exact source-to-packed-to-frozen identity for one C-eligible rung."""

    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    shell_action_receipt_hash: str
    hard_replay_endpoint_receipt_hash: str
    hard_replay_status: V1SignedWorkHardFQEndpointStatus
    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    materialization_receipt_hash: str
    frozen_action_hash: str
    canonical_candidate_identity_hash: str
    canonical_candidate_occurrence: int
    cross_branch_exact_deduplication: bool
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    D_FQ_values_reused_as_C_observations: bool = False
    authority_scope: str = SIGNED_WORK_C_AUTHORITY
    schema: str = SIGNED_WORK_C_CROSSWALK_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("C crosswalk branch must be exact")
        for name in (
            "endpoint_label",
            "cumulative_removed_count",
            "canonical_candidate_occurrence",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "shell_action_receipt_hash",
            "hard_replay_endpoint_receipt_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "materialization_receipt_hash",
            "frozen_action_hash",
            "canonical_candidate_identity_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.hard_replay_status) is not V1SignedWorkHardFQEndpointStatus
            or self.hard_replay_status
            is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        ):
            raise ValueError("only canonically Q-safe endpoints may enter C crosswalks")
        if type(self.cross_branch_exact_deduplication) is not bool:
            raise TypeError("cross-branch deduplication flag must be exact")
        object.__setattr__(
            self,
            "C_observation_count",
            _nonnegative_int(self.C_observation_count, name="C observation count"),
        )
        if self.C_observation_count != 0:
            _refuse(
                V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "pre-C crosswalk cannot contain C observations",
            )
        _closed_authority_flags(
            (
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
            ),
            owner="C action crosswalk",
        )
        if (
            self.authority_scope != SIGNED_WORK_C_AUTHORITY
            or self.schema != SIGNED_WORK_C_CROSSWALK_SCHEMA
        ):
            raise ValueError("unsupported C crosswalk semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_CROSSWALK_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCBranchReceipt:
    branch: V1RelaxedKKTBranch
    shell_execution_hash: str
    shell_execution_receipt_hash: str
    shell_branch_identity_receipt_hash: str
    shell_ladder_receipt_hash: str
    hard_replay_branch_receipt_hash: str
    coordinates_hash: str
    design_hash: str
    complete_source_endpoint_count: int
    endpoint_decision_receipt_hashes: tuple[str, ...]
    q_safe_endpoint_labels: tuple[int, ...]
    q_robust_infeasible_endpoint_labels: tuple[int, ...]
    materialization_receipt_hashes: tuple[str, ...]
    crosswalk_receipt_hashes: tuple[str, ...]
    frozen_ladder_hash: str | None
    all_source_endpoints_considered: bool = True
    q_nonmonotone_scan_early_stop_used: bool = False
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    D_FQ_values_reused_as_C_observations: bool = False
    authority_scope: str = SIGNED_WORK_C_AUTHORITY
    schema: str = SIGNED_WORK_C_BRANCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("C branch receipt branch must be exact")
        for name in (
            "shell_execution_hash",
            "shell_execution_receipt_hash",
            "shell_branch_identity_receipt_hash",
            "shell_ladder_receipt_hash",
            "hard_replay_branch_receipt_hash",
            "coordinates_hash",
            "design_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.frozen_ladder_hash is not None:
            object.__setattr__(
                self,
                "frozen_ladder_hash",
                _identity(self.frozen_ladder_hash, name="frozen ladder hash"),
            )
        count = _positive_int(
            self.complete_source_endpoint_count, name="complete source endpoint count"
        )
        object.__setattr__(self, "complete_source_endpoint_count", count)
        decisions = tuple(
            _identity(value, name="endpoint decision receipt hash")
            for value in self.endpoint_decision_receipt_hashes
        )
        safe = tuple(
            _positive_int(value, name="Q-safe label")
            for value in self.q_safe_endpoint_labels
        )
        infeasible = tuple(
            _positive_int(value, name="Q-infeasible label")
            for value in self.q_robust_infeasible_endpoint_labels
        )
        materializations = tuple(
            _identity(value, name="materialization receipt hash")
            for value in self.materialization_receipt_hashes
        )
        crosswalks = tuple(
            _identity(value, name="crosswalk receipt hash")
            for value in self.crosswalk_receipt_hashes
        )
        if (
            len(decisions) != count
            or tuple(sorted(safe)) != safe
            or tuple(sorted(infeasible)) != infeasible
            or set(safe) & set(infeasible)
            or set(safe) | set(infeasible) != set(range(1, count + 1))
            or len(materializations) != len(safe)
            or len(crosswalks) != len(safe)
            or (self.frozen_ladder_hash is None) != (not safe)
        ):
            raise ValueError("C branch endpoint partition or ladder axis is incomplete")
        object.__setattr__(self, "endpoint_decision_receipt_hashes", decisions)
        object.__setattr__(self, "q_safe_endpoint_labels", safe)
        object.__setattr__(self, "q_robust_infeasible_endpoint_labels", infeasible)
        object.__setattr__(self, "materialization_receipt_hashes", materializations)
        object.__setattr__(self, "crosswalk_receipt_hashes", crosswalks)
        object.__setattr__(
            self,
            "C_observation_count",
            _nonnegative_int(self.C_observation_count, name="C observation count"),
        )
        if (
            type(self.all_source_endpoints_considered) is not bool
            or not self.all_source_endpoints_considered
            or type(self.q_nonmonotone_scan_early_stop_used) is not bool
            or self.q_nonmonotone_scan_early_stop_used
            or self.C_observation_count != 0
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "C branch omitted endpoints, stopped early, or observed C values",
            )
        _closed_authority_flags(
            (
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
            ),
            owner="C branch receipt",
        )
        if (
            self.authority_scope != SIGNED_WORK_C_AUTHORITY
            or self.schema != SIGNED_WORK_C_BRANCH_SCHEMA
        ):
            raise ValueError("unsupported C branch receipt semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCBranchBundle:
    ladder: V1FrozenCandidateLadder | None = field(repr=False)
    endpoint_decisions: tuple[V1SignedWorkCEndpointDecisionReceipt, ...]
    materializations: tuple[V1SignedWorkCActionMaterializationReceipt, ...]
    crosswalks: tuple[V1SignedWorkCCrosswalkReceipt, ...]
    receipt: V1SignedWorkCBranchReceipt

    def __post_init__(self) -> None:
        decisions = tuple(self.endpoint_decisions)
        materials = tuple(self.materializations)
        crosswalks = tuple(self.crosswalks)
        if any(
            type(value) is not V1SignedWorkCEndpointDecisionReceipt
            for value in decisions
        ):
            raise TypeError("C branch decisions must use exact receipt types")
        if any(
            type(value) is not V1SignedWorkCActionMaterializationReceipt
            for value in materials
        ):
            raise TypeError("C branch materializations must use exact receipt types")
        if any(
            type(value) is not V1SignedWorkCCrosswalkReceipt for value in crosswalks
        ):
            raise TypeError("C branch crosswalks must use exact receipt types")
        if type(self.receipt) is not V1SignedWorkCBranchReceipt:
            raise TypeError("C branch bundle requires an exact branch receipt")
        if self.ladder is not None and type(self.ladder) is not V1FrozenCandidateLadder:
            raise TypeError("C branch ladder must use the exact frozen-ladder type")
        if (
            tuple(value.receipt_hash for value in decisions)
            != self.receipt.endpoint_decision_receipt_hashes
            or tuple(value.receipt_hash for value in materials)
            != self.receipt.materialization_receipt_hashes
            or tuple(value.receipt_hash for value in crosswalks)
            != self.receipt.crosswalk_receipt_hashes
            or (None if self.ladder is None else self.ladder.ladder_hash)
            != self.receipt.frozen_ladder_hash
            or any(value.branch is not self.receipt.branch for value in decisions)
            or any(value.branch is not self.receipt.branch for value in materials)
            or any(value.branch is not self.receipt.branch for value in crosswalks)
        ):
            raise ValueError("C branch bundle differs from its identity receipt")
        if self.ladder is not None:
            if self.ladder.action_hashes != tuple(
                value.frozen_action_hash for value in crosswalks
            ):
                raise ValueError("C ladder actions differ from their crosswalk")


@dataclass(frozen=True, slots=True)
class V1SignedWorkCFamilyReceipt:
    shell_family_hash: str
    hard_fq_bundle_hash: str
    hard_fq_family_receipt_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    coordinates_hash: str
    design_hash: str
    spectral_branch_receipt_hash: str
    diagonal_l1_branch_receipt_hash: str
    unique_candidate_identity_hashes: tuple[str, ...]
    candidate_occurrence_count: int
    labels_evicted_after_materialization: bool
    both_q_branches_preserved: bool = True
    every_source_endpoint_considered: bool = True
    ambiguous_endpoint_count: int = 0
    C_outcomes_seen: bool = False
    C_observation_count: int = 0
    D_FQ_values_reused_as_C_observations: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_C_AUTHORITY
    schema: str = SIGNED_WORK_C_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "shell_family_hash",
            "hard_fq_bundle_hash",
            "hard_fq_family_receipt_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "coordinates_hash",
            "design_hash",
            "spectral_branch_receipt_hash",
            "diagonal_l1_branch_receipt_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        identities = tuple(
            _identity(value, name="unique candidate identity hash")
            for value in self.unique_candidate_identity_hashes
        )
        if len(set(identities)) != len(identities):
            raise ValueError("unique C candidate identities repeat")
        object.__setattr__(self, "unique_candidate_identity_hashes", identities)
        occurrences = _positive_int(
            self.candidate_occurrence_count, name="candidate occurrence count"
        )
        object.__setattr__(self, "candidate_occurrence_count", occurrences)
        object.__setattr__(
            self,
            "ambiguous_endpoint_count",
            _nonnegative_int(
                self.ambiguous_endpoint_count, name="ambiguous endpoint count"
            ),
        )
        object.__setattr__(
            self,
            "C_observation_count",
            _nonnegative_int(self.C_observation_count, name="C observation count"),
        )
        if not identities or len(identities) > occurrences:
            raise ValueError("C family candidate deduplication counts are invalid")
        if (
            type(self.labels_evicted_after_materialization) is not bool
            or type(self.both_q_branches_preserved) is not bool
            or not self.both_q_branches_preserved
            or type(self.every_source_endpoint_considered) is not bool
            or not self.every_source_endpoint_considered
            or self.ambiguous_endpoint_count != 0
            or self.C_observation_count != 0
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                "C family is not a complete pre-outcome two-branch freeze",
            )
        _closed_authority_flags(
            (
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="C family receipt",
        )
        if (
            self.authority_scope != SIGNED_WORK_C_AUTHORITY
            or self.schema != SIGNED_WORK_C_FAMILY_SCHEMA
        ):
            raise ValueError("unsupported C family receipt semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FAMILY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkCBridgeFamilyBundle:
    spectral: V1SignedWorkCBranchBundle
    diagonal_l1: V1SignedWorkCBranchBundle
    receipt: V1SignedWorkCFamilyReceipt
    schema: str = SIGNED_WORK_C_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.spectral) is not V1SignedWorkCBranchBundle
            or type(self.diagonal_l1) is not V1SignedWorkCBranchBundle
            or type(self.receipt) is not V1SignedWorkCFamilyReceipt
        ):
            raise TypeError("C family bundle requires exact two-branch values")
        if (
            self.spectral.receipt.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.receipt.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
            or self.spectral.receipt.receipt_hash
            != self.receipt.spectral_branch_receipt_hash
            or self.diagonal_l1.receipt.receipt_hash
            != self.receipt.diagonal_l1_branch_receipt_hash
            or self.schema != SIGNED_WORK_C_BUNDLE_SCHEMA
        ):
            raise ValueError("C family bundle branch identity is inconsistent")
        crosswalks = self.spectral.crosswalks + self.diagonal_l1.crosswalks
        observed_unique: list[str] = []
        for value in crosswalks:
            if value.canonical_candidate_identity_hash not in observed_unique:
                observed_unique.append(value.canonical_candidate_identity_hash)
        if (
            tuple(observed_unique) != self.receipt.unique_candidate_identity_hashes
            or len(crosswalks) != self.receipt.candidate_occurrence_count
        ):
            raise ValueError("C family deduplication identity does not replay")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "family_receipt_hash": self.receipt.receipt_hash,
                "spectral_branch_receipt_hash": self.spectral.receipt.receipt_hash,
                "diagonal_l1_branch_receipt_hash": self.diagonal_l1.receipt.receipt_hash,
            },
            prefix=_BUNDLE_PREFIX,
        )

    @property
    def ladders(self) -> tuple[V1FrozenCandidateLadder, ...]:
        return tuple(
            value
            for value in (self.spectral.ladder, self.diagonal_l1.ladder)
            if value is not None
        )


@dataclass(frozen=True, slots=True)
class _ValidatedBranch:
    shell_bundle: V1SignedWorkDeviceShellExecutionBundle
    shell_ladder: V1DeviceShellFrozenLadderReceipt
    hard_branch: V1SignedWorkHardFQBranchReceipt
    decisions: tuple[V1SignedWorkCEndpointDecisionReceipt, ...]
    safe_labels: tuple[int, ...]


def _branch_pair(
    *,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    hard_fq: V1SignedWorkHardFQFamilyBundle,
) -> tuple[
    tuple[V1SignedWorkDeviceShellExecutionBundle, V1SignedWorkHardFQBranchReceipt],
    tuple[V1SignedWorkDeviceShellExecutionBundle, V1SignedWorkHardFQBranchReceipt],
]:
    return (
        (shell_family.spectral, hard_fq.receipt.spectral),
        (shell_family.diagonal_l1, hard_fq.receipt.diagonal_l1),
    )


def _validate_single_branch_sources(
    *,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    hard_fq: V1SignedWorkHardFQSingleBranchBundle,
    parent_mask: HardMask,
    design: V1SignedWorkCBridgeDesign,
) -> _ValidatedBranch:
    """Replay one exact shell/hard-FQ source axis without borrowing labels."""

    if type(shell) is not V1SignedWorkDeviceShellExecutionBundle:
        raise TypeError("single-branch C bridge requires an exact shell bundle")
    if type(hard_fq) is not V1SignedWorkHardFQSingleBranchBundle:
        raise TypeError("single-branch C bridge requires an exact hard-F/Q bundle")
    if type(parent_mask) is not HardMask:
        raise TypeError("single-branch C bridge requires an exact parent HardMask")
    if type(design) is not V1SignedWorkCBridgeDesign:
        raise TypeError("single-branch C bridge requires an exact execution design")
    parent_hash = _direct_mask_hash(parent_mask, name="single-branch parent mask")
    hard_branch = hard_fq.receipt
    hard_forecast = hard_fq.hbm_forecast
    shell_execution = shell.execution_receipt
    shell_identity = shell.branch_identity
    shell_ladder = shell.frozen_ladder
    if (
        shell_execution.selected_branch is not hard_branch.branch
        or shell_execution.receipt_hash != hard_branch.shell_execution_receipt_hash
        or shell.execution_hash != hard_branch.shell_execution_hash
        or shell_identity.receipt_hash != hard_branch.branch_identity_receipt_hash
        or shell_ladder.receipt_hash != hard_branch.frozen_ladder_receipt_hash
        or shell_ladder.label_commitment != hard_branch.label_commitment
        or hard_branch.problem_hash != shell_execution.problem_hash
        or hard_branch.policy_hash != shell_execution.policy_hash
        or hard_branch.least_work_execution_hash
        != shell_execution.least_work_execution_hash
        or hard_branch.state_identity_hash != shell_execution.state_identity_hash
        or hard_branch.hbm_forecast_hash != hard_forecast.forecast_hash
        or hard_forecast.shell_family_hash != hard_branch.shell_family_hash
        or shell_execution.parent_mask_hash != parent_hash
        or shell_ladder.parent_mask_hash != parent_hash
        or shell_ladder.route_axis_hash != parent_mask.route_axis_hash
        or shell_ladder.route_count != parent_mask.route_count
        or hard_forecast.route_count != parent_mask.route_count
        or len(shell_ladder.actions) != len(hard_branch.endpoints)
        or hard_branch.production_cuda_execution is not shell.production_cuda_execution
        or hard_branch.private_cpu_reference_execution
        is not shell.private_cpu_reference_execution
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "single-branch shell, hard-F/Q replay, forecast, and parent differ",
        )
    decisions: list[V1SignedWorkCEndpointDecisionReceipt] = []
    safe: list[int] = []
    for shell_action, endpoint in zip(
        shell_ladder.actions, hard_branch.endpoints, strict=True
    ):
        if (
            endpoint.branch is not hard_branch.branch
            or endpoint.endpoint_label != shell_action.endpoint_label
            or endpoint.cumulative_removed_count
            != shell_action.cumulative_removed_count
            or endpoint.frozen_action_receipt_hash != shell_action.receipt_hash
            or endpoint.packed_action_hash != shell_action.packed_action_hash
            or endpoint.resulting_mask_hash != shell_action.resulting_mask_hash
            or endpoint.removed_route_index_hash
            != shell_action.removed_route_index_hash
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.ENDPOINT_BIJECTION_MISMATCH,
                "single-branch shell and hard-F/Q endpoints are not bijective",
            )
        status = endpoint.status
        if type(status) is not V1SignedWorkHardFQEndpointStatus:
            _refuse(
                V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
                "hard-F/Q endpoint status is not canonical",
            )
        eligible = status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if eligible:
            safe.append(endpoint.endpoint_label)
        decisions.append(
            V1SignedWorkCEndpointDecisionReceipt(
                branch=hard_branch.branch,
                endpoint_label=endpoint.endpoint_label,
                cumulative_removed_count=endpoint.cumulative_removed_count,
                shell_action_receipt_hash=shell_action.receipt_hash,
                hard_replay_endpoint_receipt_hash=endpoint.receipt_hash,
                canonical_status=status,
                origin_branch_status=_status_value(
                    endpoint.origin_branch_status,
                    name="origin branch status",
                ),
                eligible_for_C=eligible,
            )
        )
    return _ValidatedBranch(
        shell_bundle=shell,
        shell_ladder=shell_ladder,
        hard_branch=hard_branch,
        decisions=tuple(decisions),
        safe_labels=tuple(safe),
    )


def _validate_family_sources(
    *,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    hard_fq: V1SignedWorkHardFQFamilyBundle,
    parent_mask: HardMask,
    coordinates: V1SignedWorkCBridgeCoordinates,
    design: V1SignedWorkCBridgeDesign,
) -> tuple[_ValidatedBranch, _ValidatedBranch]:
    if type(shell_family) is not V1SignedWorkDeviceShellFamilyBundle:
        raise TypeError("C bridge requires the exact signed-work shell family")
    if type(hard_fq) is not V1SignedWorkHardFQFamilyBundle:
        raise TypeError("C bridge requires the exact hard-F/Q family bundle")
    if type(parent_mask) is not HardMask:
        raise TypeError("C bridge requires the exact parent HardMask")
    if type(coordinates) is not V1SignedWorkCBridgeCoordinates:
        raise TypeError("C bridge requires exact registered D coordinates")
    if type(design) is not V1SignedWorkCBridgeDesign:
        raise TypeError("C bridge requires an exact execution design")
    parent_hash = _direct_mask_hash(parent_mask, name="C-bridge parent mask")
    hard_receipt = hard_fq.receipt
    shell_receipt = shell_family.receipt
    if (
        hard_receipt.shell_family_hash != shell_family.family_hash
        or shell_receipt.parent_mask_hash != parent_hash
        or coordinates.parent_mask_hash != parent_hash
        or coordinates.route_axis_hash != parent_mask.route_axis_hash
        or parent_mask.route_count != hard_fq.hbm_forecast.route_count
        or hard_receipt.problem_hash != shell_receipt.problem_hash
        or hard_receipt.policy_hash != shell_receipt.policy_hash
        or hard_receipt.least_work_execution_hash
        != shell_receipt.least_work_execution_hash
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "shell, complete hard-F/Q replay, parent, and D coordinates differ",
        )
    results: list[_ValidatedBranch] = []
    ambiguous: list[tuple[str, int]] = []
    for shell_bundle, hard_branch in _branch_pair(
        shell_family=shell_family, hard_fq=hard_fq
    ):
        shell_execution = shell_bundle.execution_receipt
        shell_identity = shell_bundle.branch_identity
        shell_ladder = shell_bundle.frozen_ladder
        if (
            shell_execution.selected_branch is not hard_branch.branch
            or shell_execution.receipt_hash != hard_branch.shell_execution_receipt_hash
            or shell_bundle.execution_hash != hard_branch.shell_execution_hash
            or shell_identity.receipt_hash != hard_branch.branch_identity_receipt_hash
            or shell_ladder.receipt_hash != hard_branch.frozen_ladder_receipt_hash
            or shell_ladder.label_commitment != hard_branch.label_commitment
            or shell_execution.source_registration_hash
            != coordinates.source_registration_hash
            or shell_execution.registered_d_coordinate_hash
            != coordinates.registered_d_coordinate_hash
            or shell_execution.parent_mask_hash != coordinates.parent_mask_hash
            or shell_ladder.route_axis_hash != coordinates.route_axis_hash
            or shell_ladder.route_count != parent_mask.route_count
            or len(shell_ladder.actions) != len(hard_branch.endpoints)
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "a shell branch and its complete hard-F/Q replay are not one D action axis",
            )
        decisions: list[V1SignedWorkCEndpointDecisionReceipt] = []
        safe: list[int] = []
        for shell_action, endpoint in zip(
            shell_ladder.actions, hard_branch.endpoints, strict=True
        ):
            if (
                endpoint.branch is not hard_branch.branch
                or endpoint.endpoint_label != shell_action.endpoint_label
                or endpoint.cumulative_removed_count
                != shell_action.cumulative_removed_count
                or endpoint.frozen_action_receipt_hash != shell_action.receipt_hash
                or endpoint.packed_action_hash != shell_action.packed_action_hash
                or endpoint.resulting_mask_hash != shell_action.resulting_mask_hash
                or endpoint.removed_route_index_hash
                != shell_action.removed_route_index_hash
            ):
                _refuse(
                    V1SignedWorkCBridgeRefusalCode.ENDPOINT_BIJECTION_MISMATCH,
                    "shell and hard-F/Q endpoint actions are not bijective",
                )
            status = endpoint.status
            if type(status) is not V1SignedWorkHardFQEndpointStatus:
                _refuse(
                    V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
                    "hard-F/Q endpoint status is not canonical",
                )
            if status is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS:
                ambiguous.append((hard_branch.branch.value, endpoint.endpoint_label))
            eligible = status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
            if eligible:
                safe.append(endpoint.endpoint_label)
            origin_status = getattr(endpoint, "origin_branch_status", status)
            decisions.append(
                V1SignedWorkCEndpointDecisionReceipt(
                    branch=hard_branch.branch,
                    endpoint_label=endpoint.endpoint_label,
                    cumulative_removed_count=endpoint.cumulative_removed_count,
                    shell_action_receipt_hash=shell_action.receipt_hash,
                    hard_replay_endpoint_receipt_hash=endpoint.receipt_hash,
                    canonical_status=status,
                    origin_branch_status=_status_value(
                        origin_status, name="origin branch status"
                    ),
                    eligible_for_C=eligible,
                )
            )
        results.append(
            _ValidatedBranch(
                shell_bundle=shell_bundle,
                shell_ladder=shell_ladder,
                hard_branch=hard_branch,
                decisions=tuple(decisions),
                safe_labels=tuple(safe),
            )
        )
    if ambiguous:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.CANONICAL_Q_AMBIGUOUS,
            f"canonical Q remains ambiguous at frozen endpoints {tuple(ambiguous)!r}",
        )
    if not any(value.safe_labels for value in results):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.NO_Q_SAFE_CANDIDATE,
            "neither complete Q branch contains a canonically safe C direction",
        )
    return results[0], results[1]


def _workspace_formula(
    *,
    route_count: int,
    safe_endpoint_count: int,
    route_tile_size: int,
    endpoint_lane_count: int,
    backend_kind: str,
) -> tuple[int, int, int]:
    """Replay the declared logical retained-host, peak-host, and CUDA bounds."""

    if safe_endpoint_count <= 0:
        return 0, 0, 0
    if backend_kind not in {"torch_cuda", "numpy_reference"}:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
            "unsupported single-branch packing backend",
        )
    packed_bytes = (route_count + 7) // 8
    retained_host = safe_endpoint_count * packed_bytes
    lanes = min(safe_endpoint_count, endpoint_lane_count)
    routes = min(route_count, route_tile_size)
    padded_routes = ((routes + 7) // 8) * 8
    tile_packed = padded_routes // 8
    if backend_kind == "torch_cuda":
        # Exact declared logical upper envelope for the implementation below.
        # It charges padded labels, parent bits, all three boolean membership
        # tensors, both int16 membership temporaries, reduction/cast/complement
        # and child lanes, guards and fixed bit vectors.  Views are zero-cost.
        cuda_workspace = (
            padded_routes * (3 + 8 * lanes) + tile_packed * (2 + 6 * lanes) + 8192
        )
        # The returned bytearray family and its immutable-bytes conversion can
        # coexist; the current tile also retains one parent slab and all D2H
        # child lanes.
        peak_host = 2 * retained_host + (lanes + 1) * tile_packed
    else:
        cuda_workspace = 0
        # The small-domain NumPy reference holds the parent, removal and child
        # bool axes plus the packed ndarray/bytes pair at its largest rung.
        peak_host = retained_host + 3 * route_count + 2 * packed_bytes
    return retained_host, peak_host, cuda_workspace


def _workspace_bounds(
    *,
    route_count: int,
    safe_endpoint_count: int,
    design: V1SignedWorkCBridgeDesign,
    backend_kind: str,
) -> tuple[int, int, int]:
    """Admit the exact declared workspace formula against registered caps."""

    retained_host, peak_host, cuda_workspace = _workspace_formula(
        route_count=route_count,
        safe_endpoint_count=safe_endpoint_count,
        route_tile_size=design.route_tile_size,
        endpoint_lane_count=design.endpoint_lane_count,
        backend_kind=backend_kind,
    )
    if (
        peak_host > design.maximum_host_packed_payload_bytes
        or cuda_workspace > design.maximum_cuda_logical_workspace_bytes
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
            "pre-C materialization exceeds a registered operational workspace cap",
        )
    return retained_host, peak_host, cuda_workspace


def _pack_numpy_reference(
    *,
    lease: V1DeviceShellLabelLease,
    parent_mask: HardMask,
    endpoint_count: int,
    safe_labels: tuple[int, ...],
    design: V1SignedWorkCBridgeDesign,
) -> tuple[dict[int, bytes], int]:
    if design.production_cuda_required or not design.allow_private_cpu_reference:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "production C materialization cannot use the NumPy reference",
        )
    if parent_mask.route_count > design.private_reference_route_limit:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.CPU_REFERENCE_SCALE_EXCEEDED,
            "NumPy C materialization exceeded its fixed private route cap",
        )
    labels = lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    array = np.asarray(labels)
    if (
        array.ndim != 1
        or array.dtype != np.dtype(np.uint8)
        or int(array.size) != parent_mask.route_count
        or not array.flags.c_contiguous
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
            "NumPy shell labels are not one contiguous uint8 physical axis",
        )
    if array.size and int(np.max(array)) > endpoint_count:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
            "a shell label lies outside the complete frozen endpoint axis",
        )
    parent_live = parent_mask.to_bool_array()
    if bool(np.any((array != 0) & ~parent_live)):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
            "a shell label removes a route absent from the parent",
        )
    outputs: dict[int, bytes] = {}
    for endpoint_label in safe_labels:
        removed = (array != 0) & (array <= endpoint_label)
        child_live = parent_live & ~removed
        outputs[endpoint_label] = np.packbits(child_live, bitorder="big").tobytes()
    lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    return outputs, 0


def _pack_torch_cuda(
    *,
    lease: V1DeviceShellLabelLease,
    parent_mask: HardMask,
    endpoint_count: int,
    safe_labels: tuple[int, ...],
    design: V1SignedWorkCBridgeDesign,
    cuda_workspace_bytes: int,
) -> tuple[dict[int, bytes], int]:
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1SignedWorkCBridgeRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "production C materialization requires an available CUDA runtime",
        )
    labels = lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    if (
        not isinstance(labels, torch.Tensor)
        or labels.device.type != "cuda"
        or labels.dtype is not torch.uint8
        or labels.ndim != 1
        or int(labels.numel()) != parent_mask.route_count
        or not labels.is_contiguous()
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
            "production shell labels are not one contiguous CUDA uint8 axis",
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(labels.device)
    ceiling = math.floor(design.hbm_ceiling_fraction * int(total_bytes))
    allocated = int(torch.cuda.memory_allocated(labels.device))
    if (
        cuda_workspace_bytes > cuda_reusable_memory_bytes(labels.device, framework=torch, free_bytes=free_bytes)
        or allocated + cuda_workspace_bytes > ceiling
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
            "measured CUDA occupancy invalidates the registered packing workspace",
        )
    packed_size = len(parent_mask.packed_bits)
    outputs = {label: bytearray(packed_size) for label in safe_labels}
    parent_axis = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
    bit_masks = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.uint8,
        device=labels.device,
    )
    bit_weights = bit_masks.to(torch.int16)
    lane_count = design.endpoint_lane_count
    tile = design.route_tile_size
    d2h = 0
    for start in range(0, parent_mask.route_count, tile):
        stop = min(parent_mask.route_count, start + tile)
        count = stop - start
        byte_start = start // 8
        byte_stop = byte_start + (count + 7) // 8
        parent_host = np.array(
            parent_axis[byte_start:byte_stop], dtype=np.uint8, copy=True
        )
        parent_tile = torch.as_tensor(
            parent_host, dtype=torch.uint8, device=labels.device
        )
        label_tile = labels[start:stop]
        padded_count = ((count + 7) // 8) * 8
        if padded_count != count:
            padded = torch.zeros(padded_count, dtype=torch.uint8, device=labels.device)
            padded[:count].copy_(label_tile)
            label_tile = padded
        label_bits = label_tile.reshape(-1, 8)
        parent_live = torch.bitwise_and(parent_tile.reshape(-1, 1), bit_masks) != 0
        invalid = torch.any(label_tile[:count] > endpoint_count)
        selected_dead = torch.any((label_bits != 0) & ~parent_live)
        if bool(invalid.item()) or bool(selected_dead.item()):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
                "CUDA labels left the frozen endpoint or parent-live domain",
            )
        for lane_start in range(0, len(safe_labels), lane_count):
            lane_labels = safe_labels[lane_start : lane_start + lane_count]
            endpoint_labels = torch.tensor(
                lane_labels, dtype=torch.int16, device=labels.device
            ).reshape(-1, 1, 1)
            expanded = label_bits.reshape(1, -1, 8)
            included = (expanded != 0) & (expanded <= endpoint_labels)
            clear = torch.sum(
                included.to(torch.int16) * bit_weights.reshape(1, 1, 8),
                dim=2,
                dtype=torch.int16,
            ).to(torch.uint8)
            children = torch.bitwise_and(
                parent_tile.reshape(1, -1), torch.bitwise_not(clear)
            )
            host = children.detach().to(device="cpu", non_blocking=False).numpy()
            d2h += int(host.nbytes)
            for offset, endpoint_label in enumerate(lane_labels):
                outputs[endpoint_label][byte_start:byte_stop] = host[offset].tobytes()
    lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    return {key: bytes(value) for key, value in outputs.items()}, d2h


def _materialize_validated_branch(
    *,
    validated: _ValidatedBranch,
    parent_mask: HardMask,
    design: V1SignedWorkCBridgeDesign,
) -> tuple[
    tuple[V1SignedWorkCSingleBranchPackedAction, ...],
    int,
    int,
    int,
]:
    safe_count = len(validated.safe_labels)
    expected_backend = (
        "torch_cuda"
        if validated.shell_bundle.production_cuda_execution
        else "numpy_reference"
    )
    retained_host, peak_host, cuda_workspace = _workspace_bounds(
        route_count=parent_mask.route_count,
        safe_endpoint_count=safe_count,
        design=design,
        backend_kind=expected_backend,
    )
    if not safe_count:
        return (), retained_host, peak_host, cuda_workspace
    handle = validated.shell_bundle.label_handle
    try:
        with handle.borrow_for_registered_operation(
            operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
        ) as lease:
            if (
                lease.operation
                is not V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
            ):
                _refuse(
                    V1SignedWorkCBridgeRefusalCode.AUTHORITY_VIOLATION,
                    "shell labels were not borrowed for C directional certification",
                )
            if lease.backend_kind == "torch_cuda":
                packed, d2h = _pack_torch_cuda(
                    lease=lease,
                    parent_mask=parent_mask,
                    endpoint_count=validated.shell_ladder.endpoint_count,
                    safe_labels=validated.safe_labels,
                    design=design,
                    cuda_workspace_bytes=cuda_workspace,
                )
            elif lease.backend_kind == "numpy_reference":
                packed, d2h = _pack_numpy_reference(
                    lease=lease,
                    parent_mask=parent_mask,
                    endpoint_count=validated.shell_ladder.endpoint_count,
                    safe_labels=validated.safe_labels,
                    design=design,
                )
            else:  # pragma: no cover - the lease owns this closed type axis.
                _refuse(
                    V1SignedWorkCBridgeRefusalCode.INVALID_TYPE,
                    "unsupported shell-label backend",
                )
            backend_kind = lease.backend_kind
            device_identity = lease.device_identity
            if backend_kind != expected_backend:
                _refuse(
                    V1SignedWorkCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
                    "shell execution mode and borrowed label backend differ",
                )
    except V1DeviceShellRefusal as error:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
            f"shell label custody refused C materialization: {error}",
        )
    per_action_d2h = (
        (parent_mask.route_count + 7) // 8 if backend_kind == "torch_cuda" else 0
    )
    if d2h != per_action_d2h * safe_count:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.ACTION_IDENTITY_MISMATCH,
            "packed child transfer accounting does not close",
        )
    by_label = {
        value.endpoint_label: value for value in validated.hard_branch.endpoints
    }
    shell_by_label = {
        value.endpoint_label: value for value in validated.shell_ladder.actions
    }
    decisions_by_label = {value.endpoint_label: value for value in validated.decisions}
    results: list[V1SignedWorkCSingleBranchPackedAction] = []
    for endpoint_label in validated.safe_labels:
        endpoint = by_label[endpoint_label]
        shell_action = shell_by_label[endpoint_label]
        packed_bits = packed[endpoint_label]
        packed_sha = hashlib.sha256(packed_bits).hexdigest()
        child = HardMask(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            packed_bits=packed_bits,
        )
        action = V1PackedRemovalAction.from_resulting_mask(
            parent_mask=parent_mask, resulting_mask=child
        )
        if (
            packed_sha != shell_action.resulting_mask_packed_sha256
            or action.removed_count != endpoint.cumulative_removed_count
            or action.action_hash != endpoint.packed_action_hash
            or action.resulting_mask.mask_hash != endpoint.resulting_mask_hash
            or action.removed_route_index_hash != endpoint.removed_route_index_hash
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.ACTION_IDENTITY_MISMATCH,
                f"materialized {validated.hard_branch.branch.value} endpoint "
                f"{endpoint_label} differs from its frozen D action",
            )
        materialization = V1SignedWorkCActionMaterializationReceipt(
            branch=validated.hard_branch.branch,
            endpoint_label=endpoint_label,
            cumulative_removed_count=endpoint.cumulative_removed_count,
            shell_execution_hash=validated.shell_bundle.execution_hash,
            shell_ladder_receipt_hash=validated.shell_ladder.receipt_hash,
            label_commitment=validated.shell_ladder.label_commitment,
            hard_replay_endpoint_receipt_hash=endpoint.receipt_hash,
            backend_kind=backend_kind,
            device_identity=device_identity,
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            parent_mask_hash=_direct_mask_hash(
                parent_mask, name="materialization parent"
            ),
            resulting_mask_packed_sha256=packed_sha,
            packed_action_hash=action.action_hash,
            resulting_mask_hash=action.resulting_mask.mask_hash,
            removed_route_index_hash=action.removed_route_index_hash,
            packed_payload_d2h_bytes=per_action_d2h,
        )
        results.append(
            V1SignedWorkCSingleBranchPackedAction(
                endpoint_decision=decisions_by_label[endpoint_label],
                shell_action=shell_action,
                hard_fq_endpoint=endpoint,
                packed_action=action,
                materialization=materialization,
            )
        )
    return tuple(results), retained_host, peak_host, cuda_workspace


def materialize_v1_signed_work_device_shell_c_single_branch_actions(
    *,
    shell: V1SignedWorkDeviceShellExecutionBundle,
    hard_fq: V1SignedWorkHardFQSingleBranchBundle,
    parent_mask: HardMask,
    design: V1SignedWorkCBridgeDesign = V1SignedWorkCBridgeDesign(),
) -> V1SignedWorkCSingleBranchMaterializationBundle:
    """Materialize exactly the Q-safe actions of one complete branch replay.

    Robustly infeasible and Q-ambiguous endpoints remain in the complete typed
    decision partition but never receive a packed payload.  The function does
    not evict the label handle: a multi-source caller may need atomic custody
    across several independently materialized coordinate/branch aliases.
    """

    validated = _validate_single_branch_sources(
        shell=shell,
        hard_fq=hard_fq,
        parent_mask=parent_mask,
        design=design,
    )
    packed, retained_host, peak_host, cuda_workspace = _materialize_validated_branch(
        validated=validated,
        parent_mask=parent_mask,
        design=design,
    )
    robust = tuple(
        value.endpoint_label
        for value in validated.decisions
        if value.canonical_status
        is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
    )
    ambiguous = tuple(
        value.endpoint_label
        for value in validated.decisions
        if value.canonical_status
        is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS
    )
    return V1SignedWorkCSingleBranchMaterializationBundle(
        branch=validated.hard_branch.branch,
        shell_execution_hash=shell.execution_hash,
        hard_fq_bundle_hash=hard_fq.bundle_hash,
        parent_mask_hash=_direct_mask_hash(
            parent_mask,
            name="single-branch materialization parent",
        ),
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        design_hash=design.design_hash,
        route_tile_size=design.route_tile_size,
        endpoint_lane_count=design.endpoint_lane_count,
        backend_kind=(
            "torch_cuda" if shell.production_cuda_execution else "numpy_reference"
        ),
        endpoint_decisions=validated.decisions,
        packed_actions=packed,
        q_safe_endpoint_labels=validated.safe_labels,
        q_robust_infeasible_endpoint_labels=robust,
        q_ambiguous_endpoint_labels=ambiguous,
        retained_host_packed_payload_bytes=retained_host,
        peak_host_workspace_upper_bound_bytes=peak_host,
        cuda_logical_workspace_upper_bound_bytes=cuda_workspace,
        packed_payload_d2h_bytes=sum(
            value.materialization.packed_payload_d2h_bytes for value in packed
        ),
    )


def _candidate_dedup_key(
    action: V1PackedRemovalAction,
) -> tuple[str, str, str, str, int]:
    """Exact representation/action key; equal counts alone can never deduplicate."""

    if type(action) is not V1PackedRemovalAction:
        raise TypeError("C candidate deduplication requires an exact packed action")
    return (
        hashlib.sha256(action.resulting_mask.packed_bits).hexdigest(),
        action.action_hash,
        action.resulting_mask.mask_hash,
        action.removed_route_index_hash,
        action.removed_count,
    )


def _canonical_candidate_identity(action: V1PackedRemovalAction) -> str:
    key = _candidate_dedup_key(action)
    return content_hash(
        {
            "resulting_mask_packed_sha256": key[0],
            "packed_action_hash": key[1],
            "resulting_mask_hash": key[2],
            "removed_route_index_hash": key[3],
            "removed_count": key[4],
        },
        prefix=_CANDIDATE_PREFIX,
    )


def _build_branch_bundle(
    *,
    validated: _ValidatedBranch,
    packed: tuple[V1SignedWorkCSingleBranchPackedAction, ...],
    canonical_actions: dict[tuple[str, str, str, str, int], V1PackedRemovalAction],
    occurrence_counts: dict[str, int],
    coordinates: V1SignedWorkCBridgeCoordinates,
    design: V1SignedWorkCBridgeDesign,
    parent_mask: HardMask,
) -> V1SignedWorkCBranchBundle:
    frozen_actions: list[V1FrozenLadderAction] = []
    canonical_ids: list[str] = []
    deduplicated: list[bool] = []
    for offset, candidate in enumerate(packed, start=1):
        key = _candidate_dedup_key(candidate.packed_action)
        reused = key in canonical_actions
        canonical_action = canonical_actions.setdefault(key, candidate.packed_action)
        if (
            canonical_action.resulting_mask.packed_bits
            != candidate.packed_action.resulting_mask.packed_bits
        ):
            _refuse(
                V1SignedWorkCBridgeRefusalCode.ACTION_IDENTITY_MISMATCH,
                "equal C candidate identities carried unequal packed bytes",
            )
        canonical_id = _canonical_candidate_identity(canonical_action)
        occurrence_counts[canonical_id] = occurrence_counts.get(canonical_id, 0) + 1
        frozen = V1FrozenLadderAction(
            endpoint_index=offset,
            action_id=(
                f"signed-work-independent-C:{candidate.hard_fq_endpoint.branch.value}:"
                f"source-endpoint-{candidate.hard_fq_endpoint.endpoint_label}:"
                f"{candidate.shell_action.receipt_hash}:"
                f"{candidate.hard_fq_endpoint.receipt_hash}:"
                f"{candidate.materialization.receipt_hash}"
            ),
            packed_removal_action=canonical_action,
        )
        frozen_actions.append(frozen)
        canonical_ids.append(canonical_id)
        deduplicated.append(reused)
    ladder: V1FrozenCandidateLadder | None
    if frozen_actions:
        freeze_hash = content_hash(
            {
                "branch": validated.hard_branch.branch,
                "shell_execution_hash": validated.shell_bundle.execution_hash,
                "shell_ladder_receipt_hash": validated.shell_ladder.receipt_hash,
                "hard_replay_branch_receipt_hash": validated.hard_branch.receipt_hash,
                "coordinates_hash": coordinates.coordinates_hash,
                "design_hash": design.design_hash,
                "source_endpoint_labels": tuple(
                    value.hard_fq_endpoint.endpoint_label for value in packed
                ),
                "materialization_receipt_hashes": tuple(
                    value.materialization.receipt_hash for value in packed
                ),
                "frozen_action_hashes": tuple(
                    value.action_hash for value in frozen_actions
                ),
                "C_outcomes_seen": False,
                "D_FQ_values_reused_as_C_observations": False,
            },
            prefix=_FREEZE_PREFIX,
        )
        ladder = V1FrozenCandidateLadder(
            campaign_contract_hash=coordinates.campaign_contract_hash,
            model_identity_hash=coordinates.model_identity_hash,
            query_contract_hash=coordinates.query_contract_hash,
            preprocessing_hash=coordinates.preprocessing_hash,
            current_mask_hash=coordinates.parent_mask_hash,
            raw_logit_stream_source_hash=coordinates.raw_logit_stream_source_hash,
            sampling_law_hash=coordinates.sampling_law_hash,
            proposal_generation=coordinates.proposal_generation,
            route_count=parent_mask.route_count,
            route_axis_hash=parent_mask.route_axis_hash,
            actions=tuple(frozen_actions),
            discovery_context_ids=coordinates.discovery_context_ids,
            discovery_draw_ids=coordinates.discovery_draw_ids,
            discovery_evidence_hash=validated.hard_branch.receipt_hash,
            freeze_receipt_hash=freeze_hash,
            parent_mask=parent_mask,
        )
    else:
        ladder = None
    crosswalks: list[V1SignedWorkCCrosswalkReceipt] = []
    for candidate, frozen, canonical_id, reused in zip(
        packed, frozen_actions, canonical_ids, deduplicated, strict=True
    ):
        crosswalks.append(
            V1SignedWorkCCrosswalkReceipt(
                branch=candidate.hard_fq_endpoint.branch,
                endpoint_label=candidate.hard_fq_endpoint.endpoint_label,
                cumulative_removed_count=(
                    candidate.hard_fq_endpoint.cumulative_removed_count
                ),
                shell_action_receipt_hash=candidate.shell_action.receipt_hash,
                hard_replay_endpoint_receipt_hash=(
                    candidate.hard_fq_endpoint.receipt_hash
                ),
                hard_replay_status=candidate.hard_fq_endpoint.status,
                packed_action_hash=candidate.packed_action.action_hash,
                resulting_mask_hash=(candidate.packed_action.resulting_mask.mask_hash),
                removed_route_index_hash=(
                    candidate.packed_action.removed_route_index_hash
                ),
                materialization_receipt_hash=candidate.materialization.receipt_hash,
                frozen_action_hash=frozen.action_hash,
                canonical_candidate_identity_hash=canonical_id,
                canonical_candidate_occurrence=occurrence_counts[canonical_id],
                cross_branch_exact_deduplication=reused,
            )
        )
    robust = tuple(
        value.endpoint_label
        for value in validated.decisions
        if value.canonical_status
        is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE
    )
    receipt = V1SignedWorkCBranchReceipt(
        branch=validated.hard_branch.branch,
        shell_execution_hash=validated.shell_bundle.execution_hash,
        shell_execution_receipt_hash=(
            validated.shell_bundle.execution_receipt.receipt_hash
        ),
        shell_branch_identity_receipt_hash=(
            validated.shell_bundle.branch_identity.receipt_hash
        ),
        shell_ladder_receipt_hash=validated.shell_ladder.receipt_hash,
        hard_replay_branch_receipt_hash=validated.hard_branch.receipt_hash,
        coordinates_hash=coordinates.coordinates_hash,
        design_hash=design.design_hash,
        complete_source_endpoint_count=len(validated.decisions),
        endpoint_decision_receipt_hashes=tuple(
            value.receipt_hash for value in validated.decisions
        ),
        q_safe_endpoint_labels=validated.safe_labels,
        q_robust_infeasible_endpoint_labels=robust,
        materialization_receipt_hashes=tuple(
            value.materialization.receipt_hash for value in packed
        ),
        crosswalk_receipt_hashes=tuple(value.receipt_hash for value in crosswalks),
        frozen_ladder_hash=None if ladder is None else ladder.ladder_hash,
    )
    return V1SignedWorkCBranchBundle(
        ladder=ladder,
        endpoint_decisions=validated.decisions,
        materializations=tuple(value.materialization for value in packed),
        crosswalks=tuple(crosswalks),
        receipt=receipt,
    )


def materialize_v1_signed_work_device_shell_c_family(
    *,
    shell_family: V1SignedWorkDeviceShellFamilyBundle,
    hard_fq: V1SignedWorkHardFQFamilyBundle,
    parent_mask: HardMask,
    coordinates: V1SignedWorkCBridgeCoordinates,
    design: V1SignedWorkCBridgeDesign = V1SignedWorkCBridgeDesign(),
    evict_labels_after_materialization: bool = True,
) -> V1SignedWorkCBridgeFamilyBundle:
    """Freeze the exact canonically Q-safe two-branch ladder for stream C.

    Ambiguous canonical Q status refuses before any label borrow.  Robustly
    infeasible rungs remain represented by endpoint-decision receipts but are
    not materialized.  Because Q can be nonmonotone, every endpoint is
    considered and later safe rungs survive earlier infeasible rungs.
    """

    if type(evict_labels_after_materialization) is not bool:
        raise TypeError("evict_labels_after_materialization must be an exact bool")
    spectral_source, diagonal_source = _validate_family_sources(
        shell_family=shell_family,
        hard_fq=hard_fq,
        parent_mask=parent_mask,
        coordinates=coordinates,
        design=design,
    )
    spectral_bounds = _workspace_bounds(
        route_count=parent_mask.route_count,
        safe_endpoint_count=len(spectral_source.safe_labels),
        design=design,
        backend_kind=(
            "torch_cuda"
            if shell_family.spectral.production_cuda_execution
            else "numpy_reference"
        ),
    )
    diagonal_bounds = _workspace_bounds(
        route_count=parent_mask.route_count,
        safe_endpoint_count=len(diagonal_source.safe_labels),
        design=design,
        backend_kind=(
            "torch_cuda"
            if shell_family.diagonal_l1.production_cuda_execution
            else "numpy_reference"
        ),
    )
    # The spectral payload remains live while the diagonal branch is packed.
    # This exact declared family bound refuses before either label borrow.
    combined_peak_host = max(
        spectral_bounds[1],
        spectral_bounds[0] + diagonal_bounds[1],
    )
    if combined_peak_host > design.maximum_host_packed_payload_bytes:
        _refuse(
            V1SignedWorkCBridgeRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
            "two-branch pre-C host workspace exceeds its registered cap",
        )
    spectral_materialization = (
        materialize_v1_signed_work_device_shell_c_single_branch_actions(
            shell=shell_family.spectral,
            hard_fq=V1SignedWorkHardFQSingleBranchBundle(
                receipt=hard_fq.receipt.spectral,
                hbm_forecast=hard_fq.hbm_forecast,
            ),
            parent_mask=parent_mask,
            design=design,
        )
    )
    diagonal_materialization = (
        materialize_v1_signed_work_device_shell_c_single_branch_actions(
            shell=shell_family.diagonal_l1,
            hard_fq=V1SignedWorkHardFQSingleBranchBundle(
                receipt=hard_fq.receipt.diagonal_l1,
                hbm_forecast=hard_fq.hbm_forecast,
            ),
            parent_mask=parent_mask,
            design=design,
        )
    )
    if tuple(
        value.receipt_hash for value in spectral_materialization.endpoint_decisions
    ) != tuple(value.receipt_hash for value in spectral_source.decisions) or tuple(
        value.receipt_hash for value in diagonal_materialization.endpoint_decisions
    ) != tuple(
        value.receipt_hash for value in diagonal_source.decisions
    ):
        _refuse(
            V1SignedWorkCBridgeRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "family preflight and single-branch materialization decisions differ",
        )
    spectral_packed = spectral_materialization.packed_actions
    diagonal_packed = diagonal_materialization.packed_actions
    canonical_actions: dict[tuple[str, str, str, str, int], V1PackedRemovalAction] = {}
    occurrence_counts: dict[str, int] = {}
    spectral = _build_branch_bundle(
        validated=spectral_source,
        packed=spectral_packed,
        canonical_actions=canonical_actions,
        occurrence_counts=occurrence_counts,
        coordinates=coordinates,
        design=design,
        parent_mask=parent_mask,
    )
    diagonal = _build_branch_bundle(
        validated=diagonal_source,
        packed=diagonal_packed,
        canonical_actions=canonical_actions,
        occurrence_counts=occurrence_counts,
        coordinates=coordinates,
        design=design,
        parent_mask=parent_mask,
    )
    crosswalks = spectral.crosswalks + diagonal.crosswalks
    unique: list[str] = []
    for value in crosswalks:
        if value.canonical_candidate_identity_hash not in unique:
            unique.append(value.canonical_candidate_identity_hash)
    receipt = V1SignedWorkCFamilyReceipt(
        shell_family_hash=shell_family.family_hash,
        hard_fq_bundle_hash=hard_fq.bundle_hash,
        hard_fq_family_receipt_hash=hard_fq.receipt.receipt_hash,
        parent_mask_hash=_direct_mask_hash(parent_mask, name="C-family parent mask"),
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        coordinates_hash=coordinates.coordinates_hash,
        design_hash=design.design_hash,
        spectral_branch_receipt_hash=spectral.receipt.receipt_hash,
        diagonal_l1_branch_receipt_hash=diagonal.receipt.receipt_hash,
        unique_candidate_identity_hashes=tuple(unique),
        candidate_occurrence_count=len(crosswalks),
        labels_evicted_after_materialization=evict_labels_after_materialization,
    )
    bundle = V1SignedWorkCBridgeFamilyBundle(
        spectral=spectral,
        diagonal_l1=diagonal,
        receipt=receipt,
    )
    # Every operation-typed lease has exited and the entire durable return
    # value has validated.  Constructing the family receipt or bundle must not
    # destroy the only exact label source on a late validation failure.  The
    # atomic operation below either releases both vectors or preserves both.
    if evict_labels_after_materialization:
        try:
            shell_family.evict_labels_atomically()
        except V1DeviceShellRefusal as error:
            _refuse(
                V1SignedWorkCBridgeRefusalCode.LABEL_IDENTITY_MISMATCH,
                f"atomic post-C-materialization label eviction refused: {error}",
            )
    return bundle


__all__ = [
    "SIGNED_WORK_C_AUTHORITY",
    "SIGNED_WORK_C_BRANCH_SCHEMA",
    "SIGNED_WORK_C_BRIDGE_COORDINATES_SCHEMA",
    "SIGNED_WORK_C_BRIDGE_DESIGN_SCHEMA",
    "SIGNED_WORK_C_BUNDLE_SCHEMA",
    "SIGNED_WORK_C_CROSSWALK_SCHEMA",
    "SIGNED_WORK_C_ENDPOINT_DECISION_SCHEMA",
    "SIGNED_WORK_C_FAMILY_SCHEMA",
    "SIGNED_WORK_C_MATERIALIZATION_SCHEMA",
    "SIGNED_WORK_C_SINGLE_BRANCH_ACTION_SCHEMA",
    "SIGNED_WORK_C_SINGLE_BRANCH_BUNDLE_SCHEMA",
    "V1SignedWorkCActionMaterializationReceipt",
    "V1SignedWorkCBranchBundle",
    "V1SignedWorkCBranchReceipt",
    "V1SignedWorkCBridgeCoordinates",
    "V1SignedWorkCBridgeDesign",
    "V1SignedWorkCBridgeFamilyBundle",
    "V1SignedWorkCBridgeRefusal",
    "V1SignedWorkCBridgeRefusalCode",
    "V1SignedWorkCCrosswalkReceipt",
    "V1SignedWorkCEndpointDecisionReceipt",
    "V1SignedWorkCFamilyReceipt",
    "V1SignedWorkCSingleBranchMaterializationBundle",
    "V1SignedWorkCSingleBranchPackedAction",
    "materialize_v1_signed_work_device_shell_c_family",
    "materialize_v1_signed_work_device_shell_c_single_branch_actions",
]
