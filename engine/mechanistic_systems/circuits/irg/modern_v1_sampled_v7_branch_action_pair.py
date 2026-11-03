"""Source-issued custody for the two canonical V7 branch actions.

The live pre-C common-coordinate seam already owns exactly one ``spectral``
and one ``diagonal_l1`` candidate crosswalk.  This module freezes those two
aliases without accepting a caller-selected action.  Equal physical masks are
stored once while both branch aliases remain explicit.

This is representation custody only.  In particular, the current fixed-primal
shell union contains branch-local uncertainty shells but *not* the symmetric
difference of the two branch-baseline hard actions.  Consequently neither this
pair nor that shell union is a complete action-family/decision-stability
witness, and neither can open C, K, O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
from numbers import Integral
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointReceipt,
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_pre_c_coordinator import (
    V1SignedWorkPreCLiveCommonCoordinateContext,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkCoordinateCandidateCrosswalk,
    V1SignedWorkPackedActionDedupIdentity,
    v1_signed_work_packed_action_dedup_identity,
)


SAMPLED_V7_BRANCH_ACTION_PAIR_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_source_issued_branch_action_pair_v1"
)
SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_source_issued_branch_action_alias_v1"
)
SAMPLED_V7_BRANCH_NO_ACTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_source_issued_branch_no_action_v1"
)
SAMPLED_V7_PHYSICAL_ACTION_CUSTODY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_deduplicated_physical_action_custody_v1"
)
SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY: Final = (
    "source_issued_exact_spectral_diagonal_l1_packed_action_custody_only_"
    "branch_aliases_preserved_physical_payloads_deduplicated_"
    "no_action_stability_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_V7_SHELL_UNION_LIMITATION: Final = (
    "fixed_primal_shell_union_omits_branch_baseline_symmetric_difference_"
    "and_is_not_a_complete_action_family_witness_v1"
)

_PAIR_PREFIX: Final = "partition-free-v1-sampled-v7-branch-action-pair-"
_ALIAS_PREFIX: Final = "partition-free-v1-sampled-v7-branch-action-alias-"
_NO_ACTION_PREFIX: Final = "partition-free-v1-sampled-v7-branch-no-action-"
_PHYSICAL_PREFIX: Final = "partition-free-v1-sampled-v7-physical-action-custody-"
_PHYSICAL_ID_PREFIX: Final = "partition-free-v1-sampled-v7-physical-action-id-"
_ISSUER: Final = object()
_BRANCH_AXIS: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)


class V1SampledV7BranchActionPairRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    BRANCH_ORDER_MISMATCH = "branch_order_mismatch"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    Q_STATUS_MISMATCH = "q_status_mismatch"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    INCOMPLETE_ALIAS_PARTITION = "incomplete_alias_partition"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7BranchActionPairRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7BranchActionPairRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7BranchActionPairRefusalCode:
            raise TypeError("branch-action-pair refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7BranchActionPairRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7BranchActionPairRefusal(code, message)


class V1SampledV7BranchNoActionCode(str, Enum):
    Q_ROBUST_INFEASIBLE = "q_robust_infeasible"
    Q_ENVELOPE_AMBIGUOUS = "q_envelope_ambiguous"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _closed_authority(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its source-custody boundary",
        )


def v1_sampled_v7_physical_action_identity_hash(
    identity: V1SignedWorkPackedActionDedupIdentity,
) -> str:
    """Canonical physical-action identity shared by source and replay joints."""

    if (
        type(identity) is not tuple
        or len(identity) != 5
        or any(type(value) is not str or not value for value in identity[:4])
        or type(identity[4]) is not int
        or identity[4] <= 0
    ):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            "physical action identity hashing requires the exact dedup identity",
        )
    return content_hash(identity, prefix=_PHYSICAL_ID_PREFIX)


def _no_action_code(
    status: V1SignedWorkHardFQEndpointStatus,
) -> V1SampledV7BranchNoActionCode:
    if status is V1SignedWorkHardFQEndpointStatus.Q_ROBUST_INFEASIBLE:
        return V1SampledV7BranchNoActionCode.Q_ROBUST_INFEASIBLE
    if status is V1SignedWorkHardFQEndpointStatus.Q_ENVELOPE_AMBIGUOUS:
        return V1SampledV7BranchNoActionCode.Q_ENVELOPE_AMBIGUOUS
    _refuse(
        V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
        "a Q-safe branch cannot issue a no-action receipt",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchNoActionReceipt:
    """Array-free source receipt for one unsafe or unresolved branch alias."""

    branch: V1RelaxedKKTBranch
    q_status: V1SignedWorkHardFQEndpointStatus
    code: V1SampledV7BranchNoActionCode
    candidate_crosswalk_hash: str
    endpoint_decision_receipt_hash: str
    hard_fq_endpoint_receipt_hash: str
    action_available: bool = False
    action_stability_authority: bool = False
    C_K_O_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_NO_ACTION_SCHEMA
    _issuance_token: InitVar[object] = None

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUER:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch no-action receipts are source-issued only",
            )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
                "branch no-action receipt requires an exact Q branch",
            )
        if type(self.q_status) is not V1SignedWorkHardFQEndpointStatus:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
                "branch no-action receipt requires an exact Q status",
            )
        if type(self.code) is not V1SampledV7BranchNoActionCode or (
            self.code is not _no_action_code(self.q_status)
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
                "branch no-action code does not replay its Q status",
            )
        for name in (
            "candidate_crosswalk_hash",
            "endpoint_decision_receipt_hash",
            "hard_fq_endpoint_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        _closed_authority(
            (
                self.action_available,
                self.action_stability_authority,
                self.C_K_O_or_commit_authority,
            ),
            owner="branch no-action receipt",
        )
        if self.schema != SAMPLED_V7_BRANCH_NO_ACTION_SCHEMA:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch no-action semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch,
                "q_status": self.q_status,
                "code": self.code,
                "candidate_crosswalk_hash": self.candidate_crosswalk_hash,
                "endpoint_decision_receipt_hash": (
                    self.endpoint_decision_receipt_hash
                ),
                "hard_fq_endpoint_receipt_hash": self.hard_fq_endpoint_receipt_hash,
                "action_available": self.action_available,
                "action_stability_authority": self.action_stability_authority,
                "C_K_O_or_commit_authority": self.C_K_O_or_commit_authority,
            },
            prefix=_NO_ACTION_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchActionAlias:
    """One canonical branch alias, independently retained from payload dedup."""

    branch: V1RelaxedKKTBranch
    q_status: V1SignedWorkHardFQEndpointStatus
    coordinate_candidate_key_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    source_endpoint_label: int
    candidate_crosswalk_hash: str
    source_materialization_bundle_hash: str
    endpoint_decision_receipt_hash: str
    hard_fq_bundle_hash: str
    hard_fq_branch_receipt_hash: str
    hard_fq_endpoint_receipt_hash: str
    shell_execution_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    physical_action_ordinal: int | None
    physical_action_identity_hash: str | None
    packed_action_hash: str | None
    resulting_mask_hash: str | None
    resulting_mask_packed_sha256: str | None
    removed_route_index_hash: str | None
    removed_count: int | None
    materialization_receipt_hash: str | None
    source_packed_action_hash: str | None
    no_action: V1SampledV7BranchNoActionReceipt | None
    distinct_branch_alias_preserved: bool = True
    action_stability_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY
    schema: str = SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUER:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch action aliases are source-issued only",
            )
        if type(self.branch) is not V1RelaxedKKTBranch or (
            type(self.q_status) is not V1SignedWorkHardFQEndpointStatus
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
                "branch alias lost its exact branch or Q status",
            )
        for name in (
            "coordinate_candidate_key_hash",
            "resource_coordinate_hash",
            "candidate_crosswalk_hash",
            "source_materialization_bundle_hash",
            "endpoint_decision_receipt_hash",
            "hard_fq_bundle_hash",
            "hard_fq_branch_receipt_hash",
            "hard_fq_endpoint_receipt_hash",
            "shell_execution_hash",
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
        if type(self.distinct_branch_alias_preserved) is not bool or (
            not self.distinct_branch_alias_preserved
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "branch alias was collapsed during physical-action deduplication",
            )
        _closed_authority(
            (
                self.action_stability_authority,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="branch action alias",
        )
        if (
            self.authority_scope != SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY
            or self.schema != SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch action alias semantics changed",
            )
        self._validate_payload_partition()
        observed = self._payload_hash()
        if self._sealed_hash is not None and observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "branch action alias changed after issuance",
            )
        object.__setattr__(self, "_sealed_hash", observed)

    def _validate_payload_partition(self) -> None:
        safe = self.q_status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        if safe:
            if (
                self.no_action is not None
                or self.physical_action_ordinal is None
                or self.physical_action_identity_hash is None
                or self.packed_action_hash is None
                or self.resulting_mask_hash is None
                or self.resulting_mask_packed_sha256 is None
                or self.removed_route_index_hash is None
                or self.removed_count is None
                or self.materialization_receipt_hash is None
                or self.source_packed_action_hash is None
            ):
                _refuse(
                    V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
                    "Q-safe branch alias lacks its exact action payload",
                )
            _positive_int(
                self.physical_action_ordinal,
                name="physical action ordinal",
            )
            _positive_int(self.removed_count, name="removed count")
            for name in (
                "physical_action_identity_hash",
                "packed_action_hash",
                "resulting_mask_hash",
                "resulting_mask_packed_sha256",
                "removed_route_index_hash",
                "materialization_receipt_hash",
                "source_packed_action_hash",
            ):
                _identity(getattr(self, name), name=name)
        else:
            if (
                self.physical_action_ordinal is not None
                or self.physical_action_identity_hash is not None
                or self.packed_action_hash is not None
                or self.resulting_mask_hash is not None
                or self.resulting_mask_packed_sha256 is not None
                or self.removed_route_index_hash is not None
                or self.removed_count is not None
                or self.materialization_receipt_hash is not None
                or self.source_packed_action_hash is not None
                or type(self.no_action) is not V1SampledV7BranchNoActionReceipt
                or self.no_action.branch is not self.branch
                or self.no_action.q_status is not self.q_status
                or self.no_action.candidate_crosswalk_hash
                != self.candidate_crosswalk_hash
                or self.no_action.endpoint_decision_receipt_hash
                != self.endpoint_decision_receipt_hash
                or self.no_action.hard_fq_endpoint_receipt_hash
                != self.hard_fq_endpoint_receipt_hash
            ):
                _refuse(
                    V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
                    "unsafe or ambiguous branch did not retain one typed no-action receipt",
                )

    def _payload_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch,
                "q_status": self.q_status,
                "coordinate_candidate_key_hash": self.coordinate_candidate_key_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "source_endpoint_label": self.source_endpoint_label,
                "candidate_crosswalk_hash": self.candidate_crosswalk_hash,
                "source_materialization_bundle_hash": (
                    self.source_materialization_bundle_hash
                ),
                "endpoint_decision_receipt_hash": self.endpoint_decision_receipt_hash,
                "hard_fq_bundle_hash": self.hard_fq_bundle_hash,
                "hard_fq_branch_receipt_hash": self.hard_fq_branch_receipt_hash,
                "hard_fq_endpoint_receipt_hash": self.hard_fq_endpoint_receipt_hash,
                "shell_execution_hash": self.shell_execution_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "physical_action_ordinal": self.physical_action_ordinal,
                "physical_action_identity_hash": self.physical_action_identity_hash,
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "resulting_mask_packed_sha256": (
                    self.resulting_mask_packed_sha256
                ),
                "removed_route_index_hash": self.removed_route_index_hash,
                "removed_count": self.removed_count,
                "materialization_receipt_hash": self.materialization_receipt_hash,
                "source_packed_action_hash": self.source_packed_action_hash,
                "no_action_receipt_hash": (
                    None if self.no_action is None else self.no_action.receipt_hash
                ),
                "distinct_branch_alias_preserved": (
                    self.distinct_branch_alias_preserved
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_ALIAS_PREFIX,
        )

    def assert_integrity(self) -> str:
        self._validate_payload_partition()
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "branch action alias changed after issuance",
            )
        return observed

    @property
    def alias_hash(self) -> str:
        return self.assert_integrity()

    @property
    def has_action(self) -> bool:
        return self.physical_action_ordinal is not None

    def __reduce__(self) -> None:
        raise TypeError("V7 branch action aliases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> None:
        del protocol
        raise TypeError("V7 branch action aliases are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7PhysicalActionCustody:
    """One exact packed child, shared by all physically equal branch aliases."""

    physical_action_ordinal: int
    dedup_identity: V1SignedWorkPackedActionDedupIdentity
    physical_action_identity_hash: str
    branch_alias_axis: tuple[V1RelaxedKKTBranch, ...]
    branch_alias_hashes: tuple[str, ...]
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    packed_payload_bytes: int
    route_scale_array_payload_retained: bool = False
    action_stability_authority: bool = False
    C_K_O_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_PHYSICAL_ACTION_CUSTODY_SCHEMA
    _packed_action: V1PackedRemovalAction = field(default=None, repr=False, compare=False)
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUER:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "physical action custody is source-issued only",
            )
        if type(self._packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
                "physical action custody requires one exact packed action",
            )
        object.__setattr__(
            self,
            "physical_action_ordinal",
            _positive_int(
                self.physical_action_ordinal,
                name="physical action ordinal",
            ),
        )
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed count"),
        )
        object.__setattr__(
            self,
            "packed_payload_bytes",
            _positive_int(self.packed_payload_bytes, name="packed payload bytes"),
        )
        for name in (
            "physical_action_identity_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        branches = tuple(self.branch_alias_axis)
        hashes = tuple(
            _identity(value, name="branch alias hash")
            for value in self.branch_alias_hashes
        )
        if (
            not branches
            or any(type(value) is not V1RelaxedKKTBranch for value in branches)
            or tuple(value for value in _BRANCH_AXIS if value in branches) != branches
            or len(set(branches)) != len(branches)
            or len(hashes) != len(branches)
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "physical action custody lost or reordered branch aliases",
            )
        object.__setattr__(self, "branch_alias_axis", branches)
        object.__setattr__(self, "branch_alias_hashes", hashes)
        _closed_authority(
            (
                self.route_scale_array_payload_retained,
                self.action_stability_authority,
                self.C_K_O_or_commit_authority,
            ),
            owner="physical action custody",
        )
        if self.schema != SAMPLED_V7_PHYSICAL_ACTION_CUSTODY_SCHEMA:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "physical action custody semantics changed",
            )
        observed = self._payload_hash()
        if self._sealed_hash is not None and observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "physical action custody changed after issuance",
            )
        object.__setattr__(self, "_sealed_hash", observed)

    def _payload_hash(self) -> str:
        action = self._packed_action
        identity = v1_signed_work_packed_action_dedup_identity(action)
        packed_sha = hashlib.sha256(action.resulting_mask.packed_bits).hexdigest()
        if (
            identity != self.dedup_identity
            or v1_sampled_v7_physical_action_identity_hash(identity)
            != self.physical_action_identity_hash
            or action.action_hash != self.packed_action_hash
            or action.resulting_mask.mask_hash != self.resulting_mask_hash
            or packed_sha != self.resulting_mask_packed_sha256
            or action.removed_route_index_hash != self.removed_route_index_hash
            or action.removed_count != self.removed_count
            or len(action.resulting_mask.packed_bits) != self.packed_payload_bytes
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.ACTION_IDENTITY_MISMATCH,
                "deduplicated physical action changed after source issuance",
            )
        return content_hash(
            {
                "schema": self.schema,
                "physical_action_ordinal": self.physical_action_ordinal,
                "dedup_identity": self.dedup_identity,
                "physical_action_identity_hash": self.physical_action_identity_hash,
                "branch_alias_axis": self.branch_alias_axis,
                "branch_alias_hashes": self.branch_alias_hashes,
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
                "removed_route_index_hash": self.removed_route_index_hash,
                "removed_count": self.removed_count,
                "packed_payload_bytes": self.packed_payload_bytes,
                "route_scale_array_payload_retained": (
                    self.route_scale_array_payload_retained
                ),
            },
            prefix=_PHYSICAL_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "physical action custody changed after issuance",
            )
        return observed

    @property
    def custody_hash(self) -> str:
        return self.assert_integrity()

    @property
    def packed_action(self) -> V1PackedRemovalAction:
        self.assert_integrity()
        return self._packed_action

    def __reduce__(self) -> None:
        raise TypeError("V7 physical action custody is nonserializable")

    def __reduce_ex__(self, protocol: int) -> None:
        del protocol
        raise TypeError("V7 physical action custody is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchActionPair:
    """Exact two-branch aliases plus deduplicated packed physical actions."""

    source_bundle_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    response_geometry_problem_hash: str
    response_geometry_source_hash: str
    least_work_problem_hash: str
    policy_bundle_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    endpoint_plan_hash: str
    parent_mask_hash: str
    parent_retained_count: int
    route_axis_hash: str
    route_count: int
    live_d_source_identity_hash: str | None
    branch_axis: tuple[V1RelaxedKKTBranch, V1RelaxedKKTBranch]
    branch_aliases: tuple[V1SampledV7BranchActionAlias, V1SampledV7BranchActionAlias]
    physical_actions: tuple[V1SampledV7PhysicalActionCustody, ...]
    branch_physical_action_ordinals: tuple[int | None, int | None]
    physical_action_count: int
    distinct_branch_alias_count: int
    physical_payloads_deduplicated: bool
    retains_packed_child_bytes_only: bool = True
    route_scale_array_payload_retained: bool = False
    shell_union_includes_branch_baseline_symmetric_difference: bool = False
    complete_action_family_witness: bool = False
    action_stability_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    shell_union_limitation: str = SAMPLED_V7_SHELL_UNION_LIMITATION
    authority_scope: str = SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY
    schema: str = SAMPLED_V7_BRANCH_ACTION_PAIR_SCHEMA
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUER:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch action pairs are source-issued only",
            )
        for name in (
            "source_bundle_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "response_geometry_problem_hash",
            "response_geometry_source_hash",
            "least_work_problem_hash",
            "policy_bundle_hash",
            "resource_coordinate_hash",
            "endpoint_plan_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "shell_union_limitation",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.live_d_source_identity_hash is not None:
            object.__setattr__(
                self,
                "live_d_source_identity_hash",
                _identity(
                    self.live_d_source_identity_hash,
                    name="live D source identity hash",
                ),
            )
        object.__setattr__(
            self,
            "coordinate_ordinal",
            _positive_int(self.coordinate_ordinal, name="coordinate ordinal"),
        )
        object.__setattr__(
            self,
            "parent_retained_count",
            _positive_int(self.parent_retained_count, name="parent retained count"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        aliases = tuple(self.branch_aliases)
        physical = tuple(self.physical_actions)
        ordinals = tuple(self.branch_physical_action_ordinals)
        if (
            tuple(self.branch_axis) != _BRANCH_AXIS
            or len(aliases) != 2
            or any(type(value) is not V1SampledV7BranchActionAlias for value in aliases)
            or tuple(value.branch for value in aliases) != _BRANCH_AXIS
            or len(ordinals) != 2
            or ordinals != tuple(value.physical_action_ordinal for value in aliases)
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.BRANCH_ORDER_MISMATCH,
                "branch-action pair is not canonical spectral then diagonal_l1",
            )
        if any(type(value) is not V1SampledV7PhysicalActionCustody for value in physical):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
                "branch-action pair physical payloads must be exact custody objects",
            )
        if tuple(value.physical_action_ordinal for value in physical) != tuple(
            range(1, len(physical) + 1)
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "physical action ordinals are incomplete",
            )
        object.__setattr__(self, "branch_aliases", aliases)
        object.__setattr__(self, "physical_actions", physical)
        object.__setattr__(self, "branch_physical_action_ordinals", ordinals)
        expected_count = len(physical)
        if (
            _nonnegative_int(self.physical_action_count, name="physical action count")
            != expected_count
            or _positive_int(
                self.distinct_branch_alias_count,
                name="distinct branch alias count",
            )
            != 2
            or type(self.physical_payloads_deduplicated) is not bool
            or self.physical_payloads_deduplicated
            is not (sum(value.has_action for value in aliases) > expected_count)
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                "branch alias/physical payload accounting does not replay",
            )
        flags = (
            self.retains_packed_child_bytes_only,
            self.route_scale_array_payload_retained,
            self.shell_union_includes_branch_baseline_symmetric_difference,
            self.complete_action_family_witness,
            self.action_stability_authority,
            self.C_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch action pair fabricated completeness or downstream authority",
            )
        if (
            self.shell_union_limitation != SAMPLED_V7_SHELL_UNION_LIMITATION
            or self.authority_scope != SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY
            or self.schema != SAMPLED_V7_BRANCH_ACTION_PAIR_SCHEMA
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.AUTHORITY_VIOLATION,
                "branch action pair semantics changed",
            )
        observed = self._payload_hash()
        if self._sealed_hash is not None and observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "branch action pair changed after issuance",
            )
        object.__setattr__(self, "_sealed_hash", observed)

    def _payload_hash(self) -> str:
        custody_hashes = tuple(value.assert_integrity() for value in self.physical_actions)
        alias_hashes = tuple(value.assert_integrity() for value in self.branch_aliases)
        by_ordinal = {
            value.physical_action_ordinal: value for value in self.physical_actions
        }
        for alias in self.branch_aliases:
            if alias.physical_action_ordinal is None:
                continue
            custody = by_ordinal.get(alias.physical_action_ordinal)
            if (
                custody is None
                or alias.physical_action_identity_hash
                != custody.physical_action_identity_hash
                or alias.packed_action_hash != custody.packed_action_hash
                or alias.resulting_mask_hash != custody.resulting_mask_hash
                or alias.removed_route_index_hash != custody.removed_route_index_hash
                or alias.removed_count != custody.removed_count
                or alias.branch not in custody.branch_alias_axis
                or alias.alias_hash
                != custody.branch_alias_hashes[
                    custody.branch_alias_axis.index(alias.branch)
                ]
            ):
                _refuse(
                    V1SampledV7BranchActionPairRefusalCode.INCOMPLETE_ALIAS_PARTITION,
                    "branch alias does not resolve to its deduplicated physical payload",
                )
        return content_hash(
            {
                "schema": self.schema,
                "source_bundle_hash": self.source_bundle_hash,
                "source_registration_hash": self.source_registration_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "response_geometry_problem_hash": self.response_geometry_problem_hash,
                "response_geometry_source_hash": self.response_geometry_source_hash,
                "least_work_problem_hash": self.least_work_problem_hash,
                "policy_bundle_hash": self.policy_bundle_hash,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "endpoint_plan_hash": self.endpoint_plan_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "parent_retained_count": self.parent_retained_count,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "live_d_source_identity_hash": self.live_d_source_identity_hash,
                "branch_axis": self.branch_axis,
                "branch_alias_hashes": alias_hashes,
                "physical_action_custody_hashes": custody_hashes,
                "branch_physical_action_ordinals": (
                    self.branch_physical_action_ordinals
                ),
                "physical_action_count": self.physical_action_count,
                "distinct_branch_alias_count": self.distinct_branch_alias_count,
                "physical_payloads_deduplicated": self.physical_payloads_deduplicated,
                "retains_packed_child_bytes_only": self.retains_packed_child_bytes_only,
                "route_scale_array_payload_retained": (
                    self.route_scale_array_payload_retained
                ),
                "shell_union_includes_branch_baseline_symmetric_difference": (
                    self.shell_union_includes_branch_baseline_symmetric_difference
                ),
                "complete_action_family_witness": self.complete_action_family_witness,
                "action_stability_authority": self.action_stability_authority,
                "shell_union_limitation": self.shell_union_limitation,
                "authority_scope": self.authority_scope,
            },
            prefix=_PAIR_PREFIX,
        )

    def assert_integrity(self) -> str:
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
                "branch action pair changed after issuance",
            )
        return observed

    @property
    def pair_hash(self) -> str:
        return self.assert_integrity()

    def alias(self, branch: V1RelaxedKKTBranch) -> V1SampledV7BranchActionAlias:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch alias lookup requires one exact Q branch")
        return self.branch_aliases[_BRANCH_AXIS.index(branch)]

    def packed_action(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> V1PackedRemovalAction | None:
        alias = self.alias(branch)
        if alias.physical_action_ordinal is None:
            return None
        return self.physical_actions[alias.physical_action_ordinal - 1].packed_action

    def __reduce__(self) -> None:
        raise TypeError("V7 branch action pairs are nonserializable")

    def __reduce_ex__(self, protocol: int) -> None:
        del protocol
        raise TypeError("V7 branch action pairs are nonserializable")


def _resolve_common_coordinate(
    source_context: object,
) -> tuple[V1SignedWorkPreCLiveCommonCoordinateContext, str | None]:
    if type(source_context) is V1SignedWorkPreCLiveCommonCoordinateContext:
        return source_context, None
    assert_live = getattr(source_context, "assert_live", None)
    if not callable(assert_live):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            "branch-action issuance accepts one exact common-coordinate parent "
            "or a live wrapper exposing assert_live() and that exact parent",
        )
    try:
        live_hash = assert_live()
        common = getattr(source_context, "common_coordinate", None)
    except Exception as error:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
            f"live source wrapper is stale: {type(error).__name__}: {error}",
        )
    if type(live_hash) is not str or not live_hash:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
            "live source wrapper returned no stable nonempty identity",
        )
    if type(common) is not V1SignedWorkPreCLiveCommonCoordinateContext:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.INVALID_TYPE,
            "live source wrapper lost its exact library-owned common-coordinate parent",
        )
    return common, live_hash


def _validate_common_coordinate_source(
    common: V1SignedWorkPreCLiveCommonCoordinateContext,
) -> None:
    expected = _BRANCH_AXIS
    if (
        tuple(value.branch for value in common.branch_executions) != expected
        or tuple(value.branch_identity.branch for value in common.branch_shells)
        != expected
        or tuple(value.receipt.branch for value in common.branch_hard_fq) != expected
        or tuple(value.q_branch for value in common.candidate_crosswalks) != expected
    ):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.BRANCH_ORDER_MISMATCH,
            "live source is not the canonical spectral then diagonal_l1 branch axis",
        )
    try:
        common.__post_init__()
        source_hash = common.source.assert_integrity()
        problem_hash = common.problem.assert_integrity()
    except V1SampledV7BranchActionPairRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
            f"common-coordinate source is stale: {type(error).__name__}: {error}",
        )
    geometry = common.source.response_geometry
    if (
        source_hash != common.source.bundle_hash
        or problem_hash != common.problem.problem_hash
        or common.problem.source_problem is not geometry
        or common.problem.signed_work_operator is not common.source.signed_work_operator
        or common.problem.source_registration_hash
        != common.source.source_registration_hash
        or common.problem.registered_d_coordinate_hash
        != common.source.registered_d_coordinate_hash
        or common.endpoint_plan.source_registration_hash
        != common.source.source_registration_hash
        or common.endpoint_plan.registered_d_coordinate_hash
        != common.source.registered_d_coordinate_hash
        or common.endpoint_plan.route_axis_hash != geometry.route_axis_hash
        or common.endpoint_plan.route_count != geometry.route_count
        or common.parent_mask.route_axis_hash != geometry.route_axis_hash
        or common.parent_mask.route_count != geometry.route_count
        or common.problem.resource_floor != common.coordinate.total_resource_floor
    ):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "common-coordinate action sources do not share one exact chart/parent",
        )


def _branch_endpoint(
    common: V1SignedWorkPreCLiveCommonCoordinateContext,
    branch_index: int,
    crosswalk: V1SignedWorkCoordinateCandidateCrosswalk,
) -> V1SignedWorkHardFQEndpointReceipt:
    hard_bundle = common.branch_hard_fq[branch_index]
    matches = tuple(
        value
        for value in hard_bundle.receipt.endpoints
        if value.endpoint_label == crosswalk.source_endpoint_label
    )
    if len(matches) != 1:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "candidate crosswalk does not name exactly one hard-F/Q endpoint",
        )
    endpoint = matches[0]
    decision = crosswalk.endpoint_decision
    if (
        decision.hard_replay_endpoint_receipt_hash != endpoint.receipt_hash
        or decision.canonical_status is not endpoint.status
        or decision.origin_branch_status != endpoint.origin_branch_status.value
        or decision.cumulative_removed_count != endpoint.cumulative_removed_count
        or decision.branch is not endpoint.branch
    ):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
            "crosswalk Q status does not replay its exact hard-F/Q endpoint",
        )
    safe = endpoint.status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
    if safe is not (
        type(crosswalk.packed_action) is V1PackedRemovalAction
        and crosswalk.materialization_receipt is not None
        and crosswalk.source_packed_action is not None
    ):
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
            "crosswalk action availability does not replay its Q status",
        )
    try:
        crosswalk.__post_init__()
        endpoint.__post_init__()
    except Exception as error:
        _refuse(
            V1SampledV7BranchActionPairRefusalCode.STALE_SOURCE,
            "candidate crosswalk or hard-F/Q endpoint is stale: "
            f"{type(error).__name__}: {error}",
        )
    return endpoint


def issue_v1_sampled_v7_branch_action_pair(
    *,
    source_context: object,
) -> V1SampledV7BranchActionPair:
    """Issue the canonical pair solely from exact live source custody.

    ``source_context`` must be either the exact V7 live-D context or its exact
    ``V1SignedWorkPreCLiveCommonCoordinateContext`` parent.  There is
    deliberately no action, branch, endpoint, or ranking choice parameter.
    """

    common, live_hash = _resolve_common_coordinate(source_context)
    _validate_common_coordinate_source(common)
    crosswalks = tuple(common.candidate_crosswalks)
    endpoints = tuple(
        _branch_endpoint(common, index, crosswalk)
        for index, crosswalk in enumerate(crosswalks)
    )

    physical_order: list[V1SignedWorkPackedActionDedupIdentity] = []
    physical_actions: dict[
        V1SignedWorkPackedActionDedupIdentity,
        V1PackedRemovalAction,
    ] = {}
    physical_branches: dict[
        V1SignedWorkPackedActionDedupIdentity,
        list[V1RelaxedKKTBranch],
    ] = {}
    physical_alias_hashes: dict[
        V1SignedWorkPackedActionDedupIdentity,
        list[str],
    ] = {}
    ordinal_by_identity: dict[V1SignedWorkPackedActionDedupIdentity, int] = {}

    # The alias hash includes the physical ordinal, so first establish the
    # unique-payload order from the source-owned canonical branch axis.
    for crosswalk in crosswalks:
        if crosswalk.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE:
            continue
        action = crosswalk.packed_action
        if type(action) is not V1PackedRemovalAction:
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.Q_STATUS_MISMATCH,
                "Q-safe source crosswalk lacks its exact packed action",
            )
        identity = v1_signed_work_packed_action_dedup_identity(action)
        if identity not in physical_actions:
            physical_order.append(identity)
            physical_actions[identity] = action
            physical_branches[identity] = []
            physical_alias_hashes[identity] = []
            ordinal_by_identity[identity] = len(physical_order)
        elif (
            physical_actions[identity].resulting_mask.packed_bits
            != action.resulting_mask.packed_bits
        ):
            _refuse(
                V1SampledV7BranchActionPairRefusalCode.ACTION_IDENTITY_MISMATCH,
                "matching physical identities carried unequal packed child bytes",
            )

    aliases: list[V1SampledV7BranchActionAlias] = []
    for index, (crosswalk, endpoint) in enumerate(
        zip(crosswalks, endpoints, strict=True)
    ):
        branch = _BRANCH_AXIS[index]
        status = crosswalk.q_status
        safe = status is V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE
        action = crosswalk.packed_action if safe else None
        identity = (
            None
            if action is None
            else v1_signed_work_packed_action_dedup_identity(action)
        )
        physical_ordinal = None if identity is None else ordinal_by_identity[identity]
        no_action = (
            None
            if safe
            else V1SampledV7BranchNoActionReceipt(
                branch=branch,
                q_status=status,
                code=_no_action_code(status),
                candidate_crosswalk_hash=crosswalk.crosswalk_hash,
                endpoint_decision_receipt_hash=(
                    crosswalk.endpoint_decision.receipt_hash
                ),
                hard_fq_endpoint_receipt_hash=endpoint.receipt_hash,
                _issuance_token=_ISSUER,
            )
        )
        materialization = crosswalk.materialization_receipt
        source_action = crosswalk.source_packed_action
        alias = V1SampledV7BranchActionAlias(
            branch=branch,
            q_status=status,
            coordinate_candidate_key_hash=(
                crosswalk.coordinate_candidate_key_hash
            ),
            resource_coordinate_hash=crosswalk.resource_coordinate_hash,
            coordinate_ordinal=crosswalk.coordinate_ordinal,
            source_endpoint_label=crosswalk.source_endpoint_label,
            candidate_crosswalk_hash=crosswalk.crosswalk_hash,
            source_materialization_bundle_hash=(
                crosswalk.source_materialization_bundle_hash
            ),
            endpoint_decision_receipt_hash=(
                crosswalk.endpoint_decision.receipt_hash
            ),
            hard_fq_bundle_hash=common.branch_hard_fq[index].bundle_hash,
            hard_fq_branch_receipt_hash=(
                common.branch_hard_fq[index].receipt.receipt_hash
            ),
            hard_fq_endpoint_receipt_hash=endpoint.receipt_hash,
            shell_execution_hash=common.branch_shells[index].execution_hash,
            parent_mask_hash=crosswalk.parent_mask_hash,
            route_axis_hash=crosswalk.route_axis_hash,
            route_count=crosswalk.route_count,
            physical_action_ordinal=physical_ordinal,
            physical_action_identity_hash=(
                None
                if identity is None
                else v1_sampled_v7_physical_action_identity_hash(identity)
            ),
            packed_action_hash=None if action is None else action.action_hash,
            resulting_mask_hash=(
                None if action is None else action.resulting_mask.mask_hash
            ),
            resulting_mask_packed_sha256=(
                None
                if action is None
                else hashlib.sha256(action.resulting_mask.packed_bits).hexdigest()
            ),
            removed_route_index_hash=(
                None if action is None else action.removed_route_index_hash
            ),
            removed_count=None if action is None else action.removed_count,
            materialization_receipt_hash=(
                None if materialization is None else materialization.receipt_hash
            ),
            source_packed_action_hash=(
                None if source_action is None else source_action.action_hash
            ),
            no_action=no_action,
            _issuance_token=_ISSUER,
        )
        aliases.append(alias)
        if identity is not None:
            physical_branches[identity].append(branch)
            physical_alias_hashes[identity].append(alias.alias_hash)

    physical_custody = tuple(
        V1SampledV7PhysicalActionCustody(
            physical_action_ordinal=ordinal_by_identity[identity],
            dedup_identity=identity,
            physical_action_identity_hash=(
                v1_sampled_v7_physical_action_identity_hash(identity)
            ),
            branch_alias_axis=tuple(physical_branches[identity]),
            branch_alias_hashes=tuple(physical_alias_hashes[identity]),
            packed_action_hash=physical_actions[identity].action_hash,
            resulting_mask_hash=physical_actions[identity].resulting_mask.mask_hash,
            resulting_mask_packed_sha256=hashlib.sha256(
                physical_actions[identity].resulting_mask.packed_bits
            ).hexdigest(),
            removed_route_index_hash=(
                physical_actions[identity].removed_route_index_hash
            ),
            removed_count=physical_actions[identity].removed_count,
            packed_payload_bytes=len(
                physical_actions[identity].resulting_mask.packed_bits
            ),
            _packed_action=physical_actions[identity],
            _issuance_token=_ISSUER,
        )
        for identity in physical_order
    )
    aliases_tuple = tuple(aliases)
    return V1SampledV7BranchActionPair(
        source_bundle_hash=common.source.bundle_hash,
        source_registration_hash=common.source.source_registration_hash,
        registered_d_coordinate_hash=common.source.registered_d_coordinate_hash,
        response_geometry_problem_hash=(
            common.source.response_geometry.problem_hash
        ),
        response_geometry_source_hash=(
            common.source.response_geometry.source_geometry_hash
        ),
        least_work_problem_hash=common.problem.problem_hash,
        policy_bundle_hash=common.policies.policy_hash,
        resource_coordinate_hash=common.coordinate.coordinate_hash,
        coordinate_ordinal=common.coordinate.ordinal,
        endpoint_plan_hash=common.endpoint_plan.receipt_hash,
        parent_mask_hash=common.parent_mask.mask_hash,
        parent_retained_count=common.parent_mask.retained_count,
        route_axis_hash=common.parent_mask.route_axis_hash,
        route_count=common.parent_mask.route_count,
        live_d_source_identity_hash=live_hash,
        branch_axis=_BRANCH_AXIS,
        branch_aliases=aliases_tuple,  # type: ignore[arg-type]
        physical_actions=physical_custody,
        branch_physical_action_ordinals=tuple(  # type: ignore[arg-type]
            value.physical_action_ordinal for value in aliases_tuple
        ),
        physical_action_count=len(physical_custody),
        distinct_branch_alias_count=2,
        physical_payloads_deduplicated=(
            sum(value.has_action for value in aliases_tuple) > len(physical_custody)
        ),
        _issuance_token=_ISSUER,
    )


__all__ = [
    "SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA",
    "SAMPLED_V7_BRANCH_ACTION_PAIR_AUTHORITY",
    "SAMPLED_V7_BRANCH_ACTION_PAIR_SCHEMA",
    "SAMPLED_V7_BRANCH_NO_ACTION_SCHEMA",
    "SAMPLED_V7_PHYSICAL_ACTION_CUSTODY_SCHEMA",
    "SAMPLED_V7_SHELL_UNION_LIMITATION",
    "V1SampledV7BranchActionAlias",
    "V1SampledV7BranchActionPair",
    "V1SampledV7BranchActionPairRefusal",
    "V1SampledV7BranchActionPairRefusalCode",
    "V1SampledV7BranchNoActionCode",
    "V1SampledV7BranchNoActionReceipt",
    "V1SampledV7PhysicalActionCustody",
    "issue_v1_sampled_v7_branch_action_pair",
    "v1_sampled_v7_physical_action_identity_hash",
]
