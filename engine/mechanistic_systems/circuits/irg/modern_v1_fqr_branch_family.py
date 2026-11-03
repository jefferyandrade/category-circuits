"""Pre-C two-branch hard ladders and post-C F/Q/R union selection.

The two convex KKT branches are a representation of the relaxed residual
envelope.  Their relaxed winner is not, in general, the winner after hard
rounding and finite-direction certification.  This module therefore freezes
both complete rounded ladders before any C-stream outcome, binds one complete
F/Q/R evidence graph to each retained branch, and selects only from the union
of independently certified, pre-frozen hard endpoints.

No object in this module grants calibration, BCE-oracle, mask-commit, terminal,
or hard global-optimality authority.  In particular, the family selector does
not synthesize an action across branches and does not fill a missing endpoint
from prefix monotonicity.  A branch may be omitted only when a typed pre-C
certificate proves that every one of its packed physical actions is already
present, at no smaller registered resource, in the retained branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from typing import Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_fqr_candidate_bridge import (
    V1FQRCandidateQueueAuthorization,
    V1FQRCandidateQueueRefusal,
    V1FQRCandidateQueueRefusalCode,
    authorize_v1_fqr_candidate_queue,
)
from .modern_v1_fqr_certification import (
    V1DecisionStabilityCertificate,
    V1FQRCertificationDecision,
    V1FQRCertificationDesign,
    V1FQRCertificationReceipt,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
    V1FrozenLadderCertificationReceipt,
    V1FrozenLadderDecision,
)
from .modern_v1_literal_j_operator_census import (
    V1LiteralJOperatorCensusReceipt,
)
from .modern_v1_literal_j_projection_inference import (
    V1LiteralJFixedDirectionProjectionReceipt,
)
from .modern_v1_device_shell_ladder import (
    V1DeviceShellFrozenActionReceipt,
    V1DeviceShellFrozenLadderReceipt,
)
from .modern_v1_device_shell_fq_replay_cuda import (
    V1DeviceShellHardFQCUDAReplayReceipt,
)
from .modern_v1_device_shell_fq_replay import V1DeviceShellHardFQStatus
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTBranch,
    V1RelaxedKKTClosureMode,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTDeviceShellExecutionReceipt,
    V1RelaxedKKTPolicy,
    V1RelaxedKKTProblem,
    V1RelaxedKKTResult,
    V1RelaxedKKTRoundedLadder,
    V1RelaxedKKTGeometryUncertaintySet,
)


FQR_KKT_BRANCH_BINDING_SCHEMA: Final = (
    "partition_free_v1_fqr_kkt_branch_frozen_binding_v1"
)
FQR_KKT_BRANCH_FAMILY_SCHEMA: Final = (
    "partition_free_v1_fqr_kkt_prefrozen_branch_family_v1"
)
FQR_KKT_PRE_C_DOMINANCE_SCHEMA: Final = (
    "partition_free_v1_fqr_kkt_pre_c_exact_action_dominance_v1"
)
FQR_BRANCH_EVIDENCE_BINDING_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_evidence_binding_v1"
)
FQR_BRANCH_EVIDENCE_GRAPH_KIND: Final = (
    "typed_joint_fqr_graph_transitively_retaining_full_secant_records_v1"
)
FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_family_authorization_v1"
)
FQR_KKT_BRANCH_FAMILY_SCOPE: Final = (
    "prefrozen_two_branch_hard_union_without_cross_branch_synthesis_v1"
)
FQR_PRE_C_DOMINANCE_METHOD: Final = (
    "omitted_packed_action_identity_covered_at_no_smaller_registered_resource_v1"
)
FQR_BRANCH_UNION_SELECTION_POLICY: Final = (
    "greatest_registered_resource_then_branch_enum_value_then_"
    "packed_physical_action_hash_then_endpoint_index_v1"
)
FQR_BRANCH_FAMILY_AUTHORITY_SCOPE: Final = (
    "stable_joint_fqr_hard_union_proposal_selection_only_"
    "no_calibration_oracle_commit_terminal_or_global_optimality_v1"
)
FQR_DEVICE_SHELL_ENDPOINT_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_endpoint_identity_v1"
)
FQR_DEVICE_SHELL_BRANCH_BINDING_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_branch_binding_v1"
)
FQR_DEVICE_SHELL_CROSS_BRANCH_ACTION_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_cross_branch_action_identity_v1"
)
FQR_DEVICE_SHELL_BRANCH_FAMILY_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_prefrozen_branch_family_v1"
)
FQR_DEVICE_SHELL_SOURCE_STABILITY_BINDING_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_source_stability_binding_v1"
)
FQR_DEVICE_SHELL_IDENTITY_SCOPE: Final = (
    "array_free_two_branch_endpoint_identity_only_no_fq_c_or_commit_authority_v1"
)

_BINDING_PREFIX: Final = "partition-free-v1-fqr-kkt-branch-binding-"
_FAMILY_PREFIX: Final = "partition-free-v1-fqr-kkt-branch-family-"
_DOMINANCE_PREFIX: Final = "partition-free-v1-fqr-pre-c-dominance-"
_EVIDENCE_PREFIX: Final = "partition-free-v1-fqr-branch-evidence-"
_AUTHORIZATION_PREFIX: Final = "partition-free-v1-fqr-branch-family-authorization-"
_DEVICE_SHELL_ENDPOINT_PREFIX: Final = (
    "partition-free-v1-fqr-device-shell-endpoint-"
)
_DEVICE_SHELL_BRANCH_PREFIX: Final = (
    "partition-free-v1-fqr-device-shell-branch-binding-"
)
_DEVICE_SHELL_CROSS_BRANCH_PREFIX: Final = (
    "partition-free-v1-fqr-device-shell-cross-branch-action-"
)
_DEVICE_SHELL_FAMILY_PREFIX: Final = (
    "partition-free-v1-fqr-device-shell-branch-family-"
)
_DEVICE_SHELL_SOURCE_STABILITY_PREFIX: Final = (
    "partition-free-v1-fqr-device-shell-source-stability-"
)
LiteralJFixedPrimalSourceReceipt = (
    V1LiteralJOperatorCensusReceipt | V1LiteralJFixedDirectionProjectionReceipt
)


class V1FQRBranchFamilyRefusalCode(str, Enum):
    """Stable fail-closed reasons for the hard two-branch seam."""

    INVALID_TYPE = "invalid_type"
    FQ_ONLY_EVIDENCE = "fq_only_evidence"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    STALE_CHART = "stale_chart"
    POST_SELECTION = "post_selection"
    UNSTABLE_DECISION = "unstable_decision"
    MISSING_BRANCH = "missing_branch"
    INVALID_DOMINANCE = "invalid_dominance"
    NO_CERTIFIED_ENDPOINT = "no_certified_endpoint"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FQRBranchFamilyRefusal(ValueError):
    """Typed refusal before cross-branch proposal selection."""

    def __init__(self, code: V1FQRBranchFamilyRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQRBranchFamilyRefusalCode, message: str) -> NoReturn:
    raise V1FQRBranchFamilyRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _array_digest(value: np.ndarray) -> str:
    """Reproduce the KKT module's registered physical-prefix digest."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _direct_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be an exact HardMask",
        )
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


def _translate_bridge_refusal(error: V1FQRCandidateQueueRefusal) -> NoReturn:
    mapping = {
        V1FQRCandidateQueueRefusalCode.INVALID_TYPE: (
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE
        ),
        V1FQRCandidateQueueRefusalCode.FQ_ONLY_EVIDENCE: (
            V1FQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE
        ),
        V1FQRCandidateQueueRefusalCode.COORDINATE_MISMATCH: (
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH
        ),
        V1FQRCandidateQueueRefusalCode.STALE_CHART: (
            V1FQRBranchFamilyRefusalCode.STALE_CHART
        ),
        V1FQRCandidateQueueRefusalCode.POST_SELECTION: (
            V1FQRBranchFamilyRefusalCode.POST_SELECTION
        ),
        V1FQRCandidateQueueRefusalCode.UNSTABLE_DECISION: (
            V1FQRBranchFamilyRefusalCode.UNSTABLE_DECISION
        ),
        V1FQRCandidateQueueRefusalCode.NO_CERTIFIED_ENDPOINT: (
            V1FQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT
        ),
        V1FQRCandidateQueueRefusalCode.AUTHORITY_VIOLATION: (
            V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION
        ),
    }
    _refuse(mapping[error.code], f"invalid single-branch F/Q/R graph: {error}")


def _branch_result(result: V1RelaxedKKTResult, branch: V1RelaxedKKTBranch):
    return (
        result.spectral if branch is V1RelaxedKKTBranch.SPECTRAL else result.diagonal_l1
    )


def _validated_device_shell_execution_hash(
    receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
) -> str:
    """Replay the complete durable device-shell receipt before adapting it."""

    if type(receipt) is not V1RelaxedKKTDeviceShellExecutionReceipt:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "device-shell adaptation requires the exact execution receipt",
        )
    try:
        for branch_receipt in (receipt.spectral, receipt.diagonal_l1):
            if type(branch_receipt) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
                raise TypeError("device-shell branch receipt has the wrong type")
            ladder = branch_receipt.frozen_ladder
            if type(ladder) is not V1DeviceShellFrozenLadderReceipt:
                raise TypeError("device-shell ladder receipt has the wrong type")
            for action in ladder.actions:
                if type(action) is not V1DeviceShellFrozenActionReceipt:
                    raise TypeError("device-shell action receipt has the wrong type")
                action.__post_init__()
            ladder.__post_init__()
            branch_receipt.__post_init__()
        receipt.__post_init__()
        return receipt.receipt_hash
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
            f"device-shell execution receipt does not replay: {error}",
        )


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellEndpointIdentity:
    """One compact, durable hard endpoint without a mask, order, or handle."""

    branch: V1RelaxedKKTBranch
    endpoint_index: int
    problem_hash: str
    policy_hash: str
    base_result_hash: str
    execution_receipt_hash: str
    branch_identity_receipt_hash: str
    source_rounding_hash: str
    source_ladder_receipt_hash: str
    source_action_receipt_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    shell_count: int
    cumulative_removed_count: int
    registered_resource: float
    removed_route_index_hash: str
    resulting_mask_packed_sha256: str
    resulting_mask_hash: str
    packed_action_hash: str
    schema: str = FQR_DEVICE_SHELL_ENDPOINT_IDENTITY_SCHEMA
    _identity_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell endpoint requires an exact KKT branch",
            )
        for name in (
            "problem_hash",
            "policy_hash",
            "base_result_hash",
            "execution_receipt_hash",
            "branch_identity_receipt_hash",
            "source_rounding_hash",
            "source_ladder_receipt_hash",
            "source_action_receipt_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "removed_route_index_hash",
            "resulting_mask_packed_sha256",
            "resulting_mask_hash",
            "packed_action_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "endpoint_index",
            _positive_int(self.endpoint_index, name="endpoint index"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "shell_count",
            _positive_int(self.shell_count, name="shell count"),
        )
        object.__setattr__(
            self,
            "cumulative_removed_count",
            _positive_int(
                self.cumulative_removed_count,
                name="cumulative removed count",
            ),
        )
        object.__setattr__(
            self,
            "registered_resource",
            _finite_nonnegative(
                self.registered_resource,
                name="registered endpoint resource",
            ),
        )
        if self.shell_count > self.cumulative_removed_count:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "endpoint shell count exceeds its cumulative removal count",
            )
        if self.schema != FQR_DEVICE_SHELL_ENDPOINT_IDENTITY_SCHEMA:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell endpoint schema is invalid",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "endpoint_index": self.endpoint_index,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "base_result_hash": self.base_result_hash,
            "execution_receipt_hash": self.execution_receipt_hash,
            "branch_identity_receipt_hash": self.branch_identity_receipt_hash,
            "source_rounding_hash": self.source_rounding_hash,
            "source_ladder_receipt_hash": self.source_ladder_receipt_hash,
            "source_action_receipt_hash": self.source_action_receipt_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "shell_count": self.shell_count,
            "cumulative_removed_count": self.cumulative_removed_count,
            "registered_resource": self.registered_resource,
            "removed_route_index_hash": self.removed_route_index_hash,
            "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
            "resulting_mask_hash": self.resulting_mask_hash,
            "packed_action_hash": self.packed_action_hash,
        }

    @property
    def identity_hash(self) -> str:
        cached = self._identity_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            replayed_action = V1DeviceShellFrozenActionReceipt(
                endpoint_label=self.endpoint_index,
                shell_count=self.shell_count,
                cumulative_removed_count=self.cumulative_removed_count,
                removed_route_index_hash=self.removed_route_index_hash,
                resulting_mask_packed_sha256=self.resulting_mask_packed_sha256,
                resulting_mask_hash=self.resulting_mask_hash,
                packed_action_hash=self.packed_action_hash,
            )
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                f"device-shell endpoint action does not replay: {error}",
            )
        if replayed_action.receipt_hash != self.source_action_receipt_hash:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell endpoint differs from its source action receipt",
            )
        observed = content_hash(self._payload(), prefix=_DEVICE_SHELL_ENDPOINT_PREFIX)
        cached = self._identity_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell endpoint identity was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell endpoint identity hash does not match",
            )
        object.__setattr__(self, "_identity_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellBranchBinding:
    """Array-free endpoint axis bound to one exact device-shell branch receipt."""

    branch: V1RelaxedKKTBranch
    problem_hash: str
    policy_hash: str
    base_result_hash: str
    device_shell_execution_design_hash: str
    cuda_execution_design_hash: str
    hbm_forecast_hash: str
    cuda_telemetry_hash: str
    execution_receipt_hash: str
    branch_identity_receipt_hash: str
    source_rounding_hash: str
    source_ladder_receipt_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    label_commitment: str
    eligibility_commitment: str
    block_capacity_hash: str
    block_legality_scope: str
    endpoint_axis_hash: str
    endpoint_resource_axis_hash: str
    endpoint_physical_action_axis_hash: str
    endpoints: tuple[V1FQRDeviceShellEndpointIdentity, ...]
    c_stream_outcomes_seen: bool = False
    ordinary_hard_action_materialized: bool = False
    workflow_authority: bool = False
    identity_scope: str = FQR_DEVICE_SHELL_IDENTITY_SCOPE
    schema: str = FQR_DEVICE_SHELL_BRANCH_BINDING_SCHEMA
    _binding_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell branch binding requires an exact KKT branch",
            )
        for name in (
            "problem_hash",
            "policy_hash",
            "base_result_hash",
            "device_shell_execution_design_hash",
            "cuda_execution_design_hash",
            "hbm_forecast_hash",
            "cuda_telemetry_hash",
            "execution_receipt_hash",
            "branch_identity_receipt_hash",
            "source_rounding_hash",
            "source_ladder_receipt_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "label_commitment",
            "eligibility_commitment",
            "block_capacity_hash",
            "block_legality_scope",
            "endpoint_axis_hash",
            "endpoint_resource_axis_hash",
            "endpoint_physical_action_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        endpoints = tuple(self.endpoints)
        if not endpoints or any(
            type(endpoint) is not V1FQRDeviceShellEndpointIdentity
            for endpoint in endpoints
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell branch requires a nonempty exact endpoint axis",
            )
        if tuple(endpoint.endpoint_index for endpoint in endpoints) != tuple(
            range(1, len(endpoints) + 1)
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell endpoints must form the complete one-based axis",
            )
        common = (
            self.branch,
            self.problem_hash,
            self.policy_hash,
            self.base_result_hash,
            self.execution_receipt_hash,
            self.branch_identity_receipt_hash,
            self.source_rounding_hash,
            self.source_ladder_receipt_hash,
            self.route_axis_hash,
            self.route_count,
            self.current_mask_hash,
            self.chart_identity_hash,
            self.proposal_generation,
            self.discovery_prefix_hash,
            self.discovery_evidence_hash,
        )
        if any(
            (
                endpoint.branch,
                endpoint.problem_hash,
                endpoint.policy_hash,
                endpoint.base_result_hash,
                endpoint.execution_receipt_hash,
                endpoint.branch_identity_receipt_hash,
                endpoint.source_rounding_hash,
                endpoint.source_ladder_receipt_hash,
                endpoint.route_axis_hash,
                endpoint.route_count,
                endpoint.current_mask_hash,
                endpoint.chart_identity_hash,
                endpoint.proposal_generation,
                endpoint.discovery_prefix_hash,
                endpoint.discovery_evidence_hash,
            )
            != common
            for endpoint in endpoints
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "an endpoint left its device-shell branch coordinates",
            )
        cumulative = tuple(
            endpoint.cumulative_removed_count for endpoint in endpoints
        )
        if tuple(sorted(set(cumulative))) != cumulative:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell cumulative removals must be strictly increasing",
            )
        resources = tuple(endpoint.registered_resource for endpoint in endpoints)
        if tuple(sorted(resources)) != resources:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "registered endpoint resources must be nondecreasing",
            )
        for attribute in (
            "removed_route_index_hash",
            "resulting_mask_hash",
            "packed_action_hash",
        ):
            values = tuple(getattr(endpoint, attribute) for endpoint in endpoints)
            if len(set(values)) != len(values):
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    f"one branch contains duplicate {attribute} identities",
                )
        if any(
            type(value) is not bool
            for value in (
                self.c_stream_outcomes_seen,
                self.ordinary_hard_action_materialized,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell branch authority fields must be exact booleans",
            )
        if any(
            (
                self.c_stream_outcomes_seen,
                self.ordinary_hard_action_materialized,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "device-shell branch identity crossed its pre-C authority boundary",
            )
        if (
            self.identity_scope != FQR_DEVICE_SHELL_IDENTITY_SCOPE
            or self.schema != FQR_DEVICE_SHELL_BRANCH_BINDING_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell branch identity scope or schema is invalid",
            )
        object.__setattr__(self, "endpoints", endpoints)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "base_result_hash": self.base_result_hash,
            "device_shell_execution_design_hash": (
                self.device_shell_execution_design_hash
            ),
            "cuda_execution_design_hash": self.cuda_execution_design_hash,
            "hbm_forecast_hash": self.hbm_forecast_hash,
            "cuda_telemetry_hash": self.cuda_telemetry_hash,
            "execution_receipt_hash": self.execution_receipt_hash,
            "branch_identity_receipt_hash": self.branch_identity_receipt_hash,
            "source_rounding_hash": self.source_rounding_hash,
            "source_ladder_receipt_hash": self.source_ladder_receipt_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "label_commitment": self.label_commitment,
            "eligibility_commitment": self.eligibility_commitment,
            "block_capacity_hash": self.block_capacity_hash,
            "block_legality_scope": self.block_legality_scope,
            "endpoint_axis_hash": self.endpoint_axis_hash,
            "endpoint_resource_axis_hash": self.endpoint_resource_axis_hash,
            "endpoint_physical_action_axis_hash": (
                self.endpoint_physical_action_axis_hash
            ),
            "endpoint_identity_hashes": tuple(
                endpoint.identity_hash for endpoint in self.endpoints
            ),
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "ordinary_hard_action_materialized": (
                self.ordinary_hard_action_materialized
            ),
            "workflow_authority": self.workflow_authority,
            "identity_scope": self.identity_scope,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for endpoint in self.endpoints:
            endpoint.assert_integrity()
        endpoint_axis_hash = content_hash(
            {
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "current_mask_hash": self.current_mask_hash,
                "axis": tuple(
                    (
                        endpoint.endpoint_index,
                        endpoint.shell_count,
                        endpoint.cumulative_removed_count,
                    )
                    for endpoint in self.endpoints
                ),
            },
            prefix="partition-free-v1-fqr-device-shell-endpoint-axis-",
        )
        resource_axis_hash = content_hash(
            {
                "branch": self.branch,
                "source_rounding_hash": self.source_rounding_hash,
                "label_commitment": self.label_commitment,
                "resources": tuple(
                    endpoint.registered_resource for endpoint in self.endpoints
                ),
            },
            prefix="partition-free-v1-fqr-device-shell-resource-axis-",
        )
        physical_axis_hash = content_hash(
            tuple(endpoint.identity_hash for endpoint in self.endpoints),
            prefix="partition-free-v1-fqr-device-shell-physical-action-axis-",
        )
        if (
            self.endpoint_axis_hash != endpoint_axis_hash
            or self.endpoint_resource_axis_hash != resource_axis_hash
            or self.endpoint_physical_action_axis_hash != physical_axis_hash
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell branch endpoint axes do not replay",
            )
        if any(
            (
                self.c_stream_outcomes_seen,
                self.ordinary_hard_action_materialized,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "device-shell branch identity no longer has pre-C authority",
            )
        observed = content_hash(self._payload(), prefix=_DEVICE_SHELL_BRANCH_PREFIX)
        cached = self._binding_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell branch binding was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell branch binding hash does not match",
            )
        object.__setattr__(self, "_binding_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellCrossBranchActionIdentity:
    """Explicit representation of one physical action shared by both branches."""

    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    cumulative_removed_count: int
    spectral_endpoint_index: int
    diagonal_l1_endpoint_index: int
    spectral_endpoint_identity_hash: str
    diagonal_l1_endpoint_identity_hash: str
    registered_resource: float
    schema: str = FQR_DEVICE_SHELL_CROSS_BRANCH_ACTION_SCHEMA
    _identity_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "packed_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "spectral_endpoint_identity_hash",
            "diagonal_l1_endpoint_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "cumulative_removed_count",
            "spectral_endpoint_index",
            "diagonal_l1_endpoint_index",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "registered_resource",
            _finite_nonnegative(
                self.registered_resource,
                name="cross-branch registered resource",
            ),
        )
        if self.schema != FQR_DEVICE_SHELL_CROSS_BRANCH_ACTION_SCHEMA:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "cross-branch action identity schema is invalid",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "packed_action_hash": self.packed_action_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "removed_route_index_hash": self.removed_route_index_hash,
            "cumulative_removed_count": self.cumulative_removed_count,
            "spectral_endpoint_index": self.spectral_endpoint_index,
            "diagonal_l1_endpoint_index": self.diagonal_l1_endpoint_index,
            "spectral_endpoint_identity_hash": (
                self.spectral_endpoint_identity_hash
            ),
            "diagonal_l1_endpoint_identity_hash": (
                self.diagonal_l1_endpoint_identity_hash
            ),
            "registered_resource": self.registered_resource,
        }

    @property
    def identity_hash(self) -> str:
        cached = self._identity_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_DEVICE_SHELL_CROSS_BRANCH_PREFIX
        )
        cached = self._identity_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "cross-branch action identity was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "cross-branch action identity hash does not match",
            )
        object.__setattr__(self, "_identity_hash", observed)
        return observed


def _device_shell_cross_branch_identities(
    spectral_binding: V1FQRDeviceShellBranchBinding,
    diagonal_binding: V1FQRDeviceShellBranchBinding,
) -> tuple[V1FQRDeviceShellCrossBranchActionIdentity, ...]:
    spectral_by_action = {
        endpoint.packed_action_hash: endpoint
        for endpoint in spectral_binding.endpoints
    }
    diagonal_by_action = {
        endpoint.packed_action_hash: endpoint
        for endpoint in diagonal_binding.endpoints
    }
    spectral_by_mask = {
        endpoint.resulting_mask_hash: endpoint
        for endpoint in spectral_binding.endpoints
    }
    diagonal_by_mask = {
        endpoint.resulting_mask_hash: endpoint
        for endpoint in diagonal_binding.endpoints
    }
    for mask_hash in set(spectral_by_mask).intersection(diagonal_by_mask):
        if (
            spectral_by_mask[mask_hash].packed_action_hash
            != diagonal_by_mask[mask_hash].packed_action_hash
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "the same resulting mask has conflicting physical action hashes",
            )
    rows: list[V1FQRDeviceShellCrossBranchActionIdentity] = []
    for action_hash in sorted(set(spectral_by_action).intersection(diagonal_by_action)):
        spectral = spectral_by_action[action_hash]
        diagonal = diagonal_by_action[action_hash]
        if (
            spectral.resulting_mask_hash != diagonal.resulting_mask_hash
            or spectral.removed_route_index_hash != diagonal.removed_route_index_hash
            or spectral.cumulative_removed_count != diagonal.cumulative_removed_count
            or spectral.registered_resource != diagonal.registered_resource
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "a shared packed action has conflicting cross-branch identity",
            )
        rows.append(
            V1FQRDeviceShellCrossBranchActionIdentity(
                packed_action_hash=action_hash,
                resulting_mask_hash=spectral.resulting_mask_hash,
                removed_route_index_hash=spectral.removed_route_index_hash,
                cumulative_removed_count=spectral.cumulative_removed_count,
                spectral_endpoint_index=spectral.endpoint_index,
                diagonal_l1_endpoint_index=diagonal.endpoint_index,
                spectral_endpoint_identity_hash=spectral.identity_hash,
                diagonal_l1_endpoint_identity_hash=diagonal.identity_hash,
                registered_resource=spectral.registered_resource,
            )
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellPreFrozenBranchFamily:
    """Compact two-branch device-shell identity frozen before C outcomes."""

    execution_receipt_hash: str
    problem_hash: str
    policy_hash: str
    base_result_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    endpoint_axis_hash: str
    spectral: V1FQRDeviceShellBranchBinding
    diagonal_l1: V1FQRDeviceShellBranchBinding
    cross_branch_action_identities: tuple[
        V1FQRDeviceShellCrossBranchActionIdentity, ...
    ] = ()
    c_stream_outcomes_seen: bool = False
    cross_branch_action_synthesis_authority: bool = False
    workflow_authority: bool = False
    identity_scope: str = FQR_DEVICE_SHELL_IDENTITY_SCOPE
    schema: str = FQR_DEVICE_SHELL_BRANCH_FAMILY_SCHEMA
    _family_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "execution_receipt_hash",
            "problem_hash",
            "policy_hash",
            "base_result_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "endpoint_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if (
            type(self.spectral) is not V1FQRDeviceShellBranchBinding
            or type(self.diagonal_l1) is not V1FQRDeviceShellBranchBinding
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell family requires two exact branch bindings",
            )
        if (
            self.spectral.branch is not V1RelaxedKKTBranch.SPECTRAL
            or self.diagonal_l1.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
            or self.spectral.branch is self.diagonal_l1.branch
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
                "device-shell family requires distinct spectral and diagonal branches",
            )
        duplicates = tuple(self.cross_branch_action_identities)
        if any(
            type(item) is not V1FQRDeviceShellCrossBranchActionIdentity
            for item in duplicates
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "cross-branch identities require exact durable records",
            )
        if any(
            type(value) is not bool
            for value in (
                self.c_stream_outcomes_seen,
                self.cross_branch_action_synthesis_authority,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "device-shell family authority fields must be exact booleans",
            )
        if any(
            (
                self.c_stream_outcomes_seen,
                self.cross_branch_action_synthesis_authority,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "device-shell family crossed its identity-only authority boundary",
            )
        if (
            self.identity_scope != FQR_DEVICE_SHELL_IDENTITY_SCOPE
            or self.schema != FQR_DEVICE_SHELL_BRANCH_FAMILY_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell family identity scope or schema is invalid",
            )
        object.__setattr__(self, "cross_branch_action_identities", duplicates)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "execution_receipt_hash": self.execution_receipt_hash,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "base_result_hash": self.base_result_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "endpoint_axis_hash": self.endpoint_axis_hash,
            "spectral_binding_hash": self.spectral.binding_hash,
            "diagonal_l1_binding_hash": self.diagonal_l1.binding_hash,
            "cross_branch_action_identity_hashes": tuple(
                item.identity_hash for item in self.cross_branch_action_identities
            ),
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "cross_branch_action_synthesis_authority": (
                self.cross_branch_action_synthesis_authority
            ),
            "workflow_authority": self.workflow_authority,
            "identity_scope": self.identity_scope,
        }

    def _expected_cross_branch_identities(
        self,
    ) -> tuple[V1FQRDeviceShellCrossBranchActionIdentity, ...]:
        return _device_shell_cross_branch_identities(
            self.spectral, self.diagonal_l1
        )

    @property
    def family_hash(self) -> str:
        cached = self._family_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.spectral.assert_integrity()
        self.diagonal_l1.assert_integrity()
        common = (
            self.execution_receipt_hash,
            self.problem_hash,
            self.policy_hash,
            self.base_result_hash,
            self.route_axis_hash,
            self.route_count,
            self.current_mask_hash,
            self.chart_identity_hash,
            self.proposal_generation,
            self.discovery_prefix_hash,
            self.discovery_evidence_hash,
            self.endpoint_axis_hash,
        )
        for binding in (self.spectral, self.diagonal_l1):
            if (
                binding.execution_receipt_hash,
                binding.problem_hash,
                binding.policy_hash,
                binding.base_result_hash,
                binding.route_axis_hash,
                binding.route_count,
                binding.current_mask_hash,
                binding.chart_identity_hash,
                binding.proposal_generation,
                binding.discovery_prefix_hash,
                binding.discovery_evidence_hash,
                binding.endpoint_axis_hash,
            ) != common:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "device-shell branches do not share one action axis and chart",
                )
        if (
            self.spectral.device_shell_execution_design_hash
            != self.diagonal_l1.device_shell_execution_design_hash
            or self.spectral.cuda_execution_design_hash
            != self.diagonal_l1.cuda_execution_design_hash
            or self.spectral.hbm_forecast_hash
            != self.diagonal_l1.hbm_forecast_hash
            or self.spectral.cuda_telemetry_hash
            != self.diagonal_l1.cuda_telemetry_hash
            or self.spectral.eligibility_commitment
            != self.diagonal_l1.eligibility_commitment
            or self.spectral.block_capacity_hash
            != self.diagonal_l1.block_capacity_hash
            or self.spectral.block_legality_scope
            != self.diagonal_l1.block_legality_scope
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell branches do not share one execution and legality source",
            )
        expected_duplicates = self._expected_cross_branch_identities()
        for identity in self.cross_branch_action_identities:
            identity.assert_integrity()
        if tuple(
            identity.identity_hash for identity in self.cross_branch_action_identities
        ) != tuple(identity.identity_hash for identity in expected_duplicates):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "cross-branch duplicate actions are not represented exactly once",
            )
        if any(
            (
                self.c_stream_outcomes_seen,
                self.cross_branch_action_synthesis_authority,
                self.workflow_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "device-shell family no longer has identity-only authority",
            )
        observed = content_hash(self._payload(), prefix=_DEVICE_SHELL_FAMILY_PREFIX)
        cached = self._family_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell branch family was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell branch family hash does not match",
            )
        object.__setattr__(self, "_family_hash", observed)
        return observed


def _bind_v1_fqr_device_shell_branch(
    *,
    execution_receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
    execution_receipt_hash: str,
    branch_receipt: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    chart_identity_hash: str,
    proposal_generation: int,
    discovery_prefix_hash: str,
    discovery_evidence_hash: str,
    registered_resources: tuple[float, ...],
) -> V1FQRDeviceShellBranchBinding:
    ladder = branch_receipt.frozen_ladder
    if type(registered_resources) is not tuple:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "registered resources must be an exact tuple",
        )
    resources = tuple(
        _finite_nonnegative(value, name="registered endpoint resource")
        for value in registered_resources
    )
    if len(resources) != ladder.endpoint_count or tuple(sorted(resources)) != resources:
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            "registered resources must align monotonically with every endpoint",
        )
    branch_receipt_hash = branch_receipt.receipt_hash
    ladder_receipt_hash = ladder.receipt_hash
    endpoints = tuple(
        V1FQRDeviceShellEndpointIdentity(
            branch=branch_receipt.branch,
            endpoint_index=action.endpoint_label,
            problem_hash=execution_receipt.problem_hash,
            policy_hash=execution_receipt.policy_hash,
            base_result_hash=execution_receipt.base_result_hash,
            execution_receipt_hash=execution_receipt_hash,
            branch_identity_receipt_hash=branch_receipt_hash,
            source_rounding_hash=branch_receipt.source_rounding_hash,
            source_ladder_receipt_hash=ladder_receipt_hash,
            source_action_receipt_hash=action.receipt_hash,
            route_axis_hash=ladder.route_axis_hash,
            route_count=ladder.route_count,
            current_mask_hash=ladder.parent_mask_hash,
            chart_identity_hash=chart_identity_hash,
            proposal_generation=proposal_generation,
            discovery_prefix_hash=discovery_prefix_hash,
            discovery_evidence_hash=discovery_evidence_hash,
            shell_count=action.shell_count,
            cumulative_removed_count=action.cumulative_removed_count,
            registered_resource=resource,
            removed_route_index_hash=action.removed_route_index_hash,
            resulting_mask_packed_sha256=action.resulting_mask_packed_sha256,
            resulting_mask_hash=action.resulting_mask_hash,
            packed_action_hash=action.packed_action_hash,
        )
        for action, resource in zip(ladder.actions, resources, strict=True)
    )
    endpoint_axis_hash = content_hash(
        {
            "route_axis_hash": ladder.route_axis_hash,
            "route_count": ladder.route_count,
            "current_mask_hash": ladder.parent_mask_hash,
            "axis": tuple(
                (
                    endpoint.endpoint_index,
                    endpoint.shell_count,
                    endpoint.cumulative_removed_count,
                )
                for endpoint in endpoints
            ),
        },
        prefix="partition-free-v1-fqr-device-shell-endpoint-axis-",
    )
    endpoint_resource_axis_hash = content_hash(
        {
            "branch": branch_receipt.branch,
            "source_rounding_hash": branch_receipt.source_rounding_hash,
            "label_commitment": ladder.label_commitment,
            "resources": resources,
        },
        prefix="partition-free-v1-fqr-device-shell-resource-axis-",
    )
    endpoint_physical_action_axis_hash = content_hash(
        tuple(endpoint.identity_hash for endpoint in endpoints),
        prefix="partition-free-v1-fqr-device-shell-physical-action-axis-",
    )
    return V1FQRDeviceShellBranchBinding(
        branch=branch_receipt.branch,
        problem_hash=execution_receipt.problem_hash,
        policy_hash=execution_receipt.policy_hash,
        base_result_hash=execution_receipt.base_result_hash,
        device_shell_execution_design_hash=(
            execution_receipt.device_shell_execution_design_hash
        ),
        cuda_execution_design_hash=execution_receipt.cuda_execution_design_hash,
        hbm_forecast_hash=execution_receipt.hbm_forecast_hash,
        cuda_telemetry_hash=execution_receipt.cuda_telemetry_hash,
        execution_receipt_hash=execution_receipt_hash,
        branch_identity_receipt_hash=branch_receipt_hash,
        source_rounding_hash=branch_receipt.source_rounding_hash,
        source_ladder_receipt_hash=ladder_receipt_hash,
        route_axis_hash=ladder.route_axis_hash,
        route_count=ladder.route_count,
        current_mask_hash=ladder.parent_mask_hash,
        chart_identity_hash=chart_identity_hash,
        proposal_generation=proposal_generation,
        discovery_prefix_hash=discovery_prefix_hash,
        discovery_evidence_hash=discovery_evidence_hash,
        label_commitment=ladder.label_commitment,
        eligibility_commitment=ladder.eligibility_commitment,
        block_capacity_hash=ladder.block_capacity_hash,
        block_legality_scope=ladder.block_legality_scope,
        endpoint_axis_hash=endpoint_axis_hash,
        endpoint_resource_axis_hash=endpoint_resource_axis_hash,
        endpoint_physical_action_axis_hash=endpoint_physical_action_axis_hash,
        endpoints=endpoints,
    )


def freeze_v1_fqr_device_shell_branch_family(
    *,
    execution_receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
    chart_identity_hash: str,
    proposal_generation: int,
    discovery_prefix_hash: str,
    discovery_evidence_hash: str,
    spectral_registered_resources: tuple[float, ...],
    diagonal_l1_registered_resources: tuple[float, ...],
) -> V1FQRDeviceShellPreFrozenBranchFamily:
    """Adapt exact shell receipts into a compact, pre-C two-branch identity.

    The resource axes are explicit because the v1 device-shell execution
    receipt intentionally retains only the hard endpoint identities.  This
    adapter binds those registered values but grants no F/Q, C, or workflow
    authority.
    """

    execution_hash = _validated_device_shell_execution_hash(execution_receipt)
    chart_hash = _identity(chart_identity_hash, name="chart identity hash")
    generation = _nonnegative_int(
        proposal_generation, name="proposal generation"
    )
    prefix_hash = _identity(discovery_prefix_hash, name="discovery prefix hash")
    evidence_hash = _identity(
        discovery_evidence_hash, name="discovery evidence hash"
    )
    spectral = _bind_v1_fqr_device_shell_branch(
        execution_receipt=execution_receipt,
        execution_receipt_hash=execution_hash,
        branch_receipt=execution_receipt.spectral,
        chart_identity_hash=chart_hash,
        proposal_generation=generation,
        discovery_prefix_hash=prefix_hash,
        discovery_evidence_hash=evidence_hash,
        registered_resources=spectral_registered_resources,
    )
    diagonal = _bind_v1_fqr_device_shell_branch(
        execution_receipt=execution_receipt,
        execution_receipt_hash=execution_hash,
        branch_receipt=execution_receipt.diagonal_l1,
        chart_identity_hash=chart_hash,
        proposal_generation=generation,
        discovery_prefix_hash=prefix_hash,
        discovery_evidence_hash=evidence_hash,
        registered_resources=diagonal_l1_registered_resources,
    )
    duplicates = _device_shell_cross_branch_identities(spectral, diagonal)
    return V1FQRDeviceShellPreFrozenBranchFamily(
        execution_receipt_hash=execution_hash,
        problem_hash=execution_receipt.problem_hash,
        policy_hash=execution_receipt.policy_hash,
        base_result_hash=execution_receipt.base_result_hash,
        route_axis_hash=spectral.route_axis_hash,
        route_count=spectral.route_count,
        current_mask_hash=spectral.current_mask_hash,
        chart_identity_hash=chart_hash,
        proposal_generation=generation,
        discovery_prefix_hash=prefix_hash,
        discovery_evidence_hash=evidence_hash,
        endpoint_axis_hash=spectral.endpoint_axis_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        cross_branch_action_identities=duplicates,
    )


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellSourceStabilityBinding:
    """Authority-inert join of one shell, nominal KKT diagnostics, and F/Q.

    A fixed-direction ``M_J z`` witness can diagnose margins for the nominal
    relaxed solution; it is not a solution-sensitivity theorem.  Consequently
    the nominal face and rounded-shell flags contribute to ``source_stable``
    only when every radius that perturbs the relaxed problem itself is exactly
    zero, when the adaptive-rank witness is bound to an explicit robust KKT
    solution-sensitivity certificate, or when its finite decision shell is
    separately closed by the decision-level invariance theorem.  The latter is
    decision authority, not a full relaxed-optimizer sensitivity claim.
    Solver/key arithmetic remains covered by the
    witness's outward numerical margins.  The CUDA receipt independently
    establishes complete hard F/Q replay for every registered endpoint.  This
    object cannot open C by itself.
    """

    branch_binding: V1FQRDeviceShellBranchBinding
    adaptive_rank_witness: V1RelaxedKKTAdaptiveRankBranchWitness
    cuda_hard_fq_replay: V1DeviceShellHardFQCUDAReplayReceipt
    literal_j_geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet
    literal_j_fixed_primal_census: LiteralJFixedPrimalSourceReceipt
    shell_endpoint_action_axis_hash: str
    private_primal_direction_descriptor_axis_hash: str
    routewise_kkt_face_stable: bool
    packed_endpoint_shell_stable: bool
    complete_cuda_hard_fq_replay: bool
    hard_fq_status_axis_stable: bool
    fixed_z_face_perturbation_only: bool = True
    full_kkt_solution_sensitivity_theorem: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FQR_DEVICE_SHELL_SOURCE_STABILITY_BINDING_SCHEMA
    _binding_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        branch_binding = self.branch_binding
        witness = self.adaptive_rank_witness
        replay = self.cuda_hard_fq_replay
        uncertainty = self.literal_j_geometry_uncertainty
        census = self.literal_j_fixed_primal_census
        if type(branch_binding) is not V1FQRDeviceShellBranchBinding:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source stability requires an exact shell branch binding",
            )
        if type(witness) is not V1RelaxedKKTAdaptiveRankBranchWitness:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source stability requires an exact adaptive-rank witness",
            )
        if type(replay) is not V1DeviceShellHardFQCUDAReplayReceipt:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source stability requires an exact CUDA hard-F/Q receipt",
            )
        if type(uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source stability requires exact literal-J geometry uncertainty",
            )
        if type(census) not in (
            V1LiteralJOperatorCensusReceipt,
            V1LiteralJFixedDirectionProjectionReceipt,
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source stability requires the exact fixed-primal literal-J receipt",
            )
        census.assert_integrity()
        branch_binding.assert_integrity()
        object.__setattr__(
            self,
            "shell_endpoint_action_axis_hash",
            _identity(
                self.shell_endpoint_action_axis_hash,
                name="shell endpoint action axis hash",
            ),
        )
        object.__setattr__(
            self,
            "private_primal_direction_descriptor_axis_hash",
            _identity(
                self.private_primal_direction_descriptor_axis_hash,
                name="private-primal direction descriptor axis hash",
            ),
        )
        # Re-run the CUDA receipt validator.  The witness is source-issued via
        # an InitVar token that is deliberately not replayable; its complete
        # content hash is bound below instead of calling its constructor again.
        replay.__post_init__()
        expected_face = witness.relaxed_face_stable
        expected_shell = witness.rounded_endpoint_set_stable
        expected_fq = (
            replay.complete_registered_ladder_replayed
            and not replay.nonmonotone_q_early_stop_used
            and replay.label_membership_used_directly
            and replay.stable_physical_tie_order
            and tuple(replay.packed_action_hashes)
            == tuple(
                endpoint.packed_action_hash for endpoint in branch_binding.endpoints
            )
        )
        expected_fq_status_axis = all(
            endpoint.status is not V1DeviceShellHardFQStatus.FD_AMBIGUOUS
            for endpoint in replay.endpoints
        )
        for name, expected in (
            ("routewise_kkt_face_stable", expected_face),
            ("packed_endpoint_shell_stable", expected_shell),
            ("complete_cuda_hard_fq_replay", expected_fq),
            ("hard_fq_status_axis_stable", expected_fq_status_axis),
        ):
            supplied = getattr(self, name)
            if type(supplied) is not bool:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                    f"{name} must be an exact boolean",
                )
            if supplied is not expected:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.UNSTABLE_DECISION,
                    f"{name} does not replay its source receipts",
                )
        expected_full_theorem = (
            witness.relaxed_solution_sensitivity_certificate_hash is not None
        )
        expected_fixed_z_scope = not expected_full_theorem
        scope_flags = (
            self.fixed_z_face_perturbation_only,
            self.full_kkt_solution_sensitivity_theorem,
        )
        if any(type(value) is not bool for value in scope_flags):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "fixed-z face-perturbation scope fields must be exact booleans",
            )
        if scope_flags != (expected_fixed_z_scope, expected_full_theorem):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "source-stability scope does not match the KKT sensitivity "
                "certificate bound to the witness",
            )
        authority = (
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in authority):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "source-stability authority fields must be exact booleans",
            )
        if any(authority):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "a branch source binding cannot open downstream workflow stages",
            )
        if (
            witness.branch is not branch_binding.branch
            or replay.branch is not branch_binding.branch
            or witness.problem_hash != branch_binding.problem_hash
            or replay.problem_hash != branch_binding.problem_hash
            or uncertainty.problem_hash != branch_binding.problem_hash
            or witness.policy_hash != branch_binding.policy_hash
            or replay.policy_hash != branch_binding.policy_hash
            or witness.branch_result_hash != replay.branch_result_hash
            or witness.branch_receipt_hash != replay.branch_kkt_receipt_hash
            or witness.rounded_ladder_hash != branch_binding.source_rounding_hash
            or witness.endpoint_action_axis_hash
            != self.shell_endpoint_action_axis_hash
            or witness.route_axis_hash != branch_binding.route_axis_hash
            or replay.route_axis_hash != branch_binding.route_axis_hash
            or uncertainty.route_axis_hash != branch_binding.route_axis_hash
            or witness.route_count != branch_binding.route_count
            or replay.route_count != branch_binding.route_count
            or replay.source_geometry_hash != branch_binding.chart_identity_hash
            or uncertainty.source_geometry_hash != branch_binding.chart_identity_hash
            or census.chart_identity_hash != branch_binding.chart_identity_hash
            or census.current_mask_hash != branch_binding.current_mask_hash
            or census.proposal_generation != branch_binding.proposal_generation
            or census.route_axis_hash != branch_binding.route_axis_hash
            or census.route_count != branch_binding.route_count
            or census.direction_count != 2
            or census.direction_axis_hash
            != uncertainty.literal_j_fixed_primal_direction_axis_hash
            or uncertainty.literal_j_fixed_primal_direction_count != 2
            or witness.literal_j_fixed_primal_direction_axis_hash
            != census.direction_axis_hash
            or not (
                (
                    type(census) is V1LiteralJOperatorCensusReceipt
                    and census.exact_finite_population_census
                    and census.observed_counts == census.population_counts
                    and census.directions_frozen_before_census
                )
                or (
                    type(census) is V1LiteralJFixedDirectionProjectionReceipt
                    and not census.exact_finite_population_census
                    and census.directions_frozen_before_projection
                    and census.independent_or_predictable_witness_stream
                    and census.source_derived_for_fixed_primal_projection
                    and census.fixed_z_only_no_kkt_solution_sensitivity
                )
            )
            or census.literal_j_rows_retained
            or any(
                (
                    census.face_authority,
                    census.shell_authority,
                    census.c_stream_authority,
                    census.fqr_authority,
                    census.calibration_authority,
                    census.oracle_authority,
                    census.commit_authority,
                    census.production_authority,
                    census.pareto_authority,
                )
            )
            or getattr(
                census,
                "private_primal_direction_descriptor_axis_hash",
                None,
            )
            != self.private_primal_direction_descriptor_axis_hash
            or not uncertainty.source_derived_from_registered_geometry_receipts
            or not uncertainty.literal_j_response_operator_is_statistical_estimand
            or uncertainty.gauge_dependent_low_rank_factor_is_statistical_estimand
            or witness.finite_population_geometry_uncertainty_set_hash
            != uncertainty.uncertainty_hash
            or replay.device_shell_execution_receipt_hash
            != branch_binding.execution_receipt_hash
            or replay.source_branch_identity_receipt_hash
            != branch_binding.branch_identity_receipt_hash
            or replay.frozen_ladder_receipt_hash
            != branch_binding.source_ladder_receipt_hash
            or replay.source_rounding_hash != branch_binding.source_rounding_hash
            or replay.parent_mask_hash != branch_binding.current_mask_hash
            or replay.label_commitment != branch_binding.label_commitment
            or replay.eligibility_commitment
            != branch_binding.eligibility_commitment
            or replay.block_capacity_hash != branch_binding.block_capacity_hash
            or replay.block_legality_scope != branch_binding.block_legality_scope
            or replay.endpoint_count != len(branch_binding.endpoints)
            or tuple(replay.frozen_action_receipt_hashes)
            != tuple(
                endpoint.source_action_receipt_hash
                for endpoint in branch_binding.endpoints
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "witness, CUDA replay, and packed shell left one source branch",
            )
        if self.schema != FQR_DEVICE_SHELL_SOURCE_STABILITY_BINDING_SCHEMA:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "device-shell source-stability schema is invalid",
            )
        self.assert_integrity()

    @property
    def source_stable(self) -> bool:
        return (
            self.decision_relevant_perturbation_closed
            and self.routewise_kkt_face_stable
            and self.packed_endpoint_shell_stable
            and self.complete_cuda_hard_fq_replay
            and self.hard_fq_status_axis_stable
        )

    @property
    def relaxed_problem_sensitivity_closed(self) -> bool:
        """Whether uncertainty leaves the full relaxed optimizer sensitivity closed."""

        return (
            self.relaxed_problem_perturbation_exact_zero
            or self.full_kkt_solution_sensitivity_theorem
        )

    @property
    def decision_level_shell_invariance_closed(self) -> bool:
        """Whether the narrower finite shell theorem closes this branch decision."""

        witness = self.adaptive_rank_witness
        return (
            witness.closure_mode
            is V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
            and witness.decision_level_robust_invariance_certificate_hash is not None
        )

    @property
    def decision_relevant_perturbation_closed(self) -> bool:
        """Whether perturbations are closed for the next irreversible decision."""

        return (
            self.relaxed_problem_sensitivity_closed
            or self.decision_level_shell_invariance_closed
        )

    @property
    def relaxed_problem_perturbation_exact_zero(self) -> bool:
        """Whether nominal face/shell margins need no sensitivity theorem.

        These five nonnegative radii change ``A``, ``M_J``, or the FD response
        operator.  Numerical solution/key radii are intentionally excluded:
        their existing outward-enclosed margins certify the already registered
        relaxed problem rather than a perturbed optimizer.
        """

        uncertainty = self.literal_j_geometry_uncertainty
        return all(
            value == 0.0
            for value in (
                uncertainty.corrected_work_A_sampling_linf_radius,
                uncertainty.corrected_work_A_deterministic_linf_radius,
                uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius,
                uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius,
                uncertainty.fd_q_residual_deterministic_linf_adjoint_radius,
            )
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch_binding_hash": self.branch_binding.binding_hash,
            "adaptive_rank_witness_hash": self.adaptive_rank_witness.witness_hash,
            "cuda_hard_fq_replay_hash": self.cuda_hard_fq_replay.receipt_hash,
            "literal_j_geometry_uncertainty_hash": (
                self.literal_j_geometry_uncertainty.uncertainty_hash
            ),
            "literal_j_fixed_primal_census_receipt_hash": (
                self.literal_j_fixed_primal_census.receipt_hash
            ),
            "literal_j_fixed_primal_direction_axis_hash": (
                self.literal_j_fixed_primal_census.direction_axis_hash
            ),
            "private_primal_direction_descriptor_axis_hash": (
                self.private_primal_direction_descriptor_axis_hash
            ),
            "shell_endpoint_action_axis_hash": (
                self.shell_endpoint_action_axis_hash
            ),
            "routewise_kkt_face_stable": self.routewise_kkt_face_stable,
            "packed_endpoint_shell_stable": self.packed_endpoint_shell_stable,
            "complete_cuda_hard_fq_replay": self.complete_cuda_hard_fq_replay,
            "hard_fq_status_axis_stable": self.hard_fq_status_axis_stable,
            "relaxed_problem_perturbation_exact_zero": (
                self.relaxed_problem_perturbation_exact_zero
            ),
            "fixed_z_face_perturbation_only": self.fixed_z_face_perturbation_only,
            "full_kkt_solution_sensitivity_theorem": (
                self.full_kkt_solution_sensitivity_theorem
            ),
            "c_stream_authority": self.c_stream_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_DEVICE_SHELL_SOURCE_STABILITY_PREFIX
        )
        cached = self._binding_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell source-stability binding was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "device-shell source-stability binding hash does not match",
            )
        object.__setattr__(self, "_binding_hash", observed)
        return observed


def bind_v1_fqr_device_shell_source_stability(
    *,
    branch_binding: V1FQRDeviceShellBranchBinding,
    adaptive_rank_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    cuda_hard_fq_replay: V1DeviceShellHardFQCUDAReplayReceipt,
    literal_j_geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    literal_j_fixed_primal_census: LiteralJFixedPrimalSourceReceipt,
    shell_endpoint_action_axis_hash: str,
    private_primal_direction_descriptor_axis_hash: str,
) -> V1FQRDeviceShellSourceStabilityBinding:
    """Bind exact source receipts without elevating their authority."""

    return V1FQRDeviceShellSourceStabilityBinding(
        branch_binding=branch_binding,
        adaptive_rank_witness=adaptive_rank_witness,
        cuda_hard_fq_replay=cuda_hard_fq_replay,
        literal_j_geometry_uncertainty=literal_j_geometry_uncertainty,
        literal_j_fixed_primal_census=literal_j_fixed_primal_census,
        shell_endpoint_action_axis_hash=shell_endpoint_action_axis_hash,
        private_primal_direction_descriptor_axis_hash=(
            private_primal_direction_descriptor_axis_hash
        ),
        routewise_kkt_face_stable=adaptive_rank_witness.relaxed_face_stable,
        packed_endpoint_shell_stable=(
            adaptive_rank_witness.rounded_endpoint_set_stable
        ),
        complete_cuda_hard_fq_replay=(
            cuda_hard_fq_replay.complete_registered_ladder_replayed
            and not cuda_hard_fq_replay.nonmonotone_q_early_stop_used
            and cuda_hard_fq_replay.label_membership_used_directly
            and cuda_hard_fq_replay.stable_physical_tie_order
            and tuple(cuda_hard_fq_replay.packed_action_hashes)
            == tuple(
                endpoint.packed_action_hash
                for endpoint in branch_binding.endpoints
            )
        ),
        hard_fq_status_axis_stable=all(
            endpoint.status is not V1DeviceShellHardFQStatus.FD_AMBIGUOUS
            for endpoint in cuda_hard_fq_replay.endpoints
        ),
        fixed_z_face_perturbation_only=(
            adaptive_rank_witness.relaxed_solution_sensitivity_certificate_hash
            is None
        ),
        full_kkt_solution_sensitivity_theorem=(
            adaptive_rank_witness.relaxed_solution_sensitivity_certificate_hash
            is not None
        ),
    )


@dataclass(frozen=True, slots=True)
class V1FQRKKTBranchFrozenBinding:
    """One KKT rounded ladder bound to its exact packed pre-C ladder."""

    branch: V1RelaxedKKTBranch
    rounded_ladder: V1RelaxedKKTRoundedLadder = field(repr=False, compare=False)
    frozen_ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    problem_hash: str
    policy_hash: str
    result_hash: str
    branch_result_hash: str
    rounded_ladder_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    endpoint_prefix_counts: tuple[int, ...]
    endpoint_kkt_candidate_hashes: tuple[str, ...]
    endpoint_physical_prefix_digests: tuple[str, ...]
    endpoint_removed_route_index_hashes: tuple[str, ...]
    endpoint_packed_action_hashes: tuple[str, ...]
    endpoint_registered_resources: tuple[float, ...]
    endpoint_kkt_feasibility: tuple[bool, ...]
    hard_global_optimality_claim: bool = False
    c_stream_outcomes_seen: bool = False
    schema: str = FQR_KKT_BRANCH_BINDING_SCHEMA
    _binding_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.branch, V1RelaxedKKTBranch):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch binding requires a registered KKT branch enum",
            )
        if type(self.rounded_ladder) is not V1RelaxedKKTRoundedLadder:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch binding requires the exact rounded-ladder type",
            )
        if type(self.frozen_ladder) is not V1FrozenCandidateLadder:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch binding requires the exact frozen-ladder type",
            )
        if (
            self.rounded_ladder.branch is not self.branch
            or not self.frozen_ladder.is_packed
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "branch label or packed frozen-ladder representation differs",
            )
        for name in (
            "problem_hash",
            "policy_hash",
            "result_hash",
            "branch_result_hash",
            "rounded_ladder_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        counts = tuple(
            _positive_int(value, name="endpoint prefix count")
            for value in self.endpoint_prefix_counts
        )
        hashes = tuple(
            tuple(_identity(value, name=name) for value in getattr(self, name))
            for name in (
                "endpoint_kkt_candidate_hashes",
                "endpoint_physical_prefix_digests",
                "endpoint_removed_route_index_hashes",
                "endpoint_packed_action_hashes",
            )
        )
        resources = tuple(
            _finite_nonnegative(value, name="registered endpoint resource")
            for value in self.endpoint_registered_resources
        )
        feasibility = tuple(self.endpoint_kkt_feasibility)
        if any(type(value) is not bool for value in feasibility):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "KKT feasibility values must be exact booleans",
            )
        width = len(self.rounded_ladder.candidates)
        if width == 0 or any(
            len(axis) != width for axis in (counts, *hashes, resources, feasibility)
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "branch endpoint axes must be nonempty and aligned",
            )
        if self.hard_global_optimality_claim is not False:
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "a rounded hard ladder has no global-optimality authority",
            )
        if self.c_stream_outcomes_seen is not False:
            _refuse(
                V1FQRBranchFamilyRefusalCode.POST_SELECTION,
                "branch ladder was not frozen before C-stream outcomes",
            )
        if self.schema != FQR_KKT_BRANCH_BINDING_SCHEMA:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "branch binding schema is invalid",
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "endpoint_prefix_counts", counts)
        for name, axis in zip(
            (
                "endpoint_kkt_candidate_hashes",
                "endpoint_physical_prefix_digests",
                "endpoint_removed_route_index_hashes",
                "endpoint_packed_action_hashes",
            ),
            hashes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        object.__setattr__(self, "endpoint_registered_resources", resources)
        object.__setattr__(self, "endpoint_kkt_feasibility", feasibility)
        self.assert_integrity()

    def _dynamic_axes(self) -> tuple[tuple[object, ...], ...]:
        candidates = self.rounded_ladder.candidates
        physical = tuple(
            _array_digest(self.rounded_ladder.removed_indices(index))
            for index in range(len(candidates))
        )
        actions = self.frozen_ladder.actions
        removed_hashes = tuple(
            action.packed_removal_action.removed_route_index_hash  # type: ignore[union-attr]
            for action in actions
        )
        packed_hashes = tuple(
            action.packed_removal_action.action_hash  # type: ignore[union-attr]
            for action in actions
        )
        return (
            tuple(candidate.prefix_count for candidate in candidates),
            tuple(candidate.candidate_hash for candidate in candidates),
            physical,
            removed_hashes,
            packed_hashes,
            tuple(candidate.resource_value for candidate in candidates),
            tuple(candidate.feasible for candidate in candidates),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "result_hash": self.result_hash,
            "branch_result_hash": self.branch_result_hash,
            "rounded_ladder_hash": self.rounded_ladder_hash,
            "frozen_ladder_hash": self.frozen_ladder.ladder_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "endpoint_prefix_counts": self.endpoint_prefix_counts,
            "endpoint_kkt_candidate_hashes": self.endpoint_kkt_candidate_hashes,
            "endpoint_physical_prefix_digests": (self.endpoint_physical_prefix_digests),
            "endpoint_removed_route_index_hashes": (
                self.endpoint_removed_route_index_hashes
            ),
            "endpoint_packed_action_hashes": self.endpoint_packed_action_hashes,
            "endpoint_registered_resources": self.endpoint_registered_resources,
            "endpoint_kkt_feasibility": self.endpoint_kkt_feasibility,
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.hard_global_optimality_claim is not False
            or self.c_stream_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "branch binding no longer has pre-C hard-ladder-only authority",
            )
        try:
            rounded_hash = self.rounded_ladder.ladder_hash
            frozen_hash = self.frozen_ladder.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                f"branch ladder integrity failed: {error}",
            )
        dynamic = self._dynamic_axes()
        registered = (
            self.endpoint_prefix_counts,
            self.endpoint_kkt_candidate_hashes,
            self.endpoint_physical_prefix_digests,
            self.endpoint_removed_route_index_hashes,
            self.endpoint_packed_action_hashes,
            self.endpoint_registered_resources,
            self.endpoint_kkt_feasibility,
        )
        parent = self.frozen_ladder.parent_mask
        if (
            rounded_hash != self.rounded_ladder_hash
            or dynamic != registered
            or frozen_hash != self.frozen_ladder.ladder_hash
            or self.frozen_ladder.route_axis_hash != self.route_axis_hash
            or self.frozen_ladder.route_count != self.route_count
            or self.frozen_ladder.current_mask_hash != self.current_mask_hash
            or self.frozen_ladder.proposal_generation != self.proposal_generation
            or type(parent) is not HardMask
            or _direct_mask_hash(parent, name="frozen ladder parent")
            != self.current_mask_hash
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch binding differs from its rounded or packed ladder",
            )
        observed = content_hash(self._payload(), prefix=_BINDING_PREFIX)
        cached = self._binding_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch binding was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch binding hash does not match",
            )
        object.__setattr__(self, "_binding_hash", observed)
        return observed


def _adapt_branch(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    result: V1RelaxedKKTResult,
    branch: V1RelaxedKKTBranch,
    current_mask: HardMask,
    chart_identity_hash: str,
    proposal_generation: int,
    campaign_contract_hash: str,
    model_identity_hash: str,
    query_contract_hash: str,
    preprocessing_hash: str,
    raw_logit_stream_source_hash: str,
    sampling_law_hash: str,
    discovery_context_ids: tuple[str, ...],
    discovery_draw_ids: tuple[str, ...],
    discovery_evidence_hash: str,
    family_freeze_receipt_hash: str,
) -> V1FQRKKTBranchFrozenBinding:
    branch_result = _branch_result(result, branch)
    rounded = branch_result.rounded_ladder
    if not branch_result.receipt.convergence.converged or not rounded.candidates:
        _refuse(
            V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
            f"{branch.value} did not emit a nonempty converged hard ladder",
        )
    if (
        branch_result.receipt.problem_hash != problem.problem_hash
        or branch_result.receipt.policy_hash != policy.policy_hash
        or rounded.branch is not branch
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{branch.value} result differs from its problem or policy",
        )
    actions: list[V1FrozenLadderAction] = []
    physical_digests: list[str] = []
    removed_hashes: list[str] = []
    packed_hashes: list[str] = []
    for endpoint, candidate in enumerate(rounded.candidates, start=1):
        prefix = rounded.removed_indices(endpoint - 1)
        digest = _array_digest(prefix)
        if digest != candidate.removed_index_hash:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                f"{branch.value} endpoint {endpoint} physical prefix changed",
            )
        if (
            prefix.size != candidate.prefix_count
            or np.any(prefix >= problem.route_count)
            or np.any(~problem.eligible_mask[prefix])
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                f"{branch.value} endpoint {endpoint} left the eligible route axis",
            )
        packed = V1PackedRemovalAction.from_removed_indices(
            parent_mask=current_mask,
            removed_indices=prefix,
        )
        if packed.removed_count != candidate.prefix_count:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                f"{branch.value} endpoint {endpoint} removes a non-live route",
            )
        actions.append(
            V1FrozenLadderAction(
                endpoint_index=endpoint,
                action_id=(
                    f"relaxed-kkt:{branch.value}:"
                    f"{candidate.candidate_hash}:{digest}"
                ),
                packed_removal_action=packed,
            )
        )
        physical_digests.append(digest)
        removed_hashes.append(packed.removed_route_index_hash)
        packed_hashes.append(packed.action_hash)

    branch_freeze_hash = content_hash(
        {
            "family_freeze_receipt_hash": family_freeze_receipt_hash,
            "problem_hash": problem.problem_hash,
            "policy_hash": policy.policy_hash,
            "result_hash": result.result_hash,
            "branch": branch,
            "branch_result_hash": branch_result.result_hash,
            "rounded_ladder_hash": rounded.ladder_hash,
            "physical_prefix_digests": tuple(physical_digests),
            "issued_before_c_outcomes": True,
        },
        prefix="partition-free-v1-fqr-kkt-branch-freeze-",
    )
    frozen = V1FrozenCandidateLadder(
        campaign_contract_hash=campaign_contract_hash,
        model_identity_hash=model_identity_hash,
        query_contract_hash=query_contract_hash,
        preprocessing_hash=preprocessing_hash,
        current_mask_hash=_direct_mask_hash(current_mask, name="current mask"),
        raw_logit_stream_source_hash=raw_logit_stream_source_hash,
        sampling_law_hash=sampling_law_hash,
        proposal_generation=proposal_generation,
        route_count=problem.route_count,
        route_axis_hash=problem.route_axis_hash,
        actions=tuple(actions),
        discovery_context_ids=discovery_context_ids,
        discovery_draw_ids=discovery_draw_ids,
        discovery_evidence_hash=discovery_evidence_hash,
        freeze_receipt_hash=branch_freeze_hash,
        parent_mask=current_mask,
    )
    discovery_prefix_hash = content_hash(
        {
            "context_ids": tuple(discovery_context_ids),
            "draw_ids": tuple(discovery_draw_ids),
        },
        prefix="partition-free-v1-fqr-kkt-discovery-prefix-",
    )
    return V1FQRKKTBranchFrozenBinding(
        branch=branch,
        rounded_ladder=rounded,
        frozen_ladder=frozen,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        result_hash=result.result_hash,
        branch_result_hash=branch_result.result_hash,
        rounded_ladder_hash=rounded.ladder_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        current_mask_hash=frozen.current_mask_hash,
        chart_identity_hash=chart_identity_hash,
        proposal_generation=proposal_generation,
        discovery_prefix_hash=discovery_prefix_hash,
        discovery_evidence_hash=discovery_evidence_hash,
        endpoint_prefix_counts=tuple(
            candidate.prefix_count for candidate in rounded.candidates
        ),
        endpoint_kkt_candidate_hashes=tuple(
            candidate.candidate_hash for candidate in rounded.candidates
        ),
        endpoint_physical_prefix_digests=tuple(physical_digests),
        endpoint_removed_route_index_hashes=tuple(removed_hashes),
        endpoint_packed_action_hashes=tuple(packed_hashes),
        endpoint_registered_resources=tuple(
            candidate.resource_value for candidate in rounded.candidates
        ),
        endpoint_kkt_feasibility=tuple(
            candidate.feasible for candidate in rounded.candidates
        ),
    )


@dataclass(frozen=True, slots=True)
class V1FQRPreCDominanceCertificate:
    """Pre-C proof that an omitted branch contributes no new hard action."""

    source_complete_family_hash: str
    problem_hash: str
    policy_hash: str
    result_hash: str
    retained_branch: V1RelaxedKKTBranch
    omitted_branch: V1RelaxedKKTBranch
    retained_binding_hash: str
    omitted_binding_hash: str
    omitted_ladder_hash: str
    endpoint_coverage: tuple[tuple[int, int, str, float, float], ...]
    dominance_method: str = FQR_PRE_C_DOMINANCE_METHOD
    issued_before_c_outcomes: bool = True
    fqr_outcomes_seen: bool = False
    hard_global_optimality_claim: bool = False
    schema: str = FQR_KKT_PRE_C_DOMINANCE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_complete_family_hash",
            "problem_hash",
            "policy_hash",
            "result_hash",
            "retained_binding_hash",
            "omitted_binding_hash",
            "omitted_ladder_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            not isinstance(self.retained_branch, V1RelaxedKKTBranch)
            or not isinstance(self.omitted_branch, V1RelaxedKKTBranch)
            or self.retained_branch is self.omitted_branch
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "dominance must name distinct registered branches",
            )
        coverage: list[tuple[int, int, str, float, float]] = []
        for row in self.endpoint_coverage:
            if len(row) != 5:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                    "dominance coverage rows must have five fields",
                )
            omitted = _positive_int(row[0], name="omitted endpoint")
            retained = _positive_int(row[1], name="retained endpoint")
            action_hash = _identity(row[2], name="packed physical action hash")
            omitted_resource = _finite_nonnegative(
                row[3], name="omitted registered resource"
            )
            retained_resource = _finite_nonnegative(
                row[4], name="retained registered resource"
            )
            if retained_resource < omitted_resource:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                    "retained duplicate has smaller registered resource",
                )
            coverage.append(
                (omitted, retained, action_hash, omitted_resource, retained_resource)
            )
        normalized = tuple(coverage)
        if (
            not normalized
            or tuple(row[0] for row in normalized)
            != tuple(range(1, len(normalized) + 1))
            or len({row[2] for row in normalized}) != len(normalized)
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "dominance must cover every omitted endpoint exactly once",
            )
        if (
            self.dominance_method != FQR_PRE_C_DOMINANCE_METHOD
            or self.issued_before_c_outcomes is not True
            or self.fqr_outcomes_seen is not False
            or self.hard_global_optimality_claim is not False
            or self.schema != FQR_KKT_PRE_C_DOMINANCE_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "dominance certificate is postselected or exceeds its authority",
            )
        object.__setattr__(self, "endpoint_coverage", normalized)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_complete_family_hash": self.source_complete_family_hash,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "result_hash": self.result_hash,
            "retained_branch": self.retained_branch,
            "omitted_branch": self.omitted_branch,
            "retained_binding_hash": self.retained_binding_hash,
            "omitted_binding_hash": self.omitted_binding_hash,
            "omitted_ladder_hash": self.omitted_ladder_hash,
            "endpoint_coverage": self.endpoint_coverage,
            "dominance_method": self.dominance_method,
            "issued_before_c_outcomes": self.issued_before_c_outcomes,
            "fqr_outcomes_seen": self.fqr_outcomes_seen,
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
        }

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.issued_before_c_outcomes is not True
            or self.fqr_outcomes_seen is not False
            or self.hard_global_optimality_claim is not False
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "dominance certificate is postselected or exceeds its authority",
            )
        observed = content_hash(self._payload(), prefix=_DOMINANCE_PREFIX)
        cached = self._certificate_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "pre-C dominance certificate was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "pre-C dominance certificate hash does not match",
            )
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRKKTPreFrozenBranchFamily:
    """Both hard branch ladders frozen before C, or one proved duplicate."""

    problem_hash: str
    policy_hash: str
    result_hash: str
    relaxed_selected_branch: V1RelaxedKKTBranch | None
    spectral: V1FQRKKTBranchFrozenBinding | None
    diagonal_l1: V1FQRKKTBranchFrozenBinding | None
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    dominance_certificate: V1FQRPreCDominanceCertificate | None = None
    both_ladders_frozen_before_c: bool = True
    c_stream_outcomes_seen: bool = False
    hard_global_optimality_claim: bool = False
    cross_branch_action_synthesis_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    family_scope: str = FQR_KKT_BRANCH_FAMILY_SCOPE
    schema: str = FQR_KKT_BRANCH_FAMILY_SCHEMA
    _family_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "result_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.relaxed_selected_branch is not None and not isinstance(
            self.relaxed_selected_branch, V1RelaxedKKTBranch
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "relaxed selected branch must use the registered enum",
            )
        route_count = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        bindings = tuple(
            value for value in (self.spectral, self.diagonal_l1) if value is not None
        )
        if any(type(value) is not V1FQRKKTBranchFrozenBinding for value in bindings):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "family requires exact branch bindings",
            )
        expected_labels = tuple(
            branch
            for branch, binding in (
                (V1RelaxedKKTBranch.SPECTRAL, self.spectral),
                (V1RelaxedKKTBranch.DIAGONAL_L1, self.diagonal_l1),
            )
            if binding is not None
        )
        if tuple(binding.branch for binding in bindings) != expected_labels:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "family branches are absent or mislabelled",
            )
        if len(bindings) == 2:
            if self.dominance_certificate is not None:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                    "a complete family must not carry an omission certificate",
                )
        elif len(bindings) == 1:
            certificate = self.dominance_certificate
            if type(certificate) is not V1FQRPreCDominanceCertificate:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
                    "canonical pre-C freeze requires both branches unless exact "
                    "action dominance was certified before C",
                )
            retained = bindings[0]
            if (
                certificate.retained_branch is not retained.branch
                or certificate.omitted_branch is retained.branch
                or certificate.retained_binding_hash != retained.binding_hash
                or certificate.problem_hash != self.problem_hash
                or certificate.policy_hash != self.policy_hash
                or certificate.result_hash != self.result_hash
            ):
                _refuse(
                    V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                    "omission certificate differs from the retained family branch",
                )
        else:
            _refuse(
                V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
                "a pre-C family cannot omit both branches",
            )
        if (
            self.both_ladders_frozen_before_c is not True
            or self.c_stream_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.POST_SELECTION,
                "the hard branch family was not fixed before C outcomes",
            )
        if any(
            value is not False
            for value in (
                self.hard_global_optimality_claim,
                self.cross_branch_action_synthesis_authority,
                self.calibration_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "pre-C branch family crossed its freeze-only authority boundary",
            )
        if (
            self.family_scope != FQR_KKT_BRANCH_FAMILY_SCOPE
            or self.schema != FQR_KKT_BRANCH_FAMILY_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "pre-C branch family scope or schema is invalid",
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "proposal_generation", generation)
        self.assert_integrity()

    @property
    def retained_bindings(self) -> tuple[V1FQRKKTBranchFrozenBinding, ...]:
        return tuple(
            value for value in (self.spectral, self.diagonal_l1) if value is not None
        )

    def binding_for_branch(
        self, branch: V1RelaxedKKTBranch
    ) -> V1FQRKKTBranchFrozenBinding:
        value = (
            self.spectral if branch is V1RelaxedKKTBranch.SPECTRAL else self.diagonal_l1
        )
        if value is None:
            _refuse(
                V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
                f"{branch.value} is represented only by a pre-C dominance receipt",
            )
        return value

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "result_hash": self.result_hash,
            "relaxed_selected_branch": self.relaxed_selected_branch,
            "spectral_binding_hash": (
                None if self.spectral is None else self.spectral.binding_hash
            ),
            "diagonal_l1_binding_hash": (
                None if self.diagonal_l1 is None else self.diagonal_l1.binding_hash
            ),
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "dominance_certificate_hash": (
                None
                if self.dominance_certificate is None
                else self.dominance_certificate.certificate_hash
            ),
            "both_ladders_frozen_before_c": self.both_ladders_frozen_before_c,
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
            "cross_branch_action_synthesis_authority": (
                self.cross_branch_action_synthesis_authority
            ),
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "family_scope": self.family_scope,
        }

    @property
    def family_hash(self) -> str:
        cached = self._family_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.both_ladders_frozen_before_c is not True
            or self.c_stream_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.POST_SELECTION,
                "the hard branch family no longer has a valid pre-C freeze",
            )
        if any(
            value is not False
            for value in (
                self.hard_global_optimality_claim,
                self.cross_branch_action_synthesis_authority,
                self.calibration_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
            )
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "pre-C branch family crossed its freeze-only authority boundary",
            )
        for binding in self.retained_bindings:
            binding.assert_integrity()
            if (
                binding.problem_hash != self.problem_hash
                or binding.policy_hash != self.policy_hash
                or binding.result_hash != self.result_hash
                or binding.route_axis_hash != self.route_axis_hash
                or binding.route_count != self.route_count
                or binding.current_mask_hash != self.current_mask_hash
                or binding.chart_identity_hash != self.chart_identity_hash
                or binding.proposal_generation != self.proposal_generation
                or binding.discovery_prefix_hash != self.discovery_prefix_hash
                or binding.discovery_evidence_hash != self.discovery_evidence_hash
            ):
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "a frozen branch binding left the common family coordinates",
                )
        if self.dominance_certificate is not None:
            self.dominance_certificate.assert_integrity()
        observed = content_hash(self._payload(), prefix=_FAMILY_PREFIX)
        cached = self._family_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "pre-C branch family was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "pre-C branch family hash does not match",
            )
        object.__setattr__(self, "_family_hash", observed)
        return observed


def freeze_v1_relaxed_kkt_branch_family(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    result: V1RelaxedKKTResult,
    current_mask: HardMask,
    chart_identity_hash: str,
    proposal_generation: int,
    campaign_contract_hash: str,
    model_identity_hash: str,
    query_contract_hash: str,
    preprocessing_hash: str,
    raw_logit_stream_source_hash: str,
    sampling_law_hash: str,
    discovery_context_ids: tuple[str, ...],
    discovery_draw_ids: tuple[str, ...],
    discovery_evidence_hash: str,
    family_freeze_receipt_hash: str,
) -> V1FQRKKTPreFrozenBranchFamily:
    """Freeze both KKT hard ladders before any C-stream observation."""

    if (
        type(problem) is not V1RelaxedKKTProblem
        or type(policy) is not V1RelaxedKKTPolicy
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "branch freeze requires exact KKT problem and policy types",
        )
    if type(result) is not V1RelaxedKKTResult or type(current_mask) is not HardMask:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "branch freeze requires exact KKT result and current-mask types",
        )
    if (
        result.problem_hash != problem.problem_hash
        or result.policy_hash != policy.policy_hash
        or not result.both_branches_converged
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
            "canonical branch freeze requires both converged results from this problem",
        )
    if (
        current_mask.route_count != problem.route_count
        or current_mask.route_axis_hash != problem.route_axis_hash
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            "current mask differs from the KKT route axis",
        )
    chart = _identity(chart_identity_hash, name="chart identity hash")
    generation = _nonnegative_int(proposal_generation, name="proposal generation")
    evidence = _identity(discovery_evidence_hash, name="discovery evidence hash")
    freeze_hash = _identity(family_freeze_receipt_hash, name="family freeze receipt")
    common = dict(
        problem=problem,
        policy=policy,
        result=result,
        current_mask=current_mask,
        chart_identity_hash=chart,
        proposal_generation=generation,
        campaign_contract_hash=campaign_contract_hash,
        model_identity_hash=model_identity_hash,
        query_contract_hash=query_contract_hash,
        preprocessing_hash=preprocessing_hash,
        raw_logit_stream_source_hash=raw_logit_stream_source_hash,
        sampling_law_hash=sampling_law_hash,
        discovery_context_ids=tuple(discovery_context_ids),
        discovery_draw_ids=tuple(discovery_draw_ids),
        discovery_evidence_hash=evidence,
        family_freeze_receipt_hash=freeze_hash,
    )
    spectral = _adapt_branch(branch=V1RelaxedKKTBranch.SPECTRAL, **common)
    diagonal = _adapt_branch(branch=V1RelaxedKKTBranch.DIAGONAL_L1, **common)
    return V1FQRKKTPreFrozenBranchFamily(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        result_hash=result.result_hash,
        relaxed_selected_branch=result.selected_branch,
        spectral=spectral,
        diagonal_l1=diagonal,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        current_mask_hash=_direct_mask_hash(current_mask, name="current mask"),
        chart_identity_hash=chart,
        proposal_generation=generation,
        discovery_prefix_hash=spectral.discovery_prefix_hash,
        discovery_evidence_hash=evidence,
    )


def certify_v1_pre_c_exact_action_dominance(
    *,
    complete_family: V1FQRKKTPreFrozenBranchFamily,
    retained_branch: V1RelaxedKKTBranch,
) -> V1FQRPreCDominanceCertificate:
    """Certify only exact physical-action coverage, never scalar dominance."""

    if type(complete_family) is not V1FQRKKTPreFrozenBranchFamily:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "dominance certification requires the exact complete family type",
        )
    complete_family.assert_integrity()
    if complete_family.spectral is None or complete_family.diagonal_l1 is None:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
            "dominance must be proved from the complete pre-C family",
        )
    if not isinstance(retained_branch, V1RelaxedKKTBranch):
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "retained branch must use the registered enum",
        )
    omitted_branch = (
        V1RelaxedKKTBranch.DIAGONAL_L1
        if retained_branch is V1RelaxedKKTBranch.SPECTRAL
        else V1RelaxedKKTBranch.SPECTRAL
    )
    retained = complete_family.binding_for_branch(retained_branch)
    omitted = complete_family.binding_for_branch(omitted_branch)
    retained_by_action: dict[str, tuple[int, float]] = {}
    for endpoint, (action_hash, resource) in enumerate(
        zip(
            retained.endpoint_packed_action_hashes,
            retained.endpoint_registered_resources,
            strict=True,
        ),
        start=1,
    ):
        if action_hash in retained_by_action:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "retained branch repeats a packed physical action",
            )
        retained_by_action[action_hash] = (endpoint, resource)
    coverage: list[tuple[int, int, str, float, float]] = []
    for endpoint, (action_hash, omitted_resource) in enumerate(
        zip(
            omitted.endpoint_packed_action_hashes,
            omitted.endpoint_registered_resources,
            strict=True,
        ),
        start=1,
    ):
        match = retained_by_action.get(action_hash)
        if match is None or match[1] < omitted_resource:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "an omitted packed action lacks an equal-or-better exact duplicate",
            )
        coverage.append((endpoint, match[0], action_hash, omitted_resource, match[1]))
    return V1FQRPreCDominanceCertificate(
        source_complete_family_hash=complete_family.family_hash,
        problem_hash=complete_family.problem_hash,
        policy_hash=complete_family.policy_hash,
        result_hash=complete_family.result_hash,
        retained_branch=retained_branch,
        omitted_branch=omitted_branch,
        retained_binding_hash=retained.binding_hash,
        omitted_binding_hash=omitted.binding_hash,
        omitted_ladder_hash=omitted.frozen_ladder.ladder_hash,
        endpoint_coverage=tuple(coverage),
    )


def omit_v1_pre_c_dominated_branch(
    *,
    complete_family: V1FQRKKTPreFrozenBranchFamily,
    dominance_certificate: V1FQRPreCDominanceCertificate,
) -> V1FQRKKTPreFrozenBranchFamily:
    """Produce a compact family only after exact pre-C duplicate coverage."""

    if (
        type(complete_family) is not V1FQRKKTPreFrozenBranchFamily
        or type(dominance_certificate) is not V1FQRPreCDominanceCertificate
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "branch omission requires exact family and dominance types",
        )
    complete_family.assert_integrity()
    dominance_certificate.assert_integrity()
    if (
        complete_family.spectral is None
        or complete_family.diagonal_l1 is None
        or dominance_certificate.source_complete_family_hash
        != complete_family.family_hash
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
            "dominance certificate was not issued for this complete family",
        )
    retained = complete_family.binding_for_branch(dominance_certificate.retained_branch)
    omitted = complete_family.binding_for_branch(dominance_certificate.omitted_branch)
    if (
        dominance_certificate.retained_binding_hash != retained.binding_hash
        or dominance_certificate.omitted_binding_hash != omitted.binding_hash
        or dominance_certificate.omitted_ladder_hash
        != omitted.frozen_ladder.ladder_hash
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
            "dominance certificate branch identities differ from the family",
        )
    kwargs = {
        "problem_hash": complete_family.problem_hash,
        "policy_hash": complete_family.policy_hash,
        "result_hash": complete_family.result_hash,
        "relaxed_selected_branch": complete_family.relaxed_selected_branch,
        "spectral": (
            retained if retained.branch is V1RelaxedKKTBranch.SPECTRAL else None
        ),
        "diagonal_l1": (
            retained if retained.branch is V1RelaxedKKTBranch.DIAGONAL_L1 else None
        ),
        "route_axis_hash": complete_family.route_axis_hash,
        "route_count": complete_family.route_count,
        "current_mask_hash": complete_family.current_mask_hash,
        "chart_identity_hash": complete_family.chart_identity_hash,
        "proposal_generation": complete_family.proposal_generation,
        "discovery_prefix_hash": complete_family.discovery_prefix_hash,
        "discovery_evidence_hash": complete_family.discovery_evidence_hash,
        "dominance_certificate": dominance_certificate,
    }
    return V1FQRKKTPreFrozenBranchFamily(**kwargs)


@dataclass(frozen=True, slots=True)
class V1FQRBranchEvidenceBinding:
    """One branch's complete stable F/Q/R graph and queue authorization."""

    branch_binding: V1FQRKKTBranchFrozenBinding = field(repr=False, compare=False)
    design: V1FQRCertificationDesign = field(repr=False, compare=False)
    receipt: V1FQRCertificationReceipt = field(repr=False, compare=False)
    decision: V1FQRCertificationDecision = field(repr=False, compare=False)
    stability_certificate: V1DecisionStabilityCertificate = field(
        repr=False, compare=False
    )
    candidate_queue_authorization: V1FQRCandidateQueueAuthorization = field(
        repr=False, compare=False
    )
    branch: V1RelaxedKKTBranch
    evidence_graph_kind: str = FQR_BRANCH_EVIDENCE_GRAPH_KIND
    schema: str = FQR_BRANCH_EVIDENCE_BINDING_SCHEMA
    _evidence_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.branch_binding) is not V1FQRKKTBranchFrozenBinding:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch evidence requires the exact frozen branch binding",
            )
        if self.branch is not self.branch_binding.branch:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R evidence was attached to a different KKT branch",
            )
        if (
            type(self.receipt) is V1FrozenLadderCertificationReceipt
            or type(self.decision) is V1FrozenLadderDecision
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE,
                "an F/Q-only graph cannot enter the hard branch family",
            )
        expected = (
            (self.design, V1FQRCertificationDesign),
            (self.receipt, V1FQRCertificationReceipt),
            (self.decision, V1FQRCertificationDecision),
            (self.stability_certificate, V1DecisionStabilityCertificate),
            (
                self.candidate_queue_authorization,
                V1FQRCandidateQueueAuthorization,
            ),
        )
        if any(type(value) is not expected_type for value, expected_type in expected):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch evidence requires exact F/Q/R and bridge types",
            )
        if (
            self.evidence_graph_kind != FQR_BRANCH_EVIDENCE_GRAPH_KIND
            or self.schema != FQR_BRANCH_EVIDENCE_BINDING_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "branch evidence graph kind or schema is invalid",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "branch_binding_hash": self.branch_binding.binding_hash,
            "design_hash": self.design.design_hash,
            "receipt_hash": self.receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "stability_certificate_hash": self.stability_certificate.certificate_hash,
            "candidate_queue_authorization_hash": (
                self.candidate_queue_authorization.authorization_hash
            ),
            "evidence_graph_kind": self.evidence_graph_kind,
        }

    @property
    def evidence_hash(self) -> str:
        cached = self._evidence_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.evidence_graph_kind != FQR_BRANCH_EVIDENCE_GRAPH_KIND:
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "v1 branch evidence may not reinterpret a compact receipt as "
                "the record-retaining graph",
            )
        self.branch_binding.assert_integrity()
        ladder = self.branch_binding.frozen_ladder
        parent = ladder.parent_mask
        if type(parent) is not HardMask:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "packed branch ladder lost its parent mask",
            )
        try:
            replayed = authorize_v1_fqr_candidate_queue(
                design=self.design,
                receipt=self.receipt,
                decision=self.decision,
                stability_certificate=self.stability_certificate,
                frozen_ladder=ladder,
                current_mask=parent,
                live_chart_identity_hash=self.branch_binding.chart_identity_hash,
                live_proposal_generation=self.branch_binding.proposal_generation,
                live_ladder_hash=ladder.ladder_hash,
            )
        except V1FQRCandidateQueueRefusal as error:
            _translate_bridge_refusal(error)
        if (
            replayed.authorization_hash
            != self.candidate_queue_authorization.authorization_hash
            or self.candidate_queue_authorization.ladder.ladder_hash
            != ladder.ladder_hash
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch bridge authorization differs from deterministic replay",
            )
        observed = content_hash(self._payload(), prefix=_EVIDENCE_PREFIX)
        cached = self._evidence_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch evidence was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch evidence hash does not match",
            )
        object.__setattr__(self, "_evidence_hash", observed)
        return observed


def bind_v1_fqr_branch_evidence(
    *,
    branch_binding: V1FQRKKTBranchFrozenBinding,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    stability_certificate: V1DecisionStabilityCertificate,
) -> V1FQRBranchEvidenceBinding:
    """Validate and bind one complete F/Q/R graph to one frozen branch."""

    if type(branch_binding) is not V1FQRKKTBranchFrozenBinding:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "branch evidence requires the exact frozen branch binding",
        )
    if (
        type(receipt) is V1FrozenLadderCertificationReceipt
        or type(decision) is V1FrozenLadderDecision
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE,
            "an F/Q-only receipt or decision cannot authorize a branch",
        )
    parent = branch_binding.frozen_ladder.parent_mask
    if type(parent) is not HardMask:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "packed branch ladder lost its parent mask",
        )
    try:
        authorization = authorize_v1_fqr_candidate_queue(
            design=design,
            receipt=receipt,
            decision=decision,
            stability_certificate=stability_certificate,
            frozen_ladder=branch_binding.frozen_ladder,
            current_mask=parent,
            live_chart_identity_hash=branch_binding.chart_identity_hash,
            live_proposal_generation=branch_binding.proposal_generation,
            live_ladder_hash=branch_binding.frozen_ladder.ladder_hash,
        )
    except V1FQRCandidateQueueRefusal as error:
        _translate_bridge_refusal(error)
    return V1FQRBranchEvidenceBinding(
        branch_binding=branch_binding,
        design=design,
        receipt=receipt,
        decision=decision,
        stability_certificate=stability_certificate,
        candidate_queue_authorization=authorization,
        branch=branch_binding.branch,
    )


@dataclass(frozen=True, slots=True)
class V1FQRBranchFamilyAuthorization:
    """Proposal-only choice over the independently certified hard union."""

    prefrozen_family: V1FQRKKTPreFrozenBranchFamily = field(repr=False, compare=False)
    spectral_evidence: V1FQRBranchEvidenceBinding | None = field(
        repr=False, compare=False
    )
    diagonal_l1_evidence: V1FQRBranchEvidenceBinding | None = field(
        repr=False, compare=False
    )
    selected_branch: V1RelaxedKKTBranch
    selected_endpoint_index: int
    selected_action_hash: str
    selected_packed_action_hash: str
    selected_registered_resource: float
    certified_union_coordinates: tuple[tuple[str, int, str, str, float], ...]
    selection_policy: str = FQR_BRANCH_UNION_SELECTION_POLICY
    proposal_union_selection_authority: bool = True
    cross_branch_action_synthesis_authority: bool = False
    hard_global_optimality_claim: bool = False
    candidate_freeze_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    authority_scope: str = FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
    schema: str = FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA
    _authorization_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.prefrozen_family) is not V1FQRKKTPreFrozenBranchFamily:
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "family authorization requires the exact pre-frozen family",
            )
        evidences = tuple(
            value
            for value in (self.spectral_evidence, self.diagonal_l1_evidence)
            if value is not None
        )
        if any(type(value) is not V1FQRBranchEvidenceBinding for value in evidences):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "family authorization requires exact branch evidence bindings",
            )
        if not isinstance(self.selected_branch, V1RelaxedKKTBranch):
            _refuse(
                V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
                "selected branch must use the registered enum",
            )
        endpoint = _positive_int(
            self.selected_endpoint_index, name="selected endpoint index"
        )
        action_hash = _identity(self.selected_action_hash, name="selected action hash")
        packed_hash = _identity(
            self.selected_packed_action_hash, name="selected packed action hash"
        )
        resource = _finite_nonnegative(
            self.selected_registered_resource, name="selected registered resource"
        )
        coordinates = tuple(self.certified_union_coordinates)
        if not coordinates:
            _refuse(
                V1FQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT,
                "the branch union has no certified endpoint",
            )
        for row in coordinates:
            if len(row) != 5:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "certified union coordinates must have five fields",
                )
            try:
                V1RelaxedKKTBranch(row[0])
            except ValueError:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "certified union contains an unknown branch",
                )
            _positive_int(row[1], name="certified endpoint index")
            _identity(row[2], name="certified action hash")
            _identity(row[3], name="certified packed action hash")
            _finite_nonnegative(row[4], name="certified registered resource")
        if (
            self.selection_policy != FQR_BRANCH_UNION_SELECTION_POLICY
            or self.proposal_union_selection_authority is not True
            or self.cross_branch_action_synthesis_authority is not False
            or self.hard_global_optimality_claim is not False
            or self.candidate_freeze_authority is not False
            or self.calibration_authority is not False
            or self.bce_oracle_authority is not False
            or self.irreversible_decision_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_validation_access_authority is not False
            or self.terminal_authority is not False
            or self.paired_bce_oracle_still_required is not True
            or self.authority_scope != FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
            or self.schema != FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "branch-family selection crossed its proposal-only boundary",
            )
        object.__setattr__(self, "selected_endpoint_index", endpoint)
        object.__setattr__(self, "selected_action_hash", action_hash)
        object.__setattr__(self, "selected_packed_action_hash", packed_hash)
        object.__setattr__(self, "selected_registered_resource", resource)
        object.__setattr__(self, "certified_union_coordinates", coordinates)
        self.assert_integrity()

    @property
    def retained_evidences(self) -> tuple[V1FQRBranchEvidenceBinding, ...]:
        return tuple(
            value
            for value in (self.spectral_evidence, self.diagonal_l1_evidence)
            if value is not None
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prefrozen_family_hash": self.prefrozen_family.family_hash,
            "spectral_evidence_hash": (
                None
                if self.spectral_evidence is None
                else self.spectral_evidence.evidence_hash
            ),
            "diagonal_l1_evidence_hash": (
                None
                if self.diagonal_l1_evidence is None
                else self.diagonal_l1_evidence.evidence_hash
            ),
            "selected_branch": self.selected_branch,
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "selected_packed_action_hash": self.selected_packed_action_hash,
            "selected_registered_resource": self.selected_registered_resource,
            "certified_union_coordinates": self.certified_union_coordinates,
            "selection_policy": self.selection_policy,
            "proposal_union_selection_authority": (
                self.proposal_union_selection_authority
            ),
            "cross_branch_action_synthesis_authority": (
                self.cross_branch_action_synthesis_authority
            ),
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_access_authority": (
                self.terminal_validation_access_authority
            ),
            "terminal_authority": self.terminal_authority,
            "paired_bce_oracle_still_required": self.paired_bce_oracle_still_required,
            "authority_scope": self.authority_scope,
        }

    @property
    def authorization_hash(self) -> str:
        cached = self._authorization_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.selection_policy != FQR_BRANCH_UNION_SELECTION_POLICY
            or self.proposal_union_selection_authority is not True
            or self.cross_branch_action_synthesis_authority is not False
            or self.hard_global_optimality_claim is not False
            or self.candidate_freeze_authority is not False
            or self.calibration_authority is not False
            or self.bce_oracle_authority is not False
            or self.irreversible_decision_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_validation_access_authority is not False
            or self.terminal_authority is not False
            or self.paired_bce_oracle_still_required is not True
            or self.authority_scope != FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "branch-family selection crossed its proposal-only boundary",
            )
        self.prefrozen_family.assert_integrity()
        expected = _certified_union(
            family=self.prefrozen_family,
            spectral_evidence=self.spectral_evidence,
            diagonal_l1_evidence=self.diagonal_l1_evidence,
        )
        winner = _choose_union_winner(expected)
        if (
            expected != self.certified_union_coordinates
            or (
                self.selected_branch.value,
                self.selected_endpoint_index,
                self.selected_action_hash,
                self.selected_packed_action_hash,
                self.selected_registered_resource,
            )
            != winner
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.POST_SELECTION,
                "family selection differs from the immutable certified union",
            )
        observed = content_hash(self._payload(), prefix=_AUTHORIZATION_PREFIX)
        cached = self._authorization_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch-family authorization was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "branch-family authorization hash does not match",
            )
        object.__setattr__(self, "_authorization_hash", observed)
        return observed


def _certified_union(
    *,
    family: V1FQRKKTPreFrozenBranchFamily,
    spectral_evidence: V1FQRBranchEvidenceBinding | None,
    diagonal_l1_evidence: V1FQRBranchEvidenceBinding | None,
) -> tuple[tuple[str, int, str, str, float], ...]:
    evidence_by_branch = {
        V1RelaxedKKTBranch.SPECTRAL: spectral_evidence,
        V1RelaxedKKTBranch.DIAGONAL_L1: diagonal_l1_evidence,
    }
    rows: list[tuple[str, int, str, str, float]] = []
    for branch in (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        frozen = (
            family.spectral
            if branch is V1RelaxedKKTBranch.SPECTRAL
            else family.diagonal_l1
        )
        evidence = evidence_by_branch[branch]
        if frozen is None:
            if evidence is not None:
                _refuse(
                    V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "evidence was supplied for a pre-C dominated branch",
                )
            continue
        if type(evidence) is not V1FQRBranchEvidenceBinding:
            _refuse(
                V1FQRBranchFamilyRefusalCode.MISSING_BRANCH,
                f"{branch.value} lacks its complete F/Q/R evidence graph",
            )
        evidence.assert_integrity()
        if (
            evidence.branch is not branch
            or evidence.branch_binding.binding_hash != frozen.binding_hash
        ):
            _refuse(
                V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "branch evidence differs from the pre-C family binding",
            )
        for entry in evidence.candidate_queue_authorization.queue_entries:
            endpoint = entry.endpoint_index
            if endpoint > len(frozen.endpoint_registered_resources):
                _refuse(
                    V1FQRBranchFamilyRefusalCode.POST_SELECTION,
                    "certified endpoint left the pre-frozen branch axis",
                )
            rows.append(
                (
                    branch.value,
                    endpoint,
                    entry.frozen_action_hash,
                    entry.packed_action_hash,
                    frozen.endpoint_registered_resources[endpoint - 1],
                )
            )
    if not rows:
        _refuse(
            V1FQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT,
            "no retained branch contains an independently certified endpoint",
        )
    return tuple(rows)


def _choose_union_winner(
    rows: tuple[tuple[str, int, str, str, float], ...]
) -> tuple[str, int, str, str, float]:
    # The branch tie-break is the stable enum value, not the relaxed winner.
    return min(
        rows,
        key=lambda row: (-row[4], row[0], row[3], row[1]),
    )


def authorize_v1_fqr_branch_family(
    *,
    prefrozen_family: V1FQRKKTPreFrozenBranchFamily,
    spectral_evidence: V1FQRBranchEvidenceBinding | None,
    diagonal_l1_evidence: V1FQRBranchEvidenceBinding | None,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_proposal_generation: int,
    live_family_hash: str,
) -> V1FQRBranchFamilyAuthorization:
    """Choose the greatest-resource endpoint in the certified hard union."""

    if type(prefrozen_family) is not V1FQRKKTPreFrozenBranchFamily:
        _refuse(
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE,
            "family authorization requires the exact pre-frozen family",
        )
    family_hash = prefrozen_family.assert_integrity()
    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    live_mask = _identity(live_current_mask_hash, name="live current-mask hash")
    live_generation = _nonnegative_int(
        live_proposal_generation, name="live proposal generation"
    )
    live_family = _identity(live_family_hash, name="live family hash")
    if (
        prefrozen_family.chart_identity_hash != live_chart
        or prefrozen_family.current_mask_hash != live_mask
        or prefrozen_family.proposal_generation != live_generation
        or family_hash != live_family
    ):
        _refuse(
            V1FQRBranchFamilyRefusalCode.STALE_CHART,
            "branch family does not belong to the live chart/mask/generation",
        )
    rows = _certified_union(
        family=prefrozen_family,
        spectral_evidence=spectral_evidence,
        diagonal_l1_evidence=diagonal_l1_evidence,
    )
    winner = _choose_union_winner(rows)
    return V1FQRBranchFamilyAuthorization(
        prefrozen_family=prefrozen_family,
        spectral_evidence=spectral_evidence,
        diagonal_l1_evidence=diagonal_l1_evidence,
        selected_branch=V1RelaxedKKTBranch(winner[0]),
        selected_endpoint_index=winner[1],
        selected_action_hash=winner[2],
        selected_packed_action_hash=winner[3],
        selected_registered_resource=winner[4],
        certified_union_coordinates=rows,
    )


__all__ = [
    "FQR_BRANCH_EVIDENCE_GRAPH_KIND",
    "FQR_BRANCH_FAMILY_AUTHORITY_SCOPE",
    "FQR_BRANCH_UNION_SELECTION_POLICY",
    "FQR_DEVICE_SHELL_BRANCH_BINDING_SCHEMA",
    "FQR_DEVICE_SHELL_BRANCH_FAMILY_SCHEMA",
    "FQR_DEVICE_SHELL_CROSS_BRANCH_ACTION_SCHEMA",
    "FQR_DEVICE_SHELL_ENDPOINT_IDENTITY_SCHEMA",
    "FQR_DEVICE_SHELL_IDENTITY_SCOPE",
    "FQR_DEVICE_SHELL_SOURCE_STABILITY_BINDING_SCHEMA",
    "FQR_KKT_BRANCH_FAMILY_SCOPE",
    "FQR_PRE_C_DOMINANCE_METHOD",
    "V1FQRBranchEvidenceBinding",
    "V1FQRBranchFamilyAuthorization",
    "V1FQRBranchFamilyRefusal",
    "V1FQRBranchFamilyRefusalCode",
    "V1FQRDeviceShellBranchBinding",
    "V1FQRDeviceShellCrossBranchActionIdentity",
    "V1FQRDeviceShellEndpointIdentity",
    "V1FQRDeviceShellPreFrozenBranchFamily",
    "V1FQRDeviceShellSourceStabilityBinding",
    "V1FQRKKTBranchFrozenBinding",
    "V1FQRKKTPreFrozenBranchFamily",
    "V1FQRPreCDominanceCertificate",
    "authorize_v1_fqr_branch_family",
    "bind_v1_fqr_branch_evidence",
    "bind_v1_fqr_device_shell_source_stability",
    "certify_v1_pre_c_exact_action_dominance",
    "freeze_v1_fqr_device_shell_branch_family",
    "freeze_v1_relaxed_kkt_branch_family",
    "omit_v1_pre_c_dominated_branch",
]
