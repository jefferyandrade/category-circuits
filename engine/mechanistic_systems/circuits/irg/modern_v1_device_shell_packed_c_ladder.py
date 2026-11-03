"""Bounded materialization of frozen device-shell endpoints for C measurement.

The device-shell KKT runtime keeps one uint8 first-inclusion label per route on
the device.  Independent directional C measurement, however, consumes the
historical packed :class:`V1FrozenCandidateLadder` contract.  This module is
the narrow identity bridge between those representations.

It deliberately does *not* reconstruct a route order, claim hard KKT
optimality, observe C outcomes, or grant F/Q/R, calibration, oracle, commit,
terminal, or Pareto authority.  Every endpoint was already frozen upstream;
this seam only packs those exact cumulative endpoint masks and replays their
precommitted mask, delta, and action identities.

The production path accepts only a CUDA uint8 label lease registered for
``C_DIRECTIONAL_CERTIFICATION``.  It tiles both routes and endpoint lanes and
copies only packed child-mask tensor payloads to the host.  A NumPy reference
path is available only under an explicit small-domain test registration.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

import hashlib
from dataclasses import InitVar, dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelLease,
)
from .modern_v1_fqr_branch_family import (
    V1FQRDeviceShellBranchBinding,
    V1FQRDeviceShellPreFrozenBranchFamily,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_kkt_device_shell_bridge import (
    V1SampledKKTDeviceShellProposalFreezeForC,
)


DEVICE_SHELL_PACKED_C_COORDINATES_SCHEMA: Final = (
    "partition_free_v1_device_shell_packed_C_ladder_coordinates_v1"
)
DEVICE_SHELL_PACKED_C_WORKSPACE_POLICY_SCHEMA: Final = (
    "partition_free_v1_device_shell_packed_C_workspace_policy_v1"
)
DEVICE_SHELL_PACKED_C_MATERIALIZATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_shell_packed_C_materialization_receipt_v1"
)
DEVICE_SHELL_PACKED_C_MATERIALIZATION_RESULT_SCHEMA: Final = (
    "partition_free_v1_device_shell_packed_C_materialization_result_v1"
)
DEVICE_SHELL_PACKED_C_AUTHORITY_SCOPE: Final = (
    "already_frozen_endpoint_representation_identity_for_independent_C_only_"
    "no_route_order_KKT_hard_optimality_C_outcome_FQR_K_O_calibration_oracle_"
    "commit_terminal_or_pareto_authority_v1"
)
DEVICE_SHELL_PACKED_C_DEFAULT_ROUTE_TILE_SIZE: Final = 1 << 20
DEVICE_SHELL_PACKED_C_DEFAULT_ENDPOINT_LANE_TILE_SIZE: Final = 8
DEVICE_SHELL_PACKED_C_DEFAULT_NUMPY_REFERENCE_ROUTE_LIMIT: Final = 1 << 16

_COORDINATES_PREFIX: Final = "partition-free-v1-device-shell-packed-C-coordinates-"
_WORKSPACE_POLICY_PREFIX: Final = (
    "partition-free-v1-device-shell-packed-C-workspace-policy-"
)
_MATERIALIZATION_RECEIPT_PREFIX: Final = (
    "partition-free-v1-device-shell-packed-C-materialization-receipt-"
)
_MATERIALIZATION_RESULT_PREFIX: Final = (
    "partition-free-v1-device-shell-packed-C-materialization-result-"
)
_LADDER_FREEZE_PREFIX: Final = (
    "partition-free-v1-device-shell-packed-C-ladder-freeze-"
)
_DISCOVERY_PREFIX: Final = "partition-free-v1-fqr-kkt-discovery-prefix-"
_WORKSPACE_POLICY_ISSUANCE_TOKEN: Final = object()


class V1DeviceShellPackedCRefusalCode(str, Enum):
    """Typed fail-closed outcomes for packed C-ladder materialization."""

    INVALID_TYPE = "invalid_type"
    OPERATION_MISMATCH = "operation_mismatch"
    SOURCE_MISMATCH = "source_mismatch"
    STALE_COORDINATES = "stale_coordinates"
    WORKSPACE_FORECAST_OVERFLOW = "workspace_forecast_overflow"
    PRODUCTION_CUDA_REQUIRED = "production_cuda_required"
    NUMPY_REFERENCE_LIMIT_EXCEEDED = "numpy_reference_limit_exceeded"
    UNSUPPORTED_CUDA = "unsupported_cuda"
    INVALID_LABEL = "invalid_label"
    PARENT_ROUTE_NOT_LIVE = "parent_route_not_live"
    LABEL_COMMITMENT_MISMATCH = "label_commitment_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1DeviceShellPackedCRefusal(RuntimeError):
    """An exact device-shell endpoint family could not be materialized."""

    def __init__(
        self,
        code: V1DeviceShellPackedCRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1DeviceShellPackedCRefusalCode, message: str) -> NoReturn:
    raise V1DeviceShellPackedCRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _direct_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        raise TypeError(f"{name} must use the exact HardMask type")
    replay = HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=mask.packed_bits,
    )
    return replay.mask_hash


def _new_label_digest(
    *, route_axis_hash: str, route_count: int, endpoint_count: int
) -> "hashlib._Hash":
    domain = DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN.encode("ascii")
    metadata = canonical_json(
        {
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "endpoint_count": endpoint_count,
        }
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(8, "big", signed=False))
    digest.update(domain)
    digest.update(len(metadata).to_bytes(8, "big", signed=False))
    digest.update(metadata)
    return digest


def _discovery_prefix_hash(
    *, context_ids: tuple[str, ...], draw_ids: tuple[str, ...]
) -> str:
    return content_hash(
        {"context_ids": context_ids, "draw_ids": draw_ids},
        prefix=_DISCOVERY_PREFIX,
    )


def _cuda_logical_workspace_forecast(
    *, route_tile_size: int, endpoint_lane_tile_size: int
) -> int:
    """Conservative bound for explicit tensors in one CUDA packing tile.

    This is a logical tensor-payload bound, not a claim about allocator cache
    residency.  The deliberately loose coefficient covers padded labels,
    unpacked parent bits, comparison temporaries, int16 packing temporaries,
    packed parent/child lanes, endpoint labels, and fixed bit weights.
    """

    routes = _positive_int(route_tile_size, name="route tile size")
    lanes = _positive_int(
        endpoint_lane_tile_size, name="endpoint lane tile size"
    )
    packed = (routes + 7) // 8
    return int(16 * lanes * routes + 8 * lanes * packed + 4 * routes + 4096)


@dataclass(frozen=True, slots=True)
class V1DeviceShellPackedCLadderCoordinates:
    """Missing canonical ladder coordinates registered before C outcomes."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    discovery_context_ids: tuple[str, ...]
    discovery_draw_ids: tuple[str, ...]
    registered_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    schema: str = DEVICE_SHELL_PACKED_C_COORDINATES_SCHEMA
    _coordinates_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
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
                "packed C-ladder discovery ledgers must be aligned, nonempty, "
                "and unique"
            )
        if (
            type(self.registered_before_C_outcomes) is not bool
            or type(self.C_outcomes_seen) is not bool
            or not self.registered_before_C_outcomes
            or self.C_outcomes_seen
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
                "packed C-ladder coordinates must freeze before C outcomes",
            )
        if self.schema != DEVICE_SHELL_PACKED_C_COORDINATES_SCHEMA:
            raise ValueError("packed C-ladder coordinate schema is invalid")
        object.__setattr__(self, "discovery_context_ids", contexts)
        object.__setattr__(self, "discovery_draw_ids", draws)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "discovery_context_ids": self.discovery_context_ids,
            "discovery_draw_ids": self.discovery_draw_ids,
            "registered_before_C_outcomes": self.registered_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
        }

    @property
    def coordinates_hash(self) -> str:
        cached = self._coordinates_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if not self.registered_before_C_outcomes or self.C_outcomes_seen:
            _refuse(
                V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
                "packed C-ladder coordinates were opened after C outcomes",
            )
        observed = content_hash(self._payload(), prefix=_COORDINATES_PREFIX)
        cached = self._coordinates_hash
        if cached is not None and cached != observed:
            raise ValueError("packed C-ladder coordinates were tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("packed C-ladder coordinate hash does not match")
        object.__setattr__(self, "_coordinates_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DeviceShellPackedCWorkspacePolicy:
    """Pre-C bounded workspace registration for one frozen branch axis."""

    policy_id: str
    source_branch_binding_hash: str
    source_proposal_freeze_hash: str
    route_count: int
    endpoint_count: int
    route_tile_size: int
    endpoint_lane_tile_size: int
    retained_host_packed_bytes: int
    peak_host_packed_payload_bytes: int
    cuda_logical_workspace_forecast_bytes: int
    maximum_host_packed_payload_bytes: int
    maximum_cuda_logical_workspace_bytes: int
    numpy_reference_route_limit: int
    numpy_reference_permitted_for_tests: bool
    _issuance_token: InitVar[object]
    registered_before_C_outcomes: bool = True
    empirical_C_outcomes_used: bool = False
    production_requires_cuda: bool = True
    numpy_reference_has_production_authority: bool = False
    C_outcome_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_SHELL_PACKED_C_WORKSPACE_POLICY_SCHEMA
    _policy_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _WORKSPACE_POLICY_ISSUANCE_TOKEN:
            raise PermissionError(
                "packed C-ladder workspace policy requires its registered issuer"
            )
        for name in (
            "policy_id",
            "source_branch_binding_hash",
            "source_proposal_freeze_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        integer_fields = (
            "route_count",
            "endpoint_count",
            "route_tile_size",
            "endpoint_lane_tile_size",
            "retained_host_packed_bytes",
            "peak_host_packed_payload_bytes",
            "cuda_logical_workspace_forecast_bytes",
            "maximum_host_packed_payload_bytes",
            "maximum_cuda_logical_workspace_bytes",
            "numpy_reference_route_limit",
        )
        for name in integer_fields:
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if self.route_tile_size % 8:
            raise ValueError("packed C-ladder route tile size must be byte aligned")
        if self.endpoint_lane_tile_size > self.endpoint_count:
            raise ValueError("endpoint lane tile exceeds the frozen endpoint axis")
        if (
            self.numpy_reference_route_limit
            > DEVICE_SHELL_PACKED_C_DEFAULT_NUMPY_REFERENCE_ROUTE_LIMIT
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.NUMPY_REFERENCE_LIMIT_EXCEEDED,
                "NumPy reference registration cannot expand the fixed small-domain cap",
            )
        packed_bytes = (self.route_count + 7) // 8
        retained = self.endpoint_count * packed_bytes
        peak_host = 2 * retained + self.endpoint_lane_tile_size * (
            (self.route_tile_size + 7) // 8
        )
        device = _cuda_logical_workspace_forecast(
            route_tile_size=self.route_tile_size,
            endpoint_lane_tile_size=self.endpoint_lane_tile_size,
        )
        if (
            self.retained_host_packed_bytes != retained
            or self.peak_host_packed_payload_bytes != peak_host
            or self.cuda_logical_workspace_forecast_bytes != device
        ):
            raise ValueError("packed C-ladder workspace forecast does not replay")
        if (
            peak_host > self.maximum_host_packed_payload_bytes
            or device > self.maximum_cuda_logical_workspace_bytes
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.WORKSPACE_FORECAST_OVERFLOW,
                "registered packed C-ladder workspace exceeds its declared cap",
            )
        flags = (
            self.numpy_reference_permitted_for_tests,
            self.registered_before_C_outcomes,
            self.empirical_C_outcomes_used,
            self.production_requires_cuda,
            self.numpy_reference_has_production_authority,
            self.C_outcome_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("packed C-ladder workspace flags must be exact booleans")
        if flags[1:] != (
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
                "packed C-ladder workspace policy crossed its authority boundary",
            )
        if self.schema != DEVICE_SHELL_PACKED_C_WORKSPACE_POLICY_SCHEMA:
            raise ValueError("packed C-ladder workspace policy schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "source_branch_binding_hash": self.source_branch_binding_hash,
            "source_proposal_freeze_hash": self.source_proposal_freeze_hash,
            "route_count": self.route_count,
            "endpoint_count": self.endpoint_count,
            "route_tile_size": self.route_tile_size,
            "endpoint_lane_tile_size": self.endpoint_lane_tile_size,
            "retained_host_packed_bytes": self.retained_host_packed_bytes,
            "peak_host_packed_payload_bytes": (
                self.peak_host_packed_payload_bytes
            ),
            "cuda_logical_workspace_forecast_bytes": (
                self.cuda_logical_workspace_forecast_bytes
            ),
            "maximum_host_packed_payload_bytes": (
                self.maximum_host_packed_payload_bytes
            ),
            "maximum_cuda_logical_workspace_bytes": (
                self.maximum_cuda_logical_workspace_bytes
            ),
            "numpy_reference_route_limit": self.numpy_reference_route_limit,
            "numpy_reference_permitted_for_tests": (
                self.numpy_reference_permitted_for_tests
            ),
            "registered_before_C_outcomes": self.registered_before_C_outcomes,
            "empirical_C_outcomes_used": self.empirical_C_outcomes_used,
            "production_requires_cuda": self.production_requires_cuda,
            "numpy_reference_has_production_authority": (
                self.numpy_reference_has_production_authority
            ),
            "C_outcome_authority": self.C_outcome_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_WORKSPACE_POLICY_PREFIX)
        cached = self._policy_hash
        if cached is not None and cached != observed:
            raise ValueError("packed C-ladder workspace policy was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("packed C-ladder workspace policy hash does not match")
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def _branch_from_proposal_freeze(
    *,
    proposal_freeze: V1SampledKKTDeviceShellProposalFreezeForC,
    branch: V1RelaxedKKTBranch,
) -> V1FQRDeviceShellBranchBinding:
    if type(proposal_freeze) is not V1SampledKKTDeviceShellProposalFreezeForC:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C-ladder materialization requires the exact proposal freeze",
        )
    proposal_freeze.assert_integrity()
    if (
        not proposal_freeze.C_direction_borrow_authority
        or not proposal_freeze.independent_C_FQR_required
        or proposal_freeze.C_FQR_decision_authority
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
            "proposal freeze cannot authorize independent C direction packing",
        )
    family = proposal_freeze.graph.branch_family
    if type(family) is not V1FQRDeviceShellPreFrozenBranchFamily:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "proposal freeze lost its exact device-shell family",
        )
    return (
        family.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else family.diagonal_l1
    )


def register_v1_device_shell_packed_c_workspace_policy(
    *,
    policy_id: str,
    proposal_freeze: V1SampledKKTDeviceShellProposalFreezeForC,
    branch_binding: V1FQRDeviceShellBranchBinding,
    route_tile_size: int = DEVICE_SHELL_PACKED_C_DEFAULT_ROUTE_TILE_SIZE,
    endpoint_lane_tile_size: int = (
        DEVICE_SHELL_PACKED_C_DEFAULT_ENDPOINT_LANE_TILE_SIZE
    ),
    maximum_host_packed_payload_bytes: int,
    maximum_cuda_logical_workspace_bytes: int,
    numpy_reference_route_limit: int = (
        DEVICE_SHELL_PACKED_C_DEFAULT_NUMPY_REFERENCE_ROUTE_LIMIT
    ),
    numpy_reference_permitted_for_tests: bool = False,
) -> V1DeviceShellPackedCWorkspacePolicy:
    """Register deterministic workspace bounds before independent C outcomes."""

    if type(branch_binding) is not V1FQRDeviceShellBranchBinding:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "workspace registration requires an exact device-shell branch binding",
        )
    if type(branch_binding.branch) is not V1RelaxedKKTBranch:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "branch binding has an invalid branch label",
        )
    branch_binding.assert_integrity()
    expected = _branch_from_proposal_freeze(
        proposal_freeze=proposal_freeze,
        branch=branch_binding.branch,
    )
    if expected.binding_hash != branch_binding.binding_hash:
        _refuse(
            V1DeviceShellPackedCRefusalCode.SOURCE_MISMATCH,
            "branch binding belongs to another proposal freeze",
        )
    routes = _positive_int(route_tile_size, name="route tile size")
    if routes % 8:
        raise ValueError("packed C-ladder route tile size must be byte aligned")
    lanes = min(
        _positive_int(endpoint_lane_tile_size, name="endpoint lane tile size"),
        len(branch_binding.endpoints),
    )
    route_count = branch_binding.route_count
    endpoint_count = len(branch_binding.endpoints)
    retained = endpoint_count * ((route_count + 7) // 8)
    peak_host = 2 * retained + lanes * ((routes + 7) // 8)
    cuda_workspace = _cuda_logical_workspace_forecast(
        route_tile_size=routes,
        endpoint_lane_tile_size=lanes,
    )
    maximum_host = _positive_int(
        maximum_host_packed_payload_bytes,
        name="maximum host packed payload bytes",
    )
    maximum_cuda = _positive_int(
        maximum_cuda_logical_workspace_bytes,
        name="maximum CUDA logical workspace bytes",
    )
    if peak_host > maximum_host or cuda_workspace > maximum_cuda:
        _refuse(
            V1DeviceShellPackedCRefusalCode.WORKSPACE_FORECAST_OVERFLOW,
            "pre-C packed-ladder forecast exceeds a registered workspace cap",
        )
    if type(numpy_reference_permitted_for_tests) is not bool:
        raise TypeError("numpy_reference_permitted_for_tests must be an exact bool")
    return V1DeviceShellPackedCWorkspacePolicy(
        policy_id=_identity(policy_id, name="workspace policy id"),
        source_branch_binding_hash=branch_binding.binding_hash,
        source_proposal_freeze_hash=proposal_freeze.certificate_hash,
        route_count=route_count,
        endpoint_count=endpoint_count,
        route_tile_size=routes,
        endpoint_lane_tile_size=lanes,
        retained_host_packed_bytes=retained,
        peak_host_packed_payload_bytes=peak_host,
        cuda_logical_workspace_forecast_bytes=cuda_workspace,
        maximum_host_packed_payload_bytes=maximum_host,
        maximum_cuda_logical_workspace_bytes=maximum_cuda,
        numpy_reference_route_limit=_positive_int(
            numpy_reference_route_limit,
            name="NumPy reference route limit",
        ),
        numpy_reference_permitted_for_tests=numpy_reference_permitted_for_tests,
        _issuance_token=_WORKSPACE_POLICY_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1DeviceShellPackedCMaterializationReceipt:
    """Compact identity-only receipt for one materialized packed C ladder."""

    branch: V1RelaxedKKTBranch
    source_proposal_freeze_hash: str
    source_candidate_family_hash: str
    source_branch_binding_hash: str
    source_ladder_receipt_hash: str
    source_label_commitment: str
    workspace_policy_hash: str
    ladder_coordinates_hash: str
    backend_kind: str
    device_identity: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    endpoint_count: int
    endpoint_source_identity_hashes: tuple[str, ...]
    endpoint_resulting_mask_hashes: tuple[str, ...]
    endpoint_removed_route_index_hashes: tuple[str, ...]
    endpoint_packed_action_hashes: tuple[str, ...]
    frozen_action_hashes: tuple[str, ...]
    frozen_ladder_hash: str
    retained_host_packed_bytes: int
    peak_host_packed_payload_forecast_bytes: int
    cuda_logical_workspace_forecast_bytes: int
    packed_child_tensor_d2h_bytes: int
    raw_label_tensor_d2h_bytes: int = 0
    cuda_scalar_control_transfer_count: int = 0
    route_order_materialized: bool = False
    KKT_hard_optimality_claim: bool = False
    C_outcomes_seen: bool = False
    C_FQR_decision_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = DEVICE_SHELL_PACKED_C_AUTHORITY_SCOPE
    schema: str = DEVICE_SHELL_PACKED_C_MATERIALIZATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("packed C materialization requires an exact branch")
        for name in (
            "source_proposal_freeze_hash",
            "source_candidate_family_hash",
            "source_branch_binding_hash",
            "source_ladder_receipt_hash",
            "source_label_commitment",
            "workspace_policy_hash",
            "ladder_coordinates_hash",
            "backend_kind",
            "device_identity",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "frozen_ladder_hash",
            "authority_scope",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        for name in (
            "route_count",
            "endpoint_count",
            "retained_host_packed_bytes",
            "peak_host_packed_payload_forecast_bytes",
            "cuda_logical_workspace_forecast_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if isinstance(self.proposal_generation, bool) or not isinstance(
            self.proposal_generation, Integral
        ):
            raise TypeError("proposal generation must be one exact integer")
        generation = int(self.proposal_generation)
        if generation < 0:
            raise ValueError("proposal generation must be nonnegative")
        object.__setattr__(self, "proposal_generation", generation)
        if isinstance(self.packed_child_tensor_d2h_bytes, bool) or not isinstance(
            self.packed_child_tensor_d2h_bytes, Integral
        ):
            raise TypeError("packed child D2H bytes must be one exact integer")
        packed_d2h = int(self.packed_child_tensor_d2h_bytes)
        if isinstance(self.raw_label_tensor_d2h_bytes, bool) or not isinstance(
            self.raw_label_tensor_d2h_bytes, Integral
        ):
            raise TypeError("raw label D2H bytes must be one exact integer")
        raw_label_d2h = int(self.raw_label_tensor_d2h_bytes)
        if isinstance(self.cuda_scalar_control_transfer_count, bool) or not isinstance(
            self.cuda_scalar_control_transfer_count, Integral
        ):
            raise TypeError("CUDA scalar control transfer count must be exact")
        scalar_controls = int(self.cuda_scalar_control_transfer_count)
        if packed_d2h < 0 or raw_label_d2h != 0 or scalar_controls < 0:
            _refuse(
                V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
                "packed C materialization may transfer no raw label tensor payload",
            )
        object.__setattr__(self, "packed_child_tensor_d2h_bytes", packed_d2h)
        object.__setattr__(self, "raw_label_tensor_d2h_bytes", raw_label_d2h)
        object.__setattr__(
            self, "cuda_scalar_control_transfer_count", scalar_controls
        )
        if self.backend_kind not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("packed C materialization backend kind is invalid")
        if (
            self.backend_kind == "numpy_reference" and packed_d2h != 0
        ) or (
            self.backend_kind == "torch_cuda"
            and packed_d2h != self.retained_host_packed_bytes
        ):
            raise ValueError(
                "packed C materialization backend and D2H accounting differ"
            )
        if (
            self.backend_kind == "numpy_reference" and scalar_controls != 0
        ) or (
            self.backend_kind == "torch_cuda" and scalar_controls <= 0
        ):
            raise ValueError(
                "packed C materialization backend and scalar controls differ"
            )
        expected_retained = self.endpoint_count * ((self.route_count + 7) // 8)
        if (
            self.retained_host_packed_bytes != expected_retained
            or self.peak_host_packed_payload_forecast_bytes
            < 2 * expected_retained
        ):
            raise ValueError(
                "packed C materialization host payload accounting does not replay"
            )
        axes = tuple(
            tuple(_identity(value, name=name) for value in getattr(self, name))
            for name in (
                "endpoint_source_identity_hashes",
                "endpoint_resulting_mask_hashes",
                "endpoint_removed_route_index_hashes",
                "endpoint_packed_action_hashes",
                "frozen_action_hashes",
            )
        )
        if any(len(axis) != self.endpoint_count for axis in axes):
            raise ValueError("packed C materialization endpoint axes do not align")
        if any(len(set(axis)) != len(axis) for axis in axes):
            raise ValueError("packed C materialization endpoint identities repeat")
        for name, axis in zip(
            (
                "endpoint_source_identity_hashes",
                "endpoint_resulting_mask_hashes",
                "endpoint_removed_route_index_hashes",
                "endpoint_packed_action_hashes",
                "frozen_action_hashes",
            ),
            axes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        flags = (
            self.route_order_materialized,
            self.KKT_hard_optimality_claim,
            self.C_outcomes_seen,
            self.C_FQR_decision_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            _refuse(
                V1DeviceShellPackedCRefusalCode.AUTHORITY_VIOLATION,
                "packed C materialization crossed its identity-only boundary",
            )
        if (
            self.authority_scope != DEVICE_SHELL_PACKED_C_AUTHORITY_SCOPE
            or self.schema != DEVICE_SHELL_PACKED_C_MATERIALIZATION_RECEIPT_SCHEMA
        ):
            raise ValueError("packed C materialization receipt schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "branch",
                "source_proposal_freeze_hash",
                "source_candidate_family_hash",
                "source_branch_binding_hash",
                "source_ladder_receipt_hash",
                "source_label_commitment",
                "workspace_policy_hash",
                "ladder_coordinates_hash",
                "backend_kind",
                "device_identity",
                "route_axis_hash",
                "route_count",
                "current_mask_hash",
                "chart_identity_hash",
                "proposal_generation",
                "discovery_prefix_hash",
                "discovery_evidence_hash",
                "endpoint_count",
                "endpoint_source_identity_hashes",
                "endpoint_resulting_mask_hashes",
                "endpoint_removed_route_index_hashes",
                "endpoint_packed_action_hashes",
                "frozen_action_hashes",
                "frozen_ladder_hash",
                "retained_host_packed_bytes",
                "peak_host_packed_payload_forecast_bytes",
                "cuda_logical_workspace_forecast_bytes",
                "packed_child_tensor_d2h_bytes",
                "raw_label_tensor_d2h_bytes",
                "cuda_scalar_control_transfer_count",
                "route_order_materialized",
                "KKT_hard_optimality_claim",
                "C_outcomes_seen",
                "C_FQR_decision_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_MATERIALIZATION_RECEIPT_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("packed C materialization receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("packed C materialization receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DeviceShellPackedCMaterializationResult:
    """One exact packed ladder plus its authority-inert representation receipt."""

    ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    receipt: V1DeviceShellPackedCMaterializationReceipt
    schema: str = DEVICE_SHELL_PACKED_C_MATERIALIZATION_RESULT_SCHEMA
    _result_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.ladder) is not V1FrozenCandidateLadder or type(
            self.receipt
        ) is not V1DeviceShellPackedCMaterializationReceipt:
            raise TypeError("packed C result requires exact ladder and receipt types")
        if self.schema != DEVICE_SHELL_PACKED_C_MATERIALIZATION_RESULT_SCHEMA:
            raise ValueError("packed C materialization result schema is invalid")
        self.assert_integrity()

    @property
    def result_hash(self) -> str:
        cached = self._result_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        ladder_hash = self.ladder.assert_integrity()
        receipt_hash = self.receipt.assert_integrity()
        if (
            ladder_hash != self.receipt.frozen_ladder_hash
            or self.ladder.route_axis_hash != self.receipt.route_axis_hash
            or self.ladder.route_count != self.receipt.route_count
            or self.ladder.current_mask_hash != self.receipt.current_mask_hash
            or self.ladder.proposal_generation != self.receipt.proposal_generation
            or self.ladder.action_hashes != self.receipt.frozen_action_hashes
            or tuple(
                action.packed_removal_action.action_hash  # type: ignore[union-attr]
                for action in self.ladder.actions
            )
            != self.receipt.endpoint_packed_action_hashes
            or tuple(
                action.packed_removal_action.resulting_mask.mask_hash  # type: ignore[union-attr]
                for action in self.ladder.actions
            )
            != self.receipt.endpoint_resulting_mask_hashes
            or tuple(
                action.packed_removal_action.removed_route_index_hash  # type: ignore[union-attr]
                for action in self.ladder.actions
            )
            != self.receipt.endpoint_removed_route_index_hashes
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
                "packed C materialization result differs from its ladder receipt",
            )
        observed = content_hash(
            {
                "schema": self.schema,
                "frozen_ladder_hash": ladder_hash,
                "materialization_receipt_hash": receipt_hash,
            },
            prefix=_MATERIALIZATION_RESULT_PREFIX,
        )
        cached = self._result_hash
        if cached is not None and cached != observed:
            raise ValueError("packed C materialization result was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("packed C materialization result hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed


def _validate_source_ladder_receipt(
    *,
    source_ladder_receipt: V1DeviceShellFrozenLadderReceipt | None,
    branch_binding: V1FQRDeviceShellBranchBinding,
) -> None:
    if source_ladder_receipt is None:
        return
    if type(source_ladder_receipt) is not V1DeviceShellFrozenLadderReceipt:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "source ladder receipt must use the exact device-shell type",
        )
    actions = source_ladder_receipt.actions
    endpoints = branch_binding.endpoints
    if (
        source_ladder_receipt.receipt_hash
        != branch_binding.source_ladder_receipt_hash
        or source_ladder_receipt.route_axis_hash != branch_binding.route_axis_hash
        or source_ladder_receipt.route_count != branch_binding.route_count
        or source_ladder_receipt.parent_mask_hash
        != branch_binding.current_mask_hash
        or source_ladder_receipt.label_commitment
        != branch_binding.label_commitment
        or len(actions) != len(endpoints)
        or tuple(action.receipt_hash for action in actions)
        != tuple(endpoint.source_action_receipt_hash for endpoint in endpoints)
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.SOURCE_MISMATCH,
            "source ladder receipt differs from the frozen branch binding",
        )


def _pack_numpy_reference(
    *,
    labels: object,
    parent_mask: HardMask,
    endpoint_count: int,
    policy: V1DeviceShellPackedCWorkspacePolicy,
) -> tuple[tuple[bytes, ...], int, int]:
    if not policy.numpy_reference_permitted_for_tests:
        _refuse(
            V1DeviceShellPackedCRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "production packed C-ladder materialization requires CUDA labels",
        )
    if parent_mask.route_count > policy.numpy_reference_route_limit:
        _refuse(
            V1DeviceShellPackedCRefusalCode.NUMPY_REFERENCE_LIMIT_EXCEEDED,
            "NumPy packed C-ladder reference exceeded its registered route limit",
        )
    array = np.asarray(labels)
    if (
        array.dtype != np.dtype(np.uint8)
        or array.ndim != 1
        or int(array.size) != parent_mask.route_count
        or not array.flags.c_contiguous
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "NumPy reference labels must be one contiguous uint8 route axis",
        )
    outputs = [bytearray(len(parent_mask.packed_bits)) for _ in range(endpoint_count)]
    parent_bytes = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
    tile_size = policy.route_tile_size
    lanes = policy.endpoint_lane_tile_size
    for start in range(0, parent_mask.route_count, tile_size):
        stop = min(parent_mask.route_count, start + tile_size)
        label_tile = array[start:stop]
        if label_tile.size and int(np.max(label_tile)) > endpoint_count:
            _refuse(
                V1DeviceShellPackedCRefusalCode.INVALID_LABEL,
                "a shell label lies outside the frozen endpoint axis",
            )
        byte_start = start // 8
        byte_stop = byte_start + (stop - start + 7) // 8
        parent_tile = parent_bytes[byte_start:byte_stop]
        parent_live = np.unpackbits(
            parent_tile, bitorder="big", count=stop - start
        ).astype(bool, copy=False)
        if np.any((label_tile != 0) & ~parent_live):
            _refuse(
                V1DeviceShellPackedCRefusalCode.PARENT_ROUTE_NOT_LIVE,
                "a shell label attempts to remove a route absent from its parent",
            )
        for lane_start in range(1, endpoint_count + 1, lanes):
            lane_stop = min(endpoint_count + 1, lane_start + lanes)
            endpoint_labels = np.arange(
                lane_start, lane_stop, dtype=np.int16
            ).reshape(-1, 1)
            included = (label_tile.reshape(1, -1) != 0) & (
                label_tile.reshape(1, -1) <= endpoint_labels
            )
            clear = np.packbits(included, axis=1, bitorder="big")
            children = np.bitwise_and(parent_tile.reshape(1, -1), np.bitwise_not(clear))
            for lane_offset, endpoint in enumerate(
                range(lane_start - 1, lane_stop - 1)
            ):
                outputs[endpoint][byte_start:byte_stop] = children[
                    lane_offset
                ].tobytes()
    return tuple(bytes(value) for value in outputs), 0, 0


def _pack_torch_cuda(
    *,
    labels: object,
    parent_mask: HardMask,
    endpoint_count: int,
    policy: V1DeviceShellPackedCWorkspacePolicy,
) -> tuple[tuple[bytes, ...], int, int]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent.
        _refuse(
            V1DeviceShellPackedCRefusalCode.UNSUPPORTED_CUDA,
            f"torch is unavailable for packed C-ladder CUDA materialization: {error}",
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
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "production labels must be one contiguous CUDA uint8 route axis",
        )
    outputs = [bytearray(len(parent_mask.packed_bits)) for _ in range(endpoint_count)]
    parent_bytes = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
    tile_size = policy.route_tile_size
    lanes = policy.endpoint_lane_tile_size
    device = labels.device
    try:
        available_device_bytes = cuda_reusable_memory_bytes(device, framework=torch)
    except (RuntimeError, TypeError) as error:  # pragma: no cover - driver specific.
        _refuse(
            V1DeviceShellPackedCRefusalCode.UNSUPPORTED_CUDA,
            f"CUDA free-memory forecast is unavailable: {error}",
        )
    if policy.cuda_logical_workspace_forecast_bytes > available_device_bytes:
        _refuse(
            V1DeviceShellPackedCRefusalCode.WORKSPACE_FORECAST_OVERFLOW,
            "current CUDA free memory is below the registered logical workspace",
        )
    bit_weights = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.int16,
        device=device,
    )
    bit_masks = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.uint8,
        device=device,
    )
    all_endpoint_labels = torch.arange(
        1,
        endpoint_count + 1,
        dtype=torch.int16,
        device=device,
    ).reshape(-1, 1, 1)
    packed_d2h_bytes = 0
    scalar_control_transfers = 0
    for start in range(0, parent_mask.route_count, tile_size):
        stop = min(parent_mask.route_count, start + tile_size)
        count = stop - start
        byte_start = start // 8
        byte_stop = byte_start + (count + 7) // 8
        parent_tile_host = np.array(
            parent_bytes[byte_start:byte_stop], dtype=np.uint8, copy=True
        )
        parent_tile = torch.as_tensor(
            parent_tile_host, dtype=torch.uint8, device=device
        )
        label_tile = labels[start:stop]
        padded_count = ((count + 7) // 8) * 8
        if padded_count != count:
            padded = torch.zeros(padded_count, dtype=torch.uint8, device=device)
            padded[:count].copy_(label_tile)
            label_tile = padded
        label_bits = label_tile.reshape(-1, 8)
        parent_live = (
            torch.bitwise_and(parent_tile.reshape(-1, 1), bit_masks) != 0
        )
        invalid = torch.any(label_tile[:count] > endpoint_count)
        dead_selected = torch.any(
            (label_bits != 0) & torch.logical_not(parent_live)
        )
        if bool(invalid.item()):
            _refuse(
                V1DeviceShellPackedCRefusalCode.INVALID_LABEL,
                "a CUDA shell label lies outside the frozen endpoint axis",
            )
        if bool(dead_selected.item()):
            _refuse(
                V1DeviceShellPackedCRefusalCode.PARENT_ROUTE_NOT_LIVE,
                "a CUDA shell label attempts to remove an absent parent route",
            )
        for lane_start in range(1, endpoint_count + 1, lanes):
            lane_stop = min(endpoint_count + 1, lane_start + lanes)
            endpoint_labels = all_endpoint_labels[lane_start - 1 : lane_stop - 1]
            label_lanes = label_bits.reshape(1, -1, 8)
            included = (label_lanes != 0) & (label_lanes <= endpoint_labels)
            clear = torch.sum(
                included.to(torch.int16) * bit_weights.reshape(1, 1, 8),
                dim=2,
                dtype=torch.int16,
            ).to(torch.uint8)
            children = torch.bitwise_and(
                parent_tile.reshape(1, -1), torch.bitwise_not(clear)
            )
            host_children = children.detach().to(device="cpu", non_blocking=False)
            host_array = host_children.contiguous().numpy()
            packed_d2h_bytes += int(host_array.nbytes)
            for lane_offset, endpoint in enumerate(
                range(lane_start - 1, lane_stop - 1)
            ):
                outputs[endpoint][byte_start:byte_stop] = host_array[
                    lane_offset
                ].tobytes()
        scalar_control_transfers += 2
    return (
        tuple(bytes(value) for value in outputs),
        packed_d2h_bytes,
        scalar_control_transfers,
    )


def _replay_label_commitment_and_shells(
    *,
    packed_children: tuple[bytes, ...],
    parent_mask: HardMask,
    branch_binding: V1FQRDeviceShellBranchBinding,
    route_tile_size: int,
) -> None:
    digest = _new_label_digest(
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        endpoint_count=len(packed_children),
    )
    parent_bytes = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
    child_axes = tuple(
        np.frombuffer(value, dtype=np.uint8) for value in packed_children
    )
    shell_counts = np.zeros(len(packed_children), dtype=np.int64)
    for start in range(0, parent_mask.route_count, route_tile_size):
        stop = min(parent_mask.route_count, start + route_tile_size)
        count = stop - start
        byte_start = start // 8
        byte_stop = byte_start + (count + 7) // 8
        previous = np.unpackbits(
            parent_bytes[byte_start:byte_stop], bitorder="big", count=count
        ).astype(bool, copy=False)
        reconstructed = np.zeros(count, dtype=np.uint8)
        for endpoint_offset, child_bytes in enumerate(child_axes):
            child = np.unpackbits(
                child_bytes[byte_start:byte_stop], bitorder="big", count=count
            ).astype(bool, copy=False)
            if np.any(child & ~previous):
                _refuse(
                    V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
                    "a materialized endpoint is not nested in its predecessor",
                )
            shell = previous & ~child
            shell_counts[endpoint_offset] += int(np.count_nonzero(shell))
            reconstructed[shell] = endpoint_offset + 1
            previous = child
        digest.update(memoryview(reconstructed).cast("B"))
    expected_shells = tuple(endpoint.shell_count for endpoint in branch_binding.endpoints)
    if tuple(int(value) for value in shell_counts) != expected_shells:
        _refuse(
            V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
            "materialized endpoint shells differ from their frozen counts",
        )
    observed = "partition-free-v1-device-shell-labels-" + digest.hexdigest()
    if observed != branch_binding.label_commitment:
        _refuse(
            V1DeviceShellPackedCRefusalCode.LABEL_COMMITMENT_MISMATCH,
            "materialized masks do not reproduce the frozen label commitment",
        )


def materialize_v1_device_shell_packed_c_ladder(
    *,
    label_lease: V1DeviceShellLabelLease,
    proposal_freeze: V1SampledKKTDeviceShellProposalFreezeForC,
    branch_binding: V1FQRDeviceShellBranchBinding,
    parent_mask: HardMask,
    workspace_policy: V1DeviceShellPackedCWorkspacePolicy,
    ladder_coordinates: V1DeviceShellPackedCLadderCoordinates,
    source_ladder_receipt: V1DeviceShellFrozenLadderReceipt | None = None,
) -> V1DeviceShellPackedCMaterializationResult:
    """Materialize every already-frozen endpoint as one exact packed ladder.

    The caller retains ownership of ``label_lease`` and must keep its context
    manager open for the entire call.  This function neither closes the lease
    nor grants authority to execute or interpret C outcomes.
    """

    if type(label_lease) is not V1DeviceShellLabelLease:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C materialization requires an exact label lease",
        )
    if type(branch_binding) is not V1FQRDeviceShellBranchBinding:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C materialization requires an exact branch binding",
        )
    if type(parent_mask) is not HardMask:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C materialization requires an exact parent HardMask",
        )
    if type(workspace_policy) is not V1DeviceShellPackedCWorkspacePolicy:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C materialization requires an exact workspace policy",
        )
    if type(ladder_coordinates) is not V1DeviceShellPackedCLadderCoordinates:
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            "packed C materialization requires exact ladder coordinates",
        )
    branch_binding.assert_integrity()
    workspace_policy.assert_integrity()
    ladder_coordinates.assert_integrity()
    expected_binding = _branch_from_proposal_freeze(
        proposal_freeze=proposal_freeze,
        branch=branch_binding.branch,
    )
    proposal_hash = proposal_freeze.certificate_hash
    if (
        expected_binding.binding_hash != branch_binding.binding_hash
        or workspace_policy.source_branch_binding_hash
        != branch_binding.binding_hash
        or workspace_policy.source_proposal_freeze_hash != proposal_hash
        or workspace_policy.route_count != branch_binding.route_count
        or workspace_policy.endpoint_count != len(branch_binding.endpoints)
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.SOURCE_MISMATCH,
            "proposal freeze, branch binding, and workspace registration differ",
        )
    _validate_source_ladder_receipt(
        source_ladder_receipt=source_ladder_receipt,
        branch_binding=branch_binding,
    )
    parent_hash = _direct_mask_hash(parent_mask, name="packed C parent mask")
    discovery_prefix = _discovery_prefix_hash(
        context_ids=ladder_coordinates.discovery_context_ids,
        draw_ids=ladder_coordinates.discovery_draw_ids,
    )
    if (
        parent_hash != branch_binding.current_mask_hash
        or parent_mask.route_axis_hash != branch_binding.route_axis_hash
        or parent_mask.route_count != branch_binding.route_count
        or discovery_prefix != branch_binding.discovery_prefix_hash
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.STALE_COORDINATES,
            "parent mask or discovery prefix left the frozen branch coordinates",
        )
    if label_lease.operation is not (
        V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    ):
        _refuse(
            V1DeviceShellPackedCRefusalCode.OPERATION_MISMATCH,
            "packed C materialization requires a C-direction label lease",
        )
    if label_lease.route_count != branch_binding.route_count:
        _refuse(
            V1DeviceShellPackedCRefusalCode.STALE_COORDINATES,
            "label lease route axis length differs from the frozen branch",
        )
    backend_kind = label_lease.backend_kind
    device_identity = label_lease.device_identity
    labels = label_lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    if backend_kind == "numpy_reference":
        packed_children, packed_d2h, scalar_controls = _pack_numpy_reference(
            labels=labels,
            parent_mask=parent_mask,
            endpoint_count=len(branch_binding.endpoints),
            policy=workspace_policy,
        )
    elif backend_kind == "torch_cuda":
        packed_children, packed_d2h, scalar_controls = _pack_torch_cuda(
            labels=labels,
            parent_mask=parent_mask,
            endpoint_count=len(branch_binding.endpoints),
            policy=workspace_policy,
        )
    else:  # pragma: no cover - lease constructor owns this closed axis.
        _refuse(
            V1DeviceShellPackedCRefusalCode.INVALID_TYPE,
            f"unsupported label-lease backend {backend_kind!r}",
        )
    # Replay the storage invariant after all packing.  A concurrent or illicit
    # in-place CUDA mutation increments the tensor version and fails here.
    label_lease._labels_for_registered_operation(
        operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
    )
    _replay_label_commitment_and_shells(
        packed_children=packed_children,
        parent_mask=parent_mask,
        branch_binding=branch_binding,
        route_tile_size=workspace_policy.route_tile_size,
    )

    packed_actions: list[V1PackedRemovalAction] = []
    frozen_actions: list[V1FrozenLadderAction] = []
    for endpoint, packed_bits in zip(
        branch_binding.endpoints, packed_children, strict=True
    ):
        packed_sha = hashlib.sha256(packed_bits).hexdigest()
        if packed_sha != endpoint.resulting_mask_packed_sha256:
            _refuse(
                V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
                f"endpoint {endpoint.endpoint_index} packed SHA-256 changed",
            )
        child = HardMask(
            route_axis_hash=branch_binding.route_axis_hash,
            route_count=branch_binding.route_count,
            packed_bits=packed_bits,
        )
        action = V1PackedRemovalAction.from_resulting_mask(
            parent_mask=parent_mask,
            resulting_mask=child,
        )
        if (
            child.mask_hash != endpoint.resulting_mask_hash
            or action.removed_count != endpoint.cumulative_removed_count
            or action.removed_route_index_hash != endpoint.removed_route_index_hash
            or action.action_hash != endpoint.packed_action_hash
        ):
            _refuse(
                V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
                f"endpoint {endpoint.endpoint_index} action identity changed",
            )
        frozen = V1FrozenLadderAction(
            endpoint_index=endpoint.endpoint_index,
            action_id=(
                f"device-shell-packed-C:{branch_binding.branch.value}:"
                f"{endpoint.source_action_receipt_hash}:"
                f"{endpoint.packed_action_hash}"
            ),
            packed_removal_action=action,
        )
        packed_actions.append(action)
        frozen_actions.append(frozen)

    freeze_receipt_hash = content_hash(
        {
            "source_proposal_freeze_hash": proposal_hash,
            "source_candidate_family_hash": (
                proposal_freeze.frozen_candidate_family_hash
            ),
            "source_branch_binding_hash": branch_binding.binding_hash,
            "source_ladder_receipt_hash": (
                branch_binding.source_ladder_receipt_hash
            ),
            "source_label_commitment": branch_binding.label_commitment,
            "workspace_policy_hash": workspace_policy.policy_hash,
            "ladder_coordinates_hash": ladder_coordinates.coordinates_hash,
            "endpoint_source_identity_hashes": tuple(
                endpoint.identity_hash for endpoint in branch_binding.endpoints
            ),
            "endpoint_packed_action_hashes": tuple(
                action.action_hash for action in packed_actions
            ),
            "issued_before_C_outcomes": True,
            "route_order_materialized": False,
            "KKT_hard_optimality_claim": False,
        },
        prefix=_LADDER_FREEZE_PREFIX,
    )
    ladder = V1FrozenCandidateLadder(
        campaign_contract_hash=ladder_coordinates.campaign_contract_hash,
        model_identity_hash=ladder_coordinates.model_identity_hash,
        query_contract_hash=ladder_coordinates.query_contract_hash,
        preprocessing_hash=ladder_coordinates.preprocessing_hash,
        current_mask_hash=parent_hash,
        raw_logit_stream_source_hash=(
            ladder_coordinates.raw_logit_stream_source_hash
        ),
        sampling_law_hash=ladder_coordinates.sampling_law_hash,
        proposal_generation=branch_binding.proposal_generation,
        route_count=branch_binding.route_count,
        route_axis_hash=branch_binding.route_axis_hash,
        actions=tuple(frozen_actions),
        discovery_context_ids=ladder_coordinates.discovery_context_ids,
        discovery_draw_ids=ladder_coordinates.discovery_draw_ids,
        discovery_evidence_hash=branch_binding.discovery_evidence_hash,
        freeze_receipt_hash=freeze_receipt_hash,
        parent_mask=parent_mask,
    )
    expected_packed_d2h = (
        workspace_policy.retained_host_packed_bytes
        if backend_kind == "torch_cuda"
        else 0
    )
    if packed_d2h != expected_packed_d2h:
        _refuse(
            V1DeviceShellPackedCRefusalCode.IDENTITY_MISMATCH,
            "packed child tensor transfer accounting does not close",
        )
    receipt = V1DeviceShellPackedCMaterializationReceipt(
        branch=branch_binding.branch,
        source_proposal_freeze_hash=proposal_hash,
        source_candidate_family_hash=proposal_freeze.frozen_candidate_family_hash,
        source_branch_binding_hash=branch_binding.binding_hash,
        source_ladder_receipt_hash=branch_binding.source_ladder_receipt_hash,
        source_label_commitment=branch_binding.label_commitment,
        workspace_policy_hash=workspace_policy.policy_hash,
        ladder_coordinates_hash=ladder_coordinates.coordinates_hash,
        backend_kind=backend_kind,
        device_identity=device_identity,
        route_axis_hash=branch_binding.route_axis_hash,
        route_count=branch_binding.route_count,
        current_mask_hash=parent_hash,
        chart_identity_hash=branch_binding.chart_identity_hash,
        proposal_generation=branch_binding.proposal_generation,
        discovery_prefix_hash=branch_binding.discovery_prefix_hash,
        discovery_evidence_hash=branch_binding.discovery_evidence_hash,
        endpoint_count=len(branch_binding.endpoints),
        endpoint_source_identity_hashes=tuple(
            endpoint.identity_hash for endpoint in branch_binding.endpoints
        ),
        endpoint_resulting_mask_hashes=tuple(
            action.resulting_mask.mask_hash for action in packed_actions
        ),
        endpoint_removed_route_index_hashes=tuple(
            action.removed_route_index_hash for action in packed_actions
        ),
        endpoint_packed_action_hashes=tuple(
            action.action_hash for action in packed_actions
        ),
        frozen_action_hashes=ladder.action_hashes,
        frozen_ladder_hash=ladder.ladder_hash,
        retained_host_packed_bytes=workspace_policy.retained_host_packed_bytes,
        peak_host_packed_payload_forecast_bytes=(
            workspace_policy.peak_host_packed_payload_bytes
        ),
        cuda_logical_workspace_forecast_bytes=(
            workspace_policy.cuda_logical_workspace_forecast_bytes
        ),
        packed_child_tensor_d2h_bytes=packed_d2h,
        cuda_scalar_control_transfer_count=scalar_controls,
    )
    return V1DeviceShellPackedCMaterializationResult(
        ladder=ladder,
        receipt=receipt,
    )


__all__ = [
    "DEVICE_SHELL_PACKED_C_AUTHORITY_SCOPE",
    "DEVICE_SHELL_PACKED_C_COORDINATES_SCHEMA",
    "DEVICE_SHELL_PACKED_C_DEFAULT_ENDPOINT_LANE_TILE_SIZE",
    "DEVICE_SHELL_PACKED_C_DEFAULT_NUMPY_REFERENCE_ROUTE_LIMIT",
    "DEVICE_SHELL_PACKED_C_DEFAULT_ROUTE_TILE_SIZE",
    "DEVICE_SHELL_PACKED_C_MATERIALIZATION_RECEIPT_SCHEMA",
    "DEVICE_SHELL_PACKED_C_MATERIALIZATION_RESULT_SCHEMA",
    "DEVICE_SHELL_PACKED_C_WORKSPACE_POLICY_SCHEMA",
    "V1DeviceShellPackedCLadderCoordinates",
    "V1DeviceShellPackedCMaterializationReceipt",
    "V1DeviceShellPackedCMaterializationResult",
    "V1DeviceShellPackedCRefusal",
    "V1DeviceShellPackedCRefusalCode",
    "V1DeviceShellPackedCWorkspacePolicy",
    "materialize_v1_device_shell_packed_c_ladder",
    "register_v1_device_shell_packed_c_workspace_policy",
]
