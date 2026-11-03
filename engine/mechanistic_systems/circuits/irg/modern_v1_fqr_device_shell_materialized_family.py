"""Packed pre-C adapter for a proposal-only device-shell branch family.

The CUDA selector deliberately retains first-inclusion shell labels instead of
an ordinary route order.  Independent C measurement, however, consumes exact
packed hard-mask endpoints.  This module binds those two representations
without reconstructing an order, manufacturing rounded-candidate receipts, or
claiming hard-mask optimality.

Only the registered issuer can construct the adapter.  Every packed endpoint
must replay the identities already frozen by the device-shell graph, and the
source proposal-freeze certificate remains authority-limited to opening the
independent C direction.  The adapter itself grants no C outcome, K, O,
commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_fqr_branch_family import (
    V1FQRDeviceShellBranchBinding,
    V1FQRDeviceShellPreFrozenBranchFamily,
)
from .modern_v1_device_shell_packed_c_ladder import (
    V1DeviceShellPackedCMaterializationReceipt,
    V1DeviceShellPackedCMaterializationResult,
)
from .modern_v1_ladder_statistics import V1FrozenCandidateLadder
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_kkt_device_shell_bridge import (
    V1SampledKKTDeviceShellProposalFreezeForC,
)


FQR_DEVICE_SHELL_PACKED_BRANCH_BINDING_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_packed_C_branch_binding_v1"
)
FQR_DEVICE_SHELL_PACKED_BRANCH_FAMILY_SCHEMA: Final = (
    "partition_free_v1_fqr_device_shell_packed_C_branch_family_v1"
)
FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE: Final = (
    "pre_C_device_shell_identity_adapter_only_no_route_order_hard_optimality_"
    "C_outcome_K_O_commit_terminal_or_pareto_authority_v1"
)

_BINDING_PREFIX: Final = "partition-free-v1-fqr-device-shell-packed-C-binding-"
_FAMILY_PREFIX: Final = "partition-free-v1-fqr-device-shell-packed-C-family-"
_ISSUANCE_TOKEN: Final = object()


class V1FQRDeviceShellPackedFamilyRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    POST_OUTCOME = "post_outcome"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FQRDeviceShellPackedFamilyRefusal(ValueError):
    def __init__(
        self,
        code: V1FQRDeviceShellPackedFamilyRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1FQRDeviceShellPackedFamilyRefusalCode,
    message: str,
) -> NoReturn:
    raise V1FQRDeviceShellPackedFamilyRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _assert_materialization_result_integrity(
    result: V1DeviceShellPackedCMaterializationResult,
    *,
    branch_name: str,
) -> None:
    try:
        result.assert_integrity()
    except V1FQRDeviceShellPackedFamilyRefusal:
        raise
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
            f"{branch_name} packed-C materialization does not replay: {error}",
        )


def _assert_ladder_matches_source(
    *,
    source: V1FQRDeviceShellBranchBinding,
    ladder: V1FrozenCandidateLadder,
) -> None:
    try:
        source.assert_integrity()
        ladder.assert_integrity()
    except V1FQRDeviceShellPackedFamilyRefusal:
        raise
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
            f"source binding or packed ladder does not replay: {error}",
        )
    parent = ladder.parent_mask
    if type(parent) is not HardMask or not ladder.is_packed:
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
            "device-shell C adaptation requires an exact packed ladder and parent",
        )
    if (
        ladder.current_mask_hash != source.current_mask_hash
        or parent.mask_hash != source.current_mask_hash
        or ladder.route_axis_hash != source.route_axis_hash
        or ladder.route_count != source.route_count
        or ladder.proposal_generation != source.proposal_generation
        or ladder.discovery_evidence_hash != source.discovery_evidence_hash
        or len(ladder.actions) != len(source.endpoints)
    ):
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
            "materialized ladder left its device-shell branch coordinates",
        )
    for action, endpoint in zip(ladder.actions, source.endpoints, strict=True):
        packed = action.packed_removal_action
        if packed is None:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "materialized device-shell endpoint is not packed",
            )
        child = packed.resulting_mask
        if (
            action.endpoint_index != endpoint.endpoint_index
            or packed.removed_count != endpoint.cumulative_removed_count
            or packed.removed_route_index_hash
            != endpoint.removed_route_index_hash
            or child.mask_hash != endpoint.resulting_mask_hash
            or packed.action_hash != endpoint.packed_action_hash
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "a packed C endpoint differs from its frozen shell identity",
            )


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellPackedBranchBinding:
    """One exact packed ladder derived from a frozen shell-label branch."""

    source_binding: V1FQRDeviceShellBranchBinding = field(
        repr=False,
        compare=False,
    )
    frozen_ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    materialization_receipt: V1DeviceShellPackedCMaterializationReceipt = field(
        repr=False,
        compare=False,
    )
    source_graph_hash: str
    proposal_freeze_certificate_hash: str
    _issuance_token: InitVar[object]
    route_order_reconstructed: bool = False
    hard_global_optimality_claim: bool = False
    C_outcomes_seen: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE
    schema: str = FQR_DEVICE_SHELL_PACKED_BRANCH_BINDING_SCHEMA
    _binding_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise PermissionError("packed device-shell binding requires its issuer")
        if type(self.source_binding) is not V1FQRDeviceShellBranchBinding:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "packed device-shell binding requires the exact source binding",
            )
        if type(self.frozen_ladder) is not V1FrozenCandidateLadder:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "packed device-shell binding requires the exact frozen ladder",
            )
        if type(self.materialization_receipt) is not (
            V1DeviceShellPackedCMaterializationReceipt
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "packed device-shell binding requires the exact materialization receipt",
            )
        for name in ("source_graph_hash", "proposal_freeze_certificate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        _assert_ladder_matches_source(
            source=self.source_binding,
            ladder=self.frozen_ladder,
        )
        self._assert_materialization_closure()
        flags = (
            self.route_order_reconstructed,
            self.hard_global_optimality_claim,
            self.C_outcomes_seen,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("packed device-shell authority flags must be exact booleans")
        if any(flags):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION,
                "packed device-shell binding crossed its pre-C identity boundary",
            )
        if (
            self.authority_scope != FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE
            or self.schema != FQR_DEVICE_SHELL_PACKED_BRANCH_BINDING_SCHEMA
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
                "packed device-shell binding scope or schema is invalid",
            )
        self.assert_integrity()

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        return self.source_binding.branch

    @property
    def problem_hash(self) -> str:
        return self.source_binding.problem_hash

    @property
    def policy_hash(self) -> str:
        return self.source_binding.policy_hash

    @property
    def result_hash(self) -> str:
        return self.source_binding.base_result_hash

    @property
    def route_axis_hash(self) -> str:
        return self.source_binding.route_axis_hash

    @property
    def route_count(self) -> int:
        return self.source_binding.route_count

    @property
    def current_mask_hash(self) -> str:
        return self.source_binding.current_mask_hash

    @property
    def chart_identity_hash(self) -> str:
        return self.source_binding.chart_identity_hash

    @property
    def proposal_generation(self) -> int:
        return self.source_binding.proposal_generation

    @property
    def discovery_prefix_hash(self) -> str:
        return self.source_binding.discovery_prefix_hash

    @property
    def discovery_evidence_hash(self) -> str:
        return self.source_binding.discovery_evidence_hash

    @property
    def endpoint_registered_resources(self) -> tuple[float, ...]:
        return tuple(endpoint.registered_resource for endpoint in self.source_binding.endpoints)

    @property
    def endpoint_packed_action_hashes(self) -> tuple[str, ...]:
        return tuple(endpoint.packed_action_hash for endpoint in self.source_binding.endpoints)

    def _assert_materialization_closure(self) -> None:
        receipt = self.materialization_receipt
        try:
            receipt.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                f"packed C materialization receipt does not replay: {error}",
            )
        if (
            receipt.branch is not self.branch
            or receipt.source_proposal_freeze_hash
            != self.proposal_freeze_certificate_hash
            or receipt.source_branch_binding_hash
            != self.source_binding.binding_hash
            or receipt.source_ladder_receipt_hash
            != self.source_binding.source_ladder_receipt_hash
            or receipt.source_label_commitment
            != self.source_binding.label_commitment
            or receipt.frozen_ladder_hash != self.frozen_ladder.ladder_hash
            or receipt.frozen_action_hashes != self.frozen_ladder.action_hashes
            or receipt.endpoint_packed_action_hashes
            != self.endpoint_packed_action_hashes
            or receipt.route_order_materialized
            or receipt.KKT_hard_optimality_claim
            or receipt.C_outcomes_seen
            or receipt.C_FQR_decision_authority
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "packed C materialization receipt left its source branch or authority",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_binding_hash": self.source_binding.binding_hash,
            "frozen_ladder_hash": self.frozen_ladder.ladder_hash,
            "source_graph_hash": self.source_graph_hash,
            "proposal_freeze_certificate_hash": self.proposal_freeze_certificate_hash,
            "materialization_receipt_hash": self.materialization_receipt.receipt_hash,
            "branch": self.branch,
            "endpoint_packed_action_hashes": self.endpoint_packed_action_hashes,
            "route_order_reconstructed": self.route_order_reconstructed,
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
            "C_outcomes_seen": self.C_outcomes_seen,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        _assert_ladder_matches_source(
            source=self.source_binding,
            ladder=self.frozen_ladder,
        )
        self._assert_materialization_closure()
        if any(
            (
                self.route_order_reconstructed,
                self.hard_global_optimality_claim,
                self.C_outcomes_seen,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            )
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION,
                "packed device-shell binding gained downstream authority",
            )
        observed = content_hash(self._payload(), prefix=_BINDING_PREFIX)
        cached = self._binding_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "packed device-shell binding was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "packed device-shell binding hash does not match",
            )
        object.__setattr__(self, "_binding_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRDeviceShellPackedPreFrozenBranchFamily:
    """Two exact packed branch ladders frozen before independent C outcomes."""

    source_family: V1FQRDeviceShellPreFrozenBranchFamily = field(
        repr=False,
        compare=False,
    )
    source_graph_hash: str
    proposal_freeze_certificate_hash: str
    spectral: V1FQRDeviceShellPackedBranchBinding
    diagonal_l1: V1FQRDeviceShellPackedBranchBinding
    _issuance_token: InitVar[object]
    dominance_certificate: None = None
    route_order_reconstructed: bool = False
    hard_global_optimality_claim: bool = False
    C_outcomes_seen: bool = False
    cross_branch_action_synthesis_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE
    schema: str = FQR_DEVICE_SHELL_PACKED_BRANCH_FAMILY_SCHEMA
    _family_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise PermissionError("packed device-shell family requires its issuer")
        if type(self.source_family) is not V1FQRDeviceShellPreFrozenBranchFamily:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "packed device-shell family requires the exact source family",
            )
        if (
            type(self.spectral) is not V1FQRDeviceShellPackedBranchBinding
            or type(self.diagonal_l1) is not V1FQRDeviceShellPackedBranchBinding
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
                "packed device-shell family requires two exact branch bindings",
            )
        for name in ("source_graph_hash", "proposal_freeze_certificate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.dominance_certificate is not None:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION,
                "proposal-only device-shell adaptation cannot omit a branch",
            )
        flags = (
            self.route_order_reconstructed,
            self.hard_global_optimality_claim,
            self.C_outcomes_seen,
            self.cross_branch_action_synthesis_authority,
            self.calibration_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("packed device-shell family flags must be exact booleans")
        if any(flags):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION,
                "packed device-shell family crossed its pre-C boundary",
            )
        if (
            self.authority_scope != FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE
            or self.schema != FQR_DEVICE_SHELL_PACKED_BRANCH_FAMILY_SCHEMA
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
                "packed device-shell family scope or schema is invalid",
            )
        self.assert_integrity()

    @property
    def execution_receipt_hash(self) -> str:
        return self.source_family.execution_receipt_hash

    @property
    def problem_hash(self) -> str:
        return self.source_family.problem_hash

    @property
    def policy_hash(self) -> str:
        return self.source_family.policy_hash

    @property
    def result_hash(self) -> str:
        return self.source_family.base_result_hash

    @property
    def route_axis_hash(self) -> str:
        return self.source_family.route_axis_hash

    @property
    def route_count(self) -> int:
        return self.source_family.route_count

    @property
    def current_mask_hash(self) -> str:
        return self.source_family.current_mask_hash

    @property
    def chart_identity_hash(self) -> str:
        return self.source_family.chart_identity_hash

    @property
    def proposal_generation(self) -> int:
        return self.source_family.proposal_generation

    @property
    def discovery_prefix_hash(self) -> str:
        return self.source_family.discovery_prefix_hash

    @property
    def discovery_evidence_hash(self) -> str:
        return self.source_family.discovery_evidence_hash

    @property
    def retained_bindings(self) -> tuple[V1FQRDeviceShellPackedBranchBinding, ...]:
        return (self.spectral, self.diagonal_l1)

    def binding_for_branch(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> V1FQRDeviceShellPackedBranchBinding:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch must be an exact V1RelaxedKKTBranch")
        return (
            self.spectral
            if branch is V1RelaxedKKTBranch.SPECTRAL
            else self.diagonal_l1
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_family_hash": self.source_family.family_hash,
            "source_graph_hash": self.source_graph_hash,
            "proposal_freeze_certificate_hash": self.proposal_freeze_certificate_hash,
            "spectral_binding_hash": self.spectral.binding_hash,
            "diagonal_l1_binding_hash": self.diagonal_l1.binding_hash,
            "dominance_certificate_hash": None,
            "route_order_reconstructed": self.route_order_reconstructed,
            "hard_global_optimality_claim": self.hard_global_optimality_claim,
            "C_outcomes_seen": self.C_outcomes_seen,
            "cross_branch_action_synthesis_authority": (
                self.cross_branch_action_synthesis_authority
            ),
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def family_hash(self) -> str:
        cached = self._family_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.source_family.assert_integrity()
        self.spectral.assert_integrity()
        self.diagonal_l1.assert_integrity()
        source = self.source_family
        common = (
            self.source_graph_hash,
            self.proposal_freeze_certificate_hash,
            source.route_axis_hash,
            source.route_count,
            source.current_mask_hash,
            source.chart_identity_hash,
            source.proposal_generation,
            source.discovery_prefix_hash,
            source.discovery_evidence_hash,
        )
        for binding, source_binding in (
            (self.spectral, source.spectral),
            (self.diagonal_l1, source.diagonal_l1),
        ):
            observed = (
                binding.source_graph_hash,
                binding.proposal_freeze_certificate_hash,
                binding.route_axis_hash,
                binding.route_count,
                binding.current_mask_hash,
                binding.chart_identity_hash,
                binding.proposal_generation,
                binding.discovery_prefix_hash,
                binding.discovery_evidence_hash,
            )
            if (
                observed != common
                or binding.source_binding.binding_hash != source_binding.binding_hash
                or binding.materialization_receipt.source_candidate_family_hash
                != source.family_hash
            ):
                _refuse(
                    V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
                    "packed branch left the frozen device-shell family",
                )
        left = self.spectral.frozen_ladder
        right = self.diagonal_l1.frozen_ladder
        shared_ladder_coordinates = (
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
            "discovery_context_ids",
            "discovery_draw_ids",
            "discovery_evidence_hash",
        )
        if any(
            getattr(left, name) != getattr(right, name)
            for name in shared_ladder_coordinates
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH,
                "materialized branch ladders do not share one D/chart population",
            )
        if any(
            (
                self.route_order_reconstructed,
                self.hard_global_optimality_claim,
                self.C_outcomes_seen,
                self.cross_branch_action_synthesis_authority,
                self.calibration_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            )
        ):
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION,
                "packed device-shell family gained downstream authority",
            )
        observed_hash = content_hash(self._payload(), prefix=_FAMILY_PREFIX)
        cached = self._family_hash
        if cached is not None and cached != observed_hash:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "packed device-shell family was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed_hash:
            _refuse(
                V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH,
                "packed device-shell family hash does not match",
            )
        object.__setattr__(self, "_family_hash", observed_hash)
        return observed_hash


def bind_v1_fqr_device_shell_materialized_family(
    *,
    proposal_freeze: V1SampledKKTDeviceShellProposalFreezeForC,
    spectral_materialization: V1DeviceShellPackedCMaterializationResult,
    diagonal_l1_materialization: V1DeviceShellPackedCMaterializationResult,
) -> V1FQRDeviceShellPackedPreFrozenBranchFamily:
    """Bind two exact packed ladders to one proposal-only CUDA shell freeze."""

    if type(proposal_freeze) is not V1SampledKKTDeviceShellProposalFreezeForC:
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
            "materialized family requires the exact proposal-freeze certificate",
        )
    if (
        type(spectral_materialization) is not V1DeviceShellPackedCMaterializationResult
        or type(diagonal_l1_materialization)
        is not V1DeviceShellPackedCMaterializationResult
    ):
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE,
            "materialized family requires two exact packed-C results",
        )
    _assert_materialization_result_integrity(
        spectral_materialization,
        branch_name="spectral",
    )
    _assert_materialization_result_integrity(
        diagonal_l1_materialization,
        branch_name="diagonal_l1",
    )
    proposal_freeze.assert_integrity()
    graph = proposal_freeze.graph
    source_family = graph.branch_family
    if (
        proposal_freeze.frozen_candidate_family_hash != source_family.family_hash
        or proposal_freeze.frozen_graph_hash != graph.graph_hash
        or not proposal_freeze.independent_C_FQR_required
        or proposal_freeze.C_FQR_decision_authority
        or graph.c_stream_outcomes_seen
    ):
        _refuse(
            V1FQRDeviceShellPackedFamilyRefusalCode.POST_OUTCOME,
            "proposal freeze cannot authorize this independent-C adaptation",
        )
    certificate_hash = proposal_freeze.certificate_hash
    spectral = V1FQRDeviceShellPackedBranchBinding(
        source_binding=source_family.spectral,
        frozen_ladder=spectral_materialization.ladder,
        materialization_receipt=spectral_materialization.receipt,
        source_graph_hash=graph.graph_hash,
        proposal_freeze_certificate_hash=certificate_hash,
        _issuance_token=_ISSUANCE_TOKEN,
    )
    diagonal = V1FQRDeviceShellPackedBranchBinding(
        source_binding=source_family.diagonal_l1,
        frozen_ladder=diagonal_l1_materialization.ladder,
        materialization_receipt=diagonal_l1_materialization.receipt,
        source_graph_hash=graph.graph_hash,
        proposal_freeze_certificate_hash=certificate_hash,
        _issuance_token=_ISSUANCE_TOKEN,
    )
    return V1FQRDeviceShellPackedPreFrozenBranchFamily(
        source_family=source_family,
        source_graph_hash=graph.graph_hash,
        proposal_freeze_certificate_hash=certificate_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        _issuance_token=_ISSUANCE_TOKEN,
    )


__all__ = [
    "FQR_DEVICE_SHELL_PACKED_BRANCH_BINDING_SCHEMA",
    "FQR_DEVICE_SHELL_PACKED_BRANCH_FAMILY_SCHEMA",
    "FQR_DEVICE_SHELL_PACKED_FAMILY_SCOPE",
    "V1FQRDeviceShellPackedBranchBinding",
    "V1FQRDeviceShellPackedFamilyRefusal",
    "V1FQRDeviceShellPackedFamilyRefusalCode",
    "V1FQRDeviceShellPackedPreFrozenBranchFamily",
    "bind_v1_fqr_device_shell_materialized_family",
]
