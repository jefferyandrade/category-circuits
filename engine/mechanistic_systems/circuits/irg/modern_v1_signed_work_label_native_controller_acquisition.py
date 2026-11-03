"""Controller-facing acquisition for one materialized label-native endpoint.

The label-native pre-O plan is an immutable, receipt-only fallback queue.  A
separate materializer may realize exactly one descriptor as a packed hard-mask
child.  This module closes those two objects against the live Option-A
campaign binding and exact committed-mask ancestry, then issues one narrow
controller candidate carrier.

This is intentionally *not* a ``V1SignedWorkCandidateProposal`` adapter.  The
compact label-native plan proves its native-fiber/source semantics through the
exact descriptor, certificate, crosswalk, and source action receipts, but it
does not separately expose every legacy native-fiber axis/measure field needed
to reconstruct that older type without invention.  The carrier below keeps
the stronger source receipts intact and grants no candidate-freeze,
calibration, oracle, commit, terminal, Pareto, or execution authority.

Production acquisition accepts only a CUDA materialization that exactly
replayed the plan-bound provider receipt.  The underscore factory admits the
route-limited CPU reference materializer only for focused tests and preserves
the same zero-authority firewall.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_candidate_composition import (
    V1PackedRemovalAction,
    V1RemovedRouteSetIdentity,
)
from .modern_v1_signed_work_label_native_runtime_acquisition import (
    V1LabelNativeRuntimeFallbackDescriptor,
    V1LabelNativeRuntimeFallbackPlan,
)
from .modern_v1_signed_work_label_native_runtime_materialization import (
    V1LabelNativeRuntimeSelectedEndpointProposal,
)


LABEL_NATIVE_CONTROLLER_CANDIDATE_SCHEMA: Final = (
    "partition_free_v1_label_native_controller_candidate_carrier_v1"
)
LABEL_NATIVE_CONTROLLER_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_controller_single_endpoint_receipt_v1"
)
LABEL_NATIVE_CONTROLLER_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_label_native_controller_single_endpoint_acquisition_v1"
)
LABEL_NATIVE_CONTROLLER_ACQUISITION_AUTHORITY: Final = (
    "exact_plan_selected_label_native_single_endpoint_controller_input_"
    "proposal_only_no_freeze_K_O_commit_terminal_pareto_or_execution_"
    "authority_v1"
)

_CANDIDATE_PREFIX: Final = "modern-v1-label-native-controller-candidate-"
_RECEIPT_PREFIX: Final = "modern-v1-label-native-controller-acquisition-"
_QUEUE_AXIS_PREFIX: Final = "modern-v1-label-native-runtime-plan-queue-axis-"
_LINEAGE_PREFIX: Final = "partition-free-v1-runtime-proposal-mask-lineage-"
_CANDIDATE_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_ACQUISITION_TOKEN: Final = object()


class V1LabelNativeControllerAcquisitionRefusalCode(str, Enum):
    """Typed fail-closed outcomes at the controller acquisition seam."""

    INVALID_TYPE = "invalid_type"
    PLAN_DESCRIPTOR_MISMATCH = "plan_descriptor_mismatch"
    MATERIALIZATION_MISMATCH = "materialization_mismatch"
    CAMPAIGN_BINDING_MISMATCH = "campaign_binding_mismatch"
    STALE_CURRENT_MASK = "stale_current_mask"
    LINEAGE_MISMATCH = "lineage_mismatch"
    PRODUCTION_MATERIALIZATION_REQUIRED = "production_materialization_required"
    AUTHORITY_VIOLATION = "authority_violation"
    TAMPERED = "tampered"


class V1LabelNativeControllerAcquisitionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LabelNativeControllerAcquisitionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeControllerAcquisitionRefusalCode:
            raise TypeError("controller-acquisition refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeControllerAcquisitionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeControllerAcquisitionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _fresh_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
            f"{name} must be an exact HardMask",
        )
    return HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=mask.packed_bits,
    ).mask_hash


def _lineage(
    *,
    current_mask_hash: str,
    mask_hashes: Sequence[str],
    transition_hashes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    lineage = tuple(_identity(value, name="mask lineage hash") for value in mask_hashes)
    transitions = tuple(
        _identity(value, name="mask transition receipt hash")
        for value in transition_hashes
    )
    if (
        not lineage
        or lineage[-1] != current_mask_hash
        or len(transitions) != len(lineage) - 1
        or len(set(lineage)) != len(lineage)
        or len(set(transitions)) != len(transitions)
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.LINEAGE_MISMATCH,
            "controller acquisition requires complete unique ancestry ending at the current mask",
        )
    lineage_hash = content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_mask_lineage_v1",
            "mask_lineage_hashes": lineage,
            "transition_receipt_hashes": transitions,
        },
        prefix=_LINEAGE_PREFIX,
    )
    return lineage, transitions, lineage_hash


def _queue_axis_hash(plan: V1LabelNativeRuntimeFallbackPlan) -> str:
    return content_hash(
        {
            "runtime_plan_hash": plan.plan_hash,
            "ordered_descriptor_hashes": plan.ordered_descriptor_hashes,
            "ordered_coordinate_hashes": plan.ordered_coordinate_hashes,
            "ordered_physical_candidate_identity_hashes": (
                plan.ordered_physical_candidate_identity_hashes
            ),
        },
        prefix=_QUEUE_AXIS_PREFIX,
    )


def _assert_zero_authority(materialized: V1LabelNativeRuntimeSelectedEndpointProposal) -> None:
    receipt = materialized.receipt
    required_true = (
        receipt.proposal_only,
        receipt.paired_bce_oracle_still_required,
        receipt.exactly_one_endpoint_materialized,
    )
    required_false = (
        receipt.complete_queue_materialized,
        receipt.calibration_K_open_authority,
        receipt.paired_oracle_O_open_authority,
        receipt.irreversible_decision_authority,
        receipt.mask_commit_authority,
        receipt.terminal_authority,
        receipt.pareto_authority,
        receipt.execution_authority,
    )
    if (
        any(type(value) is not bool or not value for value in required_true)
        or any(type(value) is not bool or value for value in required_false)
        or receipt.unselected_endpoint_materialization_count != 0
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.AUTHORITY_VIOLATION,
            "selected materialization crossed its proposal-only authority boundary",
        )


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeControllerCandidateCarrier:
    """One exact descriptor plus its sole materialized packed child."""

    runtime_plan_hash: str
    plan_queue_index: int
    queue_descriptor_count: int
    ordered_descriptor_axis_hash: str
    descriptor: V1LabelNativeRuntimeFallbackDescriptor = field(
        repr=False,
        compare=False,
    )
    selected_endpoint: V1LabelNativeRuntimeSelectedEndpointProposal = field(
        repr=False,
        compare=False,
    )
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    provider_handle_retained: bool
    unselected_fallback_masks_materialized: bool
    legacy_signed_work_candidate_reconstruction_claim: bool
    schema: str = LABEL_NATIVE_CONTROLLER_CANDIDATE_SCHEMA
    _candidate_id: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _CANDIDATE_TOKEN:
            raise TypeError("label-native controller candidates are issuer-only")
        expected = {
            item.name for item in fields(type(self)) if item.init and item.name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "label-native controller candidate fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", LABEL_NATIVE_CONTROLLER_CANDIDATE_SCHEMA)
        object.__setattr__(self, "_candidate_id", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        plan_hash = _identity(self.runtime_plan_hash, name="runtime plan hash")
        index = _nonnegative_int(self.plan_queue_index, name="plan queue index")
        count = _positive_int(
            self.queue_descriptor_count,
            name="queue descriptor count",
        )
        axis_hash = _identity(
            self.ordered_descriptor_axis_hash,
            name="ordered descriptor axis hash",
        )
        if type(self.descriptor) is not V1LabelNativeRuntimeFallbackDescriptor:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller candidate requires one exact runtime descriptor",
            )
        if type(self.selected_endpoint) is not V1LabelNativeRuntimeSelectedEndpointProposal:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller candidate requires one exact selected endpoint",
            )
        try:
            descriptor_hash = self.descriptor.assert_integrity()
            materialization_hash = self.selected_endpoint.assert_integrity()
        except (TypeError, ValueError) as error:
            raise V1LabelNativeControllerAcquisitionRefusal(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                f"candidate source no longer replays: {error}",
            ) from error
        receipt = self.selected_endpoint.receipt
        descriptor = self.descriptor
        if (
            index >= count
            or descriptor.queue_index != index
            or receipt.runtime_plan_hash != plan_hash
            or receipt.runtime_descriptor_hash != descriptor_hash
            or receipt.descriptor_index != index
            or receipt.queue_descriptor_count != count
            or receipt.ordered_descriptor_axis_hash != axis_hash
            or receipt.selected_coordinate_hash != descriptor.coordinate_hash
            or receipt.canonical_physical_identity_hash
            != descriptor.canonical_physical_identity_hash
            or receipt.crosswalk_receipt_hash != descriptor.crosswalk.receipt_hash
            or receipt.source_action_receipt_hash
            != descriptor.source_action_receipt_hash
            or receipt.qualified_lane_registration_hash
            != descriptor.qualified_lane_registration_hash
            or receipt.compact_provider_endpoint_label
            != descriptor.compact_provider_endpoint_label
            or receipt.source_endpoint_label != descriptor.source_endpoint_label
            or receipt.selected_endpoint_count != descriptor.cumulative_removed_count
            or receipt.selected_endpoint_action_hash
            != descriptor.source_action_receipt.packed_action_hash
            or receipt.packed_action_hash != descriptor.coordinate.action_hash
            or receipt.resulting_mask_hash
            != descriptor.source_action_receipt.resulting_mask_hash
            or receipt.removed_route_index_hash
            != descriptor.source_action_receipt.removed_route_index_hash
            or materialization_hash != receipt.receipt_hash
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.MATERIALIZATION_MISMATCH,
                "selected materialization differs from its exact plan descriptor",
            )
        _assert_zero_authority(self.selected_endpoint)
        required_true = (
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
        )
        required_false = (
            self.candidate_freeze_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.bce_oracle_authority,
            self.irreversible_decision_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.execution_authority,
            self.provider_handle_retained,
            self.unselected_fallback_masks_materialized,
            self.legacy_signed_work_candidate_reconstruction_claim,
        )
        if (
            any(type(value) is not bool or not value for value in required_true)
            or any(type(value) is not bool or value for value in required_false)
            or self.schema != LABEL_NATIVE_CONTROLLER_CANDIDATE_SCHEMA
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "label-native controller candidate crossed its proposal firewall",
            )
        object.__setattr__(self, "runtime_plan_hash", plan_hash)
        object.__setattr__(self, "plan_queue_index", index)
        object.__setattr__(self, "queue_descriptor_count", count)
        object.__setattr__(self, "ordered_descriptor_axis_hash", axis_hash)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "runtime_plan_hash": self.runtime_plan_hash,
            "plan_queue_index": self.plan_queue_index,
            "queue_descriptor_count": self.queue_descriptor_count,
            "ordered_descriptor_axis_hash": self.ordered_descriptor_axis_hash,
            "runtime_descriptor_hash": self.descriptor.descriptor_hash,
            "materialization_receipt_hash": self.selected_endpoint.proposal_hash,
            "selected_coordinate_hash": self.descriptor.coordinate_hash,
            "certificate_hash": self.descriptor.certificate_hash,
            "canonical_physical_identity_hash": (
                self.descriptor.canonical_physical_identity_hash
            ),
            "crosswalk_receipt_hash": self.descriptor.crosswalk.receipt_hash,
            "source_action_receipt_hash": self.descriptor.source_action_receipt_hash,
            "qualified_lane_registration_hash": (
                self.descriptor.qualified_lane_registration_hash
            ),
            "compact_provider_endpoint_label": (
                self.descriptor.compact_provider_endpoint_label
            ),
            "source_endpoint_label": self.descriptor.source_endpoint_label,
            "selected_endpoint_count": self.descriptor.cumulative_removed_count,
            "selected_endpoint_action_hash": self.packed_action_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "removed_route_index_hash": self.removed_route_index_hash,
            "six_role_C_design_hash": self.descriptor.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": (
                self.descriptor.six_role_C_progress_receipt_hash
            ),
            "six_role_certification_receipt_hash": (
                self.descriptor.six_role_certification_receipt_hash
            ),
            "six_role_decision_hash": self.descriptor.six_role_decision_hash,
            "fallback_authorization_hash": self.descriptor.fallback_authorization_hash,
            "proposal_only": self.proposal_only,
            "proposal_witness_only": self.proposal_witness_only,
            "fresh_relinearization_required": self.fresh_relinearization_required,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
            "provider_handle_retained": self.provider_handle_retained,
            "unselected_fallback_masks_materialized": (
                self.unselected_fallback_masks_materialized
            ),
            "legacy_signed_work_candidate_reconstruction_claim": (
                self.legacy_signed_work_candidate_reconstruction_claim
            ),
        }

    @property
    def candidate_id(self) -> str:
        return self.assert_integrity()

    @property
    def proposal_id(self) -> str:
        return self.candidate_id

    @property
    def proposal_hash(self) -> str:
        return self.candidate_id

    @property
    def queue_index(self) -> int:
        return self.plan_queue_index

    @property
    def parent_mask(self) -> HardMask:
        return self.selected_endpoint.parent_mask

    @property
    def parent_mask_hash(self) -> str:
        return self.selected_endpoint.receipt.live_current_mask_hash

    @property
    def packed_action(self) -> V1PackedRemovalAction:
        return self.selected_endpoint.packed_action

    @property
    def packed_action_hash(self) -> str:
        return self.selected_endpoint.packed_action_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.selected_endpoint.resulting_mask

    @property
    def resulting_mask_hash(self) -> str:
        return self.selected_endpoint.resulting_mask_hash

    @property
    def removed_count(self) -> int:
        return self.packed_action.removed_count

    @property
    def removed_route_index_hash(self) -> str:
        return self.packed_action.removed_route_index_hash

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return V1RemovedRouteSetIdentity(
            route_count=self.parent_mask.route_count,
            count=self.removed_count,
            index_hash=self.removed_route_index_hash,
        )

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    @property
    def selected_coordinate_hash(self) -> str:
        return self.descriptor.coordinate_hash

    @property
    def resource_gain(self) -> float:
        return float(self.descriptor.cumulative_removed_count)

    def resolve_removed_routes(self, parent_mask: HardMask) -> V1RemovedRouteSetIdentity:
        if _fresh_mask_hash(parent_mask, name="candidate parent mask") != self.parent_mask_hash:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.STALE_CURRENT_MASK,
                "candidate parent differs from its controller acquisition",
            )
        resolved = self.resulting_mask.removed_index_set_from(parent_mask)
        if (
            resolved.count != self.removed_count
            or resolved.index_hash != self.removed_route_index_hash
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "candidate delta differs from its packed action identity",
            )
        return resolved

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        observed = content_hash(self._payload(), prefix=_CANDIDATE_PREFIX)
        if self._candidate_id is not None and observed != self._candidate_id:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "label-native controller candidate was mutated after issuance",
            )
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected controller candidate hash",
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "controller candidate hash differs",
            )
        object.__setattr__(self, "_candidate_id", observed)
        return observed

    def assert_can_freeze_candidate(self) -> NoReturn:
        raise PermissionError(
            "label-native controller carriers need the explicit candidate-freeze gate"
        )

    def assert_can_open_calibration_K(self) -> NoReturn:
        raise PermissionError("label-native controller carriers cannot open calibration K")

    def assert_can_open_paired_oracle_O(self) -> NoReturn:
        raise PermissionError("label-native controller carriers cannot open paired oracle O")

    def assert_can_commit_mask(self) -> NoReturn:
        raise PermissionError("label-native controller carriers cannot commit a hard mask")

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native controller candidates are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native controller candidates are nonserializable")


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeControllerAcquisitionReceipt:
    runtime_plan_hash: str
    runtime_descriptor_hash: str
    plan_queue_index: int
    queue_descriptor_count: int
    ordered_descriptor_axis_hash: str
    selected_materialization_receipt_hash: str
    controller_candidate_id: str
    campaign_identity_hash: str
    campaign_binding_hash: str
    campaign_contract_hash: str
    campaign_registration_hash: str
    live_chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    controller_mask_lineage_hash: str
    selected_coordinate_hash: str
    certificate_hash: str
    canonical_physical_identity_hash: str
    crosswalk_receipt_hash: str
    source_action_receipt_hash: str
    qualified_lane_registration_hash: str
    compact_provider_endpoint_label: int
    source_endpoint_label: int
    selected_endpoint_count: int
    selected_endpoint_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    production_cuda_materialization_replayed: bool
    private_reference_materialization_replayed: bool
    exact_plan_descriptor_materialization_replayed: bool
    exact_campaign_binding_replayed: bool
    exact_current_mask_lineage_replayed: bool
    exactly_one_endpoint_carried: bool
    provider_handle_retained: bool
    unselected_fallback_masks_materialized: bool
    proposal_only: bool
    paired_bce_oracle_still_required: bool
    candidate_freeze_authority: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    authority_scope: str
    schema: str = LABEL_NATIVE_CONTROLLER_ACQUISITION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("label-native controller acquisition receipts are issuer-only")
        expected = {
            item.name for item in fields(type(self)) if item.init and item.name != "schema"
        }
        if set(values) != expected:
            raise TypeError(
                "controller acquisition receipt fields differ: "
                f"missing={sorted(expected.difference(values))}, "
                f"extra={sorted(set(values).difference(expected))}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(
            self,
            "schema",
            LABEL_NATIVE_CONTROLLER_ACQUISITION_RECEIPT_SCHEMA,
        )
        object.__setattr__(self, "_receipt_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        for name in (
            "runtime_plan_hash",
            "runtime_descriptor_hash",
            "ordered_descriptor_axis_hash",
            "selected_materialization_receipt_hash",
            "controller_candidate_id",
            "campaign_identity_hash",
            "campaign_binding_hash",
            "campaign_contract_hash",
            "campaign_registration_hash",
            "live_chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "controller_mask_lineage_hash",
            "selected_coordinate_hash",
            "certificate_hash",
            "canonical_physical_identity_hash",
            "crosswalk_receipt_hash",
            "source_action_receipt_hash",
            "qualified_lane_registration_hash",
            "selected_endpoint_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        index = _nonnegative_int(self.plan_queue_index, name="plan queue index")
        count = _positive_int(self.queue_descriptor_count, name="queue descriptor count")
        if index >= count:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.PLAN_DESCRIPTOR_MISMATCH,
                "controller acquisition queue index left the frozen plan",
            )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        for name in (
            "compact_provider_endpoint_label",
            "source_endpoint_label",
            "selected_endpoint_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        lineage, transitions, lineage_hash = _lineage(
            current_mask_hash=self.current_mask_hash,
            mask_hashes=self.committed_mask_lineage_hashes,
            transition_hashes=self.committed_mask_transition_receipt_hashes,
        )
        if lineage_hash != self.controller_mask_lineage_hash:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.LINEAGE_MISMATCH,
                "controller lineage hash does not replay its exact ancestry",
            )
        required_true = (
            self.exact_plan_descriptor_materialization_replayed,
            self.exact_campaign_binding_replayed,
            self.exact_current_mask_lineage_replayed,
            self.exactly_one_endpoint_carried,
            self.proposal_only,
            self.paired_bce_oracle_still_required,
        )
        required_false = (
            self.provider_handle_retained,
            self.unselected_fallback_masks_materialized,
            self.candidate_freeze_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.irreversible_decision_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
            self.execution_authority,
        )
        if (
            any(type(value) is not bool or not value for value in required_true)
            or any(type(value) is not bool or value for value in required_false)
            or type(self.production_cuda_materialization_replayed) is not bool
            or type(self.private_reference_materialization_replayed) is not bool
            or self.production_cuda_materialization_replayed
            is self.private_reference_materialization_replayed
            or self.authority_scope != LABEL_NATIVE_CONTROLLER_ACQUISITION_AUTHORITY
            or self.schema != LABEL_NATIVE_CONTROLLER_ACQUISITION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "controller acquisition crossed its single-proposal authority",
            )
        object.__setattr__(self, "plan_queue_index", index)
        object.__setattr__(self, "queue_descriptor_count", count)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(
            self,
            "committed_mask_transition_receipt_hashes",
            transitions,
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(type(self))
            if not item.name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "controller acquisition receipt was mutated after issuance",
            )
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected controller acquisition receipt hash",
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "controller acquisition receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native controller acquisition receipts are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native controller acquisition receipts are nonserializable")


@dataclass(frozen=True, slots=True, init=False)
class V1LabelNativeControllerSingleEndpointAcquisition:
    """One campaign/lineage-bound controller input; never a decision authority."""

    binding: V1OptionACampaignBinding
    current_mask: HardMask = field(repr=False, compare=False)
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    candidate: V1LabelNativeControllerCandidateCarrier
    receipt: V1LabelNativeControllerAcquisitionReceipt
    schema: str = LABEL_NATIVE_CONTROLLER_ACQUISITION_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _ACQUISITION_TOKEN:
            raise TypeError("label-native controller acquisitions are issuer-only")
        expected = {
            item.name for item in fields(type(self)) if item.init and item.name != "schema"
        }
        if set(values) != expected:
            raise TypeError("label-native controller acquisition fields differ")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", LABEL_NATIVE_CONTROLLER_ACQUISITION_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller acquisition requires one exact Option-A binding",
            )
        if type(self.current_mask) is not HardMask:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller acquisition requires one exact current mask",
            )
        if type(self.candidate) is not V1LabelNativeControllerCandidateCarrier:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller acquisition requires one exact candidate carrier",
            )
        if type(self.receipt) is not V1LabelNativeControllerAcquisitionReceipt:
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.INVALID_TYPE,
                "controller acquisition requires its exact receipt",
            )
        candidate_id = self.candidate.assert_integrity()
        receipt_hash = self.receipt.assert_integrity()
        current_hash = _fresh_mask_hash(self.current_mask, name="controller current mask")
        lineage, transitions, lineage_hash = _lineage(
            current_mask_hash=current_hash,
            mask_hashes=self.committed_mask_lineage_hashes,
            transition_hashes=self.committed_mask_transition_receipt_hashes,
        )
        receipt = self.receipt
        if (
            self.binding.binding_hash != receipt.campaign_binding_hash
            or self.binding.contract.campaign_contract_hash
            != receipt.campaign_contract_hash
            or self.binding.registration.registration_hash
            != receipt.campaign_registration_hash
            or self.binding.contract.route_axis_hash != receipt.route_axis_hash
            or current_hash != receipt.current_mask_hash
            or self.current_mask.route_axis_hash != receipt.route_axis_hash
            or self.current_mask.route_count != receipt.route_count
            or lineage != receipt.committed_mask_lineage_hashes
            or transitions != receipt.committed_mask_transition_receipt_hashes
            or lineage_hash != receipt.controller_mask_lineage_hash
            or candidate_id != receipt.controller_candidate_id
            or self.candidate.runtime_plan_hash != receipt.runtime_plan_hash
            or self.candidate.plan_queue_index != receipt.plan_queue_index
            or self.candidate.parent_mask_hash != current_hash
            or self.candidate.resulting_mask_hash != receipt.resulting_mask_hash
            or receipt_hash != self.receipt.receipt_hash
            or self.schema != LABEL_NATIVE_CONTROLLER_ACQUISITION_SCHEMA
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
                "controller acquisition no longer replays its exact frame",
            )
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(
            self,
            "committed_mask_transition_receipt_hashes",
            transitions,
        )

    @property
    def acquisition_hash(self) -> str:
        return self.assert_integrity()

    @property
    def generation_index(self) -> int:
        return self.receipt.proposal_generation

    @property
    def invalidation_chart_hash(self) -> str:
        return self.receipt.live_chart_identity_hash

    @property
    def queue(self) -> tuple[V1LabelNativeControllerCandidateCarrier, ...]:
        return (self.candidate,)

    @property
    def selected_proposal(self) -> V1LabelNativeControllerCandidateCarrier:
        return self.candidate

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        return self.receipt.assert_integrity(expected_hash)

    def assert_can_freeze_candidate(self) -> NoReturn:
        raise PermissionError(
            "label-native controller acquisitions need the explicit candidate-freeze gate"
        )

    def assert_can_open_calibration_K(self) -> NoReturn:
        raise PermissionError("label-native controller acquisitions cannot open calibration K")

    def assert_can_open_paired_oracle_O(self) -> NoReturn:
        raise PermissionError("label-native controller acquisitions cannot open paired oracle O")

    def assert_can_commit_mask(self) -> NoReturn:
        raise PermissionError("label-native controller acquisitions cannot commit a hard mask")

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native controller acquisitions are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native controller acquisitions are nonserializable")


def _acquire_selected(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    materialized_endpoint: V1LabelNativeRuntimeSelectedEndpointProposal,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    production: bool,
) -> V1LabelNativeControllerSingleEndpointAcquisition:
    if type(plan) is not V1LabelNativeRuntimeFallbackPlan:
        raise TypeError("controller acquisition requires the exact runtime plan")
    if type(materialized_endpoint) is not V1LabelNativeRuntimeSelectedEndpointProposal:
        raise TypeError("controller acquisition requires one exact selected endpoint")
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("controller acquisition requires one exact Option-A binding")
    if type(current_mask) is not HardMask:
        raise TypeError("controller acquisition requires one exact current mask")
    if type(production) is not bool:
        raise TypeError("controller acquisition production flag must be exact")
    try:
        plan_hash = plan.assert_integrity()
        materialization_hash = materialized_endpoint.assert_integrity()
    except (TypeError, ValueError) as error:
        raise V1LabelNativeControllerAcquisitionRefusal(
            V1LabelNativeControllerAcquisitionRefusalCode.TAMPERED,
            f"controller acquisition source no longer replays: {error}",
        ) from error
    index = _nonnegative_int(descriptor_index, name="descriptor index")
    if index >= len(plan.descriptors):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.PLAN_DESCRIPTOR_MISMATCH,
            "descriptor index is outside the frozen fallback plan",
        )
    descriptor = plan.descriptors[index]
    descriptor.assert_integrity(plan.ordered_descriptor_hashes[index])
    queue_axis_hash = _queue_axis_hash(plan)
    selected_receipt = materialized_endpoint.receipt
    if (
        selected_receipt.runtime_plan_hash != plan_hash
        or selected_receipt.runtime_descriptor_hash != descriptor.descriptor_hash
        or selected_receipt.descriptor_index != index
        or selected_receipt.queue_descriptor_count != len(plan.descriptors)
        or selected_receipt.ordered_descriptor_axis_hash != queue_axis_hash
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.PLAN_DESCRIPTOR_MISMATCH,
            "selected endpoint is not the named frozen plan descriptor",
        )
    if production:
        if (
            not selected_receipt.production_cuda_execution
            or selected_receipt.private_reference_execution
            or not selected_receipt.exact_planned_provider_receipt_replayed
            or selected_receipt.backend_kind != "torch_cuda"
        ):
            _refuse(
                V1LabelNativeControllerAcquisitionRefusalCode.PRODUCTION_MATERIALIZATION_REQUIRED,
                "production controller acquisition requires exact plan-bound CUDA materialization",
            )
    elif (
        selected_receipt.production_cuda_execution
        or not selected_receipt.private_reference_execution
        or selected_receipt.exact_planned_provider_receipt_replayed
        or selected_receipt.backend_kind != "numpy_reference"
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.PRODUCTION_MATERIALIZATION_REQUIRED,
            "private controller reference requires the route-limited reference materializer",
        )
    if (
        plan.campaign_identity_hash != binding.binding_hash
        or plan.campaign_contract_hash != binding.contract.campaign_contract_hash
        or plan.campaign_registration_hash != binding.registration.registration_hash
        or selected_receipt.campaign_identity_hash != binding.binding_hash
        or selected_receipt.campaign_contract_hash
        != binding.contract.campaign_contract_hash
        or selected_receipt.campaign_registration_hash
        != binding.registration.registration_hash
        or binding.contract.route_axis_hash != plan.live_route_axis_hash
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.CAMPAIGN_BINDING_MISMATCH,
            "runtime plan/materialization differ from the exact Option-A binding",
        )
    current_hash = _fresh_mask_hash(current_mask, name="controller current mask")
    materialized_parent_hash = _fresh_mask_hash(
        materialized_endpoint.parent_mask,
        name="materialized parent mask",
    )
    if (
        current_hash != plan.live_current_mask_hash
        or current_hash != selected_receipt.live_current_mask_hash
        or current_hash != materialized_parent_hash
        or current_mask.route_axis_hash != plan.live_route_axis_hash
        or current_mask.route_count != plan.live_route_count
        or selected_receipt.live_chart_identity_hash != plan.live_chart_identity_hash
        or selected_receipt.live_route_axis_hash != plan.live_route_axis_hash
        or selected_receipt.live_route_count != plan.live_route_count
        or selected_receipt.live_proposal_generation
        != plan.live_proposal_generation
    ):
        _refuse(
            V1LabelNativeControllerAcquisitionRefusalCode.STALE_CURRENT_MASK,
            "current mask/chart/route/generation left the frozen plan frame",
        )
    lineage, transitions, lineage_hash = _lineage(
        current_mask_hash=current_hash,
        mask_hashes=committed_mask_lineage_hashes,
        transition_hashes=committed_mask_transition_receipt_hashes,
    )
    _assert_zero_authority(materialized_endpoint)
    candidate = V1LabelNativeControllerCandidateCarrier(
        _issuer_token=_CANDIDATE_TOKEN,
        runtime_plan_hash=plan_hash,
        plan_queue_index=index,
        queue_descriptor_count=len(plan.descriptors),
        ordered_descriptor_axis_hash=queue_axis_hash,
        descriptor=descriptor,
        selected_endpoint=materialized_endpoint,
        proposal_only=True,
        proposal_witness_only=True,
        fresh_relinearization_required=True,
        paired_bce_oracle_required=True,
        candidate_freeze_authority=False,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        bce_oracle_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
        provider_handle_retained=False,
        unselected_fallback_masks_materialized=False,
        legacy_signed_work_candidate_reconstruction_claim=False,
    )
    candidate_id = candidate.candidate_id
    receipt = V1LabelNativeControllerAcquisitionReceipt(
        _issuer_token=_RECEIPT_TOKEN,
        runtime_plan_hash=plan_hash,
        runtime_descriptor_hash=descriptor.descriptor_hash,
        plan_queue_index=index,
        queue_descriptor_count=len(plan.descriptors),
        ordered_descriptor_axis_hash=queue_axis_hash,
        selected_materialization_receipt_hash=materialization_hash,
        controller_candidate_id=candidate_id,
        campaign_identity_hash=plan.campaign_identity_hash,
        campaign_binding_hash=binding.binding_hash,
        campaign_contract_hash=binding.contract.campaign_contract_hash,
        campaign_registration_hash=binding.registration.registration_hash,
        live_chart_identity_hash=plan.live_chart_identity_hash,
        current_mask_hash=current_hash,
        route_axis_hash=plan.live_route_axis_hash,
        route_count=plan.live_route_count,
        proposal_generation=plan.live_proposal_generation,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        controller_mask_lineage_hash=lineage_hash,
        selected_coordinate_hash=descriptor.coordinate_hash,
        certificate_hash=descriptor.certificate_hash,
        canonical_physical_identity_hash=(
            descriptor.canonical_physical_identity_hash
        ),
        crosswalk_receipt_hash=descriptor.crosswalk.receipt_hash,
        source_action_receipt_hash=descriptor.source_action_receipt_hash,
        qualified_lane_registration_hash=descriptor.qualified_lane_registration_hash,
        compact_provider_endpoint_label=descriptor.compact_provider_endpoint_label,
        source_endpoint_label=descriptor.source_endpoint_label,
        selected_endpoint_count=descriptor.cumulative_removed_count,
        selected_endpoint_action_hash=materialized_endpoint.packed_action_hash,
        resulting_mask_hash=materialized_endpoint.resulting_mask_hash,
        removed_route_index_hash=(
            materialized_endpoint.packed_action.removed_route_index_hash
        ),
        production_cuda_materialization_replayed=production,
        private_reference_materialization_replayed=not production,
        exact_plan_descriptor_materialization_replayed=True,
        exact_campaign_binding_replayed=True,
        exact_current_mask_lineage_replayed=True,
        exactly_one_endpoint_carried=True,
        provider_handle_retained=False,
        unselected_fallback_masks_materialized=False,
        proposal_only=True,
        paired_bce_oracle_still_required=True,
        candidate_freeze_authority=False,
        calibration_K_open_authority=False,
        paired_oracle_O_open_authority=False,
        irreversible_decision_authority=False,
        mask_commit_authority=False,
        terminal_authority=False,
        pareto_authority=False,
        execution_authority=False,
        authority_scope=LABEL_NATIVE_CONTROLLER_ACQUISITION_AUTHORITY,
    )
    return V1LabelNativeControllerSingleEndpointAcquisition(
        _issuer_token=_ACQUISITION_TOKEN,
        binding=binding,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        candidate=candidate,
        receipt=receipt,
    )


def acquire_v1_signed_work_label_native_controller_endpoint(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    materialized_endpoint: V1LabelNativeRuntimeSelectedEndpointProposal,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
) -> V1LabelNativeControllerSingleEndpointAcquisition:
    """Issue one production controller input from an exact CUDA endpoint."""

    return _acquire_selected(
        plan=plan,
        descriptor_index=descriptor_index,
        materialized_endpoint=materialized_endpoint,
        binding=binding,
        current_mask=current_mask,
        committed_mask_lineage_hashes=committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            committed_mask_transition_receipt_hashes
        ),
        production=True,
    )


def _acquire_v1_signed_work_label_native_controller_endpoint_reference_for_tests(
    *,
    plan: V1LabelNativeRuntimeFallbackPlan,
    descriptor_index: int,
    materialized_endpoint: V1LabelNativeRuntimeSelectedEndpointProposal,
    binding: V1OptionACampaignBinding,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
) -> V1LabelNativeControllerSingleEndpointAcquisition:
    """Route-limited private reference path; never production authority."""

    return _acquire_selected(
        plan=plan,
        descriptor_index=descriptor_index,
        materialized_endpoint=materialized_endpoint,
        binding=binding,
        current_mask=current_mask,
        committed_mask_lineage_hashes=committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            committed_mask_transition_receipt_hashes
        ),
        production=False,
    )


__all__ = [
    "LABEL_NATIVE_CONTROLLER_ACQUISITION_AUTHORITY",
    "LABEL_NATIVE_CONTROLLER_ACQUISITION_RECEIPT_SCHEMA",
    "LABEL_NATIVE_CONTROLLER_ACQUISITION_SCHEMA",
    "LABEL_NATIVE_CONTROLLER_CANDIDATE_SCHEMA",
    "V1LabelNativeControllerAcquisitionReceipt",
    "V1LabelNativeControllerAcquisitionRefusal",
    "V1LabelNativeControllerAcquisitionRefusalCode",
    "V1LabelNativeControllerCandidateCarrier",
    "V1LabelNativeControllerSingleEndpointAcquisition",
    "acquire_v1_signed_work_label_native_controller_endpoint",
]
