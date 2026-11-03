"""Preoutcome segmented literal-J view for the current-source D2 look.

The D2 coordinator already has the complete registered D/W2 graph before any
provider row exists.  This module freezes that graph as a literal-J logical
source, then binds it after execution to the exact literal-J shards published
by the shared acquisition.  It never calls the provider, materializes route
rows, or issues D-stability/KKT/C/K/O/commit authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupCommit,
    V1SampledDualChartRowGroupPacketPlan,
    V1SampledDualChartRowGroupPacketProofClosure,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_v7_current_source_d2_execution import (
    SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY,
    V1SampledV7CurrentSourceD2ExecutionResult,
    V1SampledV7CurrentSourceD2GraphCoordinator,
)
from .modern_v1_sampled_v7_current_source_d2_preoutcome_plan import (
    V1SampledV7CurrentSourceD2PreoutcomePlan,
)
from .modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyLogicalLookMeasurementIssuance,
    V1SampledV7DOnlyPacketMeasurementIssuance,
    v1_sampled_v7_d_only_transport_packet_issuances,
)
from .modern_v1_sampled_v7_shared_dw2_acquisition import (
    V1SampledV7SharedDW2AcquisitionPlan,
    V1SampledV7SharedDW2AcquisitionReceipt,
    V1SampledV7SharedDW2AcquisitionResult,
    v1_sampled_v7_nonempty_physical_roles,
)


SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_literal_J_logical_look_source_v2"
)
SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_RESULT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_literal_J_logical_look_result_v1"
)
SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY: Final = (
    "preoutcome_zero_copy_segmented_literal_J_source_view_over_current_source_D2_"
    "rows_only_no_provider_execution_literal_J_direction_Loewner_KKT_D_stability_"
    "C_K_O_commit_validation_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-shared-D-W2-literal-J-logical-look-"
_PREOUTCOME_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()
_ROLE_ORDER: Final = ("target", "complement")


class V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PREOUTCOME_REQUIRED = "preoutcome_required"
    OUTCOME_REQUIRED = "outcome_required"
    BIND_REPLAY = "bind_replay"
    CLOSURE_MISMATCH = "closure_mismatch"
    ROLE_MISMATCH = "role_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7SharedDW2LiteralJLogicalLookRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode:
            raise TypeError("literal-J logical-look refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7SharedDW2LiteralJLogicalLookRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            f"{name} must be one nonempty identity",
        )
    return value


def _authority_payload() -> dict[str, bool | str]:
    return {
        "provider_execution_authority": False,
        "literal_j_direction_authority": False,
        "loewner_or_kkt_authority": False,
        "d_stability_authority": False,
        "c_k_o_authority": False,
        "mask_commit_authority": False,
        "validation_authority": False,
        "terminal_or_pareto_authority": False,
        "authority_scope": SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY,
    }


def _closed(value: object) -> bool:
    return all(
        getattr(value, key, None) is expected
        for key, expected in _authority_payload().items()
    )


def _hash_attr(value: object, names: tuple[str, ...], *, name: str) -> str:
    for attr in names:
        candidate = getattr(value, attr, None)
        if type(candidate) is str and candidate:
            return candidate
    method = getattr(value, "assert_live", None)
    if callable(method):
        candidate = method()
        if type(candidate) is str and candidate:
            return candidate
    method = getattr(value, "assert_integrity", None)
    if callable(method):
        candidate = method()
        if type(candidate) is str and candidate:
            return candidate
    _refuse(
        V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
        f"{name} lacks a stable identity",
    )


def _validate_preoutcome_graph(
    *,
    coordinator: V1SampledV7CurrentSourceD2GraphCoordinator,
    acquisition: V1SampledV7SharedDW2AcquisitionPlan,
    d2_plan: V1SampledV7CurrentSourceD2PreoutcomePlan,
    group_plan: V1SampledDualChartRowGroupPlan,
    transport: tuple[V1SampledV7DOnlyPacketMeasurementIssuance, ...],
) -> None:
    """Replay every source-owned axis before minting the inert receipt."""

    packets = tuple(item._packet for item in transport)
    scheduled = tuple(
        coordinator.measurement_owner.schedule.packet_for(packet)
        for packet in packets
    )
    packet_hashes = tuple(item.receipt_hash for item in packets)
    schedule_hashes = tuple(item.packet_schedule_hash for item in scheduled)
    physical: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    transport_call_counts: list[int] = []
    for packet, schedule in zip(packets, scheduled, strict=True):
        target = tuple(packet.target_context_ids)
        complement = tuple(packet.complement_context_ids)
        roles = v1_sampled_v7_nonempty_physical_roles(
            target_context_ids=target,
            complement_context_ids=complement,
        )
        transport_call_counts.append(len(roles))
        draws = tuple(schedule.registered_draw_ids)
        if len(draws) != len(target) + len(complement):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
                "transport schedule and context axes differ",
            )
        by_role = {
            "target": draws[: len(target)],
            "complement": draws[len(target) :],
        }
        physical.extend((role, contexts, by_role[role]) for role, contexts in roles)
    packet_plans = tuple(group_plan.packet_plans)
    if (
        acquisition._packet_issuance is not coordinator.packet_issuance
        or acquisition._owner is not coordinator.measurement_owner
        or any(item._owner is not coordinator.measurement_owner for item in transport)
        or packet_hashes != acquisition.transport_packet_hashes
        or packet_hashes != d2_plan.transport_packet_hashes
        or schedule_hashes != acquisition.transport_schedule_packet_hashes
        or schedule_hashes != d2_plan.transport_schedule_packet_hashes
        or tuple(transport_call_counts) != acquisition.transport_physical_call_counts
        or tuple(item[0] for item in physical) != acquisition.physical_roles
        or tuple(len(item[1]) for item in physical)
        != acquisition.physical_context_counts
        or len(packet_plans) != len(physical)
        or acquisition.logical_registered_look_index
        != d2_plan.registered_look_index
        or acquisition.logical_registered_look_stratum_counts
        != d2_plan.registered_look_stratum_counts
        or acquisition.logical_residual_before_stratum_counts
        != d2_plan.logical_residual_before_stratum_counts
        or acquisition.logical_residual_after_stratum_counts
        != d2_plan.logical_residual_after_stratum_counts
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            "D2, logical-look, transport, and physical row axes diverged",
        )
    if (
        d2_plan.suffix_registration_hash
        != coordinator.d1_prefix.suffix_registration_hash
        or any(
            packet.suffix_registration_hash != d2_plan.suffix_registration_hash
            for packet in packets
        )
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            "transport packets no longer belong to the registered D suffix",
        )
    requests = tuple(acquisition.measurement_requests)
    if len(requests) != len(packet_plans):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            "physical request and row-group packet axes differ",
        )
    for ordinal, (packet_plan, request, axes) in enumerate(
        zip(packet_plans, requests, physical, strict=True)
    ):
        role, contexts, draws = axes
        if (
            type(packet_plan) is not V1SampledDualChartRowGroupPacketPlan
            or packet_plan.packet_ordinal != ordinal
            or packet_plan.context_ids != contexts
            or packet_plan.draw_ids != draws
            or packet_plan.stratum_labels != (role,) * len(contexts)
            or packet_plan.provider_plan_identity_hash != request.request_id
            or packet_plan.source_capture_identity_hash
            != request.payload.get("source_capture_identity_hash")
            or len(packet_plan.reducer_plan_identity_hashes) != 2
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
                "physical role, capture, or reducer order differs from the row-group plan",
            )
    prior = acquisition.prior_row_prefix
    prior_hash = None if prior is None else prior.manifest.manifest_hash
    prior_count = 0 if prior is None else len(prior.manifest.looks)
    if (
        group_plan.first_prefix_look_ordinal != acquisition.physical_call_offset
        or group_plan.expected_prior_manifest_hash != prior_hash
        or group_plan.registered_d_prefix_hash
        != acquisition.raw_row_prefix_identity_hash
        or prior_count != acquisition.physical_call_offset
        or (acquisition.physical_call_offset == 0) is not (prior is None)
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            "predecessor does not end at the exact current-look offset",
        )
    lineage = coordinator.measurement_owner.measurement_source_lineage
    if (
        group_plan.chart_identity_hash != lineage.chart_identity_hash
        or group_plan.current_mask_hash != lineage.current_mask_hash
        or group_plan.proposal_generation != lineage.proposal_generation
        or group_plan.route_axis_hash != lineage.route_axis_hash
        or group_plan.route_id_axis_hash != lineage.route_id_axis_hash
        or group_plan.literal_j_stream_identity_hash
        != lineage.literal_j_stream_source_hash
        or group_plan.literal_j_stream_identity_hash
        != coordinator.measurement_owner.residual_literal_j_stream_source_hash
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
            "chart, route, or literal-J stream identity changed",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2LiteralJLogicalLookPreoutcomeReceipt:
    coordinator_hash: str
    d2_preoutcome_plan_hash: str
    acquisition_plan_hash: str
    measurement_owner_hash: str
    suffix_registration_hash: str
    packet_issuance_hash: str
    logical_registered_look_index: int
    logical_registered_look_stratum_counts: tuple[int, int]
    logical_residual_before_stratum_counts: tuple[int, int]
    logical_residual_after_stratum_counts: tuple[int, int]
    transport_packet_hashes: tuple[str, ...]
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_physical_call_counts: tuple[int, ...]
    physical_roles: tuple[str, ...]
    physical_context_counts: tuple[int, ...]
    row_group_plan_hash: str
    row_group_packet_plan_hashes: tuple[str, ...]
    physical_call_offset: int
    expected_prior_manifest_hash: str | None
    raw_row_prefix_identity_hash: str
    logical_source_hash: str
    generalized_logical_prefix_identity_hash: str
    source_capture_identity_hashes: tuple[str, ...]
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    literal_j_stream_identity_hash: str
    provider_outcomes_seen: bool = False
    provider_execution_authority: bool = False
    literal_j_direction_authority: bool = False
    loewner_or_kkt_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    validation_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PREOUTCOME_TOKEN:
            raise PermissionError("literal-J logical-look receipts are source-issued")
        if (
            len(self.physical_roles) < 1
            or len(self.physical_roles) != len(self.physical_context_counts)
            or len(self.row_group_packet_plan_hashes) != len(self.physical_roles)
            or len(self.source_capture_identity_hashes) != len(self.physical_roles)
            or sum(self.transport_physical_call_counts) != len(self.physical_roles)
            or any(role not in _ROLE_ORDER for role in self.physical_roles)
            or self.provider_outcomes_seen is not False
            or not _closed(self)
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AUTHORITY_VIOLATION,
                "literal-J logical-look receipt is partial or widened authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "preoutcome-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AXIS_MISMATCH,
                "literal-J logical-look receipt was mutated",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2LiteralJStratumView:
    source_receipt_hash: str
    result_receipt_hash: str
    role: str
    packet_ordinals: tuple[int, ...]
    attachment_indices: tuple[int, ...]
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    packet_plan_hashes: tuple[str, ...]
    proof_closure_hashes: tuple[str, ...]
    source_capture_identity_hashes: tuple[str, ...]
    shard_receipt_hashes: tuple[str, ...]
    shards: tuple[V1SampledDualChartRowShardAttachment, ...] = field(
        repr=False,
        compare=False,
    )
    provider_execution_authority: bool = False
    literal_j_direction_authority: bool = False
    loewner_or_kkt_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    validation_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.role not in _ROLE_ORDER or not self.packet_ordinals:
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.ROLE_MISMATCH,
                "stratum view role is invalid or empty",
            )
        if (
            len(self.packet_ordinals) != len(self.attachment_indices)
            or len(self.shards) != len(self.packet_ordinals)
            or len(self.packet_plan_hashes) != len(self.packet_ordinals)
            or len(self.proof_closure_hashes) != len(self.packet_ordinals)
            or len(self.source_capture_identity_hashes) != len(self.packet_ordinals)
            or len(self.shard_receipt_hashes) != len(self.packet_ordinals)
            or not _closed(self)
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
                "stratum view packet and shard axes do not align",
            )

    def assert_live(self) -> str:
        for shard, expected in zip(self.shards, self.shard_receipt_hashes, strict=True):
            receipt = shard.receipt
            if getattr(receipt, "receipt_hash", None) != expected:
                _refuse(
                    V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
                    "literal-J stratum shard identity changed",
                )
            verifier = getattr(shard, "verify_live_witnesses", None)
            if callable(verifier):
                verifier(
                    expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
                    expected_chart_identity_hash=receipt.chart_identity_hash,
                    expected_current_mask_hash=receipt.current_mask_hash,
                    expected_proposal_generation=receipt.proposal_generation,
                )
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "result_receipt_hash": self.result_receipt_hash,
                "role": self.role,
                "packet_ordinals": self.packet_ordinals,
                "attachment_indices": self.attachment_indices,
                "context_ids": self.context_ids,
                "draw_ids": self.draw_ids,
                "packet_plan_hashes": self.packet_plan_hashes,
                "proof_closure_hashes": self.proof_closure_hashes,
                "source_capture_identity_hashes": self.source_capture_identity_hashes,
                "shard_receipt_hashes": self.shard_receipt_hashes,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "view-",
        )

    @property
    def view_hash(self) -> str:
        return self.assert_live()

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J stratum views carry live shard custody")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J stratum views carry live shard custody")


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2LiteralJLogicalLookResultReceipt:
    source_receipt_hash: str
    execution_result_hash: str
    acquisition_receipt_hash: str
    group_commit_receipt_hash: str
    manifest_hash: str
    physical_call_offset: int
    current_look_stop: int
    physical_roles: tuple[str, ...]
    physical_context_counts: tuple[int, ...]
    target_packet_ordinals: tuple[int, ...]
    complement_packet_ordinals: tuple[int, ...]
    proof_closure_hashes: tuple[str, ...]
    literal_j_shard_receipt_hashes: tuple[str, ...]
    provider_reinvocation_count: int = 0
    row_materialization_count: int = 0
    provider_execution_authority: bool = False
    literal_j_direction_authority: bool = False
    loewner_or_kkt_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    validation_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_RESULT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESULT_TOKEN:
            raise PermissionError("literal-J logical-look result receipts are binder-issued")
        if (
            self.current_look_stop <= self.physical_call_offset
            or len(self.proof_closure_hashes) != len(self.physical_roles)
            or len(self.literal_j_shard_receipt_hashes) != len(self.physical_roles)
            or self.provider_reinvocation_count != 0
            or self.row_materialization_count != 0
            or not _closed(self)
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AUTHORITY_VIOLATION,
                "literal-J result receipt is empty or widened authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "result-receipt-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
                "literal-J logical-look result receipt was mutated",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


class V1SampledV7SharedDW2LiteralJLogicalLookSource:
    __slots__ = (
        "coordinator",
        "receipt",
        "provider_execution_authority",
        "literal_j_direction_authority",
        "loewner_or_kkt_authority",
        "d_stability_authority",
        "c_k_o_authority",
        "mask_commit_authority",
        "validation_authority",
        "terminal_or_pareto_authority",
        "authority_scope",
        "_bound_result",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        coordinator: V1SampledV7CurrentSourceD2GraphCoordinator,
        receipt: V1SampledV7SharedDW2LiteralJLogicalLookPreoutcomeReceipt,
    ) -> None:
        self.coordinator = coordinator
        self.receipt = receipt
        for name, value in _authority_payload().items():
            setattr(self, name, value)
        self._bound_result: V1SampledV7SharedDW2LiteralJLogicalLookResult | None = None
        self.assert_live()

    @property
    def is_bound(self) -> bool:
        return self._bound_result is not None

    def assert_live(self) -> str:
        if (
            type(self.coordinator) is not V1SampledV7CurrentSourceD2GraphCoordinator
            or type(self.receipt)
            is not V1SampledV7SharedDW2LiteralJLogicalLookPreoutcomeReceipt
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.INVALID_TYPE,
                "literal-J source lost its preoutcome receipt",
            )
        self.receipt.assert_integrity()
        coordinator_hash = self.coordinator.assert_live()
        acquisition = self.coordinator.acquisition_plan
        d2_plan = self.coordinator.d2_preoutcome_plan
        if (
            type(acquisition) is not V1SampledV7SharedDW2AcquisitionPlan
            or type(d2_plan) is not V1SampledV7CurrentSourceD2PreoutcomePlan
            or type(acquisition.group_plan) is not V1SampledDualChartRowGroupPlan
            or self.receipt.coordinator_hash != coordinator_hash
            or acquisition.plan_hash != self.receipt.acquisition_plan_hash
            or acquisition.group_plan.group_plan_hash
            != self.receipt.row_group_plan_hash
            or tuple(
                item.packet_plan_hash for item in acquisition.group_plan.packet_plans
            )
            != self.receipt.row_group_packet_plan_hashes
            or acquisition.physical_call_offset != self.receipt.physical_call_offset
            or acquisition.raw_row_prefix_identity_hash
            != self.receipt.raw_row_prefix_identity_hash
            or acquisition.logical_source_hash != self.receipt.logical_source_hash
            or acquisition.group_plan.literal_j_stream_identity_hash
            != self.receipt.literal_j_stream_identity_hash
            or getattr(self.coordinator, "_component_hashes", {}).get(
                "d2_preoutcome_plan_hash"
            )
            != self.receipt.d2_preoutcome_plan_hash
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.PREOUTCOME_REQUIRED,
                "literal-J source coordinator/plan identity changed",
            )
        if not _closed(self):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AUTHORITY_VIOLATION,
                "literal-J source widened authority",
            )
        return self.receipt.receipt_hash

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J logical-look sources carry live preoutcome custody")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J logical-look sources carry live preoutcome custody")


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2LiteralJLogicalLookResult:
    source: V1SampledV7SharedDW2LiteralJLogicalLookSource = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledV7SharedDW2LiteralJLogicalLookResultReceipt
    target: V1SampledV7SharedDW2LiteralJStratumView
    complement: V1SampledV7SharedDW2LiteralJStratumView
    stratum_views: tuple[V1SampledV7SharedDW2LiteralJStratumView, ...]
    execution_result: V1SampledV7CurrentSourceD2ExecutionResult = field(
        repr=False,
        compare=False,
    )
    acquisition: V1SampledV7SharedDW2AcquisitionResult = field(
        repr=False,
        compare=False,
    )
    group_commit: V1SampledDualChartRowGroupCommit = field(
        repr=False,
        compare=False,
    )
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure, ...
    ] = field(repr=False, compare=False)
    provider_execution_authority: bool = False
    literal_j_direction_authority: bool = False
    loewner_or_kkt_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    validation_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.stratum_views != (self.target, self.complement)
            or type(self.execution_result)
            is not V1SampledV7CurrentSourceD2ExecutionResult
            or type(self.acquisition) is not V1SampledV7SharedDW2AcquisitionResult
            or type(self.group_commit) is not V1SampledDualChartRowGroupCommit
            or self.execution_result.acquisition is not self.acquisition
            or self.acquisition.group_commit is not self.group_commit
            or self.acquisition.packet_proof_closures
            is not self.packet_proof_closures
            or tuple(item.closure_hash for item in self.packet_proof_closures)
            != self.receipt.proof_closure_hashes
            or not _closed(self)
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.AUTHORITY_VIOLATION,
                "literal-J result widened authority or changed stratum order",
            )
        self.assert_live()

    @property
    def result_hash(self) -> str:
        return self.assert_live()

    def assert_live(self) -> str:
        self.source.assert_live()
        self.execution_result.assert_live()
        self.receipt.assert_integrity()
        if (
            self.source.coordinator._result is not self.execution_result
            or self.execution_result.acquisition is not self.acquisition
            or self.acquisition.group_commit is not self.group_commit
            or self.acquisition.packet_proof_closures
            is not self.packet_proof_closures
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
                "literal-J result lost its exact execution/commit/proof graph",
            )
        target_hash = self.target.assert_live()
        complement_hash = self.complement.assert_live()
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source.receipt.receipt_hash,
                "result_receipt_hash": self.receipt.receipt_hash,
                "target": target_hash,
                "complement": complement_hash,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "result-",
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J logical-look results carry live shard custody")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J logical-look results carry live shard custody")


def issue_v1_sampled_v7_shared_dw2_literal_j_logical_look_source(
    *,
    coordinator: V1SampledV7CurrentSourceD2GraphCoordinator | None = None,
    d2_execution_result: V1SampledV7CurrentSourceD2ExecutionResult | None = None,
) -> V1SampledV7SharedDW2LiteralJLogicalLookSource:
    """Freeze one current-source D2 literal-J logical source before outcomes."""

    if d2_execution_result is not None:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.PREOUTCOME_REQUIRED,
            "literal-J logical-look sources must be issued from the preoutcome coordinator",
        )
    if type(coordinator) is not V1SampledV7CurrentSourceD2GraphCoordinator:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.INVALID_TYPE,
            "literal-J logical-look source requires an exact D2 coordinator",
        )
    if (
        bool(getattr(coordinator, "has_executed", False))
        or getattr(coordinator, "_result", None) is not None
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.PREOUTCOME_REQUIRED,
            "literal-J logical-look source must be issued before D2 execution",
        )
    coordinator_hash = coordinator.assert_live()
    acquisition = coordinator.acquisition_plan
    d2_plan = coordinator.d2_preoutcome_plan
    owner = coordinator.measurement_owner
    issuance = coordinator.packet_issuance
    group_plan = acquisition.group_plan
    if (
        type(acquisition) is not V1SampledV7SharedDW2AcquisitionPlan
        or type(d2_plan) is not V1SampledV7CurrentSourceD2PreoutcomePlan
        or type(group_plan) is not V1SampledDualChartRowGroupPlan
        or type(issuance)
        not in {
            V1SampledV7DOnlyPacketMeasurementIssuance,
            V1SampledV7DOnlyLogicalLookMeasurementIssuance,
        }
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.INVALID_TYPE,
            "coordinator lost exact acquisition/preoutcome source types",
        )
    acquisition_hash = acquisition.assert_live()
    d2_hash = d2_plan.assert_live()
    transport = tuple(v1_sampled_v7_d_only_transport_packet_issuances(issuance))
    _validate_preoutcome_graph(
        coordinator=coordinator,
        acquisition=acquisition,
        d2_plan=d2_plan,
        group_plan=group_plan,
        transport=transport,
    )
    source_capture_hashes = tuple(
        _identity(
            getattr(packet, "source_capture_identity_hash", None),
            name="packet source capture identity",
        )
        for packet in group_plan.packet_plans
    )
    generalized_prefix = content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_v7_literal_J_generalized_logical_prefix_v1"
            ),
            "coordinator_hash": coordinator_hash,
            "d2_preoutcome_plan_hash": d2_hash,
            "acquisition_plan_hash": acquisition_hash,
            "logical_registered_look_index": (
                acquisition.logical_registered_look_index
            ),
            "logical_registered_look_stratum_counts": (
                acquisition.logical_registered_look_stratum_counts
            ),
            "logical_residual_before_stratum_counts": (
                acquisition.logical_residual_before_stratum_counts
            ),
            "logical_residual_after_stratum_counts": (
                acquisition.logical_residual_after_stratum_counts
            ),
            "transport_packet_hashes": acquisition.transport_packet_hashes,
            "transport_schedule_packet_hashes": (
                acquisition.transport_schedule_packet_hashes
            ),
            "physical_roles": acquisition.physical_roles,
            "physical_context_counts": acquisition.physical_context_counts,
            "physical_call_offset": acquisition.physical_call_offset,
            "expected_prior_manifest_hash": group_plan.expected_prior_manifest_hash,
            "raw_row_prefix_identity_hash": acquisition.raw_row_prefix_identity_hash,
            "logical_source_hash": acquisition.logical_source_hash,
            "row_group_plan_hash": group_plan.group_plan_hash,
            "row_group_packet_plan_hashes": tuple(
                packet.packet_plan_hash for packet in group_plan.packet_plans
            ),
            "source_execution_authority": SAMPLED_V7_CURRENT_SOURCE_D2_EXECUTION_AUTHORITY,
            "provider_outcomes_seen": False,
        },
        prefix=_PREFIX + "generalized-prefix-",
    )
    receipt = V1SampledV7SharedDW2LiteralJLogicalLookPreoutcomeReceipt(
        coordinator_hash=coordinator_hash,
        d2_preoutcome_plan_hash=d2_hash,
        acquisition_plan_hash=acquisition_hash,
        measurement_owner_hash=_hash_attr(
            owner,
            ("measurement_owner_hash",),
            name="measurement owner",
        ),
        suffix_registration_hash=_identity(
            getattr(coordinator.d1_prefix, "suffix_registration_hash", None),
            name="suffix registration",
        ),
        packet_issuance_hash=_hash_attr(
            issuance,
            ("receipt_hash",),
            name="packet issuance",
        ),
        logical_registered_look_index=acquisition.logical_registered_look_index,
        logical_registered_look_stratum_counts=(
            acquisition.logical_registered_look_stratum_counts
        ),
        logical_residual_before_stratum_counts=(
            acquisition.logical_residual_before_stratum_counts
        ),
        logical_residual_after_stratum_counts=(
            acquisition.logical_residual_after_stratum_counts
        ),
        transport_packet_hashes=tuple(getattr(item._packet, "receipt_hash") for item in transport),
        transport_schedule_packet_hashes=acquisition.transport_schedule_packet_hashes,
        transport_physical_call_counts=acquisition.transport_physical_call_counts,
        physical_roles=acquisition.physical_roles,
        physical_context_counts=acquisition.physical_context_counts,
        row_group_plan_hash=group_plan.group_plan_hash,
        row_group_packet_plan_hashes=tuple(
            packet.packet_plan_hash for packet in group_plan.packet_plans
        ),
        physical_call_offset=acquisition.physical_call_offset,
        expected_prior_manifest_hash=group_plan.expected_prior_manifest_hash,
        raw_row_prefix_identity_hash=acquisition.raw_row_prefix_identity_hash,
        logical_source_hash=acquisition.logical_source_hash,
        generalized_logical_prefix_identity_hash=generalized_prefix,
        source_capture_identity_hashes=source_capture_hashes,
        chart_identity_hash=group_plan.chart_identity_hash,
        current_mask_hash=group_plan.current_mask_hash,
        proposal_generation=group_plan.proposal_generation,
        route_axis_hash=group_plan.route_axis_hash,
        route_id_axis_hash=group_plan.route_id_axis_hash,
        route_count=group_plan.route_count,
        literal_j_stream_identity_hash=group_plan.literal_j_stream_identity_hash,
        **_authority_payload(),
        _issuance_token=_PREOUTCOME_TOKEN,
    )
    return V1SampledV7SharedDW2LiteralJLogicalLookSource(
        coordinator=coordinator,
        receipt=receipt,
    )


def _check_closure(
    *,
    packet_plan: V1SampledDualChartRowGroupPacketPlan,
    closure: V1SampledDualChartRowGroupPacketProofClosure,
    group_plan_hash: str,
) -> None:
    if (
        type(packet_plan) is not V1SampledDualChartRowGroupPacketPlan
        or type(closure) is not V1SampledDualChartRowGroupPacketProofClosure
        or closure.group_plan_hash != group_plan_hash
        or closure.packet_plan_hash != packet_plan.packet_plan_hash
        or closure.packet_ordinal != packet_plan.packet_ordinal
        or closure.provider_plan_identity_hash != packet_plan.provider_plan_identity_hash
        or closure.reducer_plan_identity_hashes != packet_plan.reducer_plan_identity_hashes
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
            "literal-J shared acquisition packet closure does not match its preoutcome plan",
        )


def _check_shard(
    *,
    packet_plan: V1SampledDualChartRowGroupPacketPlan,
    shard: V1SampledDualChartRowShardAttachment,
    attachment_index: int,
    source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
) -> None:
    if type(shard) is not V1SampledDualChartRowShardAttachment:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.INVALID_TYPE,
            "literal-J source requires exact shard attachments",
        )
    receipt = shard.receipt
    if (
        getattr(receipt, "role", None)
        is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
        or getattr(receipt, "stream_identity_hash", None)
        != source.receipt.literal_j_stream_identity_hash
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.ROLE_MISMATCH,
            "literal-J shard is not the registered literal-J stream",
        )
    if (
        getattr(receipt, "look_ordinal", None) != attachment_index
        or getattr(receipt, "registered_d_prefix_hash", None)
        != source.receipt.raw_row_prefix_identity_hash
        or getattr(receipt, "source_capture_identity_hash", None)
        != packet_plan.source_capture_identity_hash
        or getattr(receipt, "chart_identity_hash", None)
        != source.receipt.chart_identity_hash
        or getattr(receipt, "current_mask_hash", None)
        != source.receipt.current_mask_hash
        or getattr(receipt, "proposal_generation", None)
        != source.receipt.proposal_generation
        or getattr(receipt, "route_axis_hash", None)
        != source.receipt.route_axis_hash
        or getattr(receipt, "route_id_axis_hash", None)
        != source.receipt.route_id_axis_hash
        or getattr(receipt, "route_count", None) != source.receipt.route_count
        or tuple(getattr(receipt, "context_ids", ())) != packet_plan.context_ids
        or tuple(getattr(receipt, "draw_ids", ())) != packet_plan.draw_ids
        or tuple(getattr(receipt, "stratum_labels", ())) != packet_plan.stratum_labels
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
            "literal-J shard order or packet axis does not match the preoutcome plan",
        )


def bind_v1_sampled_v7_shared_dw2_literal_j_logical_look_source(
    *,
    source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    execution_result: V1SampledV7CurrentSourceD2ExecutionResult,
) -> V1SampledV7SharedDW2LiteralJLogicalLookResult:
    """Bind a preoutcome source to the exact executed D2 literal-J shards."""

    if type(source) is not V1SampledV7SharedDW2LiteralJLogicalLookSource:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.INVALID_TYPE,
            "literal-J binding requires an exact preoutcome source",
        )
    if source.is_bound:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.BIND_REPLAY,
            "literal-J logical-look sources bind exactly once",
        )
    if type(execution_result) is not V1SampledV7CurrentSourceD2ExecutionResult:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.OUTCOME_REQUIRED,
            "literal-J binding requires the exact current-source D2 execution result",
        )
    if (
        not bool(getattr(source.coordinator, "has_executed", False))
        or getattr(source.coordinator, "_result", None) is not execution_result
        or execution_result.coordinator_hash != source.receipt.coordinator_hash
        or execution_result.d2_preoutcome_plan
        is not source.coordinator.d2_preoutcome_plan
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.OUTCOME_REQUIRED,
            "literal-J binding requires the coordinator's exact landed result",
        )
    execution_result.assert_live()
    acquisition = execution_result.acquisition
    if (
        type(acquisition) is not V1SampledV7SharedDW2AcquisitionResult
        or type(acquisition.receipt) is not V1SampledV7SharedDW2AcquisitionReceipt
        or type(acquisition.group_commit) is not V1SampledDualChartRowGroupCommit
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.OUTCOME_REQUIRED,
            "execution result does not carry the exact shared acquisition result",
        )
    receipt = acquisition.receipt
    receipt.assert_integrity()
    commit = acquisition.group_commit
    commit.receipt.assert_integrity()
    if (
        receipt.plan_hash != source.receipt.acquisition_plan_hash
        or receipt.group_plan_hash != source.receipt.row_group_plan_hash
        or receipt.raw_row_prefix_identity_hash
        != source.receipt.raw_row_prefix_identity_hash
        or receipt.logical_registered_look_index
        != source.receipt.logical_registered_look_index
        or receipt.logical_registered_look_stratum_counts
        != source.receipt.logical_registered_look_stratum_counts
        or receipt.transport_packet_hashes != source.receipt.transport_packet_hashes
        or receipt.transport_schedule_packet_hashes
        != source.receipt.transport_schedule_packet_hashes
        or receipt.transport_physical_call_counts
        != source.receipt.transport_physical_call_counts
        or receipt.physical_roles != source.receipt.physical_roles
        or getattr(
            receipt,
            "physical_context_counts",
            source.receipt.physical_context_counts,
        )
        != source.receipt.physical_context_counts
        or receipt.group_commit_receipt_hash != commit.receipt.receipt_hash
        or commit.receipt.group_plan_hash != source.receipt.row_group_plan_hash
        or commit.receipt.manifest_hash != receipt.manifest_hash
        or commit.receipt.packet_plan_hashes
        != source.receipt.row_group_packet_plan_hashes
        or commit.receipt.proof_closure_hashes
        != tuple(item.closure_hash for item in acquisition.packet_proof_closures)
        or receipt.packet_proof_closure_hashes
        != tuple(item.closure_hash for item in acquisition.packet_proof_closures)
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
            "shared acquisition outcome differs from the preoutcome literal-J source",
        )
    group_plan = source.coordinator.acquisition_plan.group_plan
    closures = tuple(acquisition.packet_proof_closures)
    if len(closures) != len(group_plan.packet_plans):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
            "shared acquisition did not close every row packet",
        )
    for packet_plan, closure in zip(group_plan.packet_plans, closures, strict=True):
        _check_closure(
            packet_plan=packet_plan,
            closure=closure,
            group_plan_hash=group_plan.group_plan_hash,
        )
    attachment = commit.attachment
    verifier = getattr(attachment, "verify_live_witnesses", None)
    if callable(verifier):
        verifier()
    manifest = attachment.manifest
    current_stop = source.receipt.physical_call_offset + len(source.receipt.physical_roles)
    if (
        manifest.manifest_hash != receipt.manifest_hash
        or manifest.registered_d_prefix_hash
        != source.receipt.raw_row_prefix_identity_hash
        or manifest.literal_j_stream_identity_hash
        != source.receipt.literal_j_stream_identity_hash
        or manifest.chart_identity_hash != source.receipt.chart_identity_hash
        or manifest.current_mask_hash != source.receipt.current_mask_hash
        or manifest.proposal_generation != source.receipt.proposal_generation
        or manifest.route_axis_hash != source.receipt.route_axis_hash
        or manifest.route_id_axis_hash != source.receipt.route_id_axis_hash
        or manifest.route_count != source.receipt.route_count
        or commit.receipt.prior_manifest_hash
        != source.receipt.expected_prior_manifest_hash
        or len(manifest.looks) != current_stop
        or len(attachment.literal_j_attachments) != current_stop
    ):
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
            "published literal-J prefix does not cover the registered D2 look",
        )
    target_items: list[tuple[int, int, Any, Any]] = []
    complement_items: list[tuple[int, int, Any, Any]] = []
    shard_hashes: list[str] = []
    for packet_plan in group_plan.packet_plans:
        attachment_index = source.receipt.physical_call_offset + packet_plan.packet_ordinal
        look = manifest.looks[attachment_index]
        shard = attachment.literal_j_attachments[attachment_index]
        if (
            look.literal_j_shard is not shard.receipt
            or tuple(look.context_ids) != packet_plan.context_ids
            or tuple(look.draw_ids) != packet_plan.draw_ids
            or tuple(look.stratum_labels) != packet_plan.stratum_labels
        ):
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.CLOSURE_MISMATCH,
                "manifest look and literal-J attachment order diverged",
            )
        _check_shard(
            packet_plan=packet_plan,
            shard=shard,
            attachment_index=attachment_index,
            source=source,
        )
        shard_hashes.append(shard.receipt.receipt_hash)
        item = (packet_plan.packet_ordinal, attachment_index, look, shard)
        role = packet_plan.stratum_labels[0]
        if role == "target":
            target_items.append(item)
        elif role == "complement":
            complement_items.append(item)
        else:
            _refuse(
                V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.ROLE_MISMATCH,
                "literal-J packet plan has an unknown stratum role",
            )
    if not target_items or not complement_items:
        _refuse(
            V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode.ROLE_MISMATCH,
            "literal-J logical-look source requires target and complement views",
        )
    result_receipt = V1SampledV7SharedDW2LiteralJLogicalLookResultReceipt(
        source_receipt_hash=source.receipt.receipt_hash,
        execution_result_hash=execution_result.result_hash,
        acquisition_receipt_hash=receipt.receipt_hash,
        group_commit_receipt_hash=commit.receipt.receipt_hash,
        manifest_hash=receipt.manifest_hash,
        physical_call_offset=source.receipt.physical_call_offset,
        current_look_stop=current_stop,
        physical_roles=source.receipt.physical_roles,
        physical_context_counts=source.receipt.physical_context_counts,
        target_packet_ordinals=tuple(item[0] for item in target_items),
        complement_packet_ordinals=tuple(item[0] for item in complement_items),
        proof_closure_hashes=tuple(item.closure_hash for item in closures),
        literal_j_shard_receipt_hashes=tuple(shard_hashes),
        **_authority_payload(),
        _issuance_token=_RESULT_TOKEN,
    )

    def make_view(
        role: str,
        items: list[tuple[int, int, Any, Any]],
    ) -> V1SampledV7SharedDW2LiteralJStratumView:
        return V1SampledV7SharedDW2LiteralJStratumView(
            source_receipt_hash=source.receipt.receipt_hash,
            result_receipt_hash=result_receipt.receipt_hash,
            role=role,
            packet_ordinals=tuple(item[0] for item in items),
            attachment_indices=tuple(item[1] for item in items),
            context_ids=tuple(
                context for _p, _i, look, _s in items for context in look.context_ids
            ),
            draw_ids=tuple(draw for _p, _i, look, _s in items for draw in look.draw_ids),
            packet_plan_hashes=tuple(
                group_plan.packet_plans[item[0]].packet_plan_hash for item in items
            ),
            proof_closure_hashes=tuple(
                closures[item[0]].closure_hash for item in items
            ),
            source_capture_identity_hashes=tuple(
                group_plan.packet_plans[item[0]].source_capture_identity_hash
                for item in items
            ),
            shard_receipt_hashes=tuple(item[3].receipt.receipt_hash for item in items),
            shards=tuple(item[3] for item in items),
            **_authority_payload(),
        )

    target = make_view("target", target_items)
    complement = make_view("complement", complement_items)
    result = V1SampledV7SharedDW2LiteralJLogicalLookResult(
        source=source,
        receipt=result_receipt,
        target=target,
        complement=complement,
        stratum_views=(target, complement),
        execution_result=execution_result,
        acquisition=acquisition,
        group_commit=commit,
        packet_proof_closures=acquisition.packet_proof_closures,
        **_authority_payload(),
    )
    source._bound_result = result
    return result


V1SampledV7SharedDW2LiteralJLogicalLookSourceRefusalCode = (
    V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode
)
V1SampledV7SharedDW2LiteralJLogicalLookSourceRefusal = (
    V1SampledV7SharedDW2LiteralJLogicalLookRefusal
)
V1SampledV7SharedDW2LiteralJSegment = V1SampledV7SharedDW2LiteralJStratumView


__all__ = [
    "SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_AUTHORITY",
    "SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_RESULT_SCHEMA",
    "SAMPLED_V7_SHARED_DW2_LITERAL_J_LOGICAL_LOOK_SOURCE_SCHEMA",
    "V1SampledV7SharedDW2LiteralJLogicalLookPreoutcomeReceipt",
    "V1SampledV7SharedDW2LiteralJLogicalLookRefusal",
    "V1SampledV7SharedDW2LiteralJLogicalLookRefusalCode",
    "V1SampledV7SharedDW2LiteralJLogicalLookResult",
    "V1SampledV7SharedDW2LiteralJLogicalLookResultReceipt",
    "V1SampledV7SharedDW2LiteralJLogicalLookSource",
    "V1SampledV7SharedDW2LiteralJLogicalLookSourceRefusal",
    "V1SampledV7SharedDW2LiteralJLogicalLookSourceRefusalCode",
    "V1SampledV7SharedDW2LiteralJSegment",
    "V1SampledV7SharedDW2LiteralJStratumView",
    "bind_v1_sampled_v7_shared_dw2_literal_j_logical_look_source",
    "issue_v1_sampled_v7_shared_dw2_literal_j_logical_look_source",
]
