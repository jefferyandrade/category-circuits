"""Branch-independent same-capture A/J acquisition for sampled V7 ``D_W2``.

The logical D packet is source-issued once, before either KKT branch is
lowered.  This module turns that logical packet into the nonempty physical
strata required by the scorer and commits their corrected-A and literal-J
rows behind one atomic row-group HEAD.  Its durable row-prefix identity is
deliberately independent of branch, resource, compact support, packed action,
and append lineage.

Only raw proposal-input custody is authorized here.  D stability, C/K/O,
mask commit, terminal validation, and Pareto claims remain closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, NoReturn

try:  # pragma: no cover - exercised by production/CUDA jobs.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mechanistic_systems.circuits.irg.modern_v1_alexnet_backend import (
    prepare_v1_sampled_raw_target_logit_dual_chart_request,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupCommit,
    V1SampledDualChartRowGroupPacketPlan,
    V1SampledDualChartRowGroupPacketProofClosure,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowLivePrefixAttachment,
    begin_v1_sampled_dual_chart_row_group_transaction,
    prepare_v1_sampled_dual_chart_row_group_plan,
    reattach_v1_sampled_dual_chart_row_prefix,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_look_transaction import (
    V1SampledDualChartChunkObjectSeparationWitness,
    V1SampledDualChartCustodyProviderVerificationReceipt,
    V1SampledDualRowGroupPacketBridge,
    verify_v1_sampled_dual_chart_custody_only_provider_receipt,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyLogicalLookMeasurementIssuance,
    V1SampledV7DOnlyMeasurementOwner,
    V1SampledV7DOnlyPacketMeasurementIssuance,
    V1SampledV7DOnlyPopulationAttachment,
    v1_sampled_v7_d_only_logical_registered_look,
    v1_sampled_v7_d_only_transport_packet_issuances,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_dw2_literal_j_acquisition import (
    _canonical_route_axis,
    _provider_source_identity,
    _request_integrity_hash,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.measurement import MeasurementRequest
from mechanistic_systems.core.axis_identity import ordered_string_axis_hash
from mechanistic_systems.measurement.optimized_irg_providers import (
    OptimizedScoringResponseProvider,
    build_route_chunked_dual_chart_fanout,
)


SAMPLED_V7_SHARED_DW2_ACQUISITION_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_acquisition_plan_v2"
)
SAMPLED_V7_SHARED_DW2_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_acquisition_receipt_v2"
)
SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY: Final = (
    "branch_independent_same_capture_corrected_A_literal_J_row_custody_only_"
    "no_D_stability_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-shared-D-W2-acquisition-"
_ROLE_ORDER: Final = ("target", "complement")
_PLAN_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()
_PRIVATE_REFERENCE_TOKEN: Final = object()


class V1SampledV7SharedDW2AcquisitionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    AXIS_MISMATCH = "axis_mismatch"
    EMPTY_PACKET = "empty_packet"
    PROVIDER_MISMATCH = "provider_mismatch"
    BATCH_PREFLIGHT_FAILURE = "batch_preflight_failure"
    PREDECESSOR_MISMATCH = "predecessor_mismatch"
    PARTIAL_COVER = "partial_cover"
    ATOMIC_PUBLICATION_FAILURE = "atomic_publication_failure"
    RECOVERY_REQUIRED = "recovery_required"
    RECOVERY_MISMATCH = "recovery_mismatch"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7SharedDW2AcquisitionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7SharedDW2AcquisitionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7SharedDW2AcquisitionRefusalCode:
            raise TypeError("shared acquisition refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7SharedDW2AcquisitionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7SharedDW2AcquisitionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
            f"{name} must be one nonempty identity",
        )
    return value


def v1_sampled_v7_nonempty_physical_roles(
    *,
    target_context_ids: Sequence[str],
    complement_context_ids: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Freeze nonempty physical strata in canonical target-first order."""

    target = tuple(target_context_ids)
    complement = tuple(complement_context_ids)
    if any(type(value) is not str or not value for value in target + complement):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.AXIS_MISMATCH,
            "physical context axes require nonempty string identities",
        )
    if len(set(target + complement)) != len(target) + len(complement):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.AXIS_MISMATCH,
            "physical target/complement axes overlap or repeat",
        )
    result = tuple(
        (role, values)
        for role, values in zip(_ROLE_ORDER, (target, complement), strict=True)
        if values
    )
    if not result:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.EMPTY_PACKET,
            "a logical D_W2 packet has no nonempty physical stratum",
        )
    return result


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {"schema": "sampled-v7-shared-D-W2-axis-v1", "role": role, "values": tuple(values)},
        prefix=_PREFIX + "axis-",
    )


def _physical_call_offset(owner: V1SampledV7DOnlyMeasurementOwner, ordinal: int) -> int:
    return sum(
        int(any(label == "target" for label in packet.registered_stratum_ids))
        + int(any(label == "complement" for label in packet.registered_stratum_ids))
        for packet in owner.schedule.packets[:ordinal]
    )


def _head_manifest_hash(directory: Path) -> str | None:
    head = directory / ".modern-v1-sampled-dual-chart-row-prefix-head.json"
    if not head.exists():
        return None
    try:
        import json

        payload = json.loads(head.read_text(encoding="utf-8"))
        value = payload.get("manifest_hash")
    except Exception:
        return "<unreadable-head>"
    return value if type(value) is str and value else "<unreadable-head>"


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2AcquisitionPlan:
    measurement_owner_hash: str
    population_attachment_hash: str
    packet_issuance_hash: str
    packet_hash: str
    logical_registered_look_index: int
    logical_registered_look_stratum_counts: tuple[int, int]
    logical_residual_before_stratum_counts: tuple[int, int]
    logical_residual_after_stratum_counts: tuple[int, int]
    transport_packet_hashes: tuple[str, ...]
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_physical_call_counts: tuple[int, ...]
    raw_row_prefix_identity_hash: str
    logical_source_hash: str
    physical_call_offset: int
    physical_roles: tuple[str, ...]
    physical_context_counts: tuple[int, ...]
    physical_request_ids: tuple[str, ...]
    group_plan: V1SampledDualChartRowGroupPlan = field(repr=False, compare=False)
    measurement_requests: tuple[MeasurementRequest, ...] = field(repr=False, compare=False)
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = field(
        repr=False, compare=False
    )
    scorer: OptimizedScoringResponseProvider = field(repr=False, compare=False)
    route_ids: Sequence[str] = field(repr=False, compare=False)
    declared_device: Any = field(repr=False, compare=False)
    scratch_directory: Path = field(repr=False, compare=False)
    provider_source_identity_hash: str = ""
    destination_transport_hash: str = ""
    provider_outcomes_seen: bool = False
    branch_resource_support_or_action_identity_present: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_ACQUISITION_PLAN_SCHEMA
    _owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _population_attachment: V1SampledV7DOnlyPopulationAttachment = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _packet_issuance: object = field(repr=False, compare=False, default=None)
    _private_reference: bool = field(repr=False, compare=False, default=False)
    _issuance_token: InitVar[object | None] = None
    _plan_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLAN_TOKEN:
            raise PermissionError("shared acquisition plans are issuer-only")
        self.assert_live()
        object.__setattr__(self, "_plan_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "measurement_owner_hash": self.measurement_owner_hash,
                "population_attachment_hash": self.population_attachment_hash,
                "packet_issuance_hash": self.packet_issuance_hash,
                "packet_hash": self.packet_hash,
                "logical_registered_look_index": self.logical_registered_look_index,
                "logical_registered_look_stratum_counts": (
                    self.logical_registered_look_stratum_counts
                ),
                "logical_residual_before_stratum_counts": (
                    self.logical_residual_before_stratum_counts
                ),
                "logical_residual_after_stratum_counts": (
                    self.logical_residual_after_stratum_counts
                ),
                "transport_packet_hashes": self.transport_packet_hashes,
                "transport_schedule_packet_hashes": (
                    self.transport_schedule_packet_hashes
                ),
                "transport_physical_call_counts": self.transport_physical_call_counts,
                "raw_row_prefix_identity_hash": self.raw_row_prefix_identity_hash,
                "logical_source_hash": self.logical_source_hash,
                "physical_call_offset": self.physical_call_offset,
                "physical_roles": self.physical_roles,
                "physical_context_counts": self.physical_context_counts,
                "physical_request_ids": self.physical_request_ids,
                "group_plan_hash": self.group_plan.group_plan_hash,
                "provider_source_identity_hash": self.provider_source_identity_hash,
                "destination_transport_hash": self.destination_transport_hash,
                "branch_resource_support_or_action_identity_present": False,
                "downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "plan-",
        )

    def assert_live(self) -> str:
        exact = (
            (self._owner, V1SampledV7DOnlyMeasurementOwner),
            (self._population_attachment, V1SampledV7DOnlyPopulationAttachment),
            (self.group_plan, V1SampledDualChartRowGroupPlan),
        )
        if any(type(value) is not expected for value, expected in exact):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
                "shared acquisition lost an exact live source",
            )
        if type(self._packet_issuance) not in {
            V1SampledV7DOnlyPacketMeasurementIssuance,
            V1SampledV7DOnlyLogicalLookMeasurementIssuance,
        }:
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
                "shared acquisition lost its exact packet issuance",
            )
        owner_hash = self._owner.assert_live()
        attachment_hash = self._population_attachment.assert_live()
        try:
            issuance_hash = self._packet_issuance.assert_live()  # type: ignore[attr-defined]
            transport_issuances = v1_sampled_v7_d_only_transport_packet_issuances(
                self._packet_issuance
            )
            (
                logical_look_index,
                logical_look_counts,
                logical_residual_before,
                logical_residual_after,
            ) = v1_sampled_v7_d_only_logical_registered_look(
                self._packet_issuance,
                measurement_owner=self._owner,
            )
        except Exception as error:
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
                "packet issuance failed logical-look replay: "
                f"{type(error).__name__}: {error}",
            )
        transport_packets = tuple(value._packet for value in transport_issuances)
        scheduled_packets = tuple(
            self._owner.schedule.packet_for(packet) for packet in transport_packets
        )
        physical_call_counts = tuple(
            len(
                v1_sampled_v7_nonempty_physical_roles(
                    target_context_ids=packet.target_context_ids,
                    complement_context_ids=packet.complement_context_ids,
                )
            )
            for packet in transport_packets
        )
        if (
            self._population_attachment._owner is not self._owner
            or any(value._owner is not self._owner for value in transport_issuances)
            or owner_hash != self.measurement_owner_hash
            or attachment_hash != self.population_attachment_hash
            or issuance_hash != self.packet_issuance_hash
            or issuance_hash != self.packet_hash
            or logical_look_index != self.logical_registered_look_index
            or logical_look_counts != self.logical_registered_look_stratum_counts
            or logical_residual_before
            != self.logical_residual_before_stratum_counts
            or logical_residual_after != self.logical_residual_after_stratum_counts
            or tuple(packet.receipt_hash for packet in transport_packets)
            != self.transport_packet_hashes
            or tuple(
                packet.packet_schedule_hash for packet in scheduled_packets
            )
            != self.transport_schedule_packet_hashes
            or physical_call_counts != self.transport_physical_call_counts
            or self.scorer is not self._owner._scorer
            or self.route_ids is not self._owner._constructive_route_ids
            or self.provider_outcomes_seen is not False
            or self.branch_resource_support_or_action_identity_present is not False
            or self.d_stability_authority is not False
            or self.c_k_o_or_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.authority_scope != SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY
            or self.schema != SAMPLED_V7_SHARED_DW2_ACQUISITION_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "shared acquisition crossed its branch-independent custody boundary",
            )
        self.group_plan.assert_integrity()
        requests = tuple(self.measurement_requests)
        if (
            not requests
            or any(type(value) is not MeasurementRequest for value in requests)
            or tuple(value.request_id for value in requests) != self.physical_request_ids
            or tuple(_request_integrity_hash(value) for value in requests)
            != self.physical_request_ids
            or len(requests) != len(self.physical_roles)
            or tuple(len(value.context_selector.context_ids or ()) for value in requests)
            != self.physical_context_counts
        ):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
                "shared physical request axis changed after freeze",
            )
        prior_hash = None if self.prior_row_prefix is None else self.prior_row_prefix.manifest.manifest_hash
        prior_count = 0 if self.prior_row_prefix is None else len(self.prior_row_prefix.manifest.looks)
        if self.prior_row_prefix is not None:
            self.prior_row_prefix.verify_live_witnesses()
        if (
            prior_count != self.physical_call_offset
            or prior_hash != self.group_plan.expected_prior_manifest_hash
            or self.group_plan.first_prefix_look_ordinal != self.physical_call_offset
            or self.group_plan.registered_d_prefix_hash != self.raw_row_prefix_identity_hash
            or tuple(item.provider_plan_identity_hash for item in self.group_plan.packet_plans)
            != self.physical_request_ids
        ):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.PREDECESSOR_MISMATCH,
                "row predecessor does not equal the cumulative nonempty physical-call prefix",
            )
        observed = self._compute_hash()
        if self._plan_hash is not None and observed != self._plan_hash:
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
                "shared acquisition plan identity changed",
            )
        return observed

    @property
    def plan_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("shared acquisition plans carry live provider/path custody")


def prepare_v1_sampled_v7_shared_dw2_acquisition(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    population_attachment: V1SampledV7DOnlyPopulationAttachment,
    packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ),
    scorer: OptimizedScoringResponseProvider,
    route_ids: Sequence[str],
    declared_device: Any,
    scratch_directory: str | Path,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledV7SharedDW2AcquisitionPlan:
    return _prepare_v1_sampled_v7_shared_dw2_acquisition(
        measurement_owner=measurement_owner,
        population_attachment=population_attachment,
        packet_issuance=packet_issuance,
        scorer=scorer,
        route_ids=route_ids,
        declared_device=declared_device,
        scratch_directory=scratch_directory,
        prior_row_prefix=prior_row_prefix,
        private_reference_token=None,
    )


def _prepare_v1_sampled_v7_shared_dw2_acquisition_private_reference(
    **kwargs: Any,
) -> V1SampledV7SharedDW2AcquisitionPlan:
    return _prepare_v1_sampled_v7_shared_dw2_acquisition(
        **kwargs,
        private_reference_token=_PRIVATE_REFERENCE_TOKEN,
    )


def _prepare_v1_sampled_v7_shared_dw2_acquisition(
    *,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    population_attachment: V1SampledV7DOnlyPopulationAttachment,
    packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ),
    scorer: OptimizedScoringResponseProvider,
    route_ids: Sequence[str],
    declared_device: Any,
    scratch_directory: str | Path,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None,
    private_reference_token: object | None,
) -> V1SampledV7SharedDW2AcquisitionPlan:
    private = private_reference_token is _PRIVATE_REFERENCE_TOKEN
    exact = (
        (measurement_owner, V1SampledV7DOnlyMeasurementOwner),
        (population_attachment, V1SampledV7DOnlyPopulationAttachment),
        (scorer, OptimizedScoringResponseProvider),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
            "shared acquisition requires exact production source types",
        )
    if type(packet_issuance) not in {
        V1SampledV7DOnlyPacketMeasurementIssuance,
        V1SampledV7DOnlyLogicalLookMeasurementIssuance,
    }:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
            "shared acquisition requires a transport or logical-look packet issuance",
        )
    owner_hash = measurement_owner.assert_live()
    attachment_hash = population_attachment.assert_live()
    try:
        issuance_hash = packet_issuance.assert_live()  # type: ignore[attr-defined]
        transport_issuances = v1_sampled_v7_d_only_transport_packet_issuances(
            packet_issuance
        )
        (
            logical_look_index,
            logical_look_counts,
            logical_residual_before,
            logical_residual_after,
        ) = v1_sampled_v7_d_only_logical_registered_look(
            packet_issuance,
            measurement_owner=measurement_owner,
        )
    except Exception as error:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
            "packet issuance failed logical-look replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        population_attachment._owner is not measurement_owner
        or any(value._owner is not measurement_owner for value in transport_issuances)
        or scorer is not measurement_owner._scorer
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
            "measurement owner, population attachment, packet issuance, and scorer diverged",
        )
    if torch is None:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.PROVIDER_MISMATCH,
            "PyTorch is required for physical acquisition",
        )
    device = torch.device(declared_device)
    if not private and (
        device.type != "cuda" or device.index is None or not torch.cuda.is_available()
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "production shared acquisition requires an explicit live CUDA device",
        )
    if private and device.type != "cpu":
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "the bounded reference acquisition is CPU-only",
        )
    route_axis = _canonical_route_axis(route_ids)
    route_hash = ordered_string_axis_hash(route_axis, name="shared D_W2 route axis")
    source = measurement_owner.measurement_source_lineage
    population = population_attachment._registration
    if (
        route_ids is not measurement_owner._constructive_route_ids
        or route_hash != source.route_id_axis_hash
        or len(route_axis) != population.route_count
        or population.route_axis_hash != source.route_axis_hash
        or population.chart_identity_hash != source.chart_identity_hash
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.AXIS_MISMATCH,
            "route/chart population differs from the branch-independent source",
        )
    transport_packets = tuple(value._packet for value in transport_issuances)
    scheduled_packets = tuple(
        measurement_owner.schedule.packet_for(packet) for packet in transport_packets
    )
    physical_by_packet = tuple(
        v1_sampled_v7_nonempty_physical_roles(
            target_context_ids=packet.target_context_ids,
            complement_context_ids=packet.complement_context_ids,
        )
        for packet in transport_packets
    )
    physical = tuple(item for packet_roles in physical_by_packet for item in packet_roles)
    packet_size = int(getattr(scorer, "context_packet_size", 0))
    if packet_size <= 0 or max(len(values) for _role, values in physical) > packet_size:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.BATCH_PREFLIGHT_FAILURE,
            "one physical stratum would require more than one registered scorer call",
        )
    destination = _identity(
        getattr(scorer, "local_destination_transport_hash", None),
        name="destination transport hash",
    )
    provider_source = _provider_source_identity(
        scorer=scorer,
        route_id_axis_hash=route_hash,
        destination_transport_hash=destination,
        campaign_contract_hash=measurement_owner._campaign_binding.contract.campaign_contract_hash,
        model_identity_hash=source.model_identity_hash,
        declared_device=device,
        private_reference=private,
    )
    raw_prefix = content_hash(
        {
            "schema": "sampled-v7-shared-D-W2-raw-row-prefix-v1",
            "measurement_owner_hash": owner_hash,
            "population_attachment_hash": attachment_hash,
            "measurement_schedule_hash": measurement_owner.schedule.schedule_hash,
            "population_registration_hash": population.registration_hash,
            "provider_source_identity_hash": provider_source,
            "chart_identity_hash": source.chart_identity_hash,
            "current_mask_hash": source.current_mask_hash,
            "route_axis_hash": source.route_axis_hash,
            "route_id_axis_hash": source.route_id_axis_hash,
            "corrected_A_stream_identity_hash": source.corrected_evidence_stream_source_hash,
            "literal_J_stream_identity_hash": source.literal_j_stream_source_hash,
            "branch_resource_support_packed_action_append_coordinates": (),
        },
        prefix=_PREFIX + "raw-prefix-",
    )
    logical_source = content_hash(
        {
            "schema": "sampled-v7-shared-D-W2-logical-source-v1",
            "raw_row_prefix_identity_hash": raw_prefix,
            "packet_issuance_hash": issuance_hash,
            "logical_registered_look_index": logical_look_index,
            "logical_registered_look_stratum_counts": logical_look_counts,
            "logical_residual_before_stratum_counts": logical_residual_before,
            "logical_residual_after_stratum_counts": logical_residual_after,
            "transport_packet_hashes": tuple(
                packet.receipt_hash for packet in transport_packets
            ),
            "transport_schedule_packet_hashes": tuple(
                packet.packet_schedule_hash for packet in scheduled_packets
            ),
            "transport_request_ids": tuple(
                value.logical_measurement_request.request_id
                for value in transport_issuances
            ),
            "branch_resource_support_packed_action_append_coordinates": (),
        },
        prefix=_PREFIX + "logical-source-",
    )
    group_packets: list[V1SampledDualChartRowGroupPacketPlan] = []
    requests: list[MeasurementRequest] = []
    physical_ordinal = 0
    for transport_ordinal, (packet, scheduled, packet_physical) in enumerate(
        zip(transport_packets, scheduled_packets, physical_by_packet, strict=True)
    ):
        split = len(packet.target_context_ids)
        axes = {
            "target": slice(0, split),
            "complement": slice(split, split + len(packet.complement_context_ids)),
        }
        for role, contexts in packet_physical:
            request = prepare_v1_sampled_raw_target_logit_dual_chart_request(
                binding=measurement_owner._campaign_binding,
                current_mask=measurement_owner._current_mask,
                contexts=contexts,
                mask_calibration=measurement_owner._calibration,
                calibration=measurement_owner._calibration,
                corrected_a_stream_identity_hash=(
                    source.corrected_evidence_stream_source_hash
                ),
                literal_j_stream_identity_hash=source.literal_j_stream_source_hash,
                logical_source_hash=logical_source,
                physical_stratum=role,
            )
            requests.append(request)
            section = axes[role]
            draws = scheduled.registered_draw_ids[section]
            capture_ids = scheduled.registered_capture_ids[section]
            source_packet_hashes = scheduled.registered_source_packet_hashes[section]
            if len(contexts) != len(draws):
                _refuse(
                    V1SampledV7SharedDW2AcquisitionRefusalCode.AXIS_MISMATCH,
                    "transport subpacket physical role axes differ from the schedule",
                )
            capture = _identity(
                request.payload.get("source_capture_identity_hash"),
                name=f"{role} physical capture identity",
            )
            registered_packet = content_hash(
                {
                    "schema": "sampled-v7-shared-D-W2-physical-packet-v2",
                    "logical_source_hash": logical_source,
                    "physical_ordinal": physical_ordinal,
                    "transport_packet_ordinal": transport_ordinal,
                    "transport_suffix_packet_hash": packet.receipt_hash,
                    "transport_schedule_packet_hash": scheduled.packet_schedule_hash,
                    "physical_stratum": role,
                    "context_axis_hash": _axis_hash(role + "-context", contexts),
                    "draw_axis_hash": _axis_hash(role + "-draw", draws),
                    "capture_axis_hash": _axis_hash(role + "-capture", capture_ids),
                    "source_packet_axis_hash": _axis_hash(
                        role + "-source-packet",
                        source_packet_hashes,
                    ),
                    "provider_request_id": request.request_id,
                    "source_capture_identity_hash": capture,
                    "branch_resource_support_packed_action_append_coordinates": (),
                },
                prefix=_PREFIX + "physical-packet-",
            )
            a_plan = content_hash(
                {"registered_packet_hash": registered_packet, "role": "corrected_A"},
                prefix=_PREFIX + "reducer-",
            )
            j_plan = content_hash(
                {"registered_packet_hash": registered_packet, "role": "literal_J"},
                prefix=_PREFIX + "reducer-",
            )
            group_packets.append(
                V1SampledDualChartRowGroupPacketPlan(
                    packet_ordinal=physical_ordinal,
                    registered_packet_plan_hash=registered_packet,
                    source_capture_identity_hash=capture,
                    provider_plan_identity_hash=request.request_id,
                    reducer_plan_identity_hashes=(a_plan, j_plan),
                    context_ids=contexts,
                    draw_ids=draws,
                    stratum_labels=(role,) * len(contexts),
                )
            )
            physical_ordinal += 1
    offset = _physical_call_offset(
        measurement_owner,
        transport_packets[0].packet_ordinal,
    )
    prior_hash = None if prior_row_prefix is None else prior_row_prefix.manifest.manifest_hash
    prior_count = 0 if prior_row_prefix is None else len(prior_row_prefix.manifest.looks)
    if prior_count != offset or (offset == 0) is not (prior_row_prefix is None):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.PREDECESSOR_MISMATCH,
            "predecessor length is not the source-bound cumulative physical-call offset",
        )
    scratch = Path(scratch_directory).resolve()
    group_plan = prepare_v1_sampled_dual_chart_row_group_plan(
        registered_look_index=logical_look_index,
        expected_prior_manifest_hash=prior_hash,
        first_prefix_look_ordinal=offset,
        registered_d_prefix_hash=raw_prefix,
        chart_identity_hash=source.chart_identity_hash,
        current_mask_hash=source.current_mask_hash,
        proposal_generation=source.proposal_generation,
        route_axis_hash=source.route_axis_hash,
        route_id_axis_hash=source.route_id_axis_hash,
        route_count=population.route_count,
        corrected_a_stream_identity_hash=source.corrected_evidence_stream_source_hash,
        literal_j_stream_identity_hash=source.literal_j_stream_source_hash,
        packet_plans=tuple(group_packets),
        registered_route_chunk_size=int(getattr(scorer, "response_route_chunk_size", 0)),
        scratch_directory=scratch,
    )
    return V1SampledV7SharedDW2AcquisitionPlan(
        measurement_owner_hash=owner_hash,
        population_attachment_hash=attachment_hash,
        packet_issuance_hash=issuance_hash,
        packet_hash=issuance_hash,
        logical_registered_look_index=logical_look_index,
        logical_registered_look_stratum_counts=logical_look_counts,
        logical_residual_before_stratum_counts=logical_residual_before,
        logical_residual_after_stratum_counts=logical_residual_after,
        transport_packet_hashes=tuple(
            packet.receipt_hash for packet in transport_packets
        ),
        transport_schedule_packet_hashes=tuple(
            packet.packet_schedule_hash for packet in scheduled_packets
        ),
        transport_physical_call_counts=tuple(
            len(value) for value in physical_by_packet
        ),
        raw_row_prefix_identity_hash=raw_prefix,
        logical_source_hash=logical_source,
        physical_call_offset=offset,
        physical_roles=tuple(role for role, _contexts in physical),
        physical_context_counts=tuple(len(contexts) for _role, contexts in physical),
        physical_request_ids=tuple(request.request_id for request in requests),
        group_plan=group_plan,
        measurement_requests=tuple(requests),
        prior_row_prefix=prior_row_prefix,
        scorer=scorer,
        route_ids=route_ids,
        declared_device=device,
        scratch_directory=scratch,
        provider_source_identity_hash=provider_source,
        destination_transport_hash=destination,
        _owner=measurement_owner,
        _population_attachment=population_attachment,
        _packet_issuance=packet_issuance,
        _private_reference=private,
        _issuance_token=_PLAN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2AcquisitionReceipt:
    plan_hash: str
    raw_row_prefix_identity_hash: str
    packet_hash: str
    logical_registered_look_index: int
    logical_registered_look_stratum_counts: tuple[int, int]
    transport_packet_hashes: tuple[str, ...]
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_physical_call_counts: tuple[int, ...]
    physical_call_offset: int
    physical_roles: tuple[str, ...]
    provider_call_count: int
    capture_batch_count: int
    vjp_count: int
    group_plan_hash: str
    group_commit_receipt_hash: str
    manifest_hash: str
    provider_verification_receipt_hashes: tuple[str, ...]
    provider_proof_hashes: tuple[str, ...]
    packet_proof_closure_hashes: tuple[str, ...]
    raw_dual_stream_receipt_hashes: tuple[str, ...]
    row_object_separation_receipt_hashes: tuple[str, ...]
    physical_rows_acquired_once: bool = True
    atomic_group_publication: bool = True
    branch_resource_support_or_action_identity_present: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_ACQUISITION_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("shared acquisition receipts are executor-only")
        count = len(self.physical_roles)
        axes = (
            self.provider_verification_receipt_hashes,
            self.provider_proof_hashes,
            self.packet_proof_closure_hashes,
            self.raw_dual_stream_receipt_hashes,
            self.row_object_separation_receipt_hashes,
        )
        groups = tuple(self.transport_physical_call_counts)
        if (
            count < 1
            or not groups
            or any(type(value) is not int for value in groups)
            or sum(groups) != count
            or any(value not in (1, 2) for value in groups)
            or len(self.transport_packet_hashes) != len(groups)
            or len(self.transport_schedule_packet_hashes) != len(groups)
            or any(type(value) is not str or not value for value in self.transport_packet_hashes)
            or any(type(value) is not str or not value for value in self.transport_schedule_packet_hashes)
            or any(len(axis) != count for axis in axes)
            or any(any(type(value) is not str or not value for value in axis) for axis in axes)
            or self.provider_call_count != count
            or self.capture_batch_count != count
            or self.vjp_count != count
            or self.physical_rows_acquired_once is not True
            or self.atomic_group_publication is not True
            or self.branch_resource_support_or_action_identity_present is not False
            or self.d_stability_authority is not False
            or self.c_k_o_or_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.authority_scope != SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY
            or self.schema != SAMPLED_V7_SHARED_DW2_ACQUISITION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "shared acquisition receipt has partial cover or downstream authority",
            )
        cursor = 0
        for group_size in groups:
            segment = self.physical_roles[cursor : cursor + group_size]
            if tuple(role for role in _ROLE_ORDER if role in segment) != segment:
                _refuse(
                    V1SampledV7SharedDW2AcquisitionRefusalCode.AUTHORITY_VIOLATION,
                    "shared acquisition receipt lost target-before-complement "
                    "transport order",
                )
            cursor += group_size
        object.__setattr__(self, "_receipt_hash", self.assert_integrity())

    def assert_integrity(self) -> str:
        observed = content_hash(
            {name: getattr(self, name) for name in self.__dataclass_fields__ if not name.startswith("_")},
            prefix=_PREFIX + "receipt-",
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
                "shared acquisition receipt changed",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2AcquisitionResult:
    receipt: V1SampledV7SharedDW2AcquisitionReceipt
    group_commit: V1SampledDualChartRowGroupCommit = field(repr=False, compare=False)
    packet_proof_closures: tuple[V1SampledDualChartRowGroupPacketProofClosure, ...] = field(
        repr=False, compare=False
    )
    provider_verifications: tuple[V1SampledDualChartCustodyProviderVerificationReceipt, ...] = field(
        repr=False, compare=False
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESULT_TOKEN:
            raise PermissionError("shared acquisition results are executor-only")
        self.receipt.assert_integrity()
        if (
            type(self.group_commit) is not V1SampledDualChartRowGroupCommit
            or self.group_commit.receipt.receipt_hash != self.receipt.group_commit_receipt_hash
            or self.group_commit.receipt.manifest_hash != self.receipt.manifest_hash
            or tuple(value.closure_hash for value in self.packet_proof_closures)
            != self.receipt.packet_proof_closure_hashes
            or tuple(value.receipt_hash for value in self.provider_verifications)
            != self.receipt.provider_verification_receipt_hashes
        ):
            _refuse(
                V1SampledV7SharedDW2AcquisitionRefusalCode.STALE_SOURCE,
                "shared acquisition result lost its exact commit/proof graph",
            )


def execute_v1_sampled_v7_shared_dw2_acquisition(
    plan: V1SampledV7SharedDW2AcquisitionPlan,
) -> V1SampledV7SharedDW2AcquisitionResult:
    if type(plan) is not V1SampledV7SharedDW2AcquisitionPlan:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
            "execution requires an exact shared acquisition plan",
        )
    plan.assert_live()
    current_head = _head_manifest_hash(plan.scratch_directory)
    expected = plan.group_plan.expected_prior_manifest_hash
    if current_head != expected:
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.RECOVERY_REQUIRED,
            "row HEAD already differs from the planned predecessor; exact receipt reattachment is required and provider work will not be reissued",
        )
    verifications: list[V1SampledDualChartCustodyProviderVerificationReceipt] = []
    closures: list[V1SampledDualChartRowGroupPacketProofClosure] = []
    try:
        with begin_v1_sampled_dual_chart_row_group_transaction(
            plan=plan.group_plan,
            scratch_directory=plan.scratch_directory,
            prior=plan.prior_row_prefix,
        ) as transaction:
            for ordinal, request in enumerate(plan.measurement_requests):
                plan.assert_live()
                packet_plan = plan.group_plan.packet_plans[ordinal]
                witness = V1SampledDualChartChunkObjectSeparationWitness()
                bridge = V1SampledDualRowGroupPacketBridge(
                    group_plan_hash=plan.group_plan.group_plan_hash,
                    group_packet_plan=packet_plan,
                    staging_consumer=transaction.open_packet_staging_consumer(packet_ordinal=ordinal),
                    route_ids=plan.route_ids,
                    device=plan.declared_device,
                    destination_transport_hash=plan.destination_transport_hash,
                    registered_route_chunk_size=plan.group_plan.registered_route_chunk_size,
                    corrected_a_stream_identity_hash=plan._owner.corrected_evidence_stream_source_hash,
                    literal_j_stream_identity_hash=plan._owner.residual_literal_j_stream_source_hash,
                    dual_chunk_observer=witness.observe,
                )
                fanout = build_route_chunked_dual_chart_fanout(
                    evidence_consumers=(bridge.evidence_consumer,),
                    jacobian_consumers=(bridge.jacobian_consumer,),
                )
                provider_receipt = plan.scorer.accumulate_device_tripartite(
                    request, accumulator=fanout
                )
                plan.assert_live()
                if not isinstance(provider_receipt, Mapping):
                    raise TypeError("shared D_W2 scorer returned no mapping receipt")
                branch = dict(bridge.branch_receipt("corrected_a"))
                separation = witness.close(
                    expected_pair_count=len(tuple(branch.get("route_windows", ())))
                )
                verification = verify_v1_sampled_dual_chart_custody_only_provider_receipt(
                    provider_receipt,
                    group_plan=plan.group_plan,
                    packet_ordinal=ordinal,
                    row_custody_bridge=bridge,
                    row_object_separation_receipt_hash=separation,
                )
                bridge.seal()
                closure = transaction.mark_packet_proofs_closed(
                    packet_ordinal=ordinal,
                    provider_proof_hash=verification.provider_receipt_hash,
                    reducer_proof_hashes=(
                        verification.corrected_custody_branch_receipt_hash,
                        verification.literal_j_custody_branch_receipt_hash,
                    ),
                )
                verifications.append(verification)
                closures.append(closure)
            prepared = transaction.prepare_commit()
            commit = transaction.publish_prepared(
                expected_receipt_hash=prepared.receipt.receipt_hash
            )
    except V1SampledV7SharedDW2AcquisitionRefusal:
        raise
    except BaseException as error:
        raise V1SampledV7SharedDW2AcquisitionRefusal(
            V1SampledV7SharedDW2AcquisitionRefusalCode.ATOMIC_PUBLICATION_FAILURE,
            f"shared D_W2 acquisition aborted before atomic publication: {error}",
        ) from error
    if len(verifications) != len(plan.physical_roles) or len(closures) != len(plan.physical_roles):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.PARTIAL_COVER,
            "execution did not cover every preregistered nonempty physical stratum",
        )
    verification_axis = tuple(verifications)
    closure_axis = tuple(closures)
    receipt = V1SampledV7SharedDW2AcquisitionReceipt(
        plan_hash=plan.plan_hash,
        raw_row_prefix_identity_hash=plan.raw_row_prefix_identity_hash,
        packet_hash=plan.packet_hash,
        logical_registered_look_index=plan.logical_registered_look_index,
        logical_registered_look_stratum_counts=(
            plan.logical_registered_look_stratum_counts
        ),
        transport_packet_hashes=plan.transport_packet_hashes,
        transport_schedule_packet_hashes=plan.transport_schedule_packet_hashes,
        transport_physical_call_counts=plan.transport_physical_call_counts,
        physical_call_offset=plan.physical_call_offset,
        physical_roles=plan.physical_roles,
        provider_call_count=len(verification_axis),
        capture_batch_count=len(verification_axis),
        vjp_count=len(verification_axis),
        group_plan_hash=plan.group_plan.group_plan_hash,
        group_commit_receipt_hash=commit.receipt.receipt_hash,
        manifest_hash=commit.receipt.manifest_hash,
        provider_verification_receipt_hashes=tuple(value.receipt_hash for value in verification_axis),
        provider_proof_hashes=tuple(value.provider_receipt_hash for value in verification_axis),
        packet_proof_closure_hashes=tuple(value.closure_hash for value in closure_axis),
        raw_dual_stream_receipt_hashes=tuple(value.raw_dual_stream_receipt_hash for value in verification_axis),
        row_object_separation_receipt_hashes=tuple(value.row_object_separation_receipt_hash for value in verification_axis),
        _issuance_token=_RECEIPT_TOKEN,
    )
    return V1SampledV7SharedDW2AcquisitionResult(
        receipt=receipt,
        group_commit=commit,
        packet_proof_closures=closure_axis,
        provider_verifications=verification_axis,
        _issuance_token=_RESULT_TOKEN,
    )


def reattach_v1_sampled_v7_shared_dw2_acquisition(
    *,
    plan: V1SampledV7SharedDW2AcquisitionPlan,
    receipt: V1SampledV7SharedDW2AcquisitionReceipt,
) -> V1SampledDualChartRowLivePrefixAttachment:
    if (
        type(plan) is not V1SampledV7SharedDW2AcquisitionPlan
        or type(receipt) is not V1SampledV7SharedDW2AcquisitionReceipt
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.INVALID_TYPE,
            "recovery requires the exact shared plan and durable receipt",
        )
    plan.assert_live()
    receipt.assert_integrity()
    if (
        receipt.plan_hash != plan.plan_hash
        or receipt.group_plan_hash != plan.group_plan.group_plan_hash
        or receipt.raw_row_prefix_identity_hash != plan.raw_row_prefix_identity_hash
        or receipt.physical_call_offset != plan.physical_call_offset
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.RECOVERY_MISMATCH,
            "recovery receipt belongs to another shared acquisition",
        )
    try:
        attachment = reattach_v1_sampled_dual_chart_row_prefix(
            scratch_directory=plan.scratch_directory
        )
        attachment.verify_live_witnesses()
    except Exception as error:
        raise V1SampledV7SharedDW2AcquisitionRefusal(
            V1SampledV7SharedDW2AcquisitionRefusalCode.RECOVERY_MISMATCH,
            f"published shared acquisition could not be reattached: {error}",
        ) from error
    manifest = attachment.manifest
    if (
        manifest.manifest_hash != receipt.manifest_hash
        or len(manifest.looks) != plan.physical_call_offset + len(plan.physical_roles)
        or manifest.registered_d_prefix_hash != plan.raw_row_prefix_identity_hash
        or manifest.chart_identity_hash != plan.group_plan.chart_identity_hash
        or manifest.current_mask_hash != plan.group_plan.current_mask_hash
    ):
        _refuse(
            V1SampledV7SharedDW2AcquisitionRefusalCode.RECOVERY_MISMATCH,
            "reattached HEAD differs from the exact shared acquisition receipt",
        )
    return attachment


__all__ = [
    "SAMPLED_V7_SHARED_DW2_ACQUISITION_AUTHORITY",
    "SAMPLED_V7_SHARED_DW2_ACQUISITION_PLAN_SCHEMA",
    "SAMPLED_V7_SHARED_DW2_ACQUISITION_RECEIPT_SCHEMA",
    "V1SampledV7SharedDW2AcquisitionPlan",
    "V1SampledV7SharedDW2AcquisitionReceipt",
    "V1SampledV7SharedDW2AcquisitionRefusal",
    "V1SampledV7SharedDW2AcquisitionRefusalCode",
    "V1SampledV7SharedDW2AcquisitionResult",
    "execute_v1_sampled_v7_shared_dw2_acquisition",
    "prepare_v1_sampled_v7_shared_dw2_acquisition",
    "reattach_v1_sampled_v7_shared_dw2_acquisition",
    "v1_sampled_v7_nonempty_physical_roles",
]
