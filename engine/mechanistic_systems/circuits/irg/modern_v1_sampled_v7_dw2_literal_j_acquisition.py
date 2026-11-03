"""Physical same-capture A/J acquisition for one logical sampled ``D_W2`` packet.

One source-issued variable-shell packet is a logical target/complement unit.
The optimized scorer, however, must execute one homogeneous physical request
per stratum.  This module freezes that logical-to-physical crosswalk before
rows exist, executes exactly target then complement, and publishes all four
row shards behind one row-group HEAD update.

The result owns proposal-input row custody only.  It does not evaluate a KKT
face, stop D, open C/K/O, certify F/Q/R, commit a mask, or support a terminal
or Pareto claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, NoReturn

from mechanistic_systems.circuits.irg.modern_v1_alexnet_backend import (
    prepare_v1_sampled_raw_target_logit_dual_chart_request,
)
from mechanistic_systems.circuits.irg.modern_v1_campaign_contract import (
    V1OptionACampaignBinding,
)
from mechanistic_systems.circuits.irg.modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from mechanistic_systems.circuits.irg.modern_v1_neutral_runtime import (
    FixedAffineBCECalibration,
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
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixRegistration,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_packet_registration import (
    V1SampledVariableShellPacketRegistrationResult,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.measurement import MeasurementRequest
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    canonical_unique_string_axis,
    compact_ordinal_route_axis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    OptimizedScoringResponseProvider,
    build_route_chunked_dual_chart_fanout,
)


SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_W2_literal_J_acquisition_crosswalk_v1"
)
SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_W2_literal_J_acquisition_receipt_v1"
)
SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY: Final = (
    "same_capture_corrected_A_and_literal_J_row_custody_only_no_D_stability_"
    "F_Q_R_C_K_O_mask_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-W2-literal-J-acquisition-"
_PHYSICAL_ORDER: Final = ("target", "complement")
_CROSSWALK_TOKEN: Final = object()
_PHYSICAL_CROSSWALK_TOKEN: Final = object()
_PLAN_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()
_PRIVATE_REFERENCE_PROVIDER_TOKEN: Final = object()


class V1SampledV7DW2LiteralJAcquisitionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    AXIS_MISMATCH = "axis_mismatch"
    PROVIDER_MISMATCH = "provider_mismatch"
    PARTIAL_COVER = "partial_cover"
    REQUEST_MISMATCH = "request_mismatch"
    BATCH_PREFLIGHT_FAILURE = "batch_preflight_failure"
    PREDECESSOR_MISMATCH = "predecessor_mismatch"
    RECOVERY_MISMATCH = "recovery_mismatch"
    ATOMIC_PUBLICATION_FAILURE = "atomic_publication_failure"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7DW2LiteralJAcquisitionRefusal(RuntimeError):
    """Typed fail-closed refusal at the logical-to-physical acquisition seam."""

    def __init__(
        self,
        code: V1SampledV7DW2LiteralJAcquisitionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DW2LiteralJAcquisitionRefusalCode:
            raise TypeError("acquisition refusal code must have its exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DW2LiteralJAcquisitionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DW2LiteralJAcquisitionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_W2_ordered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + "axis-",
    )


def _canonical_route_axis(values: Sequence[str]) -> Sequence[str]:
    """Validate route custody without expanding a constructive 57M axis."""

    if type(values) is ConstructiveOrdinalRouteAxis:
        return values
    if isinstance(values, ConstructiveOrdinalRouteAxis):
        # Canonicalize subclasses back to the exact immutable grammar owner.
        # Length is O(1); no route ID is generated or retained here.
        return compact_ordinal_route_axis(len(values))
    return canonical_unique_string_axis(
        values,
        name="sampled V7 D_W2 physical route-ID axis",
    )


def _request_integrity_hash(request: MeasurementRequest) -> str:
    """Recompute the mutable request's canonical ID immediately before use."""

    if type(request) is not MeasurementRequest:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            "physical scorer request must have the exact MeasurementRequest type",
        )
    observed = content_hash(
        {
            "kind": request.kind,
            "state_hash": request.state_hash,
            "version": request.version,
            "context_selector": request.context_selector,
            "candidate_ids": tuple(request.candidate_ids),
            "expected_cost": request.expected_cost,
            "reuse_keys": tuple(request.reuse_keys),
            "provenance": request.provenance,
            "payload": dict(request.payload),
        },
        prefix="irg-request-",
    )
    if observed != request.request_id:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.REQUEST_MISMATCH,
            "physical scorer request changed after preregistration",
        )
    return observed


def _assert_packet_registration_integrity(
    packet_registration: V1SampledVariableShellPacketRegistrationResult,
) -> str:
    """Close the source-issued packet/result/append graph without caller claims."""

    if type(packet_registration) is not V1SampledVariableShellPacketRegistrationResult:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            "acquisition requires the exact source-issued packet registration result",
        )
    receipt = packet_registration.receipt
    try:
        receipt_hash = receipt.assert_integrity()
        source = packet_registration.source_registration
        append = packet_registration.append_packet_registration
        source_hash = source.registration_hash
        append_hash = append.registration_hash
    except Exception as error:
        raise V1SampledV7DW2LiteralJAcquisitionRefusal(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
            f"packet-registration graph failed live integrity: {error}",
        ) from error
    if (
        source_hash != receipt.source_registration_hash
        or append_hash != receipt.append_packet_registration_hash
        or append.lineage_hash != receipt.append_lineage_hash
        or append.packet_ordinal != receipt.packet_ordinal
        or append.registered_d_w2_prefix_hash != receipt.planned_d_w2_prefix_hash
        or source.registered_d_w2_prefix_hash != receipt.planned_d_w2_prefix_hash
        or source.registered_context_ids != receipt.registered_context_ids
        or source.registered_draw_ids != receipt.registered_draw_ids
        or source.registered_measurement_ids != receipt.registered_measurement_ids
        or source.registered_capture_ids != receipt.registered_capture_ids
        or source.registered_source_packet_hashes
        != receipt.registered_source_packet_hashes
        or append.registered_context_ids != receipt.registered_context_ids
        or append.registered_draw_ids != receipt.registered_draw_ids
        or append.registered_measurement_ids != receipt.registered_measurement_ids
        or append.registered_capture_ids != receipt.registered_capture_ids
        or append.registered_source_packet_hashes
        != receipt.registered_source_packet_hashes
        or append.registered_logical_stratum_ids != receipt.registered_stratum_ids
        or source.corrected_evidence_stream_source_hash
        != receipt.corrected_evidence_stream_source_hash
        or source.literal_j_stream_source_hash != receipt.literal_j_stream_source_hash
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
            "packet registration, append reservation, and compact source diverged",
        )
    return receipt_hash


def _provider_source_identity(
    *,
    scorer: OptimizedScoringResponseProvider,
    route_id_axis_hash: str,
    destination_transport_hash: str,
    campaign_contract_hash: str,
    model_identity_hash: str,
    declared_device: Any,
    private_reference: bool = False,
) -> str:
    campaign = _identity(
        campaign_contract_hash,
        name="campaign contract hash",
    )
    model = _identity(
        model_identity_hash,
        name="campaign model seal hash",
    )
    study = _identity(getattr(scorer, "study_hash", None), name="scorer study hash")
    chunk = getattr(scorer, "response_route_chunk_size", None)
    try:
        context_packet_size = int(getattr(scorer, "context_packet_size"))
    except Exception:
        context_packet_size = int(getattr(scorer, "registered_context_packet_size", 0))
    bindings = tuple(getattr(scorer, "conserving_activation_bindings", ()))
    if private_reference:
        evaluator_identity = "private-bounded-provider-reference"
        provider_device = str(getattr(scorer, "device_name", declared_device))
    else:
        evaluator = getattr(scorer, "evaluator", None)
        provider_device = str(getattr(evaluator, "device", ""))
        evaluator_identity = content_hash(
            {
                "provider_type": (
                    f"{type(scorer).__module__}.{type(scorer).__qualname__}"
                ),
                "evaluator_type": (
                    f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
                ),
                "topology_hash": getattr(evaluator, "topology_hash", None),
                "route_schema_hash": getattr(evaluator, "route_schema_hash", None),
                "parity_receipt_hash": getattr(evaluator, "parity_receipt_hash", None),
            },
            prefix=_PREFIX + "evaluator-source-",
        )
    if (
        isinstance(chunk, bool)
        or not isinstance(chunk, int)
        or chunk <= 0
        or context_packet_size <= 0
        or not bindings
        or provider_device != str(declared_device)
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.PROVIDER_MISMATCH,
            "scorer lacks exact chunk, activation, or declared-device custody",
        )
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_W2_scorer_source_v1",
            "provider_type": (
                f"{type(scorer).__module__}.{type(scorer).__qualname__}"
            ),
            "campaign_contract_hash": campaign,
            "model_identity_hash": model,
            "study_hash": study,
            "route_id_axis_hash": route_id_axis_hash,
            "response_route_chunk_size": chunk,
            "destination_transport_hash": destination_transport_hash,
            "context_packet_size": context_packet_size,
            "conserving_activation_binding_axis_hash": content_hash(
                bindings,
                prefix=_PREFIX + "conserving-activation-bindings-",
            ),
            "query_axis_hash": content_hash(
                tuple(getattr(scorer, "queries", ())),
                prefix=_PREFIX + "query-axis-",
            ),
            "evaluator_identity_hash": evaluator_identity,
            "declared_device": str(declared_device),
            "production_exact_provider": not private_reference,
            "private_bounded_reference_provider": private_reference,
            "optimized_tripartite_provider": True,
        },
        prefix=_PREFIX + "provider-source-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DW2LiteralJPhysicalPacketCrosswalk:
    physical_ordinal: int
    stratum_id: str
    context_count: int
    context_axis_hash: str
    draw_axis_hash: str
    measurement_request_id: str
    source_capture_identity_hash: str
    registered_packet_plan_hash: str
    row_group_packet_plan_hash: str
    corrected_a_branch_plan_identity_hash: str
    literal_j_branch_plan_identity_hash: str
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _packet_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _PHYSICAL_CROSSWALK_TOKEN:
            raise PermissionError("physical packet crosswalks are source-issued only")
        ordinal = int(self.physical_ordinal)
        count = int(self.context_count)
        if (
            isinstance(self.physical_ordinal, bool)
            or ordinal not in (0, 1)
            or self.stratum_id != _PHYSICAL_ORDER[ordinal]
            or isinstance(self.context_count, bool)
            or count <= 0
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.AXIS_MISMATCH,
                "physical packet is not the exact nonempty target/complement axis",
            )
        for name in (
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_request_id",
            "source_capture_identity_hash",
            "registered_packet_plan_hash",
            "row_group_packet_plan_hash",
            "corrected_a_branch_plan_identity_hash",
            "literal_j_branch_plan_identity_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        if (
            self.corrected_a_branch_plan_identity_hash
            == self.literal_j_branch_plan_identity_hash
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.PROVIDER_MISMATCH,
                "physical A/J branch plan identities collapsed",
            )
        object.__setattr__(self, "physical_ordinal", ordinal)
        object.__setattr__(self, "context_count", count)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "physical-crosswalk-")
        if self._packet_hash is not None and observed != self._packet_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "physical packet crosswalk changed after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "physical packet crosswalk hash differs",
            )
        object.__setattr__(self, "_packet_hash", observed)
        return observed

    @property
    def packet_crosswalk_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7DW2LiteralJAcquisitionCrosswalk:
    population_projection_registration_hash: str
    suffix_registration_hash: str
    logical_packet_hash: str
    logical_packet_ordinal: int
    packet_registration_receipt_hash: str
    source_registration_hash: str
    append_packet_registration_hash: str
    append_lineage_hash: str
    logical_source_capture_identity_hash: str
    planned_d_w2_prefix_hash: str
    stable_registered_d_prefix_hash: str
    expected_prior_manifest_hash: str | None
    predecessor_look_count: int
    logical_target_context_axis_hash: str
    logical_complement_context_axis_hash: str
    logical_pair_axis_hash: str
    campaign_contract_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    provider_source_identity_hash: str
    destination_transport_hash: str
    group_plan_hash: str
    physical_packets: tuple[V1SampledV7DW2LiteralJPhysicalPacketCrosswalk, ...]
    measurement_request_integrity_hashes: tuple[str, str]
    physical_request_order: tuple[str, str] = _PHYSICAL_ORDER
    physical_request_count: int = 2
    one_capture_and_one_vjp_per_physical_request: bool = True
    array_payload_present: bool = False
    provider_outcomes_seen: bool = False
    row_custody_authority: bool = True
    d_stability_authority: bool = False
    c_k_o_or_mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY
    schema: str = SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_CROSSWALK_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _crosswalk_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _CROSSWALK_TOKEN:
            raise PermissionError("D_W2 acquisition crosswalks are source-issued only")
        for name in (
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "logical_packet_hash",
            "packet_registration_receipt_hash",
            "source_registration_hash",
            "append_packet_registration_hash",
            "append_lineage_hash",
            "logical_source_capture_identity_hash",
            "planned_d_w2_prefix_hash",
            "stable_registered_d_prefix_hash",
            "logical_target_context_axis_hash",
            "logical_complement_context_axis_hash",
            "logical_pair_axis_hash",
            "campaign_contract_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "provider_source_identity_hash",
            "destination_transport_hash",
            "group_plan_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        packets = tuple(self.physical_packets)
        if self.expected_prior_manifest_hash is not None:
            _identity(
                self.expected_prior_manifest_hash,
                name="expected prior manifest hash",
            )
        request_hashes = tuple(self.measurement_request_integrity_hashes)
        if (
            isinstance(self.logical_packet_ordinal, bool)
            or self.logical_packet_ordinal < 0
            or isinstance(self.predecessor_look_count, bool)
            or self.predecessor_look_count != 2 * self.logical_packet_ordinal
            or (
                self.logical_packet_ordinal == 0
                and self.expected_prior_manifest_hash is not None
            )
            or (
                self.logical_packet_ordinal > 0
                and self.expected_prior_manifest_hash is None
            )
            or self.proposal_generation < 0
            or self.route_count <= 0
            or len(packets) != 2
            or tuple(packet.physical_ordinal for packet in packets) != (0, 1)
            or tuple(packet.stratum_id for packet in packets) != _PHYSICAL_ORDER
            or len(request_hashes) != 2
            or any(type(value) is not str or not value for value in request_hashes)
            or request_hashes
            != tuple(packet.measurement_request_id for packet in packets)
            or self.physical_request_order != _PHYSICAL_ORDER
            or self.physical_request_count != 2
            or self.one_capture_and_one_vjp_per_physical_request is not True
            or self.array_payload_present is not False
            or self.provider_outcomes_seen is not False
            or self.row_custody_authority is not True
            or self.d_stability_authority is not False
            or self.c_k_o_or_mask_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.authority_scope
            != SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY
            or self.schema
            != SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_CROSSWALK_SCHEMA
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "D_W2 crosswalk changed its two-request or custody-only boundary",
            )
        object.__setattr__(self, "physical_packets", packets)
        object.__setattr__(self, "measurement_request_integrity_hashes", request_hashes)
        for packet in packets:
            packet.assert_integrity()
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "physical_packets"
        }
        payload["physical_packet_crosswalk_hashes"] = tuple(
            packet.packet_crosswalk_hash for packet in self.physical_packets
        )
        return payload

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for packet in self.physical_packets:
            packet.assert_integrity()
        payload = self._payload()
        observed = content_hash(payload, prefix=_PREFIX + "crosswalk-")
        if self._crosswalk_hash is not None and observed != self._crosswalk_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "logical-to-physical crosswalk changed after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "logical-to-physical crosswalk hash differs",
            )
        object.__setattr__(self, "_crosswalk_hash", observed)
        return observed

    @property
    def crosswalk_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7DW2LiteralJAcquisitionPlan:
    crosswalk: V1SampledV7DW2LiteralJAcquisitionCrosswalk
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration = field(
        repr=False,
        compare=False,
    )
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration = field(
        repr=False,
        compare=False,
    )
    packet: V1SampledVariableShellDW2SuffixPacket = field(
        repr=False,
        compare=False,
    )
    packet_registration: V1SampledVariableShellPacketRegistrationResult = field(
        repr=False,
        compare=False,
    )
    group_plan: V1SampledDualChartRowGroupPlan = field(repr=False, compare=False)
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = field(
        repr=False,
        compare=False,
    )
    measurement_requests: tuple[MeasurementRequest, MeasurementRequest] = field(
        repr=False,
        compare=False,
    )
    scorer: OptimizedScoringResponseProvider = field(repr=False, compare=False)
    route_ids: Sequence[str] = field(repr=False, compare=False)
    declared_device: Any = field(repr=False, compare=False)
    scratch_directory: Path = field(repr=False, compare=False)
    private_reference_provider: bool = field(default=False, repr=False, compare=False)
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _plan_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _PLAN_TOKEN:
            raise PermissionError("D_W2 acquisition plans are source-issued only")
        self.assert_integrity()

    def assert_integrity(self) -> str:
        if (
            type(self.crosswalk) is not V1SampledV7DW2LiteralJAcquisitionCrosswalk
            or type(self.population_registration)
            is not V1LiteralJDFixedResidualPopulationProjectionRegistration
            or type(self.suffix_registration)
            is not V1SampledVariableShellDW2SuffixRegistration
            or type(self.packet) is not V1SampledVariableShellDW2SuffixPacket
            or type(self.packet_registration)
            is not V1SampledVariableShellPacketRegistrationResult
            or type(self.group_plan) is not V1SampledDualChartRowGroupPlan
            or (
                type(self.scorer) is not OptimizedScoringResponseProvider
                and not (
                    self.private_reference_provider
                    and isinstance(self.scorer, OptimizedScoringResponseProvider)
                )
            )
            or (
                self.prior_row_prefix is not None
                and type(self.prior_row_prefix)
                is not V1SampledDualChartRowLivePrefixAttachment
            )
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
                "D_W2 acquisition plan lost an exact source/provider type",
            )
        self.population_registration.assert_integrity(
            self.crosswalk.population_projection_registration_hash
        )
        self.suffix_registration.assert_integrity(
            self.crosswalk.suffix_registration_hash
        )
        self.packet.assert_integrity(self.crosswalk.logical_packet_hash)
        packet_registration_hash = _assert_packet_registration_integrity(
            self.packet_registration
        )
        if packet_registration_hash != self.crosswalk.packet_registration_receipt_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "packet registration differs from the acquisition crosswalk",
            )
        self.group_plan.assert_integrity(self.crosswalk.group_plan_hash)
        self.crosswalk.assert_integrity()
        prior = self.prior_row_prefix
        if prior is not None:
            prior.verify_live_witnesses()
        expected_prior_hash = None if prior is None else prior.manifest.manifest_hash
        expected_prior_looks = 0 if prior is None else len(prior.manifest.looks)
        requests = tuple(self.measurement_requests)
        if len(requests) != 2 or any(type(item) is not MeasurementRequest for item in requests):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
                "D_W2 acquisition plan lost its exact two MeasurementRequests",
            )
        physical = self.crosswalk.physical_packets
        route_hash = ordered_string_axis_hash(
            self.route_ids,
            name="sampled V7 D_W2 physical route-ID axis",
        )
        scorer_route_ids = getattr(self.scorer, "route_ids", ())
        scorer_route_hash = ordered_string_axis_hash(
            scorer_route_ids,
            name="sampled V7 D_W2 scorer route-ID axis",
        )
        destination = getattr(self.scorer, "local_destination_transport_hash", None)
        if (
            route_hash != self.crosswalk.route_id_axis_hash
            or len(self.route_ids) != self.crosswalk.route_count
            or len(scorer_route_ids) != self.crosswalk.route_count
            or scorer_route_hash != self.crosswalk.route_id_axis_hash
            or destination != self.crosswalk.destination_transport_hash
            or expected_prior_hash != self.crosswalk.expected_prior_manifest_hash
            or expected_prior_looks != self.crosswalk.predecessor_look_count
            or self.group_plan.expected_prior_manifest_hash != expected_prior_hash
            or self.group_plan.first_prefix_look_ordinal != expected_prior_looks
            or tuple(request.request_id for request in requests)
            != tuple(item.measurement_request_id for item in physical)
            or tuple(
                request.payload.get("source_capture_identity_hash")
                for request in requests
            )
            != tuple(item.source_capture_identity_hash for item in physical)
            or tuple(tuple(request.context_selector.context_ids or ()) for request in requests)
            != (self.packet.target_context_ids, self.packet.complement_context_ids)
            or _provider_source_identity(
                scorer=self.scorer,
                route_id_axis_hash=route_hash,
                destination_transport_hash=self.crosswalk.destination_transport_hash,
                campaign_contract_hash=self.crosswalk.campaign_contract_hash,
                model_identity_hash=self.crosswalk.model_identity_hash,
                declared_device=self.declared_device,
                private_reference=self.private_reference_provider,
            )
            != self.crosswalk.provider_source_identity_hash
            or tuple(_request_integrity_hash(request) for request in requests)
            != self.crosswalk.measurement_request_integrity_hashes
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "D_W2 acquisition source, scorer, request, or route identity changed",
            )
        observed = content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D_W2_live_plan_v1",
                "crosswalk_hash": self.crosswalk.crosswalk_hash,
                "measurement_request_ids": tuple(item.request_id for item in requests),
                "provider_source_identity_hash": self.crosswalk.provider_source_identity_hash,
                "group_plan_hash": self.group_plan.group_plan_hash,
                "packet_registration_receipt_hash": packet_registration_hash,
                "expected_prior_manifest_hash": expected_prior_hash,
                "predecessor_look_count": expected_prior_looks,
                "live_provider_custody": True,
            },
            prefix=_PREFIX + "plan-",
        )
        if self._plan_hash is not None and self._plan_hash != observed:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "live acquisition plan changed after issuance",
            )
        object.__setattr__(self, "_plan_hash", observed)
        return observed

    def __reduce__(self) -> NoReturn:
        raise TypeError("D_W2 acquisition plans carry live provider/path custody")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("D_W2 acquisition plans carry live provider/path custody")


@dataclass(frozen=True, slots=True)
class V1SampledV7DW2LiteralJAcquisitionReceipt:
    crosswalk_hash: str
    packet_registration_receipt_hash: str
    append_packet_registration_hash: str
    logical_packet_ordinal: int
    expected_prior_manifest_hash: str | None
    predecessor_look_count: int
    group_plan_hash: str
    group_commit_receipt_hash: str
    manifest_hash: str
    provider_verification_receipt_hashes: tuple[str, str]
    provider_proof_hashes: tuple[str, str]
    packet_proof_closure_hashes: tuple[str, str]
    raw_dual_stream_receipt_hashes: tuple[str, str]
    row_object_separation_receipt_hashes: tuple[str, str]
    corrected_custody_branch_receipt_hashes: tuple[str, str]
    literal_j_custody_branch_receipt_hashes: tuple[str, str]
    physical_request_order: tuple[str, str] = _PHYSICAL_ORDER
    provider_call_count: int = 2
    capture_batch_count: int = 2
    vjp_count: int = 2
    role_shard_count: int = 4
    atomic_group_publication: bool = True
    head_publication_is_custody_commit_boundary: bool = True
    pre_head_failure_restores_exact_predecessor: bool = True
    post_head_resume_requires_exact_prefix_reattachment: bool = True
    automatic_result_reissuance_after_process_death: bool = False
    array_payload_present: bool = False
    row_custody_authority: bool = True
    d_stability_authority: bool = False
    c_k_o_or_mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY
    schema: str = SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("acquisition receipts are executor-issued only")
        for name in (
            "crosswalk_hash",
            "packet_registration_receipt_hash",
            "append_packet_registration_hash",
            "group_plan_hash",
            "group_commit_receipt_hash",
            "manifest_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        if self.expected_prior_manifest_hash is not None:
            _identity(
                self.expected_prior_manifest_hash,
                name="expected prior manifest hash",
            )
        axes = (
            self.provider_verification_receipt_hashes,
            self.provider_proof_hashes,
            self.packet_proof_closure_hashes,
            self.raw_dual_stream_receipt_hashes,
            self.row_object_separation_receipt_hashes,
            self.corrected_custody_branch_receipt_hashes,
            self.literal_j_custody_branch_receipt_hashes,
        )
        if any(
            len(axis) != 2 or any(type(value) is not str or not value for value in axis)
            for axis in axes
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.PARTIAL_COVER,
                "D_W2 execution receipt has a partial two-packet proof axis",
            )
        if (
            self.physical_request_order != _PHYSICAL_ORDER
            or isinstance(self.logical_packet_ordinal, bool)
            or self.logical_packet_ordinal < 0
            or isinstance(self.predecessor_look_count, bool)
            or self.predecessor_look_count != 2 * self.logical_packet_ordinal
            or (
                self.logical_packet_ordinal == 0
                and self.expected_prior_manifest_hash is not None
            )
            or (
                self.logical_packet_ordinal > 0
                and self.expected_prior_manifest_hash is None
            )
            or self.provider_call_count != 2
            or self.capture_batch_count != 2
            or self.vjp_count != 2
            or self.role_shard_count != 4
            or self.atomic_group_publication is not True
            or self.head_publication_is_custody_commit_boundary is not True
            or self.pre_head_failure_restores_exact_predecessor is not True
            or self.post_head_resume_requires_exact_prefix_reattachment is not True
            or self.automatic_result_reissuance_after_process_death is not False
            or self.array_payload_present is not False
            or self.row_custody_authority is not True
            or self.d_stability_authority is not False
            or self.c_k_o_or_mask_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.authority_scope
            != SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY
            or self.schema != SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.AUTHORITY_VIOLATION,
                "D_W2 execution receipt changed its exact-two custody boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "acquisition receipt changed after issuance",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "acquisition receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7DW2LiteralJAcquisitionResult:
    receipt: V1SampledV7DW2LiteralJAcquisitionReceipt
    group_commit: V1SampledDualChartRowGroupCommit = field(repr=False, compare=False)
    packet_proof_closures: tuple[
        V1SampledDualChartRowGroupPacketProofClosure,
        V1SampledDualChartRowGroupPacketProofClosure,
    ] = field(repr=False, compare=False)
    provider_verifications: tuple[
        V1SampledDualChartCustodyProviderVerificationReceipt,
        V1SampledDualChartCustodyProviderVerificationReceipt,
    ] = field(repr=False, compare=False)
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RESULT_TOKEN:
            raise PermissionError("D_W2 acquisition results are executor-issued only")
        self.receipt.assert_integrity()
        if (
            type(self.receipt) is not V1SampledV7DW2LiteralJAcquisitionReceipt
            or type(self.group_commit) is not V1SampledDualChartRowGroupCommit
            or len(self.packet_proof_closures) != 2
            or len(self.provider_verifications) != 2
            or any(
                type(value) is not V1SampledDualChartRowGroupPacketProofClosure
                for value in self.packet_proof_closures
            )
            or any(
                type(value)
                is not V1SampledDualChartCustodyProviderVerificationReceipt
                for value in self.provider_verifications
            )
            or self.group_commit.receipt.receipt_hash
            != self.receipt.group_commit_receipt_hash
            or self.group_commit.receipt.manifest_hash != self.receipt.manifest_hash
            or tuple(value.closure_hash for value in self.packet_proof_closures)
            != self.receipt.packet_proof_closure_hashes
            or tuple(value.receipt_hash for value in self.provider_verifications)
            != self.receipt.provider_verification_receipt_hashes
            or self.group_commit.receipt.prior_manifest_hash
            != self.receipt.expected_prior_manifest_hash
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
                "D_W2 acquisition result no longer binds its exact commit/proofs",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("D_W2 acquisition results carry live row-shard attachments")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("D_W2 acquisition results carry live row-shard attachments")


def prepare_v1_sampled_v7_dw2_literal_j_acquisition(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    packet_registration: V1SampledVariableShellPacketRegistrationResult,
    binding: V1OptionACampaignBinding,
    current_mask: Any,
    mask_calibration: FixedAffineBCECalibration,
    calibration: FixedAffineBCECalibration,
    scorer: OptimizedScoringResponseProvider,
    route_ids: Sequence[str],
    declared_device: Any,
    scratch_directory: str | Path,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledV7DW2LiteralJAcquisitionPlan:
    """Freeze one production packet from its source-issued append reservation."""

    return _prepare_v1_sampled_v7_dw2_literal_j_acquisition(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        packet_registration=packet_registration,
        binding=binding,
        current_mask=current_mask,
        mask_calibration=mask_calibration,
        calibration=calibration,
        scorer=scorer,
        route_ids=route_ids,
        declared_device=declared_device,
        scratch_directory=scratch_directory,
        prior_row_prefix=prior_row_prefix,
        private_reference_token=None,
    )


def _prepare_v1_sampled_v7_dw2_literal_j_acquisition_private_reference(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    packet_registration: V1SampledVariableShellPacketRegistrationResult,
    binding: V1OptionACampaignBinding,
    current_mask: Any,
    mask_calibration: FixedAffineBCECalibration,
    calibration: FixedAffineBCECalibration,
    scorer: OptimizedScoringResponseProvider,
    route_ids: Sequence[str],
    declared_device: Any,
    scratch_directory: str | Path,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None = None,
) -> V1SampledV7DW2LiteralJAcquisitionPlan:
    """Bounded numerical-oracle seam; never a production provider capability."""

    return _prepare_v1_sampled_v7_dw2_literal_j_acquisition(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        packet_registration=packet_registration,
        binding=binding,
        current_mask=current_mask,
        mask_calibration=mask_calibration,
        calibration=calibration,
        scorer=scorer,
        route_ids=route_ids,
        declared_device=declared_device,
        scratch_directory=scratch_directory,
        prior_row_prefix=prior_row_prefix,
        private_reference_token=_PRIVATE_REFERENCE_PROVIDER_TOKEN,
    )


def _prepare_v1_sampled_v7_dw2_literal_j_acquisition(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    packet_registration: V1SampledVariableShellPacketRegistrationResult,
    binding: V1OptionACampaignBinding,
    current_mask: Any,
    mask_calibration: FixedAffineBCECalibration,
    calibration: FixedAffineBCECalibration,
    scorer: OptimizedScoringResponseProvider,
    route_ids: Sequence[str],
    declared_device: Any,
    scratch_directory: str | Path,
    prior_row_prefix: V1SampledDualChartRowLivePrefixAttachment | None,
    private_reference_token: object | None,
) -> V1SampledV7DW2LiteralJAcquisitionPlan:
    """Shared issuer after exact production/private-reference separation."""

    private_reference = private_reference_token is _PRIVATE_REFERENCE_PROVIDER_TOKEN
    if private_reference_token not in (None, _PRIVATE_REFERENCE_PROVIDER_TOKEN):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.AUTHORITY_VIOLATION,
            "unknown provider capability reached the private acquisition issuer",
        )

    if (
        type(population_registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
        or type(suffix_registration)
        is not V1SampledVariableShellDW2SuffixRegistration
        or type(packet) is not V1SampledVariableShellDW2SuffixPacket
        or type(packet_registration)
        is not V1SampledVariableShellPacketRegistrationResult
        or type(binding) is not V1OptionACampaignBinding
        or type(mask_calibration) is not FixedAffineBCECalibration
        or type(calibration) is not FixedAffineBCECalibration
        or (
            type(scorer) is not OptimizedScoringResponseProvider
            and not (
                private_reference
                and isinstance(scorer, OptimizedScoringResponseProvider)
            )
        )
        or (
            prior_row_prefix is not None
            and type(prior_row_prefix)
            is not V1SampledDualChartRowLivePrefixAttachment
        )
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            "D_W2 acquisition preparation received an inexact source/provider type",
        )
    population_registration.assert_integrity()
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    packet_registration_hash = _assert_packet_registration_integrity(
        packet_registration
    )
    source_receipt = packet_registration.receipt
    compact_source = packet_registration.source_registration
    append_registration = packet_registration.append_packet_registration
    observed = source_receipt.packet_stratum_counts
    target_count = observed[0]
    contexts_by_stratum = (
        source_receipt.registered_context_ids[:target_count],
        source_receipt.registered_context_ids[target_count:],
    )
    draw_axes = (
        source_receipt.registered_draw_ids[:target_count],
        source_receipt.registered_draw_ids[target_count:],
    )
    route_axis = _canonical_route_axis(route_ids)
    route_count = len(route_axis)
    route_id_axis_hash = ordered_string_axis_hash(
        route_axis,
        name="sampled V7 D_W2 physical route-ID axis",
    )
    current_mask_hash = _identity(
        getattr(current_mask, "mask_hash", None),
        name="current mask hash",
    )
    contract = getattr(binding, "contract", None)
    campaign_contract_hash = _identity(
        getattr(contract, "campaign_contract_hash", None),
        name="campaign contract hash",
    )
    destination = _identity(
        getattr(scorer, "local_destination_transport_hash", None),
        name="scorer destination transport hash",
    )
    corrected_stream = source_receipt.corrected_evidence_stream_source_hash
    literal_stream = source_receipt.literal_j_stream_source_hash
    if corrected_stream == literal_stream:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.PROVIDER_MISMATCH,
            "source-issued corrected-A and literal-J identities collapsed",
        )
    provider_source = _provider_source_identity(
        scorer=scorer,
        route_id_axis_hash=route_id_axis_hash,
        destination_transport_hash=destination,
        campaign_contract_hash=campaign_contract_hash,
        model_identity_hash=population_registration.model_identity_hash,
        declared_device=declared_device,
        private_reference=private_reference,
    )
    try:
        context_packet_size = int(getattr(scorer, "context_packet_size"))
    except Exception:
        context_packet_size = int(
            getattr(scorer, "registered_context_packet_size", 0)
        )
    if (
        population_registration.suffix_registration_hash
        != suffix_registration.receipt_hash
        or packet.suffix_registration_hash != suffix_registration.receipt_hash
        or packet.generation_frame_hash != suffix_registration.generation_frame_hash
        or packet.discovery_stream_hash != suffix_registration.discovery_stream_hash
        or source_receipt.suffix_registration_hash != suffix_registration.receipt_hash
        or source_receipt.packet_hash != packet.receipt_hash
        or source_receipt.packet_ordinal != packet.packet_ordinal
        or source_receipt.generation_frame_hash != packet.generation_frame_hash
        or source_receipt.discovery_stream_hash != packet.discovery_stream_hash
        or source_receipt.planned_d_w2_prefix_hash
        != compact_source.registered_d_w2_prefix_hash
        or source_receipt.append_lineage_hash != append_registration.lineage_hash
        or tuple(
            packet.after_suffix_stratum_counts[index]
            - packet.before_suffix_stratum_counts[index]
            for index in (0, 1)
        )
        != observed
        or source_receipt.after_suffix_stratum_counts
        != packet.after_suffix_stratum_counts
        or source_receipt.registered_context_ids
        != packet.target_context_ids + packet.complement_context_ids
        or contexts_by_stratum
        != (packet.target_context_ids, packet.complement_context_ids)
        or source_receipt.registered_stratum_ids
        != ("target",) * observed[0] + ("complement",) * observed[1]
        or min(observed) <= 0
        or any(
            observed[index] > population_registration.residual_stratum_counts[index]
            for index in (0, 1)
        )
        or population_registration.current_mask_hash != current_mask_hash
        or population_registration.proposal_generation
        != suffix_registration.generation_index
        or population_registration.route_count != route_count
        or population_registration.route_id_axis_hash != route_id_axis_hash
        or compact_source.model_identity_hash
        != population_registration.model_identity_hash
        or compact_source.chart_identity_hash
        != population_registration.chart_identity_hash
        or compact_source.current_mask_hash != current_mask_hash
        or compact_source.proposal_generation
        != population_registration.proposal_generation
        or compact_source.route_axis_hash != population_registration.route_axis_hash
        or compact_source.route_id_axis_hash != route_id_axis_hash
        or compact_source.literal_j_stream_source_hash
        != population_registration.residual_literal_j_stream_source_hash
        or source_receipt.literal_j_stream_source_hash
        != population_registration.residual_literal_j_stream_source_hash
        or source_receipt.corrected_evidence_stream_source_hash
        == population_registration.residual_witness_stream_source_hash
        or getattr(contract, "model_seal_hash", None)
        != population_registration.model_identity_hash
        or len(getattr(scorer, "route_ids", ())) != route_count
        or ordered_string_axis_hash(
            getattr(scorer, "route_ids", ()),
            name="sampled V7 D_W2 scorer route-ID axis",
        )
        != route_id_axis_hash
        or not tuple(getattr(scorer, "conserving_activation_bindings", ()))
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.STALE_SOURCE,
            "logical packet, population chart, scorer, mask, or route source diverged",
        )
    if max(observed) > context_packet_size:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.BATCH_PREFLIGHT_FAILURE,
            "a physical stratum would require more than one capture/VJP batch",
        )

    logical_ordinal = source_receipt.packet_ordinal
    expected_prior_look_count = 2 * logical_ordinal
    expected_prior_manifest_hash = (
        None if prior_row_prefix is None else prior_row_prefix.manifest.manifest_hash
    )
    if prior_row_prefix is not None:
        prior_row_prefix.verify_live_witnesses()
    if (
        (logical_ordinal == 0 and prior_row_prefix is not None)
        or (logical_ordinal == 0 and packet.before_suffix_stratum_counts != (0, 0))
        or (logical_ordinal > 0 and prior_row_prefix is None)
        or (
            prior_row_prefix is not None
            and len(prior_row_prefix.manifest.looks) != expected_prior_look_count
        )
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.PREDECESSOR_MISMATCH,
            "adaptive packet does not extend its exact cumulative row predecessor",
        )

    stable_registered_prefix_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_W2_row_lineage_v2",
            "population_projection_registration_hash": (
                population_registration.registration_hash
            ),
            "suffix_registration_hash": suffix_registration.receipt_hash,
            "append_lineage_hash": source_receipt.append_lineage_hash,
            "campaign_contract_hash": campaign_contract_hash,
            "provider_source_identity_hash": provider_source,
            "chart_identity_hash": population_registration.chart_identity_hash,
            "current_mask_hash": current_mask_hash,
            "route_axis_hash": population_registration.route_axis_hash,
            "route_id_axis_hash": route_id_axis_hash,
            "corrected_a_stream_identity_hash": corrected_stream,
            "literal_j_stream_identity_hash": literal_stream,
        },
        prefix=_PREFIX + "stable-row-prefix-",
    )
    if prior_row_prefix is not None:
        manifest = prior_row_prefix.manifest
        if (
            manifest.registered_d_prefix_hash != stable_registered_prefix_hash
            or manifest.chart_identity_hash
            != population_registration.chart_identity_hash
            or manifest.current_mask_hash != current_mask_hash
            or manifest.proposal_generation
            != population_registration.proposal_generation
            or manifest.route_axis_hash != population_registration.route_axis_hash
            or manifest.route_id_axis_hash != route_id_axis_hash
            or manifest.route_count != route_count
            or manifest.corrected_a_stream_identity_hash != corrected_stream
            or manifest.literal_j_stream_identity_hash != literal_stream
        ):
            _refuse(
                V1SampledV7DW2LiteralJAcquisitionRefusalCode.PREDECESSOR_MISMATCH,
                "row predecessor belongs to another source/chart/provider lineage",
            )

    source_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_D_W2_logical_source_v2",
            "population_projection_registration_hash": (
                population_registration.registration_hash
            ),
            "suffix_registration_hash": suffix_registration.receipt_hash,
            "packet_hash": packet.receipt_hash,
            "packet_registration_receipt_hash": packet_registration_hash,
            "source_registration_hash": compact_source.registration_hash,
            "append_packet_registration_hash": append_registration.registration_hash,
            "append_lineage_hash": source_receipt.append_lineage_hash,
            "logical_source_capture_identity_hash": (
                source_receipt.source_capture_identity_hash
            ),
            "planned_d_w2_prefix_hash": source_receipt.planned_d_w2_prefix_hash,
            "target_context_axis_hash": packet.target_packet_axis_hash,
            "complement_context_axis_hash": packet.complement_packet_axis_hash,
            "pair_axis_hash": packet.packet_pair_axis_hash,
            "chart_identity_hash": population_registration.chart_identity_hash,
            "current_mask_hash": current_mask_hash,
        },
        prefix=_PREFIX + "logical-source-",
    )

    requests = tuple(
        prepare_v1_sampled_raw_target_logit_dual_chart_request(
            binding=binding,
            current_mask=current_mask,
            contexts=contexts,
            mask_calibration=mask_calibration,
            calibration=calibration,
            corrected_a_stream_identity_hash=corrected_stream,
            literal_j_stream_identity_hash=literal_stream,
            logical_source_hash=source_hash,
            physical_stratum=role,
        )
        for role, contexts in zip(_PHYSICAL_ORDER, contexts_by_stratum, strict=True)
    )
    request_integrity_hashes = tuple(
        _request_integrity_hash(request) for request in requests
    )
    if (
        len(set(request_integrity_hashes)) != 2
        or len(
            {
                _identity(
                    request.payload.get("source_capture_identity_hash"),
                    name="physical source capture identity hash",
                )
                for request in requests
            }
        )
        != 2
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.REQUEST_MISMATCH,
            "target/complement requests or capture plans collapsed",
        )
    group_packets: list[V1SampledDualChartRowGroupPacketPlan] = []
    preliminary: list[tuple[str, str, str]] = []
    for ordinal, (role, contexts, draws, request) in enumerate(
        zip(
            _PHYSICAL_ORDER,
            contexts_by_stratum,
            draw_axes,
            requests,
            strict=True,
        )
    ):
        capture = _identity(
            request.payload.get("source_capture_identity_hash"),
            name=f"{role} source capture identity hash",
        )
        registered_packet = content_hash(
            {
                "schema": "partition_free_v1_sampled_v7_D_W2_physical_packet_plan_v1",
                "logical_source_hash": source_hash,
                "physical_ordinal": ordinal,
                "stratum_id": role,
                "context_axis_hash": _axis_hash(f"{role}-contexts", contexts),
                "draw_axis_hash": _axis_hash(f"{role}-draws", draws),
                "provider_plan_identity_hash": request.request_id,
                "source_capture_identity_hash": capture,
                "route_axis_hash": population_registration.route_axis_hash,
                "route_id_axis_hash": route_id_axis_hash,
                "corrected_a_stream_identity_hash": corrected_stream,
                "literal_j_stream_identity_hash": literal_stream,
            },
            prefix=_PREFIX + "physical-packet-plan-",
        )
        corrected_branch = content_hash(
            {
                "registered_packet_plan_hash": registered_packet,
                "role": "corrected_a_row_custody",
                "stream_identity_hash": corrected_stream,
            },
            prefix=_PREFIX + "branch-plan-",
        )
        literal_branch = content_hash(
            {
                "registered_packet_plan_hash": registered_packet,
                "role": "literal_j_row_custody",
                "stream_identity_hash": literal_stream,
            },
            prefix=_PREFIX + "branch-plan-",
        )
        group_packets.append(
            V1SampledDualChartRowGroupPacketPlan(
                packet_ordinal=ordinal,
                registered_packet_plan_hash=registered_packet,
                source_capture_identity_hash=capture,
                provider_plan_identity_hash=request.request_id,
                reducer_plan_identity_hashes=(corrected_branch, literal_branch),
                context_ids=contexts,
                draw_ids=draws,
                stratum_labels=(role,) * len(contexts),
            )
        )
        preliminary.append((registered_packet, corrected_branch, literal_branch))

    scratch = Path(scratch_directory).resolve()
    chunk_size = int(getattr(scorer, "response_route_chunk_size", 0))
    group_plan = prepare_v1_sampled_dual_chart_row_group_plan(
        registered_look_index=logical_ordinal + 1,
        expected_prior_manifest_hash=expected_prior_manifest_hash,
        first_prefix_look_ordinal=expected_prior_look_count,
        registered_d_prefix_hash=stable_registered_prefix_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=route_count,
        corrected_a_stream_identity_hash=corrected_stream,
        literal_j_stream_identity_hash=literal_stream,
        packet_plans=tuple(group_packets),
        registered_route_chunk_size=chunk_size,
        scratch_directory=scratch,
    )
    physical_crosswalk = tuple(
        V1SampledV7DW2LiteralJPhysicalPacketCrosswalk(
            physical_ordinal=ordinal,
            stratum_id=role,
            context_count=len(contexts),
            context_axis_hash=_axis_hash(f"{role}-contexts", contexts),
            draw_axis_hash=_axis_hash(f"{role}-draws", draws),
            measurement_request_id=request.request_id,
            source_capture_identity_hash=group_packet.source_capture_identity_hash,
            registered_packet_plan_hash=registered_packet,
            row_group_packet_plan_hash=group_packet.packet_plan_hash,
            corrected_a_branch_plan_identity_hash=corrected_branch,
            literal_j_branch_plan_identity_hash=literal_branch,
            _issuance_token=_PHYSICAL_CROSSWALK_TOKEN,
        )
        for ordinal, (
            role,
            contexts,
            draws,
            request,
            group_packet,
            (registered_packet, corrected_branch, literal_branch),
        ) in enumerate(
            zip(
                _PHYSICAL_ORDER,
                contexts_by_stratum,
                draw_axes,
                requests,
                group_packets,
                preliminary,
                strict=True,
            )
        )
    )
    crosswalk = V1SampledV7DW2LiteralJAcquisitionCrosswalk(
        population_projection_registration_hash=population_registration.registration_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        logical_packet_hash=packet.receipt_hash,
        logical_packet_ordinal=packet.packet_ordinal,
        packet_registration_receipt_hash=packet_registration_hash,
        source_registration_hash=compact_source.registration_hash,
        append_packet_registration_hash=append_registration.registration_hash,
        append_lineage_hash=source_receipt.append_lineage_hash,
        logical_source_capture_identity_hash=(
            source_receipt.source_capture_identity_hash
        ),
        planned_d_w2_prefix_hash=source_receipt.planned_d_w2_prefix_hash,
        stable_registered_d_prefix_hash=stable_registered_prefix_hash,
        expected_prior_manifest_hash=expected_prior_manifest_hash,
        predecessor_look_count=expected_prior_look_count,
        logical_target_context_axis_hash=packet.target_packet_axis_hash,
        logical_complement_context_axis_hash=packet.complement_packet_axis_hash,
        logical_pair_axis_hash=packet.packet_pair_axis_hash,
        campaign_contract_hash=campaign_contract_hash,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=route_count,
        corrected_a_stream_identity_hash=corrected_stream,
        literal_j_stream_identity_hash=literal_stream,
        provider_source_identity_hash=provider_source,
        destination_transport_hash=destination,
        group_plan_hash=group_plan.group_plan_hash,
        physical_packets=physical_crosswalk,
        measurement_request_integrity_hashes=request_integrity_hashes,  # type: ignore[arg-type]
        _issuance_token=_CROSSWALK_TOKEN,
    )
    plan = V1SampledV7DW2LiteralJAcquisitionPlan(
        crosswalk=crosswalk,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        packet_registration=packet_registration,
        group_plan=group_plan,
        prior_row_prefix=prior_row_prefix,
        measurement_requests=requests,  # type: ignore[arg-type]
        scorer=scorer,
        route_ids=route_axis,
        declared_device=declared_device,
        scratch_directory=scratch,
        private_reference_provider=private_reference,
        _issuance_token=_PLAN_TOKEN,
    )
    return plan


def execute_v1_sampled_v7_dw2_literal_j_acquisition(
    plan: V1SampledV7DW2LiteralJAcquisitionPlan,
) -> V1SampledV7DW2LiteralJAcquisitionResult:
    """Execute exactly two scorer calls and atomically publish four shards."""

    if type(plan) is not V1SampledV7DW2LiteralJAcquisitionPlan:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            "D_W2 acquisition execution requires its exact source-issued plan",
        )
    plan.assert_integrity()
    verifications: list[V1SampledDualChartCustodyProviderVerificationReceipt] = []
    closures: list[V1SampledDualChartRowGroupPacketProofClosure] = []
    try:
        with begin_v1_sampled_dual_chart_row_group_transaction(
            plan=plan.group_plan,
            scratch_directory=plan.scratch_directory,
            prior=plan.prior_row_prefix,
        ) as transaction:
            for ordinal, request in enumerate(plan.measurement_requests):
                # The request payload is a mutable Mapping by historical ABI;
                # revalidate its canonical ID, scorer source, and predecessor
                # immediately before each irreversible provider invocation.
                plan.assert_integrity()
                _request_integrity_hash(request)
                packet_plan = plan.group_plan.packet_plans[ordinal]
                witness = V1SampledDualChartChunkObjectSeparationWitness()
                bridge = V1SampledDualRowGroupPacketBridge(
                    group_plan_hash=plan.group_plan.group_plan_hash,
                    group_packet_plan=packet_plan,
                    staging_consumer=transaction.open_packet_staging_consumer(
                        packet_ordinal=ordinal
                    ),
                    route_ids=plan.route_ids,
                    device=plan.declared_device,
                    destination_transport_hash=plan.crosswalk.destination_transport_hash,
                    registered_route_chunk_size=(
                        plan.group_plan.registered_route_chunk_size
                    ),
                    corrected_a_stream_identity_hash=(
                        plan.crosswalk.corrected_a_stream_identity_hash
                    ),
                    literal_j_stream_identity_hash=(
                        plan.crosswalk.literal_j_stream_identity_hash
                    ),
                    dual_chunk_observer=witness.observe,
                )
                fanout = build_route_chunked_dual_chart_fanout(
                    evidence_consumers=(bridge.evidence_consumer,),
                    jacobian_consumers=(bridge.jacobian_consumer,),
                )
                provider_receipt = plan.scorer.accumulate_device_tripartite(
                    request,
                    accumulator=fanout,
                )
                # A historical MeasurementRequest carries a mutable payload,
                # and a provider is a live object.  Close the time-of-check /
                # time-of-use interval before accepting any returned proof;
                # this is especially necessary for the final physical call,
                # which has no subsequent loop-boundary preflight.
                plan.assert_integrity()
                _request_integrity_hash(request)
                if not isinstance(provider_receipt, Mapping):
                    raise TypeError("D_W2 scorer returned no typed provider receipt")
                branch = dict(bridge.branch_receipt("corrected_a"))
                windows = tuple(branch.get("route_windows", ()))
                separation_hash = witness.close(expected_pair_count=len(windows))
                verification = (
                    verify_v1_sampled_dual_chart_custody_only_provider_receipt(
                        provider_receipt,
                        group_plan=plan.group_plan,
                        packet_ordinal=ordinal,
                        row_custody_bridge=bridge,
                        row_object_separation_receipt_hash=separation_hash,
                    )
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
    except V1SampledV7DW2LiteralJAcquisitionRefusal:
        raise
    except BaseException as error:
        raise V1SampledV7DW2LiteralJAcquisitionRefusal(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.ATOMIC_PUBLICATION_FAILURE,
            f"D_W2 acquisition aborted before atomic publication: {error}",
        ) from error

    verification_axis = tuple(verifications)
    closure_axis = tuple(closures)
    if len(verification_axis) != 2 or len(closure_axis) != 2:
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.PARTIAL_COVER,
            "D_W2 execution did not close exactly two physical packets",
        )
    receipt = V1SampledV7DW2LiteralJAcquisitionReceipt(
        crosswalk_hash=plan.crosswalk.crosswalk_hash,
        packet_registration_receipt_hash=(
            plan.crosswalk.packet_registration_receipt_hash
        ),
        append_packet_registration_hash=(
            plan.crosswalk.append_packet_registration_hash
        ),
        logical_packet_ordinal=plan.crosswalk.logical_packet_ordinal,
        expected_prior_manifest_hash=(
            plan.crosswalk.expected_prior_manifest_hash
        ),
        predecessor_look_count=plan.crosswalk.predecessor_look_count,
        group_plan_hash=plan.group_plan.group_plan_hash,
        group_commit_receipt_hash=commit.receipt.receipt_hash,
        manifest_hash=commit.receipt.manifest_hash,
        provider_verification_receipt_hashes=tuple(
            value.receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        provider_proof_hashes=tuple(
            value.provider_receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        packet_proof_closure_hashes=tuple(
            value.closure_hash for value in closure_axis
        ),  # type: ignore[arg-type]
        raw_dual_stream_receipt_hashes=tuple(
            value.raw_dual_stream_receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        row_object_separation_receipt_hashes=tuple(
            value.row_object_separation_receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        corrected_custody_branch_receipt_hashes=tuple(
            value.corrected_custody_branch_receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        literal_j_custody_branch_receipt_hashes=tuple(
            value.literal_j_custody_branch_receipt_hash for value in verification_axis
        ),  # type: ignore[arg-type]
        _issuance_token=_RECEIPT_TOKEN,
    )
    return V1SampledV7DW2LiteralJAcquisitionResult(
        receipt=receipt,
        group_commit=commit,
        packet_proof_closures=closure_axis,  # type: ignore[arg-type]
        provider_verifications=verification_axis,  # type: ignore[arg-type]
        _issuance_token=_RESULT_TOKEN,
    )


def reattach_v1_sampled_v7_dw2_literal_j_acquisition_prefix(
    *,
    plan: V1SampledV7DW2LiteralJAcquisitionPlan,
    receipt: V1SampledV7DW2LiteralJAcquisitionReceipt,
) -> V1SampledDualChartRowLivePrefixAttachment:
    """Recover exact published row custody, never reissue execution authority.

    HEAD replacement is the custody commit boundary.  A pre-HEAD failure is
    rolled back by the row-group transaction; after HEAD, resume must present
    the exact path-free plan and executor-issued receipt and reattach the
    content-addressed prefix.  This function intentionally cannot manufacture
    provider verifications, proof closures, or a new acquisition result.
    """

    if (
        type(plan) is not V1SampledV7DW2LiteralJAcquisitionPlan
        or type(receipt) is not V1SampledV7DW2LiteralJAcquisitionReceipt
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.INVALID_TYPE,
            "recovery requires one exact acquisition plan and receipt",
        )
    plan.assert_integrity()
    receipt.assert_integrity()
    if (
        receipt.crosswalk_hash != plan.crosswalk.crosswalk_hash
        or receipt.group_plan_hash != plan.group_plan.group_plan_hash
        or receipt.packet_registration_receipt_hash
        != plan.crosswalk.packet_registration_receipt_hash
        or receipt.append_packet_registration_hash
        != plan.crosswalk.append_packet_registration_hash
        or receipt.expected_prior_manifest_hash
        != plan.crosswalk.expected_prior_manifest_hash
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.RECOVERY_MISMATCH,
            "recovery receipt belongs to another packet, plan, or predecessor",
        )
    try:
        attachment = reattach_v1_sampled_dual_chart_row_prefix(
            scratch_directory=plan.scratch_directory,
        )
        attachment.verify_live_witnesses()
    except Exception as error:
        raise V1SampledV7DW2LiteralJAcquisitionRefusal(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.RECOVERY_MISMATCH,
            f"published acquisition prefix could not be reattached: {error}",
        ) from error
    manifest = attachment.manifest
    if (
        manifest.manifest_hash != receipt.manifest_hash
        or len(manifest.looks) != receipt.predecessor_look_count + 2
        or manifest.looks[receipt.predecessor_look_count].prior_manifest_hash
        != receipt.expected_prior_manifest_hash
        or manifest.registered_d_prefix_hash
        != plan.crosswalk.stable_registered_d_prefix_hash
        or manifest.chart_identity_hash != plan.crosswalk.chart_identity_hash
        or manifest.current_mask_hash != plan.crosswalk.current_mask_hash
        or manifest.route_axis_hash != plan.crosswalk.route_axis_hash
        or manifest.route_id_axis_hash != plan.crosswalk.route_id_axis_hash
        or manifest.corrected_a_stream_identity_hash
        != plan.crosswalk.corrected_a_stream_identity_hash
        or manifest.literal_j_stream_identity_hash
        != plan.crosswalk.literal_j_stream_identity_hash
    ):
        _refuse(
            V1SampledV7DW2LiteralJAcquisitionRefusalCode.RECOVERY_MISMATCH,
            "reattached HEAD differs from the exact committed acquisition prefix",
        )
    return attachment


__all__ = (
    "SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_AUTHORITY",
    "SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_CROSSWALK_SCHEMA",
    "SAMPLED_V7_DW2_LITERAL_J_ACQUISITION_RECEIPT_SCHEMA",
    "V1SampledV7DW2LiteralJAcquisitionCrosswalk",
    "V1SampledV7DW2LiteralJAcquisitionPlan",
    "V1SampledV7DW2LiteralJAcquisitionReceipt",
    "V1SampledV7DW2LiteralJAcquisitionRefusal",
    "V1SampledV7DW2LiteralJAcquisitionRefusalCode",
    "V1SampledV7DW2LiteralJAcquisitionResult",
    "V1SampledV7DW2LiteralJPhysicalPacketCrosswalk",
    "execute_v1_sampled_v7_dw2_literal_j_acquisition",
    "prepare_v1_sampled_v7_dw2_literal_j_acquisition",
    "reattach_v1_sampled_v7_dw2_literal_j_acquisition_prefix",
)
