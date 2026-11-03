"""Exact row-free device-D bridge into compact post-D replay.

This module is the ordinary source-native sibling of the historical V7 row
custody bridge.  It executes only the existing V4 zero-publication device-D
transaction, binds each pre-outcome provider/capture request before execution,
and retains scalar/hash boundaries only.  No row shard, manifest, scratch
directory, HEAD, or route-scale intermediate is created.

The final bridge receipt consumes the exact live materialization, its exact
authority-inert rank receipt, the registered-D transaction and prefix binding,
and the complete ordered look request/result/proof chain.  Only that receipt
may ask the compact replay module to issue its canonical final-source type.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_composition_backend import (
    V1SampledCompositionRegisteredDPrefixBinding,
    V1SampledCompositionRegisteredDTransaction,
    build_v1_sampled_composition_registered_d_prefix_binding,
)
from .modern_v1_sampled_device_resident_d_session import (
    V1SampledDeviceResidentDLookReceipt,
    V1SampledDeviceResidentDMaterializationRankReceipt,
    V1SampledDeviceResidentDSession,
    V1SampledDeviceResidentDSessionMaterialization,
    V1SampledDeviceResidentDSessionMode,
    V1SampledDeviceResidentDRowFreeFinalizationReceipt,
)
from .modern_v1_sampled_look_transaction import (
    SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4,
    V1SampledDeviceResidentDTransactionResult,
    V1SampledRegisteredLookTransaction,
    V1SampledRegisteredLookTransactionRequest,
    _provider_plan_identity_hash,
    _source_capture_identity_hash,
)
from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_sampled_v7_compact_post_d_replay import (
    V1SampledV7CompactDeviceResidentDFinalSourceReceipt,
    V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2,
    issue_v1_sampled_v7_compact_device_final_source_from_exact_bridge,
    issue_v1_sampled_v7_compact_row_free_final_source_from_exact_bridge_v2,
    v1_sampled_v7_compact_sampling_measure_hash,
)
from .modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)


COMPACT_DEVICE_D_PACKET_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_packet_registration_v1"
)
COMPACT_DEVICE_D_LOOK_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_look_registration_v1"
)
COMPACT_DEVICE_D_PACKET_BOUNDARY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_packet_boundary_v1"
)
COMPACT_DEVICE_D_LOOK_BOUNDARY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_look_boundary_v1"
)
COMPACT_DEVICE_D_LOOK_COMMIT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_look_commit_v1"
)
COMPACT_DEVICE_D_FINAL_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_final_bridge_v1"
)
COMPACT_DEVICE_D_ROW_FREE_FINAL_BRIDGE_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_v7_compact_device_D_row_free_final_bridge_v2"
)
COMPACT_DEVICE_D_AUTHORITY_SCOPE: Final = (
    "row_free_device_D_source_custody_only_no_D_stability_C_K_O_commit_"
    "runtime_terminal_validation_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-compact-device-D-bridge-"
_DEVICE_D_CHUNK_SOURCE_PREFIX: Final = (
    "partition-free-v1-sampled-device-resident-D-same-capture-chunk-"
)
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-context-axis-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-draw-axis-"
_STRATUM_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-stratum-axis-"
_PACKET_AXIS_PREFIX: Final = "partition-free-v1-literal-j-replay-packet-axis-"
_SOURCE_MEASUREMENT_AXIS_PREFIX: Final = (
    "partition-free-v1-literal-j-replay-source-measurement-axis-"
)
_UNWEIGHTED_PREFIX: Final = (
    "partition-free-v1-sampled-device-final-rank-source-unweighted-prefix-"
)
_PACKET_REGISTRATION_TOKEN: Final = object()
_LOOK_REGISTRATION_TOKEN: Final = object()
_PACKET_BOUNDARY_TOKEN: Final = object()
_LOOK_BOUNDARY_TOKEN: Final = object()
_LOOK_COMMIT_TOKEN: Final = object()
_FINAL_BRIDGE_TOKEN: Final = object()
_FINAL_BRIDGE_V2_TOKEN: Final = object()


class V1SampledV7CompactDeviceDBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    REQUEST_DRIFT = "request_drift"
    PREFIX_DRIFT = "prefix_drift"
    PRIOR_CHAIN_DRIFT = "prior_chain_drift"
    PROVIDER_DRIFT = "provider_drift"
    CAPTURE_DRIFT = "capture_drift"
    STREAM_DRIFT = "stream_drift"
    PACKET_DRIFT = "packet_drift"
    MATERIALIZATION_DRIFT = "materialization_drift"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7CompactDeviceDBridgeRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CompactDeviceDBridgeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CompactDeviceDBridgeRefusalCode:
            raise TypeError("compact device-D refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CompactDeviceDBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CompactDeviceDBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if not item.name.startswith("_")
    }


def _axis_hash(role: str, values: Sequence[object]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_device_D_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + role + "-axis-",
    )


def _all_authority_closed(value: object) -> None:
    for name in (
        "d_stability_authority",
        "c_stream_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "runtime_authority",
        "terminal_authority",
        "pareto_authority",
    ):
        if getattr(value, name, None) is not False:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                f"{name} must remain closed",
            )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceDPacketRegistration:
    source_packet_ordinal: int
    logical_look_index: int
    packet_ordinal_within_logical_look: int
    transaction_request_hash: str
    packet_plan_hash: str
    provider_plan_identity_hash: str
    source_capture_identity_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    source_measurement_hashes: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    route_axis_declaration_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    population_stratum_counts: tuple[int, ...]
    sealed_population_order_hashes: tuple[str, ...]
    outcome_independent: bool = True
    row_attachment_present: bool = False
    schema: str = COMPACT_DEVICE_D_PACKET_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PACKET_REGISTRATION_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "packet registrations are bridge-issued",
            )
        for name in (
            "transaction_request_hash",
            "packet_plan_hash",
            "provider_plan_identity_hash",
            "source_capture_identity_hash",
            "route_axis_declaration_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_packet_ordinal",
            _nonnegative_int(self.source_packet_ordinal, name="source packet ordinal"),
        )
        object.__setattr__(
            self,
            "logical_look_index",
            _positive_int(self.logical_look_index, name="logical look index"),
        )
        object.__setattr__(
            self,
            "packet_ordinal_within_logical_look",
            _nonnegative_int(
                self.packet_ordinal_within_logical_look,
                name="packet ordinal within logical look",
            ),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        axes = (
            tuple(self.context_ids),
            tuple(self.draw_ids),
            tuple(self.source_measurement_hashes),
            tuple(self.stratum_labels),
        )
        if (
            any(not axis for axis in axes)
            or len({len(axis) for axis in axes}) != 1
            or any(type(item) is not str or not item for axis in axes for item in axis)
            or len(set(axes[0])) != len(axes[0])
            or len(set(axes[1])) != len(axes[1])
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.route_axis_hash == self.route_id_axis_hash
            or self.outcome_independent is not True
            or self.row_attachment_present is not False
            or self.schema != COMPACT_DEVICE_D_PACKET_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
                "packet registration axes, streams, or outcome boundary are invalid",
            )
        counts = tuple(
            _positive_int(value, name="population stratum count")
            for value in self.population_stratum_counts
        )
        orders = tuple(
            _identity(value, name="sealed population order hash")
            for value in self.sealed_population_order_hashes
        )
        if len(counts) != len(orders):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
                "population counts and orders do not align",
            )
        object.__setattr__(self, "context_ids", axes[0])
        object.__setattr__(self, "draw_ids", axes[1])
        object.__setattr__(self, "source_measurement_hashes", axes[2])
        object.__setattr__(self, "stratum_labels", axes[3])
        object.__setattr__(self, "population_stratum_counts", counts)
        object.__setattr__(self, "sealed_population_order_hashes", orders)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "packet-registration-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "packet registration changed after issuance",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceDLookRegistration:
    transaction_request_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    route_axis_declaration_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    logical_look_index: int
    source_packet_ordinal_start: int
    packets: tuple[V1SampledV7CompactDeviceDPacketRegistration, ...]
    request_prefix_frozen: bool = True
    row_scratch_bytes: int = 0
    downstream_authority: bool = False
    schema: str = COMPACT_DEVICE_D_LOOK_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _LOOK_REGISTRATION_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "look registrations are bridge-issued",
            )
        for name in (
            "transaction_request_hash",
            "session_design_hash",
            "hbm_forecast_hash",
            "route_axis_declaration_hash",
            "route_axis_hash",
            "route_id_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        look = _positive_int(self.logical_look_index, name="logical look index")
        start = _nonnegative_int(
            self.source_packet_ordinal_start,
            name="source packet ordinal start",
        )
        packets = tuple(self.packets)
        route_count = _positive_int(self.route_count, name="route count")
        if (
            not packets
            or any(
                type(packet) is not V1SampledV7CompactDeviceDPacketRegistration
                for packet in packets
            )
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(start, start + len(packets)))
            or tuple(
                packet.packet_ordinal_within_logical_look for packet in packets
            )
            != tuple(range(len(packets)))
            or any(
                packet.logical_look_index != look
                or packet.transaction_request_hash
                != self.transaction_request_hash
                or packet.route_axis_declaration_hash
                != self.route_axis_declaration_hash
                or packet.route_axis_hash != self.route_axis_hash
                or packet.route_id_axis_hash != self.route_id_axis_hash
                or packet.route_count != route_count
                for packet in packets
            )
            or self.route_axis_hash == self.route_id_axis_hash
            or self.request_prefix_frozen is not True
            or self.row_scratch_bytes != 0
            or self.downstream_authority is not False
            or self.schema != COMPACT_DEVICE_D_LOOK_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
                "logical-look packet registration is partial, reordered, or privileged",
            )
        for packet in packets:
            packet.assert_integrity()
        object.__setattr__(self, "logical_look_index", look)
        object.__setattr__(self, "source_packet_ordinal_start", start)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packets"] = tuple(packet.registration_hash for packet in self.packets)
        return content_hash(value, prefix=_PREFIX + "look-registration-")

    def assert_integrity(self) -> str:
        for packet in self.packets:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "look registration changed after issuance",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()


def register_v1_sampled_v7_compact_device_d_look(
    *,
    request: V1SampledRegisteredLookTransactionRequest,
    measurement_requests: tuple[object, ...],
    session: V1SampledDeviceResidentDSession,
    physical_route_axis: DeclaredRouteAxis,
    source_packet_ordinal_start: int,
) -> V1SampledV7CompactDeviceDLookRegistration:
    """Freeze one V4 logical look before provider outcomes exist."""

    if (
        type(request) is not V1SampledRegisteredLookTransactionRequest
        or type(measurement_requests) is not tuple
        or type(session) is not V1SampledDeviceResidentDSession
        or type(physical_route_axis) is not DeclaredRouteAxis
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "look registration requires exact request/session/physical-axis owners and a tuple request axis",
        )
    if request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4:
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
            "compact row-free bridge admits only the existing V4 transaction",
        )
    if len(measurement_requests) != len(request.packet_plans):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
            "measurement-request axis is partial",
        )
    session_design = session.design
    first = request.packet_plans[0]
    if (
        session_design.design_hash != session.hbm_forecast.design_hash
        or session_design.fd_design.design_hash != request.design.design_hash
        or session_design.route_axis_hash != physical_route_axis.route_id_axis_hash
        or session_design.route_count != physical_route_axis.route_count
        or physical_route_axis.route_axis_hash
        == physical_route_axis.route_id_axis_hash
        or session_design.corrected_a_stream_identity_hash
        != first.corrected_evidence_stream_identity_hash
        or session_design.literal_j_stream_identity_hash
        != first.literal_jacobian_stream_identity_hash
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.STREAM_DRIFT,
            "look request and live session do not share one A/J chart",
        )
    start = _nonnegative_int(
        source_packet_ordinal_start,
        name="source packet ordinal start",
    )
    packets = []
    provider_ids = []
    capture_ids = []
    for local_ordinal, (plan, measurement) in enumerate(
        zip(request.packet_plans, measurement_requests, strict=True)
    ):
        try:
            provider_id = _provider_plan_identity_hash(measurement)
        except (TypeError, ValueError) as error:
            raise V1SampledV7CompactDeviceDBridgeRefusal(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PROVIDER_DRIFT,
                "measurement request lacks one unambiguous provider-plan identity",
            ) from error
        try:
            capture_id = _source_capture_identity_hash(measurement)
        except (TypeError, ValueError) as error:
            raise V1SampledV7CompactDeviceDBridgeRefusal(
                V1SampledV7CompactDeviceDBridgeRefusalCode.CAPTURE_DRIFT,
                "measurement request lacks one unambiguous source-capture identity",
            ) from error
        packet = plan.reduced_request.packet_request
        if (
            not plan.same_capture_and_vjp_required
            or plan.corrected_evidence_stream_identity_hash
            != first.corrected_evidence_stream_identity_hash
            or plan.literal_jacobian_stream_identity_hash
            != first.literal_jacobian_stream_identity_hash
            or packet.route_axis_hash != session_design.route_axis_hash
            or packet.route_axis_hash != physical_route_axis.route_id_axis_hash
            or packet.route_count != session_design.route_count
            or packet.population_stratum_counts
            != session_design.population_stratum_counts
            or packet.sealed_population_order_hashes
            != session_design.sealed_population_order_hashes
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.STREAM_DRIFT,
                "one packet changed the registered session or A/J chart",
            )
        packets.append(
            V1SampledV7CompactDeviceDPacketRegistration(
                source_packet_ordinal=start + local_ordinal,
                logical_look_index=request.registered_look_index,
                packet_ordinal_within_logical_look=local_ordinal,
                transaction_request_hash=request.request_hash,
                packet_plan_hash=plan.plan_hash,
                provider_plan_identity_hash=provider_id,
                source_capture_identity_hash=capture_id,
                context_ids=tuple(packet.expected_context_ids),
                draw_ids=tuple(packet.expected_draw_ids),
                source_measurement_hashes=tuple(
                    plan.fd_request.source_measurement_hashes
                ),
                stratum_labels=tuple(plan.fd_request.stratum_labels),
                route_axis_declaration_hash=physical_route_axis.content_hash,
                route_axis_hash=physical_route_axis.route_axis_hash,
                route_id_axis_hash=physical_route_axis.route_id_axis_hash,
                route_count=packet.route_count,
                corrected_a_stream_identity_hash=(
                    plan.corrected_evidence_stream_identity_hash
                ),
                literal_j_stream_identity_hash=(
                    plan.literal_jacobian_stream_identity_hash
                ),
                population_stratum_counts=tuple(
                    packet.population_stratum_counts
                ),
                sealed_population_order_hashes=tuple(
                    packet.sealed_population_order_hashes
                ),
                _issuance_token=_PACKET_REGISTRATION_TOKEN,
            )
        )
        provider_ids.append(provider_id)
        capture_ids.append(capture_id)
    if len(set(provider_ids)) != len(provider_ids) or len(set(capture_ids)) != len(
        capture_ids
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
            "one logical look aliases provider or capture identities across packets",
        )
    return V1SampledV7CompactDeviceDLookRegistration(
        transaction_request_hash=request.request_hash,
        session_design_hash=session_design.design_hash,
        hbm_forecast_hash=session.hbm_forecast.forecast_hash,
        route_axis_declaration_hash=physical_route_axis.content_hash,
        route_axis_hash=physical_route_axis.route_axis_hash,
        route_id_axis_hash=physical_route_axis.route_id_axis_hash,
        route_count=physical_route_axis.route_count,
        logical_look_index=request.registered_look_index,
        source_packet_ordinal_start=start,
        packets=tuple(packets),
        _issuance_token=_LOOK_REGISTRATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceDPacketBoundary:
    packet_registration_hash: str
    source_packet_ordinal: int
    logical_look_index: int
    packet_ordinal_within_logical_look: int
    transaction_result_hash: str
    look_receipt_hash: str
    provider_proof_hash: str
    raw_dual_proof_hash: str
    same_capture_vjp_source_hash: str
    route_windows: tuple[tuple[int, int], ...]
    same_capture_chunk_source_axis_hash: str
    route_scale_d2h_bytes: int = 0
    row_scratch_bytes: int = 0
    row_head_present: bool = False
    row_manifest_present: bool = False
    row_shard_count: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_DEVICE_D_AUTHORITY_SCOPE
    schema: str = COMPACT_DEVICE_D_PACKET_BOUNDARY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PACKET_BOUNDARY_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "packet boundaries are bridge-issued",
            )
        for name in (
            "packet_registration_hash",
            "transaction_result_hash",
            "look_receipt_hash",
            "provider_proof_hash",
            "raw_dual_proof_hash",
            "same_capture_vjp_source_hash",
            "same_capture_chunk_source_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_packet_ordinal",
            _nonnegative_int(self.source_packet_ordinal, name="source packet ordinal"),
        )
        object.__setattr__(
            self,
            "logical_look_index",
            _positive_int(self.logical_look_index, name="logical look index"),
        )
        object.__setattr__(
            self,
            "packet_ordinal_within_logical_look",
            _nonnegative_int(
                self.packet_ordinal_within_logical_look,
                name="packet ordinal within logical look",
            ),
        )
        windows = tuple((int(start), int(stop)) for start, stop in self.route_windows)
        if (
            not windows
            or windows[0][0] != 0
            or any(
                start < 0
                or stop <= start
                or (index > 0 and start != windows[index - 1][1])
                for index, (start, stop) in enumerate(windows)
            )
            or self.route_scale_d2h_bytes != 0
            or self.row_scratch_bytes != 0
            or self.row_head_present is not False
            or self.row_manifest_present is not False
            or self.row_shard_count != 0
            or self.authority_scope != COMPACT_DEVICE_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_DEVICE_D_PACKET_BOUNDARY_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PACKET_DRIFT,
                "packet boundary is partial or retained row custody",
            )
        _all_authority_closed(self)
        object.__setattr__(self, "route_windows", windows)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "packet-boundary-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "packet boundary changed after issuance",
            )
        return observed

    @property
    def boundary_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceDLookBoundary:
    look_registration_hash: str
    transaction_request_hash: str
    transaction_result_hash: str
    prior_boundary_hash: str | None
    prior_transaction_result_hash: str | None
    session_design_hash: str
    hbm_forecast_hash: str
    logical_look_index: int
    source_packet_ordinal_start: int
    source_packet_ordinal_stop: int
    packet_boundaries: tuple[V1SampledV7CompactDeviceDPacketBoundary, ...]
    provider_vjp_count: int
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    row_scratch_bytes: int = 0
    row_head_present: bool = False
    row_manifest_present: bool = False
    row_shard_count: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_DEVICE_D_AUTHORITY_SCOPE
    schema: str = COMPACT_DEVICE_D_LOOK_BOUNDARY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _LOOK_BOUNDARY_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "look boundaries are bridge-issued",
            )
        for name in (
            "look_registration_hash",
            "transaction_request_hash",
            "transaction_result_hash",
            "session_design_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("prior_boundary_hash", "prior_transaction_result_hash"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        look = _positive_int(self.logical_look_index, name="logical look index")
        start = _nonnegative_int(
            self.source_packet_ordinal_start,
            name="source packet ordinal start",
        )
        stop = _positive_int(
            self.source_packet_ordinal_stop,
            name="source packet ordinal stop",
        )
        packets = tuple(self.packet_boundaries)
        if (
            stop <= start
            or not packets
            or stop - start != len(packets)
            or any(
                type(packet) is not V1SampledV7CompactDeviceDPacketBoundary
                for packet in packets
            )
            or tuple(packet.source_packet_ordinal for packet in packets)
            != tuple(range(start, stop))
            or tuple(
                packet.packet_ordinal_within_logical_look for packet in packets
            )
            != tuple(range(len(packets)))
            or any(packet.logical_look_index != look for packet in packets)
            or _positive_int(self.provider_vjp_count, name="provider VJP count")
            != len(packets)
            or (look == 1)
            is not (
                self.prior_boundary_hash is None
                and self.prior_transaction_result_hash is None
            )
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or self.row_scratch_bytes != 0
            or self.row_head_present is not False
            or self.row_manifest_present is not False
            or self.row_shard_count != 0
            or self.authority_scope != COMPACT_DEVICE_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_DEVICE_D_LOOK_BOUNDARY_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                "logical-look boundary is reordered, partial, or row-backed",
            )
        for packet in packets:
            packet.assert_integrity()
        _all_authority_closed(self)
        object.__setattr__(self, "logical_look_index", look)
        object.__setattr__(self, "source_packet_ordinal_start", start)
        object.__setattr__(self, "source_packet_ordinal_stop", stop)
        object.__setattr__(self, "packet_boundaries", packets)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packet_boundaries"] = tuple(
            packet.boundary_hash for packet in self.packet_boundaries
        )
        return content_hash(value, prefix=_PREFIX + "look-boundary-")

    def assert_integrity(self) -> str:
        for packet in self.packet_boundaries:
            packet.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "look boundary changed after issuance",
            )
        return observed

    @property
    def boundary_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceDLookCommit:
    registration: V1SampledV7CompactDeviceDLookRegistration
    boundary: V1SampledV7CompactDeviceDLookBoundary
    resident_result: V1SampledDeviceResidentDTransactionResult = field(
        repr=False,
        compare=False,
    )
    downstream_authority: bool = False
    schema: str = COMPACT_DEVICE_D_LOOK_COMMIT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _LOOK_COMMIT_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "look commits are bridge-issued",
            )
        if (
            type(self.registration) is not V1SampledV7CompactDeviceDLookRegistration
            or type(self.boundary) is not V1SampledV7CompactDeviceDLookBoundary
            or type(self.resident_result)
            is not V1SampledDeviceResidentDTransactionResult
            or self.registration.registration_hash
            != self.boundary.look_registration_hash
            or self.resident_result.result_hash
            != self.boundary.transaction_result_hash
            or self.downstream_authority is not False
            or self.schema != COMPACT_DEVICE_D_LOOK_COMMIT_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "look commit does not bind its exact registration/result/boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "registration_hash": self.registration.registration_hash,
                "boundary_hash": self.boundary.boundary_hash,
                "resident_result_hash": self.resident_result.result_hash,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "look-commit-",
        )

    def assert_integrity(self) -> str:
        self.registration.assert_integrity()
        self.boundary.assert_integrity()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "look commit changed after issuance",
            )
        return observed

    @property
    def commit_hash(self) -> str:
        return self.assert_integrity()


def _same_capture_chunk_source_hashes(
    *,
    request_hash: str,
    packet_plan_hash: str,
    packet_ordinal_within_logical_look: int,
    route_windows: Sequence[tuple[int, int]],
    corrected_a_stream_identity_hash: str,
    literal_j_stream_identity_hash: str,
) -> tuple[str, ...]:
    return tuple(
        content_hash(
            {
                "schema": "partition_free_v1_device_resident_D_same_capture_chunk_v1",
                "transaction_request_hash": request_hash,
                "packet_plan_hash": packet_plan_hash,
                "packet_ordinal": packet_ordinal_within_logical_look,
                "route_window": (int(start), int(stop)),
                "corrected_evidence_stream_identity_hash": (
                    corrected_a_stream_identity_hash
                ),
                "literal_jacobian_stream_identity_hash": (
                    literal_j_stream_identity_hash
                ),
                "same_capture_and_vjp_required": True,
                "route_scale_values_not_hashed": True,
            },
            prefix=_DEVICE_D_CHUNK_SOURCE_PREFIX,
        )
        for start, stop in route_windows
    )


def execute_v1_sampled_v7_compact_device_d_look(
    *,
    registration: V1SampledV7CompactDeviceDLookRegistration,
    request: V1SampledRegisteredLookTransactionRequest,
    measurement_requests: tuple[object, ...],
    provider: object,
    session: V1SampledDeviceResidentDSession,
    prior_commit: V1SampledV7CompactDeviceDLookCommit | None = None,
    allow_cpu_reference_test: bool = False,
) -> V1SampledV7CompactDeviceDLookCommit:
    """Execute one frozen logical look through the existing row-free V4 path."""

    if (
        type(registration) is not V1SampledV7CompactDeviceDLookRegistration
        or type(request) is not V1SampledRegisteredLookTransactionRequest
        or type(measurement_requests) is not tuple
        or type(session) is not V1SampledDeviceResidentDSession
        or type(allow_cpu_reference_test) is not bool
        or (
            prior_commit is not None
            and type(prior_commit) is not V1SampledV7CompactDeviceDLookCommit
        )
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "compact look execution requires exact frozen owners",
        )
    try:
        registration.assert_integrity()
        if (
            session.design.execution_mode
            is V1SampledDeviceResidentDSessionMode.CPU_REFERENCE_TEST_ONLY
            and not allow_cpu_reference_test
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.CPU_PRODUCTION_FALLBACK,
                "production compact device-D bridge refuses CPU fallback",
            )
        if (
            request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4
            or request.request_hash != registration.transaction_request_hash
            or request.registered_look_index != registration.logical_look_index
            or session.design.design_hash != registration.session_design_hash
            or session.hbm_forecast.forecast_hash
            != registration.hbm_forecast_hash
            or len(measurement_requests) != len(registration.packets)
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.REQUEST_DRIFT,
                "execution request/session differs from preregistration",
            )
        observed_provider = tuple(
            _provider_plan_identity_hash(value) for value in measurement_requests
        )
        observed_capture = tuple(
            _source_capture_identity_hash(value) for value in measurement_requests
        )
        if observed_provider != tuple(
            packet.provider_plan_identity_hash for packet in registration.packets
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PROVIDER_DRIFT,
                "provider-plan axis changed after freeze",
            )
        if observed_capture != tuple(
            packet.source_capture_identity_hash for packet in registration.packets
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.CAPTURE_DRIFT,
                "source-capture axis changed after freeze",
            )
        if registration.logical_look_index == 1:
            if prior_commit is not None or request.prior_device_resident_result is not None:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                    "first logical look cannot carry a prior",
                )
            prior_boundary_hash = None
            prior_result_hash = None
        else:
            if prior_commit is None:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                    "continued logical look requires the exact prior commit",
                )
            prior_commit.assert_integrity()
            prior_result = request.prior_device_resident_result
            if (
                type(prior_result) is not V1SampledDeviceResidentDTransactionResult
                or prior_commit.boundary.logical_look_index + 1
                != registration.logical_look_index
                or prior_commit.boundary.source_packet_ordinal_stop
                != registration.source_packet_ordinal_start
                or prior_result.result_hash != prior_commit.resident_result.result_hash
                or prior_commit.boundary.session_design_hash
                != registration.session_design_hash
                or prior_commit.boundary.hbm_forecast_hash
                != registration.hbm_forecast_hash
            ):
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                    "continued logical look changed its prior result or packet cursor",
                )
            prior_boundary_hash = prior_commit.boundary.boundary_hash
            prior_result_hash = prior_commit.resident_result.result_hash
        result = V1SampledRegisteredLookTransaction(
            request
        ).execute_into_device_resident_d_session(
            provider=provider,
            measurement_requests=measurement_requests,
            device_resident_d_session=session,
        )
        if (
            result.transaction_request_hash != request.request_hash
            or result.registered_look_index != registration.logical_look_index
            or result.prior_result_hash != prior_result_hash
            or len(result.look_receipts) != len(registration.packets)
            or len(result.provider_receipt_hashes) != len(registration.packets)
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                "device-resident result changed request, prior, or packet count",
            )
        packet_boundaries = []
        for packet_registration, look, provider_proof in zip(
            registration.packets,
            result.look_receipts,
            result.provider_receipt_hashes,
            strict=True,
        ):
            packet_registration.assert_integrity()
            local = packet_registration.packet_ordinal_within_logical_look
            chunk_sources = _same_capture_chunk_source_hashes(
                request_hash=request.request_hash,
                packet_plan_hash=packet_registration.packet_plan_hash,
                packet_ordinal_within_logical_look=local,
                route_windows=look.route_windows,
                corrected_a_stream_identity_hash=(
                    packet_registration.corrected_a_stream_identity_hash
                ),
                literal_j_stream_identity_hash=(
                    packet_registration.literal_j_stream_identity_hash
                ),
            )
            expected_chunk_axis = content_hash(
                chunk_sources,
                prefix=(
                    "partition-free-v1-sampled-device-resident-D-"
                    "same-capture-chunk-axis-"
                ),
            )
            if (
                look.registered_look_index != registration.logical_look_index
                or look.packet_ordinal != local
                or look.route_axis_hash != packet_registration.route_id_axis_hash
                or look.route_count != packet_registration.route_count
                or look.context_axis_hash
                != content_hash(
                    packet_registration.context_ids,
                    prefix=(
                        "partition-free-v1-sampled-device-resident-D-"
                        "look-context-axis-"
                    ),
                )
                or look.draw_axis_hash
                != content_hash(
                    packet_registration.draw_ids,
                    prefix=(
                        "partition-free-v1-sampled-device-resident-D-"
                        "look-draw-axis-"
                    ),
                )
                or look.source_measurement_axis_hash
                != content_hash(
                    packet_registration.source_measurement_hashes,
                    prefix=(
                        "partition-free-v1-sampled-device-resident-D-"
                        "look-source-measurement-axis-"
                    ),
                )
                or look.same_capture_chunk_source_axis_hash != expected_chunk_axis
                or look.route_scale_d2h_bytes != 0
                or look.route_scale_host_tensor_materializations != 0
            ):
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PACKET_DRIFT,
                    "one live look receipt differs from its frozen packet",
                )
            raw_dual_proof = content_hash(
                {
                    "schema": (
                        "partition_free_v1_sampled_v7_compact_device_D_"
                        "raw_dual_proof_v1"
                    ),
                    "packet_registration_hash": (
                        packet_registration.registration_hash
                    ),
                    "look_receipt_hash": look.receipt_hash,
                    "provider_proof_hash": provider_proof,
                    "source_capture_identity_hash": (
                        packet_registration.source_capture_identity_hash
                    ),
                    "corrected_a_stream_identity_hash": (
                        packet_registration.corrected_a_stream_identity_hash
                    ),
                    "literal_j_stream_identity_hash": (
                        packet_registration.literal_j_stream_identity_hash
                    ),
                    "same_capture_and_vjp_verified_by_V4_provider_proof": True,
                },
                prefix=_PREFIX + "raw-dual-proof-",
            )
            same_capture_source = content_hash(
                {
                    "packet_registration_hash": packet_registration.registration_hash,
                    "provider_proof_hash": provider_proof,
                    "raw_dual_proof_hash": raw_dual_proof,
                    "same_capture_chunk_source_axis_hash": expected_chunk_axis,
                },
                prefix=_PREFIX + "same-capture-vjp-source-",
            )
            packet_boundaries.append(
                V1SampledV7CompactDeviceDPacketBoundary(
                    packet_registration_hash=packet_registration.registration_hash,
                    source_packet_ordinal=(
                        packet_registration.source_packet_ordinal
                    ),
                    logical_look_index=registration.logical_look_index,
                    packet_ordinal_within_logical_look=local,
                    transaction_result_hash=result.result_hash,
                    look_receipt_hash=look.receipt_hash,
                    provider_proof_hash=provider_proof,
                    raw_dual_proof_hash=raw_dual_proof,
                    same_capture_vjp_source_hash=same_capture_source,
                    route_windows=look.route_windows,
                    same_capture_chunk_source_axis_hash=expected_chunk_axis,
                    _issuance_token=_PACKET_BOUNDARY_TOKEN,
                )
            )
        boundary = V1SampledV7CompactDeviceDLookBoundary(
            look_registration_hash=registration.registration_hash,
            transaction_request_hash=request.request_hash,
            transaction_result_hash=result.result_hash,
            prior_boundary_hash=prior_boundary_hash,
            prior_transaction_result_hash=prior_result_hash,
            session_design_hash=registration.session_design_hash,
            hbm_forecast_hash=registration.hbm_forecast_hash,
            logical_look_index=registration.logical_look_index,
            source_packet_ordinal_start=registration.source_packet_ordinal_start,
            source_packet_ordinal_stop=(
                registration.source_packet_ordinal_start
                + len(registration.packets)
            ),
            packet_boundaries=tuple(packet_boundaries),
            provider_vjp_count=len(packet_boundaries),
            _issuance_token=_LOOK_BOUNDARY_TOKEN,
        )
        return V1SampledV7CompactDeviceDLookCommit(
            registration=registration,
            boundary=boundary,
            resident_result=result,
            _issuance_token=_LOOK_COMMIT_TOKEN,
        )
    except BaseException:
        # Any refused continuation is one deterministic replay boundary.  The
        # underlying V4 executor already evicts on provider failure; this
        # idempotent guard also covers preflight identity refusals.
        if type(session) is V1SampledDeviceResidentDSession:
            session.evict()
        raise


def _flatten_packet_axis(
    registrations: Sequence[V1SampledV7CompactDeviceDPacketRegistration],
    name: str,
) -> tuple[str, ...]:
    return tuple(
        value
        for registration in registrations
        for value in getattr(registration, name)
    )


def build_v1_sampled_v7_compact_registered_d_prefix_binding(
    *,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    registered_requests: tuple[V1SampledRegisteredLookTransactionRequest, ...],
    look_commits: tuple[V1SampledV7CompactDeviceDLookCommit, ...],
    corrected_moments: V1StratifiedCorrectedAttributionMoments,
) -> V1SampledCompositionRegisteredDPrefixBinding:
    """Derive the ordinary registered-D prefix from exact row-free owners."""

    if (
        type(registered_d_transaction)
        is not V1SampledCompositionRegisteredDTransaction
        or type(registered_requests) is not tuple
        or type(look_commits) is not tuple
        or type(corrected_moments)
        is not V1StratifiedCorrectedAttributionMoments
        or not registered_requests
        or len(registered_requests) != len(look_commits)
        or any(
            type(request) is not V1SampledRegisteredLookTransactionRequest
            for request in registered_requests
        )
        or any(
            type(commit) is not V1SampledV7CompactDeviceDLookCommit
            for commit in look_commits
        )
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "compact registered-D binding requires exact transaction, requests, commits, and moments",
        )
    corrected_moments.assert_integrity()
    for commit in look_commits:
        commit.assert_integrity()
    registrations = tuple(
        packet
        for commit in look_commits
        for packet in commit.registration.packets
    )
    if (
        tuple(request.request_hash for request in registered_requests)
        != tuple(commit.registration.transaction_request_hash for commit in look_commits)
        or tuple(request.registered_look_index for request in registered_requests)
        != tuple(range(1, len(registered_requests) + 1))
        or registered_d_transaction.registered_look_count
        != len(registered_requests)
        or registered_d_transaction.replay_look_indices
        != tuple(request.registered_look_index for request in registered_requests)
        or tuple(packet.source_packet_ordinal for packet in registrations)
        != tuple(range(len(registrations)))
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.PREFIX_DRIFT,
            "registered requests/commits do not form the exact transaction prefix",
        )
    contexts = _flatten_packet_axis(registrations, "context_ids")
    draws = _flatten_packet_axis(registrations, "draw_ids")
    measurements = _flatten_packet_axis(registrations, "source_measurement_hashes")
    first = registrations[0]
    if (
        len(contexts) != len(draws)
        or len(contexts) != len(measurements)
        or len(contexts) != sum(corrected_moments.stratum_counts)
        or corrected_moments.current_mask_hash
        != registered_requests[0].packet_plans[0].reduced_request.packet_request.current_mask_hash
        or corrected_moments.proposal_generation
        != registered_requests[0].packet_plans[0].reduced_request.packet_request.proposal_generation
        or corrected_moments.route_axis_hash != first.route_id_axis_hash
        or corrected_moments.route_count != first.route_count
        or corrected_moments.raw_logit_stream_source_hash
        != first.corrected_a_stream_identity_hash
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.PREFIX_DRIFT,
            "corrected moments differ from the exact ordered row-free prefix",
        )
    weight_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_device_D_unweighted_prefix_v1",
            "stratum_counts": tuple(corrected_moments.stratum_counts),
            "packet_stratum_labels": tuple(
                packet.stratum_labels for packet in registrations
            ),
            "unweighted_within_stratum": True,
        },
        prefix=_UNWEIGHTED_PREFIX,
    )
    return build_v1_sampled_composition_registered_d_prefix_binding(
        rank_invariant_raw_prefix_coordinate=(
            registered_d_transaction.rank_invariant_raw_prefix_coordinate
        ),
        registered_look_count=registered_d_transaction.registered_look_count,
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        current_mask_hash=corrected_moments.current_mask_hash,
        route_axis_hash=first.route_axis_hash,
        route_count=first.route_count,
        proposal_generation=corrected_moments.proposal_generation,
        cumulative_stratum_counts=tuple(corrected_moments.stratum_counts),
        D_context_axis_hash=content_hash(contexts, prefix=_CONTEXT_AXIS_PREFIX),
        D_draw_axis_hash=content_hash(draws, prefix=_DRAW_AXIS_PREFIX),
        D_source_measurement_axis_hash=content_hash(
            measurements,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        D_weight_hash=weight_hash,
    )


def build_v1_sampled_v7_compact_registered_d_prefix_binding_row_free_v2(
    *,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    registered_requests: tuple[V1SampledRegisteredLookTransactionRequest, ...],
    look_commits: tuple[V1SampledV7CompactDeviceDLookCommit, ...],
    row_free_finalization: V1SampledDeviceResidentDRowFreeFinalizationReceipt,
) -> V1SampledCompositionRegisteredDPrefixBinding:
    """Bind the exact D prefix from scalar session custody, never moments."""

    if (
        type(registered_d_transaction)
        is not V1SampledCompositionRegisteredDTransaction
        or type(registered_requests) is not tuple
        or type(look_commits) is not tuple
        or type(row_free_finalization)
        is not V1SampledDeviceResidentDRowFreeFinalizationReceipt
        or not registered_requests
        or len(registered_requests) != len(look_commits)
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "row-free v2 prefix binding requires exact scalar owners",
        )
    for commit in look_commits:
        if type(commit) is not V1SampledV7CompactDeviceDLookCommit:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
                "row-free v2 prefix contains another commit type",
            )
        commit.assert_integrity()
    registrations = tuple(
        packet
        for commit in look_commits
        for packet in commit.registration.packets
    )
    if (
        any(
            type(request) is not V1SampledRegisteredLookTransactionRequest
            for request in registered_requests
        )
        or tuple(request.request_hash for request in registered_requests)
        != tuple(commit.registration.transaction_request_hash for commit in look_commits)
        or tuple(request.registered_look_index for request in registered_requests)
        != tuple(range(1, len(registered_requests) + 1))
        or registered_d_transaction.registered_look_count != len(registered_requests)
        or tuple(packet.source_packet_ordinal for packet in registrations)
        != tuple(range(len(registrations)))
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.PREFIX_DRIFT,
            "row-free v2 request/commit prefix is partial or reordered",
        )
    first = registrations[0]
    fd_design = registered_requests[0].design
    strata = tuple(fd_design.stratum_names)
    contexts = _flatten_packet_axis(registrations, "context_ids")
    draws = _flatten_packet_axis(registrations, "draw_ids")
    measurements = _flatten_packet_axis(registrations, "source_measurement_hashes")
    counts = tuple(
        sum(label == stratum for packet in registrations for label in packet.stratum_labels)
        for stratum in strata
    )
    session_prefix = "partition-free-v1-sampled-device-resident-D-"
    grouped_context_hashes = tuple(
        content_hash(
            tuple(
                context
                for packet in registrations
                for context, label in zip(
                    packet.context_ids,
                    packet.stratum_labels,
                    strict=True,
                )
                if label == stratum
            ),
            prefix=session_prefix + "final-context-axis-",
        )
        for stratum in strata
    )
    grouped_draw_hashes = tuple(
        content_hash(
            tuple(
                draw
                for packet in registrations
                for draw, label in zip(
                    packet.draw_ids,
                    packet.stratum_labels,
                    strict=True,
                )
                if label == stratum
            ),
            prefix=session_prefix + "final-draw-axis-",
        )
        for stratum in strata
    )
    grouped_measurement_hashes = tuple(
        content_hash(
            tuple(
                measurement
                for packet in registrations
                for measurement, label in zip(
                    packet.source_measurement_hashes,
                    packet.stratum_labels,
                    strict=True,
                )
                if label == stratum
            ),
            prefix=session_prefix + "final-source-measurement-axis-",
        )
        for stratum in strata
    )
    look_receipts = tuple(
        receipt
        for commit in look_commits
        for receipt in commit.resident_result.look_receipts
    )
    if (
        row_free_finalization.final_logical_look_index != len(registered_requests)
        or row_free_finalization.stratum_counts != counts
        or row_free_finalization.context_axis_hashes_by_stratum
        != grouped_context_hashes
        or row_free_finalization.draw_axis_hashes_by_stratum != grouped_draw_hashes
        or row_free_finalization.source_measurement_axis_hashes_by_stratum
        != grouped_measurement_hashes
        or row_free_finalization.look_receipt_axis_hash
        != content_hash(
            tuple(receipt.receipt_hash for receipt in look_receipts),
            prefix=session_prefix + "final-look-receipt-axis-",
        )
        or row_free_finalization.source_packet_axis_hash
        != content_hash(
            tuple(receipt.same_capture_chunk_source_axis_hash for receipt in look_receipts),
            prefix=session_prefix + "final-source-packet-axis-",
        )
        or row_free_finalization.route_axis_hash != first.route_id_axis_hash
        or row_free_finalization.route_count != first.route_count
        or row_free_finalization.corrected_a_stream_identity_hash
        != first.corrected_a_stream_identity_hash
        or row_free_finalization.literal_j_stream_identity_hash
        != first.literal_j_stream_identity_hash
        or row_free_finalization.route_scale_d2h_bytes != 0
        or row_free_finalization.portable_corrected_moments_present is not False
        or row_free_finalization.portable_fd_snapshot_present is not False
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.PREFIX_DRIFT,
            "row-free scalar finalization differs from exact request/commit custody",
        )
    packet = registered_requests[0].packet_plans[0].reduced_request.packet_request
    weight_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_device_D_unweighted_prefix_v2",
            "stratum_counts": counts,
            "packet_stratum_labels": tuple(
                value.stratum_labels for value in registrations
            ),
            "row_free_finalization_receipt_hash": row_free_finalization.receipt_hash,
            "unweighted_within_stratum": True,
        },
        prefix=_UNWEIGHTED_PREFIX,
    )
    return build_v1_sampled_composition_registered_d_prefix_binding(
        rank_invariant_raw_prefix_coordinate=(
            registered_d_transaction.rank_invariant_raw_prefix_coordinate
        ),
        registered_look_count=registered_d_transaction.registered_look_count,
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        current_mask_hash=packet.current_mask_hash,
        route_axis_hash=first.route_axis_hash,
        route_count=first.route_count,
        proposal_generation=packet.proposal_generation,
        cumulative_stratum_counts=counts,
        D_context_axis_hash=content_hash(contexts, prefix=_CONTEXT_AXIS_PREFIX),
        D_draw_axis_hash=content_hash(draws, prefix=_DRAW_AXIS_PREFIX),
        D_source_measurement_axis_hash=content_hash(
            measurements,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        D_weight_hash=weight_hash,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceResidentDFinalSourceBridge:
    """Compact exact-owner closure; it retains no route-scale source rows."""

    session_materialization_hash: str
    materialization_rank_receipt_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    corrected_moments_state_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    physical_route_axis_declaration_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    packet_registrations: tuple[V1SampledV7CompactDeviceDPacketRegistration, ...]
    packet_boundaries: tuple[V1SampledV7CompactDeviceDPacketBoundary, ...]
    look_receipts: tuple[V1SampledDeviceResidentDLookReceipt, ...]
    route_scale_source_row_d2h_bytes: int = 0
    row_scratch_bytes: int = 0
    row_head_present: bool = False
    row_manifest_present: bool = False
    row_shard_count: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_DEVICE_D_AUTHORITY_SCOPE
    schema: str = COMPACT_DEVICE_D_FINAL_BRIDGE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FINAL_BRIDGE_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "final compact bridges are exact-owner issued",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        looks = _positive_int(self.registered_look_count, name="registered look count")
        packets = tuple(self.packet_registrations)
        boundaries = tuple(self.packet_boundaries)
        look_receipts = tuple(self.look_receipts)
        if (
            not packets
            or len(packets) != len(boundaries)
            or len(packets) != len(look_receipts)
            or any(
                type(value) is not V1SampledV7CompactDeviceDPacketRegistration
                for value in packets
            )
            or any(
                type(value) is not V1SampledV7CompactDeviceDPacketBoundary
                for value in boundaries
            )
            or any(
                type(value) is not V1SampledDeviceResidentDLookReceipt
                for value in look_receipts
            )
            or tuple(value.source_packet_ordinal for value in packets)
            != tuple(range(len(packets)))
            or tuple(value.source_packet_ordinal for value in boundaries)
            != tuple(range(len(boundaries)))
            or any(
                packet.registration_hash != boundary.packet_registration_hash
                or packet.logical_look_index != boundary.logical_look_index
                or packet.packet_ordinal_within_logical_look
                != boundary.packet_ordinal_within_logical_look
                or boundary.look_receipt_hash != look.receipt_hash
                or packet.logical_look_index != look.registered_look_index
                or packet.packet_ordinal_within_logical_look != look.packet_ordinal
                for packet, boundary, look in zip(
                    packets,
                    boundaries,
                    look_receipts,
                    strict=True,
                )
            )
            or any(
                packet.route_axis_declaration_hash
                != self.physical_route_axis_declaration_hash
                or packet.route_axis_hash != self.route_axis_hash
                or packet.route_id_axis_hash != self.route_id_axis_hash
                or packet.route_count != self.route_count
                for packet in packets
            )
            or max(packet.logical_look_index for packet in packets) != looks
            or _positive_int(self.route_count, name="route count")
            != packets[0].route_count
            or _positive_int(self.population_count, name="population count")
            < sum(len(packet.context_ids) for packet in packets)
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.route_axis_hash == self.route_id_axis_hash
            or self.route_scale_source_row_d2h_bytes != 0
            or self.row_scratch_bytes != 0
            or self.row_head_present is not False
            or self.row_manifest_present is not False
            or self.row_shard_count != 0
            or self.authority_scope != COMPACT_DEVICE_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_DEVICE_D_FINAL_BRIDGE_SCHEMA
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MATERIALIZATION_DRIFT,
                "final compact bridge is partial, reordered, row-backed, or privileged",
            )
        for packet, boundary, look in zip(
            packets,
            boundaries,
            look_receipts,
            strict=True,
        ):
            packet.assert_integrity()
            boundary.assert_integrity()
            if look.assert_integrity() != boundary.look_receipt_hash:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PACKET_DRIFT,
                    "look receipt changed after packet-boundary issuance",
                )
        _all_authority_closed(self)
        object.__setattr__(self, "proposal_generation", _nonnegative_int(self.proposal_generation, name="proposal generation"))
        object.__setattr__(self, "registered_look_count", looks)
        object.__setattr__(self, "packet_registrations", packets)
        object.__setattr__(self, "packet_boundaries", boundaries)
        object.__setattr__(self, "look_receipts", look_receipts)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packet_registrations"] = tuple(
            packet.registration_hash for packet in self.packet_registrations
        )
        value["packet_boundaries"] = tuple(
            boundary.boundary_hash for boundary in self.packet_boundaries
        )
        value["look_receipts"] = tuple(
            receipt.receipt_hash for receipt in self.look_receipts
        )
        return content_hash(value, prefix=_PREFIX + "final-bridge-")

    def assert_integrity(self) -> str:
        for packet, boundary, look in zip(
            self.packet_registrations,
            self.packet_boundaries,
            self.look_receipts,
            strict=True,
        ):
            packet.assert_integrity()
            boundary.assert_integrity()
            if look.assert_integrity() != boundary.look_receipt_hash:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                    "look receipt changed after final-bridge issuance",
                )
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "final compact bridge changed after exact-owner issuance",
            )
        return observed

    @property
    def bridge_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7CompactDeviceResidentDRowFreeFinalSourceBridgeV2:
    """Exact scalar/live-device D closure without portable route arrays."""

    row_free_finalization_receipt_hash: str
    materialization_rank_receipt_hash: str
    session_design_hash: str
    hbm_forecast_hash: str
    registered_d_transaction_hash: str
    registered_d_prefix_binding_hash: str
    registered_request_prefix_hash: str
    literal_j_source_receipt_hash: str
    fd_design_hash: str
    corrected_reacquisition_source_hash: str
    live_fd_source_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    physical_route_axis_declaration_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    registered_look_count: int
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    measurement_axis_hash: str
    packet_axis_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_order_axis_hash: str
    sampling_measure_hash: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    population_stratum_counts: tuple[int, ...]
    registered_stratum_counts: tuple[int, ...]
    packet_registrations: tuple[V1SampledV7CompactDeviceDPacketRegistration, ...]
    packet_boundaries: tuple[V1SampledV7CompactDeviceDPacketBoundary, ...]
    look_receipts: tuple[V1SampledDeviceResidentDLookReceipt, ...]
    corrected_f_reacquisition_required: bool = True
    live_device_fd_owner_required: bool = True
    portable_corrected_moments_present: bool = False
    portable_fd_snapshot_present: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    row_scratch_bytes: int = 0
    row_head_present: bool = False
    row_manifest_present: bool = False
    row_shard_count: int = 0
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = COMPACT_DEVICE_D_AUTHORITY_SCOPE
    schema: str = COMPACT_DEVICE_D_ROW_FREE_FINAL_BRIDGE_SCHEMA_V2
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FINAL_BRIDGE_V2_TOKEN:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.AUTHORITY_VIOLATION,
                "row-free v2 bridges are exact scalar/live-device issued",
            )
        for item in fields(self):
            if not item.name.startswith("_") and item.name.endswith("_hash"):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        packets = tuple(self.packet_registrations)
        boundaries = tuple(self.packet_boundaries)
        looks = tuple(self.look_receipts)
        names = tuple(_identity(value, name="stratum name") for value in self.stratum_names)
        weights = tuple(float(value) for value in self.stratum_weights)
        population = tuple(
            _positive_int(value, name="population stratum count")
            for value in self.population_stratum_counts
        )
        registered = tuple(
            _positive_int(value, name="registered stratum count")
            for value in self.registered_stratum_counts
        )
        observed_registered = tuple(
            sum(label == name for packet in packets for label in packet.stratum_labels)
            for name in names
        )
        expected_measure_hash = v1_sampled_v7_compact_sampling_measure_hash(
            fd_design_hash=self.fd_design_hash,
            stratum_names=names,
            stratum_weights=weights,
            population_stratum_counts=population,
            registered_stratum_counts=registered,
            population_order_axis_hash=self.population_order_axis_hash,
        )
        flags = (
            self.corrected_f_reacquisition_required,
            self.live_device_fd_owner_required,
            self.portable_corrected_moments_present,
            self.portable_fd_snapshot_present,
            self.row_head_present,
            self.row_manifest_present,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        look_count = _positive_int(
            self.registered_look_count,
            name="registered look count",
        )
        if (
            not packets
            or len(packets) != len(boundaries)
            or len(packets) != len(looks)
            or any(
                type(value) is not V1SampledV7CompactDeviceDPacketRegistration
                for value in packets
            )
            or any(
                type(value) is not V1SampledV7CompactDeviceDPacketBoundary
                for value in boundaries
            )
            or any(
                type(value) is not V1SampledDeviceResidentDLookReceipt
                for value in looks
            )
            or tuple(value.source_packet_ordinal for value in packets)
            != tuple(range(len(packets)))
            or tuple(value.source_packet_ordinal for value in boundaries)
            != tuple(range(len(boundaries)))
            or any(
                packet.registration_hash != boundary.packet_registration_hash
                or packet.logical_look_index != boundary.logical_look_index
                or packet.packet_ordinal_within_logical_look
                != boundary.packet_ordinal_within_logical_look
                or boundary.look_receipt_hash != look.receipt_hash
                or packet.logical_look_index != look.registered_look_index
                or packet.packet_ordinal_within_logical_look != look.packet_ordinal
                or packet.route_axis_declaration_hash
                != self.physical_route_axis_declaration_hash
                or packet.route_axis_hash != self.route_axis_hash
                or packet.route_id_axis_hash != self.route_id_axis_hash
                or packet.route_count != self.route_count
                for packet, boundary, look in zip(
                    packets,
                    boundaries,
                    looks,
                    strict=True,
                )
            )
            or max(packet.logical_look_index for packet in packets) != look_count
            or _positive_int(self.route_count, name="route count")
            != packets[0].route_count
            or _positive_int(self.population_count, name="population count")
            < sum(len(packet.context_ids) for packet in packets)
            or len({len(names), len(weights), len(population), len(registered)}) != 1
            or len(set(names)) != len(names)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or not math.isclose(
                math.fsum(weights),
                1.0,
                rel_tol=0.0,
                abs_tol=128.0 * math.ulp(1.0),
            )
            or sum(population) != self.population_count
            or sum(registered) != sum(len(packet.context_ids) for packet in packets)
            or registered != observed_registered
            or self.sampling_measure_hash != expected_measure_hash
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.route_axis_hash == self.route_id_axis_hash
            or any(type(flag) is not bool for flag in flags)
            or flags[:2] != (True, True)
            or any(flags[2:])
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or self.row_scratch_bytes != 0
            or self.row_shard_count != 0
            or self.authority_scope != COMPACT_DEVICE_D_AUTHORITY_SCOPE
            or self.schema != COMPACT_DEVICE_D_ROW_FREE_FINAL_BRIDGE_SCHEMA_V2
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MATERIALIZATION_DRIFT,
                "row-free v2 bridge is partial, portable, row-backed, or privileged",
            )
        for packet, boundary, look in zip(
            packets,
            boundaries,
            looks,
            strict=True,
        ):
            packet.assert_integrity()
            boundary.assert_integrity()
            if look.assert_integrity() != boundary.look_receipt_hash:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.PACKET_DRIFT,
                    "look receipt changed after row-free packet boundary",
                )
        _all_authority_closed(self)
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(self, "registered_look_count", look_count)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", population)
        object.__setattr__(self, "registered_stratum_counts", registered)
        object.__setattr__(self, "packet_registrations", packets)
        object.__setattr__(self, "packet_boundaries", boundaries)
        object.__setattr__(self, "look_receipts", looks)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        value = _payload(self)
        value["packet_registrations"] = tuple(
            packet.registration_hash for packet in self.packet_registrations
        )
        value["packet_boundaries"] = tuple(
            boundary.boundary_hash for boundary in self.packet_boundaries
        )
        value["look_receipts"] = tuple(
            receipt.receipt_hash for receipt in self.look_receipts
        )
        return content_hash(value, prefix=_PREFIX + "row-free-final-bridge-v2-")

    def assert_integrity(self) -> str:
        for packet, boundary, look in zip(
            self.packet_registrations,
            self.packet_boundaries,
            self.look_receipts,
            strict=True,
        ):
            packet.assert_integrity()
            boundary.assert_integrity()
            if look.assert_integrity() != boundary.look_receipt_hash:
                _refuse(
                    V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                    "row-free v2 look receipt changed after bridge issuance",
                )
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.MUTATED_RECEIPT,
                "row-free v2 bridge changed after exact-owner issuance",
            )
        return observed

    @property
    def bridge_hash(self) -> str:
        return self.assert_integrity()


def issue_v1_sampled_v7_compact_device_resident_d_row_free_final_source_v2(
    *,
    row_free_finalization: V1SampledDeviceResidentDRowFreeFinalizationReceipt,
    materialization_rank_receipt: V1SampledDeviceResidentDMaterializationRankReceipt,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    registered_d_prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
    physical_route_axis: DeclaredRouteAxis,
    registered_requests: tuple[V1SampledRegisteredLookTransactionRequest, ...],
    look_commits: tuple[V1SampledV7CompactDeviceDLookCommit, ...],
) -> V1SampledV7CompactDeviceResidentDRowFreeFinalSourceReceiptV2:
    """Close scalar/live-device D custody without moments or FD snapshots."""

    exact = (
        (row_free_finalization, V1SampledDeviceResidentDRowFreeFinalizationReceipt),
        (
            materialization_rank_receipt,
            V1SampledDeviceResidentDMaterializationRankReceipt,
        ),
        (registered_d_transaction, V1SampledCompositionRegisteredDTransaction),
        (
            registered_d_prefix_binding,
            V1SampledCompositionRegisteredDPrefixBinding,
        ),
        (physical_route_axis, DeclaredRouteAxis),
    )
    if (
        any(type(value) is not expected for value, expected in exact)
        or type(registered_requests) is not tuple
        or type(look_commits) is not tuple
        or not registered_requests
        or len(registered_requests) != len(look_commits)
        or any(
            type(value) is not V1SampledRegisteredLookTransactionRequest
            for value in registered_requests
        )
        or any(
            type(value) is not V1SampledV7CompactDeviceDLookCommit
            for value in look_commits
        )
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "row-free v2 final issuance requires every exact scalar owner",
        )
    for commit in look_commits:
        commit.assert_integrity()
    registrations = tuple(
        packet
        for commit in look_commits
        for packet in commit.registration.packets
    )
    boundaries = tuple(
        packet
        for commit in look_commits
        for packet in commit.boundary.packet_boundaries
    )
    look_receipts = tuple(
        receipt
        for commit in look_commits
        for receipt in commit.resident_result.look_receipts
    )
    requests = tuple(registered_requests)
    first_packet = requests[0].packet_plans[0].reduced_request.packet_request
    fd_design = requests[0].design
    context_axis = _flatten_packet_axis(registrations, "context_ids")
    draw_axis = _flatten_packet_axis(registrations, "draw_ids")
    stratum_axis = _flatten_packet_axis(registrations, "stratum_labels")
    measurement_axis = _flatten_packet_axis(
        registrations,
        "source_measurement_hashes",
    )
    population_counts = tuple(first_packet.population_stratum_counts)
    population_orders = tuple(first_packet.sealed_population_order_hashes)
    stratum_names = tuple(fd_design.stratum_names)
    stratum_weights = tuple(float(value) for value in fd_design.stratum_weights)
    registered_stratum_counts = tuple(
        sum(label == name for label in stratum_axis) for name in stratum_names
    )
    population_order_axis_hash = content_hash(
        population_orders,
        prefix=_PREFIX + "population-order-axis-",
    )
    sampling_measure_hash = v1_sampled_v7_compact_sampling_measure_hash(
        fd_design_hash=fd_design.design_hash,
        stratum_names=stratum_names,
        stratum_weights=stratum_weights,
        population_stratum_counts=population_counts,
        registered_stratum_counts=registered_stratum_counts,
        population_order_axis_hash=population_order_axis_hash,
    )
    packet_axis = tuple(
        (
            registration.packet_plan_hash,
            registration.source_capture_identity_hash,
            boundary.provider_proof_hash,
            boundary.raw_dual_proof_hash,
            boundary.same_capture_vjp_source_hash,
        )
        for registration, boundary in zip(
            registrations,
            boundaries,
            strict=True,
        )
    )
    expected_prior_result = None
    expected_prior_boundary = None
    prior_chain_closed = True
    for look_index, (request, commit) in enumerate(
        zip(requests, look_commits, strict=True),
        start=1,
    ):
        prior_chain_closed = prior_chain_closed and (
            request.schema == SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4
            and request.registered_look_index == look_index
            and commit.registration.logical_look_index == look_index
            and commit.boundary.logical_look_index == look_index
            and commit.boundary.prior_transaction_result_hash
            == expected_prior_result
            and commit.boundary.prior_boundary_hash == expected_prior_boundary
            and request.prior_device_resident_result_hash == expected_prior_result
        )
        expected_prior_result = commit.resident_result.result_hash
        expected_prior_boundary = commit.boundary.boundary_hash
    final_result = look_commits[-1].resident_result
    closure = {
        "request_order": tuple(request.request_hash for request in requests)
        == tuple(
            commit.registration.transaction_request_hash for commit in look_commits
        ),
        "prior_chain": prior_chain_closed,
        "look_count": len(requests)
        == registered_d_transaction.registered_look_count
        == registered_d_prefix_binding.registered_look_count
        == row_free_finalization.final_logical_look_index,
        "packet_global_axis": tuple(
            packet.source_packet_ordinal for packet in registrations
        )
        == tuple(range(len(registrations))),
        "packet_boundary_axis": tuple(
            packet.source_packet_ordinal for packet in boundaries
        )
        == tuple(range(len(boundaries))),
        "rank": row_free_finalization.materialization_rank_receipt_hash
        == materialization_rank_receipt.receipt_hash
        and row_free_finalization.registered_proposal_fd_rank
        == materialization_rank_receipt.registered_proposal_fd_rank
        == registered_d_transaction.physical_fd_rank,
        "session": row_free_finalization.session_design_hash
        == materialization_rank_receipt.session_design_hash
        == look_commits[0].boundary.session_design_hash,
        "forecast": row_free_finalization.hbm_forecast_hash
        == materialization_rank_receipt.hbm_forecast_hash
        == look_commits[0].boundary.hbm_forecast_hash,
        "fd_design": registered_d_transaction.fd_design_hash
        == fd_design.design_hash,
        "sampling_measure": (
            tuple(first_packet.population_stratum_counts) == population_counts
            and tuple(fd_design.stratum_names) == stratum_names
            and tuple(float(value) for value in fd_design.stratum_weights)
            == stratum_weights
            and set(stratum_axis) == set(stratum_names)
            and sum(registered_stratum_counts) == len(context_axis)
        ),
        "physical_axis": physical_route_axis.route_count
        == row_free_finalization.route_count
        == registrations[0].route_count
        and physical_route_axis.route_axis_hash
        == registrations[0].route_axis_hash
        == registered_d_prefix_binding.route_axis_hash
        and physical_route_axis.route_id_axis_hash
        == registrations[0].route_id_axis_hash
        == row_free_finalization.route_axis_hash
        == fd_design.route_axis_hash
        and physical_route_axis.content_hash
        == registrations[0].route_axis_declaration_hash
        and physical_route_axis.route_axis_hash
        != physical_route_axis.route_id_axis_hash,
        "final_result": final_result.cumulative_packet_count_after
        == len(registrations)
        and final_result.cumulative_look_receipt_hashes
        == tuple(receipt.receipt_hash for receipt in look_receipts)
        and final_result.cumulative_stratum_counts_after
        == row_free_finalization.stratum_counts
        == registered_d_prefix_binding.cumulative_stratum_counts,
        "prefix": registered_d_transaction.rank_invariant_raw_prefix_hash
        == registered_d_prefix_binding.rank_invariant_raw_prefix_hash
        and registered_d_transaction.literal_j_source_receipt_hash
        == registered_d_prefix_binding.literal_j_source_receipt_hash
        and registered_d_transaction.source_replay_axis_binding_hash
        == registered_d_prefix_binding.binding_hash
        and registered_d_prefix_binding.D_context_axis_hash
        == content_hash(context_axis, prefix=_CONTEXT_AXIS_PREFIX)
        and registered_d_prefix_binding.D_draw_axis_hash
        == content_hash(draw_axis, prefix=_DRAW_AXIS_PREFIX)
        and registered_d_prefix_binding.D_source_measurement_axis_hash
        == content_hash(
            measurement_axis,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        "chart": registered_d_prefix_binding.current_mask_hash
        == first_packet.current_mask_hash
        and registered_d_prefix_binding.route_count == first_packet.route_count
        and registered_d_prefix_binding.proposal_generation
        == first_packet.proposal_generation,
        "streams": row_free_finalization.corrected_a_stream_identity_hash
        == registrations[0].corrected_a_stream_identity_hash
        == first_packet.raw_logit_stream_source_hash
        and row_free_finalization.literal_j_stream_identity_hash
        == registrations[0].literal_j_stream_identity_hash
        == fd_design.raw_logit_stream_source_hash,
        "row_free": row_free_finalization.portable_corrected_moments_present
        is False
        and row_free_finalization.portable_fd_snapshot_present is False
        and row_free_finalization.route_scale_d2h_bytes == 0
        and row_free_finalization.route_scale_host_tensor_materializations == 0
        and all(
            boundary.route_scale_d2h_bytes == 0
            and boundary.row_scratch_bytes == 0
            and boundary.row_head_present is False
            and boundary.row_manifest_present is False
            and boundary.row_shard_count == 0
            for boundary in boundaries
        ),
    }
    if not all(closure.values()):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.MATERIALIZATION_DRIFT,
            "exact row-free v2 closure failed="
            f"{tuple(name for name, closed in closure.items() if not closed)}",
        )
    chart_identity_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_dual_chart_identity_v2",
            "campaign_contract_hash": first_packet.campaign_contract_hash,
            "model_identity_hash": first_packet.model_identity_hash,
            "query_contract_hash": first_packet.query_contract_hash,
            "preprocessing_hash": first_packet.preprocessing_hash,
            "sampling_law_hash": first_packet.sampling_law_hash,
            "current_mask_hash": first_packet.current_mask_hash,
            "proposal_generation": first_packet.proposal_generation,
            "route_axis_hash": physical_route_axis.route_axis_hash,
            "route_id_axis_hash": physical_route_axis.route_id_axis_hash,
            "corrected_a_stream_identity_hash": (
                registrations[0].corrected_a_stream_identity_hash
            ),
            "literal_j_stream_identity_hash": (
                registrations[0].literal_j_stream_identity_hash
            ),
            "row_free_finalization_receipt_hash": (
                row_free_finalization.receipt_hash
            ),
        },
        prefix=_PREFIX + "dual-chart-identity-v2-",
    )
    bridge = V1SampledV7CompactDeviceResidentDRowFreeFinalSourceBridgeV2(
        row_free_finalization_receipt_hash=row_free_finalization.receipt_hash,
        materialization_rank_receipt_hash=(
            materialization_rank_receipt.receipt_hash
        ),
        session_design_hash=row_free_finalization.session_design_hash,
        hbm_forecast_hash=row_free_finalization.hbm_forecast_hash,
        registered_d_transaction_hash=registered_d_transaction.transaction_hash,
        registered_d_prefix_binding_hash=registered_d_prefix_binding.binding_hash,
        registered_request_prefix_hash=(
            registered_d_transaction.rank_invariant_raw_prefix_hash
        ),
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        fd_design_hash=fd_design.design_hash,
        corrected_reacquisition_source_hash=(
            row_free_finalization.corrected_reacquisition_source_hash
        ),
        live_fd_source_hash=row_free_finalization.live_fd_source_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=first_packet.current_mask_hash,
        proposal_generation=first_packet.proposal_generation,
        route_axis_hash=physical_route_axis.route_axis_hash,
        route_id_axis_hash=physical_route_axis.route_id_axis_hash,
        physical_route_axis_declaration_hash=physical_route_axis.content_hash,
        route_count=first_packet.route_count,
        corrected_a_stream_identity_hash=(
            registrations[0].corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=(
            registrations[0].literal_j_stream_identity_hash
        ),
        registered_look_count=len(requests),
        context_axis_hash=content_hash(context_axis, prefix=_CONTEXT_AXIS_PREFIX),
        draw_axis_hash=content_hash(draw_axis, prefix=_DRAW_AXIS_PREFIX),
        stratum_axis_hash=content_hash(stratum_axis, prefix=_STRATUM_AXIS_PREFIX),
        measurement_axis_hash=content_hash(
            measurement_axis,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        packet_axis_hash=content_hash(packet_axis, prefix=_PACKET_AXIS_PREFIX),
        population_count=sum(population_counts),
        population_context_axis_hash=content_hash(
            {
                "population_stratum_counts": population_counts,
                "sealed_population_order_hashes": population_orders,
                "axis_role": "sealed_population_context_owner",
            },
            prefix=_PREFIX + "population-context-axis-",
        ),
        population_draw_axis_hash=content_hash(
            {
                "population_stratum_counts": population_counts,
                "sealed_population_order_hashes": population_orders,
                "axis_role": "sealed_population_draw_owner",
            },
            prefix=_PREFIX + "population-draw-axis-",
        ),
        population_order_axis_hash=population_order_axis_hash,
        sampling_measure_hash=sampling_measure_hash,
        stratum_names=stratum_names,
        stratum_weights=stratum_weights,
        population_stratum_counts=population_counts,
        registered_stratum_counts=registered_stratum_counts,
        packet_registrations=registrations,
        packet_boundaries=boundaries,
        look_receipts=look_receipts,
        _issuance_token=_FINAL_BRIDGE_V2_TOKEN,
    )
    return issue_v1_sampled_v7_compact_row_free_final_source_from_exact_bridge_v2(
        exact_bridge=bridge,
    )


def issue_v1_sampled_v7_compact_device_resident_d_final_source(
    *,
    materialization: V1SampledDeviceResidentDSessionMaterialization,
    materialization_rank_receipt: V1SampledDeviceResidentDMaterializationRankReceipt,
    registered_d_transaction: V1SampledCompositionRegisteredDTransaction,
    registered_d_prefix_binding: V1SampledCompositionRegisteredDPrefixBinding,
    physical_route_axis: DeclaredRouteAxis,
    registered_requests: tuple[V1SampledRegisteredLookTransactionRequest, ...],
    look_commits: tuple[V1SampledV7CompactDeviceDLookCommit, ...],
) -> V1SampledV7CompactDeviceResidentDFinalSourceReceipt:
    """Close exact row-free D custody and issue the compact replay parent."""

    exact = (
        (materialization, V1SampledDeviceResidentDSessionMaterialization),
        (
            materialization_rank_receipt,
            V1SampledDeviceResidentDMaterializationRankReceipt,
        ),
        (registered_d_transaction, V1SampledCompositionRegisteredDTransaction),
        (
            registered_d_prefix_binding,
            V1SampledCompositionRegisteredDPrefixBinding,
        ),
        (physical_route_axis, DeclaredRouteAxis),
    )
    if (
        any(type(value) is not expected for value, expected in exact)
        or type(registered_requests) is not tuple
        or type(look_commits) is not tuple
        or not registered_requests
        or len(registered_requests) != len(look_commits)
        or any(
            type(value) is not V1SampledRegisteredLookTransactionRequest
            for value in registered_requests
        )
        or any(
            type(value) is not V1SampledV7CompactDeviceDLookCommit
            for value in look_commits
        )
    ):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.INVALID_TYPE,
            "device-final issuance requires every exact source-native owner",
        )
    for commit in look_commits:
        commit.assert_integrity()
    moments = materialization.corrected_attribution_moments
    moments.assert_integrity()
    snapshots = tuple(materialization.fd_snapshots_by_rank)
    if len(snapshots) != 1:
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.MATERIALIZATION_DRIFT,
            "compact device-final issuance requires one registered FD snapshot",
        )
    rank, snapshot = snapshots[0]
    snapshot.assert_integrity()
    registrations = tuple(
        packet
        for commit in look_commits
        for packet in commit.registration.packets
    )
    boundaries = tuple(
        packet
        for commit in look_commits
        for packet in commit.boundary.packet_boundaries
    )
    look_receipts = tuple(
        receipt
        for commit in look_commits
        for receipt in commit.resident_result.look_receipts
    )
    requests = tuple(registered_requests)
    first_request = requests[0]
    first_plan = first_request.packet_plans[0]
    first_packet = first_plan.reduced_request.packet_request
    fd_design = first_request.design
    request_hashes = tuple(request.request_hash for request in requests)
    commit_request_hashes = tuple(
        commit.registration.transaction_request_hash for commit in look_commits
    )
    expected_prior_result = None
    expected_prior_boundary = None
    for look_index, (request, commit) in enumerate(
        zip(requests, look_commits, strict=True),
        start=1,
    ):
        if (
            request.schema != SAMPLED_REGISTERED_LOOK_REQUEST_SCHEMA_V4
            or request.registered_look_index != look_index
            or commit.registration.logical_look_index != look_index
            or commit.boundary.logical_look_index != look_index
            or commit.boundary.prior_transaction_result_hash
            != expected_prior_result
            or commit.boundary.prior_boundary_hash != expected_prior_boundary
            or request.prior_device_resident_result_hash != expected_prior_result
        ):
            _refuse(
                V1SampledV7CompactDeviceDBridgeRefusalCode.PRIOR_CHAIN_DRIFT,
                "logical-look request/result/boundary chain is stale or reordered",
            )
        expected_prior_result = commit.resident_result.result_hash
        expected_prior_boundary = commit.boundary.boundary_hash
    final_result = look_commits[-1].resident_result
    grouped_measurements = tuple(
        (
            name,
            tuple(
                measurement
                for packet in registrations
                for measurement, label in zip(
                    packet.source_measurement_hashes,
                    packet.stratum_labels,
                    strict=True,
                )
                if label == name
            ),
        )
        for name in fd_design.stratum_names
    )
    context_axis = _flatten_packet_axis(registrations, "context_ids")
    draw_axis = _flatten_packet_axis(registrations, "draw_ids")
    stratum_axis = _flatten_packet_axis(registrations, "stratum_labels")
    measurement_axis = _flatten_packet_axis(
        registrations,
        "source_measurement_hashes",
    )
    population_counts = tuple(first_packet.population_stratum_counts)
    population_orders = tuple(first_packet.sealed_population_order_hashes)
    route_id_axis_hash = physical_route_axis.route_id_axis_hash
    chart_identity_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_compact_dual_chart_identity_v1",
            "campaign_contract_hash": moments.campaign_contract_hash,
            "model_identity_hash": moments.model_identity_hash,
            "query_contract_hash": moments.query_contract_hash,
            "preprocessing_hash": moments.preprocessing_hash,
            "sampling_law_hash": moments.sampling_law_hash,
            "current_mask_hash": moments.current_mask_hash,
            "proposal_generation": moments.proposal_generation,
            "route_axis_hash": physical_route_axis.route_axis_hash,
            "route_id_axis_hash": route_id_axis_hash,
            "corrected_a_stream_identity_hash": (
                registrations[0].corrected_a_stream_identity_hash
            ),
            "literal_j_stream_identity_hash": (
                registrations[0].literal_j_stream_identity_hash
            ),
        },
        prefix=_PREFIX + "dual-chart-identity-",
    )
    packet_axis = tuple(
        (
            registration.packet_plan_hash,
            registration.source_capture_identity_hash,
            boundary.provider_proof_hash,
            boundary.raw_dual_proof_hash,
            boundary.same_capture_vjp_source_hash,
        )
        for registration, boundary in zip(
            registrations,
            boundaries,
            strict=True,
        )
    )
    closure = {
        "request_order": request_hashes == commit_request_hashes,
        "look_count": (
            len(requests)
            == registered_d_transaction.registered_look_count
            == registered_d_prefix_binding.registered_look_count
            == materialization.final_look_count
            == moments.look_index
        ),
        "packet_global_axis": tuple(
            packet.source_packet_ordinal for packet in registrations
        )
        == tuple(range(len(registrations))),
        "packet_boundary_axis": tuple(
            packet.source_packet_ordinal for packet in boundaries
        )
        == tuple(range(len(boundaries))),
        "packet_source_uniqueness": (
            len({packet.provider_plan_identity_hash for packet in registrations})
            == len(registrations)
            and len(
                {packet.source_capture_identity_hash for packet in registrations}
            )
            == len(registrations)
            and len(set(context_axis)) == len(context_axis)
            and len(set(draw_axis)) == len(draw_axis)
            and len(set(measurement_axis)) == len(measurement_axis)
        ),
        "materialization_rank": (
            materialization.materialization_rank_receipt_hash
            == materialization_rank_receipt.receipt_hash
        ),
        "materialization_session": (
            materialization.session_design_hash
            == materialization_rank_receipt.session_design_hash
            == look_commits[0].boundary.session_design_hash
        ),
        "materialization_forecast": (
            materialization.hbm_forecast_hash
            == materialization_rank_receipt.hbm_forecast_hash
            == look_commits[0].boundary.hbm_forecast_hash
        ),
        "rank": (
            materialization.selected_physical_fd_rank
            == materialization_rank_receipt.registered_proposal_fd_rank
            == registered_d_transaction.physical_fd_rank
            == snapshot.sketch_rank
            == rank
        ),
        "fd_design": (
            registered_d_transaction.fd_design_hash
            == fd_design.design_hash
            == snapshot.design_hash
        ),
        "snapshot_packets": snapshot.packet_boundary_index == len(registrations),
        "snapshot_measurements": (
            snapshot.source_measurement_hashes_by_stratum == grouped_measurements
        ),
        "snapshot_route": (
            snapshot.route_axis_hash
            == materialization.route_axis_hash
            == fd_design.route_axis_hash
            == registrations[0].route_id_axis_hash
        ),
        "physical_route_axis": (
            physical_route_axis.route_count == moments.route_count
            and physical_route_axis.route_axis_hash
            == registrations[0].route_axis_hash
            == registered_d_prefix_binding.route_axis_hash
            and physical_route_axis.route_id_axis_hash
            == registrations[0].route_id_axis_hash
            == moments.route_axis_hash
            and physical_route_axis.content_hash
            == registrations[0].route_axis_declaration_hash
            and physical_route_axis.route_axis_hash
            != physical_route_axis.route_id_axis_hash
        ),
        "final_result_packets": (
            final_result.cumulative_packet_count_after == len(registrations)
            and final_result.cumulative_look_receipt_hashes
            == tuple(
                receipt.receipt_hash
                for commit in look_commits
                for receipt in commit.resident_result.look_receipts
            )
        ),
        "final_counts": (
            final_result.cumulative_stratum_counts_after
            == moments.stratum_counts
            == registered_d_prefix_binding.cumulative_stratum_counts
        ),
        "prefix_coordinate": (
            registered_d_transaction.rank_invariant_raw_prefix_hash
            == registered_d_prefix_binding.rank_invariant_raw_prefix_hash
        ),
        "prefix_literal_j": (
            registered_d_transaction.literal_j_source_receipt_hash
            == registered_d_prefix_binding.literal_j_source_receipt_hash
        ),
        "prefix_binding": (
            registered_d_transaction.source_replay_axis_binding_hash
            == registered_d_prefix_binding.binding_hash
        ),
        "prefix_axes": (
            registered_d_prefix_binding.D_context_axis_hash
            == content_hash(context_axis, prefix=_CONTEXT_AXIS_PREFIX)
            and registered_d_prefix_binding.D_draw_axis_hash
            == content_hash(draw_axis, prefix=_DRAW_AXIS_PREFIX)
            and registered_d_prefix_binding.D_source_measurement_axis_hash
            == content_hash(
                measurement_axis,
                prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
            )
        ),
        "prefix_chart": (
            registered_d_prefix_binding.current_mask_hash
            == moments.current_mask_hash
            and registered_d_prefix_binding.route_axis_hash
            == registrations[0].route_axis_hash
            and registered_d_prefix_binding.route_count == moments.route_count
            and registered_d_prefix_binding.proposal_generation
            == moments.proposal_generation
        ),
        "streams": (
            moments.raw_logit_stream_source_hash
            == registrations[0].corrected_a_stream_identity_hash
            and fd_design.raw_logit_stream_source_hash
            == registrations[0].literal_j_stream_identity_hash
        ),
        "context_count": len(context_axis) == sum(moments.stratum_counts),
        "zero_intermediate_rows": all(
            boundary.route_scale_d2h_bytes == 0
            and boundary.row_scratch_bytes == 0
            and boundary.row_head_present is False
            and boundary.row_manifest_present is False
            and boundary.row_shard_count == 0
            for boundary in boundaries
        ),
    }
    if not all(closure.values()):
        _refuse(
            V1SampledV7CompactDeviceDBridgeRefusalCode.MATERIALIZATION_DRIFT,
            "exact source-native device-D closure failed="
            f"{tuple(name for name, closed in closure.items() if not closed)}",
        )
    bridge = V1SampledV7CompactDeviceResidentDFinalSourceBridge(
        session_materialization_hash=materialization.materialization_hash,
        materialization_rank_receipt_hash=materialization_rank_receipt.receipt_hash,
        session_design_hash=materialization.session_design_hash,
        hbm_forecast_hash=materialization.hbm_forecast_hash,
        registered_d_transaction_hash=registered_d_transaction.transaction_hash,
        registered_d_prefix_binding_hash=registered_d_prefix_binding.binding_hash,
        registered_request_prefix_hash=(
            registered_d_transaction.rank_invariant_raw_prefix_hash
        ),
        literal_j_source_receipt_hash=(
            registered_d_transaction.literal_j_source_receipt_hash
        ),
        fd_design_hash=fd_design.design_hash,
        fd_snapshot_hash=snapshot.snapshot_hash,
        corrected_moments_state_hash=moments.state_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=moments.current_mask_hash,
        proposal_generation=moments.proposal_generation,
        route_axis_hash=registrations[0].route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        physical_route_axis_declaration_hash=physical_route_axis.content_hash,
        route_count=moments.route_count,
        corrected_a_stream_identity_hash=(
            registrations[0].corrected_a_stream_identity_hash
        ),
        literal_j_stream_identity_hash=(
            registrations[0].literal_j_stream_identity_hash
        ),
        registered_look_count=len(requests),
        context_axis_hash=content_hash(context_axis, prefix=_CONTEXT_AXIS_PREFIX),
        draw_axis_hash=content_hash(draw_axis, prefix=_DRAW_AXIS_PREFIX),
        stratum_axis_hash=content_hash(stratum_axis, prefix=_STRATUM_AXIS_PREFIX),
        measurement_axis_hash=content_hash(
            measurement_axis,
            prefix=_SOURCE_MEASUREMENT_AXIS_PREFIX,
        ),
        packet_axis_hash=content_hash(packet_axis, prefix=_PACKET_AXIS_PREFIX),
        population_count=sum(population_counts),
        population_context_axis_hash=content_hash(
            {
                "population_stratum_counts": population_counts,
                "sealed_population_order_hashes": population_orders,
                "axis_role": "sealed_population_context_owner",
            },
            prefix=_PREFIX + "population-context-axis-",
        ),
        population_draw_axis_hash=content_hash(
            {
                "population_stratum_counts": population_counts,
                "sealed_population_order_hashes": population_orders,
                "axis_role": "sealed_population_draw_owner",
            },
            prefix=_PREFIX + "population-draw-axis-",
        ),
        population_order_axis_hash=content_hash(
            population_orders,
            prefix=_PREFIX + "population-order-axis-",
        ),
        packet_registrations=registrations,
        packet_boundaries=boundaries,
        look_receipts=look_receipts,
        _issuance_token=_FINAL_BRIDGE_TOKEN,
    )
    return issue_v1_sampled_v7_compact_device_final_source_from_exact_bridge(
        exact_bridge=bridge,
    )


__all__ = [
    "COMPACT_DEVICE_D_AUTHORITY_SCOPE",
    "COMPACT_DEVICE_D_FINAL_BRIDGE_SCHEMA",
    "COMPACT_DEVICE_D_ROW_FREE_FINAL_BRIDGE_SCHEMA_V2",
    "COMPACT_DEVICE_D_LOOK_BOUNDARY_SCHEMA",
    "COMPACT_DEVICE_D_LOOK_COMMIT_SCHEMA",
    "COMPACT_DEVICE_D_LOOK_REGISTRATION_SCHEMA",
    "COMPACT_DEVICE_D_PACKET_BOUNDARY_SCHEMA",
    "COMPACT_DEVICE_D_PACKET_REGISTRATION_SCHEMA",
    "V1SampledV7CompactDeviceDBridgeRefusal",
    "V1SampledV7CompactDeviceDBridgeRefusalCode",
    "V1SampledV7CompactDeviceDLookBoundary",
    "V1SampledV7CompactDeviceDLookCommit",
    "V1SampledV7CompactDeviceDLookRegistration",
    "V1SampledV7CompactDeviceDPacketBoundary",
    "V1SampledV7CompactDeviceDPacketRegistration",
    "V1SampledV7CompactDeviceResidentDFinalSourceBridge",
    "V1SampledV7CompactDeviceResidentDRowFreeFinalSourceBridgeV2",
    "build_v1_sampled_v7_compact_registered_d_prefix_binding",
    "build_v1_sampled_v7_compact_registered_d_prefix_binding_row_free_v2",
    "execute_v1_sampled_v7_compact_device_d_look",
    "issue_v1_sampled_v7_compact_device_resident_d_final_source",
    "issue_v1_sampled_v7_compact_device_resident_d_row_free_final_source_v2",
    "register_v1_sampled_v7_compact_device_d_look",
]
