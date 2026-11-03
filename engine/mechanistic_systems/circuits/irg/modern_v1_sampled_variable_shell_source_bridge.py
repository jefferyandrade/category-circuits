"""Bounded same-VJP lowering from sampled dual-chart rows into ``D_W2``.

This module is the producer adapter between the raw-target-logit
dual-chart scorer and :class:`V1SignedWorkVariableShellSourceCUDAAccumulator`.
The scorer emits corrected Aumann--Shapley evidence and literal relaxed-gate
Jacobian chunks from one capture/VJP.  The adapter retains at most one
corrected chunk until its matching Jacobian chunk arrives, then immediately
lowers the pair into the compact native-fiber/shell representation.

No context-by-route row bank survives a chunk.  The adapter owns execution
custody only; it cannot select an action, open C/K/O, commit a mask, or make a
Pareto claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, NoReturn, Sequence

try:  # pragma: no cover - import refusal is exercised by runtime construction
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mechanistic_systems.measurement.optimized_irg_providers import (
    _RouteChunkedDualChartConsumer,
    _RouteChunkedLossOnlyConsumer,
)

from .contracts import content_hash
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAAccumulator,
    V1SignedWorkVariableShellSourceDeviceHandle,
)


SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_sampled_variable_shell_D_W2_same_vjp_bridge_v1"
)
SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_AUTHORITY: Final = (
    "execution_only_no_action_C_K_O_commit_terminal_or_pareto_authority"
)
_RAW_TARGET_LOGIT_COORDINATE: Final = "raw_target_logit"
_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-bridge-"
_EMPTY_SINK_AXIS_HASH: Final = content_hash(
    {
        "schema": "partition_free_v1_sampled_variable_shell_bound_sink_axis_v1",
        "sink_identity_hashes": (),
    },
    prefix=_PREFIX + "bound-sink-axis-",
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one exact nonnegative integer")
    return value


def _sink_axis_hash(values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_bound_sink_axis_v1",
            "sink_identity_hashes": tuple(values),
        },
        prefix=_PREFIX + "bound-sink-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellBoundChunkSinkBinding:
    """One execution-only fanout sink for already-bound same-VJP chunks.

    The variable-shell accumulator remains the source of the durable chunk
    binding.  A sink receives that exact issuer-typed binding only after the
    canonical compact source has consumed the row pair.  This permits a
    source-local replay to build an additional bounded numerical object (for
    example a block-support coefficient vector) without another VJP and
    without granting the sink any action or downstream authority.
    """

    sink_identity_hash: str
    sink: Any = field(repr=False, compare=False)
    bound_source_registration_hash: str | None = None
    execution_only: bool = True
    action_authority: bool = False
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sink_identity_hash",
            _identity(self.sink_identity_hash, name="bound sink identity hash"),
        )
        if self.bound_source_registration_hash is not None:
            object.__setattr__(
                self,
                "bound_source_registration_hash",
                _identity(
                    self.bound_source_registration_hash,
                    name="bound sink source-registration hash",
                ),
            )
        required = (
            "begin_bound_packet",
            "consume_bound_route_chunk",
            "end_bound_packet",
        )
        if any(not callable(getattr(self.sink, name, None)) for name in required):
            raise TypeError(
                "bound chunk sink must implement begin/consume/end execution hooks"
            )
        if self.execution_only is not True or any(
            value is not False
            for value in (
                self.action_authority,
                self.C_stream_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            )
        ):
            raise PermissionError("bound chunk sink crossed its execution boundary")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live bound chunk sink bindings are nonserializable")

    @property
    def singleton_sink_axis_hash(self) -> str:
        return _sink_axis_hash((self.sink_identity_hash,))


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellSourceBridgeReceipt:
    registration_hash: str
    compact_source_receipt_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    destination_transport_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    context_count: int
    packet_count: int
    paired_route_chunk_count: int
    bound_chunk_sink_axis_hash: str = _EMPTY_SINK_AXIS_HASH
    bound_chunk_sink_count: int = 0
    bound_chunk_sink_event_count: int = 0
    bound_chunk_fanout_complete: bool = True
    maximum_live_provider_chunk_pair_count: int = 1
    maximum_native_fiber_carry_routes_per_context: int = 0
    native_fiber_rechunking_complete: bool = True
    same_capture_and_vjp_verified: bool = True
    complete_registered_context_prefix_consumed: bool = True
    complete_route_axis_consumed_per_packet: bool = True
    context_by_route_rows_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    action_selection_authority: bool = False
    C_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_AUTHORITY
    schema: str = SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "compact_source_receipt_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "destination_transport_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "bound_chunk_sink_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "context_count",
            "packet_count",
            "paired_route_chunk_count",
            "bound_chunk_sink_count",
            "bound_chunk_sink_event_count",
            "maximum_native_fiber_carry_routes_per_context",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if self.context_count <= 0 or self.packet_count <= 0:
            raise ValueError(
                "a completed D_W2 bridge must contain contexts and packets"
            )
        if self.paired_route_chunk_count < self.packet_count:
            raise ValueError("every D_W2 packet must contain a paired route chunk")
        if self.maximum_live_provider_chunk_pair_count != 1:
            raise PermissionError("the D_W2 bridge may retain only one live chunk pair")
        required_true = (
            self.same_capture_and_vjp_verified,
            self.complete_registered_context_prefix_consumed,
            self.complete_route_axis_consumed_per_packet,
            self.native_fiber_rechunking_complete,
            self.bound_chunk_fanout_complete,
        )
        required_false = (
            self.context_by_route_rows_retained,
            self.route_scale_device_to_host_transfer,
            self.action_selection_authority,
            self.C_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in required_true + required_false)
            or not all(required_true)
            or any(required_false)
            or self.authority_scope != SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_AUTHORITY
            or self.schema != SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_SCHEMA
        ):
            raise PermissionError("D_W2 bridge receipt crossed its execution boundary")
        expected_sink_events = self.bound_chunk_sink_count * (
            2 * self.packet_count + self.paired_route_chunk_count
        )
        if self.bound_chunk_sink_event_count != expected_sink_events or (
            self.bound_chunk_sink_count == 0
            and self.bound_chunk_sink_axis_hash != _EMPTY_SINK_AXIS_HASH
        ):
            raise ValueError("D_W2 bound-chunk fanout telemetry does not close")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_receipt_hash"
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("D_W2 bridge receipt was mutated")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("D_W2 bridge receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledVariableShellSourceBridgeResult:
    source_handle: V1SignedWorkVariableShellSourceDeviceHandle = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledVariableShellSourceBridgeReceipt

    def __post_init__(self) -> None:
        if type(self.source_handle) is not V1SignedWorkVariableShellSourceDeviceHandle:
            raise TypeError(
                "D_W2 bridge result requires the exact compact source handle"
            )
        if type(self.receipt) is not V1SampledVariableShellSourceBridgeReceipt:
            raise TypeError("D_W2 bridge result requires the exact execution receipt")
        if (
            self.source_handle.receipt.receipt_hash
            != self.receipt.compact_source_receipt_hash
        ):
            raise ValueError("D_W2 bridge result changed its compact source")
        self.receipt.assert_integrity()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D_W2 bridge results are nonserializable")


class _PairedSourceState:
    """One synchronous, bounded corrected/J packet join."""

    __slots__ = (
        "accumulator",
        "route_ids",
        "route_axis_hash",
        "route_id_axis_hash",
        "destination_transport_hash",
        "context_cursor",
        "packet_count",
        "paired_route_chunk_count",
        "packet_context_ids",
        "opened_roles",
        "closed_roles",
        "provider_next_route",
        "lowered_next_route",
        "pending_corrected",
        "pending_span",
        "carry_corrected",
        "carry_literal",
        "carry_start",
        "maximum_carry_routes",
        "bound_chunk_sinks",
        "bound_chunk_sink_axis_hash",
        "bound_chunk_sink_event_count",
        "sealed",
    )

    def __init__(
        self,
        *,
        accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
        route_ids: Sequence[str],
        route_axis_hash: str,
        route_id_axis_hash: str,
        destination_transport_hash: str,
        bound_chunk_sinks: Sequence[V1SampledVariableShellBoundChunkSinkBinding] = (),
    ) -> None:
        if type(accumulator) is not V1SignedWorkVariableShellSourceCUDAAccumulator:
            raise TypeError("D_W2 bridge requires the exact variable-shell accumulator")
        registration = accumulator.registration
        self.accumulator = accumulator
        self.route_ids = route_ids
        self.route_axis_hash = _identity(route_axis_hash, name="route axis hash")
        self.route_id_axis_hash = _identity(
            route_id_axis_hash,
            name="route-ID axis hash",
        )
        self.destination_transport_hash = _identity(
            destination_transport_hash,
            name="destination transport hash",
        )
        if (
            len(route_ids) != accumulator.native_fiber_axis_receipt.route_count
            or registration.route_axis_hash != self.route_axis_hash
            or registration.route_id_axis_hash != self.route_id_axis_hash
            or accumulator.native_fiber_axis_receipt.route_axis_hash
            != self.route_axis_hash
            or accumulator.native_fiber_axis_receipt.route_id_axis_hash
            != self.route_id_axis_hash
        ):
            raise ValueError("D_W2 bridge route identity differs from its source")
        self.context_cursor = 0
        self.packet_count = 0
        self.paired_route_chunk_count = 0
        self.packet_context_ids: tuple[str, ...] = ()
        self.opened_roles: set[str] = set()
        self.closed_roles: set[str] = set()
        self.provider_next_route = 0
        self.lowered_next_route = 0
        self.pending_corrected: Any | None = None
        self.pending_span: tuple[int, int] | None = None
        self.carry_corrected: Any | None = None
        self.carry_literal: Any | None = None
        self.carry_start: int | None = None
        self.maximum_carry_routes = 0
        if type(bound_chunk_sinks) not in {tuple, list} or any(
            type(item) is not V1SampledVariableShellBoundChunkSinkBinding
            for item in bound_chunk_sinks
        ):
            raise TypeError("bound chunk sinks must be one exact binding sequence")
        sinks = tuple(bound_chunk_sinks)
        identities = tuple(item.sink_identity_hash for item in sinks)
        if len(identities) != len(set(identities)):
            raise ValueError("bound chunk sink identities must be unique")
        if any(
            item.bound_source_registration_hash is not None
            and item.bound_source_registration_hash
            != registration.registration_hash
            for item in sinks
        ):
            raise PermissionError(
                "bound chunk sink was preregistered for another physical source"
            )
        self.bound_chunk_sinks = sinks
        self.bound_chunk_sink_axis_hash = _sink_axis_hash(identities)
        self.bound_chunk_sink_event_count = 0
        self.sealed = False

    def begin(
        self,
        *,
        role: str,
        context_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        stream_identity_hash: str | None,
    ) -> None:
        if self.sealed or role in self.opened_roles or self.closed_roles:
            raise RuntimeError("D_W2 bridge packet role was reopened or already sealed")
        if route_ids is not self.route_ids:
            raise PermissionError(
                "D_W2 production bridge requires the registered lazy route-axis owner"
            )
        if destination_transport_hash != self.destination_transport_hash:
            raise PermissionError("D_W2 bridge received another destination transport")
        registration = self.accumulator.registration
        expected_stream = (
            registration.corrected_evidence_stream_source_hash
            if role == "corrected"
            else registration.literal_j_stream_source_hash
        )
        if stream_identity_hash != expected_stream:
            raise PermissionError("D_W2 bridge received another dual-chart stream")
        contexts = tuple(context_ids)
        if not contexts:
            raise ValueError("D_W2 bridge packet cannot be empty")
        expected = registration.registered_context_ids[
            self.context_cursor : self.context_cursor + len(contexts)
        ]
        if contexts != expected:
            raise PermissionError(
                "D_W2 packet is not the next registered context prefix"
            )
        if self.packet_context_ids and contexts != self.packet_context_ids:
            raise PermissionError("corrected and literal-J packet axes differ")
        self.packet_context_ids = contexts
        self.opened_roles.add(role)
        if self.opened_roles == {"corrected", "literal_j"}:
            stop = self.context_cursor + len(contexts)
            self.accumulator.begin_packet(
                context_ids=registration.registered_context_ids[
                    self.context_cursor : stop
                ],
                draw_ids=registration.registered_draw_ids[self.context_cursor : stop],
                measurement_ids=registration.registered_measurement_ids[
                    self.context_cursor : stop
                ],
                capture_ids=registration.registered_capture_ids[
                    self.context_cursor : stop
                ],
                source_packet_hashes=registration.registered_source_packet_hashes[
                    self.context_cursor : stop
                ],
            )
            for binding in self.bound_chunk_sinks:
                binding.sink.begin_bound_packet(
                    source_registration=registration,
                    context_start_ordinal=self.context_cursor,
                    context_ids=registration.registered_context_ids[
                        self.context_cursor : stop
                    ],
                    draw_ids=registration.registered_draw_ids[
                        self.context_cursor : stop
                    ],
                    measurement_ids=registration.registered_measurement_ids[
                        self.context_cursor : stop
                    ],
                    capture_ids=registration.registered_capture_ids[
                        self.context_cursor : stop
                    ],
                    source_packet_hashes=registration.registered_source_packet_hashes[
                        self.context_cursor : stop
                    ],
                )
                self.bound_chunk_sink_event_count += 1

    def _lower_aligned_pair(
        self,
        *,
        start: int,
        corrected: Any,
        literal: Any,
    ) -> None:
        """Lower one fiber-aligned pair into the compact accumulator."""

        width = int(corrected.shape[1])
        if start != self.lowered_next_route or width <= 0:
            raise RuntimeError("D_W2 native-fiber rechunking lost its route cursor")
        registration = self.accumulator.registration
        corrected_chunk = corrected.contiguous()
        literal_chunk = literal.contiguous()
        binding = self.accumulator.issue_route_chunk_binding(
            route_start=start,
            corrected_evidence_rows=corrected_chunk,
            literal_j_rows=literal_chunk,
            route_axis_hash=self.route_axis_hash,
            route_id_axis_hash=self.route_id_axis_hash,
            same_capture_vjp_source_hash=registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                registration.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
        )
        self.accumulator.consume_route_chunk(
            route_start=start,
            corrected_evidence_rows=corrected_chunk,
            literal_j_rows=literal_chunk,
            route_axis_hash=self.route_axis_hash,
            route_id_axis_hash=self.route_id_axis_hash,
            same_capture_vjp_source_hash=registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                registration.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
            chunk_binding=binding,
        )
        tensor_versions = (corrected_chunk._version, literal_chunk._version)
        for sink_binding in self.bound_chunk_sinks:
            sink_binding.sink.consume_bound_route_chunk(
                source_registration=registration,
                route_start=start,
                corrected_evidence_rows=corrected_chunk,
                literal_j_rows=literal_chunk,
                chunk_binding=binding,
            )
            self.bound_chunk_sink_event_count += 1
            if tensor_versions != (corrected_chunk._version, literal_chunk._version):
                raise PermissionError("bound chunk sink mutated source tensor custody")
        self.lowered_next_route += width
        self.paired_route_chunk_count += 1

    def _consume_paired_rows(
        self,
        *,
        start: int,
        stop: int,
        corrected: Any,
        literal: Any,
    ) -> None:
        """Rechunk one provider pair onto exact native-fiber boundaries."""

        if torch is None:
            raise RuntimeError("D_W2 bridge requires PyTorch")
        contexts = len(self.packet_context_ids)
        width = stop - start
        device = self.accumulator.device
        for name, value in (("corrected", corrected), ("literal-J", literal)):
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float64
                or value.device != device
                or tuple(value.shape) != (contexts, width)
                or not value.is_contiguous()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise TypeError(
                    f"D_W2 {name} provider chunk must be contiguous finite float64"
                )

        cursor = start
        source_offset = 0
        if self.carry_corrected is not None:
            if (
                self.carry_literal is None
                or self.carry_start != self.lowered_next_route
                or cursor != self.provider_next_route
            ):
                raise RuntimeError("D_W2 native-fiber carry is stale")
            carry_start = self.carry_start
            cut = self.accumulator.native_fiber_axis_receipt.cut_for_route(carry_start)
            carried = int(self.carry_corrected.shape[1])
            need = cut.routes_per_fiber - carried
            if need <= 0:
                raise RuntimeError("D_W2 native-fiber carry is not a strict tail")
            take = min(need, width)
            merged_corrected = torch.cat(
                (self.carry_corrected, corrected[:, :take]),
                dim=1,
            )
            merged_literal = torch.cat(
                (self.carry_literal, literal[:, :take]),
                dim=1,
            )
            cursor += take
            source_offset += take
            if int(merged_corrected.shape[1]) == cut.routes_per_fiber:
                self.carry_corrected = None
                self.carry_literal = None
                self.carry_start = None
                self._lower_aligned_pair(
                    start=carry_start,
                    corrected=merged_corrected,
                    literal=merged_literal,
                )
            else:
                self.carry_corrected = merged_corrected.contiguous()
                self.carry_literal = merged_literal.contiguous()
                self.maximum_carry_routes = max(
                    self.maximum_carry_routes,
                    int(merged_corrected.shape[1]),
                )
                return

        while cursor < stop:
            cut = self.accumulator.native_fiber_axis_receipt.cut_for_route(cursor)
            segment_stop = min(stop, cut.route_stop)
            segment_width = segment_stop - cursor
            aligned_width = (
                segment_width // cut.routes_per_fiber
            ) * cut.routes_per_fiber
            if aligned_width:
                local_stop = source_offset + aligned_width
                self._lower_aligned_pair(
                    start=cursor,
                    corrected=corrected[:, source_offset:local_stop],
                    literal=literal[:, source_offset:local_stop],
                )
                cursor += aligned_width
                source_offset = local_stop
            tail_width = segment_stop - cursor
            if tail_width:
                if segment_stop == cut.route_stop:
                    raise RuntimeError(
                        "native-fiber cut ended with an incomplete fiber"
                    )
                local_stop = source_offset + tail_width
                self.carry_start = cursor
                self.carry_corrected = corrected[
                    :, source_offset:local_stop
                ].contiguous()
                self.carry_literal = literal[:, source_offset:local_stop].contiguous()
                self.maximum_carry_routes = max(
                    self.maximum_carry_routes,
                    tail_width,
                )
                cursor = segment_stop
                source_offset = local_stop
        if source_offset != width:
            raise RuntimeError("D_W2 provider pair was not consumed exactly once")

    def consume(self, *, role: str, start: int, stop: int, rows: Any) -> None:
        if self.opened_roles != {"corrected", "literal_j"} or self.closed_roles:
            raise RuntimeError("D_W2 chunks require both live packet roles")
        if role == "corrected":
            if self.pending_corrected is not None or start != self.provider_next_route:
                raise RuntimeError(
                    "corrected D_W2 chunks are duplicated or out of order"
                )
            self.pending_corrected = rows
            self.pending_span = (start, stop)
            return
        if role != "literal_j":
            raise ValueError("unknown D_W2 dual-chart role")
        if (
            self.pending_corrected is None
            or self.pending_span != (start, stop)
            or start != self.provider_next_route
        ):
            raise RuntimeError(
                "literal-J D_W2 chunk lacks its matching corrected chunk"
            )
        self._consume_paired_rows(
            start=start,
            stop=stop,
            corrected=self.pending_corrected,
            literal=rows,
        )
        self.pending_corrected = None
        self.pending_span = None
        self.provider_next_route = stop

    def end(self, *, role: str) -> None:
        if role not in self.opened_roles or role in self.closed_roles:
            raise RuntimeError("D_W2 packet role was not open or was closed twice")
        if self.pending_corrected is not None:
            raise RuntimeError("D_W2 packet ended with an unmatched corrected chunk")
        route_count = self.accumulator.native_fiber_axis_receipt.route_count
        if (
            self.provider_next_route != route_count
            or self.lowered_next_route != route_count
            or self.carry_corrected is not None
            or self.carry_literal is not None
            or self.carry_start is not None
        ):
            raise RuntimeError("D_W2 packet did not cover the full route axis")
        self.closed_roles.add(role)
        if self.closed_roles == {"corrected", "literal_j"}:
            self.accumulator.end_packet()
            registration = self.accumulator.registration
            for binding in self.bound_chunk_sinks:
                binding.sink.end_bound_packet(source_registration=registration)
                self.bound_chunk_sink_event_count += 1
            self.context_cursor += len(self.packet_context_ids)
            self.packet_count += 1
            self.packet_context_ids = ()
            self.opened_roles.clear()
            self.closed_roles.clear()
            self.provider_next_route = 0
            self.lowered_next_route = 0


class _VariableShellRoleConsumer(_RouteChunkedLossOnlyConsumer):
    positive_removal_stream_coordinate = _RAW_TARGET_LOGIT_COORDINATE

    def __init__(self, *, state: _PairedSourceState, role: str) -> None:
        if role not in {"corrected", "literal_j"}:
            raise ValueError("unknown D_W2 bridge role")
        self._state = state
        self._role = role
        self.route_ids = state.route_ids
        self.device = state.accumulator.device
        registration = state.accumulator.registration
        self.loss_pullback_hash = (
            registration.corrected_evidence_stream_source_hash
            if role == "corrected"
            else registration.literal_j_stream_source_hash
        )
        self.chunk_size = 0

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        super().bind_route_chunk_size(chunk_size)
        if self.chunk_size > self._state.accumulator.design.route_chunk_size:
            raise ValueError("provider route chunk exceeds the registered D_W2 design")

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        self._state.begin(
            role=self._role,
            context_ids=context_ids,
            route_ids=route_ids,
            destination_transport_hash=destination_transport_hash,
            stream_identity_hash=loss_pullback_hash,
        )

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        self._state.consume(role=self._role, start=start, stop=stop, rows=rows)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        raise PermissionError("D_W2 bridge accepts only explicit positive-removal rows")

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        raise PermissionError("D_W2 bridge refuses a collapsed/fused scientific stream")

    def end_packet(self) -> None:
        self._state.end(role=self._role)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        return {
            "route_chunked_streaming": True,
            "route_chunk_size": int(self.chunk_size),
            "loss_only_route_streaming": True,
            "sampled_variable_shell_D_W2_lowering": True,
            "dual_chart_role": self._role,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "positive_removal_stream_identity_hash": self.loss_pullback_hash,
            "packet_count": self._state.packet_count,
            "paired_route_chunk_count": self._state.paired_route_chunk_count,
            "maximum_live_provider_chunk_pair_count": 1,
            "maximum_native_fiber_carry_routes_per_context": (
                self._state.maximum_carry_routes
            ),
            "native_fiber_rechunking_complete": True,
            "full_tripartite_rows_materialized": False,
            "host_context_route_rows_transferred": False,
            "host_context_route_row_transfer_count": 0,
            "host_context_route_row_transfer_bytes": 0,
            "host_context_route_transfer_role": "none",
            "context_by_route_rows_retained": False,
        }


class V1SampledVariableShellSourceBridge:
    """Live source owner for one registered, predictable ``D_W2`` prefix."""

    __slots__ = ("_state", "_consumer", "_result")

    def __init__(
        self,
        *,
        accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
        route_ids: Sequence[str],
        route_axis_hash: str,
        route_id_axis_hash: str,
        destination_transport_hash: str,
        bound_chunk_sinks: Sequence[V1SampledVariableShellBoundChunkSinkBinding] = (),
    ) -> None:
        state = _PairedSourceState(
            accumulator=accumulator,
            route_ids=route_ids,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            destination_transport_hash=destination_transport_hash,
            bound_chunk_sinks=bound_chunk_sinks,
        )
        corrected = _VariableShellRoleConsumer(state=state, role="corrected")
        literal = _VariableShellRoleConsumer(state=state, role="literal_j")
        self._state = state
        self._consumer = _RouteChunkedDualChartConsumer(
            evidence_consumer=corrected,
            jacobian_consumer=literal,
        )
        self._result: V1SampledVariableShellSourceBridgeResult | None = None

    @property
    def producer_consumer(self) -> _RouteChunkedDualChartConsumer:
        if self._result is not None:
            raise RuntimeError("sealed D_W2 bridge has no producer authority")
        return self._consumer

    def seal(self) -> V1SampledVariableShellSourceBridgeResult:
        if self._result is not None or self._state.sealed:
            raise RuntimeError("D_W2 bridge was already sealed")
        registration = self._state.accumulator.registration
        if (
            self._state.opened_roles
            or self._state.closed_roles
            or self._state.pending_corrected is not None
            or self._state.context_cursor != registration.context_count
        ):
            raise RuntimeError(
                "D_W2 bridge cannot seal an incomplete registered prefix"
            )
        handle = self._state.accumulator.seal()
        receipt = V1SampledVariableShellSourceBridgeReceipt(
            registration_hash=registration.registration_hash,
            compact_source_receipt_hash=handle.receipt.receipt_hash,
            route_axis_hash=self._state.route_axis_hash,
            route_id_axis_hash=self._state.route_id_axis_hash,
            destination_transport_hash=self._state.destination_transport_hash,
            same_capture_vjp_source_hash=registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                registration.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
            context_count=self._state.context_cursor,
            packet_count=self._state.packet_count,
            paired_route_chunk_count=self._state.paired_route_chunk_count,
            bound_chunk_sink_axis_hash=self._state.bound_chunk_sink_axis_hash,
            bound_chunk_sink_count=len(self._state.bound_chunk_sinks),
            bound_chunk_sink_event_count=(self._state.bound_chunk_sink_event_count),
            maximum_native_fiber_carry_routes_per_context=(
                self._state.maximum_carry_routes
            ),
        )
        self._state.sealed = True
        self._result = V1SampledVariableShellSourceBridgeResult(
            source_handle=handle,
            receipt=receipt,
        )
        return self._result

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D_W2 bridges are nonserializable")


__all__ = [
    "SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_AUTHORITY",
    "SAMPLED_VARIABLE_SHELL_SOURCE_BRIDGE_SCHEMA",
    "V1SampledVariableShellBoundChunkSinkBinding",
    "V1SampledVariableShellSourceBridge",
    "V1SampledVariableShellSourceBridgeReceipt",
    "V1SampledVariableShellSourceBridgeResult",
]
