"""Source-bound replay fanout into the multi-segment block-support CUDA source.

The sampled D-side chart is physically owned by several immutable row shards:
conditioned D1 first, followed by every residual-fit packet that could have
influenced the frozen anchor.  Replaying those shards through the canonical
variable-shell bridge reissues exact same-source chunk bindings without a new
VJP.  This module fans each already-bound corrected-A chunk into one logical
``D_fixed`` block-support accumulator while the ordinary compact replay source
is consumed and immediately evicted.

Only one route window and one route-scale coefficient vector are resident.
No context-by-route row bank survives the replay and this execution receipt
cannot select an action or open C/K/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
    V1SampledVariableShellSourceBridgeResult,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator,
    V1SignedWorkBlockSupportSourceDeviceHandle,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceChunkBinding,
    V1SignedWorkVariableShellSourceRegistration,
)


SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_SCHEMA: Final = (
    "partition_free_v1_sampled_block_support_replay_fanout_v1"
)
SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_AUTHORITY: Final = (
    "D_fixed_numerical_source_execution_only_no_action_C_K_O_commit_terminal_"
    "or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-sampled-block-support-replay-fanout-"
_RECEIPT_ISSUER: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be one exact nonnegative integer")
    return value


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_block_support_replay_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + role + "-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportReplayFanoutReceipt:
    multi_segment_registration_hash: str
    structural_source_registration_hash: str
    multi_segment_registration_axis_hash: str
    replay_segment_registration_axis_hash: str
    numerical_source_chain_hash: str
    segment_source_registration_hashes: tuple[str, ...]
    segment_sink_identity_hashes: tuple[str, ...]
    segment_bridge_receipt_hashes: tuple[str, ...]
    segment_count: int
    context_count: int
    packet_count: int
    paired_route_chunk_count: int
    fanout_event_count: int
    block_support_source_receipt_hash: str
    _issuer: object = field(default=None, repr=False, compare=False)
    all_influencing_segments_consumed_once_in_order: bool = True
    source_local_chunk_bindings_verified: bool = True
    replay_compact_sources_evicted: bool = True
    one_route_window_live_per_source: bool = True
    route_scale_block_support_vector_retained: bool = True
    context_by_route_rows_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    action_authority: bool = False
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_AUTHORITY
    schema: str = SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _RECEIPT_ISSUER:
            raise TypeError("block-support replay receipts are issuer-only")
        for name in (
            "multi_segment_registration_hash",
            "structural_source_registration_hash",
            "multi_segment_registration_axis_hash",
            "replay_segment_registration_axis_hash",
            "numerical_source_chain_hash",
            "block_support_source_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        axes = tuple(
            tuple(_identity(value, name=name) for value in getattr(self, name))
            for name in (
                "segment_source_registration_hashes",
                "segment_sink_identity_hashes",
                "segment_bridge_receipt_hashes",
            )
        )
        for name, axis in zip(
            (
                "segment_source_registration_hashes",
                "segment_sink_identity_hashes",
                "segment_bridge_receipt_hashes",
            ),
            axes,
            strict=True,
        ):
            object.__setattr__(self, name, axis)
        for name in (
            "segment_count",
            "context_count",
            "packet_count",
            "paired_route_chunk_count",
            "fanout_event_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.segment_count <= 0
            or self.context_count <= 0
            or self.packet_count < self.segment_count
            or self.paired_route_chunk_count < self.packet_count
            or any(len(axis) != self.segment_count for axis in axes)
            or any(len(set(axis)) != len(axis) for axis in axes)
            or self.fanout_event_count
            != 2 * self.packet_count + self.paired_route_chunk_count
            or self.replay_segment_registration_axis_hash
            != _axis_hash(
                "segment_registration",
                self.segment_source_registration_hashes,
            )
        ):
            raise ValueError("block-support replay segment/event axes do not close")
        required_true = (
            self.all_influencing_segments_consumed_once_in_order,
            self.source_local_chunk_bindings_verified,
            self.replay_compact_sources_evicted,
            self.one_route_window_live_per_source,
            self.route_scale_block_support_vector_retained,
        )
        required_false = (
            self.context_by_route_rows_retained,
            self.route_scale_device_to_host_transfer,
            self.action_authority,
            self.C_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(value is not True for value in required_true)
            or any(value is not False for value in required_false)
            or self.authority_scope != SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_AUTHORITY
            or self.schema != SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_SCHEMA
        ):
            raise PermissionError("block-support replay receipt crossed its boundary")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"_issuer", "_receipt_hash"}
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("block-support replay receipt was mutated")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("block-support replay receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportReplayFanoutResult:
    source_handle: V1SignedWorkBlockSupportSourceDeviceHandle = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledBlockSupportReplayFanoutReceipt

    def __post_init__(self) -> None:
        if type(self.source_handle) is not V1SignedWorkBlockSupportSourceDeviceHandle:
            raise TypeError("fanout result requires the exact block-support handle")
        if type(self.receipt) is not V1SampledBlockSupportReplayFanoutReceipt:
            raise TypeError("fanout result requires the exact replay receipt")
        self.receipt.assert_integrity()
        if self.source_handle.receipt.receipt_hash != (
            self.receipt.block_support_source_receipt_hash
        ):
            raise ValueError("fanout result changed its block-support source")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live block-support replay results are nonserializable")


class _SegmentSink:
    __slots__ = (
        "coordinator",
        "segment_ordinal",
        "source_registration",
        "sink_identity_hash",
        "context_cursor",
        "active_count",
        "packet_count",
        "chunk_count",
        "event_count",
        "completed",
    )

    def __init__(
        self,
        *,
        coordinator: "V1SampledBlockSupportReplayFanoutCoordinator",
        segment_ordinal: int,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        sink_identity_hash: str,
    ) -> None:
        self.coordinator = coordinator
        self.segment_ordinal = segment_ordinal
        self.source_registration = source_registration
        self.sink_identity_hash = sink_identity_hash
        self.context_cursor = 0
        self.active_count = 0
        self.packet_count = 0
        self.chunk_count = 0
        self.event_count = 0
        self.completed = False

    def _validate_source(
        self,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
    ) -> None:
        self.coordinator._require_unaborted()
        if (
            type(source_registration) is not V1SignedWorkVariableShellSourceRegistration
            or source_registration.registration_hash
            != self.source_registration.registration_hash
        ):
            raise PermissionError("replay sink received another physical source")

    def abort_unsealed(self) -> str:
        """Abort the owning unpublished replay family exactly once.

        The shared row fanout holds only this source-bound sink, not the
        coordinator itself.  Exposing the coordinator's typed abort through
        the sink closes failure-atomic cleanup without granting any replay or
        downstream decision authority.
        """

        return self.coordinator.abort_unsealed()

    def begin_bound_packet(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        context_start_ordinal: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
    ) -> None:
        self._validate_source(source_registration)
        if (
            self.completed
            or self.active_count
            or context_start_ordinal != self.context_cursor
        ):
            raise RuntimeError(
                "replay sink packet is stale, duplicated, or out of order"
            )
        axes = tuple(
            tuple(axis)
            for axis in (
                context_ids,
                draw_ids,
                measurement_ids,
                capture_ids,
                source_packet_hashes,
            )
        )
        count = len(axes[0])
        if len({len(axis) for axis in axes}) != 1 or count <= 0:
            raise ValueError("replay sink packet axes do not align")
        self.coordinator._accumulator.begin_packet(
            segment_ordinal=self.segment_ordinal,
            context_ids=axes[0],
            draw_ids=axes[1],
            measurement_ids=axes[2],
            capture_ids=axes[3],
            source_packet_hashes=axes[4],
        )
        self.active_count = count
        self.event_count += 1

    def consume_bound_route_chunk(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        chunk_binding: V1SignedWorkVariableShellSourceChunkBinding,
    ) -> None:
        del literal_j_rows
        self._validate_source(source_registration)
        if not self.active_count:
            raise RuntimeError("replay sink route chunk lacks an active packet")
        self.coordinator._accumulator.consume_route_chunk(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            chunk_binding=chunk_binding,
        )
        self.chunk_count += 1
        self.event_count += 1

    def end_bound_packet(
        self,
        *,
        source_registration: V1SignedWorkVariableShellSourceRegistration,
    ) -> None:
        self._validate_source(source_registration)
        if not self.active_count:
            raise RuntimeError("replay sink packet was closed without opening")
        completed = self.active_count
        self.coordinator._accumulator.end_packet()
        self.context_cursor += completed
        self.active_count = 0
        self.packet_count += 1
        self.event_count += 1
        if self.context_cursor == self.source_registration.context_count:
            self.completed = True
        elif self.context_cursor > self.source_registration.context_count:
            raise RuntimeError("replay sink exceeded its physical source axis")


class V1SampledBlockSupportReplayFanoutCoordinator:
    """Join ordered replay bridges into one logical block-support source."""

    __slots__ = (
        "_accumulator",
        "_segments",
        "_next_segment",
        "_active_sink",
        "_sink_identities",
        "_bridge_receipt_hashes",
        "_packet_count",
        "_chunk_count",
        "_event_count",
        "_abort_hash",
        "_aborted",
        "_sealed",
    )

    def __init__(
        self,
        *,
        accumulator: V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator,
        segment_source_registrations: Sequence[
            V1SignedWorkVariableShellSourceRegistration
        ],
    ) -> None:
        if (
            type(accumulator)
            is not V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator
        ):
            raise TypeError(
                "replay fanout requires the exact multi-segment accumulator"
            )
        if type(segment_source_registrations) not in {tuple, list} or any(
            type(item) is not V1SignedWorkVariableShellSourceRegistration
            for item in segment_source_registrations
        ):
            raise TypeError("replay fanout segment registrations have wrong type")
        segments = tuple(segment_source_registrations)
        if tuple(item.registration_hash for item in segments) != tuple(
            item.registration_hash for item in accumulator.segment_source_registrations
        ):
            raise ValueError("replay fanout segment registration axis changed")
        self._accumulator = accumulator
        self._segments = segments
        self._next_segment = 0
        self._active_sink: _SegmentSink | None = None
        self._sink_identities: list[str] = []
        self._bridge_receipt_hashes: list[str] = []
        self._packet_count = 0
        self._chunk_count = 0
        self._event_count = 0
        self._abort_hash: str | None = None
        self._aborted = False
        self._sealed = False

    @property
    def is_aborted(self) -> bool:
        return bool(self._aborted)

    def _require_unaborted(self) -> None:
        if self._aborted:
            raise RuntimeError("block-support replay fanout was aborted")

    def issue_next_segment_sink(
        self,
    ) -> V1SampledVariableShellBoundChunkSinkBinding:
        self._require_unaborted()
        if self._sealed or self._active_sink is not None:
            raise RuntimeError("replay fanout cannot issue another live segment sink")
        if self._next_segment >= len(self._segments):
            raise RuntimeError("every replay fanout segment was already issued")
        ordinal = self._next_segment
        source = self._segments[ordinal]
        identity = content_hash(
            {
                "schema": "partition_free_v1_sampled_block_support_segment_sink_v1",
                "multi_segment_registration_hash": (
                    self._accumulator.multi_segment_registration.registration_hash
                ),
                "segment_ordinal": ordinal,
                "source_registration_hash": source.registration_hash,
            },
            prefix=_PREFIX + "segment-sink-",
        )
        sink = _SegmentSink(
            coordinator=self,
            segment_ordinal=ordinal,
            source_registration=source,
            sink_identity_hash=identity,
        )
        self._active_sink = sink
        return V1SampledVariableShellBoundChunkSinkBinding(
            sink_identity_hash=identity,
            sink=sink,
            bound_source_registration_hash=source.registration_hash,
        )

    def accept_segment_bridge(
        self,
        result: V1SampledVariableShellSourceBridgeResult,
    ) -> None:
        self._require_unaborted()
        if type(result) is not V1SampledVariableShellSourceBridgeResult:
            raise TypeError("replay fanout accepts the exact bridge result only")
        sink = self._active_sink
        if self._sealed or sink is None or not sink.completed or sink.active_count:
            raise RuntimeError("replay fanout segment is incomplete")
        receipt = result.receipt
        receipt.assert_integrity()
        source = self._segments[self._next_segment]
        binding_axis = V1SampledVariableShellBoundChunkSinkBinding(
            sink_identity_hash=sink.sink_identity_hash,
            sink=sink,
            bound_source_registration_hash=source.registration_hash,
        ).singleton_sink_axis_hash
        if (
            receipt.registration_hash != source.registration_hash
            or receipt.context_count != source.context_count
            or receipt.bound_chunk_sink_count != 1
            or receipt.bound_chunk_sink_axis_hash != binding_axis
        ):
            raise PermissionError("replay bridge differs from its issued segment sink")
        if (
            receipt.packet_count != sink.packet_count
            or receipt.paired_route_chunk_count != sink.chunk_count
            or receipt.bound_chunk_sink_event_count != sink.event_count
        ):
            raise ValueError("replay bridge and segment sink event counts differ")
        result.source_handle.evict()
        self._sink_identities.append(sink.sink_identity_hash)
        self._bridge_receipt_hashes.append(receipt.receipt_hash)
        self._packet_count += sink.packet_count
        self._chunk_count += sink.chunk_count
        self._event_count += sink.event_count
        self._next_segment += 1
        self._active_sink = None

    def seal(self) -> V1SampledBlockSupportReplayFanoutResult:
        self._require_unaborted()
        if (
            self._sealed
            or self._active_sink is not None
            or self._next_segment != len(self._segments)
        ):
            raise RuntimeError("replay fanout cannot seal an incomplete source family")
        handle = self._accumulator.seal()
        multi = self._accumulator.multi_segment_registration
        receipt = V1SampledBlockSupportReplayFanoutReceipt(
            multi_segment_registration_hash=multi.registration_hash,
            structural_source_registration_hash=(
                multi.structural_source_registration_hash
            ),
            multi_segment_registration_axis_hash=(multi.segment_registration_axis_hash),
            replay_segment_registration_axis_hash=_axis_hash(
                "segment_registration",
                tuple(item.registration_hash for item in self._segments),
            ),
            numerical_source_chain_hash=multi.numerical_source_chain_hash,
            segment_source_registration_hashes=tuple(
                item.registration_hash for item in self._segments
            ),
            segment_sink_identity_hashes=tuple(self._sink_identities),
            segment_bridge_receipt_hashes=tuple(self._bridge_receipt_hashes),
            segment_count=len(self._segments),
            context_count=sum(item.context_count for item in self._segments),
            packet_count=self._packet_count,
            paired_route_chunk_count=self._chunk_count,
            fanout_event_count=self._event_count,
            block_support_source_receipt_hash=handle.receipt.receipt_hash,
            _issuer=_RECEIPT_ISSUER,
        )
        self._sealed = True
        return V1SampledBlockSupportReplayFanoutResult(
            source_handle=handle,
            receipt=receipt,
        )

    def abort_unsealed(self) -> str:
        """Idempotently unwind an unpublished replay family and its tensors."""

        if self._aborted:
            assert self._abort_hash is not None
            return self._abort_hash
        if self._sealed:
            raise RuntimeError(
                "sealed replay fanout transferred custody to its result handle"
            )
        active_sink_hash = (
            None
            if self._active_sink is None
            else self._active_sink.sink_identity_hash
        )
        abort_hash = content_hash(
            {
                "schema": (
                    "partition_free_v1_sampled_block_support_replay_abort_v1"
                ),
                "multi_segment_registration_hash": (
                    self._accumulator.multi_segment_registration.registration_hash
                ),
                "segment_registration_hashes": tuple(
                    value.registration_hash for value in self._segments
                ),
                "next_segment_ordinal": self._next_segment,
                "active_sink_identity_hash": active_sink_hash,
                "accepted_sink_identity_hashes": tuple(self._sink_identities),
                "accepted_bridge_receipt_hashes": tuple(
                    self._bridge_receipt_hashes
                ),
                "packet_count_before_abort": self._packet_count,
                "paired_route_chunk_count_before_abort": self._chunk_count,
                "fanout_event_count_before_abort": self._event_count,
                "unpublished_source_released": True,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "abort-",
        )
        self._accumulator.abort_unsealed()
        sink = self._active_sink
        if sink is not None:
            sink.active_count = 0
            sink.completed = False
        self._active_sink = None
        self._sink_identities.clear()
        self._bridge_receipt_hashes.clear()
        self._packet_count = 0
        self._chunk_count = 0
        self._event_count = 0
        self._abort_hash = abort_hash
        self._aborted = True
        return abort_hash

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live block-support replay coordinators are nonserializable")


__all__ = [
    "SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_AUTHORITY",
    "SAMPLED_BLOCK_SUPPORT_REPLAY_FANOUT_SCHEMA",
    "V1SampledBlockSupportReplayFanoutCoordinator",
    "V1SampledBlockSupportReplayFanoutReceipt",
    "V1SampledBlockSupportReplayFanoutResult",
]
