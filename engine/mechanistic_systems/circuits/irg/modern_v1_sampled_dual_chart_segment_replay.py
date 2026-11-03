"""Bounded zero-VJP replay of one sealed dual-chart source segment.

This is the residual-packet sibling of the conditioned-D1 replay.  It reads
one immutable corrected-A/literal-J shard pair in registered route windows,
selects a frozen ordered row subset, and sends the pair through the canonical
variable-shell source bridge.  Optional bound-chunk sinks receive the exact
issuer-typed chunk bindings, allowing the same read to construct a block-
support coefficient source.

The operation performs no model forward or backward pass and never
materializes a complete context-by-route matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, NoReturn, Sequence

import numpy as np

try:  # pragma: no cover - refusal is exercised at runtime construction
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .contracts import content_hash
from .modern_v1_neutral_runtime import DeclaredRouteAxis
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
    V1SampledVariableShellSourceBridge,
    V1SampledVariableShellSourceBridgeResult,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceCUDAAccumulator,
)


SAMPLED_DUAL_CHART_SEGMENT_REPLAY_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_segment_replay_v1"
)
SAMPLED_DUAL_CHART_SEGMENT_REPLAY_AUTHORITY: Final = (
    "zero_VJP_source_replay_execution_only_no_action_C_K_O_commit_terminal_or_"
    "pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-sampled-dual-chart-segment-replay-"
_RECEIPT_ISSUER: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be one exact positive integer")
    return value


def _row_selection_hash(values: Sequence[int]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_dual_chart_row_selection_v1",
            "row_indices": tuple(values),
        },
        prefix=_PREFIX + "row-selection-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledDualChartSegmentReplayReceipt:
    source_registration_hash: str
    corrected_a_shard_receipt_hash: str
    literal_j_shard_receipt_hash: str
    corrected_a_content_sha256: str
    literal_j_content_sha256: str
    row_selection_hash: str
    bridge_receipt_hash: str
    context_count: int
    route_count: int
    route_window_count: int
    source_descriptor_pass_count: int
    source_route_window_read_count: int
    source_payload_read_bytes: int
    selected_tensor_materialization_bytes: int
    maximum_owned_host_staging_bytes: int
    _issuer: object = field(default=None, repr=False, compare=False)
    raw_dual_roles_replayed_together: bool = True
    complete_role_domain_digest_previously_admitted: bool = True
    zero_additional_model_forwards: bool = True
    zero_additional_vjps: bool = True
    bounded_route_window_streaming: bool = True
    one_source_packet_replayed: bool = True
    context_by_route_rows_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    action_authority: bool = False
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_DUAL_CHART_SEGMENT_REPLAY_AUTHORITY
    schema: str = SAMPLED_DUAL_CHART_SEGMENT_REPLAY_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _RECEIPT_ISSUER:
            raise TypeError("dual-chart segment replay receipts are issuer-only")
        for name in (
            "source_registration_hash",
            "corrected_a_shard_receipt_hash",
            "literal_j_shard_receipt_hash",
            "corrected_a_content_sha256",
            "literal_j_content_sha256",
            "row_selection_hash",
            "bridge_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "context_count",
            "route_count",
            "route_window_count",
            "source_descriptor_pass_count",
            "source_route_window_read_count",
            "source_payload_read_bytes",
            "selected_tensor_materialization_bytes",
            "maximum_owned_host_staging_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        expected_selected_bytes = 2 * self.context_count * self.route_count * 8
        if (
            self.source_descriptor_pass_count != 2
            or self.source_route_window_read_count != 2 * self.route_window_count
            or self.selected_tensor_materialization_bytes != expected_selected_bytes
            or self.maximum_owned_host_staging_bytes
            > self.source_payload_read_bytes
            + self.selected_tensor_materialization_bytes
        ):
            raise ValueError("dual-chart replay byte/window telemetry does not close")
        required_true = (
            self.raw_dual_roles_replayed_together,
            self.complete_role_domain_digest_previously_admitted,
            self.zero_additional_model_forwards,
            self.zero_additional_vjps,
            self.bounded_route_window_streaming,
            self.one_source_packet_replayed,
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
            or self.authority_scope != SAMPLED_DUAL_CHART_SEGMENT_REPLAY_AUTHORITY
            or self.schema != SAMPLED_DUAL_CHART_SEGMENT_REPLAY_SCHEMA
        ):
            raise PermissionError("dual-chart segment replay crossed its boundary")
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
            raise ValueError("dual-chart segment replay receipt was mutated")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("dual-chart segment replay receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledDualChartSegmentReplayResult:
    bridge_result: V1SampledVariableShellSourceBridgeResult = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledDualChartSegmentReplayReceipt

    def __post_init__(self) -> None:
        if type(self.bridge_result) is not V1SampledVariableShellSourceBridgeResult:
            raise TypeError("segment replay requires the exact bridge result")
        if type(self.receipt) is not V1SampledDualChartSegmentReplayReceipt:
            raise TypeError("segment replay requires the exact replay receipt")
        self.receipt.assert_integrity()
        if self.bridge_result.receipt.receipt_hash != self.receipt.bridge_receipt_hash:
            raise ValueError("segment replay changed its bridge result")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live dual-chart segment replay results are nonserializable")


def replay_v1_sampled_dual_chart_segment(
    *,
    corrected_a_shard: V1SampledDualChartRowShardAttachment,
    literal_j_shard: V1SampledDualChartRowShardAttachment,
    row_indices: Sequence[int],
    accumulator: V1SignedWorkVariableShellSourceCUDAAccumulator,
    compact_route_axis: DeclaredRouteAxis,
    destination_transport_hash: str,
    bound_chunk_sinks: Sequence[V1SampledVariableShellBoundChunkSinkBinding] = (),
) -> V1SampledDualChartSegmentReplayResult:
    """Replay one frozen shard pair through one source-local registration."""

    if torch is None:
        raise RuntimeError("dual-chart segment replay requires PyTorch")
    if (
        type(corrected_a_shard) is not V1SampledDualChartRowShardAttachment
        or type(literal_j_shard) is not V1SampledDualChartRowShardAttachment
        or type(accumulator) is not V1SignedWorkVariableShellSourceCUDAAccumulator
        or type(compact_route_axis) is not DeclaredRouteAxis
    ):
        raise TypeError("dual-chart replay inputs have wrong types")
    corrected_a_shard.verify_live_witnesses(
        expected_role=V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY,
    )
    literal_j_shard.verify_live_witnesses(
        expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
    )
    a_receipt = corrected_a_shard.receipt
    j_receipt = literal_j_shard.receipt
    registration = accumulator.registration
    if type(row_indices) not in {tuple, list}:
        raise TypeError("dual-chart row selection must be one ordered sequence")
    selected = tuple(row_indices)
    if (
        not selected
        or any(type(value) is not int or value < 0 for value in selected)
        or len(selected) != len(set(selected))
        or any(value >= len(a_receipt.context_ids) for value in selected)
    ):
        raise ValueError("dual-chart row selection is empty, repeated, or out of range")
    shared = (
        a_receipt.registered_d_prefix_hash == j_receipt.registered_d_prefix_hash,
        a_receipt.source_capture_identity_hash
        == j_receipt.source_capture_identity_hash,
        a_receipt.chart_identity_hash == j_receipt.chart_identity_hash,
        a_receipt.current_mask_hash == j_receipt.current_mask_hash,
        a_receipt.proposal_generation == j_receipt.proposal_generation,
        a_receipt.route_axis_hash == j_receipt.route_axis_hash,
        a_receipt.route_id_axis_hash == j_receipt.route_id_axis_hash,
        a_receipt.route_count == j_receipt.route_count,
        a_receipt.context_ids == j_receipt.context_ids,
        a_receipt.draw_ids == j_receipt.draw_ids,
        a_receipt.stratum_labels == j_receipt.stratum_labels,
        a_receipt.route_windows == j_receipt.route_windows,
    )
    selected_contexts = tuple(a_receipt.context_ids[index] for index in selected)
    selected_draws = tuple(a_receipt.draw_ids[index] for index in selected)
    if (
        not all(shared)
        or a_receipt.role is not V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY
        or j_receipt.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
        or a_receipt.registered_d_prefix_hash != registration.registered_d_prefix_hash
        or a_receipt.source_capture_identity_hash
        != registration.same_capture_vjp_source_hash
        or a_receipt.chart_identity_hash != registration.chart_identity_hash
        or a_receipt.current_mask_hash != registration.current_mask_hash
        or a_receipt.proposal_generation != registration.proposal_generation
        or a_receipt.route_axis_hash != registration.route_axis_hash
        or a_receipt.route_id_axis_hash != registration.route_id_axis_hash
        or a_receipt.route_axis_hash != compact_route_axis.route_axis_hash
        or a_receipt.route_id_axis_hash != compact_route_axis.route_id_axis_hash
        or a_receipt.route_count != len(compact_route_axis.route_ids)
        or a_receipt.stream_identity_hash
        != registration.corrected_evidence_stream_source_hash
        or j_receipt.stream_identity_hash != registration.literal_j_stream_source_hash
        or selected_contexts != registration.registered_context_ids
        or selected_draws != registration.registered_draw_ids
        or len(selected) != registration.context_count
        or registration.context_count > accumulator.design.maximum_batch_size
    ):
        raise PermissionError("dual-chart shards differ from the replay registration")
    bridge = V1SampledVariableShellSourceBridge(
        accumulator=accumulator,
        route_ids=compact_route_axis.route_ids,
        route_axis_hash=registration.route_axis_hash,
        route_id_axis_hash=registration.route_id_axis_hash,
        destination_transport_hash=destination_transport_hash,
        bound_chunk_sinks=bound_chunk_sinks,
    )
    consumer = bridge.producer_consumer
    chunk_size = accumulator.design.route_chunk_size
    consumer.bind_route_chunk_size(chunk_size)
    evidence = consumer.evidence_consumer
    literal = consumer.jacobian_consumer
    evidence.begin_packet(
        registration.registered_context_ids,
        route_ids=compact_route_axis.route_ids,
        destination_transport_hash=destination_transport_hash,
        loss_pullback_hash=registration.corrected_evidence_stream_source_hash,
    )
    literal.begin_packet(
        registration.registered_context_ids,
        route_ids=compact_route_axis.route_ids,
        destination_transport_hash=destination_transport_hash,
        loss_pullback_hash=registration.literal_j_stream_source_hash,
    )
    window_count = 0
    payload_bytes = 0
    selected_bytes = 0
    maximum_staging = 0
    a_windows = corrected_a_shard.iter_route_windows(verify=True)
    j_windows = literal_j_shard.iter_route_windows(verify=True)
    for a_window, j_window in zip(a_windows, j_windows, strict=True):
        a_start, a_stop, a_rows = a_window
        j_start, j_stop, j_rows = j_window
        if (a_start, a_stop) != (j_start, j_stop):
            raise PermissionError("dual-chart replay route windows differ")
        window_count += 1
        payload_bytes += int(a_rows.nbytes + j_rows.nbytes)
        maximum_staging = max(maximum_staging, int(a_rows.nbytes + j_rows.nbytes))
        width = a_stop - a_start
        for local_start in range(0, width, chunk_size):
            local_stop = min(width, local_start + chunk_size)
            start = a_start + local_start
            stop = a_start + local_stop
            a_selected = np.array(
                a_rows[selected, local_start:local_stop],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            j_selected = np.array(
                j_rows[selected, local_start:local_stop],
                dtype=np.float64,
                order="C",
                copy=True,
            )
            if not (
                bool(np.isfinite(a_selected).all())
                and bool(np.isfinite(j_selected).all())
            ):
                raise ValueError("dual-chart replay encountered nonfinite source rows")
            selected_bytes += int(a_selected.nbytes + j_selected.nbytes)
            maximum_staging = max(
                maximum_staging,
                int(
                    a_rows.nbytes
                    + j_rows.nbytes
                    + a_selected.nbytes
                    + j_selected.nbytes
                ),
            )
            a_tensor = torch.from_numpy(a_selected).to(
                device=accumulator.device,
                dtype=torch.float64,
            )
            j_tensor = torch.from_numpy(j_selected).to(
                device=accumulator.device,
                dtype=torch.float64,
            )
            evidence.consume_positive_removal_chunk(start, stop, a_tensor.contiguous())
            literal.consume_positive_removal_chunk(start, stop, j_tensor.contiguous())
    evidence.end_packet()
    literal.end_packet()
    result = bridge.seal()
    if (
        window_count != len(a_receipt.route_windows)
        or selected_bytes != 2 * len(selected) * a_receipt.route_count * 8
    ):
        raise RuntimeError("dual-chart replay did not consume its exact bounded cover")
    receipt = V1SampledDualChartSegmentReplayReceipt(
        source_registration_hash=registration.registration_hash,
        corrected_a_shard_receipt_hash=a_receipt.receipt_hash,
        literal_j_shard_receipt_hash=j_receipt.receipt_hash,
        corrected_a_content_sha256=a_receipt.content_sha256,
        literal_j_content_sha256=j_receipt.content_sha256,
        row_selection_hash=_row_selection_hash(selected),
        bridge_receipt_hash=result.receipt.receipt_hash,
        context_count=len(selected),
        route_count=a_receipt.route_count,
        route_window_count=window_count,
        source_descriptor_pass_count=2,
        source_route_window_read_count=2 * window_count,
        source_payload_read_bytes=payload_bytes,
        selected_tensor_materialization_bytes=selected_bytes,
        maximum_owned_host_staging_bytes=maximum_staging,
        _issuer=_RECEIPT_ISSUER,
    )
    return V1SampledDualChartSegmentReplayResult(
        bridge_result=result,
        receipt=receipt,
    )


__all__ = [
    "SAMPLED_DUAL_CHART_SEGMENT_REPLAY_AUTHORITY",
    "SAMPLED_DUAL_CHART_SEGMENT_REPLAY_SCHEMA",
    "V1SampledDualChartSegmentReplayReceipt",
    "V1SampledDualChartSegmentReplayResult",
    "replay_v1_sampled_dual_chart_segment",
]
