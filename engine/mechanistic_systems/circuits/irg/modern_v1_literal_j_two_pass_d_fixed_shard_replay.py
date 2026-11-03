"""Immutable-source replay for the two-pass literal-J ``D_fixed`` owner.

The numerical owner in :mod:`modern_v1_literal_j_two_pass_d_fixed_projection_cuda`
deliberately knows nothing about files or sampled-D custody. This module is
its production source adapter. It binds the exact block-support plan and its
live numerical result to the conditioned-D1 compact-source receipt, every raw
literal-J shard that produced it, and one source-issued residual segment
replay receipt per later physical segment.

Conditioned D1 is one logical packet even when target-then-complement order
spans several raw capture shards. Each pass reads the immutable shards in the
same bounded route-window order, gathers only the registered rows into one
transient tile, submits it, and immediately releases the tile. No VJP,
context-by-route HBM bank, route-scale D2H, or caller-authored digest is
available at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_direction_cuda import (
    V1LiteralJAffineShellDirectionDeviceHandle,
)

from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJPrivatePrimalDirectionCUDAHandle,
    V1LiteralJPrivatePrimalDirectionCUDALease,
    V1LiteralJPrivatePrimalDirectionDescriptorAxis,
)
from .modern_v1_literal_j_two_pass_d_fixed_projection_cuda import (
    V1LiteralJTwoPassDFixedProjectionCUDAAccumulator,
    V1LiteralJTwoPassDFixedProjectionDesign,
    V1LiteralJTwoPassDFixedProjectionExecution,
    V1LiteralJTwoPassDFixedProjectionRegistration,
    register_v1_literal_j_two_pass_d_fixed_projection,
)
from .modern_v1_sampled_conditioned_d1_compact_source import (
    V1SampledConditionedD1CompactSourceReceipt,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_dual_chart_segment_replay import (
    V1SampledDualChartSegmentReplayReceipt,
)
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellSourceBridgeReceipt,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportCoefficientLease,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceReceipt,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceReceipt,
)

if TYPE_CHECKING:
    from .modern_v1_sampled_block_support_numeric_source import (
        V1SampledBlockSupportNumericSourceResult,
    )
    from .modern_v1_sampled_block_support_replay_registration import (
        V1SampledBlockSupportReplayRegistrationPlan,
    )

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_source_bound_shard_replay_v2"
)
LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_AUTHORITY: Final = (
    "source_bound_D_fixed_projection_only_no_D_decision_C_K_O_commit_terminal_"
    "or_pareto_authority_v2"
)
LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_ALGORITHM: Final = (
    "conditioned_D1_multishard_target_then_complement_plus_ordered_residual_"
    "selected_shards_two_verified_reads_literal_J_projection_v2"
)
LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_source_bound_shard_replay_v2"
)
LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_AUTHORITY: Final = (
    "source_bound_structured_projection_only_no_population_FD_solver_D_C_K_O_"
    "commit_terminal_or_pareto_authority_v1"
)
LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_ALGORITHM: Final = (
    "immutable_D_fixed_two_pass_literal_J_outside_plus_shell_basis_gather_v1"
)
_PREFIX: Final = "partition-free-v1-literal-j-two-pass-D-fixed-shard-replay-"
_D1_PREFIX: Final = "partition-free-v1-sampled-conditioned-D1-compact-source-"
_SEGMENT_PREFIX: Final = "partition-free-v1-sampled-dual-chart-segment-replay-"
_VARIABLE_PREFIX: Final = "partition-free-v1-variable-shell-source-"
_ISSUER: Final = object()


class V1LiteralJTwoPassDFixedShardReplayRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    PARTIAL_REPLAY = "partial_replay"
    NONFINITE_ROW = "nonfinite_row"
    DIRECTION_CUSTODY_MISMATCH = "direction_custody_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJTwoPassDFixedShardReplayRefusal(RuntimeError):
    """Typed fail-closed refusal at the immutable source boundary."""

    def __init__(
        self,
        code: V1LiteralJTwoPassDFixedShardReplayRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJTwoPassDFixedShardReplayRefusalCode:
            raise TypeError("literal-J shard replay refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJTwoPassDFixedShardReplayRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJTwoPassDFixedShardReplayRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _identity_axis(
    values: Sequence[str],
    *,
    name: str,
    unique: bool = True,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be one ordered string axis")
    axis = tuple(_identity(value, name=f"{name} member") for value in values)
    if (not axis and not allow_empty) or (unique and len(axis) != len(set(axis))):
        raise ValueError(f"{name} must be nonempty and correctly unique")
    return axis


def _variable_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_VARIABLE_PREFIX + "axis-",
    )


def _d1_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_conditioned_D1_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_D1_PREFIX + role + "-axis-",
    )


def _segment_row_selection_hash(values: Sequence[int]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_dual_chart_row_selection_v1",
            "row_indices": tuple(values),
        },
        prefix=_SEGMENT_PREFIX + "row-selection-",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJConditionedD1ShardReplaySource:
    """Exact durable D1 result plus its still-live raw-J attachments."""

    replay_receipt: V1SampledConditionedD1CompactSourceReceipt
    compact_source_receipt: V1SignedWorkVariableShellSourceReceipt
    literal_j_shards: tuple[V1SampledDualChartRowShardAttachment, ...] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            type(self.replay_receipt) is not V1SampledConditionedD1CompactSourceReceipt
            or type(self.compact_source_receipt)
            is not V1SignedWorkVariableShellSourceReceipt
            or type(self.literal_j_shards) is not tuple
            or not self.literal_j_shards
            or any(
                type(item) is not V1SampledDualChartRowShardAttachment
                for item in self.literal_j_shards
            )
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
                "conditioned-D1 replay requires exact receipt and attachment types",
            )
        replay = self.replay_receipt
        replay.assert_integrity()
        source = self.compact_source_receipt
        if (
            source.receipt_hash != replay.compact_source_receipt_hash
            or tuple(item.receipt.receipt_hash for item in self.literal_j_shards)
            != replay.literal_j_shard_receipt_hashes
            or tuple(item.receipt.content_sha256 for item in self.literal_j_shards)
            != replay.literal_j_content_sha256s
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
                "conditioned-D1 compact result and raw literal-J custody differ",
            )
        for shard in self.literal_j_shards:
            shard.verify_live_witnesses(
                expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
                expected_chart_identity_hash=replay.chart_identity_hash,
                expected_current_mask_hash=replay.current_mask_hash,
                expected_proposal_generation=replay.proposal_generation,
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live conditioned-D1 shard replay sources are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualShardReplaySource:
    """One frozen selected residual shard and its complete replay chain."""

    replay_receipt: V1SampledDualChartSegmentReplayReceipt
    bridge_receipt: V1SampledVariableShellSourceBridgeReceipt
    compact_source_receipt: V1SignedWorkVariableShellSourceReceipt
    literal_j_shard: V1SampledDualChartRowShardAttachment = field(
        repr=False,
        compare=False,
    )
    selected_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.replay_receipt) is not V1SampledDualChartSegmentReplayReceipt
            or type(self.bridge_receipt)
            is not V1SampledVariableShellSourceBridgeReceipt
            or type(self.compact_source_receipt)
            is not V1SignedWorkVariableShellSourceReceipt
            or type(self.literal_j_shard) is not V1SampledDualChartRowShardAttachment
            or type(self.selected_row_indices) is not tuple
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
                "residual replay requires exact replay, bridge, source, and shard types",
            )
        selected = tuple(
            _nonnegative_int(item, name="selected residual row")
            for item in self.selected_row_indices
        )
        if not selected or len(selected) != len(set(selected)):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.ORDER_MISMATCH,
                "residual row selection must be nonempty and unique",
            )
        object.__setattr__(self, "selected_row_indices", selected)
        replay = self.replay_receipt
        replay.assert_integrity()
        bridge = self.bridge_receipt
        bridge.assert_integrity()
        source = self.compact_source_receipt
        shard = self.literal_j_shard
        shard.verify_live_witnesses(
            expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
        )
        receipt = shard.receipt
        if (
            replay.bridge_receipt_hash != bridge.receipt_hash
            or bridge.compact_source_receipt_hash != source.receipt_hash
            or replay.source_registration_hash != source.registration_hash
            or replay.literal_j_shard_receipt_hash != receipt.receipt_hash
            or replay.literal_j_content_sha256 != receipt.content_sha256
            or replay.row_selection_hash != _segment_row_selection_hash(selected)
            or replay.context_count != len(selected)
            or any(item >= len(receipt.context_ids) for item in selected)
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
                "residual selected shard does not replay its source-issued receipt chain",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live residual shard replay sources are nonserializable")


@dataclass(frozen=True, slots=True)
class _ShardSlice:
    shard: V1SampledDualChartRowShardAttachment = field(repr=False, compare=False)
    source_rows: tuple[int, ...]
    destination_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _PacketSource:
    packet_index: int
    slices: tuple[_ShardSlice, ...]
    route_windows: tuple[tuple[int, int], ...]
    source_axis_hash: str


def _source_receipt_matches_packet(
    *,
    source: V1SignedWorkVariableShellSourceReceipt,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    packet: Any,
) -> bool:
    axes = (
        ("context", packet.context_ids, source.context_axis_hash),
        ("draw", packet.draw_ids, source.draw_axis_hash),
        ("measurement", packet.measurement_ids, source.measurement_axis_hash),
        ("capture", packet.capture_ids, source.capture_axis_hash),
        ("source_packet", packet.source_packet_hashes, source.source_packet_axis_hash),
    )
    return (
        source.registration_hash == packet.physical_source_registration_hash
        and source.model_identity_hash == registration.model_identity_hash
        and source.chart_identity_hash == registration.chart_identity_hash
        and source.current_mask_hash == registration.current_mask_hash
        and source.proposal_generation == registration.proposal_generation
        and source.route_axis_hash == registration.route_axis_hash
        and source.route_id_axis_hash == registration.route_id_axis_hash
        and source.route_count == registration.route_count
        and source.registered_d_prefix_hash == packet.registered_d_prefix_hash
        and source.registered_d_w2_prefix_hash == packet.registered_d_w2_prefix_hash
        and source.witness_stream_source_hash == packet.witness_stream_source_hash
        and source.same_capture_vjp_source_hash == packet.same_capture_vjp_source_hash
        and source.literal_j_stream_source_hash == packet.literal_j_stream_source_hash
        and source.context_count == packet.context_count
        and all(
            _variable_axis_hash(role, values) == observed
            for role, values, observed in axes
        )
    )


def _conditioned_packet_source(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    packet: Any,
    source: V1LiteralJConditionedD1ShardReplaySource,
) -> _PacketSource:
    replay = source.replay_receipt
    compact = source.compact_source_receipt
    if packet.segment_role != "conditioned_D1" or not _source_receipt_matches_packet(
        source=compact,
        registration=registration,
        packet=packet,
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "conditioned-D1 compact source left the registered physical packet",
        )
    expected_d1_hashes = (
        replay.canonical_context_axis_hash,
        replay.canonical_draw_axis_hash,
        replay.canonical_measurement_axis_hash,
        replay.canonical_capture_axis_hash,
        replay.canonical_source_packet_axis_hash,
    )
    expected_d1_axes = (
        ("context", packet.context_ids),
        ("draw", packet.draw_ids),
        ("measurement", packet.measurement_ids),
        ("capture", packet.capture_ids),
        ("source_packet", packet.source_packet_hashes),
    )
    if (
        replay.route_axis_hash != registration.route_axis_hash
        or replay.route_id_axis_hash != registration.route_id_axis_hash
        or replay.route_count != registration.route_count
        or replay.chart_identity_hash != registration.chart_identity_hash
        or replay.current_mask_hash != registration.current_mask_hash
        or replay.proposal_generation != registration.proposal_generation
        or replay.registered_d_prefix_hash != packet.registered_d_prefix_hash
        or replay.literal_j_stream_source_hash != packet.literal_j_stream_source_hash
        or replay.context_count != packet.context_count
        or tuple(_d1_axis_hash(role, axis) for role, axis in expected_d1_axes)
        != expected_d1_hashes
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "conditioned-D1 semantic receipt does not replay the packet axes",
        )

    locations: dict[str, tuple[int, int, str, str]] = {}
    route_windows: tuple[tuple[int, int], ...] | None = None
    for shard_index, shard in enumerate(source.literal_j_shards):
        receipt = shard.receipt
        if (
            receipt.registered_d_prefix_hash != packet.registered_d_prefix_hash
            or receipt.chart_identity_hash != registration.chart_identity_hash
            or receipt.current_mask_hash != registration.current_mask_hash
            or receipt.proposal_generation != registration.proposal_generation
            or receipt.route_axis_hash != registration.route_axis_hash
            or receipt.route_id_axis_hash != registration.route_id_axis_hash
            or receipt.route_count != registration.route_count
            or receipt.stream_identity_hash != packet.literal_j_stream_source_hash
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
                "one D1 literal-J shard left the registered chart or route axis",
            )
        if route_windows is None:
            route_windows = receipt.route_windows
        elif receipt.route_windows != route_windows:
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.ORDER_MISMATCH,
                "D1 source shards do not expose one congruent route-window cover",
            )
        for row, (context, draw, stratum) in enumerate(
            zip(
                receipt.context_ids,
                receipt.draw_ids,
                receipt.stratum_labels,
                strict=True,
            )
        ):
            if context in locations:
                _refuse(
                    V1LiteralJTwoPassDFixedShardReplayRefusalCode.ORDER_MISMATCH,
                    "D1 raw custody duplicates one registered context",
                )
            locations[context] = (shard_index, row, draw, stratum)
    if set(locations) != set(packet.context_ids):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.PARTIAL_REPLAY,
            "D1 raw custody does not exactly cover the logical packet",
        )
    strata = tuple(locations[context][3] for context in packet.context_ids)
    expected_strata = ("target",) * replay.d1_stratum_counts[0] + (
        "complement",
    ) * replay.d1_stratum_counts[1]
    if (
        strata != expected_strata
        or tuple(locations[context][2] for context in packet.context_ids)
        != packet.draw_ids
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.ORDER_MISMATCH,
            "D1 rows do not replay canonical target-then-complement context/draw order",
        )
    slices: list[_ShardSlice] = []
    for shard_index, shard in enumerate(source.literal_j_shards):
        destinations = tuple(
            index
            for index, context in enumerate(packet.context_ids)
            if locations[context][0] == shard_index
        )
        if destinations:
            slices.append(
                _ShardSlice(
                    shard=shard,
                    source_rows=tuple(
                        locations[packet.context_ids[index]][1]
                        for index in destinations
                    ),
                    destination_rows=destinations,
                )
            )
    assert route_windows is not None
    source_axis_hash = content_hash(
        {
            "schema": "literal_j_D_fixed_conditioned_D1_source_axis_v2",
            "packet_descriptor_hash": packet.descriptor_hash,
            "conditioned_D1_replay_receipt_hash": replay.receipt_hash,
            "compact_source_receipt_hash": compact.receipt_hash,
            "literal_j_shards": tuple(
                (
                    item.shard.receipt.receipt_hash,
                    item.shard.receipt.logical_identity_hash,
                    item.shard.receipt.content_sha256,
                    item.source_rows,
                    item.destination_rows,
                )
                for item in slices
            ),
            "route_windows": route_windows,
        },
        prefix=_PREFIX + "D1-source-axis-",
    )
    return _PacketSource(
        packet_index=packet.packet_index,
        slices=tuple(slices),
        route_windows=route_windows,
        source_axis_hash=source_axis_hash,
    )


def _residual_packet_source(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    packet: Any,
    source: V1LiteralJResidualShardReplaySource,
) -> _PacketSource:
    compact = source.compact_source_receipt
    replay = source.replay_receipt
    bridge = source.bridge_receipt
    shard = source.literal_j_shard
    receipt = shard.receipt
    selected = source.selected_row_indices
    if (
        not packet.segment_role.startswith("residual_fit:")
        or not _source_receipt_matches_packet(
            source=compact,
            registration=registration,
            packet=packet,
        )
        or bridge.registration_hash != packet.physical_source_registration_hash
        or bridge.route_axis_hash != registration.route_axis_hash
        or bridge.route_id_axis_hash != registration.route_id_axis_hash
        or bridge.same_capture_vjp_source_hash != packet.same_capture_vjp_source_hash
        or bridge.literal_j_stream_source_hash != packet.literal_j_stream_source_hash
        or replay.route_count != registration.route_count
        or receipt.registered_d_prefix_hash != packet.registered_d_prefix_hash
        or receipt.source_capture_identity_hash != packet.same_capture_vjp_source_hash
        or receipt.chart_identity_hash != registration.chart_identity_hash
        or receipt.current_mask_hash != registration.current_mask_hash
        or receipt.proposal_generation != registration.proposal_generation
        or receipt.route_axis_hash != registration.route_axis_hash
        or receipt.route_id_axis_hash != registration.route_id_axis_hash
        or receipt.route_count != registration.route_count
        or receipt.stream_identity_hash != packet.literal_j_stream_source_hash
        or tuple(receipt.context_ids[index] for index in selected) != packet.context_ids
        or tuple(receipt.draw_ids[index] for index in selected) != packet.draw_ids
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "residual selected shard does not replay the next physical packet",
        )
    selection_axis_hash = content_hash(
        {
            "schema": "literal_j_D_fixed_residual_source_axis_v2",
            "packet_descriptor_hash": packet.descriptor_hash,
            "segment_replay_receipt_hash": replay.receipt_hash,
            "bridge_receipt_hash": bridge.receipt_hash,
            "compact_source_receipt_hash": compact.receipt_hash,
            "literal_j_shard_receipt_hash": receipt.receipt_hash,
            "literal_j_logical_identity_hash": receipt.logical_identity_hash,
            "literal_j_content_sha256": receipt.content_sha256,
            "selected_row_indices": selected,
            "route_windows": receipt.route_windows,
        },
        prefix=_PREFIX + "residual-source-axis-",
    )
    return _PacketSource(
        packet_index=packet.packet_index,
        slices=(
            _ShardSlice(
                shard=shard,
                source_rows=selected,
                destination_rows=tuple(range(packet.context_count)),
            ),
        ),
        route_windows=receipt.route_windows,
        source_axis_hash=selection_axis_hash,
    )


def _validate_numeric_and_plan(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
) -> int:
    from .modern_v1_sampled_block_support_numeric_source import (
        V1SampledBlockSupportNumericSourceResult,
    )
    from .modern_v1_sampled_block_support_replay_registration import (
        V1SampledBlockSupportReplayRegistrationPlan,
    )

    if (
        type(replay_plan) is not V1SampledBlockSupportReplayRegistrationPlan
        or type(numeric_source) is not V1SampledBlockSupportNumericSourceResult
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "source replay requires the exact plan and live numeric-source result",
        )
    plan = replay_plan.receipt
    plan.assert_integrity()
    numeric_source.receipt.assert_integrity()
    numeric_source.source_handle.assert_live()
    live = numeric_source.source_handle.receipt
    durable = numeric_source.receipt
    if (
        durable.replay_registration_receipt_hash != plan.receipt_hash
        or durable.block_support_source_receipt_hash != live.receipt_hash
        or live.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or live.source_registration_hash != plan.structural_registration_hash
        or live.model_identity_hash != plan.model_identity_hash
        or live.chart_identity_hash != plan.chart_identity_hash
        or live.current_mask_hash != plan.current_mask_hash
        or live.proposal_generation != plan.proposal_generation
        or live.route_axis_hash != plan.route_axis_hash
        or live.route_id_axis_hash != plan.route_id_axis_hash
        or live.context_count != plan.structural_context_count
        or live.segment_registration_count != len(plan.segment_roles)
        or live.multi_segment_numerical_source is not True
        or live.route_count <= 0
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "live numeric block-support source does not replay the exact structural plan",
        )
    return live.route_count


def _direction_axis(
    custody: (
        V1LiteralJPrivatePrimalDirectionCUDAHandle
        | V1LiteralJPrivatePrimalDirectionCUDALease
    ),
) -> V1LiteralJPrivatePrimalDirectionDescriptorAxis:
    if type(custody) not in {
        V1LiteralJPrivatePrimalDirectionCUDAHandle,
        V1LiteralJPrivatePrimalDirectionCUDALease,
    }:
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "production replay accepts only a typed KKT private-primal handle or lease",
        )
    return custody.descriptor_axis


def _same_vjp_proof_hash(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
) -> str:
    return content_hash(
        {
            "schema": "literal_j_D_fixed_source_owned_same_VJP_proof_v2",
            "replay_registration_receipt_hash": replay_plan.receipt.receipt_hash,
            "numeric_source_receipt_hash": numeric_source.receipt.receipt_hash,
            "conditioned_D1_replay_receipt_hash": (
                conditioned_d1.replay_receipt.receipt_hash
            ),
            "conditioned_D1_compact_source_receipt_hash": (
                conditioned_d1.compact_source_receipt.receipt_hash
            ),
            "conditioned_D1_corrected_shard_receipts": (
                conditioned_d1.replay_receipt.corrected_a_shard_receipt_hashes
            ),
            "conditioned_D1_corrected_content": (
                conditioned_d1.replay_receipt.corrected_a_content_sha256s
            ),
            "conditioned_D1_literal_shard_receipts": tuple(
                item.receipt.receipt_hash for item in conditioned_d1.literal_j_shards
            ),
            "conditioned_D1_literal_content": tuple(
                item.receipt.content_sha256 for item in conditioned_d1.literal_j_shards
            ),
            "residual_replay_receipt_hashes": tuple(
                item.replay_receipt.receipt_hash for item in residual_segments
            ),
            "residual_corrected_shard_receipts": tuple(
                item.replay_receipt.corrected_a_shard_receipt_hash
                for item in residual_segments
            ),
            "residual_corrected_content": tuple(
                item.replay_receipt.corrected_a_content_sha256
                for item in residual_segments
            ),
            "residual_literal_shard_receipts": tuple(
                item.literal_j_shard.receipt.receipt_hash for item in residual_segments
            ),
            "residual_literal_content": tuple(
                item.literal_j_shard.receipt.content_sha256
                for item in residual_segments
            ),
            "row_selection_hashes": tuple(
                item.replay_receipt.row_selection_hash for item in residual_segments
            ),
            "no_caller_authored_digest": True,
        },
        prefix=_PREFIX + "same-VJP-proof-",
    )


def _packet_begin_kwargs(
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    packet: Any,
) -> dict[str, object]:
    return {
        "context_ids": packet.context_ids,
        "draw_ids": packet.draw_ids,
        "measurement_ids": packet.measurement_ids,
        "capture_ids": packet.capture_ids,
        "source_packet_hashes": packet.source_packet_hashes,
        "model_identity_hash": registration.model_identity_hash,
        "chart_identity_hash": registration.chart_identity_hash,
        "current_mask_hash": registration.current_mask_hash,
        "proposal_generation": registration.proposal_generation,
        "route_axis_hash": registration.route_axis_hash,
        "route_id_axis_hash": registration.route_id_axis_hash,
        "same_capture_and_vjp_execution_proof_hash": (
            registration.same_capture_and_vjp_execution_proof_hash
        ),
        "same_capture_vjp_source_hash": packet.same_capture_vjp_source_hash,
        "literal_j_stream_source_hash": packet.literal_j_stream_source_hash,
    }


def _source_tile_hash(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    packet: Any,
    source: _PacketSource,
    parent_window: tuple[int, int],
    route_window: tuple[int, int],
) -> str:
    return content_hash(
        {
            "schema": "literal_j_D_fixed_immutable_selected_source_tile_v2",
            "registration_hash": registration.registration_hash,
            "packet_descriptor_hash": packet.descriptor_hash,
            "packet_source_axis_hash": source.source_axis_hash,
            "parent_route_window": parent_window,
            "submitted_route_window": route_window,
            "literal_j_sources": tuple(
                (
                    item.shard.receipt.receipt_hash,
                    item.shard.receipt.logical_identity_hash,
                    item.shard.receipt.content_sha256,
                    item.source_rows,
                    item.destination_rows,
                )
                for item in source.slices
            ),
            "same_capture_and_vjp_execution_proof_hash": (
                registration.same_capture_and_vjp_execution_proof_hash
            ),
            "no_caller_authored_digest": True,
        },
        prefix=_PREFIX + "source-tile-",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedShardReplayReceipt:
    registration_hash: str
    projection_receipt_hash: str
    projection_telemetry_hash: str
    replay_registration_receipt_hash: str
    numeric_source_receipt_hash: str
    live_block_support_source_receipt_hash: str
    conditioned_d1_replay_receipt_hash: str
    residual_replay_receipt_hashes: tuple[str, ...]
    packet_source_axis_hashes: tuple[str, ...]
    direction_descriptor_axis_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    literal_j_tile_source_ledger_hash: str
    context_count: int
    route_count: int
    packet_count: int
    raw_shard_count: int
    immutable_shard_read_pass_count: int
    shard_route_window_read_count: int
    accumulator_tile_submit_count: int
    source_payload_read_bytes: int
    host_to_device_tile_bytes: int
    maximum_owned_host_staging_bytes: int
    maximum_device_tile_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    _issuer: object = field(default=None, repr=False, compare=False)
    conditioned_d1_is_one_logical_packet: bool = True
    conditioned_d1_target_then_complement_order: bool = True
    residual_packet_order_frozen: bool = True
    immutable_source_value_identity_bound: bool = True
    caller_authored_digest_seam_present: bool = False
    zero_additional_model_forwards: bool = True
    zero_additional_vjps: bool = True
    context_route_bank_retained: bool = False
    route_scale_rows_d2h: bool = False
    array_free: bool = True
    projection_authority: bool = True
    d_decision_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_ALGORITHM
    authority_scope: str = LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_AUTHORITY
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise TypeError("literal-J shard replay receipts are issuer-only")
        for name in (
            "registration_hash",
            "projection_receipt_hash",
            "projection_telemetry_hash",
            "replay_registration_receipt_hash",
            "numeric_source_receipt_hash",
            "live_block_support_source_receipt_hash",
            "conditioned_d1_replay_receipt_hash",
            "direction_descriptor_axis_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "literal_j_tile_source_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("residual_replay_receipt_hashes", "packet_source_axis_hashes"):
            object.__setattr__(
                self,
                name,
                _identity_axis(
                    getattr(self, name),
                    name=name,
                    allow_empty=name == "residual_replay_receipt_hashes",
                ),
            )
        for name in (
            "context_count",
            "route_count",
            "packet_count",
            "raw_shard_count",
            "immutable_shard_read_pass_count",
            "shard_route_window_read_count",
            "accumulator_tile_submit_count",
            "source_payload_read_bytes",
            "host_to_device_tile_bytes",
            "maximum_owned_host_staging_bytes",
            "maximum_device_tile_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            min(
                self.context_count,
                self.route_count,
                self.packet_count,
                self.raw_shard_count,
                self.shard_route_window_read_count,
                self.accumulator_tile_submit_count,
            )
            <= 0
            or self.packet_count != len(self.packet_source_axis_hashes)
            or self.packet_count != len(self.residual_replay_receipt_hashes) + 1
            or self.immutable_shard_read_pass_count != 2
            or self.production_cuda_execution == self.private_cpu_reference_execution
        ):
            raise ValueError(
                "literal-J source replay counts or execution mode do not close"
            )
        required_true = (
            self.conditioned_d1_is_one_logical_packet,
            self.conditioned_d1_target_then_complement_order,
            self.residual_packet_order_frozen,
            self.immutable_source_value_identity_bound,
            self.zero_additional_model_forwards,
            self.zero_additional_vjps,
            self.array_free,
            self.projection_authority,
        )
        required_false = (
            self.caller_authored_digest_seam_present,
            self.context_route_bank_retained,
            self.route_scale_rows_d2h,
            self.d_decision_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.algorithm != LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_ALGORITHM
            or self.authority_scope != LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_AUTHORITY
            or self.schema != LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_SCHEMA
        ):
            raise PermissionError(
                "literal-J source replay crossed its authority boundary"
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"_issuer", "_receipt_hash"}
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("literal-J source replay receipt was mutated")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("literal-J source replay receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedShardReplayResult:
    registration: V1LiteralJTwoPassDFixedProjectionRegistration
    projection: V1LiteralJTwoPassDFixedProjectionExecution = field(
        repr=False,
        compare=False,
    )
    receipt: V1LiteralJTwoPassDFixedShardReplayReceipt

    def __post_init__(self) -> None:
        if (
            type(self.registration) is not V1LiteralJTwoPassDFixedProjectionRegistration
            or type(self.projection) is not V1LiteralJTwoPassDFixedProjectionExecution
            or type(self.receipt) is not V1LiteralJTwoPassDFixedShardReplayReceipt
        ):
            raise TypeError("literal-J shard replay result requires exact types")
        self.registration.assert_integrity()
        self.receipt.assert_integrity()
        if (
            self.registration.registration_hash != self.receipt.registration_hash
            or self.projection.receipt.registration_hash
            != self.registration.registration_hash
            or self.projection.receipt.source_replay_axis_hash
            != self.registration.source_replay_axis_hash
            or self.projection.receipt.receipt_hash
            != self.receipt.projection_receipt_hash
            or self.projection.telemetry.telemetry_hash
            != self.receipt.projection_telemetry_hash
        ):
            raise ValueError("literal-J shard replay projection identity changed")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live literal-J shard replay results are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellShardReplayReceipt:
    source_bound_replay_receipt_hash: str
    registration_hash: str
    projection_receipt_hash: str
    projection_telemetry_hash: str
    direction_receipt_hash: str
    direction_source_hash: str
    direction_lineage_hash: str
    preoutcome_registration_hash: str
    source_registration_hash: str
    postsource_receipt_hash: str
    exact_postsource_custody_hash: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_stream_binding_hash: str
    preoutcome_fanout_receipt_hash: str
    preoutcome_destination_binding_hash: str
    preoutcome_binding_axis_hash: str
    preoutcome_physical_action_axis_hash: str
    projection_replay_registration_receipt_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    alpha_lineage_hash: str
    replay_registration_receipt_hash: str
    numeric_source_receipt_hash: str
    literal_j_tile_source_ledger_hash: str
    restart_replay_hash: str
    route_count: int
    shell_size: int
    immutable_shard_read_pass_count: int
    structured_direction_custody: bool = True
    outside_zero_on_shell: bool = True
    no_dense_basis_bank: bool = True
    zero_additional_model_forwards: bool = True
    zero_additional_vjps: bool = True
    context_route_bank_retained: bool = False
    route_scale_rows_d2h: bool = False
    array_free: bool = True
    source_projection_authority: bool = True
    exact_preoutcome_postsource_custody_bound: bool = True
    population_moment_authority: bool = False
    registered_fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_ALGORITHM
    authority_scope: str = LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise TypeError("affine-shell shard replay receipts are issuer-only")
        for name in (
            "source_bound_replay_receipt_hash",
            "registration_hash",
            "projection_receipt_hash",
            "projection_telemetry_hash",
            "direction_receipt_hash",
            "direction_source_hash",
            "direction_lineage_hash",
            "preoutcome_registration_hash",
            "source_registration_hash",
            "postsource_receipt_hash",
            "exact_postsource_custody_hash",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "literal_j_stream_binding_hash",
            "preoutcome_fanout_receipt_hash",
            "preoutcome_destination_binding_hash",
            "preoutcome_binding_axis_hash",
            "preoutcome_physical_action_axis_hash",
            "projection_replay_registration_receipt_hash",
            "shell_freeze_receipt_hash",
            "alpha_lineage_hash",
            "replay_registration_receipt_hash",
            "numeric_source_receipt_hash",
            "literal_j_tile_source_ledger_hash",
            "restart_replay_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        object.__setattr__(
            self, "shell_size", _positive_int(self.shell_size, name="shell_size")
        )
        object.__setattr__(
            self,
            "immutable_shard_read_pass_count",
            _positive_int(
                self.immutable_shard_read_pass_count,
                name="immutable_shard_read_pass_count",
            ),
        )
        indices = tuple(
            _nonnegative_int(item, name="affine-shell physical index")
            for item in self.shell_indices
        )
        ids = _identity_axis(self.shell_route_ids, name="affine-shell route ids")
        if (
            not 1 <= len(indices) <= 8
            or len(indices) != len(ids)
            or len(indices) != len(set(indices))
            or any(item >= self.route_count for item in indices)
            or self.shell_size != len(indices)
            or self.immutable_shard_read_pass_count != 2
        ):
            raise ValueError("affine-shell source replay axis does not close")
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", ids)
        required_true = (
            self.structured_direction_custody,
            self.outside_zero_on_shell,
            self.no_dense_basis_bank,
            self.zero_additional_model_forwards,
            self.zero_additional_vjps,
            self.array_free,
            self.source_projection_authority,
            self.exact_preoutcome_postsource_custody_bound,
        )
        required_false = (
            self.context_route_bank_retained,
            self.route_scale_rows_d2h,
            self.population_moment_authority,
            self.registered_fd_replacement_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.algorithm != LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_ALGORITHM
            or self.authority_scope != LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_AUTHORITY
            or self.schema != LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_SCHEMA
        ):
            raise PermissionError("affine-shell shard replay crossed authority")
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"_issuer", "_receipt_hash"}
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_PREFIX + "affine-shell-receipt-"
        )
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("affine-shell shard replay receipt was mutated")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("affine-shell shard replay receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellShardReplayResult:
    source_bound_replay: V1LiteralJTwoPassDFixedShardReplayResult = field(
        repr=False,
        compare=False,
    )
    receipt: V1LiteralJAffineShellShardReplayReceipt

    def __post_init__(self) -> None:
        if (
            type(self.source_bound_replay)
            is not V1LiteralJTwoPassDFixedShardReplayResult
            or type(self.receipt) is not V1LiteralJAffineShellShardReplayReceipt
        ):
            raise TypeError("affine-shell shard replay result requires exact types")
        source = self.source_bound_replay
        self.receipt.assert_integrity()
        if (
            source.receipt.receipt_hash
            != self.receipt.source_bound_replay_receipt_hash
            or source.registration.registration_hash != self.receipt.registration_hash
            or source.projection.receipt.receipt_hash
            != self.receipt.projection_receipt_hash
            or source.projection.telemetry.telemetry_hash
            != self.receipt.projection_telemetry_hash
            or source.receipt.literal_j_tile_source_ledger_hash
            != self.receipt.literal_j_tile_source_ledger_hash
        ):
            raise ValueError("affine-shell replay wrapper changed source identity")

    @property
    def registration(self) -> V1LiteralJTwoPassDFixedProjectionRegistration:
        return self.source_bound_replay.registration

    @property
    def projection(self) -> V1LiteralJTwoPassDFixedProjectionExecution:
        return self.source_bound_replay.projection

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live affine-shell shard replay results are nonserializable")


def _execute_registered_sources_with_live_block_support_lease(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    block_support_receipt: V1SignedWorkBlockSupportSourceReceipt,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis | None,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign | None,
    coefficient_lease: V1SignedWorkBlockSupportCoefficientLease,
    structured_direction_custody: (
        V1LiteralJAffineShellDirectionDeviceHandle | None
    ) = None,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    """Run the immutable two-pass kernel from one pre-outcome registration.

    Both the legacy multi-segment owner and the V7 D1-only owner enter here.
    The latter is intentionally the only path that can present one packet and
    an empty residual axis; no sampled-D plan is synthesized to impersonate a
    multi-segment anchor.
    """

    if torch is None:
        raise RuntimeError("literal-J shard replay requires PyTorch")
    exact_types = (
        (registration, V1LiteralJTwoPassDFixedProjectionRegistration),
        (block_support_receipt, V1SignedWorkBlockSupportSourceReceipt),
        (conditioned_d1, V1LiteralJConditionedD1ShardReplaySource),
        (coefficient_lease, V1SignedWorkBlockSupportCoefficientLease),
    )
    if any(type(value) is not expected for value, expected in exact_types) or type(
        residual_segments
    ) not in {tuple, list} or any(
        type(item) is not V1LiteralJResidualShardReplaySource
        for item in residual_segments
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "registered source replay requires exact source and direction custody",
        )
    legacy_direction = (
        type(descriptor_axis) is V1LiteralJPrivatePrimalDirectionDescriptorAxis
        and structured_direction_custody is None
    )
    structured_direction = (
        descriptor_axis is None
        and directions is None
        and type(structured_direction_custody)
        is V1LiteralJAffineShellDirectionDeviceHandle
    )
    if legacy_direction == structured_direction:
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "source replay requires exactly one legacy or structured direction owner",
        )
    registration.assert_integrity()
    block_support_receipt.assert_integrity()
    residual = tuple(residual_segments)
    route_count = registration.route_count
    if (
        coefficient_lease.tensor is None
        or tuple(int(item) for item in coefficient_lease.tensor.shape)
        != (route_count,)
        or block_support_receipt.receipt_hash
        != registration.block_support_source_receipt_hash
        or block_support_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or block_support_receipt.source_registration_hash
        != registration.structural_source_registration_hash
        or block_support_receipt.model_identity_hash != registration.model_identity_hash
        or block_support_receipt.chart_identity_hash != registration.chart_identity_hash
        or block_support_receipt.current_mask_hash != registration.current_mask_hash
        or block_support_receipt.proposal_generation
        != registration.proposal_generation
        or block_support_receipt.route_axis_hash != registration.route_axis_hash
        or block_support_receipt.route_id_axis_hash != registration.route_id_axis_hash
        or block_support_receipt.route_count != route_count
        or block_support_receipt.context_count
        != registration.structural_context_count
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "live D_fixed coefficient source left the registered structural measure",
        )
    if (
        len(registration.packets) != len(residual) + 1
        or registration.segment_roles
        != ("conditioned_D1",)
        + tuple(item.segment_role for item in registration.packets[1:])
        or any(
            not item.segment_role.startswith("residual_fit:")
            for item in registration.packets[1:]
        )
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.PARTIAL_REPLAY,
            "registered packet axis is not exactly D1 followed by supplied residuals",
        )
    if legacy_direction:
        assert descriptor_axis is not None
        if (
            descriptor_axis.route_axis_hash != registration.route_axis_hash
            or descriptor_axis.route_count != route_count
            or descriptor_axis.direction_ids != registration.direction_ids
            or descriptor_axis.descriptor_axis_hash
            != registration.direction_source_hash
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
                "KKT private-primal direction lineage differs from the replay registration",
            )
        direction_axis_source_hash = descriptor_axis.descriptor_axis_hash
    else:
        assert structured_direction_custody is not None
        structured_receipt = structured_direction_custody.receipt
        if (
            structured_receipt.route_axis_hash != registration.route_axis_hash
            or structured_receipt.route_id_axis_hash != registration.route_id_axis_hash
            or structured_receipt.route_count != route_count
            or structured_receipt.direction_ids != registration.direction_ids
            or structured_receipt.direction_source_hash
            != registration.direction_source_hash
        ):
            _refuse(
                V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
                "affine-shell direction lineage differs from the replay registration",
            )
        direction_axis_source_hash = structured_receipt.direction_source_hash

    packet_sources = (
        _conditioned_packet_source(
            registration=registration,
            packet=registration.packets[0],
            source=conditioned_d1,
        ),
    ) + tuple(
        _residual_packet_source(
            registration=registration,
            packet=packet,
            source=source,
        )
        for packet, source in zip(registration.packets[1:], residual, strict=True)
    )
    accumulator = V1LiteralJTwoPassDFixedProjectionCUDAAccumulator(
        registration=registration,
        directions=directions,
        direction_source_hash=direction_axis_source_hash,
        design=design,
        structured_direction_custody=structured_direction_custody,
    )
    route_tile_size = accumulator.design.route_tile_size
    source_read_count = 0
    source_read_bytes = 0
    submitted_tiles = 0
    h2d_bytes = 0
    max_host_staging = 0
    max_device_tile = 0
    tile_source_hashes: list[str] = []

    for pass_index in (1, 2):
        if pass_index == 2:
            accumulator.rewind_for_pass2()
        for packet, packet_source in zip(
            registration.packets,
            packet_sources,
            strict=True,
        ):
            begin = (
                accumulator.begin_pass1_packet
                if pass_index == 1
                else accumulator.begin_pass2_packet
            )
            consume = (
                accumulator.consume_pass1_tile
                if pass_index == 1
                else accumulator.consume_pass2_tile
            )
            end = (
                accumulator.end_pass1_packet
                if pass_index == 1
                else accumulator.end_pass2_packet
            )
            begin(**_packet_begin_kwargs(registration, packet))
            iterators = tuple(
                item.shard.iter_route_windows(verify=True)
                for item in packet_source.slices
            )
            observed_windows = 0
            expected_start = 0
            for source_windows in zip(*iterators, strict=True):
                window_axis = tuple((item[0], item[1]) for item in source_windows)
                if (
                    observed_windows >= len(packet_source.route_windows)
                    or len(set(window_axis)) != 1
                    or window_axis[0] != packet_source.route_windows[observed_windows]
                    or window_axis[0][0] != expected_start
                ):
                    _refuse(
                        V1LiteralJTwoPassDFixedShardReplayRefusalCode.ORDER_MISMATCH,
                        "immutable shards departed from the registered route-window order",
                    )
                window_start, window_stop = window_axis[0]
                expected_start = window_stop
                observed_windows += 1
                raw_bytes = sum(int(item[2].nbytes) for item in source_windows)
                source_read_count += len(source_windows)
                source_read_bytes += raw_bytes
                width = window_stop - window_start
                for local_start in range(0, width, route_tile_size):
                    local_stop = min(width, local_start + route_tile_size)
                    start = window_start + local_start
                    stop = window_start + local_stop
                    host_tile = np.empty(
                        (packet.context_count, stop - start),
                        dtype=np.float64,
                    )
                    written = np.zeros(packet.context_count, dtype=np.bool_)
                    for shard_slice, source_window in zip(
                        packet_source.slices,
                        source_windows,
                        strict=True,
                    ):
                        selected = np.asarray(
                            source_window[2][
                                shard_slice.source_rows,
                                local_start:local_stop,
                            ],
                            dtype=np.float64,
                            order="C",
                        )
                        host_tile[np.asarray(shard_slice.destination_rows)] = selected
                        written[np.asarray(shard_slice.destination_rows)] = True
                    if not bool(written.all()) or not bool(
                        np.isfinite(host_tile).all()
                    ):
                        _refuse(
                            V1LiteralJTwoPassDFixedShardReplayRefusalCode.NONFINITE_ROW,
                            "source tile is partial or contains a nonfinite literal-J value",
                        )
                    max_host_staging = max(
                        max_host_staging,
                        raw_bytes + int(host_tile.nbytes) + int(written.nbytes),
                    )
                    device_tile = torch.from_numpy(host_tile).to(
                        device=accumulator.device,
                        dtype=torch.float64,
                    )
                    device_tile = device_tile.contiguous()
                    if accumulator.device.type == "cuda":
                        h2d_bytes += int(host_tile.nbytes)
                    max_device_tile = max(max_device_tile, int(host_tile.nbytes))
                    tile_source_hash = _source_tile_hash(
                        registration=registration,
                        packet=packet,
                        source=packet_source,
                        parent_window=(window_start, window_stop),
                        route_window=(start, stop),
                    )
                    tile_source_hashes.append(tile_source_hash)
                    consume(
                        start,
                        stop,
                        device_tile,
                        literal_j_tile_source_hash=tile_source_hash,
                    )
                    submitted_tiles += 1
                    del device_tile, host_tile, written
                del source_windows
            if (
                observed_windows != len(packet_source.route_windows)
                or expected_start != registration.route_count
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedShardReplayRefusalCode.PARTIAL_REPLAY,
                    "source replay did not consume the complete registered route cover",
                )
            end()

    projection = accumulator.finalize()
    half = len(tile_source_hashes) // 2
    tile_ledger_hash = content_hash(
        {
            "schema": "literal_j_D_fixed_source_tile_ledger_v2",
            "registration_hash": registration.registration_hash,
            "packet_source_axis_hashes": tuple(
                item.source_axis_hash for item in packet_sources
            ),
            "pass1_tile_source_hashes": tuple(tile_source_hashes[:half]),
            "pass2_tile_source_hashes": tuple(tile_source_hashes[half:]),
            "pass_values_identical": (
                tile_source_hashes[:half] == tile_source_hashes[half:]
            ),
        },
        prefix=_PREFIX + "tile-ledger-",
    )
    receipt = V1LiteralJTwoPassDFixedShardReplayReceipt(
        registration_hash=registration.registration_hash,
        projection_receipt_hash=projection.receipt.receipt_hash,
        projection_telemetry_hash=projection.telemetry.telemetry_hash,
        replay_registration_receipt_hash=(
            registration.replay_registration_receipt_hash
        ),
        numeric_source_receipt_hash=registration.numeric_source_receipt_hash,
        live_block_support_source_receipt_hash=block_support_receipt.receipt_hash,
        conditioned_d1_replay_receipt_hash=(conditioned_d1.replay_receipt.receipt_hash),
        residual_replay_receipt_hashes=tuple(
            item.replay_receipt.receipt_hash for item in residual
        ),
        packet_source_axis_hashes=tuple(
            item.source_axis_hash for item in packet_sources
        ),
        direction_descriptor_axis_hash=direction_axis_source_hash,
        same_capture_and_vjp_execution_proof_hash=(
            registration.same_capture_and_vjp_execution_proof_hash
        ),
        literal_j_tile_source_ledger_hash=tile_ledger_hash,
        context_count=registration.structural_context_count,
        route_count=registration.route_count,
        packet_count=len(registration.packets),
        raw_shard_count=len(conditioned_d1.literal_j_shards) + len(residual),
        immutable_shard_read_pass_count=2,
        shard_route_window_read_count=source_read_count,
        accumulator_tile_submit_count=submitted_tiles,
        source_payload_read_bytes=source_read_bytes,
        host_to_device_tile_bytes=h2d_bytes,
        maximum_owned_host_staging_bytes=max_host_staging,
        maximum_device_tile_bytes=max_device_tile,
        production_cuda_execution=projection.telemetry.production_cuda_execution,
        private_cpu_reference_execution=(
            projection.telemetry.private_cpu_reference_execution
        ),
        _issuer=_ISSUER,
    )
    return V1LiteralJTwoPassDFixedShardReplayResult(
        registration=registration,
        projection=projection,
        receipt=receipt,
    )


def _execute_with_live_block_support_lease(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign | None,
    coefficient_lease: V1SignedWorkBlockSupportCoefficientLease,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    if torch is None:
        raise RuntimeError("literal-J shard replay requires PyTorch")
    route_count = _validate_numeric_and_plan(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
    )
    if (
        type(coefficient_lease) is not V1SignedWorkBlockSupportCoefficientLease
        or coefficient_lease.tensor is None
        or tuple(int(item) for item in coefficient_lease.tensor.shape) != (route_count,)
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "literal-J replay must hold the live D_fixed coefficient lease",
        )
    if (
        type(conditioned_d1) is not V1LiteralJConditionedD1ShardReplaySource
        or type(residual_segments) not in {tuple, list}
        or any(
            type(item) is not V1LiteralJResidualShardReplaySource
            for item in residual_segments
        )
        or type(descriptor_axis) is not V1LiteralJPrivatePrimalDirectionDescriptorAxis
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "source and direction axes must use exact typed custody",
        )
    residual = tuple(residual_segments)
    if len(residual) != len(replay_plan.physical_segment_registrations) - 1:
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.PARTIAL_REPLAY,
            "residual replay axis omits or adds a physical D_fixed segment",
        )
    expected_segment_bridge_hashes = (
        conditioned_d1.replay_receipt.compact_bridge_receipt_hash,
    ) + tuple(item.replay_receipt.bridge_receipt_hash for item in residual)
    if (
        numeric_source.receipt.segment_bridge_receipt_hashes
        != expected_segment_bridge_hashes
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "D1/residual bridge receipts do not match the sealed numeric-source axis",
        )
    if (
        descriptor_axis.route_axis_hash != replay_plan.receipt.route_axis_hash
        or descriptor_axis.route_count != route_count
        or len(descriptor_axis.direction_ids) != 2
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "KKT private-primal direction axis left the registered route geometry",
        )
    same_vjp_proof = _same_vjp_proof_hash(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
        conditioned_d1=conditioned_d1,
        residual_segments=residual,
    )
    registration = register_v1_literal_j_two_pass_d_fixed_projection(
        replay_plan=replay_plan,
        numeric_source_receipt=numeric_source.receipt,
        route_count=route_count,
        direction_ids=descriptor_axis.direction_ids,
        direction_source_hash=descriptor_axis.descriptor_axis_hash,
        same_capture_and_vjp_execution_proof_hash=same_vjp_proof,
    )
    return _execute_registered_sources_with_live_block_support_lease(
        registration=registration,
        block_support_receipt=numeric_source.source_handle.receipt,
        conditioned_d1=conditioned_d1,
        residual_segments=residual,
        descriptor_axis=descriptor_axis,
        directions=directions,
        design=design,
        coefficient_lease=coefficient_lease,
    )


def _execute(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign | None,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    from .modern_v1_sampled_block_support_numeric_source import (
        V1SampledBlockSupportNumericSourceResult,
    )

    if type(numeric_source) is not V1SampledBlockSupportNumericSourceResult:
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "source replay requires the exact live numeric-source result",
        )
    with numeric_source.source_handle.open_coefficient_lease() as coefficient_lease:
        return _execute_with_live_block_support_lease(
            replay_plan=replay_plan,
            numeric_source=numeric_source,
            conditioned_d1=conditioned_d1,
            residual_segments=residual_segments,
            descriptor_axis=descriptor_axis,
            directions=directions,
            design=design,
            coefficient_lease=coefficient_lease,
        )


def replay_v1_literal_j_two_pass_d_fixed_from_shards(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    direction_custody: (
        V1LiteralJPrivatePrimalDirectionCUDAHandle
        | V1LiteralJPrivatePrimalDirectionCUDALease
    ),
    design: V1LiteralJTwoPassDFixedProjectionDesign | None = None,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    """Execute production replay from typed KKT direction custody only."""

    axis = _direction_axis(direction_custody)
    if type(direction_custody) is V1LiteralJPrivatePrimalDirectionCUDAHandle:
        with direction_custody.borrow_for_literal_j_projection() as lease:
            return _execute(
                replay_plan=replay_plan,
                numeric_source=numeric_source,
                conditioned_d1=conditioned_d1,
                residual_segments=residual_segments,
                descriptor_axis=axis,
                directions=lease.directions_for_literal_j_projection(),
                design=design,
            )
    return _execute(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
        conditioned_d1=conditioned_d1,
        residual_segments=residual_segments,
        descriptor_axis=axis,
        directions=direction_custody.directions_for_literal_j_projection(),
        design=design,
    )


def replay_v1_literal_j_two_pass_d_fixed_affine_shell_from_shards(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    direction_custody: V1LiteralJAffineShellDirectionDeviceHandle,
    design: V1LiteralJTwoPassDFixedProjectionDesign | None = None,
) -> V1LiteralJAffineShellShardReplayResult:
    """Replay immutable D-fixed shards for ``(z_outside, e_B)``.

    Registration and all parent checks finish before the first shard value is
    read.  The source adapter then performs the same verified two reads as the
    legacy path while the numerical accumulator gathers shell basis columns
    directly from each literal-J tile.
    """

    from .modern_v1_sampled_block_support_numeric_source import (
        V1SampledBlockSupportNumericSourceResult,
    )
    from .modern_v1_sampled_block_support_replay_registration import (
        V1SampledBlockSupportReplayRegistrationPlan,
    )

    if (
        type(replay_plan) is not V1SampledBlockSupportReplayRegistrationPlan
        or type(numeric_source) is not V1SampledBlockSupportNumericSourceResult
        or type(conditioned_d1) is not V1LiteralJConditionedD1ShardReplaySource
        or type(residual_segments) not in {tuple, list}
        or any(
            type(item) is not V1LiteralJResidualShardReplaySource
            for item in residual_segments
        )
        or type(direction_custody)
        is not V1LiteralJAffineShellDirectionDeviceHandle
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "structured shard replay requires exact plan, source, attachment, and direction custody",
        )
    # This check intentionally precedes plan/numeric validation and every raw
    # shard iterator.  The compatibility direction API may exercise the local
    # numerical owner, but a caller-authored preoutcome hash or outside-source
    # string can never open source-bound structured replay.
    direction = direction_custody.receipt
    plan = replay_plan.receipt
    if (
        direction.exact_preoutcome_postsource_custody_bound is not True
        or direction.outside_direction_exact_action_custody is not True
        or direction.preoutcome_source_registration_hash is None
        or direction.source_registration_hash is None
        or direction.exact_postsource_custody_hash is None
        or direction.literal_j_cell_hash is None
        or direction.literal_j_cell_scope_hash is None
        or direction.literal_j_stream_binding_hash is None
        or direction.preoutcome_fanout_receipt_hash is None
        or direction.preoutcome_destination_binding_hash is None
        or direction.preoutcome_binding_axis_hash is None
        or direction.preoutcome_physical_action_axis_hash is None
        or direction.postsource_receipt_hash is None
        or direction.projection_replay_registration_receipt_hash is None
        or direction.projection_replay_registration_receipt_hash
        != plan.receipt_hash
        or direction.source_registration_hash != plan.structural_registration_hash
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "structured replay requires exact pre-outcome, post-D_W2 source, action, and J-cell custody",
        )
    route_count = _validate_numeric_and_plan(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
    )
    residual = tuple(residual_segments)
    if len(residual) != len(replay_plan.physical_segment_registrations) - 1:
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.PARTIAL_REPLAY,
            "structured residual replay axis omits or adds a D-fixed segment",
        )
    expected_segment_bridge_hashes = (
        conditioned_d1.replay_receipt.compact_bridge_receipt_hash,
    ) + tuple(item.replay_receipt.bridge_receipt_hash for item in residual)
    if numeric_source.receipt.segment_bridge_receipt_hashes != (
        expected_segment_bridge_hashes
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "structured source attachments differ from the numeric-source segment axis",
        )

    common_source_receipts = (
        conditioned_d1.compact_source_receipt,
    ) + tuple(item.compact_source_receipt for item in residual)
    if (
        direction.model_identity_hash != plan.model_identity_hash
        or direction.chart_identity_hash != plan.chart_identity_hash
        or direction.current_mask_hash != plan.current_mask_hash
        or direction.proposal_generation != plan.proposal_generation
        or direction.route_axis_hash != plan.route_axis_hash
        or direction.route_id_axis_hash != plan.route_id_axis_hash
        or direction.route_count != route_count
        or direction.registered_d_prefix_hash
        != plan.conditioned_d1_registered_prefix_hash
        or direction.registered_d_w2_prefix_hash
        != plan.residual_registered_dw2_prefix_hash
        or direction.shell_freeze_receipt_hash != plan.shell_freeze_receipt_hash
        or direction.shell_indices != plan.shell_indices
        or direction.shell_route_ids != plan.shell_route_ids
        or direction.literal_j_stream_source_hash
        != plan.composite_literal_j_stream_source_hash
        or direction.same_capture_vjp_source_hash
        != plan.composite_same_capture_vjp_source_hash
        or direction.source_registration_hash != plan.structural_registration_hash
        or direction.alpha_lineage_hash != plan.alpha_lineage_hash
        or any(
            source.shell_freeze_receipt_hash != plan.shell_freeze_receipt_hash
            or source.shell_indices != plan.shell_indices
            or source.shell_route_ids != plan.shell_route_ids
            or source.alpha_lineage_hash != plan.alpha_lineage_hash
            for source in common_source_receipts
        )
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.STALE_IDENTITY,
            "affine-shell direction or compact source left the plan-owned shell/alpha lineage",
        )
    same_vjp_proof = _same_vjp_proof_hash(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
        conditioned_d1=conditioned_d1,
        residual_segments=residual,
    )
    registration = register_v1_literal_j_two_pass_d_fixed_projection(
        replay_plan=replay_plan,
        numeric_source_receipt=numeric_source.receipt,
        route_count=route_count,
        direction_ids=direction.direction_ids,
        direction_source_hash=direction.direction_source_hash,
        same_capture_and_vjp_execution_proof_hash=same_vjp_proof,
    )
    with numeric_source.source_handle.open_coefficient_lease() as coefficient_lease:
        source_bound = _execute_registered_sources_with_live_block_support_lease(
            registration=registration,
            block_support_receipt=numeric_source.source_handle.receipt,
            conditioned_d1=conditioned_d1,
            residual_segments=residual,
            descriptor_axis=None,
            directions=None,
            design=design,
            coefficient_lease=coefficient_lease,
            structured_direction_custody=direction_custody,
        )
    restart_replay_hash = content_hash(
        {
            "schema": "literal_J_affine_shell_shard_restart_replay_v1",
            "replay_registration_receipt_hash": plan.receipt_hash,
            "numeric_source_receipt_hash": numeric_source.receipt.receipt_hash,
            "direction_receipt_hash": direction.receipt_hash,
            "preoutcome_registration_hash": (
                direction.preoutcome_source_registration_hash
            ),
            "source_registration_hash": direction.source_registration_hash,
            "postsource_receipt_hash": direction.postsource_receipt_hash,
            "exact_postsource_custody_hash": direction.exact_postsource_custody_hash,
            "literal_j_cell_hash": direction.literal_j_cell_hash,
            "literal_j_cell_scope_hash": direction.literal_j_cell_scope_hash,
            "literal_j_stream_binding_hash": direction.literal_j_stream_binding_hash,
            "preoutcome_fanout_receipt_hash": (
                direction.preoutcome_fanout_receipt_hash
            ),
            "preoutcome_destination_binding_hash": (
                direction.preoutcome_destination_binding_hash
            ),
            "preoutcome_binding_axis_hash": direction.preoutcome_binding_axis_hash,
            "preoutcome_physical_action_axis_hash": (
                direction.preoutcome_physical_action_axis_hash
            ),
            "projection_replay_registration_receipt_hash": (
                direction.projection_replay_registration_receipt_hash
            ),
            "registration_hash": registration.registration_hash,
            "source_bound_replay_receipt_hash": source_bound.receipt.receipt_hash,
            "literal_j_tile_source_ledger_hash": (
                source_bound.receipt.literal_j_tile_source_ledger_hash
            ),
            "zero_additional_vjps": True,
            "no_dense_basis_bank": True,
        },
        prefix=_PREFIX + "affine-shell-restart-",
    )
    receipt = V1LiteralJAffineShellShardReplayReceipt(
        source_bound_replay_receipt_hash=source_bound.receipt.receipt_hash,
        registration_hash=registration.registration_hash,
        projection_receipt_hash=source_bound.projection.receipt.receipt_hash,
        projection_telemetry_hash=(
            source_bound.projection.telemetry.telemetry_hash
        ),
        direction_receipt_hash=direction.receipt_hash,
        direction_source_hash=direction.direction_source_hash,
        direction_lineage_hash=direction.lineage_hash,
        preoutcome_registration_hash=direction.preoutcome_source_registration_hash,
        source_registration_hash=direction.source_registration_hash,
        postsource_receipt_hash=direction.postsource_receipt_hash,
        exact_postsource_custody_hash=direction.exact_postsource_custody_hash,
        literal_j_cell_hash=direction.literal_j_cell_hash,
        literal_j_cell_scope_hash=direction.literal_j_cell_scope_hash,
        literal_j_stream_binding_hash=direction.literal_j_stream_binding_hash,
        preoutcome_fanout_receipt_hash=direction.preoutcome_fanout_receipt_hash,
        preoutcome_destination_binding_hash=(
            direction.preoutcome_destination_binding_hash
        ),
        preoutcome_binding_axis_hash=direction.preoutcome_binding_axis_hash,
        preoutcome_physical_action_axis_hash=(
            direction.preoutcome_physical_action_axis_hash
        ),
        projection_replay_registration_receipt_hash=(
            direction.projection_replay_registration_receipt_hash
        ),
        shell_freeze_receipt_hash=direction.shell_freeze_receipt_hash,
        shell_indices=direction.shell_indices,
        shell_route_ids=direction.shell_route_ids,
        alpha_lineage_hash=direction.alpha_lineage_hash,
        replay_registration_receipt_hash=plan.receipt_hash,
        numeric_source_receipt_hash=numeric_source.receipt.receipt_hash,
        literal_j_tile_source_ledger_hash=(
            source_bound.receipt.literal_j_tile_source_ledger_hash
        ),
        restart_replay_hash=restart_replay_hash,
        route_count=route_count,
        shell_size=len(direction.shell_indices),
        immutable_shard_read_pass_count=2,
        _issuer=_ISSUER,
    )
    return V1LiteralJAffineShellShardReplayResult(
        source_bound_replay=source_bound,
        receipt=receipt,
    )


def _replay_v1_literal_j_two_pass_d_fixed_from_shards_cpu_reference(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source: V1SampledBlockSupportNumericSourceResult,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    residual_segments: Sequence[V1LiteralJResidualShardReplaySource],
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    """Small-domain test oracle; deliberately private and CPU-only."""

    if (
        type(design) is not V1LiteralJTwoPassDFixedProjectionDesign
        or design.production_cuda_required
        or not design.allow_private_cpu_reference
        or torch is None
        or type(directions) is not torch.Tensor
        or directions.device.type != "cpu"
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.AUTHORITY_VIOLATION,
            "private CPU replay cannot impersonate production direction custody",
        )
    return _execute(
        replay_plan=replay_plan,
        numeric_source=numeric_source,
        conditioned_d1=conditioned_d1,
        residual_segments=residual_segments,
        descriptor_axis=descriptor_axis,
        directions=directions,
        design=design,
    )


def _execute_d1_only(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    block_support_source: V1SignedWorkBlockSupportSourceDeviceHandle,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign | None,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    if (
        type(registration) is not V1LiteralJTwoPassDFixedProjectionRegistration
        or type(block_support_source)
        is not V1SignedWorkBlockSupportSourceDeviceHandle
        or type(conditioned_d1) is not V1LiteralJConditionedD1ShardReplaySource
        or type(descriptor_axis)
        is not V1LiteralJPrivatePrimalDirectionDescriptorAxis
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.INVALID_TYPE,
            "D1-only replay requires exact registration and live source types",
        )
    block_support_source.assert_live()
    receipt = block_support_source.receipt
    registration.assert_integrity()
    if (
        registration.segment_roles != ("conditioned_D1",)
        or registration.segment_context_counts != (
            registration.structural_context_count,
        )
        or len(registration.packets) != 1
        or receipt.segment_registration_count != 1
        or receipt.multi_segment_numerical_source is not False
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.AUTHORITY_VIOLATION,
            "D1-only replay cannot accept residual-fit or multi-segment custody",
        )
    with block_support_source.open_coefficient_lease() as coefficient_lease:
        return _execute_registered_sources_with_live_block_support_lease(
            registration=registration,
            block_support_receipt=receipt,
            conditioned_d1=conditioned_d1,
            residual_segments=(),
            descriptor_axis=descriptor_axis,
            directions=directions,
            design=design,
            coefficient_lease=coefficient_lease,
        )


def replay_v1_literal_j_two_pass_d1_only_from_shards(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    block_support_source: V1SignedWorkBlockSupportSourceDeviceHandle,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    direction_custody: (
        V1LiteralJPrivatePrimalDirectionCUDAHandle
        | V1LiteralJPrivatePrimalDirectionCUDALease
    ),
    design: V1LiteralJTwoPassDFixedProjectionDesign | None = None,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    """Replay the source-issued D1-only anchor from typed KKT custody.

    This grants projection custody only.  In particular it does not fabricate
    a residual-fit segment or admit a fresh certification ``D_W2`` packet into
    the anchor history.
    """

    axis = _direction_axis(direction_custody)
    if type(direction_custody) is V1LiteralJPrivatePrimalDirectionCUDAHandle:
        with direction_custody.borrow_for_literal_j_projection() as lease:
            return _execute_d1_only(
                registration=registration,
                block_support_source=block_support_source,
                conditioned_d1=conditioned_d1,
                descriptor_axis=axis,
                directions=lease.directions_for_literal_j_projection(),
                design=design,
            )
    return _execute_d1_only(
        registration=registration,
        block_support_source=block_support_source,
        conditioned_d1=conditioned_d1,
        descriptor_axis=axis,
        directions=direction_custody.directions_for_literal_j_projection(),
        design=design,
    )


def _replay_v1_literal_j_two_pass_d1_only_from_shards_cpu_reference(
    *,
    registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    block_support_source: V1SignedWorkBlockSupportSourceDeviceHandle,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    design: V1LiteralJTwoPassDFixedProjectionDesign,
) -> V1LiteralJTwoPassDFixedShardReplayResult:
    """Small-domain D1-only oracle; never a production fallback."""

    if (
        type(design) is not V1LiteralJTwoPassDFixedProjectionDesign
        or design.production_cuda_required
        or not design.allow_private_cpu_reference
        or torch is None
        or type(directions) is not torch.Tensor
        or directions.device.type != "cpu"
    ):
        _refuse(
            V1LiteralJTwoPassDFixedShardReplayRefusalCode.AUTHORITY_VIOLATION,
            "private CPU D1-only replay cannot impersonate production custody",
        )
    return _execute_d1_only(
        registration=registration,
        block_support_source=block_support_source,
        conditioned_d1=conditioned_d1,
        descriptor_axis=descriptor_axis,
        directions=directions,
        design=design,
    )


__all__ = [
    "LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_ALGORITHM",
    "LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_AUTHORITY",
    "LITERAL_J_AFFINE_SHELL_SHARD_REPLAY_SCHEMA",
    "LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_ALGORITHM",
    "LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_AUTHORITY",
    "LITERAL_J_TWO_PASS_D_FIXED_SHARD_REPLAY_SCHEMA",
    "V1LiteralJConditionedD1ShardReplaySource",
    "V1LiteralJAffineShellShardReplayReceipt",
    "V1LiteralJAffineShellShardReplayResult",
    "V1LiteralJResidualShardReplaySource",
    "V1LiteralJTwoPassDFixedShardReplayReceipt",
    "V1LiteralJTwoPassDFixedShardReplayRefusal",
    "V1LiteralJTwoPassDFixedShardReplayRefusalCode",
    "V1LiteralJTwoPassDFixedShardReplayResult",
    "replay_v1_literal_j_two_pass_d1_only_from_shards",
    "replay_v1_literal_j_two_pass_d_fixed_affine_shell_from_shards",
    "replay_v1_literal_j_two_pass_d_fixed_from_shards",
]
