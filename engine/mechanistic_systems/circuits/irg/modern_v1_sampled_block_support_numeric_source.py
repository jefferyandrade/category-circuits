"""Production-shaped owner for the exact sampled ``D_fixed`` support source.

The registration plan already separates one canonical structural ``D_fixed``
mean from its ordered physical replay owners.  This module closes the next
execution seam: it binds that plan to one preferred hard action, the exact
multi-segment block-support registration, its bounded numerical accumulator,
and the source-local replay fanout.

The owner exposes a deliberately narrow lifecycle::

    issue next segment sink -> accept its exact bridge result -> ... -> seal

Sealing returns the existing live block-support coefficient handle and one
array-free execution receipt.  It does not certify sampling, action stability,
Q, C, calibration, oracle loss, a commit, terminality, or Pareto superiority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn

from .contracts import content_hash
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_block_support_replay_fanout import (
    V1SampledBlockSupportReplayFanoutCoordinator,
    V1SampledBlockSupportReplayFanoutResult,
)
from .modern_v1_sampled_block_support_replay_registration import (
    V1SampledBlockSupportReplayRegistrationPlan,
)
from .modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
    V1SampledVariableShellSourceBridgeResult,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportCUDAExecutionDesign,
    V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator,
    V1SignedWorkBlockSupportMultiSegmentSourceRegistration,
    V1SignedWorkBlockSupportPreferredActionBinding,
    V1SignedWorkBlockSupportSourceDeviceHandle,
    V1SignedWorkBlockSupportSourceRole,
    register_v1_signed_work_block_support_multi_segment_source,
)


SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_SCHEMA: Final = (
    "partition_free_v1_sampled_block_support_numeric_source_receipt_v1"
)
SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_ALGORITHM: Final = (
    "exact_structural_D_fixed_mean_from_ordered_conditioned_D1_and_residual_"
    "same_source_replay_with_native_fiber_signed_plane_v1"
)
SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_AUTHORITY: Final = (
    "D_fixed_numerical_block_support_source_only_no_sampling_separation_action_"
    "Q_C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-block-support-numeric-source-"
_RECEIPT_ISSUER: Final = object()


class V1SampledBlockSupportNumericSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    INCOMPLETE = "incomplete"
    SOURCE_EXECUTION_FAILURE = "source_execution_failure"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledBlockSupportNumericSourceRefusal(RuntimeError):
    """Typed refusal at the exact D_fixed numerical-source boundary."""

    def __init__(
        self,
        code: V1SampledBlockSupportNumericSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledBlockSupportNumericSourceRefusalCode:
            raise TypeError("numeric-source refusal code has the wrong type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledBlockSupportNumericSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledBlockSupportNumericSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != len(set(result)):
        _refuse(
            V1SampledBlockSupportNumericSourceRefusalCode.AXIS_MISMATCH,
            f"{name} must be unique",
        )
    return result


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportNumericSourceReceipt:
    replay_registration_receipt_hash: str
    anchor_influencing_axis_receipt_hash: str
    full_population_compact_source_receipt_hash: str
    structural_source_registration_hash: str
    physical_source_registration_hashes: tuple[str, ...]
    multi_segment_registration_hash: str
    segment_registration_axis_hash: str
    numerical_source_chain_hash: str
    preferred_action_binding_hash: str
    preferred_action_hash: str
    preferred_resulting_mask_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    execution_design_hash: str
    hbm_forecast_hash: str
    replay_fanout_receipt_hash: str
    block_support_source_receipt_hash: str
    segment_bridge_receipt_hashes: tuple[str, ...]
    segment_sink_identity_hashes: tuple[str, ...]
    segment_count: int
    context_count: int
    packet_count: int
    paired_route_chunk_count: int
    structural_context_axis_hash: str
    structural_draw_axis_hash: str
    structural_measurement_axis_hash: str
    structural_capture_axis_hash: str
    structural_source_packet_axis_hash: str
    _issuer: object = field(default=None, repr=False, compare=False)
    exact_D_fixed_structural_mean_executed: bool = True
    conditioned_D1_then_residual_segment_order: bool = True
    source_local_chunk_binding_replayed: bool = True
    preferred_action_content_bound: bool = True
    native_fiber_signed_plane_constructed: bool = True
    replay_compact_sources_evicted: bool = True
    live_route_scale_block_support_handle_returned: bool = True
    array_free: bool = True
    raw_rows_present: bool = False
    tensor_payload_present: bool = False
    context_by_route_rows_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    sampling_separation_authority: bool = False
    action_authority: bool = False
    q_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_ALGORITHM
    authority_scope: str = SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_AUTHORITY
    schema: str = SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuer is not _RECEIPT_ISSUER:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "numeric-source receipts are issuer-only",
            )
        for name in (
            "replay_registration_receipt_hash",
            "anchor_influencing_axis_receipt_hash",
            "full_population_compact_source_receipt_hash",
            "structural_source_registration_hash",
            "multi_segment_registration_hash",
            "segment_registration_axis_hash",
            "numerical_source_chain_hash",
            "preferred_action_binding_hash",
            "preferred_action_hash",
            "preferred_resulting_mask_hash",
            "preferred_removed_route_index_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "execution_design_hash",
            "hbm_forecast_hash",
            "replay_fanout_receipt_hash",
            "block_support_source_receipt_hash",
            "structural_context_axis_hash",
            "structural_draw_axis_hash",
            "structural_measurement_axis_hash",
            "structural_capture_axis_hash",
            "structural_source_packet_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "physical_source_registration_hashes",
            "segment_bridge_receipt_hashes",
            "segment_sink_identity_hashes",
        ):
            object.__setattr__(
                self,
                name,
                _identity_axis(getattr(self, name), name=name),
            )
        for name in (
            "preferred_selected_count",
            "proposal_generation",
            "segment_count",
            "context_count",
            "packet_count",
            "paired_route_chunk_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.segment_count < 2
            or self.context_count <= 0
            or self.packet_count < self.segment_count
            or self.paired_route_chunk_count < self.packet_count
            or len(self.physical_source_registration_hashes) != self.segment_count
            or len(self.segment_bridge_receipt_hashes) != self.segment_count
            or len(self.segment_sink_identity_hashes) != self.segment_count
            or self.preferred_selected_count <= 0
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.AXIS_MISMATCH,
                "numeric-source segment, context, packet, or action extents do not close",
            )
        required_true = (
            self.exact_D_fixed_structural_mean_executed,
            self.conditioned_D1_then_residual_segment_order,
            self.source_local_chunk_binding_replayed,
            self.preferred_action_content_bound,
            self.native_fiber_signed_plane_constructed,
            self.replay_compact_sources_evicted,
            self.live_route_scale_block_support_handle_returned,
            self.array_free,
        )
        required_false = (
            self.raw_rows_present,
            self.tensor_payload_present,
            self.context_by_route_rows_retained,
            self.route_scale_device_to_host_transfer,
            self.sampling_separation_authority,
            self.action_authority,
            self.q_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in required_true + required_false)
            or any(value is not True for value in required_true)
            or any(value is not False for value in required_false)
            or self.algorithm != SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_ALGORITHM
            or self.authority_scope != SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_AUTHORITY
            or self.schema != SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_SCHEMA
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.AUTHORITY_VIOLATION,
                "numeric-source receipt crossed its source-only authority boundary",
            )
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
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "numeric-source receipt was mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "numeric-source receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledBlockSupportNumericSourceResult:
    source_handle: V1SignedWorkBlockSupportSourceDeviceHandle = field(
        repr=False,
        compare=False,
    )
    receipt: V1SampledBlockSupportNumericSourceReceipt

    def __post_init__(self) -> None:
        if type(self.source_handle) is not V1SignedWorkBlockSupportSourceDeviceHandle:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "numeric-source result requires the existing block-support handle",
            )
        if type(self.receipt) is not V1SampledBlockSupportNumericSourceReceipt:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "numeric-source result requires its exact receipt",
            )
        self.receipt.assert_integrity()
        self.source_handle.assert_live()
        if (
            self.source_handle.receipt.receipt_hash
            != self.receipt.block_support_source_receipt_hash
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "numeric-source result changed its live coefficient source",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D_fixed numerical-source results are nonserializable")


class V1SampledBlockSupportNumericSourceOwner:
    """Own the exact ordered replay lifecycle for one frozen preferred action."""

    __slots__ = (
        "plan",
        "preferred_action_binding",
        "native_fiber_axis_receipt",
        "execution_design",
        "multi_segment_registration",
        "_accumulator",
        "_fanout",
        "_plan_receipt_hash",
        "_physical_registration_hashes",
        "_active_sink",
        "_accepted_segments",
        "_sealed",
    )

    def __init__(
        self,
        *,
        replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
        preferred_action_binding: V1SignedWorkBlockSupportPreferredActionBinding,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        execution_design: V1SignedWorkBlockSupportCUDAExecutionDesign,
        declared_device: str,
        device_total_memory_bytes: int,
        device_free_memory_bytes: int,
        current_reserved_device_bytes: int,
    ) -> None:
        if (
            type(replay_plan) is not V1SampledBlockSupportReplayRegistrationPlan
            or type(preferred_action_binding)
            is not V1SignedWorkBlockSupportPreferredActionBinding
            or type(native_fiber_axis_receipt)
            is not V1NativeDestinationFiberAxisReceipt
            or type(execution_design) is not V1SignedWorkBlockSupportCUDAExecutionDesign
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "numeric-source owner requires exact plan, action, axis, and design types",
            )
        receipt = replay_plan.receipt
        receipt_hash = receipt.assert_integrity()
        structural = replay_plan.structural_registration
        physical = replay_plan.physical_segment_registrations
        physical_hashes = tuple(item.registration_hash for item in physical)
        if (
            structural.registration_hash != receipt.structural_registration_hash
            or physical_hashes != receipt.physical_registration_hashes
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "replay plan registrations changed after registration",
            )
        self._validate_plan_axes(replay_plan)
        native_fiber_axis_receipt.assert_integrity()
        if (
            native_fiber_axis_receipt.receipt_hash
            != receipt.native_fiber_axis_receipt_hash
            or native_fiber_axis_receipt.measure_hash
            != receipt.native_fiber_measure_hash
            or native_fiber_axis_receipt.route_axis_hash != receipt.route_axis_hash
            or native_fiber_axis_receipt.route_id_axis_hash
            != receipt.route_id_axis_hash
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "native-fiber route geometry differs from the replay plan",
            )
        preferred_action_binding.assert_live()
        if (
            preferred_action_binding.source_registration_hash
            != structural.registration_hash
            or preferred_action_binding.model_identity_hash
            != structural.model_identity_hash
            or preferred_action_binding.chart_identity_hash
            != structural.chart_identity_hash
            or preferred_action_binding.current_mask_hash
            != structural.current_mask_hash
            or preferred_action_binding.proposal_generation
            != structural.proposal_generation
            or preferred_action_binding.route_axis_hash != structural.route_axis_hash
            or preferred_action_binding.route_count
            != native_fiber_axis_receipt.route_count
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "preferred action is not bound to the structural replay registration",
            )
        context_group_id = content_hash(
            {
                "schema": "partition_free_v1_sampled_block_support_D_fixed_group_v1",
                "replay_registration_receipt_hash": receipt_hash,
                "preferred_action_binding_hash": preferred_action_binding.binding_hash,
            },
            prefix=_PREFIX + "D-fixed-group-",
        )
        partition_hash = content_hash(
            {
                "schema": "partition_free_v1_sampled_block_support_D_fixed_partition_v1",
                "segment_roles": receipt.segment_roles,
                "segment_context_counts": receipt.segment_context_counts,
                "conditioned_D1_then_residual": True,
            },
            prefix=_PREFIX + "D-fixed-partition-",
        )
        try:
            multi = register_v1_signed_work_block_support_multi_segment_source(
                structural_source_registration=structural,
                segment_source_registrations=physical,
                source_role=V1SignedWorkBlockSupportSourceRole.D_FIXED,
                context_group_id=context_group_id,
                block_partition_hash=partition_hash,
                block_ordinal=0,
                preferred_action_binding=preferred_action_binding,
                context_ids=structural.registered_context_ids,
                draw_ids=structural.registered_draw_ids,
                measurement_ids=structural.registered_measurement_ids,
                capture_ids=structural.registered_capture_ids,
                source_packet_hashes=structural.registered_source_packet_hashes,
            )
            accumulator = V1SignedWorkBlockSupportMultiSegmentSourceCUDAAccumulator(
                registration=multi,
                structural_source_registration=structural,
                segment_source_registrations=physical,
                native_fiber_axis_receipt=native_fiber_axis_receipt,
                preferred_action_binding=preferred_action_binding,
                design=execution_design,
                declared_device=declared_device,
                device_total_memory_bytes=device_total_memory_bytes,
                device_free_memory_bytes=device_free_memory_bytes,
                current_reserved_device_bytes=current_reserved_device_bytes,
            )
            fanout = V1SampledBlockSupportReplayFanoutCoordinator(
                accumulator=accumulator,
                segment_source_registrations=physical,
            )
        except Exception as error:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.SOURCE_EXECUTION_FAILURE,
                f"could not open the exact multi-segment numerical source: {error}",
            )
        self.plan = replay_plan
        self.preferred_action_binding = preferred_action_binding
        self.native_fiber_axis_receipt = native_fiber_axis_receipt
        self.execution_design = execution_design
        self.multi_segment_registration = multi
        self._accumulator = accumulator
        self._fanout = fanout
        self._plan_receipt_hash = receipt_hash
        self._physical_registration_hashes = physical_hashes
        self._active_sink: V1SampledVariableShellBoundChunkSinkBinding | None = None
        self._accepted_segments = 0
        self._sealed = False

    @staticmethod
    def _validate_plan_axes(
        replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    ) -> None:
        receipt = replay_plan.receipt
        structural = replay_plan.structural_registration
        physical = replay_plan.physical_segment_registrations
        axis_fields = (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
            "registered_source_packet_hashes",
        )
        for name in axis_fields:
            concatenated = tuple(
                value for segment in physical for value in getattr(segment, name)
            )
            if concatenated != tuple(getattr(structural, name)):
                _refuse(
                    V1SampledBlockSupportNumericSourceRefusalCode.AXIS_MISMATCH,
                    f"physical {name} no longer exactly partitions the structural axis",
                )
        if (
            tuple(item.context_count for item in physical)
            != receipt.segment_context_counts
            or sum(item.context_count for item in physical)
            != receipt.structural_context_count
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.AXIS_MISMATCH,
                "physical segment counts no longer close the structural D_fixed mean",
            )

    def _assert_live(self) -> None:
        if self._sealed:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.ORDER_MISMATCH,
                "numeric-source owner is already sealed",
            )
        if self.plan.receipt.assert_integrity() != self._plan_receipt_hash:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "replay registration receipt changed",
            )
        if (
            tuple(
                item.registration_hash
                for item in self.plan.physical_segment_registrations
            )
            != self._physical_registration_hashes
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "physical replay registration axis changed",
            )
        self._validate_plan_axes(self.plan)
        self.multi_segment_registration.assert_integrity()
        self.preferred_action_binding.assert_live()
        self.native_fiber_axis_receipt.assert_integrity()

    def issue_next_segment_sink(
        self,
    ) -> V1SampledVariableShellBoundChunkSinkBinding:
        self._assert_live()
        if self._active_sink is not None:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.ORDER_MISMATCH,
                "one physical replay segment sink is already active",
            )
        if self._accepted_segments >= len(self.plan.physical_segment_registrations):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.ORDER_MISMATCH,
                "every registered physical replay segment was already accepted",
            )
        try:
            sink = self._fanout.issue_next_segment_sink()
        except Exception as error:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.SOURCE_EXECUTION_FAILURE,
                f"could not issue the next exact replay sink: {error}",
            )
        self._active_sink = sink
        return sink

    def accept_segment_bridge(
        self,
        result: V1SampledVariableShellSourceBridgeResult,
    ) -> None:
        self._assert_live()
        if type(result) is not V1SampledVariableShellSourceBridgeResult:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "numeric-source owner accepts exact bridge results only",
            )
        sink = self._active_sink
        if sink is None:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.ORDER_MISMATCH,
                "no issued replay segment sink is awaiting acceptance",
            )
        expected = self.plan.physical_segment_registrations[self._accepted_segments]
        receipt = result.receipt
        receipt.assert_integrity()
        if (
            receipt.registration_hash != expected.registration_hash
            or receipt.bound_chunk_sink_count != 1
            or receipt.bound_chunk_sink_axis_hash != sink.singleton_sink_axis_hash
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "bridge result differs from the issued physical replay segment",
            )
        try:
            self._fanout.accept_segment_bridge(result)
        except Exception as error:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INCOMPLETE,
                f"physical replay segment is incomplete or inconsistent: {error}",
            )
        self._accepted_segments += 1
        self._active_sink = None

    def seal(self) -> V1SampledBlockSupportNumericSourceResult:
        self._assert_live()
        if self._active_sink is not None or self._accepted_segments != len(
            self.plan.physical_segment_registrations
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INCOMPLETE,
                "cannot seal before every exact replay segment is accepted",
            )
        try:
            fanout = self._fanout.seal()
        except Exception as error:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.SOURCE_EXECUTION_FAILURE,
                f"could not seal the block-support numerical source: {error}",
            )
        self._validate_sealed_fanout(fanout)
        source = fanout.source_handle
        source_receipt = source.receipt
        replay = self.plan.receipt
        structural = self.plan.structural_registration
        fanout_receipt = fanout.receipt
        receipt = V1SampledBlockSupportNumericSourceReceipt(
            replay_registration_receipt_hash=replay.receipt_hash,
            anchor_influencing_axis_receipt_hash=(
                replay.anchor_influencing_axis_receipt_hash
            ),
            full_population_compact_source_receipt_hash=(
                replay.full_population_compact_source_receipt_hash
            ),
            structural_source_registration_hash=structural.registration_hash,
            physical_source_registration_hashes=(replay.physical_registration_hashes),
            multi_segment_registration_hash=(
                self.multi_segment_registration.registration_hash
            ),
            segment_registration_axis_hash=(
                self.multi_segment_registration.segment_registration_axis_hash
            ),
            numerical_source_chain_hash=(
                self.multi_segment_registration.numerical_source_chain_hash
            ),
            preferred_action_binding_hash=(self.preferred_action_binding.binding_hash),
            preferred_action_hash=self.preferred_action_binding.preferred_action_hash,
            preferred_resulting_mask_hash=(
                self.preferred_action_binding.preferred_resulting_mask_hash
            ),
            preferred_removed_route_index_hash=(
                self.preferred_action_binding.preferred_removed_route_index_hash
            ),
            preferred_selected_count=(
                self.preferred_action_binding.preferred_selected_count
            ),
            model_identity_hash=structural.model_identity_hash,
            chart_identity_hash=structural.chart_identity_hash,
            current_mask_hash=structural.current_mask_hash,
            proposal_generation=structural.proposal_generation,
            route_axis_hash=structural.route_axis_hash,
            route_id_axis_hash=structural.route_id_axis_hash,
            native_fiber_axis_receipt_hash=(
                self.native_fiber_axis_receipt.receipt_hash
            ),
            native_fiber_measure_hash=self.native_fiber_axis_receipt.measure_hash,
            execution_design_hash=self.execution_design.design_hash,
            hbm_forecast_hash=source_receipt.hbm_forecast_hash,
            replay_fanout_receipt_hash=fanout_receipt.receipt_hash,
            block_support_source_receipt_hash=source_receipt.receipt_hash,
            segment_bridge_receipt_hashes=(
                fanout_receipt.segment_bridge_receipt_hashes
            ),
            segment_sink_identity_hashes=(fanout_receipt.segment_sink_identity_hashes),
            segment_count=fanout_receipt.segment_count,
            context_count=fanout_receipt.context_count,
            packet_count=fanout_receipt.packet_count,
            paired_route_chunk_count=fanout_receipt.paired_route_chunk_count,
            structural_context_axis_hash=source_receipt.context_axis_hash,
            structural_draw_axis_hash=source_receipt.draw_axis_hash,
            structural_measurement_axis_hash=(source_receipt.measurement_axis_hash),
            structural_capture_axis_hash=source_receipt.capture_axis_hash,
            structural_source_packet_axis_hash=(source_receipt.source_packet_axis_hash),
            _issuer=_RECEIPT_ISSUER,
        )
        self._sealed = True
        return V1SampledBlockSupportNumericSourceResult(
            source_handle=source,
            receipt=receipt,
        )

    def _validate_sealed_fanout(
        self,
        result: V1SampledBlockSupportReplayFanoutResult,
    ) -> None:
        if type(result) is not V1SampledBlockSupportReplayFanoutResult:
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.INVALID_TYPE,
                "fanout returned a noncanonical result",
            )
        fanout = result.receipt
        source = result.source_handle.receipt
        multi = self.multi_segment_registration
        if (
            fanout.multi_segment_registration_hash != multi.registration_hash
            or fanout.structural_source_registration_hash
            != multi.structural_source_registration_hash
            or fanout.multi_segment_registration_axis_hash
            != multi.segment_registration_axis_hash
            or fanout.numerical_source_chain_hash != multi.numerical_source_chain_hash
            or fanout.segment_source_registration_hashes
            != self._physical_registration_hashes
            or source.registration_hash != multi.registration_hash
            or source.source_registration_hash
            != multi.structural_source_registration_hash
            or source.structural_group_registration_hash
            != multi.structural_group_registration_hash
            or source.segment_registration_axis_hash
            != multi.segment_registration_axis_hash
            or source.numerical_source_chain_hash != multi.numerical_source_chain_hash
            or source.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
            or source.preferred_action_binding_hash
            != self.preferred_action_binding.binding_hash
            or source.retained_context_route_rows != 0
            or source.route_scale_d2h_values != 0
            or source.route_scale_device_tensor_content_hashes != 0
        ):
            _refuse(
                V1SampledBlockSupportNumericSourceRefusalCode.STALE_IDENTITY,
                "sealed fanout does not reproduce the registered D_fixed numerical source",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D_fixed numerical-source owners are nonserializable")


__all__ = [
    "SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_ALGORITHM",
    "SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_AUTHORITY",
    "SAMPLED_BLOCK_SUPPORT_NUMERIC_SOURCE_SCHEMA",
    "V1SampledBlockSupportNumericSourceOwner",
    "V1SampledBlockSupportNumericSourceReceipt",
    "V1SampledBlockSupportNumericSourceRefusal",
    "V1SampledBlockSupportNumericSourceRefusalCode",
    "V1SampledBlockSupportNumericSourceResult",
]
