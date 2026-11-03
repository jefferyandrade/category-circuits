"""Zero-VJP fan-out of one shared sampled-V7 A/J row group.

The destination axis is frozen before acquisition outcomes and is keyed by
exact packed physical actions.  Branch aliases may share one physical action,
but distinct packed actions can never collapse.  Durable A/J route windows
are read and transferred once, then every action-local compact reducer issues
its own typed chunk binding against those same tensors.

This seam produces compact proposal-input sources and append receipts only.
It does not certify D stability or open C/K/O, commit, terminal, or Pareto
authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import zip_longest
import math
from threading import RLock
from typing import Any, Final, NoReturn, Sequence

import numpy as np

try:  # pragma: no cover - CUDA availability is environment-specific.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from mechanistic_systems.circuits.irg.modern_v1_candidate_composition import (
    V1PackedRemovalAction,
)
from mechanistic_systems.circuits.irg.candidates import HardMask
from mechanistic_systems.circuits.irg.modern_v1_relaxed_kkt import (
    V1RelaxedKKTBranch,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7BranchActionPair,
    V1SampledV7PhysicalActionCustody,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyBranchSourceBinding,
    v1_sampled_v7_d_only_transport_packet_issuances,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_v7_shared_dw2_acquisition import (
    V1SampledV7SharedDW2AcquisitionPlan,
    V1SampledV7SharedDW2AcquisitionResult,
    v1_sampled_v7_nonempty_physical_roles,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_packet_registration import (
    _binary64_fraction_enclosure,
    _issue_measure_weights,
    v1_sampled_variable_shell_common_measure_unit_hash,
    v1_sampled_variable_shell_planned_prefix_hash,
)
from mechanistic_systems.circuits.irg.modern_v1_sampled_variable_shell_source_bridge import (
    V1SampledVariableShellBoundChunkSinkBinding,
    V1SampledVariableShellSourceBridgeReceipt,
    V1SampledVariableShellSourceBridgeResult,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendLineage,
    V1SignedWorkVariableShellSourceAppendOwner,
    V1SignedWorkVariableShellSourceAppendPacketRegistration,
    V1SignedWorkVariableShellSourceAppendSegmentedHandle,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellExecutionMode,
    V1SignedWorkVariableShellSourceCUDAAccumulator,
    V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    V1SignedWorkVariableShellSourceDeviceHandle,
    V1SignedWorkVariableShellSourceRegistration,
)
from mechanistic_systems.contracts.content_address import content_hash


SAMPLED_V7_SHARED_DW2_FANOUT_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_physical_action_fanout_plan_v1"
)
SAMPLED_V7_SHARED_DW2_FANOUT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_physical_action_fanout_receipt_v1"
)
SAMPLED_V7_SHARED_DW2_BLOCK_SUPPORT_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_shared_D_W2_block_support_destination_binding_v1"
)
SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY: Final = (
    "preoutcome_deduplicated_physical_action_compact_source_fanout_only_"
    "optional_exact_action_specific_residual_block_support_sink_no_D1_replay_"
    "zero_extra_provider_VJP_no_D_stability_C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID: Final = (
    "sampled-v7-deduplicated-physical-action-fanout-v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-shared-D-W2-fanout-"
_BRANCH_AXIS: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)
_ACTION_TOKEN: Final = object()
_DESTINATION_TOKEN: Final = object()
_PLAN_TOKEN: Final = object()
_RESULT_TOKEN: Final = object()
_PRIVATE_REFERENCE_TOKEN: Final = object()
_EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH: Final = content_hash(
    {
        "schema": "partition_free_v1_sampled_v7_block_support_sink_axis_v1",
        "destination_binding_hashes": (),
    },
    prefix=_PREFIX + "block-support-sink-axis-",
)


def _block_support_sink_axis_hash(
    values: Sequence[V1SampledV7SharedDW2BlockSupportDestinationBinding],
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_v7_block_support_sink_axis_v1",
            "destination_binding_hashes": tuple(value.binding_hash for value in values),
        },
        prefix=_PREFIX + "block-support-sink-axis-",
    )


class V1SampledV7SharedDW2FanoutRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    EMPTY_DESTINATION_AXIS = "empty_destination_axis"
    ACTION_COLLAPSE = "action_collapse"
    ALIAS_MISMATCH = "alias_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    DESTINATION_NOT_FROZEN = "destination_not_frozen"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    PARTIAL_COVER = "partial_cover"
    FAILURE_ATOMIC_ABORT = "failure_atomic_abort"
    FAILURE_ATOMIC_CLEANUP_INCOMPLETE = "failure_atomic_cleanup_incomplete"
    REPLAY = "replay"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7SharedDW2FanoutRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7SharedDW2FanoutRefusalCode,
        message: str,
        *,
        cleanup_receipt: object | None = None,
    ) -> None:
        if type(code) is not V1SampledV7SharedDW2FanoutRefusalCode:
            raise TypeError("shared fanout refusal code must be exact")
        self.code = code
        self.cleanup_receipt = cleanup_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SampledV7SharedDW2FanoutRefusalCode, message: str) -> NoReturn:
    raise V1SampledV7SharedDW2FanoutRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            f"{name} must be one nonempty identity",
        )
    return value


def _logical_transport_packets_and_schedule(
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    """Replay the hardware transport axis beneath one registered D look."""

    transport_issuances = v1_sampled_v7_d_only_transport_packet_issuances(
        acquisition_plan._packet_issuance
    )
    packets = tuple(value._packet for value in transport_issuances)
    scheduled = tuple(
        acquisition_plan._owner.schedule.packet_for(packet) for packet in packets
    )
    if (
        not packets
        or tuple(packet.receipt_hash for packet in packets)
        != acquisition_plan.transport_packet_hashes
        or tuple(item.packet_schedule_hash for item in scheduled)
        != acquisition_plan.transport_schedule_packet_hashes
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "fanout transport packet axis no longer replays the frozen logical look",
        )
    return packets, scheduled


def _logical_transport_axes(
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
) -> tuple[
    tuple[object, ...],
    tuple[object, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, int],
]:
    """Return logical-look axes plus physical call slices in scorer order."""

    packets, scheduled_packets = _logical_transport_packets_and_schedule(
        acquisition_plan
    )
    context_axis: list[str] = []
    draw_axis: list[str] = []
    measurement_axis: list[str] = []
    capture_axis: list[str] = []
    source_packet_axis: list[str] = []
    stratum_axis: list[str] = []
    physical_axes: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    target_count = 0
    complement_count = 0
    for packet, scheduled in zip(packets, scheduled_packets, strict=True):
        split = len(packet.target_context_ids)
        target_count += split
        complement_count += len(packet.complement_context_ids)
        slices = {
            "target": slice(0, split),
            "complement": slice(split, split + len(packet.complement_context_ids)),
        }
        packet_contexts = tuple(scheduled.registered_context_ids)
        packet_draws = tuple(scheduled.registered_draw_ids)
        packet_measurements = tuple(scheduled.registered_measurement_ids)
        packet_captures = tuple(scheduled.registered_capture_ids)
        packet_sources = tuple(scheduled.registered_source_packet_hashes)
        packet_strata = tuple(scheduled.registered_stratum_ids)
        context_axis.extend(packet_contexts)
        draw_axis.extend(packet_draws)
        measurement_axis.extend(packet_measurements)
        capture_axis.extend(packet_captures)
        source_packet_axis.extend(packet_sources)
        stratum_axis.extend(packet_strata)
        for role, _contexts in v1_sampled_v7_nonempty_physical_roles(
            target_context_ids=packet.target_context_ids,
            complement_context_ids=packet.complement_context_ids,
        ):
            section = slices[role]
            physical_axes.append(
                (
                    role,
                    (
                        packet_contexts[section],
                        packet_draws[section],
                        packet_measurements[section],
                        packet_captures[section],
                        packet_sources[section],
                    ),
                )
            )
    counts = (target_count, complement_count)
    expected_counts = tuple(
        acquisition_plan.logical_residual_after_stratum_counts[index]
        - acquisition_plan.logical_residual_before_stratum_counts[index]
        for index in (0, 1)
    )
    if (
        counts != expected_counts
        or tuple(role for role, _axes in physical_axes)
        != acquisition_plan.physical_roles
        or len(physical_axes) != len(acquisition_plan.physical_roles)
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "logical-look axes do not replay the frozen physical acquisition order",
        )
    return (
        tuple(physical_axes),
        packets,
        tuple(context_axis),
        tuple(draw_axis),
        tuple(measurement_axis),
        tuple(capture_axis),
        tuple(source_packet_axis),
        tuple(stratum_axis),
        counts,
    )


def _logical_source_local_weights_and_quanta(
    *,
    weight_receipt: object,
    logical_stratum_counts: tuple[int, int],
    registered_stratum_ids: Sequence[str],
) -> tuple[tuple[Fraction, ...], tuple[int, ...], int]:
    """Lower the exact residual measure into one logical-look transport unit."""

    weight_receipt.assert_integrity()
    labels = tuple(_identity(value, name="registered logical stratum") for value in registered_stratum_ids)
    target_count, complement_count = logical_stratum_counts
    if target_count + complement_count <= 0:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "a compact logical-look source cannot be empty",
        )
    if (
        len(labels) != target_count + complement_count
        or labels.count("target") != target_count
        or labels.count("complement") != complement_count
        or any(value not in {"target", "complement"} for value in labels)
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "registered logical strata do not replay the logical-look counts",
        )
    residual_masses = tuple(
        Fraction(*pair) for pair in weight_receipt.residual_stratum_mass_fractions
    )
    if (
        len(residual_masses) != 2
        or sum(residual_masses, start=Fraction(0)) <= 0
        or residual_masses
        != tuple(
            Fraction(
                count,
                weight_receipt.population_stratum_counts[0]
                + weight_receipt.population_stratum_counts[1],
            )
            for count in weight_receipt.residual_stratum_counts
        )
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "source-issued residual masses do not replay their registered counts",
        )
    present_masses = tuple(
        residual_masses[index]
        for index, count in enumerate((target_count, complement_count))
        if count > 0
    )
    present_mass = sum(present_masses, start=Fraction(0))
    if present_mass <= 0:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "a nonempty logical look cannot represent zero residual mass",
        )
    per_context = {
        "target": (
            residual_masses[0] / present_mass / target_count
            if target_count
            else Fraction(0)
        ),
        "complement": (
            residual_masses[1] / present_mass / complement_count
            if complement_count
            else Fraction(0)
        ),
    }
    fractions = tuple(per_context[label] for label in labels)
    if sum(fractions, start=Fraction(0)) != Fraction(1):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "logical-look compact-source weights do not close unit mass",
        )
    denominator = math.lcm(*(value.denominator for value in fractions))
    quanta = tuple(int(value * denominator) for value in fractions)
    if sum(quanta) != denominator or any(
        Fraction(quantum, denominator) != weight
        for quantum, weight in zip(quanta, fractions, strict=True)
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "logical-look integer quanta do not replay exact source weights",
        )
    return fractions, quanta, denominator


class _FanoutLease:
    __slots__ = ("state", "_lock")

    def __init__(self) -> None:
        self.state = "prepared"
        self._lock = RLock()

    def begin(self) -> None:
        with self._lock:
            if self.state != "prepared":
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.REPLAY,
                    f"fanout plan is already {self.state}",
                )
            self.state = "running"

    def complete(self) -> None:
        with self._lock:
            if self.state != "running":
                raise RuntimeError("fanout lifecycle did not enter running state")
            self.state = "complete"

    def abort(self) -> None:
        with self._lock:
            self.state = "aborted"

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("fanout lifecycle leases are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2PhysicalAction:
    physical_action_ordinal: int
    source_custody_hash: str
    physical_action_identity_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    parent_mask_hash: str
    removed_count: int
    removed_route_index_hash: str
    branch_alias_axis: tuple[V1RelaxedKKTBranch, ...]
    branch_alias_hashes: tuple[str, ...]
    packed_payload_bytes: int
    exact_packed_byte_custody: bool = True
    destination_frozen_pre_outcome: bool = True
    action_or_decision_authority: bool = False
    _packed_action: V1PackedRemovalAction = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _source_custody: V1SampledV7PhysicalActionCustody | None = field(
        repr=False, compare=False, default=None
    )
    _issuance_token: InitVar[object | None] = None
    _action_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ACTION_TOKEN:
            raise PermissionError("fanout physical actions are issuer-only")
        if type(self._packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
                "fanout action requires exact packed removal custody",
            )
        branches = tuple(self.branch_alias_axis)
        aliases = tuple(self.branch_alias_hashes)
        if (
            isinstance(self.physical_action_ordinal, bool)
            or int(self.physical_action_ordinal) <= 0
            or not branches
            or any(type(value) is not V1RelaxedKKTBranch for value in branches)
            or tuple(value for value in _BRANCH_AXIS if value in branches) != branches
            or len(branches) != len(set(branches))
            or len(aliases) != len(branches)
            or len(aliases) != len(set(aliases))
            or any(type(value) is not str or not value for value in aliases)
            or isinstance(self.packed_payload_bytes, bool)
            or int(self.packed_payload_bytes) <= 0
            or isinstance(self.removed_count, bool)
            or int(self.removed_count) <= 0
            or self.exact_packed_byte_custody is not True
            or self.destination_frozen_pre_outcome is not True
            or self.action_or_decision_authority is not False
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
                "physical action or alias partition is not exact",
            )
        for name in (
            "source_custody_hash",
            "physical_action_identity_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "parent_mask_hash",
            "removed_route_index_hash",
        ):
            _identity(getattr(self, name), name=name)
        action = self._packed_action
        if (
            action.action_hash != self.packed_action_hash
            or action.resulting_mask.mask_hash != self.resulting_mask_hash
            or action.removed_count != int(self.removed_count)
            or action.removed_route_index_hash != self.removed_route_index_hash
            or len(action.resulting_mask.packed_bits) != int(self.packed_payload_bytes)
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
                "packed action bytes differ from fanout custody",
            )
        if self._source_custody is not None:
            if (
                type(self._source_custody) is not V1SampledV7PhysicalActionCustody
                or self._source_custody.assert_integrity() != self.source_custody_hash
                or self._source_custody.packed_action is not action
            ):
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                    "physical action differs from its source action-pair custody",
                )
        object.__setattr__(self, "physical_action_ordinal", int(self.physical_action_ordinal))
        object.__setattr__(self, "packed_payload_bytes", int(self.packed_payload_bytes))
        object.__setattr__(self, "removed_count", int(self.removed_count))
        object.__setattr__(self, "branch_alias_axis", branches)
        object.__setattr__(self, "branch_alias_hashes", aliases)
        object.__setattr__(self, "_action_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "physical_action_ordinal": self.physical_action_ordinal,
                "source_custody_hash": self.source_custody_hash,
                "physical_action_identity_hash": self.physical_action_identity_hash,
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
                "parent_mask_hash": self.parent_mask_hash,
                "removed_count": self.removed_count,
                "removed_route_index_hash": self.removed_route_index_hash,
                "branch_alias_axis": self.branch_alias_axis,
                "branch_alias_hashes": self.branch_alias_hashes,
                "packed_payload_bytes": self.packed_payload_bytes,
                "exact_packed_byte_custody": True,
                "destination_frozen_pre_outcome": True,
            },
            prefix=_PREFIX + "physical-action-",
        )

    def assert_integrity(self) -> str:
        if self._source_custody is not None:
            self._source_custody.assert_integrity()
        observed = self._compute_hash()
        if observed != self._action_hash:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout physical action changed after freeze",
            )
        return observed

    @property
    def action_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2BlockSupportDestinationBinding:
    """Pre-outcome residual block-support sink for one physical action.

    The supplied registration is the exact residual segment registration
    expected by the already-open replay coordinator.  It is intentionally
    separate from the action's F/Q compact-source registration because their
    statistical context weights need not agree.
    """

    physical_action_identity_hash: str
    source_registration: V1SignedWorkVariableShellSourceRegistration
    sink_binding: V1SampledVariableShellBoundChunkSinkBinding = field(
        repr=False, compare=False
    )
    frozen_before_row_outcomes: bool = True
    residual_segment_only: bool = True
    d1_or_prior_segment_replay_authority: bool = False
    downstream_authority: bool = False
    schema: str = SAMPLED_V7_SHARED_DW2_BLOCK_SUPPORT_BINDING_SCHEMA
    _binding_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        _identity(
            self.physical_action_identity_hash,
            name="block-support physical action identity",
        )
        if (
            type(self.source_registration)
            is not V1SignedWorkVariableShellSourceRegistration
            or type(self.sink_binding)
            is not V1SampledVariableShellBoundChunkSinkBinding
            or self.sink_binding.bound_source_registration_hash
            != self.source_registration.registration_hash
            or self.frozen_before_row_outcomes is not True
            or self.residual_segment_only is not True
            or self.d1_or_prior_segment_replay_authority is not False
            or self.downstream_authority is not False
            or self.schema != SAMPLED_V7_SHARED_DW2_BLOCK_SUPPORT_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.DESTINATION_NOT_FROZEN,
                "block-support destination lacks an exact source-bound residual sink",
            )
        object.__setattr__(self, "_binding_hash", self.assert_integrity())

    def assert_integrity(self) -> str:
        observed = content_hash(
            {
                "schema": self.schema,
                "physical_action_identity_hash": self.physical_action_identity_hash,
                "source_registration_hash": self.source_registration.registration_hash,
                "sink_identity_hash": self.sink_binding.sink_identity_hash,
                "bound_source_registration_hash": (
                    self.sink_binding.bound_source_registration_hash
                ),
                "frozen_before_row_outcomes": True,
                "residual_segment_only": True,
                "d1_or_prior_segment_replay_authority": False,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "block-support-destination-binding-",
        )
        if self._binding_hash is not None and observed != self._binding_hash:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "block-support destination binding changed after freeze",
            )
        return observed

    @property
    def binding_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutDestination:
    action: V1SampledV7SharedDW2PhysicalAction
    support_binding_hashes: tuple[str, ...]
    action_resource_axis_hash: str
    action_resource_source_hash: str
    append_lineage: V1SignedWorkVariableShellSourceAppendLineage
    source_registration: V1SignedWorkVariableShellSourceRegistration
    append_registration: V1SignedWorkVariableShellSourceAppendPacketRegistration
    append_owner: V1SignedWorkVariableShellSourceAppendOwner = field(repr=False, compare=False)
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign = field(repr=False, compare=False)
    block_support_binding: V1SampledV7SharedDW2BlockSupportDestinationBinding | None = field(
        repr=False, compare=False, default=None
    )
    physical_action_axis_bound: bool = True
    destination_frozen_before_rows: bool = True
    outcomes_seen: bool = False
    downstream_authority: bool = False
    block_support_sink_axis_preregistered: bool = False
    action_specific_block_support_sink_integrated: bool = False
    d1_replay_integrated: bool = False
    full_action_specific_sink_fanout_complete: bool = False
    _support_bindings: tuple[V1SampledV7DOnlyBranchSourceBinding, ...] = field(
        repr=False, compare=False, default=()
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESTINATION_TOKEN:
            raise PermissionError("fanout destinations are issuer-only")
        if (
            type(self.action) is not V1SampledV7SharedDW2PhysicalAction
            or type(self.append_lineage) is not V1SignedWorkVariableShellSourceAppendLineage
            or type(self.source_registration) is not V1SignedWorkVariableShellSourceRegistration
            or type(self.append_registration) is not V1SignedWorkVariableShellSourceAppendPacketRegistration
            or type(self.append_owner) is not V1SignedWorkVariableShellSourceAppendOwner
            or type(self.design) is not V1SignedWorkVariableShellSourceCUDAExecutionDesign
            or any(type(value) is not V1SampledV7DOnlyBranchSourceBinding for value in self._support_bindings)
            or self.physical_action_axis_bound is not True
            or self.destination_frozen_before_rows is not True
            or self.outcomes_seen is not False
            or self.downstream_authority is not False
            or self.block_support_sink_axis_preregistered
            is not (self.block_support_binding is not None)
            or self.action_specific_block_support_sink_integrated is not False
            or self.d1_replay_integrated is not False
            or self.full_action_specific_sink_fanout_complete is not False
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.AUTHORITY_VIOLATION,
                "fanout destination crossed its pre-outcome action-local boundary",
            )
        self.assert_static_integrity(require_pending=True)

    def assert_static_integrity(self, *, require_pending: bool) -> str:
        action_hash = self.action.assert_integrity()
        for binding in self._support_bindings:
            binding.assert_live()
        block_hash = None
        if self.block_support_binding is not None:
            block_hash = self.block_support_binding.assert_integrity()
            if (
                self.block_support_binding.physical_action_identity_hash
                != self.action.physical_action_identity_hash
            ):
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.DESTINATION_NOT_FROZEN,
                    "block-support sink belongs to another physical action",
                )
        if (
            self.support_binding_hashes != tuple(value.binding_hash for value in self._support_bindings)
            or self.append_owner.lineage is not self.append_lineage
            or self.append_lineage.kkt_branch_source_receipt_hash != action_hash
            or self.append_lineage.resource_axis_hash != self.action_resource_axis_hash
            or self.append_lineage.resource_source_receipt_hash != self.action_resource_source_hash
            or self.source_registration.kkt_branch_source_receipt_hash != action_hash
            or self.source_registration.resource_axis_hash != self.action_resource_axis_hash
            or self.source_registration.resource_source_receipt_hash != self.action_resource_source_hash
            or self.append_lineage.design_hash != self.design.design_hash
            or self.append_registration.lineage_hash != self.append_lineage.lineage_hash
            or (require_pending and not self.append_owner.has_pending_packet)
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.DESTINATION_NOT_FROZEN,
                "action-local source, append, support, or resource custody changed",
            )
        return content_hash(
            {
                "action_hash": action_hash,
                "support_binding_hashes": self.support_binding_hashes,
                "action_resource_axis_hash": self.action_resource_axis_hash,
                "action_resource_source_hash": self.action_resource_source_hash,
                "append_lineage_hash": self.append_lineage.lineage_hash,
                "source_registration_hash": self.source_registration.registration_hash,
                "append_registration_hash": self.append_registration.registration_hash,
                "design_hash": self.design.design_hash,
                "block_support_binding_hash": block_hash,
                "block_support_sink_axis_preregistered": (
                    self.block_support_binding is not None
                ),
                "destination_frozen_before_rows": True,
            },
            prefix=_PREFIX + "destination-",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutPlan:
    acquisition_plan_hash: str
    measurement_owner_hash: str
    population_attachment_hash: str
    packet_hash: str
    physical_action_axis_hash: str
    branch_alias_to_physical_action: tuple[tuple[str, str, str], ...]
    destinations: tuple[V1SampledV7SharedDW2FanoutDestination, ...] = field(
        repr=False, compare=False
    )
    destination_axis_frozen_pre_outcome: bool = True
    extra_provider_call_budget: int = 0
    extra_vjp_budget: int = 0
    block_support_sink_axis_hash: str = _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
    block_support_sink_count: int = 0
    block_support_sink_axis_preregistered: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    action_specific_block_support_sink_integrated: bool = False
    d1_replay_integrated: bool = False
    full_action_specific_sink_fanout_complete: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_FANOUT_PLAN_SCHEMA
    _acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _lease: _FanoutLease = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _private_reference: bool = field(repr=False, compare=False, default=False)
    _failure_injection: str | None = field(repr=False, compare=False, default=None)
    _issuance_token: InitVar[object | None] = None
    _plan_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLAN_TOKEN:
            raise PermissionError("fanout plans are issuer-only")
        if type(self._lease) is not _FanoutLease:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
                "fanout plan lost its lifecycle lease",
            )
        self.assert_preoutcome()
        object.__setattr__(self, "_plan_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "acquisition_plan_hash": self.acquisition_plan_hash,
                "measurement_owner_hash": self.measurement_owner_hash,
                "population_attachment_hash": self.population_attachment_hash,
                "packet_hash": self.packet_hash,
                "physical_action_axis_hash": self.physical_action_axis_hash,
                "branch_alias_to_physical_action": self.branch_alias_to_physical_action,
                "destination_hashes": tuple(
                    value.assert_static_integrity(require_pending=self._lease.state == "prepared")
                    for value in self.destinations
                ),
                "destination_axis_frozen_pre_outcome": True,
                "extra_provider_call_budget": 0,
                "extra_vjp_budget": 0,
                "block_support_sink_axis_hash": self.block_support_sink_axis_hash,
                "block_support_sink_count": self.block_support_sink_count,
                "block_support_sink_axis_preregistered": (
                    self.block_support_sink_axis_preregistered
                ),
                "downstream_authority": False,
                "action_specific_block_support_sink_integrated": False,
                "d1_replay_integrated": False,
                "full_action_specific_sink_fanout_complete": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "plan-",
        )

    def assert_preoutcome(self) -> str:
        if (
            type(self._acquisition_plan) is not V1SampledV7SharedDW2AcquisitionPlan
            or not self.destinations
            or any(type(value) is not V1SampledV7SharedDW2FanoutDestination for value in self.destinations)
            or self._lease.state != "prepared"
            or self.destination_axis_frozen_pre_outcome is not True
            or self.extra_provider_call_budget != 0
            or self.extra_vjp_budget != 0
            or type(self.block_support_sink_count) is not int
            or self.block_support_sink_count < 0
            or self.block_support_sink_axis_preregistered
            is not (self.block_support_sink_count == len(self.destinations) > 0)
            or (
                self.block_support_sink_count == 0
                and self.block_support_sink_axis_hash
                != _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
            )
            or any(
                (value.block_support_binding is not None)
                is not self.block_support_sink_axis_preregistered
                for value in self.destinations
            )
            or self.block_support_sink_axis_hash
            != (
                _block_support_sink_axis_hash(
                    tuple(
                        value.block_support_binding
                        for value in self.destinations
                        if value.block_support_binding is not None
                    )
                )
                if self.block_support_sink_axis_preregistered
                else _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
            )
            or self.d_stability_authority is not False
            or self.c_k_o_or_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or self.action_specific_block_support_sink_integrated is not False
            or self.d1_replay_integrated is not False
            or self.full_action_specific_sink_fanout_complete is not False
            or self.authority_scope != SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY
            or self.schema != SAMPLED_V7_SHARED_DW2_FANOUT_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.AUTHORITY_VIOLATION,
                "fanout plan is not a frozen zero-provider pre-outcome axis",
            )
        if (
            self._acquisition_plan.plan_hash != self.acquisition_plan_hash
            or self.measurement_owner_hash != self._acquisition_plan.measurement_owner_hash
            or self.population_attachment_hash != self._acquisition_plan.population_attachment_hash
            or self.packet_hash != self._acquisition_plan.packet_hash
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout belongs to another shared acquisition source",
            )
        observed = self._compute_hash()
        if self._plan_hash is not None and observed != self._plan_hash:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout plan identity changed after destination freeze",
            )
        return observed

    @property
    def plan_hash(self) -> str:
        if self._plan_hash is None:
            return self.assert_preoutcome()
        return self._plan_hash


def _action_from_source(
    custody: V1SampledV7PhysicalActionCustody,
    *,
    parent_mask_hash: str,
) -> V1SampledV7SharedDW2PhysicalAction:
    import hashlib

    action = custody.packed_action
    return V1SampledV7SharedDW2PhysicalAction(
        physical_action_ordinal=custody.physical_action_ordinal,
        source_custody_hash=custody.custody_hash,
        physical_action_identity_hash=custody.physical_action_identity_hash,
        packed_action_hash=custody.packed_action_hash,
        resulting_mask_hash=custody.resulting_mask_hash,
        resulting_mask_packed_sha256=hashlib.sha256(action.resulting_mask.packed_bits).hexdigest(),
        parent_mask_hash=parent_mask_hash,
        removed_count=custody.removed_count,
        removed_route_index_hash=custody.removed_route_index_hash,
        branch_alias_axis=custody.branch_alias_axis,
        branch_alias_hashes=custody.branch_alias_hashes,
        packed_payload_bytes=custody.packed_payload_bytes,
        _packed_action=action,
        _source_custody=custody,
        _issuance_token=_ACTION_TOKEN,
    )


def issue_v1_sampled_v7_shared_dw2_physical_action(
    *,
    custody: V1SampledV7PhysicalActionCustody,
    parent_mask: HardMask,
) -> V1SampledV7SharedDW2PhysicalAction:
    """Lower exact source custody into one public fanout action.

    The parent object, rather than a caller-authored hash, is required so the
    issuer can replay the irreversible parent-to-child delta before sealing
    the fanout action.  This is representation custody only and grants no
    proposal, stability, C/K/O, or commit authority.
    """

    if (
        type(custody) is not V1SampledV7PhysicalActionCustody
        or type(parent_mask) is not HardMask
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "physical-action lowering requires exact source custody and parent mask",
        )
    try:
        custody.assert_integrity()
        action = custody.packed_action
        removed_count, removed_hash = action.resulting_mask.removed_index_count_and_hash_from(
            parent_mask
        )
    except Exception as error:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
            "source-custodied action is not an irreversible child of the supplied "
            f"parent: {type(error).__name__}: {error}",
        )
    if (
        removed_count != custody.removed_count
        or removed_hash != custody.removed_route_index_hash
        or action.resulting_mask.route_axis_hash != parent_mask.route_axis_hash
        or action.resulting_mask.route_count != parent_mask.route_count
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
            "source custody and exact parent-to-child removal delta disagree",
        )
    return _action_from_source(custody, parent_mask_hash=parent_mask.mask_hash)


def _private_actions(
    actions_by_branch: tuple[V1PackedRemovalAction, V1PackedRemovalAction],
    *,
    parent_mask: HardMask,
) -> tuple[V1SampledV7SharedDW2PhysicalAction, ...]:
    import hashlib

    if len(actions_by_branch) != 2 or any(type(value) is not V1PackedRemovalAction for value in actions_by_branch):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "private reference requires one exact action per canonical branch",
        )
    order: list[str] = []
    grouped: dict[str, list[tuple[V1RelaxedKKTBranch, V1PackedRemovalAction]]] = {}
    for branch, action in zip(_BRANCH_AXIS, actions_by_branch, strict=True):
        key = content_hash(
            {
                "action_hash": action.action_hash,
                "mask_hash": action.resulting_mask.mask_hash,
                "packed_sha256": hashlib.sha256(action.resulting_mask.packed_bits).hexdigest(),
            },
            prefix=_PREFIX + "private-dedup-",
        )
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append((branch, action))
    result = []
    for ordinal, key in enumerate(order, start=1):
        entries = grouped[key]
        action = entries[0][1]
        aliases = tuple(
            content_hash(
                {"private_action_key": key, "branch": branch},
                prefix=_PREFIX + "private-alias-",
            )
            for branch, _ in entries
        )
        result.append(
            V1SampledV7SharedDW2PhysicalAction(
                physical_action_ordinal=ordinal,
                source_custody_hash=content_hash(
                    {"private_action_key": key, "exact_packed_bytes": action.resulting_mask.packed_bits},
                    prefix=_PREFIX + "private-custody-",
                ),
                physical_action_identity_hash=key,
                packed_action_hash=action.action_hash,
                resulting_mask_hash=action.resulting_mask.mask_hash,
                resulting_mask_packed_sha256=hashlib.sha256(action.resulting_mask.packed_bits).hexdigest(),
                parent_mask_hash=parent_mask.mask_hash,
                removed_count=action.removed_count,
                removed_route_index_hash=action.removed_route_index_hash,
                branch_alias_axis=tuple(branch for branch, _ in entries),
                branch_alias_hashes=aliases,
                packed_payload_bytes=len(action.resulting_mask.packed_bits),
                _packed_action=action,
                _source_custody=None,
                _issuance_token=_ACTION_TOKEN,
            )
        )
    return tuple(result)


def prepare_v1_sampled_v7_shared_dw2_fanout(
    *,
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
    action_pair: V1SampledV7BranchActionPair,
    spectral_support: V1SampledV7DOnlyBranchSourceBinding,
    diagonal_l1_support: V1SampledV7DOnlyBranchSourceBinding,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
) -> V1SampledV7SharedDW2FanoutPlan:
    if (
        type(acquisition_plan) is not V1SampledV7SharedDW2AcquisitionPlan
        or type(action_pair) is not V1SampledV7BranchActionPair
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "production fanout requires exact source-issued branch-action custody",
        )
    action_pair.assert_integrity()
    current_mask = acquisition_plan._owner._current_mask
    if (
        action_pair.parent_mask_hash != current_mask.mask_hash
        or action_pair.route_axis_hash != current_mask.route_axis_hash
        or action_pair.route_count != current_mask.route_count
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "source-issued physical actions belong to another current parent/route axis",
        )
    return _prepare_fanout(
        acquisition_plan=acquisition_plan,
        actions=tuple(
            _action_from_source(value, parent_mask_hash=action_pair.parent_mask_hash)
            for value in action_pair.physical_actions
        ),
        spectral_support=spectral_support,
        diagonal_l1_support=diagonal_l1_support,
        design=design,
        private_reference=False,
        failure_injection=None,
        block_support_bindings=None,
    )


def prepare_v1_sampled_v7_shared_dw2_block_support_fanout(
    *,
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
    action_pair: V1SampledV7BranchActionPair,
    spectral_support: V1SampledV7DOnlyBranchSourceBinding,
    diagonal_l1_support: V1SampledV7DOnlyBranchSourceBinding,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_destination_axis: tuple[
        V1SampledV7SharedDW2BlockSupportDestinationBinding, ...
    ],
) -> V1SampledV7SharedDW2FanoutPlan:
    """Freeze the production residual block-support sink axis before rows exist.

    This preparation path is deliberately separate from the compatibility
    path above: every unique physical action must have one exact residual
    registration/sink binding, in physical-action order.  The supplied sinks
    may already contain accepted D1/prior segments, but this seam does not
    claim that replay; only the downstream coordinator's numerical-source
    seal can do so.
    """

    if (
        type(acquisition_plan) is not V1SampledV7SharedDW2AcquisitionPlan
        or type(action_pair) is not V1SampledV7BranchActionPair
        or type(block_support_destination_axis) is not tuple
        or not block_support_destination_axis
        or any(
            type(value)
            is not V1SampledV7SharedDW2BlockSupportDestinationBinding
            for value in block_support_destination_axis
        )
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "strict production fanout requires a complete exact block-support destination axis",
        )
    action_pair.assert_integrity()
    current_mask = acquisition_plan._owner._current_mask
    if (
        action_pair.parent_mask_hash != current_mask.mask_hash
        or action_pair.route_axis_hash != current_mask.route_axis_hash
        or action_pair.route_count != current_mask.route_count
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "source-issued physical actions belong to another current parent/route axis",
        )
    return _prepare_fanout(
        acquisition_plan=acquisition_plan,
        actions=tuple(
            _action_from_source(value, parent_mask_hash=action_pair.parent_mask_hash)
            for value in action_pair.physical_actions
        ),
        spectral_support=spectral_support,
        diagonal_l1_support=diagonal_l1_support,
        design=design,
        private_reference=False,
        failure_injection=None,
        block_support_bindings=block_support_destination_axis,
    )


def _prepare_v1_sampled_v7_shared_dw2_fanout_private_reference(
    *,
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
    actions_by_branch: tuple[V1PackedRemovalAction, V1PackedRemovalAction],
    spectral_support: V1SampledV7DOnlyBranchSourceBinding,
    diagonal_l1_support: V1SampledV7DOnlyBranchSourceBinding,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    failure_injection: str | None = None,
) -> V1SampledV7SharedDW2FanoutPlan:
    if type(acquisition_plan) is not V1SampledV7SharedDW2AcquisitionPlan:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "private fanout requires one exact shared acquisition plan",
        )
    return _prepare_fanout(
        acquisition_plan=acquisition_plan,
        actions=_private_actions(
            actions_by_branch,
            parent_mask=acquisition_plan._owner._current_mask,
        ),
        spectral_support=spectral_support,
        diagonal_l1_support=diagonal_l1_support,
        design=design,
        private_reference=True,
        failure_injection=failure_injection,
        block_support_bindings=None,
    )


def _prepare_v1_sampled_v7_shared_dw2_block_support_fanout_private_reference(
    *,
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
    actions_by_branch: tuple[V1PackedRemovalAction, V1PackedRemovalAction],
    spectral_support: V1SampledV7DOnlyBranchSourceBinding,
    diagonal_l1_support: V1SampledV7DOnlyBranchSourceBinding,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    block_support_destination_axis: tuple[
        V1SampledV7SharedDW2BlockSupportDestinationBinding, ...
    ],
    failure_injection: str | None = None,
) -> V1SampledV7SharedDW2FanoutPlan:
    """CPU small-domain constructor for the strict residual sink path."""

    if (
        type(acquisition_plan) is not V1SampledV7SharedDW2AcquisitionPlan
        or type(block_support_destination_axis) is not tuple
        or not block_support_destination_axis
        or any(
            type(value)
            is not V1SampledV7SharedDW2BlockSupportDestinationBinding
            for value in block_support_destination_axis
        )
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "private strict fanout requires a complete exact block-support destination axis",
        )
    return _prepare_fanout(
        acquisition_plan=acquisition_plan,
        actions=_private_actions(
            actions_by_branch,
            parent_mask=acquisition_plan._owner._current_mask,
        ),
        spectral_support=spectral_support,
        diagonal_l1_support=diagonal_l1_support,
        design=design,
        private_reference=True,
        failure_injection=failure_injection,
        block_support_bindings=block_support_destination_axis,
    )


def _prepare_fanout(
    *,
    acquisition_plan: V1SampledV7SharedDW2AcquisitionPlan,
    actions: tuple[V1SampledV7SharedDW2PhysicalAction, ...],
    spectral_support: V1SampledV7DOnlyBranchSourceBinding,
    diagonal_l1_support: V1SampledV7DOnlyBranchSourceBinding,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    private_reference: bool,
    failure_injection: str | None,
    block_support_bindings: tuple[
        V1SampledV7SharedDW2BlockSupportDestinationBinding, ...
    ]
    | None,
) -> V1SampledV7SharedDW2FanoutPlan:
    exact = (
        (acquisition_plan, V1SampledV7SharedDW2AcquisitionPlan),
        (spectral_support, V1SampledV7DOnlyBranchSourceBinding),
        (diagonal_l1_support, V1SampledV7DOnlyBranchSourceBinding),
        (design, V1SignedWorkVariableShellSourceCUDAExecutionDesign),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "fanout requires exact acquisition, support, and CUDA design owners",
        )
    acquisition_plan.assert_live()
    if bool(acquisition_plan._private_reference) is not bool(private_reference):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.CPU_PRODUCTION_FALLBACK,
            "production and private-reference acquisition/fanout modes may not mix",
        )
    if not actions:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.EMPTY_DESTINATION_AXIS,
            "no safe physical action exists to receive shared rows",
        )
    strict_block_support = block_support_bindings is not None
    frozen_block_axis = () if block_support_bindings is None else block_support_bindings
    if strict_block_support and (
        type(frozen_block_axis) is not tuple
        or len(frozen_block_axis) != len(actions)
        or any(
            type(value)
            is not V1SampledV7SharedDW2BlockSupportDestinationBinding
            for value in frozen_block_axis
        )
        or tuple(value.physical_action_identity_hash for value in frozen_block_axis)
        != tuple(value.physical_action_identity_hash for value in actions)
        or len(
            {
                value.physical_action_identity_hash
                for value in frozen_block_axis
            }
        )
        != len(frozen_block_axis)
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.DESTINATION_NOT_FROZEN,
            "strict residual block-support sinks must exactly cover the unique physical-action axis in order",
        )
    if private_reference:
        if design.execution_mode is not V1SignedWorkVariableShellExecutionMode.CPU_REFERENCE_TEST_ONLY:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.CPU_PRODUCTION_FALLBACK,
                "private fanout requires CPU reference mode",
            )
    elif design.execution_mode is not V1SignedWorkVariableShellExecutionMode.CUDA_PRODUCTION:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.CPU_PRODUCTION_FALLBACK,
            "production fanout has no CPU fallback",
        )
    bindings = {
        V1RelaxedKKTBranch.SPECTRAL: spectral_support,
        V1RelaxedKKTBranch.DIAGONAL_L1: diagonal_l1_support,
    }
    owner = acquisition_plan._owner
    attachment = acquisition_plan._population_attachment
    for binding in bindings.values():
        binding.assert_live()
        if (
            binding._measurement_owner is not owner
            or binding._population_attachment is not attachment
            or binding._design.design_hash != design.design_hash
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "branch-local supports belong to another measurement/population/design source",
            )
    source = owner.measurement_source_lineage
    (
        _physical_axes,
        transport_packets,
        registered_context_ids,
        registered_draw_ids,
        registered_measurement_ids,
        registered_capture_ids,
        registered_source_packet_hashes,
        registered_stratum_ids,
        logical_stratum_counts,
    ) = _logical_transport_axes(acquisition_plan)
    suffix = owner._suffix_registration
    weights = _issue_measure_weights(
        design=owner._residual_measure_design,
        registration=suffix,
        packet=transport_packets[-1],
    )
    fractions, quanta, _common_denominator = _logical_source_local_weights_and_quanta(
        weight_receipt=weights,
        logical_stratum_counts=logical_stratum_counts,
        registered_stratum_ids=registered_stratum_ids,
    )
    enclosures = tuple(_binary64_fraction_enclosure(value) for value in fractions)
    planned_prefix = v1_sampled_variable_shell_planned_prefix_hash(
        registration=suffix,
        packet=transport_packets[-1],
    )
    shell = owner._shell_union
    native = owner._native_fiber_axis_receipt
    shell_fibers = tuple(
        native.route_to_fiber(index).global_fiber_index for index in shell.shell_indices
    )
    destinations: list[V1SampledV7SharedDW2FanoutDestination] = []
    reservations: list[
        tuple[
            str,
            V1SignedWorkVariableShellSourceAppendOwner,
            V1SignedWorkVariableShellSourceAppendPacketRegistration,
        ]
    ] = []
    try:
        for action_index, action in enumerate(actions):
            action.assert_integrity()
            packed = action._packed_action
            if (
                packed.resulting_mask.route_axis_hash != source.route_axis_hash
                or packed.resulting_mask.route_count != native.route_count
            ):
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
                    "physical action packed bytes belong to another route axis",
                )
            try:
                removed_count, removed_hash = (
                    packed.resulting_mask.removed_index_count_and_hash_from(
                        owner._current_mask
                    )
                )
            except ValueError as error:
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
                    f"physical action is not an irreversible child of the current parent: {error}",
                )
            if (
                owner._current_mask.mask_hash != action.parent_mask_hash
                or removed_count != action.removed_count
                or removed_hash != action.removed_route_index_hash
            ):
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
                    "physical action delta does not replay against the current parent",
                )
            action_bindings = tuple(bindings[branch] for branch in action.branch_alias_axis)
            support_hashes = tuple(value.binding_hash for value in action_bindings)
            resource_axis = content_hash(
                {
                    "physical_action_custody_hash": action.action_hash,
                    "branch_support_resource_axes": tuple(
                        (branch, binding.append_lineage.resource_axis_hash)
                        for branch, binding in zip(action.branch_alias_axis, action_bindings, strict=True)
                    ),
                },
                prefix=_PREFIX + "action-resource-axis-",
            )
            resource_source = content_hash(
                {
                    "physical_action_custody_hash": action.action_hash,
                    "branch_support_resource_sources": tuple(
                        (branch, binding.append_lineage.resource_source_receipt_hash)
                        for branch, binding in zip(action.branch_alias_axis, action_bindings, strict=True)
                    ),
                },
                prefix=_PREFIX + "action-resource-source-",
            )
            lineage = V1SignedWorkVariableShellSourceAppendLineage(
                model_identity_hash=source.model_identity_hash,
                chart_identity_hash=source.chart_identity_hash,
                current_mask_hash=source.current_mask_hash,
                proposal_generation=source.proposal_generation,
                route_axis_hash=source.route_axis_hash,
                route_id_axis_hash=source.route_id_axis_hash,
                registered_d_prefix_hash=suffix.d1_pair_prefix_axis_hash,
                shell_freeze_receipt_hash=shell.receipt_hash,
                shell_indices=shell.shell_indices,
                shell_route_ids=shell.shell_route_ids,
                shell_native_fiber_indices=shell_fibers,
                native_fiber_axis_receipt_hash=native.receipt_hash,
                native_fiber_measure_hash=native.measure_hash,
                kkt_branch_id=SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID,
                kkt_branch_source_receipt_hash=action.action_hash,
                resource_axis_hash=resource_axis,
                resource_source_receipt_hash=resource_source,
                witness_stream_source_hash=source.witness_stream_source_hash,
                same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
                corrected_evidence_stream_source_hash=source.corrected_evidence_stream_source_hash,
                literal_j_stream_source_hash=source.literal_j_stream_source_hash,
                alpha_ledger_id=source.alpha_ledger_id,
                alpha_coordinate_id=source.alpha_coordinate_id,
                alpha_lineage_hash=source.alpha_lineage_hash,
                design_hash=design.design_hash,
                population_count=owner.schedule.population_count,
                population_context_axis_hash=owner.schedule.population_context_axis_hash,
                population_draw_axis_hash=owner.schedule.population_draw_axis_hash,
                route_count=native.route_count,
                fiber_count=native.fiber_count,
                cut_count=native.cut_count,
                predictable_packet_order_source_hash=owner.schedule.schedule_hash,
                stratum_measure_source_hash=owner._residual_measure_design.design_hash,
                common_integer_measure_unit_hash=v1_sampled_variable_shell_common_measure_unit_hash(registration=suffix),
            )
            append_owner = V1SignedWorkVariableShellSourceAppendOwner(lineage=lineage)
            registration = V1SignedWorkVariableShellSourceRegistration(
                registration_id=content_hash(
                    {
                        "acquisition_plan_hash": acquisition_plan.plan_hash,
                        "physical_action_custody_hash": action.action_hash,
                        "planned_D_W2_prefix_hash": planned_prefix,
                        "action_resource_axis_hash": resource_axis,
                    },
                    prefix=_PREFIX + "source-registration-id-",
                ),
                model_identity_hash=source.model_identity_hash,
                chart_identity_hash=source.chart_identity_hash,
                current_mask_hash=source.current_mask_hash,
                proposal_generation=source.proposal_generation,
                route_axis_hash=source.route_axis_hash,
                route_id_axis_hash=source.route_id_axis_hash,
                registered_d_prefix_hash=suffix.d1_pair_prefix_axis_hash,
                registered_d_w2_prefix_hash=planned_prefix,
                parent_d_source_receipt_hash=attachment.population_registration_hash,
                shell_freeze_receipt_hash=shell.receipt_hash,
                shell_indices=shell.shell_indices,
                shell_route_ids=shell.shell_route_ids,
                native_fiber_axis_receipt_hash=native.receipt_hash,
                native_fiber_measure_hash=native.measure_hash,
                kkt_branch_id=SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID,
                kkt_branch_source_receipt_hash=action.action_hash,
                resource_axis_hash=resource_axis,
                resource_source_receipt_hash=resource_source,
                witness_stream_source_hash=source.witness_stream_source_hash,
                same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
                corrected_evidence_stream_source_hash=source.corrected_evidence_stream_source_hash,
                literal_j_stream_source_hash=source.literal_j_stream_source_hash,
                alpha_ledger_id=source.alpha_ledger_id,
                alpha_coordinate_id=source.alpha_coordinate_id,
                alpha_lineage_hash=source.alpha_lineage_hash,
                population_count=owner.schedule.population_count,
                population_context_axis_hash=owner.schedule.population_context_axis_hash,
                population_draw_axis_hash=owner.schedule.population_draw_axis_hash,
                registered_context_ids=registered_context_ids,
                registered_draw_ids=registered_draw_ids,
                registered_measurement_ids=registered_measurement_ids,
                registered_capture_ids=registered_capture_ids,
                registered_source_packet_hashes=registered_source_packet_hashes,
                context_weights=tuple(value[1] for value in enclosures),
            )
            block_binding = (
                frozen_block_axis[action_index] if strict_block_support else None
            )
            if block_binding is not None:
                block_binding.assert_integrity()
                residual = block_binding.source_registration
                differing_fields = tuple(
                    name
                    for name in registration.__dataclass_fields__
                    if name not in {"registration_id", "context_weights"}
                    and getattr(residual, name) != getattr(registration, name)
                )
                uniform_weight = 1.0 / float(registration.context_count)
                if (
                    differing_fields
                    or residual.registration_id == registration.registration_id
                    or residual.context_weights
                    != (uniform_weight,) * registration.context_count
                    or block_binding.physical_action_identity_hash
                    != action.physical_action_identity_hash
                ):
                    _refuse(
                        V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
                        "residual block-support registration is not the exact uniform-weight physical D_W2 action segment"
                        + (
                            ":" + ",".join(differing_fields)
                            if differing_fields
                            else ""
                        ),
                    )
            stratum_ids = tuple(
                content_hash(
                    {
                        "physical_action_custody_hash": action.action_hash,
                        "logical_packet_issuance_hash": acquisition_plan.packet_hash,
                        "planned_D_W2_prefix_hash": planned_prefix,
                        "source_packet_hash": source_packet_hash,
                        "context_id": context_id,
                        "logical_stratum": stratum,
                    },
                    prefix=_PREFIX + "append-stratum-",
                )
                for context_id, source_packet_hash, stratum in zip(
                    registered_context_ids,
                    registered_source_packet_hashes,
                    registered_stratum_ids,
                    strict=True,
                )
            )
            append_registration = append_owner.register_next_packet(
                registered_d_w2_prefix_hash=planned_prefix,
                context_ids=registered_context_ids,
                draw_ids=registered_draw_ids,
                measurement_ids=registered_measurement_ids,
                capture_ids=registered_capture_ids,
                source_packet_hashes=registered_source_packet_hashes,
                stratum_ids=stratum_ids,
                context_measure_quanta=quanta,
                logical_stratum_ids=registered_stratum_ids,
            )
            reservations.append((action.action_hash, append_owner, append_registration))
            destinations.append(
                V1SampledV7SharedDW2FanoutDestination(
                    action=action,
                    support_binding_hashes=support_hashes,
                    action_resource_axis_hash=resource_axis,
                    action_resource_source_hash=resource_source,
                    append_lineage=lineage,
                    source_registration=registration,
                    append_registration=append_registration,
                    append_owner=append_owner,
                    design=design,
                    block_support_binding=block_binding,
                    block_support_sink_axis_preregistered=(block_binding is not None),
                    _support_bindings=action_bindings,
                    _issuance_token=_DESTINATION_TOKEN,
                )
            )
    except BaseException as original_error:
        cleanup_failures: list[str] = []
        for action_hash, append_owner, append_registration in reservations:
            try:
                if append_owner.has_pending_packet:
                    append_owner.abort_pending_packet(registration=append_registration)
                append_owner.evict()
                if not bool(getattr(append_owner, "_evicted", False)):
                    raise RuntimeError("append owner did not enter the evicted state")
            except BaseException as cleanup_error:
                cleanup_failures.append(
                    content_hash(
                        {
                            "action_hash": action_hash,
                            "registration_hash": append_registration.registration_hash,
                            "cleanup_error_type": type(cleanup_error).__qualname__,
                            "cleanup_error_message": str(cleanup_error),
                        },
                        prefix=_PREFIX + "preparation-cleanup-failure-",
                    )
                )
        if cleanup_failures:
            raise V1SampledV7SharedDW2FanoutRefusal(
                V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE,
                "destination preparation failed and exact pending-owner cleanup is incomplete: "
                + ",".join(cleanup_failures),
            ) from original_error
        raise
    if len({value.action.physical_action_identity_hash for value in destinations}) != len(destinations):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
            "destination axis repeated a physical action identity",
        )
    alias_map = tuple(
        (branch.value, alias_hash, destination.action.physical_action_identity_hash)
        for destination in destinations
        for branch, alias_hash in zip(
            destination.action.branch_alias_axis,
            destination.action.branch_alias_hashes,
            strict=True,
        )
    )
    alias_branches = tuple(row[0] for row in alias_map)
    expected_alias_branches = tuple(
        value.value for value in _BRANCH_AXIS if value.value in alias_branches
    )
    if (
        not alias_map
        or alias_branches != expected_alias_branches
        or len(set(alias_branches)) != len(alias_branches)
        or len({row[1] for row in alias_map}) != len(alias_map)
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ALIAS_MISMATCH,
            "physical destinations do not preserve the canonical unique action-alias subset",
        )
    axis_hash = content_hash(
        {
            "physical_action_hashes": tuple(value.action.action_hash for value in destinations),
            "branch_alias_to_physical_action": alias_map,
            "pre_outcome": True,
        },
        prefix=_PREFIX + "destination-axis-",
    )
    block_axis_hash = (
        _block_support_sink_axis_hash(frozen_block_axis)
        if strict_block_support
        else _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
    )
    return V1SampledV7SharedDW2FanoutPlan(
        acquisition_plan_hash=acquisition_plan.plan_hash,
        measurement_owner_hash=acquisition_plan.measurement_owner_hash,
        population_attachment_hash=acquisition_plan.population_attachment_hash,
        packet_hash=acquisition_plan.packet_hash,
        physical_action_axis_hash=axis_hash,
        branch_alias_to_physical_action=alias_map,
        destinations=tuple(destinations),
        block_support_sink_axis_hash=block_axis_hash,
        block_support_sink_count=len(frozen_block_axis),
        block_support_sink_axis_preregistered=strict_block_support,
        _acquisition_plan=acquisition_plan,
        _lease=_FanoutLease(),
        _private_reference=private_reference,
        _failure_injection=failure_injection,
        _issuance_token=_PLAN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutDestinationResult:
    action_hash: str
    physical_action_identity_hash: str
    source_receipt_hash: str
    append_receipt_hash: str
    chunk_binding_axis_hash: str
    branch_alias_axis: tuple[V1RelaxedKKTBranch, ...]
    branch_alias_hashes: tuple[str, ...]
    source_handle: V1SignedWorkVariableShellSourceDeviceHandle = field(repr=False, compare=False)
    append_owner: V1SignedWorkVariableShellSourceAppendOwner = field(repr=False, compare=False)
    block_support_bridge_result: V1SampledVariableShellSourceBridgeResult | None = field(
        repr=False, compare=False, default=None
    )
    source_or_append_authority_only: bool = True
    downstream_authority: bool = False
    action_specific_block_support_sink_integrated: bool = False
    d1_replay_integrated: bool = False
    full_action_specific_sink_fanout_complete: bool = False

    def __post_init__(self) -> None:
        if (
            self.block_support_bridge_result is not None
            and type(self.block_support_bridge_result)
            is not V1SampledVariableShellSourceBridgeResult
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
                "block-support fanout result must be the exact bridge result type",
            )
        block_integrated = self.block_support_bridge_result is not None
        if (
            type(self.source_handle) is not V1SignedWorkVariableShellSourceDeviceHandle
            or type(self.append_owner) is not V1SignedWorkVariableShellSourceAppendOwner
            or self.source_handle.receipt.receipt_hash != self.source_receipt_hash
            or self.append_owner.receipt.receipt_hash != self.append_receipt_hash
            or self.source_handle.receipt.chunk_binding_axis_hash != self.chunk_binding_axis_hash
            or self.source_or_append_authority_only is not True
            or self.downstream_authority is not False
            or self.action_specific_block_support_sink_integrated
            is not block_integrated
            or self.d1_replay_integrated is not False
            or self.full_action_specific_sink_fanout_complete
            is not block_integrated
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout destination result lost its sealed compact/append custody",
            )
        if block_integrated:
            bridge = self.block_support_bridge_result
            assert bridge is not None
            bridge.receipt.assert_integrity()
            if (
                bridge.receipt.bound_chunk_sink_count != 1
                or bridge.receipt.bound_chunk_fanout_complete is not True
            ):
                _refuse(
                    V1SampledV7SharedDW2FanoutRefusalCode.PARTIAL_COVER,
                    "action-specific residual block-support bridge telemetry is incomplete",
                )

    def as_segmented_source(self) -> V1SignedWorkVariableShellSourceAppendSegmentedHandle:
        """Expose only the action-local residual compact prefix.

        This does not add the action-specific block-support/D1 contribution;
        that extension remains a separately typed downstream obligation.
        """

        if (
            self.source_handle.receipt.receipt_hash != self.source_receipt_hash
            or self.append_owner.receipt.receipt_hash != self.append_receipt_hash
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout destination changed before segmented-source exposure",
            )
        return V1SignedWorkVariableShellSourceAppendSegmentedHandle(
            owner=self.append_owner,
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutReceipt:
    plan_hash: str
    acquisition_receipt_hash: str
    physical_action_axis_hash: str
    source_receipt_hashes: tuple[str, ...]
    append_receipt_hashes: tuple[str, ...]
    chunk_binding_axis_hashes: tuple[str, ...]
    unique_physical_action_count: int
    distinct_branch_alias_count: int
    raw_role_tile_read_count: int
    device_role_tile_upload_count: int
    destination_local_chunk_binding_count: int
    block_support_bridge_receipt_hashes: tuple[str, ...] = ()
    block_support_sink_axis_hash: str = _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
    block_support_destination_local_chunk_binding_count: int = 0
    block_support_bound_sink_event_count: int = 0
    extra_provider_call_count: int = 0
    extra_vjp_count: int = 0
    each_raw_role_tile_read_once: bool = True
    each_raw_role_tile_uploaded_once: bool = True
    failure_atomic_destination_publication: bool = True
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    action_specific_block_support_sink_integrated: bool = False
    d1_replay_integrated: bool = False
    full_action_specific_sink_fanout_complete: bool = False
    authority_scope: str = SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY
    schema: str = SAMPLED_V7_SHARED_DW2_FANOUT_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESULT_TOKEN:
            raise PermissionError("fanout receipts are executor-issued only")
        count = int(self.unique_physical_action_count)
        axes = (
            self.source_receipt_hashes,
            self.append_receipt_hashes,
            self.chunk_binding_axis_hashes,
        )
        strict_block_support = self.action_specific_block_support_sink_integrated
        if (
            count <= 0
            or any(
                type(value) is not int
                for value in (
                    self.unique_physical_action_count,
                    self.distinct_branch_alias_count,
                    self.raw_role_tile_read_count,
                    self.device_role_tile_upload_count,
                    self.destination_local_chunk_binding_count,
                    self.block_support_destination_local_chunk_binding_count,
                    self.block_support_bound_sink_event_count,
                    self.extra_provider_call_count,
                    self.extra_vjp_count,
                )
            )
            or any(type(axis) is not tuple for axis in axes)
            or any(len(axis) != count for axis in axes)
            or any(any(type(value) is not str or not value for value in axis) for axis in axes)
            or int(self.distinct_branch_alias_count) < count
            or int(self.raw_role_tile_read_count) <= 0
            or self.raw_role_tile_read_count != self.device_role_tile_upload_count
            or int(self.destination_local_chunk_binding_count) <= 0
            or type(self.block_support_bridge_receipt_hashes) is not tuple
            or (
                strict_block_support
                and (
                    len(self.block_support_bridge_receipt_hashes) != count
                    or any(
                        type(value) is not str or not value
                        for value in self.block_support_bridge_receipt_hashes
                    )
                    or self.block_support_sink_axis_hash
                    == _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
                    or self.block_support_destination_local_chunk_binding_count
                    <= 0
                    or self.block_support_bound_sink_event_count <= 0
                    or self.full_action_specific_sink_fanout_complete is not True
                )
            )
            or (
                not strict_block_support
                and (
                    self.block_support_bridge_receipt_hashes
                    or self.block_support_sink_axis_hash
                    != _EMPTY_BLOCK_SUPPORT_SINK_AXIS_HASH
                    or self.block_support_destination_local_chunk_binding_count != 0
                    or self.block_support_bound_sink_event_count != 0
                    or self.full_action_specific_sink_fanout_complete is not False
                )
            )
            or self.extra_provider_call_count != 0
            or self.extra_vjp_count != 0
            or self.each_raw_role_tile_read_once is not True
            or self.each_raw_role_tile_uploaded_once is not True
            or self.failure_atomic_destination_publication is not True
            or self.d_stability_authority is not False
            or self.c_k_o_or_commit_authority is not False
            or self.terminal_or_pareto_authority is not False
            or type(self.action_specific_block_support_sink_integrated) is not bool
            or self.d1_replay_integrated is not False
            or type(self.full_action_specific_sink_fanout_complete) is not bool
            or self.authority_scope != SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY
            or self.schema != SAMPLED_V7_SHARED_DW2_FANOUT_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.AUTHORITY_VIOLATION,
                "fanout receipt has partial cover or overstates its compact-source seam",
            )
        for name in (
            "plan_hash",
            "acquisition_receipt_hash",
            "physical_action_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        object.__setattr__(self, "_receipt_hash", self.assert_integrity())

    def assert_integrity(self) -> str:
        observed = content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "receipt-",
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "fanout receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutResult:
    receipt: V1SampledV7SharedDW2FanoutReceipt
    destinations: tuple[V1SampledV7SharedDW2FanoutDestinationResult, ...] = field(
        repr=False, compare=False
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESULT_TOKEN:
            raise PermissionError("fanout results are executor-only")
        self.receipt.assert_integrity()
        block_bridges = tuple(
            value.block_support_bridge_result
            for value in self.destinations
            if value.block_support_bridge_result is not None
        )
        if (
            len(self.destinations) != self.receipt.unique_physical_action_count
            or tuple(value.source_receipt_hash for value in self.destinations)
            != self.receipt.source_receipt_hashes
            or tuple(value.append_receipt_hash for value in self.destinations)
            != self.receipt.append_receipt_hashes
            or tuple(
                value.receipt.receipt_hash for value in block_bridges
            )
            != self.receipt.block_support_bridge_receipt_hashes
            or sum(
                value.receipt.paired_route_chunk_count for value in block_bridges
            )
            != self.receipt.block_support_destination_local_chunk_binding_count
            or sum(
                value.receipt.bound_chunk_sink_event_count for value in block_bridges
            )
            != self.receipt.block_support_bound_sink_event_count
            or any(
                value.action_specific_block_support_sink_integrated
                is not self.receipt.action_specific_block_support_sink_integrated
                for value in self.destinations
            )
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.PARTIAL_COVER,
                "fanout result does not cover its physical destination axis",
            )


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutCleanupItem:
    action_hash: str
    append_registration_hash: str
    pending_reservation_observed: bool
    abort_receipt_hash: str | None
    append_owner_evicted: bool
    compact_handle_present: bool
    compact_handle_evicted: bool
    block_support_handle_present: bool
    block_support_handle_evicted: bool
    block_support_sink_touched: bool
    block_support_abort_hash: str | None
    block_support_cleanup_debt_hash: str | None
    cleanup_failure_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.action_hash, name="cleanup action hash")
        _identity(self.append_registration_hash, name="cleanup append registration hash")
        if self.abort_receipt_hash is not None:
            _identity(self.abort_receipt_hash, name="cleanup abort receipt hash")
        if self.block_support_cleanup_debt_hash is not None:
            _identity(
                self.block_support_cleanup_debt_hash,
                name="block-support cleanup debt hash",
            )
        if self.block_support_abort_hash is not None:
            _identity(
                self.block_support_abort_hash,
                name="block-support abort hash",
            )
        if any(
            type(value) is not bool
            for value in (
                self.pending_reservation_observed,
                self.append_owner_evicted,
                self.compact_handle_present,
                self.compact_handle_evicted,
                self.block_support_handle_present,
                self.block_support_handle_evicted,
                self.block_support_sink_touched,
            )
        ) or any(type(value) is not str or not value for value in self.cleanup_failure_hashes):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE,
                "cleanup evidence is malformed",
            )
        if (
            (not self.pending_reservation_observed and self.abort_receipt_hash is not None)
            or (
                self.pending_reservation_observed
                and self.abort_receipt_hash is None
                and not self.cleanup_failure_hashes
            )
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE,
                "pending-reservation cleanup lacks its exact abort receipt",
            )
        if (
            not self.block_support_sink_touched
            and (
                self.block_support_abort_hash is not None
                or self.block_support_cleanup_debt_hash is not None
            )
        ) or (
            self.block_support_sink_touched
            and (
                (self.block_support_abort_hash is None)
                == (self.block_support_cleanup_debt_hash is None)
            )
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE,
                "touched block-support sink must have exactly one abort witness or cleanup debt",
            )

    @property
    def cleanup_complete(self) -> bool:
        return (
            self.append_owner_evicted
            and self.compact_handle_evicted
            and self.block_support_handle_evicted
            and self.block_support_cleanup_debt_hash is None
            and (
                not self.block_support_sink_touched
                or self.block_support_abort_hash is not None
            )
            and not self.cleanup_failure_hashes
        )

    @property
    def item_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "cleanup-item-")


@dataclass(frozen=True, slots=True)
class V1SampledV7SharedDW2FanoutCleanupReceipt:
    plan_hash: str
    original_failure_hash: str
    items: tuple[V1SampledV7SharedDW2FanoutCleanupItem, ...]
    no_result_published: bool = True
    all_destinations_cleaned: bool = True
    downstream_authority: bool = False

    def __post_init__(self) -> None:
        _identity(self.plan_hash, name="cleanup plan hash")
        _identity(self.original_failure_hash, name="original fanout failure hash")
        if (
            not self.items
            or any(type(value) is not V1SampledV7SharedDW2FanoutCleanupItem for value in self.items)
            or self.no_result_published is not True
            or self.all_destinations_cleaned
            is not all(value.cleanup_complete for value in self.items)
            or self.downstream_authority is not False
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE,
                "cleanup receipt does not truthfully describe all destinations",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "plan_hash": self.plan_hash,
                "original_failure_hash": self.original_failure_hash,
                "item_hashes": tuple(value.item_hash for value in self.items),
                "no_result_published": True,
                "all_destinations_cleaned": self.all_destinations_cleaned,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "cleanup-receipt-",
        )


def _cleanup_failure_hash(*, phase: str, error: BaseException) -> str:
    return content_hash(
        {
            "phase": phase,
            "error_type": type(error).__qualname__,
            "error_message": str(error),
        },
        prefix=_PREFIX + "cleanup-failure-",
    )


def _cleanup_failed_fanout(
    *,
    plan: V1SampledV7SharedDW2FanoutPlan,
    handles: Sequence[V1SignedWorkVariableShellSourceDeviceHandle],
    block_support_handles: Sequence[
        V1SignedWorkVariableShellSourceDeviceHandle | None
    ],
    block_support_sink_touched: Sequence[bool],
    original_error: BaseException,
) -> V1SampledV7SharedDW2FanoutCleanupReceipt:
    items: list[V1SampledV7SharedDW2FanoutCleanupItem] = []
    for index, destination in enumerate(plan.destinations):
        owner = destination.append_owner
        handle = handles[index] if index < len(handles) else None
        block_handle = (
            block_support_handles[index]
            if index < len(block_support_handles)
            else None
        )
        sink_touched = (
            bool(block_support_sink_touched[index])
            if index < len(block_support_sink_touched)
            else False
        )
        pending = False
        abort_hash: str | None = None
        failures: list[str] = []
        try:
            pending = owner.has_pending_packet
        except BaseException as error:
            failures.append(_cleanup_failure_hash(phase="inspect-pending", error=error))
        if pending:
            try:
                abort = owner.abort_pending_packet(
                    registration=destination.append_registration
                )
                abort_hash = abort.receipt_hash
            except BaseException as error:
                failures.append(_cleanup_failure_hash(phase="abort-pending", error=error))
        try:
            owner.evict()
        except BaseException as error:
            failures.append(_cleanup_failure_hash(phase="evict-append-owner", error=error))
        owner_evicted = bool(getattr(owner, "_evicted", False))
        if handle is not None and not handle.is_evicted:
            try:
                handle.evict()
            except BaseException as error:
                failures.append(_cleanup_failure_hash(phase="evict-compact-handle", error=error))
        handle_evicted = handle is None or handle.is_evicted
        if block_handle is not None and not block_handle.is_evicted:
            try:
                block_handle.evict()
            except BaseException as error:
                failures.append(
                    _cleanup_failure_hash(
                        phase="evict-block-support-compact-handle", error=error
                    )
                )
        block_handle_evicted = block_handle is None or block_handle.is_evicted
        block_abort_hash: str | None = None
        cleanup_debt_hash: str | None = None
        if sink_touched:
            binding = destination.block_support_binding
            sink = None if binding is None else binding.sink_binding.sink
            abort = getattr(sink, "abort_unsealed", None)
            if callable(abort):
                try:
                    observed = abort()
                    block_abort_hash = _identity(
                        observed,
                        name="block-support replay abort hash",
                    )
                except BaseException as error:
                    failures.append(
                        _cleanup_failure_hash(
                            phase="abort-block-support-replay",
                            error=error,
                        )
                    )
            if block_abort_hash is None:
                cleanup_debt_hash = content_hash(
                    {
                        "schema": (
                            "partition_free_v1_sampled_v7_block_support_sink_"
                            "cleanup_debt_v2"
                        ),
                        "action_hash": destination.action.action_hash,
                        "sink_identity_hash": (
                            None
                            if binding is None
                            else binding.sink_binding.sink_identity_hash
                        ),
                        "abort_transition_exposed": callable(abort),
                        "partially_consumed_sink_must_not_be_reused": True,
                    },
                    prefix=_PREFIX + "block-support-cleanup-debt-",
                )
        items.append(
            V1SampledV7SharedDW2FanoutCleanupItem(
                action_hash=destination.action.action_hash,
                append_registration_hash=destination.append_registration.registration_hash,
                pending_reservation_observed=pending,
                abort_receipt_hash=abort_hash,
                append_owner_evicted=owner_evicted,
                compact_handle_present=handle is not None,
                compact_handle_evicted=handle_evicted,
                block_support_handle_present=block_handle is not None,
                block_support_handle_evicted=block_handle_evicted,
                block_support_sink_touched=sink_touched,
                block_support_abort_hash=block_abort_hash,
                block_support_cleanup_debt_hash=cleanup_debt_hash,
                cleanup_failure_hashes=tuple(failures),
            )
        )
    return V1SampledV7SharedDW2FanoutCleanupReceipt(
        plan_hash=plan.plan_hash,
        original_failure_hash=content_hash(
            {
                "error_type": type(original_error).__qualname__,
                "error_message": str(original_error),
            },
            prefix=_PREFIX + "original-failure-",
        ),
        items=tuple(items),
        all_destinations_cleaned=all(value.cleanup_complete for value in items),
    )


def _exact_parent_to_child_removal_primal(
    *,
    parent_mask: HardMask,
    action: V1SampledV7SharedDW2PhysicalAction,
    device: Any,
) -> Any:
    """Materialize ``z = parent_live & ~child_live`` with exact delta custody.

    The compact source's fixed-outside vector is a removal direction, not a
    child live-mask.  Streamed packed-mask scans keep the CPU workspace bounded
    while proving that the vector's count and canonical index hash equal the
    source-issued action.
    """

    if type(parent_mask) is not HardMask or type(action) is not V1SampledV7SharedDW2PhysicalAction:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "removal-primal construction requires exact parent-mask/action custody",
        )
    if torch is None:
        raise RuntimeError("PyTorch is unavailable")
    action.assert_integrity()
    packed = action._packed_action
    child = packed.resulting_mask
    try:
        parent_mask.assert_same_axis(child)
        removed_count, removed_hash = child.removed_index_count_and_hash_from(parent_mask)
    except (TypeError, ValueError) as error:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            f"physical action is not an irreversible child of the current parent: {error}",
        )
    if parent_mask.mask_hash != action.parent_mask_hash:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "source-issued action belongs to another current parent mask",
        )
    if (
        removed_count != action.removed_count
        or removed_hash != action.removed_route_index_hash
        or removed_count != packed.removed_count
        or removed_hash != packed.removed_route_index_hash
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
            "parent-to-child removal direction differs from action count/hash custody",
        )
    frozen = torch.zeros(
        (parent_mask.route_count,),
        dtype=torch.float64,
        device=torch.device(device),
    )
    materialized_count = 0
    for chunk in child.iter_removed_index_chunks_from(parent_mask):
        indices = torch.as_tensor(
            np.asarray(chunk, dtype=np.int64),
            dtype=torch.int64,
            device=frozen.device,
        )
        frozen[indices] = 1.0
        materialized_count += int(indices.numel())
    if materialized_count != removed_count or int(torch.count_nonzero(frozen).item()) != removed_count:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.ACTION_COLLAPSE,
            "materialized removal primal does not replay the exact packed delta",
        )
    return frozen


def _next_native_fiber_stop(
    native: Any,
    start: int,
    available_stop: int,
    maximum: int,
) -> int | None:
    cut = native.cut_for_route(start)
    local = start - cut.route_offset
    if local % cut.routes_per_fiber:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "fanout route cursor begins inside a native fiber",
        )
    if maximum < cut.routes_per_fiber:
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
            "fanout design cannot form one complete native-fiber route tile",
        )
    candidate = min(available_stop, cut.route_stop, start + maximum)
    width = candidate - start
    width -= width % cut.routes_per_fiber
    if width <= 0:
        return None
    return start + width


def execute_v1_sampled_v7_shared_dw2_fanout(
    *,
    plan: V1SampledV7SharedDW2FanoutPlan,
    acquisition: V1SampledV7SharedDW2AcquisitionResult,
) -> V1SampledV7SharedDW2FanoutResult:
    if (
        type(plan) is not V1SampledV7SharedDW2FanoutPlan
        or type(acquisition) is not V1SampledV7SharedDW2AcquisitionResult
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.INVALID_TYPE,
            "fanout execution requires the exact frozen plan and shared acquisition",
        )
    plan.assert_preoutcome()
    if (
        acquisition.receipt.plan_hash != plan.acquisition_plan_hash
        or acquisition.receipt.packet_hash != plan.packet_hash
    ):
        _refuse(
            V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
            "shared row acquisition differs from the frozen destination axis",
        )
    plan._lease.begin()
    handles: list[V1SignedWorkVariableShellSourceDeviceHandle] = []
    block_support_handles: list[
        V1SignedWorkVariableShellSourceDeviceHandle | None
    ] = [None] * len(plan.destinations)
    block_support_sink_touched = [False] * len(plan.destinations)
    try:
        if torch is None:
            raise RuntimeError("PyTorch is unavailable")
        native = plan._acquisition_plan._owner._native_fiber_axis_receipt
        device = torch.device(plan._acquisition_plan.declared_device)
        accumulators: list[V1SignedWorkVariableShellSourceCUDAAccumulator] = []
        block_support_accumulators: list[
            V1SignedWorkVariableShellSourceCUDAAccumulator | None
        ] = []
        for destination in plan.destinations:
            frozen = _exact_parent_to_child_removal_primal(
                parent_mask=plan._acquisition_plan._owner._current_mask,
                action=destination.action,
                device=device,
            )
            kwargs: dict[str, Any] = {}
            if plan._private_reference:
                kwargs = {
                    "device_total_memory_bytes": 1 << 40,
                    "device_free_memory_bytes": 1 << 40,
                    "current_reserved_device_bytes": 0,
                }
            accumulator = V1SignedWorkVariableShellSourceCUDAAccumulator(
                registration=destination.source_registration,
                native_fiber_axis_receipt=native,
                frozen_primal=frozen,
                design=destination.design,
                declared_device=str(device),
                **kwargs,
            )
            block_accumulator = None
            if destination.block_support_binding is not None:
                block_accumulator = V1SignedWorkVariableShellSourceCUDAAccumulator(
                    registration=(
                        destination.block_support_binding.source_registration
                    ),
                    native_fiber_axis_receipt=native,
                    frozen_primal=frozen,
                    design=destination.design,
                    declared_device=str(device),
                    **kwargs,
                )
            del frozen
            accumulators.append(accumulator)
            block_support_accumulators.append(block_accumulator)

        offset = acquisition.receipt.physical_call_offset
        count = len(acquisition.receipt.physical_roles)
        attachment = acquisition.group_commit.attachment
        a_shards = attachment.corrected_a_attachments[offset : offset + count]
        j_shards = attachment.literal_j_attachments[offset : offset + count]
        physical_axes = _logical_transport_axes(plan._acquisition_plan)[0]
        if (
            len(physical_axes) != count
            or tuple(role for role, _axes in physical_axes)
            != acquisition.receipt.physical_roles
        ):
            _refuse(
                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                "acquired physical rows no longer replay the logical-look transport axis",
            )
        raw_reads = 0
        uploads = 0
        binding_count = 0
        block_binding_count = 0
        block_context_cursors = [0] * len(plan.destinations)
        block_packet_counts = [0] * len(plan.destinations)
        block_chunk_counts = [0] * len(plan.destinations)
        block_event_counts = [0] * len(plan.destinations)
        maximum_native_fiber_carry = 0
        maximum_route_chunk = min(
            value.design.route_chunk_size for value in plan.destinations
        )
        for (role, axes), a_shard, j_shard in zip(
            physical_axes, a_shards, j_shards, strict=True
        ):
            for accumulator in accumulators:
                accumulator.begin_packet(
                    context_ids=axes[0],
                    draw_ids=axes[1],
                    measurement_ids=axes[2],
                    capture_ids=axes[3],
                    source_packet_hashes=axes[4],
                )
            for index, (destination, accumulator) in enumerate(
                zip(
                    plan.destinations,
                    block_support_accumulators,
                    strict=True,
                )
            ):
                if accumulator is None:
                    continue
                block = destination.block_support_binding
                if block is None:
                    raise RuntimeError("block-support accumulator lost its sink binding")
                registration = block.source_registration
                accumulator.begin_packet(
                    context_ids=axes[0],
                    draw_ids=axes[1],
                    measurement_ids=axes[2],
                    capture_ids=axes[3],
                    source_packet_hashes=axes[4],
                )
                # Mark the external replay coordinator as touched before the
                # hook.  Failure cleanup must then obtain its exact abort
                # witness; a throwing sink is never reported reusable.
                block_support_sink_touched[index] = True
                block.sink_binding.sink.begin_bound_packet(
                    source_registration=registration,
                    context_start_ordinal=block_context_cursors[index],
                    context_ids=axes[0],
                    draw_ids=axes[1],
                    measurement_ids=axes[2],
                    capture_ids=axes[3],
                    source_packet_hashes=axes[4],
                )
                block_event_counts[index] += 1
            a_iter = a_shard.iter_route_windows(verify=True)
            j_iter = j_shard.iter_route_windows(verify=True)
            sentinel = object()
            pending_start: int | None = None
            pending_stop: int | None = None
            pending_a_parts: list[Any] = []
            pending_j_parts: list[Any] = []

            def emit_available_chunks(*, final: bool) -> None:
                nonlocal binding_count, block_binding_count
                nonlocal pending_start, pending_stop
                nonlocal pending_a_parts, pending_j_parts
                nonlocal maximum_native_fiber_carry
                while pending_start is not None and pending_stop is not None:
                    stop = _next_native_fiber_stop(
                        native,
                        pending_start,
                        pending_stop,
                        maximum_route_chunk,
                    )
                    if stop is None:
                        break
                    if len(pending_a_parts) == 1:
                        pending_a = pending_a_parts[0]
                        pending_j = pending_j_parts[0]
                    else:
                        pending_a = torch.cat(tuple(pending_a_parts), dim=1)
                        pending_j = torch.cat(tuple(pending_j_parts), dim=1)
                    width = stop - pending_start
                    a_chunk = pending_a[:, :width].contiguous()
                    j_chunk = pending_j[:, :width].contiguous()
                    for accumulator in accumulators:
                        binding = accumulator.issue_route_chunk_binding(
                            route_start=pending_start,
                            corrected_evidence_rows=a_chunk,
                            literal_j_rows=j_chunk,
                            route_axis_hash=plan._acquisition_plan.group_plan.route_axis_hash,
                            route_id_axis_hash=plan._acquisition_plan.group_plan.route_id_axis_hash,
                            same_capture_vjp_source_hash=plan._acquisition_plan._owner.same_capture_vjp_source_hash,
                            corrected_evidence_stream_source_hash=plan._acquisition_plan._owner.corrected_evidence_stream_source_hash,
                            literal_j_stream_source_hash=plan._acquisition_plan._owner.residual_literal_j_stream_source_hash,
                        )
                        accumulator.consume_route_chunk(
                            route_start=pending_start,
                            corrected_evidence_rows=a_chunk,
                            literal_j_rows=j_chunk,
                            route_axis_hash=plan._acquisition_plan.group_plan.route_axis_hash,
                            route_id_axis_hash=plan._acquisition_plan.group_plan.route_id_axis_hash,
                            same_capture_vjp_source_hash=plan._acquisition_plan._owner.same_capture_vjp_source_hash,
                            corrected_evidence_stream_source_hash=plan._acquisition_plan._owner.corrected_evidence_stream_source_hash,
                            literal_j_stream_source_hash=plan._acquisition_plan._owner.residual_literal_j_stream_source_hash,
                            chunk_binding=binding,
                        )
                        binding_count += 1
                    for index, (destination, accumulator) in enumerate(
                        zip(
                            plan.destinations,
                            block_support_accumulators,
                            strict=True,
                        )
                    ):
                        if accumulator is None:
                            continue
                        block = destination.block_support_binding
                        if block is None:
                            raise RuntimeError(
                                "block-support accumulator lost its sink binding"
                            )
                        registration = block.source_registration
                        block_chunk_binding = accumulator.issue_route_chunk_binding(
                            route_start=pending_start,
                            corrected_evidence_rows=a_chunk,
                            literal_j_rows=j_chunk,
                            route_axis_hash=plan._acquisition_plan.group_plan.route_axis_hash,
                            route_id_axis_hash=plan._acquisition_plan.group_plan.route_id_axis_hash,
                            same_capture_vjp_source_hash=plan._acquisition_plan._owner.same_capture_vjp_source_hash,
                            corrected_evidence_stream_source_hash=plan._acquisition_plan._owner.corrected_evidence_stream_source_hash,
                            literal_j_stream_source_hash=plan._acquisition_plan._owner.residual_literal_j_stream_source_hash,
                        )
                        accumulator.consume_route_chunk(
                            route_start=pending_start,
                            corrected_evidence_rows=a_chunk,
                            literal_j_rows=j_chunk,
                            route_axis_hash=plan._acquisition_plan.group_plan.route_axis_hash,
                            route_id_axis_hash=plan._acquisition_plan.group_plan.route_id_axis_hash,
                            same_capture_vjp_source_hash=plan._acquisition_plan._owner.same_capture_vjp_source_hash,
                            corrected_evidence_stream_source_hash=plan._acquisition_plan._owner.corrected_evidence_stream_source_hash,
                            literal_j_stream_source_hash=plan._acquisition_plan._owner.residual_literal_j_stream_source_hash,
                            chunk_binding=block_chunk_binding,
                        )
                        versions = (
                            int(getattr(a_chunk, "_version", -1)),
                            int(getattr(j_chunk, "_version", -1)),
                        )
                        block_support_sink_touched[index] = True
                        block.sink_binding.sink.consume_bound_route_chunk(
                            source_registration=registration,
                            route_start=pending_start,
                            corrected_evidence_rows=a_chunk,
                            literal_j_rows=j_chunk,
                            chunk_binding=block_chunk_binding,
                        )
                        block_event_counts[index] += 1
                        block_chunk_counts[index] += 1
                        block_binding_count += 1
                        if versions != (
                            int(getattr(a_chunk, "_version", -1)),
                            int(getattr(j_chunk, "_version", -1)),
                        ):
                            _refuse(
                                V1SampledV7SharedDW2FanoutRefusalCode.STALE_SOURCE,
                                "block-support sink mutated shared A/J tensor custody",
                            )
                        if (
                            plan._failure_injection
                            == "after_first_block_sink_chunk"
                            and index == 0
                            and block_chunk_counts[index] == 1
                        ):
                            raise RuntimeError(
                                "injected failure after first block-support sink chunk"
                            )
                    if width == int(pending_a.shape[1]):
                        pending_start = None
                        pending_stop = None
                        pending_a_parts = []
                        pending_j_parts = []
                    else:
                        pending_start = stop
                        pending_a_parts = [pending_a[:, width:].contiguous()]
                        pending_j_parts = [pending_j[:, width:].contiguous()]
                    del a_chunk, j_chunk, pending_a, pending_j
                if final and pending_start is not None:
                    _refuse(
                        V1SampledV7SharedDW2FanoutRefusalCode.AXIS_MISMATCH,
                        "durable row windows ended before a complete native-fiber tile",
                    )
                if pending_start is not None and pending_stop is not None:
                    maximum_native_fiber_carry = max(
                        maximum_native_fiber_carry,
                        pending_stop - pending_start,
                    )

            for a_item, j_item in zip_longest(a_iter, j_iter, fillvalue=sentinel):
                if a_item is sentinel or j_item is sentinel:
                    raise RuntimeError("corrected-A/literal-J durable window counts differ")
                a_start, a_stop, a_rows = a_item
                j_start, j_stop, j_rows = j_item
                if (a_start, a_stop) != (j_start, j_stop):
                    raise RuntimeError("corrected-A/literal-J durable windows are misaligned")
                a_tensor = torch.tensor(a_rows, dtype=torch.float64, device=device)
                j_tensor = torch.tensor(j_rows, dtype=torch.float64, device=device)
                raw_reads += 2
                uploads += 2
                if pending_start is None:
                    pending_start = a_start
                    pending_stop = a_stop
                    pending_a_parts = [a_tensor]
                    pending_j_parts = [j_tensor]
                else:
                    if pending_stop != a_start:
                        raise RuntimeError("durable row windows are not contiguous")
                    pending_stop = a_stop
                    pending_a_parts.append(a_tensor)
                    pending_j_parts.append(j_tensor)
                emit_available_chunks(final=False)
                del a_tensor, j_tensor
            emit_available_chunks(final=True)
            for accumulator in accumulators:
                accumulator.end_packet()
            for index, (destination, accumulator) in enumerate(
                zip(
                    plan.destinations,
                    block_support_accumulators,
                    strict=True,
                )
            ):
                if accumulator is None:
                    continue
                block = destination.block_support_binding
                if block is None:
                    raise RuntimeError("block-support accumulator lost its sink binding")
                accumulator.end_packet()
                block_support_sink_touched[index] = True
                block.sink_binding.sink.end_bound_packet(
                    source_registration=block.source_registration,
                )
                block_event_counts[index] += 1
                block_packet_counts[index] += 1
                block_context_cursors[index] += len(axes[0])
        for index, accumulator in enumerate(accumulators):
            handles.append(accumulator.seal())
            if plan._failure_injection == "after_first_source_seal" and index == 0:
                raise RuntimeError("injected failure after first compact source seal")
        block_bridge_results: list[
            V1SampledVariableShellSourceBridgeResult | None
        ] = []
        for index, (destination, accumulator) in enumerate(
            zip(plan.destinations, block_support_accumulators, strict=True)
        ):
            if accumulator is None:
                block_bridge_results.append(None)
                continue
            block = destination.block_support_binding
            if block is None:
                raise RuntimeError("block-support accumulator lost its sink binding")
            registration = block.source_registration
            if block_context_cursors[index] != registration.context_count:
                raise RuntimeError(
                    "block-support residual source did not consume its exact context axis"
                )
            block_handle = accumulator.seal()
            block_support_handles[index] = block_handle
            bridge_receipt = V1SampledVariableShellSourceBridgeReceipt(
                registration_hash=registration.registration_hash,
                compact_source_receipt_hash=block_handle.receipt.receipt_hash,
                route_axis_hash=registration.route_axis_hash,
                route_id_axis_hash=registration.route_id_axis_hash,
                destination_transport_hash=(
                    plan._acquisition_plan.destination_transport_hash
                ),
                same_capture_vjp_source_hash=(
                    registration.same_capture_vjp_source_hash
                ),
                corrected_evidence_stream_source_hash=(
                    registration.corrected_evidence_stream_source_hash
                ),
                literal_j_stream_source_hash=registration.literal_j_stream_source_hash,
                context_count=block_context_cursors[index],
                packet_count=block_packet_counts[index],
                paired_route_chunk_count=block_chunk_counts[index],
                bound_chunk_sink_axis_hash=(
                    block.sink_binding.singleton_sink_axis_hash
                ),
                bound_chunk_sink_count=1,
                bound_chunk_sink_event_count=block_event_counts[index],
                maximum_native_fiber_carry_routes_per_context=(
                    maximum_native_fiber_carry
                ),
            )
            block_bridge_results.append(
                V1SampledVariableShellSourceBridgeResult(
                    source_handle=block_handle,
                    receipt=bridge_receipt,
                )
            )
        results: list[V1SampledV7SharedDW2FanoutDestinationResult] = []
        for index, (destination, handle, block_bridge) in enumerate(
            zip(
                plan.destinations,
                handles,
                block_bridge_results,
                strict=True,
            )
        ):
            append_receipt = destination.append_owner.append_registered_packet(
                registration=destination.append_registration,
                source=handle,
            )
            if plan._failure_injection == "after_first_append" and index == 0:
                raise RuntimeError("injected failure after first destination append")
            results.append(
                V1SampledV7SharedDW2FanoutDestinationResult(
                    action_hash=destination.action.action_hash,
                    physical_action_identity_hash=destination.action.physical_action_identity_hash,
                    source_receipt_hash=handle.receipt.receipt_hash,
                    append_receipt_hash=append_receipt.receipt_hash,
                    chunk_binding_axis_hash=handle.receipt.chunk_binding_axis_hash,
                    branch_alias_axis=destination.action.branch_alias_axis,
                    branch_alias_hashes=destination.action.branch_alias_hashes,
                    source_handle=handle,
                    append_owner=destination.append_owner,
                    block_support_bridge_result=block_bridge,
                    action_specific_block_support_sink_integrated=(
                        block_bridge is not None
                    ),
                    full_action_specific_sink_fanout_complete=(
                        block_bridge is not None
                    ),
                )
            )
        if len(results) != len(plan.destinations):
            raise RuntimeError("fanout destination append axis is partial")
    except BaseException as error:
        cleanup = _cleanup_failed_fanout(
            plan=plan,
            handles=handles,
            block_support_handles=block_support_handles,
            block_support_sink_touched=block_support_sink_touched,
            original_error=error,
        )
        plan._lease.abort()
        code = (
            V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_ABORT
            if cleanup.all_destinations_cleaned
            else V1SampledV7SharedDW2FanoutRefusalCode.FAILURE_ATOMIC_CLEANUP_INCOMPLETE
        )
        raise V1SampledV7SharedDW2FanoutRefusal(
            code,
            (
                "fanout produced no result and exact destination cleanup completed"
                if cleanup.all_destinations_cleaned
                else "fanout produced no result but one or more exact destinations retain cleanup debt"
            ),
            cleanup_receipt=cleanup,
        ) from error

    result_axis = tuple(results)
    receipt = V1SampledV7SharedDW2FanoutReceipt(
        plan_hash=plan.plan_hash,
        acquisition_receipt_hash=acquisition.receipt.receipt_hash,
        physical_action_axis_hash=plan.physical_action_axis_hash,
        source_receipt_hashes=tuple(value.source_receipt_hash for value in result_axis),
        append_receipt_hashes=tuple(value.append_receipt_hash for value in result_axis),
        chunk_binding_axis_hashes=tuple(value.chunk_binding_axis_hash for value in result_axis),
        unique_physical_action_count=len(result_axis),
        distinct_branch_alias_count=len(plan.branch_alias_to_physical_action),
        raw_role_tile_read_count=raw_reads,
        device_role_tile_upload_count=uploads,
        destination_local_chunk_binding_count=binding_count,
        block_support_bridge_receipt_hashes=tuple(
            value.block_support_bridge_result.receipt.receipt_hash
            for value in result_axis
            if value.block_support_bridge_result is not None
        ),
        block_support_sink_axis_hash=plan.block_support_sink_axis_hash,
        block_support_destination_local_chunk_binding_count=block_binding_count,
        block_support_bound_sink_event_count=sum(block_event_counts),
        action_specific_block_support_sink_integrated=(
            plan.block_support_sink_axis_preregistered
        ),
        full_action_specific_sink_fanout_complete=(
            plan.block_support_sink_axis_preregistered
        ),
        _issuance_token=_RESULT_TOKEN,
    )
    plan._lease.complete()
    return V1SampledV7SharedDW2FanoutResult(
        receipt=receipt,
        destinations=result_axis,
        _issuance_token=_RESULT_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID",
    "SAMPLED_V7_SHARED_DW2_FANOUT_AUTHORITY",
    "SAMPLED_V7_SHARED_DW2_FANOUT_PLAN_SCHEMA",
    "SAMPLED_V7_SHARED_DW2_FANOUT_RECEIPT_SCHEMA",
    "SAMPLED_V7_SHARED_DW2_BLOCK_SUPPORT_BINDING_SCHEMA",
    "V1SampledV7SharedDW2BlockSupportDestinationBinding",
    "V1SampledV7SharedDW2FanoutDestination",
    "V1SampledV7SharedDW2FanoutDestinationResult",
    "V1SampledV7SharedDW2FanoutCleanupItem",
    "V1SampledV7SharedDW2FanoutCleanupReceipt",
    "V1SampledV7SharedDW2FanoutPlan",
    "V1SampledV7SharedDW2FanoutReceipt",
    "V1SampledV7SharedDW2FanoutRefusal",
    "V1SampledV7SharedDW2FanoutRefusalCode",
    "V1SampledV7SharedDW2FanoutResult",
    "V1SampledV7SharedDW2PhysicalAction",
    "execute_v1_sampled_v7_shared_dw2_fanout",
    "issue_v1_sampled_v7_shared_dw2_physical_action",
    "prepare_v1_sampled_v7_shared_dw2_block_support_fanout",
    "prepare_v1_sampled_v7_shared_dw2_fanout",
]
