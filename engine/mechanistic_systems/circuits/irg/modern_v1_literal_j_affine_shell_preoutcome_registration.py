"""Pre-outcome registration of one structured literal-J direction family.

The affine shell family ``(z_outside, e_b1, ..., e_bB)`` must be frozen before
the residual D_W2 rows that will be projected onto it are observed.  This
module seals that identity from issuer-owned sampled-v7 parents.  It stores no
route-scale arrays and does not allocate or spend alpha: the already-registered
sampled-v7 literal-J cell is reused verbatim.

The receipt is identity custody only.  In particular, it is not a population
projection, confidence statement, FD replacement, D-stability decision, or
C/K/O/commit authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import math
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_shared_dw2_fanout import (
    SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID,
    V1SampledV7SharedDW2PhysicalAction,
)
from .modern_v1_sampled_variable_shell_packet_registration import (
    v1_sampled_variable_shell_planned_prefix_hash,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    open_v1_sampled_variable_shell_dw2_suffix_cursor,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRegistration,
)


LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_J_affine_shell_direction_preoutcome_"
    "registration_v3"
)
LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY: Final = (
    "preoutcome_structured_literal_J_direction_family_identity_only_existing_"
    "sampled_v7_J_cell_reused_no_arrays_outcomes_new_alpha_population_FD_"
    "solver_D_C_K_O_action_commit_terminal_performance_or_pareto_authority_v1"
)
LITERAL_J_AFFINE_SHELL_PREOUTCOME_DIRECTION_RULE: Final = (
    "parent_to_child_removal_indicator_zero_on_frozen_shell_then_ordered_shell_"
    "coordinate_basis_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-affine-shell-preoutcome-"
_ISSUER_TOKEN: Final = object()
_MAXIMUM_SHELL_SIZE: Final = 8


class V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    ACTION_MISMATCH = "action_mismatch"
    CHART_MISMATCH = "chart_mismatch"
    ROUTE_AXIS_MISMATCH = "route_axis_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    SHELL_MISMATCH = "shell_mismatch"
    SOURCE_STREAM_MISMATCH = "source_stream_mismatch"
    ALPHA_CELL_MISMATCH = "alpha_cell_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_REGISTRATION = "mutated_registration"


class V1LiteralJAffineShellPreoutcomeRegistrationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode:
            raise TypeError("affine-shell pre-outcome refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJAffineShellPreoutcomeRegistrationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _alpha(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ALPHA_CELL_MISMATCH,
            "the existing literal-J cell alpha must be one positive finite probability",
        )
    return value


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            f"{name} must be one immutable target/complement count pair",
        )
    result = tuple(
        _nonnegative_int(item, name=f"{name} item") for item in value
    )
    return result  # type: ignore[return-value]


def _sha256(value: object, *, name: str) -> str:
    digest = _identity(value, name=name)
    if len(digest) != 64 or digest != digest.lower():
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one lowercase SHA-256 digest",
        )
    try:
        int(digest, 16)
    except ValueError:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one lowercase SHA-256 digest",
        )
    return digest


def _ordered_direction_ids(shell_route_ids: tuple[str, ...]) -> tuple[str, ...]:
    return ("literal-j:outside",) + tuple(
        f"literal-j:shell-basis:{route_id}" for route_id in shell_route_ids
    )


def _packed_sha256(mask: HardMask) -> str:
    return hashlib.sha256(mask.packed_bits).hexdigest()


def _mask_bit(mask: HardMask, route_index: int) -> bool:
    byte = mask.packed_bits[route_index // 8]
    return bool(byte & (1 << (7 - (route_index % 8))))


def _matching_schedule_packet_axes(
    *,
    schedule_packets: tuple[object, ...],
    source_registration: V1SignedWorkVariableShellSourceRegistration,
) -> tuple[object, ...]:
    """Resolve one exact contiguous transport axis beneath a logical D look.

    A statistical look may be physically transported by several bounded
    scorer packets.  The variable-shell source registration stores their
    packet-major concatenated axes, so identity must be established against
    the complete contiguous schedule interval rather than a single packet.
    """

    fields = (
        "registered_context_ids",
        "registered_draw_ids",
        "registered_measurement_ids",
        "registered_capture_ids",
        "registered_source_packet_hashes",
    )
    expected = tuple(tuple(getattr(source_registration, name)) for name in fields)
    matches: list[tuple[object, ...]] = []
    for start in range(len(schedule_packets)):
        for stop in range(start + 1, len(schedule_packets) + 1):
            candidate = schedule_packets[start:stop]
            observed = tuple(
                tuple(
                    value
                    for packet in candidate
                    for value in tuple(getattr(packet, name))
                )
                for name in fields
            )
            if observed == expected:
                matches.append(candidate)
    if len(matches) != 1:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            "source axes do not select exactly one contiguous issuer-owned "
            "residual transport interval",
        )
    return matches[0]


def _coordinates(
    *,
    source_registration: V1SignedWorkVariableShellSourceRegistration,
    action: V1SampledV7SharedDW2PhysicalAction,
    parent_mask: HardMask,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
    literal_j_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
) -> dict[str, object]:
    exact = (
        (
            source_registration,
            V1SignedWorkVariableShellSourceRegistration,
            "variable-shell source registration",
        ),
        (action, V1SampledV7SharedDW2PhysicalAction, "physical action"),
        (parent_mask, HardMask, "parent/current hard mask"),
        (
            native_fiber_axis,
            V1NativeDestinationFiberAxisReceipt,
            "native-fiber axis receipt",
        ),
        (
            literal_j_owner_binding,
            V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
            "sampled-v7 literal-J owner binding",
        ),
    )
    if any(type(value) is not expected for value, expected, _ in exact):
        wrong = next(
            name for value, expected, name in exact if type(value) is not expected
        )
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
            f"registration requires the exact {wrong}",
        )

    try:
        action_hash = action.assert_integrity()
        native_fiber_axis.assert_integrity()
        owner_hash = literal_j_owner_binding.assert_live()
        source_hash = source_registration.registration_hash
    except V1LiteralJAffineShellPreoutcomeRegistrationRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PARENT_INTEGRITY,
            "one issuer-owned parent failed integrity replay: "
            f"{type(error).__name__}: {error}",
        )

    owner = literal_j_owner_binding._measurement_owner
    lineage = owner.measurement_source_lineage
    suffix = owner._suffix_registration
    schedule = owner.schedule
    shell = owner._shell_union
    owner_native = owner._native_fiber_axis_receipt
    owner_parent = owner._current_mask

    if (
        source_registration.model_identity_hash
        != literal_j_owner_binding.model_identity_hash
        or source_registration.chart_identity_hash
        != literal_j_owner_binding.chart_identity_hash
        or source_registration.current_mask_hash
        != literal_j_owner_binding.current_mask_hash
        or source_registration.proposal_generation
        != literal_j_owner_binding.proposal_generation
        or source_registration.model_identity_hash != lineage.model_identity_hash
        or source_registration.chart_identity_hash != lineage.chart_identity_hash
        or source_registration.current_mask_hash != lineage.current_mask_hash
        or source_registration.proposal_generation != lineage.proposal_generation
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.CHART_MISMATCH,
            "source, action owner, and sampled-v7 literal-J cell do not share one chart",
        )

    if (
        parent_mask.mask_hash != source_registration.current_mask_hash
        or parent_mask.mask_hash != action.parent_mask_hash
        or parent_mask.mask_hash != owner_parent.mask_hash
        or parent_mask.route_axis_hash != source_registration.route_axis_hash
        or parent_mask.route_count != native_fiber_axis.route_count
        or parent_mask.route_axis_hash != native_fiber_axis.route_axis_hash
        or source_registration.route_axis_hash
        != literal_j_owner_binding.physical_route_axis_hash
        or source_registration.route_axis_hash != lineage.route_axis_hash
        or source_registration.route_id_axis_hash != lineage.route_id_axis_hash
        or source_registration.route_id_axis_hash
        != native_fiber_axis.route_id_axis_hash
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ROUTE_AXIS_MISMATCH,
            "parent mask, source, native fibers, and literal-J owner left one physical route axis",
        )
    if (
        parent_mask.packed_bits != owner_parent.packed_bits
        or native_fiber_axis.receipt_hash != owner_native.receipt_hash
        or native_fiber_axis.measure_hash != owner_native.measure_hash
        or source_registration.native_fiber_axis_receipt_hash
        != native_fiber_axis.receipt_hash
        or source_registration.native_fiber_measure_hash
        != native_fiber_axis.measure_hash
        or native_fiber_axis.route_count != literal_j_owner_binding.route_count
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ROUTE_AXIS_MISMATCH,
            "exact parent bytes, native-fiber measure, or route count changed",
        )

    packed_action = action._packed_action
    child = packed_action.resulting_mask
    try:
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            parent_mask
        )
    except Exception as error:
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ACTION_MISMATCH,
            f"action is not an irreversible child of the parent: {error}",
        )
    if (
        source_registration.kkt_branch_id
        != SAMPLED_V7_PHYSICAL_ACTION_FANOUT_BRANCH_ID
        or source_registration.kkt_branch_source_receipt_hash != action_hash
        or action.action_hash != action_hash
        or packed_action.action_hash != action.packed_action_hash
        or child.mask_hash != action.resulting_mask_hash
        or removed_count != action.removed_count
        or removed_hash != action.removed_route_index_hash
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ACTION_MISMATCH,
            "source registration, packed child, and physical-action custody differ",
        )

    shell_indices = tuple(source_registration.shell_indices)
    shell_route_ids = tuple(source_registration.shell_route_ids)
    if (
        not 1 <= len(shell_indices) <= _MAXIMUM_SHELL_SIZE
        or shell_indices != tuple(shell.shell_indices)
        or shell_route_ids != tuple(shell.shell_route_ids)
        or source_registration.shell_freeze_receipt_hash != shell.receipt_hash
        or any(index >= parent_mask.route_count for index in shell_indices)
        or any(not _mask_bit(parent_mask, index) for index in shell_indices)
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SHELL_MISMATCH,
            "the ordered shell is not the live, pre-outcome, retained-route shell",
        )
    direction_ids = _ordered_direction_ids(shell_route_ids)

    stream_fields = (
        (
            source_registration.witness_stream_source_hash,
            literal_j_owner_binding.residual_witness_stream_source_hash,
            lineage.witness_stream_source_hash,
        ),
        (
            source_registration.same_capture_vjp_source_hash,
            literal_j_owner_binding.same_capture_vjp_source_hash,
            lineage.same_capture_vjp_source_hash,
        ),
        (
            source_registration.corrected_evidence_stream_source_hash,
            literal_j_owner_binding.corrected_evidence_stream_source_hash,
            lineage.corrected_evidence_stream_source_hash,
        ),
        (
            source_registration.literal_j_stream_source_hash,
            literal_j_owner_binding.literal_j_stream_source_hash,
            lineage.literal_j_stream_source_hash,
        ),
    )
    if (
        any(len(set(axis)) != 1 for axis in stream_fields)
        or source_registration.alpha_lineage_hash
        != literal_j_owner_binding.historical_owner_alpha_lineage_hash
        or len(
            {
                source_registration.witness_stream_source_hash,
                source_registration.same_capture_vjp_source_hash,
                source_registration.corrected_evidence_stream_source_hash,
                source_registration.literal_j_stream_source_hash,
            }
        )
        != 4
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SOURCE_STREAM_MISMATCH,
            "witness, corrected-evidence, same-capture VJP, or literal-J stream changed",
        )

    if (
        schedule.schedule_hash != literal_j_owner_binding.measurement_schedule_hash
        or suffix.receipt_hash != literal_j_owner_binding.suffix_registration_hash
        or source_registration.registered_d_prefix_hash
        != suffix.d1_pair_prefix_axis_hash
        or source_registration.population_count != schedule.population_count
        or source_registration.population_context_axis_hash
        != schedule.population_context_axis_hash
        or source_registration.population_draw_axis_hash
        != schedule.population_draw_axis_hash
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            "source registration left the issuer-owned D1/residual population axes",
        )

    scheduled_packets = _matching_schedule_packet_axes(
        schedule_packets=tuple(schedule.packets),
        source_registration=source_registration,
    )
    first_scheduled_packet = scheduled_packets[0]
    scheduled_packet = scheduled_packets[-1]
    scheduled_ordinals = tuple(
        int(getattr(item, "packet_ordinal")) for item in scheduled_packets
    )
    if (
        scheduled_ordinals
        != tuple(range(scheduled_ordinals[0], scheduled_ordinals[0] + len(scheduled_ordinals)))
        or any(
            left.after_suffix_stratum_counts != right.before_suffix_stratum_counts
            for left, right in zip(scheduled_packets, scheduled_packets[1:], strict=False)
        )
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            "logical-look transport packets are not one contiguous residual prefix",
        )
    cursor = open_v1_sampled_variable_shell_dw2_suffix_cursor(
        registration=suffix,
        frame=owner._frame,
        discovery_stream=owner._discovery_stream,
    )
    suffix_packets: list[object] = []
    selected_ordinals = set(scheduled_ordinals)
    for ordinal in range(scheduled_packet.packet_ordinal + 1):
        current = cursor.begin_next_packet()
        if ordinal in selected_ordinals:
            suffix_packets.append(current)
        if ordinal != scheduled_packet.packet_ordinal:
            cursor.accept_active_packet(current)
    if len(suffix_packets) != len(scheduled_packets):  # pragma: no cover
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            "registered residual transport interval could not be replayed",
        )
    suffix_packet = suffix_packets[-1]
    planned_prefix_hash = v1_sampled_variable_shell_planned_prefix_hash(
        registration=suffix,
        packet=suffix_packet,
    )
    expected_look_counts = tuple(
        d1 + residual
        for d1, residual in zip(
            literal_j_owner_binding.resident_d1_stratum_counts,
            scheduled_packet.after_suffix_stratum_counts,
            strict=True,
        )
    )
    if (
        tuple(item.suffix_packet_hash for item in scheduled_packets)
        != tuple(item.receipt_hash for item in suffix_packets)
        or source_registration.registered_d_w2_prefix_hash != planned_prefix_hash
        or literal_j_owner_binding.residual_prefix_stratum_counts
        != scheduled_packet.after_suffix_stratum_counts
        or literal_j_owner_binding.registered_look_stratum_counts
        != expected_look_counts
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
            "literal-J alpha look does not end at the registered D_W2 packet boundary",
        )

    literal_j_alpha = _alpha(literal_j_owner_binding.literal_j_cell_alpha)
    for name in (
        "literal_j_cell_id",
        "literal_j_cell_hash",
        "literal_j_cell_scope_hash",
        "literal_j_stream_binding_hash",
    ):
        _identity(getattr(literal_j_owner_binding, name), name=name)
    if (
        literal_j_owner_binding.alpha_spend_authority is not False
        or literal_j_owner_binding.registered_a_f_j_cells_bound is not True
        or literal_j_owner_binding.same_capture_streams_bound is not True
    ):
        _refuse(
            V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ALPHA_CELL_MISMATCH,
            "literal-J owner binding does not expose one inert preregistered J cell",
        )

    outside_source_hash = content_hash(
        {
            "schema": LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA,
            "direction_rule": LITERAL_J_AFFINE_SHELL_PREOUTCOME_DIRECTION_RULE,
            "physical_action_hash": action_hash,
            "parent_mask_hash": parent_mask.mask_hash,
            "resulting_mask_hash": action.resulting_mask_hash,
            "removed_route_index_hash": action.removed_route_index_hash,
            "shell_indices": shell_indices,
            "route_axis_hash": source_registration.route_axis_hash,
        },
        prefix=_PREFIX + "outside-direction-source-",
    )
    direction_family_hash = content_hash(
        {
            "schema": LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA,
            "source_registration_hash": source_hash,
            "physical_action_hash": action_hash,
            "outside_direction_source_hash": outside_source_hash,
            "ordered_direction_ids": direction_ids,
            "literal_j_cell_hash": literal_j_owner_binding.literal_j_cell_hash,
            "literal_j_cell_alpha": literal_j_alpha,
            "historical_alpha_ledger_hash": (
                literal_j_owner_binding.historical_owner_alpha_ledger_hash
            ),
            "historical_alpha_coordinate_hash": (
                literal_j_owner_binding.historical_owner_alpha_coordinate_hash
            ),
            "historical_alpha_lineage_hash": (
                literal_j_owner_binding.historical_owner_alpha_lineage_hash
            ),
            "registered_d_w2_prefix_hash": planned_prefix_hash,
        },
        prefix=_PREFIX + "direction-family-",
    )

    return {
        "source_registration_hash": source_hash,
        "source_registration_id": source_registration.registration_id,
        "physical_action_hash": action_hash,
        "physical_action_identity_hash": action.physical_action_identity_hash,
        "packed_action_hash": action.packed_action_hash,
        "parent_mask_hash": parent_mask.mask_hash,
        "parent_mask_packed_sha256": _packed_sha256(parent_mask),
        "resulting_mask_hash": action.resulting_mask_hash,
        "resulting_mask_packed_sha256": action.resulting_mask_packed_sha256,
        "removed_count": action.removed_count,
        "removed_route_index_hash": action.removed_route_index_hash,
        "model_identity_hash": source_registration.model_identity_hash,
        "chart_identity_hash": source_registration.chart_identity_hash,
        "current_mask_hash": source_registration.current_mask_hash,
        "proposal_generation": source_registration.proposal_generation,
        "route_axis_hash": source_registration.route_axis_hash,
        "route_id_axis_hash": source_registration.route_id_axis_hash,
        "route_count": native_fiber_axis.route_count,
        "native_fiber_axis_receipt_hash": native_fiber_axis.receipt_hash,
        "native_fiber_measure_hash": native_fiber_axis.measure_hash,
        "shell_freeze_receipt_hash": source_registration.shell_freeze_receipt_hash,
        "shell_indices": shell_indices,
        "shell_route_ids": shell_route_ids,
        "ordered_direction_ids": direction_ids,
        "outside_direction_source_hash": outside_source_hash,
        "direction_family_hash": direction_family_hash,
        "registered_d_prefix_hash": source_registration.registered_d_prefix_hash,
        "registered_d_w2_prefix_hash": planned_prefix_hash,
        "measurement_schedule_hash": schedule.schedule_hash,
        "suffix_registration_hash": suffix.receipt_hash,
        "schedule_packet_hash": scheduled_packet.packet_schedule_hash,
        "schedule_packet_ordinal": scheduled_packet.packet_ordinal,
        "transport_schedule_packet_hashes": tuple(
            item.packet_schedule_hash for item in scheduled_packets
        ),
        "transport_schedule_packet_ordinals": scheduled_ordinals,
        "transport_suffix_packet_hashes": tuple(
            item.receipt_hash for item in suffix_packets
        ),
        "before_suffix_stratum_counts": (
            first_scheduled_packet.before_suffix_stratum_counts
        ),
        "after_suffix_stratum_counts": scheduled_packet.after_suffix_stratum_counts,
        "registered_look_index": literal_j_owner_binding.registered_look_index,
        "registered_look_stratum_counts": (
            literal_j_owner_binding.registered_look_stratum_counts
        ),
        "registered_prefix_axis_hash": (
            literal_j_owner_binding.registered_prefix_axis_hash
        ),
        "residual_prefix_axis_hash": (
            literal_j_owner_binding.residual_prefix_axis_hash
        ),
        "witness_stream_source_hash": source_registration.witness_stream_source_hash,
        "same_capture_vjp_source_hash": (
            source_registration.same_capture_vjp_source_hash
        ),
        "corrected_evidence_stream_source_hash": (
            source_registration.corrected_evidence_stream_source_hash
        ),
        "literal_j_stream_source_hash": (
            source_registration.literal_j_stream_source_hash
        ),
        "literal_j_stream_binding_hash": (
            literal_j_owner_binding.literal_j_stream_binding_hash
        ),
        "literal_j_owner_binding_receipt_hash": owner_hash,
        "literal_j_cell_id": literal_j_owner_binding.literal_j_cell_id,
        "literal_j_cell_hash": literal_j_owner_binding.literal_j_cell_hash,
        "literal_j_cell_scope_hash": (
            literal_j_owner_binding.literal_j_cell_scope_hash
        ),
        "literal_j_cell_alpha": literal_j_alpha,
        "historical_alpha_ledger_hash": (
            literal_j_owner_binding.historical_owner_alpha_ledger_hash
        ),
        "historical_alpha_coordinate_hash": (
            literal_j_owner_binding.historical_owner_alpha_coordinate_hash
        ),
        "historical_alpha_lineage_hash": (
            literal_j_owner_binding.historical_owner_alpha_lineage_hash
        ),
    }


@dataclass(frozen=True, slots=True)
class V1LiteralJAffineShellDirectionPreoutcomeRegistration:
    source_registration_hash: str
    source_registration_id: str
    physical_action_hash: str
    physical_action_identity_hash: str
    packed_action_hash: str
    parent_mask_hash: str
    parent_mask_packed_sha256: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_count: int
    removed_route_index_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    ordered_direction_ids: tuple[str, ...]
    outside_direction_source_hash: str
    direction_family_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    measurement_schedule_hash: str
    suffix_registration_hash: str
    schedule_packet_hash: str
    schedule_packet_ordinal: int
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_schedule_packet_ordinals: tuple[int, ...]
    transport_suffix_packet_hashes: tuple[str, ...]
    before_suffix_stratum_counts: tuple[int, int]
    after_suffix_stratum_counts: tuple[int, int]
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    residual_prefix_axis_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    literal_j_stream_binding_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_cell_alpha: float
    historical_alpha_ledger_hash: str
    historical_alpha_coordinate_hash: str
    historical_alpha_lineage_hash: str
    registered_before_d_w2_outcomes: bool = True
    exact_typed_parent_graph_bound: bool = True
    ordered_direction_family_frozen: bool = True
    existing_literal_j_cell_reused: bool = True
    array_free_durable_registration: bool = True
    array_payload_present: bool = False
    outcome_payload_present: bool = False
    alpha_allocation_performed: bool = False
    alpha_spend_authority: bool = False
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    direction_rule: str = LITERAL_J_AFFINE_SHELL_PREOUTCOME_DIRECTION_RULE
    authority_scope: str = LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY
    schema: str = LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.INVALID_TYPE,
                "affine-shell pre-outcome registrations are issuer-sealed",
            )
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            if item.type is str or isinstance(getattr(self, item.name), str):
                _identity(getattr(self, item.name), name=item.name)
        removed = _positive_int(self.removed_count, name="removed_count")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        _nonnegative_int(self.schedule_packet_ordinal, name="schedule_packet_ordinal")
        _positive_int(self.registered_look_index, name="registered_look_index")
        if any(
            type(axis) is not tuple
            for axis in (
                self.transport_schedule_packet_hashes,
                self.transport_schedule_packet_ordinals,
                self.transport_suffix_packet_hashes,
            )
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
                "logical-look transport coordinates must be immutable tuple axes",
            )
        transport_schedule_hashes = tuple(
            _identity(value, name="transport schedule packet hash")
            for value in self.transport_schedule_packet_hashes
        )
        transport_ordinals = tuple(
            _nonnegative_int(value, name="transport schedule packet ordinal")
            for value in self.transport_schedule_packet_ordinals
        )
        transport_suffix_hashes = tuple(
            _identity(value, name="transport suffix packet hash")
            for value in self.transport_suffix_packet_hashes
        )
        if (
            not transport_ordinals
            or len(transport_schedule_hashes) != len(transport_ordinals)
            or len(transport_suffix_hashes) != len(transport_ordinals)
            or transport_ordinals
            != tuple(
                range(
                    transport_ordinals[0],
                    transport_ordinals[0] + len(transport_ordinals),
                )
            )
            or transport_schedule_hashes[-1] != self.schedule_packet_hash
            or transport_ordinals[-1] != self.schedule_packet_ordinal
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
                "logical-look transport axes are empty, partial, reordered, or "
                "disagree with their endpoint alias",
            )
        _sha256(
            self.parent_mask_packed_sha256,
            name="parent_mask_packed_sha256",
        )
        _sha256(
            self.resulting_mask_packed_sha256,
            name="resulting_mask_packed_sha256",
        )
        before = _count_pair(
            self.before_suffix_stratum_counts,
            name="before_suffix_stratum_counts",
        )
        after = _count_pair(
            self.after_suffix_stratum_counts,
            name="after_suffix_stratum_counts",
        )
        registered_look = _count_pair(
            self.registered_look_stratum_counts,
            name="registered_look_stratum_counts",
        )
        if (
            any(after[index] < before[index] for index in (0, 1))
            or sum(after[index] - before[index] for index in (0, 1)) <= 0
            or any(registered_look[index] < after[index] for index in (0, 1))
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
                "durable D1/D_W2 count axes are not nested at one nonempty packet boundary",
            )
        if removed > routes:
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ACTION_MISMATCH,
                "removed route count exceeds the registered route axis",
            )
        if type(self.shell_indices) is not tuple:
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SHELL_MISMATCH,
                "shell indices must be one immutable ordered axis",
            )
        shell_indices = tuple(
            _nonnegative_int(item, name="shell index") for item in self.shell_indices
        )
        if type(self.shell_route_ids) is not tuple:
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SHELL_MISMATCH,
                "shell route IDs must be one immutable ordered axis",
            )
        shell_route_ids = tuple(
            _identity(item, name="shell route ID") for item in self.shell_route_ids
        )
        if type(self.ordered_direction_ids) is not tuple:
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SHELL_MISMATCH,
                "ordered direction IDs must be one immutable axis",
            )
        direction_ids = tuple(
            _identity(item, name="ordered direction ID")
            for item in self.ordered_direction_ids
        )
        if (
            not 1 <= len(shell_indices) <= _MAXIMUM_SHELL_SIZE
            or len(shell_indices) != len(shell_route_ids)
            or len(shell_indices) != len(set(shell_indices))
            or len(shell_route_ids) != len(set(shell_route_ids))
            or any(index >= routes for index in shell_indices)
            or len(direction_ids) != len(set(direction_ids))
            or len(direction_ids) != len(shell_indices) + 1
            or direction_ids != _ordered_direction_ids(shell_route_ids)
            or self.current_mask_hash != self.parent_mask_hash
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.SHELL_MISMATCH,
                "durable structured-direction coordinates do not replay",
            )
        _alpha(self.literal_j_cell_alpha)
        positive = (
            self.registered_before_d_w2_outcomes,
            self.exact_typed_parent_graph_bound,
            self.ordered_direction_family_frozen,
            self.existing_literal_j_cell_reused,
            self.array_free_durable_registration,
        )
        closed = (
            self.array_payload_present,
            self.outcome_payload_present,
            self.alpha_allocation_performed,
            self.alpha_spend_authority,
            self.population_projection_authority,
            self.confidence_authority,
            self.fd_replacement_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.action_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.direction_rule
            != LITERAL_J_AFFINE_SHELL_PREOUTCOME_DIRECTION_RULE
            or self.authority_scope
            != LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY
            or self.schema
            != LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "pre-outcome registration crossed its identity-only authority boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "registration-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.MUTATED_REGISTRATION,
                "durable pre-outcome registration hash does not replay",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> dict[str, object]:
        """Return the array-free durable payload; hash is verified first."""

        self.assert_integrity()
        return self._payload()


def register_v1_literal_j_affine_shell_direction_preoutcome(
    *,
    source_registration: V1SignedWorkVariableShellSourceRegistration,
    action: V1SampledV7SharedDW2PhysicalAction,
    parent_mask: HardMask,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
    literal_j_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
) -> V1LiteralJAffineShellDirectionPreoutcomeRegistration:
    """Seal one structured J family under an existing sampled-v7 J cell."""

    coordinates = _coordinates(
        source_registration=source_registration,
        action=action,
        parent_mask=parent_mask,
        native_fiber_axis=native_fiber_axis,
        literal_j_owner_binding=literal_j_owner_binding,
    )
    return V1LiteralJAffineShellDirectionPreoutcomeRegistration(
        **coordinates,  # type: ignore[arg-type]
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "LITERAL_J_AFFINE_SHELL_PREOUTCOME_DIRECTION_RULE",
    "LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY",
    "LITERAL_J_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA",
    "V1LiteralJAffineShellDirectionPreoutcomeRegistration",
    "V1LiteralJAffineShellPreoutcomeRegistrationRefusal",
    "V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode",
    "register_v1_literal_j_affine_shell_direction_preoutcome",
]
