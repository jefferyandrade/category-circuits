"""Atomic current-source D2 pre-outcome custody for sampled V7.

The D1 producer freezes the fitting-chart prefix.  The next physical provider
call may begin only after the current two-branch actions, one predictable
``D_W2`` packet, the registered A/F/J look, every action-local destination,
and the literal-J affine-shell direction family all name one live source.

This module performs exactly that join.  It does not create statistical
bands, reconstruct the retired Cartesian ``source_image``, accept provider
outcomes, spend alpha, certify D, open C/K/O, or authorize a mask.  The
result retains its live parents and is intentionally nonserializable.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    V1SampledV7AffineShellPreoutcomeFanoutPlan,
)
from .modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7BranchActionPair,
    issue_v1_sampled_v7_branch_action_pair,
)
from .modern_v1_sampled_v7_current_source_d1_prefix import (
    V1SampledV7CurrentSourceD1PrefixResult,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyLogicalLookMeasurementIssuance,
    V1SampledV7DOnlyMeasurementOwner,
    V1SampledV7DOnlyPacketMeasurementIssuance,
    v1_sampled_v7_d_only_logical_registered_look,
    v1_sampled_v7_d_only_transport_packet_issuances,
)
from .modern_v1_sampled_v7_shared_dw2_fanout import (
    V1SampledV7SharedDW2FanoutPlan,
)


SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_current_source_D2_preoutcome_plan_v2"
)
SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_AUTHORITY: Final = (
    "current_source_D2_preoutcome_identity_and_live_custody_only_no_provider_"
    "outcome_alpha_spend_D_C_K_O_action_commit_terminal_performance_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-current-source-D2-preoutcome-"
_ISSUER_TOKEN: Final = object()


class V1SampledV7CurrentSourceD2PreoutcomeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    D1_MISMATCH = "d1_mismatch"
    PACKET_MISMATCH = "packet_mismatch"
    ALPHA_LOOK_MISMATCH = "alpha_look_mismatch"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    DESTINATION_AXIS_MISMATCH = "destination_axis_mismatch"
    DIRECTION_AXIS_MISMATCH = "direction_axis_mismatch"
    OUTCOME_LEAKAGE = "outcome_leakage"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_PLAN = "mutated_plan"


class V1SampledV7CurrentSourceD2PreoutcomeRefusal(RuntimeError):
    """Typed fail-closed refusal at the current-source D1-to-D2 seam."""

    def __init__(
        self,
        code: V1SampledV7CurrentSourceD2PreoutcomeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CurrentSourceD2PreoutcomeRefusalCode:
            raise TypeError("D2 pre-outcome refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CurrentSourceD2PreoutcomeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CurrentSourceD2PreoutcomeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _assert_live(value: object, *, name: str) -> str:
    method = getattr(value, "assert_live", None)
    if not callable(method):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            f"{name} lacks assert_live()",
        )
    try:
        observed = method()
    except V1SampledV7CurrentSourceD2PreoutcomeRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.STALE_SOURCE,
            f"{name} failed live replay: {type(error).__name__}: {error}",
        )
    return _identity(observed, name=f"{name} hash")


def _assert_integrity(value: object, *, name: str) -> str:
    method = getattr(value, "assert_integrity", None)
    if not callable(method):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            f"{name} lacks assert_integrity()",
        )
    try:
        observed = method()
    except V1SampledV7CurrentSourceD2PreoutcomeRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.STALE_SOURCE,
            f"{name} failed integrity replay: {type(error).__name__}: {error}",
        )
    return _identity(observed, name=f"{name} hash")


def _source_coordinates_unchecked(
    *,
    live_d_context: object,
    d1_prefix: V1SampledV7CurrentSourceD1PrefixResult,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ),
    alpha_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    affine_shell_fanout_plan: V1SampledV7AffineShellPreoutcomeFanoutPlan,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> tuple[dict[str, object], V1SampledV7BranchActionPair, V1SampledV7SharedDW2FanoutPlan]:
    exact = (
        (d1_prefix, V1SampledV7CurrentSourceD1PrefixResult, "D1 prefix"),
        (measurement_owner, V1SampledV7DOnlyMeasurementOwner, "measurement owner"),
        (
            packet_issuance,
            (
                V1SampledV7DOnlyPacketMeasurementIssuance,
                V1SampledV7DOnlyLogicalLookMeasurementIssuance,
            ),
            "packet issuance",
        ),
        (
            alpha_owner_binding,
            V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
            "A/F/J owner binding",
        ),
        (
            affine_shell_fanout_plan,
            V1SampledV7AffineShellPreoutcomeFanoutPlan,
            "affine-shell fanout plan",
        ),
        (
            joint_preoutcome_registration,
            V1LiteralJJointAffineShellPreoutcomeRegistration,
            "joint affine-shell registration",
        ),
    )
    for value, expected, name in exact:
        if (
            isinstance(expected, tuple)
            and type(value) not in expected
            or not isinstance(expected, tuple)
            and type(value) is not expected
        ):
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
                f"{name} requires an exact canonical source type",
            )

    live_hash = _assert_live(live_d_context, name="live V7 D context")
    d1_hash = d1_prefix.assert_live()
    owner_hash = measurement_owner.assert_live()
    issuance_hash = packet_issuance.assert_live()
    alpha_hash = alpha_owner_binding.assert_live()
    fanout_wrapper_hash = _assert_live(
        affine_shell_fanout_plan,
        name="affine-shell fanout plan",
    )
    joint_hash = joint_preoutcome_registration.assert_integrity()

    current_source = getattr(live_d_context, "current_source", None)
    if current_source is None:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            "live V7 D context lost current-source custody",
        )
    current_source_hash = _assert_live(current_source, name="current-source D owner")
    if (
        getattr(live_d_context, "current_source_d1_prefix", None) is not d1_prefix
        or getattr(live_d_context, "d_only_measurement_owner", None)
        is not measurement_owner
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.D1_MISMATCH,
            "D1 prefix and D-only owner must be the exact leaves attached to the live context",
        )

    if (
        measurement_owner._frame
        is not getattr(current_source, "generation_sampling_frame", None)
        or measurement_owner._discovery_stream
        is not getattr(current_source, "discovery_sampling_stream", None)
        or measurement_owner._current_mask
        is not getattr(current_source, "current_mask", None)
        or measurement_owner._suffix_registration is not d1_prefix.suffix_registration
        or measurement_owner._residual_measure_design
        is not d1_prefix.residual_measure_design
        or measurement_owner._resident_d1_population
        is not d1_prefix.resident_d1_population
        or measurement_owner._shell_union is not d1_prefix.shell_union
        or d1_prefix.suffix_registration_hash
        != measurement_owner._suffix_registration.receipt_hash
        or d1_prefix.shell_union_receipt_hash
        != measurement_owner._shell_union.receipt_hash
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.D1_MISMATCH,
            "live source, D1 prefix, and measurement owner do not share one exact chart",
        )

    transport_issuances = v1_sampled_v7_d_only_transport_packet_issuances(
        packet_issuance
    )
    transport_packets = tuple(issuance._packet for issuance in transport_issuances)
    schedule_packets = tuple(
        measurement_owner.schedule.packet_for(packet) for packet in transport_packets
    )
    (
        logical_look_index,
        expected_endpoint_counts,
        logical_residual_before,
        logical_residual_after,
    ) = v1_sampled_v7_d_only_logical_registered_look(
        packet_issuance,
        measurement_owner=measurement_owner,
    )
    packet = transport_packets[-1]
    packet_hash = packet.assert_integrity()
    schedule_packet = schedule_packets[-1]
    if (
        any(issuance._owner is not measurement_owner for issuance in transport_issuances)
        or transport_packets[0].packet_ordinal != 0
        or logical_residual_before != (0, 0)
        or logical_residual_after == (0, 0)
        or packet.after_suffix_stratum_counts != logical_residual_after
        or packet.suffix_registration_hash != d1_prefix.suffix_registration_hash
        or any(
            transport_packet.suffix_registration_hash
            != d1_prefix.suffix_registration_hash
            for transport_packet in transport_packets
        )
        or getattr(packet_issuance, "measurement_owner_hash", None) != owner_hash
        or getattr(packet_issuance, "measurement_schedule_hash", None)
        != measurement_owner.schedule.schedule_hash
        or schedule_packet.suffix_packet_hash != packet_hash
        or tuple(item.suffix_packet_hash for item in schedule_packets)
        != tuple(item.receipt_hash for item in transport_packets)
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.PACKET_MISMATCH,
            "D2 logical look is not the exact predictable prefix of the D1 suffix",
        )

    alpha_family_plan = getattr(current_source, "alpha_family_plan", None)
    registered_looks = tuple(
        tuple(value)
        for value in getattr(alpha_family_plan, "registered_look_stratum_counts", ())
    )
    endpoint_look_indices = tuple(
        index
        for index, counts in enumerate(registered_looks, start=1)
        if counts == expected_endpoint_counts
    )
    if (
        alpha_owner_binding._measurement_owner is not measurement_owner
        or alpha_owner_binding._alpha_family_plan is not alpha_family_plan
        or alpha_owner_binding._alpha_family_registry
        is not getattr(current_source, "alpha_family_registry", None)
        or alpha_owner_binding._alpha_head_cas
        is not getattr(current_source, "alpha_head_cas", None)
        or alpha_owner_binding._ordinal_physical_axis_bridge
        is not getattr(current_source, "ordinal_physical_axis_bridge", None)
        or len(endpoint_look_indices) != 1
        or alpha_owner_binding.registered_look_index != logical_look_index
        or endpoint_look_indices[0] != logical_look_index
        or alpha_owner_binding.registered_look_stratum_counts
        != expected_endpoint_counts
        or alpha_owner_binding.registered_a_f_j_cells_bound is not True
        or alpha_owner_binding.same_capture_streams_bound is not True
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.ALPHA_LOOK_MISMATCH,
            "A/F/J binding is not the unique frozen logical-look endpoint",
        )

    try:
        action_pair = issue_v1_sampled_v7_branch_action_pair(
            source_context=live_d_context
        )
    except Exception as error:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.ACTION_AXIS_MISMATCH,
            "two-branch source action pair could not be replayed: "
            f"{type(error).__name__}: {error}",
        )
    if type(action_pair) is not V1SampledV7BranchActionPair:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            "source action issuer returned a noncanonical pair",
        )
    action_pair_hash = action_pair.assert_integrity()
    common = getattr(live_d_context, "common_coordinate", None)
    if common is None:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            "live V7 D context lost common-coordinate custody",
        )
    branch_execution_hashes = tuple(
        _identity(getattr(value, "execution_hash", None), name="branch execution hash")
        for value in tuple(getattr(common, "branch_executions", ()))
    )
    branch_shell_hashes = tuple(
        _identity(getattr(value, "execution_hash", None), name="branch shell hash")
        for value in tuple(getattr(common, "branch_shells", ()))
    )
    shell_receipt = getattr(d1_prefix.shell_result, "receipt", None)
    if (
        len(action_pair.branch_aliases) != 2
        or len(branch_execution_hashes) != 2
        or len(branch_shell_hashes) != 2
        or tuple(value.shell_execution_hash for value in action_pair.branch_aliases)
        != branch_shell_hashes
        or tuple(getattr(shell_receipt, "branch_execution_hashes", ()))
        != branch_execution_hashes
        or tuple(getattr(shell_receipt, "branch_shell_hashes", ()))
        != branch_shell_hashes
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.ACTION_AXIS_MISMATCH,
            "D1 fixed directions and the live two-branch action pair diverged",
        )

    base_fanout = affine_shell_fanout_plan._base_plan
    if type(base_fanout) is not V1SampledV7SharedDW2FanoutPlan:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
            "affine-shell wrapper lost its exact shared fanout",
        )
    base_hash = base_fanout.assert_preoutcome()
    acquisition = base_fanout._acquisition_plan
    destinations = tuple(base_fanout.destinations)
    destination_hashes = tuple(
        value.assert_static_integrity(require_pending=True) for value in destinations
    )
    action_identity_axis = tuple(
        value.physical_action_identity_hash for value in action_pair.physical_actions
    )
    packed_action_axis = tuple(
        value.packed_action_hash for value in action_pair.physical_actions
    )
    source_custody_axis = tuple(
        value.custody_hash for value in action_pair.physical_actions
    )
    branch_alias_crosswalk = tuple(
        (branch.value, alias_hash, custody.physical_action_identity_hash)
        for custody in action_pair.physical_actions
        for branch, alias_hash in zip(
            custody.branch_alias_axis,
            custody.branch_alias_hashes,
            strict=True,
        )
    )
    if (
        acquisition._owner is not measurement_owner
        or acquisition._packet_issuance is not packet_issuance
        or base_fanout.measurement_owner_hash != owner_hash
        or base_fanout.packet_hash != issuance_hash
        or not destinations
        or len(destinations) != len(action_pair.physical_actions)
        or tuple(value.action.physical_action_identity_hash for value in destinations)
        != action_identity_axis
        or tuple(value.action.packed_action_hash for value in destinations)
        != packed_action_axis
        or tuple(value.action.source_custody_hash for value in destinations)
        != source_custody_axis
        or base_fanout.branch_alias_to_physical_action
        != branch_alias_crosswalk
        or tuple(value.action.physical_action_ordinal for value in destinations)
        != tuple(range(1, len(destinations) + 1))
        or base_fanout.block_support_sink_axis_preregistered is not True
        or base_fanout.block_support_sink_count != len(destinations)
        or any(value.block_support_binding is None for value in destinations)
        or affine_shell_fanout_plan.base_plan_hash != base_hash
        or affine_shell_fanout_plan.physical_action_axis_hash
        != base_fanout.physical_action_axis_hash
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.DESTINATION_AXIS_MISMATCH,
            "shared fanout does not cover the exact current-source physical actions",
        )

    durable = affine_shell_fanout_plan.durable_receipt
    durable_hash = durable.assert_integrity()
    if (
        joint_preoutcome_registration.fanout_receipt_hash != durable_hash
        or joint_preoutcome_registration.fanout_wrapper_plan_hash
        != fanout_wrapper_hash
        or joint_preoutcome_registration.fanout_base_plan_hash != base_hash
        or joint_preoutcome_registration.physical_action_axis_hash
        != base_fanout.physical_action_axis_hash
        or joint_preoutcome_registration.literal_j_owner_binding_receipt_hash
        != alpha_hash
        or joint_preoutcome_registration.action_count != len(destinations)
        or joint_preoutcome_registration.physical_action_identity_hashes
        != action_identity_axis
        or joint_preoutcome_registration.packed_action_hashes != packed_action_axis
        or joint_preoutcome_registration.shell_freeze_receipt_hash
        != d1_prefix.shell_union_receipt_hash
        or joint_preoutcome_registration.shell_indices
        != d1_prefix.shell_union.shell_indices
        or joint_preoutcome_registration.shell_route_ids
        != d1_prefix.shell_union.shell_route_ids
        or joint_preoutcome_registration.registered_before_postsource_outcomes
        is not True
        or joint_preoutcome_registration.postsource_identity_present is not False
        or joint_preoutcome_registration.outcome_identity_present is not False
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.DIRECTION_AXIS_MISMATCH,
            "affine-shell directions do not replay the D1 shell, action, and A/F/J axes",
        )

    append_registration_hashes = tuple(
        value.append_registration.registration_hash for value in destinations
    )
    source_registration_hashes = tuple(
        value.source_registration.registration_hash for value in destinations
    )
    if (
        joint_preoutcome_registration.source_registration_hashes
        != source_registration_hashes
        or joint_preoutcome_registration.source_registration_hashes
        != tuple(
            value.source_registration_hash
            for value in affine_shell_fanout_plan.preoutcome_registrations
        )
    ):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.DIRECTION_AXIS_MISMATCH,
            "action-local source sinks and affine-shell directions are not co-registered",
        )

    flags_with_outcomes = (
        getattr(packet_issuance, "source_outcomes_seen", False),
        getattr(base_fanout, "provider_outcomes_seen", False),
        getattr(affine_shell_fanout_plan, "provider_outcomes_seen", False),
        getattr(joint_preoutcome_registration, "postsource_identity_present", False),
        getattr(joint_preoutcome_registration, "outcome_identity_present", False),
    )
    if any(value is not False for value in flags_with_outcomes):
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.OUTCOME_LEAKAGE,
            "D2 pre-outcome plan observed a provider or postsource identity",
        )

    fixed_direction_ids = tuple(
        _identity(value, name="fixed-primal direction ID")
        for value in tuple(getattr(shell_receipt, "fixed_primal_direction_ids", ()))
    )
    if len(fixed_direction_ids) != 2:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.ACTION_AXIS_MISMATCH,
            "D1 must expose exactly two source fixed-primal directions",
        )

    coordinates: dict[str, object] = {
        "live_d_context_hash": live_hash,
        "current_source_context_hash": current_source_hash,
        "d1_prefix_result_hash": d1_hash,
        "d1_axis_plan_hash": d1_prefix.d1_axis_plan_hash,
        "d1_shell_result_hash": d1_prefix.shell_result_hash,
        "d1_shell_union_receipt_hash": d1_prefix.shell_union_receipt_hash,
        "measurement_owner_hash": owner_hash,
        "suffix_registration_hash": d1_prefix.suffix_registration_hash,
        "packet_issuance_hash": issuance_hash,
        "suffix_packet_hash": packet_hash,
        "schedule_packet_hash": schedule_packet.packet_schedule_hash,
        "packet_ordinal": packet.packet_ordinal,
        "packet_before_suffix_stratum_counts": packet.before_suffix_stratum_counts,
        "packet_after_suffix_stratum_counts": packet.after_suffix_stratum_counts,
        "logical_residual_before_stratum_counts": logical_residual_before,
        "logical_residual_after_stratum_counts": logical_residual_after,
        "transport_packet_hashes": tuple(
            item.receipt_hash for item in transport_packets
        ),
        "transport_schedule_packet_hashes": tuple(
            item.packet_schedule_hash for item in schedule_packets
        ),
        "transport_packet_ordinals": tuple(
            item.packet_ordinal for item in transport_packets
        ),
        "alpha_owner_binding_receipt_hash": alpha_hash,
        "registered_look_index": alpha_owner_binding.registered_look_index,
        "registered_look_stratum_counts": (
            alpha_owner_binding.registered_look_stratum_counts
        ),
        "corrected_a_cell_hash": alpha_owner_binding.corrected_a_cell_hash,
        "signed_f_cell_hash": alpha_owner_binding.signed_f_cell_hash,
        "literal_j_cell_hash": alpha_owner_binding.literal_j_cell_hash,
        "dual_stream_binding_hash": alpha_owner_binding.dual_stream_binding_hash,
        "source_action_pair_hash": action_pair_hash,
        "source_branch_axis": tuple(value.branch.value for value in action_pair.branch_aliases),
        "source_branch_alias_hashes": tuple(
            value.alias_hash for value in action_pair.branch_aliases
        ),
        "source_branch_execution_hashes": branch_execution_hashes,
        "source_branch_shell_execution_hashes": branch_shell_hashes,
        "fixed_primal_direction_ids": fixed_direction_ids,
        "physical_action_axis_hash": base_fanout.physical_action_axis_hash,
        "physical_action_identity_hashes": action_identity_axis,
        "packed_action_hashes": packed_action_axis,
        "physical_action_source_custody_hashes": source_custody_axis,
        "branch_alias_to_physical_action": branch_alias_crosswalk,
        "shared_fanout_plan_hash": base_hash,
        "destination_integrity_hashes": destination_hashes,
        "destination_source_registration_hashes": source_registration_hashes,
        "destination_append_registration_hashes": append_registration_hashes,
        "block_support_sink_axis_hash": base_fanout.block_support_sink_axis_hash,
        "block_support_sink_identity_hashes": tuple(
            value.block_support_binding.sink_binding.sink_identity_hash
            for value in destinations
            if value.block_support_binding is not None
        ),
        "affine_shell_fanout_plan_hash": fanout_wrapper_hash,
        "affine_shell_fanout_receipt_hash": durable_hash,
        "affine_shell_binding_axis_hash": affine_shell_fanout_plan.binding_axis_hash,
        "joint_preoutcome_registration_hash": joint_hash,
        "joint_direction_axis_hash": joint_preoutcome_registration.joint_direction_axis_hash,
        "joint_direction_ids": joint_preoutcome_registration.joint_direction_ids,
        "preoutcome_registration_hashes": (
            joint_preoutcome_registration.preoutcome_registration_hashes
        ),
        "destination_binding_hashes": (
            joint_preoutcome_registration.destination_binding_hashes
        ),
    }
    return coordinates, action_pair, base_fanout


def _source_coordinates(
    *,
    live_d_context: object,
    d1_prefix: V1SampledV7CurrentSourceD1PrefixResult,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ),
    alpha_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    affine_shell_fanout_plan: V1SampledV7AffineShellPreoutcomeFanoutPlan,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> tuple[dict[str, object], V1SampledV7BranchActionPair, V1SampledV7SharedDW2FanoutPlan]:
    """Normalize every replay failure to this seam's typed refusal surface."""

    try:
        return _source_coordinates_unchecked(
            live_d_context=live_d_context,
            d1_prefix=d1_prefix,
            measurement_owner=measurement_owner,
            packet_issuance=packet_issuance,
            alpha_owner_binding=alpha_owner_binding,
            affine_shell_fanout_plan=affine_shell_fanout_plan,
            joint_preoutcome_registration=joint_preoutcome_registration,
        )
    except V1SampledV7CurrentSourceD2PreoutcomeRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.STALE_SOURCE,
            "current-source D2 parent replay failed: "
            f"{type(error).__name__}: {error}",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7CurrentSourceD2PreoutcomePlan:
    """One live, replayable, outcome-free D2 source plan."""

    live_d_context_hash: str
    current_source_context_hash: str
    d1_prefix_result_hash: str
    d1_axis_plan_hash: str
    d1_shell_result_hash: str
    d1_shell_union_receipt_hash: str
    measurement_owner_hash: str
    suffix_registration_hash: str
    packet_issuance_hash: str
    suffix_packet_hash: str
    schedule_packet_hash: str
    packet_ordinal: int
    packet_before_suffix_stratum_counts: tuple[int, int]
    packet_after_suffix_stratum_counts: tuple[int, int]
    logical_residual_before_stratum_counts: tuple[int, int]
    logical_residual_after_stratum_counts: tuple[int, int]
    transport_packet_hashes: tuple[str, ...]
    transport_schedule_packet_hashes: tuple[str, ...]
    transport_packet_ordinals: tuple[int, ...]
    alpha_owner_binding_receipt_hash: str
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    corrected_a_cell_hash: str
    signed_f_cell_hash: str
    literal_j_cell_hash: str
    dual_stream_binding_hash: str
    source_action_pair_hash: str
    source_branch_axis: tuple[str, str]
    source_branch_alias_hashes: tuple[str, str]
    source_branch_execution_hashes: tuple[str, str]
    source_branch_shell_execution_hashes: tuple[str, str]
    fixed_primal_direction_ids: tuple[str, str]
    physical_action_axis_hash: str
    physical_action_identity_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    physical_action_source_custody_hashes: tuple[str, ...]
    branch_alias_to_physical_action: tuple[tuple[str, str, str], ...]
    shared_fanout_plan_hash: str
    destination_integrity_hashes: tuple[str, ...]
    destination_source_registration_hashes: tuple[str, ...]
    destination_append_registration_hashes: tuple[str, ...]
    block_support_sink_axis_hash: str
    block_support_sink_identity_hashes: tuple[str, ...]
    affine_shell_fanout_plan_hash: str
    affine_shell_fanout_receipt_hash: str
    affine_shell_binding_axis_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    preoutcome_registration_hashes: tuple[str, ...]
    destination_binding_hashes: tuple[str, ...]
    exact_two_branch_source_direction_axis_frozen: bool = True
    exactly_one_predictable_logical_look_frozen: bool = True
    packet_endpoint_a_f_j_look_bound: bool = True
    complete_physical_action_destination_axis_frozen: bool = True
    affine_shell_direction_and_sink_axes_frozen: bool = True
    provider_outcomes_seen: bool = False
    outcome_payload_present: bool = False
    alpha_spend_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    action_or_commit_authority: bool = False
    terminal_performance_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_AUTHORITY
    schema: str = SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_SCHEMA
    _live_d_context: object = field(repr=False, compare=False, default=None)
    _d1_prefix: V1SampledV7CurrentSourceD1PrefixResult = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _measurement_owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ) = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _alpha_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _source_action_pair: V1SampledV7BranchActionPair = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _shared_fanout_plan: V1SampledV7SharedDW2FanoutPlan = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _affine_shell_fanout_plan: V1SampledV7AffineShellPreoutcomeFanoutPlan = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.INVALID_TYPE,
                "D2 pre-outcome plans are issuer-only",
            )
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "plan-")

    def assert_live(self, expected_hash: str | None = None) -> str:
        coordinates, action_pair, fanout = _source_coordinates(
            live_d_context=self._live_d_context,
            d1_prefix=self._d1_prefix,
            measurement_owner=self._measurement_owner,
            packet_issuance=self._packet_issuance,
            alpha_owner_binding=self._alpha_owner_binding,
            affine_shell_fanout_plan=self._affine_shell_fanout_plan,
            joint_preoutcome_registration=self._joint_preoutcome_registration,
        )
        if action_pair.assert_integrity() != self._source_action_pair.assert_integrity():
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.ACTION_AXIS_MISMATCH,
                "replayed source action pair differs from the retained pair",
            )
        if fanout is not self._shared_fanout_plan:
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.DESTINATION_AXIS_MISMATCH,
                "another shared fanout was substituted after D2 freeze",
            )
        for name, value in coordinates.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.MUTATED_PLAN,
                    f"D2 coordinate {name} changed after freeze",
                )
        positive = (
            self.exact_two_branch_source_direction_axis_frozen,
            self.exactly_one_predictable_logical_look_frozen,
            self.packet_endpoint_a_f_j_look_bound,
            self.complete_physical_action_destination_axis_frozen,
            self.affine_shell_direction_and_sink_axes_frozen,
        )
        closed = (
            self.provider_outcomes_seen,
            self.outcome_payload_present,
            self.alpha_spend_authority,
            self.d_stability_authority,
            self.c_k_o_authority,
            self.action_or_commit_authority,
            self.terminal_performance_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_AUTHORITY
            or self.schema != SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.AUTHORITY_VIOLATION,
                "D2 plan crossed its pre-outcome identity-only authority",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.MUTATED_PLAN,
                "D2 plan hash changed after freeze",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7CurrentSourceD2PreoutcomeRefusalCode.MUTATED_PLAN,
                "D2 plan hash does not match expected replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def require_physical_acquisition(self) -> NoReturn:
        raise PermissionError("D2 pre-outcome plan has no acquisition authority")

    def require_alpha_spend(self) -> NoReturn:
        raise PermissionError("D2 pre-outcome plan has no alpha-spend authority")

    def require_d_stability(self) -> NoReturn:
        raise PermissionError("D2 pre-outcome plan has no D-stability authority")

    def require_c_k_o(self) -> NoReturn:
        raise PermissionError("D2 pre-outcome plan has no C/K/O authority")

    def require_mask_commit(self) -> NoReturn:
        raise PermissionError("D2 pre-outcome plan cannot commit masks")

    def __reduce__(self) -> NoReturn:
        raise TypeError("live current-source D2 pre-outcome plans are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live current-source D2 pre-outcome plans are nonserializable")


def issue_v1_sampled_v7_current_source_d2_preoutcome_plan(
    *,
    live_d_context: object,
    d1_prefix: V1SampledV7CurrentSourceD1PrefixResult,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    packet_issuance: (
        V1SampledV7DOnlyPacketMeasurementIssuance
        | V1SampledV7DOnlyLogicalLookMeasurementIssuance
    ),
    alpha_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    affine_shell_fanout_plan: V1SampledV7AffineShellPreoutcomeFanoutPlan,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> V1SampledV7CurrentSourceD2PreoutcomePlan:
    """Freeze the complete D2 source graph before the first provider outcome."""

    coordinates, action_pair, fanout = _source_coordinates(
        live_d_context=live_d_context,
        d1_prefix=d1_prefix,
        measurement_owner=measurement_owner,
        packet_issuance=packet_issuance,
        alpha_owner_binding=alpha_owner_binding,
        affine_shell_fanout_plan=affine_shell_fanout_plan,
        joint_preoutcome_registration=joint_preoutcome_registration,
    )
    return V1SampledV7CurrentSourceD2PreoutcomePlan(
        **coordinates,  # type: ignore[arg-type]
        _live_d_context=live_d_context,
        _d1_prefix=d1_prefix,
        _measurement_owner=measurement_owner,
        _packet_issuance=packet_issuance,
        _alpha_owner_binding=alpha_owner_binding,
        _source_action_pair=action_pair,
        _shared_fanout_plan=fanout,
        _affine_shell_fanout_plan=affine_shell_fanout_plan,
        _joint_preoutcome_registration=joint_preoutcome_registration,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_AUTHORITY",
    "SAMPLED_V7_CURRENT_SOURCE_D2_PREOUTCOME_PLAN_SCHEMA",
    "V1SampledV7CurrentSourceD2PreoutcomePlan",
    "V1SampledV7CurrentSourceD2PreoutcomeRefusal",
    "V1SampledV7CurrentSourceD2PreoutcomeRefusalCode",
    "issue_v1_sampled_v7_current_source_d2_preoutcome_plan",
]
