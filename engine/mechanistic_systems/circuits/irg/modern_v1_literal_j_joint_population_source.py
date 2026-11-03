"""Causal dual-source custody for one literal-J action direction.

The conditioned discovery prefix ``D1`` and the fresh residual ``D_W2`` look
are different physical sources.  Their registrations, and the structural
registration which owns the packed parent-to-child action, are therefore not
required to be equal.  What *is* required is one exact scientific join:

* the D1 raw-J attachment axis and its fixed-prefix counts;
* the residual source registered before provider outcomes;
* the exact finite-population D1/residual measure;
* the live packed action custody; and
* the complete pre-registered action/direction family.

This module makes the temporal split explicit.  A preoutcome owner can be
issued only while the causal logical-look source is still unbound.  After the
provider result lands, a separate binder attaches the exact logical result and
residual postsource receipt.  A durable ``provider_outcomes_seen=False`` field
can consequently never impersonate temporal preregistration.

Both receipts are array-free and authority-inert.  The process-local owners
retain live source objects only so every later consumer can replay custody.
They do not project rows, spend alpha, certify D, open C/K/O, or commit a mask.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
from threading import RLock
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_preoutcome_registration import (
    V1LiteralJAffineShellDirectionPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
    V1LiteralJConditionedD1ShardReplaySource,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    V1SampledV7AffineShellPreoutcomeDestinationBinding,
    V1SampledV7AffineShellPreoutcomeFanoutReceipt,
)
from .modern_v1_sampled_v7_shared_dw2_literal_j_logical_look_source import (
    V1SampledV7SharedDW2LiteralJLogicalLookResult,
    V1SampledV7SharedDW2LiteralJLogicalLookSource,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportPreferredActionBinding,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceReceipt,
)


LITERAL_J_JOINT_POPULATION_PREOUTCOME_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_preoutcome_source_v1"
)
LITERAL_J_JOINT_POPULATION_SOURCE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_population_source_v1"
)
LITERAL_J_JOINT_POPULATION_SOURCE_AUTHORITY: Final = (
    "causal_D1_residual_measure_action_direction_source_custody_only_no_"
    "projection_confidence_FD_solver_D_C_K_O_commit_terminal_or_pareto_"
    "authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-joint-population-source-"
_PREOUTCOME_TOKEN: Final = object()
_POSTSOURCE_TOKEN: Final = object()
_PREOUTCOME_OWNER_TOKEN: Final = object()
_POSTSOURCE_OWNER_TOKEN: Final = object()


class V1LiteralJJointPopulationSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PREOUTCOME_REQUIRED = "preoutcome_required"
    OUTCOME_REQUIRED = "outcome_required"
    RESIDUAL_SOURCE_MISMATCH = "residual_source_mismatch"
    D1_SOURCE_MISMATCH = "d1_source_mismatch"
    MEASURE_MISMATCH = "measure_mismatch"
    ACTION_CUSTODY_MISMATCH = "action_custody_mismatch"
    JOINT_AXIS_MISMATCH = "joint_axis_mismatch"
    STALE_SOURCE = "stale_source"
    BIND_REPLAY = "bind_replay"
    MATERIALIZATION_FAILURE = "materialization_failure"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointPopulationSourceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointPopulationSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointPopulationSourceRefusalCode:
            raise TypeError("joint-population refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointPopulationSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointPopulationSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _identity_axis(
    value: object,
    *,
    name: str,
    allow_empty: bool = False,
    unique: bool = True,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} member") for item in value)
    if (not allow_empty and not result) or (unique and len(set(result)) != len(result)):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} is empty or not correctly unique",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one target/complement count pair",
        )
    result = tuple(_nonnegative_int(item, name=f"{name} member") for item in value)
    return result  # type: ignore[return-value]


def _fraction_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact rational numerator/denominator pair",
        )
    numerator = _nonnegative_int(value[0], name=f"{name} numerator")
    denominator = _positive_int(value[1], name=f"{name} denominator")
    if Fraction(numerator, denominator) != Fraction(*value):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            f"{name} is not an exact rational pair",
        )
    return numerator, denominator


def _authority_payload() -> dict[str, bool | str]:
    return {
        "provider_execution_authority": False,
        "provider_result_authority": False,
        "alpha_allocation_authority": False,
        "alpha_spend_authority": False,
        "literal_j_projection_authority": False,
        "population_projection_authority": False,
        "confidence_authority": False,
        "fd_replacement_authority": False,
        "loewner_or_kkt_authority": False,
        "solver_authority": False,
        "d_sampling_authority": False,
        "d_stability_authority": False,
        "c_stream_authority": False,
        "calibration_authority": False,
        "oracle_authority": False,
        "action_selection_authority": False,
        "c_k_o_authority": False,
        "mask_commit_authority": False,
        "terminal_or_pareto_authority": False,
        "authority_scope": LITERAL_J_JOINT_POPULATION_SOURCE_AUTHORITY,
    }


def _closed_authority(value: object) -> bool:
    return all(
        getattr(value, name, None) is expected
        if type(expected) is bool
        else getattr(value, name, None) == expected
        for name, expected in _authority_payload().items()
    )


def _d1_source_axis_hash(
    source: V1LiteralJConditionedD1ShardReplaySource,
) -> str:
    replay = source.replay_receipt
    compact = source.compact_source_receipt
    return content_hash(
        {
            "schema": "literal_J_conditioned_D1_raw_source_axis_v1",
            "replay_receipt_hash": replay.receipt_hash,
            "compact_source_receipt_hash": compact.receipt_hash,
            "compact_source_registration_hash": compact.registration_hash,
            "registered_D_prefix_hash": replay.registered_d_prefix_hash,
            "D1_pair_prefix_axis_hash": replay.d1_pair_prefix_axis_hash,
            "canonical_context_axis_hash": replay.canonical_context_axis_hash,
            "canonical_draw_axis_hash": replay.canonical_draw_axis_hash,
            "canonical_measurement_axis_hash": replay.canonical_measurement_axis_hash,
            "canonical_capture_axis_hash": replay.canonical_capture_axis_hash,
            "canonical_source_packet_axis_hash": (
                replay.canonical_source_packet_axis_hash
            ),
            "literal_J_shard_receipt_hashes": (
                replay.literal_j_shard_receipt_hashes
            ),
            "literal_J_content_sha256s": replay.literal_j_content_sha256s,
            "D1_stratum_counts": replay.d1_stratum_counts,
            "population_stratum_counts": replay.population_stratum_counts,
            "context_count": replay.context_count,
            "population_count": replay.population_count,
            "route_count": replay.route_count,
            "raw_attachment_axis_bound": True,
        },
        prefix=_PREFIX + "d1-axis-",
    )


def _validate_complete_destination_member(
    *,
    fanout: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    destination: V1SampledV7AffineShellPreoutcomeDestinationBinding,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
) -> tuple[str, str, str]:
    try:
        fanout_hash = fanout.assert_integrity()
        destination_hash = destination.assert_integrity()
        registration_hash = registration.assert_integrity()
    except V1LiteralJJointPopulationSourceRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
            "one residual preoutcome parent failed integrity replay: "
            f"{type(error).__name__}: {error}",
        )
    ordinal = destination.physical_action_ordinal
    if (
        ordinal <= 0
        or ordinal > fanout.destination_count
        or fanout.destination_bindings[ordinal - 1] is not destination
        or fanout.preoutcome_registrations[ordinal - 1] is not registration
        or destination.preregistration_hash != registration_hash
        or destination.source_registration_hash
        != registration.source_registration_hash
        or destination.physical_action_hash != registration.physical_action_hash
        or destination.physical_action_identity_hash
        != registration.physical_action_identity_hash
        or destination.direction_family_hash != registration.direction_family_hash
        or destination.literal_j_owner_binding_receipt_hash
        != registration.literal_j_owner_binding_receipt_hash
        or destination.literal_j_cell_hash != registration.literal_j_cell_hash
        or destination.registered_look_index != registration.registered_look_index
        or fanout.literal_j_owner_binding_receipt_hash
        != registration.literal_j_owner_binding_receipt_hash
        or fanout.literal_j_cell_hash != registration.literal_j_cell_hash
        or fanout.registered_look_index != registration.registered_look_index
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.JOINT_AXIS_MISMATCH,
            "residual preregistration is not the exact complete-fanout member",
        )
    return fanout_hash, destination_hash, registration_hash


def _validate_joint_member(
    *,
    fanout_hash: str,
    fanout: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    destination: V1SampledV7AffineShellPreoutcomeDestinationBinding,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    joint: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> str:
    try:
        joint_hash = joint.assert_integrity()
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
            "joint action/direction preregistration failed replay: "
            f"{type(error).__name__}: {error}",
        )
    index = destination.physical_action_ordinal - 1
    common = (
        (joint.fanout_receipt_hash, fanout_hash),
        (joint.physical_action_axis_hash, fanout.physical_action_axis_hash),
        (joint.fanout_binding_axis_hash, fanout.binding_axis_hash),
        (joint.model_identity_hash, registration.model_identity_hash),
        (joint.chart_identity_hash, registration.chart_identity_hash),
        (joint.current_mask_hash, registration.current_mask_hash),
        (joint.proposal_generation, registration.proposal_generation),
        (joint.route_axis_hash, registration.route_axis_hash),
        (joint.route_id_axis_hash, registration.route_id_axis_hash),
        (joint.route_count, registration.route_count),
        (joint.registered_d_prefix_hash, registration.registered_d_prefix_hash),
        (joint.registered_d_w2_prefix_hash, registration.registered_d_w2_prefix_hash),
        (joint.shell_freeze_receipt_hash, registration.shell_freeze_receipt_hash),
        (joint.shell_indices, registration.shell_indices),
        (joint.shell_route_ids, registration.shell_route_ids),
        (joint.literal_j_stream_source_hash, registration.literal_j_stream_source_hash),
        (joint.same_capture_vjp_source_hash, registration.same_capture_vjp_source_hash),
        (joint.literal_j_stream_binding_hash, registration.literal_j_stream_binding_hash),
        (joint.literal_j_cell_hash, registration.literal_j_cell_hash),
        (joint.literal_j_cell_scope_hash, registration.literal_j_cell_scope_hash),
        (joint.historical_alpha_lineage_hash, registration.historical_alpha_lineage_hash),
        (joint.registered_look_index, registration.registered_look_index),
    )
    if (
        any(left != right for left, right in common)
        or joint.physical_action_ordinals[index]
        != destination.physical_action_ordinal
        or joint.physical_action_identity_hashes[index]
        != registration.physical_action_identity_hash
        or joint.physical_action_hashes[index] != registration.physical_action_hash
        or joint.packed_action_hashes[index] != registration.packed_action_hash
        or joint.source_registration_hashes[index]
        != registration.source_registration_hash
        or joint.preoutcome_registration_hashes[index]
        != registration.registration_hash
        or joint.destination_binding_hashes[index] != destination.binding_hash
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.JOINT_AXIS_MISMATCH,
            "joint action/direction axis is missing, swapped, or stale",
        )
    return joint_hash


def _validate_d1_measure_action(
    *,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    measure_design: V1SampledResidualPopulationMeasureDesign,
    measure_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    action_custody: V1SignedWorkBlockSupportPreferredActionBinding,
) -> tuple[str, str, str, str]:
    try:
        # Reconstructing the exact typed wrapper replays every raw attachment.
        conditioned_d1.__post_init__()
        d1_hash = conditioned_d1.replay_receipt.assert_integrity()
        design_hash = measure_design.assert_integrity()
        weight_hash = measure_weight_receipt.assert_integrity()
        action_custody.assert_live()
        action_hash = action_custody.binding_hash
    except V1LiteralJJointPopulationSourceRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
            "one D1, measure, or action parent failed live replay: "
            f"{type(error).__name__}: {error}",
        )

    d1 = conditioned_d1.replay_receipt
    compact = conditioned_d1.compact_source_receipt
    d1_common = (
        (compact.model_identity_hash, registration.model_identity_hash),
        (d1.chart_identity_hash, registration.chart_identity_hash),
        (compact.chart_identity_hash, registration.chart_identity_hash),
        (d1.current_mask_hash, registration.current_mask_hash),
        (compact.current_mask_hash, registration.current_mask_hash),
        (d1.proposal_generation, registration.proposal_generation),
        (compact.proposal_generation, registration.proposal_generation),
        (d1.route_axis_hash, registration.route_axis_hash),
        (compact.route_axis_hash, registration.route_axis_hash),
        (d1.route_id_axis_hash, registration.route_id_axis_hash),
        (compact.route_id_axis_hash, registration.route_id_axis_hash),
        (d1.route_count, registration.route_count),
        (compact.route_count, registration.route_count),
        (d1.native_fiber_axis_receipt_hash, registration.native_fiber_axis_receipt_hash),
        (compact.native_fiber_axis_receipt_hash, registration.native_fiber_axis_receipt_hash),
        (d1.native_fiber_measure_hash, registration.native_fiber_measure_hash),
        (compact.native_fiber_measure_hash, registration.native_fiber_measure_hash),
        (compact.shell_freeze_receipt_hash, registration.shell_freeze_receipt_hash),
        (compact.shell_indices, registration.shell_indices),
        (compact.shell_route_ids, registration.shell_route_ids),
        (compact.alpha_lineage_hash, registration.historical_alpha_lineage_hash),
        (d1.d1_pair_prefix_axis_hash, registration.registered_d_prefix_hash),
        (compact.registered_d_w2_prefix_hash, d1.d1_pair_prefix_axis_hash),
        (compact.literal_j_stream_source_hash, d1.literal_j_stream_source_hash),
        (compact.context_count, d1.context_count),
    )
    if any(left != right for left, right in d1_common):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.D1_SOURCE_MISMATCH,
            "conditioned D1 raw source left the chart, mask, route, prefix, shell, J, or alpha lineage",
        )
    if (
        d1.residual_measure_design_hash != design_hash
        or d1.residual_weight_receipt_hash != weight_hash
        or measure_weight_receipt.measure_design_hash != design_hash
        or measure_weight_receipt.suffix_registration_hash
        != measure_design.suffix_registration_hash
        or measure_design.suffix_registration_hash != registration.suffix_registration_hash
        or measure_design.current_mask_hash != registration.current_mask_hash
        or measure_design.generation_index != registration.proposal_generation
        or measure_design.population_stratum_counts != d1.population_stratum_counts
        or measure_design.d1_stratum_counts != d1.d1_stratum_counts
        or measure_design.population_count != d1.population_count
        or measure_design.d1_pair_prefix_axis_hash != d1.d1_pair_prefix_axis_hash
        or measure_weight_receipt.population_stratum_counts
        != measure_design.population_stratum_counts
        or measure_weight_receipt.d1_stratum_counts != measure_design.d1_stratum_counts
        or measure_weight_receipt.residual_stratum_counts
        != measure_design.residual_stratum_counts
        or measure_weight_receipt.cumulative_dw2_stratum_counts
        != registration.after_suffix_stratum_counts
        or registration.registered_look_stratum_counts
        != tuple(
            d1_count + residual_count
            for d1_count, residual_count in zip(
                d1.d1_stratum_counts,
                measure_weight_receipt.cumulative_dw2_stratum_counts,
                strict=True,
            )
        )
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.MEASURE_MISMATCH,
            "D1, residual suffix, and exact population measure do not form one prefix",
        )
    if (
        action_custody.model_identity_hash != registration.model_identity_hash
        or action_custody.chart_identity_hash != registration.chart_identity_hash
        or action_custody.current_mask_hash != registration.parent_mask_hash
        or action_custody.proposal_generation != registration.proposal_generation
        or action_custody.route_axis_hash != registration.route_axis_hash
        or action_custody.route_count != registration.route_count
        or action_custody.preferred_action_hash != registration.packed_action_hash
        or action_custody.preferred_resulting_mask_hash
        != registration.resulting_mask_hash
        or action_custody.preferred_removed_route_index_hash
        != registration.removed_route_index_hash
        or action_custody.preferred_selected_count != registration.removed_count
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.ACTION_CUSTODY_MISMATCH,
            "packed action custody differs from the registered irreversible action",
        )
    return d1_hash, design_hash, weight_hash, action_hash


def _validate_logical_source(
    *,
    source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    require_bound: bool,
) -> str:
    if type(source) is not V1SampledV7SharedDW2LiteralJLogicalLookSource:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            "logical source must have the exact causal source type",
        )
    try:
        source_hash = source.assert_live()
    except Exception as error:
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
            "causal logical source failed live replay: "
            f"{type(error).__name__}: {error}",
        )
    if source.is_bound is not require_bound:
        _refuse(
            (
                V1LiteralJJointPopulationSourceRefusalCode.OUTCOME_REQUIRED
                if require_bound
                else V1LiteralJJointPopulationSourceRefusalCode.PREOUTCOME_REQUIRED
            ),
            "logical source has the wrong temporal binding state",
        )
    receipt = source.receipt
    if (
        receipt.provider_outcomes_seen is not False
        or receipt.chart_identity_hash != registration.chart_identity_hash
        or receipt.current_mask_hash != registration.current_mask_hash
        or receipt.proposal_generation != registration.proposal_generation
        or receipt.route_axis_hash != registration.route_axis_hash
        or receipt.route_id_axis_hash != registration.route_id_axis_hash
        or receipt.route_count != registration.route_count
        or receipt.literal_j_stream_identity_hash
        != registration.literal_j_stream_source_hash
        or receipt.suffix_registration_hash != registration.suffix_registration_hash
        or receipt.logical_registered_look_index != registration.registered_look_index
        or receipt.logical_registered_look_stratum_counts
        != registration.registered_look_stratum_counts
        or receipt.logical_residual_before_stratum_counts
        != registration.before_suffix_stratum_counts
        or receipt.logical_residual_after_stratum_counts
        != registration.after_suffix_stratum_counts
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.RESIDUAL_SOURCE_MISMATCH,
            "logical source and residual preregistration name different look axes",
        )
    return source_hash


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationPreoutcomeReceipt:
    logical_source_receipt_hash: str
    fanout_receipt_hash: str
    destination_binding_hash: str
    fanout_binding_axis_hash: str
    physical_action_axis_hash: str
    residual_preoutcome_registration_hash: str
    residual_source_registration_hash: str
    d1_replay_receipt_hash: str
    d1_compact_source_receipt_hash: str
    d1_source_registration_hash: str
    d1_source_axis_hash: str
    d1_corrected_evidence_stream_source_hash: str
    d1_literal_j_stream_source_hash: str
    d1_same_capture_vjp_source_hash: str
    d1_literal_j_shard_receipt_hashes: tuple[str, ...]
    d1_literal_j_content_sha256s: tuple[str, ...]
    measure_design_hash: str
    measure_weight_receipt_hash: str
    action_custody_hash: str
    action_source_registration_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    joint_outside_direction_id: str
    physical_action_ordinal: int
    physical_action_identity_hash: str
    physical_action_hash: str
    packed_action_hash: str
    parent_mask_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    removed_count: int
    outside_direction_source_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    literal_j_stream_binding_hash: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    alpha_lineage_hash: str
    d1_registered_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    d1_context_count: int
    d1_stratum_counts: tuple[int, int]
    residual_sample_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    d1_context_weight_fraction: tuple[int, int]
    residual_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_mass_fraction: tuple[int, int]
    common_measure_denominator: int
    d1_context_quantum: int
    residual_context_quanta: tuple[int, int]
    total_measure_quanta: int
    provider_outcomes_seen: bool = False
    logical_source_bound_at_registration: bool = False
    independently_bound_source_roles: bool = True
    source_registration_equality_required: bool = False
    array_free_durable_receipt: bool = True
    provider_execution_authority: bool = False
    provider_result_authority: bool = False
    alpha_allocation_authority: bool = False
    alpha_spend_authority: bool = False
    literal_j_projection_authority: bool = False
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    loewner_or_kkt_authority: bool = False
    solver_authority: bool = False
    d_sampling_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_selection_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_POPULATION_SOURCE_AUTHORITY
    schema: str = LITERAL_J_JOINT_POPULATION_PREOUTCOME_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PREOUTCOME_TOKEN:
            raise PermissionError("joint-population preoutcome receipts are issuer-only")
        for name in (
            "logical_source_receipt_hash",
            "fanout_receipt_hash",
            "destination_binding_hash",
            "fanout_binding_axis_hash",
            "physical_action_axis_hash",
            "residual_preoutcome_registration_hash",
            "residual_source_registration_hash",
            "d1_replay_receipt_hash",
            "d1_compact_source_receipt_hash",
            "d1_source_registration_hash",
            "d1_source_axis_hash",
            "d1_corrected_evidence_stream_source_hash",
            "d1_literal_j_stream_source_hash",
            "d1_same_capture_vjp_source_hash",
            "measure_design_hash",
            "measure_weight_receipt_hash",
            "action_custody_hash",
            "action_source_registration_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "joint_outside_direction_id",
            "physical_action_identity_hash",
            "physical_action_hash",
            "packed_action_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "outside_direction_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "same_capture_vjp_source_hash",
            "literal_j_stream_binding_hash",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "alpha_lineage_hash",
            "d1_registered_prefix_hash",
            "d1_pair_prefix_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        _identity_axis(
            self.d1_literal_j_shard_receipt_hashes,
            name="D1 literal-J shard receipts",
        )
        _identity_axis(
            self.d1_literal_j_content_sha256s,
            name="D1 literal-J content digests",
        )
        _positive_int(self.physical_action_ordinal, name="physical action ordinal")
        _positive_int(self.removed_count, name="removed count")
        _positive_int(self.route_count, name="route count")
        _positive_int(self.d1_context_count, name="D1 context count")
        _nonnegative_int(self.proposal_generation, name="proposal generation")
        d1 = _count_pair(self.d1_stratum_counts, name="D1 stratum counts")
        sampled = _count_pair(
            self.residual_sample_stratum_counts,
            name="residual sample stratum counts",
        )
        population = _count_pair(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        d1_weight = _fraction_pair(
            self.d1_context_weight_fraction,
            name="D1 context weight",
        )
        if (
            type(self.residual_context_weight_fractions) is not tuple
            or len(self.residual_context_weight_fractions) != 2
            or type(self.residual_stratum_mass_fractions) is not tuple
            or len(self.residual_stratum_mass_fractions) != 2
        ):
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
                "residual rational mass axes must have target/complement members",
            )
        residual_weights = tuple(
            _fraction_pair(value, name="residual context weight")
            for value in self.residual_context_weight_fractions
        )
        residual_masses = tuple(
            _fraction_pair(value, name="residual stratum mass")
            for value in self.residual_stratum_mass_fractions
        )
        residual_population_mass = _fraction_pair(
            self.residual_population_mass_fraction,
            name="residual population mass",
        )
        common_denominator = _positive_int(
            self.common_measure_denominator,
            name="common measure denominator",
        )
        d1_quantum = _positive_int(
            self.d1_context_quantum,
            name="D1 context quantum",
        )
        residual_quanta = _count_pair(
            self.residual_context_quanta,
            name="residual context quanta",
        )
        total_quanta = _positive_int(
            self.total_measure_quanta,
            name="total measure quanta",
        )
        if (
            self.d1_context_count != sum(d1)
            or any(d1[index] + sampled[index] > population[index] for index in (0, 1))
            or Fraction(*d1_weight) != Fraction(1, sum(population))
            or any(
                Fraction(*residual_masses[index])
                != Fraction(population[index] - d1[index], sum(population))
                for index in (0, 1)
            )
            or Fraction(*residual_population_mass)
            != sum(Fraction(*value) for value in residual_masses)
            or any(
                sampled[index] > 0
                and Fraction(*residual_weights[index])
                != Fraction(
                    population[index] - d1[index],
                    sum(population) * sampled[index],
                )
                for index in (0, 1)
            )
            or any(
                sampled[index] == 0 and Fraction(*residual_weights[index]) != 0
                for index in (0, 1)
            )
            or Fraction(d1_quantum, common_denominator) != Fraction(*d1_weight)
            or any(
                Fraction(residual_quanta[index], common_denominator)
                != Fraction(*residual_weights[index])
                for index in (0, 1)
            )
            or total_quanta != common_denominator
            or total_quanta
            != sum(d1) * d1_quantum
            + sum(
                sampled[index] * residual_quanta[index] for index in (0, 1)
            )
            or type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
            or not 1 <= len(self.shell_indices) <= 8
            or len(self.shell_indices) != len(self.shell_route_ids)
            or len(set(self.shell_indices)) != len(self.shell_indices)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or item >= self.route_count
                for item in self.shell_indices
            )
            or self.provider_outcomes_seen is not False
            or self.logical_source_bound_at_registration is not False
            or self.independently_bound_source_roles is not True
            or self.source_registration_equality_required is not False
            or self.array_free_durable_receipt is not True
            or not _closed_authority(self)
            or self.schema != LITERAL_J_JOINT_POPULATION_PREOUTCOME_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.AUTHORITY_VIOLATION,
                "preoutcome receipt is inconsistent, temporally late, or widened",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def exact_d1_context_weight(self) -> Fraction:
        return Fraction(*self.d1_context_weight_fraction)

    def exact_residual_context_weight(self, stratum: str) -> Fraction:
        if stratum not in {"target", "complement"}:
            raise ValueError("stratum must be 'target' or 'complement'")
        return Fraction(
            *self.residual_context_weight_fractions[
                0 if stratum == "target" else 1
            ]
        )

    def exact_residual_stratum_mass(self, stratum: str) -> Fraction:
        if stratum not in {"target", "complement"}:
            raise ValueError("stratum must be 'target' or 'complement'")
        return Fraction(
            *self.residual_stratum_mass_fractions[
                0 if stratum == "target" else 1
            ]
        )

    def exact_residual_population_mass(self) -> Fraction:
        return Fraction(*self.residual_population_mass_fraction)

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "preoutcome-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                "joint-population preoutcome receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointPopulationPreoutcomeSource:
    """Process-local owner of exact parents frozen before residual outcomes."""

    __slots__ = (
        "receipt",
        "_logical_source",
        "_fanout",
        "_destination",
        "_registration",
        "_conditioned_d1",
        "_measure_design",
        "_measure_weight_receipt",
        "_action_custody",
        "_joint",
        "_state",
        "_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointPopulationPreoutcomeReceipt,
        logical_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
        fanout: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
        destination: V1SampledV7AffineShellPreoutcomeDestinationBinding,
        registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
        conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
        measure_design: V1SampledResidualPopulationMeasureDesign,
        measure_weight_receipt: V1SampledResidualPopulationWeightReceipt,
        action_custody: V1SignedWorkBlockSupportPreferredActionBinding,
        joint: V1LiteralJJointAffineShellPreoutcomeRegistration,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _PREOUTCOME_OWNER_TOKEN:
            raise PermissionError("joint-population preoutcome owners are issuer-only")
        self.receipt = receipt
        self._logical_source = logical_source
        self._fanout = fanout
        self._destination = destination
        self._registration = registration
        self._conditioned_d1 = conditioned_d1
        self._measure_design = measure_design
        self._measure_weight_receipt = measure_weight_receipt
        self._action_custody = action_custody
        self._joint = joint
        self._state = "preoutcome"
        self._lock = RLock()
        self.assert_preoutcome_live()

    @property
    def is_consumed(self) -> bool:
        return self._state == "consumed"

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def _replay_common(self) -> None:
        fanout_hash, destination_hash, registration_hash = (
            _validate_complete_destination_member(
                fanout=self._fanout,
                destination=self._destination,
                registration=self._registration,
            )
        )
        joint_hash = _validate_joint_member(
            fanout_hash=fanout_hash,
            fanout=self._fanout,
            destination=self._destination,
            registration=self._registration,
            joint=self._joint,
        )
        d1_hash, design_hash, weight_hash, action_hash = _validate_d1_measure_action(
            registration=self._registration,
            conditioned_d1=self._conditioned_d1,
            measure_design=self._measure_design,
            measure_weight_receipt=self._measure_weight_receipt,
            action_custody=self._action_custody,
        )
        if (
            fanout_hash != self.receipt.fanout_receipt_hash
            or destination_hash != self.receipt.destination_binding_hash
            or registration_hash
            != self.receipt.residual_preoutcome_registration_hash
            or joint_hash != self.receipt.joint_preoutcome_registration_hash
            or d1_hash != self.receipt.d1_replay_receipt_hash
            or design_hash != self.receipt.measure_design_hash
            or weight_hash != self.receipt.measure_weight_receipt_hash
            or action_hash != self.receipt.action_custody_hash
            or _d1_source_axis_hash(self._conditioned_d1)
            != self.receipt.d1_source_axis_hash
        ):
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                "one preoutcome parent changed after source issuance",
            )

    def assert_preoutcome_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.EVICTED,
                    "joint-population preoutcome owner was evicted",
                )
            if self._state != "preoutcome":
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.BIND_REPLAY,
                    "preoutcome owner was already consumed",
                )
            self.receipt.assert_integrity()
            source_hash = _validate_logical_source(
                source=self._logical_source,
                registration=self._registration,
                require_bound=False,
            )
            if source_hash != self.receipt.logical_source_receipt_hash:
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                    "logical source identity changed after preoutcome issuance",
                )
            self._replay_common()
            return self.receipt.receipt_hash

    def materialize_action_float64(
        self,
        *,
        device: object,
        tile_bytes: int,
        production_cuda_required: bool,
    ) -> tuple[Any, int, int]:
        """Materialize only the issuer-bound packed irreversible action.

        This operation is deliberately available before residual outcomes so
        analytic literal-J support may be derived from route bounds and the
        frozen action.  It reads neither a provider nor any D1/D2 row and
        grants no projection or decision authority.  A consumed PRE owner may
        continue to serve its exact POST owner, but an evicted owner refuses.
        """

        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.EVICTED,
                    "joint-population preoutcome owner was evicted",
                )
            if self._state == "preoutcome":
                self.assert_preoutcome_live()
            elif self._state == "consumed":
                self.receipt.assert_integrity()
                self._replay_common()
            else:
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                    "joint-population preoutcome owner has an unknown state",
                )
            try:
                return self._action_custody.materialize_float64(
                    device=device,
                    tile_bytes=_positive_int(
                        tile_bytes,
                        name="packed-byte tile size",
                    ),
                    production_cuda_required=production_cuda_required,
                )
            except V1LiteralJJointPopulationSourceRefusal:
                raise
            except Exception as error:
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.MATERIALIZATION_FAILURE,
                    "packed action materialization failed: "
                    f"{type(error).__name__}: {error}",
                )

    def evict(self) -> None:
        with self._lock:
            if self._state == "consumed":
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.BIND_REPLAY,
                    "consumed preoutcome custody is retained by its postsource owner",
                )
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint-population preoutcome owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("joint-population preoutcome owners are nonserializable")


def issue_v1_literal_j_joint_population_preoutcome_source(
    *,
    logical_look_source: V1SampledV7SharedDW2LiteralJLogicalLookSource,
    preoutcome_fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    preoutcome_destination_binding: V1SampledV7AffineShellPreoutcomeDestinationBinding,
    preoutcome_registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    measure_design: V1SampledResidualPopulationMeasureDesign,
    measure_weight_receipt: V1SampledResidualPopulationWeightReceipt,
    action_custody: V1SignedWorkBlockSupportPreferredActionBinding,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> V1LiteralJJointPopulationPreoutcomeSource:
    """Freeze one exact action/population source before residual outcomes."""

    exact = (
        (preoutcome_fanout_receipt, V1SampledV7AffineShellPreoutcomeFanoutReceipt),
        (
            preoutcome_destination_binding,
            V1SampledV7AffineShellPreoutcomeDestinationBinding,
        ),
        (
            preoutcome_registration,
            V1LiteralJAffineShellDirectionPreoutcomeRegistration,
        ),
        (conditioned_d1, V1LiteralJConditionedD1ShardReplaySource),
        (measure_design, V1SampledResidualPopulationMeasureDesign),
        (measure_weight_receipt, V1SampledResidualPopulationWeightReceipt),
        (action_custody, V1SignedWorkBlockSupportPreferredActionBinding),
        (
            joint_preoutcome_registration,
            V1LiteralJJointAffineShellPreoutcomeRegistration,
        ),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            "preoutcome issuance requires every exact typed source parent",
        )
    fanout_hash, destination_hash, registration_hash = (
        _validate_complete_destination_member(
            fanout=preoutcome_fanout_receipt,
            destination=preoutcome_destination_binding,
            registration=preoutcome_registration,
        )
    )
    joint_hash = _validate_joint_member(
        fanout_hash=fanout_hash,
        fanout=preoutcome_fanout_receipt,
        destination=preoutcome_destination_binding,
        registration=preoutcome_registration,
        joint=joint_preoutcome_registration,
    )
    d1_hash, design_hash, weight_hash, action_hash = _validate_d1_measure_action(
        registration=preoutcome_registration,
        conditioned_d1=conditioned_d1,
        measure_design=measure_design,
        measure_weight_receipt=measure_weight_receipt,
        action_custody=action_custody,
    )
    logical_hash = _validate_logical_source(
        source=logical_look_source,
        registration=preoutcome_registration,
        require_bound=False,
    )
    d1 = conditioned_d1.replay_receipt
    compact = conditioned_d1.compact_source_receipt
    ordinal = preoutcome_destination_binding.physical_action_ordinal
    receipt = V1LiteralJJointPopulationPreoutcomeReceipt(
        logical_source_receipt_hash=logical_hash,
        fanout_receipt_hash=fanout_hash,
        destination_binding_hash=destination_hash,
        fanout_binding_axis_hash=preoutcome_fanout_receipt.binding_axis_hash,
        physical_action_axis_hash=preoutcome_fanout_receipt.physical_action_axis_hash,
        residual_preoutcome_registration_hash=registration_hash,
        residual_source_registration_hash=(
            preoutcome_registration.source_registration_hash
        ),
        d1_replay_receipt_hash=d1_hash,
        d1_compact_source_receipt_hash=compact.receipt_hash,
        d1_source_registration_hash=compact.registration_hash,
        d1_source_axis_hash=_d1_source_axis_hash(conditioned_d1),
        d1_corrected_evidence_stream_source_hash=(
            compact.corrected_evidence_stream_source_hash
        ),
        d1_literal_j_stream_source_hash=compact.literal_j_stream_source_hash,
        d1_same_capture_vjp_source_hash=compact.same_capture_vjp_source_hash,
        d1_literal_j_shard_receipt_hashes=d1.literal_j_shard_receipt_hashes,
        d1_literal_j_content_sha256s=d1.literal_j_content_sha256s,
        measure_design_hash=design_hash,
        measure_weight_receipt_hash=weight_hash,
        action_custody_hash=action_hash,
        action_source_registration_hash=action_custody.source_registration_hash,
        joint_preoutcome_registration_hash=joint_hash,
        joint_direction_axis_hash=joint_preoutcome_registration.joint_direction_axis_hash,
        joint_direction_ids=joint_preoutcome_registration.joint_direction_ids,
        joint_outside_direction_id=(
            joint_preoutcome_registration.outside_direction_ids[ordinal - 1]
        ),
        physical_action_ordinal=ordinal,
        physical_action_identity_hash=(
            preoutcome_registration.physical_action_identity_hash
        ),
        physical_action_hash=preoutcome_registration.physical_action_hash,
        packed_action_hash=preoutcome_registration.packed_action_hash,
        parent_mask_hash=preoutcome_registration.parent_mask_hash,
        resulting_mask_hash=preoutcome_registration.resulting_mask_hash,
        removed_route_index_hash=preoutcome_registration.removed_route_index_hash,
        removed_count=preoutcome_registration.removed_count,
        outside_direction_source_hash=(
            preoutcome_registration.outside_direction_source_hash
        ),
        model_identity_hash=preoutcome_registration.model_identity_hash,
        chart_identity_hash=preoutcome_registration.chart_identity_hash,
        current_mask_hash=preoutcome_registration.current_mask_hash,
        proposal_generation=preoutcome_registration.proposal_generation,
        route_axis_hash=preoutcome_registration.route_axis_hash,
        route_id_axis_hash=preoutcome_registration.route_id_axis_hash,
        route_count=preoutcome_registration.route_count,
        registered_d_prefix_hash=preoutcome_registration.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=(
            preoutcome_registration.registered_d_w2_prefix_hash
        ),
        shell_freeze_receipt_hash=(
            preoutcome_registration.shell_freeze_receipt_hash
        ),
        shell_indices=preoutcome_registration.shell_indices,
        shell_route_ids=preoutcome_registration.shell_route_ids,
        native_fiber_axis_receipt_hash=(
            preoutcome_registration.native_fiber_axis_receipt_hash
        ),
        native_fiber_measure_hash=preoutcome_registration.native_fiber_measure_hash,
        corrected_evidence_stream_source_hash=(
            preoutcome_registration.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=(
            preoutcome_registration.literal_j_stream_source_hash
        ),
        same_capture_vjp_source_hash=(
            preoutcome_registration.same_capture_vjp_source_hash
        ),
        literal_j_stream_binding_hash=(
            preoutcome_registration.literal_j_stream_binding_hash
        ),
        literal_j_cell_hash=preoutcome_registration.literal_j_cell_hash,
        literal_j_cell_scope_hash=preoutcome_registration.literal_j_cell_scope_hash,
        alpha_lineage_hash=preoutcome_registration.historical_alpha_lineage_hash,
        d1_registered_prefix_hash=d1.registered_d_prefix_hash,
        d1_pair_prefix_axis_hash=d1.d1_pair_prefix_axis_hash,
        d1_context_count=d1.context_count,
        d1_stratum_counts=d1.d1_stratum_counts,
        residual_sample_stratum_counts=(
            measure_weight_receipt.cumulative_dw2_stratum_counts
        ),
        population_stratum_counts=measure_design.population_stratum_counts,
        d1_context_weight_fraction=(
            measure_weight_receipt.d1_context_weight_fraction
        ),
        residual_context_weight_fractions=(
            measure_weight_receipt.dw2_context_weight_fractions
        ),
        residual_stratum_mass_fractions=(
            measure_weight_receipt.residual_stratum_mass_fractions
        ),
        residual_population_mass_fraction=(
            measure_weight_receipt.residual_population_scale_fraction
        ),
        common_measure_denominator=measure_weight_receipt.common_denominator,
        d1_context_quantum=measure_weight_receipt.d1_context_quantum,
        residual_context_quanta=measure_weight_receipt.dw2_context_quanta,
        total_measure_quanta=measure_weight_receipt.total_weight_quanta,
        **_authority_payload(),
        _issuance_token=_PREOUTCOME_TOKEN,
    )
    return V1LiteralJJointPopulationPreoutcomeSource(
        receipt=receipt,
        logical_source=logical_look_source,
        fanout=preoutcome_fanout_receipt,
        destination=preoutcome_destination_binding,
        registration=preoutcome_registration,
        conditioned_d1=conditioned_d1,
        measure_design=measure_design,
        measure_weight_receipt=measure_weight_receipt,
        action_custody=action_custody,
        joint=joint_preoutcome_registration,
        _issuance_token=_PREOUTCOME_OWNER_TOKEN,
    )


def _validate_postsource_receipt(
    *,
    pre: V1LiteralJJointPopulationPreoutcomeReceipt,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    postsource: V1SignedWorkVariableShellSourceReceipt,
    measure_design: V1SampledResidualPopulationMeasureDesign,
    measure_weight_receipt: V1SampledResidualPopulationWeightReceipt,
) -> str:
    source_hash = postsource.receipt_hash
    common = (
        (postsource.registration_hash, pre.residual_source_registration_hash),
        (postsource.model_identity_hash, pre.model_identity_hash),
        (postsource.chart_identity_hash, pre.chart_identity_hash),
        (postsource.current_mask_hash, pre.current_mask_hash),
        (postsource.proposal_generation, pre.proposal_generation),
        (postsource.route_axis_hash, pre.route_axis_hash),
        (postsource.route_id_axis_hash, pre.route_id_axis_hash),
        (postsource.route_count, pre.route_count),
        (postsource.registered_d_prefix_hash, pre.registered_d_prefix_hash),
        (postsource.registered_d_w2_prefix_hash, pre.registered_d_w2_prefix_hash),
        (postsource.shell_freeze_receipt_hash, pre.shell_freeze_receipt_hash),
        (postsource.shell_indices, pre.shell_indices),
        (postsource.shell_route_ids, pre.shell_route_ids),
        (postsource.native_fiber_axis_receipt_hash, pre.native_fiber_axis_receipt_hash),
        (postsource.native_fiber_measure_hash, pre.native_fiber_measure_hash),
        (postsource.kkt_branch_source_receipt_hash, pre.physical_action_hash),
        (
            postsource.corrected_evidence_stream_source_hash,
            pre.corrected_evidence_stream_source_hash,
        ),
        (postsource.literal_j_stream_source_hash, pre.literal_j_stream_source_hash),
        (postsource.same_capture_vjp_source_hash, pre.same_capture_vjp_source_hash),
        (postsource.alpha_lineage_hash, pre.alpha_lineage_hash),
        (postsource.population_count, sum(measure_design.residual_stratum_counts)),
        (
            postsource.context_count,
            sum(
                after - before
                for before, after in zip(
                    registration.before_suffix_stratum_counts,
                    registration.after_suffix_stratum_counts,
                    strict=True,
                )
            ),
        ),
    )
    if (
        any(left != right for left, right in common)
        or measure_design.design_hash != pre.measure_design_hash
        or measure_weight_receipt.receipt_hash != pre.measure_weight_receipt_hash
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.RESIDUAL_SOURCE_MISMATCH,
            "postsource residual left the registered chart, population prefix, shell, stream, alpha, or action",
        )
    return source_hash


@dataclass(frozen=True, slots=True)
class V1LiteralJJointPopulationSourceReceipt:
    preoutcome_receipt_hash: str
    logical_source_receipt_hash: str
    logical_result_hash: str
    logical_result_receipt_hash: str
    residual_preoutcome_registration_hash: str
    residual_source_registration_hash: str
    residual_postsource_receipt_hash: str
    d1_source_axis_hash: str
    d1_source_registration_hash: str
    measure_design_hash: str
    measure_weight_receipt_hash: str
    action_custody_hash: str
    action_source_registration_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    physical_action_ordinal: int
    physical_action_identity_hash: str
    physical_action_hash: str
    packed_action_hash: str
    provider_outcomes_seen: bool = True
    exact_logical_result_bound: bool = True
    independently_bound_source_roles: bool = True
    source_registration_equality_required: bool = False
    array_free_durable_receipt: bool = True
    provider_execution_authority: bool = False
    provider_result_authority: bool = False
    alpha_allocation_authority: bool = False
    alpha_spend_authority: bool = False
    literal_j_projection_authority: bool = False
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    loewner_or_kkt_authority: bool = False
    solver_authority: bool = False
    d_sampling_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_selection_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_POPULATION_SOURCE_AUTHORITY
    schema: str = LITERAL_J_JOINT_POPULATION_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _POSTSOURCE_TOKEN:
            raise PermissionError("joint-population source receipts are binder-only")
        for name in (
            "preoutcome_receipt_hash",
            "logical_source_receipt_hash",
            "logical_result_hash",
            "logical_result_receipt_hash",
            "residual_preoutcome_registration_hash",
            "residual_source_registration_hash",
            "residual_postsource_receipt_hash",
            "d1_source_axis_hash",
            "d1_source_registration_hash",
            "measure_design_hash",
            "measure_weight_receipt_hash",
            "action_custody_hash",
            "action_source_registration_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "physical_action_identity_hash",
            "physical_action_hash",
            "packed_action_hash",
        ):
            _identity(getattr(self, name), name=name)
        _positive_int(self.physical_action_ordinal, name="physical action ordinal")
        if (
            self.provider_outcomes_seen is not True
            or self.exact_logical_result_bound is not True
            or self.independently_bound_source_roles is not True
            or self.source_registration_equality_required is not False
            or self.array_free_durable_receipt is not True
            or not _closed_authority(self)
            or self.schema != LITERAL_J_JOINT_POPULATION_SOURCE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.AUTHORITY_VIOLATION,
                "postsource receipt is incomplete or widened",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "postsource-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                "joint-population source receipt changed after binding",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointPopulationSource:
    """Live postsource dual-source/action binding consumed by direction issuers."""

    __slots__ = (
        "receipt",
        "preoutcome_source",
        "logical_result",
        "postsource_receipt",
        "_state",
        "_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointPopulationSourceReceipt,
        preoutcome_source: V1LiteralJJointPopulationPreoutcomeSource,
        logical_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
        postsource_receipt: V1SignedWorkVariableShellSourceReceipt,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _POSTSOURCE_OWNER_TOKEN:
            raise PermissionError("joint-population source owners are binder-only")
        self.receipt = receipt
        self.preoutcome_source = preoutcome_source
        self.logical_result = logical_result
        self.postsource_receipt = postsource_receipt
        self._state = "live"
        self._lock = RLock()
        self.assert_live()

    @property
    def preoutcome_receipt(self) -> V1LiteralJJointPopulationPreoutcomeReceipt:
        return self.preoutcome_source.receipt

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def assert_live(self) -> str:
        with self._lock:
            if self._state == "evicted":
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.EVICTED,
                    "joint-population source was evicted",
                )
            self.receipt.assert_integrity()
            pre_owner = self.preoutcome_source
            pre = pre_owner.receipt
            pre.assert_integrity()
            pre_owner._replay_common()
            logical_hash = _validate_logical_source(
                source=pre_owner._logical_source,
                registration=pre_owner._registration,
                require_bound=True,
            )
            try:
                result_hash = self.logical_result.assert_live()
                result_receipt_hash = self.logical_result.receipt.assert_integrity()
            except Exception as error:
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                    "logical result failed live replay: "
                    f"{type(error).__name__}: {error}",
                )
            if (
                self.logical_result.source is not pre_owner._logical_source
                or pre_owner._logical_source._bound_result is not self.logical_result
                or self.logical_result.receipt.source_receipt_hash
                != pre.logical_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.OUTCOME_REQUIRED,
                    "postsource binding does not own the exact result of its preregistered logical source",
                )
            post_hash = _validate_postsource_receipt(
                pre=pre,
                registration=pre_owner._registration,
                postsource=self.postsource_receipt,
                measure_design=pre_owner._measure_design,
                measure_weight_receipt=pre_owner._measure_weight_receipt,
            )
            if (
                self.receipt.preoutcome_receipt_hash != pre.receipt_hash
                or self.receipt.logical_source_receipt_hash != logical_hash
                or self.receipt.logical_result_hash != result_hash
                or self.receipt.logical_result_receipt_hash != result_receipt_hash
                or self.receipt.residual_postsource_receipt_hash != post_hash
            ):
                _refuse(
                    V1LiteralJJointPopulationSourceRefusalCode.STALE_SOURCE,
                    "postsource binding changed one exact source parent",
                )
            return self.receipt.receipt_hash

    def materialize_action_float64(
        self,
        *,
        device: object,
        tile_bytes: int,
        production_cuda_required: bool,
    ) -> tuple[Any, int, int]:
        with self._lock:
            self.assert_live()
            return self.preoutcome_source.materialize_action_float64(
                device=device,
                tile_bytes=tile_bytes,
                production_cuda_required=production_cuda_required,
            )

    def evict(self) -> None:
        with self._lock:
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint-population source owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("joint-population source owners are nonserializable")


def bind_v1_literal_j_joint_population_source(
    *,
    preoutcome_source: V1LiteralJJointPopulationPreoutcomeSource,
    logical_look_result: V1SampledV7SharedDW2LiteralJLogicalLookResult,
    postsource_receipt: V1SignedWorkVariableShellSourceReceipt,
) -> V1LiteralJJointPopulationSource:
    """Bind the exact residual result without redefining its frozen action axis."""

    if (
        type(preoutcome_source) is not V1LiteralJJointPopulationPreoutcomeSource
        or type(logical_look_result)
        is not V1SampledV7SharedDW2LiteralJLogicalLookResult
        or type(postsource_receipt) is not V1SignedWorkVariableShellSourceReceipt
    ):
        _refuse(
            V1LiteralJJointPopulationSourceRefusalCode.INVALID_TYPE,
            "postsource binding requires the exact preoutcome owner, logical result, and residual receipt",
        )
    with preoutcome_source._lock:
        if preoutcome_source._state != "preoutcome":
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.BIND_REPLAY,
                "preoutcome source may be bound exactly once",
            )
        pre = preoutcome_source.receipt
        pre.assert_integrity()
        # The expected transition is unbound -> exact bound result.  Replay
        # common parents without calling assert_preoutcome_live(), which must
        # now fail by construction because provider outcomes have landed.
        preoutcome_source._replay_common()
        logical_hash = _validate_logical_source(
            source=preoutcome_source._logical_source,
            registration=preoutcome_source._registration,
            require_bound=True,
        )
        try:
            result_hash = logical_look_result.assert_live()
            result_receipt_hash = logical_look_result.receipt.assert_integrity()
        except Exception as error:
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.OUTCOME_REQUIRED,
                "logical result failed exact postsource replay: "
                f"{type(error).__name__}: {error}",
            )
        if (
            logical_look_result.source is not preoutcome_source._logical_source
            or preoutcome_source._logical_source._bound_result is not logical_look_result
            or logical_look_result.receipt.source_receipt_hash
            != pre.logical_source_receipt_hash
            or logical_hash != pre.logical_source_receipt_hash
        ):
            _refuse(
                V1LiteralJJointPopulationSourceRefusalCode.OUTCOME_REQUIRED,
                "logical result is not the exact outcome of the preregistered source",
            )
        postsource_hash = _validate_postsource_receipt(
            pre=pre,
            registration=preoutcome_source._registration,
            postsource=postsource_receipt,
            measure_design=preoutcome_source._measure_design,
            measure_weight_receipt=preoutcome_source._measure_weight_receipt,
        )
        receipt = V1LiteralJJointPopulationSourceReceipt(
            preoutcome_receipt_hash=pre.receipt_hash,
            logical_source_receipt_hash=logical_hash,
            logical_result_hash=result_hash,
            logical_result_receipt_hash=result_receipt_hash,
            residual_preoutcome_registration_hash=(
                pre.residual_preoutcome_registration_hash
            ),
            residual_source_registration_hash=pre.residual_source_registration_hash,
            residual_postsource_receipt_hash=postsource_hash,
            d1_source_axis_hash=pre.d1_source_axis_hash,
            d1_source_registration_hash=pre.d1_source_registration_hash,
            measure_design_hash=pre.measure_design_hash,
            measure_weight_receipt_hash=pre.measure_weight_receipt_hash,
            action_custody_hash=pre.action_custody_hash,
            action_source_registration_hash=pre.action_source_registration_hash,
            joint_preoutcome_registration_hash=(
                pre.joint_preoutcome_registration_hash
            ),
            joint_direction_axis_hash=pre.joint_direction_axis_hash,
            physical_action_ordinal=pre.physical_action_ordinal,
            physical_action_identity_hash=pre.physical_action_identity_hash,
            physical_action_hash=pre.physical_action_hash,
            packed_action_hash=pre.packed_action_hash,
            **_authority_payload(),
            _issuance_token=_POSTSOURCE_TOKEN,
        )
        preoutcome_source._state = "consumed"
        return V1LiteralJJointPopulationSource(
            receipt=receipt,
            preoutcome_source=preoutcome_source,
            logical_result=logical_look_result,
            postsource_receipt=postsource_receipt,
            _issuance_token=_POSTSOURCE_OWNER_TOKEN,
        )


__all__ = [
    "LITERAL_J_JOINT_POPULATION_PREOUTCOME_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_SOURCE_SCHEMA",
    "LITERAL_J_JOINT_POPULATION_SOURCE_AUTHORITY",
    "V1LiteralJJointPopulationSourceRefusalCode",
    "V1LiteralJJointPopulationSourceRefusal",
    "V1LiteralJJointPopulationPreoutcomeReceipt",
    "V1LiteralJJointPopulationPreoutcomeSource",
    "V1LiteralJJointPopulationSourceReceipt",
    "V1LiteralJJointPopulationSource",
    "issue_v1_literal_j_joint_population_preoutcome_source",
    "bind_v1_literal_j_joint_population_source",
]
