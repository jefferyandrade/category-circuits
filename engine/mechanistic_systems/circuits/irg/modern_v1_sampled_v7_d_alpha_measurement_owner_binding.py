"""Bind registered sampled-V7 D alpha cells to one live measurement owner.

The alpha-family registry and its durable compare-and-swap receipt establish
that one pre-outcome A/F/J family was registered once.  The branch-independent
``D`` measurement owner establishes a live full-population prefix/capture
source, but its historical alpha strings were minted before that registry
existed and deliberately have no spend authority.  This module joins those
seams without reinterpreting the historical strings.

The exact :class:`V1SampledV7DAlphaFamilyPlan` is required only as a semantic
replay witness.  The v2 registry exposes the plan hash and scalar draw counts,
but not the two per-stratum look counts needed to reject a target/complement
swap.  The plan supplies those hidden coordinates; it grants no additional
authority.

The alpha plan and its A/F/J cells live on the ordered ordinal axis, whereas
the measurement owner exposes the physical mask-bit axis.  An exact
:class:`V1OrdinalPhysicalAxisBridgeReceipt` is consequently mandatory.  A
same-axis shortcut is refused: the receipt must prove the registered nontrivial
position-preserving bijection, chart mask, and route count explicitly.

The resulting live receipt binds one registered look's signed-F and literal-J
cells to the resident-D1 plus residual-suffix population prefix and to the
owner's same-capture streams.  It authorizes no acquisition, alpha spend,
statistical decision, C/K/O work, action, commit, terminal claim, performance
claim, or Pareto claim.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_d_alpha_family_registry import (
    V1SampledV7DAlphaCellRole,
    V1SampledV7DAlphaFamilyCell,
    V1SampledV7DAlphaFamilyPlan,
    V1SampledV7DAlphaFamilyRegistryReceipt,
)
from .modern_v1_sampled_v7_d_alpha_head_cas import (
    V1SampledV7DAlphaHeadCASReceipt,
)
from .modern_v1_response_exactification import (
    V1OrdinalPhysicalAxisBridgeReceipt,
)
from .modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyMeasurementOwner,
)


SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_measurement_owner_binding_v1"
)
SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_AUTHORITY: Final = (
    "sampled_v7_D_registered_F_J_cell_to_live_measurement_owner_identity_"
    "binding_only_no_acquisition_alpha_spend_D_decision_C_K_O_action_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_D_HISTORICAL_OWNER_ALPHA_SEMANTICS: Final = (
    "inert_compatibility_coordinates_only_not_the_registered_alpha_family_"
    "ledger_cell_or_spend_authority_v1"
)

_NON_A_SCOPE_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_v7_D_non_A_alpha_cell_scope_v1"
)
_NON_A_SCOPE_PREFIX: Final = (
    "partition-free-v1-alexnet-sampled-v7-D-non-A-alpha-scope-"
)
_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-alpha-measurement-owner-binding-"
)
_PREFIX_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-alpha-owner-prefix-axis-"
)
_STREAM_BINDING_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-alpha-owner-stream-binding-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    REGISTRY_CAS_MISMATCH = "registry_cas_mismatch"
    PLAN_MISMATCH = "plan_mismatch"
    CAMPAIGN_MISMATCH = "campaign_mismatch"
    GENERATION_MISMATCH = "generation_mismatch"
    LOOK_MISMATCH = "look_mismatch"
    POPULATION_MISMATCH = "population_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    STREAM_MISMATCH = "stream_mismatch"
    OWNER_SUBSTITUTION = "owner_substitution"
    AXIS_BRIDGE_MISMATCH = "axis_bridge_mismatch"
    AXIS_BRIDGE_SUBSTITUTION = "axis_bridge_substitution"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7DAlphaMeasurementOwnerBindingRefusal(RuntimeError):
    """Typed fail-closed refusal at the alpha/measurement-owner join."""

    def __init__(
        self,
        code: V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode:
            raise TypeError("alpha-owner binding refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DAlphaMeasurementOwnerBindingRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _look_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
            "registered look index must be one exact integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.LOOK_MISMATCH,
            "registered look indices are one-based and positive",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable target/complement count pair",
        )
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
                f"{name} entries must be exact integers",
            )
        count = int(item)
        if count < 0:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
                f"{name} entries must be nonnegative",
            )
        result.append(count)
    return result[0], result[1]


def _assert_parent(
    value: object,
    expected_type: type,
    *,
    name: str,
) -> str:
    if type(value) is not expected_type:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
            f"{name} requires the exact {expected_type.__name__} type",
        )
    try:
        if expected_type is V1SampledV7DOnlyMeasurementOwner:
            return value.assert_live()  # type: ignore[union-attr]
        return value.assert_integrity()  # type: ignore[union-attr]
    except V1SampledV7DAlphaMeasurementOwnerBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PARENT_INTEGRITY,
            f"{name} failed replay: {type(error).__name__}: {error}",
        )


def _cell_for(
    cells: tuple[V1SampledV7DAlphaFamilyCell, ...],
    *,
    role: V1SampledV7DAlphaCellRole,
    look_index: int,
) -> V1SampledV7DAlphaFamilyCell:
    cell_id = (
        f"corrected-A-order-look:{look_index:08d}"
        if role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
        else f"{role.value}-look:{look_index:08d}"
    )
    matches = tuple(
        cell for cell in cells if cell.role is role and cell.cell_id == cell_id
    )
    if len(matches) != 1:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.LOOK_MISMATCH,
            f"registered look lacks exactly one {role.value} cell",
        )
    return matches[0]


def _expected_non_a_scope_hash(
    *,
    plan: V1SampledV7DAlphaFamilyPlan,
    frame_hash: str,
    role: V1SampledV7DAlphaCellRole,
    look_index: int,
    look_counts: tuple[int, int],
) -> str:
    estimand_by_role = {
        V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS: (
            "same_capture_native_fiber_signed_corrected_A_projection_v1"
        ),
        V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING: (
            "same_capture_literal_relaxed_gate_raw_target_logit_J_projection_v1"
        ),
    }
    if role not in estimand_by_role:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
            "the owner join accepts only signed-F and literal-J cells",
        )
    return content_hash(
        {
            "schema": _NON_A_SCOPE_SCHEMA,
            "role": role.value,
            "estimand": estimand_by_role[role],
            "same_capture_dual_chart_required": True,
            "campaign_contract_hash": plan.campaign_contract_hash,
            "model_identity_hash": plan.model_identity_hash,
            "query_contract_hash": plan.query_contract_hash,
            "preprocessing_hash": plan.preprocessing_hash,
            "current_mask_hash": plan.current_mask_hash,
            "raw_logit_stream_source_hash": plan.raw_logit_stream_source_hash,
            "proposal_generation": plan.proposal_generation,
            "route_count": plan.route_count,
            "route_axis_hash": plan.route_axis_hash,
            "stratum_names": plan.stratum_names,
            "population_stratum_counts": plan.population_stratum_counts,
            "sealed_population_order_hashes": plan.sealed_population_order_hashes,
            "sampling_law_hash": plan.sampling_law_hash,
            "generation_sampling_frame_receipt_hash": frame_hash,
            "look_index": look_index,
            "registered_look_stratum_counts": look_counts,
        },
        prefix=_NON_A_SCOPE_PREFIX,
    )


def _joined_coordinates(
    *,
    alpha_family_plan: V1SampledV7DAlphaFamilyPlan,
    alpha_family_registry: V1SampledV7DAlphaFamilyRegistryReceipt,
    alpha_head_cas: V1SampledV7DAlphaHeadCASReceipt,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    ordinal_physical_axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt,
    registered_look_index: int,
    registered_look_stratum_counts: tuple[int, int],
) -> dict[str, object]:
    plan_hash = _assert_parent(
        alpha_family_plan,
        V1SampledV7DAlphaFamilyPlan,
        name="alpha-family plan witness",
    )
    registry_hash = _assert_parent(
        alpha_family_registry,
        V1SampledV7DAlphaFamilyRegistryReceipt,
        name="alpha-family registry",
    )
    cas_hash = _assert_parent(
        alpha_head_cas,
        V1SampledV7DAlphaHeadCASReceipt,
        name="durable alpha-head CAS",
    )
    owner_hash = _assert_parent(
        measurement_owner,
        V1SampledV7DOnlyMeasurementOwner,
        name="live D measurement owner",
    )
    axis_bridge_hash = _assert_parent(
        ordinal_physical_axis_bridge,
        V1OrdinalPhysicalAxisBridgeReceipt,
        name="ordinal-to-physical route-axis bridge",
    )
    look = _look_index(registered_look_index)
    supplied_counts = _count_pair(
        registered_look_stratum_counts,
        name="registered look stratum counts",
    )

    if (
        alpha_family_registry.alpha_family_plan_hash != plan_hash
        or alpha_head_cas.alpha_family_plan_hash != plan_hash
        or alpha_family_registry.master_registration_hash
        != alpha_family_plan.master_registration_hash
        or alpha_head_cas.master_registration_hash
        != alpha_family_plan.master_registration_hash
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PLAN_MISMATCH,
            "registry and durable head do not replay the exact semantic plan witness",
        )
    if (
        alpha_head_cas.alpha_family_registry_receipt_hash != registry_hash
        or alpha_head_cas.published_ledger_head_hash
        != alpha_family_registry.ledger_after_hash
        or alpha_head_cas.expected_predecessor_ledger_hash
        != alpha_family_registry.ledger_before_hash
        or alpha_head_cas.durable_controller_head_closed is not True
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.REGISTRY_CAS_MISMATCH,
            "durable head is not the exact published successor of this registry",
        )
    if (
        alpha_family_registry.cell_axis_hash != alpha_family_plan.cell_axis_hash
        or tuple(cell.cell_hash for cell in alpha_family_registry.cell_axis)
        != tuple(cell.cell_hash for cell in alpha_family_plan.cell_axis)
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PLAN_MISMATCH,
            "registry cell axis differs from the semantic plan witness",
        )

    frame = measurement_owner._frame
    discovery = measurement_owner._discovery_stream
    suffix = measurement_owner._suffix_registration
    measure = measurement_owner._residual_measure_design
    lineage = measurement_owner.measurement_source_lineage
    resident = measurement_owner._resident_d1_population.receipt
    contract = measurement_owner._campaign_binding.contract
    calibration = measurement_owner._calibration
    physical_axis = measurement_owner._physical_route_axis

    if (
        alpha_family_plan.campaign_contract_hash
        != contract.campaign_contract_hash
        or alpha_family_registry.campaign_contract_hash
        != contract.campaign_contract_hash
        or alpha_head_cas.campaign_contract_hash
        != contract.campaign_contract_hash
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.CAMPAIGN_MISMATCH,
            "plan, registry, durable head, and live owner name different campaigns",
        )
    if (
        alpha_family_plan.proposal_generation != frame.generation_index
        or alpha_family_registry.proposal_generation != frame.generation_index
        or alpha_head_cas.proposal_generation != frame.generation_index
        or lineage.proposal_generation != frame.generation_index
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.GENERATION_MISMATCH,
            "alpha parents and live owner name different proposal generations",
        )

    population_counts = frame.full_population_stratum_counts
    order_hashes = (
        discovery.target_assignment_receipt.content_hash,
        discovery.complement_assignment_receipt.content_hash,
    )
    population_membership_hashes = (
        discovery.target_population_membership_hash,
        discovery.complement_population_membership_hash,
    )
    if (
        alpha_family_plan.stratum_names != ("target", "complement")
        or alpha_family_plan.population_stratum_counts != population_counts
        or alpha_family_plan.sealed_population_order_hashes != order_hashes
        or measure.population_stratum_counts != population_counts
        or suffix.population_stratum_counts != population_counts
        or discovery.full_population_stratum_counts != population_counts
        or measure.population_count != sum(population_counts)
        or alpha_family_plan.sampling_law_hash != contract.sampling_law_hash
        or alpha_family_plan.model_identity_hash != lineage.model_identity_hash
        or alpha_family_plan.query_contract_hash != calibration.query_hash
        or alpha_family_plan.current_mask_hash != lineage.current_mask_hash
        or alpha_family_plan.current_mask_hash != frame.current_mask_hash
        or alpha_family_plan.raw_logit_stream_source_hash
        != frame.frozen_transform_hash
        or alpha_family_plan.route_count != physical_axis.route_count
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.POPULATION_MISMATCH,
            "plan coordinates do not replay the owner's exact D population/chart",
        )
    if (
        ordinal_physical_axis_bridge.ordered_route_axis_hash
        == ordinal_physical_axis_bridge.physical_route_axis_hash
        or ordinal_physical_axis_bridge.ordered_route_axis_hash
        != alpha_family_plan.route_axis_hash
        or ordinal_physical_axis_bridge.physical_route_axis_hash
        != lineage.route_axis_hash
        or ordinal_physical_axis_bridge.physical_route_axis_hash
        != physical_axis.route_axis_hash
        or ordinal_physical_axis_bridge.current_mask_hash
        != alpha_family_plan.current_mask_hash
        or ordinal_physical_axis_bridge.current_mask_hash
        != lineage.current_mask_hash
        or ordinal_physical_axis_bridge.route_count != alpha_family_plan.route_count
        or ordinal_physical_axis_bridge.route_count != physical_axis.route_count
        or ordinal_physical_axis_bridge.position_rule
        != "ordered_ordinal_i_maps_to_physical_mask_bit_i"
        or ordinal_physical_axis_bridge.full_position_bijection_verified is not True
        or ordinal_physical_axis_bridge.prior_removed_routes_fixed is not True
        or ordinal_physical_axis_bridge.endpoint_actions_contain_new_current_mask_deltas_only
        is not True
        or ordinal_physical_axis_bridge.mask_commit_authority is not False
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.AXIS_BRIDGE_MISMATCH,
            "ordered alpha cells and physical measurement routes lack their exact "
            "nontrivial position-preserving bijection",
        )
    if look > len(alpha_family_plan.registered_look_stratum_counts):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.LOOK_MISMATCH,
            "registered look index exceeds the pre-outcome plan axis",
        )
    plan_counts = alpha_family_plan.registered_look_stratum_counts[look - 1]
    if supplied_counts != plan_counts:
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.LOOK_MISMATCH,
            "supplied target/complement counts do not equal the selected plan look",
        )

    d1_counts = measure.d1_stratum_counts
    if any(plan_counts[index] < d1_counts[index] for index in (0, 1)):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PREFIX_MISMATCH,
            "selected look precedes the resident-D1 prefix owned by this source",
        )
    residual_prefix_counts = tuple(
        plan_counts[index] - d1_counts[index] for index in (0, 1)
    )
    if any(
        residual_prefix_counts[index] > measure.residual_stratum_counts[index]
        for index in (0, 1)
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PREFIX_MISMATCH,
            "selected look exceeds the source-owned residual suffix",
        )

    target_order = tuple(discovery.target_assignment_receipt.generated_order)
    complement_order = tuple(
        discovery.complement_assignment_receipt.generated_order
    )
    scheduled_target = tuple(
        context
        for packet in measurement_owner.schedule.packets
        for context, stratum in zip(
            packet.registered_context_ids,
            packet.registered_stratum_ids,
            strict=True,
        )
        if stratum == "target"
    )
    scheduled_complement = tuple(
        context
        for packet in measurement_owner.schedule.packets
        for context, stratum in zip(
            packet.registered_context_ids,
            packet.registered_stratum_ids,
            strict=True,
        )
        if stratum == "complement"
    )
    target_suffix = target_order[d1_counts[0] :]
    complement_suffix = complement_order[d1_counts[1] :]
    if (
        scheduled_target != target_suffix
        or scheduled_complement != complement_suffix
        or tuple(len(axis) for axis in (scheduled_target, scheduled_complement))
        != measure.residual_stratum_counts
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.PREFIX_MISMATCH,
            "measurement schedule does not replay the exact discovery suffix axes",
        )
    target_prefix = target_order[: plan_counts[0]]
    complement_prefix = complement_order[: plan_counts[1]]
    residual_target_prefix = scheduled_target[: residual_prefix_counts[0]]
    residual_complement_prefix = scheduled_complement[: residual_prefix_counts[1]]
    prefix_axis_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
            "look_index": look,
            "registered_look_stratum_counts": plan_counts,
            "population_stratum_counts": population_counts,
            "population_membership_hashes": population_membership_hashes,
            "sealed_population_order_hashes": order_hashes,
            "target_prefix_context_ids": target_prefix,
            "complement_prefix_context_ids": complement_prefix,
        },
        prefix=_PREFIX_AXIS_PREFIX + "full-",
    )
    residual_prefix_axis_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
            "resident_d1_stratum_counts": d1_counts,
            "residual_prefix_stratum_counts": residual_prefix_counts,
            "target_residual_prefix_context_ids": residual_target_prefix,
            "complement_residual_prefix_context_ids": residual_complement_prefix,
            "measurement_schedule_hash": measurement_owner.schedule.schedule_hash,
            "suffix_registration_hash": suffix.receipt_hash,
        },
        prefix=_PREFIX_AXIS_PREFIX + "residual-",
    )

    corrected = _cell_for(
        alpha_family_registry.cell_axis,
        role=V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER,
        look_index=look,
    )
    signed_f = _cell_for(
        alpha_family_registry.cell_axis,
        role=V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS,
        look_index=look,
    )
    literal_j = _cell_for(
        alpha_family_registry.cell_axis,
        role=V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING,
        look_index=look,
    )
    expected_population = sum(population_counts)
    expected_draws = sum(plan_counts)
    expected_census = plan_counts == population_counts
    for cell in (corrected, signed_f, literal_j):
        if (
            cell.population_size != expected_population
            or cell.draw_count != expected_draws
            or cell.exact_census is not expected_census
        ):
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.POPULATION_MISMATCH,
                "selected A/F/J cells do not share the exact registered D look",
            )
    for cell, role in (
        (signed_f, V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS),
        (literal_j, V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING),
    ):
        expected_scope = _expected_non_a_scope_hash(
            plan=alpha_family_plan,
            frame_hash=frame.receipt_hash,
            role=role,
            look_index=look,
            look_counts=plan_counts,
        )
        if cell.cell_scope_hash != expected_scope:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.LOOK_MISMATCH,
                f"{role.value} cell scope does not replay the selected plan look",
            )

    stream_axis = (
        measurement_owner.residual_witness_stream_source_hash,
        measurement_owner.same_capture_vjp_source_hash,
        measurement_owner.corrected_evidence_stream_source_hash,
        measurement_owner.residual_literal_j_stream_source_hash,
    )
    if (
        len(set(stream_axis)) != len(stream_axis)
        or lineage.witness_stream_source_hash != stream_axis[0]
        or lineage.same_capture_vjp_source_hash != stream_axis[1]
        or lineage.corrected_evidence_stream_source_hash != stream_axis[2]
        or lineage.literal_j_stream_source_hash != stream_axis[3]
        or lineage.capture_policy_identity_hash
        != measurement_owner.capture_policy_identity_hash
    ):
        _refuse(
            V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.STREAM_MISMATCH,
            "owner lost its exact distinct same-capture residual stream axis",
        )
    signed_f_stream_binding_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
            "cell_hash": signed_f.cell_hash,
            "cell_scope_hash": signed_f.cell_scope_hash,
            "registered_prefix_axis_hash": prefix_axis_hash,
            "residual_prefix_axis_hash": residual_prefix_axis_hash,
            "resident_d1_receipt_hash": resident.receipt_hash,
            "resident_witness_stream_source_hash": resident.witness_stream_source_hash,
            "resident_corrected_evidence_stream_source_hash": (
                resident.corrected_evidence_stream_source_hash
            ),
            "residual_witness_stream_source_hash": stream_axis[0],
            "same_capture_vjp_source_hash": stream_axis[1],
            "residual_corrected_evidence_stream_source_hash": stream_axis[2],
            "capture_policy_identity_hash": (
                measurement_owner.capture_policy_identity_hash
            ),
        },
        prefix=_STREAM_BINDING_PREFIX + "signed-F-",
    )
    literal_j_stream_binding_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
            "cell_hash": literal_j.cell_hash,
            "cell_scope_hash": literal_j.cell_scope_hash,
            "registered_prefix_axis_hash": prefix_axis_hash,
            "residual_prefix_axis_hash": residual_prefix_axis_hash,
            "resident_d1_receipt_hash": resident.receipt_hash,
            "resident_literal_j_stream_source_hash": (
                resident.literal_j_stream_source_hash
            ),
            "residual_witness_stream_source_hash": stream_axis[0],
            "same_capture_vjp_source_hash": stream_axis[1],
            "residual_literal_j_stream_source_hash": stream_axis[3],
            "capture_policy_identity_hash": (
                measurement_owner.capture_policy_identity_hash
            ),
        },
        prefix=_STREAM_BINDING_PREFIX + "literal-J-",
    )
    dual_stream_binding_hash = content_hash(
        {
            "signed_f_stream_binding_hash": signed_f_stream_binding_hash,
            "literal_j_stream_binding_hash": literal_j_stream_binding_hash,
            "same_capture_vjp_source_hash": stream_axis[1],
            "same_registered_prefix": True,
        },
        prefix=_STREAM_BINDING_PREFIX + "dual-",
    )

    return {
        "alpha_family_plan_hash": plan_hash,
        "alpha_family_registry_receipt_hash": registry_hash,
        "alpha_head_cas_receipt_hash": cas_hash,
        "published_ledger_head_hash": alpha_head_cas.published_ledger_head_hash,
        "measurement_owner_hash": owner_hash,
        "ordinal_physical_axis_bridge_receipt_hash": axis_bridge_hash,
        "campaign_contract_hash": contract.campaign_contract_hash,
        "controller_run_hash": alpha_family_registry.controller_run_hash,
        "proposal_generation": frame.generation_index,
        "generation_frame_receipt_hash": frame.receipt_hash,
        "discovery_stream_receipt_hash": discovery.receipt_hash,
        "measurement_schedule_hash": measurement_owner.schedule.schedule_hash,
        "measurement_source_lineage_hash": lineage.lineage_hash,
        "residual_population_measure_design_hash": measure.design_hash,
        "resident_d1_receipt_hash": resident.receipt_hash,
        "suffix_registration_hash": suffix.receipt_hash,
        "model_identity_hash": lineage.model_identity_hash,
        "chart_identity_hash": lineage.chart_identity_hash,
        "current_mask_hash": lineage.current_mask_hash,
        "query_contract_hash": alpha_family_plan.query_contract_hash,
        "preprocessing_hash": alpha_family_plan.preprocessing_hash,
        "raw_logit_stream_source_hash": alpha_family_plan.raw_logit_stream_source_hash,
        "ordered_route_axis_hash": alpha_family_plan.route_axis_hash,
        "physical_route_axis_hash": physical_axis.route_axis_hash,
        "route_count": alpha_family_plan.route_count,
        "axis_bridge_already_removed_route_index_hash": (
            ordinal_physical_axis_bridge.already_removed_route_index_hash
        ),
        "axis_bridge_endpoint_new_removal_counts": (
            ordinal_physical_axis_bridge.endpoint_new_removal_counts
        ),
        "axis_bridge_ordered_endpoint_index_hashes": (
            ordinal_physical_axis_bridge.ordered_endpoint_index_hashes
        ),
        "axis_bridge_physical_endpoint_index_hashes": (
            ordinal_physical_axis_bridge.physical_endpoint_index_hashes
        ),
        "axis_bridge_position_rule": ordinal_physical_axis_bridge.position_rule,
        "axis_bridge_full_position_bijection_verified": (
            ordinal_physical_axis_bridge.full_position_bijection_verified
        ),
        "sampling_law_hash": alpha_family_plan.sampling_law_hash,
        "stratum_names": alpha_family_plan.stratum_names,
        "population_stratum_counts": population_counts,
        "population_membership_hashes": population_membership_hashes,
        "sealed_population_order_hashes": order_hashes,
        "registered_look_index": look,
        "registered_look_stratum_counts": plan_counts,
        "resident_d1_stratum_counts": d1_counts,
        "residual_prefix_stratum_counts": residual_prefix_counts,
        "registered_prefix_axis_hash": prefix_axis_hash,
        "residual_prefix_axis_hash": residual_prefix_axis_hash,
        "corrected_a_cell_id": corrected.cell_id,
        "corrected_a_cell_hash": corrected.cell_hash,
        "corrected_a_cell_scope_hash": corrected.cell_scope_hash,
        "corrected_a_cell_alpha": corrected.alpha_spend,
        "signed_f_cell_id": signed_f.cell_id,
        "signed_f_cell_hash": signed_f.cell_hash,
        "signed_f_cell_scope_hash": signed_f.cell_scope_hash,
        "signed_f_cell_alpha": signed_f.alpha_spend,
        "literal_j_cell_id": literal_j.cell_id,
        "literal_j_cell_hash": literal_j.cell_hash,
        "literal_j_cell_scope_hash": literal_j.cell_scope_hash,
        "literal_j_cell_alpha": literal_j.alpha_spend,
        "capture_policy_identity_hash": measurement_owner.capture_policy_identity_hash,
        "residual_witness_stream_source_hash": stream_axis[0],
        "same_capture_vjp_source_hash": stream_axis[1],
        "corrected_evidence_stream_source_hash": stream_axis[2],
        "literal_j_stream_source_hash": stream_axis[3],
        "signed_f_stream_binding_hash": signed_f_stream_binding_hash,
        "literal_j_stream_binding_hash": literal_j_stream_binding_hash,
        "dual_stream_binding_hash": dual_stream_binding_hash,
        "historical_owner_alpha_ledger_hash": measurement_owner.alpha_ledger_hash,
        "historical_owner_alpha_coordinate_hash": (
            measurement_owner.alpha_coordinate_hash
        ),
        "historical_owner_alpha_lineage_hash": measurement_owner.alpha_lineage_hash,
        "historical_owner_familywise_alpha": measurement_owner.familywise_alpha,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
    alpha_family_plan_hash: str
    alpha_family_registry_receipt_hash: str
    alpha_head_cas_receipt_hash: str
    published_ledger_head_hash: str
    measurement_owner_hash: str
    ordinal_physical_axis_bridge_receipt_hash: str
    campaign_contract_hash: str
    controller_run_hash: str
    proposal_generation: int
    generation_frame_receipt_hash: str
    discovery_stream_receipt_hash: str
    measurement_schedule_hash: str
    measurement_source_lineage_hash: str
    residual_population_measure_design_hash: str
    resident_d1_receipt_hash: str
    suffix_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    raw_logit_stream_source_hash: str
    ordered_route_axis_hash: str
    physical_route_axis_hash: str
    route_count: int
    axis_bridge_already_removed_route_index_hash: str
    axis_bridge_endpoint_new_removal_counts: tuple[int, ...]
    axis_bridge_ordered_endpoint_index_hashes: tuple[str, ...]
    axis_bridge_physical_endpoint_index_hashes: tuple[str, ...]
    axis_bridge_position_rule: str
    axis_bridge_full_position_bijection_verified: bool
    sampling_law_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    population_membership_hashes: tuple[str, str]
    sealed_population_order_hashes: tuple[str, str]
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    resident_d1_stratum_counts: tuple[int, int]
    residual_prefix_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    residual_prefix_axis_hash: str
    corrected_a_cell_id: str
    corrected_a_cell_hash: str
    corrected_a_cell_scope_hash: str
    corrected_a_cell_alpha: float
    signed_f_cell_id: str
    signed_f_cell_hash: str
    signed_f_cell_scope_hash: str
    signed_f_cell_alpha: float
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_cell_alpha: float
    capture_policy_identity_hash: str
    residual_witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    signed_f_stream_binding_hash: str
    literal_j_stream_binding_hash: str
    dual_stream_binding_hash: str
    historical_owner_alpha_ledger_hash: str
    historical_owner_alpha_coordinate_hash: str
    historical_owner_alpha_lineage_hash: str
    historical_owner_familywise_alpha: float
    historical_owner_alpha_coordinates_inert: bool = True
    plan_is_semantic_witness_only: bool = True
    durable_controller_head_closed: bool = True
    downstream_measurement_owner_binding_closed: bool = True
    registered_a_f_j_cells_bound: bool = True
    same_capture_streams_bound: bool = True
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    historical_owner_alpha_semantics: str = (
        SAMPLED_V7_D_HISTORICAL_OWNER_ALPHA_SEMANTICS
    )
    authority_scope: str = SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA
    _alpha_family_plan: V1SampledV7DAlphaFamilyPlan = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _alpha_family_registry: V1SampledV7DAlphaFamilyRegistryReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _alpha_head_cas: V1SampledV7DAlphaHeadCASReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _measurement_owner: V1SampledV7DOnlyMeasurementOwner = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _ordinal_physical_axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.INVALID_TYPE,
                "sampled-V7 D alpha-owner bindings are issuer-only",
            )
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_RECEIPT_PREFIX)

    def assert_live(self, expected_hash: str | None = None) -> str:
        observed_owner_hash = _assert_parent(
            self._measurement_owner,
            V1SampledV7DOnlyMeasurementOwner,
            name="live D measurement owner",
        )
        if observed_owner_hash != self.measurement_owner_hash:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.OWNER_SUBSTITUTION,
                "another live measurement owner was substituted after issuance",
            )
        observed_axis_bridge_hash = _assert_parent(
            self._ordinal_physical_axis_bridge,
            V1OrdinalPhysicalAxisBridgeReceipt,
            name="ordinal-to-physical route-axis bridge",
        )
        if observed_axis_bridge_hash != self.ordinal_physical_axis_bridge_receipt_hash:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.AXIS_BRIDGE_SUBSTITUTION,
                "another ordinal-to-physical bridge was substituted after issuance",
            )
        expected = _joined_coordinates(
            alpha_family_plan=self._alpha_family_plan,
            alpha_family_registry=self._alpha_family_registry,
            alpha_head_cas=self._alpha_head_cas,
            measurement_owner=self._measurement_owner,
            ordinal_physical_axis_bridge=self._ordinal_physical_axis_bridge,
            registered_look_index=self.registered_look_index,
            registered_look_stratum_counts=self.registered_look_stratum_counts,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                code = (
                    V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.OWNER_SUBSTITUTION
                    if name == "measurement_owner_hash"
                    else V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.MUTATED_RECEIPT
                )
                _refuse(code, f"bound coordinate {name} changed after issuance")
        positive = (
            self.historical_owner_alpha_coordinates_inert,
            self.plan_is_semantic_witness_only,
            self.durable_controller_head_closed,
            self.downstream_measurement_owner_binding_closed,
            self.registered_a_f_j_cells_bound,
            self.same_capture_streams_bound,
        )
        closed = (
            self.physical_acquisition_authority,
            self.alpha_spend_authority,
            self.d_decision_authority,
            self.c_measurement_authority,
            self.calibration_k_authority,
            self.paired_oracle_o_authority,
            self.action_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.historical_owner_alpha_semantics
            != SAMPLED_V7_D_HISTORICAL_OWNER_ALPHA_SEMANTICS
            or self.authority_scope
            != SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.AUTHORITY_VIOLATION,
                "alpha-owner binding crossed its identity-only authority boundary",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.MUTATED_RECEIPT,
                "alpha-owner binding identity changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode.MUTATED_RECEIPT,
                "alpha-owner binding hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def require_physical_acquisition(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no acquisition authority")

    def require_alpha_spend(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no alpha-spend authority")

    def require_d_decision(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no D-decision authority")

    def require_c_measurement(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no C authority")

    def require_calibration_k(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no K authority")

    def require_paired_oracle_o(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding has no O authority")

    def require_mask_commit(self) -> NoReturn:
        raise PermissionError("alpha-owner identity binding cannot commit masks")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live sampled-V7 D alpha-owner bindings are nonserializable")


def issue_v1_sampled_v7_d_alpha_measurement_owner_binding(
    *,
    alpha_family_plan: V1SampledV7DAlphaFamilyPlan,
    alpha_family_registry: V1SampledV7DAlphaFamilyRegistryReceipt,
    alpha_head_cas: V1SampledV7DAlphaHeadCASReceipt,
    measurement_owner: V1SampledV7DOnlyMeasurementOwner,
    ordinal_physical_axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt,
    registered_look_index: int,
    registered_look_stratum_counts: tuple[int, int],
) -> V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
    """Bind one registered F/J look to an exact live D measurement owner."""

    coordinates = _joined_coordinates(
        alpha_family_plan=alpha_family_plan,
        alpha_family_registry=alpha_family_registry,
        alpha_head_cas=alpha_head_cas,
        measurement_owner=measurement_owner,
        ordinal_physical_axis_bridge=ordinal_physical_axis_bridge,
        registered_look_index=registered_look_index,
        registered_look_stratum_counts=registered_look_stratum_counts,
    )
    return V1SampledV7DAlphaMeasurementOwnerBindingReceipt(
        **coordinates,  # type: ignore[arg-type]
        _alpha_family_plan=alpha_family_plan,
        _alpha_family_registry=alpha_family_registry,
        _alpha_head_cas=alpha_head_cas,
        _measurement_owner=measurement_owner,
        _ordinal_physical_axis_bridge=ordinal_physical_axis_bridge,
        _issuance_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_AUTHORITY",
    "SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA",
    "SAMPLED_V7_D_HISTORICAL_OWNER_ALPHA_SEMANTICS",
    "V1SampledV7DAlphaMeasurementOwnerBindingReceipt",
    "V1SampledV7DAlphaMeasurementOwnerBindingRefusal",
    "V1SampledV7DAlphaMeasurementOwnerBindingRefusalCode",
    "issue_v1_sampled_v7_d_alpha_measurement_owner_binding",
]
