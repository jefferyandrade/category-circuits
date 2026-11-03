"""Bind one sampled-V7 literal-J observation to its registered alpha cell.

The pre-outcome literal-J alpha receipt names one finite registered D look.
Producing its fixed-D1 plus residual-population projection is a consuming
transition: the resident D1 numeric handle is deliberately burned.  A normal
``assert_live`` on the pre-outcome owner is therefore no longer meaningful
after the observation exists.  This module accepts only that one transition.

It replays the immutable alpha, population-registration, resident-D1, chart,
physical-route, stream, prefix, and scalar-alpha coordinates and requires the
actual source-bound population projection to contain exactly the registered
residual prefix.  The sampled-V7 J cell is marked observation-bound, but alpha
is not spent a second time and no durable cell-consumption or downstream
decision authority is created.  Legacy alpha strings remain inert provenance.

The landed legacy population producer cannot express the zero-alpha exact
census cell.  That case remains an explicit refusal until an alpha-free census
producer lands.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionReceipt,
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_d_alpha_family_registry import (
    V1SampledV7DAlphaFamilyPlan,
    V1SampledV7DAlphaFamilyRegistryReceipt,
)
from .modern_v1_sampled_v7_d_alpha_head_cas import (
    V1SampledV7DAlphaHeadCASReceipt,
)
from .modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyMeasurementOwner,
)
from .modern_v1_sampled_v7_literal_j_alpha_registration import (
    SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS,
    V1SampledV7LiteralJAlphaRegistrationReceipt,
)
from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
    V1SampledV7ResidentD1LiteralJPopulationReceipt,
)
from .modern_v1_response_exactification import (
    V1OrdinalPhysicalAxisBridgeReceipt,
)


SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_literal_J_observation_alpha_cell_binding_v2"
)
SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_AUTHORITY: Final = (
    "sampled_v7_literal_J_population_projection_to_preregistered_alpha_cell_"
    "observation_binding_only_no_second_alpha_spend_durable_ledger_D_C_K_O_"
    "action_commit_terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-literal-J-observation-binding-"
_ALPHA_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-v7-literal-J-alpha-registration-receipt-"
)
_DW2_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_POPULATION_PREFIX: Final = (
    "partition-free-v1-literal-j-D-fixed-residual-population-"
)
_ISSUER_TOKEN: Final = object()
_POPULATION_SOURCE_RESIDENT_D1: Final = "resident_d1"
_POPULATION_SOURCE_AFFINE_SHELL: Final = "affine_shell"


class V1SampledV7LiteralJObservationBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    REGISTRATION_MISMATCH = "registration_mismatch"
    OBSERVATION_IDENTITY_MISMATCH = "observation_identity_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    SOURCE_LINEAGE_MISMATCH = "source_lineage_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    EXACT_CENSUS_PRODUCER_REQUIRED = "exact_census_producer_required"
    POSTSELECTION = "postselection"
    SOURCE_SUBSTITUTION = "source_substitution"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7LiteralJObservationBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7LiteralJObservationBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7LiteralJObservationBindingRefusalCode:
            raise TypeError("literal-J observation refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7LiteralJObservationBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7LiteralJObservationBindingRefusal(code, message)


def _dw2_axis_hash(*, role: str, context_ids: tuple[str, ...]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_variable_shell_D_W2_axis_v1",
            "role": role,
            "ordered_context_ids": context_ids,
        },
        prefix=_DW2_PREFIX + role + "-axis-",
    )


def _residual_observed_prefix_hash(
    *,
    registration_hash: str,
    stratum_id: str,
    observed_count: int,
    source_axis_hash: str,
) -> str:
    return content_hash(
        {
            "schema": "registered_residual_prefix_identity_v1",
            "registration_hash": registration_hash,
            "stratum_id": stratum_id,
            "observed_count": observed_count,
            "source_hash": source_axis_hash,
        },
        prefix=_POPULATION_PREFIX + "residual-prefix-",
    )


def _static_alpha_hash(
    alpha_registration: V1SampledV7LiteralJAlphaRegistrationReceipt,
) -> str:
    observed = content_hash(
        alpha_registration._payload(),
        prefix=_ALPHA_RECEIPT_PREFIX,
    )
    if (
        alpha_registration._receipt_hash is None
        or observed != alpha_registration._receipt_hash
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.PARENT_INTEGRITY,
            "pre-outcome literal-J alpha receipt changed before observation binding",
        )
    return observed


def _assert_transition_parents(
    *,
    alpha_registration: V1SampledV7LiteralJAlphaRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> tuple[
    str,
    str,
    str,
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    V1SampledV7DOnlyMeasurementOwner,
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
]:
    if type(alpha_registration) is not V1SampledV7LiteralJAlphaRegistrationReceipt:
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J observation binding requires the exact alpha registration",
        )
    if type(population_projection) is not (
        V1LiteralJDFixedResidualPopulationProjectionReceipt
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J observation binding requires the exact population projection receipt",
        )
    if type(resident_d1_receipt) is not (
        V1SampledV7ResidentD1LiteralJPopulationReceipt
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J observation binding requires the exact resident-D1 receipt",
        )

    # The zero-alpha census must be refused before generic mutation handling so
    # the absence of an alpha-free producer remains an explicit typed boundary.
    if (
        alpha_registration.literal_j_cell_alpha == 0.0
        or alpha_registration.registered_look_stratum_counts
        == alpha_registration.population_stratum_counts
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.EXACT_CENSUS_PRODUCER_REQUIRED,
            "sampled-V7 zero-alpha literal-J census requires an alpha-free producer",
        )

    binding = alpha_registration._owner_binding
    registration = alpha_registration._population_registration
    if type(binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "alpha registration lost its exact alpha-owner binding",
        )
    if type(registration) is not (
        V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "alpha registration lost its exact population registration",
        )
    owner = binding._measurement_owner
    if type(owner) is not V1SampledV7DOnlyMeasurementOwner:
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.INVALID_TYPE,
            "alpha-owner binding lost its exact measurement owner",
        )
    exact_authority_parents = (
        (binding._alpha_family_plan, V1SampledV7DAlphaFamilyPlan),
        (
            binding._alpha_family_registry,
            V1SampledV7DAlphaFamilyRegistryReceipt,
        ),
        (binding._alpha_head_cas, V1SampledV7DAlphaHeadCASReceipt),
        (
            binding._ordinal_physical_axis_bridge,
            V1OrdinalPhysicalAxisBridgeReceipt,
        ),
    )
    if any(type(value) is not expected for value, expected in exact_authority_parents):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "alpha plan, registry, CAS, or route-axis bridge parent was substituted",
        )

    try:
        alpha_hash = _static_alpha_hash(alpha_registration)
        registration_hash = registration.assert_integrity()
        projection_hash = population_projection.assert_integrity()
        resident_hash = resident_d1_receipt.assert_integrity()
        plan_hash = binding._alpha_family_plan.assert_integrity()
        registry_hash = binding._alpha_family_registry.assert_integrity()
        cas_hash = binding._alpha_head_cas.assert_integrity()
        bridge_hash = binding._ordinal_physical_axis_bridge.assert_integrity()
        frame_hash = owner._frame.assert_integrity()
        discovery_hash = owner._discovery_stream.assert_integrity()
        suffix_hash = owner._suffix_registration.assert_integrity()
        measure_hash = owner._residual_measure_design.assert_integrity()
        owner.schedule.assert_parent_graph(
            frame=owner._frame,
            discovery_stream=owner._discovery_stream,
            suffix_registration=owner._suffix_registration,
        )
        lineage_hash = owner.measurement_source_lineage.assert_integrity()
    except V1SampledV7LiteralJObservationBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.PARENT_INTEGRITY,
            f"one literal-J observation parent failed replay: {type(error).__name__}: {error}",
        )

    # Replaying the pre-outcome binding via ``assert_live`` would incorrectly
    # reject the authorized consuming transition.  Its sealed payload and every
    # still-immutable authority parent are replayed directly instead.
    try:
        observed_binding_hash = binding._compute_hash()
        observed_owner_hash = owner._compute_hash()
    except Exception as error:
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.PARENT_INTEGRITY,
            f"pre-outcome source identity failed static replay: {type(error).__name__}: {error}",
        )
    if (
        binding._receipt_hash is None
        or observed_binding_hash != binding._receipt_hash
        or observed_binding_hash != alpha_registration.owner_binding_receipt_hash
        or owner._owner_hash is None
        or observed_owner_hash != owner._owner_hash
        or observed_owner_hash != binding.measurement_owner_hash
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.PARENT_INTEGRITY,
            "pre-outcome alpha-owner or measurement-owner identity changed",
        )

    plan = binding._alpha_family_plan
    registry = binding._alpha_family_registry
    cas = binding._alpha_head_cas
    if (
        plan_hash != binding.alpha_family_plan_hash
        or registry_hash != binding.alpha_family_registry_receipt_hash
        or cas_hash != binding.alpha_head_cas_receipt_hash
        or bridge_hash != binding.ordinal_physical_axis_bridge_receipt_hash
        or registry.alpha_family_plan_hash != plan_hash
        or cas.alpha_family_plan_hash != plan_hash
        or cas.alpha_family_registry_receipt_hash != registry_hash
        or cas.published_ledger_head_hash != binding.published_ledger_head_hash
        or frame_hash != binding.generation_frame_receipt_hash
        or discovery_hash != binding.discovery_stream_receipt_hash
        or owner.schedule.schedule_hash != binding.measurement_schedule_hash
        or lineage_hash != binding.measurement_source_lineage_hash
        or suffix_hash != binding.suffix_registration_hash
        or measure_hash != binding.residual_population_measure_design_hash
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "plan, registry, CAS, axis bridge, or retained D source was substituted after preregistration",
        )

    source_family = alpha_registration.population_source_family
    source_resident_result = owner._resident_d1_population
    if (
        source_family
        not in {_POPULATION_SOURCE_RESIDENT_D1, _POPULATION_SOURCE_AFFINE_SHELL}
        or resident_d1_receipt is not source_resident_result.receipt
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "literal-J source family or retained resident-D1 witness was substituted",
        )
    if (
        source_family == _POPULATION_SOURCE_RESIDENT_D1
        and not source_resident_result.device_handle.is_evicted
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_LINEAGE_MISMATCH,
            "resident-D1 source was not consumed by the population transition",
        )

    if (
        alpha_registration.owner_binding_receipt_hash != observed_binding_hash
        or alpha_registration.population_registration_hash != registration_hash
        or alpha_registration.population_source_family != source_family
        or alpha_registration.literal_j_cell_id != binding.literal_j_cell_id
        or alpha_registration.literal_j_cell_hash != binding.literal_j_cell_hash
        or alpha_registration.literal_j_cell_scope_hash
        != binding.literal_j_cell_scope_hash
        or alpha_registration.literal_j_cell_alpha != binding.literal_j_cell_alpha
        or alpha_registration.registered_look_index
        != binding.registered_look_index
        or alpha_registration.registered_look_stratum_counts
        != binding.registered_look_stratum_counts
        or alpha_registration.resident_d1_stratum_counts
        != binding.resident_d1_stratum_counts
        or alpha_registration.residual_prefix_stratum_counts
        != binding.residual_prefix_stratum_counts
        or alpha_registration.population_stratum_counts
        != binding.population_stratum_counts
        or alpha_registration.registered_prefix_axis_hash
        != binding.registered_prefix_axis_hash
        or alpha_registration.residual_prefix_axis_hash
        != binding.residual_prefix_axis_hash
        or alpha_registration.literal_j_stream_binding_hash
        != binding.literal_j_stream_binding_hash
        or alpha_registration.legacy_alpha_ledger_hash
        != registration.alpha_ledger_hash
        or alpha_registration.legacy_alpha_coordinate_hash
        != registration.alpha_coordinate_hash
        or alpha_registration.legacy_alpha_lineage_hash
        != registration.alpha_lineage_hash
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.REGISTRATION_MISMATCH,
            "alpha registration no longer replays its exact J cell, look, or population registration",
        )

    return (
        alpha_hash,
        projection_hash,
        resident_hash,
        binding,
        owner,
        registration,
    )


def _observation_coordinates(
    *,
    alpha_registration: V1SampledV7LiteralJAlphaRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> dict[str, object]:
    (
        alpha_hash,
        projection_hash,
        resident_hash,
        binding,
        owner,
        registration,
    ) = _assert_transition_parents(
        alpha_registration=alpha_registration,
        population_projection=population_projection,
        resident_d1_receipt=resident_d1_receipt,
    )

    projection = population_projection
    source_family = alpha_registration.population_source_family
    lineage = owner.measurement_source_lineage
    identity_pairs = (
        (projection.registration_hash, registration.registration_hash),
        (projection.model_identity_hash, binding.model_identity_hash),
        (projection.chart_identity_hash, binding.chart_identity_hash),
        (projection.current_mask_hash, binding.current_mask_hash),
        (projection.route_axis_hash, binding.physical_route_axis_hash),
        (registration.route_id_axis_hash, lineage.route_id_axis_hash),
        (registration.suffix_registration_hash, binding.suffix_registration_hash),
        (
            registration.population_measure_design_hash,
            binding.residual_population_measure_design_hash,
        ),
        (
            registration.residual_literal_j_stream_source_hash,
            binding.literal_j_stream_source_hash,
        ),
        (
            registration.residual_witness_stream_source_hash,
            binding.residual_witness_stream_source_hash,
        ),
    )
    if (
        any(left != right for left, right in identity_pairs)
        or projection.proposal_generation != binding.proposal_generation
        or projection.route_count != binding.route_count
        or registration.route_count != binding.route_count
        or projection.stratum_ids != binding.stratum_names
        or registration.stratum_ids != binding.stratum_names
        or projection.population_stratum_counts
        != binding.population_stratum_counts
        or projection.population_count != sum(binding.population_stratum_counts)
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.OBSERVATION_IDENTITY_MISMATCH,
            "literal-J projection and registered alpha look name different chart, route, or population identities",
        )

    resident_identity_ok = (
        binding.resident_d1_receipt_hash == resident_hash
        and resident_d1_receipt.model_identity_hash == binding.model_identity_hash
        and resident_d1_receipt.chart_identity_hash == binding.chart_identity_hash
        and resident_d1_receipt.current_mask_hash == binding.current_mask_hash
        and resident_d1_receipt.proposal_generation == binding.proposal_generation
        and resident_d1_receipt.route_axis_hash == binding.physical_route_axis_hash
        and resident_d1_receipt.route_id_axis_hash == lineage.route_id_axis_hash
        and resident_d1_receipt.route_count == binding.route_count
        and resident_d1_receipt.stratum_ids == binding.stratum_names
        and resident_d1_receipt.d1_stratum_counts
        == binding.resident_d1_stratum_counts
    )
    direction_projection_ok = (
        projection.direction_ids == registration.direction_ids
        and projection.d_fixed_projection_direction_axis_hash
        == registration.d_fixed_projection_direction_axis_hash
        and projection.private_primal_direction_axis_hash
        == registration.private_primal_direction_axis_hash
    )
    legacy_source_ok = (
        source_family == _POPULATION_SOURCE_RESIDENT_D1
        and registration.affine_shell_shard_replay_receipt_hash is None
        and projection.affine_shell_shard_replay_receipt_hash is None
        and registration.d_fixed_registration_hash == resident_hash
        and registration.d_fixed_shard_replay_receipt_hash == resident_hash
        and projection.d_fixed_projection_receipt_hash == resident_hash
        and projection.d_fixed_shard_replay_receipt_hash == resident_hash
        and resident_d1_receipt.direction_ids == registration.direction_ids
        and resident_d1_receipt.private_primal_source_hash
        == registration.direction_source_hash
        and resident_d1_receipt.private_primal_descriptor_axis_hash
        == registration.private_primal_direction_axis_hash
    )
    affine_source_ok = (
        source_family == _POPULATION_SOURCE_AFFINE_SHELL
        and registration.affine_shell_shard_replay_receipt_hash is not None
        and projection.affine_shell_shard_replay_receipt_hash
        == registration.affine_shell_shard_replay_receipt_hash
        and registration.d_fixed_shard_replay_receipt_hash is not None
        and projection.d_fixed_shard_replay_receipt_hash
        == registration.d_fixed_shard_replay_receipt_hash
        and registration.d_fixed_registration_hash != resident_hash
        and registration.d_fixed_shard_replay_receipt_hash != resident_hash
        and projection.d_fixed_projection_receipt_hash != resident_hash
    )
    if not (resident_identity_ok and direction_projection_ok and (legacy_source_ok or affine_source_ok)):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_LINEAGE_MISMATCH,
            "literal-J observation did not descend from its registered D-fixed source family and direction axis",
        )

    d1 = alpha_registration.resident_d1_stratum_counts
    residual_prefix = alpha_registration.residual_prefix_stratum_counts
    registered_look = alpha_registration.registered_look_stratum_counts
    if (
        projection.d_fixed_stratum_counts != d1
        or projection.residual_population_counts != registration.residual_stratum_counts
        or projection.residual_observed_counts != residual_prefix
        or tuple(d1[h] + residual_prefix[h] for h in (0, 1))
        != registered_look
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.POSTSELECTION,
            "population projection observed a different residual prefix than the preregistered J look",
        )
    if (
        projection.exact_full_population_census
        or projection.residual_projection_receipt_hash is None
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.POSTSELECTION,
            "a noncensus registered J look cannot consume a census or D1-only projection",
        )

    discovery = owner._discovery_stream
    target_order = tuple(discovery.target_assignment_receipt.generated_order)
    complement_order = tuple(
        discovery.complement_assignment_receipt.generated_order
    )
    residual_contexts = (
        target_order[d1[0] : registered_look[0]],
        complement_order[d1[1] : registered_look[1]],
    )
    target_source_axis = _dw2_axis_hash(
        role="D-W2-target-prefix-freeze",
        context_ids=residual_contexts[0],
    )
    complement_source_axis = _dw2_axis_hash(
        role="D-W2-complement-prefix-freeze",
        context_ids=residual_contexts[1],
    )
    expected_observed_prefix_hashes = (
        _residual_observed_prefix_hash(
            registration_hash=registration.registration_hash,
            stratum_id=registration.stratum_ids[0],
            observed_count=residual_prefix[0],
            source_axis_hash=target_source_axis,
        ),
        _residual_observed_prefix_hash(
            registration_hash=registration.registration_hash,
            stratum_id=registration.stratum_ids[1],
            observed_count=residual_prefix[1],
            source_axis_hash=complement_source_axis,
        ),
    )
    if (
        projection.residual_observed_prefix_axis_hashes
        != expected_observed_prefix_hashes
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.PREFIX_MISMATCH,
            "literal-J residual observation is not the exact predictable registered suffix prefix",
        )

    if (
        registration.familywise_alpha
        != alpha_registration.literal_j_cell_alpha
        or projection.registered_residual_familywise_alpha
        != alpha_registration.literal_j_cell_alpha
        or projection.alpha_spent != alpha_registration.literal_j_cell_alpha
        or projection.d_fixed_numerical_alpha_spent != 0.0
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.ALPHA_MISMATCH,
            "source observation alpha does not equal the preregistered J cell",
        )

    legacy = (
        registration.alpha_ledger_hash,
        registration.alpha_coordinate_hash,
        registration.alpha_lineage_hash,
    )
    if (
        legacy
        != (
            projection.alpha_ledger_hash,
            projection.alpha_coordinate_hash,
            projection.alpha_lineage_hash,
        )
        or legacy
        != (
            alpha_registration.legacy_alpha_ledger_hash,
            alpha_registration.legacy_alpha_coordinate_hash,
            alpha_registration.legacy_alpha_lineage_hash,
        )
        or any(
            value
            in {
                alpha_registration.literal_j_cell_id,
                alpha_registration.literal_j_cell_hash,
                alpha_registration.literal_j_cell_scope_hash,
                alpha_hash,
            }
            for value in legacy
        )
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.ALPHA_MISMATCH,
            "legacy alpha coordinates changed or aliased the sampled-V7 J cell",
        )

    if (
        projection.d_fixed_source_bound is not True
        or projection.private_unbound_cpu_reference_only is not False
        or projection.full_population_literal_j_projection_authority is not True
        or projection.kkt_geometry_uncertainty_input_authority is not True
        or projection.fd_rank_authority is not False
        or projection.kkt_face_or_solution_authority is not False
        or projection.c_stream_authority is not False
        or projection.calibration_authority is not False
        or projection.oracle_authority is not False
        or projection.commit_authority is not False
        or projection.terminal_authority is not False
        or projection.pareto_authority is not False
    ):
        _refuse(
            V1SampledV7LiteralJObservationBindingRefusalCode.AUTHORITY_VIOLATION,
            "source projection is not the exact source-bound geometry-only authority",
        )

    return {
        "alpha_registration_receipt_hash": alpha_hash,
        "owner_binding_receipt_hash": alpha_registration.owner_binding_receipt_hash,
        "population_registration_hash": registration.registration_hash,
        "population_source_family": source_family,
        "population_projection_receipt_hash": projection_hash,
        "resident_d1_receipt_hash": resident_hash,
        "literal_j_cell_id": alpha_registration.literal_j_cell_id,
        "literal_j_cell_hash": alpha_registration.literal_j_cell_hash,
        "literal_j_cell_scope_hash": alpha_registration.literal_j_cell_scope_hash,
        "literal_j_cell_alpha": alpha_registration.literal_j_cell_alpha,
        "registered_look_index": alpha_registration.registered_look_index,
        "registered_look_stratum_counts": registered_look,
        "resident_d1_stratum_counts": d1,
        "residual_prefix_stratum_counts": residual_prefix,
        "population_stratum_counts": alpha_registration.population_stratum_counts,
        "registered_prefix_axis_hash": alpha_registration.registered_prefix_axis_hash,
        "residual_prefix_axis_hash": alpha_registration.residual_prefix_axis_hash,
        "residual_observed_prefix_axis_hashes": expected_observed_prefix_hashes,
        "model_identity_hash": binding.model_identity_hash,
        "chart_identity_hash": binding.chart_identity_hash,
        "current_mask_hash": binding.current_mask_hash,
        "proposal_generation": binding.proposal_generation,
        "physical_route_axis_hash": binding.physical_route_axis_hash,
        "route_id_axis_hash": registration.route_id_axis_hash,
        "route_count": binding.route_count,
        "direction_ids": registration.direction_ids,
        "d_fixed_projection_direction_axis_hash": (
            registration.d_fixed_projection_direction_axis_hash
        ),
        "private_primal_direction_axis_hash": (
            registration.private_primal_direction_axis_hash
        ),
        "literal_j_stream_source_hash": binding.literal_j_stream_source_hash,
        "literal_j_stream_binding_hash": binding.literal_j_stream_binding_hash,
        "source_projection_alpha_spent": projection.alpha_spent,
        "second_alpha_spend": 0.0,
        "legacy_alpha_ledger_hash": legacy[0],
        "legacy_alpha_coordinate_hash": legacy[1],
        "legacy_alpha_lineage_hash": legacy[2],
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7LiteralJObservationBindingReceipt:
    alpha_registration_receipt_hash: str
    owner_binding_receipt_hash: str
    population_registration_hash: str
    population_source_family: str
    population_projection_receipt_hash: str
    resident_d1_receipt_hash: str
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_cell_alpha: float
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    resident_d1_stratum_counts: tuple[int, int]
    residual_prefix_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    residual_prefix_axis_hash: str
    residual_observed_prefix_axis_hashes: tuple[str, str]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    physical_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    literal_j_stream_source_hash: str
    literal_j_stream_binding_hash: str
    source_projection_alpha_spent: float
    second_alpha_spend: float
    legacy_alpha_ledger_hash: str
    legacy_alpha_coordinate_hash: str
    legacy_alpha_lineage_hash: str
    source_observation_bound: bool = True
    literal_j_cell_observation_bound: bool = True
    cell_durably_consumed: bool = False
    second_alpha_spend_authority: bool = False
    durable_alpha_ledger_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    legacy_alpha_coordinates_inert: bool = True
    legacy_alpha_semantics: str = SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS
    authority_scope: str = SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_SCHEMA
    _alpha_registration: V1SampledV7LiteralJAlphaRegistrationReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)
    _alpha_parent_identity: int = field(init=False, default=0, repr=False)
    _projection_parent_identity: int = field(init=False, default=0, repr=False)
    _resident_parent_identity: int = field(init=False, default=0, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 literal-J observation bindings are issuer-only")
        object.__setattr__(self, "_alpha_parent_identity", id(self._alpha_registration))
        object.__setattr__(
            self,
            "_projection_parent_identity",
            id(self._population_projection),
        )
        object.__setattr__(
            self,
            "_resident_parent_identity",
            id(self._resident_d1_receipt),
        )
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_live(self, expected_hash: str | None = None) -> str:
        if (
            id(self._alpha_registration) != self._alpha_parent_identity
            or id(self._population_projection) != self._projection_parent_identity
            or id(self._resident_d1_receipt) != self._resident_parent_identity
        ):
            _refuse(
                V1SampledV7LiteralJObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
                "one literal-J observation parent was substituted after issuance",
            )
        expected = _observation_coordinates(
            alpha_registration=self._alpha_registration,
            population_projection=self._population_projection,
            resident_d1_receipt=self._resident_d1_receipt,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7LiteralJObservationBindingRefusalCode.MUTATED_RECEIPT,
                    f"literal-J observation coordinate {name} changed",
                )
        positive = (
            self.source_observation_bound,
            self.literal_j_cell_observation_bound,
            self.legacy_alpha_coordinates_inert,
        )
        closed = (
            self.cell_durably_consumed,
            self.second_alpha_spend_authority,
            self.durable_alpha_ledger_authority,
            self.d_decision_authority,
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
            or self.second_alpha_spend != 0.0
            or self.legacy_alpha_semantics
            != SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS
            or self.authority_scope
            != SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7LiteralJObservationBindingRefusalCode.AUTHORITY_VIOLATION,
                "literal-J observation binding crossed its identity-only boundary",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7LiteralJObservationBindingRefusalCode.MUTATED_RECEIPT,
                "literal-J observation binding changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7LiteralJObservationBindingRefusalCode.MUTATED_RECEIPT,
                "literal-J observation binding hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live literal-J observation bindings are nonserializable")


def issue_v1_sampled_v7_literal_j_observation_binding(
    *,
    alpha_registration: V1SampledV7LiteralJAlphaRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> V1SampledV7LiteralJObservationBindingReceipt:
    coordinates = _observation_coordinates(
        alpha_registration=alpha_registration,
        population_projection=population_projection,
        resident_d1_receipt=resident_d1_receipt,
    )
    return V1SampledV7LiteralJObservationBindingReceipt(
        **coordinates,  # type: ignore[arg-type]
        _alpha_registration=alpha_registration,
        _population_projection=population_projection,
        _resident_d1_receipt=resident_d1_receipt,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_AUTHORITY",
    "SAMPLED_V7_LITERAL_J_OBSERVATION_BINDING_SCHEMA",
    "V1SampledV7LiteralJObservationBindingReceipt",
    "V1SampledV7LiteralJObservationBindingRefusal",
    "V1SampledV7LiteralJObservationBindingRefusalCode",
    "issue_v1_sampled_v7_literal_j_observation_binding",
]
