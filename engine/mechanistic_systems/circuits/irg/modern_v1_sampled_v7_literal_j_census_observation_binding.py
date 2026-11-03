"""Bind an alpha-free literal-J exact census to its source projection.

This is the census sibling of
``modern_v1_sampled_v7_literal_j_observation_binding``.  The sampled sibling
consumes a positive-alpha residual prefix and deliberately refuses the
zero-alpha full-population cell.  Here the only accepted transition is the
opposite one: an issuer-bound alpha-free census registration plus a
source-bound resident-D1/residual literal-J projection whose residual component
is an exact census and whose spent alpha is exactly zero.

The receipt is still only an observation binding.  It preserves the sampled-V7
D source graph, ordered/physical route axes, and fixed-direction projection
identity; it does not authorize D/C/K/O decisions, actions, commits,
performance claims, or Pareto claims.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionReceipt,
)
from .modern_v1_sampled_v7_d_alpha_family_registry import (
    V1SampledV7DAlphaFamilyPlan,
    V1SampledV7DAlphaFamilyRegistryReceipt,
)
from .modern_v1_sampled_v7_d_alpha_head_cas import (
    V1SampledV7DAlphaHeadCASReceipt,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_d_only_source_issuance import (
    V1SampledV7DOnlyMeasurementOwner,
)
from .modern_v1_sampled_v7_literal_j_census_registration import (
    V1SampledV7LiteralJCensusRegistrationReceipt,
)
from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
    V1SampledV7ResidentD1LiteralJPopulationReceipt,
)
from .modern_v1_response_exactification import (
    V1OrdinalPhysicalAxisBridgeReceipt,
)


SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_literal_J_alpha_free_census_observation_binding_v1"
)
SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_AUTHORITY: Final = (
    "sampled_v7_literal_J_alpha_free_full_train_census_population_projection_"
    "observation_binding_only_no_alpha_spend_D_C_K_O_action_commit_terminal_"
    "performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-literal-J-census-observation-"
_DW2_PREFIX: Final = "partition-free-v1-sampled-variable-shell-D-W2-suffix-"
_POPULATION_PREFIX: Final = (
    "partition-free-v1-literal-j-D-fixed-residual-population-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledV7LiteralJCensusObservationBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    REGISTRATION_MISMATCH = "registration_mismatch"
    OBSERVATION_IDENTITY_MISMATCH = "observation_identity_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    SOURCE_LINEAGE_MISMATCH = "source_lineage_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    NONCENSUS_PROJECTION = "noncensus_projection"
    SOURCE_SUBSTITUTION = "source_substitution"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7LiteralJCensusObservationBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7LiteralJCensusObservationBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7LiteralJCensusObservationBindingRefusalCode:
            raise TypeError("literal-J census observation refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7LiteralJCensusObservationBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7LiteralJCensusObservationBindingRefusal(code, message)


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


def _static_census_registration_hash(
    census_registration: V1SampledV7LiteralJCensusRegistrationReceipt,
) -> str:
    observed = census_registration._compute_hash()
    if (
        census_registration._receipt_hash is None
        or observed != census_registration._receipt_hash
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.PARENT_INTEGRITY,
            "pre-outcome literal-J census registration changed before observation",
        )
    return observed


def _assert_transition_parents(
    *,
    census_registration: V1SampledV7LiteralJCensusRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> tuple[
    str,
    str,
    str,
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    V1SampledV7DOnlyMeasurementOwner,
]:
    if type(census_registration) is not V1SampledV7LiteralJCensusRegistrationReceipt:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J census observation requires the exact census registration",
        )
    if type(population_projection) is not (
        V1LiteralJDFixedResidualPopulationProjectionReceipt
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J census observation requires the exact population projection",
        )
    if type(resident_d1_receipt) is not (
        V1SampledV7ResidentD1LiteralJPopulationReceipt
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.INVALID_TYPE,
            "literal-J census observation requires the exact resident-D1 receipt",
        )

    binding = census_registration._owner_binding
    if type(binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.INVALID_TYPE,
            "census registration lost its exact alpha-owner binding",
        )
    owner = binding._measurement_owner
    if type(owner) is not V1SampledV7DOnlyMeasurementOwner:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.INVALID_TYPE,
            "alpha-owner binding lost its exact measurement owner",
        )
    authority_parents = (
        (binding._alpha_family_plan, V1SampledV7DAlphaFamilyPlan),
        (binding._alpha_family_registry, V1SampledV7DAlphaFamilyRegistryReceipt),
        (binding._alpha_head_cas, V1SampledV7DAlphaHeadCASReceipt),
        (binding._ordinal_physical_axis_bridge, V1OrdinalPhysicalAxisBridgeReceipt),
    )
    if any(type(value) is not expected for value, expected in authority_parents):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "alpha plan, registry, CAS, or route-axis bridge parent was substituted",
        )

    try:
        census_hash = _static_census_registration_hash(census_registration)
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
        observed_binding_hash = binding._compute_hash()
        observed_owner_hash = owner._compute_hash()
    except V1SampledV7LiteralJCensusObservationBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.PARENT_INTEGRITY,
            f"one literal-J census observation parent failed replay: {type(error).__name__}: {error}",
        )

    if (
        binding._receipt_hash is None
        or observed_binding_hash != binding._receipt_hash
        or observed_binding_hash != census_registration.owner_binding_receipt_hash
        or owner._owner_hash is None
        or observed_owner_hash != owner._owner_hash
        or observed_owner_hash != binding.measurement_owner_hash
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.PARENT_INTEGRITY,
            "pre-outcome alpha-owner or measurement-owner identity changed",
        )
    if (
        plan_hash != binding.alpha_family_plan_hash
        or registry_hash != binding.alpha_family_registry_receipt_hash
        or cas_hash != binding.alpha_head_cas_receipt_hash
        or bridge_hash != binding.ordinal_physical_axis_bridge_receipt_hash
        or frame_hash != binding.generation_frame_receipt_hash
        or discovery_hash != binding.discovery_stream_receipt_hash
        or owner.schedule.schedule_hash != binding.measurement_schedule_hash
        or lineage_hash != binding.measurement_source_lineage_hash
        or suffix_hash != binding.suffix_registration_hash
        or measure_hash != binding.residual_population_measure_design_hash
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "plan, registry, CAS, axis bridge, or retained D source was substituted",
        )
    if resident_d1_receipt is not owner._resident_d1_population.receipt:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
            "literal-J census observation did not use the retained resident-D1 witness",
        )
    if not owner._resident_d1_population.device_handle.is_evicted:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_LINEAGE_MISMATCH,
            "resident-D1 source was not consumed by the census population transition",
        )

    return census_hash, projection_hash, resident_hash, binding, owner


def _observation_coordinates(
    *,
    census_registration: V1SampledV7LiteralJCensusRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> dict[str, object]:
    census_hash, projection_hash, resident_hash, binding, owner = (
        _assert_transition_parents(
            census_registration=census_registration,
            population_projection=population_projection,
            resident_d1_receipt=resident_d1_receipt,
        )
    )
    projection = population_projection
    lineage = owner.measurement_source_lineage
    identity_pairs = (
        (projection.model_identity_hash, binding.model_identity_hash),
        (projection.chart_identity_hash, binding.chart_identity_hash),
        (projection.current_mask_hash, binding.current_mask_hash),
        (projection.route_axis_hash, binding.physical_route_axis_hash),
        (projection.stratum_ids, binding.stratum_names),
        (projection.population_stratum_counts, binding.population_stratum_counts),
    )
    if (
        any(left != right for left, right in identity_pairs)
        or projection.proposal_generation != binding.proposal_generation
        or projection.route_count != binding.route_count
        or projection.population_count != sum(binding.population_stratum_counts)
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.OBSERVATION_IDENTITY_MISMATCH,
            "literal-J census projection and registration name different chart, route, or population identities",
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
        projection.direction_ids == resident_d1_receipt.direction_ids
        and projection.private_primal_direction_axis_hash
        == resident_d1_receipt.private_primal_descriptor_axis_hash
    )
    source_bound_ok = (
        projection.d_fixed_projection_receipt_hash == resident_hash
        and projection.d_fixed_shard_replay_receipt_hash == resident_hash
        and projection.affine_shell_shard_replay_receipt_hash is None
    )
    if not (resident_identity_ok and direction_projection_ok and source_bound_ok):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_LINEAGE_MISMATCH,
            "literal-J census observation did not descend from the registered resident-D1 direction source",
        )

    d1 = census_registration.resident_d1_stratum_counts
    residual_prefix = census_registration.residual_prefix_stratum_counts
    population = census_registration.population_stratum_counts
    registered_look = census_registration.registered_look_stratum_counts
    if (
        projection.exact_full_population_census is not True
        or projection.residual_projection_receipt_hash is None
        or projection.residual_observed_prefix_axis_hashes is None
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.NONCENSUS_PROJECTION,
            "literal-J census observation requires an exact full-population projection",
        )
    if (
        registered_look != population
        or projection.d_fixed_stratum_counts != d1
        or projection.residual_population_counts != residual_prefix
        or projection.residual_observed_counts != residual_prefix
        or tuple(d1[h] + residual_prefix[h] for h in (0, 1)) != population
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.PREFIX_MISMATCH,
            "literal-J census projection is not exactly the registered full population",
        )

    discovery = owner._discovery_stream
    target_order = tuple(discovery.target_assignment_receipt.generated_order)
    complement_order = tuple(
        discovery.complement_assignment_receipt.generated_order
    )
    residual_contexts = (
        target_order[d1[0] : population[0]],
        complement_order[d1[1] : population[1]],
    )
    target_source_axis = _dw2_axis_hash(
        role="D-W2-target-prefix-freeze",
        context_ids=residual_contexts[0],
    )
    complement_source_axis = _dw2_axis_hash(
        role="D-W2-complement-prefix-freeze",
        context_ids=residual_contexts[1],
    )
    expected_prefix_hashes = (
        _residual_observed_prefix_hash(
            registration_hash=projection.registration_hash,
            stratum_id=projection.stratum_ids[0],
            observed_count=residual_prefix[0],
            source_axis_hash=target_source_axis,
        ),
        _residual_observed_prefix_hash(
            registration_hash=projection.registration_hash,
            stratum_id=projection.stratum_ids[1],
            observed_count=residual_prefix[1],
            source_axis_hash=complement_source_axis,
        ),
    )
    if projection.residual_observed_prefix_axis_hashes != expected_prefix_hashes:
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.PREFIX_MISMATCH,
            "literal-J census residual observation is not the exact predictable full suffix",
        )

    if (
        census_registration.literal_j_cell_alpha != 0.0
        or projection.alpha_spent != 0.0
        or projection.d_fixed_numerical_alpha_spent != 0.0
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.ALPHA_MISMATCH,
            "literal-J census observation must spend no sampling alpha",
        )
    if (
        not math.isfinite(projection.registered_residual_familywise_alpha)
        or not 0.0 < projection.registered_residual_familywise_alpha < 1.0
    ):
        _refuse(
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.ALPHA_MISMATCH,
            "legacy residual-registration alpha must remain inert positive provenance",
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
            V1SampledV7LiteralJCensusObservationBindingRefusalCode.AUTHORITY_VIOLATION,
            "source projection is not exact source-bound geometry-only authority",
        )

    return {
        "census_registration_receipt_hash": census_hash,
        "owner_binding_receipt_hash": census_registration.owner_binding_receipt_hash,
        "source_projection_registration_hash": projection.registration_hash,
        "population_projection_receipt_hash": projection_hash,
        "resident_d1_receipt_hash": resident_hash,
        "literal_j_cell_id": census_registration.literal_j_cell_id,
        "literal_j_cell_hash": census_registration.literal_j_cell_hash,
        "literal_j_cell_scope_hash": census_registration.literal_j_cell_scope_hash,
        "literal_j_cell_alpha": census_registration.literal_j_cell_alpha,
        "registered_look_index": census_registration.registered_look_index,
        "registered_look_stratum_counts": registered_look,
        "resident_d1_stratum_counts": d1,
        "residual_prefix_stratum_counts": residual_prefix,
        "population_stratum_counts": population,
        "registered_prefix_axis_hash": census_registration.registered_prefix_axis_hash,
        "residual_prefix_axis_hash": census_registration.residual_prefix_axis_hash,
        "residual_observed_prefix_axis_hashes": expected_prefix_hashes,
        "model_identity_hash": binding.model_identity_hash,
        "chart_identity_hash": binding.chart_identity_hash,
        "current_mask_hash": binding.current_mask_hash,
        "proposal_generation": binding.proposal_generation,
        "ordered_route_axis_hash": binding.ordered_route_axis_hash,
        "physical_route_axis_hash": binding.physical_route_axis_hash,
        "route_id_axis_hash": lineage.route_id_axis_hash,
        "route_count": binding.route_count,
        "direction_ids": projection.direction_ids,
        "d_fixed_projection_direction_axis_hash": (
            projection.d_fixed_projection_direction_axis_hash
        ),
        "private_primal_direction_axis_hash": (
            projection.private_primal_direction_axis_hash
        ),
        "literal_j_stream_source_hash": binding.literal_j_stream_source_hash,
        "literal_j_stream_binding_hash": binding.literal_j_stream_binding_hash,
        "source_projection_alpha_spent": projection.alpha_spent,
        "registered_residual_familywise_alpha_inert": (
            projection.registered_residual_familywise_alpha
        ),
        "second_alpha_spend": 0.0,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7LiteralJCensusObservationBindingReceipt:
    census_registration_receipt_hash: str
    owner_binding_receipt_hash: str
    source_projection_registration_hash: str
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
    ordered_route_axis_hash: str
    physical_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    literal_j_stream_source_hash: str
    literal_j_stream_binding_hash: str
    source_projection_alpha_spent: float
    registered_residual_familywise_alpha_inert: float
    second_alpha_spend: float
    source_observation_bound: bool = True
    literal_j_cell_observation_bound: bool = True
    alpha_free_census_observation: bool = True
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
    authority_scope: str = SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_SCHEMA
    _census_registration: V1SampledV7LiteralJCensusRegistrationReceipt = field(
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
    _parent_identity_axis: tuple[int, int, int] = field(
        init=False, default=(0, 0, 0), repr=False
    )

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 literal-J census observations are issuer-only")
        object.__setattr__(
            self,
            "_parent_identity_axis",
            (
                id(self._census_registration),
                id(self._population_projection),
                id(self._resident_d1_receipt),
            ),
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
        if self._parent_identity_axis != (
            id(self._census_registration),
            id(self._population_projection),
            id(self._resident_d1_receipt),
        ):
            _refuse(
                V1SampledV7LiteralJCensusObservationBindingRefusalCode.SOURCE_SUBSTITUTION,
                "one literal-J census observation parent was substituted after issuance",
            )
        expected = _observation_coordinates(
            census_registration=self._census_registration,
            population_projection=self._population_projection,
            resident_d1_receipt=self._resident_d1_receipt,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7LiteralJCensusObservationBindingRefusalCode.MUTATED_RECEIPT,
                    f"literal-J census observation coordinate {name} changed",
                )
        positive = (
            self.source_observation_bound,
            self.literal_j_cell_observation_bound,
            self.alpha_free_census_observation,
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
            self.literal_j_cell_alpha != 0.0
            or self.source_projection_alpha_spent != 0.0
            or self.second_alpha_spend != 0.0
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7LiteralJCensusObservationBindingRefusalCode.AUTHORITY_VIOLATION,
                "literal-J census observation crossed its identity-only boundary",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7LiteralJCensusObservationBindingRefusalCode.MUTATED_RECEIPT,
                "literal-J census observation changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7LiteralJCensusObservationBindingRefusalCode.MUTATED_RECEIPT,
                "literal-J census observation hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live literal-J census observations are nonserializable")


def issue_v1_sampled_v7_literal_j_alpha_free_census_observation_binding(
    *,
    census_registration: V1SampledV7LiteralJCensusRegistrationReceipt,
    population_projection: V1LiteralJDFixedResidualPopulationProjectionReceipt,
    resident_d1_receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
) -> V1SampledV7LiteralJCensusObservationBindingReceipt:
    coordinates = _observation_coordinates(
        census_registration=census_registration,
        population_projection=population_projection,
        resident_d1_receipt=resident_d1_receipt,
    )
    return V1SampledV7LiteralJCensusObservationBindingReceipt(
        **coordinates,  # type: ignore[arg-type]
        _census_registration=census_registration,
        _population_projection=population_projection,
        _resident_d1_receipt=resident_d1_receipt,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_AUTHORITY",
    "SAMPLED_V7_LITERAL_J_CENSUS_OBSERVATION_BINDING_SCHEMA",
    "V1SampledV7LiteralJCensusObservationBindingReceipt",
    "V1SampledV7LiteralJCensusObservationBindingRefusal",
    "V1SampledV7LiteralJCensusObservationBindingRefusalCode",
    "issue_v1_sampled_v7_literal_j_alpha_free_census_observation_binding",
]
