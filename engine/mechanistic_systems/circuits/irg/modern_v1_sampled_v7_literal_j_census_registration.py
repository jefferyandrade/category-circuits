"""Alpha-free exact-census literal-J registration for sampled-V7 D.

The positive-alpha literal-J population registration intentionally cannot
encode the exact-census cell: its legacy population producer requires a
strictly positive familywise alpha, while sampled-V7 reserves the full-train
literal-J census as an alpha-free registration coordinate.

This module lands that missing seam without retrofitting legacy alpha
coordinates.  It binds the already-issued sampled-V7 D alpha-owner receipt to
the exact full-population literal-J cell, preserves both the ordered ordinal
axis and the physical mask axis, and grants only census literal-J population
registration authority.  It does not authorize observation, D/C/K/O decisions,
mask action, commit, terminal, performance, or Pareto claims.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)


SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_literal_J_alpha_free_exact_census_registration_v1"
)
SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_AUTHORITY: Final = (
    "sampled_v7_literal_J_alpha_free_exact_full_train_census_population_"
    "registration_only_no_observation_D_C_K_O_action_commit_terminal_"
    "performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-literal-J-census-registration-"
_ISSUER_TOKEN: Final = object()


class V1SampledV7LiteralJCensusRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    NONCENSUS_LOOK = "noncensus_look"
    ALPHA_MISMATCH = "alpha_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7LiteralJCensusRegistrationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7LiteralJCensusRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7LiteralJCensusRegistrationRefusalCode:
            raise TypeError("literal-J census registration refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7LiteralJCensusRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7LiteralJCensusRegistrationRefusal(code, message)


def _coordinates(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
) -> dict[str, object]:
    if type(owner_binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.INVALID_TYPE,
            "literal-J census registration requires the exact live owner binding",
        )
    try:
        owner_hash = owner_binding.assert_live()
    except V1SampledV7LiteralJCensusRegistrationRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.PARENT_INTEGRITY,
            f"literal-J census parent failed replay: {type(error).__name__}: {error}",
        )

    if owner_binding.registered_look_stratum_counts != owner_binding.population_stratum_counts:
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.NONCENSUS_LOOK,
            "literal-J census registration requires the registered look to be "
            "the sealed full population, not a sampled prefix",
        )
    if owner_binding.literal_j_cell_alpha != 0.0:
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.ALPHA_MISMATCH,
            "literal-J census registration is alpha-free and requires the "
            "registered J cell alpha to be exactly zero",
        )

    expected_full = tuple(
        fixed + residual
        for fixed, residual in zip(
            owner_binding.resident_d1_stratum_counts,
            owner_binding.residual_prefix_stratum_counts,
            strict=True,
        )
    )
    if expected_full != owner_binding.population_stratum_counts:
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.PREFIX_MISMATCH,
            "resident D1 plus residual prefix does not reconstruct the full "
            "population census",
        )
    if (
        owner_binding.axis_bridge_full_position_bijection_verified is not True
        or not owner_binding.axis_bridge_position_rule
        or not owner_binding.ordered_route_axis_hash
        or not owner_binding.physical_route_axis_hash
    ):
        _refuse(
            V1SampledV7LiteralJCensusRegistrationRefusalCode.AXIS_MISMATCH,
            "literal-J census requires the live registered ordinal/physical "
            "route-axis bridge",
        )

    return {
        "owner_binding_receipt_hash": owner_hash,
        "alpha_family_plan_hash": owner_binding.alpha_family_plan_hash,
        "alpha_family_registry_receipt_hash": (
            owner_binding.alpha_family_registry_receipt_hash
        ),
        "alpha_head_cas_receipt_hash": owner_binding.alpha_head_cas_receipt_hash,
        "published_ledger_head_hash": owner_binding.published_ledger_head_hash,
        "measurement_owner_hash": owner_binding.measurement_owner_hash,
        "ordinal_physical_axis_bridge_receipt_hash": (
            owner_binding.ordinal_physical_axis_bridge_receipt_hash
        ),
        "campaign_contract_hash": owner_binding.campaign_contract_hash,
        "controller_run_hash": owner_binding.controller_run_hash,
        "proposal_generation": owner_binding.proposal_generation,
        "generation_frame_receipt_hash": owner_binding.generation_frame_receipt_hash,
        "discovery_stream_receipt_hash": owner_binding.discovery_stream_receipt_hash,
        "measurement_schedule_hash": owner_binding.measurement_schedule_hash,
        "measurement_source_lineage_hash": owner_binding.measurement_source_lineage_hash,
        "residual_population_measure_design_hash": (
            owner_binding.residual_population_measure_design_hash
        ),
        "resident_d1_receipt_hash": owner_binding.resident_d1_receipt_hash,
        "suffix_registration_hash": owner_binding.suffix_registration_hash,
        "model_identity_hash": owner_binding.model_identity_hash,
        "chart_identity_hash": owner_binding.chart_identity_hash,
        "current_mask_hash": owner_binding.current_mask_hash,
        "query_contract_hash": owner_binding.query_contract_hash,
        "preprocessing_hash": owner_binding.preprocessing_hash,
        "raw_logit_stream_source_hash": owner_binding.raw_logit_stream_source_hash,
        "ordered_route_axis_hash": owner_binding.ordered_route_axis_hash,
        "physical_route_axis_hash": owner_binding.physical_route_axis_hash,
        "route_count": owner_binding.route_count,
        "axis_bridge_already_removed_route_index_hash": (
            owner_binding.axis_bridge_already_removed_route_index_hash
        ),
        "axis_bridge_endpoint_new_removal_counts": (
            owner_binding.axis_bridge_endpoint_new_removal_counts
        ),
        "axis_bridge_ordered_endpoint_index_hashes": (
            owner_binding.axis_bridge_ordered_endpoint_index_hashes
        ),
        "axis_bridge_physical_endpoint_index_hashes": (
            owner_binding.axis_bridge_physical_endpoint_index_hashes
        ),
        "axis_bridge_position_rule": owner_binding.axis_bridge_position_rule,
        "axis_bridge_full_position_bijection_verified": (
            owner_binding.axis_bridge_full_position_bijection_verified
        ),
        "sampling_law_hash": owner_binding.sampling_law_hash,
        "stratum_names": owner_binding.stratum_names,
        "population_stratum_counts": owner_binding.population_stratum_counts,
        "population_membership_hashes": owner_binding.population_membership_hashes,
        "sealed_population_order_hashes": owner_binding.sealed_population_order_hashes,
        "registered_look_index": owner_binding.registered_look_index,
        "registered_look_stratum_counts": owner_binding.registered_look_stratum_counts,
        "resident_d1_stratum_counts": owner_binding.resident_d1_stratum_counts,
        "residual_prefix_stratum_counts": owner_binding.residual_prefix_stratum_counts,
        "registered_prefix_axis_hash": owner_binding.registered_prefix_axis_hash,
        "residual_prefix_axis_hash": owner_binding.residual_prefix_axis_hash,
        "literal_j_cell_id": owner_binding.literal_j_cell_id,
        "literal_j_cell_hash": owner_binding.literal_j_cell_hash,
        "literal_j_cell_scope_hash": owner_binding.literal_j_cell_scope_hash,
        "literal_j_cell_alpha": owner_binding.literal_j_cell_alpha,
        "capture_policy_identity_hash": owner_binding.capture_policy_identity_hash,
        "residual_witness_stream_source_hash": (
            owner_binding.residual_witness_stream_source_hash
        ),
        "same_capture_vjp_source_hash": owner_binding.same_capture_vjp_source_hash,
        "literal_j_stream_source_hash": owner_binding.literal_j_stream_source_hash,
        "literal_j_stream_binding_hash": owner_binding.literal_j_stream_binding_hash,
        "dual_stream_binding_hash": owner_binding.dual_stream_binding_hash,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7LiteralJCensusRegistrationReceipt:
    owner_binding_receipt_hash: str
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
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_cell_alpha: float
    capture_policy_identity_hash: str
    residual_witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    literal_j_stream_source_hash: str
    literal_j_stream_binding_hash: str
    dual_stream_binding_hash: str
    alpha_spent: float = 0.0
    alpha_free: bool = True
    exact_full_train_census: bool = True
    preoutcome_census_registration_bound: bool = True
    census_literal_j_population_registration_authority: bool = True
    observation_seen: bool = False
    physical_acquisition_authority: bool = False
    alpha_spend_authority: bool = False
    second_alpha_spend_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_SCHEMA
    _owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 literal-J census registrations are issuer-only")
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
        expected = _coordinates(owner_binding=self._owner_binding)
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7LiteralJCensusRegistrationRefusalCode.MUTATED_RECEIPT,
                    f"literal-J census coordinate {name} changed after issuance",
                )
        positive = (
            self.alpha_free,
            self.exact_full_train_census,
            self.preoutcome_census_registration_bound,
            self.census_literal_j_population_registration_authority,
        )
        closed = (
            self.observation_seen,
            self.physical_acquisition_authority,
            self.alpha_spend_authority,
            self.second_alpha_spend_authority,
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
            self.alpha_spent != 0.0
            or self.literal_j_cell_alpha != 0.0
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7LiteralJCensusRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "literal-J census registration crossed its alpha-free authority",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7LiteralJCensusRegistrationRefusalCode.MUTATED_RECEIPT,
                "literal-J census registration changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7LiteralJCensusRegistrationRefusalCode.MUTATED_RECEIPT,
                "literal-J census registration hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def require_census_literal_j_population_registration(self) -> str:
        """Return the live receipt hash for the one authority this receipt grants."""

        return self.assert_live()

    def require_physical_acquisition(self) -> NoReturn:
        raise PermissionError("literal-J census registration has no acquisition authority")

    def require_alpha_spend(self) -> NoReturn:
        raise PermissionError("literal-J census registration spends no alpha")

    def require_d_decision(self) -> NoReturn:
        raise PermissionError("literal-J census registration has no D-decision authority")

    def require_c_measurement(self) -> NoReturn:
        raise PermissionError("literal-J census registration has no C authority")

    def require_calibration_k(self) -> NoReturn:
        raise PermissionError("literal-J census registration has no K authority")

    def require_paired_oracle_o(self) -> NoReturn:
        raise PermissionError("literal-J census registration has no O authority")

    def require_mask_commit(self) -> NoReturn:
        raise PermissionError("literal-J census registration cannot commit masks")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live literal-J census registrations are nonserializable")


def register_v1_sampled_v7_literal_j_alpha_free_census_cell(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
) -> V1SampledV7LiteralJCensusRegistrationReceipt:
    coordinates = _coordinates(owner_binding=owner_binding)
    return V1SampledV7LiteralJCensusRegistrationReceipt(
        **coordinates,  # type: ignore[arg-type]
        _owner_binding=owner_binding,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_AUTHORITY",
    "SAMPLED_V7_LITERAL_J_CENSUS_REGISTRATION_SCHEMA",
    "V1SampledV7LiteralJCensusRegistrationReceipt",
    "V1SampledV7LiteralJCensusRegistrationRefusal",
    "V1SampledV7LiteralJCensusRegistrationRefusalCode",
    "register_v1_sampled_v7_literal_j_alpha_free_census_cell",
]
