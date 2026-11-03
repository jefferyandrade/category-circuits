"""Pre-outcome sampled-V7 literal-J attempt binding to its alpha cell.

The resident-D1 plus residual-population projection registers the entire
literal-J population design, but its legacy alpha strings predate the sampled-
V7 A/F/J family.  A registered family look additionally fixes how much of the
residual suffix may be observed before the decision is revisited.  This module
binds that attempt coordinate before residual outcomes exist.

The landed legacy projection registration requires a strictly positive alpha,
so it cannot represent the alpha-free exact-census cell.  This wrapper refuses
that case explicitly; a versioned alpha-free census producer remains required.
No observation, second spend, D/C/K/O/action/commit/terminal/performance or
Pareto authority is granted.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)


SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_literal_J_preregistered_alpha_attempt_v2"
)
SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_AUTHORITY: Final = (
    "sampled_v7_literal_J_residual_population_attempt_to_preregistered_alpha_"
    "cell_identity_only_no_observation_second_spend_D_C_K_O_action_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS: Final = (
    "legacy_literal_J_alpha_ledger_coordinate_and_lineage_hashes_remain_inert_"
    "provenance_and_are_not_sampled_v7_alpha_cell_identities_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-literal-J-alpha-registration-"
_ISSUER_TOKEN: Final = object()
_POPULATION_SOURCE_RESIDENT_D1: Final = "resident_d1"
_POPULATION_SOURCE_AFFINE_SHELL: Final = "affine_shell"


class V1SampledV7LiteralJAlphaRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    LOOK_OR_PREFIX_MISMATCH = "look_or_prefix_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    EXACT_CENSUS_PRODUCER_REQUIRED = "exact_census_producer_required"
    LEGACY_ALIAS = "legacy_alpha_coordinate_alias"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7LiteralJAlphaRegistrationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7LiteralJAlphaRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7LiteralJAlphaRegistrationRefusalCode:
            raise TypeError("literal-J alpha registration refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7LiteralJAlphaRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7LiteralJAlphaRegistrationRefusal(code, message)


def _coordinates(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
) -> dict[str, object]:
    if type(owner_binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.INVALID_TYPE,
            "literal-J alpha binding requires the exact live owner join",
        )
    if type(population_registration) is not (
        V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.INVALID_TYPE,
            "literal-J alpha binding requires the exact population registration",
        )
    try:
        owner_hash = owner_binding.assert_live()
        registration_hash = population_registration.assert_integrity()
    except V1SampledV7LiteralJAlphaRegistrationRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.PARENT_INTEGRITY,
            f"one literal-J registration parent failed replay: {type(error).__name__}: {error}",
        )

    registration = population_registration
    population_source_family = (
        _POPULATION_SOURCE_AFFINE_SHELL
        if registration.affine_shell_shard_replay_receipt_hash is not None
        else _POPULATION_SOURCE_RESIDENT_D1
    )
    lineage = owner_binding._measurement_owner.measurement_source_lineage
    if (
        registration.model_identity_hash != owner_binding.model_identity_hash
        or registration.chart_identity_hash != owner_binding.chart_identity_hash
        or registration.current_mask_hash != owner_binding.current_mask_hash
        or registration.proposal_generation != owner_binding.proposal_generation
        or registration.route_axis_hash != owner_binding.physical_route_axis_hash
        or registration.route_id_axis_hash != lineage.route_id_axis_hash
        or registration.route_count != owner_binding.route_count
        or registration.stratum_ids != owner_binding.stratum_names
        or registration.population_stratum_counts
        != owner_binding.population_stratum_counts
        or registration.d_fixed_stratum_counts
        != owner_binding.resident_d1_stratum_counts
        or registration.residual_witness_stream_source_hash
        != owner_binding.residual_witness_stream_source_hash
        or registration.residual_literal_j_stream_source_hash
        != owner_binding.literal_j_stream_source_hash
        or registration.suffix_registration_hash
        != owner_binding.suffix_registration_hash
        or registration.population_measure_design_hash
        != owner_binding.residual_population_measure_design_hash
    ):
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "literal-J population registration left the live sampled-V7 D chart",
        )

    residual_prefix = owner_binding.residual_prefix_stratum_counts
    if any(
        observed < 0 or observed > total
        for observed, total in zip(
            residual_prefix, registration.residual_stratum_counts, strict=True
        )
    ):
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.LOOK_OR_PREFIX_MISMATCH,
            "registered literal-J look lies outside the sealed residual suffix",
        )
    if tuple(
        d1 + residual
        for d1, residual in zip(
            registration.d_fixed_stratum_counts, residual_prefix, strict=True
        )
    ) != owner_binding.registered_look_stratum_counts:
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.LOOK_OR_PREFIX_MISMATCH,
            "literal-J D1 plus residual attempt does not reconstruct the family look",
        )

    exact_census = (
        owner_binding.registered_look_stratum_counts
        == owner_binding.population_stratum_counts
    )
    if exact_census:
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.EXACT_CENSUS_PRODUCER_REQUIRED,
            "the legacy literal-J registration requires positive alpha and cannot "
            "represent the sampled-V7 zero-alpha census cell",
        )
    if registration.familywise_alpha != owner_binding.literal_j_cell_alpha:
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.ALPHA_MISMATCH,
            "literal-J familywise alpha differs from the preregistered J cell",
        )

    forbidden_aliases = {
        owner_binding.literal_j_cell_id,
        owner_binding.literal_j_cell_hash,
        owner_binding.literal_j_cell_scope_hash,
        owner_hash,
    }
    if any(
        value in forbidden_aliases
        for value in (
            registration.alpha_ledger_hash,
            registration.alpha_coordinate_hash,
            registration.alpha_lineage_hash,
        )
    ):
        _refuse(
            V1SampledV7LiteralJAlphaRegistrationRefusalCode.LEGACY_ALIAS,
            "legacy literal-J alpha hashes cannot impersonate the sampled-V7 cell",
        )

    return {
        "owner_binding_receipt_hash": owner_hash,
        "population_registration_hash": registration_hash,
        "population_source_family": population_source_family,
        "literal_j_cell_id": owner_binding.literal_j_cell_id,
        "literal_j_cell_hash": owner_binding.literal_j_cell_hash,
        "literal_j_cell_scope_hash": owner_binding.literal_j_cell_scope_hash,
        "literal_j_cell_alpha": owner_binding.literal_j_cell_alpha,
        "registered_look_index": owner_binding.registered_look_index,
        "registered_look_stratum_counts": (
            owner_binding.registered_look_stratum_counts
        ),
        "resident_d1_stratum_counts": owner_binding.resident_d1_stratum_counts,
        "residual_prefix_stratum_counts": residual_prefix,
        "population_stratum_counts": owner_binding.population_stratum_counts,
        "registered_prefix_axis_hash": owner_binding.registered_prefix_axis_hash,
        "residual_prefix_axis_hash": owner_binding.residual_prefix_axis_hash,
        "literal_j_stream_binding_hash": owner_binding.literal_j_stream_binding_hash,
        "legacy_alpha_ledger_hash": registration.alpha_ledger_hash,
        "legacy_alpha_coordinate_hash": registration.alpha_coordinate_hash,
        "legacy_alpha_lineage_hash": registration.alpha_lineage_hash,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7LiteralJAlphaRegistrationReceipt:
    owner_binding_receipt_hash: str
    population_registration_hash: str
    population_source_family: str
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
    literal_j_stream_binding_hash: str
    legacy_alpha_ledger_hash: str
    legacy_alpha_coordinate_hash: str
    legacy_alpha_lineage_hash: str
    preoutcome_attempt_bound: bool = True
    legacy_alpha_coordinates_inert: bool = True
    exact_census: bool = False
    observation_seen: bool = False
    second_alpha_spend_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    legacy_alpha_semantics: str = SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS
    authority_scope: str = SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_SCHEMA
    _owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 literal-J alpha registrations are issuer-only")
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def assert_live(self, expected_hash: str | None = None) -> str:
        expected = _coordinates(
            owner_binding=self._owner_binding,
            population_registration=self._population_registration,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7LiteralJAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                    f"literal-J alpha coordinate {name} changed after issuance",
                )
        closed = (
            self.exact_census,
            self.observation_seen,
            self.second_alpha_spend_authority,
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
            self.preoutcome_attempt_bound is not True
            or self.legacy_alpha_coordinates_inert is not True
            or any(type(value) is not bool or value is not False for value in closed)
            or self.legacy_alpha_semantics
            != SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS
            or self.authority_scope != SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7LiteralJAlphaRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "literal-J alpha registration crossed its pre-outcome authority",
            )
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7LiteralJAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                "literal-J alpha registration changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7LiteralJAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                "literal-J alpha registration hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live literal-J alpha registrations are nonserializable")


def register_v1_sampled_v7_literal_j_alpha_cell(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
) -> V1SampledV7LiteralJAlphaRegistrationReceipt:
    coordinates = _coordinates(
        owner_binding=owner_binding,
        population_registration=population_registration,
    )
    return V1SampledV7LiteralJAlphaRegistrationReceipt(
        **coordinates,  # type: ignore[arg-type]
        _owner_binding=owner_binding,
        _population_registration=population_registration,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_AUTHORITY",
    "SAMPLED_V7_LITERAL_J_ALPHA_REGISTRATION_SCHEMA",
    "SAMPLED_V7_LITERAL_J_LEGACY_ALPHA_SEMANTICS",
    "V1SampledV7LiteralJAlphaRegistrationReceipt",
    "V1SampledV7LiteralJAlphaRegistrationRefusal",
    "V1SampledV7LiteralJAlphaRegistrationRefusalCode",
    "register_v1_sampled_v7_literal_j_alpha_cell",
]
