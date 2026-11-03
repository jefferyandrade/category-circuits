"""Pre-outcome binding of sampled-V7 signed-F inference to its alpha cell.

The generic fixed-primal projection registration predates the sampled-V7 A/F/J
family registry.  Its legacy alpha ledger strings cannot be reinterpreted as a
new family cell.  This module wraps that exact registration *before* rows are
observed and proves that its look, population prefix, same-capture streams, and
numeric alpha are the signed-F cell already selected by the live D owner join.

The wrapper is registration evidence only.  It neither spends alpha again nor
proves that a projection outcome consumed the registration.  No D/C/K/O,
action, commit, terminal, performance, or Pareto authority is granted.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_signed_work_fixed_primal_projection_cuda import (
    V1SignedWorkFixedPrimalProjectionRegistration,
)


SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_signed_F_preregistered_alpha_cell_binding_v1"
)
SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_AUTHORITY: Final = (
    "sampled_v7_signed_F_fixed_primal_registration_to_preregistered_alpha_cell_"
    "identity_only_no_observation_second_spend_D_C_K_O_action_commit_terminal_"
    "performance_or_pareto_authority_v1"
)
SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS: Final = (
    "legacy_fixed_primal_alpha_ledger_and_coordinate_strings_remain_inert_"
    "provenance_and_are_not_sampled_v7_alpha_cell_identities_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-signed-F-alpha-registration-"
_OWNER_PREFIX_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-alpha-owner-prefix-axis-full-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledV7SignedFAlphaRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    LOOK_OR_PREFIX_MISMATCH = "look_or_prefix_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    CENSUS_MISMATCH = "census_mismatch"
    LEGACY_ALIAS = "legacy_alpha_coordinate_alias"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7SignedFAlphaRegistrationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7SignedFAlphaRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7SignedFAlphaRegistrationRefusalCode:
            raise TypeError("signed-F alpha registration refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7SignedFAlphaRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7SignedFAlphaRegistrationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _coordinates(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    projection_registration: V1SignedWorkFixedPrimalProjectionRegistration,
) -> dict[str, object]:
    if type(owner_binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.INVALID_TYPE,
            "signed-F alpha binding requires the exact live owner join",
        )
    if type(projection_registration) is not V1SignedWorkFixedPrimalProjectionRegistration:
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.INVALID_TYPE,
            "signed-F alpha binding requires the exact fixed-primal registration",
        )
    try:
        owner_hash = owner_binding.assert_live()
        registration_hash = projection_registration.registration_hash
    except V1SampledV7SignedFAlphaRegistrationRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.PARENT_INTEGRITY,
            f"one signed-F registration parent failed replay: {type(error).__name__}: {error}",
        )

    lineage = owner_binding._measurement_owner.measurement_source_lineage
    registration = projection_registration
    if (
        registration.model_identity_hash != owner_binding.model_identity_hash
        or registration.chart_identity_hash != owner_binding.chart_identity_hash
        or registration.current_mask_hash != owner_binding.current_mask_hash
        or registration.proposal_generation != owner_binding.proposal_generation
        or registration.route_axis_hash != owner_binding.physical_route_axis_hash
        or registration.route_id_axis_hash != lineage.route_id_axis_hash
        or registration.witness_stream_source_hash
        != owner_binding.residual_witness_stream_source_hash
        or registration.same_capture_vjp_source_hash
        != owner_binding.same_capture_vjp_source_hash
        or registration.corrected_evidence_stream_source_hash
        != owner_binding.corrected_evidence_stream_source_hash
        or registration.literal_j_stream_source_hash
        != owner_binding.literal_j_stream_source_hash
        or tuple(item.stratum_id for item in registration.strata)
        != owner_binding.stratum_names
    ):
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "fixed-primal registration left the live signed-F D chart",
        )

    if (
        registration.look_ordinal != owner_binding.registered_look_index
        or registration.population_counts != owner_binding.population_stratum_counts
        or registration.registered_counts
        != owner_binding.registered_look_stratum_counts
    ):
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.LOOK_OR_PREFIX_MISMATCH,
            "fixed-primal registration uses a different signed-F look or count axis",
        )
    contexts = tuple(item.registered_context_ids for item in registration.strata)
    prefix_axis_hash = content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
            "look_index": owner_binding.registered_look_index,
            "registered_look_stratum_counts": (
                owner_binding.registered_look_stratum_counts
            ),
            "population_stratum_counts": owner_binding.population_stratum_counts,
            "population_membership_hashes": owner_binding.population_membership_hashes,
            "sealed_population_order_hashes": (
                owner_binding.sealed_population_order_hashes
            ),
            "target_prefix_context_ids": contexts[0],
            "complement_prefix_context_ids": contexts[1],
        },
        prefix=_OWNER_PREFIX_AXIS_PREFIX,
    )
    if prefix_axis_hash != owner_binding.registered_prefix_axis_hash:
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.LOOK_OR_PREFIX_MISMATCH,
            "fixed-primal context axis is not the registered live D prefix",
        )

    exact_census = (
        owner_binding.registered_look_stratum_counts
        == owner_binding.population_stratum_counts
    )
    stratum_weights = tuple(item.stratum_weight for item in registration.strata)
    if (
        registration.exact_census_mode is not exact_census
        or registration.sampling_alpha_spent is exact_census
    ):
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.CENSUS_MISMATCH,
            "fixed-primal census and sampling-alpha semantics do not replay",
        )
    if exact_census:
        fractions = registration.exact_census_stratum_weight_fractions
        if (
            type(fractions) is not tuple
            or len(fractions) != len(stratum_weights)
            or stratum_weights
            != tuple(numerator / denominator for numerator, denominator in fractions)
        ):
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "exact-census signed F must use finite-population stratum weights",
            )
        if registration.familywise_alpha is not None or owner_binding.signed_f_cell_alpha != 0.0:
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.ALPHA_MISMATCH,
                "exact-census signed F must consume zero sampling alpha",
            )
    else:
        if stratum_weights != (0.5, 0.5):
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "sampled signed F must use the registered balanced stratum weights",
            )
        if registration.familywise_alpha != owner_binding.signed_f_cell_alpha:
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.ALPHA_MISMATCH,
                "fixed-primal familywise alpha differs from the preregistered F cell",
            )

    forbidden_aliases = {
        owner_binding.signed_f_cell_id,
        owner_binding.signed_f_cell_hash,
        owner_binding.signed_f_cell_scope_hash,
        owner_hash,
    }
    if (
        registration.alpha_ledger_id in forbidden_aliases
        or registration.alpha_coordinate_id in forbidden_aliases
    ):
        _refuse(
            V1SampledV7SignedFAlphaRegistrationRefusalCode.LEGACY_ALIAS,
            "legacy fixed-primal alpha strings cannot impersonate the sampled-V7 cell",
        )

    return {
        "owner_binding_receipt_hash": owner_hash,
        "projection_registration_hash": registration_hash,
        "signed_f_cell_id": owner_binding.signed_f_cell_id,
        "signed_f_cell_hash": owner_binding.signed_f_cell_hash,
        "signed_f_cell_scope_hash": owner_binding.signed_f_cell_scope_hash,
        "signed_f_cell_alpha": owner_binding.signed_f_cell_alpha,
        "registered_look_index": owner_binding.registered_look_index,
        "registered_look_stratum_counts": (
            owner_binding.registered_look_stratum_counts
        ),
        "population_stratum_counts": owner_binding.population_stratum_counts,
        "registered_prefix_axis_hash": prefix_axis_hash,
        "registered_d_prefix_hash": owner_binding.registered_prefix_axis_hash,
        "signed_f_stream_binding_hash": owner_binding.signed_f_stream_binding_hash,
        "legacy_alpha_ledger_id": registration.alpha_ledger_id,
        "legacy_alpha_coordinate_id": registration.alpha_coordinate_id,
        "exact_census": exact_census,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7SignedFAlphaRegistrationReceipt:
    owner_binding_receipt_hash: str
    projection_registration_hash: str
    signed_f_cell_id: str
    signed_f_cell_hash: str
    signed_f_cell_scope_hash: str
    signed_f_cell_alpha: float
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    registered_d_prefix_hash: str
    signed_f_stream_binding_hash: str
    legacy_alpha_ledger_id: str
    legacy_alpha_coordinate_id: str
    exact_census: bool
    preoutcome_binding_closed: bool = True
    legacy_alpha_coordinates_inert: bool = True
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
    legacy_alpha_semantics: str = SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS
    authority_scope: str = SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_SCHEMA
    _owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _projection_registration: V1SignedWorkFixedPrimalProjectionRegistration = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 signed-F alpha registrations are issuer-only")
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
            projection_registration=self._projection_registration,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7SignedFAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                    f"signed-F alpha coordinate {name} changed after issuance",
                )
        closed = (
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
            self.preoutcome_binding_closed is not True
            or self.legacy_alpha_coordinates_inert is not True
            or any(type(value) is not bool or value is not False for value in closed)
            or self.legacy_alpha_semantics
            != SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS
            or self.authority_scope != SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "signed-F alpha registration crossed its pre-outcome authority",
            )
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                "signed-F alpha registration changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7SignedFAlphaRegistrationRefusalCode.MUTATED_RECEIPT,
                "signed-F alpha registration hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live signed-F alpha registrations are nonserializable")


def register_v1_sampled_v7_signed_f_alpha_cell(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    projection_registration: V1SignedWorkFixedPrimalProjectionRegistration,
) -> V1SampledV7SignedFAlphaRegistrationReceipt:
    coordinates = _coordinates(
        owner_binding=owner_binding,
        projection_registration=projection_registration,
    )
    return V1SampledV7SignedFAlphaRegistrationReceipt(
        **coordinates,  # type: ignore[arg-type]
        _owner_binding=owner_binding,
        _projection_registration=projection_registration,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_AUTHORITY",
    "SAMPLED_V7_SIGNED_F_ALPHA_REGISTRATION_SCHEMA",
    "SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS",
    "V1SampledV7SignedFAlphaRegistrationReceipt",
    "V1SampledV7SignedFAlphaRegistrationRefusal",
    "V1SampledV7SignedFAlphaRegistrationRefusalCode",
    "register_v1_sampled_v7_signed_f_alpha_cell",
]
