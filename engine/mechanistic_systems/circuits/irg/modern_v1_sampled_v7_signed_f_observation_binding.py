"""Bind one signed-F projection observation to its sampled-V7 alpha cell.

The generic fixed-primal projection runtime predates the sampled-V7 A/F/J
alpha-family registry.  ``V1SampledV7SignedFAlphaRegistrationReceipt`` closes
the pre-outcome half of that seam.  This module closes only the observation
half: an actual fixed-primal projection receipt must have been issued from the
*same* retained registration, physical route axis, predictable prefix, and
same-capture measurement streams that were bound to the signed-F cell.

The binding does not spend alpha again and does not durably consume the cell.
Legacy alpha ledger strings remain inert provenance.  A later atomic A/F/J
observation-family transaction must provide nonrecycling consumption and any
D-decision authority.  Nothing here authorizes C/K/O, an action, commit,
terminal, performance, or Pareto claims.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_sampled_v7_signed_f_alpha_registration import (
    SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS,
    V1SampledV7SignedFAlphaRegistrationReceipt,
)
from .modern_v1_signed_work_fixed_primal_projection_cuda import (
    V1SignedWorkFixedPrimalProjectionReceipt,
)


SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_signed_F_observation_alpha_cell_binding_v1"
)
SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_AUTHORITY: Final = (
    "sampled_v7_signed_F_fixed_primal_projection_observation_to_preregistered_"
    "alpha_cell_identity_only_no_second_alpha_spend_durable_cell_consumption_"
    "D_C_K_O_action_commit_terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-signed-F-observation-binding-"
_PROJECTION_AXIS_PREFIX: Final = (
    "partition-free-v1-signed-work-fixed-primal-projection-axis-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledV7SignedFObservationBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    PREREGISTRATION_MISMATCH = "preregistration_or_postselection_mismatch"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    CENSUS_MISMATCH = "census_mismatch"
    LEGACY_ALIAS = "legacy_alpha_coordinate_alias"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7SignedFObservationBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7SignedFObservationBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7SignedFObservationBindingRefusalCode:
            raise TypeError("signed-F observation refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7SignedFObservationBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7SignedFObservationBindingRefusal(code, message)


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "signed_work_fixed_primal_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PROJECTION_AXIS_PREFIX,
    )


def _assert_parents(
    *,
    alpha_registration: V1SampledV7SignedFAlphaRegistrationReceipt,
    projection_receipt: V1SignedWorkFixedPrimalProjectionReceipt,
) -> tuple[str, str]:
    if type(alpha_registration) is not V1SampledV7SignedFAlphaRegistrationReceipt:
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.INVALID_TYPE,
            "signed-F observation binding requires the exact alpha registration",
        )
    if type(projection_receipt) is not V1SignedWorkFixedPrimalProjectionReceipt:
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.INVALID_TYPE,
            "signed-F observation binding requires the exact projection receipt",
        )
    try:
        alpha_hash = alpha_registration.assert_live()
        projection_hash = projection_receipt.assert_integrity()
    except V1SampledV7SignedFObservationBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.PARENT_INTEGRITY,
            "one signed-F observation parent failed replay: "
            f"{type(error).__name__}: {error}",
        )
    return alpha_hash, projection_hash


def _observation_coordinates(
    *,
    alpha_registration: V1SampledV7SignedFAlphaRegistrationReceipt,
    projection_receipt: V1SignedWorkFixedPrimalProjectionReceipt,
) -> dict[str, object]:
    alpha_hash, projection_hash = _assert_parents(
        alpha_registration=alpha_registration,
        projection_receipt=projection_receipt,
    )
    registration = alpha_registration._projection_registration
    owner = alpha_registration._owner_binding
    try:
        owner_hash = owner.assert_live()
        registration_hash = registration.registration_hash
    except Exception as error:
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.PARENT_INTEGRITY,
            "retained signed-F registration lineage failed replay: "
            f"{type(error).__name__}: {error}",
        )

    if (
        alpha_registration.projection_registration_hash != registration_hash
        or projection_receipt.registration_hash != registration_hash
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.PREREGISTRATION_MISMATCH,
            "projection outcome was not issued from the exact pre-outcome registration",
        )

    # The corrected-A order lives on the registered ordinal axis; signed-F is
    # deliberately measured on the distinct physical mask axis.  Conflating
    # the two is not a harmless shortcut even when their route counts agree.
    if (
        owner.ordered_route_axis_hash == owner.physical_route_axis_hash
        or projection_receipt.route_axis_hash != owner.physical_route_axis_hash
        or projection_receipt.route_axis_hash == owner.ordered_route_axis_hash
        or projection_receipt.route_count != owner.route_count
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.AXIS_MISMATCH,
            "signed-F observation did not remain on the bridged physical route axis",
        )

    source_pairs = (
        (projection_receipt.model_identity_hash, registration.model_identity_hash),
        (projection_receipt.chart_identity_hash, registration.chart_identity_hash),
        (projection_receipt.current_mask_hash, registration.current_mask_hash),
        (projection_receipt.route_id_axis_hash, registration.route_id_axis_hash),
        (
            projection_receipt.witness_stream_source_hash,
            registration.witness_stream_source_hash,
        ),
        (
            projection_receipt.same_capture_vjp_source_hash,
            registration.same_capture_vjp_source_hash,
        ),
        (
            projection_receipt.corrected_evidence_stream_source_hash,
            registration.corrected_evidence_stream_source_hash,
        ),
        (
            projection_receipt.literal_j_stream_source_hash,
            registration.literal_j_stream_source_hash,
        ),
        (
            projection_receipt.causal_registration_receipt_hash,
            registration.causal_registration_receipt_hash,
        ),
        (
            projection_receipt.corrected_support_source_receipt_hash,
            registration.corrected_support_source_receipt_hash,
        ),
        (
            projection_receipt.literal_j_support_source_receipt_hash,
            registration.literal_j_support_source_receipt_hash,
        ),
        (projection_receipt.fixed_primal_id, registration.fixed_primal_id),
        (
            projection_receipt.fixed_primal_source_receipt_hash,
            registration.fixed_primal_source_receipt_hash,
        ),
    )
    if (
        any(left != right for left, right in source_pairs)
        or projection_receipt.proposal_generation
        != registration.proposal_generation
        or projection_receipt.stratum_ids
        != tuple(item.stratum_id for item in registration.strata)
        or projection_receipt.stratum_weights
        != tuple(item.stratum_weight for item in registration.strata)
        or projection_receipt.population_counts != registration.population_counts
        or projection_receipt.observed_counts != registration.registered_counts
        or projection_receipt.look_ordinal != registration.look_ordinal
        or projection_receipt.predictable_before_witness_outcomes is not True
        or projection_receipt.sealed_without_replacement_permutations is not True
        or projection_receipt.source_projection_authority is not True
        or projection_receipt.fixed_z_only_no_solution_sensitivity is not True
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "projection outcome left its fixed-primal same-capture measurement source",
        )

    prefix_context = tuple(
        _axis_hash("registered_context_ids", item.registered_context_ids)
        for item in registration.strata
    )
    prefix_draw = tuple(
        _axis_hash("registered_draw_ids", item.registered_draw_ids)
        for item in registration.strata
    )
    prefix_measurement = tuple(
        _axis_hash("registered_measurement_ids", item.registered_measurement_ids)
        for item in registration.strata
    )
    prefix_capture = tuple(
        _axis_hash("registered_capture_ids", item.registered_capture_ids)
        for item in registration.strata
    )
    if (
        projection_receipt.registered_prefix_context_axis_hashes != prefix_context
        or projection_receipt.registered_prefix_draw_axis_hashes != prefix_draw
        or projection_receipt.registered_prefix_measurement_axis_hashes
        != prefix_measurement
        or projection_receipt.registered_prefix_capture_axis_hashes != prefix_capture
        or projection_receipt.look_ordinal
        != alpha_registration.registered_look_index
        or projection_receipt.observed_counts
        != alpha_registration.registered_look_stratum_counts
        or projection_receipt.population_counts
        != alpha_registration.population_stratum_counts
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.PREFIX_MISMATCH,
            "projection outcome is not the exact registered predictable D prefix",
        )

    exact_census = alpha_registration.exact_census
    if (
        projection_receipt.exact_census_mode is not exact_census
        or projection_receipt.exact_finite_population_census is not exact_census
        or projection_receipt.sampling_alpha_spent is exact_census
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.CENSUS_MISMATCH,
            "projection census and sampling-alpha semantics do not replay",
        )
    if exact_census:
        if (
            projection_receipt.familywise_alpha is not None
            or alpha_registration.signed_f_cell_alpha != 0.0
            or projection_receipt.signed_f_sampling_radius_identically_zero
            is not True
        ):
            _refuse(
                V1SampledV7SignedFObservationBindingRefusalCode.ALPHA_MISMATCH,
                "exact-census signed F must use zero sampling alpha and radius",
            )
    elif (
        projection_receipt.familywise_alpha
        != alpha_registration.signed_f_cell_alpha
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.ALPHA_MISMATCH,
            "observed signed-F familywise alpha differs from its registered cell",
        )

    if (
        projection_receipt.alpha_ledger_id
        != alpha_registration.legacy_alpha_ledger_id
        or projection_receipt.alpha_coordinate_id
        != alpha_registration.legacy_alpha_coordinate_id
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "projection outcome changed its inert legacy alpha coordinates",
        )
    forbidden_aliases = {
        alpha_registration.signed_f_cell_id,
        alpha_registration.signed_f_cell_hash,
        alpha_registration.signed_f_cell_scope_hash,
        alpha_hash,
    }
    if (
        projection_receipt.alpha_ledger_id in forbidden_aliases
        or projection_receipt.alpha_coordinate_id in forbidden_aliases
    ):
        _refuse(
            V1SampledV7SignedFObservationBindingRefusalCode.LEGACY_ALIAS,
            "legacy projection alpha strings cannot impersonate the sampled-V7 cell",
        )

    measurement_source_binding_hash = content_hash(
        {
            "projection_registration_hash": registration_hash,
            "projection_receipt_hash": projection_hash,
            "model_identity_hash": projection_receipt.model_identity_hash,
            "chart_identity_hash": projection_receipt.chart_identity_hash,
            "current_mask_hash": projection_receipt.current_mask_hash,
            "proposal_generation": projection_receipt.proposal_generation,
            "physical_route_axis_hash": projection_receipt.route_axis_hash,
            "route_id_axis_hash": projection_receipt.route_id_axis_hash,
            "witness_stream_source_hash": (
                projection_receipt.witness_stream_source_hash
            ),
            "same_capture_vjp_source_hash": (
                projection_receipt.same_capture_vjp_source_hash
            ),
            "corrected_evidence_stream_source_hash": (
                projection_receipt.corrected_evidence_stream_source_hash
            ),
            "literal_j_stream_source_hash": (
                projection_receipt.literal_j_stream_source_hash
            ),
            "causal_registration_receipt_hash": (
                projection_receipt.causal_registration_receipt_hash
            ),
            "registered_prefix_context_axis_hashes": prefix_context,
            "registered_prefix_draw_axis_hashes": prefix_draw,
            "registered_prefix_measurement_axis_hashes": prefix_measurement,
            "registered_prefix_capture_axis_hashes": prefix_capture,
        },
        prefix=_PREFIX + "measurement-source-",
    )
    return {
        "signed_f_alpha_registration_receipt_hash": alpha_hash,
        "owner_binding_receipt_hash": owner_hash,
        "projection_registration_hash": registration_hash,
        "projection_receipt_hash": projection_hash,
        "signed_f_cell_id": alpha_registration.signed_f_cell_id,
        "signed_f_cell_hash": alpha_registration.signed_f_cell_hash,
        "signed_f_cell_scope_hash": alpha_registration.signed_f_cell_scope_hash,
        "signed_f_cell_alpha": alpha_registration.signed_f_cell_alpha,
        "registered_look_index": alpha_registration.registered_look_index,
        "registered_look_stratum_counts": (
            alpha_registration.registered_look_stratum_counts
        ),
        "population_stratum_counts": alpha_registration.population_stratum_counts,
        "registered_prefix_axis_hash": (
            alpha_registration.registered_prefix_axis_hash
        ),
        "ordered_route_axis_hash": owner.ordered_route_axis_hash,
        "physical_route_axis_hash": owner.physical_route_axis_hash,
        "route_count": owner.route_count,
        "signed_f_stream_binding_hash": (
            alpha_registration.signed_f_stream_binding_hash
        ),
        "measurement_source_binding_hash": measurement_source_binding_hash,
        "observed_prefix_context_axis_hashes": prefix_context,
        "observed_prefix_draw_axis_hashes": prefix_draw,
        "observed_prefix_measurement_axis_hashes": prefix_measurement,
        "observed_prefix_capture_axis_hashes": prefix_capture,
        "legacy_alpha_ledger_id": projection_receipt.alpha_ledger_id,
        "legacy_alpha_coordinate_id": projection_receipt.alpha_coordinate_id,
        "familywise_alpha_observed": projection_receipt.familywise_alpha,
        "exact_census": exact_census,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7SignedFObservationBindingReceipt:
    signed_f_alpha_registration_receipt_hash: str
    owner_binding_receipt_hash: str
    projection_registration_hash: str
    projection_receipt_hash: str
    signed_f_cell_id: str
    signed_f_cell_hash: str
    signed_f_cell_scope_hash: str
    signed_f_cell_alpha: float
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    ordered_route_axis_hash: str
    physical_route_axis_hash: str
    route_count: int
    signed_f_stream_binding_hash: str
    measurement_source_binding_hash: str
    observed_prefix_context_axis_hashes: tuple[str, str]
    observed_prefix_draw_axis_hashes: tuple[str, str]
    observed_prefix_measurement_axis_hashes: tuple[str, str]
    observed_prefix_capture_axis_hashes: tuple[str, str]
    legacy_alpha_ledger_id: str
    legacy_alpha_coordinate_id: str
    familywise_alpha_observed: float | None
    exact_census: bool
    source_observation_bound: bool = True
    legacy_alpha_coordinates_inert: bool = True
    cell_durably_consumed: bool = False
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
    authority_scope: str = SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_SCHEMA
    _alpha_registration: V1SampledV7SignedFAlphaRegistrationReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _projection_receipt: V1SignedWorkFixedPrimalProjectionReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("sampled-V7 signed-F observation bindings are issuer-only")
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def assert_live(self, expected_hash: str | None = None) -> str:
        expected = _observation_coordinates(
            alpha_registration=self._alpha_registration,
            projection_receipt=self._projection_receipt,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7SignedFObservationBindingRefusalCode.MUTATED_RECEIPT,
                    f"signed-F observation coordinate {name} changed after issuance",
                )
        closed = (
            self.cell_durably_consumed,
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
            self.source_observation_bound is not True
            or self.legacy_alpha_coordinates_inert is not True
            or any(type(value) is not bool or value is not False for value in closed)
            or self.legacy_alpha_semantics
            != SAMPLED_V7_SIGNED_F_LEGACY_ALPHA_SEMANTICS
            or self.authority_scope
            != SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7SignedFObservationBindingRefusalCode.AUTHORITY_VIOLATION,
                "signed-F observation binding crossed its identity-only boundary",
            )
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7SignedFObservationBindingRefusalCode.MUTATED_RECEIPT,
                "signed-F observation binding changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7SignedFObservationBindingRefusalCode.MUTATED_RECEIPT,
                "signed-F observation binding hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live signed-F observation bindings are nonserializable")


def issue_v1_sampled_v7_signed_f_observation_binding(
    *,
    alpha_registration: V1SampledV7SignedFAlphaRegistrationReceipt,
    projection_receipt: V1SignedWorkFixedPrimalProjectionReceipt,
) -> V1SampledV7SignedFObservationBindingReceipt:
    coordinates = _observation_coordinates(
        alpha_registration=alpha_registration,
        projection_receipt=projection_receipt,
    )
    return V1SampledV7SignedFObservationBindingReceipt(
        **coordinates,  # type: ignore[arg-type]
        _alpha_registration=alpha_registration,
        _projection_receipt=projection_receipt,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_AUTHORITY",
    "SAMPLED_V7_SIGNED_F_OBSERVATION_BINDING_SCHEMA",
    "V1SampledV7SignedFObservationBindingReceipt",
    "V1SampledV7SignedFObservationBindingRefusal",
    "V1SampledV7SignedFObservationBindingRefusalCode",
    "issue_v1_sampled_v7_signed_f_observation_binding",
]
