"""Bind one sampled-V7 corrected-A confidence observation to its alpha cell.

The sampled-V7 alpha-family registry charges a finite A/F/J family before D
measurements exist.  The live measurement-owner join identifies the exact
corrected-A cell for one registered look.  This module closes only the next
identity seam: the actual hybrid Serfling confidence box must be the box built
from that pre-registered design, prefix, and scalar alpha.

This receipt does not spend alpha a second time and does not mark the cell as
durably consumed.  A later A/F/J observation-family transaction must provide
that nonrecycling consumption authority.  No D/C/K/O/action/commit/terminal or
Pareto authority is created here.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_proposal_statistics import (
    V1CorrectedAttributionConfidenceBoxV2Receipt,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    SAMPLED_V7_D_ALPHA_MEASUREMENT_OWNER_BINDING_SCHEMA,
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)


SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_corrected_A_observation_alpha_cell_binding_v1"
)
SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_AUTHORITY: Final = (
    "sampled_v7_corrected_A_confidence_box_to_preregistered_alpha_cell_identity_"
    "binding_only_no_second_alpha_spend_cell_consumption_D_C_K_O_action_commit_"
    "terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-corrected-A-observation-binding-"
_OWNER_PREFIX_AXIS_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-alpha-owner-prefix-axis-full-"
)
_ISSUER_TOKEN: Final = object()


class V1SampledV7CorrectedAObservationBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    DESIGN_MISMATCH = "design_mismatch"
    OBSERVATION_IDENTITY_MISMATCH = "observation_identity_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    CENSUS_MISMATCH = "census_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"


class V1SampledV7CorrectedAObservationBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CorrectedAObservationBindingRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CorrectedAObservationBindingRefusalCode:
            raise TypeError("corrected-A observation refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CorrectedAObservationBindingRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CorrectedAObservationBindingRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty identity",
        )
    return value


def _assert_parents(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    confidence_box: V1CorrectedAttributionConfidenceBoxV2Receipt,
) -> tuple[str, str]:
    if type(owner_binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.INVALID_TYPE,
            "corrected-A observation binding requires the exact live owner join",
        )
    if type(confidence_box) is not V1CorrectedAttributionConfidenceBoxV2Receipt:
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.INVALID_TYPE,
            "corrected-A observation binding requires the exact v2 confidence box",
        )
    try:
        owner_hash = owner_binding.assert_live()
        box_hash = confidence_box.assert_integrity()
    except V1SampledV7CorrectedAObservationBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.PARENT_INTEGRITY,
            f"one corrected-A parent failed replay: {type(error).__name__}: {error}",
        )
    return owner_hash, box_hash


def _observation_coordinates(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    confidence_box: V1CorrectedAttributionConfidenceBoxV2Receipt,
) -> dict[str, object]:
    owner_hash, box_hash = _assert_parents(
        owner_binding=owner_binding,
        confidence_box=confidence_box,
    )
    plan = owner_binding._alpha_family_plan
    registry = owner_binding._alpha_family_registry
    plan_hash = plan.assert_integrity()
    registry_hash = registry.assert_integrity()

    identity_pairs = (
        (confidence_box.campaign_contract_hash, owner_binding.campaign_contract_hash),
        (confidence_box.model_identity_hash, owner_binding.model_identity_hash),
        (confidence_box.query_contract_hash, owner_binding.query_contract_hash),
        (confidence_box.preprocessing_hash, owner_binding.preprocessing_hash),
        (confidence_box.current_mask_hash, owner_binding.current_mask_hash),
        (
            confidence_box.raw_logit_stream_source_hash,
            owner_binding.raw_logit_stream_source_hash,
        ),
        (confidence_box.sampling_law_hash, owner_binding.sampling_law_hash),
        (confidence_box.route_axis_hash, owner_binding.ordered_route_axis_hash),
    )
    if (
        any(left != right for left, right in identity_pairs)
        or confidence_box.proposal_generation != owner_binding.proposal_generation
        or confidence_box.route_count != owner_binding.route_count
        or confidence_box.stratum_names != owner_binding.stratum_names
        or confidence_box.population_stratum_counts
        != owner_binding.population_stratum_counts
        or confidence_box.sealed_population_order_hashes
        != owner_binding.sealed_population_order_hashes
        or confidence_box.look_index != owner_binding.registered_look_index
        or confidence_box.stratum_counts
        != owner_binding.registered_look_stratum_counts
    ):
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.OBSERVATION_IDENTITY_MISMATCH,
            "confidence box and live alpha-owner join name different D observations",
        )

    if (
        confidence_box.base_serfling_design_hash
        != registry.charged_base_design_hash
        or (
            registry.hybrid_wrapper_design_hash is not None
            and confidence_box.design_hash != registry.hybrid_wrapper_design_hash
        )
        or confidence_box.registered_look_stratum_counts
        != plan.registered_look_stratum_counts
        or confidence_box.alpha_by_look != plan.corrected_a_alpha_by_look
        or confidence_box.trajectory_alpha_ledger_hash != plan.ledger_before_hash
        or confidence_box.trajectory_alpha_ledger_generation
        != plan.proposal_generation
        or confidence_box.generation_alpha_allocation
        != plan.generation_alpha_allocation
    ):
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.DESIGN_MISMATCH,
            "confidence box did not consume the registered corrected-A design",
        )

    if confidence_box.look_alpha != owner_binding.corrected_a_cell_alpha:
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.ALPHA_MISMATCH,
            "confidence-box look alpha differs from the preregistered A cell",
        )
    exact_census = (
        owner_binding.registered_look_stratum_counts
        == owner_binding.population_stratum_counts
    )
    observed_census = confidence_box.census_by_stratum == (True, True)
    if (
        observed_census is not exact_census
        or (exact_census and confidence_box.look_alpha != 0.0)
        or (not exact_census and confidence_box.look_alpha <= 0.0)
    ):
        _refuse(
            V1SampledV7CorrectedAObservationBindingRefusalCode.CENSUS_MISMATCH,
            "corrected-A census and alpha-spend semantics do not replay",
        )

    contexts = confidence_box.observed_context_ids_by_stratum
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
            V1SampledV7CorrectedAObservationBindingRefusalCode.PREFIX_MISMATCH,
            "confidence-box context ledger is not the registered live D prefix",
        )

    draw_axis_hash = content_hash(
        {
            "look_index": confidence_box.look_index,
            "draw_ids_by_stratum": confidence_box.observed_draw_ids_by_stratum,
        },
        prefix=_PREFIX + "observed-draw-axis-",
    )
    return {
        "owner_binding_receipt_hash": owner_hash,
        "alpha_family_plan_hash": plan_hash,
        "alpha_family_registry_receipt_hash": registry_hash,
        "alpha_head_cas_receipt_hash": owner_binding.alpha_head_cas_receipt_hash,
        "corrected_a_cell_id": owner_binding.corrected_a_cell_id,
        "corrected_a_cell_hash": owner_binding.corrected_a_cell_hash,
        "corrected_a_cell_scope_hash": owner_binding.corrected_a_cell_scope_hash,
        "corrected_a_cell_alpha": owner_binding.corrected_a_cell_alpha,
        "confidence_box_receipt_hash": box_hash,
        "confidence_box_state_hash": confidence_box.state_hash,
        "base_serfling_design_hash": confidence_box.base_serfling_design_hash,
        "hybrid_design_hash": confidence_box.design_hash,
        "registered_look_index": confidence_box.look_index,
        "registered_look_stratum_counts": confidence_box.stratum_counts,
        "population_stratum_counts": confidence_box.population_stratum_counts,
        "registered_prefix_axis_hash": prefix_axis_hash,
        "observed_draw_axis_hash": draw_axis_hash,
        "look_alpha_observed": confidence_box.look_alpha,
        "exact_census": exact_census,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7CorrectedAObservationBindingReceipt:
    owner_binding_receipt_hash: str
    alpha_family_plan_hash: str
    alpha_family_registry_receipt_hash: str
    alpha_head_cas_receipt_hash: str
    corrected_a_cell_id: str
    corrected_a_cell_hash: str
    corrected_a_cell_scope_hash: str
    corrected_a_cell_alpha: float
    confidence_box_receipt_hash: str
    confidence_box_state_hash: str
    base_serfling_design_hash: str
    hybrid_design_hash: str
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    registered_prefix_axis_hash: str
    observed_draw_axis_hash: str
    look_alpha_observed: float
    exact_census: bool
    source_observation_bound: bool = True
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
    authority_scope: str = SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_SCHEMA
    _owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _confidence_box: V1CorrectedAttributionConfidenceBoxV2Receipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("corrected-A observation bindings are issuer-only")
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
        expected = _observation_coordinates(
            owner_binding=self._owner_binding,
            confidence_box=self._confidence_box,
        )
        for name, value in expected.items():
            if getattr(self, name) != value:
                _refuse(
                    V1SampledV7CorrectedAObservationBindingRefusalCode.MUTATED_RECEIPT,
                    f"corrected-A observation coordinate {name} changed",
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
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7CorrectedAObservationBindingRefusalCode.AUTHORITY_VIOLATION,
                "corrected-A observation binding crossed its identity-only boundary",
            )
        observed = self._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7CorrectedAObservationBindingRefusalCode.MUTATED_RECEIPT,
                "corrected-A observation binding changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7CorrectedAObservationBindingRefusalCode.MUTATED_RECEIPT,
                "corrected-A observation binding hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_live()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live corrected-A observation bindings are nonserializable")


def issue_v1_sampled_v7_corrected_a_observation_binding(
    *,
    owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
    confidence_box: V1CorrectedAttributionConfidenceBoxV2Receipt,
) -> V1SampledV7CorrectedAObservationBindingReceipt:
    coordinates = _observation_coordinates(
        owner_binding=owner_binding,
        confidence_box=confidence_box,
    )
    return V1SampledV7CorrectedAObservationBindingReceipt(
        **coordinates,  # type: ignore[arg-type]
        _owner_binding=owner_binding,
        _confidence_box=confidence_box,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_AUTHORITY",
    "SAMPLED_V7_CORRECTED_A_OBSERVATION_BINDING_SCHEMA",
    "V1SampledV7CorrectedAObservationBindingReceipt",
    "V1SampledV7CorrectedAObservationBindingRefusal",
    "V1SampledV7CorrectedAObservationBindingRefusalCode",
    "issue_v1_sampled_v7_corrected_a_observation_binding",
]
