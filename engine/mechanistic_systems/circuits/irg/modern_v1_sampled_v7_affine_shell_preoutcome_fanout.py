"""Pre-provider affine-shell registration over a sampled-v7 D_W2 fanout.

The stable shared-D_W2 fanout owns the physical-action axis and the live
action-local compact-source registrations.  This versioned wrapper freezes one
structured literal-J direction family for every unique physical action before
the provider is allowed to run, without changing that stable fanout ABI.

This module is identity custody only.  It stores no arrays or outcomes, spends
no alpha, and grants no execution, population, decision, or commit authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import json
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_preoutcome_registration import (
    V1LiteralJAffineShellDirectionPreoutcomeRegistration,
    V1LiteralJAffineShellPreoutcomeRegistrationRefusal,
    V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode,
    register_v1_literal_j_affine_shell_direction_preoutcome,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_shared_dw2_fanout import (
    V1SampledV7SharedDW2FanoutDestination,
    V1SampledV7SharedDW2FanoutPlan,
)


SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_affine_shell_preoutcome_fanout_"
    "destination_binding_v1"
)
SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_affine_shell_preoutcome_fanout_plan_v1"
)
SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_affine_shell_preoutcome_fanout_receipt_v1"
)
SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY: Final = (
    "preprovider_complete_physical_action_axis_affine_shell_identity_custody_"
    "only_existing_literal_J_cell_reused_no_arrays_outcomes_new_alpha_provider_"
    "execution_population_projection_D_C_K_O_action_commit_terminal_performance_"
    "or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-affine-shell-preoutcome-fanout-"
_BINDING_TOKEN: Final = object()
_PLAN_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()


class V1SampledV7AffineShellPreoutcomeFanoutRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    BASE_PLAN_NOT_PREPARED = "base_plan_not_prepared"
    PROVIDER_OUTCOME_SEEN = "provider_outcome_seen"
    OWNER_MISMATCH = "owner_mismatch"
    WRONG_LITERAL_J_LOOK = "wrong_literal_j_look"
    DESTINATION_AXIS_MISMATCH = "destination_axis_mismatch"
    PARENT_INTEGRITY = "parent_integrity"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_BINDING = "mutated_binding"
    MUTATED_PLAN = "mutated_plan"


class V1SampledV7AffineShellPreoutcomeFanoutRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7AffineShellPreoutcomeFanoutRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7AffineShellPreoutcomeFanoutRefusalCode:
            raise TypeError("affine-shell preoutcome fanout refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7AffineShellPreoutcomeFanoutRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7AffineShellPreoutcomeFanoutRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _published_row_head(plan: V1SampledV7SharedDW2FanoutPlan) -> str | None:
    """Read the durable row-prefix head without touching provider state."""

    acquisition = plan._acquisition_plan
    head = acquisition.scratch_directory / (
        ".modern-v1-sampled-dual-chart-row-prefix-head.json"
    )
    if not head.exists():
        return None
    try:
        payload = json.loads(head.read_text(encoding="utf-8"))
        value = payload.get("manifest_hash")
    except Exception:
        return "<unreadable-row-head>"
    return value if type(value) is str and value else "<unreadable-row-head>"


def _provider_outcome_is_absent(plan: V1SampledV7SharedDW2FanoutPlan) -> bool:
    """Require both inert plan claims and the exact unpublished predecessor."""

    acquisition = plan._acquisition_plan
    return (
        acquisition.provider_outcomes_seen is False
        and all(destination.outcomes_seen is False for destination in plan.destinations)
        and _published_row_head(plan)
        == acquisition.group_plan.expected_prior_manifest_hash
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7AffineShellPreoutcomeDestinationBinding:
    """Array-free durable binding for one physical-action destination."""

    physical_action_ordinal: int
    physical_action_identity_hash: str
    base_destination_integrity_hash: str
    source_registration_hash: str
    physical_action_hash: str
    preregistration_hash: str
    direction_family_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_hash: str
    registered_look_index: int
    destination_frozen_before_provider: bool = True
    exact_source_action_preregistration_join: bool = True
    array_free_durable_binding: bool = True
    array_payload_present: bool = False
    outcome_payload_present: bool = False
    alpha_allocation_performed: bool = False
    downstream_authority: bool = False
    schema: str = SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_BINDING_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _BINDING_TOKEN:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
                "destination bindings are issuer-sealed",
            )
        _positive_int(self.physical_action_ordinal, name="physical_action_ordinal")
        _positive_int(self.registered_look_index, name="registered_look_index")
        for name in (
            "physical_action_identity_hash",
            "base_destination_integrity_hash",
            "source_registration_hash",
            "physical_action_hash",
            "preregistration_hash",
            "direction_family_hash",
            "literal_j_owner_binding_receipt_hash",
            "literal_j_cell_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            self.destination_frozen_before_provider is not True
            or self.exact_source_action_preregistration_join is not True
            or self.array_free_durable_binding is not True
            or self.array_payload_present is not False
            or self.outcome_payload_present is not False
            or self.alpha_allocation_performed is not False
            or self.downstream_authority is not False
            or self.schema
            != SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.AUTHORITY_VIOLATION,
                "destination binding crossed its array-free identity-only boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "destination-binding-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.MUTATED_BINDING,
                "destination binding hash does not replay",
            )
        return observed

    @property
    def binding_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> dict[str, object]:
        self.assert_integrity()
        return self._payload()


def _binding_axis_hash(
    *,
    base_plan_hash: str,
    bindings: tuple[V1SampledV7AffineShellPreoutcomeDestinationBinding, ...],
) -> str:
    return content_hash(
        {
            "schema": SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_PLAN_SCHEMA,
            "base_plan_hash": base_plan_hash,
            "ordered_complete_binding_axis": tuple(
                (
                    binding.physical_action_ordinal,
                    binding.physical_action_identity_hash,
                    binding.binding_hash,
                )
                for binding in bindings
            ),
        },
        prefix=_PREFIX + "binding-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7AffineShellPreoutcomeFanoutReceipt:
    """Durable proof that the complete action axis was preregistered.

    Unlike the live wrapper, this receipt remains replayable after the base
    fanout leaves ``prepared`` state.  It is only an identity witness: retaining
    it does not authorize provider execution or any downstream decision.
    """

    wrapper_plan_hash: str
    base_plan_hash: str
    physical_action_axis_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_hash: str
    registered_look_index: int
    destination_count: int
    binding_axis_hash: str
    destination_bindings: tuple[
        V1SampledV7AffineShellPreoutcomeDestinationBinding, ...
    ]
    preoutcome_registrations: tuple[
        V1LiteralJAffineShellDirectionPreoutcomeRegistration, ...
    ]
    ordered_complete_physical_action_axis: bool = True
    frozen_before_provider_outcomes: bool = True
    array_free_durable_receipt: bool = True
    array_payload_present: bool = False
    outcome_payload_present: bool = False
    alpha_allocation_performed: bool = False
    provider_execution_authority: bool = False
    population_projection_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY
    schema: str = SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
                "durable affine-shell fanout receipts are issuer-sealed",
            )
        self._validate_axes()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate_axes(self) -> None:
        for name in (
            "wrapper_plan_hash",
            "base_plan_hash",
            "physical_action_axis_hash",
            "literal_j_owner_binding_receipt_hash",
            "literal_j_cell_hash",
            "binding_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        count = _positive_int(self.destination_count, name="destination_count")
        _positive_int(self.registered_look_index, name="registered_look_index")
        if (
            type(self.destination_bindings) is not tuple
            or type(self.preoutcome_registrations) is not tuple
            or count != len(self.destination_bindings)
            or count != len(self.preoutcome_registrations)
            or any(
                type(value)
                is not V1SampledV7AffineShellPreoutcomeDestinationBinding
                for value in self.destination_bindings
            )
            or any(
                type(value)
                is not V1LiteralJAffineShellDirectionPreoutcomeRegistration
                for value in self.preoutcome_registrations
            )
            or tuple(
                value.physical_action_ordinal for value in self.destination_bindings
            )
            != tuple(range(1, count + 1))
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                "durable receipt does not carry one complete ordered action axis",
            )
        for binding, registration in zip(
            self.destination_bindings,
            self.preoutcome_registrations,
            strict=True,
        ):
            binding.assert_integrity()
            registration.assert_integrity()
            if (
                binding.preregistration_hash != registration.registration_hash
                or binding.source_registration_hash
                != registration.source_registration_hash
                or binding.physical_action_hash != registration.physical_action_hash
                or binding.physical_action_identity_hash
                != registration.physical_action_identity_hash
                or binding.direction_family_hash != registration.direction_family_hash
                or binding.literal_j_owner_binding_receipt_hash
                != self.literal_j_owner_binding_receipt_hash
                or registration.literal_j_owner_binding_receipt_hash
                != self.literal_j_owner_binding_receipt_hash
                or binding.literal_j_cell_hash != self.literal_j_cell_hash
                or registration.literal_j_cell_hash != self.literal_j_cell_hash
                or binding.registered_look_index != self.registered_look_index
                or registration.registered_look_index != self.registered_look_index
            ):
                _refuse(
                    V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                    "durable destination binding and preregistration axes diverge",
                )
        if self.binding_axis_hash != _binding_axis_hash(
            base_plan_hash=self.base_plan_hash,
            bindings=self.destination_bindings,
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                "durable complete binding-axis hash does not replay",
            )
        positive = (
            self.ordered_complete_physical_action_axis,
            self.frozen_before_provider_outcomes,
            self.array_free_durable_receipt,
        )
        closed = (
            self.array_payload_present,
            self.outcome_payload_present,
            self.alpha_allocation_performed,
            self.provider_execution_authority,
            self.population_projection_authority,
            self.d_stability_authority,
            self.c_k_o_or_commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.authority_scope
            != SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY
            or self.schema
            != SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.AUTHORITY_VIOLATION,
                "durable receipt crossed its identity-only authority boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "wrapper_plan_hash": self.wrapper_plan_hash,
            "base_plan_hash": self.base_plan_hash,
            "physical_action_axis_hash": self.physical_action_axis_hash,
            "literal_j_owner_binding_receipt_hash": (
                self.literal_j_owner_binding_receipt_hash
            ),
            "literal_j_cell_hash": self.literal_j_cell_hash,
            "registered_look_index": self.registered_look_index,
            "destination_count": self.destination_count,
            "binding_axis_hash": self.binding_axis_hash,
            "ordered_destination_binding_hashes": tuple(
                value.binding_hash for value in self.destination_bindings
            ),
            "ordered_preregistration_hashes": tuple(
                value.registration_hash for value in self.preoutcome_registrations
            ),
            "ordered_complete_physical_action_axis": (
                self.ordered_complete_physical_action_axis
            ),
            "frozen_before_provider_outcomes": self.frozen_before_provider_outcomes,
            "array_free_durable_receipt": self.array_free_durable_receipt,
            "array_payload_present": self.array_payload_present,
            "outcome_payload_present": self.outcome_payload_present,
            "alpha_allocation_performed": self.alpha_allocation_performed,
            "provider_execution_authority": self.provider_execution_authority,
            "population_projection_authority": self.population_projection_authority,
            "d_stability_authority": self.d_stability_authority,
            "c_k_o_or_commit_authority": self.c_k_o_or_commit_authority,
            "terminal_or_pareto_authority": self.terminal_or_pareto_authority,
            "authority_scope": self.authority_scope,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate_axes()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.MUTATED_PLAN,
                "durable affine-shell fanout receipt hash does not replay",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> dict[str, object]:
        """Return the complete nested array-free durable custody payload."""

        self.assert_integrity()
        return {
            **self._payload(),
            "destination_binding_payloads": tuple(
                value.to_payload() for value in self.destination_bindings
            ),
            "preoutcome_registration_payloads": tuple(
                value.to_payload() for value in self.preoutcome_registrations
            ),
        }


@dataclass(frozen=True, slots=True)
class V1SampledV7AffineShellPreoutcomeFanoutPlan:
    """Live wrapper retaining one prepared base fanout and its registrations."""

    base_plan_hash: str
    physical_action_axis_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_hash: str
    registered_look_index: int
    destination_count: int
    binding_axis_hash: str
    destination_bindings: tuple[
        V1SampledV7AffineShellPreoutcomeDestinationBinding, ...
    ]
    base_plan_prepared_at_freeze: bool = True
    provider_outcomes_seen: bool = False
    ordered_complete_binding_axis: bool = True
    existing_literal_j_cell_reused: bool = True
    extra_provider_call_budget: int = 0
    extra_vjp_budget: int = 0
    array_payload_present: bool = False
    outcome_payload_present: bool = False
    alpha_allocation_performed: bool = False
    provider_execution_authority: bool = False
    population_projection_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY
    schema: str = SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_PLAN_SCHEMA
    _base_plan: V1SampledV7SharedDW2FanoutPlan = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _literal_j_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt = field(
        repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _preoutcome_registrations: tuple[
        V1LiteralJAffineShellDirectionPreoutcomeRegistration, ...
    ] = field(repr=False, compare=False, default=())
    _durable_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLAN_TOKEN:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
                "affine-shell fanout wrappers are issuer-sealed",
            )
        self.assert_live()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())
        receipt = V1SampledV7AffineShellPreoutcomeFanoutReceipt(
            wrapper_plan_hash=self._sealed_hash,  # type: ignore[arg-type]
            base_plan_hash=self.base_plan_hash,
            physical_action_axis_hash=self.physical_action_axis_hash,
            literal_j_owner_binding_receipt_hash=(
                self.literal_j_owner_binding_receipt_hash
            ),
            literal_j_cell_hash=self.literal_j_cell_hash,
            registered_look_index=self.registered_look_index,
            destination_count=self.destination_count,
            binding_axis_hash=self.binding_axis_hash,
            destination_bindings=self.destination_bindings,
            preoutcome_registrations=self._preoutcome_registrations,
            _issuance_token=_RECEIPT_TOKEN,
        )
        object.__setattr__(self, "_durable_receipt", receipt)
        self.assert_live()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "base_plan_hash": self.base_plan_hash,
            "physical_action_axis_hash": self.physical_action_axis_hash,
            "literal_j_owner_binding_receipt_hash": (
                self.literal_j_owner_binding_receipt_hash
            ),
            "literal_j_cell_hash": self.literal_j_cell_hash,
            "registered_look_index": self.registered_look_index,
            "destination_count": self.destination_count,
            "binding_axis_hash": self.binding_axis_hash,
            "ordered_destination_binding_hashes": tuple(
                binding.binding_hash for binding in self.destination_bindings
            ),
            "base_plan_prepared_at_freeze": self.base_plan_prepared_at_freeze,
            "provider_outcomes_seen": self.provider_outcomes_seen,
            "ordered_complete_binding_axis": self.ordered_complete_binding_axis,
            "existing_literal_j_cell_reused": self.existing_literal_j_cell_reused,
            "extra_provider_call_budget": self.extra_provider_call_budget,
            "extra_vjp_budget": self.extra_vjp_budget,
            "array_payload_present": self.array_payload_present,
            "outcome_payload_present": self.outcome_payload_present,
            "alpha_allocation_performed": self.alpha_allocation_performed,
            "provider_execution_authority": self.provider_execution_authority,
            "population_projection_authority": self.population_projection_authority,
            "d_stability_authority": self.d_stability_authority,
            "c_k_o_or_commit_authority": self.c_k_o_or_commit_authority,
            "terminal_or_pareto_authority": self.terminal_or_pareto_authority,
            "authority_scope": self.authority_scope,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "plan-")

    def assert_live(self, expected_hash: str | None = None) -> str:
        if (
            type(self._base_plan) is not V1SampledV7SharedDW2FanoutPlan
            or type(self._literal_j_owner_binding)
            is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
            or type(self.destination_bindings) is not tuple
            or type(self._preoutcome_registrations) is not tuple
            or any(
                type(value)
                is not V1SampledV7AffineShellPreoutcomeDestinationBinding
                for value in self.destination_bindings
            )
            or any(
                type(value)
                is not V1LiteralJAffineShellDirectionPreoutcomeRegistration
                for value in self._preoutcome_registrations
            )
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
                "wrapper lost an exact live base plan, owner binding, or registration axis",
            )
        if self._base_plan._lease.state != "prepared":
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.BASE_PLAN_NOT_PREPARED,
                "base shared-D_W2 fanout is no longer in prepared state",
            )
        acquisition = self._base_plan._acquisition_plan
        if (
            not _provider_outcome_is_absent(self._base_plan)
            or self.provider_outcomes_seen is not False
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PROVIDER_OUTCOME_SEEN,
                "affine-shell registration must precede every provider outcome",
            )
        try:
            observed_base_hash = self._base_plan.assert_preoutcome()
        except Exception as error:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.BASE_PLAN_NOT_PREPARED,
                "base shared-D_W2 fanout is no longer prepared: "
                f"{type(error).__name__}: {error}",
            )
        try:
            owner_hash = self._literal_j_owner_binding.assert_live()
        except Exception as error:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY,
                "literal-J owner binding failed live replay: "
                f"{type(error).__name__}: {error}",
            )
        if (
            self._literal_j_owner_binding._measurement_owner
            is not acquisition._owner
            or self._literal_j_owner_binding.measurement_owner_hash
            != self._base_plan.measurement_owner_hash
            or owner_hash != self.literal_j_owner_binding_receipt_hash
            or self._literal_j_owner_binding.literal_j_cell_hash
            != self.literal_j_cell_hash
            or self._literal_j_owner_binding.registered_look_index
            != self.registered_look_index
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.OWNER_MISMATCH,
                "literal-J look does not belong to the exact base measurement owner",
            )
        count = _positive_int(self.destination_count, name="destination_count")
        if (
            observed_base_hash != self.base_plan_hash
            or self.physical_action_axis_hash
            != self._base_plan.physical_action_axis_hash
            or count != len(self._base_plan.destinations)
            or count != len(self.destination_bindings)
            or count != len(self._preoutcome_registrations)
            or tuple(
                destination.action.physical_action_ordinal
                for destination in self._base_plan.destinations
            )
            != tuple(range(1, count + 1))
            or tuple(
                binding.physical_action_ordinal
                for binding in self.destination_bindings
            )
            != tuple(range(1, count + 1))
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                "wrapper no longer covers the complete physical-action axis in order",
            )
        for destination, binding, registration in zip(
            self._base_plan.destinations,
            self.destination_bindings,
            self._preoutcome_registrations,
            strict=True,
        ):
            try:
                destination_hash = destination.assert_static_integrity(
                    require_pending=True
                )
                binding.assert_integrity()
                registration.assert_integrity()
            except V1SampledV7AffineShellPreoutcomeFanoutRefusal:
                raise
            except Exception as error:
                _refuse(
                    V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY,
                    "one destination binding parent failed integrity replay: "
                    f"{type(error).__name__}: {error}",
                )
            if (
                binding.base_destination_integrity_hash != destination_hash
                or binding.source_registration_hash
                != destination.source_registration.registration_hash
                or binding.physical_action_hash != destination.action.action_hash
                or binding.physical_action_identity_hash
                != destination.action.physical_action_identity_hash
                or binding.preregistration_hash != registration.registration_hash
                or binding.direction_family_hash != registration.direction_family_hash
                or binding.literal_j_owner_binding_receipt_hash != owner_hash
                or binding.literal_j_cell_hash
                != self._literal_j_owner_binding.literal_j_cell_hash
                or binding.registered_look_index != self.registered_look_index
                or registration.source_registration_hash
                != destination.source_registration.registration_hash
                or registration.physical_action_hash != destination.action.action_hash
                or registration.physical_action_identity_hash
                != destination.action.physical_action_identity_hash
                or registration.literal_j_owner_binding_receipt_hash != owner_hash
            ):
                _refuse(
                    V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                    "one destination, source, action, and preregistration join changed",
                )
        expected_axis_hash = _binding_axis_hash(
            base_plan_hash=observed_base_hash,
            bindings=self.destination_bindings,
        )
        positive = (
            self.base_plan_prepared_at_freeze,
            self.ordered_complete_binding_axis,
            self.existing_literal_j_cell_reused,
        )
        closed = (
            self.provider_outcomes_seen,
            self.array_payload_present,
            self.outcome_payload_present,
            self.alpha_allocation_performed,
            self.provider_execution_authority,
            self.population_projection_authority,
            self.d_stability_authority,
            self.c_k_o_or_commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            self.binding_axis_hash != expected_axis_hash
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.extra_provider_call_budget != 0
            or self.extra_vjp_budget != 0
            or self.authority_scope
            != SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY
            or self.schema != SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.AUTHORITY_VIOLATION,
                "wrapper crossed its pre-provider identity-only authority boundary",
            )
        observed = self._compute_hash()
        if self._sealed_hash is not None and observed != self._sealed_hash:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.MUTATED_PLAN,
                "wrapper plan hash changed after freeze",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.MUTATED_PLAN,
                "wrapper plan hash does not match the expected identity",
            )
        if self._sealed_hash is not None:
            if type(self._durable_receipt) is not V1SampledV7AffineShellPreoutcomeFanoutReceipt:
                _refuse(
                    V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY,
                    "sealed live wrapper lost its durable complete-axis receipt",
                )
            receipt = self._durable_receipt
            receipt.assert_integrity()
            if (
                receipt.wrapper_plan_hash != observed
                or receipt.base_plan_hash != self.base_plan_hash
                or receipt.physical_action_axis_hash != self.physical_action_axis_hash
                or receipt.binding_axis_hash != self.binding_axis_hash
                or receipt.destination_bindings != self.destination_bindings
                or receipt.preoutcome_registrations
                != self._preoutcome_registrations
            ):
                _refuse(
                    V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
                    "durable receipt no longer matches the live complete-axis wrapper",
                )
        return observed

    @property
    def plan_hash(self) -> str:
        return self.assert_live()

    @property
    def base_plan(self) -> V1SampledV7SharedDW2FanoutPlan:
        self.assert_live()
        return self._base_plan

    @property
    def preoutcome_registrations(
        self,
    ) -> tuple[V1LiteralJAffineShellDirectionPreoutcomeRegistration, ...]:
        self.assert_live()
        return self._preoutcome_registrations

    @property
    def durable_receipt(self) -> V1SampledV7AffineShellPreoutcomeFanoutReceipt:
        """Return the authority-inert receipt even after base-plan completion."""

        if type(self._durable_receipt) is not V1SampledV7AffineShellPreoutcomeFanoutReceipt:
            _refuse(
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY,
                "live wrapper lost its durable complete-axis receipt",
            )
        self._durable_receipt.assert_integrity()
        return self._durable_receipt

    def to_payload(self) -> dict[str, object]:
        """Return a durable array-free summary without live plan ownership."""

        self.assert_live()
        return self._payload()

    def require_provider_execution(self) -> NoReturn:
        raise PermissionError("preoutcome fanout wrapper has no provider authority")

    def require_population_projection(self) -> NoReturn:
        raise PermissionError("preoutcome fanout wrapper has no population authority")

    def require_mask_commit(self) -> NoReturn:
        raise PermissionError("preoutcome fanout wrapper cannot commit masks")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live affine-shell fanout wrappers are nonserializable")


def _issue_destination_binding(
    *,
    destination: V1SampledV7SharedDW2FanoutDestination,
    registration: V1LiteralJAffineShellDirectionPreoutcomeRegistration,
) -> V1SampledV7AffineShellPreoutcomeDestinationBinding:
    destination_hash = destination.assert_static_integrity(require_pending=True)
    return V1SampledV7AffineShellPreoutcomeDestinationBinding(
        physical_action_ordinal=destination.action.physical_action_ordinal,
        physical_action_identity_hash=(
            destination.action.physical_action_identity_hash
        ),
        base_destination_integrity_hash=destination_hash,
        source_registration_hash=destination.source_registration.registration_hash,
        physical_action_hash=destination.action.action_hash,
        preregistration_hash=registration.registration_hash,
        direction_family_hash=registration.direction_family_hash,
        literal_j_owner_binding_receipt_hash=(
            registration.literal_j_owner_binding_receipt_hash
        ),
        literal_j_cell_hash=registration.literal_j_cell_hash,
        registered_look_index=registration.registered_look_index,
        _issuance_token=_BINDING_TOKEN,
    )


def prepare_v1_sampled_v7_affine_shell_preoutcome_fanout(
    *,
    base_plan: V1SampledV7SharedDW2FanoutPlan,
    literal_j_owner_binding: V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
) -> V1SampledV7AffineShellPreoutcomeFanoutPlan:
    """Freeze all action-local structured-J families before provider execution."""

    if (
        type(base_plan) is not V1SampledV7SharedDW2FanoutPlan
        or type(literal_j_owner_binding)
        is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
    ):
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.INVALID_TYPE,
            "preoutcome fanout requires one exact base plan and literal-J binding",
        )
    if base_plan._lease.state != "prepared":
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.BASE_PLAN_NOT_PREPARED,
            "base shared-D_W2 fanout must be in prepared state",
        )
    acquisition = base_plan._acquisition_plan
    if not _provider_outcome_is_absent(base_plan):
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PROVIDER_OUTCOME_SEEN,
            "provider outcomes were already observed",
        )
    try:
        base_plan_hash = base_plan.assert_preoutcome()
    except Exception as error:
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.BASE_PLAN_NOT_PREPARED,
            "base shared-D_W2 fanout is not prepared: "
            f"{type(error).__name__}: {error}",
        )
    try:
        owner_hash = literal_j_owner_binding.assert_live()
    except Exception as error:
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY,
            "literal-J owner binding failed live replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        literal_j_owner_binding._measurement_owner is not acquisition._owner
        or literal_j_owner_binding.measurement_owner_hash
        != base_plan.measurement_owner_hash
    ):
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.OWNER_MISMATCH,
            "literal-J binding belongs to another D measurement owner",
        )
    destinations = tuple(base_plan.destinations)
    if (
        not destinations
        or tuple(destination.action.physical_action_ordinal for destination in destinations)
        != tuple(range(1, len(destinations) + 1))
    ):
        _refuse(
            V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.DESTINATION_AXIS_MISMATCH,
            "base fanout physical-action axis is empty, partial, or reordered",
        )
    parent_mask = acquisition._owner._current_mask
    native_fiber_axis = acquisition._owner._native_fiber_axis_receipt
    registrations: list[V1LiteralJAffineShellDirectionPreoutcomeRegistration] = []
    bindings: list[V1SampledV7AffineShellPreoutcomeDestinationBinding] = []
    for destination in destinations:
        try:
            registration = register_v1_literal_j_affine_shell_direction_preoutcome(
                source_registration=destination.source_registration,
                action=destination.action,
                parent_mask=parent_mask,
                native_fiber_axis=native_fiber_axis,
                literal_j_owner_binding=literal_j_owner_binding,
            )
        except V1LiteralJAffineShellPreoutcomeRegistrationRefusal as error:
            code = (
                V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.WRONG_LITERAL_J_LOOK
                if error.code
                in {
                    V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.PREFIX_MISMATCH,
                    V1LiteralJAffineShellPreoutcomeRegistrationRefusalCode.ALPHA_CELL_MISMATCH,
                }
                else V1SampledV7AffineShellPreoutcomeFanoutRefusalCode.PARENT_INTEGRITY
            )
            _refuse(
                code,
                "structured-J preregistration failed for physical action "
                f"{destination.action.physical_action_ordinal}: {error}",
            )
        registrations.append(registration)
        bindings.append(
            _issue_destination_binding(
                destination=destination,
                registration=registration,
            )
        )
    binding_axis = tuple(bindings)
    registration_axis = tuple(registrations)
    axis_hash = _binding_axis_hash(
        base_plan_hash=base_plan_hash,
        bindings=binding_axis,
    )
    return V1SampledV7AffineShellPreoutcomeFanoutPlan(
        base_plan_hash=base_plan_hash,
        physical_action_axis_hash=base_plan.physical_action_axis_hash,
        literal_j_owner_binding_receipt_hash=owner_hash,
        literal_j_cell_hash=literal_j_owner_binding.literal_j_cell_hash,
        registered_look_index=literal_j_owner_binding.registered_look_index,
        destination_count=len(destinations),
        binding_axis_hash=axis_hash,
        destination_bindings=binding_axis,
        _base_plan=base_plan,
        _literal_j_owner_binding=literal_j_owner_binding,
        _preoutcome_registrations=registration_axis,
        _issuance_token=_PLAN_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_AUTHORITY",
    "SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_BINDING_SCHEMA",
    "SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_PLAN_SCHEMA",
    "SAMPLED_V7_AFFINE_SHELL_PREOUTCOME_FANOUT_RECEIPT_SCHEMA",
    "V1SampledV7AffineShellPreoutcomeDestinationBinding",
    "V1SampledV7AffineShellPreoutcomeFanoutPlan",
    "V1SampledV7AffineShellPreoutcomeFanoutReceipt",
    "V1SampledV7AffineShellPreoutcomeFanoutRefusal",
    "V1SampledV7AffineShellPreoutcomeFanoutRefusalCode",
    "prepare_v1_sampled_v7_affine_shell_preoutcome_fanout",
]
