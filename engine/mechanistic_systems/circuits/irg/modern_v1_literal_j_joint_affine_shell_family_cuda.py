"""Joint, action-complete structured literal-J direction custody.

One sampled-v7 literal-J alpha cell decides among the complete physical-action
axis.  Consequently its confidence family is registered once over

``(z_outside^1, ..., z_outside^A, e_b1, ..., e_bB)``

rather than once per action.  This owner joins the already pre-registered
fanout receipt to exactly one canonical postsource direction handle per action.
It retains the parent leases and exposes their outside vectors as an ordered
tuple plus one shared shell-index vector.  It never stacks the outside vectors
and never materialises a shell basis bank.

This is process-local direction custody and statistical-family registration.
It allocates no alpha and grants no population, D, C, K, O, commit, terminal,
performance, or Pareto authority.  A later joint shard/residual producer must
bind this receipt hash and use the registered statement denominator unchanged.
The outer 85% H200 admission forecast also belongs to that allocating producer:
this zero-allocation custody join neither forecasts nor authorizes HBM admission.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_direction_cuda import (
    V1LiteralJAffineShellDirectionConstructionCustodyKind,
    V1LiteralJAffineShellDirectionDeviceHandle,
    V1LiteralJAffineShellDirectionExecutionMode,
    V1LiteralJAffineShellDirectionLease,
)
from .modern_v1_sampled_v7_affine_shell_preoutcome_fanout import (
    V1SampledV7AffineShellPreoutcomeFanoutPlan,
    V1SampledV7AffineShellPreoutcomeFanoutReceipt,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_preoutcome_registration_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY: Final = (
    "preoutcome_action_complete_literal_J_statistical_family_identity_and_"
    "existing_alpha_denominator_only_no_outcomes_new_alpha_population_D_C_K_O_"
    "commit_terminal_performance_or_pareto_authority_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_family_receipt_v4"
)
LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_AUTHORITY: Final = (
    "joint_action_complete_structured_literal_J_direction_and_existing_alpha_"
    "denominator_custody_only_no_new_alpha_population_FD_solver_D_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_ALGORITHM: Final = (
    "ordered_action_outside_tuple_plus_one_shared_shell_no_dense_action_route_"
    "stack_or_basis_bank_v1"
)
_PREFIX: Final = "partition-free-v1-literal-j-joint-affine-shell-family-"
_PREOUTCOME_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()


class V1LiteralJJointAffineShellFamilyExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1LiteralJJointAffineShellFamilyRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    PREOUTCOME_REGISTRATION_MISMATCH = "preoutcome_registration_mismatch"
    DESTINATION_AXIS_MISMATCH = "destination_axis_mismatch"
    UNBOUND_DIRECTION = "unbound_direction"
    SOURCE_CUSTODY_MISMATCH = "source_custody_mismatch"
    J_CELL_MISMATCH = "j_cell_mismatch"
    CHART_MISMATCH = "chart_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    SHELL_MISMATCH = "shell_mismatch"
    EXECUTION_MODE_MISMATCH = "execution_mode_mismatch"
    DEVICE_REQUIRED = "device_required"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    STALE_IDENTITY = "stale_identity"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointAffineShellFamilyRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointAffineShellFamilyRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointAffineShellFamilyRefusalCode:
            raise TypeError("joint affine-shell refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointAffineShellFamilyRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointAffineShellFamilyRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty and unique",
        )
    return result


def _identity_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty",
        )
    return result


def _optional_identity_tuple(
    value: object,
    *,
    name: str,
) -> tuple[str | None, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty immutable tuple",
        )
    result: list[str | None] = []
    for item in value:
        if item is None:
            result.append(None)
        else:
            result.append(_identity(item, name=f"{name} entry"))
    return tuple(result)


def _finite_alpha(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be one finite binary64 probability in (0, 1)",
        )
    return value


def _divide_probability_down(value: float, count: int) -> float:
    """Return a binary64 quotient that cannot exceed the exact quotient."""

    alpha = _finite_alpha(value, name="literal_j_cell_alpha")
    divisor = _positive_int(count, name="simultaneous_statement_count")
    exact = Fraction.from_float(alpha) / divisor
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, 0.0)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.NUMERICAL_REFUSAL,
            "the action-complete per-statement alpha underflowed binary64",
        )
    return result


def _outside_direction_id(ordinal: int, action_identity_hash: str) -> str:
    return f"literal-j:outside:action:{ordinal}:{action_identity_hash}"


def _shell_direction_id(route_id: str) -> str:
    return f"literal-j:shared-shell-basis:{route_id}"


def _require_exact_common_device(
    tensors: tuple[Any, ...],
    *,
    expected_type: str,
) -> object:
    """Require exact device equality, including a CUDA device ordinal."""

    if not tensors:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_DESIGN,
            "joint device validation requires a nonempty tensor tuple",
        )
    reference = tensors[0].device
    if reference.type != expected_type or any(
        tensor.device != reference for tensor in tensors[1:]
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DEVICE_REQUIRED,
            "all joint outside vectors and the shared shell must occupy the "
            "same exact device, including CUDA ordinal",
        )
    return reference


def _preoutcome_joint_axis_hash(
    *,
    fanout_receipt_hash: str,
    physical_action_axis_hash: str,
    fanout_binding_axis_hash: str,
    physical_action_identity_hashes: tuple[str, ...],
    physical_action_hashes: tuple[str, ...],
    source_registration_hashes: tuple[str, ...],
    preoutcome_registration_hashes: tuple[str, ...],
    destination_binding_hashes: tuple[str, ...],
    joint_direction_ids: tuple[str, ...],
    literal_j_cell_hash: str,
    literal_j_cell_scope_hash: str,
    historical_alpha_ledger_hash: str,
    historical_alpha_coordinate_hash: str,
    historical_alpha_lineage_hash: str,
    registered_d_prefix_hash: str,
    registered_d_w2_prefix_hash: str,
    route_axis_hash: str,
    route_count: int,
    simultaneous_statement_count: int,
    familywise_alpha: float,
) -> str:
    """Hash only coordinates frozen before any postsource direction exists."""

    return content_hash(
        {
            "schema": LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA,
            "fanout_receipt_hash": fanout_receipt_hash,
            "physical_action_axis_hash": physical_action_axis_hash,
            "fanout_binding_axis_hash": fanout_binding_axis_hash,
            "physical_action_identity_hashes": physical_action_identity_hashes,
            "physical_action_hashes": physical_action_hashes,
            "source_registration_hashes": source_registration_hashes,
            "preoutcome_registration_hashes": preoutcome_registration_hashes,
            "destination_binding_hashes": destination_binding_hashes,
            "joint_direction_ids": joint_direction_ids,
            "literal_j_cell_hash": literal_j_cell_hash,
            "literal_j_cell_scope_hash": literal_j_cell_scope_hash,
            "historical_alpha_ledger_hash": historical_alpha_ledger_hash,
            "historical_alpha_coordinate_hash": historical_alpha_coordinate_hash,
            "historical_alpha_lineage_hash": historical_alpha_lineage_hash,
            "registered_d_prefix_hash": registered_d_prefix_hash,
            "registered_d_w2_prefix_hash": registered_d_w2_prefix_hash,
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "simultaneous_statement_count": simultaneous_statement_count,
            "familywise_alpha": familywise_alpha,
            "postsource_receipt_or_outcome_hash_present": False,
        },
        prefix=_PREFIX + "preoutcome-direction-axis-",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellPreoutcomeRegistration:
    """Action-complete literal-J family fixed before postsource outcomes."""

    fanout_receipt_hash: str
    fanout_wrapper_plan_hash: str
    fanout_base_plan_hash: str
    physical_action_axis_hash: str
    fanout_binding_axis_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_stream_binding_hash: str
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    historical_alpha_ledger_hash: str
    historical_alpha_coordinate_hash: str
    historical_alpha_lineage_hash: str
    registered_look_index: int
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
    action_count: int
    shell_size: int
    direction_count: int
    physical_action_ordinals: tuple[int, ...]
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    source_registration_hashes: tuple[str, ...]
    preoutcome_registration_hashes: tuple[str, ...]
    destination_binding_hashes: tuple[str, ...]
    outside_direction_ids: tuple[str, ...]
    shared_shell_direction_ids: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    registered_before_postsource_outcomes: bool = True
    exact_complete_fanout_parent_bound: bool = True
    action_qualified_outside_axis_frozen: bool = True
    one_shared_shell_axis_frozen: bool = True
    existing_literal_j_cell_reused: bool = True
    alpha_allocation_performed: bool = False
    postsource_identity_present: bool = False
    outcome_identity_present: bool = False
    array_free_durable_registration: bool = True
    population_projection_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_or_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = (
        LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY
    )
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PREOUTCOME_TOKEN:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
                "joint preoutcome registrations are issuer-sealed",
            )
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "fanout_receipt_hash",
            "fanout_wrapper_plan_hash",
            "fanout_base_plan_hash",
            "physical_action_axis_hash",
            "fanout_binding_axis_hash",
            "literal_j_owner_binding_receipt_hash",
            "literal_j_cell_id",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "literal_j_stream_binding_hash",
            "literal_j_stream_source_hash",
            "same_capture_vjp_source_hash",
            "historical_alpha_ledger_hash",
            "historical_alpha_coordinate_hash",
            "historical_alpha_lineage_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        actions = _positive_int(self.action_count, name="action_count")
        shell_size = _positive_int(self.shell_size, name="shell_size")
        directions = _positive_int(self.direction_count, name="direction_count")
        routes = _positive_int(self.route_count, name="route_count")
        _positive_int(self.registered_look_index, name="registered_look_index")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        if (
            type(self.physical_action_ordinals) is not tuple
            or any(type(item) is not int for item in self.physical_action_ordinals)
            or self.physical_action_ordinals != tuple(range(1, actions + 1))
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "preoutcome physical-action ordinals are not complete and ordered",
            )
        action_axes = (
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
            self.source_registration_hashes,
            self.preoutcome_registration_hashes,
            self.destination_binding_hashes,
            self.outside_direction_ids,
        )
        if any(len(_identity_tuple(axis, name="preoutcome action axis")) != actions for axis in action_axes):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "one preoutcome action axis is incomplete",
            )
        if any(
            len(set(axis)) != actions
            for axis in (
                self.physical_action_identity_hashes,
                self.physical_action_hashes,
                self.preoutcome_registration_hashes,
                self.destination_binding_hashes,
                self.outside_direction_ids,
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "preoutcome action-defining axes must be unique",
            )
        if (
            type(self.shell_indices) is not tuple
            or any(type(item) is not int for item in self.shell_indices)
            or len(self.shell_indices) != shell_size
            or not 1 <= shell_size <= 8
            or len(set(self.shell_indices)) != shell_size
            or any(item < 0 or item >= routes for item in self.shell_indices)
            or len(_identity_axis(self.shell_route_ids, name="shell_route_ids"))
            != shell_size
            or len(
                _identity_axis(
                    self.shared_shell_direction_ids,
                    name="shared_shell_direction_ids",
                )
            )
            != shell_size
            or directions != actions + shell_size
            or len(_identity_axis(self.joint_direction_ids, name="joint_direction_ids"))
            != directions
            or self.outside_direction_ids
            != tuple(
                _outside_direction_id(ordinal, identity)
                for ordinal, identity in zip(
                    self.physical_action_ordinals,
                    self.physical_action_identity_hashes,
                    strict=True,
                )
            )
            or self.shared_shell_direction_ids
            != tuple(_shell_direction_id(item) for item in self.shell_route_ids)
            or self.joint_direction_ids
            != self.outside_direction_ids + self.shared_shell_direction_ids
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH,
                "preoutcome outside-first/shared-shell family does not replay",
            )
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(
            self.simultaneous_statement_count,
            name="simultaneous_statement_count",
        )
        per_statement = _finite_alpha(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        expected_axis_hash = _preoutcome_joint_axis_hash(
            fanout_receipt_hash=self.fanout_receipt_hash,
            physical_action_axis_hash=self.physical_action_axis_hash,
            fanout_binding_axis_hash=self.fanout_binding_axis_hash,
            physical_action_identity_hashes=self.physical_action_identity_hashes,
            physical_action_hashes=self.physical_action_hashes,
            source_registration_hashes=self.source_registration_hashes,
            preoutcome_registration_hashes=self.preoutcome_registration_hashes,
            destination_binding_hashes=self.destination_binding_hashes,
            joint_direction_ids=self.joint_direction_ids,
            literal_j_cell_hash=self.literal_j_cell_hash,
            literal_j_cell_scope_hash=self.literal_j_cell_scope_hash,
            historical_alpha_ledger_hash=self.historical_alpha_ledger_hash,
            historical_alpha_coordinate_hash=(
                self.historical_alpha_coordinate_hash
            ),
            historical_alpha_lineage_hash=self.historical_alpha_lineage_hash,
            registered_d_prefix_hash=self.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=self.registered_d_w2_prefix_hash,
            route_axis_hash=self.route_axis_hash,
            route_count=routes,
            simultaneous_statement_count=statements,
            familywise_alpha=alpha,
        )
        positive = (
            self.registered_before_postsource_outcomes,
            self.exact_complete_fanout_parent_bound,
            self.action_qualified_outside_axis_frozen,
            self.one_shared_shell_axis_frozen,
            self.existing_literal_j_cell_reused,
            self.array_free_durable_registration,
        )
        closed = (
            self.alpha_allocation_performed,
            self.postsource_identity_present,
            self.outcome_identity_present,
            self.population_projection_authority,
            self.d_stability_authority,
            self.c_k_o_or_commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            statements != 2 * directions * (routes + 1)
            or per_statement != _divide_probability_down(alpha, statements)
            or self.joint_direction_axis_hash != expected_axis_hash
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.authority_scope
            != LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY
            or self.schema
            != LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.AUTHORITY_VIOLATION,
                "joint preoutcome family crossed its alpha or authority boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "preoutcome-registration-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "joint preoutcome registration hash does not replay",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> dict[str, object]:
        self.assert_integrity()
        return self._payload()


def _derive_v1_literal_j_joint_affine_shell_preoutcome_coordinates(
    *,
    fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
) -> dict[str, object]:
    """Authority-inert deterministic replay from one durable fanout receipt."""

    if type(fanout_receipt) is not V1SampledV7AffineShellPreoutcomeFanoutReceipt:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            "joint preoutcome registration requires the exact fanout receipt",
        )
    try:
        fanout_hash = fanout_receipt.assert_integrity()
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "the complete preoutcome fanout receipt failed replay: "
            f"{type(error).__name__}: {error}",
        )
    registrations = fanout_receipt.preoutcome_registrations
    bindings = fanout_receipt.destination_bindings
    actions = fanout_receipt.destination_count
    if (
        type(registrations) is not tuple
        or type(bindings) is not tuple
        or len(registrations) != actions
        or len(bindings) != actions
        or actions <= 0
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "fanout does not expose one complete preoutcome action axis",
        )
    first = registrations[0]
    common_fields = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "route_count",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "shell_freeze_receipt_hash",
        "shell_indices",
        "shell_route_ids",
        "literal_j_stream_source_hash",
        "same_capture_vjp_source_hash",
        "literal_j_stream_binding_hash",
        "literal_j_owner_binding_receipt_hash",
        "literal_j_cell_id",
        "literal_j_cell_hash",
        "literal_j_cell_scope_hash",
        "literal_j_cell_alpha",
        "historical_alpha_ledger_hash",
        "historical_alpha_coordinate_hash",
        "historical_alpha_lineage_hash",
        "registered_look_index",
    )
    for registration in registrations:
        try:
            registration.assert_integrity()
        except Exception as error:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                "one preoutcome member failed replay: "
                f"{type(error).__name__}: {error}",
            )
        mismatches = tuple(
            name
            for name in common_fields
            if getattr(registration, name) != getattr(first, name)
        )
        if mismatches:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
                "fanout members cannot define one common joint J family: "
                + ", ".join(mismatches),
            )
    identities = tuple(item.physical_action_identity_hash for item in bindings)
    action_hashes = tuple(item.physical_action_hash for item in bindings)
    if len(set(identities)) != actions or len(set(action_hashes)) != actions:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "joint preoutcome actions must be unique",
        )
    outside_ids = tuple(
        _outside_direction_id(ordinal, identity)
        for ordinal, identity in enumerate(identities, start=1)
    )
    shell_ids = tuple(_shell_direction_id(item) for item in first.shell_route_ids)
    joint_ids = outside_ids + shell_ids
    statements = 2 * len(joint_ids) * (first.route_count + 1)
    source_hashes = tuple(item.source_registration_hash for item in registrations)
    prereg_hashes = tuple(item.registration_hash for item in registrations)
    binding_hashes = tuple(item.binding_hash for item in bindings)
    axis_hash = _preoutcome_joint_axis_hash(
        fanout_receipt_hash=fanout_hash,
        physical_action_axis_hash=fanout_receipt.physical_action_axis_hash,
        fanout_binding_axis_hash=fanout_receipt.binding_axis_hash,
        physical_action_identity_hashes=identities,
        physical_action_hashes=action_hashes,
        source_registration_hashes=source_hashes,
        preoutcome_registration_hashes=prereg_hashes,
        destination_binding_hashes=binding_hashes,
        joint_direction_ids=joint_ids,
        literal_j_cell_hash=first.literal_j_cell_hash,
        literal_j_cell_scope_hash=first.literal_j_cell_scope_hash,
        historical_alpha_ledger_hash=first.historical_alpha_ledger_hash,
        historical_alpha_coordinate_hash=first.historical_alpha_coordinate_hash,
        historical_alpha_lineage_hash=first.historical_alpha_lineage_hash,
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        route_axis_hash=first.route_axis_hash,
        route_count=first.route_count,
        simultaneous_statement_count=statements,
        familywise_alpha=first.literal_j_cell_alpha,
    )
    return dict(
        fanout_receipt_hash=fanout_hash,
        fanout_wrapper_plan_hash=fanout_receipt.wrapper_plan_hash,
        fanout_base_plan_hash=fanout_receipt.base_plan_hash,
        physical_action_axis_hash=fanout_receipt.physical_action_axis_hash,
        fanout_binding_axis_hash=fanout_receipt.binding_axis_hash,
        literal_j_owner_binding_receipt_hash=(
            fanout_receipt.literal_j_owner_binding_receipt_hash
        ),
        literal_j_cell_id=first.literal_j_cell_id,
        literal_j_cell_hash=first.literal_j_cell_hash,
        literal_j_cell_scope_hash=first.literal_j_cell_scope_hash,
        literal_j_stream_binding_hash=first.literal_j_stream_binding_hash,
        literal_j_stream_source_hash=first.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=first.same_capture_vjp_source_hash,
        historical_alpha_ledger_hash=first.historical_alpha_ledger_hash,
        historical_alpha_coordinate_hash=first.historical_alpha_coordinate_hash,
        historical_alpha_lineage_hash=first.historical_alpha_lineage_hash,
        registered_look_index=first.registered_look_index,
        model_identity_hash=first.model_identity_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.current_mask_hash,
        proposal_generation=first.proposal_generation,
        route_axis_hash=first.route_axis_hash,
        route_id_axis_hash=first.route_id_axis_hash,
        route_count=first.route_count,
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=first.shell_freeze_receipt_hash,
        shell_indices=first.shell_indices,
        shell_route_ids=first.shell_route_ids,
        action_count=actions,
        shell_size=len(first.shell_indices),
        direction_count=len(joint_ids),
        physical_action_ordinals=tuple(range(1, actions + 1)),
        physical_action_identity_hashes=identities,
        physical_action_hashes=action_hashes,
        packed_action_hashes=tuple(item.packed_action_hash for item in registrations),
        source_registration_hashes=source_hashes,
        preoutcome_registration_hashes=prereg_hashes,
        destination_binding_hashes=binding_hashes,
        outside_direction_ids=outside_ids,
        shared_shell_direction_ids=shell_ids,
        joint_direction_ids=joint_ids,
        joint_direction_axis_hash=axis_hash,
        familywise_alpha=first.literal_j_cell_alpha,
        simultaneous_statement_count=statements,
        per_statement_two_sided_alpha=_divide_probability_down(
            first.literal_j_cell_alpha,
            statements,
        ),
    )


def register_v1_literal_j_joint_affine_shell_preoutcome(
    *,
    fanout_plan: V1SampledV7AffineShellPreoutcomeFanoutPlan,
) -> V1LiteralJJointAffineShellPreoutcomeRegistration:
    """Canonically freeze the joint family while the provider is still absent."""

    if type(fanout_plan) is not V1SampledV7AffineShellPreoutcomeFanoutPlan:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            "canonical joint preregistration requires the exact live fanout plan",
        )
    try:
        fanout_plan.assert_live()
        durable = fanout_plan.durable_receipt
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "joint family registration must occur while the fanout remains live "
            "and before every provider outcome: "
            f"{type(error).__name__}: {error}",
        )
    coordinates = _derive_v1_literal_j_joint_affine_shell_preoutcome_coordinates(
        fanout_receipt=durable,
    )
    return V1LiteralJJointAffineShellPreoutcomeRegistration(
        **coordinates,  # type: ignore[arg-type]
        _issuance_token=_PREOUTCOME_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellFamilyReceipt:
    """Array-free custody and single-family denominator receipt."""

    fanout_receipt_hash: str
    joint_preoutcome_registration_hash: str
    fanout_wrapper_plan_hash: str
    fanout_base_plan_hash: str
    physical_action_axis_hash: str
    fanout_binding_axis_hash: str
    literal_j_owner_binding_receipt_hash: str
    literal_j_cell_id: str
    literal_j_cell_hash: str
    literal_j_cell_scope_hash: str
    literal_j_stream_binding_hash: str
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hash: str
    historical_alpha_ledger_hash: str
    historical_alpha_coordinate_hash: str
    historical_alpha_lineage_hash: str
    registered_look_index: int
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
    action_count: int
    shell_size: int
    direction_count: int
    physical_action_ordinals: tuple[int, ...]
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    source_registration_hashes: tuple[str, ...]
    preoutcome_registration_hashes: tuple[str, ...]
    destination_binding_hashes: tuple[str, ...]
    direction_receipt_hashes: tuple[str, ...]
    exact_postsource_custody_hashes: tuple[str, ...]
    postsource_receipt_hashes: tuple[str, ...]
    construction_custody_kinds: tuple[str, ...]
    construction_custody_receipt_hashes: tuple[str, ...]
    uniform_construction_custody_kind: str
    projection_replay_registration_receipt_hashes: tuple[str | None, ...]
    outside_direction_ids: tuple[str, ...]
    shared_shell_direction_ids: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    preoutcome_joint_direction_axis_hash: str
    postsource_custody_axis_hash: str
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    execution_mode: str
    retained_parent_device_bytes: int
    additional_joint_device_bytes: int = 0
    existing_literal_j_cell_reused: bool = True
    alpha_allocation_performed: bool = False
    complete_unique_physical_action_axis: bool = True
    exactly_one_direction_per_destination: bool = True
    exact_preoutcome_postsource_action_custody: bool = True
    one_shared_shell_basis_family: bool = True
    outside_vectors_exposed_as_tuple: bool = True
    action_route_dense_stack_materialized: bool = False
    dense_shell_basis_bank_materialized: bool = False
    route_scale_d2h: bool = False
    array_free_durable_receipt: bool = True
    live_parent_leases_required: bool = True
    population_projection_authority: bool = False
    fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    outer_hbm_admission_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
                "joint family receipts are issuer-sealed",
            )
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "fanout_receipt_hash",
            "joint_preoutcome_registration_hash",
            "fanout_wrapper_plan_hash",
            "fanout_base_plan_hash",
            "physical_action_axis_hash",
            "fanout_binding_axis_hash",
            "literal_j_owner_binding_receipt_hash",
            "literal_j_cell_id",
            "literal_j_cell_hash",
            "literal_j_cell_scope_hash",
            "literal_j_stream_binding_hash",
            "literal_j_stream_source_hash",
            "same_capture_vjp_source_hash",
            "historical_alpha_ledger_hash",
            "historical_alpha_coordinate_hash",
            "historical_alpha_lineage_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "preoutcome_joint_direction_axis_hash",
            "postsource_custody_axis_hash",
            "uniform_construction_custody_kind",
        ):
            _identity(getattr(self, name), name=name)
        action_count = _positive_int(self.action_count, name="action_count")
        shell_size = _positive_int(self.shell_size, name="shell_size")
        direction_count = _positive_int(self.direction_count, name="direction_count")
        routes = _positive_int(self.route_count, name="route_count")
        _positive_int(self.registered_look_index, name="registered_look_index")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        _nonnegative_int(
            self.retained_parent_device_bytes,
            name="retained_parent_device_bytes",
        )
        _nonnegative_int(
            self.additional_joint_device_bytes,
            name="additional_joint_device_bytes",
        )
        if type(self.shell_indices) is not tuple or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in self.shell_indices
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
                "shell_indices must be one exact integer tuple",
            )
        if (
            len(self.shell_indices) != shell_size
            or not 1 <= shell_size <= 8
            or len(set(self.shell_indices)) != shell_size
            or any(item < 0 or item >= routes for item in self.shell_indices)
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH,
                "joint shell must contain one to eight unique in-axis indices",
            )
        axes = {
            "physical_action_identity_hashes": self.physical_action_identity_hashes,
            "physical_action_hashes": self.physical_action_hashes,
            "packed_action_hashes": self.packed_action_hashes,
            "source_registration_hashes": self.source_registration_hashes,
            "preoutcome_registration_hashes": self.preoutcome_registration_hashes,
            "destination_binding_hashes": self.destination_binding_hashes,
            "direction_receipt_hashes": self.direction_receipt_hashes,
            "exact_postsource_custody_hashes": self.exact_postsource_custody_hashes,
            "postsource_receipt_hashes": self.postsource_receipt_hashes,
            "construction_custody_kinds": self.construction_custody_kinds,
            "construction_custody_receipt_hashes": (
                self.construction_custody_receipt_hashes
            ),
            "outside_direction_ids": self.outside_direction_ids,
        }
        for name, axis in axes.items():
            if len(_identity_tuple(axis, name=name)) != action_count:
                _refuse(
                    V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                    f"{name} does not cover the complete action axis",
                )
        projection_axis = _optional_identity_tuple(
            self.projection_replay_registration_receipt_hashes,
            name="projection_replay_registration_receipt_hashes",
        )
        if len(projection_axis) != action_count:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "legacy projection-replay custody axis does not cover every action",
            )
        try:
            custody_kinds = tuple(
                V1LiteralJAffineShellDirectionConstructionCustodyKind(item)
                for item in self.construction_custody_kinds
            )
            uniform_kind = V1LiteralJAffineShellDirectionConstructionCustodyKind(
                self.uniform_construction_custody_kind
            )
        except (TypeError, ValueError):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                "joint family contains an unsupported construction custody kind",
            )
        if (
            len(custody_kinds) != action_count
            or any(item is not uniform_kind for item in custody_kinds)
            or uniform_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            CALLER_VECTOR_COMPATIBILITY
            or (
                uniform_kind
                is V1LiteralJAffineShellDirectionConstructionCustodyKind.
                LEGACY_PROJECTION_REPLAY
                and any(
                    projection is None or projection != construction
                    for projection, construction in zip(
                        projection_axis,
                        self.construction_custody_receipt_hashes,
                        strict=True,
                    )
                )
            )
            or (
                uniform_kind
                is V1LiteralJAffineShellDirectionConstructionCustodyKind.
                JOINT_POPULATION_SOURCE
                and any(item is not None for item in projection_axis)
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                "one family must use one coherent canonical construction custody kind",
            )
        if any(
            len(set(axis)) != action_count
            for axis in (
                self.physical_action_identity_hashes,
                self.physical_action_hashes,
                self.preoutcome_registration_hashes,
                self.destination_binding_hashes,
                self.direction_receipt_hashes,
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "action-defining parent axes must each be complete and unique",
            )
        if (
            type(self.physical_action_ordinals) is not tuple
            or any(type(item) is not int for item in self.physical_action_ordinals)
            or self.physical_action_ordinals != tuple(range(1, action_count + 1))
            or len(_identity_axis(self.shell_route_ids, name="shell_route_ids"))
            != shell_size
            or len(
                _identity_axis(
                    self.shared_shell_direction_ids,
                    name="shared_shell_direction_ids",
                )
            )
            != shell_size
            or len(_identity_axis(self.joint_direction_ids, name="joint_direction_ids"))
            != direction_count
            or direction_count != action_count + shell_size
            or self.joint_direction_ids
            != self.outside_direction_ids + self.shared_shell_direction_ids
            or self.outside_direction_ids
            != tuple(
                _outside_direction_id(ordinal, action_hash)
                for ordinal, action_hash in zip(
                    self.physical_action_ordinals,
                    self.physical_action_identity_hashes,
                    strict=True,
                )
            )
            or self.shared_shell_direction_ids
            != tuple(_shell_direction_id(item) for item in self.shell_route_ids)
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
                "joint outside-first/shared-shell direction axis does not replay",
            )
        alpha = _finite_alpha(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(
            self.simultaneous_statement_count,
            name="simultaneous_statement_count",
        )
        per_statement = _finite_alpha(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        expected_preoutcome_axis_hash = _preoutcome_joint_axis_hash(
            fanout_receipt_hash=self.fanout_receipt_hash,
            physical_action_axis_hash=self.physical_action_axis_hash,
            fanout_binding_axis_hash=self.fanout_binding_axis_hash,
            physical_action_identity_hashes=self.physical_action_identity_hashes,
            physical_action_hashes=self.physical_action_hashes,
            source_registration_hashes=self.source_registration_hashes,
            preoutcome_registration_hashes=self.preoutcome_registration_hashes,
            destination_binding_hashes=self.destination_binding_hashes,
            joint_direction_ids=self.joint_direction_ids,
            literal_j_cell_hash=self.literal_j_cell_hash,
            literal_j_cell_scope_hash=self.literal_j_cell_scope_hash,
            historical_alpha_ledger_hash=self.historical_alpha_ledger_hash,
            historical_alpha_coordinate_hash=(
                self.historical_alpha_coordinate_hash
            ),
            historical_alpha_lineage_hash=self.historical_alpha_lineage_hash,
            registered_d_prefix_hash=self.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=self.registered_d_w2_prefix_hash,
            route_axis_hash=self.route_axis_hash,
            route_count=routes,
            simultaneous_statement_count=statements,
            familywise_alpha=alpha,
        )
        expected_postsource_axis_hash = content_hash(
            {
                "schema": "literal_J_joint_affine_shell_postsource_custody_axis_v2",
                "joint_preoutcome_registration_hash": (
                    self.joint_preoutcome_registration_hash
                ),
                "preoutcome_joint_direction_axis_hash": (
                    self.preoutcome_joint_direction_axis_hash
                ),
                "direction_receipt_hashes": self.direction_receipt_hashes,
                "exact_postsource_custody_hashes": (
                    self.exact_postsource_custody_hashes
                ),
                "postsource_receipt_hashes": self.postsource_receipt_hashes,
                "construction_custody_kinds": self.construction_custody_kinds,
                "construction_custody_receipt_hashes": (
                    self.construction_custody_receipt_hashes
                ),
                "uniform_construction_custody_kind": (
                    self.uniform_construction_custody_kind
                ),
                "projection_replay_registration_receipt_hashes": (
                    self.projection_replay_registration_receipt_hashes
                ),
            },
            prefix=_PREFIX + "postsource-custody-axis-",
        )
        positive = (
            self.existing_literal_j_cell_reused,
            self.complete_unique_physical_action_axis,
            self.exactly_one_direction_per_destination,
            self.exact_preoutcome_postsource_action_custody,
            self.one_shared_shell_basis_family,
            self.outside_vectors_exposed_as_tuple,
            self.array_free_durable_receipt,
            self.live_parent_leases_required,
        )
        closed = (
            self.alpha_allocation_performed,
            self.action_route_dense_stack_materialized,
            self.dense_shell_basis_bank_materialized,
            self.route_scale_d2h,
            self.population_projection_authority,
            self.fd_replacement_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.outer_hbm_admission_authority,
            self.pareto_authority,
        )
        if (
            statements != 2 * direction_count * (routes + 1)
            or per_statement != _divide_probability_down(alpha, statements)
            or self.preoutcome_joint_direction_axis_hash
            != expected_preoutcome_axis_hash
            or self.postsource_custody_axis_hash != expected_postsource_axis_hash
            or self.additional_joint_device_bytes != 0
            or self.execution_mode
            not in tuple(item.value for item in V1LiteralJJointAffineShellFamilyExecutionMode)
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.AUTHORITY_VIOLATION,
                "joint family crossed its direction, alpha, memory, or authority boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "joint family receipt hash does not replay",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> dict[str, object]:
        self.assert_integrity()
        return self._payload()


class V1LiteralJJointAffineShellFamilyLease:
    __slots__ = ("_owner", "_token", "_closed", "receipt_hash")
    __hash__ = None

    def __init__(
        self,
        owner: "V1LiteralJJointAffineShellFamilyDeviceHandle",
        token: object,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.receipt_hash = owner.receipt.receipt_hash

    def projection_custody(self) -> tuple[tuple[Any, ...], Any]:
        """Return ordered outside vectors and the single shared shell index."""

        if self._closed:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.EVICTED,
                "joint family lease is closed",
            )
        if self._owner.receipt.receipt_hash != self.receipt_hash:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.STALE_IDENTITY,
                "joint family receipt changed while leased",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        self._owner._close_lease(self._token)
        self._closed = True

    def __enter__(self) -> "V1LiteralJJointAffineShellFamilyLease":
        self.projection_custody()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint affine-shell leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("joint affine-shell leases are nonserializable")


class V1LiteralJJointAffineShellFamilyDeviceHandle:
    """Live parent-lease owner; it creates no route-scale tensor."""

    __slots__ = (
        "_receipt",
        "_receipt_hash",
        "_parent_leases",
        "_parent_receipt_hashes",
        "_tokens",
    )

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAffineShellFamilyReceipt,
        parent_leases: tuple[V1LiteralJAffineShellDirectionLease, ...],
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _HANDLE_TOKEN:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
                "joint family handles are issuer-sealed",
            )
        if (
            type(receipt) is not V1LiteralJJointAffineShellFamilyReceipt
            or type(parent_leases) is not tuple
            or len(parent_leases) != receipt.action_count
            or any(type(item) is not V1LiteralJAffineShellDirectionLease for item in parent_leases)
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
                "joint handle requires one exact retained parent lease per action",
            )
        self._receipt = receipt
        self._receipt_hash = receipt.assert_integrity()
        self._parent_leases = parent_leases
        self._parent_receipt_hashes = tuple(item.receipt_hash for item in parent_leases)
        self._tokens: set[object] = set()
        self._require_live()

    def _require_live(self) -> tuple[tuple[Any, ...], Any]:
        self._receipt.assert_integrity(self._receipt_hash)
        if self._parent_leases is None:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.EVICTED,
                "joint family custody was evicted",
            )
        if tuple(item.receipt_hash for item in self._parent_leases) != self._parent_receipt_hashes:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "a retained parent receipt identity changed",
            )
        outside: list[Any] = []
        shared_shell: Any | None = None
        try:
            for lease in self._parent_leases:
                parent_outside, parent_shell = lease.structured_tensors()
                if shared_shell is None:
                    shared_shell = parent_shell
                elif not bool(torch.equal(parent_shell, shared_shell)):
                    _refuse(
                        V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                        "retained parent shell tensors no longer agree",
                    )
                outside.append(parent_outside)
        except V1LiteralJJointAffineShellFamilyRefusal:
            raise
        except Exception as error:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "a retained direction parent failed live replay: "
                f"{type(error).__name__}: {error}",
            )
        if shared_shell is None or len(outside) != self._receipt.action_count:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "joint family lost its complete live action axis",
            )
        return tuple(outside), shared_shell

    @property
    def receipt(self) -> V1LiteralJJointAffineShellFamilyReceipt:
        self._require_live()
        return self._receipt

    @property
    def is_evicted(self) -> bool:
        return self._parent_leases is None

    def borrow(self) -> V1LiteralJJointAffineShellFamilyLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellFamilyLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.STALE_IDENTITY,
                "joint family lease token is stale",
            )
        try:
            self._require_live()
        except Exception:
            self._tokens.remove(token)
            raise
        self._tokens.remove(token)

    def evict(self) -> None:
        replay_error: Exception | None = None
        try:
            self._require_live()
        except V1LiteralJJointAffineShellFamilyRefusal as error:
            if error.code is V1LiteralJJointAffineShellFamilyRefusalCode.EVICTED:
                raise
            replay_error = error
        if self._tokens:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.ACTIVE_LEASE,
                "cannot evict joint family custody while leased",
            )
        leases = self._parent_leases
        close_error: Exception | None = None
        if leases is not None:
            for lease in leases:
                try:
                    lease.close()
                except Exception as error:  # release every remaining parent.
                    if close_error is None:
                        close_error = error
        self._parent_leases = None
        self._parent_receipt_hashes = ()
        if replay_error is not None:
            raise replay_error
        if close_error is not None:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                "a mutated parent was released during joint eviction: "
                f"{type(close_error).__name__}: {close_error}",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint affine-shell handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("joint affine-shell handles are nonserializable")


def _close_parent_leases(
    leases: list[V1LiteralJAffineShellDirectionLease],
) -> None:
    for lease in leases:
        try:
            lease.close()
        except Exception:
            # Constructor cleanup has no authority to hide the triggering
            # refusal, but must attempt every release.
            pass


def _construct_joint_receipt_with_cleanup(
    *,
    leases: list[V1LiteralJAffineShellDirectionLease],
    **coordinates: object,
) -> V1LiteralJJointAffineShellFamilyReceipt:
    try:
        return V1LiteralJJointAffineShellFamilyReceipt(**coordinates)  # type: ignore[arg-type]
    except Exception:
        _close_parent_leases(leases)
        raise


def issue_v1_literal_j_joint_affine_shell_family_cuda(
    *,
    fanout_receipt: V1SampledV7AffineShellPreoutcomeFanoutReceipt,
    direction_handles: tuple[V1LiteralJAffineShellDirectionDeviceHandle, ...],
    execution_mode: V1LiteralJJointAffineShellFamilyExecutionMode,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> V1LiteralJJointAffineShellFamilyDeviceHandle:
    """Join the complete action axis under its one already-spent J cell."""

    if (
        torch is None
        or type(fanout_receipt) is not V1SampledV7AffineShellPreoutcomeFanoutReceipt
        or type(joint_preoutcome_registration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
        or type(direction_handles) is not tuple
        or type(execution_mode) is not V1LiteralJJointAffineShellFamilyExecutionMode
        or any(
            type(item) is not V1LiteralJAffineShellDirectionDeviceHandle
            for item in direction_handles
        )
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.INVALID_TYPE,
            "joint issuance requires the exact preoutcome registration, fanout "
            "receipt, handle tuple, and mode",
        )
    try:
        fanout_hash = fanout_receipt.assert_integrity()
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "the durable complete-action fanout receipt failed replay: "
            f"{type(error).__name__}: {error}",
        )
    try:
        provided_preoutcome_hash = joint_preoutcome_registration.assert_integrity()
        expected_preoutcome_coordinates = (
            _derive_v1_literal_j_joint_affine_shell_preoutcome_coordinates(
            fanout_receipt=fanout_receipt,
            )
        )
    except V1LiteralJJointAffineShellFamilyRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "joint preoutcome registration failed replay: "
            f"{type(error).__name__}: {error}",
        )
    if any(
        getattr(joint_preoutcome_registration, name) != value
        for name, value in expected_preoutcome_coordinates.items()
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "joint preoutcome registration is not the deterministic family of "
            "this exact fanout receipt",
        )
    action_count = fanout_receipt.destination_count
    if len(direction_handles) != action_count:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "direction tuple must contain exactly one member per fanout destination",
        )
    registrations = fanout_receipt.preoutcome_registrations
    bindings = fanout_receipt.destination_bindings
    if (
        type(registrations) is not tuple
        or type(bindings) is not tuple
        or len(registrations) != action_count
        or len(bindings) != action_count
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "fanout receipt lost its complete registration/binding axes",
        )
    action_identity_axis = tuple(item.physical_action_identity_hash for item in bindings)
    action_hash_axis = tuple(item.physical_action_hash for item in bindings)
    if (
        len(set(action_identity_axis)) != action_count
        or len(set(action_hash_axis)) != action_count
    ):
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.DESTINATION_AXIS_MISMATCH,
            "the physical-action axis must be complete and unique",
        )
    first = registrations[0]
    shared_coordinate_fields = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "route_count",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "shell_freeze_receipt_hash",
        "shell_indices",
        "shell_route_ids",
        "literal_j_stream_source_hash",
        "same_capture_vjp_source_hash",
        "literal_j_stream_binding_hash",
        "literal_j_owner_binding_receipt_hash",
        "literal_j_cell_id",
        "literal_j_cell_hash",
        "literal_j_cell_scope_hash",
        "literal_j_cell_alpha",
        "historical_alpha_ledger_hash",
        "historical_alpha_coordinate_hash",
        "historical_alpha_lineage_hash",
        "registered_look_index",
    )
    for registration in registrations:
        try:
            registration.assert_integrity()
        except Exception as error:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                "one preoutcome action registration failed replay: "
                f"{type(error).__name__}: {error}",
            )
        mismatches = tuple(
            name
            for name in shared_coordinate_fields
            if getattr(registration, name) != getattr(first, name)
        )
        if mismatches:
            if any("literal_j" in name for name in mismatches):
                code = V1LiteralJJointAffineShellFamilyRefusalCode.J_CELL_MISMATCH
            elif any("prefix" in name for name in mismatches):
                code = V1LiteralJJointAffineShellFamilyRefusalCode.PREFIX_MISMATCH
            elif any("shell" in name for name in mismatches):
                code = V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH
            else:
                code = V1LiteralJJointAffineShellFamilyRefusalCode.CHART_MISMATCH
            _refuse(
                code,
                "action registrations do not share one chart/J/shell family: "
                + ", ".join(mismatches),
            )
    expected_parent_mode = (
        V1LiteralJAffineShellDirectionExecutionMode.CUDA_PRODUCTION.value
        if execution_mode
        is V1LiteralJJointAffineShellFamilyExecutionMode.CUDA_PRODUCTION
        else V1LiteralJAffineShellDirectionExecutionMode.CPU_REFERENCE_TEST_ONLY.value
    )
    parent_receipts = []
    for ordinal, (binding, registration, handle) in enumerate(
        zip(bindings, registrations, direction_handles, strict=True),
        start=1,
    ):
        try:
            receipt = handle.receipt
            receipt.assert_integrity()
        except Exception as error:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                f"direction {ordinal} failed live replay: {type(error).__name__}: {error}",
            )
        if (
            receipt.exact_preoutcome_postsource_custody_bound is not True
            or receipt.outside_direction_exact_action_custody is not True
            or receipt.preoutcome_source_registration_bound is not True
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.UNBOUND_DIRECTION,
                f"direction {ordinal} is a compatibility direction without canonical custody",
            )
        try:
            custody_kind = V1LiteralJAffineShellDirectionConstructionCustodyKind(
                receipt.construction_custody_kind
            )
        except (TypeError, ValueError):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                f"direction {ordinal} has an unsupported construction custody kind",
            )
        if (
            custody_kind
            is V1LiteralJAffineShellDirectionConstructionCustodyKind.
            CALLER_VECTOR_COMPATIBILITY
            or receipt.construction_custody_receipt_hash is None
            or (
                custody_kind
                is V1LiteralJAffineShellDirectionConstructionCustodyKind.
                LEGACY_PROJECTION_REPLAY
                and receipt.projection_replay_registration_receipt_hash
                != receipt.construction_custody_receipt_hash
            )
            or (
                custody_kind
                is V1LiteralJAffineShellDirectionConstructionCustodyKind.
                JOINT_POPULATION_SOURCE
                and receipt.projection_replay_registration_receipt_hash is not None
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                f"direction {ordinal} has incoherent construction custody",
            )
        if receipt.execution_mode != expected_parent_mode:
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.EXECUTION_MODE_MISMATCH,
                f"direction {ordinal} does not use the registered joint execution mode",
            )
        if (
            receipt.preoutcome_fanout_receipt_hash != fanout_hash
            or receipt.preoutcome_binding_axis_hash != fanout_receipt.binding_axis_hash
            or receipt.preoutcome_physical_action_axis_hash
            != fanout_receipt.physical_action_axis_hash
            or receipt.preoutcome_destination_binding_hash != binding.binding_hash
            or receipt.preoutcome_source_registration_hash
            != registration.registration_hash
            or receipt.source_registration_hash != registration.source_registration_hash
            or receipt.physical_action_hash != registration.physical_action_hash
            or receipt.packed_action_hash != registration.packed_action_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
                f"direction {ordinal} is not the corresponding fanout action member",
            )
        if (
            receipt.literal_j_cell_hash != registration.literal_j_cell_hash
            or receipt.literal_j_cell_scope_hash != registration.literal_j_cell_scope_hash
            or receipt.literal_j_stream_binding_hash
            != registration.literal_j_stream_binding_hash
            or receipt.literal_j_stream_source_hash
            != registration.literal_j_stream_source_hash
            or receipt.same_capture_vjp_source_hash
            != registration.same_capture_vjp_source_hash
            or receipt.alpha_lineage_hash != registration.historical_alpha_lineage_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.J_CELL_MISMATCH,
                f"direction {ordinal} left the shared literal-J cell",
            )
        if (
            receipt.model_identity_hash != registration.model_identity_hash
            or receipt.chart_identity_hash != registration.chart_identity_hash
            or receipt.current_mask_hash != registration.current_mask_hash
            or receipt.proposal_generation != registration.proposal_generation
            or receipt.route_axis_hash != registration.route_axis_hash
            or receipt.route_id_axis_hash != registration.route_id_axis_hash
            or receipt.route_count != registration.route_count
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.CHART_MISMATCH,
                f"direction {ordinal} left the registered model/chart/route axis",
            )
        if (
            receipt.registered_d_prefix_hash != registration.registered_d_prefix_hash
            or receipt.registered_d_w2_prefix_hash
            != registration.registered_d_w2_prefix_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.PREFIX_MISMATCH,
                f"direction {ordinal} left the registered D/D_W2 prefix",
            )
        if (
            receipt.shell_freeze_receipt_hash
            != registration.shell_freeze_receipt_hash
            or receipt.shell_indices != registration.shell_indices
            or receipt.shell_route_ids != registration.shell_route_ids
        ):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH,
                f"direction {ordinal} left the common frozen shell",
            )
        parent_receipts.append(receipt)
    construction_kinds = tuple(
        item.construction_custody_kind for item in parent_receipts
    )
    if len(set(construction_kinds)) != 1:
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.SOURCE_CUSTODY_MISMATCH,
            "one joint family cannot mix legacy projection and current joint-population construction custody",
        )
    uniform_construction_kind = construction_kinds[0]
    leases: list[V1LiteralJAffineShellDirectionLease] = []
    try:
        for handle in direction_handles:
            lease = handle.borrow()
            leases.append(lease)
        outside_vectors: list[Any] = []
        shared_shell: Any | None = None
        shared_device: Any | None = None
        for ordinal, lease in enumerate(leases, start=1):
            outside, shell = lease.structured_tensors()
            if (
                tuple(outside.shape) != (first.route_count,)
                or outside.dtype != torch.float64
                or not outside.is_contiguous()
                or bool(outside.requires_grad)
            ):
                _refuse(
                    V1LiteralJJointAffineShellFamilyRefusalCode.MUTATED_CUSTODY,
                    f"direction {ordinal} is no longer one canonical route vector",
                )
            if outside.device != shell.device or (
                shared_device is not None and outside.device != shared_device
            ):
                _refuse(
                    V1LiteralJJointAffineShellFamilyRefusalCode.DEVICE_REQUIRED,
                    "every action outside vector and the shared shell must use "
                    "one exact device ordinal",
                )
            if shared_device is None:
                shared_device = outside.device
            if shared_shell is None:
                shared_shell = shell
            elif not bool(torch.equal(shell, shared_shell)):
                _refuse(
                    V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH,
                    "direction parents do not expose the same shell index vector",
                )
            outside_vectors.append(outside)
        if shared_shell is None or tuple(shared_shell.shape) != (len(first.shell_indices),):
            _refuse(
                V1LiteralJJointAffineShellFamilyRefusalCode.SHELL_MISMATCH,
                "the shared shell tensor has the wrong bounded shape",
            )
        _require_exact_common_device(
            (*outside_vectors, shared_shell),
            expected_type=(
                "cuda"
                if execution_mode
                is V1LiteralJJointAffineShellFamilyExecutionMode.CUDA_PRODUCTION
                else "cpu"
            ),
        )
    except Exception:
        _close_parent_leases(leases)
        raise
    outside_ids = tuple(
        _outside_direction_id(ordinal, identity)
        for ordinal, identity in enumerate(action_identity_axis, start=1)
    )
    shell_ids = tuple(_shell_direction_id(item) for item in first.shell_route_ids)
    joint_ids = outside_ids + shell_ids
    direction_receipt_hashes = tuple(item.receipt_hash for item in parent_receipts)
    postsource_custody_axis_hash = content_hash(
        {
            "schema": "literal_J_joint_affine_shell_postsource_custody_axis_v2",
            "joint_preoutcome_registration_hash": provided_preoutcome_hash,
            "preoutcome_joint_direction_axis_hash": (
                joint_preoutcome_registration.joint_direction_axis_hash
            ),
            "direction_receipt_hashes": direction_receipt_hashes,
            "exact_postsource_custody_hashes": tuple(
                item.exact_postsource_custody_hash for item in parent_receipts
            ),
            "postsource_receipt_hashes": tuple(
                item.postsource_receipt_hash for item in parent_receipts
            ),
            "construction_custody_kinds": construction_kinds,
            "construction_custody_receipt_hashes": tuple(
                item.construction_custody_receipt_hash for item in parent_receipts
            ),
            "uniform_construction_custody_kind": uniform_construction_kind,
            "projection_replay_registration_receipt_hashes": tuple(
                item.projection_replay_registration_receipt_hash
                for item in parent_receipts
            ),
        },
        prefix=_PREFIX + "postsource-custody-axis-",
    )
    direction_count = action_count + len(first.shell_indices)
    statements = 2 * direction_count * (first.route_count + 1)
    if (
        joint_ids != joint_preoutcome_registration.joint_direction_ids
        or direction_count != joint_preoutcome_registration.direction_count
        or statements
        != joint_preoutcome_registration.simultaneous_statement_count
    ):
        _close_parent_leases(leases)
        _refuse(
            V1LiteralJJointAffineShellFamilyRefusalCode.PREOUTCOME_REGISTRATION_MISMATCH,
            "postsource custody attempted to redefine the frozen statistical axis",
        )
    receipt = _construct_joint_receipt_with_cleanup(
        leases=leases,
        fanout_receipt_hash=fanout_hash,
        joint_preoutcome_registration_hash=provided_preoutcome_hash,
        fanout_wrapper_plan_hash=fanout_receipt.wrapper_plan_hash,
        fanout_base_plan_hash=fanout_receipt.base_plan_hash,
        physical_action_axis_hash=fanout_receipt.physical_action_axis_hash,
        fanout_binding_axis_hash=fanout_receipt.binding_axis_hash,
        literal_j_owner_binding_receipt_hash=(
            fanout_receipt.literal_j_owner_binding_receipt_hash
        ),
        literal_j_cell_id=first.literal_j_cell_id,
        literal_j_cell_hash=first.literal_j_cell_hash,
        literal_j_cell_scope_hash=first.literal_j_cell_scope_hash,
        literal_j_stream_binding_hash=first.literal_j_stream_binding_hash,
        literal_j_stream_source_hash=first.literal_j_stream_source_hash,
        same_capture_vjp_source_hash=first.same_capture_vjp_source_hash,
        historical_alpha_ledger_hash=first.historical_alpha_ledger_hash,
        historical_alpha_coordinate_hash=first.historical_alpha_coordinate_hash,
        historical_alpha_lineage_hash=first.historical_alpha_lineage_hash,
        registered_look_index=first.registered_look_index,
        model_identity_hash=first.model_identity_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.current_mask_hash,
        proposal_generation=first.proposal_generation,
        route_axis_hash=first.route_axis_hash,
        route_id_axis_hash=first.route_id_axis_hash,
        route_count=first.route_count,
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=first.shell_freeze_receipt_hash,
        shell_indices=first.shell_indices,
        shell_route_ids=first.shell_route_ids,
        action_count=action_count,
        shell_size=len(first.shell_indices),
        direction_count=direction_count,
        physical_action_ordinals=tuple(range(1, action_count + 1)),
        physical_action_identity_hashes=action_identity_axis,
        physical_action_hashes=action_hash_axis,
        packed_action_hashes=tuple(item.packed_action_hash for item in registrations),
        source_registration_hashes=tuple(
            item.source_registration_hash for item in registrations
        ),
        preoutcome_registration_hashes=tuple(
            item.registration_hash for item in registrations
        ),
        destination_binding_hashes=tuple(item.binding_hash for item in bindings),
        direction_receipt_hashes=direction_receipt_hashes,
        exact_postsource_custody_hashes=tuple(
            item.exact_postsource_custody_hash for item in parent_receipts
        ),  # type: ignore[arg-type]
        postsource_receipt_hashes=tuple(
            item.postsource_receipt_hash for item in parent_receipts
        ),  # type: ignore[arg-type]
        construction_custody_kinds=construction_kinds,
        construction_custody_receipt_hashes=tuple(
            item.construction_custody_receipt_hash for item in parent_receipts
        ),  # type: ignore[arg-type]
        uniform_construction_custody_kind=uniform_construction_kind,
        projection_replay_registration_receipt_hashes=tuple(
            item.projection_replay_registration_receipt_hash
            for item in parent_receipts
        ),  # type: ignore[arg-type]
        outside_direction_ids=outside_ids,
        shared_shell_direction_ids=shell_ids,
        joint_direction_ids=joint_ids,
        preoutcome_joint_direction_axis_hash=(
            joint_preoutcome_registration.joint_direction_axis_hash
        ),
        postsource_custody_axis_hash=postsource_custody_axis_hash,
        familywise_alpha=first.literal_j_cell_alpha,
        simultaneous_statement_count=statements,
        per_statement_two_sided_alpha=_divide_probability_down(
            first.literal_j_cell_alpha,
            statements,
        ),
        execution_mode=execution_mode.value,
        retained_parent_device_bytes=sum(
            item.retained_device_bytes for item in parent_receipts
        ),
        _issuance_token=_RECEIPT_TOKEN,
    )
    try:
        return V1LiteralJJointAffineShellFamilyDeviceHandle(
            receipt=receipt,
            parent_leases=tuple(leases),
            issuance_token=_HANDLE_TOKEN,
        )
    except Exception:
        _close_parent_leases(leases)
        raise


__all__ = [
    "LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_PREOUTCOME_REGISTRATION_AUTHORITY",
    "LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_AUTHORITY",
    "LITERAL_J_JOINT_AFFINE_SHELL_FAMILY_ALGORITHM",
    "V1LiteralJJointAffineShellFamilyExecutionMode",
    "V1LiteralJJointAffineShellFamilyRefusalCode",
    "V1LiteralJJointAffineShellFamilyRefusal",
    "V1LiteralJJointAffineShellPreoutcomeRegistration",
    "V1LiteralJJointAffineShellFamilyReceipt",
    "V1LiteralJJointAffineShellFamilyLease",
    "V1LiteralJJointAffineShellFamilyDeviceHandle",
    "register_v1_literal_j_joint_affine_shell_preoutcome",
    "issue_v1_literal_j_joint_affine_shell_family_cuda",
]
