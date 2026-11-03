"""Continuum-robust enclosure for the signed-work CUDA D witness.

The CUDA decision witness proves a finite-grid statement: every registered
signed-F x literal-J x FD x solver vertex produced the same resource-cell and
byte-deduplicated physical-action image.  That is necessary, but it is not yet
the continuum statement needed by canonical v1.  The missing question is
whether the source-derived uncertainty sets around those vertices can move a
resource cell, relaxed face/support, signed-F winner, hard-rounding key, or
physical action identity.

This module is that fail-closed promotion layer.  It consumes compact
source-derived owner receipts and compact decision-margin receipts only.  It
never receives route-scale arrays, never materializes host route tensors, and
it grants no C, K, O, commit, terminal, performance, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Mapping, NoReturn

from .contracts import content_hash
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
    V1SignedWorkSamplingUncertaintyRole,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA,
    V1SignedWorkCUDADecisionWitnessQualification,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceReceipt,
)


SIGNED_WORK_D_W_CONTINUUM_OWNER_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_continuum_owner_uncertainty_v1"
)
SIGNED_WORK_D_W_CONTINUUM_MARGIN_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_continuum_decision_margin_v1"
)
SIGNED_WORK_D_W_CONTINUUM_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_continuum_robustness_certificate_v1"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_complete_action_image_continuum_closure_v2"
)
SIGNED_WORK_D_W_CONTINUUM_AUTHORITY_SCOPE: Final = (
    "source_derived_continuum_D_W_robustness_only_no_C_K_O_commit_terminal_"
    "performance_or_pareto_authority_v1"
)
SIGNED_WORK_D_W_CONTINUUM_RULE: Final = (
    "finite_cuda_D_W_grid_complete_plus_all_four_owner_uncertainty_receipts_"
    "and_all_phase_A_phase_B_rounding_and_physical_action_margins_enclosed_v1"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE: Final = (
    "complete_closed_top_k_action_cells_cover_the_relaxed_order_continuum_with_"
    "ties_in_every_compatible_cell_then_aggregate_source_group_corrected_F_"
    "bounds_select_one_action_by_Gamma_lower_equals_min_alternate_L_minus_max_"
    "preferred_U_and_exact_packed_bytes_form_a_categorical_action_quotient_v2"
)

_PREFIX: Final = "partition-free-v1-signed-work-D-W-continuum-"
_CERTIFICATE_TOKEN: Final = object()


class V1SignedWorkDecisionOwnerRole(str, Enum):
    """The four uncertainty owners whose product defines D_W."""

    SIGNED_F_SAMPLING = "signed_F_sampling"
    LITERAL_J_SAMPLING = "literal_J_sampling"
    FD_RESIDUAL = "deterministic_FD_residual"
    SOLVER_ROUNDING = "solver_rounding_numerics"


class V1SignedWorkDecisionMarginRole(str, Enum):
    """Continuum-sensitive decisions made by Phase A/B before C opens."""

    PHASE_A_RESOURCE_CELL = "phase_A_resource_cell"
    ALIAS_SAFE_BRANCH_SUPPORT_FACE = "alias_safe_branch_support_face"
    PHASE_B_SIGNED_F_WINNER = "phase_B_signed_F_winner"
    HARD_ROUNDING_KEY = "hard_rounding_key"
    BYTE_DEDUP_PHYSICAL_ACTION = "byte_dedup_physical_action"


class V1SignedWorkDecisionContinuumClosureMode(str, Enum):
    """Versioned interpretation of the pre-C continuum closure.

    ``LEGACY_LOCAL_SENSITIVITY`` keeps the historical five-positive-margin
    reader intact.  ``COMPLETE_ACTION_IMAGE`` is the canonical mode: order
    boundaries are covered as closed action cells and byte equality is an
    exact categorical quotient, so neither is represented by an invented
    positive scalar distance.
    """

    LEGACY_LOCAL_SENSITIVITY = "legacy_local_sensitivity_five_margin_v1"
    COMPLETE_ACTION_IMAGE = "complete_action_image_categorical_quotient_v2"


class V1SignedWorkDecisionBoundaryClosureKind(str, Enum):
    """Mathematical object that closes one legacy decision-boundary role."""

    OUTWARD_POSITIVE_SCALAR = "outward_positive_scalar_margin"
    AGGREGATE_ACTION_SEPARATION = "aggregate_action_cell_separation_Gamma_lower"
    COMPLETE_CLOSED_ACTION_CELL_COVER = "complete_closed_top_k_action_cell_cover"
    EXACT_CATEGORICAL_PHYSICAL_ACTION_QUOTIENT = (
        "exact_categorical_packed_physical_action_quotient"
    )


class V1SignedWorkDecisionEnclosureRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    FINITE_GRID_INCOMPLETE = "finite_grid_incomplete"
    FINITE_GRID_ONLY = "finite_grid_only"
    MISSING_OWNER_RECEIPT = "missing_owner_receipt"
    MISSING_MARGIN_RECEIPT = "missing_margin_receipt"
    STALE_IDENTITY = "stale_identity"
    UNENCLOSED_OWNER = "unclosed_owner_uncertainty"
    UNENCLOSED_MARGIN = "unclosed_decision_margin"
    NONROBUST_MARGIN = "nonrobust_decision_margin"
    BOUNDEDNESS_VIOLATION = "boundedness_violation"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkDecisionEnclosureRefusal(RuntimeError):
    """Fail-closed result for continuum D_W promotion attempts."""

    def __init__(
        self,
        code: V1SignedWorkDecisionEnclosureRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkDecisionEnclosureRefusalCode:
            raise TypeError("decision-enclosure refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkDecisionEnclosureRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkDecisionEnclosureRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty and unique",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            f"{name} must be one exact bool",
        )
    return value


def _next_down(value: float) -> float:
    result = math.nextafter(float(value), -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            "outward lower bound left finite binary64",
        )
    return result


def _false_downstream_authority(value: object) -> None:
    for name in (
        "generic_C_stream_open_authority",
        "production_C_stream_open_authority",
        "C_measurement_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "performance_authority",
        "pareto_authority",
    ):
        if bool(getattr(value, name, False)):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.AUTHORITY_VIOLATION,
                f"finite CUDA qualification already opened downstream authority: {name}",
            )


def _qualification_hash(value: object) -> str:
    return _identity(
        getattr(value, "qualification_hash", None), name="qualification hash"
    )


_REQUIRED_OWNER_ROLES: Final = (
    V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
    V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
    V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
    V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
)
_REQUIRED_MARGIN_ROLES: Final = (
    V1SignedWorkDecisionMarginRole.PHASE_A_RESOURCE_CELL,
    V1SignedWorkDecisionMarginRole.ALIAS_SAFE_BRANCH_SUPPORT_FACE,
    V1SignedWorkDecisionMarginRole.PHASE_B_SIGNED_F_WINNER,
    V1SignedWorkDecisionMarginRole.HARD_ROUNDING_KEY,
    V1SignedWorkDecisionMarginRole.BYTE_DEDUP_PHYSICAL_ACTION,
)
_MARGIN_OWNER_ROLES: Final = {
    V1SignedWorkDecisionMarginRole.PHASE_A_RESOURCE_CELL: (
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
    ),
    V1SignedWorkDecisionMarginRole.ALIAS_SAFE_BRANCH_SUPPORT_FACE: (
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
    ),
    V1SignedWorkDecisionMarginRole.PHASE_B_SIGNED_F_WINNER: _REQUIRED_OWNER_ROLES,
    V1SignedWorkDecisionMarginRole.HARD_ROUNDING_KEY: _REQUIRED_OWNER_ROLES,
    V1SignedWorkDecisionMarginRole.BYTE_DEDUP_PHYSICAL_ACTION: _REQUIRED_OWNER_ROLES,
}

_COMPLETE_ACTION_IMAGE_BOUNDARY_KINDS: Final = {
    V1SignedWorkDecisionMarginRole.PHASE_A_RESOURCE_CELL: (
        V1SignedWorkDecisionBoundaryClosureKind.OUTWARD_POSITIVE_SCALAR
    ),
    V1SignedWorkDecisionMarginRole.ALIAS_SAFE_BRANCH_SUPPORT_FACE: (
        V1SignedWorkDecisionBoundaryClosureKind.OUTWARD_POSITIVE_SCALAR
    ),
    V1SignedWorkDecisionMarginRole.PHASE_B_SIGNED_F_WINNER: (
        V1SignedWorkDecisionBoundaryClosureKind.AGGREGATE_ACTION_SEPARATION
    ),
    V1SignedWorkDecisionMarginRole.HARD_ROUNDING_KEY: (
        V1SignedWorkDecisionBoundaryClosureKind.COMPLETE_CLOSED_ACTION_CELL_COVER
    ),
    V1SignedWorkDecisionMarginRole.BYTE_DEDUP_PHYSICAL_ACTION: (
        V1SignedWorkDecisionBoundaryClosureKind.EXACT_CATEGORICAL_PHYSICAL_ACTION_QUOTIENT
    ),
}


def v1_signed_work_decision_margin_owner_roles(
    margin_role: V1SignedWorkDecisionMarginRole,
) -> tuple[V1SignedWorkDecisionOwnerRole, ...]:
    """Return the canonical owner family that can move one decision margin."""

    if type(margin_role) is not V1SignedWorkDecisionMarginRole:
        raise TypeError("margin role must be exact")
    return _MARGIN_OWNER_ROLES[margin_role]


def v1_signed_work_decision_boundary_closure_kind(
    margin_role: V1SignedWorkDecisionMarginRole,
    *,
    closure_mode: V1SignedWorkDecisionContinuumClosureMode,
) -> V1SignedWorkDecisionBoundaryClosureKind:
    """Return the versioned mathematical closure for a legacy boundary name.

    The legacy schema remains a five-scalar reader.  Canonical v2 deliberately
    changes the *kind* of evidence for order and byte identity; it does not
    reinterpret an old caller-authored scalar as categorical evidence.
    """

    if type(margin_role) is not V1SignedWorkDecisionMarginRole:
        raise TypeError("margin role must be exact")
    if type(closure_mode) is not V1SignedWorkDecisionContinuumClosureMode:
        raise TypeError("continuum closure mode must be exact")
    if (
        closure_mode
        is V1SignedWorkDecisionContinuumClosureMode.LEGACY_LOCAL_SENSITIVITY
    ):
        return V1SignedWorkDecisionBoundaryClosureKind.OUTWARD_POSITIVE_SCALAR
    return _COMPLETE_ACTION_IMAGE_BOUNDARY_KINDS[margin_role]


def _expected_owner_hashes(
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    qualification: object,
) -> dict[V1SignedWorkDecisionOwnerRole, str]:
    qualification_hashes = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: _identity(
            getattr(qualification, "signed_f_uncertainty_set_hash", None),
            name="finite qualification signed-F owner hash",
        ),
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: _identity(
            getattr(qualification, "literal_j_uncertainty_set_hash", None),
            name="finite qualification literal-J owner hash",
        ),
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: _identity(
            getattr(qualification, "fd_residual_uncertainty_set_hash", None),
            name="finite qualification FD owner hash",
        ),
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: _identity(
            getattr(qualification, "solver_uncertainty_set_hash", None),
            name="finite qualification solver owner hash",
        ),
    }
    uncertainty_hashes = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: (
            uncertainty.signed_f_sampling.set_hash
        ),
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: (
            uncertainty.literal_j_sampling.set_hash
        ),
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: (uncertainty.fd_residual.set_hash),
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: (
            uncertainty.solver_rounding.set_hash
        ),
    }
    for role in _REQUIRED_OWNER_ROLES:
        if qualification_hashes[role] != uncertainty_hashes[role]:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "finite CUDA qualification owner hash is stale",
            )
    return qualification_hashes


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionOwnerUncertaintyReceipt:
    """Compact source-derived enclosure for one D_W uncertainty owner."""

    owner_role: V1SignedWorkDecisionOwnerRole
    uncertainty_set_hash: str
    owner_set_hash: str
    registered_d_w_prefix_hash: str
    route_axis_hash: str
    realization_hashes: tuple[str, ...]
    source_derivation_hash: str
    outward_radius_upper_bound: float
    source_derived_from_registered_owner: bool = True
    finite_grid_covered: bool = True
    continuum_enclosure_provided: bool = True
    statistical_uncertainty: bool = False
    deterministic_uncertainty: bool = False
    spends_alpha: bool = False
    exact_finite_population_census: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    schema: str = SIGNED_WORK_D_W_CONTINUUM_OWNER_SCHEMA

    def __post_init__(self) -> None:
        if type(self.owner_role) is not V1SignedWorkDecisionOwnerRole:
            raise TypeError("owner role must be exact")
        for name in (
            "uncertainty_set_hash",
            "owner_set_hash",
            "registered_d_w_prefix_hash",
            "route_axis_hash",
            "source_derivation_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "realization_hashes",
            _identity_axis(self.realization_hashes, name="owner realizations"),
        )
        object.__setattr__(
            self,
            "outward_radius_upper_bound",
            _nonnegative(
                self.outward_radius_upper_bound,
                name="owner uncertainty radius",
            ),
        )
        flags = (
            _exact_bool(
                self.source_derived_from_registered_owner,
                name="source-derived owner",
            ),
            _exact_bool(self.finite_grid_covered, name="finite grid covered"),
            _exact_bool(
                self.continuum_enclosure_provided,
                name="continuum enclosure provided",
            ),
        )
        if flags != (True, True, True):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
                "owner uncertainty is not source-derived and continuum-enclosed",
            )
        sampling_role = self.owner_role in {
            V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
            V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
        }
        exact_census = _exact_bool(
            self.exact_finite_population_census,
            name="exact finite-population census",
        )
        if exact_census and not sampling_role:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
                "only sampling owners may be marked as exact finite-population census",
            )
        expected = (
            (False, False, False)
            if exact_census
            else (sampling_role, not sampling_role, sampling_role)
        )
        observed = (
            _exact_bool(self.statistical_uncertainty, name="statistical uncertainty"),
            _exact_bool(
                self.deterministic_uncertainty,
                name="deterministic uncertainty",
            ),
            _exact_bool(self.spends_alpha, name="spends alpha"),
        )
        if observed != expected:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
                "owner receipt changed statistical/deterministic ownership",
            )
        if exact_census and self.outward_radius_upper_bound != 0.0:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
                "an exact finite-population census owner must have zero uncertainty radius",
            )
        d2h = _nonnegative_int(self.route_scale_d2h_bytes, name="route-scale D2H bytes")
        host = _nonnegative_int(
            self.route_scale_host_tensor_materializations,
            name="route-scale host tensor materializations",
        )
        if d2h != 0 or host != 0:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.BOUNDEDNESS_VIOLATION,
                "owner enclosure crossed the route-scale host-materialization seam",
            )
        if self.schema != SIGNED_WORK_D_W_CONTINUUM_OWNER_SCHEMA:
            raise ValueError("unsupported D_W continuum owner schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "owner-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionMarginReceipt:
    """Outward finite margin for one continuum-sensitive D_W decision."""

    margin_role: V1SignedWorkDecisionMarginRole
    uncertainty_set_hash: str
    finite_cuda_qualification_hash: str
    registered_d_w_prefix_hash: str
    route_axis_hash: str
    parent_mask_hash: str
    registered_d_coordinate_hash: str
    owner_roles: tuple[V1SignedWorkDecisionOwnerRole, ...]
    source_boundary_hash: str
    source_derivation_hash: str
    outward_enclosure_policy_hash: str
    margin_lower_bound: float
    margin_upper_bound: float
    owner_receipt_hashes: tuple[str, ...] = ()
    owner_perturbation_upper_bound: float = 0.0
    numerical_perturbation_upper_bound: float = 0.0
    source_derived_from_registered_receipts: bool = True
    outward_finite_bounds: bool = True
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    schema: str = SIGNED_WORK_D_W_CONTINUUM_MARGIN_SCHEMA

    def __post_init__(self) -> None:
        if type(self.margin_role) is not V1SignedWorkDecisionMarginRole:
            raise TypeError("margin role must be exact")
        for name in (
            "uncertainty_set_hash",
            "finite_cuda_qualification_hash",
            "registered_d_w_prefix_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "registered_d_coordinate_hash",
            "source_boundary_hash",
            "source_derivation_hash",
            "outward_enclosure_policy_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        roles = tuple(self.owner_roles)
        if (
            not roles
            or any(type(role) is not V1SignedWorkDecisionOwnerRole for role in roles)
            or len(roles) != len(set(roles))
        ):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
                "margin owner roles must be unique exact roles",
            )
        expected_roles = _MARGIN_OWNER_ROLES[self.margin_role]
        if roles != expected_roles:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "margin receipt is not bound to its canonical owner-role family",
            )
        owner_receipts = tuple(self.owner_receipt_hashes)
        if owner_receipts:
            object.__setattr__(
                self,
                "owner_receipt_hashes",
                _identity_axis(owner_receipts, name="margin owner receipt hashes"),
            )
        lower = _finite(self.margin_lower_bound, name="margin lower bound")
        upper = _finite(self.margin_upper_bound, name="margin upper bound")
        if lower > upper:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
                "margin interval is reversed",
            )
        object.__setattr__(self, "margin_lower_bound", lower)
        object.__setattr__(self, "margin_upper_bound", upper)
        object.__setattr__(
            self,
            "owner_perturbation_upper_bound",
            _nonnegative(
                self.owner_perturbation_upper_bound,
                name="owner perturbation upper bound",
            ),
        )
        object.__setattr__(
            self,
            "numerical_perturbation_upper_bound",
            _nonnegative(
                self.numerical_perturbation_upper_bound,
                name="numerical perturbation upper bound",
            ),
        )
        flags = (
            _exact_bool(
                self.source_derived_from_registered_receipts,
                name="source-derived margin",
            ),
            _exact_bool(self.outward_finite_bounds, name="outward finite bounds"),
        )
        if flags != (True, True):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_MARGIN,
                "margin is not source-derived with outward finite bounds",
            )
        d2h = _nonnegative_int(self.route_scale_d2h_bytes, name="route-scale D2H bytes")
        host = _nonnegative_int(
            self.route_scale_host_tensor_materializations,
            name="route-scale host tensor materializations",
        )
        if d2h != 0 or host != 0:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.BOUNDEDNESS_VIOLATION,
                "margin enclosure crossed the route-scale host-materialization seam",
            )
        if self.schema != SIGNED_WORK_D_W_CONTINUUM_MARGIN_SCHEMA:
            raise ValueError("unsupported D_W continuum margin schema")

    @property
    def total_perturbation_upper_bound(self) -> float:
        return math.nextafter(
            self.owner_perturbation_upper_bound
            + self.numerical_perturbation_upper_bound,
            math.inf,
        )

    @property
    def continuum_margin_lower_bound(self) -> float:
        return _next_down(self.margin_lower_bound - self.total_perturbation_upper_bound)

    @property
    def robustly_positive(self) -> bool:
        return self.continuum_margin_lower_bound > 0.0

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "margin_role": self.margin_role.value,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "finite_cuda_qualification_hash": self.finite_cuda_qualification_hash,
                "registered_d_w_prefix_hash": self.registered_d_w_prefix_hash,
                "route_axis_hash": self.route_axis_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "owner_roles": tuple(role.value for role in self.owner_roles),
                "owner_receipt_hashes": self.owner_receipt_hashes,
                "source_boundary_hash": self.source_boundary_hash,
                "source_derivation_hash": self.source_derivation_hash,
                "outward_enclosure_policy_hash": self.outward_enclosure_policy_hash,
                "margin_interval": (self.margin_lower_bound, self.margin_upper_bound),
                "perturbation_upper_bound": self.total_perturbation_upper_bound,
                "continuum_margin_lower_bound": self.continuum_margin_lower_bound,
                "route_scale_d2h_bytes": 0,
            },
            prefix=_PREFIX + "margin-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionContinuumRobustnessCertificate:
    """Continuum-stable finite CUDA D witness with all authority still closed."""

    finite_cuda_qualification_hash: str
    uncertainty_set_hash: str
    owner_receipt_hashes: tuple[str, ...]
    margin_receipt_hashes: tuple[str, ...]
    resource_cell_family_hash: str
    physical_action_family_hash: str
    preferred_physical_decision_family_hash: str
    minimum_continuum_margin_lower_bound: float
    source_owner_family_hash: str
    source_margin_family_hash: str
    source_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        repr=False,
        compare=False,
    )
    source_owner_receipts: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...] = (
        field(repr=False, compare=False)
    )
    source_margin_receipts: tuple[V1SignedWorkDecisionMarginReceipt, ...] = field(
        repr=False,
        compare=False,
    )
    _issuer_token: InitVar[object]
    _issued_certificate_hash: str = field(init=False, repr=False, compare=False)
    finite_grid_complete: bool = True
    continuum_robust: bool = True
    every_registered_owner_set_enclosed: bool = True
    every_registered_decision_margin_enclosed: bool = True
    route_scale_d2h_bytes: int = 0
    generic_C_stream_open_authority: bool = False
    production_C_stream_open_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_CONTINUUM_AUTHORITY_SCOPE
    robustness_rule: str = SIGNED_WORK_D_W_CONTINUUM_RULE
    schema: str = SIGNED_WORK_D_W_CONTINUUM_CERTIFICATE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _CERTIFICATE_TOKEN:
            raise TypeError("continuum robustness certificates are issuer-only")
        if type(self.source_qualification) is not (
            V1SignedWorkCUDADecisionWitnessQualification
        ):
            raise TypeError(
                "continuum certificate requires its exact live qualification"
            )
        owners = tuple(self.source_owner_receipts)
        margins = tuple(self.source_margin_receipts)
        if (
            len(owners) != len(_REQUIRED_OWNER_ROLES)
            or any(
                type(value) is not V1SignedWorkDecisionOwnerUncertaintyReceipt
                for value in owners
            )
            or len(margins) != len(_REQUIRED_MARGIN_ROLES)
            or any(
                type(value) is not V1SignedWorkDecisionMarginReceipt
                for value in margins
            )
        ):
            raise TypeError(
                "continuum certificate requires its exact owner and margin receipts"
            )
        object.__setattr__(self, "source_owner_receipts", owners)
        object.__setattr__(self, "source_margin_receipts", margins)
        self.source_qualification.assert_live_identity()
        for name in (
            "finite_cuda_qualification_hash",
            "uncertainty_set_hash",
            "resource_cell_family_hash",
            "physical_action_family_hash",
            "preferred_physical_decision_family_hash",
            "source_owner_family_hash",
            "source_margin_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "owner_receipt_hashes",
            _identity_axis(self.owner_receipt_hashes, name="owner receipt hashes"),
        )
        object.__setattr__(
            self,
            "margin_receipt_hashes",
            _identity_axis(self.margin_receipt_hashes, name="margin receipt hashes"),
        )
        minimum = _finite(
            self.minimum_continuum_margin_lower_bound,
            name="minimum continuum margin lower bound",
        )
        if minimum <= 0.0:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.NONROBUST_MARGIN,
                "continuum certificate requires strictly positive decision margins",
            )
        object.__setattr__(self, "minimum_continuum_margin_lower_bound", minimum)
        flags = (
            _exact_bool(self.finite_grid_complete, name="finite grid complete"),
            _exact_bool(self.continuum_robust, name="continuum robust"),
            _exact_bool(
                self.every_registered_owner_set_enclosed,
                name="every owner set enclosed",
            ),
            _exact_bool(
                self.every_registered_decision_margin_enclosed,
                name="every margin enclosed",
            ),
            _exact_bool(
                self.generic_C_stream_open_authority,
                name="generic C stream authority",
            ),
            _exact_bool(
                self.production_C_stream_open_authority,
                name="production C stream authority",
            ),
            _exact_bool(self.C_measurement_authority, name="C measurement authority"),
            _exact_bool(self.calibration_authority, name="calibration authority"),
            _exact_bool(self.oracle_authority, name="oracle authority"),
            _exact_bool(self.mask_commit_authority, name="mask commit authority"),
            _exact_bool(self.terminal_authority, name="terminal authority"),
            _exact_bool(self.performance_authority, name="performance authority"),
            _exact_bool(self.pareto_authority, name="Pareto authority"),
        )
        if flags != (
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.AUTHORITY_VIOLATION,
                "continuum robustness certificate crossed its authority boundary",
            )
        d2h = _nonnegative_int(self.route_scale_d2h_bytes, name="route-scale D2H bytes")
        if d2h != 0:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.BOUNDEDNESS_VIOLATION,
                "continuum certificate performed a route-scale host transfer",
            )
        if (
            self.authority_scope != SIGNED_WORK_D_W_CONTINUUM_AUTHORITY_SCOPE
            or self.robustness_rule != SIGNED_WORK_D_W_CONTINUUM_RULE
            or self.schema != SIGNED_WORK_D_W_CONTINUUM_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported D_W continuum certificate semantics")
        self._assert_source_identity()
        object.__setattr__(self, "_issued_certificate_hash", self.certificate_hash)

    @property
    def certificate_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "finite_cuda_qualification_hash": (self.finite_cuda_qualification_hash),
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "owner_receipt_hashes": self.owner_receipt_hashes,
                "margin_receipt_hashes": self.margin_receipt_hashes,
                "resource_cell_family_hash": self.resource_cell_family_hash,
                "physical_action_family_hash": self.physical_action_family_hash,
                "preferred_physical_decision_family_hash": (
                    self.preferred_physical_decision_family_hash
                ),
                "minimum_continuum_margin_lower_bound": (
                    self.minimum_continuum_margin_lower_bound
                ),
                "source_owner_family_hash": self.source_owner_family_hash,
                "source_margin_family_hash": self.source_margin_family_hash,
                "finite_grid_complete": self.finite_grid_complete,
                "continuum_robust": self.continuum_robust,
                "every_registered_owner_set_enclosed": (
                    self.every_registered_owner_set_enclosed
                ),
                "every_registered_decision_margin_enclosed": (
                    self.every_registered_decision_margin_enclosed
                ),
                "route_scale_d2h_bytes": self.route_scale_d2h_bytes,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
                "robustness_rule": self.robustness_rule,
            },
            prefix=_PREFIX + "certificate-",
        )

    def _assert_source_identity(self) -> None:
        qualification = self.source_qualification
        qualification.assert_live_identity()
        if qualification.qualification_hash != self.finite_cuda_qualification_hash:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "continuum certificate's live CUDA qualification changed",
            )
        owner_hashes = tuple(value.receipt_hash for value in self.source_owner_receipts)
        margin_hashes = tuple(
            value.receipt_hash for value in self.source_margin_receipts
        )
        if (
            owner_hashes != self.owner_receipt_hashes
            or margin_hashes != self.margin_receipt_hashes
            or content_hash(owner_hashes, prefix=_PREFIX + "owner-family-")
            != self.source_owner_family_hash
            or content_hash(margin_hashes, prefix=_PREFIX + "margin-family-")
            != self.source_margin_family_hash
        ):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "continuum certificate's owner or margin source changed",
            )

    def assert_live_identity(self) -> None:
        """Replay the exact live CUDA/source graph before downstream use."""

        self._assert_source_identity()
        if self.certificate_hash != self._issued_certificate_hash:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "continuum certificate identity changed after issuance",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("continuum robustness certificates are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("continuum robustness certificates are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("continuum robustness certificates cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("continuum robustness certificates cannot be copied")


def derive_v1_signed_work_decision_owner_uncertainty_receipt(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_role: V1SignedWorkDecisionOwnerRole,
    source_derivation_hash: str,
    outward_radius_upper_bound: float,
) -> V1SignedWorkDecisionOwnerUncertaintyReceipt:
    """Build one owner receipt directly from the registered uncertainty set."""

    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("owner receipt derivation requires the exact uncertainty set")
    if type(owner_role) is not V1SignedWorkDecisionOwnerRole:
        raise TypeError("owner role must be exact")
    source = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: (
            uncertainty.signed_f_sampling,
            True,
        ),
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: (
            uncertainty.literal_j_sampling,
            True,
        ),
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: (
            uncertainty.fd_residual,
            False,
        ),
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: (
            uncertainty.solver_rounding,
            False,
        ),
    }[owner_role]
    owner, statistical = source
    if statistical:
        expected_role = (
            V1SignedWorkSamplingUncertaintyRole.SIGNED_F
            if owner_role is V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING
            else V1SignedWorkSamplingUncertaintyRole.LITERAL_J
        )
        if owner.role is not expected_role:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "sampling owner role does not match its uncertainty receipt role",
            )
    exact_census = (
        bool(getattr(owner, "full_train_census", False)) if statistical else False
    )
    if exact_census and float(outward_radius_upper_bound) != 0.0:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
            "an exact finite-population census owner must be derived with zero radius",
        )
    return V1SignedWorkDecisionOwnerUncertaintyReceipt(
        owner_role=owner_role,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        owner_set_hash=owner.set_hash,
        registered_d_w_prefix_hash=uncertainty.registered_d_w_prefix_hash,
        route_axis_hash=uncertainty.route_axis_hash,
        realization_hashes=tuple(owner.realization_hashes),
        source_derivation_hash=source_derivation_hash,
        outward_radius_upper_bound=outward_radius_upper_bound,
        statistical_uncertainty=(statistical and not exact_census),
        deterministic_uncertainty=(not statistical),
        spends_alpha=(statistical and not exact_census),
        exact_finite_population_census=exact_census,
    )


def derive_v1_signed_work_decision_owner_receipts(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    source_derivation_hashes: Mapping[V1SignedWorkDecisionOwnerRole, str],
    outward_radius_upper_bounds: (
        Mapping[V1SignedWorkDecisionOwnerRole, float] | None
    ) = None,
) -> tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]:
    """Build all four canonical owner receipts in deterministic role order."""

    radii = (
        {} if outward_radius_upper_bounds is None else dict(outward_radius_upper_bounds)
    )
    if set(source_derivation_hashes) != set(_REQUIRED_OWNER_ROLES):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.MISSING_OWNER_RECEIPT,
            "source derivation hashes do not cover the four owner sets",
        )
    return tuple(
        derive_v1_signed_work_decision_owner_uncertainty_receipt(
            uncertainty=uncertainty,
            owner_role=role,
            source_derivation_hash=source_derivation_hashes[role],
            outward_radius_upper_bound=float(radii.get(role, 0.0)),
        )
        for role in _REQUIRED_OWNER_ROLES
    )


def derive_v1_signed_work_decision_owner_receipts_from_variable_shell_source(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    variable_shell_source_receipt: V1SignedWorkVariableShellSourceReceipt,
    deterministic_source_derivation_hashes: Mapping[
        V1SignedWorkDecisionOwnerRole,
        str,
    ],
    outward_radius_upper_bounds: (
        Mapping[V1SignedWorkDecisionOwnerRole, float] | None
    ) = None,
) -> tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]:
    """Derive sampling-owner receipts from the registered variable-shell source.

    The variable-shell source owns the same-capture corrected-evidence and
    literal-J D_W2 stream identities. This helper binds signed-F/literal-J
    owner receipts to that source object instead of accepting caller-authored
    sampling source hashes. FD residual and solver/rounding owners remain
    separate deterministic uncertainty sets and must provide their own source
    derivation hashes.
    """

    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("variable-shell owner derivation requires exact uncertainty")
    if (
        type(variable_shell_source_receipt)
        is not V1SignedWorkVariableShellSourceReceipt
    ):
        raise TypeError("variable-shell owner derivation requires exact source receipt")
    if (
        variable_shell_source_receipt.route_axis_hash != uncertainty.route_axis_hash
        or variable_shell_source_receipt.registered_d_w2_prefix_hash
        != uncertainty.registered_d_w_prefix_hash
    ):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
            "variable-shell source receipt is not bound to this D_W uncertainty set",
        )

    source_census = bool(variable_shell_source_receipt.finite_census_complete)
    for owner in (uncertainty.signed_f_sampling, uncertainty.literal_j_sampling):
        if bool(owner.full_train_census) is not source_census:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.UNENCLOSED_OWNER,
                "sampling owner census status does not match the source receipt",
            )

    deterministic_roles = {
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
    }
    if set(deterministic_source_derivation_hashes) != deterministic_roles:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.MISSING_OWNER_RECEIPT,
            "deterministic source hashes must cover FD residual and solver owners",
        )

    sampling_schema = (
        "partition_free_v1_signed_work_D_W_variable_shell_"
        "sampling_owner_source_derivation_v1"
    )
    source_hashes = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: content_hash(
            {
                "schema": sampling_schema,
                "owner_role": V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING.value,
                "uncertainty_hash": uncertainty.uncertainty_hash,
                "owner_set_hash": uncertainty.signed_f_sampling.set_hash,
                "variable_shell_source_receipt_hash": (
                    variable_shell_source_receipt.receipt_hash
                ),
                "source_replay_hash": variable_shell_source_receipt.source_replay_hash,
                "stream_source_hash": (
                    variable_shell_source_receipt.corrected_evidence_stream_source_hash
                ),
                "finite_census_complete": source_census,
            },
            prefix=_PREFIX + "variable-shell-owner-source-",
        ),
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: content_hash(
            {
                "schema": sampling_schema,
                "owner_role": V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING.value,
                "uncertainty_hash": uncertainty.uncertainty_hash,
                "owner_set_hash": uncertainty.literal_j_sampling.set_hash,
                "variable_shell_source_receipt_hash": (
                    variable_shell_source_receipt.receipt_hash
                ),
                "source_replay_hash": variable_shell_source_receipt.source_replay_hash,
                "stream_source_hash": variable_shell_source_receipt.literal_j_stream_source_hash,
                "finite_census_complete": source_census,
            },
            prefix=_PREFIX + "variable-shell-owner-source-",
        ),
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: deterministic_source_derivation_hashes[
            V1SignedWorkDecisionOwnerRole.FD_RESIDUAL
        ],
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: deterministic_source_derivation_hashes[
            V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING
        ],
    }
    return derive_v1_signed_work_decision_owner_receipts(
        uncertainty=uncertainty,
        source_derivation_hashes=source_hashes,
        outward_radius_upper_bounds=outward_radius_upper_bounds,
    )


def _validate_finite_cuda_qualification(
    *,
    qualification: object,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> str:
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("continuum promotion requires the exact uncertainty set")
    if type(qualification) is not V1SignedWorkCUDADecisionWitnessQualification:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.INVALID_TYPE,
            "continuum promotion requires a finite CUDA D_W qualification",
        )
    qualification.assert_live_identity()
    if qualification.schema != SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA:
        raise ValueError("unsupported finite CUDA D_W qualification schema")
    finite_flags = (
        getattr(qualification, "complete_registered_uncertainty_product", False),
        getattr(qualification, "resource_cell_physical_action_invariant", False),
        getattr(qualification, "byte_deduplicated_physical_actions_verified", False),
        getattr(qualification, "final_solver_dual_custody_verified", False),
    )
    if finite_flags != (True, True, True, True):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.FINITE_GRID_INCOMPLETE,
            "finite CUDA D_W qualification is not complete",
        )
    if (
        _nonnegative_int(
            getattr(qualification, "route_scale_d2h_bytes", 0),
            name="finite qualification route-scale D2H bytes",
        )
        != 0
    ):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.BOUNDEDNESS_VIOLATION,
            "finite CUDA D_W qualification performed route-scale host transfer",
        )
    _false_downstream_authority(qualification)
    if (
        _identity(
            getattr(qualification, "uncertainty_set_hash", None),
            name="finite qualification uncertainty hash",
        )
        != uncertainty.uncertainty_hash
        or _identity(
            getattr(qualification, "route_axis_hash", None),
            name="finite qualification route axis",
        )
        != uncertainty.route_axis_hash
        or _identity(
            getattr(qualification, "registered_d_w_prefix_hash", None),
            name="finite qualification D_W prefix",
        )
        != uncertainty.registered_d_w_prefix_hash
    ):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
            "finite CUDA qualification does not match the requested uncertainty set",
        )
    for role, expected in (
        (
            V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
            uncertainty.signed_f_sampling.set_hash,
        ),
        (
            V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
            uncertainty.literal_j_sampling.set_hash,
        ),
        (
            V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
            uncertainty.fd_residual.set_hash,
        ),
        (
            V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
            uncertainty.solver_rounding.set_hash,
        ),
    ):
        observed = _expected_owner_hashes(uncertainty, qualification)[role]
        if observed != expected:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "finite CUDA qualification owner hash is stale",
            )
    return _qualification_hash(qualification)


def _tuple_family_hash(value: object, *, name: str, prefix: str) -> str:
    family = tuple(getattr(value, name, ()))
    if not family:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
            f"finite CUDA qualification lacks {name}",
        )
    return content_hash(family, prefix=_PREFIX + prefix)


def certify_v1_signed_work_decision_continuum_robustness(
    *,
    qualification: object,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_receipts: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...],
    margin_receipts: tuple[V1SignedWorkDecisionMarginReceipt, ...],
) -> V1SignedWorkDecisionContinuumRobustnessCertificate:
    """Promote a finite CUDA D_W witness only when the continuum is enclosed."""

    qualification_hash = _validate_finite_cuda_qualification(
        qualification=qualification,
        uncertainty=uncertainty,
    )
    owners = tuple(owner_receipts)
    margins = tuple(margin_receipts)
    if not owners and not margins:
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.FINITE_GRID_ONLY,
            "finite-grid D_W stability lacks continuum owner and margin enclosures",
        )
    by_owner = {value.owner_role: value for value in owners}
    if (
        len(by_owner) != len(owners)
        or tuple(by_owner.get(role) for role in _REQUIRED_OWNER_ROLES).count(None) != 0
    ):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.MISSING_OWNER_RECEIPT,
            "continuum promotion requires exactly one receipt for each owner",
        )
    expected_owner_hashes = _expected_owner_hashes(uncertainty, qualification)
    for role in _REQUIRED_OWNER_ROLES:
        receipt = by_owner[role]
        if (
            receipt.uncertainty_set_hash != uncertainty.uncertainty_hash
            or receipt.owner_set_hash != expected_owner_hashes[role]
            or receipt.registered_d_w_prefix_hash
            != uncertainty.registered_d_w_prefix_hash
            or receipt.route_axis_hash != uncertainty.route_axis_hash
        ):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "an owner uncertainty receipt is not source-bound to this D_W witness",
            )
    by_margin = {value.margin_role: value for value in margins}
    if (
        len(by_margin) != len(margins)
        or tuple(by_margin.get(role) for role in _REQUIRED_MARGIN_ROLES).count(None)
        != 0
    ):
        _refuse(
            V1SignedWorkDecisionEnclosureRefusalCode.MISSING_MARGIN_RECEIPT,
            "continuum promotion requires every registered D decision margin",
        )
    parent = _identity(
        getattr(qualification, "parent_mask_hash", None),
        name="finite qualification parent mask",
    )
    coordinate = _identity(
        getattr(qualification, "registered_d_coordinate_hash", None),
        name="finite qualification D coordinate",
    )
    minimum_margin = math.inf
    for role in _REQUIRED_MARGIN_ROLES:
        receipt = by_margin[role]
        expected_margin_owner_hashes = tuple(
            by_owner[owner_role].receipt_hash for owner_role in receipt.owner_roles
        )
        if (
            receipt.uncertainty_set_hash != uncertainty.uncertainty_hash
            or receipt.finite_cuda_qualification_hash != qualification_hash
            or receipt.registered_d_w_prefix_hash
            != uncertainty.registered_d_w_prefix_hash
            or receipt.route_axis_hash != uncertainty.route_axis_hash
            or receipt.parent_mask_hash != parent
            or receipt.registered_d_coordinate_hash != coordinate
            or (
                bool(receipt.owner_receipt_hashes)
                and receipt.owner_receipt_hashes != expected_margin_owner_hashes
            )
        ):
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.STALE_IDENTITY,
                "a decision-margin receipt is not source-bound to this D_W witness",
            )
        if not receipt.robustly_positive:
            _refuse(
                V1SignedWorkDecisionEnclosureRefusalCode.NONROBUST_MARGIN,
                f"{role.value} has no strictly positive continuum margin",
            )
        minimum_margin = min(minimum_margin, receipt.continuum_margin_lower_bound)
    return V1SignedWorkDecisionContinuumRobustnessCertificate(
        finite_cuda_qualification_hash=qualification_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        owner_receipt_hashes=tuple(
            by_owner[role].receipt_hash for role in _REQUIRED_OWNER_ROLES
        ),
        margin_receipt_hashes=tuple(
            by_margin[role].receipt_hash for role in _REQUIRED_MARGIN_ROLES
        ),
        resource_cell_family_hash=_tuple_family_hash(
            qualification,
            name="resource_cell_family",
            prefix="resource-cell-family-",
        ),
        physical_action_family_hash=_tuple_family_hash(
            qualification,
            name="physical_action_family",
            prefix="physical-action-family-",
        ),
        preferred_physical_decision_family_hash=_tuple_family_hash(
            qualification,
            name="preferred_physical_decision_family",
            prefix="preferred-physical-family-",
        ),
        minimum_continuum_margin_lower_bound=minimum_margin,
        source_owner_family_hash=content_hash(
            tuple(by_owner[role].receipt_hash for role in _REQUIRED_OWNER_ROLES),
            prefix=_PREFIX + "owner-family-",
        ),
        source_margin_family_hash=content_hash(
            tuple(by_margin[role].receipt_hash for role in _REQUIRED_MARGIN_ROLES),
            prefix=_PREFIX + "margin-family-",
        ),
        source_qualification=qualification,
        source_owner_receipts=tuple(by_owner[role] for role in _REQUIRED_OWNER_ROLES),
        source_margin_receipts=tuple(
            by_margin[role] for role in _REQUIRED_MARGIN_ROLES
        ),
        _issuer_token=_CERTIFICATE_TOKEN,
    )


__all__ = [
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE",
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_SCHEMA",
    "SIGNED_WORK_D_W_CONTINUUM_AUTHORITY_SCOPE",
    "SIGNED_WORK_D_W_CONTINUUM_CERTIFICATE_SCHEMA",
    "SIGNED_WORK_D_W_CONTINUUM_MARGIN_SCHEMA",
    "SIGNED_WORK_D_W_CONTINUUM_OWNER_SCHEMA",
    "V1SignedWorkDecisionContinuumRobustnessCertificate",
    "V1SignedWorkDecisionBoundaryClosureKind",
    "V1SignedWorkDecisionContinuumClosureMode",
    "V1SignedWorkDecisionEnclosureRefusal",
    "V1SignedWorkDecisionEnclosureRefusalCode",
    "V1SignedWorkDecisionMarginReceipt",
    "V1SignedWorkDecisionMarginRole",
    "V1SignedWorkDecisionOwnerRole",
    "V1SignedWorkDecisionOwnerUncertaintyReceipt",
    "certify_v1_signed_work_decision_continuum_robustness",
    "derive_v1_signed_work_decision_owner_receipts",
    "derive_v1_signed_work_decision_owner_receipts_from_variable_shell_source",
    "derive_v1_signed_work_decision_owner_uncertainty_receipt",
    "v1_signed_work_decision_margin_owner_roles",
    "v1_signed_work_decision_boundary_closure_kind",
]
