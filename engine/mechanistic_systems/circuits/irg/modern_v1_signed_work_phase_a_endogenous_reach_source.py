"""Typed source requirements for the missing Phase-A continuum-reach theorem.

This module deliberately does not issue a positive reach certificate.  It
records the native coordinate, norm, and causal role of every source that a
future certificate must consume.  In particular, it prevents the historical
four-owner scalar axis from being mistaken for one dimensionally homogeneous
uncertainty ball.

The only admissible future positive producers for branch-endpoint motion are
certified robust re-solves over the registered uncertainty envelopes or a full
KKT strong-regularity theorem.  Fixed-direction ``M_J z`` evidence and caller
floats are not solution-sensitivity proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash


PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENT_SCHEMA: Final = (
    "partition_free_v1_phase_A_endogenous_typed_source_requirement_v1"
)
PHASE_A_BRANCH_ENDPOINT_MOTION_REQUIREMENT_SCHEMA: Final = (
    "partition_free_v1_phase_A_branch_endpoint_Q_motion_requirement_v1"
)
PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE: Final = (
    "V1PhaseAEndogenousTypedSourceDecisionBoundaryReachCertificate"
)
PHASE_A_ENDOGENOUS_REACH_API_NAME: Final = (
    "derive_from_registered_typed_A_F_J_FD_solver_and_phase_A_boundary_operators"
)

_PREFIX: Final = "partition-free-v1-phase-A-endogenous-reach-source-"


class V1PhaseAEndogenousReachSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    DIMENSION_MISMATCH = "dimension_mismatch"
    DOUBLE_CHARGE = "double_charge"
    UNIVERSAL_RADIUS_FORBIDDEN = "universal_radius_forbidden"
    MISSING_BRANCH_ENDPOINT_MOTION_PRODUCER = (
        "missing_branch_endpoint_motion_producer"
    )


class V1PhaseAEndogenousReachSourceRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAEndogenousReachSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAEndogenousReachSourceRefusalCode:
            raise TypeError("endogenous reach-source refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAEndogenousReachSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAEndogenousReachSourceRefusal(code, message)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real scalar",
        )
    scalar = float(value)
    if not math.isfinite(scalar):
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
            f"{name} must be finite",
        )
    return scalar


def _positive(value: object, *, name: str) -> float:
    scalar = _finite(value, name=name)
    if scalar <= 0.0:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
            f"{name} must be strictly positive",
        )
    return scalar


def _nonzero(value: object, *, name: str) -> float:
    scalar = _finite(value, name=name)
    if scalar == 0.0:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
            f"{name} must be nonzero",
        )
    return scalar


def _outward_abs_difference_upper(left: float, right: float) -> float:
    radius = abs(left - right)
    if not math.isfinite(radius):
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
            "interval difference overflowed FP64",
        )
    if radius == 0.0:
        return 0.0
    return math.nextafter(radius, math.inf)


class V1PhaseANativeScalarUnit(str, Enum):
    """Native units; distinct members may never share an untyped radius."""

    CORRECTED_LOGIT_PER_GATE = "corrected_logit_per_gate"
    QUADRATIC_RESPONSE_PER_GATE = "raw_logit_squared_per_gate"
    QUADRATIC_RESPONSE = "raw_logit_squared"
    QUADRATIC_RESPONSE_OPERATOR = "raw_logit_squared_per_gate_squared"
    PHYSICAL_RESOURCE_ENDPOINT = "physical_resource_endpoint"

    @property
    def logit_homogeneity_degree(self) -> int:
        if self is V1PhaseANativeScalarUnit.CORRECTED_LOGIT_PER_GATE:
            return 1
        if self in {
            V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_PER_GATE,
            V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE,
            V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_OPERATOR,
        }:
            return 2
        return 0

    @property
    def resource_homogeneity_degree(self) -> int:
        return int(self is V1PhaseANativeScalarUnit.PHYSICAL_RESOURCE_ENDPOINT)

    def rescale(
        self,
        value: object,
        *,
        logit_scale: object = 1.0,
        resource_scale: object = 1.0,
    ) -> float:
        """Pure unit arithmetic; this never issues statistical authority."""

        scalar = _finite(value, name="native-unit value")
        logit = abs(_nonzero(logit_scale, name="logit scale"))
        resource = _positive(resource_scale, name="resource scale")
        return scalar * (logit**self.logit_homogeneity_degree) * (
            resource**self.resource_homogeneity_degree
        )


class V1PhaseASourceNorm(str, Enum):
    ROUTE_LINF = "route_linf"
    SCALAR_ABSOLUTE = "scalar_absolute"
    SPECTRAL_L2_TO_L2 = "spectral_l2_to_l2"
    NONNEGATIVE_DIAGONAL_COORDINATEWISE = (
        "nonnegative_diagonal_coordinatewise"
    )
    BRANCH_ENDPOINT_LINF = "branch_endpoint_linf"


@dataclass(frozen=True, slots=True)
class V1PhaseACenteredIntervalEnclosure:
    """Non-authoritative arithmetic enclosure about one registered center.

    The current-look corrected-A point need not be the midpoint of the
    cumulative confidence interval.  Its radius must therefore be computed
    about the exact center used by the solver, not silently replaced by the
    interval half-width.  Choosing the midpoint is valid only after re-solving
    the nominal problem at that midpoint.
    """

    center: float
    lower: float
    upper: float
    native_unit: V1PhaseANativeScalarUnit
    outward_radius_upper_bound: float = field(init=False)
    midpoint: float = field(init=False)
    midpoint_radius: float = field(init=False)
    decision_authority: bool = False

    def __post_init__(self) -> None:
        center = _finite(self.center, name="registered interval center")
        lower = _finite(self.lower, name="interval lower endpoint")
        upper = _finite(self.upper, name="interval upper endpoint")
        if lower > upper:
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
                "interval lower endpoint exceeds its upper endpoint",
            )
        if type(self.native_unit) is not V1PhaseANativeScalarUnit:
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
                "centered interval requires one exact native unit",
            )
        if self.decision_authority is not False:
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.UNIVERSAL_RADIUS_FORBIDDEN,
                "scalar interval arithmetic cannot issue continuum authority",
            )
        radius = max(
            _outward_abs_difference_upper(center, lower),
            _outward_abs_difference_upper(upper, center),
        )
        midpoint = lower + 0.5 * (upper - lower)
        midpoint_radius = max(
            _outward_abs_difference_upper(midpoint, lower),
            _outward_abs_difference_upper(upper, midpoint),
        )
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "outward_radius_upper_bound", radius)
        object.__setattr__(self, "midpoint", midpoint)
        object.__setattr__(self, "midpoint_radius", midpoint_radius)

    @property
    def interval_contained_about_registered_center(self) -> bool:
        radius = self.outward_radius_upper_bound
        return self.center - radius <= self.lower and self.upper <= self.center + radius

    def rescaled(
        self,
        *,
        logit_scale: object = 1.0,
        resource_scale: object = 1.0,
    ) -> V1PhaseACenteredIntervalEnclosure:
        return V1PhaseACenteredIntervalEnclosure(
            center=self.native_unit.rescale(
                self.center,
                logit_scale=logit_scale,
                resource_scale=resource_scale,
            ),
            lower=self.native_unit.rescale(
                self.lower,
                logit_scale=logit_scale,
                resource_scale=resource_scale,
            ),
            upper=self.native_unit.rescale(
                self.upper,
                logit_scale=logit_scale,
                resource_scale=resource_scale,
            ),
            native_unit=self.native_unit,
        )


@dataclass(frozen=True, slots=True)
class V1PhaseACorrectedAFBandSourceRequirement:
    """Corrected-A and signed-F bands; Phase-B only by causal construction."""

    corrected_a_source_type: str = "V1CorrectedAttributionConfidenceBoxV2Receipt"
    corrected_a_center_field: str = "class_balanced_absolute_point"
    corrected_a_lower_field: str = "cumulative_class_balanced_absolute_lower"
    corrected_a_upper_field: str = "cumulative_class_balanced_absolute_upper"
    signed_f_source_type: str = "V1SignedWorkFixedPrimalProjectionCUDAExecution"
    signed_f_sampling_radius_field: str = "signed_f_sampling_radius"
    signed_f_numerical_radius_field: str = "signed_f_numerical_radius"
    native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.CORRECTED_LOGIT_PER_GATE
    )
    source_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.ROUTE_LINF
    phase_b_only: bool = True
    phase_a_causal_sensitivity_exactly_zero: bool = True
    direct_product_components_may_not_be_summed: bool = True
    sampling_component_must_be_retained_not_only_fingerprinted: bool = True
    action_f_L_U_already_encloses_sampling_and_numerics: bool = True
    additional_action_L_U_owner_charge_forbidden: bool = True
    universal_scalar_radius_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.native_unit is not V1PhaseANativeScalarUnit.CORRECTED_LOGIT_PER_GATE
            or self.source_norm is not V1PhaseASourceNorm.ROUTE_LINF
            or self.phase_b_only is not True
            or self.phase_a_causal_sensitivity_exactly_zero is not True
            or self.direct_product_components_may_not_be_summed is not True
            or self.sampling_component_must_be_retained_not_only_fingerprinted
            is not True
            or self.action_f_L_U_already_encloses_sampling_and_numerics is not True
            or self.additional_action_L_U_owner_charge_forbidden is not True
            or self.universal_scalar_radius_authority is not False
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.DIMENSION_MISMATCH,
                "corrected A/F lost its Phase-B-only typed-band contract",
            )

    def phase_a_boundary_reach_upper_bound(self) -> float:
        return 0.0

    def assert_additional_action_L_U_owner_charge_allowed(self) -> NoReturn:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.DOUBLE_CHARGE,
            "Phase-B action F L/U already encloses sampling and numerical uncertainty",
        )


@dataclass(frozen=True, slots=True)
class V1PhaseALiteralJBandSourceRequirement:
    """Distinct fixed-direction ``M_J z`` and scalar-Q statistical bands."""

    source_type: str = "V1LiteralJDFixedResidualPopulationProjectionReceipt"
    mjz_radius_field: str = "mjz_sampling_linf_radius_upper_bound"
    q_radius_field: str = "q_sampling_linf_radius_upper_bound"
    mjz_native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_PER_GATE
    )
    q_native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE
    )
    mjz_source_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.ROUTE_LINF
    q_source_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.SCALAR_ABSOLUTE
    fixed_primal_q_gradient_factor_from_mjz: int = 2
    fixed_direction_only_not_full_operator_sensitivity: bool = True
    universal_scalar_radius_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.mjz_native_unit
            is not V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_PER_GATE
            or self.q_native_unit
            is not V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE
            or self.mjz_source_norm is not V1PhaseASourceNorm.ROUTE_LINF
            or self.q_source_norm is not V1PhaseASourceNorm.SCALAR_ABSOLUTE
            or self.fixed_primal_q_gradient_factor_from_mjz != 2
            or self.fixed_direction_only_not_full_operator_sensitivity is not True
            or self.universal_scalar_radius_authority is not False
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.DIMENSION_MISMATCH,
                "literal-J MJz/Q units or the exact Q-gradient factor changed",
            )

    def q_gradient_from_mjz(self, value: object) -> float:
        return self.fixed_primal_q_gradient_factor_from_mjz * _finite(
            value,
            name="M_J z coordinate",
        )


@dataclass(frozen=True, slots=True)
class V1PhaseAFDResidualEnvelopeSourceRequirement:
    """Deterministic FD envelope with both spectral and diagonal owners."""

    source_type: str = "V1StratifiedFDGeometryResult"
    spectral_radius_receipt_field: str = "combined_residual_spectral_bound"
    diagonal_live_field: str = "geometry.residual_diagonal_bound"
    diagonal_hash_receipt_field: str = "residual_diagonal_hash"
    native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_OPERATOR
    )
    spectral_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.SPECTRAL_L2_TO_L2
    diagonal_norm: V1PhaseASourceNorm = (
        V1PhaseASourceNorm.NONNEGATIVE_DIAGONAL_COORDINATEWISE
    )
    spectral_and_diagonal_branches_both_required: bool = True
    already_embedded_in_phase_a_q_branches: bool = True
    additional_phase_a_q_charge_forbidden: bool = True
    deterministic_not_alpha_spending: bool = True
    universal_scalar_radius_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.native_unit
            is not V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE_OPERATOR
            or self.spectral_norm is not V1PhaseASourceNorm.SPECTRAL_L2_TO_L2
            or self.diagonal_norm
            is not V1PhaseASourceNorm.NONNEGATIVE_DIAGONAL_COORDINATEWISE
            or self.spectral_and_diagonal_branches_both_required is not True
            or self.already_embedded_in_phase_a_q_branches is not True
            or self.additional_phase_a_q_charge_forbidden is not True
            or self.deterministic_not_alpha_spending is not True
            or self.universal_scalar_radius_authority is not False
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.DIMENSION_MISMATCH,
                "FD residual lost rho+d or single-charge semantics",
            )

    def assert_additional_phase_a_q_charge_allowed(self) -> NoReturn:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.DOUBLE_CHARGE,
            "rho and d are already embedded in the registered Phase-A Q branches",
        )


@dataclass(frozen=True, slots=True)
class V1PhaseASolverNativeIntervalSourceRequirement:
    """Solver uncertainty remains in separate resource and Q coordinates."""

    source_type: str = "V1SignedWorkResourceFrontierCUDAResult"
    resource_center_field: str = "physical_resource_center"
    resource_lower_field: str = "physical_resource_lower_bound"
    resource_upper_field: str = "physical_resource_upper_bound"
    q_candidate_field: str = "q_min_candidate"
    q_lower_bound_field: str = "q_min_lower_bound"
    resource_native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.PHYSICAL_RESOURCE_ENDPOINT
    )
    q_native_unit: V1PhaseANativeScalarUnit = (
        V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE
    )
    resource_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.BRANCH_ENDPOINT_LINF
    q_norm: V1PhaseASourceNorm = V1PhaseASourceNorm.SCALAR_ABSOLUTE
    scalar_replay_already_enclosed_in_resource_cell_slack: bool = True
    primal_dual_q_interval_already_enclosed: bool = True
    universal_scalar_radius_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.resource_native_unit
            is not V1PhaseANativeScalarUnit.PHYSICAL_RESOURCE_ENDPOINT
            or self.q_native_unit
            is not V1PhaseANativeScalarUnit.QUADRATIC_RESPONSE
            or self.resource_norm is not V1PhaseASourceNorm.BRANCH_ENDPOINT_LINF
            or self.q_norm is not V1PhaseASourceNorm.SCALAR_ABSOLUTE
            or self.scalar_replay_already_enclosed_in_resource_cell_slack is not True
            or self.primal_dual_q_interval_already_enclosed is not True
            or self.universal_scalar_radius_authority is not False
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.DIMENSION_MISMATCH,
                "solver resource/Q intervals lost native-unit separation",
            )

    def assert_universal_radius_allowed(self) -> NoReturn:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.UNIVERSAL_RADIUS_FORBIDDEN,
            "solver resource-endpoint and Q intervals have incompatible units",
        )

    def assert_additional_resource_scalar_replay_charge_allowed(self) -> NoReturn:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.DOUBLE_CHARGE,
            "resource-cell slack already encloses CUDA scalar replay uncertainty",
        )


class V1PhaseABranchEndpointMotionPositiveProducer(str, Enum):
    SELF_CONSISTENT_BOUNDED_ROBUST_VARIABLE_SHELL_RESOLVE = (
        "self_consistent_B_le_8_robust_variable_shell_resolve_over_registered_"
        "typed_uncertainty_envelopes"
    )
    FULL_KKT_STRONG_REGULARITY = (
        "full_KKT_strong_regularity_and_solution_sensitivity_theorem"
    )


@dataclass(frozen=True, slots=True)
class V1PhaseARobustVariableShellResolveRequirement:
    """Natural tractable producer for endpoint/Q motion, not its implementation."""

    maximum_shell_route_count: int = 8
    same_capture_shell_column_covariance_supports_required: bool = True
    loewner_lower_and_upper_shell_Q_envelopes_required: bool = True
    strict_outside_coordinate_no_entry_screen_required: bool = True
    within_shell_degeneracy_delegated_to_categorical_action_image: bool = True
    expand_shell_on_any_admissible_ambiguity: bool = True
    expansion_must_reach_self_consistent_fixed_point: bool = True
    ambiguity_at_shell_cap_refuses: bool = True
    fixed_z_mjz_alone_is_insufficient: bool = True
    resource_cell_exit_endpoint_linf_sensitivity: int = 1
    optimal_interval_merge_split_endpoint_linf_sensitivity: int = 2
    certified_infeasible_q_boundary_sensitivity: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.maximum_shell_route_count) is not int
            or self.maximum_shell_route_count != 8
            or any(
                getattr(self, name) is not True
                for name in (
                    "same_capture_shell_column_covariance_supports_required",
                    "loewner_lower_and_upper_shell_Q_envelopes_required",
                    "strict_outside_coordinate_no_entry_screen_required",
                    "within_shell_degeneracy_delegated_to_categorical_action_image",
                    "expand_shell_on_any_admissible_ambiguity",
                    "expansion_must_reach_self_consistent_fixed_point",
                    "ambiguity_at_shell_cap_refuses",
                    "fixed_z_mjz_alone_is_insufficient",
                )
            )
            or self.resource_cell_exit_endpoint_linf_sensitivity != 1
            or self.optimal_interval_merge_split_endpoint_linf_sensitivity != 2
            or self.certified_infeasible_q_boundary_sensitivity != 1
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
                "robust variable-shell producer lost its B<=8 fixed-point or "
                "1/2/1 boundary contract",
            )


@dataclass(frozen=True, slots=True)
class V1PhaseABranchEndpointQMotionMapRequirement:
    """Fail-closed interface for the genuinely absent sensitivity map."""

    accepted_positive_producers: tuple[
        V1PhaseABranchEndpointMotionPositiveProducer, ...
    ] = tuple(V1PhaseABranchEndpointMotionPositiveProducer)
    required_input_coordinates: tuple[str, ...] = (
        "literal_J_MJz_route_linf_and_Q_scalar_absolute_bands",
        "FD_spectral_rho_and_coordinatewise_diagonal_d",
        "solver_resource_endpoint_and_Q_native_intervals",
        "same_capture_shell_column_covariance_supports",
    )
    required_output_coordinates: tuple[str, ...] = (
        "resource_endpoint_linf_motion_upper_bound_in_physical_resource_units",
        "Q_lower_bound_motion_upper_bound_in_raw_logit_squared_units",
    )
    robust_variable_shell_resolve: V1PhaseARobustVariableShellResolveRequirement = (
        field(default_factory=V1PhaseARobustVariableShellResolveRequirement)
    )
    fixed_direction_mjz_alone_is_sufficient: bool = False
    caller_scalar_radius_is_sufficient: bool = False
    positive_producer_present: bool = False
    schema: str = PHASE_A_BRANCH_ENDPOINT_MOTION_REQUIREMENT_SCHEMA
    requirement_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.accepted_positive_producers
            != tuple(V1PhaseABranchEndpointMotionPositiveProducer)
            or type(self.required_input_coordinates) is not tuple
            or not self.required_input_coordinates
            or len(self.required_input_coordinates)
            != len(set(self.required_input_coordinates))
            or type(self.required_output_coordinates) is not tuple
            or not self.required_output_coordinates
            or len(self.required_output_coordinates)
            != len(set(self.required_output_coordinates))
            or type(self.robust_variable_shell_resolve)
            is not V1PhaseARobustVariableShellResolveRequirement
            or self.fixed_direction_mjz_alone_is_sufficient is not False
            or self.caller_scalar_radius_is_sufficient is not False
            or self.positive_producer_present is not False
            or self.schema != PHASE_A_BRANCH_ENDPOINT_MOTION_REQUIREMENT_SCHEMA
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
                "branch endpoint-motion requirement overstated positive authority",
            )
        object.__setattr__(
            self,
            "requirement_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "requirement_hash"
                },
                prefix=_PREFIX + "branch-endpoint-motion-requirement-",
            ),
        )

    def require_positive_producer(self) -> NoReturn:
        _refuse(
            V1PhaseAEndogenousReachSourceRefusalCode.MISSING_BRANCH_ENDPOINT_MOTION_PRODUCER,
            "only a self-consistent B<=8 robust variable-shell re-solve with "
            "same-capture covariance supports, Loewner Q envelopes, strict "
            "outside-shell no-entry/no-exit screening, categorical closure of "
            "within-shell degeneracy, and ambiguity-driven shell expansion to "
            "a fixed point (or refusal at the cap), or a full KKT "
            "strong-regularity theorem, may map typed J/FD/solver uncertainty "
            "to branch endpoint motion; fixed-z M_J z alone refuses",
        )


@dataclass(frozen=True, slots=True)
class V1PhaseAEndogenousTypedSourceRequirementSet:
    corrected_a_and_signed_f: V1PhaseACorrectedAFBandSourceRequirement = field(
        default_factory=V1PhaseACorrectedAFBandSourceRequirement
    )
    literal_j: V1PhaseALiteralJBandSourceRequirement = field(
        default_factory=V1PhaseALiteralJBandSourceRequirement
    )
    fd_residual: V1PhaseAFDResidualEnvelopeSourceRequirement = field(
        default_factory=V1PhaseAFDResidualEnvelopeSourceRequirement
    )
    solver: V1PhaseASolverNativeIntervalSourceRequirement = field(
        default_factory=V1PhaseASolverNativeIntervalSourceRequirement
    )
    branch_endpoint_motion: V1PhaseABranchEndpointQMotionMapRequirement = field(
        default_factory=V1PhaseABranchEndpointQMotionMapRequirement
    )
    legacy_four_scalar_axis_is_provenance_only: bool = True
    no_universal_owner_radius_authority: bool = True
    schema: str = PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENT_SCHEMA
    requirement_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.corrected_a_and_signed_f)
            is not V1PhaseACorrectedAFBandSourceRequirement
            or type(self.literal_j) is not V1PhaseALiteralJBandSourceRequirement
            or type(self.fd_residual)
            is not V1PhaseAFDResidualEnvelopeSourceRequirement
            or type(self.solver) is not V1PhaseASolverNativeIntervalSourceRequirement
            or type(self.branch_endpoint_motion)
            is not V1PhaseABranchEndpointQMotionMapRequirement
            or self.legacy_four_scalar_axis_is_provenance_only is not True
            or self.no_universal_owner_radius_authority is not True
            or self.schema != PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENT_SCHEMA
        ):
            _refuse(
                V1PhaseAEndogenousReachSourceRefusalCode.INVALID_TYPE,
                "typed source requirement set lost a canonical source owner",
            )
        object.__setattr__(
            self,
            "requirement_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "requirement_hash"
                },
                prefix=_PREFIX + "typed-source-requirement-set-",
            ),
        )

    @property
    def typed_supply_axis(self) -> tuple[str, ...]:
        return (
            "corrected_A_and_signed_F_route_linf_bands_are_Phase_B_only_with_"
            "zero_Phase_A_causal_sensitivity",
            "literal_J_keeps_MJz_route_linf_and_Q_scalar_absolute_bands_with_"
            "exact_Q_gradient_factor_2",
            "FD_keeps_spectral_rho_and_coordinatewise_diagonal_d_with_embedded_"
            "single_charge_semantics",
            "solver_keeps_physical_resource_endpoint_and_quadratic_response_"
            "intervals_in_native_units",
            "branch_endpoint_motion_requires_self_consistent_B_le_8_robust_"
            "variable_shell_resolve_with_outside_strictness_and_categorical_"
            "inside_shell_degeneracy_or_full_KKT_strong_regularity",
        )


PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS: Final = (
    V1PhaseAEndogenousTypedSourceRequirementSet()
)
PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES: Final = (
    PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS.typed_supply_axis
    + (
        "role_norm_and_unit_compatible_motion_to_each_decision_changing_"
        "resource_exit_Q_support_or_action_face",
        "outward_FP64_reach_lower_bound_to_every_noninvariant_boundary",
        "complete_continuum_coverage_between_registered_source_images",
    )
)
PHASE_A_ENDOGENOUS_REACH_REASON: Final = (
    "the complete categorical action quotient is already source bound, but the "
    "legacy four-scalar owner axis is provenance only: corrected A/F has zero "
    "Phase-A causal sensitivity; literal-J MJz and Q, FD rho and d, and solver "
    "resource/Q intervals have different native units and norms. The missing "
    "positive theorem is the branch-endpoint/Q-motion map, which may be issued "
    "only by a self-consistent B<=8 robust variable-shell re-solve using the "
    "same-capture covariance supports, Loewner Q envelopes, strict outside-"
    "shell no-entry/no-exit screening, categorical closure of within-shell "
    "degeneracy, and fixed-point expansion (or refusal at the cap), or by a "
    "full KKT strong-regularity proof. Same-action tie and byte-alias faces "
    "remain categorical and require no scalar margin"
)


__all__ = [
    "PHASE_A_BRANCH_ENDPOINT_MOTION_REQUIREMENT_SCHEMA",
    "PHASE_A_ENDOGENOUS_REACH_API_NAME",
    "PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE",
    "PHASE_A_ENDOGENOUS_REACH_REASON",
    "PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES",
    "PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENT_SCHEMA",
    "PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS",
    "V1PhaseABranchEndpointMotionPositiveProducer",
    "V1PhaseABranchEndpointQMotionMapRequirement",
    "V1PhaseARobustVariableShellResolveRequirement",
    "V1PhaseACenteredIntervalEnclosure",
    "V1PhaseACorrectedAFBandSourceRequirement",
    "V1PhaseAEndogenousReachSourceRefusal",
    "V1PhaseAEndogenousReachSourceRefusalCode",
    "V1PhaseAEndogenousTypedSourceRequirementSet",
    "V1PhaseAFDResidualEnvelopeSourceRequirement",
    "V1PhaseALiteralJBandSourceRequirement",
    "V1PhaseANativeScalarUnit",
    "V1PhaseASolverNativeIntervalSourceRequirement",
    "V1PhaseASourceNorm",
]
