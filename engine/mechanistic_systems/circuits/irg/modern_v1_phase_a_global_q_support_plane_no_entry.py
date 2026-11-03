"""Exact small-domain reference proof for global true-``Q`` no-entry.

For a positive-semidefinite literal-J response matrix ``M`` and any frozen
anchor ``z0``, convexity gives the global support inequality

``Q(z) >= Q(z0) + 2 (M z0)^T (z - z0)``.

Consequently every ``Q(z) <= tau_Q`` action obeys

``(2 M z0)^T z <= tau_Q + Q(z0)``.

When removal gates are nonnegative, a simultaneous coordinatewise lower
enclosure of ``2 M z0`` and an upper enclosure of the right-hand side produce
an *outer* half-space.  Intersecting any number of those half-spaces with the
irreversible gate box and disjoint one-live-survivor capacities therefore
contains every true-Q-feasible action.  Maximizing pruning resource over that
polytope gives a sound global resource upper bound; it does not freeze a set
of "outside" routes and so covers swaps and coalitions.

This module is intentionally a CPU, exact-rational, small-domain theorem
kernel.  Its caller supplies the already-enclosed rows explicitly, and the
result has ``production_source_authority=False``.  It neither constructs the
rows from CUDA D-stream evidence nor grants D stability, F, C, K/O, commit,
runtime, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import itertools
import math
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash


PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_SCHEMA: Final = (
    "partition_free_v1_phase_A_global_true_Q_support_plane_no_entry_reference_v1"
)
PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_AUTHORITY: Final = (
    "exact_private_small_domain_outer_LP_reference_only_no_production_source_D_F_"
    "C_K_O_commit_terminal_runtime_or_pareto_authority_v1"
)
PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_ALGORITHM: Final = (
    "exact_fraction_vertex_enumeration_and_dual_normal_cone_certificate_v1"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_SCHEMA_V2: Final = (
    "partition_free_v1_phase_A_global_true_Q_outside_face_invariance_reference_v2"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_AUTHORITY_V2: Final = (
    "exact_private_small_domain_outside_face_invariance_premise_only_local_shell_"
    "join_required_no_production_D_F_C_K_O_commit_runtime_or_pareto_authority_v2"
)
PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_ALGORITHM_V2: Final = (
    "exact_fraction_full_and_outside_fixed_face_vertex_enumeration_with_averaged_"
    "exact_basic_full_duals_at_face_optimizer_sufficient_not_max_margin_v2"
)
_PREFIX: Final = "partition-free-v1-phase-A-global-Q-support-plane-no-entry-"
_PLANE_TOKEN: Final = object()
_BLOCK_TOKEN: Final = object()
_PROBLEM_TOKEN: Final = object()
_DESIGN_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_FIXED_POINT_TOKEN: Final = object()
_OUTSIDE_FACE_REGISTRATION_TOKEN: Final = object()
_OUTSIDE_FACE_RECEIPT_TOKEN: Final = object()
_MAXIMUM_REFERENCE_DIMENSION: Final = 8
_MAXIMUM_REFERENCE_CONSTRAINTS: Final = 64


class V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_IDENTITY = "invalid_identity"
    AXIS_MISMATCH = "axis_mismatch"
    INVALID_OUTER_HALFSPACE = "invalid_outer_halfspace"
    INVALID_CAPACITY = "invalid_capacity"
    INCUMBENT_INFEASIBLE = "incumbent_infeasible"
    ENGINEERING_LIMIT = "engineering_limit"
    EXACT_SOLVER_FAILURE = "exact_solver_failure"
    INVALID_SHELL_REGISTRATION = "invalid_shell_registration"
    OUTSIDE_FACE_RESOURCE_GAP = "outside_face_resource_gap"
    OUTSIDE_REDUCED_ADVANTAGE_NOT_STRICT = (
        "outside_reduced_advantage_not_strict"
    )
    OUTWARD_ENCLOSURE_ERASED_STRICTNESS = (
        "outward_enclosure_erased_strictness"
    )
    STALE_OR_MUTATED = "stale_or_mutated"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQSupportPlaneNoEntryRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode:
            raise TypeError("global-Q no-entry refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAGlobalQSupportPlaneNoEntryRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_IDENTITY,
            f"{name} must be one nonempty exact string",
        )
    return value


def _finite_binary64(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            f"{name} must be one finite exact binary64 value",
        )
    return value


def _nonnegative_binary64(value: object, *, name: str) -> float:
    result = _finite_binary64(value, name=name)
    if result < 0.0:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or type(value) is not int or value < minimum:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            f"{name} must be an exact integer >= {minimum}",
        )
    return value


def _float_down(value: Fraction) -> float:
    try:
        result = float(value)
    except OverflowError:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "exact endpoint cannot be enclosed by a finite binary64 value",
        )
    if not math.isfinite(result):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "exact endpoint cannot be enclosed by a finite binary64 value",
        )
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _float_up(value: Fraction) -> float:
    try:
        result = float(value)
    except OverflowError:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "exact endpoint cannot be enclosed by a finite binary64 value",
        )
    if not math.isfinite(result):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "exact endpoint cannot be enclosed by a finite binary64 value",
        )
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _payload_hash(payload: dict[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _fraction_pair(value: Fraction) -> tuple[int, int]:
    """Return the canonical, exactly replayable numerator/denominator pair."""

    return (value.numerator, value.denominator)


def _validate_fraction_point(
    point: object,
    *,
    dimension: int,
    name: str,
) -> tuple[tuple[int, int], ...]:
    if type(point) is not tuple or len(point) != dimension:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
            f"{name} must contain exactly one exact fraction pair per route",
        )
    for pair in point:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or isinstance(pair[0], bool)
            or type(pair[0]) is not int
            or isinstance(pair[1], bool)
            or type(pair[1]) is not int
            or pair[1] <= 0
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                f"{name} contains a noncanonical exact fraction pair",
            )
    return point


class V1PhaseAGlobalQCoordinateState(str, Enum):
    """Exact current-chart bounds for one irreversible removal coordinate."""

    ELIGIBLE_LIVE = "eligible_live_free_in_0_1"
    INELIGIBLE_LIVE = "ineligible_live_fixed_zero"
    ALREADY_REMOVED = "already_removed_fixed_zero"
    REQUIRED_LIVE = "required_live_fixed_zero"


class V1PhaseAGlobalQReducedAdvantageClassification(str, Enum):
    STRICT_SELECTED_POSITIVE = "strict_selected_positive"
    STRICT_EXCLUDED_NEGATIVE = "strict_excluded_negative"
    TOUCHING_ZERO = "touching_zero_nonpositive_certificate"
    MISALIGNED_WITH_INCUMBENT = "misaligned_with_incumbent_nonpositive_certificate"
    DELEGATED_TO_LOCAL_SHELL = "delegated_to_local_shell_not_sign_tested"
    FIXED_INELIGIBLE_LIVE = "fixed_ineligible_live"
    FIXED_ALREADY_REMOVED = "fixed_already_removed"
    FIXED_REQUIRED_LIVE = "fixed_required_live"


class V1PhaseAGlobalQNoEntryStatus(str, Enum):
    CERTIFIED_STRICT_GLOBAL_OUTER_OPTIMUM = (
        "certified_strict_global_outer_LP_optimum_reference_only"
    )
    INCUMBENT_BELOW_GLOBAL_OUTER_BOUND = (
        "incumbent_below_global_outer_LP_bound_nonpositive"
    )
    TOUCHING_OR_AMBIGUOUS_REDUCED_ADVANTAGE = (
        "touching_or_ambiguous_reduced_advantage_nonpositive"
    )
    DUAL_MISALIGNED_WITH_INCUMBENT = (
        "dual_reduced_advantage_misaligned_with_incumbent_nonpositive"
    )


class V1PhaseAOutsideShellFixedPointStatus(str, Enum):
    CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE = (
        "certified_outside_strict_shell_image_closed_reference_only"
    )
    OUTSIDE_TOUCHING_OR_MISALIGNED = "outside_touching_or_misaligned_nonpositive"
    SHELL_IMAGE_NOT_CLOSED_OR_MISALIGNED = (
        "shell_action_image_not_closed_or_misaligned_nonpositive"
    )
    PRIVATE_NO_ENTRY_NOT_SOLVED = "private_global_q_no_entry_reference_not_solved"


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlane:
    """One preregistered outward support plane for the true-Q sublevel set."""

    anchor_identity_hash: str
    coefficient_lower: tuple[float, ...]
    rhs_upper: float
    tangent_factor: int = 2
    coefficient_is_simultaneous_coordinatewise_lower: bool = True
    rhs_is_simultaneous_upper: bool = True
    psd_true_q_support_identity_attested: bool = True
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLANE_TOKEN:
            raise PermissionError("global-Q support planes require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        _identity(self.anchor_identity_hash, name="anchor_identity_hash")
        if type(self.coefficient_lower) is not tuple or not self.coefficient_lower:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "support coefficients must be one nonempty immutable tuple",
            )
        for value in self.coefficient_lower:
            _finite_binary64(value, name="support coefficient lower endpoint")
        _nonnegative_binary64(self.rhs_upper, name="support rhs upper endpoint")
        if (
            self.tangent_factor != 2
            or type(self.tangent_factor) is not int
            or self.coefficient_is_simultaneous_coordinatewise_lower is not True
            or self.rhs_is_simultaneous_upper is not True
            or self.psd_true_q_support_identity_attested is not True
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_OUTER_HALFSPACE,
                "a support plane must attest c_L <= 2Mz0 and rhs_U >= tau_Q+Q(z0)",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "anchor_identity_hash": self.anchor_identity_hash,
            "coefficient_lower": self.coefficient_lower,
            "rhs_upper": self.rhs_upper,
            "tangent_factor": self.tangent_factor,
            "coefficient_is_simultaneous_coordinatewise_lower": (
                self.coefficient_is_simultaneous_coordinatewise_lower
            ),
            "rhs_is_simultaneous_upper": self.rhs_is_simultaneous_upper,
            "psd_true_q_support_identity_attested": (
                self.psd_true_q_support_identity_attested
            ),
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="support-plane")

    @property
    def support_plane_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "global-Q support plane no longer replays",
            )
        return observed


def register_v1_phase_a_global_q_support_plane(
    *,
    anchor_identity_hash: str,
    coefficient_lower: Sequence[float],
    rhs_upper: float,
) -> V1PhaseAGlobalQSupportPlane:
    return V1PhaseAGlobalQSupportPlane(
        anchor_identity_hash=anchor_identity_hash,
        coefficient_lower=tuple(coefficient_lower),
        rhs_upper=rhs_upper,
        _issuance_token=_PLANE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOneLiveSurvivorCapacity:
    """One disjoint native block constraint ``sum(z_j) <= capacity``."""

    block_identity_hash: str
    coordinate_indices: tuple[int, ...]
    removal_capacity: float
    one_live_survivor_semantics: bool = True
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _BLOCK_TOKEN:
            raise PermissionError("global-Q block capacities require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        _identity(self.block_identity_hash, name="block_identity_hash")
        if type(self.coordinate_indices) is not tuple or not self.coordinate_indices:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "block coordinate indices must be one nonempty immutable tuple",
            )
        for index in self.coordinate_indices:
            _exact_int(index, name="block coordinate index")
        if tuple(sorted(set(self.coordinate_indices))) != self.coordinate_indices:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_CAPACITY,
                "block coordinate indices must be sorted and unique",
            )
        capacity = _nonnegative_binary64(
            self.removal_capacity,
            name="block removal capacity",
        )
        if capacity > float(len(self.coordinate_indices)) or not capacity.is_integer():
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_CAPACITY,
                "one-live-survivor removal capacity must be an exact integer count",
            )
        if self.one_live_survivor_semantics is not True:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "block constraint lost one-live-survivor semantics",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "block_identity_hash": self.block_identity_hash,
            "coordinate_indices": self.coordinate_indices,
            "removal_capacity": self.removal_capacity,
            "one_live_survivor_semantics": self.one_live_survivor_semantics,
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="block-capacity")

    @property
    def block_capacity_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "global-Q block capacity no longer replays",
            )
        return observed


def register_v1_phase_a_global_q_one_live_survivor_capacity(
    *,
    block_identity_hash: str,
    coordinate_indices: Sequence[int],
    removal_capacity: float,
) -> V1PhaseAGlobalQOneLiveSurvivorCapacity:
    return V1PhaseAGlobalQOneLiveSurvivorCapacity(
        block_identity_hash=block_identity_hash,
        coordinate_indices=tuple(coordinate_indices),
        removal_capacity=removal_capacity,
        _issuance_token=_BLOCK_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
    """Explicit private arrays for the exact small-domain reference theorem."""

    problem_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    support_family_registration_hash: str
    true_q_geometry_registration_hash: str
    route_identity_hashes: tuple[str, ...]
    coordinate_states: tuple[V1PhaseAGlobalQCoordinateState, ...]
    resource_coefficients: tuple[float, ...]
    support_planes: tuple[V1PhaseAGlobalQSupportPlane, ...]
    block_capacities: tuple[V1PhaseAGlobalQOneLiveSurvivorCapacity, ...]
    incumbent_selected: tuple[bool, ...]
    support_planes_preregistered: bool = True
    caller_supplied_private_reference_arrays: bool = True
    production_source_authority: bool = False
    downstream_authority: bool = False
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PROBLEM_TOKEN:
            raise PermissionError("global-Q private problems require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "problem_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "support_family_registration_hash",
            "true_q_geometry_registration_hash",
        ):
            _identity(getattr(self, name), name=name)
        if type(self.route_identity_hashes) is not tuple:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "route identities must be one immutable tuple",
            )
        dimension = len(self.route_identity_hashes)
        if not 1 <= dimension <= _MAXIMUM_REFERENCE_DIMENSION:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
                "the exact reference problem requires 1 <= dimension <= 8",
            )
        for route_identity in self.route_identity_hashes:
            _identity(route_identity, name="route identity hash")
        if len(set(self.route_identity_hashes)) != dimension:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "route identities must be unique",
            )
        axes = (
            self.coordinate_states,
            self.resource_coefficients,
            self.incumbent_selected,
        )
        if any(type(axis) is not tuple or len(axis) != dimension for axis in axes):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "coordinate state, resource, incumbent, and route axes must align",
            )
        for state in self.coordinate_states:
            if type(state) is not V1PhaseAGlobalQCoordinateState:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                    "coordinate states must use the exact state enum",
                )
        if V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE not in self.coordinate_states:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "a no-entry reference problem requires at least one eligible live route",
            )
        for resource in self.resource_coefficients:
            _nonnegative_binary64(resource, name="resource coefficient")
        for selected, state in zip(
            self.incumbent_selected,
            self.coordinate_states,
            strict=True,
        ):
            if type(selected) is not bool:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                    "incumbent selection values must be exact booleans",
                )
            if selected and state is not V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INCUMBENT_INFEASIBLE,
                    "the incumbent may select only eligible live coordinates",
                )
        if type(self.support_planes) is not tuple or not self.support_planes:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "at least one preregistered support plane is required",
            )
        anchor_ids: list[str] = []
        for plane in self.support_planes:
            if type(plane) is not V1PhaseAGlobalQSupportPlane:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                    "support planes must use the exact issued type",
                )
            plane.support_plane_hash
            if len(plane.coefficient_lower) != dimension:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                    "support plane coefficient axis does not match routes",
                )
            anchor_ids.append(plane.anchor_identity_hash)
        if len(set(anchor_ids)) != len(anchor_ids):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_OUTER_HALFSPACE,
                "support anchor identities must be unique",
            )
        if type(self.block_capacities) is not tuple:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "block capacities must be one immutable tuple",
            )
        seen_coordinates: set[int] = set()
        seen_blocks: set[str] = set()
        for block in self.block_capacities:
            if type(block) is not V1PhaseAGlobalQOneLiveSurvivorCapacity:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                    "block capacities must use the exact issued type",
                )
            block.block_capacity_hash
            if block.block_identity_hash in seen_blocks:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_CAPACITY,
                    "block identities must be unique",
                )
            seen_blocks.add(block.block_identity_hash)
            if any(index >= dimension for index in block.coordinate_indices):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                    "block coordinate lies outside the route axis",
                )
            if seen_coordinates.intersection(block.coordinate_indices):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_CAPACITY,
                    "one-live-survivor block capacities must be disjoint",
                )
            seen_coordinates.update(block.coordinate_indices)
        if (
            self.support_planes_preregistered is not True
            or self.caller_supplied_private_reference_arrays is not True
            or self.production_source_authority is not False
            or self.downstream_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "private reference inputs crossed their source or downstream authority",
            )
        incumbent = tuple(Fraction(int(value)) for value in self.incumbent_selected)
        for plane in self.support_planes:
            lhs = sum(
                Fraction.from_float(coefficient) * gate
                for coefficient, gate in zip(
                    plane.coefficient_lower,
                    incumbent,
                    strict=True,
                )
            )
            if lhs > Fraction.from_float(plane.rhs_upper):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INCUMBENT_INFEASIBLE,
                    "incumbent violates an outward support half-space",
                )
        for block in self.block_capacities:
            lhs = sum(incumbent[index] for index in block.coordinate_indices)
            if lhs > Fraction.from_float(block.removal_capacity):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INCUMBENT_INFEASIBLE,
                    "incumbent violates a one-live-survivor block capacity",
                )

    def _payload(self) -> dict[str, object]:
        return {
            "problem_id": self.problem_id,
            "model_identity_hash": self.model_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "support_family_registration_hash": self.support_family_registration_hash,
            "true_q_geometry_registration_hash": self.true_q_geometry_registration_hash,
            "route_identity_hashes": self.route_identity_hashes,
            "coordinate_states": tuple(value.value for value in self.coordinate_states),
            "resource_coefficients": self.resource_coefficients,
            "support_planes": tuple(
                plane.support_plane_hash for plane in self.support_planes
            ),
            "block_capacities": tuple(
                block.block_capacity_hash for block in self.block_capacities
            ),
            "incumbent_selected": self.incumbent_selected,
            "support_planes_preregistered": self.support_planes_preregistered,
            "caller_supplied_private_reference_arrays": (
                self.caller_supplied_private_reference_arrays
            ),
            "production_source_authority": self.production_source_authority,
            "downstream_authority": self.downstream_authority,
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="private-problem")

    @property
    def problem_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "global-Q private problem no longer replays",
            )
        return observed


def issue_v1_phase_a_global_q_support_plane_no_entry_private_problem(
    *,
    problem_id: str,
    model_identity_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    support_family_registration_hash: str,
    true_q_geometry_registration_hash: str,
    route_identity_hashes: Sequence[str],
    coordinate_states: Sequence[V1PhaseAGlobalQCoordinateState],
    resource_coefficients: Sequence[float],
    support_planes: Sequence[V1PhaseAGlobalQSupportPlane],
    incumbent_selected: Sequence[bool],
    block_capacities: Sequence[V1PhaseAGlobalQOneLiveSurvivorCapacity] = (),
) -> V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
    return V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem(
        problem_id=problem_id,
        model_identity_hash=model_identity_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        support_family_registration_hash=support_family_registration_hash,
        true_q_geometry_registration_hash=true_q_geometry_registration_hash,
        route_identity_hashes=tuple(route_identity_hashes),
        coordinate_states=tuple(coordinate_states),
        resource_coefficients=tuple(resource_coefficients),
        support_planes=tuple(support_planes),
        block_capacities=tuple(block_capacities),
        incumbent_selected=tuple(incumbent_selected),
        _issuance_token=_PROBLEM_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlaneNoEntryDesign:
    maximum_dimension: int = _MAXIMUM_REFERENCE_DIMENSION
    maximum_total_inequalities: int = _MAXIMUM_REFERENCE_CONSTRAINTS
    maximum_basis_combinations: int = 2_000_000
    exact_fraction_vertex_enumeration: bool = True
    private_reference_only: bool = True
    production_source_authority: bool = False
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("global-Q no-entry designs require their issuer")
        _exact_int(self.maximum_dimension, name="maximum_dimension", minimum=1)
        _exact_int(
            self.maximum_total_inequalities,
            name="maximum_total_inequalities",
            minimum=1,
        )
        _exact_int(
            self.maximum_basis_combinations,
            name="maximum_basis_combinations",
            minimum=1,
        )
        if (
            self.maximum_dimension > _MAXIMUM_REFERENCE_DIMENSION
            or self.maximum_total_inequalities > _MAXIMUM_REFERENCE_CONSTRAINTS
            or self.exact_fraction_vertex_enumeration is not True
            or self.private_reference_only is not True
            or self.production_source_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "reference design crossed its exact small-domain boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="design")

    @property
    def design_hash(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "global-Q no-entry design no longer replays",
            )
        return observed


def register_v1_phase_a_global_q_support_plane_no_entry_design(
    *,
    maximum_basis_combinations: int = 2_000_000,
) -> V1PhaseAGlobalQSupportPlaneNoEntryDesign:
    return V1PhaseAGlobalQSupportPlaneNoEntryDesign(
        maximum_basis_combinations=maximum_basis_combinations,
        _issuance_token=_DESIGN_TOKEN,
    )


def _resource_coordinate_content_hash(
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
) -> str:
    return _payload_hash(
        {
            "problem_hash": problem.problem_hash,
            "route_axis_hash": problem.route_axis_hash,
            "route_identity_hashes": problem.route_identity_hashes,
            "coordinate_states": tuple(
                state.value for state in problem.coordinate_states
            ),
            "resource_coefficients": problem.resource_coefficients,
        },
        role="outside-face-resource-coordinate",
    )


def _outside_membership_hash(
    *,
    problem_hash: str,
    route_axis_hash: str,
    coordinate_indices: tuple[int, ...],
    route_identity_hashes: tuple[str, ...],
    incumbent_selected: tuple[bool, ...],
) -> str:
    return _payload_hash(
        {
            "problem_hash": problem_hash,
            "route_axis_hash": route_axis_hash,
            "coordinate_indices": coordinate_indices,
            "route_identity_hashes": route_identity_hashes,
            "incumbent_selected": incumbent_selected,
        },
        role="outside-face-incumbent-membership",
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceRegistrationV2:
    """Issuer-only frozen shell/outside-face registration for the v2 theorem.

    The private source object is retained solely to enforce exact object
    ancestry at solve time.  It is deliberately excluded from the stable
    registration hash; every source value that affects the theorem is already
    sealed by ``problem_hash`` and the explicit coordinate hashes below.
    """

    problem_hash: str
    resource_coordinate_hash: str
    resource_coordinate_content_hash: str
    shell_freeze_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    shell_coordinate_indices: tuple[int, ...]
    shell_route_identity_hashes: tuple[str, ...]
    outside_eligible_coordinate_indices: tuple[int, ...]
    outside_eligible_route_identity_hashes: tuple[str, ...]
    outside_eligible_incumbent_selected: tuple[bool, ...]
    outside_selected_route_identity_hashes: tuple[str, ...]
    outside_excluded_route_identity_hashes: tuple[str, ...]
    outside_eligible_incumbent_membership_hash: str
    shell_preregistered_and_frozen: bool = True
    outside_membership_derived_from_exact_problem: bool = True
    local_shell_action_join_required: bool = True
    private_reference_only: bool = True
    production_source_authority: bool = False
    _source_problem_input: InitVar[object | None] = None
    _issuance_token: InitVar[object | None] = None
    _source_problem: object | None = field(init=False, default=None, repr=False)
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(
        self,
        _source_problem_input: object | None,
        _issuance_token: object | None,
    ) -> None:
        if _issuance_token is not _OUTSIDE_FACE_REGISTRATION_TOKEN:
            raise PermissionError("outside-face registrations require their issuer")
        object.__setattr__(self, "_source_problem", _source_problem_input)
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "problem_hash",
            "resource_coordinate_hash",
            "resource_coordinate_content_hash",
            "shell_freeze_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "outside_eligible_incumbent_membership_hash",
        ):
            _identity(getattr(self, name), name=name)
        source = self._source_problem
        if type(source) is not V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "outside-face registration lost its exact private source problem",
            )
        source_hash = source.problem_hash
        if (
            self.problem_hash != source_hash
            or self.model_identity_hash != source.model_identity_hash
            or self.chart_identity_hash != source.chart_identity_hash
            or self.current_mask_hash != source.current_mask_hash
            or self.route_axis_hash != source.route_axis_hash
            or self.resource_coordinate_content_hash
            != _resource_coordinate_content_hash(source)
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-face registration no longer matches its exact problem",
            )
        shell_axes = (
            self.shell_coordinate_indices,
            self.shell_route_identity_hashes,
        )
        if any(type(axis) is not tuple for axis in shell_axes):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "shell coordinates and route identities must be immutable tuples",
            )
        if not self.shell_coordinate_indices:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_SHELL_REGISTRATION,
                "the outside-face theorem requires a nonempty local shell",
            )
        if len(self.shell_coordinate_indices) > _MAXIMUM_REFERENCE_DIMENSION:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
                "registered shell exceeds the exact-reference dimension cap",
            )
        for index in self.shell_coordinate_indices:
            _exact_int(index, name="shell coordinate index")
        if tuple(sorted(set(self.shell_coordinate_indices))) != self.shell_coordinate_indices:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_SHELL_REGISTRATION,
                "shell coordinate indices must be sorted and unique",
            )
        if len(self.shell_route_identity_hashes) != len(self.shell_coordinate_indices):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "shell coordinate and route-identity axes must align",
            )
        dimension = len(source.route_identity_hashes)
        if any(index >= dimension for index in self.shell_coordinate_indices):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "registered shell coordinate lies outside the route axis",
            )
        expected_shell_routes = tuple(
            source.route_identity_hashes[index]
            for index in self.shell_coordinate_indices
        )
        if self.shell_route_identity_hashes != expected_shell_routes:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "shell route identities must exactly follow the sorted source indices",
            )
        if any(
            source.coordinate_states[index]
            is not V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
            for index in self.shell_coordinate_indices
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_SHELL_REGISTRATION,
                "every registered shell coordinate must be eligible and live",
            )
        shell_set = set(self.shell_coordinate_indices)
        expected_outside_indices = tuple(
            index
            for index, state in enumerate(source.coordinate_states)
            if state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
            and index not in shell_set
        )
        expected_outside_routes = tuple(
            source.route_identity_hashes[index] for index in expected_outside_indices
        )
        expected_membership = tuple(
            source.incumbent_selected[index] for index in expected_outside_indices
        )
        expected_selected = tuple(
            route
            for route, selected in zip(
                expected_outside_routes,
                expected_membership,
                strict=True,
            )
            if selected
        )
        expected_excluded = tuple(
            route
            for route, selected in zip(
                expected_outside_routes,
                expected_membership,
                strict=True,
            )
            if not selected
        )
        if (
            type(self.outside_eligible_coordinate_indices) is not tuple
            or type(self.outside_eligible_route_identity_hashes) is not tuple
            or type(self.outside_eligible_incumbent_selected) is not tuple
            or type(self.outside_selected_route_identity_hashes) is not tuple
            or type(self.outside_excluded_route_identity_hashes) is not tuple
            or self.outside_eligible_coordinate_indices != expected_outside_indices
            or self.outside_eligible_route_identity_hashes != expected_outside_routes
            or self.outside_eligible_incumbent_selected != expected_membership
            or self.outside_selected_route_identity_hashes != expected_selected
            or self.outside_excluded_route_identity_hashes != expected_excluded
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside eligible incumbent membership no longer matches the problem",
            )
        expected_membership_hash = _outside_membership_hash(
            problem_hash=source_hash,
            route_axis_hash=source.route_axis_hash,
            coordinate_indices=expected_outside_indices,
            route_identity_hashes=expected_outside_routes,
            incumbent_selected=expected_membership,
        )
        if self.outside_eligible_incumbent_membership_hash != expected_membership_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside eligible incumbent-membership hash no longer replays",
            )
        flags = (
            self.shell_preregistered_and_frozen,
            self.outside_membership_derived_from_exact_problem,
            self.local_shell_action_join_required,
            self.private_reference_only,
            self.production_source_authority,
        )
        if any(type(value) is not bool for value in flags) or (
            self.shell_preregistered_and_frozen is not True
            or self.outside_membership_derived_from_exact_problem is not True
            or self.local_shell_action_join_required is not True
            or self.private_reference_only is not True
            or self.production_source_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "outside-face registration crossed its frozen reference boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="outside-face-registration-v2")

    @property
    def registration_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-face registration no longer replays",
            )
        return observed

    def _assert_exact_problem(
        self,
        problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    ) -> None:
        self.registration_hash
        if problem is not self._source_problem:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-face theorem requires the exact registered problem object",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("outside-face registrations are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("outside-face registrations are nonserializable")


def register_v1_phase_a_global_q_outside_face_invariance_v2(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    resource_coordinate_hash: str,
    shell_freeze_hash: str,
    shell_coordinate_indices: Sequence[int],
    shell_route_identity_hashes: Sequence[str],
) -> V1PhaseAGlobalQOutsideFaceRegistrationV2:
    if type(problem) is not V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "outside-face registration requires the exact private problem type",
        )
    problem_hash = problem.problem_hash
    _identity(resource_coordinate_hash, name="resource_coordinate_hash")
    _identity(shell_freeze_hash, name="shell_freeze_hash")
    shell_indices = tuple(shell_coordinate_indices)
    shell_routes = tuple(shell_route_identity_hashes)
    if len(shell_indices) > _MAXIMUM_REFERENCE_DIMENSION:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "registered shell exceeds the exact-reference dimension cap",
        )
    for index in shell_indices:
        _exact_int(index, name="shell coordinate index")
    if tuple(sorted(set(shell_indices))) != shell_indices:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_SHELL_REGISTRATION,
            "shell coordinate indices must be sorted and unique",
        )
    shell_set = set(shell_indices)
    outside_indices = tuple(
        index
        for index, state in enumerate(problem.coordinate_states)
        if state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
        and index not in shell_set
    )
    outside_routes = tuple(
        problem.route_identity_hashes[index] for index in outside_indices
    )
    outside_membership = tuple(
        problem.incumbent_selected[index] for index in outside_indices
    )
    registration = V1PhaseAGlobalQOutsideFaceRegistrationV2(
        problem_hash=problem_hash,
        resource_coordinate_hash=resource_coordinate_hash,
        resource_coordinate_content_hash=_resource_coordinate_content_hash(problem),
        shell_freeze_hash=shell_freeze_hash,
        model_identity_hash=problem.model_identity_hash,
        chart_identity_hash=problem.chart_identity_hash,
        current_mask_hash=problem.current_mask_hash,
        route_axis_hash=problem.route_axis_hash,
        shell_coordinate_indices=shell_indices,
        shell_route_identity_hashes=shell_routes,
        outside_eligible_coordinate_indices=outside_indices,
        outside_eligible_route_identity_hashes=outside_routes,
        outside_eligible_incumbent_selected=outside_membership,
        outside_selected_route_identity_hashes=tuple(
            route
            for route, selected in zip(
                outside_routes,
                outside_membership,
                strict=True,
            )
            if selected
        ),
        outside_excluded_route_identity_hashes=tuple(
            route
            for route, selected in zip(
                outside_routes,
                outside_membership,
                strict=True,
            )
            if not selected
        ),
        outside_eligible_incumbent_membership_hash=_outside_membership_hash(
            problem_hash=problem_hash,
            route_axis_hash=problem.route_axis_hash,
            coordinate_indices=outside_indices,
            route_identity_hashes=outside_routes,
            incumbent_selected=outside_membership,
        ),
        _source_problem_input=problem,
        _issuance_token=_OUTSIDE_FACE_REGISTRATION_TOKEN,
    )
    registration.registration_hash
    return registration


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQReducedAdvantageRecord:
    route_identity_hash: str
    coordinate_state: V1PhaseAGlobalQCoordinateState
    incumbent_selected: bool
    reduced_advantage_lower: float
    reduced_advantage_upper: float
    classification: V1PhaseAGlobalQReducedAdvantageClassification

    def __post_init__(self) -> None:
        _identity(self.route_identity_hash, name="route_identity_hash")
        if type(self.coordinate_state) is not V1PhaseAGlobalQCoordinateState:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "reduced-advantage state must be exact",
            )
        if type(self.incumbent_selected) is not bool:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "reduced-advantage incumbent flag must be exact",
            )
        lower = _finite_binary64(
            self.reduced_advantage_lower,
            name="reduced advantage lower endpoint",
        )
        upper = _finite_binary64(
            self.reduced_advantage_upper,
            name="reduced advantage upper endpoint",
        )
        if lower > upper:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "reduced-advantage interval is reversed",
            )
        if type(self.classification) is not V1PhaseAGlobalQReducedAdvantageClassification:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "reduced-advantage classification must be exact",
            )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlaneNoEntryReceipt:
    problem_hash: str
    design_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    support_family_registration_hash: str
    true_q_geometry_registration_hash: str
    dimension: int
    support_plane_count: int
    block_capacity_count: int
    total_inequality_count: int
    enumerated_basis_count: int
    feasible_vertex_count: int
    exact_dual_basis_count: int
    global_resource_primal_lower_bound: float
    global_resource_dual_upper_bound: float
    incumbent_resource_lower_bound: float
    incumbent_resource_upper_bound: float
    exact_primal_dual_equality: bool
    incumbent_is_exact_global_outer_lp_optimum: bool
    strict_signed_reduced_advantage_margin_lower: float
    reduced_advantage_records: tuple[V1PhaseAGlobalQReducedAdvantageRecord, ...]
    status: V1PhaseAGlobalQNoEntryStatus
    global_true_q_outer_no_entry_reference_certified: bool
    all_support_planes_used_as_outer_halfspaces_only: bool = True
    exact_fraction_arithmetic: bool = True
    support_tangent_factor_two_enforced: bool = True
    box_bounds_enforced: bool = True
    disjoint_one_live_survivor_capacities_enforced: bool = True
    swaps_and_coalitions_in_global_outer_lp: bool = True
    typed_array_free_receipt: bool = True
    caller_supplied_private_reference_arrays: bool = True
    production_source_authority: bool = False
    d_stability_authority: bool = False
    f_authority: bool = False
    c_k_o_commit_terminal_runtime_pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("global-Q no-entry receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "problem_hash",
            "design_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "support_family_registration_hash",
            "true_q_geometry_registration_hash",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "dimension",
            "support_plane_count",
            "block_capacity_count",
            "total_inequality_count",
            "enumerated_basis_count",
            "feasible_vertex_count",
            "exact_dual_basis_count",
        ):
            minimum = 1 if name in {"dimension", "support_plane_count", "feasible_vertex_count", "exact_dual_basis_count"} else 0
            _exact_int(getattr(self, name), name=name, minimum=minimum)
        if self.dimension > _MAXIMUM_REFERENCE_DIMENSION:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
                "receipt exceeds the exact reference dimension cap",
            )
        for name in (
            "global_resource_primal_lower_bound",
            "global_resource_dual_upper_bound",
            "incumbent_resource_lower_bound",
            "incumbent_resource_upper_bound",
            "strict_signed_reduced_advantage_margin_lower",
        ):
            _finite_binary64(getattr(self, name), name=name)
        if (
            self.global_resource_primal_lower_bound
            > self.global_resource_dual_upper_bound
            or self.incumbent_resource_lower_bound
            > self.incumbent_resource_upper_bound
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
                "receipt contains a reversed outward resource interval",
            )
        if (
            type(self.reduced_advantage_records) is not tuple
            or len(self.reduced_advantage_records) != self.dimension
            or len({item.route_identity_hash for item in self.reduced_advantage_records})
            != self.dimension
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "reduced-advantage records must cover each route exactly once",
            )
        if type(self.status) is not V1PhaseAGlobalQNoEntryStatus:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "no-entry status must be exact",
            )
        flags = (
            self.exact_primal_dual_equality,
            self.incumbent_is_exact_global_outer_lp_optimum,
            self.global_true_q_outer_no_entry_reference_certified,
            self.all_support_planes_used_as_outer_halfspaces_only,
            self.exact_fraction_arithmetic,
            self.support_tangent_factor_two_enforced,
            self.box_bounds_enforced,
            self.disjoint_one_live_survivor_capacities_enforced,
            self.swaps_and_coalitions_in_global_outer_lp,
            self.typed_array_free_receipt,
            self.caller_supplied_private_reference_arrays,
            self.production_source_authority,
            self.d_stability_authority,
            self.f_authority,
            self.c_k_o_commit_terminal_runtime_pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "receipt authority and proof flags must be exact booleans",
            )
        positive = (
            self.status
            is V1PhaseAGlobalQNoEntryStatus.CERTIFIED_STRICT_GLOBAL_OUTER_OPTIMUM
        )
        if (
            self.exact_primal_dual_equality is not True
            or self.global_true_q_outer_no_entry_reference_certified != positive
            or positive and not self.incumbent_is_exact_global_outer_lp_optimum
            or positive and self.strict_signed_reduced_advantage_margin_lower <= 0.0
            or self.all_support_planes_used_as_outer_halfspaces_only is not True
            or self.exact_fraction_arithmetic is not True
            or self.support_tangent_factor_two_enforced is not True
            or self.box_bounds_enforced is not True
            or self.disjoint_one_live_survivor_capacities_enforced is not True
            or self.swaps_and_coalitions_in_global_outer_lp is not True
            or self.typed_array_free_receipt is not True
            or self.caller_supplied_private_reference_arrays is not True
            or self.production_source_authority is not False
            or self.d_stability_authority is not False
            or self.f_authority is not False
            or self.c_k_o_commit_terminal_runtime_pareto_authority is not False
            or self.algorithm != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "global-Q no-entry receipt crossed its proof or authority boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    @property
    def receipt_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "global-Q no-entry receipt no longer replays",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQOutsideFaceInvarianceReceiptV2:
    """Positive exact-reference premise that outside membership is invariant.

    This theorem deliberately says nothing about which in-shell action should
    be chosen.  The shell remains a free face and requires a separately
    source-bound local action join before any downstream D decision.
    """

    registration_hash: str
    problem_hash: str
    design_hash: str
    resource_coordinate_hash: str
    resource_coordinate_content_hash: str
    shell_freeze_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    shell_coordinate_indices: tuple[int, ...]
    shell_route_identity_hashes: tuple[str, ...]
    outside_eligible_coordinate_indices: tuple[int, ...]
    outside_eligible_route_identity_hashes: tuple[str, ...]
    outside_eligible_incumbent_selected: tuple[bool, ...]
    outside_eligible_incumbent_membership_hash: str
    dimension: int
    outside_eligible_count: int
    vacuous_outside: bool
    full_constraint_content_hash: str
    outside_fixed_face_constraint_content_hash: str
    full_total_inequality_count: int
    outside_fixed_face_total_inequality_count: int
    full_enumerated_basis_count: int
    outside_fixed_face_enumerated_basis_count: int
    full_feasible_vertex_count: int
    outside_fixed_face_feasible_vertex_count: int
    exact_full_dual_basis_count_at_face_optimizer: int
    support_plane_count: int
    block_capacity_count: int
    support_plane_dual_multipliers_exact: tuple[tuple[int, int], ...]
    block_capacity_dual_multipliers_exact: tuple[tuple[int, int], ...]
    structural_dual_content_hash: str
    full_optimal_vertex_exact: tuple[tuple[int, int], ...]
    outside_fixed_face_optimal_vertex_exact: tuple[tuple[int, int], ...]
    full_resource_exact: tuple[int, int]
    outside_fixed_face_resource_exact: tuple[int, int]
    full_resource_primal_lower_bound: float
    full_resource_dual_upper_bound: float
    outside_fixed_face_resource_primal_lower_bound: float
    outside_fixed_face_resource_upper_bound: float
    hard_incumbent_resource_lower_bound: float
    hard_incumbent_resource_upper_bound: float
    exact_full_primal_dual_equality: bool
    exact_outside_fixed_face_resource_enumeration: bool
    exact_full_outside_fixed_face_resource_equality: bool
    outside_fixed_face_optimizer_is_full_optimal: bool
    hard_incumbent_is_exact_global_outer_lp_optimum: bool
    strict_outside_signed_reduced_advantage_margin_lower: float
    reduced_advantage_records: tuple[V1PhaseAGlobalQReducedAdvantageRecord, ...]
    outside_face_invariance_reference_certified: bool = True
    local_shell_action_join_required: bool = True
    exact_fraction_constraints: bool = True
    exact_fraction_basis_enumeration: bool = True
    exact_fraction_vertices: bool = True
    all_exact_basic_full_duals_at_face_optimizer_averaged: bool = True
    sufficient_reference_not_complete_max_margin: bool = True
    swaps_and_coalitions_present_in_full_outer_lp: bool = True
    private_reference_only: bool = True
    production_source_authority: bool = False
    d_stability_authority: bool = False
    f_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_ALGORITHM_V2
    authority_scope: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_AUTHORITY_V2
    schema: str = PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_SCHEMA_V2
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _OUTSIDE_FACE_RECEIPT_TOKEN:
            raise PermissionError("outside-face invariance receipts require issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "registration_hash",
            "problem_hash",
            "design_hash",
            "resource_coordinate_hash",
            "resource_coordinate_content_hash",
            "shell_freeze_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "outside_eligible_incumbent_membership_hash",
            "full_constraint_content_hash",
            "outside_fixed_face_constraint_content_hash",
            "structural_dual_content_hash",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "dimension",
            "outside_eligible_count",
            "full_total_inequality_count",
            "outside_fixed_face_total_inequality_count",
            "full_enumerated_basis_count",
            "outside_fixed_face_enumerated_basis_count",
            "full_feasible_vertex_count",
            "outside_fixed_face_feasible_vertex_count",
            "exact_full_dual_basis_count_at_face_optimizer",
            "support_plane_count",
            "block_capacity_count",
        ):
            minimum = (
                0
                if name
                in {
                    "outside_eligible_count",
                    "support_plane_count",
                    "block_capacity_count",
                }
                else 1
            )
            _exact_int(getattr(self, name), name=name, minimum=minimum)
        if self.dimension > _MAXIMUM_REFERENCE_DIMENSION:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
                "outside-face receipt exceeds the exact reference dimension cap",
            )
        if (
            type(self.shell_coordinate_indices) is not tuple
            or type(self.shell_route_identity_hashes) is not tuple
            or type(self.outside_eligible_coordinate_indices) is not tuple
            or type(self.outside_eligible_route_identity_hashes) is not tuple
            or type(self.outside_eligible_incumbent_selected) is not tuple
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "outside-face receipt axes must be immutable tuples",
            )
        if (
            not self.shell_coordinate_indices
            or tuple(sorted(set(self.shell_coordinate_indices)))
            != self.shell_coordinate_indices
            or tuple(sorted(set(self.outside_eligible_coordinate_indices)))
            != self.outside_eligible_coordinate_indices
            or len(self.shell_coordinate_indices)
            != len(self.shell_route_identity_hashes)
            or len(self.outside_eligible_coordinate_indices)
            != self.outside_eligible_count
            or len(self.outside_eligible_route_identity_hashes)
            != self.outside_eligible_count
            or len(self.outside_eligible_incumbent_selected)
            != self.outside_eligible_count
            or any(type(value) is not bool for value in self.outside_eligible_incumbent_selected)
            or set(self.shell_coordinate_indices).intersection(
                self.outside_eligible_coordinate_indices
            )
            or set(self.shell_route_identity_hashes).intersection(
                self.outside_eligible_route_identity_hashes
            )
            or any(
                index < 0 or index >= self.dimension
                for index in (
                    self.shell_coordinate_indices
                    + self.outside_eligible_coordinate_indices
                )
            )
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "outside-face receipt shell/outside axes are incoherent",
            )
        for route_identity in (
            self.shell_route_identity_hashes
            + self.outside_eligible_route_identity_hashes
        ):
            _identity(route_identity, name="outside-face route identity hash")
        if type(self.vacuous_outside) is not bool or self.vacuous_outside != (
            self.outside_eligible_count == 0
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "vacuous-outside flag must exactly track the outside count",
            )
        _validate_fraction_point(
            self.full_optimal_vertex_exact,
            dimension=self.dimension,
            name="full optimal vertex",
        )
        _validate_fraction_point(
            self.outside_fixed_face_optimal_vertex_exact,
            dimension=self.dimension,
            name="outside-fixed-face optimal vertex",
        )
        dual_axes = (
            (
                self.support_plane_dual_multipliers_exact,
                self.support_plane_count,
                "support-plane dual multipliers",
            ),
            (
                self.block_capacity_dual_multipliers_exact,
                self.block_capacity_count,
                "block-capacity dual multipliers",
            ),
        )
        for axis, expected_count, name in dual_axes:
            if type(axis) is not tuple or len(axis) != expected_count:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                    f"{name} must cover its complete structural row axis",
                )
            for pair in axis:
                if (
                    type(pair) is not tuple
                    or len(pair) != 2
                    or isinstance(pair[0], bool)
                    or type(pair[0]) is not int
                    or isinstance(pair[1], bool)
                    or type(pair[1]) is not int
                    or pair[0] < 0
                    or pair[1] <= 0
                    or _fraction_pair(Fraction(pair[0], pair[1])) != pair
                ):
                    _refuse(
                        V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                        f"{name} contains a noncanonical nonnegative fraction",
                    )
        expected_dual_hash = _payload_hash(
            {
                "problem_hash": self.problem_hash,
                "support_plane_dual_multipliers_exact": (
                    self.support_plane_dual_multipliers_exact
                ),
                "block_capacity_dual_multipliers_exact": (
                    self.block_capacity_dual_multipliers_exact
                ),
            },
            role="outside-face-structural-dual-v2",
        )
        if self.structural_dual_content_hash != expected_dual_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-face structural dual witness no longer replays",
            )
        for name in ("full_resource_exact", "outside_fixed_face_resource_exact"):
            pair = getattr(self, name)
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or isinstance(pair[0], bool)
                or type(pair[0]) is not int
                or isinstance(pair[1], bool)
                or type(pair[1]) is not int
                or pair[1] <= 0
            ):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                    f"{name} must be one canonical exact fraction pair",
                )
        for name in (
            "full_resource_primal_lower_bound",
            "full_resource_dual_upper_bound",
            "outside_fixed_face_resource_primal_lower_bound",
            "outside_fixed_face_resource_upper_bound",
            "hard_incumbent_resource_lower_bound",
            "hard_incumbent_resource_upper_bound",
            "strict_outside_signed_reduced_advantage_margin_lower",
        ):
            _finite_binary64(getattr(self, name), name=name)
        if (
            self.full_resource_primal_lower_bound
            > self.full_resource_dual_upper_bound
            or self.outside_fixed_face_resource_primal_lower_bound
            > self.outside_fixed_face_resource_upper_bound
            or self.hard_incumbent_resource_lower_bound
            > self.hard_incumbent_resource_upper_bound
            or self.full_resource_exact != self.outside_fixed_face_resource_exact
            or self.full_resource_primal_lower_bound
            != self.outside_fixed_face_resource_primal_lower_bound
            or self.full_resource_dual_upper_bound
            != self.outside_fixed_face_resource_upper_bound
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
                "outside-face receipt lost exact full/face resource equality",
            )
        if self.vacuous_outside:
            if self.strict_outside_signed_reduced_advantage_margin_lower != 0.0:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                    "a vacuous outside must carry exactly zero strict margin",
                )
        elif self.strict_outside_signed_reduced_advantage_margin_lower <= 0.0:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "a nonvacuous outside requires a positive outward strict margin",
            )
        if (
            type(self.reduced_advantage_records) is not tuple
            or len(self.reduced_advantage_records) != self.dimension
            or len(
                {record.route_identity_hash for record in self.reduced_advantage_records}
            )
            != self.dimension
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "outside-face reduced-advantage records must cover every route once",
            )
        records_by_route = {
            record.route_identity_hash: record
            for record in self.reduced_advantage_records
        }
        registered_eligible_routes = set(self.shell_route_identity_hashes).union(
            self.outside_eligible_route_identity_hashes
        )
        if not registered_eligible_routes.issubset(records_by_route):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "outside-face records omit a registered shell or outside route",
            )
        if any(
            records_by_route[route].classification
            is not V1PhaseAGlobalQReducedAdvantageClassification.DELEGATED_TO_LOCAL_SHELL
            for route in self.shell_route_identity_hashes
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "every shell route must be delegated and never sign-tested",
            )
        for route, selected in zip(
            self.outside_eligible_route_identity_hashes,
            self.outside_eligible_incumbent_selected,
            strict=True,
        ):
            expected = (
                V1PhaseAGlobalQReducedAdvantageClassification.STRICT_SELECTED_POSITIVE
                if selected
                else V1PhaseAGlobalQReducedAdvantageClassification.STRICT_EXCLUDED_NEGATIVE
            )
            if records_by_route[route].classification is not expected:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                    "outside route lacks the required strict incumbent-aligned sign",
                )
        flags = (
            self.exact_full_primal_dual_equality,
            self.exact_outside_fixed_face_resource_enumeration,
            self.exact_full_outside_fixed_face_resource_equality,
            self.outside_fixed_face_optimizer_is_full_optimal,
            self.hard_incumbent_is_exact_global_outer_lp_optimum,
            self.outside_face_invariance_reference_certified,
            self.local_shell_action_join_required,
            self.exact_fraction_constraints,
            self.exact_fraction_basis_enumeration,
            self.exact_fraction_vertices,
            self.all_exact_basic_full_duals_at_face_optimizer_averaged,
            self.sufficient_reference_not_complete_max_margin,
            self.swaps_and_coalitions_present_in_full_outer_lp,
            self.private_reference_only,
            self.production_source_authority,
            self.d_stability_authority,
            self.f_authority,
            self.c_authority,
            self.k_authority,
            self.o_authority,
            self.commit_authority,
            self.runtime_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or (
            self.exact_full_primal_dual_equality is not True
            or self.exact_outside_fixed_face_resource_enumeration is not True
            or self.exact_full_outside_fixed_face_resource_equality is not True
            or self.outside_fixed_face_optimizer_is_full_optimal is not True
            or self.outside_face_invariance_reference_certified is not True
            or self.local_shell_action_join_required is not True
            or self.exact_fraction_constraints is not True
            or self.exact_fraction_basis_enumeration is not True
            or self.exact_fraction_vertices is not True
            or self.all_exact_basic_full_duals_at_face_optimizer_averaged is not True
            or self.sufficient_reference_not_complete_max_margin is not True
            or self.swaps_and_coalitions_present_in_full_outer_lp is not True
            or self.private_reference_only is not True
            or self.production_source_authority is not False
            or self.d_stability_authority is not False
            or self.f_authority is not False
            or self.c_authority is not False
            or self.k_authority is not False
            or self.o_authority is not False
            or self.commit_authority is not False
            or self.runtime_authority is not False
            or self.pareto_authority is not False
            or self.algorithm != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_ALGORITHM_V2
            or self.authority_scope
            != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_AUTHORITY_V2
            or self.schema != PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_SCHEMA_V2
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "outside-face invariance receipt crossed its premise-only boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="outside-face-receipt-v2")

    @property
    def receipt_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-face invariance receipt no longer replays",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1PhaseAOutsideShellFixedPointReferenceReceipt:
    """Weaker fixed-point theorem shape for the production no-entry sibling.

    The exact private global-Q theorem above requires strict two-sided reduced
    signs everywhere and therefore rejects harmless degeneracy on the compact
    shell.  Canonical sampled-v7 needs only strict membership stability outside
    the registered shell; inside-shell touching is delegated to the categorical
    resource-cell action image.  This receipt records that weaker algebraic
    decomposition without gaining production-source, D-stability, C/K/O,
    commit, runtime, terminal, or Pareto authority.
    """

    no_entry_receipt_hash: str
    route_axis_hash: str
    current_mask_hash: str
    shell_route_identity_hashes: tuple[str, ...]
    outside_route_identity_hashes: tuple[str, ...]
    shell_touching_route_identity_hashes: tuple[str, ...]
    shell_misaligned_route_identity_hashes: tuple[str, ...]
    outside_strict_signed_margin_lower: float
    shell_action_image_hash: str
    shell_action_image_closed: bool
    outside_membership_strictly_stable: bool
    status: V1PhaseAOutsideShellFixedPointStatus
    outside_no_entry_reference_certified: bool
    adaptive_shell_fixed_point_reference_certified: bool
    private_reference_only: bool = True
    production_source_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_terminal_runtime_pareto_authority: bool = False
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FIXED_POINT_TOKEN:
            raise PermissionError("outside-shell fixed-point receipts require issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "no_entry_receipt_hash",
            "route_axis_hash",
            "current_mask_hash",
            "shell_action_image_hash",
        ):
            _identity(getattr(self, name), name=name)
        axes = (
            self.shell_route_identity_hashes,
            self.outside_route_identity_hashes,
            self.shell_touching_route_identity_hashes,
            self.shell_misaligned_route_identity_hashes,
        )
        if any(type(axis) is not tuple for axis in axes):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "fixed-point route identity axes must be immutable tuples",
            )
        all_routes = self.shell_route_identity_hashes + self.outside_route_identity_hashes
        if not self.shell_route_identity_hashes or len(set(all_routes)) != len(all_routes):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                "fixed-point shell/outside route axes must be nonempty and disjoint",
            )
        for route_identity in all_routes:
            _identity(route_identity, name="fixed-point route identity hash")
        shell_set = set(self.shell_route_identity_hashes)
        for route_identity in (
            self.shell_touching_route_identity_hashes
            + self.shell_misaligned_route_identity_hashes
        ):
            _identity(route_identity, name="fixed-point shell exceptional route")
            if route_identity not in shell_set:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
                    "shell exceptional route lies outside the registered shell",
                )
        _finite_binary64(
            self.outside_strict_signed_margin_lower,
            name="outside strict signed margin lower endpoint",
        )
        if type(self.status) is not V1PhaseAOutsideShellFixedPointStatus:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "outside-shell fixed-point status must be exact",
            )
        flags = (
            self.shell_action_image_closed,
            self.outside_membership_strictly_stable,
            self.outside_no_entry_reference_certified,
            self.adaptive_shell_fixed_point_reference_certified,
            self.private_reference_only,
            self.production_source_authority,
            self.d_stability_authority,
            self.c_k_o_commit_terminal_runtime_pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
                "fixed-point proof and authority flags must be exact booleans",
            )
        positive = (
            self.status
            is V1PhaseAOutsideShellFixedPointStatus.CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE
        )
        if (
            self.outside_no_entry_reference_certified != positive
            or self.adaptive_shell_fixed_point_reference_certified != positive
            or positive and not self.shell_action_image_closed
            or positive and not self.outside_membership_strictly_stable
            or positive and self.outside_strict_signed_margin_lower < 0.0
            or self.private_reference_only is not True
            or self.production_source_authority is not False
            or self.d_stability_authority is not False
            or self.c_k_o_commit_terminal_runtime_pareto_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AUTHORITY_VIOLATION,
                "outside-shell fixed-point receipt crossed its reference boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="outside-shell-fixed-point")

    @property
    def receipt_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.STALE_OR_MUTATED,
                "outside-shell fixed-point receipt no longer replays",
            )
        return observed


def derive_v1_phase_a_outside_shell_fixed_point_reference(
    *,
    no_entry_receipt: V1PhaseAGlobalQSupportPlaneNoEntryReceipt,
    shell_route_identity_hashes: Sequence[str],
    shell_action_image_hash: str,
    shell_action_image_closed: bool,
) -> V1PhaseAOutsideShellFixedPointReferenceReceipt:
    """Derive the canonical weaker no-entry shape from the exact reference.

    This is still a private/reference derivation.  It accepts touching
    degeneracy only on the registered shell and only when the caller supplies a
    closed shell action image.  Any outside touching or sign misalignment keeps
    the fixed point unresolved.
    """

    if type(no_entry_receipt) is not V1PhaseAGlobalQSupportPlaneNoEntryReceipt:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "outside-shell fixed point requires the exact no-entry receipt",
        )
    no_entry_hash = no_entry_receipt.receipt_hash
    if type(shell_action_image_closed) is not bool:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "shell_action_image_closed must be an exact boolean",
        )
    _identity(shell_action_image_hash, name="shell_action_image_hash")
    shell = tuple(shell_route_identity_hashes)
    if not shell or len(set(shell)) != len(shell):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
            "shell route identity axis must be nonempty and unique",
        )
    records_by_route = {
        record.route_identity_hash: record
        for record in no_entry_receipt.reduced_advantage_records
    }
    if any(route not in records_by_route for route in shell):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.AXIS_MISMATCH,
            "shell route identity lies outside the no-entry route axis",
        )
    shell_set = set(shell)
    outside_records = tuple(
        record
        for record in no_entry_receipt.reduced_advantage_records
        if record.route_identity_hash not in shell_set
        and record.coordinate_state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
    )
    shell_records = tuple(records_by_route[route] for route in shell)
    touching_shell = tuple(
        record.route_identity_hash
        for record in shell_records
        if record.classification
        is V1PhaseAGlobalQReducedAdvantageClassification.TOUCHING_ZERO
    )
    misaligned_shell = tuple(
        record.route_identity_hash
        for record in shell_records
        if record.classification
        is V1PhaseAGlobalQReducedAdvantageClassification.MISALIGNED_WITH_INCUMBENT
    )
    outside_margins: list[float] = []
    outside_stable = True
    for record in outside_records:
        if (
            record.classification
            is V1PhaseAGlobalQReducedAdvantageClassification.STRICT_SELECTED_POSITIVE
        ):
            outside_margins.append(record.reduced_advantage_lower)
        elif (
            record.classification
            is V1PhaseAGlobalQReducedAdvantageClassification.STRICT_EXCLUDED_NEGATIVE
        ):
            outside_margins.append(-record.reduced_advantage_upper)
        else:
            outside_stable = False
            outside_margins.append(0.0)
    margin = min(outside_margins) if outside_margins else 0.0
    if not outside_stable:
        status = V1PhaseAOutsideShellFixedPointStatus.OUTSIDE_TOUCHING_OR_MISALIGNED
    elif not shell_action_image_closed or misaligned_shell:
        status = (
            V1PhaseAOutsideShellFixedPointStatus.SHELL_IMAGE_NOT_CLOSED_OR_MISALIGNED
        )
    elif (
        no_entry_receipt.exact_primal_dual_equality
        and not no_entry_receipt.production_source_authority
    ):
        status = (
            V1PhaseAOutsideShellFixedPointStatus.CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE
        )
    else:
        status = V1PhaseAOutsideShellFixedPointStatus.PRIVATE_NO_ENTRY_NOT_SOLVED
    positive = (
        status
        is V1PhaseAOutsideShellFixedPointStatus.CERTIFIED_OUTSIDE_STRICT_SHELL_IMAGE_CLOSED_REFERENCE
    )
    receipt = V1PhaseAOutsideShellFixedPointReferenceReceipt(
        no_entry_receipt_hash=no_entry_hash,
        route_axis_hash=no_entry_receipt.route_axis_hash,
        current_mask_hash=no_entry_receipt.current_mask_hash,
        shell_route_identity_hashes=shell,
        outside_route_identity_hashes=tuple(
            record.route_identity_hash for record in outside_records
        ),
        shell_touching_route_identity_hashes=touching_shell,
        shell_misaligned_route_identity_hashes=misaligned_shell,
        outside_strict_signed_margin_lower=margin,
        shell_action_image_hash=shell_action_image_hash,
        shell_action_image_closed=shell_action_image_closed,
        outside_membership_strictly_stable=outside_stable,
        status=status,
        outside_no_entry_reference_certified=positive,
        adaptive_shell_fixed_point_reference_certified=positive,
        _issuance_token=_FIXED_POINT_TOKEN,
    )
    receipt.receipt_hash
    return receipt


class _ConstraintKind(str, Enum):
    SUPPORT = "support"
    BLOCK = "block"
    UPPER = "upper"
    LOWER = "lower"


@dataclass(frozen=True, slots=True)
class _Constraint:
    kind: _ConstraintKind
    source_index: int
    row: tuple[Fraction, ...]
    rhs: Fraction


def _solve_square_exact(
    matrix: Sequence[Sequence[Fraction]],
    rhs: Sequence[Fraction],
) -> tuple[Fraction, ...] | None:
    dimension = len(rhs)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        return None
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(dimension):
        pivot = next(
            (row for row in range(column, dimension) if augmented[row][column]),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row],
                        augmented[column],
                        strict=True,
                    )
                ]
    return tuple(augmented[row][-1] for row in range(dimension))


def _dot(left: Sequence[Fraction], right: Sequence[Fraction]) -> Fraction:
    return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))


def _build_constraints(
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
) -> tuple[_Constraint, ...]:
    dimension = len(problem.route_identity_hashes)
    constraints: list[_Constraint] = []
    for index, plane in enumerate(problem.support_planes):
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.SUPPORT,
                source_index=index,
                row=tuple(Fraction.from_float(value) for value in plane.coefficient_lower),
                rhs=Fraction.from_float(plane.rhs_upper),
            )
        )
    for index, block in enumerate(problem.block_capacities):
        row = tuple(
            Fraction(1) if coordinate in block.coordinate_indices else Fraction(0)
            for coordinate in range(dimension)
        )
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.BLOCK,
                source_index=index,
                row=row,
                rhs=Fraction.from_float(block.removal_capacity),
            )
        )
    for coordinate, state in enumerate(problem.coordinate_states):
        row = tuple(
            Fraction(1) if index == coordinate else Fraction(0)
            for index in range(dimension)
        )
        upper = (
            Fraction(1)
            if state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
            else Fraction(0)
        )
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.UPPER,
                source_index=coordinate,
                row=row,
                rhs=upper,
            )
        )
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.LOWER,
                source_index=coordinate,
                row=tuple(-value for value in row),
                rhs=Fraction(0),
            )
        )
    return tuple(constraints)


def _build_outside_fixed_face_constraints(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    shell_coordinate_indices: tuple[int, ...],
) -> tuple[_Constraint, ...]:
    """Build ``P_S`` by fixing eligible outside gates to the hard incumbent.

    The structural rows are byte-for-byte the full problem rows.  Only the box
    endpoint opposite each incumbent-selected outside bound changes, so this
    is the exact intersection of ``P`` with the registered outside box face.
    """

    dimension = len(problem.route_identity_hashes)
    shell_set = set(shell_coordinate_indices)
    constraints: list[_Constraint] = []
    for index, plane in enumerate(problem.support_planes):
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.SUPPORT,
                source_index=index,
                row=tuple(
                    Fraction.from_float(value)
                    for value in plane.coefficient_lower
                ),
                rhs=Fraction.from_float(plane.rhs_upper),
            )
        )
    for index, block in enumerate(problem.block_capacities):
        row = tuple(
            Fraction(1) if coordinate in block.coordinate_indices else Fraction(0)
            for coordinate in range(dimension)
        )
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.BLOCK,
                source_index=index,
                row=row,
                rhs=Fraction.from_float(block.removal_capacity),
            )
        )
    for coordinate, state in enumerate(problem.coordinate_states):
        row = tuple(
            Fraction(1) if index == coordinate else Fraction(0)
            for index in range(dimension)
        )
        eligible_shell = (
            state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
            and coordinate in shell_set
        )
        eligible_outside = (
            state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
            and coordinate not in shell_set
        )
        if eligible_shell:
            upper = Fraction(1)
            lower_rhs = Fraction(0)
        elif eligible_outside and problem.incumbent_selected[coordinate]:
            upper = Fraction(1)
            lower_rhs = Fraction(-1)
        else:
            upper = Fraction(0)
            lower_rhs = Fraction(0)
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.UPPER,
                source_index=coordinate,
                row=row,
                rhs=upper,
            )
        )
        constraints.append(
            _Constraint(
                kind=_ConstraintKind.LOWER,
                source_index=coordinate,
                row=tuple(-value for value in row),
                rhs=lower_rhs,
            )
        )
    return tuple(constraints)


def _constraint_content_hash(
    constraints: tuple[_Constraint, ...],
    *,
    role: str,
) -> str:
    return _payload_hash(
        {
            "constraints": tuple(
                {
                    "kind": constraint.kind.value,
                    "source_index": constraint.source_index,
                    "row": tuple(_fraction_pair(value) for value in constraint.row),
                    "rhs": _fraction_pair(constraint.rhs),
                }
                for constraint in constraints
            )
        },
        role=role,
    )


def _is_feasible(
    point: tuple[Fraction, ...],
    constraints: tuple[_Constraint, ...],
) -> bool:
    return all(_dot(constraint.row, point) <= constraint.rhs for constraint in constraints)


def _enumerate_vertices(
    *,
    constraints: tuple[_Constraint, ...],
    dimension: int,
    resource: tuple[Fraction, ...],
    maximum_basis_combinations: int,
) -> tuple[
    tuple[Fraction, ...],
    Fraction,
    int,
    int,
]:
    basis_count = math.comb(len(constraints), dimension)
    if basis_count > maximum_basis_combinations:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "exact vertex basis enumeration exceeds the preregistered cap",
        )
    vertices: set[tuple[Fraction, ...]] = set()
    for basis in itertools.combinations(range(len(constraints)), dimension):
        point = _solve_square_exact(
            tuple(constraints[index].row for index in basis),
            tuple(constraints[index].rhs for index in basis),
        )
        if point is not None and _is_feasible(point, constraints):
            vertices.add(point)
    if not vertices:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "bounded outer polytope produced no exact vertex",
        )
    objective_by_vertex = {point: _dot(resource, point) for point in vertices}
    optimum = max(objective_by_vertex.values())
    optimal_vertices = tuple(
        sorted(point for point, value in objective_by_vertex.items() if value == optimum)
    )
    chosen = optimal_vertices[-1]
    return chosen, optimum, len(vertices), basis_count


def _dual_basis_candidates(
    *,
    point: tuple[Fraction, ...],
    constraints: tuple[_Constraint, ...],
    resource: tuple[Fraction, ...],
    optimum: Fraction,
) -> tuple[tuple[Fraction, ...], ...]:
    dimension = len(point)
    active = tuple(
        index
        for index, constraint in enumerate(constraints)
        if _dot(constraint.row, point) == constraint.rhs
    )
    candidates: set[tuple[Fraction, ...]] = set()
    for basis in itertools.combinations(active, dimension):
        # A_B^T y_B = resource.
        transposed = tuple(
            tuple(constraints[basis[column]].row[row] for column in range(dimension))
            for row in range(dimension)
        )
        multipliers = _solve_square_exact(transposed, resource)
        if multipliers is None or any(value < 0 for value in multipliers):
            continue
        full = [Fraction(0)] * len(constraints)
        for constraint_index, multiplier in zip(basis, multipliers, strict=True):
            full[constraint_index] = multiplier
        full_tuple = tuple(full)
        if any(
            sum(
                full_tuple[index] * constraints[index].row[coordinate]
                for index in range(len(constraints))
            )
            != resource[coordinate]
            for coordinate in range(dimension)
        ):
            continue
        if sum(
            full_tuple[index] * constraints[index].rhs
            for index in range(len(constraints))
        ) != optimum:
            continue
        candidates.add(full_tuple)
    if not candidates:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "no exact nonnegative dual basis certifies the enumerated optimum",
        )
    # Retain complete full vectors for exact verification.  The caller later
    # averages only their structural multipliers; convexity preserves dual
    # feasibility and often exposes strict complementarity hidden at vertices.
    return tuple(sorted(candidates))


def _average_structural_multipliers(
    *,
    dual_candidates: tuple[tuple[Fraction, ...], ...],
    structural_count: int,
) -> tuple[Fraction, ...]:
    count = Fraction(len(dual_candidates))
    return tuple(
        sum((candidate[index] for candidate in dual_candidates), Fraction(0)) / count
        for index in range(structural_count)
    )


def _structural_reduced_advantages(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    structural_multipliers: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    support_count = len(problem.support_planes)
    advantages: list[Fraction] = []
    for coordinate, resource_value in enumerate(problem.resource_coefficients):
        penalty = Fraction(0)
        for index, plane in enumerate(problem.support_planes):
            penalty += structural_multipliers[index] * Fraction.from_float(
                plane.coefficient_lower[coordinate]
            )
        for block_index, block in enumerate(problem.block_capacities):
            if coordinate in block.coordinate_indices:
                penalty += structural_multipliers[support_count + block_index]
        advantages.append(Fraction.from_float(resource_value) - penalty)
    return tuple(advantages)


def _structural_dual_upper(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    structural_multipliers: tuple[Fraction, ...],
    advantages: tuple[Fraction, ...],
) -> Fraction:
    support_count = len(problem.support_planes)
    result = sum(
        structural_multipliers[index] * Fraction.from_float(plane.rhs_upper)
        for index, plane in enumerate(problem.support_planes)
    )
    result += sum(
        structural_multipliers[support_count + index]
        * Fraction.from_float(block.removal_capacity)
        for index, block in enumerate(problem.block_capacities)
    )
    result += sum(
        max(Fraction(0), advantage)
        for state, advantage in zip(
            problem.coordinate_states,
            advantages,
            strict=True,
        )
        if state is V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE
    )
    return result


def _fixed_classification(
    state: V1PhaseAGlobalQCoordinateState,
) -> V1PhaseAGlobalQReducedAdvantageClassification:
    if state is V1PhaseAGlobalQCoordinateState.INELIGIBLE_LIVE:
        return V1PhaseAGlobalQReducedAdvantageClassification.FIXED_INELIGIBLE_LIVE
    if state is V1PhaseAGlobalQCoordinateState.ALREADY_REMOVED:
        return V1PhaseAGlobalQReducedAdvantageClassification.FIXED_ALREADY_REMOVED
    if state is V1PhaseAGlobalQCoordinateState.REQUIRED_LIVE:
        return V1PhaseAGlobalQReducedAdvantageClassification.FIXED_REQUIRED_LIVE
    raise AssertionError("eligible state is not fixed")


def solve_v1_phase_a_global_q_support_plane_no_entry_reference(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    design: V1PhaseAGlobalQSupportPlaneNoEntryDesign,
) -> V1PhaseAGlobalQSupportPlaneNoEntryReceipt:
    """Solve and certify the complete small-domain outer LP exactly."""

    if type(problem) is not V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "problem must use the exact issued private-reference type",
        )
    if type(design) is not V1PhaseAGlobalQSupportPlaneNoEntryDesign:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "design must use the exact issued type",
        )
    problem_hash = problem.problem_hash
    design_hash = design.design_hash
    dimension = len(problem.route_identity_hashes)
    if dimension > design.maximum_dimension:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "problem dimension exceeds the registered exact design",
        )
    constraints = _build_constraints(problem)
    if len(constraints) > design.maximum_total_inequalities:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "outer LP inequality count exceeds the registered exact design",
        )
    resource = tuple(Fraction.from_float(value) for value in problem.resource_coefficients)
    optimizer, optimum, feasible_vertex_count, enumerated_basis_count = _enumerate_vertices(
        constraints=constraints,
        dimension=dimension,
        resource=resource,
        maximum_basis_combinations=design.maximum_basis_combinations,
    )
    incumbent = tuple(Fraction(int(value)) for value in problem.incumbent_selected)
    incumbent_resource = _dot(resource, incumbent)
    incumbent_optimal = incumbent_resource == optimum
    dual_point = incumbent if incumbent_optimal else optimizer
    dual_candidates = _dual_basis_candidates(
        point=dual_point,
        constraints=constraints,
        resource=resource,
        optimum=optimum,
    )
    structural_count = len(problem.support_planes) + len(problem.block_capacities)
    structural_multipliers = _average_structural_multipliers(
        dual_candidates=dual_candidates,
        structural_count=structural_count,
    )
    advantages = _structural_reduced_advantages(
        problem=problem,
        structural_multipliers=structural_multipliers,
    )
    dual_upper = _structural_dual_upper(
        problem=problem,
        structural_multipliers=structural_multipliers,
        advantages=advantages,
    )
    if dual_upper != optimum:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "averaged structural dual failed exact primal-dual replay",
        )

    records: list[V1PhaseAGlobalQReducedAdvantageRecord] = []
    signed_margins: list[Fraction] = []
    has_touching = False
    has_misaligned = False
    for route_identity, state, selected, advantage in zip(
        problem.route_identity_hashes,
        problem.coordinate_states,
        problem.incumbent_selected,
        advantages,
        strict=True,
    ):
        if state is not V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE:
            classification = _fixed_classification(state)
        elif selected and advantage > 0:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.STRICT_SELECTED_POSITIVE
            )
            signed_margins.append(advantage)
        elif not selected and advantage < 0:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.STRICT_EXCLUDED_NEGATIVE
            )
            signed_margins.append(-advantage)
        elif advantage == 0:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.TOUCHING_ZERO
            )
            signed_margins.append(Fraction(0))
            has_touching = True
        else:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.MISALIGNED_WITH_INCUMBENT
            )
            signed_margins.append(advantage if selected else -advantage)
            has_misaligned = True
        records.append(
            V1PhaseAGlobalQReducedAdvantageRecord(
                route_identity_hash=route_identity,
                coordinate_state=state,
                incumbent_selected=selected,
                reduced_advantage_lower=_float_down(advantage),
                reduced_advantage_upper=_float_up(advantage),
                classification=classification,
            )
        )
    exact_margin = min(signed_margins) if signed_margins else Fraction(0)
    margin_lower = _float_down(exact_margin)
    outward_strict = margin_lower > 0.0
    if not incumbent_optimal:
        status = V1PhaseAGlobalQNoEntryStatus.INCUMBENT_BELOW_GLOBAL_OUTER_BOUND
    elif has_misaligned:
        status = V1PhaseAGlobalQNoEntryStatus.DUAL_MISALIGNED_WITH_INCUMBENT
    elif has_touching or not outward_strict:
        status = (
            V1PhaseAGlobalQNoEntryStatus.TOUCHING_OR_AMBIGUOUS_REDUCED_ADVANTAGE
        )
    else:
        status = (
            V1PhaseAGlobalQNoEntryStatus.CERTIFIED_STRICT_GLOBAL_OUTER_OPTIMUM
        )
    certified = (
        status
        is V1PhaseAGlobalQNoEntryStatus.CERTIFIED_STRICT_GLOBAL_OUTER_OPTIMUM
    )
    receipt = V1PhaseAGlobalQSupportPlaneNoEntryReceipt(
        problem_hash=problem_hash,
        design_hash=design_hash,
        model_identity_hash=problem.model_identity_hash,
        chart_identity_hash=problem.chart_identity_hash,
        current_mask_hash=problem.current_mask_hash,
        route_axis_hash=problem.route_axis_hash,
        support_family_registration_hash=problem.support_family_registration_hash,
        true_q_geometry_registration_hash=problem.true_q_geometry_registration_hash,
        dimension=dimension,
        support_plane_count=len(problem.support_planes),
        block_capacity_count=len(problem.block_capacities),
        total_inequality_count=len(constraints),
        enumerated_basis_count=enumerated_basis_count,
        feasible_vertex_count=feasible_vertex_count,
        exact_dual_basis_count=len(dual_candidates),
        global_resource_primal_lower_bound=_float_down(optimum),
        global_resource_dual_upper_bound=_float_up(dual_upper),
        incumbent_resource_lower_bound=_float_down(incumbent_resource),
        incumbent_resource_upper_bound=_float_up(incumbent_resource),
        exact_primal_dual_equality=True,
        incumbent_is_exact_global_outer_lp_optimum=incumbent_optimal,
        strict_signed_reduced_advantage_margin_lower=margin_lower,
        reduced_advantage_records=tuple(records),
        status=status,
        global_true_q_outer_no_entry_reference_certified=certified,
        _issuance_token=_RECEIPT_TOKEN,
    )
    receipt.receipt_hash
    return receipt


def solve_v1_phase_a_global_q_outside_face_invariance_reference_v2(
    *,
    problem: V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem,
    registration: V1PhaseAGlobalQOutsideFaceRegistrationV2,
    design: V1PhaseAGlobalQSupportPlaneNoEntryDesign,
) -> V1PhaseAGlobalQOutsideFaceInvarianceReceiptV2:
    """Certify exact full/face equality and strict *outside-only* signs.

    This is a sufficient exact private-reference theorem, not a complete
    maximum-margin search over the dual optimal face.  It averages every exact
    basic full-problem dual found at the chosen face optimizer.  Convexity
    preserves dual optimality; a strictly aligned outside sign in that average
    is sufficient.  Failure to expose such a sign refuses rather than claiming
    that no suitable dual exists.
    """

    if type(problem) is not V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "outside-face theorem requires the exact private problem type",
        )
    if type(registration) is not V1PhaseAGlobalQOutsideFaceRegistrationV2:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "outside-face theorem requires the exact issued registration type",
        )
    if type(design) is not V1PhaseAGlobalQSupportPlaneNoEntryDesign:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.INVALID_TYPE,
            "outside-face theorem requires the exact issued design type",
        )
    registration._assert_exact_problem(problem)
    registration_hash = registration.registration_hash
    design_hash = design.design_hash
    dimension = len(problem.route_identity_hashes)
    if dimension > design.maximum_dimension:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "outside-face problem exceeds the registered exact dimension",
        )
    full_constraints = _build_constraints(problem)
    face_constraints = _build_outside_fixed_face_constraints(
        problem=problem,
        shell_coordinate_indices=registration.shell_coordinate_indices,
    )
    if (
        len(full_constraints) > design.maximum_total_inequalities
        or len(face_constraints) > design.maximum_total_inequalities
    ):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.ENGINEERING_LIMIT,
            "full or outside-fixed-face LP exceeds the inequality cap",
        )
    resource = tuple(
        Fraction.from_float(value) for value in problem.resource_coefficients
    )
    (
        full_optimizer,
        full_optimum,
        full_vertex_count,
        full_basis_count,
    ) = _enumerate_vertices(
        constraints=full_constraints,
        dimension=dimension,
        resource=resource,
        maximum_basis_combinations=design.maximum_basis_combinations,
    )
    (
        face_optimizer,
        face_optimum,
        face_vertex_count,
        face_basis_count,
    ) = _enumerate_vertices(
        constraints=face_constraints,
        dimension=dimension,
        resource=resource,
        maximum_basis_combinations=design.maximum_basis_combinations,
    )
    if face_optimum != full_optimum:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.OUTSIDE_FACE_RESOURCE_GAP,
            "outside-fixed shell-free face does not intersect the full outer optimum",
        )
    if not _is_feasible(face_optimizer, full_constraints):
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "outside-fixed-face optimizer failed full-problem feasibility replay",
        )
    dual_candidates = _dual_basis_candidates(
        point=face_optimizer,
        constraints=full_constraints,
        resource=resource,
        optimum=full_optimum,
    )
    structural_count = len(problem.support_planes) + len(problem.block_capacities)
    structural_multipliers = _average_structural_multipliers(
        dual_candidates=dual_candidates,
        structural_count=structural_count,
    )
    advantages = _structural_reduced_advantages(
        problem=problem,
        structural_multipliers=structural_multipliers,
    )
    full_dual_upper = _structural_dual_upper(
        problem=problem,
        structural_multipliers=structural_multipliers,
        advantages=advantages,
    )
    if full_dual_upper != full_optimum:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "averaged exact basic duals failed full primal-dual replay",
        )

    shell_set = set(registration.shell_coordinate_indices)
    outside_set = set(registration.outside_eligible_coordinate_indices)
    records: list[V1PhaseAGlobalQReducedAdvantageRecord] = []
    exact_outside_margins: list[Fraction] = []
    for coordinate, (route_identity, state, selected, advantage) in enumerate(
        zip(
            problem.route_identity_hashes,
            problem.coordinate_states,
            problem.incumbent_selected,
            advantages,
            strict=True,
        )
    ):
        if state is not V1PhaseAGlobalQCoordinateState.ELIGIBLE_LIVE:
            classification = _fixed_classification(state)
        elif coordinate in shell_set:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.DELEGATED_TO_LOCAL_SHELL
            )
        elif coordinate in outside_set and selected and advantage > 0:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.STRICT_SELECTED_POSITIVE
            )
            exact_outside_margins.append(advantage)
        elif coordinate in outside_set and not selected and advantage < 0:
            classification = (
                V1PhaseAGlobalQReducedAdvantageClassification.STRICT_EXCLUDED_NEGATIVE
            )
            exact_outside_margins.append(-advantage)
        else:
            _refuse(
                V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.OUTSIDE_REDUCED_ADVANTAGE_NOT_STRICT,
                "an eligible outside route lacks a strict incumbent-aligned reduced sign",
            )
        records.append(
            V1PhaseAGlobalQReducedAdvantageRecord(
                route_identity_hash=route_identity,
                coordinate_state=state,
                incumbent_selected=selected,
                reduced_advantage_lower=_float_down(advantage),
                reduced_advantage_upper=_float_up(advantage),
                classification=classification,
            )
        )
    outside_count = len(registration.outside_eligible_coordinate_indices)
    if len(exact_outside_margins) != outside_count:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.EXACT_SOLVER_FAILURE,
            "outside strict-margin axis does not cover every eligible outside route",
        )
    exact_margin = (
        min(exact_outside_margins) if exact_outside_margins else Fraction(0)
    )
    margin_lower = _float_down(exact_margin)
    if outside_count and margin_lower <= 0.0:
        _refuse(
            V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode.OUTWARD_ENCLOSURE_ERASED_STRICTNESS,
            "exact outside strictness rounds outward to a nonpositive binary64 margin",
        )
    incumbent = tuple(Fraction(int(value)) for value in problem.incumbent_selected)
    incumbent_resource = _dot(resource, incumbent)
    receipt = V1PhaseAGlobalQOutsideFaceInvarianceReceiptV2(
        registration_hash=registration_hash,
        problem_hash=problem.problem_hash,
        design_hash=design_hash,
        resource_coordinate_hash=registration.resource_coordinate_hash,
        resource_coordinate_content_hash=(
            registration.resource_coordinate_content_hash
        ),
        shell_freeze_hash=registration.shell_freeze_hash,
        model_identity_hash=problem.model_identity_hash,
        chart_identity_hash=problem.chart_identity_hash,
        current_mask_hash=problem.current_mask_hash,
        route_axis_hash=problem.route_axis_hash,
        shell_coordinate_indices=registration.shell_coordinate_indices,
        shell_route_identity_hashes=registration.shell_route_identity_hashes,
        outside_eligible_coordinate_indices=(
            registration.outside_eligible_coordinate_indices
        ),
        outside_eligible_route_identity_hashes=(
            registration.outside_eligible_route_identity_hashes
        ),
        outside_eligible_incumbent_selected=(
            registration.outside_eligible_incumbent_selected
        ),
        outside_eligible_incumbent_membership_hash=(
            registration.outside_eligible_incumbent_membership_hash
        ),
        dimension=dimension,
        outside_eligible_count=outside_count,
        vacuous_outside=outside_count == 0,
        full_constraint_content_hash=_constraint_content_hash(
            full_constraints,
            role="outside-face-full-constraints-v2",
        ),
        outside_fixed_face_constraint_content_hash=_constraint_content_hash(
            face_constraints,
            role="outside-face-fixed-constraints-v2",
        ),
        full_total_inequality_count=len(full_constraints),
        outside_fixed_face_total_inequality_count=len(face_constraints),
        full_enumerated_basis_count=full_basis_count,
        outside_fixed_face_enumerated_basis_count=face_basis_count,
        full_feasible_vertex_count=full_vertex_count,
        outside_fixed_face_feasible_vertex_count=face_vertex_count,
        exact_full_dual_basis_count_at_face_optimizer=len(dual_candidates),
        support_plane_count=len(problem.support_planes),
        block_capacity_count=len(problem.block_capacities),
        support_plane_dual_multipliers_exact=tuple(
            _fraction_pair(value)
            for value in structural_multipliers[: len(problem.support_planes)]
        ),
        block_capacity_dual_multipliers_exact=tuple(
            _fraction_pair(value)
            for value in structural_multipliers[len(problem.support_planes) :]
        ),
        structural_dual_content_hash=_payload_hash(
            {
                "problem_hash": problem.problem_hash,
                "support_plane_dual_multipliers_exact": tuple(
                    _fraction_pair(value)
                    for value in structural_multipliers[
                        : len(problem.support_planes)
                    ]
                ),
                "block_capacity_dual_multipliers_exact": tuple(
                    _fraction_pair(value)
                    for value in structural_multipliers[
                        len(problem.support_planes) :
                    ]
                ),
            },
            role="outside-face-structural-dual-v2",
        ),
        full_optimal_vertex_exact=tuple(
            _fraction_pair(value) for value in full_optimizer
        ),
        outside_fixed_face_optimal_vertex_exact=tuple(
            _fraction_pair(value) for value in face_optimizer
        ),
        full_resource_exact=_fraction_pair(full_optimum),
        outside_fixed_face_resource_exact=_fraction_pair(face_optimum),
        full_resource_primal_lower_bound=_float_down(full_optimum),
        full_resource_dual_upper_bound=_float_up(full_dual_upper),
        outside_fixed_face_resource_primal_lower_bound=_float_down(face_optimum),
        outside_fixed_face_resource_upper_bound=_float_up(face_optimum),
        hard_incumbent_resource_lower_bound=_float_down(incumbent_resource),
        hard_incumbent_resource_upper_bound=_float_up(incumbent_resource),
        exact_full_primal_dual_equality=True,
        exact_outside_fixed_face_resource_enumeration=True,
        exact_full_outside_fixed_face_resource_equality=True,
        outside_fixed_face_optimizer_is_full_optimal=True,
        hard_incumbent_is_exact_global_outer_lp_optimum=(
            incumbent_resource == full_optimum
        ),
        strict_outside_signed_reduced_advantage_margin_lower=margin_lower,
        reduced_advantage_records=tuple(records),
        _issuance_token=_OUTSIDE_FACE_RECEIPT_TOKEN,
    )
    receipt.receipt_hash
    return receipt


__all__ = [
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_ALGORITHM_V2",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_AUTHORITY_V2",
    "PHASE_A_GLOBAL_Q_OUTSIDE_FACE_INVARIANCE_SCHEMA_V2",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_ALGORITHM",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_AUTHORITY",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_NO_ENTRY_SCHEMA",
    "V1PhaseAGlobalQCoordinateState",
    "V1PhaseAGlobalQNoEntryStatus",
    "V1PhaseAGlobalQOneLiveSurvivorCapacity",
    "V1PhaseAGlobalQOutsideFaceInvarianceReceiptV2",
    "V1PhaseAGlobalQOutsideFaceRegistrationV2",
    "V1PhaseAOutsideShellFixedPointReferenceReceipt",
    "V1PhaseAOutsideShellFixedPointStatus",
    "V1PhaseAGlobalQReducedAdvantageClassification",
    "V1PhaseAGlobalQReducedAdvantageRecord",
    "V1PhaseAGlobalQSupportPlane",
    "V1PhaseAGlobalQSupportPlaneNoEntryDesign",
    "V1PhaseAGlobalQSupportPlaneNoEntryPrivateProblem",
    "V1PhaseAGlobalQSupportPlaneNoEntryReceipt",
    "V1PhaseAGlobalQSupportPlaneNoEntryRefusal",
    "V1PhaseAGlobalQSupportPlaneNoEntryRefusalCode",
    "derive_v1_phase_a_outside_shell_fixed_point_reference",
    "issue_v1_phase_a_global_q_support_plane_no_entry_private_problem",
    "register_v1_phase_a_global_q_outside_face_invariance_v2",
    "register_v1_phase_a_global_q_one_live_survivor_capacity",
    "register_v1_phase_a_global_q_support_plane",
    "register_v1_phase_a_global_q_support_plane_no_entry_design",
    "solve_v1_phase_a_global_q_support_plane_no_entry_reference",
    "solve_v1_phase_a_global_q_outside_face_invariance_reference_v2",
]
