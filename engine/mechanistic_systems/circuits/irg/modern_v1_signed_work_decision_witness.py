"""Pure reference witness for the canonical two-phase signed-work D decision.

Canonical signed-work v1 does not make one scalar-salience decision.  It makes
two causally ordered decisions on one registered D decision-witness stream
(``D_W``):

* Phase A uses registered resource and literal-J response geometry only.  It
  exposes the endogenous resource-coordinate/Q-branch family.
* Phase B conditions on that closed family and uses native-fiber signed
  corrected-evidence work to select and round an action family.

This module is a CPU/small-domain theorem reference for the missing decision
stability seam.  It enumerates the complete Cartesian product of four distinct
uncertainty owners: signed-F sampling, literal-J sampling, deterministic FD
residual, and solver/rounding numerics.  C may open only if every admissible
realization yields the same Phase-A resource-cell family *and* the same
Phase-B resource-cell/physical-action decision.  Nonunique Q-branch and
private-face aliases remain diagnostic provenance when their complete hard
rounding image is identical.  Even a stable result opens only private
reference C admissibility, never generic or production C authority.

The singleton statistic ``A_j = E|a_j(x)|`` is intentionally absent from the
certificate.  It is only a triangle envelope.  It cannot recover the
within-context, within-native-fiber dependence that determines
``E|sum_j a_j(x) z_j|``.  The small reference evaluators at the bottom of this
module make that insufficiency directly testable.

No object here grants production C, K, O, mask-commit, terminal, performance,
or Pareto authority.  The stable certificate establishes only private
small-domain reference admissibility for continuing to a reference C check.
Production issuance must replace the exhaustive reference enumeration with an
equivalent source-custody-bound CUDA witness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from itertools import product
import math
from numbers import Integral, Real
from typing import Final, NoReturn

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch


SIGNED_WORK_D_W_SAMPLING_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_sampling_uncertainty_v1"
)
SIGNED_WORK_D_W_FD_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_FD_residual_uncertainty_v1"
)
SIGNED_WORK_D_W_SOLVER_UNCERTAINTY_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_solver_rounding_uncertainty_v1"
)
SIGNED_WORK_D_W_UNCERTAINTY_SET_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_four_owner_uncertainty_set_v1"
)
SIGNED_WORK_D_W_DIRECTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_private_direction_descriptor_v1"
)
SIGNED_WORK_D_W_FIBER_INTERVAL_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_native_fiber_interval_v1"
)
SIGNED_WORK_D_W_REALIZATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_two_phase_realization_v1"
)
SIGNED_WORK_D_W_PHYSICAL_DEDUP_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_physical_dedup_provenance_v1"
)
SIGNED_WORK_D_W_PHASE_A_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_phase_A_coordinate_invariance_v1"
)
SIGNED_WORK_D_W_PHASE_B_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_phase_B_action_invariance_v1"
)
SIGNED_WORK_D_W_STABILITY_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_two_phase_stability_certificate_v1"
)
SIGNED_WORK_D_W_REFUSAL_SCHEMA: Final = "partition_free_v1_signed_work_D_W_decision_refusal_v1"
SIGNED_WORK_D_W_AUTHORITY_SCOPE: Final = (
    "private_small_domain_exact_uncertainty_product_two_phase_D_W_reference_"
    "C_admissibility_only_no_generic_or_production_C_K_O_commit_terminal_or_"
    "pareto_v1"
)
SIGNED_WORK_D_W_REFERENCE_RULE: Final = (
    "phase_A_literal_J_Q_registered_resource_cell_family_invariant_then_"
    "phase_B_native_fiber_signed_F_resource_cell_times_byte_deduplicated_"
    "physical_action_family_and_preferred_action_invariant_over_complete_"
    "registered_uncertainty_product_nonunique_face_aliases_diagnostic_v1"
)

_PREFIX: Final = "partition-free-v1-signed-work-D-W-decision-witness-"
_PHASE_A_ISSUANCE_TOKEN: Final = object()
_PHASE_B_ISSUANCE_TOKEN: Final = object()
_STABILITY_ISSUANCE_TOKEN: Final = object()


class V1SignedWorkDecisionWitnessRefusalCode(str, Enum):
    """Typed invalid-input and causal-boundary refusals."""

    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    INCOMPLETE_UNCERTAINTY_PRODUCT = "incomplete_uncertainty_product"
    REFERENCE_DOMAIN_EXCEEDED = "reference_domain_exceeded"
    CAUSAL_LEAKAGE = "causal_leakage"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkDecisionWitnessRefusal(ValueError):
    """Malformed source evidence cannot be interpreted as ambiguity."""

    def __init__(
        self,
        code: V1SignedWorkDecisionWitnessRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkDecisionWitnessRefusalCode:
            raise TypeError("decision-witness refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkDecisionWitnessRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkDecisionWitnessRefusal(code, message)


class V1SignedWorkSamplingUncertaintyRole(str, Enum):
    """The two statistical owners retained separately inside D_W."""

    SIGNED_F = "signed_F_sampling"
    LITERAL_J = "literal_J_sampling"


class V1SignedWorkDWAxisBindingRole(str, Enum):
    """The optimizer phase to which a private direction is bound."""

    PHASE_A_Q_RESOURCE = "phase_A_literal_J_Q_resource"
    PHASE_B_NATIVE_FIBER_SIGNED_F = "phase_B_native_fiber_signed_F"


class V1SignedWorkDWShellEndpointRole(str, Enum):
    """A direction's exact role in the two-stage proposal image."""

    PHASE_A_EXPOSED_RESOURCE_ENDPOINT = "phase_A_exposed_resource_endpoint"
    PHASE_B_RELAXED_FACE = "phase_B_relaxed_face"
    PHASE_B_ROUNDED_SHELL = "phase_B_rounded_shell"
    PHASE_B_HARD_ENDPOINT = "phase_B_hard_endpoint"


class V1SignedWorkIntervalSubgradientRegion(str, Enum):
    """Set-valued L1 subgradient region of one signed projection interval."""

    STRICTLY_NEGATIVE = "strictly_negative"
    CONTAINS_ZERO = "contains_zero"
    STRICTLY_POSITIVE = "strictly_positive"


class V1SignedWorkDecisionWitnessDisposition(str, Enum):
    """Only lawful outcomes of the pure robust reference."""

    STABLE = "stable"
    ACQUIRE_MORE_D_W = "acquire_more_D_W"
    HIGHER_RANK_REPLAY = "higher_rank_replay"
    MORE_SOLVER_ITERATIONS = "more_solver_iterations"
    FAIL_CLOSED = "fail_closed"


class V1SignedWorkDecisionPhase(str, Enum):
    PHASE_A = "phase_A"
    PHASE_B = "phase_B"


class V1SignedWorkDecisionAmbiguitySource(str, Enum):
    SIGNED_F_SAMPLING = "signed_F_sampling"
    LITERAL_J_SAMPLING = "literal_J_sampling"
    FD_RESIDUAL = "deterministic_FD_residual"
    SOLVER_ROUNDING = "solver_rounding_numerics"
    CAUSAL_LEAKAGE = "phase_A_signed_F_causal_leakage"
    STRUCTURAL = "unattributed_structural_ambiguity"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonempty and unique",
        )
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one exact bool",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSamplingUncertaintySet:
    """One registered D_W sampling owner, or its exact finite-population census."""

    role: V1SignedWorkSamplingUncertaintyRole
    uncertainty_set_hash: str
    realization_hashes: tuple[str, ...]
    registered_d_w_prefix_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    direction_axis_hash: str
    alpha_ledger_family_hash: str
    additional_registered_samples_available: bool
    full_train_census: bool
    statistical_uncertainty: bool = True
    deterministic_uncertainty: bool = False
    spends_alpha: bool = True
    schema: str = SIGNED_WORK_D_W_SAMPLING_UNCERTAINTY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.role) is not V1SignedWorkSamplingUncertaintyRole:
            raise TypeError("sampling uncertainty role must be exact")
        for name in (
            "uncertainty_set_hash",
            "registered_d_w_prefix_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "direction_axis_hash",
            "alpha_ledger_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "realization_hashes",
            _identity_axis(self.realization_hashes, name="sampling realizations"),
        )
        flags = (
            _exact_bool(
                self.additional_registered_samples_available,
                name="additional registered samples available",
            ),
            _exact_bool(self.full_train_census, name="full train census"),
            _exact_bool(self.statistical_uncertainty, name="statistical uncertainty"),
            _exact_bool(
                self.deterministic_uncertainty,
                name="deterministic uncertainty",
            ),
            _exact_bool(self.spends_alpha, name="spends alpha"),
        )
        if flags[1] and flags[0]:
            raise ValueError("a full train census has no additional sample to acquire")
        if flags[1] and len(self.realization_hashes) != 1:
            raise ValueError("a full train census has one exact sampling realization")
        expected_ownership = (False, False, False) if flags[1] else (True, False, True)
        if flags[2:] != expected_ownership:
            raise PermissionError("sampling uncertainty changed statistical ownership")
        if self.schema != SIGNED_WORK_D_W_SAMPLING_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported D_W sampling uncertainty schema")

    @property
    def set_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "sampling-set-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFDResidualUncertaintySet:
    """Deterministic literal-J FD residual set at one admitted rank."""

    uncertainty_set_hash: str
    realization_hashes: tuple[str, ...]
    registered_d_w_prefix_hash: str
    same_capture_vjp_lineage_hash: str
    current_rank: int
    admitted_rank_ladder: tuple[int, ...]
    same_registered_prefix_required: bool = True
    deterministic_uncertainty: bool = True
    spends_alpha: bool = False
    schema: str = SIGNED_WORK_D_W_FD_UNCERTAINTY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "uncertainty_set_hash",
            "registered_d_w_prefix_hash",
            "same_capture_vjp_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "realization_hashes",
            _identity_axis(self.realization_hashes, name="FD residual realizations"),
        )
        current = _positive_int(self.current_rank, name="current FD rank")
        ladder = tuple(
            _positive_int(value, name="admitted FD rank") for value in self.admitted_rank_ladder
        )
        if not ladder or ladder != tuple(sorted(set(ladder))) or current not in ladder:
            raise ValueError("FD rank must belong to one increasing admitted ladder")
        flags = (
            _exact_bool(
                self.same_registered_prefix_required,
                name="same registered prefix required",
            ),
            _exact_bool(
                self.deterministic_uncertainty,
                name="deterministic uncertainty",
            ),
            _exact_bool(self.spends_alpha, name="spends alpha"),
        )
        if flags != (True, True, False):
            raise PermissionError("FD residual uncertainty changed ownership")
        if self.schema != SIGNED_WORK_D_W_FD_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported D_W FD uncertainty schema")
        object.__setattr__(self, "current_rank", current)
        object.__setattr__(self, "admitted_rank_ladder", ladder)

    @property
    def higher_rank_replay_available(self) -> bool:
        return self.current_rank != self.admitted_rank_ladder[-1]

    @property
    def next_rank(self) -> int | None:
        index = self.admitted_rank_ladder.index(self.current_rank)
        if index + 1 == len(self.admitted_rank_ladder):
            return None
        return self.admitted_rank_ladder[index + 1]

    @property
    def set_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "FD-set-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkSolverRoundingUncertaintySet:
    """Deterministic numerical uncertainty of Phase A/B solve and rounding."""

    uncertainty_set_hash: str
    realization_hashes: tuple[str, ...]
    solver_hash: str
    policy_hash: str
    completed_iterations: int
    maximum_registered_iterations: int
    deterministic_uncertainty: bool = True
    spends_alpha: bool = False
    schema: str = SIGNED_WORK_D_W_SOLVER_UNCERTAINTY_SCHEMA

    def __post_init__(self) -> None:
        for name in ("uncertainty_set_hash", "solver_hash", "policy_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "realization_hashes",
            _identity_axis(
                self.realization_hashes,
                name="solver/rounding realizations",
            ),
        )
        completed = _nonnegative_int(
            self.completed_iterations,
            name="completed solver iterations",
        )
        maximum = _positive_int(
            self.maximum_registered_iterations,
            name="maximum registered solver iterations",
        )
        if completed > maximum:
            raise ValueError("completed solver iterations exceed the registered maximum")
        flags = (
            _exact_bool(
                self.deterministic_uncertainty,
                name="deterministic uncertainty",
            ),
            _exact_bool(self.spends_alpha, name="spends alpha"),
        )
        if flags != (True, False):
            raise PermissionError("solver uncertainty changed deterministic ownership")
        if self.schema != SIGNED_WORK_D_W_SOLVER_UNCERTAINTY_SCHEMA:
            raise ValueError("unsupported D_W solver uncertainty schema")
        object.__setattr__(self, "completed_iterations", completed)
        object.__setattr__(self, "maximum_registered_iterations", maximum)

    @property
    def more_iterations_available(self) -> bool:
        return self.completed_iterations < self.maximum_registered_iterations

    @property
    def set_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "solver-set-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionUncertaintySet:
    """The complete, explicitly separated uncertainty product for one chart."""

    chart_identity_hash: str
    route_axis_hash: str
    registered_d_w_prefix_hash: str
    signed_f_sampling: V1SignedWorkSamplingUncertaintySet
    literal_j_sampling: V1SignedWorkSamplingUncertaintySet
    fd_residual: V1SignedWorkFDResidualUncertaintySet
    solver_rounding: V1SignedWorkSolverRoundingUncertaintySet
    maximum_reference_realizations: int = 4_096
    scalar_a_certificate_authority: bool = False
    complete_cartesian_product_required: bool = True
    schema: str = SIGNED_WORK_D_W_UNCERTAINTY_SET_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "chart_identity_hash",
            "route_axis_hash",
            "registered_d_w_prefix_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.signed_f_sampling) is not V1SignedWorkSamplingUncertaintySet
            or self.signed_f_sampling.role is not V1SignedWorkSamplingUncertaintyRole.SIGNED_F
        ):
            raise TypeError("signed_f_sampling requires the exact signed-F owner")
        if (
            type(self.literal_j_sampling) is not V1SignedWorkSamplingUncertaintySet
            or self.literal_j_sampling.role is not V1SignedWorkSamplingUncertaintyRole.LITERAL_J
        ):
            raise TypeError("literal_j_sampling requires the exact literal-J owner")
        if type(self.fd_residual) is not V1SignedWorkFDResidualUncertaintySet:
            raise TypeError("fd_residual requires its exact deterministic owner")
        if type(self.solver_rounding) is not V1SignedWorkSolverRoundingUncertaintySet:
            raise TypeError("solver_rounding requires its exact deterministic owner")
        prefixes = {
            self.registered_d_w_prefix_hash,
            self.signed_f_sampling.registered_d_w_prefix_hash,
            self.literal_j_sampling.registered_d_w_prefix_hash,
            self.fd_residual.registered_d_w_prefix_hash,
        }
        if len(prefixes) != 1:
            _refuse(
                V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
                "the four uncertainty owners do not share one registered D_W prefix",
            )
        if (
            self.signed_f_sampling.direction_axis_hash
            == self.literal_j_sampling.direction_axis_hash
        ):
            _refuse(
                V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
                "signed corrected evidence and literal J require distinct typed axes",
            )
        object.__setattr__(
            self,
            "maximum_reference_realizations",
            _positive_int(
                self.maximum_reference_realizations,
                name="maximum reference realizations",
            ),
        )
        flags = (
            _exact_bool(
                self.scalar_a_certificate_authority,
                name="scalar-A certificate authority",
            ),
            _exact_bool(
                self.complete_cartesian_product_required,
                name="complete Cartesian product required",
            ),
        )
        if flags != (False, True):
            raise PermissionError(
                "canonical signed-work stability cannot be delegated to scalar A "
                "or an incomplete uncertainty product"
            )
        if self.schema != SIGNED_WORK_D_W_UNCERTAINTY_SET_SCHEMA:
            raise ValueError("unsupported signed-work decision uncertainty schema")

    @property
    def expected_realization_count(self) -> int:
        return math.prod(len(axis) for axis in self.realization_axes)

    @property
    def realization_axes(self) -> tuple[tuple[str, ...], ...]:
        return (
            self.signed_f_sampling.realization_hashes,
            self.literal_j_sampling.realization_hashes,
            self.fd_residual.realization_hashes,
            self.solver_rounding.realization_hashes,
        )

    @property
    def uncertainty_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "chart_identity_hash": self.chart_identity_hash,
                "route_axis_hash": self.route_axis_hash,
                "registered_d_w_prefix_hash": self.registered_d_w_prefix_hash,
                "signed_f_sampling_set_hash": self.signed_f_sampling.set_hash,
                "literal_j_sampling_set_hash": self.literal_j_sampling.set_hash,
                "fd_residual_set_hash": self.fd_residual.set_hash,
                "solver_rounding_set_hash": self.solver_rounding.set_hash,
                "maximum_reference_realizations": (self.maximum_reference_realizations),
                "scalar_a_certificate_authority": False,
                "complete_cartesian_product_required": True,
            },
            prefix=_PREFIX + "uncertainty-product-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDWDirectionDescriptor:
    """Array-free identity of one variable private D_W optimizer direction."""

    direction_id: str
    chart_identity_hash: str
    route_axis_hash: str
    variable_direction_axis_hash: str
    axis_binding_role: V1SignedWorkDWAxisBindingRole
    fd_rank: int
    resource_coordinate: float
    resource_coordinate_hash: str
    resource_coordinate_cell_hash: str
    q_branch: V1RelaxedKKTBranch
    solver_hash: str
    policy_hash: str
    problem_hash: str
    result_hash: str
    private_direction_hash: str
    shell_endpoint_role: V1SignedWorkDWShellEndpointRole
    source_endpoint_label: int = 0
    native_fiber_axis_hash: str | None = None
    physical_action_hash: str | None = None
    packed_bits_sha256: str | None = None
    resulting_mask_hash: str | None = None
    removed_route_index_hash: str | None = None
    removed_count: int | None = None
    packed_bytes_equal_verified: bool = False
    schema: str = SIGNED_WORK_D_W_DIRECTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "direction_id",
            "chart_identity_hash",
            "route_axis_hash",
            "variable_direction_axis_hash",
            "resource_coordinate_hash",
            "resource_coordinate_cell_hash",
            "solver_hash",
            "policy_hash",
            "problem_hash",
            "result_hash",
            "private_direction_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.axis_binding_role) is not V1SignedWorkDWAxisBindingRole:
            raise TypeError("D_W axis binding role must be exact")
        if type(self.q_branch) is not V1RelaxedKKTBranch:
            raise TypeError("D_W direction requires one exact Q branch")
        if type(self.shell_endpoint_role) is not V1SignedWorkDWShellEndpointRole:
            raise TypeError("D_W shell/endpoint role must be exact")
        object.__setattr__(self, "fd_rank", _positive_int(self.fd_rank, name="FD rank"))
        object.__setattr__(
            self,
            "resource_coordinate",
            _nonnegative(self.resource_coordinate, name="resource coordinate"),
        )
        label = _nonnegative_int(self.source_endpoint_label, name="endpoint label")
        object.__setattr__(self, "source_endpoint_label", label)
        for name in (
            "native_fiber_axis_hash",
            "physical_action_hash",
            "packed_bits_sha256",
            "resulting_mask_hash",
            "removed_route_index_hash",
        ):
            object.__setattr__(
                self,
                name,
                _optional_identity(getattr(self, name), name=name),
            )
        packed_verified = _exact_bool(
            self.packed_bytes_equal_verified,
            name="packed bytes equal verified",
        )
        phase_a = self.axis_binding_role is V1SignedWorkDWAxisBindingRole.PHASE_A_Q_RESOURCE
        if phase_a:
            if self.shell_endpoint_role is not (
                V1SignedWorkDWShellEndpointRole.PHASE_A_EXPOSED_RESOURCE_ENDPOINT
            ):
                raise ValueError("Phase A requires the exposed-resource endpoint role")
            if (
                self.native_fiber_axis_hash is not None
                or any(
                    value is not None
                    for value in (
                        self.physical_action_hash,
                        self.packed_bits_sha256,
                        self.resulting_mask_hash,
                        self.removed_route_index_hash,
                        self.removed_count,
                    )
                )
                or packed_verified
                or label != 0
            ):
                raise ValueError("Phase A cannot carry Phase-B physical action state")
        else:
            if self.shell_endpoint_role is (
                V1SignedWorkDWShellEndpointRole.PHASE_A_EXPOSED_RESOURCE_ENDPOINT
            ):
                raise ValueError("Phase B cannot impersonate a Phase-A endpoint")
            if self.native_fiber_axis_hash is None:
                raise ValueError("Phase B requires the native-fiber axis identity")
            hard = self.shell_endpoint_role is (
                V1SignedWorkDWShellEndpointRole.PHASE_B_HARD_ENDPOINT
            )
            physical_values = (
                self.physical_action_hash,
                self.packed_bits_sha256,
                self.resulting_mask_hash,
                self.removed_route_index_hash,
                self.removed_count,
            )
            if hard:
                if any(value is None for value in physical_values):
                    raise ValueError("a hard endpoint requires its complete physical identity")
                if label == 0:
                    raise ValueError("a hard endpoint requires a positive endpoint label")
                removed = _positive_int(self.removed_count, name="removed route count")
                object.__setattr__(self, "removed_count", removed)
                if not packed_verified:
                    raise PermissionError(
                        "physical dedup requires byte equality, not matching hashes alone"
                    )
            elif any(value is not None for value in physical_values) or packed_verified:
                raise ValueError("only a hard endpoint may carry physical action state")
        if self.schema != SIGNED_WORK_D_W_DIRECTION_SCHEMA:
            raise ValueError("unsupported D_W direction descriptor schema")

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "direction-")

    @property
    def decision_member(self) -> tuple[object, ...]:
        """Diagnostic face identity on one preregistered resource cell.

        The exact numeric resource coordinate and receipt hash remain in the
        descriptor provenance.  A continuous perturbation inside one cell is
        not a new irreversible decision.  This face identity is diagnostic;
        joint authority below uses its complete hard rounding image.
        """

        return (
            self.axis_binding_role.value,
            self.resource_coordinate_cell_hash,
            self.q_branch.value,
            self.private_direction_hash,
            self.shell_endpoint_role.value,
            self.source_endpoint_label,
            self.native_fiber_axis_hash,
        )

    @property
    def resource_cell_member(self) -> tuple[object, ...]:
        """Discrete coordinate-cell/shell identity independent of face alias."""

        return (
            self.resource_coordinate_cell_hash,
            self.shell_endpoint_role.value,
            self.source_endpoint_label,
        )

    @property
    def physical_dedup_identity_hash(self) -> str | None:
        if self.shell_endpoint_role is not (V1SignedWorkDWShellEndpointRole.PHASE_B_HARD_ENDPOINT):
            return None
        assert self.physical_action_hash is not None
        assert self.packed_bits_sha256 is not None
        assert self.resulting_mask_hash is not None
        assert self.removed_route_index_hash is not None
        assert self.removed_count is not None
        return v1_signed_work_d_w_physical_dedup_identity_hash(
            physical_action_hash=self.physical_action_hash,
            packed_bits_sha256=self.packed_bits_sha256,
            resulting_mask_hash=self.resulting_mask_hash,
            removed_route_index_hash=self.removed_route_index_hash,
            removed_count=self.removed_count,
        )


def v1_signed_work_d_w_physical_dedup_identity_hash(
    *,
    physical_action_hash: str,
    packed_bits_sha256: str,
    resulting_mask_hash: str,
    removed_route_index_hash: str,
    removed_count: int,
) -> str:
    """Replay the Phase-B hard-endpoint physical dedup identity.

    This public helper is intentionally the same identity map used by
    :class:`V1SignedWorkDWDirectionDescriptor`.  Downstream joins that bind an
    issued packed action to a D_W stability certificate must not duplicate the
    prefix/payload by hand or compare only same-cardinality metadata.
    """

    return content_hash(
        {
            "physical_action_hash": _identity(
                physical_action_hash,
                name="physical action hash",
            ),
            "packed_bits_sha256": _identity(
                packed_bits_sha256,
                name="packed bits SHA-256",
            ),
            "resulting_mask_hash": _identity(
                resulting_mask_hash,
                name="resulting mask hash",
            ),
            "removed_route_index_hash": _identity(
                removed_route_index_hash,
                name="removed route index hash",
            ),
            "removed_count": _positive_int(removed_count, name="removed route count"),
            "packed_bytes_equal_verified": True,
        },
        prefix=_PREFIX + "physical-identity-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkNativeFiberProjectionInterval:
    """Outward signed interval and its exact set-valued L1 subgradient."""

    direction_descriptor_hash: str
    context_axis_hash: str
    native_fiber_id: str
    lower: float
    upper: float
    fiber_weight: float = 1.0
    schema: str = SIGNED_WORK_D_W_FIBER_INTERVAL_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "direction_descriptor_hash",
            "context_axis_hash",
            "native_fiber_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        lower = _finite(self.lower, name="fiber projection lower bound")
        upper = _finite(self.upper, name="fiber projection upper bound")
        if lower > upper:
            raise ValueError("fiber projection interval is reversed")
        weight = _nonnegative(self.fiber_weight, name="fiber weight")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "fiber_weight", weight)
        if self.schema != SIGNED_WORK_D_W_FIBER_INTERVAL_SCHEMA:
            raise ValueError("unsupported native-fiber interval schema")

    @property
    def subgradient_region(self) -> V1SignedWorkIntervalSubgradientRegion:
        if self.upper < 0.0:
            return V1SignedWorkIntervalSubgradientRegion.STRICTLY_NEGATIVE
        if self.lower > 0.0:
            return V1SignedWorkIntervalSubgradientRegion.STRICTLY_POSITIVE
        return V1SignedWorkIntervalSubgradientRegion.CONTAINS_ZERO

    @property
    def subgradient_interval(self) -> tuple[float, float]:
        region = self.subgradient_region
        if region is V1SignedWorkIntervalSubgradientRegion.STRICTLY_NEGATIVE:
            return (-1.0, -1.0)
        if region is V1SignedWorkIntervalSubgradientRegion.STRICTLY_POSITIVE:
            return (1.0, 1.0)
        return (-1.0, 1.0)

    @property
    def absolute_interval(self) -> tuple[float, float]:
        if self.upper < 0.0:
            return (-self.upper, -self.lower)
        if self.lower > 0.0:
            return (self.lower, self.upper)
        return (0.0, max(-self.lower, self.upper))

    @property
    def interval_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "direction_descriptor_hash": self.direction_descriptor_hash,
                "context_axis_hash": self.context_axis_hash,
                "native_fiber_id": self.native_fiber_id,
                "lower": self.lower,
                "upper": self.upper,
                "fiber_weight": self.fiber_weight,
                "absolute_interval": self.absolute_interval,
                "subgradient_interval": self.subgradient_interval,
            },
            prefix=_PREFIX + "fiber-interval-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalDedupProvenance:
    """One exact physical action plus every coordinate/branch alias."""

    physical_dedup_identity_hash: str
    physical_action_hash: str
    packed_bits_sha256: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    removed_count: int
    alias_direction_descriptor_hashes: tuple[str, ...]
    packed_bytes_equal_verified: bool = True
    schema: str = SIGNED_WORK_D_W_PHYSICAL_DEDUP_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "physical_dedup_identity_hash",
            "physical_action_hash",
            "packed_bits_sha256",
            "resulting_mask_hash",
            "removed_route_index_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed count"),
        )
        aliases = tuple(
            sorted(
                _identity_axis(
                    self.alias_direction_descriptor_hashes,
                    name="physical-action aliases",
                )
            )
        )
        object.__setattr__(self, "alias_direction_descriptor_hashes", aliases)
        if not _exact_bool(
            self.packed_bytes_equal_verified,
            name="packed bytes equal verified",
        ):
            raise PermissionError("physical dedup cannot rely on digests alone")
        expected = content_hash(
            {
                "physical_action_hash": self.physical_action_hash,
                "packed_bits_sha256": self.packed_bits_sha256,
                "resulting_mask_hash": self.resulting_mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
                "removed_count": self.removed_count,
                "packed_bytes_equal_verified": True,
            },
            prefix=_PREFIX + "physical-identity-",
        )
        if self.physical_dedup_identity_hash != expected:
            raise ValueError("physical dedup identity does not replay")
        if self.schema != SIGNED_WORK_D_W_PHYSICAL_DEDUP_SCHEMA:
            raise ValueError("unsupported physical dedup schema")

    @property
    def provenance_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "physical-provenance-")


def _physical_dedup_provenance(
    directions: Sequence[V1SignedWorkDWDirectionDescriptor],
) -> tuple[V1SignedWorkPhysicalDedupProvenance, ...]:
    groups: dict[str, list[V1SignedWorkDWDirectionDescriptor]] = {}
    for direction in directions:
        identity = direction.physical_dedup_identity_hash
        if identity is not None:
            groups.setdefault(identity, []).append(direction)
    records: list[V1SignedWorkPhysicalDedupProvenance] = []
    for identity, aliases in sorted(groups.items()):
        first = aliases[0]
        physical = (
            first.physical_action_hash,
            first.packed_bits_sha256,
            first.resulting_mask_hash,
            first.removed_route_index_hash,
            first.removed_count,
        )
        if any(
            (
                item.physical_action_hash,
                item.packed_bits_sha256,
                item.resulting_mask_hash,
                item.removed_route_index_hash,
                item.removed_count,
            )
            != physical
            for item in aliases[1:]
        ):
            raise ValueError("physical dedup aliases disagree despite one identity")
        assert first.physical_action_hash is not None
        assert first.packed_bits_sha256 is not None
        assert first.resulting_mask_hash is not None
        assert first.removed_route_index_hash is not None
        assert first.removed_count is not None
        records.append(
            V1SignedWorkPhysicalDedupProvenance(
                physical_dedup_identity_hash=identity,
                physical_action_hash=first.physical_action_hash,
                packed_bits_sha256=first.packed_bits_sha256,
                resulting_mask_hash=first.resulting_mask_hash,
                removed_route_index_hash=first.removed_route_index_hash,
                removed_count=first.removed_count,
                alias_direction_descriptor_hashes=tuple(item.descriptor_hash for item in aliases),
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionRealization:
    """One optimizer image at one vertex of the four-owner uncertainty set."""

    signed_f_sampling_realization_hash: str
    literal_j_sampling_realization_hash: str
    fd_residual_realization_hash: str
    solver_rounding_realization_hash: str
    directions: tuple[V1SignedWorkDWDirectionDescriptor, ...]
    preferred_physical_dedup_identity_hash: str
    native_fiber_projection_intervals: tuple[V1SignedWorkNativeFiberProjectionInterval, ...] = ()
    schema: str = SIGNED_WORK_D_W_REALIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "signed_f_sampling_realization_hash",
            "literal_j_sampling_realization_hash",
            "fd_residual_realization_hash",
            "solver_rounding_realization_hash",
            "preferred_physical_dedup_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        directions = tuple(self.directions)
        if not directions or any(
            type(value) is not V1SignedWorkDWDirectionDescriptor for value in directions
        ):
            raise TypeError("a decision realization requires exact D_W directions")
        hashes = tuple(value.descriptor_hash for value in directions)
        if len(hashes) != len(set(hashes)):
            raise ValueError("a decision realization repeats a direction descriptor")
        phase_a = tuple(
            value
            for value in directions
            if value.axis_binding_role is V1SignedWorkDWAxisBindingRole.PHASE_A_Q_RESOURCE
        )
        phase_b = tuple(
            value
            for value in directions
            if value.axis_binding_role
            is V1SignedWorkDWAxisBindingRole.PHASE_B_NATIVE_FIBER_SIGNED_F
        )
        if not phase_a or not phase_b:
            raise ValueError("a realization must contain both optimizer phases")
        phase_a_cells = {value.resource_coordinate_cell_hash for value in phase_a}
        phase_b_cells = {value.resource_coordinate_cell_hash for value in phase_b}
        if not phase_b_cells.issubset(phase_a_cells):
            raise ValueError("Phase B contains a resource cell not exposed by Phase A")
        physical = _physical_dedup_provenance(phase_b)
        if not physical:
            raise ValueError("Phase B requires at least one exact hard endpoint")
        physical_ids = {value.physical_dedup_identity_hash for value in physical}
        if self.preferred_physical_dedup_identity_hash not in physical_ids:
            raise ValueError("the preferred Phase-B action is absent after physical dedup")
        intervals = tuple(self.native_fiber_projection_intervals)
        if any(type(value) is not V1SignedWorkNativeFiberProjectionInterval for value in intervals):
            raise TypeError("native-fiber intervals require their exact type")
        phase_b_hashes = {value.descriptor_hash for value in phase_b}
        if any(value.direction_descriptor_hash not in phase_b_hashes for value in intervals):
            raise ValueError("a native-fiber interval is not bound to Phase B")
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "native_fiber_projection_intervals", intervals)
        if self.schema != SIGNED_WORK_D_W_REALIZATION_SCHEMA:
            raise ValueError("unsupported D_W decision realization schema")

    @property
    def grid_key(self) -> tuple[str, str, str, str]:
        return (
            self.signed_f_sampling_realization_hash,
            self.literal_j_sampling_realization_hash,
            self.fd_residual_realization_hash,
            self.solver_rounding_realization_hash,
        )

    @property
    def phase_a_directions(self) -> tuple[V1SignedWorkDWDirectionDescriptor, ...]:
        return tuple(
            value
            for value in self.directions
            if value.axis_binding_role is V1SignedWorkDWAxisBindingRole.PHASE_A_Q_RESOURCE
        )

    @property
    def phase_b_directions(self) -> tuple[V1SignedWorkDWDirectionDescriptor, ...]:
        return tuple(
            value
            for value in self.directions
            if value.axis_binding_role
            is V1SignedWorkDWAxisBindingRole.PHASE_B_NATIVE_FIBER_SIGNED_F
        )

    @property
    def physical_dedup_provenance(
        self,
    ) -> tuple[V1SignedWorkPhysicalDedupProvenance, ...]:
        return _physical_dedup_provenance(self.phase_b_directions)

    @property
    def phase_a_decision_family_hash(self) -> str:
        return content_hash(
            tuple(sorted({value.resource_cell_member for value in self.phase_a_directions})),
            prefix=_PREFIX + "phase-A-decision-family-",
        )

    @property
    def phase_b_decision_family_hash(self) -> str:
        hard = tuple(
            value
            for value in self.phase_b_directions
            if value.physical_dedup_identity_hash is not None
        )
        return content_hash(
            {
                "resource_cell_physical_action_family": tuple(
                    sorted(
                        {
                            (
                                value.resource_coordinate_cell_hash,
                                value.physical_dedup_identity_hash,
                            )
                            for value in hard
                        }
                    )
                ),
                "physical_action_family": tuple(
                    value.physical_dedup_identity_hash for value in self.physical_dedup_provenance
                ),
                "preferred_physical_action": (self.preferred_physical_dedup_identity_hash),
            },
            prefix=_PREFIX + "phase-B-decision-family-",
        )

    @property
    def phase_a_face_family_hash(self) -> str:
        """Diagnostic Phase-A branch/face image; not a stability requirement."""

        return content_hash(
            tuple(sorted(value.decision_member for value in self.phase_a_directions)),
            prefix=_PREFIX + "phase-A-face-family-",
        )

    @property
    def phase_b_face_family_hash(self) -> str:
        """Diagnostic Phase-B branch/face image; not a stability requirement."""

        return content_hash(
            tuple(sorted(value.decision_member for value in self.phase_b_directions)),
            prefix=_PREFIX + "phase-B-face-family-",
        )

    @property
    def realization_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "grid_key": self.grid_key,
                "direction_descriptor_hashes": tuple(
                    value.descriptor_hash for value in self.directions
                ),
                "native_fiber_interval_hashes": tuple(
                    value.interval_hash for value in self.native_fiber_projection_intervals
                ),
                "physical_dedup_provenance_hashes": tuple(
                    value.provenance_hash for value in self.physical_dedup_provenance
                ),
                "phase_a_decision_family_hash": self.phase_a_decision_family_hash,
                "phase_b_decision_family_hash": self.phase_b_decision_family_hash,
                "phase_a_face_family_hash": self.phase_a_face_family_hash,
                "phase_b_face_family_hash": self.phase_b_face_family_hash,
            },
            prefix=_PREFIX + "realization-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhaseACoordinateInvarianceCertificate:
    """Every admissible uncertainty vertex exposes one Phase-A family."""

    uncertainty_set_hash: str
    phase_a_decision_family_hash: str
    realization_hashes: tuple[str, ...]
    direction_descriptor_hashes: tuple[str, ...]
    diagnostic_face_family_hashes: tuple[str, ...]
    _issuance_token: InitVar[object]
    phase_a_invariant: bool = True
    private_reference_C_admissibility: bool = False
    C_stream_open_authority: bool = False
    production_C_stream_open_authority: bool = False
    schema: str = SIGNED_WORK_D_W_PHASE_A_CERTIFICATE_SCHEMA

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _PHASE_A_ISSUANCE_TOKEN:
            raise PermissionError("Phase-A invariance requires exhaustive issuance")
        for name in ("uncertainty_set_hash", "phase_a_decision_family_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "realization_hashes",
            tuple(
                sorted(
                    _identity_axis(
                        self.realization_hashes,
                        name="Phase-A realization hashes",
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "direction_descriptor_hashes",
            tuple(
                sorted(
                    _identity_axis(
                        self.direction_descriptor_hashes,
                        name="Phase-A descriptor hashes",
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "diagnostic_face_family_hashes",
            tuple(
                sorted(
                    _identity_axis(
                        self.diagnostic_face_family_hashes,
                        name="Phase-A diagnostic face-family hashes",
                    )
                )
            ),
        )
        flags = (
            _exact_bool(self.phase_a_invariant, name="Phase-A invariant"),
            _exact_bool(
                self.private_reference_C_admissibility,
                name="private reference C admissibility",
            ),
            _exact_bool(self.C_stream_open_authority, name="C stream open authority"),
            _exact_bool(
                self.production_C_stream_open_authority,
                name="production C stream open authority",
            ),
        )
        if flags != (True, False, False, False):
            raise PermissionError("Phase A alone cannot open C")
        if self.schema != SIGNED_WORK_D_W_PHASE_A_CERTIFICATE_SCHEMA:
            raise ValueError("unsupported Phase-A certificate schema")

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "phase-A-certificate-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhaseBActionInvarianceCertificate:
    """Every admissible uncertainty vertex yields one Phase-B action family."""

    uncertainty_set_hash: str
    phase_a_certificate_hash: str
    phase_b_decision_family_hash: str
    preferred_physical_dedup_identity_hash: str
    realization_hashes: tuple[str, ...]
    direction_descriptor_hashes: tuple[str, ...]
    physical_dedup_provenance_hashes: tuple[str, ...]
    diagnostic_face_family_hashes: tuple[str, ...]
    _issuance_token: InitVar[object]
    phase_b_invariant: bool = True
    private_reference_C_admissibility: bool = False
    C_stream_open_authority: bool = False
    production_C_stream_open_authority: bool = False
    schema: str = SIGNED_WORK_D_W_PHASE_B_CERTIFICATE_SCHEMA

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _PHASE_B_ISSUANCE_TOKEN:
            raise PermissionError("Phase-B invariance requires exhaustive issuance")
        for name in (
            "uncertainty_set_hash",
            "phase_a_certificate_hash",
            "phase_b_decision_family_hash",
            "preferred_physical_dedup_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "realization_hashes",
            "direction_descriptor_hashes",
            "physical_dedup_provenance_hashes",
            "diagnostic_face_family_hashes",
        ):
            object.__setattr__(
                self,
                name,
                tuple(sorted(_identity_axis(getattr(self, name), name=name))),
            )
        flags = (
            _exact_bool(self.phase_b_invariant, name="Phase-B invariant"),
            _exact_bool(
                self.private_reference_C_admissibility,
                name="private reference C admissibility",
            ),
            _exact_bool(self.C_stream_open_authority, name="C stream open authority"),
            _exact_bool(
                self.production_C_stream_open_authority,
                name="production C stream open authority",
            ),
        )
        if flags != (True, False, False, False):
            raise PermissionError("Phase B alone cannot open C")
        if self.schema != SIGNED_WORK_D_W_PHASE_B_CERTIFICATE_SCHEMA:
            raise ValueError("unsupported Phase-B certificate schema")

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "phase-B-certificate-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionStabilityCertificate:
    """Joint private-reference admissibility; never production C authority."""

    uncertainty_set_hash: str
    enumeration_hash: str
    phase_a: V1SignedWorkPhaseACoordinateInvarianceCertificate
    phase_b: V1SignedWorkPhaseBActionInvarianceCertificate
    _issuance_token: InitVar[object]
    disposition: V1SignedWorkDecisionWitnessDisposition = (
        V1SignedWorkDecisionWitnessDisposition.STABLE
    )
    reference_rule: str = SIGNED_WORK_D_W_REFERENCE_RULE
    private_reference_C_admissibility: bool = True
    C_stream_open_authority: bool = False
    production_C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_D_W_STABILITY_CERTIFICATE_SCHEMA

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _STABILITY_ISSUANCE_TOKEN:
            raise PermissionError("joint D_W stability requires exhaustive issuance")
        for name in ("uncertainty_set_hash", "enumeration_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.phase_a) is not V1SignedWorkPhaseACoordinateInvarianceCertificate:
            raise TypeError("joint stability requires the exact Phase-A certificate")
        if type(self.phase_b) is not V1SignedWorkPhaseBActionInvarianceCertificate:
            raise TypeError("joint stability requires the exact Phase-B certificate")
        if (
            self.phase_a.uncertainty_set_hash != self.uncertainty_set_hash
            or self.phase_b.uncertainty_set_hash != self.uncertainty_set_hash
            or self.phase_b.phase_a_certificate_hash != self.phase_a.certificate_hash
        ):
            raise ValueError("the two phase certificates do not share one uncertainty set")
        if self.disposition is not V1SignedWorkDecisionWitnessDisposition.STABLE:
            raise ValueError("a stability certificate must have stable disposition")
        flags = (
            _exact_bool(
                self.private_reference_C_admissibility,
                name="private reference C admissibility",
            ),
            _exact_bool(self.C_stream_open_authority, name="C stream authority"),
            _exact_bool(
                self.production_C_stream_open_authority,
                name="production C stream authority",
            ),
            _exact_bool(
                self.calibration_K_open_authority,
                name="calibration K authority",
            ),
            _exact_bool(
                self.paired_oracle_O_open_authority,
                name="paired oracle O authority",
            ),
            _exact_bool(self.mask_commit_authority, name="mask commit authority"),
            _exact_bool(self.terminal_authority, name="terminal authority"),
            _exact_bool(self.pareto_authority, name="Pareto authority"),
        )
        if flags != (True, False, False, False, False, False, False, False):
            raise PermissionError("D_W reference stability crossed its authority")
        if (
            self.reference_rule != SIGNED_WORK_D_W_REFERENCE_RULE
            or self.authority_scope != SIGNED_WORK_D_W_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_D_W_STABILITY_CERTIFICATE_SCHEMA
        ):
            raise PermissionError("D_W stability semantics changed")

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "joint-certificate-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionStabilityRefusal:
    """A typed no-C result plus the only justified next computation."""

    disposition: V1SignedWorkDecisionWitnessDisposition
    unresolved_phase: V1SignedWorkDecisionPhase
    reason: str
    uncertainty_set_hash: str
    enumeration_hash: str
    ambiguity_sources: tuple[V1SignedWorkDecisionAmbiguitySource, ...]
    phase_a_certificate: V1SignedWorkPhaseACoordinateInvarianceCertificate | None = None
    next_rank: int | None = None
    private_reference_C_admissibility: bool = False
    C_stream_open_authority: bool = False
    production_C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_D_W_REFUSAL_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.disposition) is not V1SignedWorkDecisionWitnessDisposition
            or self.disposition is V1SignedWorkDecisionWitnessDisposition.STABLE
        ):
            raise TypeError("a refusal requires one non-stable disposition")
        if type(self.unresolved_phase) is not V1SignedWorkDecisionPhase:
            raise TypeError("a refusal requires one exact unresolved phase")
        for name in ("reason", "uncertainty_set_hash", "enumeration_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        sources = tuple(self.ambiguity_sources)
        if not sources or any(
            type(value) is not V1SignedWorkDecisionAmbiguitySource for value in sources
        ):
            raise TypeError("a refusal requires typed ambiguity sources")
        if len(sources) != len(set(sources)):
            raise ValueError("a refusal repeats an ambiguity source")
        object.__setattr__(self, "ambiguity_sources", sources)
        if self.phase_a_certificate is not None:
            if type(self.phase_a_certificate) is not (
                V1SignedWorkPhaseACoordinateInvarianceCertificate
            ):
                raise TypeError("phase_a_certificate type is invalid")
            if self.phase_a_certificate.uncertainty_set_hash != self.uncertainty_set_hash:
                raise ValueError("a refusal carries a stale Phase-A certificate")
        if self.unresolved_phase is V1SignedWorkDecisionPhase.PHASE_A:
            if self.phase_a_certificate is not None:
                raise ValueError("unresolved Phase A cannot carry its certificate")
        elif self.phase_a_certificate is None:
            raise ValueError("unresolved Phase B requires the closed Phase-A certificate")
        if self.next_rank is not None:
            object.__setattr__(
                self,
                "next_rank",
                _positive_int(self.next_rank, name="next FD rank"),
            )
        flags = (
            self.private_reference_C_admissibility,
            self.C_stream_open_authority,
            self.production_C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            raise PermissionError("an unresolved D_W witness cannot open C")
        if self.schema != SIGNED_WORK_D_W_REFUSAL_SCHEMA:
            raise ValueError("unsupported D_W decision refusal schema")

    @property
    def refusal_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "refusal-")


V1SignedWorkDecisionWitnessResult = (
    V1SignedWorkDecisionStabilityCertificate | V1SignedWorkDecisionStabilityRefusal
)


def _validate_reference_grid(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    realizations: Sequence[V1SignedWorkDecisionRealization],
) -> tuple[V1SignedWorkDecisionRealization, ...]:
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("decision witness requires the exact uncertainty contract")
    values = tuple(realizations)
    if any(type(value) is not V1SignedWorkDecisionRealization for value in values):
        raise TypeError("decision witness requires exact realization objects")
    expected_count = uncertainty.expected_realization_count
    if expected_count > uncertainty.maximum_reference_realizations:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.REFERENCE_DOMAIN_EXCEEDED,
            "the exhaustive Cartesian product exceeds the registered reference limit",
        )
    expected = set(product(*uncertainty.realization_axes))
    observed: dict[tuple[str, str, str, str], V1SignedWorkDecisionRealization] = {}
    for value in values:
        if value.grid_key in observed:
            _refuse(
                V1SignedWorkDecisionWitnessRefusalCode.INCOMPLETE_UNCERTAINTY_PRODUCT,
                "the reference grid repeats an uncertainty vertex",
            )
        observed[value.grid_key] = value
    if set(observed) != expected:
        _refuse(
            V1SignedWorkDecisionWitnessRefusalCode.INCOMPLETE_UNCERTAINTY_PRODUCT,
            "the reference grid is not the complete registered Cartesian product",
        )
    signed_axis = uncertainty.signed_f_sampling.direction_axis_hash
    literal_axis = uncertainty.literal_j_sampling.direction_axis_hash
    solver = uncertainty.solver_rounding
    rank = uncertainty.fd_residual.current_rank
    for value in values:
        for direction in value.directions:
            if (
                direction.chart_identity_hash != uncertainty.chart_identity_hash
                or direction.route_axis_hash != uncertainty.route_axis_hash
                or direction.fd_rank != rank
                or direction.solver_hash != solver.solver_hash
                or direction.policy_hash != solver.policy_hash
            ):
                _refuse(
                    V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
                    "a private direction left its chart, rank, solver, or policy",
                )
            expected_axis = (
                literal_axis
                if direction.axis_binding_role is V1SignedWorkDWAxisBindingRole.PHASE_A_Q_RESOURCE
                else signed_axis
            )
            if direction.variable_direction_axis_hash != expected_axis:
                _refuse(
                    V1SignedWorkDecisionWitnessRefusalCode.AXIS_MISMATCH,
                    "a private direction impersonated the other optimizer phase axis",
                )
    positions = [
        {value: index for index, value in enumerate(axis)} for axis in uncertainty.realization_axes
    ]
    return tuple(
        sorted(
            values,
            key=lambda value: tuple(
                positions[index][coordinate] for index, coordinate in enumerate(value.grid_key)
            ),
        )
    )


def _ambiguity_sources(
    *,
    realizations: Sequence[V1SignedWorkDecisionRealization],
    decision_hash_name: str,
) -> tuple[V1SignedWorkDecisionAmbiguitySource, ...]:
    source_axis = (
        V1SignedWorkDecisionAmbiguitySource.SIGNED_F_SAMPLING,
        V1SignedWorkDecisionAmbiguitySource.LITERAL_J_SAMPLING,
        V1SignedWorkDecisionAmbiguitySource.FD_RESIDUAL,
        V1SignedWorkDecisionAmbiguitySource.SOLVER_ROUNDING,
    )
    result: list[V1SignedWorkDecisionAmbiguitySource] = []
    for dimension, source in enumerate(source_axis):
        grouped: dict[tuple[str, ...], set[str]] = {}
        for realization in realizations:
            other = tuple(
                value for index, value in enumerate(realization.grid_key) if index != dimension
            )
            grouped.setdefault(other, set()).add(str(getattr(realization, decision_hash_name)))
        if any(len(decisions) > 1 for decisions in grouped.values()):
            result.append(source)
    if not result:
        result.append(V1SignedWorkDecisionAmbiguitySource.STRUCTURAL)
    return tuple(result)


def _enumeration_hash(
    *,
    uncertainty_hash: str,
    realizations: Sequence[V1SignedWorkDecisionRealization],
) -> str:
    return content_hash(
        {
            "uncertainty_set_hash": uncertainty_hash,
            "reference_rule": SIGNED_WORK_D_W_REFERENCE_RULE,
            "realization_hashes": tuple(value.realization_hash for value in realizations),
            "complete_cartesian_product_enumerated": True,
            "scalar_A_certificate_used": False,
        },
        prefix=_PREFIX + "enumeration-",
    )


def _phase_a_certificate(
    *,
    uncertainty_hash: str,
    realizations: Sequence[V1SignedWorkDecisionRealization],
) -> V1SignedWorkPhaseACoordinateInvarianceCertificate:
    family = realizations[0].phase_a_decision_family_hash
    return V1SignedWorkPhaseACoordinateInvarianceCertificate(
        uncertainty_set_hash=uncertainty_hash,
        phase_a_decision_family_hash=family,
        realization_hashes=tuple(value.realization_hash for value in realizations),
        direction_descriptor_hashes=tuple(
            sorted(
                {
                    item.descriptor_hash
                    for value in realizations
                    for item in value.phase_a_directions
                }
            )
        ),
        diagnostic_face_family_hashes=tuple(
            sorted({value.phase_a_face_family_hash for value in realizations})
        ),
        _issuance_token=_PHASE_A_ISSUANCE_TOKEN,
    )


def _phase_b_certificate(
    *,
    uncertainty_hash: str,
    phase_a: V1SignedWorkPhaseACoordinateInvarianceCertificate,
    realizations: Sequence[V1SignedWorkDecisionRealization],
) -> V1SignedWorkPhaseBActionInvarianceCertificate:
    first = realizations[0]
    return V1SignedWorkPhaseBActionInvarianceCertificate(
        uncertainty_set_hash=uncertainty_hash,
        phase_a_certificate_hash=phase_a.certificate_hash,
        phase_b_decision_family_hash=first.phase_b_decision_family_hash,
        preferred_physical_dedup_identity_hash=(first.preferred_physical_dedup_identity_hash),
        realization_hashes=tuple(value.realization_hash for value in realizations),
        direction_descriptor_hashes=tuple(
            sorted(
                {
                    item.descriptor_hash
                    for value in realizations
                    for item in value.phase_b_directions
                }
            )
        ),
        physical_dedup_provenance_hashes=tuple(
            sorted(
                {
                    item.provenance_hash
                    for value in realizations
                    for item in value.physical_dedup_provenance
                }
            )
        ),
        diagnostic_face_family_hashes=tuple(
            sorted({value.phase_b_face_family_hash for value in realizations})
        ),
        _issuance_token=_PHASE_B_ISSUANCE_TOKEN,
    )


def _refinement_disposition(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    sources: tuple[V1SignedWorkDecisionAmbiguitySource, ...],
) -> tuple[V1SignedWorkDecisionWitnessDisposition, int | None, str]:
    actions: set[V1SignedWorkDecisionWitnessDisposition] = set()
    blocked: list[str] = []
    if V1SignedWorkDecisionAmbiguitySource.SIGNED_F_SAMPLING in sources:
        if uncertainty.signed_f_sampling.additional_registered_samples_available:
            actions.add(V1SignedWorkDecisionWitnessDisposition.ACQUIRE_MORE_D_W)
        else:
            blocked.append("signed-F sampling is exhausted")
    if V1SignedWorkDecisionAmbiguitySource.LITERAL_J_SAMPLING in sources:
        if uncertainty.literal_j_sampling.additional_registered_samples_available:
            actions.add(V1SignedWorkDecisionWitnessDisposition.ACQUIRE_MORE_D_W)
        else:
            blocked.append("literal-J sampling is exhausted")
    if V1SignedWorkDecisionAmbiguitySource.FD_RESIDUAL in sources:
        if uncertainty.fd_residual.higher_rank_replay_available:
            actions.add(V1SignedWorkDecisionWitnessDisposition.HIGHER_RANK_REPLAY)
        else:
            blocked.append("the admitted FD-rank ladder is exhausted")
    if V1SignedWorkDecisionAmbiguitySource.SOLVER_ROUNDING in sources:
        if uncertainty.solver_rounding.more_iterations_available:
            actions.add(V1SignedWorkDecisionWitnessDisposition.MORE_SOLVER_ITERATIONS)
        else:
            blocked.append("registered solver iterations are exhausted")
    if any(
        source
        in {
            V1SignedWorkDecisionAmbiguitySource.CAUSAL_LEAKAGE,
            V1SignedWorkDecisionAmbiguitySource.STRUCTURAL,
        }
        for source in sources
    ):
        blocked.append("ambiguity has no lawful single-owner refinement")
    if blocked or len(actions) != 1:
        detail = "; ".join(blocked)
        if len(actions) > 1:
            detail = (detail + "; " if detail else "") + (
                "more than one computation class can change the decision"
            )
        return (
            V1SignedWorkDecisionWitnessDisposition.FAIL_CLOSED,
            None,
            detail or "no registered refinement can close the decision",
        )
    action = next(iter(actions))
    next_rank = (
        uncertainty.fd_residual.next_rank
        if action is V1SignedWorkDecisionWitnessDisposition.HIGHER_RANK_REPLAY
        else None
    )
    return action, next_rank, "the ambiguity is isolated to one lawful computation"


def certify_v1_signed_work_d_decision_witness_reference(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    realizations: Sequence[V1SignedWorkDecisionRealization],
) -> V1SignedWorkDecisionWitnessResult:
    """Exhaustively certify or refuse the two-phase D_W optimizer image.

    This function is deliberately finite and exact at the supplied decision
    image.  It does not infer unenumerated vertices and it never substitutes
    singleton ``A`` intervals for native-fiber signed work.
    """

    ordered = _validate_reference_grid(
        uncertainty=uncertainty,
        realizations=realizations,
    )
    uncertainty_hash = uncertainty.uncertainty_hash
    enumeration_hash = _enumeration_hash(
        uncertainty_hash=uncertainty_hash,
        realizations=ordered,
    )

    phase_a_families = {value.phase_a_decision_family_hash for value in ordered}
    if len(phase_a_families) != 1:
        sources = _ambiguity_sources(
            realizations=ordered,
            decision_hash_name="phase_a_decision_family_hash",
        )
        if V1SignedWorkDecisionAmbiguitySource.SIGNED_F_SAMPLING in sources:
            sources = tuple(
                dict.fromkeys((*sources, V1SignedWorkDecisionAmbiguitySource.CAUSAL_LEAKAGE))
            )
            disposition = V1SignedWorkDecisionWitnessDisposition.FAIL_CLOSED
            next_rank = None
            detail = "Phase A changed with signed-F-only uncertainty"
        else:
            disposition, next_rank, detail = _refinement_disposition(
                uncertainty=uncertainty,
                sources=sources,
            )
        return V1SignedWorkDecisionStabilityRefusal(
            disposition=disposition,
            unresolved_phase=V1SignedWorkDecisionPhase.PHASE_A,
            reason=f"Phase-A Q/resource family is not invariant: {detail}",
            uncertainty_set_hash=uncertainty_hash,
            enumeration_hash=enumeration_hash,
            ambiguity_sources=sources,
            next_rank=next_rank,
        )

    phase_a = _phase_a_certificate(
        uncertainty_hash=uncertainty_hash,
        realizations=ordered,
    )
    phase_b_families = {value.phase_b_decision_family_hash for value in ordered}
    if len(phase_b_families) != 1:
        sources = _ambiguity_sources(
            realizations=ordered,
            decision_hash_name="phase_b_decision_family_hash",
        )
        disposition, next_rank, detail = _refinement_disposition(
            uncertainty=uncertainty,
            sources=sources,
        )
        return V1SignedWorkDecisionStabilityRefusal(
            disposition=disposition,
            unresolved_phase=V1SignedWorkDecisionPhase.PHASE_B,
            reason=f"Phase-B signed-work action family is not invariant: {detail}",
            uncertainty_set_hash=uncertainty_hash,
            enumeration_hash=enumeration_hash,
            ambiguity_sources=sources,
            phase_a_certificate=phase_a,
            next_rank=next_rank,
        )

    phase_b = _phase_b_certificate(
        uncertainty_hash=uncertainty_hash,
        phase_a=phase_a,
        realizations=ordered,
    )
    return V1SignedWorkDecisionStabilityCertificate(
        uncertainty_set_hash=uncertainty_hash,
        enumeration_hash=enumeration_hash,
        phase_a=phase_a,
        phase_b=phase_b,
        _issuance_token=_STABILITY_ISSUANCE_TOKEN,
    )


def _matrix(value: object, *, name: str) -> tuple[tuple[float, ...], ...]:
    if type(value) not in {tuple, list} or not value:
        raise TypeError(f"{name} must be one nonempty matrix")
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for raw_row in value:
        if type(raw_row) not in {tuple, list} or not raw_row:
            raise TypeError(f"{name} rows must be nonempty sequences")
        row = tuple(_finite(item, name=f"{name} entry") for item in raw_row)
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        rows.append(row)
    return tuple(rows)


def _probability_weights(
    value: object,
    *,
    count: int,
) -> tuple[float, ...]:
    if type(value) not in {tuple, list}:
        raise TypeError("context weights must be one sequence")
    result = tuple(_nonnegative(item, name="context weight") for item in value)
    if len(result) != count:
        raise ValueError("context weights do not align with rows")
    if not math.isclose(math.fsum(result), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("context weights must sum to one")
    return result


def _action(value: object, *, route_count: int) -> tuple[float, ...]:
    if type(value) not in {tuple, list}:
        raise TypeError("action must be one route-axis sequence")
    action = tuple(_nonnegative(item, name="action gate") for item in value)
    if len(action) != route_count or any(item > 1.0 for item in action):
        raise ValueError("action must lie on the route-axis box 0<=z<=1")
    return action


def v1_singleton_corrected_evidence_magnitudes(
    *,
    corrected_evidence_rows: Sequence[Sequence[float]],
    context_weights: Sequence[float],
) -> tuple[float, ...]:
    """Return ``A_j=E|a_j|`` for diagnostics, never certificate authority."""

    rows = _matrix(corrected_evidence_rows, name="corrected evidence rows")
    weights = _probability_weights(context_weights, count=len(rows))
    return tuple(
        math.fsum(weight * abs(row[route]) for weight, row in zip(weights, rows))
        for route in range(len(rows[0]))
    )


def v1_native_fiber_signed_work_reference(
    *,
    corrected_evidence_rows: Sequence[Sequence[float]],
    context_weights: Sequence[float],
    route_native_fiber_ids: Sequence[str],
    native_fiber_weights: Mapping[str, float],
    action: Sequence[float],
) -> float:
    """Evaluate exact small-domain native-fiber signed corrected work.

    Cancellation occurs inside one context and one registered native fiber.
    Absolute value is applied before averaging contexts and before summing
    distinct fibers, so neither cross-context nor cross-fiber cancellation is
    admitted.
    """

    rows = _matrix(corrected_evidence_rows, name="corrected evidence rows")
    weights = _probability_weights(context_weights, count=len(rows))
    route_count = len(rows[0])
    fibers = tuple(_identity(item, name="native fiber id") for item in route_native_fiber_ids)
    if len(fibers) != route_count:
        raise ValueError("native-fiber axis does not align with routes")
    unique_fibers = tuple(dict.fromkeys(fibers))
    if set(native_fiber_weights) != set(unique_fibers):
        raise ValueError("native-fiber weights must cover the exact fiber axis")
    fiber_weights = {
        fiber: _nonnegative(native_fiber_weights[fiber], name="native-fiber weight")
        for fiber in unique_fibers
    }
    gates = _action(action, route_count=route_count)
    total = 0.0
    for context_weight, row in zip(weights, rows):
        projections = {fiber: 0.0 for fiber in unique_fibers}
        for route, fiber in enumerate(fibers):
            projections[fiber] += row[route] * gates[route]
        total += context_weight * math.fsum(
            fiber_weights[fiber] * abs(projections[fiber]) for fiber in unique_fibers
        )
    return math.fsum((total,))


def v1_literal_j_quadratic_response_reference(
    *,
    literal_j_rows: Sequence[Sequence[float]],
    context_weights: Sequence[float],
    action: Sequence[float],
) -> float:
    """Evaluate ``Q(z)=E[(j(x)^T z)^2]`` in a literal small-domain chart."""

    rows = _matrix(literal_j_rows, name="literal-J rows")
    weights = _probability_weights(context_weights, count=len(rows))
    gates = _action(action, route_count=len(rows[0]))
    return math.fsum(
        weight * math.fsum(entry * gate for entry, gate in zip(row, gates)) ** 2
        for weight, row in zip(weights, rows)
    )


__all__ = [
    "SIGNED_WORK_D_W_AUTHORITY_SCOPE",
    "SIGNED_WORK_D_W_REFERENCE_RULE",
    "V1SignedWorkDWAxisBindingRole",
    "V1SignedWorkDWDirectionDescriptor",
    "V1SignedWorkDWShellEndpointRole",
    "V1SignedWorkDecisionAmbiguitySource",
    "V1SignedWorkDecisionPhase",
    "V1SignedWorkDecisionRealization",
    "V1SignedWorkDecisionStabilityCertificate",
    "V1SignedWorkDecisionStabilityRefusal",
    "V1SignedWorkDecisionUncertaintySet",
    "V1SignedWorkDecisionWitnessDisposition",
    "V1SignedWorkDecisionWitnessRefusal",
    "V1SignedWorkDecisionWitnessRefusalCode",
    "V1SignedWorkDecisionWitnessResult",
    "V1SignedWorkFDResidualUncertaintySet",
    "V1SignedWorkIntervalSubgradientRegion",
    "V1SignedWorkNativeFiberProjectionInterval",
    "V1SignedWorkPhaseACoordinateInvarianceCertificate",
    "V1SignedWorkPhaseBActionInvarianceCertificate",
    "V1SignedWorkPhysicalDedupProvenance",
    "V1SignedWorkSamplingUncertaintyRole",
    "V1SignedWorkSamplingUncertaintySet",
    "V1SignedWorkSolverRoundingUncertaintySet",
    "certify_v1_signed_work_d_decision_witness_reference",
    "v1_signed_work_d_w_physical_dedup_identity_hash",
    "v1_literal_j_quadratic_response_reference",
    "v1_native_fiber_signed_work_reference",
    "v1_singleton_corrected_evidence_magnitudes",
]
