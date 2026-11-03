"""Independent-C six-role acquisition for canonical signed-work modern v1.

The D stream may freeze candidate ladders, but none of its numerical F/Q
values may enter C inference.  This module binds the exact signed-work
``D -> frozen ladder -> C`` bridge to a fresh same-row measurement of six
distinct observables:

``F_native, Q_J, R_half, R_one, D_half, D_one``.

Every accepted C row advances all six roles for every endpoint on every
registered branch in one transaction.  F is native-fiber signed work and is
reported only as objective evidence; this contract intentionally contains no
loss-derived F threshold.  Q and R expose local-chart trust, while D is the
sharp finite raw-logit displacement measurement.  All receipts remain
evidence-only: they cannot select a candidate, open K/O, commit a mask, or
claim terminal/Pareto authority.

The statistical controller is scalar and may run on host, but the route-scale
producer is not.  Production designs accept only ``torch_cuda`` bridge
materializations.  The bounded NumPy bridge is admitted solely through an
explicit small-domain reference flag for focused tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, Literal, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_directional_remainder import V1DirectionalSecantDesign
from .modern_v1_fibered_direct_displacement import (
    V1FiberedDirectDisplacementObservation,
)
from .modern_v1_fibered_directional_record import V1FiberedDirectionalRecord
from .modern_v1_fqr_sequential_acquisition import (
    _binary64_scaled_integer,
    _downward_fraction_to_probability,
    _downward_weighted_sum,
    _fraction_to_float_outward,
    _outward_weighted_sum,
    _response_burden,
    _response_burden_lower,
    _scaled_mean,
    _serfling_radius_upper,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_device_shell_c_bridge import (
    V1SignedWorkCBranchBundle,
    V1SignedWorkCBridgeFamilyBundle,
)


F_NATIVE_ROLE: Final = "F_native"
Q_J_ROLE: Final = "Q_J"
R_HALF_ROLE: Final = "R_half"
R_ONE_ROLE: Final = "R_one"
D_HALF_ROLE: Final = "D_half"
D_ONE_ROLE: Final = "D_one"
SIGNED_WORK_SIX_ROLES: Final = (
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    D_HALF_ROLE,
    D_ONE_ROLE,
)
SIGNED_WORK_SIX_ROLE_DOSES: Final = {
    F_NATIVE_ROLE: None,
    Q_J_ROLE: None,
    R_HALF_ROLE: 0.5,
    R_ONE_ROLE: 1.0,
    D_HALF_ROLE: 0.5,
    D_ONE_ROLE: 1.0,
}

SIGNED_WORK_SIX_ROLE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_design_v1"
)
SIGNED_WORK_SIX_ROLE_BRANCH_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_branch_design_v1"
)
SIGNED_WORK_SIX_ROLE_CELL_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_cell_v1"
)
SIGNED_WORK_SIX_ROLE_ENDPOINT_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_endpoint_v1"
)
SIGNED_WORK_SIX_ROLE_BRANCH_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_branch_progress_v1"
)
SIGNED_WORK_SIX_ROLE_PROGRESS_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_progress_v1"
)
SIGNED_WORK_SIX_ROLE_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_signed_work_independent_C_six_role_snapshot_v1"
)
SIGNED_WORK_SIX_ROLE_AUTHORITY: Final = (
    "fresh_independent_C_native_F_literal_J_Q_directional_R_and_direct_D_"
    "evidence_only_no_D_value_recycling_no_candidate_selection_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
SIGNED_WORK_SIX_ROLE_ALPHA_POLICY: Final = (
    "pre_C_branch_stratum_endpoint_role_two_tail_nonrecycling_all_prefix_v1"
)
SIGNED_WORK_SIX_ROLE_SAMPLING_LAW: Final = (
    "sealed_without_replacement_prefix_of_complete_registered_train_C_v1"
)
SIGNED_WORK_SIX_ROLE_METHOD: Final = (
    "separate_one_sided_lower_and_upper_hoeffding_serfling_v1"
)
SIGNED_WORK_SIX_ROLE_CENSUS_METHOD: Final = (
    "exact_finite_population_census_of_outward_row_envelopes_v1"
)

_DESIGN_PREFIX: Final = "modern-v1-signed-work-six-role-C-design-"
_BRANCH_DESIGN_PREFIX: Final = "modern-v1-signed-work-six-role-C-branch-design-"
_CELL_PREFIX: Final = "modern-v1-signed-work-six-role-C-cell-"
_ENDPOINT_PREFIX: Final = "modern-v1-signed-work-six-role-C-endpoint-"
_BRANCH_PROGRESS_PREFIX: Final = "modern-v1-signed-work-six-role-C-branch-progress-"
_PROGRESS_PREFIX: Final = "modern-v1-signed-work-six-role-C-progress-"
_SNAPSHOT_PREFIX: Final = "modern-v1-signed-work-six-role-C-snapshot-"
_CHAIN_PREFIX: Final = "modern-v1-signed-work-six-role-C-chain-"

SixRoleSamplingStatus = Literal[
    "awaiting_C",
    "partial_C_evidence",
    "full_C_census",
]


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


def _immutable_nonnegative_f64(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=np.dtype("<f8"), order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a binary64 array") from error
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or bool(np.any(array < 0.0))
    ):
        raise ValueError(f"{name} must be finite/nonnegative with shape {shape}")
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(shape)
    result.setflags(write=False)
    return result


def _axis_pairs(value: object, *, name: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        raise TypeError(f"{name} must be an ordered sequence")
    result: list[tuple[str, tuple[str, ...]]] = []
    for raw in value:
        if type(raw) not in {tuple, list} or len(raw) != 2:
            raise ValueError(f"{name} rows must be name/axis pairs")
        axis_name = _identity(raw[0], name=f"{name} name")
        if type(raw[1]) not in {tuple, list}:
            raise TypeError(f"{name} values must be ordered")
        axis = tuple(_identity(item, name=f"{name} identity") for item in raw[1])
        if not axis or len(set(axis)) != len(axis):
            raise ValueError(f"{name} axes must be nonempty and unique")
        result.append((axis_name, axis))
    rows = tuple(result)
    if not rows or len({name for name, _axis in rows}) != len(rows):
        raise ValueError(f"{name} names must be nonempty and unique")
    return rows


def _role_bound_matrix(branch: "V1SignedWorkSixRoleBranchDesign") -> np.ndarray:
    return np.stack(
        (
            branch.analytic_f_native_upper_by_stratum_endpoint,
            branch.analytic_q_j_upper_by_stratum_endpoint,
            branch.analytic_r_upper_by_stratum_endpoint_and_dose[:, :, 0],
            branch.analytic_r_upper_by_stratum_endpoint_and_dose[:, :, 1],
            branch.analytic_d_upper_by_stratum_endpoint_and_dose[:, :, 0],
            branch.analytic_d_upper_by_stratum_endpoint_and_dose[:, :, 1],
        ),
        axis=2,
    )


def _observation_matrices(
    record: V1FiberedDirectionalRecord,
    direct: V1FiberedDirectDisplacementObservation,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.stack(
        (
            record.f_lower_observation_by_endpoint,
            record.q_lower_observation_by_endpoint,
            record.r_lower_observation_by_endpoint[:, 0],
            record.r_lower_observation_by_endpoint[:, 1],
            direct.d_half_lower_by_endpoint,
            direct.d_one_lower_by_endpoint,
        ),
        axis=1,
    )
    upper = np.stack(
        (
            record.f_upper_observation_by_endpoint,
            record.q_upper_observation_by_endpoint,
            record.r_upper_observation_by_endpoint[:, 0],
            record.r_upper_observation_by_endpoint[:, 1],
            direct.d_half_upper_by_endpoint,
            direct.d_one_upper_by_endpoint,
        ),
        axis=1,
    )
    return lower, upper


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleBranchDesign:
    """One pre-C signed-work branch and its six analytic support families."""

    branch: V1RelaxedKKTBranch
    bridge_branch: V1SignedWorkCBranchBundle = field(repr=False, compare=False)
    secant_design: V1DirectionalSecantDesign = field(repr=False, compare=False)
    sealed_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    analytic_f_native_upper_by_stratum_endpoint: np.ndarray = field(repr=False, compare=False)
    analytic_q_j_upper_by_stratum_endpoint: np.ndarray = field(repr=False, compare=False)
    analytic_r_upper_by_stratum_endpoint_and_dose: np.ndarray = field(repr=False, compare=False)
    analytic_d_upper_by_stratum_endpoint_and_dose: np.ndarray = field(repr=False, compare=False)
    f_native_bound_source_hash: str
    q_j_bound_source_hash: str
    directional_remainder_bound_source_hash: str
    direct_displacement_bound_source_hash: str
    production_route_scale_cuda_required: bool = True
    allow_private_small_cpu_reference: bool = False
    private_reference_route_limit: int = 4096
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    D_FQ_values_reused_as_C_observations: bool = False
    schema: str = SIGNED_WORK_SIX_ROLE_BRANCH_DESIGN_SCHEMA
    _design_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("six-role branch must use the exact KKT branch enum")
        if type(self.bridge_branch) is not V1SignedWorkCBranchBundle:
            raise TypeError("six-role branch requires the exact signed-work C bridge")
        if type(self.secant_design) is not V1DirectionalSecantDesign:
            raise TypeError("six-role branch requires the exact secant design")
        self.secant_design.assert_integrity()
        if self.bridge_branch.ladder is None:
            raise ValueError("six-role C acquisition requires a nonempty frozen ladder")
        self.bridge_branch.ladder.assert_integrity()
        receipt = self.bridge_branch.receipt
        if (
            receipt.branch is not self.branch
            or receipt.C_outcomes_seen
            or receipt.C_observation_count != 0
            or receipt.D_FQ_values_reused_as_C_observations
            or self.secant_design.ladder_hash != self.bridge_branch.ladder.ladder_hash
            or self.secant_design.endpoint_action_hashes
            != tuple(value.frozen_action_hash for value in self.bridge_branch.crosswalks)
        ):
            raise ValueError("six-role branch left its exact pre-C bridge identity")
        contexts = self.secant_design.certification_context_ids_by_stratum
        draws = _axis_pairs(self.sealed_draw_ids_by_stratum, name="sealed C draw ledger")
        if (
            tuple(name for name, _axis in draws) != tuple(name for name, _axis in contexts)
            or tuple(len(axis) for _name, axis in draws)
            != tuple(len(axis) for _name, axis in contexts)
        ):
            raise ValueError("six-role C draw ledger differs from its context population")
        shape = (len(contexts), len(self.secant_design.endpoint_action_hashes))
        f_bound = _immutable_nonnegative_f64(
            self.analytic_f_native_upper_by_stratum_endpoint,
            shape=shape,
            name="native-F analytic bounds",
        )
        q_bound = _immutable_nonnegative_f64(
            self.analytic_q_j_upper_by_stratum_endpoint,
            shape=shape,
            name="literal-J Q analytic bounds",
        )
        r_bound = _immutable_nonnegative_f64(
            self.analytic_r_upper_by_stratum_endpoint_and_dose,
            shape=(*shape, 2),
            name="R analytic bounds",
        )
        d_bound = _immutable_nonnegative_f64(
            self.analytic_d_upper_by_stratum_endpoint_and_dose,
            shape=(*shape, 2),
            name="D analytic bounds",
        )
        for name in (
            "f_native_bound_source_hash",
            "q_j_bound_source_hash",
            "directional_remainder_bound_source_hash",
            "direct_displacement_bound_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        limit = _nonnegative_int(
            self.private_reference_route_limit,
            name="private reference route limit",
        )
        if any(
            type(value) is not bool
            for value in (
                self.production_route_scale_cuda_required,
                self.allow_private_small_cpu_reference,
                self.issued_before_C_outcomes,
                self.C_outcomes_seen,
                self.D_FQ_values_reused_as_C_observations,
            )
        ):
            raise TypeError("six-role branch flags must be exact booleans")
        if (
            not self.issued_before_C_outcomes
            or self.C_outcomes_seen
            or self.D_FQ_values_reused_as_C_observations
        ):
            raise PermissionError("six-role branch was post-selected or recycled D values")
        backends = {value.backend_kind for value in self.bridge_branch.materializations}
        route_count = self.secant_design.ladder.route_count
        if self.production_route_scale_cuda_required:
            if self.allow_private_small_cpu_reference or backends != {"torch_cuda"}:
                raise PermissionError("production six-role C requires the CUDA route-scale bridge")
        elif backends != {"torch_cuda"}:
            if (
                not self.allow_private_small_cpu_reference
                or backends != {"numpy_reference"}
                or route_count > limit
            ):
                raise PermissionError("CPU route-scale fallback is not authorized")
        if self.schema != SIGNED_WORK_SIX_ROLE_BRANCH_DESIGN_SCHEMA:
            raise ValueError("unsupported six-role branch design schema")
        object.__setattr__(self, "sealed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "analytic_f_native_upper_by_stratum_endpoint", f_bound)
        object.__setattr__(self, "analytic_q_j_upper_by_stratum_endpoint", q_bound)
        object.__setattr__(self, "analytic_r_upper_by_stratum_endpoint_and_dose", r_bound)
        object.__setattr__(self, "analytic_d_upper_by_stratum_endpoint_and_dose", d_bound)
        object.__setattr__(self, "private_reference_route_limit", limit)
        self.assert_integrity()

    @property
    def stratum_names(self) -> tuple[str, ...]:
        return tuple(name for name, _axis in self.secant_design.certification_context_ids_by_stratum)

    @property
    def population_sizes(self) -> tuple[int, ...]:
        return tuple(len(axis) for _name, axis in self.secant_design.certification_context_ids_by_stratum)

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.secant_design.endpoint_action_hashes

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch.value,
            "bridge_branch_receipt_hash": self.bridge_branch.receipt.receipt_hash,
            "frozen_ladder_hash": self.bridge_branch.ladder.ladder_hash,
            "secant_design_hash": self.secant_design.design_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "certification_context_ids_by_stratum": self.secant_design.certification_context_ids_by_stratum,
            "sealed_draw_ids_by_stratum": self.sealed_draw_ids_by_stratum,
            "analytic_f_native_upper_by_stratum_endpoint": self.analytic_f_native_upper_by_stratum_endpoint,
            "analytic_q_j_upper_by_stratum_endpoint": self.analytic_q_j_upper_by_stratum_endpoint,
            "analytic_r_upper_by_stratum_endpoint_and_dose": self.analytic_r_upper_by_stratum_endpoint_and_dose,
            "analytic_d_upper_by_stratum_endpoint_and_dose": self.analytic_d_upper_by_stratum_endpoint_and_dose,
            "f_native_bound_source_hash": self.f_native_bound_source_hash,
            "q_j_bound_source_hash": self.q_j_bound_source_hash,
            "directional_remainder_bound_source_hash": self.directional_remainder_bound_source_hash,
            "direct_displacement_bound_source_hash": self.direct_displacement_bound_source_hash,
            "production_route_scale_cuda_required": self.production_route_scale_cuda_required,
            "allow_private_small_cpu_reference": self.allow_private_small_cpu_reference,
            "private_reference_route_limit": self.private_reference_route_limit,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "D_FQ_values_reused_as_C_observations": self.D_FQ_values_reused_as_C_observations,
        }

    @property
    def design_hash(self) -> str:
        return self.assert_integrity() if self._design_hash is None else self._design_hash

    def _assert_bridge_binding(self) -> None:
        receipt = self.bridge_branch.receipt
        ladder = self.bridge_branch.ladder
        if ladder is None:
            raise ValueError("six-role branch lost its frozen C ladder")
        ladder.assert_integrity()
        if (
            receipt.branch is not self.branch
            or receipt.C_outcomes_seen
            or receipt.C_observation_count != 0
            or receipt.D_FQ_values_reused_as_C_observations
            or tuple(value.receipt_hash for value in self.bridge_branch.endpoint_decisions)
            != receipt.endpoint_decision_receipt_hashes
            or tuple(value.receipt_hash for value in self.bridge_branch.materializations)
            != receipt.materialization_receipt_hashes
            or tuple(value.receipt_hash for value in self.bridge_branch.crosswalks)
            != receipt.crosswalk_receipt_hashes
            or ladder.ladder_hash != receipt.frozen_ladder_hash
            or self.secant_design.ladder_hash != ladder.ladder_hash
            or self.secant_design.endpoint_action_hashes
            != tuple(value.frozen_action_hash for value in self.bridge_branch.crosswalks)
        ):
            raise ValueError("six-role branch bridge binding is stale")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.secant_design.assert_integrity()
        self._assert_bridge_binding()
        observed = content_hash(self._payload(), prefix=_BRANCH_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError("six-role branch design was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("six-role branch design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleCDesign:
    """Pre-register all branch/stratum/endpoint/role/tail coordinates."""

    design_id: str
    bridge_family: V1SignedWorkCBridgeFamilyBundle = field(repr=False, compare=False)
    branch_designs: tuple[V1SignedWorkSixRoleBranchDesign, ...] = field(repr=False, compare=False)
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_C_six_role: float
    six_role_family_alpha_allocation: float
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    alpha_policy: str = SIGNED_WORK_SIX_ROLE_ALPHA_POLICY
    sampling_law: str = SIGNED_WORK_SIX_ROLE_SAMPLING_LAW
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    validation_outcomes_seen: bool = False
    signed_F_is_objective_not_default_feasibility_constraint: bool = True
    D_FQ_values_reused_as_C_observations: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY
    schema: str = SIGNED_WORK_SIX_ROLE_DESIGN_SCHEMA
    cell_count: int = field(init=False)
    tail_count: int = field(init=False)
    tail_family_alpha: float = field(init=False)
    registered_tail_alpha_total: float = field(init=False)
    _design_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _identity(self.design_id, name="six-role design id"))
        if type(self.bridge_family) is not V1SignedWorkCBridgeFamilyBundle:
            raise TypeError("six-role family requires the exact signed-work C bridge family")
        designs = tuple(self.branch_designs)
        if (
            not designs
            or any(type(value) is not V1SignedWorkSixRoleBranchDesign for value in designs)
            or len({value.branch for value in designs}) != len(designs)
        ):
            raise ValueError("six-role branch designs must be nonempty, exact, and unique")
        bridge_by_branch = {
            self.bridge_family.spectral.receipt.branch: self.bridge_family.spectral,
            self.bridge_family.diagonal_l1.receipt.branch: self.bridge_family.diagonal_l1,
        }
        for design in designs:
            design.assert_integrity()
            if design.bridge_branch.receipt.receipt_hash != bridge_by_branch[design.branch].receipt.receipt_hash:
                raise ValueError("six-role branch names another C bridge family")
        expected_branches = tuple(
            branch for branch, bundle in bridge_by_branch.items() if bundle.ladder is not None
        )
        if tuple(value.branch for value in designs) != expected_branches:
            raise ValueError("six-role branch axis must cover every nonempty C bridge branch in order")
        first = designs[0].secant_design
        for design in designs[1:]:
            secant = design.secant_design
            if (
                secant.certification_context_ids_by_stratum
                != first.certification_context_ids_by_stratum
                or secant.stratum_weights != first.stratum_weights
                or secant.certification_context_population_hash
                != first.certification_context_population_hash
                or secant.certification_stream_commitment_hash
                != first.certification_stream_commitment_hash
                or design.sealed_draw_ids_by_stratum != designs[0].sealed_draw_ids_by_stratum
            ):
                raise ValueError("six-role branches must share one sealed C population and prefix")
        for name in (
            "trajectory_alpha_ledger_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha-ledger generation",
        )
        if generation != first.ladder.proposal_generation:
            raise ValueError("six-role alpha ledger belongs to another proposal generation")
        familywise = _finite_probability(self.trajectory_familywise_alpha, name="trajectory familywise alpha")
        committed = float(self.trajectory_alpha_committed_before_C_six_role)
        if not math.isfinite(committed) or committed < 0.0:
            raise ValueError("committed alpha must be finite and nonnegative")
        allocation = _finite_probability(self.six_role_family_alpha_allocation, name="six-role family alpha")
        if Fraction.from_float(committed) + Fraction.from_float(allocation) > Fraction.from_float(familywise):
            raise ValueError("six-role C allocation overspends the trajectory ledger")
        cells = sum(
            len(design.stratum_names) * len(design.endpoint_action_hashes) * len(SIGNED_WORK_SIX_ROLES)
            for design in designs
        )
        tails = 2 * cells
        tail_alpha = _downward_fraction_to_probability(Fraction.from_float(allocation) / tails)
        registered = math.fsum(tail_alpha for _ in range(tails))
        if registered > allocation:
            raise ValueError("rounded six-role tail allocation overspends its family")
        required_true = (
            self.issued_before_C_outcomes,
            self.signed_F_is_objective_not_default_feasibility_constraint,
        )
        required_false = (
            self.C_outcomes_seen,
            self.validation_outcomes_seen,
            self.D_FQ_values_reused_as_C_observations,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in required_true + required_false):
            raise TypeError("six-role authority flags must be exact booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError("six-role C design crossed its pre-outcome evidence seam")
        if (
            self.alpha_policy != SIGNED_WORK_SIX_ROLE_ALPHA_POLICY
            or self.sampling_law != SIGNED_WORK_SIX_ROLE_SAMPLING_LAW
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY
            or self.schema != SIGNED_WORK_SIX_ROLE_DESIGN_SCHEMA
        ):
            raise ValueError("unsupported six-role sampling or authority semantics")
        object.__setattr__(self, "branch_designs", designs)
        object.__setattr__(self, "trajectory_alpha_ledger_generation", generation)
        object.__setattr__(self, "trajectory_familywise_alpha", familywise)
        object.__setattr__(self, "trajectory_alpha_committed_before_C_six_role", committed)
        object.__setattr__(self, "six_role_family_alpha_allocation", allocation)
        object.__setattr__(self, "cell_count", cells)
        object.__setattr__(self, "tail_count", tails)
        object.__setattr__(self, "tail_family_alpha", tail_alpha)
        object.__setattr__(self, "registered_tail_alpha_total", registered)
        self.assert_integrity()

    @property
    def branches(self) -> tuple[V1RelaxedKKTBranch, ...]:
        return tuple(value.branch for value in self.branch_designs)

    @property
    def alpha_coordinates(self) -> tuple[tuple[str, str, str, str, str], ...]:
        return tuple(
            (design.branch.value, stratum, action, role, tail)
            for design in self.branch_designs
            for stratum in design.stratum_names
            for action in design.endpoint_action_hashes
            for role in SIGNED_WORK_SIX_ROLES
            for tail in ("lower", "upper")
        )

    def branch_design(self, branch: V1RelaxedKKTBranch) -> V1SignedWorkSixRoleBranchDesign:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("six-role branch lookup requires the exact enum")
        try:
            return self.branch_designs[self.branches.index(branch)]
        except ValueError as error:
            raise ValueError("branch is not registered for six-role C") from error

    def look_alpha(self, *, branch: V1RelaxedKKTBranch, stratum_index: int, sample_size: int) -> float:
        design = self.branch_design(branch)
        index = _nonnegative_int(stratum_index, name="stratum index")
        if index >= len(design.population_sizes):
            raise ValueError("stratum index left the registered axis")
        count = _positive_int(sample_size, name="sample size")
        population = design.population_sizes[index]
        if count > population:
            raise ValueError("sample size exceeds the registered population")
        if count == population:
            return 0.0
        schedule = Fraction(population, count * (count + 1) * (population - 1))
        return _downward_fraction_to_probability(Fraction.from_float(self.tail_family_alpha) * schedule)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            "bridge_family_bundle_hash": self.bridge_family.bundle_hash,
            "branch_design_hashes": tuple(value.design_hash for value in self.branch_designs),
            "branches": tuple(value.value for value in self.branches),
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": self.trajectory_alpha_ledger_generation,
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_C_six_role": self.trajectory_alpha_committed_before_C_six_role,
            "six_role_family_alpha_allocation": self.six_role_family_alpha_allocation,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "roles": SIGNED_WORK_SIX_ROLES,
            "alpha_coordinates": self.alpha_coordinates,
            "cell_count": self.cell_count,
            "tail_count": self.tail_count,
            "tail_family_alpha": self.tail_family_alpha,
            "registered_tail_alpha_total": self.registered_tail_alpha_total,
            "alpha_policy": self.alpha_policy,
            "sampling_law": self.sampling_law,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "signed_F_is_objective_not_default_feasibility_constraint": self.signed_F_is_objective_not_default_feasibility_constraint,
            "D_FQ_values_reused_as_C_observations": self.D_FQ_values_reused_as_C_observations,
            "candidate_selection_authority": self.candidate_selection_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def design_hash(self) -> str:
        return self.assert_integrity() if self._design_hash is None else self._design_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.branch_designs:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError("six-role C design was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("six-role C design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleCellSummary:
    branch: V1RelaxedKKTBranch
    stratum_id: str
    endpoint_index: int
    action_hash: str
    role: str
    dose: float | None
    population_size: int
    sample_size: int
    lower: float
    upper: float
    value_upper_bound: float
    current_lower_look_alpha: float
    current_upper_look_alpha: float
    full_census: bool
    method: str
    observation_chain_hash: str
    schema: str = SIGNED_WORK_SIX_ROLE_CELL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("six-role cell branch must be exact")
        _identity(self.stratum_id, name="cell stratum")
        _positive_int(self.endpoint_index, name="cell endpoint index")
        _identity(self.action_hash, name="cell action hash")
        if self.role not in SIGNED_WORK_SIX_ROLES or self.dose != SIGNED_WORK_SIX_ROLE_DOSES[self.role]:
            raise ValueError("six-role cell role/dose is invalid")
        population = _positive_int(self.population_size, name="cell population")
        count = _nonnegative_int(self.sample_size, name="cell sample size")
        if count > population:
            raise ValueError("six-role cell sample exceeds population")
        values = tuple(float(value) for value in (self.lower, self.upper, self.value_upper_bound))
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not values[0] <= values[1] <= values[2]:
            raise ValueError("six-role cell interval exceeds its analytic support")
        for value in (self.current_lower_look_alpha, self.current_upper_look_alpha):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("six-role look alpha must be finite/nonnegative")
        if type(self.full_census) is not bool or self.full_census != (count == population):
            raise ValueError("six-role census flag is inconsistent")
        expected_method = SIGNED_WORK_SIX_ROLE_CENSUS_METHOD if self.full_census else SIGNED_WORK_SIX_ROLE_METHOD
        if count == 0:
            expected_method = "unobserved_registered_support_v1"
        if self.method != expected_method:
            raise ValueError("six-role cell method is inconsistent")
        _identity(self.observation_chain_hash, name="cell observation chain")
        if self.schema != SIGNED_WORK_SIX_ROLE_CELL_SCHEMA:
            raise ValueError("unsupported six-role cell schema")

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_CELL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleEndpointSummary:
    branch: V1RelaxedKKTBranch
    endpoint_index: int
    action_hash: str
    f_native_lower: float
    f_native_upper: float
    q_j_lower: float
    q_j_upper: float
    r_half_lower: float
    r_half_upper: float
    r_one_lower: float
    r_one_upper: float
    d_half_lower: float
    d_half_upper: float
    d_one_lower: float
    d_one_upper: float
    chart_half_burden_lower: float
    chart_half_burden_upper: float
    chart_one_burden_lower: float
    chart_one_burden_upper: float
    signed_F_is_objective_evidence: bool = True
    Q_and_R_are_chart_trust_evidence: bool = True
    D_is_finite_response_admission_evidence: bool = True
    candidate_selection_authority: bool = False
    commit_authority: bool = False
    schema: str = SIGNED_WORK_SIX_ROLE_ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("six-role endpoint branch must be exact")
        _positive_int(self.endpoint_index, name="endpoint index")
        _identity(self.action_hash, name="endpoint action hash")
        pairs = (
            (self.f_native_lower, self.f_native_upper),
            (self.q_j_lower, self.q_j_upper),
            (self.r_half_lower, self.r_half_upper),
            (self.r_one_lower, self.r_one_upper),
            (self.d_half_lower, self.d_half_upper),
            (self.d_one_lower, self.d_one_upper),
            (self.chart_half_burden_lower, self.chart_half_burden_upper),
            (self.chart_one_burden_lower, self.chart_one_burden_upper),
        )
        if any(
            not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) < 0.0
            or float(lower) > float(upper)
            for lower, upper in pairs
        ):
            raise ValueError("six-role endpoint intervals are invalid")
        if (
            self.signed_F_is_objective_evidence is not True
            or self.Q_and_R_are_chart_trust_evidence is not True
            or self.D_is_finite_response_admission_evidence is not True
            or self.candidate_selection_authority is not False
            or self.commit_authority is not False
        ):
            raise PermissionError("six-role endpoint changed its evidence-only role split")
        if self.schema != SIGNED_WORK_SIX_ROLE_ENDPOINT_SCHEMA:
            raise ValueError("unsupported six-role endpoint schema")

    @property
    def summary_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleBranchProgress:
    branch: V1RelaxedKKTBranch
    branch_design_hash: str
    observed_counts_by_stratum: tuple[int, ...]
    next_expected_context_by_stratum: tuple[str | None, ...]
    next_expected_draw_by_stratum: tuple[str | None, ...]
    cell_summaries: tuple[V1SignedWorkSixRoleCellSummary, ...]
    endpoint_summaries: tuple[V1SignedWorkSixRoleEndpointSummary, ...]
    branch_observation_chain_hash: str
    schema: str = SIGNED_WORK_SIX_ROLE_BRANCH_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("six-role branch progress branch must be exact")
        _identity(self.branch_design_hash, name="branch design hash")
        counts = tuple(_nonnegative_int(value, name="observed C count") for value in self.observed_counts_by_stratum)
        object.__setattr__(self, "observed_counts_by_stratum", counts)
        if any(value is not None and (type(value) is not str or not value) for value in self.next_expected_context_by_stratum + self.next_expected_draw_by_stratum):
            raise ValueError("six-role next C coordinates are invalid")
        if any(type(value) is not V1SignedWorkSixRoleCellSummary for value in self.cell_summaries):
            raise TypeError("six-role cells must use exact summaries")
        if any(type(value) is not V1SignedWorkSixRoleEndpointSummary for value in self.endpoint_summaries):
            raise TypeError("six-role endpoints must use exact summaries")
        _identity(self.branch_observation_chain_hash, name="branch observation chain")
        if self.schema != SIGNED_WORK_SIX_ROLE_BRANCH_PROGRESS_SCHEMA:
            raise ValueError("unsupported six-role branch progress schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PROGRESS_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleProgressReceipt:
    design_hash: str
    bridge_family_bundle_hash: str
    branch_receipts: tuple[V1SignedWorkSixRoleBranchProgress, ...]
    shared_record_count: int
    transaction_chain_hash: str
    sampling_status: SixRoleSamplingStatus
    tail_count: int
    registered_tail_alpha_total: float
    D_FQ_values_reused_as_C_observations: bool = False
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY
    schema: str = SIGNED_WORK_SIX_ROLE_PROGRESS_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.design_hash, name="six-role design hash")
        _identity(self.bridge_family_bundle_hash, name="C bridge family hash")
        if not self.branch_receipts or any(type(value) is not V1SignedWorkSixRoleBranchProgress for value in self.branch_receipts):
            raise TypeError("six-role progress requires exact branch receipts")
        _nonnegative_int(self.shared_record_count, name="shared C record count")
        _identity(self.transaction_chain_hash, name="six-role transaction chain")
        if self.sampling_status not in {"awaiting_C", "partial_C_evidence", "full_C_census"}:
            raise ValueError("unknown six-role sampling status")
        _positive_int(self.tail_count, name="tail count")
        if not math.isfinite(self.registered_tail_alpha_total) or self.registered_tail_alpha_total <= 0.0:
            raise ValueError("six-role registered alpha must be finite/positive")
        if any(
            getattr(self, name) is not False
            for name in (
                "D_FQ_values_reused_as_C_observations",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
            )
        ):
            raise PermissionError("six-role progress crossed its evidence seam")
        if self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY or self.schema != SIGNED_WORK_SIX_ROLE_PROGRESS_SCHEMA:
            raise ValueError("unsupported six-role progress authority/schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PROGRESS_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleAccumulatorSnapshot:
    design_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    counts_by_branch_and_stratum: tuple[tuple[int, ...], ...]
    lower_scaled_sums_by_branch: tuple[tuple[int, ...], ...]
    upper_scaled_sums_by_branch: tuple[tuple[int, ...], ...]
    branch_observation_chain_hashes: tuple[str, ...]
    shared_record_count: int
    transaction_chain_hash: str
    schema: str = SIGNED_WORK_SIX_ROLE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.design_hash, name="snapshot design hash")
        if not self.branches or any(type(value) is not V1RelaxedKKTBranch for value in self.branches):
            raise TypeError("snapshot branches must use exact enums")
        if not (
            len(self.branches)
            == len(self.counts_by_branch_and_stratum)
            == len(self.lower_scaled_sums_by_branch)
            == len(self.upper_scaled_sums_by_branch)
            == len(self.branch_observation_chain_hashes)
        ):
            raise ValueError("snapshot branch axes are misaligned")
        for row in self.counts_by_branch_and_stratum:
            tuple(_nonnegative_int(value, name="snapshot observed count") for value in row)
        for row in self.lower_scaled_sums_by_branch + self.upper_scaled_sums_by_branch:
            if any(isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0 for value in row):
                raise ValueError("snapshot sufficient sums must be nonnegative integers")
        for value in self.branch_observation_chain_hashes:
            _identity(value, name="snapshot branch chain")
        _nonnegative_int(self.shared_record_count, name="snapshot shared record count")
        _identity(self.transaction_chain_hash, name="snapshot transaction chain")
        if self.schema != SIGNED_WORK_SIX_ROLE_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported six-role snapshot schema")

    @property
    def snapshot_hash(self) -> str:
        return content_hash(self, prefix=_SNAPSHOT_PREFIX)


class V1SignedWorkSixRoleAccumulator:
    """Atomic same-C acquisition across every registered signed-work branch."""

    def __init__(self, design: V1SignedWorkSixRoleCDesign) -> None:
        if type(design) is not V1SignedWorkSixRoleCDesign:
            raise TypeError("six-role accumulator requires its exact design")
        design_hash = design.assert_integrity()
        self.design = design
        self._counts = [np.zeros(len(value.stratum_names), dtype=np.int64) for value in design.branch_designs]
        self._lower_sums = [
            np.zeros(
                (len(value.stratum_names), len(value.endpoint_action_hashes), len(SIGNED_WORK_SIX_ROLES)),
                dtype=object,
            )
            for value in design.branch_designs
        ]
        self._upper_sums = [np.zeros_like(value, dtype=object) for value in self._lower_sums]
        self._branch_chains = [
            content_hash(
                {"design_hash": design_hash, "branch": value.branch.value, "record_count": 0},
                prefix=_CHAIN_PREFIX,
            )
            for value in design.branch_designs
        ]
        self._shared_record_count = 0
        self._transaction_chain = content_hash(
            {"design_hash": design_hash, "bridge_family_bundle_hash": design.bridge_family.bundle_hash, "record_count": 0},
            prefix=_CHAIN_PREFIX,
        )

    @classmethod
    def from_snapshot(
        cls,
        design: V1SignedWorkSixRoleCDesign,
        snapshot: V1SignedWorkSixRoleAccumulatorSnapshot,
        *,
        expected_snapshot_hash: str | None = None,
    ) -> "V1SignedWorkSixRoleAccumulator":
        if type(snapshot) is not V1SignedWorkSixRoleAccumulatorSnapshot:
            raise TypeError("six-role restore requires the exact snapshot")
        if snapshot.shared_record_count and expected_snapshot_hash is None:
            raise ValueError("nonempty six-role restore requires an expected snapshot hash")
        if expected_snapshot_hash is not None and snapshot.snapshot_hash != expected_snapshot_hash:
            raise ValueError("six-role snapshot hash does not match")
        if snapshot.design_hash != design.design_hash or snapshot.branches != design.branches:
            raise ValueError("six-role snapshot belongs to another design")
        result = cls(design)
        for index, branch_design in enumerate(design.branch_designs):
            shape = result._lower_sums[index].shape
            expected = math.prod(shape)
            if (
                len(snapshot.counts_by_branch_and_stratum[index]) != len(branch_design.stratum_names)
                or len(snapshot.lower_scaled_sums_by_branch[index]) != expected
                or len(snapshot.upper_scaled_sums_by_branch[index]) != expected
            ):
                raise ValueError("six-role snapshot sufficient-statistic shape is stale")
            result._counts[index] = np.asarray(snapshot.counts_by_branch_and_stratum[index], dtype=np.int64)
            result._lower_sums[index] = np.asarray(snapshot.lower_scaled_sums_by_branch[index], dtype=object).reshape(shape)
            result._upper_sums[index] = np.asarray(snapshot.upper_scaled_sums_by_branch[index], dtype=object).reshape(shape)
        result._branch_chains = list(snapshot.branch_observation_chain_hashes)
        result._shared_record_count = snapshot.shared_record_count
        result._transaction_chain = snapshot.transaction_chain_hash
        result.progress_receipt()
        return result

    @property
    def shared_record_count(self) -> int:
        return self._shared_record_count

    def _validate_row(
        self,
        *,
        branch_index: int,
        record: V1FiberedDirectionalRecord,
        direct: V1FiberedDirectDisplacementObservation,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        design = self.design.branch_designs[branch_index]
        if type(record) is not V1FiberedDirectionalRecord or type(direct) is not V1FiberedDirectDisplacementObservation:
            raise TypeError("six-role C requires exact fibered-record/direct-displacement pairs")
        record_hash = record.assert_integrity()
        direct.assert_integrity()
        if (
            direct.directional_record_hash != record_hash
            or record.design_hash != design.secant_design.design_hash
            or direct.design_hash != design.secant_design.design_hash
            or record.ladder_hash != design.secant_design.ladder_hash
            or direct.ladder_hash != design.secant_design.ladder_hash
            or record.endpoint_action_hashes != design.endpoint_action_hashes
            or direct.endpoint_action_hashes != design.endpoint_action_hashes
            or record.native_fiber_axis_receipt_hash != self.design.native_fiber_axis_receipt_hash
            or direct.native_fiber_axis_receipt_hash != self.design.native_fiber_axis_receipt_hash
            or record.native_fiber_measure_hash != self.design.native_fiber_measure_hash
            or direct.native_fiber_measure_hash != self.design.native_fiber_measure_hash
            or record.certification_stream_commitment_hash
            != design.secant_design.certification_stream_commitment_hash
            or direct.certification_stream_commitment_hash
            != design.secant_design.certification_stream_commitment_hash
            or record.joint_fqr_family_registration_hash
            != design.secant_design.joint_fqr_family_registration_hash
            or direct.joint_fqr_family_registration_hash
            != design.secant_design.joint_fqr_family_registration_hash
            or direct.source_bound_source_hash
            != design.direct_displacement_bound_source_hash
        ):
            raise ValueError("six-role C row left its branch/ladder/fiber/stream identity")
        try:
            stratum = design.stratum_names.index(record.stratum_id)
        except ValueError as error:
            raise ValueError("six-role C row names an unregistered stratum") from error
        count = int(self._counts[branch_index][stratum])
        contexts = design.secant_design.certification_context_ids_by_stratum[stratum][1]
        draws = design.sealed_draw_ids_by_stratum[stratum][1]
        if count >= len(contexts):
            raise ValueError("six-role C stratum is already censused")
        if (
            record.sealed_order_index != count
            or direct.sealed_order_index != count
            or record.context_id != contexts[count]
            or direct.context_id != contexts[count]
            or record.draw_id != draws[count]
            or direct.draw_id != draws[count]
            or direct.context_id != record.context_id
            or direct.draw_id != record.draw_id
        ):
            raise ValueError("six-role C row is not the next sealed prefix coordinate")
        lower, upper = _observation_matrices(record, direct)
        bound = _role_bound_matrix(design)[stratum]
        if bool(np.any(lower > upper)) or bool(np.any(upper > bound)):
            raise ValueError("six-role C observation exceeds its pre-outcome analytic support")
        return stratum, lower, upper

    def record_atomic(
        self,
        rows_by_branch: Sequence[
            tuple[V1RelaxedKKTBranch, V1FiberedDirectionalRecord, V1FiberedDirectDisplacementObservation]
        ],
    ) -> V1SignedWorkSixRoleProgressReceipt:
        """Advance every branch and every role, or leave live state unchanged."""

        self.design.assert_integrity()
        rows = tuple(rows_by_branch)
        if tuple(value[0] for value in rows) != self.design.branches:
            raise ValueError("six-role transaction must cover branches in frozen order")
        validated = tuple(
            self._validate_row(branch_index=index, record=row[1], direct=row[2])
            for index, row in enumerate(rows)
        )
        shared_coordinates = tuple(
            (
                row[1].stratum_id,
                row[1].sealed_order_index,
                row[1].context_id,
                row[1].draw_id,
                row[1].parent_raw_target_logit_center,
                row[1].parent_raw_target_logit_radius,
                row[1].parent_execution_identity_hash,
                row[1].corrected_evidence_stream_identity_hash,
                row[1].literal_jacobian_stream_identity_hash,
                row[1].raw_logit_stream_identity_hash,
                row[1].same_capture_and_vjp_execution_proof_hash,
            )
            for row in rows
        )
        if any(value != shared_coordinates[0] for value in shared_coordinates[1:]):
            raise ValueError("six-role branches did not measure the same atomic C row")

        counts = [value.copy() for value in self._counts]
        lower_sums = [value.copy() for value in self._lower_sums]
        upper_sums = [value.copy() for value in self._upper_sums]
        branch_chains = list(self._branch_chains)
        for index, (row, (stratum, lower, upper)) in enumerate(zip(rows, validated, strict=True)):
            for endpoint in range(lower.shape[0]):
                for role in range(lower.shape[1]):
                    lower_sums[index][stratum, endpoint, role] += _binary64_scaled_integer(float(lower[endpoint, role]))
                    upper_sums[index][stratum, endpoint, role] += _binary64_scaled_integer(float(upper[endpoint, role]))
            counts[index][stratum] += 1
            branch_chains[index] = content_hash(
                {
                    "previous_chain_hash": branch_chains[index],
                    "branch": row[0].value,
                    "record_hash": row[1].record_hash,
                    "direct_displacement_hash": row[2].observation_hash,
                    "sealed_C_coordinate": shared_coordinates[0][:4],
                    "six_roles_advanced": SIGNED_WORK_SIX_ROLES,
                },
                prefix=_CHAIN_PREFIX,
            )
        new_count = self._shared_record_count + 1
        transaction_chain = content_hash(
            {
                "previous_chain_hash": self._transaction_chain,
                "shared_record_count": new_count,
                "sealed_C_coordinate": shared_coordinates[0],
                "branch_record_pairs": tuple(
                    (row[0].value, row[1].record_hash, row[2].observation_hash)
                    for row in rows
                ),
                "six_roles_advanced_atomically": SIGNED_WORK_SIX_ROLES,
            },
            prefix=_CHAIN_PREFIX,
        )
        self._counts = counts
        self._lower_sums = lower_sums
        self._upper_sums = upper_sums
        self._branch_chains = branch_chains
        self._shared_record_count = new_count
        self._transaction_chain = transaction_chain
        return self.progress_receipt()

    def _cell(
        self,
        *,
        branch_index: int,
        stratum: int,
        endpoint: int,
        role: int,
    ) -> V1SignedWorkSixRoleCellSummary:
        design = self.design.branch_designs[branch_index]
        count = int(self._counts[branch_index][stratum])
        population = design.population_sizes[stratum]
        bound = float(_role_bound_matrix(design)[stratum, endpoint, role])
        lower_sum = int(self._lower_sums[branch_index][stratum, endpoint, role])
        upper_sum = int(self._upper_sums[branch_index][stratum, endpoint, role])
        if count == 0:
            lower, upper, look_alpha = 0.0, bound, 0.0
            method = "unobserved_registered_support_v1"
        elif count == population:
            lower = _fraction_to_float_outward(_scaled_mean(lower_sum, count), direction="lower")
            upper = _fraction_to_float_outward(_scaled_mean(upper_sum, count), direction="upper")
            look_alpha = 0.0
            method = SIGNED_WORK_SIX_ROLE_CENSUS_METHOD
        else:
            lower_mean = _fraction_to_float_outward(_scaled_mean(lower_sum, count), direction="lower")
            upper_mean = _fraction_to_float_outward(_scaled_mean(upper_sum, count), direction="upper")
            look_alpha = self.design.look_alpha(branch=design.branch, stratum_index=stratum, sample_size=count)
            radius = _serfling_radius_upper(
                upper_bound=bound,
                count=count,
                population=population,
                look_alpha=look_alpha,
            )
            lower = max(0.0, math.nextafter(lower_mean - radius, -math.inf))
            upper = min(bound, math.nextafter(upper_mean + radius, math.inf))
            method = SIGNED_WORK_SIX_ROLE_METHOD
        return V1SignedWorkSixRoleCellSummary(
            branch=design.branch,
            stratum_id=design.stratum_names[stratum],
            endpoint_index=endpoint + 1,
            action_hash=design.endpoint_action_hashes[endpoint],
            role=SIGNED_WORK_SIX_ROLES[role],
            dose=SIGNED_WORK_SIX_ROLE_DOSES[SIGNED_WORK_SIX_ROLES[role]],
            population_size=population,
            sample_size=count,
            lower=lower,
            upper=upper,
            value_upper_bound=bound,
            current_lower_look_alpha=look_alpha,
            current_upper_look_alpha=look_alpha,
            full_census=count == population,
            method=method,
            observation_chain_hash=self._branch_chains[branch_index],
        )

    def progress_receipt(self) -> V1SignedWorkSixRoleProgressReceipt:
        branch_receipts: list[V1SignedWorkSixRoleBranchProgress] = []
        all_counts: list[tuple[int, ...]] = []
        for branch_index, design in enumerate(self.design.branch_designs):
            cells = tuple(
                self._cell(branch_index=branch_index, stratum=stratum, endpoint=endpoint, role=role)
                for stratum in range(len(design.stratum_names))
                for endpoint in range(len(design.endpoint_action_hashes))
                for role in range(len(SIGNED_WORK_SIX_ROLES))
            )
            weights = tuple(weight for _name, weight in design.secant_design.stratum_weights)
            endpoints: list[V1SignedWorkSixRoleEndpointSummary] = []
            for endpoint, action in enumerate(design.endpoint_action_hashes):
                lowers: dict[str, float] = {}
                uppers: dict[str, float] = {}
                for role_index, role in enumerate(SIGNED_WORK_SIX_ROLES):
                    role_cells = tuple(
                        cells[
                            (stratum * len(design.endpoint_action_hashes) + endpoint)
                            * len(SIGNED_WORK_SIX_ROLES)
                            + role_index
                        ]
                        for stratum in range(len(design.stratum_names))
                    )
                    lowers[role] = _downward_weighted_sum(weights, tuple(value.lower for value in role_cells))
                    uppers[role] = _outward_weighted_sum(weights, tuple(value.upper for value in role_cells))
                endpoints.append(
                    V1SignedWorkSixRoleEndpointSummary(
                        branch=design.branch,
                        endpoint_index=endpoint + 1,
                        action_hash=action,
                        f_native_lower=lowers[F_NATIVE_ROLE],
                        f_native_upper=uppers[F_NATIVE_ROLE],
                        q_j_lower=lowers[Q_J_ROLE],
                        q_j_upper=uppers[Q_J_ROLE],
                        r_half_lower=lowers[R_HALF_ROLE],
                        r_half_upper=uppers[R_HALF_ROLE],
                        r_one_lower=lowers[R_ONE_ROLE],
                        r_one_upper=uppers[R_ONE_ROLE],
                        d_half_lower=lowers[D_HALF_ROLE],
                        d_half_upper=uppers[D_HALF_ROLE],
                        d_one_lower=lowers[D_ONE_ROLE],
                        d_one_upper=uppers[D_ONE_ROLE],
                        chart_half_burden_lower=_response_burden_lower(
                            eta=0.5,
                            q_lower=lowers[Q_J_ROLE],
                            r_lower=lowers[R_HALF_ROLE],
                        ),
                        chart_half_burden_upper=_response_burden(
                            eta=0.5,
                            q_upper=uppers[Q_J_ROLE],
                            r_upper=uppers[R_HALF_ROLE],
                        ),
                        chart_one_burden_lower=_response_burden_lower(
                            eta=1.0,
                            q_lower=lowers[Q_J_ROLE],
                            r_lower=lowers[R_ONE_ROLE],
                        ),
                        chart_one_burden_upper=_response_burden(
                            eta=1.0,
                            q_upper=uppers[Q_J_ROLE],
                            r_upper=uppers[R_ONE_ROLE],
                        ),
                    )
                )
            counts = tuple(int(value) for value in self._counts[branch_index])
            all_counts.append(counts)
            contexts = design.secant_design.certification_context_ids_by_stratum
            draws = design.sealed_draw_ids_by_stratum
            branch_receipts.append(
                V1SignedWorkSixRoleBranchProgress(
                    branch=design.branch,
                    branch_design_hash=design.design_hash,
                    observed_counts_by_stratum=counts,
                    next_expected_context_by_stratum=tuple(
                        None if count == len(contexts[index][1]) else contexts[index][1][count]
                        for index, count in enumerate(counts)
                    ),
                    next_expected_draw_by_stratum=tuple(
                        None if count == len(draws[index][1]) else draws[index][1][count]
                        for index, count in enumerate(counts)
                    ),
                    cell_summaries=cells,
                    endpoint_summaries=tuple(endpoints),
                    branch_observation_chain_hash=self._branch_chains[branch_index],
                )
            )
        total_observed = sum(sum(values) for values in all_counts)
        total_population = sum(sum(value.population_sizes) for value in self.design.branch_designs)
        status: SixRoleSamplingStatus
        if total_observed == 0:
            status = "awaiting_C"
        elif total_observed == total_population:
            status = "full_C_census"
        else:
            status = "partial_C_evidence"
        receipt = V1SignedWorkSixRoleProgressReceipt(
            design_hash=self.design.design_hash,
            bridge_family_bundle_hash=self.design.bridge_family.bundle_hash,
            branch_receipts=tuple(branch_receipts),
            shared_record_count=self._shared_record_count,
            transaction_chain_hash=self._transaction_chain,
            sampling_status=status,
            tail_count=self.design.tail_count,
            registered_tail_alpha_total=self.design.registered_tail_alpha_total,
        )
        return receipt

    def export_snapshot(self) -> V1SignedWorkSixRoleAccumulatorSnapshot:
        snapshot = V1SignedWorkSixRoleAccumulatorSnapshot(
            design_hash=self.design.design_hash,
            branches=self.design.branches,
            counts_by_branch_and_stratum=tuple(
                tuple(int(value) for value in row) for row in self._counts
            ),
            lower_scaled_sums_by_branch=tuple(
                tuple(int(value) for value in row.reshape(-1)) for row in self._lower_sums
            ),
            upper_scaled_sums_by_branch=tuple(
                tuple(int(value) for value in row.reshape(-1)) for row in self._upper_sums
            ),
            branch_observation_chain_hashes=tuple(self._branch_chains),
            shared_record_count=self._shared_record_count,
            transaction_chain_hash=self._transaction_chain,
        )
        return snapshot


__all__ = [
    "D_HALF_ROLE",
    "D_ONE_ROLE",
    "F_NATIVE_ROLE",
    "Q_J_ROLE",
    "R_HALF_ROLE",
    "R_ONE_ROLE",
    "SIGNED_WORK_SIX_ROLES",
    "SIGNED_WORK_SIX_ROLE_ALPHA_POLICY",
    "SIGNED_WORK_SIX_ROLE_AUTHORITY",
    "SIGNED_WORK_SIX_ROLE_DESIGN_SCHEMA",
    "V1SignedWorkSixRoleAccumulator",
    "V1SignedWorkSixRoleAccumulatorSnapshot",
    "V1SignedWorkSixRoleBranchDesign",
    "V1SignedWorkSixRoleBranchProgress",
    "V1SignedWorkSixRoleCDesign",
    "V1SignedWorkSixRoleCellSummary",
    "V1SignedWorkSixRoleEndpointSummary",
    "V1SignedWorkSixRoleProgressReceipt",
]
