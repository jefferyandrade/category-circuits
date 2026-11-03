"""Pre-C joint F/Q/R design factory for the retained KKT branch family.

Every retained convex KKT branch receives its own frozen ladder, directional
secant design, F/Q/R certification design, sequential C-stream acquisition,
and compact resource registration.  The C population, sealed stream/draw
axes, strata, chart, generation, and family registration are shared exactly.
D and C may use distinct sealed permutations, but both are bound to the same
complete training-population identity.

The statistical allocation is made once over the complete
branch-by-stratum-by-endpoint-by-role family.  Branch-local sequential designs
receive only their registered portion of that allocation; no branch may reuse
the full family alpha.  Everything here is fixed before C outcomes and grants
no K, O, commit, terminal-validation, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_directional_remainder import V1DirectionalSecantDesign
from .modern_v1_fqr_branch_family import (
    V1FQRKKTBranchFrozenBinding,
    V1FQRKKTPreFrozenBranchFamily,
)
from .modern_v1_fqr_device_shell_materialized_family import (
    V1FQRDeviceShellPackedBranchBinding,
    V1FQRDeviceShellPackedPreFrozenBranchFamily,
)
from .modern_v1_fqr_certification import V1FQRCertificationDesign
from .modern_v1_fqr_compact_certification import (
    V1CompactFQRPreCResourceRegistration,
)
from .modern_v1_fqr_sequential_acquisition import (
    FQR_SEQUENTIAL_ROLES,
    V1FQRSequentialAcquisitionDesign,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch


FQR_BRANCH_ANALYTIC_INPUT_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_analytic_input_v1"
)
FQR_JOINT_ALPHA_CELL_SCHEMA: Final = (
    "partition_free_v1_fqr_joint_branch_alpha_cell_v1"
)
FQR_JOINT_FAMILY_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_fqr_joint_branch_family_registration_v1"
)
FQR_BRANCH_DESIGN_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_design_bundle_v1"
)
FQR_BRANCH_FAMILY_DESIGN_SCHEMA: Final = (
    "partition_free_v1_fqr_branch_family_design_v1"
)
FQR_BRANCH_FAMILY_ALPHA_POLICY: Final = (
    "single_uniform_nonrecycling_allocation_over_retained_branch_stratum_"
    "endpoint_role_cells_v1"
)
FQR_BRANCH_FAMILY_AUTHORITY_SCOPE: Final = (
    "pre_C_joint_branch_FQR_design_only_no_C_outcome_K_O_commit_terminal_"
    "or_pareto_authority_v1"
)
FQR_BRANCH_ORDER: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)

_INPUT_PREFIX: Final = "modern-v1-fqr-branch-analytic-input-"
_CELL_PREFIX: Final = "modern-v1-fqr-joint-alpha-cell-"
_REGISTRATION_PREFIX: Final = "modern-v1-fqr-joint-family-registration-"
_BUNDLE_PREFIX: Final = "modern-v1-fqr-branch-design-bundle-"
_DESIGN_PREFIX: Final = "modern-v1-fqr-branch-family-design-"


class V1FQRBranchDesignRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    STALE_CHART = "stale_chart"
    POST_OUTCOME = "post_outcome"
    MISSING_BRANCH_INPUT = "missing_branch_input"
    EXTRA_BRANCH_INPUT = "extra_branch_input"
    ARRAY_MISMATCH = "array_mismatch"
    ALPHA_OVERSPEND = "alpha_overspend"
    DESIGN_COLLISION = "design_collision"
    INVALID_DOMINANCE = "invalid_dominance"
    RESOURCE_DRIFT = "resource_drift"
    AUTHORITY_VIOLATION = "authority_violation"
    TRAIN_POPULATION_MISMATCH = "train_population_mismatch"


class V1FQRBranchDesignRefusal(ValueError):
    def __init__(self, code: V1FQRBranchDesignRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQRBranchDesignRefusalCode, message: str) -> NoReturn:
    raise V1FQRBranchDesignRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite_positive(value, name=name)
    if result > 1.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must not exceed one",
        )
    return result


def _immutable_f64(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        _refuse(
            V1FQRBranchDesignRefusalCode.ARRAY_MISMATCH,
            f"{name} cannot be represented as float64: {error}",
        )
    if not np.isfinite(array).all() or np.any(array < 0.0):
        _refuse(
            V1FQRBranchDesignRefusalCode.ARRAY_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    snapshot = np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C"),
        dtype=np.float64,
    ).reshape(array.shape)
    snapshot.setflags(write=False)
    return snapshot


def _axis_pairs(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be an ordered sequence",
        )
    result: list[tuple[str, tuple[str, ...]]] = []
    all_values: list[str] = []
    for row in value:
        if type(row) not in {tuple, list} or len(row) != 2:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                f"{name} rows must be name/axis pairs",
            )
        axis_name = _identity(row[0], name=f"{name} axis name")
        if type(row[1]) not in {tuple, list}:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                f"{name} axis values must be ordered sequences",
            )
        axis = tuple(
            _identity(item, name=f"{name} value") for item in tuple(row[1])
        )
        if not axis or len(set(axis)) != len(axis):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                f"{name} axes must be unique and nonempty",
            )
        result.append((axis_name, axis))
        all_values.extend(axis)
    normalized = tuple(result)
    if (
        not normalized
        or len({axis_name for axis_name, _axis in normalized}) != len(normalized)
        or len(set(all_values)) != len(all_values)
    ):
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            f"{name} must have globally unique nonempty axes",
        )
    return normalized


def _weight_axis(
    value: object,
    *,
    names: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            "stratum weights must be an ordered sequence",
        )
    rows: list[tuple[str, float]] = []
    for row in value:
        if type(row) not in {tuple, list} or len(row) != 2:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "stratum weights must be name/value pairs",
            )
        rows.append(
            (
                _identity(row[0], name="stratum weight name"),
                _finite_positive(row[1], name="stratum weight"),
            )
        )
    result = tuple(rows)
    if tuple(name for name, _weight in result) != names:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            "stratum weights differ from the shared C strata",
        )
    total = math.fsum(weight for _name, weight in result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            "stratum weights must sum to one",
        )
    return tuple((name, weight / total) for name, weight in result)


def _fraction_to_float_downward(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
            "joint alpha cell underflowed binary64",
        )
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, 0.0)
    if result <= 0.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
            "joint alpha cell underflowed after outward rounding",
        )
    return result


def _fraction_to_float_upward(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
            "branch alpha allocation overflowed or underflowed binary64",
        )
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        _refuse(
            V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
            "branch alpha allocation overflowed binary64",
        )
    return result


def _joint_alpha_plan(
    *,
    allocation: float,
    branch_cell_counts: tuple[tuple[V1RelaxedKKTBranch, int], ...],
) -> tuple[float, tuple[tuple[V1RelaxedKKTBranch, float], ...]]:
    total_cells = sum(count for _branch, count in branch_cell_counts)
    exact_allocation = Fraction.from_float(allocation)
    cell_alpha = _fraction_to_float_downward(exact_allocation / total_cells)
    for _attempt in range(64):
        rows: list[tuple[V1RelaxedKKTBranch, float]] = []
        valid = True
        for branch, count in branch_cell_counts:
            target = Fraction.from_float(cell_alpha) * count
            branch_allocation = _fraction_to_float_upward(target)
            derived = _fraction_to_float_downward(
                Fraction.from_float(branch_allocation) / count
            )
            if derived != cell_alpha:
                valid = False
                break
            rows.append((branch, branch_allocation))
        exact_spend = sum(
            (Fraction.from_float(value) for _branch, value in rows),
            Fraction(0, 1),
        )
        registered_total = math.fsum(
            cell_alpha for _index in range(total_cells)
        )
        if (
            valid
            and exact_spend <= exact_allocation
            and registered_total <= allocation
        ):
            return cell_alpha, tuple(rows)
        cell_alpha = math.nextafter(cell_alpha, 0.0)
        if cell_alpha <= 0.0:
            break
    _refuse(
        V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
        "no binary64 joint alpha plan fits the registered family allocation",
    )


def _retained_branch_order(
    family: (
        V1FQRKKTPreFrozenBranchFamily
        | V1FQRDeviceShellPackedPreFrozenBranchFamily
    ),
) -> tuple[V1RelaxedKKTBranch, ...]:
    return tuple(
        branch
        for branch, binding in (
            (V1RelaxedKKTBranch.SPECTRAL, family.spectral),
            (V1RelaxedKKTBranch.DIAGONAL_L1, family.diagonal_l1),
        )
        if binding is not None
    )


@dataclass(frozen=True, slots=True)
class V1FQRBranchAnalyticInput:
    """Branch-specific analytic envelopes and deterministic source identities."""

    input_id: str
    branch: V1RelaxedKKTBranch
    analytic_f_upper_by_stratum_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    analytic_q_upper_by_stratum_endpoint: np.ndarray = field(
        repr=False,
        compare=False,
    )
    analytic_r_upper_by_stratum_endpoint_and_dose: np.ndarray = field(
        repr=False,
        compare=False,
    )
    corrected_work_statistical_design_hash: str
    literal_jacobian_statistical_design_hash: str
    corrected_work_bound_source_hash: str
    literal_jacobian_bound_source_hash: str
    directional_remainder_bound_source_hash: str
    fd_rank_uncertainty_set_hash: str
    kkt_numerical_uncertainty_set_hash: str
    numerical_enclosure_policy_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    bounds_mutable_after_C: bool = False
    validation_outcomes_seen: bool = False
    downstream_authority: bool = False
    schema: str = FQR_BRANCH_ANALYTIC_INPUT_SCHEMA
    _input_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_id", _identity(self.input_id, name="input id"))
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "analytic input requires the exact KKT branch enum",
            )
        f_bound = _immutable_f64(
            self.analytic_f_upper_by_stratum_endpoint,
            name="analytic F bounds",
        )
        q_bound = _immutable_f64(
            self.analytic_q_upper_by_stratum_endpoint,
            name="analytic Q bounds",
        )
        r_bound = _immutable_f64(
            self.analytic_r_upper_by_stratum_endpoint_and_dose,
            name="analytic R bounds",
        )
        if (
            f_bound.ndim != 2
            or q_bound.shape != f_bound.shape
            or r_bound.shape != (*f_bound.shape, 2)
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.ARRAY_MISMATCH,
                "branch F/Q/R analytic arrays do not share strata/endpoint axes",
            )
        for name in (
            "corrected_work_statistical_design_hash",
            "literal_jacobian_statistical_design_hash",
            "corrected_work_bound_source_hash",
            "literal_jacobian_bound_source_hash",
            "directional_remainder_bound_source_hash",
            "fd_rank_uncertainty_set_hash",
            "kkt_numerical_uncertainty_set_hash",
            "numerical_enclosure_policy_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.bounds_mutable_after_C is not False
            or self.validation_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch analytic envelopes were not frozen before C outcomes",
            )
        if self.downstream_authority is not False:
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch analytic input cannot grant downstream authority",
            )
        if self.schema != FQR_BRANCH_ANALYTIC_INPUT_SCHEMA:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch analytic input schema is invalid",
            )
        object.__setattr__(self, "analytic_f_upper_by_stratum_endpoint", f_bound)
        object.__setattr__(self, "analytic_q_upper_by_stratum_endpoint", q_bound)
        object.__setattr__(
            self,
            "analytic_r_upper_by_stratum_endpoint_and_dose",
            r_bound,
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "input_id",
                "branch",
                "analytic_f_upper_by_stratum_endpoint",
                "analytic_q_upper_by_stratum_endpoint",
                "analytic_r_upper_by_stratum_endpoint_and_dose",
                "corrected_work_statistical_design_hash",
                "literal_jacobian_statistical_design_hash",
                "corrected_work_bound_source_hash",
                "literal_jacobian_bound_source_hash",
                "directional_remainder_bound_source_hash",
                "fd_rank_uncertainty_set_hash",
                "kkt_numerical_uncertainty_set_hash",
                "numerical_enclosure_policy_hash",
                "issued_before_C_outcomes",
                "C_outcomes_seen",
                "bounds_mutable_after_C",
                "validation_outcomes_seen",
                "downstream_authority",
            )
        }

    @property
    def input_hash(self) -> str:
        cached = self._input_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.bounds_mutable_after_C is not False
            or self.validation_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch analytic input is no longer pre-C",
            )
        if self.downstream_authority is not False:
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch analytic input gained downstream authority",
            )
        observed = content_hash(self._payload(), prefix=_INPUT_PREFIX)
        cached = self._input_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch analytic input was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch analytic input hash does not match",
            )
        object.__setattr__(self, "_input_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRJointAlphaCell:
    branch: V1RelaxedKKTBranch
    stratum_id: str
    endpoint_index: int
    action_hash: str
    role: str
    cell_family_alpha: float
    nonrecycling: bool = True
    schema: str = FQR_JOINT_ALPHA_CELL_SCHEMA
    _cell_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "joint alpha cell requires the exact KKT branch enum",
            )
        object.__setattr__(
            self,
            "stratum_id",
            _identity(self.stratum_id, name="alpha-cell stratum"),
        )
        object.__setattr__(
            self,
            "endpoint_index",
            _positive_int(self.endpoint_index, name="alpha-cell endpoint"),
        )
        object.__setattr__(
            self,
            "action_hash",
            _identity(self.action_hash, name="alpha-cell action hash"),
        )
        if self.role not in FQR_SEQUENTIAL_ROLES:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "joint alpha cell names an unknown F/Q/R role",
            )
        object.__setattr__(
            self,
            "cell_family_alpha",
            _probability(self.cell_family_alpha, name="cell family alpha"),
        )
        if self.nonrecycling is not True or self.schema != FQR_JOINT_ALPHA_CELL_SCHEMA:
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "joint alpha cell must be nonrecycling and versioned",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "branch",
                "stratum_id",
                "endpoint_index",
                "action_hash",
                "role",
                "cell_family_alpha",
                "nonrecycling",
            )
        }

    @property
    def cell_hash(self) -> str:
        cached = self._cell_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.nonrecycling is not True:
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "joint alpha cell lost its nonrecycling guarantee",
            )
        observed = content_hash(self._payload(), prefix=_CELL_PREFIX)
        cached = self._cell_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "joint alpha cell was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "joint alpha cell hash does not match",
            )
        object.__setattr__(self, "_cell_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRJointFamilyRegistration:
    """One pre-C family identity and nonrecycling alpha allocation."""

    design_id: str
    prefrozen_family: (
        V1FQRKKTPreFrozenBranchFamily
        | V1FQRDeviceShellPackedPreFrozenBranchFamily
    ) = field(
        repr=False,
        compare=False,
    )
    branch_inputs: tuple[V1FQRBranchAnalyticInput, ...] = field(
        repr=False,
        compare=False,
    )
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    stratum_weights: tuple[tuple[str, float], ...]
    sealed_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    discovery_context_population_hash: str
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    packed_direction_provider_source_hash: str
    relaxed_gate_executor_source_hash: str
    raw_logit_interval_verifier_source_hash: str
    bce_damage_budget: float
    positive_affine_calibration_slope: float
    trajectory_alpha_ledger_hash: str
    trajectory_alpha_ledger_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_fqr: float
    fqr_family_alpha_allocation: float
    planned_cell_family_alpha: float
    branch_alpha_allocations: tuple[tuple[V1RelaxedKKTBranch, float], ...]
    alpha_cells: tuple[V1FQRJointAlphaCell, ...]
    total_cell_count: int
    registered_cell_alpha_total: float
    resource_units: str
    resource_source_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    K_outcomes_seen: bool = False
    O_outcomes_seen: bool = False
    allocation_nonrecycling: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    alpha_policy: str = FQR_BRANCH_FAMILY_ALPHA_POLICY
    authority_scope: str = FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
    schema: str = FQR_JOINT_FAMILY_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_id", _identity(self.design_id, name="design id"))
        if type(self.prefrozen_family) not in {
            V1FQRKKTPreFrozenBranchFamily,
            V1FQRDeviceShellPackedPreFrozenBranchFamily,
        }:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "joint registration requires an exact admitted pre-frozen family",
            )
        try:
            self.prefrozen_family.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                f"pre-frozen branch family integrity failed: {error}",
            )
        inputs = tuple(self.branch_inputs)
        if any(type(value) is not V1FQRBranchAnalyticInput for value in inputs):
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "joint registration requires exact branch analytic inputs",
            )
        for value in inputs:
            value.assert_integrity()
        retained = _retained_branch_order(self.prefrozen_family)
        observed = tuple(value.branch for value in inputs)
        if len(set(observed)) != len(observed):
            _refuse(
                V1FQRBranchDesignRefusalCode.DESIGN_COLLISION,
                "more than one analytic input names the same branch",
            )
        missing = tuple(branch for branch in retained if branch not in observed)
        extra = tuple(branch for branch in observed if branch not in retained)
        if missing:
            _refuse(
                V1FQRBranchDesignRefusalCode.MISSING_BRANCH_INPUT,
                f"retained branches lack analytic inputs: {missing}",
            )
        if extra:
            _refuse(
                V1FQRBranchDesignRefusalCode.EXTRA_BRANCH_INPUT,
                f"analytic inputs were supplied for omitted branches: {extra}",
            )
        if observed != retained:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch analytic inputs are not in retained-family order",
            )
        if len({value.input_id for value in inputs}) != len(inputs):
            _refuse(
                V1FQRBranchDesignRefusalCode.DESIGN_COLLISION,
                "branch analytic input IDs collide",
            )

        contexts = _axis_pairs(
            self.certification_context_ids_by_stratum,
            name="shared C population",
        )
        draws = _axis_pairs(
            self.sealed_draw_ids_by_stratum,
            name="shared sealed draws",
        )
        if (
            tuple(name for name, _axis in contexts)
            != tuple(name for name, _axis in draws)
            or tuple(len(axis) for _name, axis in contexts)
            != tuple(len(axis) for _name, axis in draws)
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "shared C context and draw populations differ",
            )
        weights = _weight_axis(
            self.stratum_weights,
            names=tuple(name for name, _axis in contexts),
        )
        for name in (
            "discovery_context_population_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
            "packed_direction_provider_source_hash",
            "relaxed_gate_executor_source_hash",
            "raw_logit_interval_verifier_source_hash",
            "trajectory_alpha_ledger_hash",
            "resource_units",
            "resource_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.discovery_context_population_hash
            != self.certification_context_population_hash
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.TRAIN_POPULATION_MISMATCH,
                "D discovery and C certification must be sealed permutations "
                "of the same complete training population",
            )
        budget = _finite_nonnegative(self.bce_damage_budget, name="BCE damage budget")
        slope = _finite_positive(
            self.positive_affine_calibration_slope,
            name="positive affine calibration slope",
        )
        generation = _nonnegative_int(
            self.trajectory_alpha_ledger_generation,
            name="trajectory alpha-ledger generation",
        )
        familywise = _probability(
            self.trajectory_familywise_alpha,
            name="trajectory familywise alpha",
        )
        committed = _finite_nonnegative(
            self.trajectory_alpha_committed_before_fqr,
            name="trajectory alpha committed before F/Q/R",
        )
        allocation = _probability(
            self.fqr_family_alpha_allocation,
            name="F/Q/R family alpha allocation",
        )
        planned = _probability(
            self.planned_cell_family_alpha,
            name="planned joint cell alpha",
        )
        registered = _finite_positive(
            self.registered_cell_alpha_total,
            name="registered cell alpha total",
        )
        if (
            Fraction.from_float(committed) + Fraction.from_float(allocation)
            > Fraction.from_float(familywise)
            or Fraction.from_float(registered) > Fraction.from_float(allocation)
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
                "joint F/Q/R family overspends the trajectory alpha ledger",
            )
        if (
            generation != self.prefrozen_family.proposal_generation
            or self.prefrozen_family.chart_identity_hash == ""
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.STALE_CHART,
                "joint alpha ledger belongs to another chart generation",
            )

        strata = len(contexts)
        expected_cell_axis: list[tuple[V1RelaxedKKTBranch, str, int, str, str]] = []
        for branch_input in inputs:
            binding = self.prefrozen_family.binding_for_branch(branch_input.branch)
            endpoints = len(binding.frozen_ladder.actions)
            expected_shape = (strata, endpoints)
            if (
                branch_input.analytic_f_upper_by_stratum_endpoint.shape
                != expected_shape
                or branch_input.analytic_q_upper_by_stratum_endpoint.shape
                != expected_shape
                or branch_input.analytic_r_upper_by_stratum_endpoint_and_dose.shape
                != (*expected_shape, 2)
            ):
                _refuse(
                    V1FQRBranchDesignRefusalCode.ARRAY_MISMATCH,
                    f"{branch_input.branch.value} analytic arrays differ from its "
                    "strata/ladder axes",
                )
            for stratum, _axis in contexts:
                for endpoint, action_hash in enumerate(
                    binding.frozen_ladder.action_hashes,
                    start=1,
                ):
                    for role in FQR_SEQUENTIAL_ROLES:
                        expected_cell_axis.append(
                            (
                                branch_input.branch,
                                stratum,
                                endpoint,
                                action_hash,
                                role,
                            )
                        )
        cells = tuple(self.alpha_cells)
        if any(type(value) is not V1FQRJointAlphaCell for value in cells):
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "joint registration requires exact alpha cells",
            )
        for value in cells:
            value.assert_integrity()
        observed_cell_axis = tuple(
            (
                value.branch,
                value.stratum_id,
                value.endpoint_index,
                value.action_hash,
                value.role,
            )
            for value in cells
        )
        total_cells = _positive_int(self.total_cell_count, name="total alpha cells")
        if (
            observed_cell_axis != tuple(expected_cell_axis)
            or total_cells != len(expected_cell_axis)
            or any(value.cell_family_alpha != planned for value in cells)
            or len({value.cell_hash for value in cells}) != len(cells)
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "joint alpha cells do not exactly cover the registered family",
            )
        expected_registered = math.fsum(planned for _cell in cells)
        if registered != expected_registered:
            _refuse(
                V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
                "registered alpha total does not replay from the joint cells",
            )
        allocations = tuple(self.branch_alpha_allocations)
        if (
            len(allocations) != len(inputs)
            or tuple(branch for branch, _value in allocations) != retained
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch alpha allocation axis differs from retained branches",
            )
        normalized_allocations: list[tuple[V1RelaxedKKTBranch, float]] = []
        for branch, value in allocations:
            if type(branch) is not V1RelaxedKKTBranch:
                _refuse(
                    V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                    "branch alpha allocation requires exact branch enums",
                )
            branch_allocation = _probability(
                value,
                name="branch F/Q/R alpha allocation",
            )
            branch_cells = sum(1 for cell in cells if cell.branch is branch)
            derived = _fraction_to_float_downward(
                Fraction.from_float(branch_allocation) / branch_cells
            )
            if derived != planned:
                _refuse(
                    V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
                    "branch allocation does not reproduce the joint cell alpha",
                )
            normalized_allocations.append((branch, branch_allocation))
        if sum(
            (Fraction.from_float(value) for _branch, value in normalized_allocations),
            Fraction(0, 1),
        ) > Fraction.from_float(allocation):
            _refuse(
                V1FQRBranchDesignRefusalCode.ALPHA_OVERSPEND,
                "branch allocations overspend the joint F/Q/R family",
            )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.K_outcomes_seen is not False
            or self.O_outcomes_seen is not False
            or self.allocation_nonrecycling is not True
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "joint family registration is not a pre-C nonrecycling design",
            )
        if any(
            value is not False
            for value in (
                self.calibration_K_open_authority,
                self.C_stream_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "joint family registration crossed its design-only authority",
            )
        if (
            self.alpha_policy != FQR_BRANCH_FAMILY_ALPHA_POLICY
            or self.authority_scope != FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
            or self.schema != FQR_JOINT_FAMILY_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "joint family registration policy or schema is invalid",
            )
        if len(inputs) == 1 and self.prefrozen_family.dominance_certificate is None:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_DOMINANCE,
                "one retained branch requires an exact pre-C dominance certificate",
            )
        object.__setattr__(self, "branch_inputs", inputs)
        object.__setattr__(self, "certification_context_ids_by_stratum", contexts)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "sealed_draw_ids_by_stratum", draws)
        object.__setattr__(self, "bce_damage_budget", budget)
        object.__setattr__(self, "positive_affine_calibration_slope", slope)
        object.__setattr__(self, "trajectory_alpha_ledger_generation", generation)
        object.__setattr__(self, "trajectory_familywise_alpha", familywise)
        object.__setattr__(self, "trajectory_alpha_committed_before_fqr", committed)
        object.__setattr__(self, "fqr_family_alpha_allocation", allocation)
        object.__setattr__(self, "planned_cell_family_alpha", planned)
        object.__setattr__(self, "branch_alpha_allocations", tuple(normalized_allocations))
        object.__setattr__(self, "alpha_cells", cells)
        object.__setattr__(self, "total_cell_count", total_cells)
        object.__setattr__(self, "registered_cell_alpha_total", registered)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        family = self.prefrozen_family
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            "prefrozen_family_hash": family.family_hash,
            "problem_hash": family.problem_hash,
            "policy_hash": family.policy_hash,
            "result_hash": family.result_hash,
            "retained_branches": _retained_branch_order(family),
            "branch_input_hashes": tuple(value.input_hash for value in self.branch_inputs),
            "route_axis_hash": family.route_axis_hash,
            "route_count": family.route_count,
            "current_mask_hash": family.current_mask_hash,
            "chart_identity_hash": family.chart_identity_hash,
            "proposal_generation": family.proposal_generation,
            "certification_context_ids_by_stratum": self.certification_context_ids_by_stratum,
            "stratum_weights": self.stratum_weights,
            "sealed_draw_ids_by_stratum": self.sealed_draw_ids_by_stratum,
            "discovery_context_population_hash": self.discovery_context_population_hash,
            "certification_context_population_hash": self.certification_context_population_hash,
            "certification_stream_commitment_hash": self.certification_stream_commitment_hash,
            "packed_direction_provider_source_hash": self.packed_direction_provider_source_hash,
            "relaxed_gate_executor_source_hash": self.relaxed_gate_executor_source_hash,
            "raw_logit_interval_verifier_source_hash": self.raw_logit_interval_verifier_source_hash,
            "bce_damage_budget": self.bce_damage_budget,
            "positive_affine_calibration_slope": self.positive_affine_calibration_slope,
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": self.trajectory_alpha_ledger_generation,
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_fqr": self.trajectory_alpha_committed_before_fqr,
            "fqr_family_alpha_allocation": self.fqr_family_alpha_allocation,
            "planned_cell_family_alpha": self.planned_cell_family_alpha,
            "branch_alpha_allocations": self.branch_alpha_allocations,
            "alpha_cell_hashes": tuple(value.cell_hash for value in self.alpha_cells),
            "total_cell_count": self.total_cell_count,
            "registered_cell_alpha_total": self.registered_cell_alpha_total,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "K_outcomes_seen": self.K_outcomes_seen,
            "O_outcomes_seen": self.O_outcomes_seen,
            "allocation_nonrecycling": self.allocation_nonrecycling,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "alpha_policy": self.alpha_policy,
            "authority_scope": self.authority_scope,
        }

    @property
    def registration_hash(self) -> str:
        cached = self._registration_hash
        return self.assert_integrity() if cached is None else cached

    def alpha_allocation_for_branch(self, branch: V1RelaxedKKTBranch) -> float:
        for registered_branch, value in self.branch_alpha_allocations:
            if registered_branch is branch:
                return value
        _refuse(
            V1FQRBranchDesignRefusalCode.EXTRA_BRANCH_INPUT,
            f"{branch.value} is not retained by this joint registration",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            self.prefrozen_family.assert_integrity()
            for value in self.branch_inputs:
                value.assert_integrity()
            for value in self.alpha_cells:
                value.assert_integrity()
        except V1FQRBranchDesignRefusal:
            raise
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                f"joint registration source integrity failed: {error}",
            )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.K_outcomes_seen is not False
            or self.O_outcomes_seen is not False
            or self.allocation_nonrecycling is not True
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "joint registration is no longer pre-C",
            )
        if any(
            value is not False
            for value in (
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "joint registration gained downstream authority",
            )
        observed = content_hash(self._payload(), prefix=_REGISTRATION_PREFIX)
        cached = self._registration_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "joint family registration was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "joint family registration hash does not match",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRBranchDesignBundle:
    """Exact per-branch child designs under one joint family registration."""

    registration_hash: str
    branch: V1RelaxedKKTBranch
    branch_binding: (
        V1FQRKKTBranchFrozenBinding | V1FQRDeviceShellPackedBranchBinding
    ) = field(repr=False, compare=False)
    secant_design: V1DirectionalSecantDesign = field(repr=False, compare=False)
    fqr_design: V1FQRCertificationDesign = field(repr=False, compare=False)
    sequential_design: V1FQRSequentialAcquisitionDesign = field(
        repr=False,
        compare=False,
    )
    resource_registration: V1CompactFQRPreCResourceRegistration = field(
        repr=False,
        compare=False,
    )
    branch_alpha_allocation: float
    branch_cell_count: int
    cell_family_alpha: float
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = FQR_BRANCH_DESIGN_BUNDLE_SCHEMA
    _bundle_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registration_hash",
            _identity(self.registration_hash, name="joint registration hash"),
        )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "branch bundle requires the exact KKT branch enum",
            )
        if type(self.branch_binding) not in {
            V1FQRKKTBranchFrozenBinding,
            V1FQRDeviceShellPackedBranchBinding,
        }:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "branch bundle requires an exact admitted branch binding",
            )
        expected = (
            (self.secant_design, V1DirectionalSecantDesign),
            (self.fqr_design, V1FQRCertificationDesign),
            (self.sequential_design, V1FQRSequentialAcquisitionDesign),
            (
                self.resource_registration,
                V1CompactFQRPreCResourceRegistration,
            ),
        )
        if any(type(value) is not expected_type for value, expected_type in expected):
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "branch bundle child design type drifted",
            )
        allocation = _probability(
            self.branch_alpha_allocation,
            name="branch alpha allocation",
        )
        cells = _positive_int(self.branch_cell_count, name="branch alpha cells")
        cell_alpha = _probability(
            self.cell_family_alpha,
            name="branch cell family alpha",
        )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch design bundle is not pre-C",
            )
        if any(
            value is not False
            for value in (
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch design bundle crossed its pre-C authority boundary",
            )
        if self.schema != FQR_BRANCH_DESIGN_BUNDLE_SCHEMA:
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch design bundle schema is invalid",
            )
        object.__setattr__(self, "branch_alpha_allocation", allocation)
        object.__setattr__(self, "branch_cell_count", cells)
        object.__setattr__(self, "cell_family_alpha", cell_alpha)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration_hash,
            "branch": self.branch,
            "branch_binding_hash": self.branch_binding.binding_hash,
            "secant_design_hash": self.secant_design.design_hash,
            "fqr_design_hash": self.fqr_design.design_hash,
            "sequential_design_hash": self.sequential_design.design_hash,
            "resource_registration_hash": self.resource_registration.registration_hash,
            "branch_alpha_allocation": self.branch_alpha_allocation,
            "branch_cell_count": self.branch_cell_count,
            "cell_family_alpha": self.cell_family_alpha,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
        }

    @property
    def bundle_hash(self) -> str:
        cached = self._bundle_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.issued_before_C_outcomes is not True or self.C_outcomes_seen is not False:
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch design bundle is no longer pre-C",
            )
        if any(
            value is not False
            for value in (
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch design bundle gained downstream authority",
            )
        try:
            self.branch_binding.assert_integrity()
            self.secant_design.assert_integrity()
            self.fqr_design.assert_integrity()
            self.sequential_design.assert_integrity()
            self.resource_registration.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                f"branch child design integrity failed: {error}",
            )
        binding = self.branch_binding
        ladder = binding.frozen_ladder
        resources = self.resource_registration
        if (
            binding.branch is not self.branch
            or self.secant_design.ladder_hash != ladder.ladder_hash
            or self.fqr_design.secant_design.design_hash
            != self.secant_design.design_hash
            or self.sequential_design.fqr_design.design_hash
            != self.fqr_design.design_hash
            or self.secant_design.joint_fqr_family_registration_hash
            != self.registration_hash
            or self.fqr_design.joint_fqr_family_registration_hash
            != self.registration_hash
            or self.sequential_design.joint_fqr_family_registration_hash
            != self.registration_hash
            or resources.acquisition_design_hash != self.sequential_design.design_hash
            or resources.ladder_hash != ladder.ladder_hash
            or resources.chart_identity_hash != binding.chart_identity_hash
            or resources.current_mask_hash != binding.current_mask_hash
            or resources.proposal_generation != binding.proposal_generation
            or resources.route_axis_hash != binding.route_axis_hash
            or resources.route_count != binding.route_count
            or resources.endpoint_action_hashes != ladder.action_hashes
            or tuple(row[2] for row in resources.resource_by_endpoint)
            != binding.endpoint_registered_resources
            or self.sequential_design.fqr_family_alpha_allocation
            != self.branch_alpha_allocation
            or self.sequential_design.cell_count != self.branch_cell_count
            or self.sequential_design.cell_family_alpha != self.cell_family_alpha
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch child designs left their registered ladder/family coordinates",
            )
        observed = content_hash(self._payload(), prefix=_BUNDLE_PREFIX)
        cached = self._bundle_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch design bundle was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch design bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRBranchFamilyDesign:
    """Complete pre-C child-design family under one common registration hash."""

    registration: V1FQRJointFamilyRegistration = field(repr=False, compare=False)
    branch_bundles: tuple[V1FQRBranchDesignBundle, ...] = field(
        repr=False,
        compare=False,
    )
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
    schema: str = FQR_BRANCH_FAMILY_DESIGN_SCHEMA
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.registration) is not V1FQRJointFamilyRegistration:
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "branch-family design requires the exact joint registration",
            )
        bundles = tuple(self.branch_bundles)
        if any(type(value) is not V1FQRBranchDesignBundle for value in bundles):
            _refuse(
                V1FQRBranchDesignRefusalCode.INVALID_TYPE,
                "branch-family design requires exact branch bundles",
            )
        for value in bundles:
            value.assert_integrity()
        retained = _retained_branch_order(self.registration.prefrozen_family)
        if tuple(value.branch for value in bundles) != retained:
            _refuse(
                V1FQRBranchDesignRefusalCode.MISSING_BRANCH_INPUT,
                "branch bundles do not exactly cover retained family branches",
            )
        child_hash_axes = (
            tuple(value.secant_design.design_hash for value in bundles),
            tuple(value.fqr_design.design_hash for value in bundles),
            tuple(value.sequential_design.design_hash for value in bundles),
            tuple(value.resource_registration.registration_hash for value in bundles),
            tuple(value.branch_binding.frozen_ladder.ladder_hash for value in bundles),
        )
        if any(len(set(axis)) != len(axis) for axis in child_hash_axes):
            _refuse(
                V1FQRBranchDesignRefusalCode.DESIGN_COLLISION,
                "retained branches collide on a child design or ladder identity",
            )
        if any(
            value.registration_hash != self.registration.registration_hash
            for value in bundles
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "one branch bundle names another joint family",
            )
        for branch_input, bundle in zip(
            self.registration.branch_inputs,
            bundles,
            strict=True,
        ):
            registered_binding = self.registration.prefrozen_family.binding_for_branch(
                bundle.branch
            )
            secant = bundle.secant_design
            fqr = bundle.fqr_design
            sequential = bundle.sequential_design
            resources = bundle.resource_registration
            if (
                branch_input.branch is not bundle.branch
                or bundle.branch_binding.binding_hash
                != registered_binding.binding_hash
                or secant.certification_context_ids_by_stratum
                != self.registration.certification_context_ids_by_stratum
                or secant.stratum_weights != self.registration.stratum_weights
                or sequential.sealed_draw_ids_by_stratum
                != self.registration.sealed_draw_ids_by_stratum
                or secant.certification_context_population_hash
                != self.registration.certification_context_population_hash
                or secant.certification_stream_commitment_hash
                != self.registration.certification_stream_commitment_hash
                or secant.packed_direction_provider_source_hash
                != self.registration.packed_direction_provider_source_hash
                or secant.relaxed_gate_executor_source_hash
                != self.registration.relaxed_gate_executor_source_hash
                or secant.raw_logit_interval_verifier_source_hash
                != self.registration.raw_logit_interval_verifier_source_hash
                or fqr.bce_damage_budget != self.registration.bce_damage_budget
                or fqr.positive_affine_calibration_slope
                != self.registration.positive_affine_calibration_slope
                or fqr.corrected_work_statistical_design_hash
                != branch_input.corrected_work_statistical_design_hash
                or fqr.literal_jacobian_statistical_design_hash
                != branch_input.literal_jacobian_statistical_design_hash
                or not np.array_equal(
                    sequential.analytic_f_upper_by_stratum_endpoint,
                    branch_input.analytic_f_upper_by_stratum_endpoint,
                )
                or not np.array_equal(
                    sequential.analytic_q_upper_by_stratum_endpoint,
                    branch_input.analytic_q_upper_by_stratum_endpoint,
                )
                or not np.array_equal(
                    sequential.analytic_r_upper_by_stratum_endpoint_and_dose,
                    branch_input.analytic_r_upper_by_stratum_endpoint_and_dose,
                )
                or sequential.corrected_work_bound_source_hash
                != branch_input.corrected_work_bound_source_hash
                or sequential.literal_jacobian_bound_source_hash
                != branch_input.literal_jacobian_bound_source_hash
                or sequential.directional_remainder_bound_source_hash
                != branch_input.directional_remainder_bound_source_hash
                or sequential.fd_rank_uncertainty_set_hash
                != branch_input.fd_rank_uncertainty_set_hash
                or sequential.kkt_numerical_uncertainty_set_hash
                != branch_input.kkt_numerical_uncertainty_set_hash
                or sequential.numerical_enclosure_policy_hash
                != branch_input.numerical_enclosure_policy_hash
                or sequential.trajectory_alpha_ledger_hash
                != self.registration.trajectory_alpha_ledger_hash
                or sequential.trajectory_alpha_ledger_generation
                != self.registration.trajectory_alpha_ledger_generation
                or sequential.trajectory_familywise_alpha
                != self.registration.trajectory_familywise_alpha
                or sequential.trajectory_alpha_committed_before_fqr
                != self.registration.trajectory_alpha_committed_before_fqr
                or bundle.branch_alpha_allocation
                != self.registration.alpha_allocation_for_branch(bundle.branch)
                or resources.resource_units != self.registration.resource_units
                or resources.resource_source_hash
                != self.registration.resource_source_hash
            ):
                _refuse(
                    V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                    f"{bundle.branch.value} child designs differ from the joint "
                    "family registration or branch analytic input",
                )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch-family design is not pre-C",
            )
        if any(
            value is not False
            for value in (
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch-family design crossed its design-only authority",
            )
        if (
            self.authority_scope != FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
            or self.schema != FQR_BRANCH_FAMILY_DESIGN_SCHEMA
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
                "branch-family design scope or schema is invalid",
            )
        object.__setattr__(self, "branch_bundles", bundles)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "registration_hash": self.registration.registration_hash,
            "branch_bundle_hashes": tuple(
                value.bundle_hash for value in self.branch_bundles
            ),
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def common_family_hash(self) -> str:
        return self.registration.registration_hash

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def bundle_for_branch(
        self,
        branch: V1RelaxedKKTBranch,
    ) -> V1FQRBranchDesignBundle:
        for value in self.branch_bundles:
            if value.branch is branch:
                return value
        _refuse(
            V1FQRBranchDesignRefusalCode.EXTRA_BRANCH_INPUT,
            f"{branch.value} is not retained by this branch-family design",
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.registration.assert_integrity()
        for value in self.branch_bundles:
            value.assert_integrity()
        if self.issued_before_C_outcomes is not True or self.C_outcomes_seen is not False:
            _refuse(
                V1FQRBranchDesignRefusalCode.POST_OUTCOME,
                "branch-family design is no longer pre-C",
            )
        if any(
            value is not False
            for value in (
                self.C_stream_open_authority,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.irreversible_decision_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_claim_authority,
            )
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.AUTHORITY_VIOLATION,
                "branch-family design gained downstream authority",
            )
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        cached = self._design_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch-family design was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
                "branch-family design hash does not match",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


def build_v1_fqr_branch_family_design(
    *,
    design_id: str,
    prefrozen_family: object,
    branch_inputs: Sequence[V1FQRBranchAnalyticInput],
    certification_context_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    stratum_weights: Sequence[tuple[str, float]],
    sealed_draw_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    discovery_context_population_hash: str,
    certification_context_population_hash: str,
    certification_stream_commitment_hash: str,
    packed_direction_provider_source_hash: str,
    relaxed_gate_executor_source_hash: str,
    raw_logit_interval_verifier_source_hash: str,
    bce_damage_budget: float,
    positive_affine_calibration_slope: float,
    trajectory_alpha_ledger_hash: str,
    trajectory_familywise_alpha: float,
    trajectory_alpha_committed_before_fqr: float,
    fqr_family_alpha_allocation: float,
    resource_units: str,
    resource_source_hash: str,
) -> V1FQRBranchFamilyDesign:
    """Build every retained branch design under one pre-outcome registration."""

    if type(prefrozen_family) not in {
        V1FQRKKTPreFrozenBranchFamily,
        V1FQRDeviceShellPackedPreFrozenBranchFamily,
    }:
        _refuse(
            V1FQRBranchDesignRefusalCode.INVALID_TYPE,
            "branch-family factory requires an exact admitted pre-frozen family",
        )
    try:
        prefrozen_family.assert_integrity()
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1FQRBranchDesignRefusalCode.HASH_MISMATCH,
            f"pre-frozen family integrity failed: {error}",
        )
    inputs = tuple(branch_inputs)
    if any(type(value) is not V1FQRBranchAnalyticInput for value in inputs):
        _refuse(
            V1FQRBranchDesignRefusalCode.INVALID_TYPE,
            "factory branch inputs must use the exact analytic-input type",
        )
    retained = _retained_branch_order(prefrozen_family)
    observed = tuple(value.branch for value in inputs)
    if len(set(observed)) != len(observed):
        _refuse(
            V1FQRBranchDesignRefusalCode.DESIGN_COLLISION,
            "factory received duplicate branch inputs",
        )
    missing = tuple(branch for branch in retained if branch not in observed)
    extra = tuple(branch for branch in observed if branch not in retained)
    if missing:
        _refuse(
            V1FQRBranchDesignRefusalCode.MISSING_BRANCH_INPUT,
            f"factory is missing retained branches: {missing}",
        )
    if extra:
        _refuse(
            V1FQRBranchDesignRefusalCode.EXTRA_BRANCH_INPUT,
            f"factory received inputs for omitted branches: {extra}",
        )
    if observed != retained:
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            "factory branch inputs are not in retained-family order",
        )
    contexts = _axis_pairs(
        certification_context_ids_by_stratum,
        name="shared C population",
    )
    draws = _axis_pairs(sealed_draw_ids_by_stratum, name="shared sealed draws")
    weights = _weight_axis(
        stratum_weights,
        names=tuple(name for name, _axis in contexts),
    )
    if (
        tuple(name for name, _axis in contexts)
        != tuple(name for name, _axis in draws)
        or tuple(len(axis) for _name, axis in contexts)
        != tuple(len(axis) for _name, axis in draws)
    ):
        _refuse(
            V1FQRBranchDesignRefusalCode.COORDINATE_MISMATCH,
            "shared C context and draw populations differ",
        )
    allocation = _probability(
        fqr_family_alpha_allocation,
        name="F/Q/R family alpha allocation",
    )
    branch_cell_counts = tuple(
        (
            branch,
            len(contexts)
            * len(prefrozen_family.binding_for_branch(branch).frozen_ladder.actions)
            * len(FQR_SEQUENTIAL_ROLES),
        )
        for branch in retained
    )
    planned_cell_alpha, branch_allocations = _joint_alpha_plan(
        allocation=allocation,
        branch_cell_counts=branch_cell_counts,
    )
    cells: list[V1FQRJointAlphaCell] = []
    for branch in retained:
        binding = prefrozen_family.binding_for_branch(branch)
        for stratum, _axis in contexts:
            for endpoint, action_hash in enumerate(
                binding.frozen_ladder.action_hashes,
                start=1,
            ):
                for role in FQR_SEQUENTIAL_ROLES:
                    cells.append(
                        V1FQRJointAlphaCell(
                            branch=branch,
                            stratum_id=stratum,
                            endpoint_index=endpoint,
                            action_hash=action_hash,
                            role=role,
                            cell_family_alpha=planned_cell_alpha,
                        )
                    )
    registration = V1FQRJointFamilyRegistration(
        design_id=design_id,
        prefrozen_family=prefrozen_family,
        branch_inputs=inputs,
        certification_context_ids_by_stratum=contexts,
        stratum_weights=weights,
        sealed_draw_ids_by_stratum=draws,
        discovery_context_population_hash=discovery_context_population_hash,
        certification_context_population_hash=certification_context_population_hash,
        certification_stream_commitment_hash=certification_stream_commitment_hash,
        packed_direction_provider_source_hash=packed_direction_provider_source_hash,
        relaxed_gate_executor_source_hash=relaxed_gate_executor_source_hash,
        raw_logit_interval_verifier_source_hash=(
            raw_logit_interval_verifier_source_hash
        ),
        bce_damage_budget=bce_damage_budget,
        positive_affine_calibration_slope=positive_affine_calibration_slope,
        trajectory_alpha_ledger_hash=trajectory_alpha_ledger_hash,
        trajectory_alpha_ledger_generation=prefrozen_family.proposal_generation,
        trajectory_familywise_alpha=trajectory_familywise_alpha,
        trajectory_alpha_committed_before_fqr=(
            trajectory_alpha_committed_before_fqr
        ),
        fqr_family_alpha_allocation=allocation,
        planned_cell_family_alpha=planned_cell_alpha,
        branch_alpha_allocations=branch_allocations,
        alpha_cells=tuple(cells),
        total_cell_count=len(cells),
        registered_cell_alpha_total=math.fsum(
            planned_cell_alpha for _cell in cells
        ),
        resource_units=resource_units,
        resource_source_hash=resource_source_hash,
    )
    common_hash = registration.registration_hash
    bundles: list[V1FQRBranchDesignBundle] = []
    for branch_input in inputs:
        branch = branch_input.branch
        binding = prefrozen_family.binding_for_branch(branch)
        ladder = binding.frozen_ladder
        secant = V1DirectionalSecantDesign(
            design_id=f"{design_id}:secant:{branch.value}",
            ladder=ladder,
            chart_identity_hash=prefrozen_family.chart_identity_hash,
            certification_context_ids_by_stratum=contexts,
            stratum_weights=weights,
            certification_context_population_hash=(
                registration.certification_context_population_hash
            ),
            certification_stream_commitment_hash=(
                registration.certification_stream_commitment_hash
            ),
            packed_direction_provider_source_hash=(
                registration.packed_direction_provider_source_hash
            ),
            relaxed_gate_executor_source_hash=(
                registration.relaxed_gate_executor_source_hash
            ),
            raw_logit_interval_verifier_source_hash=(
                registration.raw_logit_interval_verifier_source_hash
            ),
            joint_fqr_family_registration_hash=common_hash,
        )
        fqr = V1FQRCertificationDesign(
            design_id=f"{design_id}:FQR:{branch.value}",
            secant_design=secant,
            bce_damage_budget=registration.bce_damage_budget,
            positive_affine_calibration_slope=(
                registration.positive_affine_calibration_slope
            ),
            corrected_work_statistical_design_hash=(
                branch_input.corrected_work_statistical_design_hash
            ),
            literal_jacobian_statistical_design_hash=(
                branch_input.literal_jacobian_statistical_design_hash
            ),
            joint_fqr_family_registration_hash=common_hash,
        )
        branch_allocation = registration.alpha_allocation_for_branch(branch)
        sequential = V1FQRSequentialAcquisitionDesign(
            design_id=f"{design_id}:sequential-C:{branch.value}",
            fqr_design=fqr,
            sealed_draw_ids_by_stratum=draws,
            analytic_f_upper_by_stratum_endpoint=(
                branch_input.analytic_f_upper_by_stratum_endpoint
            ),
            analytic_q_upper_by_stratum_endpoint=(
                branch_input.analytic_q_upper_by_stratum_endpoint
            ),
            analytic_r_upper_by_stratum_endpoint_and_dose=(
                branch_input.analytic_r_upper_by_stratum_endpoint_and_dose
            ),
            corrected_work_bound_source_hash=(
                branch_input.corrected_work_bound_source_hash
            ),
            literal_jacobian_bound_source_hash=(
                branch_input.literal_jacobian_bound_source_hash
            ),
            directional_remainder_bound_source_hash=(
                branch_input.directional_remainder_bound_source_hash
            ),
            trajectory_alpha_ledger_hash=registration.trajectory_alpha_ledger_hash,
            trajectory_alpha_ledger_generation=(
                registration.trajectory_alpha_ledger_generation
            ),
            trajectory_familywise_alpha=registration.trajectory_familywise_alpha,
            trajectory_alpha_committed_before_fqr=(
                registration.trajectory_alpha_committed_before_fqr
            ),
            fqr_family_alpha_allocation=branch_allocation,
            fd_rank_uncertainty_set_hash=(
                branch_input.fd_rank_uncertainty_set_hash
            ),
            kkt_numerical_uncertainty_set_hash=(
                branch_input.kkt_numerical_uncertainty_set_hash
            ),
            numerical_enclosure_policy_hash=(
                branch_input.numerical_enclosure_policy_hash
            ),
            joint_fqr_family_registration_hash=common_hash,
        )
        resources = binding.endpoint_registered_resources
        if (
            any(value <= 0.0 or not math.isfinite(value) for value in resources)
            or any(right <= left for left, right in zip(resources, resources[1:]))
        ):
            _refuse(
                V1FQRBranchDesignRefusalCode.RESOURCE_DRIFT,
                f"{branch.value} registered resources are not strictly increasing",
            )
        resource_registration = V1CompactFQRPreCResourceRegistration(
            acquisition_design_hash=sequential.design_hash,
            ladder_hash=ladder.ladder_hash,
            chart_identity_hash=prefrozen_family.chart_identity_hash,
            current_mask_hash=prefrozen_family.current_mask_hash,
            proposal_generation=prefrozen_family.proposal_generation,
            route_count=prefrozen_family.route_count,
            route_axis_hash=prefrozen_family.route_axis_hash,
            endpoint_action_hashes=ladder.action_hashes,
            resource_by_endpoint=tuple(
                (endpoint, action_hash, resources[endpoint - 1])
                for endpoint, action_hash in enumerate(
                    ladder.action_hashes,
                    start=1,
                )
            ),
            resource_units=registration.resource_units,
            resource_source_hash=registration.resource_source_hash,
        )
        bundles.append(
            V1FQRBranchDesignBundle(
                registration_hash=common_hash,
                branch=branch,
                branch_binding=binding,
                secant_design=secant,
                fqr_design=fqr,
                sequential_design=sequential,
                resource_registration=resource_registration,
                branch_alpha_allocation=branch_allocation,
                branch_cell_count=sequential.cell_count,
                cell_family_alpha=sequential.cell_family_alpha,
            )
        )
    return V1FQRBranchFamilyDesign(
        registration=registration,
        branch_bundles=tuple(bundles),
    )


__all__ = [
    "FQR_BRANCH_FAMILY_ALPHA_POLICY",
    "FQR_BRANCH_FAMILY_AUTHORITY_SCOPE",
    "V1FQRBranchAnalyticInput",
    "V1FQRBranchDesignBundle",
    "V1FQRBranchDesignRefusal",
    "V1FQRBranchDesignRefusalCode",
    "V1FQRBranchFamilyDesign",
    "V1FQRJointAlphaCell",
    "V1FQRJointFamilyRegistration",
    "build_v1_fqr_branch_family_design",
]
