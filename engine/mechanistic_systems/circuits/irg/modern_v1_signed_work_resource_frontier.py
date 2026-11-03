"""Endogenous resource-frontier reference joint for canonical modern-v1.

The signed-work least-work problem needs a resource coordinate, but that
coordinate must not become an operator-chosen pruning step.  This module
closes that seam for small domains in two causally ordered phases:

Phase A solves, independently for each canonical Q branch,

    maximize    resource.T @ z
    subject to  Q_branch(z) <= tau_Q
                required_lower_bound <= z <= eligible
                compact live-block removal capacities.

Phase B does not choose a new scalar.  It emits the distinct, numerically
certified exposed resource endpoints from Phase A as registered coordinates
for subsequent least-signed-work solves.  Overlapping certified endpoint
intervals are collapsed to their least witnessed endpoint; separated
intervals remain separate.  Hence every emitted coordinate is witnessed
feasible, branch membership is closed, and no cardinality schedule, geometric
grid, or hand-selected step size enters the construction.

This is deliberately a deterministic FP64 CPU/small-domain theorem reference.
It uses convex SLSQP candidates followed by independent first-order dual
enclosures.  It does *not* claim route-scale CUDA execution, hard-mask global
optimality, F/Q/R certification, C/K/O/oracle access, mask-commit authority,
terminality, or Pareto authority.  Production callers fail closed at the
registered small-domain route limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import math
from numbers import Integral, Real
from typing import Final, Literal, Sequence

import numpy as np

from .contracts import content_hash
from .modern_v1_signed_work_least_work_reference import (
    V1SignedWorkKKTReferenceProblem,
    V1SignedWorkLeastWorkBranch,
)


SIGNED_WORK_RESOURCE_FRONTIER_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_reference_policy_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_PROBLEM_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_reference_problem_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_BRANCH_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_branch_receipt_v1"
)
SIGNED_WORK_RESOURCE_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_endogenous_resource_coordinate_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_work_resource_frontier_result_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE: Final = (
    "cpu_small_domain_relaxed_Q_resource_frontier_proposal_only_no_route_scale_"
    "cuda_F_Q_R_C_K_O_oracle_mask_commit_terminal_or_pareto_authority_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_OBJECTIVE_ID: Final = (
    "phase_A_maximize_registered_resource_per_independent_convex_Q_branch_then_"
    "phase_B_emit_only_certified_exposed_endpoint_coordinates_v1"
)
SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID: Final = (
    "sorted_distinct_phase_A_endpoint_intervals_collapse_overlap_to_least_"
    "witnessed_feasible_resource_no_external_step_schedule_v1"
)

_POLICY_PREFIX: Final = "partition-free-v1-signed-work-resource-frontier-policy-"
_PROBLEM_PREFIX: Final = "partition-free-v1-signed-work-resource-frontier-problem-"
_BRANCH_PREFIX: Final = "partition-free-v1-signed-work-resource-frontier-branch-"
_COORDINATE_PREFIX: Final = "partition-free-v1-signed-work-resource-coordinate-"
_RESULT_PREFIX: Final = "partition-free-v1-signed-work-resource-frontier-result-"


class V1SignedWorkResourceBranchStatus(str, Enum):
    """Closed and unresolved outcomes for one independent Q branch."""

    OPTIMAL = "optimal"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNRESOLVED = "unresolved"


class V1SignedWorkResourceUnionStatus(str, Enum):
    """Status of the nonconvex union of the two convex Q branches."""

    CERTIFIED_COORDINATES = "certified_coordinates"
    CERTIFIED_NO_NONEMPTY_ACTION = "certified_no_nonempty_action"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNRESOLVED = "unresolved"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be one nonempty string")
    return value


def _exact_bool(value: object, *, name: str, expected: bool | None = None) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    if expected is not None and value is not expected:
        raise PermissionError(f"{name} crossed registered authority")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return content_hash(
        {
            "dtype": str(array.dtype),
            "shape": tuple(int(item) for item in array.shape),
            "bytes": array.tobytes(order="C").hex(),
        },
        prefix="partition-free-v1-signed-work-resource-frontier-array-",
    )


def _required_axis(value: object, *, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_) or raw.ndim != 1 or raw.shape != (size,):
        raise ValueError("required lower bound must be an exact bool route-axis vector")
    result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierPolicy:
    """Numerical and authority policy for the CPU convex reference."""

    policy_id: str = "modern-v1-signed-work-resource-frontier-reference-default"
    solver: Literal["scipy_slsqp_plus_dual_enclosure"] = (
        "scipy_slsqp_plus_dual_enclosure"
    )
    max_iterations: int = 4_000
    ftol: float = 1e-13
    primal_feasibility_tolerance: float = 2e-9
    optimality_absolute_tolerance: float = 5e-6
    optimality_relative_tolerance: float = 2e-8
    small_domain_route_limit: int = 32
    deterministic_start_count: int = 5
    registered_numerical_tolerances_only: bool = True
    external_resource_coordinate_allowed: bool = False
    route_scale_cuda_authority: bool = False
    fqr_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        if self.solver != "scipy_slsqp_plus_dual_enclosure":
            raise ValueError("unsupported resource-frontier reference solver")
        for name in ("max_iterations", "small_domain_route_limit", "deterministic_start_count"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in (
            "ftol",
            "primal_feasibility_tolerance",
            "optimality_absolute_tolerance",
            "optimality_relative_tolerance",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        for name, expected in (
            ("registered_numerical_tolerances_only", True),
            ("external_resource_coordinate_allowed", False),
            ("route_scale_cuda_authority", False),
            ("fqr_authority", False),
            ("mask_commit_authority", False),
        ):
            _exact_bool(getattr(self, name), name=name, expected=expected)
        if self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE:
            raise ValueError("resource-frontier policy authority scope is invalid")
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_POLICY_SCHEMA:
            raise ValueError("resource-frontier policy schema is invalid")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierProblem:
    """Geometry-only Phase-A problem with a mandatory nested hard lower rung."""

    problem_id: str
    geometry: V1SignedWorkKKTReferenceProblem = field(repr=False, compare=False)
    required_lower_bound: np.ndarray = field(repr=False, compare=False)
    objective_id: str = SIGNED_WORK_RESOURCE_FRONTIER_OBJECTIVE_ID
    external_resource_coordinate_supplied: bool = False
    route_scale_cuda_authority: bool = False
    fqr_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_PROBLEM_SCHEMA
    required_lower_bound_hash: str = field(init=False)
    baseline_resource: float = field(init=False)
    _problem_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _identity(self.problem_id, name="problem id"))
        if type(self.geometry) is not V1SignedWorkKKTReferenceProblem:
            raise TypeError("resource frontier requires exact signed-work reference geometry")
        if self.geometry.resource_floor != 0.0:
            raise ValueError(
                "resource-frontier geometry must have zero resource floor; Phase A owns "
                "all endogenous coordinates"
            )
        lower = _required_axis(self.required_lower_bound, size=self.geometry.route_count)
        eligible = self.geometry.eligible_mask
        if bool(np.any(np.logical_and(lower, np.logical_not(eligible)))):
            raise ValueError("required lower rung must be a subset of eligible live routes")
        lower_f64 = lower.astype(np.float64)
        for capacity in self.geometry.block_capacities:
            used = float(np.sum(lower_f64[capacity.block.start : capacity.block.stop]))
            if used > float(capacity.removal_capacity):
                raise ValueError("required lower rung exceeds a live-block removal capacity")
        if self.objective_id != SIGNED_WORK_RESOURCE_FRONTIER_OBJECTIVE_ID:
            raise ValueError("resource-frontier objective id is invalid")
        for name, expected in (
            ("external_resource_coordinate_supplied", False),
            ("route_scale_cuda_authority", False),
            ("fqr_authority", False),
            ("mask_commit_authority", False),
        ):
            _exact_bool(getattr(self, name), name=name, expected=expected)
        if self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE:
            raise ValueError("resource-frontier problem authority scope is invalid")
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_PROBLEM_SCHEMA:
            raise ValueError("resource-frontier problem schema is invalid")
        lower_hash = _digest_array(lower.astype(np.uint8))
        baseline = float(np.dot(self.geometry.resource, lower_f64))
        object.__setattr__(self, "required_lower_bound", lower)
        object.__setattr__(self, "required_lower_bound_hash", lower_hash)
        object.__setattr__(self, "baseline_resource", baseline)
        object.__setattr__(
            self,
            "_problem_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "problem_id": self.problem_id,
                    "geometry_problem_hash": self.geometry.problem_hash,
                    "required_lower_bound_hash": lower_hash,
                    "baseline_resource": baseline,
                    "objective_id": self.objective_id,
                    "external_resource_coordinate_supplied": False,
                    "authority_scope": self.authority_scope,
                },
                prefix=_PROBLEM_PREFIX,
            ),
        )

    @property
    def problem_hash(self) -> str:
        return self._problem_hash


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceBranchReceipt:
    """Primal/dual-enclosed Phase-A outcome for one Q branch."""

    problem_hash: str
    policy_hash: str
    branch: V1SignedWorkLeastWorkBranch
    status: V1SignedWorkResourceBranchStatus
    required_lower_bound_hash: str
    baseline_resource: float
    quadratic_budget: float
    q_min_candidate: float
    q_min_lower_bound: float
    achieved_resource: float | None
    resource_upper_bound: float | None
    resource_optimality_gap: float | None
    achieved_q: float | None
    z: tuple[float, ...] | None
    primal_feasible: bool
    dual_enclosure_closed: bool
    status_reason: str
    primal_witness_summary_hash: str | None = None
    primal_witness_route_count: int | None = None
    primal_witness_scalar_d2h_bytes: int = 0
    route_vector_d2h_bytes: int = 0
    route_vector_retained: bool = True
    ephemeral_cuda_witness_certified: bool = False
    relaxed_branch_only: bool = True
    hard_global_optimality_claim: bool = False
    route_scale_cuda_authority: bool = False
    fqr_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_BRANCH_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "required_lower_bound_hash", "status_reason"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("resource branch receipt requires exact Q branch")
        if type(self.status) is not V1SignedWorkResourceBranchStatus:
            raise TypeError("resource branch receipt requires exact branch status")
        for name in (
            "baseline_resource",
            "quadratic_budget",
            "q_min_candidate",
            "q_min_lower_bound",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        for name in (
            "achieved_resource",
            "resource_upper_bound",
            "resource_optimality_gap",
            "achieved_q",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name=name))
        if self.z is not None:
            z = tuple(_finite(item, name="relaxed route gate") for item in self.z)
            object.__setattr__(self, "z", z)
        summary_hash = self.primal_witness_summary_hash
        if summary_hash is not None:
            summary_hash = _identity(
                summary_hash,
                name="primal witness summary hash",
            )
            object.__setattr__(self, "primal_witness_summary_hash", summary_hash)
        route_count = self.primal_witness_route_count
        if route_count is not None:
            route_count = _positive_int(
                route_count,
                name="primal witness route count",
            )
            object.__setattr__(self, "primal_witness_route_count", route_count)
        for name in ("primal_witness_scalar_d2h_bytes", "route_vector_d2h_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be one exact nonnegative integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in ("route_vector_retained", "ephemeral_cuda_witness_certified"):
            _exact_bool(getattr(self, name), name=name)
        for name in ("primal_feasible", "dual_enclosure_closed"):
            _exact_bool(getattr(self, name), name=name)
        for name, expected in (
            ("relaxed_branch_only", True),
            ("hard_global_optimality_claim", False),
            ("route_scale_cuda_authority", False),
            ("fqr_authority", False),
            ("mask_commit_authority", False),
        ):
            _exact_bool(getattr(self, name), name=name, expected=expected)
        if self.status is V1SignedWorkResourceBranchStatus.OPTIMAL:
            if not self.primal_feasible or not self.dual_enclosure_closed:
                raise ValueError("optimal branch requires closed primal and dual receipts")
            if any(
                item is None
                for item in (
                    self.achieved_resource,
                    self.resource_upper_bound,
                    self.resource_optimality_gap,
                    self.achieved_q,
                )
            ):
                raise ValueError("optimal branch is missing its relaxed optimum")
            if self.resource_upper_bound < self.achieved_resource:
                raise ValueError("resource upper bound cannot be below its witness")
            if self.z is not None:
                if (
                    self.ephemeral_cuda_witness_certified
                    or not self.route_vector_retained
                    or self.route_vector_d2h_bytes != 0
                    or self.primal_witness_scalar_d2h_bytes != 0
                    or self.primal_witness_summary_hash is not None
                    or self.primal_witness_route_count not in {None, len(self.z)}
                ):
                    raise ValueError(
                        "retained small-domain witness has CUDA-summary fields"
                    )
                object.__setattr__(
                    self,
                    "primal_witness_route_count",
                    len(self.z),
                )
            elif (
                not self.ephemeral_cuda_witness_certified
                or self.route_vector_retained
                or self.primal_witness_summary_hash is None
                or self.primal_witness_route_count is None
                or self.primal_witness_scalar_d2h_bytes <= 0
                or self.route_vector_d2h_bytes != 0
            ):
                raise ValueError(
                    "vector-free optimal branch requires a bounded CUDA witness summary"
                )
        elif self.status is V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE:
            if not self.dual_enclosure_closed or self.q_min_lower_bound <= self.quadratic_budget:
                raise ValueError("infeasible branch requires a strict Q lower-bound certificate")
            if self.primal_feasible or self.z is not None:
                raise ValueError("infeasible branch cannot carry a feasible resource witness")
        if self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE:
            raise ValueError("resource branch authority scope is invalid")
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_BRANCH_SCHEMA:
            raise ValueError("resource branch schema is invalid")

    @property
    def branch_receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkEndogenousResourceCoordinate:
    """One Phase-A-derived resource floor for later least-work Phase B."""

    problem_hash: str
    policy_hash: str
    ordinal: int
    total_resource_floor: float
    incremental_resource: float
    baseline_resource: float
    required_lower_bound_hash: str
    witnessed_feasible_branches: tuple[V1SignedWorkLeastWorkBranch, ...]
    certified_excluded_branches: tuple[V1SignedWorkLeastWorkBranch, ...]
    source_branch_receipt_hashes: tuple[str, ...]
    branch_membership_closed: bool = True
    externally_supplied_step_size: bool = False
    phase_b_least_work_proposal_only: bool = True
    fqr_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE
    coordinate_rule_id: str = SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID
    schema: str = SIGNED_WORK_RESOURCE_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "required_lower_bound_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        ordinal = _positive_int(self.ordinal, name="coordinate ordinal")
        object.__setattr__(self, "ordinal", ordinal)
        for name in ("total_resource_floor", "incremental_resource", "baseline_resource"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if self.incremental_resource <= 0.0:
            raise ValueError("an endogenous Phase-B coordinate must be a nonempty increment")
        reconstruction = self.baseline_resource + self.incremental_resource
        equality_tolerance = 16.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(self.total_resource_floor),
            abs(reconstruction),
        )
        if abs(self.total_resource_floor - reconstruction) > equality_tolerance:
            raise ValueError("resource coordinate total and increment are inconsistent")
        witnessed = tuple(self.witnessed_feasible_branches)
        excluded = tuple(self.certified_excluded_branches)
        canonical = (
            V1SignedWorkLeastWorkBranch.SPECTRAL,
            V1SignedWorkLeastWorkBranch.DIAGONAL_L1,
        )
        if not witnessed or any(
            type(item) is not V1SignedWorkLeastWorkBranch for item in witnessed
        ):
            raise TypeError("coordinate requires witnessed feasible exact branches")
        if any(type(item) is not V1SignedWorkLeastWorkBranch for item in excluded):
            raise TypeError("coordinate exclusions require exact branches")
        if set(witnessed).intersection(excluded) or set(witnessed).union(excluded) != set(
            canonical
        ):
            raise ValueError("coordinate branch membership must close the canonical union")
        object.__setattr__(self, "witnessed_feasible_branches", witnessed)
        object.__setattr__(self, "certified_excluded_branches", excluded)
        hashes = tuple(
            _identity(item, name="source branch receipt hash")
            for item in self.source_branch_receipt_hashes
        )
        if not hashes:
            raise ValueError("coordinate requires source branch receipts")
        object.__setattr__(self, "source_branch_receipt_hashes", hashes)
        for name, expected in (
            ("branch_membership_closed", True),
            ("externally_supplied_step_size", False),
            ("phase_b_least_work_proposal_only", True),
            ("fqr_authority", False),
            ("mask_commit_authority", False),
        ):
            _exact_bool(getattr(self, name), name=name, expected=expected)
        if self.coordinate_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID:
            raise ValueError("resource-coordinate rule id is invalid")
        if self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE:
            raise ValueError("resource-coordinate authority scope is invalid")
        if self.schema != SIGNED_WORK_RESOURCE_COORDINATE_SCHEMA:
            raise ValueError("resource-coordinate schema is invalid")

    @property
    def coordinate_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkResourceFrontierResult:
    """Closed Phase-A branch union and its endogenous Phase-B coordinates."""

    problem_hash: str
    policy_hash: str
    spectral: V1SignedWorkResourceBranchReceipt
    diagonal_l1: V1SignedWorkResourceBranchReceipt
    union_status: V1SignedWorkResourceUnionStatus
    coordinates: tuple[V1SignedWorkEndogenousResourceCoordinate, ...]
    all_branch_alternatives_closed: bool
    nonconvex_branch_union_not_one_convex_problem: bool = True
    coordinate_rule_id: str = SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID
    route_scale_cuda_authority: bool = False
    fqr_authority: bool = False
    cko_authority: bool = False
    mask_commit_authority: bool = False
    terminality_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_RESOURCE_FRONTIER_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_hash", _identity(self.problem_hash, name="problem hash"))
        object.__setattr__(self, "policy_hash", _identity(self.policy_hash, name="policy hash"))
        if type(self.spectral) is not V1SignedWorkResourceBranchReceipt or (
            self.spectral.branch is not V1SignedWorkLeastWorkBranch.SPECTRAL
        ):
            raise TypeError("resource result spectral branch is invalid")
        if type(self.diagonal_l1) is not V1SignedWorkResourceBranchReceipt or (
            self.diagonal_l1.branch is not V1SignedWorkLeastWorkBranch.DIAGONAL_L1
        ):
            raise TypeError("resource result diagonal branch is invalid")
        if type(self.union_status) is not V1SignedWorkResourceUnionStatus:
            raise TypeError("resource result union status is invalid")
        coordinates = tuple(self.coordinates)
        if any(type(item) is not V1SignedWorkEndogenousResourceCoordinate for item in coordinates):
            raise TypeError("resource result coordinates must be exact")
        if any(item.ordinal != index for index, item in enumerate(coordinates, start=1)):
            raise ValueError("resource result coordinates must have canonical ordinals")
        if any(
            a.total_resource_floor >= b.total_resource_floor
            for a, b in zip(coordinates, coordinates[1:])
        ):
            raise ValueError("resource result coordinates must be strictly increasing")
        object.__setattr__(self, "coordinates", coordinates)
        _exact_bool(self.all_branch_alternatives_closed, name="all branch alternatives closed")
        for name, expected in (
            ("nonconvex_branch_union_not_one_convex_problem", True),
            ("route_scale_cuda_authority", False),
            ("fqr_authority", False),
            ("cko_authority", False),
            ("mask_commit_authority", False),
            ("terminality_authority", False),
            ("pareto_authority", False),
        ):
            _exact_bool(getattr(self, name), name=name, expected=expected)
        if self.union_status is V1SignedWorkResourceUnionStatus.UNRESOLVED:
            if self.all_branch_alternatives_closed or coordinates:
                raise ValueError("unresolved union cannot emit Phase-B coordinates")
        else:
            if not self.all_branch_alternatives_closed:
                raise ValueError("certified union requires all branch alternatives closed")
        if (
            self.union_status is V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
            and not coordinates
        ):
            raise ValueError("certified-coordinate union requires coordinates")
        if (
            self.union_status is not V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
            and coordinates
        ):
            raise ValueError("only certified-coordinate union may emit coordinates")
        if self.coordinate_rule_id != SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID:
            raise ValueError("resource result coordinate rule is invalid")
        if self.authority_scope != SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE:
            raise ValueError("resource result authority scope is invalid")
        if self.schema != SIGNED_WORK_RESOURCE_FRONTIER_RESULT_SCHEMA:
            raise ValueError("resource result schema is invalid")

    @property
    def result_hash(self) -> str:
        return content_hash(self, prefix=_RESULT_PREFIX)

    def require_phase_b_coordinates(self) -> tuple[V1SignedWorkEndogenousResourceCoordinate, ...]:
        """Return proposal coordinates only when the entire branch union closed."""

        if self.union_status is not V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES:
            raise PermissionError("resource frontier has no certified Phase-B coordinates")
        return self.coordinates


def _q_value_gradient(
    geometry: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
    z: np.ndarray,
) -> tuple[float, np.ndarray]:
    hz = geometry.low_rank_factor @ z
    common_value = float(np.dot(hz, hz))
    common_gradient = 2.0 * (geometry.low_rank_factor.T @ hz)
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        value = common_value + geometry.residual_spectral_bound * float(np.dot(z, z))
        gradient = common_gradient + 2.0 * geometry.residual_spectral_bound * z
        return value, gradient
    if branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        root = np.sqrt(geometry.residual_diagonal)
        projection = float(np.dot(root, z))
        return common_value + projection * projection, common_gradient + 2.0 * projection * root
    raise TypeError("unknown signed-work Q branch")


def _block_matrix_and_capacity(
    geometry: V1SignedWorkKKTReferenceProblem,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize only in the registered CPU/small-domain reference."""

    matrix = np.zeros((len(geometry.block_capacities), geometry.route_count), dtype=np.float64)
    capacity = np.empty(len(geometry.block_capacities), dtype=np.float64)
    for row, item in enumerate(geometry.block_capacities):
        matrix[row, item.block.start : item.block.stop] = 1.0
        capacity[row] = float(item.removal_capacity)
    return matrix, capacity


def _outward_guard(*, scale: float, dimension: int) -> float:
    return 512.0 * np.finfo(np.float64).eps * max(1.0, abs(scale)) * max(1, dimension)


def _linear_feasible(
    z: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    block_matrix: np.ndarray,
    capacity: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    return bool(
        np.all(z >= lower - tolerance)
        and np.all(z <= upper + tolerance)
        and np.all(block_matrix @ z <= capacity + tolerance)
    )


def _repair_linear_inward(
    z: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    geometry: V1SignedWorkKKTReferenceProblem,
) -> np.ndarray:
    """Project tiny numerical violations inward without changing the lower rung.

    This is not an optimizer.  It clips boxes and removes only the floating
    excess above a compact block capacity, in reverse physical order.  Q is
    always recomputed afterwards because decreasing a gate can destroy a
    cancellation and increase response length.
    """

    result = np.clip(np.asarray(z, dtype=np.float64), lower, upper)
    for item in geometry.block_capacities:
        start = item.block.start
        stop = item.block.stop
        capacity = float(item.removal_capacity)
        excess = float(np.sum(result[start:stop])) - capacity
        if excess <= 0.0:
            continue
        for index in range(stop - 1, start - 1, -1):
            available = result[index] - lower[index]
            if available <= 0.0:
                continue
            decrement = min(available, excess)
            result[index] -= decrement
            excess = float(np.sum(result[start:stop])) - capacity
            if excess <= 0.0:
                break
        # A final nextafter closes a possible one-ulp summation overshoot.
        if float(np.sum(result[start:stop])) > capacity:
            for index in range(stop - 1, start - 1, -1):
                if result[index] > lower[index]:
                    result[index] = math.nextafter(result[index], lower[index])
                    if float(np.sum(result[start:stop])) <= capacity:
                        break
    return result


def _resource_cap_point(
    resource: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    geometry: V1SignedWorkKKTReferenceProblem,
) -> np.ndarray:
    """Exact compact-block resource maximizer without the Q constraint."""

    z = np.array(lower, dtype=np.float64, copy=True)
    for item in geometry.block_capacities:
        sl = slice(item.block.start, item.block.stop)
        remaining = float(item.removal_capacity) - float(np.sum(z[sl]))
        if remaining <= 0.0:
            continue
        indices = np.arange(item.block.start, item.block.stop, dtype=np.int64)
        indices = indices[upper[indices] > lower[indices]]
        order = sorted((int(i) for i in indices), key=lambda i: (-resource[i], i))
        for index in order:
            amount = min(upper[index] - z[index], remaining)
            if amount <= 0.0:
                continue
            z[index] += amount
            remaining -= amount
            if remaining <= 0.0:
                break
    return z


def _furthest_q_feasible_on_segment(
    geometry: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
    feasible: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    budget = geometry.quadratic_budget
    target_q, _ = _q_value_gradient(geometry, branch, target)
    if target_q <= budget:
        return np.array(target, copy=True)
    low = 0.0
    high = 1.0
    delta = target - feasible
    for _ in range(96):
        mid = (low + high) * 0.5
        q, _ = _q_value_gradient(geometry, branch, feasible + mid * delta)
        if q <= budget:
            low = mid
        else:
            high = mid
    alpha = math.nextafter(low, 0.0)
    result = feasible + alpha * delta
    for _ in range(16):
        q, _ = _q_value_gradient(geometry, branch, result)
        if q <= budget:
            break
        alpha = math.nextafter(alpha, 0.0)
        result = feasible + alpha * delta
    return result


def _min_q_dual_lower_bound(
    z: np.ndarray,
    q_value: float,
    q_gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    block_matrix: np.ndarray,
    capacity: np.ndarray,
) -> tuple[float, float, bool]:
    """Return an outward lower bound on min Q over the linear feasible set."""

    from scipy.optimize import linprog

    n = int(z.size)
    b = int(capacity.size)
    # Variables are [mu_blocks, alpha_lower, beta_upper, t_abs_residual].
    mu = slice(0, b)
    alpha = slice(b, b + n)
    beta = slice(b + n, b + 2 * n)
    t = slice(b + 2 * n, b + 3 * n)
    width = b + 3 * n
    block_slack = np.maximum(0.0, capacity - block_matrix @ z)
    lower_slack = np.maximum(0.0, z - lower)
    upper_slack = np.maximum(0.0, upper - z)
    radius = np.maximum(lower_slack, upper_slack)
    objective = np.zeros(width, dtype=np.float64)
    objective[mu] = block_slack
    objective[alpha] = lower_slack
    objective[beta] = upper_slack
    objective[t] = radius
    a_ub = np.zeros((2 * n, width), dtype=np.float64)
    a_ub[:n, mu] = block_matrix.T
    a_ub[:n, alpha] = -np.eye(n)
    a_ub[:n, beta] = np.eye(n)
    a_ub[:n, t] = -np.eye(n)
    a_ub[n:, mu] = -block_matrix.T
    a_ub[n:, alpha] = np.eye(n)
    a_ub[n:, beta] = -np.eye(n)
    a_ub[n:, t] = -np.eye(n)
    b_ub = np.concatenate((-q_gradient, q_gradient))
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * width,
        method="highs",
    )
    if solved.success and solved.x is not None and np.isfinite(solved.x).all():
        vector = np.maximum(0.0, np.asarray(solved.x, dtype=np.float64))
        multipliers_available = True
    else:
        vector = np.zeros(width, dtype=np.float64)
        multipliers_available = False
    mu_v = vector[mu]
    alpha_v = vector[alpha]
    beta_v = vector[beta]
    stationarity = q_gradient + block_matrix.T @ mu_v - alpha_v + beta_v
    error = float(
        np.dot(mu_v, block_slack)
        + np.dot(alpha_v, lower_slack)
        + np.dot(beta_v, upper_slack)
        + np.dot(np.abs(stationarity), radius)
    )
    guard = _outward_guard(
        scale=q_value + error + float(np.linalg.norm(q_gradient, ord=1)),
        dimension=width,
    )
    lower_bound = math.nextafter(q_value - error - guard, -math.inf)
    return lower_bound, float(np.max(np.abs(stationarity), initial=0.0)), multipliers_available


def _resource_dual_upper_bound(
    z: np.ndarray,
    resource_value: float,
    resource: np.ndarray,
    q_value: float,
    q_gradient: np.ndarray,
    q_budget: float,
    lower: np.ndarray,
    upper: np.ndarray,
    block_matrix: np.ndarray,
    capacity: np.ndarray,
) -> tuple[float, float, bool]:
    """Return a valid first-order dual enclosure of the branch maximum."""

    from scipy.optimize import linprog

    n = int(z.size)
    b = int(capacity.size)
    # Variables are [lambda_Q, mu_blocks, alpha_lower, beta_upper, t_abs_residual].
    lam = 0
    mu = slice(1, 1 + b)
    alpha = slice(1 + b, 1 + b + n)
    beta = slice(1 + b + n, 1 + b + 2 * n)
    t = slice(1 + b + 2 * n, 1 + b + 3 * n)
    width = 1 + b + 3 * n
    q_slack = max(0.0, q_budget - q_value)
    block_slack = np.maximum(0.0, capacity - block_matrix @ z)
    lower_slack = np.maximum(0.0, z - lower)
    upper_slack = np.maximum(0.0, upper - z)
    radius = np.maximum(lower_slack, upper_slack)
    objective = np.zeros(width, dtype=np.float64)
    objective[lam] = q_slack
    objective[mu] = block_slack
    objective[alpha] = lower_slack
    objective[beta] = upper_slack
    objective[t] = radius
    a_ub = np.zeros((2 * n, width), dtype=np.float64)
    a_ub[:n, lam] = q_gradient
    a_ub[:n, mu] = block_matrix.T
    a_ub[:n, alpha] = -np.eye(n)
    a_ub[:n, beta] = np.eye(n)
    a_ub[:n, t] = -np.eye(n)
    a_ub[n:, lam] = -q_gradient
    a_ub[n:, mu] = -block_matrix.T
    a_ub[n:, alpha] = np.eye(n)
    a_ub[n:, beta] = -np.eye(n)
    a_ub[n:, t] = -np.eye(n)
    b_ub = np.concatenate((resource, -resource))
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * width,
        method="highs",
    )
    if solved.success and solved.x is not None and np.isfinite(solved.x).all():
        vector = np.maximum(0.0, np.asarray(solved.x, dtype=np.float64))
        multipliers_available = True
    else:
        vector = np.zeros(width, dtype=np.float64)
        multipliers_available = False
    lambda_v = float(vector[lam])
    mu_v = vector[mu]
    alpha_v = vector[alpha]
    beta_v = vector[beta]
    stationarity = (
        -resource
        + lambda_v * q_gradient
        + block_matrix.T @ mu_v
        - alpha_v
        + beta_v
    )
    excess = float(
        lambda_v * q_slack
        + np.dot(mu_v, block_slack)
        + np.dot(alpha_v, lower_slack)
        + np.dot(beta_v, upper_slack)
        + np.dot(np.abs(stationarity), radius)
    )
    guard = _outward_guard(
        scale=resource_value
        + excess
        + float(np.linalg.norm(resource, ord=1))
        + lambda_v * (q_value + q_budget),
        dimension=width,
    )
    upper_bound = math.nextafter(resource_value + excess + guard, math.inf)
    return upper_bound, float(np.max(np.abs(stationarity), initial=0.0)), multipliers_available


def _strong_convexity_resource_upper_bound(
    geometry: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
    q_min_z: np.ndarray,
    q_min_lower_bound: float,
) -> float:
    """Bound resource using the certified Q minimum and Q's strong curvature.

    The first-order Q-min dual receipt gives, for every linearly admissible
    ``y``, ``grad Q(z_min).(y-z_min) >= -(Q(z_min)-Q_lower)``.  Because Q is
    exactly quadratic, adding its smallest eigenvalue times
    ``||y-z_min||^2`` yields the stated radius.  A rank-deficient branch has no
    such strengthening and returns infinity.
    """

    factor = geometry.low_rank_factor
    gram = factor.T @ factor
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        gram = gram + geometry.residual_spectral_bound * np.eye(
            geometry.route_count,
            dtype=np.float64,
        )
    elif branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        root = np.sqrt(geometry.residual_diagonal)
        gram = gram + np.outer(root, root)
    else:
        raise TypeError("unknown signed-work Q branch")
    eigenvalue = float(np.linalg.eigvalsh(gram)[0])
    eigen_guard = _outward_guard(
        scale=float(np.linalg.norm(gram, ord=2)),
        dimension=geometry.route_count,
    )
    eigenvalue_lower = eigenvalue - eigen_guard
    if eigenvalue_lower <= 0.0:
        return math.inf
    radius = math.sqrt(
        max(0.0, geometry.quadratic_budget - q_min_lower_bound) / eigenvalue_lower
    )
    center_resource = float(np.dot(geometry.resource, q_min_z))
    bound = center_resource + float(np.linalg.norm(geometry.resource, ord=2)) * radius
    guard = _outward_guard(
        scale=bound + float(np.linalg.norm(geometry.resource, ord=1)),
        dimension=geometry.route_count,
    )
    return math.nextafter(bound + guard, math.inf)


def _minimize_q_candidate(
    problem: V1SignedWorkResourceFrontierProblem,
    policy: V1SignedWorkResourceFrontierPolicy,
    branch: V1SignedWorkLeastWorkBranch,
    lower: np.ndarray,
    upper: np.ndarray,
    block_matrix: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, float, float, bool]:
    from scipy.optimize import Bounds, minimize

    geometry = problem.geometry

    def objective(z: np.ndarray) -> float:
        return _q_value_gradient(geometry, branch, z)[0]

    def jacobian(z: np.ndarray) -> np.ndarray:
        return _q_value_gradient(geometry, branch, z)[1]

    constraints = (
        {
            "type": "ineq",
            "fun": lambda z: capacity - block_matrix @ z,
            "jac": lambda z: -block_matrix,
        },
    )
    starts = [np.array(lower, copy=True)]
    cap_point = _resource_cap_point(geometry.resource, lower, upper, geometry)
    starts.extend(
        lower + fraction * (cap_point - lower)
        for fraction in np.linspace(0.25, 1.0, policy.deterministic_start_count - 1)
    )
    best = np.array(lower, copy=True)
    best_q = objective(best)
    any_solver_success = False
    for start in starts:
        solved = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="SLSQP",
            jac=jacobian,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"maxiter": policy.max_iterations, "ftol": policy.ftol, "disp": False},
        )
        candidate = _repair_linear_inward(
            np.asarray(solved.x, dtype=np.float64),
            lower,
            upper,
            geometry,
        )
        if not np.isfinite(candidate).all() or not _linear_feasible(
            candidate,
            lower,
            upper,
            block_matrix,
            capacity,
            tolerance=0.0,
        ):
            continue
        any_solver_success = any_solver_success or bool(solved.success)
        q = objective(candidate)
        if q < best_q:
            best = np.array(candidate, copy=True)
            best_q = q
    q, gradient = _q_value_gradient(geometry, branch, best)
    lower_bound, _stationarity, dual_available = _min_q_dual_lower_bound(
        best,
        q,
        gradient,
        lower,
        upper,
        block_matrix,
        capacity,
    )
    return best, q, lower_bound, bool(any_solver_success or dual_available)


def _maximize_resource_candidate(
    problem: V1SignedWorkResourceFrontierProblem,
    policy: V1SignedWorkResourceFrontierPolicy,
    branch: V1SignedWorkLeastWorkBranch,
    feasible_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    block_matrix: np.ndarray,
    capacity: np.ndarray,
) -> tuple[np.ndarray, float, float, float, bool]:
    from scipy.optimize import Bounds, minimize

    geometry = problem.geometry
    resource = geometry.resource

    def objective(z: np.ndarray) -> float:
        return -float(np.dot(resource, z))

    def jacobian(_z: np.ndarray) -> np.ndarray:
        return -resource

    def q_constraint(z: np.ndarray) -> float:
        return geometry.quadratic_budget - _q_value_gradient(geometry, branch, z)[0]

    def q_constraint_jacobian(z: np.ndarray) -> np.ndarray:
        return -_q_value_gradient(geometry, branch, z)[1]

    constraints = (
        {
            "type": "ineq",
            "fun": lambda z: capacity - block_matrix @ z,
            "jac": lambda z: -block_matrix,
        },
        {"type": "ineq", "fun": q_constraint, "jac": q_constraint_jacobian},
    )
    cap_point = _resource_cap_point(resource, lower, upper, geometry)
    segment_point = _furthest_q_feasible_on_segment(
        geometry,
        branch,
        feasible_seed,
        cap_point,
    )
    starts = [np.array(feasible_seed, copy=True), np.array(segment_point, copy=True)]
    starts.extend(
        feasible_seed + fraction * (segment_point - feasible_seed)
        for fraction in np.linspace(0.25, 0.75, max(0, policy.deterministic_start_count - 2))
    )
    best = np.array(feasible_seed, copy=True)
    best_resource = float(np.dot(resource, best))
    any_solver_success = False
    for start in starts:
        solved = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="SLSQP",
            jac=jacobian,
            bounds=Bounds(lower, upper),
            constraints=constraints,
            options={"maxiter": policy.max_iterations, "ftol": policy.ftol, "disp": False},
        )
        candidate = _repair_linear_inward(
            np.asarray(solved.x, dtype=np.float64),
            lower,
            upper,
            geometry,
        )
        if not np.isfinite(candidate).all() or not _linear_feasible(
            candidate,
            lower,
            upper,
            block_matrix,
            capacity,
            tolerance=0.0,
        ):
            continue
        q, _ = _q_value_gradient(geometry, branch, candidate)
        if q > geometry.quadratic_budget:
            candidate = _furthest_q_feasible_on_segment(
                geometry,
                branch,
                feasible_seed,
                candidate,
            )
            q, _ = _q_value_gradient(geometry, branch, candidate)
        if q > geometry.quadratic_budget or not _linear_feasible(
            candidate,
            lower,
            upper,
            block_matrix,
            capacity,
            tolerance=0.0,
        ):
            continue
        any_solver_success = any_solver_success or bool(solved.success)
        candidate_resource = float(np.dot(resource, candidate))
        if candidate_resource > best_resource:
            best = np.array(candidate, copy=True)
            best_resource = candidate_resource
    q, gradient = _q_value_gradient(geometry, branch, best)
    upper_bound, _stationarity, dual_available = _resource_dual_upper_bound(
        best,
        best_resource,
        resource,
        q,
        gradient,
        geometry.quadratic_budget,
        lower,
        upper,
        block_matrix,
        capacity,
    )
    return best, best_resource, q, upper_bound, bool(any_solver_success or dual_available)


def _solve_branch(
    problem: V1SignedWorkResourceFrontierProblem,
    policy: V1SignedWorkResourceFrontierPolicy,
    branch: V1SignedWorkLeastWorkBranch,
) -> V1SignedWorkResourceBranchReceipt:
    geometry = problem.geometry
    lower = problem.required_lower_bound.astype(np.float64)
    upper = geometry.eligible_mask.astype(np.float64)
    block_matrix, capacity = _block_matrix_and_capacity(geometry)
    q_min_z, q_min, q_lower, q_min_numerical = _minimize_q_candidate(
        problem,
        policy,
        branch,
        lower,
        upper,
        block_matrix,
        capacity,
    )
    if q_min > geometry.quadratic_budget:
        if q_lower > geometry.quadratic_budget:
            return V1SignedWorkResourceBranchReceipt(
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                branch=branch,
                status=V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE,
                required_lower_bound_hash=problem.required_lower_bound_hash,
                baseline_resource=problem.baseline_resource,
                quadratic_budget=geometry.quadratic_budget,
                q_min_candidate=q_min,
                q_min_lower_bound=q_lower,
                achieved_resource=None,
                resource_upper_bound=None,
                resource_optimality_gap=None,
                achieved_q=None,
                z=None,
                primal_feasible=False,
                dual_enclosure_closed=True,
                status_reason="minimum_Q_dual_lower_bound_exceeds_budget",
            )
        return V1SignedWorkResourceBranchReceipt(
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            branch=branch,
            status=V1SignedWorkResourceBranchStatus.UNRESOLVED,
            required_lower_bound_hash=problem.required_lower_bound_hash,
            baseline_resource=problem.baseline_resource,
            quadratic_budget=geometry.quadratic_budget,
            q_min_candidate=q_min,
            q_min_lower_bound=q_lower,
            achieved_resource=None,
            resource_upper_bound=None,
            resource_optimality_gap=None,
            achieved_q=None,
            z=None,
            primal_feasible=False,
            dual_enclosure_closed=False,
            status_reason=(
                "no_feasible_Q_witness_and_minimum_Q_enclosure_does_not_certify_infeasibility"
            ),
        )
    z, achieved, achieved_q, upper_bound, numerical = _maximize_resource_candidate(
        problem,
        policy,
        branch,
        q_min_z,
        lower,
        upper,
        block_matrix,
        capacity,
    )
    upper_bound = min(
        upper_bound,
        _strong_convexity_resource_upper_bound(
            geometry,
            branch,
            q_min_z,
            q_lower,
        ),
    )
    # Both enclosures are theoretical upper bounds, but outward rounding can
    # place a witness a few ulps above one of them.  The witness itself is a
    # valid lower bound, so close the interval conservatively at its maximum.
    upper_bound = max(upper_bound, achieved)
    gap = max(0.0, upper_bound - achieved)
    tolerance = policy.optimality_absolute_tolerance + policy.optimality_relative_tolerance * max(
        1.0,
        abs(upper_bound),
        abs(achieved),
    )
    primal = bool(
        _linear_feasible(
            z,
            lower,
            upper,
            block_matrix,
            capacity,
            tolerance=0.0,
        )
        and achieved_q <= geometry.quadratic_budget
    )
    closed = bool(primal and numerical and q_min_numerical and gap <= tolerance)
    return V1SignedWorkResourceBranchReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch=branch,
        status=(
            V1SignedWorkResourceBranchStatus.OPTIMAL
            if closed
            else V1SignedWorkResourceBranchStatus.UNRESOLVED
        ),
        required_lower_bound_hash=problem.required_lower_bound_hash,
        baseline_resource=problem.baseline_resource,
        quadratic_budget=geometry.quadratic_budget,
        q_min_candidate=q_min,
        q_min_lower_bound=q_lower,
        achieved_resource=achieved,
        resource_upper_bound=upper_bound,
        resource_optimality_gap=gap,
        achieved_q=achieved_q,
        z=tuple(float(item) for item in z),
        primal_feasible=primal,
        dual_enclosure_closed=closed,
        status_reason=(
            "relaxed_branch_resource_maximum_dual_enclosed"
            if closed
            else "relaxed_branch_resource_maximum_numerically_unresolved"
        ),
    )


def _derive_coordinates(
    problem: V1SignedWorkResourceFrontierProblem,
    policy: V1SignedWorkResourceFrontierPolicy,
    receipts: tuple[V1SignedWorkResourceBranchReceipt, ...],
) -> tuple[V1SignedWorkEndogenousResourceCoordinate, ...]:
    optimal = [item for item in receipts if item.status is V1SignedWorkResourceBranchStatus.OPTIMAL]
    nonempty_resolution = policy.optimality_absolute_tolerance + (
        policy.optimality_relative_tolerance * max(1.0, abs(problem.baseline_resource))
    )
    if not optimal:
        return ()
    optimal.sort(key=lambda item: (float(item.achieved_resource), item.branch.value))
    # Merge endpoint certificates whose [witness, dual upper] intervals overlap.
    groups: list[list[V1SignedWorkResourceBranchReceipt]] = []
    current: list[V1SignedWorkResourceBranchReceipt] = []
    current_upper = -math.inf
    for receipt in optimal:
        lower_endpoint = float(receipt.achieved_resource)
        if not current or lower_endpoint <= current_upper:
            current.append(receipt)
            current_upper = max(current_upper, float(receipt.resource_upper_bound))
        else:
            groups.append(current)
            current = [receipt]
            current_upper = float(receipt.resource_upper_bound)
    groups.append(current)
    coordinates: list[V1SignedWorkEndogenousResourceCoordinate] = []
    all_hashes = tuple(item.branch_receipt_hash for item in receipts)
    canonical = (
        V1SignedWorkLeastWorkBranch.SPECTRAL,
        V1SignedWorkLeastWorkBranch.DIAGONAL_L1,
    )
    for group in groups:
        rho = min(float(item.achieved_resource) for item in group)
        # If an endpoint interval still overlaps the registered numerical
        # neighborhood of the mandatory lower rung, Phase A has not resolved a
        # nonempty irreversible action there.  Withhold that entire face rather
        # than turning numerical noise into a step.
        if rho <= problem.baseline_resource + nonempty_resolution:
            continue
        witnessed = tuple(
            branch
            for branch in canonical
            if next(item for item in receipts if item.branch is branch).status
            is V1SignedWorkResourceBranchStatus.OPTIMAL
            and float(
                next(item for item in receipts if item.branch is branch).achieved_resource
            )
            >= rho
        )
        excluded = tuple(branch for branch in canonical if branch not in witnessed)
        # Every excluded optimum must have a dual upper endpoint strictly below
        # rho; certified-infeasible branches are excluded by their Q receipt.
        for branch in excluded:
            receipt = next(item for item in receipts if item.branch is branch)
            if receipt.status is V1SignedWorkResourceBranchStatus.OPTIMAL and not (
                float(receipt.resource_upper_bound) < rho
            ):
                raise RuntimeError("resource endpoint interval clustering failed closed")
        increment = rho - problem.baseline_resource
        coordinates.append(
            V1SignedWorkEndogenousResourceCoordinate(
                problem_hash=problem.problem_hash,
                policy_hash=policy.policy_hash,
                ordinal=len(coordinates) + 1,
                total_resource_floor=rho,
                incremental_resource=increment,
                baseline_resource=problem.baseline_resource,
                required_lower_bound_hash=problem.required_lower_bound_hash,
                witnessed_feasible_branches=witnessed,
                certified_excluded_branches=excluded,
                source_branch_receipt_hashes=all_hashes,
            )
        )
    return tuple(coordinates)


def solve_v1_signed_work_resource_frontier_reference(
    problem: V1SignedWorkResourceFrontierProblem,
    policy: V1SignedWorkResourceFrontierPolicy | None = None,
) -> V1SignedWorkResourceFrontierResult:
    """Solve the two Phase-A branches and freeze endogenous Phase-B endpoints."""

    if type(problem) is not V1SignedWorkResourceFrontierProblem:
        raise TypeError("resource-frontier solve requires the exact problem")
    policy = V1SignedWorkResourceFrontierPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkResourceFrontierPolicy:
        raise TypeError("resource-frontier solve requires the exact policy")
    if problem.geometry.route_count > policy.small_domain_route_limit:
        raise PermissionError(
            "CPU resource-frontier reference exceeds its registered small-domain route limit; "
            "no route-scale CPU fallback is authorized"
        )
    spectral = _solve_branch(problem, policy, V1SignedWorkLeastWorkBranch.SPECTRAL)
    diagonal = _solve_branch(problem, policy, V1SignedWorkLeastWorkBranch.DIAGONAL_L1)
    receipts = (spectral, diagonal)
    if any(item.status is V1SignedWorkResourceBranchStatus.UNRESOLVED for item in receipts):
        union_status = V1SignedWorkResourceUnionStatus.UNRESOLVED
        coordinates: tuple[V1SignedWorkEndogenousResourceCoordinate, ...] = ()
        closed = False
    elif all(
        item.status is V1SignedWorkResourceBranchStatus.CERTIFIED_INFEASIBLE
        for item in receipts
    ):
        union_status = V1SignedWorkResourceUnionStatus.CERTIFIED_INFEASIBLE
        coordinates = ()
        closed = True
    else:
        coordinates = _derive_coordinates(problem, policy, receipts)
        union_status = (
            V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
            if coordinates
            else V1SignedWorkResourceUnionStatus.CERTIFIED_NO_NONEMPTY_ACTION
        )
        closed = True
    return V1SignedWorkResourceFrontierResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        union_status=union_status,
        coordinates=coordinates,
        all_branch_alternatives_closed=closed,
    )


def brute_force_hard_resource_maximum(
    problem: V1SignedWorkResourceFrontierProblem,
    branch: V1SignedWorkLeastWorkBranch,
) -> tuple[tuple[int, ...], float, float] | None:
    """Enumerate the tiny hard nested frontier for reference-only tests."""

    if type(problem) is not V1SignedWorkResourceFrontierProblem:
        raise TypeError("hard resource frontier requires exact problem")
    if type(branch) is not V1SignedWorkLeastWorkBranch:
        raise TypeError("hard resource frontier requires exact Q branch")
    geometry = problem.geometry
    if geometry.route_count > 24:
        raise PermissionError(
            "hard resource enumeration exceeds its registered tiny-domain limit"
        )
    required = tuple(int(i) for i in np.flatnonzero(problem.required_lower_bound))
    optional = tuple(
        int(i)
        for i in np.flatnonzero(
            np.logical_and(geometry.eligible_mask, np.logical_not(problem.required_lower_bound))
        )
    )
    best: tuple[tuple[int, ...], float, float] | None = None
    for count in range(len(optional) + 1):
        for addition in combinations(optional, count):
            subset = tuple(sorted((*required, *addition)))
            z = np.zeros(geometry.route_count, dtype=np.float64)
            if subset:
                z[list(subset)] = 1.0
            legal = True
            for item in geometry.block_capacities:
                if float(np.sum(z[item.block.start : item.block.stop])) > item.removal_capacity:
                    legal = False
                    break
            if not legal:
                continue
            q, _ = _q_value_gradient(geometry, branch, z)
            if q > geometry.quadratic_budget:
                continue
            resource = float(np.dot(geometry.resource, z))
            candidate = (subset, resource, q)
            if best is None or (resource, tuple(-i for i in subset)) > (
                best[1],
                tuple(-i for i in best[0]),
            ):
                best = candidate
    return best


__all__ = [
    "SIGNED_WORK_RESOURCE_FRONTIER_AUTHORITY_SCOPE",
    "SIGNED_WORK_RESOURCE_FRONTIER_COORDINATE_RULE_ID",
    "SIGNED_WORK_RESOURCE_FRONTIER_OBJECTIVE_ID",
    "V1SignedWorkEndogenousResourceCoordinate",
    "V1SignedWorkResourceBranchReceipt",
    "V1SignedWorkResourceBranchStatus",
    "V1SignedWorkResourceFrontierPolicy",
    "V1SignedWorkResourceFrontierProblem",
    "V1SignedWorkResourceFrontierResult",
    "V1SignedWorkResourceUnionStatus",
    "brute_force_hard_resource_maximum",
    "solve_v1_signed_work_resource_frontier_reference",
]
