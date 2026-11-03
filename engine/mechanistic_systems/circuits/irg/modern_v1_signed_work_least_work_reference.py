"""Small-domain signed-work least-work reference for canonical modern-v1.

This module is deliberately CPU/reference-only.  It solves the frozen-chart
least-work relaxation

    minimize    F_U(z) = sum_i w_i |(Bz)_i| + epsilon_F
    subject to  resource^T z >= rho
                Q_branch(z) <= tau_Q
                0 <= z <= parent_live * selectable
                block survivor capacities

for the two canonical Q branches independently.  It does not contain, derive,
or authorize a ``tau_F`` budget.  The signed native-fiber work operator ``B``
owns cancellation in its registered rows; the literal-J response operator owns
Q; and every result is proposal/theorem-reference evidence only, with no C,
K, O, mask-commit, terminal, or Pareto authority.

The numerical optimizer uses deterministic FP64 SciPy SLSQP on an epigraph
form.  The returned primal is then checked by a separate linear KKT feasibility
problem over active multipliers and the set-valued L1 subgradient.  Exact
zero-row cancellations are therefore admitted only when the subgradient
certificate closes; otherwise the branch refuses rather than pretending a
chosen sign was authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any, Final, Literal, Sequence

import numpy as np

from .contracts import content_hash


SIGNED_WORK_LEAST_WORK_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_reference_policy_v1"
)
SIGNED_WORK_LEAST_WORK_BLOCK_SCHEMA: Final = (
    "partition_free_v1_signed_work_live_block_v1"
)
SIGNED_WORK_LEAST_WORK_BLOCK_CAPACITY_SCHEMA: Final = (
    "partition_free_v1_signed_work_live_block_capacity_v1"
)
SIGNED_WORK_LEAST_WORK_PROBLEM_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_reference_problem_v1"
)
SIGNED_WORK_LEAST_WORK_PRIMAL_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_primal_receipt_v1"
)
SIGNED_WORK_LEAST_WORK_DUAL_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_dual_kkt_receipt_v1"
)
SIGNED_WORK_LEAST_WORK_BRANCH_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_branch_result_v1"
)
SIGNED_WORK_LEAST_WORK_RESULT_SCHEMA: Final = (
    "partition_free_v1_signed_work_least_work_result_v1"
)
SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE: Final = (
    "cpu_small_domain_signed_native_fiber_least_work_reference_only_no_tau_F_"
    "C_K_O_oracle_mask_commit_terminal_or_pareto_authority_v1"
)
SIGNED_WORK_LEAST_WORK_OBJECTIVE_ID: Final = (
    "minimize_certified_native_fiber_signed_work_upper_at_fixed_resource_floor_v1"
)
SIGNED_WORK_LEAST_WORK_F_ID: Final = (
    "F_upper_equals_sum_native_fiber_weights_abs_Bz_plus_registered_additive_"
    "upper_bound_v1"
)
SIGNED_WORK_LEAST_WORK_Q_SPECTRAL_ID: Final = (
    "Q_spectral_equals_norm_Hz_squared_plus_rho_norm_z_squared_v1"
)
SIGNED_WORK_LEAST_WORK_Q_DIAGONAL_ID: Final = (
    "Q_diagonal_l1_equals_norm_Hz_squared_plus_sqrt_d_dot_z_squared_v1"
)

_POLICY_PREFIX: Final = "partition-free-v1-signed-work-least-work-policy-"
_BLOCK_PREFIX: Final = "partition-free-v1-signed-work-least-work-block-"
_CAPACITY_PREFIX: Final = "partition-free-v1-signed-work-least-work-capacity-"
_PROBLEM_PREFIX: Final = "partition-free-v1-signed-work-least-work-problem-"
_PRIMAL_PREFIX: Final = "partition-free-v1-signed-work-least-work-primal-"
_DUAL_PREFIX: Final = "partition-free-v1-signed-work-least-work-dual-"
_BRANCH_PREFIX: Final = "partition-free-v1-signed-work-least-work-branch-"
_RESULT_PREFIX: Final = "partition-free-v1-signed-work-least-work-result-"

BranchName = Literal["spectral", "diagonal_l1"]
BranchStatus = Literal[
    "certified_relaxed_branch_optimum",
    "refuse_primal_infeasible",
    "refuse_optimizer_failed",
    "refuse_kkt_uncertified",
    "refuse_zero_row_subgradient_uncertified",
]
ZeroRowSubgradientStatus = Literal[
    "no_zero_work_rows",
    "zero_row_subgradient_certified",
    "zero_row_subgradient_uncertified",
]


class V1SignedWorkLeastWorkBranch(str, Enum):
    """The two independent convex Q branches."""

    SPECTRAL = "spectral"
    DIAGONAL_L1 = "diagonal_l1"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be one nonempty string")
    return value


def _bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
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


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array_f64(
    value: object,
    *,
    name: str,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be representable as float64") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim {ndim}")
    if shape is not None:
        if len(shape) != array.ndim:
            raise ValueError(f"{name} shape contract is malformed")
        for observed, expected in zip(array.shape, shape, strict=True):
            if expected is not None and observed != expected:
                raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _array_bool(value: object, *, name: str, size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (size,) or raw.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must be an exact bool route-axis vector")
    result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _tuple_f64(value: np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(value, dtype=np.float64).reshape(-1))


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = {
        "dtype": str(array.dtype),
        "shape": tuple(int(item) for item in array.shape),
        "bytes_sha256": content_hash(
            array.tobytes(order="C").hex(),
            prefix="partition-free-v1-signed-work-array-bytes-",
        ),
    }
    return content_hash(payload, prefix="partition-free-v1-signed-work-array-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkPolicy:
    """Numerical policy for the CPU small-domain reference."""

    policy_id: str = "modern-v1-signed-work-least-work-reference-default"
    solver: Literal["scipy_slsqp_epigraph"] = "scipy_slsqp_epigraph"
    max_iterations: int = 2_000
    ftol: float = 1e-12
    primal_feasibility_tolerance: float = 2e-8
    kkt_stationarity_tolerance: float = 2e-6
    zero_row_tolerance: float = 2e-9
    small_domain_route_limit: int = 32
    deterministic_start_count: int = 10
    relaxed_branch_optimality_only: bool = True
    hard_global_optimality_claim: bool = False
    tau_f_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_LEAST_WORK_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        if self.solver != "scipy_slsqp_epigraph":
            raise ValueError("unsupported signed-work least-work solver")
        for name in ("max_iterations", "small_domain_route_limit", "deterministic_start_count"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in ("ftol", "primal_feasibility_tolerance", "kkt_stationarity_tolerance"):
            object.__setattr__(self, name, _positive(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "zero_row_tolerance",
            _nonnegative(self.zero_row_tolerance, name="zero row tolerance"),
        )
        for name, expected in (
            ("relaxed_branch_optimality_only", True),
            ("hard_global_optimality_claim", False),
            ("tau_f_authority", False),
            ("mask_commit_authority", False),
        ):
            if _bool(getattr(self, name), name=name) is not expected:
                raise PermissionError("signed-work least-work policy crossed authority")
        if self.authority_scope != SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE:
            raise ValueError("signed-work policy authority scope is invalid")
        if self.schema != SIGNED_WORK_LEAST_WORK_POLICY_SCHEMA:
            raise ValueError("signed-work policy schema is invalid")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiveBlock:
    """Static contiguous block with required live survivors."""

    block_id: str
    start: int
    stop: int
    required_live_survivors: int = 1
    schema: str = SIGNED_WORK_LEAST_WORK_BLOCK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _identity(self.block_id, name="block id"))
        start = _nonnegative_int(self.start, name="block start")
        stop = _positive_int(self.stop, name="block stop")
        required = _nonnegative_int(
            self.required_live_survivors,
            name="required live survivors",
        )
        if stop <= start:
            raise ValueError("block stop must exceed block start")
        if required > stop - start:
            raise ValueError("required live survivors cannot exceed block size")
        if self.schema != SIGNED_WORK_LEAST_WORK_BLOCK_SCHEMA:
            raise ValueError("signed-work block schema is invalid")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "required_live_survivors", required)

    @property
    def block_hash(self) -> str:
        return content_hash(self, prefix=_BLOCK_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiveBlockCapacity:
    """Live/selectable-derived relaxed removal capacity for one block."""

    block: V1SignedWorkLiveBlock
    parent_live_count: int
    selectable_live_count: int
    removal_capacity: int
    schema: str = SIGNED_WORK_LEAST_WORK_BLOCK_CAPACITY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.block) is not V1SignedWorkLiveBlock:
            raise TypeError("block capacity requires an exact signed-work block")
        live = _nonnegative_int(self.parent_live_count, name="parent live count")
        selectable = _nonnegative_int(self.selectable_live_count, name="selectable live count")
        capacity = _nonnegative_int(self.removal_capacity, name="removal capacity")
        if live > self.block.stop - self.block.start or selectable > live:
            raise ValueError("block live/selectable counts are inconsistent")
        expected = min(selectable, max(0, live - self.block.required_live_survivors))
        if capacity != expected:
            raise ValueError("removal capacity must be live/selectable-derived")
        if self.schema != SIGNED_WORK_LEAST_WORK_BLOCK_CAPACITY_SCHEMA:
            raise ValueError("signed-work capacity schema is invalid")
        object.__setattr__(self, "parent_live_count", live)
        object.__setattr__(self, "selectable_live_count", selectable)
        object.__setattr__(self, "removal_capacity", capacity)

    @property
    def capacity_hash(self) -> str:
        return content_hash(self, prefix=_CAPACITY_PREFIX)


def _normalize_blocks(
    blocks: Sequence[V1SignedWorkLiveBlock],
    *,
    route_count: int,
) -> tuple[V1SignedWorkLiveBlock, ...]:
    result = tuple(blocks)
    if not result or any(type(block) is not V1SignedWorkLiveBlock for block in result):
        raise TypeError("live blocks must be exact signed-work block objects")
    expected = 0
    seen: set[str] = set()
    for block in result:
        if block.start != expected:
            raise ValueError("live blocks must be contiguous and ordered")
        if block.block_id in seen:
            raise ValueError("live block ids must be unique")
        seen.add(block.block_id)
        expected = block.stop
    if expected != route_count:
        raise ValueError("live blocks must cover the route axis")
    return result


def _capacity_for(
    block: V1SignedWorkLiveBlock,
    *,
    parent_live_axis: np.ndarray,
    selectable_axis: np.ndarray,
) -> V1SignedWorkLiveBlockCapacity:
    live = int(np.count_nonzero(parent_live_axis[block.start : block.stop]))
    selectable = int(
        np.count_nonzero(
            np.logical_and(
                parent_live_axis[block.start : block.stop],
                selectable_axis[block.start : block.stop],
            )
        )
    )
    return V1SignedWorkLiveBlockCapacity(
        block=block,
        parent_live_count=live,
        selectable_live_count=selectable,
        removal_capacity=min(selectable, max(0, live - block.required_live_survivors)),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkKKTReferenceProblem:
    """Frozen small-domain signed-work/Q least-work reference problem."""

    problem_id: str
    resource: np.ndarray = field(repr=False, compare=False)
    signed_work_matrix: np.ndarray = field(repr=False, compare=False)
    work_row_weights: np.ndarray = field(repr=False, compare=False)
    low_rank_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    residual_spectral_bound: float
    parent_live_axis: np.ndarray = field(repr=False, compare=False)
    selectable_axis: np.ndarray = field(repr=False, compare=False)
    live_blocks: tuple[V1SignedWorkLiveBlock, ...]
    resource_floor: float
    quadratic_budget: float
    route_axis_hash: str
    signed_work_operator_hash: str
    literal_j_operator_hash: str
    signed_work_additive_upper_bound: float = 0.0
    objective_id: str = SIGNED_WORK_LEAST_WORK_OBJECTIVE_ID
    signed_f_equation_id: str = SIGNED_WORK_LEAST_WORK_F_ID
    spectral_q_equation_id: str = SIGNED_WORK_LEAST_WORK_Q_SPECTRAL_ID
    diagonal_q_equation_id: str = SIGNED_WORK_LEAST_WORK_Q_DIAGONAL_ID
    tau_f_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_LEAST_WORK_PROBLEM_SCHEMA
    block_capacities: tuple[V1SignedWorkLiveBlockCapacity, ...] = field(init=False)
    eligible_axis_hash: str = field(init=False)
    _problem_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _identity(self.problem_id, name="problem id"))
        resource = _array_f64(self.resource, name="resource", ndim=1, nonnegative=True)
        route_count = int(resource.size)
        if route_count <= 0:
            raise ValueError("resource route axis must be nonempty")
        if bool(np.any(resource <= 0.0)):
            raise ValueError("resource must be positive on the route axis")
        signed = _array_f64(
            self.signed_work_matrix,
            name="signed work matrix",
            ndim=2,
            shape=(None, route_count),
        )
        weights = _array_f64(
            self.work_row_weights,
            name="work row weights",
            ndim=1,
            shape=(signed.shape[0],),
            nonnegative=True,
        )
        factor = _array_f64(
            self.low_rank_factor,
            name="low rank factor",
            ndim=2,
            shape=(None, route_count),
        )
        diagonal = _array_f64(
            self.residual_diagonal,
            name="residual diagonal",
            ndim=1,
            shape=(route_count,),
            nonnegative=True,
        )
        parent_live = _array_bool(self.parent_live_axis, name="parent live axis", size=route_count)
        selectable = _array_bool(self.selectable_axis, name="selectable axis", size=route_count)
        blocks = _normalize_blocks(self.live_blocks, route_count=route_count)
        capacities = tuple(
            _capacity_for(block, parent_live_axis=parent_live, selectable_axis=selectable)
            for block in blocks
        )
        rho = _nonnegative(self.residual_spectral_bound, name="residual spectral bound")
        if diagonal.size and rho + 128.0 * np.finfo(np.float64).eps < float(np.max(diagonal)):
            raise ValueError("residual spectral bound must cover residual diagonal")
        floor = _nonnegative(self.resource_floor, name="resource floor")
        budget = _nonnegative(self.quadratic_budget, name="quadratic budget")
        additive = _nonnegative(
            self.signed_work_additive_upper_bound,
            name="signed work additive upper bound",
        )
        for name in ("route_axis_hash", "signed_work_operator_hash", "literal_j_operator_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.objective_id != SIGNED_WORK_LEAST_WORK_OBJECTIVE_ID:
            raise ValueError("signed-work least-work objective id is invalid")
        if (
            self.signed_f_equation_id != SIGNED_WORK_LEAST_WORK_F_ID
            or self.spectral_q_equation_id != SIGNED_WORK_LEAST_WORK_Q_SPECTRAL_ID
            or self.diagonal_q_equation_id != SIGNED_WORK_LEAST_WORK_Q_DIAGONAL_ID
        ):
            raise ValueError("signed-work least-work equations are invalid")
        for name in ("tau_f_authority", "mask_commit_authority"):
            if _bool(getattr(self, name), name=name):
                raise PermissionError("signed-work problem crossed authority")
        if self.authority_scope != SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE:
            raise ValueError("signed-work problem authority scope is invalid")
        if self.schema != SIGNED_WORK_LEAST_WORK_PROBLEM_SCHEMA:
            raise ValueError("signed-work problem schema is invalid")
        eligible = np.logical_and(parent_live, selectable)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "signed_work_matrix", signed)
        object.__setattr__(self, "work_row_weights", weights)
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_diagonal", diagonal)
        object.__setattr__(self, "residual_spectral_bound", max(rho, float(np.max(diagonal, initial=0.0))))
        object.__setattr__(self, "parent_live_axis", parent_live)
        object.__setattr__(self, "selectable_axis", selectable)
        object.__setattr__(self, "live_blocks", blocks)
        object.__setattr__(self, "block_capacities", capacities)
        object.__setattr__(self, "resource_floor", floor)
        object.__setattr__(self, "quadratic_budget", budget)
        object.__setattr__(self, "signed_work_additive_upper_bound", additive)
        object.__setattr__(
            self,
            "eligible_axis_hash",
            content_hash(
                {
                    "route_axis_hash": self.route_axis_hash,
                    "parent_live": tuple(bool(x) for x in parent_live),
                    "selectable": tuple(bool(x) for x in selectable),
                },
                prefix="partition-free-v1-signed-work-eligible-axis-",
            ),
        )
        object.__setattr__(self, "_problem_hash", content_hash(self._payload(), prefix=_PROBLEM_PREFIX))

    @property
    def route_count(self) -> int:
        return int(self.resource.size)

    @property
    def work_row_count(self) -> int:
        return int(self.signed_work_matrix.shape[0])

    @property
    def eligible_mask(self) -> np.ndarray:
        result = np.logical_and(self.parent_live_axis, self.selectable_axis)
        result.setflags(write=False)
        return result

    @property
    def problem_hash(self) -> str:
        return self._problem_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "problem_id": self.problem_id,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "resource_digest": _digest_array(self.resource),
            "signed_work_matrix_digest": _digest_array(self.signed_work_matrix),
            "work_row_weights_digest": _digest_array(self.work_row_weights),
            "low_rank_factor_digest": _digest_array(self.low_rank_factor),
            "residual_diagonal_digest": _digest_array(self.residual_diagonal),
            "residual_spectral_bound": self.residual_spectral_bound,
            "parent_live_digest": _digest_array(self.parent_live_axis.astype(np.uint8)),
            "selectable_digest": _digest_array(self.selectable_axis.astype(np.uint8)),
            "eligible_axis_hash": self.eligible_axis_hash,
            "block_capacity_hashes": tuple(item.capacity_hash for item in self.block_capacities),
            "resource_floor": self.resource_floor,
            "quadratic_budget": self.quadratic_budget,
            "signed_work_operator_hash": self.signed_work_operator_hash,
            "literal_j_operator_hash": self.literal_j_operator_hash,
            "signed_work_additive_upper_bound": self.signed_work_additive_upper_bound,
            "objective_id": self.objective_id,
            "signed_f_equation_id": self.signed_f_equation_id,
            "spectral_q_equation_id": self.spectral_q_equation_id,
            "diagonal_q_equation_id": self.diagonal_q_equation_id,
            "tau_f_authority": self.tau_f_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "authority_scope": self.authority_scope,
        }


def _signed_work(problem: V1SignedWorkKKTReferenceProblem, z: np.ndarray) -> tuple[float, np.ndarray]:
    projections = problem.signed_work_matrix @ z
    value = float(np.dot(problem.work_row_weights, np.abs(projections)))
    return value + problem.signed_work_additive_upper_bound, projections


def _q_value_and_gradient(
    problem: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
    z: np.ndarray,
) -> tuple[float, np.ndarray]:
    low = problem.low_rank_factor @ z
    low_q = float(np.dot(low, low))
    grad = 2.0 * (problem.low_rank_factor.T @ low)
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        q = low_q + problem.residual_spectral_bound * float(np.dot(z, z))
        grad = grad + 2.0 * problem.residual_spectral_bound * z
    else:
        sqrt_d = np.sqrt(problem.residual_diagonal)
        projection = float(np.dot(sqrt_d, z))
        q = low_q + projection * projection
        grad = grad + 2.0 * projection * sqrt_d
    return q, np.asarray(grad, dtype=np.float64)


def _block_sums(problem: V1SignedWorkKKTReferenceProblem, z: np.ndarray) -> tuple[float, ...]:
    return tuple(
        float(np.sum(z[item.block.start : item.block.stop], dtype=np.float64))
        for item in problem.block_capacities
    )


def _maximum_resource_under_blocks(problem: V1SignedWorkKKTReferenceProblem) -> float:
    total = 0.0
    eligible = problem.eligible_mask
    for capacity in problem.block_capacities:
        if capacity.removal_capacity <= 0:
            continue
        indices = [
            index
            for index in range(capacity.block.start, capacity.block.stop)
            if bool(eligible[index])
        ]
        resources = sorted((float(problem.resource[index]) for index in indices), reverse=True)
        total += sum(resources[: capacity.removal_capacity])
    return float(total)


def _initial_zs(problem: V1SignedWorkKKTReferenceProblem, policy: V1SignedWorkLeastWorkPolicy) -> list[np.ndarray]:
    n = problem.route_count
    eligible = problem.eligible_mask
    starts: list[np.ndarray] = [np.zeros(n, dtype=np.float64)]

    uniform = np.zeros(n, dtype=np.float64)
    if problem.resource_floor > 0.0:
        remaining = problem.resource_floor
        for capacity in problem.block_capacities:
            indices = [
                index
                for index in range(capacity.block.start, capacity.block.stop)
                if bool(eligible[index])
            ]
            if not indices or capacity.removal_capacity <= 0:
                continue
            share = min(1.0, capacity.removal_capacity / max(1, len(indices)))
            for index in indices:
                uniform[index] = share
        resource = float(np.dot(problem.resource, uniform))
        if resource > 0.0:
            uniform *= min(1.0, problem.resource_floor / resource)
        starts.append(uniform)

    singleton_cost = np.sum(
        np.abs(problem.signed_work_matrix) * problem.work_row_weights[:, None],
        axis=0,
    )
    ratio = np.where(eligible, singleton_cost / problem.resource, np.inf)
    order = tuple(int(item) for item in np.lexsort((np.arange(n), ratio)))
    greedy = np.zeros(n, dtype=np.float64)
    used = [0.0 for _ in problem.block_capacities]
    block_stops = np.asarray([item.block.stop for item in problem.block_capacities], dtype=np.int64)
    remaining = problem.resource_floor
    for index in order:
        if not eligible[index] or remaining <= 0.0:
            continue
        block_index = int(np.searchsorted(block_stops, index, side="right"))
        cap_left = problem.block_capacities[block_index].removal_capacity - used[block_index]
        if cap_left <= 0.0:
            continue
        amount = min(1.0, cap_left, remaining / float(problem.resource[index]))
        greedy[index] = amount
        used[block_index] += amount
        remaining -= amount * float(problem.resource[index])
    starts.append(greedy)

    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                pair = np.zeros(n, dtype=np.float64)
                if eligible[i] and eligible[j]:
                    pair[i] = 0.5
                    pair[j] = 0.5
                    starts.append(pair)
    rng = np.random.default_rng(20260828)
    for _ in range(max(0, policy.deterministic_start_count - len(starts))):
        raw = rng.random(n)
        raw[~eligible] = 0.0
        for capacity in problem.block_capacities:
            segment = raw[capacity.block.start : capacity.block.stop]
            total = float(np.sum(segment))
            if total > capacity.removal_capacity and total > 0.0:
                segment *= capacity.removal_capacity / total
        starts.append(raw)
    # Stable unique starts.
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for z in starts:
        clipped = np.clip(z, 0.0, 1.0)
        clipped[~eligible] = 0.0
        unique.setdefault(tuple(float(x) for x in np.round(clipped, 15)), clipped)
    return list(unique.values())


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkPrimalReceipt:
    branch: V1SignedWorkLeastWorkBranch
    z: tuple[float, ...]
    resource_value: float
    signed_work_upper: float
    quadratic_value: float
    block_sums: tuple[float, ...]
    maximum_box_violation: float
    resource_floor_violation: float
    quadratic_violation: float
    maximum_block_violation: float
    feasible: bool
    schema: str = SIGNED_WORK_LEAST_WORK_PRIMAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("primal receipt branch must be exact")
        z = tuple(_nonnegative(item, name="z value") for item in self.z)
        if any(value > 1.0 for value in z):
            raise ValueError("z values must lie in [0,1]")
        object.__setattr__(self, "z", z)
        for name in (
            "resource_value",
            "signed_work_upper",
            "quadratic_value",
            "maximum_box_violation",
            "resource_floor_violation",
            "quadratic_violation",
            "maximum_block_violation",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "block_sums",
            tuple(_nonnegative(item, name="block sum") for item in self.block_sums),
        )
        if type(self.feasible) is not bool:
            raise TypeError("primal feasible must be exact bool")
        if self.schema != SIGNED_WORK_LEAST_WORK_PRIMAL_SCHEMA:
            raise ValueError("primal receipt schema is invalid")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PRIMAL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkDualKKTReceipt:
    branch: V1SignedWorkLeastWorkBranch
    resource_dual: float
    q_dual: float
    block_duals: tuple[float, ...]
    stationarity_linf_residual: float
    complementarity_linf_residual: float
    zero_work_row_count: int
    zero_row_subgradient_status: ZeroRowSubgradientStatus
    kkt_certified: bool
    scipy_success: bool
    scipy_status: int
    scipy_message: str
    relaxed_branch_optimality_only: bool = True
    hard_global_optimality_claim: bool = False
    tau_f_authority: bool = False
    mask_commit_authority: bool = False
    authority_scope: str = SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_LEAST_WORK_DUAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("dual receipt branch must be exact")
        for name in (
            "resource_dual",
            "q_dual",
            "stationarity_linf_residual",
            "complementarity_linf_residual",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "block_duals",
            tuple(_nonnegative(item, name="block dual") for item in self.block_duals),
        )
        object.__setattr__(
            self,
            "zero_work_row_count",
            _nonnegative_int(self.zero_work_row_count, name="zero work row count"),
        )
        if self.zero_row_subgradient_status not in {
            "no_zero_work_rows",
            "zero_row_subgradient_certified",
            "zero_row_subgradient_uncertified",
        }:
            raise ValueError("unsupported zero-row subgradient status")
        for name in ("kkt_certified", "scipy_success"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be exact bool")
        object.__setattr__(self, "scipy_status", _nonnegative_int(self.scipy_status, name="scipy status"))
        object.__setattr__(self, "scipy_message", _identity(self.scipy_message, name="scipy message"))
        for name, expected in (
            ("relaxed_branch_optimality_only", True),
            ("hard_global_optimality_claim", False),
            ("tau_f_authority", False),
            ("mask_commit_authority", False),
        ):
            if _bool(getattr(self, name), name=name) is not expected:
                raise PermissionError("dual receipt crossed authority")
        if self.authority_scope != SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE:
            raise ValueError("dual receipt authority scope is invalid")
        if self.schema != SIGNED_WORK_LEAST_WORK_DUAL_SCHEMA:
            raise ValueError("dual receipt schema is invalid")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DUAL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkBranchResult:
    branch: V1SignedWorkLeastWorkBranch
    status: BranchStatus
    problem_hash: str
    policy_hash: str
    primal: V1SignedWorkLeastWorkPrimalReceipt | None
    dual_kkt: V1SignedWorkLeastWorkDualKKTReceipt | None
    refusal_reason: str | None = None
    relaxed_branch_optimality_only: bool = True
    hard_global_optimality_claim: bool = False
    tau_f_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SIGNED_WORK_LEAST_WORK_BRANCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("branch result branch must be exact")
        if self.status not in {
            "certified_relaxed_branch_optimum",
            "refuse_primal_infeasible",
            "refuse_optimizer_failed",
            "refuse_kkt_uncertified",
            "refuse_zero_row_subgradient_uncertified",
        }:
            raise ValueError("unsupported signed-work branch status")
        object.__setattr__(self, "problem_hash", _identity(self.problem_hash, name="problem hash"))
        object.__setattr__(self, "policy_hash", _identity(self.policy_hash, name="policy hash"))
        if self.primal is not None and type(self.primal) is not V1SignedWorkLeastWorkPrimalReceipt:
            raise TypeError("branch primal must be exact or None")
        if self.dual_kkt is not None and type(self.dual_kkt) is not V1SignedWorkLeastWorkDualKKTReceipt:
            raise TypeError("branch dual receipt must be exact or None")
        if self.status == "certified_relaxed_branch_optimum":
            if self.primal is None or self.dual_kkt is None:
                raise ValueError("certified branch requires primal and dual receipts")
            if not (self.primal.feasible and self.dual_kkt.kkt_certified):
                raise ValueError("certified branch lacks primal/KKT certification")
            if self.refusal_reason is not None:
                raise ValueError("certified branch cannot carry refusal reason")
        elif self.refusal_reason is None:
            raise ValueError("refused branch requires a reason")
        for name, expected in (
            ("relaxed_branch_optimality_only", True),
            ("hard_global_optimality_claim", False),
            ("tau_f_authority", False),
            ("mask_commit_authority", False),
        ):
            if _bool(getattr(self, name), name=name) is not expected:
                raise PermissionError("branch result crossed authority")
        if self.schema != SIGNED_WORK_LEAST_WORK_BRANCH_SCHEMA:
            raise ValueError("branch result schema is invalid")

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "branch": self.branch.value,
                "status": self.status,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "primal_hash": None if self.primal is None else self.primal.receipt_hash,
                "dual_hash": None if self.dual_kkt is None else self.dual_kkt.receipt_hash,
                "refusal_reason": self.refusal_reason,
                "relaxed_branch_optimality_only": self.relaxed_branch_optimality_only,
                "hard_global_optimality_claim": self.hard_global_optimality_claim,
                "tau_f_authority": self.tau_f_authority,
                "mask_commit_authority": self.mask_commit_authority,
            },
            prefix=_BRANCH_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLeastWorkResult:
    problem_hash: str
    policy_hash: str
    spectral: V1SignedWorkLeastWorkBranchResult
    diagonal_l1: V1SignedWorkLeastWorkBranchResult
    selected_branch: V1SignedWorkLeastWorkBranch | None
    selected_branch_choice_certified: bool
    relaxed_branch_optimality_only: bool = True
    hard_global_optimality_claim: bool = False
    tau_f_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SIGNED_WORK_LEAST_WORK_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_hash", _identity(self.problem_hash, name="problem hash"))
        object.__setattr__(self, "policy_hash", _identity(self.policy_hash, name="policy hash"))
        if self.spectral.branch is not V1SignedWorkLeastWorkBranch.SPECTRAL:
            raise ValueError("spectral result is mislabelled")
        if self.diagonal_l1.branch is not V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
            raise ValueError("diagonal result is mislabelled")
        if self.selected_branch is not None and type(self.selected_branch) is not V1SignedWorkLeastWorkBranch:
            raise TypeError("selected branch must be exact branch or None")
        if type(self.selected_branch_choice_certified) is not bool:
            raise TypeError("selected branch choice flag must be exact bool")
        for name, expected in (
            ("relaxed_branch_optimality_only", True),
            ("hard_global_optimality_claim", False),
            ("tau_f_authority", False),
            ("mask_commit_authority", False),
        ):
            if _bool(getattr(self, name), name=name) is not expected:
                raise PermissionError("result crossed authority")
        if self.schema != SIGNED_WORK_LEAST_WORK_RESULT_SCHEMA:
            raise ValueError("result schema is invalid")

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "spectral_hash": self.spectral.result_hash,
                "diagonal_l1_hash": self.diagonal_l1.result_hash,
                "selected_branch": None if self.selected_branch is None else self.selected_branch.value,
                "selected_branch_choice_certified": self.selected_branch_choice_certified,
                "relaxed_branch_optimality_only": self.relaxed_branch_optimality_only,
                "hard_global_optimality_claim": self.hard_global_optimality_claim,
                "tau_f_authority": self.tau_f_authority,
                "mask_commit_authority": self.mask_commit_authority,
            },
            prefix=_RESULT_PREFIX,
        )


def _primal_receipt(
    problem: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
    z: np.ndarray,
    *,
    tolerance: float,
) -> V1SignedWorkLeastWorkPrimalReceipt:
    z = np.asarray(z, dtype=np.float64).reshape(problem.route_count)
    work, _ = _signed_work(problem, z)
    q, _ = _q_value_and_gradient(problem, branch, z)
    resource = float(np.dot(problem.resource, z))
    eligible = problem.eligible_mask
    box_violation = max(
        0.0,
        float(np.max(np.maximum(-z, 0.0), initial=0.0)),
        float(np.max(np.maximum(z - 1.0, 0.0), initial=0.0)),
        float(np.max(np.abs(z[~eligible]), initial=0.0)) if bool(np.any(~eligible)) else 0.0,
    )
    block_sums = _block_sums(problem, z)
    block_violation = max(
        (
            max(0.0, value - capacity.removal_capacity)
            for value, capacity in zip(block_sums, problem.block_capacities, strict=True)
        ),
        default=0.0,
    )
    resource_violation = max(0.0, problem.resource_floor - resource)
    q_violation = max(0.0, q - problem.quadratic_budget)
    feasible = max(box_violation, block_violation, resource_violation, q_violation) <= tolerance
    clipped = np.clip(z, 0.0, 1.0)
    clipped[~eligible] = 0.0
    clipped[np.abs(clipped) <= 10.0 * np.finfo(np.float64).eps] = 0.0
    clipped[np.abs(clipped - 1.0) <= 10.0 * np.finfo(np.float64).eps] = 1.0
    return V1SignedWorkLeastWorkPrimalReceipt(
        branch=branch,
        z=_tuple_f64(clipped),
        resource_value=resource,
        signed_work_upper=work,
        quadratic_value=q,
        block_sums=block_sums,
        maximum_box_violation=box_violation,
        resource_floor_violation=resource_violation,
        quadratic_violation=q_violation,
        maximum_block_violation=block_violation,
        feasible=feasible,
    )


def _solve_epigraph(
    problem: V1SignedWorkKKTReferenceProblem,
    policy: V1SignedWorkLeastWorkPolicy,
    branch: V1SignedWorkLeastWorkBranch,
) -> tuple[np.ndarray | None, Any]:
    from scipy.optimize import minimize

    n = problem.route_count
    m = problem.work_row_count
    eligible = problem.eligible_mask
    bounds: list[tuple[float, float]] = [
        (0.0, 1.0 if bool(eligible[index]) else 0.0) for index in range(n)
    ] + [(0.0, None) for _ in range(m)]

    def unpack(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return v[:n], v[n:]

    def objective(v: np.ndarray) -> float:
        _z, t = unpack(v)
        return float(np.dot(problem.work_row_weights, t)) + problem.signed_work_additive_upper_bound

    def objective_jac(_v: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(n, dtype=np.float64), problem.work_row_weights))

    def resource_constraint(v: np.ndarray) -> float:
        z, _t = unpack(v)
        return float(np.dot(problem.resource, z) - problem.resource_floor)

    def resource_jac(_v: np.ndarray) -> np.ndarray:
        return np.concatenate((problem.resource, np.zeros(m, dtype=np.float64)))

    def q_constraint(v: np.ndarray) -> float:
        z, _t = unpack(v)
        q, _g = _q_value_and_gradient(problem, branch, z)
        return float(problem.quadratic_budget - q)

    def q_jac(v: np.ndarray) -> np.ndarray:
        z, _t = unpack(v)
        _q, g = _q_value_and_gradient(problem, branch, z)
        return np.concatenate((-g, np.zeros(m, dtype=np.float64)))

    constraints: list[dict[str, Any]] = [
        {"type": "ineq", "fun": resource_constraint, "jac": resource_jac},
        {"type": "ineq", "fun": q_constraint, "jac": q_jac},
    ]
    for capacity in problem.block_capacities:
        start = capacity.block.start
        stop = capacity.block.stop
        bound = float(capacity.removal_capacity)

        def block_fun(v: np.ndarray, start: int = start, stop: int = stop, bound: float = bound) -> float:
            z, _t = unpack(v)
            return bound - float(np.sum(z[start:stop], dtype=np.float64))

        def block_jac(_v: np.ndarray, start: int = start, stop: int = stop) -> np.ndarray:
            jac = np.zeros(n + m, dtype=np.float64)
            jac[start:stop] = -1.0
            return jac

        constraints.append({"type": "ineq", "fun": block_fun, "jac": block_jac})

    for row in range(m):
        b = problem.signed_work_matrix[row]

        def plus_fun(v: np.ndarray, row: int = row, b: np.ndarray = b) -> float:
            z, t = unpack(v)
            return float(t[row] - np.dot(b, z))

        def plus_jac(_v: np.ndarray, row: int = row, b: np.ndarray = b) -> np.ndarray:
            jac = np.zeros(n + m, dtype=np.float64)
            jac[:n] = -b
            jac[n + row] = 1.0
            return jac

        def minus_fun(v: np.ndarray, row: int = row, b: np.ndarray = b) -> float:
            z, t = unpack(v)
            return float(t[row] + np.dot(b, z))

        def minus_jac(_v: np.ndarray, row: int = row, b: np.ndarray = b) -> np.ndarray:
            jac = np.zeros(n + m, dtype=np.float64)
            jac[:n] = b
            jac[n + row] = 1.0
            return jac

        constraints.append({"type": "ineq", "fun": plus_fun, "jac": plus_jac})
        constraints.append({"type": "ineq", "fun": minus_fun, "jac": minus_jac})

    best: tuple[float, np.ndarray, Any] | None = None
    for start_z in _initial_zs(problem, policy):
        start_t = np.abs(problem.signed_work_matrix @ start_z)
        start = np.concatenate((start_z, start_t))
        result = minimize(
            objective,
            start,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": policy.max_iterations,
                "ftol": policy.ftol,
                "disp": False,
            },
        )
        if not np.isfinite(result.fun):
            continue
        z, _t = unpack(np.asarray(result.x, dtype=np.float64))
        receipt = _primal_receipt(problem, branch, z, tolerance=policy.primal_feasibility_tolerance)
        if not receipt.feasible:
            continue
        key_z = tuple(round(value, 14) for value in receipt.z)
        candidate_key = (receipt.signed_work_upper, tuple(abs(value) for value in key_z), key_z)
        if best is None or candidate_key < best[0]:  # type: ignore[operator]
            best = (candidate_key, np.asarray(receipt.z, dtype=np.float64), result)  # type: ignore[assignment]
    if best is None:
        return None, None
    return best[1], best[2]


def _kkt_receipt(
    problem: V1SignedWorkKKTReferenceProblem,
    policy: V1SignedWorkLeastWorkPolicy,
    branch: V1SignedWorkLeastWorkBranch,
    z: np.ndarray,
    scipy_result: Any,
) -> V1SignedWorkLeastWorkDualKKTReceipt:
    from scipy.optimize import linprog

    n = problem.route_count
    m = problem.work_row_count
    B = problem.signed_work_matrix
    w = problem.work_row_weights
    projections = B @ z
    q, gq = _q_value_and_gradient(problem, branch, z)
    zero_mask = np.abs(projections) <= policy.zero_row_tolerance
    zero_count = int(np.count_nonzero(zero_mask))

    # Variables: y_m, gamma_resource, mu_q, beta_blocks, alpha_low_n,
    # alpha_upper_n, epsilon.
    block_count = len(problem.block_capacities)
    y_offset = 0
    gamma_index = y_offset + m
    mu_index = gamma_index + 1
    beta_offset = mu_index + 1
    alpha_low_offset = beta_offset + block_count
    alpha_upper_offset = alpha_low_offset + n
    eps_index = alpha_upper_offset + n
    variable_count = eps_index + 1

    bounds: list[tuple[float | None, float | None]] = []
    for row in range(m):
        if zero_mask[row]:
            bounds.append((-float(w[row]), float(w[row])))
        else:
            value = float(w[row] * math.copysign(1.0, projections[row]))
            bounds.append((value, value))
    resource_active = problem.resource_floor - float(np.dot(problem.resource, z)) >= -policy.primal_feasibility_tolerance
    bounds.append((0.0, None) if resource_active else (0.0, 0.0))
    q_active = q - problem.quadratic_budget >= -policy.primal_feasibility_tolerance
    bounds.append((0.0, None) if q_active else (0.0, 0.0))
    block_sums = _block_sums(problem, z)
    for value, capacity in zip(block_sums, problem.block_capacities, strict=True):
        active = value - capacity.removal_capacity >= -policy.primal_feasibility_tolerance
        bounds.append((0.0, None) if active else (0.0, 0.0))
    eligible = problem.eligible_mask
    for index in range(n):
        lower_active = z[index] <= policy.primal_feasibility_tolerance
        bounds.append((0.0, None) if lower_active else (0.0, 0.0))
    for index in range(n):
        upper = 1.0 if bool(eligible[index]) else 0.0
        upper_active = upper - z[index] <= policy.primal_feasibility_tolerance
        bounds.append((0.0, None) if upper_active else (0.0, 0.0))
    bounds.append((0.0, None))

    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for index in range(n):
        row = np.zeros(variable_count, dtype=np.float64)
        row[y_offset : y_offset + m] = B[:, index]
        row[gamma_index] = -float(problem.resource[index])
        row[mu_index] = float(gq[index])
        for block_index, capacity in enumerate(problem.block_capacities):
            if capacity.block.start <= index < capacity.block.stop:
                row[beta_offset + block_index] = 1.0
                break
        row[alpha_low_offset + index] = -1.0
        row[alpha_upper_offset + index] = 1.0
        row[eps_index] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
        a_ub.append(-row)
        b_ub.append(0.0)

    c = np.zeros(variable_count, dtype=np.float64)
    c[eps_index] = 1.0
    lp = linprog(
        c,
        A_ub=np.asarray(a_ub, dtype=np.float64),
        b_ub=np.asarray(b_ub, dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not lp.success:
        residual = float("inf")
        gamma = 0.0
        mu = 0.0
        beta = np.zeros(block_count, dtype=np.float64)
    else:
        x = np.asarray(lp.x, dtype=np.float64)
        residual = float(x[eps_index])
        gamma = float(x[gamma_index])
        mu = float(x[mu_index])
        beta = np.asarray(x[beta_offset : beta_offset + block_count], dtype=np.float64)

    complementarity_terms = [
        abs(gamma * max(0.0, float(np.dot(problem.resource, z)) - problem.resource_floor)),
        abs(mu * max(0.0, problem.quadratic_budget - q)),
    ]
    complementarity_terms.extend(
        abs(float(beta_i) * max(0.0, capacity.removal_capacity - value))
        for beta_i, value, capacity in zip(beta, block_sums, problem.block_capacities, strict=True)
    )
    complementarity = max(complementarity_terms, default=0.0)
    kkt_certified = bool(
        lp.success
        and math.isfinite(residual)
        and residual <= policy.kkt_stationarity_tolerance
        and complementarity <= max(policy.kkt_stationarity_tolerance, policy.primal_feasibility_tolerance)
    )
    if zero_count == 0:
        zero_status: ZeroRowSubgradientStatus = "no_zero_work_rows"
    elif kkt_certified:
        zero_status = "zero_row_subgradient_certified"
    else:
        zero_status = "zero_row_subgradient_uncertified"
    message = str(getattr(scipy_result, "message", "scipy_result_unavailable")) or "scipy_result_unavailable"
    status = int(getattr(scipy_result, "status", 0))
    if status < 0:
        status = 0
    return V1SignedWorkLeastWorkDualKKTReceipt(
        branch=branch,
        resource_dual=gamma,
        q_dual=mu,
        block_duals=tuple(float(item) for item in beta),
        stationarity_linf_residual=residual if math.isfinite(residual) else np.finfo(np.float64).max,
        complementarity_linf_residual=complementarity,
        zero_work_row_count=zero_count,
        zero_row_subgradient_status=zero_status,
        kkt_certified=kkt_certified,
        scipy_success=bool(getattr(scipy_result, "success", False)),
        scipy_status=status,
        scipy_message=message,
    )


def _solve_branch(
    problem: V1SignedWorkKKTReferenceProblem,
    policy: V1SignedWorkLeastWorkPolicy,
    branch: V1SignedWorkLeastWorkBranch,
) -> V1SignedWorkLeastWorkBranchResult:
    if problem.route_count > policy.small_domain_route_limit:
        return V1SignedWorkLeastWorkBranchResult(
            branch=branch,
            status="refuse_optimizer_failed",
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            primal=None,
            dual_kkt=None,
            refusal_reason="route_count_exceeds_cpu_small_domain_reference_limit",
        )
    if problem.resource_floor > _maximum_resource_under_blocks(problem) + policy.primal_feasibility_tolerance:
        return V1SignedWorkLeastWorkBranchResult(
            branch=branch,
            status="refuse_primal_infeasible",
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            primal=None,
            dual_kkt=None,
            refusal_reason="resource_floor_exceeds_live_selectable_block_capacity",
        )
    z, scipy_result = _solve_epigraph(problem, policy, branch)
    if z is None or scipy_result is None:
        return V1SignedWorkLeastWorkBranchResult(
            branch=branch,
            status="refuse_primal_infeasible",
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            primal=None,
            dual_kkt=None,
            refusal_reason="no_primal_point_certified_for_resource_Q_and_legality",
        )
    primal = _primal_receipt(problem, branch, z, tolerance=policy.primal_feasibility_tolerance)
    if not primal.feasible:
        return V1SignedWorkLeastWorkBranchResult(
            branch=branch,
            status="refuse_primal_infeasible",
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            primal=primal,
            dual_kkt=None,
            refusal_reason="returned_primal_failed_outward_feasibility",
        )
    dual = _kkt_receipt(problem, policy, branch, np.asarray(primal.z, dtype=np.float64), scipy_result)
    if not dual.kkt_certified:
        status: BranchStatus = (
            "refuse_zero_row_subgradient_uncertified"
            if dual.zero_row_subgradient_status == "zero_row_subgradient_uncertified"
            else "refuse_kkt_uncertified"
        )
        return V1SignedWorkLeastWorkBranchResult(
            branch=branch,
            status=status,
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            primal=primal,
            dual_kkt=dual,
            refusal_reason=dual.zero_row_subgradient_status
            if status == "refuse_zero_row_subgradient_uncertified"
            else "kkt_stationarity_or_complementarity_uncertified",
        )
    return V1SignedWorkLeastWorkBranchResult(
        branch=branch,
        status="certified_relaxed_branch_optimum",
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        primal=primal,
        dual_kkt=dual,
    )


def solve_v1_signed_work_least_work_reference(
    problem: V1SignedWorkKKTReferenceProblem,
    policy: V1SignedWorkLeastWorkPolicy | None = None,
) -> V1SignedWorkLeastWorkResult:
    """Solve both fixed-resource least-work Q branches and receipt the result."""

    if type(problem) is not V1SignedWorkKKTReferenceProblem:
        raise TypeError("signed-work least-work solve requires the exact problem")
    policy = V1SignedWorkLeastWorkPolicy() if policy is None else policy
    if type(policy) is not V1SignedWorkLeastWorkPolicy:
        raise TypeError("signed-work least-work solve requires the exact policy")
    spectral = _solve_branch(problem, policy, V1SignedWorkLeastWorkBranch.SPECTRAL)
    diagonal = _solve_branch(problem, policy, V1SignedWorkLeastWorkBranch.DIAGONAL_L1)
    certified = [
        branch
        for branch in (spectral, diagonal)
        if branch.status == "certified_relaxed_branch_optimum"
    ]
    if not certified:
        selected = None
        choice = False
    else:
        best = min(
            certified,
            key=lambda item: (
                item.primal.signed_work_upper if item.primal is not None else math.inf,
                item.primal.quadratic_value if item.primal is not None else math.inf,
                item.branch.value,
            ),
        )
        selected = best.branch
        choice = True
    return V1SignedWorkLeastWorkResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        spectral=spectral,
        diagonal_l1=diagonal,
        selected_branch=selected,
        selected_branch_choice_certified=choice,
    )


def brute_force_hard_least_work_frontier(
    problem: V1SignedWorkKKTReferenceProblem,
    branch: V1SignedWorkLeastWorkBranch,
) -> tuple[tuple[int, ...], float, float, float] | None:
    """Return the best hard-mask feasible point for tiny comparison tests."""

    from itertools import combinations

    if type(problem) is not V1SignedWorkKKTReferenceProblem:
        raise TypeError("hard frontier requires exact signed-work problem")
    eligible_indices = tuple(int(i) for i in np.flatnonzero(problem.eligible_mask))
    best: tuple[tuple[int, ...], float, float, float] | None = None
    block_stops = np.asarray([item.block.stop for item in problem.block_capacities], dtype=np.int64)
    for count in range(0, len(eligible_indices) + 1):
        for subset in combinations(eligible_indices, count):
            z = np.zeros(problem.route_count, dtype=np.float64)
            z[list(subset)] = 1.0
            used = [0 for _ in problem.block_capacities]
            legal = True
            for index in subset:
                block_index = int(np.searchsorted(block_stops, index, side="right"))
                used[block_index] += 1
                if used[block_index] > problem.block_capacities[block_index].removal_capacity:
                    legal = False
                    break
            if not legal:
                continue
            resource = float(np.dot(problem.resource, z))
            if resource + 1e-12 < problem.resource_floor:
                continue
            q, _ = _q_value_and_gradient(problem, branch, z)
            if q > problem.quadratic_budget + 1e-12:
                continue
            work, _ = _signed_work(problem, z)
            key = (work, q, resource, subset)
            if best is None or key < (best[1], best[3], best[2], best[0]):
                best = (tuple(int(i) for i in subset), work, resource, q)
    return best


__all__ = [
    "SIGNED_WORK_LEAST_WORK_AUTHORITY_SCOPE",
    "SIGNED_WORK_LEAST_WORK_F_ID",
    "SIGNED_WORK_LEAST_WORK_OBJECTIVE_ID",
    "SIGNED_WORK_LEAST_WORK_Q_DIAGONAL_ID",
    "SIGNED_WORK_LEAST_WORK_Q_SPECTRAL_ID",
    "V1SignedWorkKKTReferenceProblem",
    "V1SignedWorkLeastWorkBranch",
    "V1SignedWorkLeastWorkBranchResult",
    "V1SignedWorkLeastWorkDualKKTReceipt",
    "V1SignedWorkLeastWorkPolicy",
    "V1SignedWorkLeastWorkPrimalReceipt",
    "V1SignedWorkLeastWorkResult",
    "V1SignedWorkLiveBlock",
    "V1SignedWorkLiveBlockCapacity",
    "brute_force_hard_least_work_frontier",
    "solve_v1_signed_work_least_work_reference",
]
