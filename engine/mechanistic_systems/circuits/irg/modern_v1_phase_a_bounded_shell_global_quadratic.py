"""Certified small-shell quadratic resource bounds for canonical v1 Phase A.

The population Loewner lower matrix is a valid quadratic lower envelope even
when interval construction makes it indefinite.  Projecting that matrix back
to the PSD cone would raise the lower envelope and can therefore manufacture
feasibility or stability.  This owner instead solves the resulting bounded
quadratic problem globally over the registered ``B <= 8`` gate box.

The implementation is deliberately compact-domain only.  It uses exact
rational interval arithmetic for the branch-and-bound proof and exposes no
route-scale tensors.  A finite engineering box budget may stop refinement,
but it cannot turn an unresolved interval into a decision.  This module owns
only a certified resource-endpoint interval; it grants no D, C, K, O, commit,
terminal, performance, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import heapq
import itertools
import math
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash


PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_SCHEMA: Final = (
    "partition_free_v1_phase_A_bounded_shell_global_quadratic_v2"
)
PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_AUTHORITY: Final = (
    "compact_B_le_8_global_quadratic_resource_interval_only_no_D_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-phase-A-bounded-shell-global-quadratic-"
_PROBLEM_TOKEN: Final = object()
_DESIGN_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_MAXIMUM_SHELL: Final = 8
_CAPACITY_CONSTRAINT_SCHEMA: Final = (
    "partition_free_v1_phase_A_disjoint_shell_block_capacity_constraint_v1"
)


class V1PhaseABoundedShellGlobalQuadraticRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_PROBLEM = "invalid_problem"
    NONFINITE = "nonfinite"
    ASYMMETRIC_MATRIX = "asymmetric_matrix"
    SHELL_CAP_EXCEEDED = "shell_cap_exceeded"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseABoundedShellGlobalQuadraticRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseABoundedShellGlobalQuadraticRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseABoundedShellGlobalQuadraticRefusalCode:
            raise TypeError("bounded-shell refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseABoundedShellGlobalQuadraticRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseABoundedShellGlobalQuadraticRefusal(code, message)


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _refuse(
            V1PhaseABoundedShellGlobalQuadraticRefusalCode.NONFINITE,
            f"{name} must be one finite binary64 value",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _float_down(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result


def _float_up(value: Fraction) -> float:
    result = float(value)
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _fraction_axis(value: Sequence[float], *, name: str) -> tuple[Fraction, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _refuse(
            V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
            f"{name} must be one finite sequence",
        )
    return tuple(Fraction.from_float(_finite(item, name=f"{name} entry")) for item in value)


def _payload_hash(payload: dict[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


@dataclass(frozen=True, slots=True)
class V1PhaseAShellBlockCapacityConstraint:
    """One relaxed shell sum cap induced by an executable live-route block."""

    block_id: str
    shell_indices: tuple[int, ...]
    removal_capacity: float
    one_live_survivor_block: bool = True
    schema: str = _CAPACITY_CONSTRAINT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.block_id) is not str or not self.block_id:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                "capacity block_id must be one nonempty exact string",
            )
        if type(self.shell_indices) is not tuple or not self.shell_indices:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                "capacity shell_indices must be one nonempty immutable tuple",
            )
        indices: list[int] = []
        for index in self.shell_indices:
            if isinstance(index, bool) or type(index) is not int or index < 0:
                _refuse(
                    V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                    "capacity shell indices must be exact nonnegative integers",
                )
            indices.append(index)
        if len(indices) != len(set(indices)) or tuple(indices) != tuple(sorted(indices)):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "capacity shell indices must be sorted and unique",
            )
        _finite(self.removal_capacity, name="removal_capacity")
        if self.removal_capacity < 0.0 or self.removal_capacity > float(len(indices)):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "capacity must lie in [0, shell routes covered by the block]",
            )
        if (
            type(self.one_live_survivor_block) is not bool
            or self.one_live_survivor_block is not True
            or self.schema != _CAPACITY_CONSTRAINT_SCHEMA
        ):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.AUTHORITY_VIOLATION,
                "capacity constraint crossed one-live-survivor authority",
            )


def _capacity_constraints(
    value: object,
    *,
    shell: int,
    lower_gate_bounds: Sequence[float],
) -> tuple[V1PhaseAShellBlockCapacityConstraint, ...]:
    if type(value) is not tuple:
        _refuse(
            V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
            "shell block capacity constraints must be one immutable tuple",
        )
    seen_indices: set[int] = set()
    seen_blocks: set[str] = set()
    normalized: list[V1PhaseAShellBlockCapacityConstraint] = []
    lower = tuple(Fraction.from_float(_finite(item, name="lower gate")) for item in lower_gate_bounds)
    for constraint in value:
        if type(constraint) is not V1PhaseAShellBlockCapacityConstraint:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                "capacity constraints require exact V1PhaseAShellBlockCapacityConstraint values",
            )
        if constraint.block_id in seen_blocks:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "capacity block IDs must be unique",
            )
        seen_blocks.add(constraint.block_id)
        if any(index >= shell for index in constraint.shell_indices):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "capacity shell index is outside the registered shell",
            )
        overlap = seen_indices.intersection(constraint.shell_indices)
        if overlap:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "capacity constraints must be disjoint on the shell axis",
            )
        seen_indices.update(constraint.shell_indices)
        capacity = Fraction.from_float(constraint.removal_capacity)
        required = sum(lower[index] for index in constraint.shell_indices)
        if required > capacity:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "lower shell gates already exceed a block capacity",
            )
        normalized.append(constraint)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class V1PhaseABoundedShellGlobalQuadraticProblem:
    """One affine-shell quadratic ``[y,1]^T M [y,1]`` and resource objective."""

    problem_id: str
    loewner_matrix: tuple[tuple[float, ...], ...]
    resource_offset: float
    resource_coefficients: tuple[float, ...]
    q_budget: float
    lower_gate_bounds: tuple[float, ...]
    upper_gate_bounds: tuple[float, ...]
    shell_block_capacity_constraints: tuple[V1PhaseAShellBlockCapacityConstraint, ...] = ()
    loewner_lower_psd_projection_performed: bool = False
    downstream_authority: bool = False
    schema: str = PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PROBLEM_TOKEN:
            raise PermissionError("bounded-shell problems require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        if type(self.problem_id) is not str or not self.problem_id:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                "problem_id must be one nonempty exact string",
            )
        shell = len(self.resource_coefficients)
        if not 1 <= shell <= _MAXIMUM_SHELL:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.SHELL_CAP_EXCEEDED,
                "the certified global solver requires 1 <= B <= 8",
            )
        if (
            type(self.loewner_matrix) is not tuple
            or len(self.loewner_matrix) != shell + 1
            or any(type(row) is not tuple or len(row) != shell + 1 for row in self.loewner_matrix)
            or type(self.resource_coefficients) is not tuple
            or type(self.lower_gate_bounds) is not tuple
            or type(self.upper_gate_bounds) is not tuple
            or len(self.lower_gate_bounds) != shell
            or len(self.upper_gate_bounds) != shell
        ):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "matrix, resource, and gate axes do not align",
            )
        for index, row in enumerate(self.loewner_matrix):
            for other, value in enumerate(row):
                _finite(value, name="loewner entry")
                if value != self.loewner_matrix[other][index]:
                    _refuse(
                        V1PhaseABoundedShellGlobalQuadraticRefusalCode.ASYMMETRIC_MATRIX,
                        "the registered Loewner matrix must be exactly symmetric",
                    )
        _finite(self.resource_offset, name="resource_offset")
        _finite(self.q_budget, name="q_budget")
        for name, axis in (
            ("resource_coefficients", self.resource_coefficients),
            ("lower_gate_bounds", self.lower_gate_bounds),
            ("upper_gate_bounds", self.upper_gate_bounds),
        ):
            for value in axis:
                _finite(value, name=f"{name} entry")
        if any(
            lower < 0.0 or upper > 1.0 or lower > upper
            for lower, upper in zip(
                self.lower_gate_bounds,
                self.upper_gate_bounds,
                strict=True,
            )
        ):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "relaxed irreversible shell gates must obey 0 <= lower <= upper <= 1",
            )
        constraints = _capacity_constraints(
            self.shell_block_capacity_constraints,
            shell=shell,
            lower_gate_bounds=self.lower_gate_bounds,
        )
        object.__setattr__(
            self,
            "shell_block_capacity_constraints",
            constraints,
        )
        if (
            type(self.loewner_lower_psd_projection_performed) is not bool
            or type(self.downstream_authority) is not bool
            or self.loewner_lower_psd_projection_performed is not False
            or self.downstream_authority is not False
            or self.schema != PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_SCHEMA
        ):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.AUTHORITY_VIOLATION,
                "bounded-shell problem crossed its Loewner or authority boundary",
            )

    def _payload(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="problem")

    @property
    def problem_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "bounded-shell problem no longer replays",
            )
        return observed


def issue_v1_phase_a_bounded_shell_global_quadratic_problem(
    *,
    problem_id: str,
    loewner_matrix: Sequence[Sequence[float]],
    resource_offset: float,
    resource_coefficients: Sequence[float],
    q_budget: float,
    lower_gate_bounds: Sequence[float] | None = None,
    upper_gate_bounds: Sequence[float] | None = None,
    shell_block_capacity_constraints: Sequence[V1PhaseAShellBlockCapacityConstraint] | None = None,
) -> V1PhaseABoundedShellGlobalQuadraticProblem:
    coefficients = tuple(resource_coefficients)
    shell = len(coefficients)
    lower = (0.0,) * shell if lower_gate_bounds is None else tuple(lower_gate_bounds)
    upper = (1.0,) * shell if upper_gate_bounds is None else tuple(upper_gate_bounds)
    constraints = (
        ()
        if shell_block_capacity_constraints is None
        else tuple(shell_block_capacity_constraints)
    )
    return V1PhaseABoundedShellGlobalQuadraticProblem(
        problem_id=problem_id,
        loewner_matrix=tuple(tuple(row) for row in loewner_matrix),
        resource_offset=resource_offset,
        resource_coefficients=coefficients,
        q_budget=q_budget,
        lower_gate_bounds=lower,
        upper_gate_bounds=upper,
        shell_block_capacity_constraints=constraints,
        _issuance_token=_PROBLEM_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseABoundedShellGlobalQuadraticSearchDesign:
    maximum_expanded_boxes: int = 262_144
    binary64_adjacency_stopping: bool = True
    outcome_independent_split_rule: str = "widest_gate_then_lowest_index"
    downstream_authority: bool = False
    schema: str = "partition_free_v1_phase_A_bounded_shell_global_search_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("bounded-shell search designs require their issuer")
        _positive_int(self.maximum_expanded_boxes, name="maximum_expanded_boxes")
        if (
            self.maximum_expanded_boxes > 16_777_216
            or self.binary64_adjacency_stopping is not True
            or self.outcome_independent_split_rule != "widest_gate_then_lowest_index"
            or self.downstream_authority is not False
        ):
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.AUTHORITY_VIOLATION,
                "search design crossed its deterministic engineering boundary",
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
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM,
                "bounded-shell search design no longer replays",
            )
        return observed


def register_v1_phase_a_bounded_shell_global_quadratic_search_design(
    *, maximum_expanded_boxes: int = 262_144,
) -> V1PhaseABoundedShellGlobalQuadraticSearchDesign:
    return V1PhaseABoundedShellGlobalQuadraticSearchDesign(
        maximum_expanded_boxes=maximum_expanded_boxes,
        _issuance_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseABoundedShellGlobalQuadraticReceipt:
    problem_hash: str
    design_hash: str
    shell_size: int
    q_min_expanded_box_count: int
    q_min_pending_box_count: int
    q_min_lower_bound: float
    q_min_upper_bound: float
    q_min_witness_shell_gates: tuple[float, ...]
    q_min_binary64_closed: bool
    block_capacity_constraint_count: int
    block_capacity_constraints_hash: str
    disjoint_block_capacity_constraints_enforced: bool
    expanded_box_count: int
    pending_box_count: int
    feasible_witness_present: bool
    certified_infeasible: bool
    resource_lower_bound: float | None
    resource_upper_bound: float | None
    feasible_witness_shell_gates: tuple[float, ...] | None
    binary64_endpoint_closed: bool
    engineering_budget_exhausted: bool
    exact_rational_interval_arithmetic: bool = True
    indefinite_lower_matrix_supported_without_psd_projection: bool = True
    loewner_lower_psd_projection_performed: bool = False
    global_resource_interval_authority: bool = True
    d_stability_authority: bool = False
    c_k_o_commit_terminal_performance_pareto_authority: bool = False
    authority_scope: str = PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_AUTHORITY
    schema: str = PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("bounded-shell receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        if type(self.problem_hash) is not str or not self.problem_hash or type(self.design_hash) is not str or not self.design_hash:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE,
                "receipt identities must be nonempty exact strings",
            )
        _positive_int(self.shell_size, name="shell_size")
        if self.shell_size > _MAXIMUM_SHELL:
            _refuse(
                V1PhaseABoundedShellGlobalQuadraticRefusalCode.SHELL_CAP_EXCEEDED,
                "receipt exceeds the B<=8 shell cap",
            )
        if isinstance(self.expanded_box_count, bool) or not isinstance(self.expanded_box_count, int) or self.expanded_box_count < 0:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "expanded_box_count must be nonnegative")
        if isinstance(self.pending_box_count, bool) or not isinstance(self.pending_box_count, int) or self.pending_box_count < 0:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "pending_box_count must be nonnegative")
        if isinstance(self.q_min_expanded_box_count, bool) or not isinstance(self.q_min_expanded_box_count, int) or self.q_min_expanded_box_count < 0:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "q_min_expanded_box_count must be nonnegative")
        if isinstance(self.q_min_pending_box_count, bool) or not isinstance(self.q_min_pending_box_count, int) or self.q_min_pending_box_count < 0:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "q_min_pending_box_count must be nonnegative")
        _finite(self.q_min_lower_bound, name="q_min_lower_bound")
        _finite(self.q_min_upper_bound, name="q_min_upper_bound")
        if self.q_min_lower_bound > self.q_min_upper_bound or type(self.q_min_witness_shell_gates) is not tuple or len(self.q_min_witness_shell_gates) != self.shell_size:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM, "Q-min interval or witness does not align")
        for value in self.q_min_witness_shell_gates:
            _finite(value, name="q_min witness gate")
        if isinstance(self.block_capacity_constraint_count, bool) or not isinstance(self.block_capacity_constraint_count, int) or self.block_capacity_constraint_count < 0:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "block_capacity_constraint_count must be nonnegative")
        if type(self.block_capacity_constraints_hash) is not str or not self.block_capacity_constraints_hash:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "block_capacity_constraints_hash must be one nonempty exact string")
        flags = (
            self.feasible_witness_present,
            self.certified_infeasible,
            self.binary64_endpoint_closed,
            self.q_min_binary64_closed,
            self.disjoint_block_capacity_constraints_enforced,
            self.engineering_budget_exhausted,
            self.exact_rational_interval_arithmetic,
            self.indefinite_lower_matrix_supported_without_psd_projection,
            self.loewner_lower_psd_projection_performed,
            self.global_resource_interval_authority,
            self.d_stability_authority,
            self.c_k_o_commit_terminal_performance_pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "receipt flags must be exact booleans")
        if self.feasible_witness_present:
            if self.resource_lower_bound is None or self.resource_upper_bound is None or self.feasible_witness_shell_gates is None:
                _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM, "feasible receipt lost its endpoint or witness")
            _finite(self.resource_lower_bound, name="resource_lower_bound")
            _finite(self.resource_upper_bound, name="resource_upper_bound")
            if self.resource_lower_bound > self.resource_upper_bound:
                _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM, "resource interval is reversed")
        elif self.resource_lower_bound is not None or self.feasible_witness_shell_gates is not None:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM, "receipt has a lower endpoint without a feasible witness")
        if self.resource_upper_bound is not None:
            _finite(self.resource_upper_bound, name="resource_upper_bound")
        if (
            (self.certified_infeasible and self.feasible_witness_present)
            or self.certified_infeasible != (self.pending_box_count == 0 and not self.feasible_witness_present)
            or self.binary64_endpoint_closed and not self.feasible_witness_present
            or self.disjoint_block_capacity_constraints_enforced is not True
            or self.engineering_budget_exhausted != (
                (self.pending_box_count > 0 and not self.binary64_endpoint_closed)
                or (self.q_min_pending_box_count > 0 and not self.q_min_binary64_closed)
            )
            or self.exact_rational_interval_arithmetic is not True
            or self.indefinite_lower_matrix_supported_without_psd_projection is not True
            or self.loewner_lower_psd_projection_performed is not False
            or self.global_resource_interval_authority is not True
            or self.d_stability_authority is not False
            or self.c_k_o_commit_terminal_performance_pareto_authority is not False
            or self.authority_scope != PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_AUTHORITY
        ):
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.AUTHORITY_VIOLATION, "bounded-shell receipt crossed its proof or authority boundary")

    def _payload(self) -> dict[str, object]:
        return {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    @property
    def receipt_hash(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_PROBLEM, "bounded-shell receipt no longer replays")
        return observed


@dataclass(frozen=True, slots=True)
class _Box:
    lower: tuple[Fraction, ...]
    upper: tuple[Fraction, ...]


def _quadratic_value(matrix: tuple[tuple[Fraction, ...], ...], gates: tuple[Fraction, ...]) -> Fraction:
    axis = gates + (Fraction(1),)
    return sum(matrix[i][j] * axis[i] * axis[j] for i in range(len(axis)) for j in range(len(axis)))


def _quadratic_interval(matrix: tuple[tuple[Fraction, ...], ...], box: _Box) -> tuple[Fraction, Fraction]:
    lower_axis = box.lower + (Fraction(1),)
    upper_axis = box.upper + (Fraction(1),)
    lower = Fraction(0)
    upper = Fraction(0)
    dimension = len(lower_axis)
    for i in range(dimension):
        for j in range(i, dimension):
            coefficient = matrix[i][j] if i == j else 2 * matrix[i][j]
            product_lower = lower_axis[i] * lower_axis[j]
            product_upper = upper_axis[i] * upper_axis[j]
            if coefficient >= 0:
                lower += coefficient * product_lower
                upper += coefficient * product_upper
            else:
                lower += coefficient * product_upper
                upper += coefficient * product_lower
    return lower, upper


def _resource_value(offset: Fraction, coefficients: tuple[Fraction, ...], gates: tuple[Fraction, ...]) -> Fraction:
    return offset + sum(coefficient * gate for coefficient, gate in zip(coefficients, gates, strict=True))


def _resource_best_point(coefficients: tuple[Fraction, ...], box: _Box) -> tuple[Fraction, ...]:
    return tuple(upper if coefficient >= 0 else lower for coefficient, lower, upper in zip(coefficients, box.lower, box.upper, strict=True))


def _resource_upper(offset: Fraction, coefficients: tuple[Fraction, ...], box: _Box) -> Fraction:
    return _resource_value(offset, coefficients, _resource_best_point(coefficients, box))


_FractionCapacityConstraints = tuple[tuple[tuple[int, ...], Fraction], ...]


def _fraction_capacity_constraints(
    constraints: tuple[V1PhaseAShellBlockCapacityConstraint, ...],
) -> _FractionCapacityConstraints:
    return tuple(
        (
            constraint.shell_indices,
            Fraction.from_float(constraint.removal_capacity),
        )
        for constraint in constraints
    )


def _capacity_feasible_point(
    gates: tuple[Fraction, ...],
    constraints: _FractionCapacityConstraints,
) -> bool:
    return all(
        sum(gates[index] for index in indices) <= capacity
        for indices, capacity in constraints
    )


def _capacity_possible_box(
    box: _Box,
    constraints: _FractionCapacityConstraints,
) -> bool:
    return all(
        sum(box.lower[index] for index in indices) <= capacity
        for indices, capacity in constraints
    )


def _constrained_resource_best_point(
    coefficients: tuple[Fraction, ...],
    box: _Box,
    constraints: _FractionCapacityConstraints,
) -> tuple[Fraction, ...] | None:
    if not _capacity_possible_box(box, constraints):
        return None
    point = list(box.lower)
    constrained: set[int] = set()
    for indices, capacity in constraints:
        constrained.update(indices)
        remaining = capacity - sum(box.lower[index] for index in indices)
        for index in sorted(indices, key=lambda item: (-coefficients[item], item)):
            if coefficients[index] <= 0 or remaining <= 0:
                continue
            step = min(box.upper[index] - box.lower[index], remaining)
            point[index] += step
            remaining -= step
    for index, coefficient in enumerate(coefficients):
        if index in constrained:
            continue
        point[index] = box.upper[index] if coefficient >= 0 else box.lower[index]
    result = tuple(point)
    if not _capacity_feasible_point(result, constraints):
        return None
    return result


def _constrained_resource_upper(
    offset: Fraction,
    coefficients: tuple[Fraction, ...],
    box: _Box,
    constraints: _FractionCapacityConstraints,
) -> Fraction | None:
    best = _constrained_resource_best_point(coefficients, box, constraints)
    if best is None:
        return None
    return _resource_value(offset, coefficients, best)


def _midpoint(box: _Box) -> tuple[Fraction, ...]:
    return tuple((lower + upper) / 2 for lower, upper in zip(box.lower, box.upper, strict=True))


def _global_q_minimum(
    *,
    matrix: tuple[tuple[Fraction, ...], ...],
    root: _Box,
    capacity_constraints: _FractionCapacityConstraints,
    maximum_expanded_boxes: int,
) -> tuple[Fraction, Fraction, tuple[Fraction, ...], int, int]:
    """Enclose the global Q minimum without assuming convexity or PSD."""

    shell = len(root.lower)
    witness = root.lower
    incumbent = _quadratic_value(matrix, witness)

    def consider(point: tuple[Fraction, ...]) -> None:
        nonlocal incumbent, witness
        if not _capacity_feasible_point(point, capacity_constraints):
            return
        value = _quadratic_value(matrix, point)
        if value < incumbent or (value == incumbent and point < witness):
            incumbent = value
            witness = point

    for bits in itertools.product((0, 1), repeat=shell):
        consider(tuple(root.upper[index] if bit else root.lower[index] for index, bit in enumerate(bits)))

    heap: list[tuple[Fraction, int, _Box]] = []
    serial = 0

    def enqueue(box: _Box) -> None:
        nonlocal serial
        if not _capacity_possible_box(box, capacity_constraints):
            return
        lower, _upper = _quadratic_interval(matrix, box)
        if lower >= incumbent:
            return
        consider(_midpoint(box))
        if lower >= incumbent:
            return
        heapq.heappush(heap, (lower, serial, box))
        serial += 1

    enqueue(root)
    expanded = 0
    while heap and expanded < maximum_expanded_boxes:
        lower, _serial, box = heapq.heappop(heap)
        if lower >= incumbent:
            continue
        widths = tuple(high - low for low, high in zip(box.lower, box.upper, strict=True))
        split = max(range(shell), key=lambda index: (widths[index], -index))
        if widths[split] == 0:
            continue
        middle = (box.lower[split] + box.upper[split]) / 2
        left_upper = list(box.upper)
        left_upper[split] = middle
        right_lower = list(box.lower)
        right_lower[split] = middle
        enqueue(_Box(lower=box.lower, upper=tuple(left_upper)))
        enqueue(_Box(lower=tuple(right_lower), upper=box.upper))
        expanded += 1
        if heap:
            lower_float = _float_down(min(incumbent, heap[0][0]))
            upper_float = _float_up(incumbent)
            if upper_float <= math.nextafter(lower_float, math.inf):
                break
    while heap and heap[0][0] >= incumbent:
        heapq.heappop(heap)
    global_lower = min(incumbent, heap[0][0]) if heap else incumbent
    return global_lower, incumbent, witness, expanded, len(heap)


def solve_v1_phase_a_bounded_shell_global_quadratic(
    *,
    problem: V1PhaseABoundedShellGlobalQuadraticProblem,
    design: V1PhaseABoundedShellGlobalQuadraticSearchDesign,
    failure_injection: str | None = None,
) -> V1PhaseABoundedShellGlobalQuadraticReceipt:
    """Return a certified global resource interval; never infer D stability."""

    if type(problem) is not V1PhaseABoundedShellGlobalQuadraticProblem or type(design) is not V1PhaseABoundedShellGlobalQuadraticSearchDesign:
        _refuse(V1PhaseABoundedShellGlobalQuadraticRefusalCode.INVALID_TYPE, "global solve requires exact problem and design types")
    if failure_injection not in (None, "after_first_expansion"):
        raise ValueError("unknown bounded-shell failure injection")
    problem_hash = problem.problem_hash
    design_hash = design.design_hash
    matrix = tuple(tuple(Fraction.from_float(value) for value in row) for row in problem.loewner_matrix)
    coefficients = _fraction_axis(problem.resource_coefficients, name="resource_coefficients")
    offset = Fraction.from_float(problem.resource_offset)
    budget = Fraction.from_float(problem.q_budget)
    capacity_constraints = _fraction_capacity_constraints(
        problem.shell_block_capacity_constraints,
    )
    capacity_hash = _payload_hash(
        {
            "schema": _CAPACITY_CONSTRAINT_SCHEMA,
            "constraints": problem.shell_block_capacity_constraints,
        },
        role="capacity-constraints",
    )
    root = _Box(
        lower=_fraction_axis(problem.lower_gate_bounds, name="lower_gate_bounds"),
        upper=_fraction_axis(problem.upper_gate_bounds, name="upper_gate_bounds"),
    )

    (
        q_min_lower_exact,
        q_min_upper_exact,
        q_min_witness,
        q_min_expanded,
        q_min_pending,
    ) = _global_q_minimum(
        matrix=matrix,
        root=root,
        capacity_constraints=capacity_constraints,
        maximum_expanded_boxes=design.maximum_expanded_boxes,
    )
    q_min_lower = _float_down(q_min_lower_exact)
    q_min_upper = _float_up(q_min_upper_exact)
    q_min_closed = q_min_upper <= math.nextafter(q_min_lower, math.inf)

    incumbent: Fraction | None = None
    witness: tuple[Fraction, ...] | None = None

    def consider(point: tuple[Fraction, ...]) -> None:
        nonlocal incumbent, witness
        if not _capacity_feasible_point(point, capacity_constraints):
            return
        if _quadratic_value(matrix, point) <= budget:
            value = _resource_value(offset, coefficients, point)
            if incumbent is None or value > incumbent or (value == incumbent and point < witness):
                incumbent = value
                witness = point

    shell = len(coefficients)
    for bits in itertools.product((0, 1), repeat=shell):
        consider(tuple(root.upper[index] if bit else root.lower[index] for index, bit in enumerate(bits)))
    consider(_midpoint(root))

    heap: list[tuple[Fraction, int, _Box]] = []
    serial = 0

    def enqueue(box: _Box) -> None:
        nonlocal serial
        if not _capacity_possible_box(box, capacity_constraints):
            return
        q_lower, q_upper = _quadratic_interval(matrix, box)
        if q_lower > budget:
            return
        best = _constrained_resource_best_point(
            coefficients,
            box,
            capacity_constraints,
        )
        if best is None:
            return
        if q_upper <= budget:
            consider(best)
            return
        consider(best)
        consider(_midpoint(box))
        upper = _constrained_resource_upper(
            offset,
            coefficients,
            box,
            capacity_constraints,
        )
        if upper is None:
            return
        if incumbent is not None and upper <= incumbent:
            return
        heapq.heappush(heap, (-upper, serial, box))
        serial += 1

    if q_min_lower_exact <= budget:
        enqueue(root)
    expanded = 0
    while heap and expanded < design.maximum_expanded_boxes:
        negative_upper, _serial, box = heapq.heappop(heap)
        upper = -negative_upper
        if incumbent is not None and upper <= incumbent:
            continue
        widths = tuple(high - low for low, high in zip(box.lower, box.upper, strict=True))
        split = max(range(shell), key=lambda index: (widths[index], -index))
        if widths[split] == 0:
            continue
        middle = (box.lower[split] + box.upper[split]) / 2
        left_upper = list(box.upper)
        left_upper[split] = middle
        right_lower = list(box.lower)
        right_lower[split] = middle
        enqueue(_Box(lower=box.lower, upper=tuple(left_upper)))
        enqueue(_Box(lower=tuple(right_lower), upper=box.upper))
        expanded += 1
        if failure_injection == "after_first_expansion" and expanded == 1:
            raise RuntimeError("injected bounded-shell failure after first expansion")
        if incumbent is not None and heap:
            candidate_upper = -heap[0][0]
            lower_float = _float_down(incumbent)
            upper_float = _float_up(max(incumbent, candidate_upper))
            if upper_float <= math.nextafter(lower_float, math.inf):
                break

    while heap and incumbent is not None and -heap[0][0] <= incumbent:
        heapq.heappop(heap)
    pending = len(heap)
    certified_infeasible = incumbent is None and pending == 0
    if incumbent is None:
        resource_lower = None
        witness_float = None
        resource_upper = _float_up(-heap[0][0]) if heap else None
        closed = False
    else:
        resource_lower = _float_down(incumbent)
        witness_float = tuple(float(value) for value in witness) if witness is not None else None
        exact_upper = max(incumbent, -heap[0][0]) if heap else incumbent
        resource_upper = _float_up(exact_upper)
        closed = resource_upper <= math.nextafter(resource_lower, math.inf)
    exhausted = (pending > 0 and not closed) or (
        q_min_pending > 0 and not q_min_closed
    )
    return V1PhaseABoundedShellGlobalQuadraticReceipt(
        problem_hash=problem_hash,
        design_hash=design_hash,
        shell_size=shell,
        q_min_expanded_box_count=q_min_expanded,
        q_min_pending_box_count=q_min_pending,
        q_min_lower_bound=q_min_lower,
        q_min_upper_bound=q_min_upper,
        q_min_witness_shell_gates=tuple(float(value) for value in q_min_witness),
        q_min_binary64_closed=q_min_closed,
        block_capacity_constraint_count=len(problem.shell_block_capacity_constraints),
        block_capacity_constraints_hash=capacity_hash,
        disjoint_block_capacity_constraints_enforced=True,
        expanded_box_count=expanded,
        pending_box_count=pending,
        feasible_witness_present=incumbent is not None,
        certified_infeasible=certified_infeasible,
        resource_lower_bound=resource_lower,
        resource_upper_bound=resource_upper,
        feasible_witness_shell_gates=witness_float,
        binary64_endpoint_closed=closed,
        engineering_budget_exhausted=exhausted,
        _issuance_token=_RECEIPT_TOKEN,
    )


__all__ = (
    "PHASE_A_BOUNDED_SHELL_GLOBAL_QUADRATIC_AUTHORITY",
    "V1PhaseAShellBlockCapacityConstraint",
    "V1PhaseABoundedShellGlobalQuadraticProblem",
    "V1PhaseABoundedShellGlobalQuadraticReceipt",
    "V1PhaseABoundedShellGlobalQuadraticRefusal",
    "V1PhaseABoundedShellGlobalQuadraticRefusalCode",
    "V1PhaseABoundedShellGlobalQuadraticSearchDesign",
    "issue_v1_phase_a_bounded_shell_global_quadratic_problem",
    "register_v1_phase_a_bounded_shell_global_quadratic_search_design",
    "solve_v1_phase_a_bounded_shell_global_quadratic",
)
