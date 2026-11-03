"""Identity-bound CUDA qualification issuer for the signed-work D witness.

The canonical D decision is the image of two source-bound optimizer phases,
not the identity of one private relaxed face.  Phase A exposes registered
resource cells from literal-J/Q geometry.  Phase B maps those cells through
native-fiber signed-F optimization and exact packed hard-action rounding.  A
decision is stable when every registered vertex of the four-owner uncertainty
product has the same

``resource cell x byte-deduplicated physical action``

image and the same preferred physical decision.  Q-branch, solver-face, and
rounding-alias identities are retained as diagnostics but are deliberately not
part of that equality.

This module is the live-state CUDA *qualification* seam.  It borrows the exact
Phase-A theorem-state lease and each Phase-B signed-work-decision lease, checks
literal-J/signed-F/FD/solver ownership and final-dual custody using tensor
metadata only, and never materializes a route-scale tensor on the host.  The
result remains nonserializable and process-identity-bound.  Until the real
H200 completeness suite promotes a later authority-bearing issuer, every C,
K, O, commit, terminal, performance, and Pareto authority flag here is false.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from itertools import product
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAExecutionBundle,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    V1SignedWorkLeastWorkCUDAStateHandle,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkPackedActionDedupIdentity,
    V1SignedWorkPreCPackedActionFamily,
)
from .modern_v1_signed_work_resource_frontier_cuda import (
    V1SignedWorkResourceFrontierCUDAResult,
    V1SignedWorkResourceFrontierCUDATheoremStateHandle,
)


SIGNED_WORK_D_W_CUDA_RESOURCE_CELL_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_cuda_resource_cell_v1"
)
SIGNED_WORK_D_W_CUDA_PHASE_B_BINDING_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_cuda_phase_B_state_binding_v1"
)
SIGNED_WORK_D_W_CUDA_REGISTERED_CELL_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_cuda_registered_uncertainty_cell_v1"
)
SIGNED_WORK_D_W_CUDA_CUSTODY_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_cuda_tensor_custody_v1"
)
SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_cuda_decision_qualification_v1"
)
SIGNED_WORK_D_W_CUDA_AUTHORITY_SCOPE: Final = (
    "identity_bound_live_cuda_D_W_completeness_qualification_only_no_generic_"
    "or_production_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
SIGNED_WORK_D_W_CUDA_DECISION_RULE: Final = (
    "complete_registered_signed_F_x_literal_J_x_FD_x_solver_cells_then_"
    "invariant_resource_cell_family_and_resource_cell_times_exact_packed_"
    "physical_action_family_and_preferred_physical_decision_with_branch_and_"
    "private_face_aliases_diagnostic_only_v1"
)

_PREFIX: Final = "partition-free-v1-signed-work-D-W-cuda-witness-"
_CELL_ISSUER_TOKEN: Final = object()
_QUALIFICATION_ISSUER_TOKEN: Final = object()
_EXECUTION_TYPES: Final = {
    V1SignedWorkLeastWorkCUDAExecutionBundle,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
}


class V1SignedWorkCUDAWitnessRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    CUDA_REQUIRED = "cuda_required"
    MISSING_SOLVER_DUAL_CUSTODY = "missing_solver_dual_custody"
    INCOMPLETE_UNCERTAINTY_PRODUCT = "incomplete_uncertainty_product"
    INCOMPLETE_RESOURCE_CELL_REGISTRATION = "incomplete_resource_cell_registration"
    INCOMPLETE_PHASE_B_IMAGE = "incomplete_phase_B_image"
    PACKED_ACTION_PROVENANCE_MISMATCH = "packed_action_provenance_mismatch"
    DECISION_UNSTABLE = "decision_unstable"
    BOUNDEDNESS_VIOLATION = "boundedness_violation"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkCUDAWitnessRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkCUDAWitnessRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkCUDAWitnessRefusalCode:
            raise TypeError("CUDA witness refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkCUDAWitnessRefusalCode, message: str) -> NoReturn:
    raise V1SignedWorkCUDAWitnessRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty and unique",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be finite",
        )
    return result


def _physical_identity_hash(value: V1SignedWorkPackedActionDedupIdentity) -> str:
    return content_hash(tuple(value), prefix=_PREFIX + "physical-action-")


def _device_name(value: object) -> str:
    result = str(value)
    if not result:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            "tensor custody has no device identity",
        )
    return result


def _tensor_metadata(value: object, *, name: str) -> tuple[object, ...]:
    """Read identity/shape metadata only; never read or transfer tensor data."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    ndim = getattr(value, "ndim", None)
    if shape is None or dtype is None or device is None or ndim is None:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            f"{name} is not one retained tensor",
        )
    contiguous = getattr(value, "is_contiguous", None)
    if not callable(contiguous) or not bool(contiguous()):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            f"{name} is not contiguous retained custody",
        )
    if bool(getattr(value, "requires_grad", True)):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            f"{name} unexpectedly requires gradients",
        )
    version = getattr(value, "_version", None)
    data_ptr = getattr(value, "data_ptr", None)
    if version is None or not callable(data_ptr):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            f"{name} lacks identity-bound tensor metadata",
        )
    return (
        name,
        id(value),
        int(version),
        int(data_ptr()),
        tuple(int(item) for item in tuple(shape)),
        str(dtype),
        _device_name(device),
    )


def _expected_q_branches(coordinate: object) -> tuple[V1RelaxedKKTBranch, ...]:
    witnessed = getattr(coordinate, "witnessed_feasible_branches", None)
    if type(witnessed) is not tuple or not witnessed:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "Phase-A coordinate lost its witnessed Q-branch family",
        )
    try:
        return tuple(V1RelaxedKKTBranch(item.value) for item in witnessed)
    except (AttributeError, ValueError) as error:
        raise V1SignedWorkCUDAWitnessRefusal(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "Phase-A coordinate contains a noncanonical Q branch",
        ) from error


def _execution_state_binding_hashes(execution: object) -> tuple[str, ...]:
    if type(execution) is V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
        return tuple(execution.result.solver_state_binding_hashes)
    if type(execution) is V1SignedWorkLeastWorkCUDAExecutionBundle:
        return tuple(value.receipt_hash for value in execution.result.branch_receipts)
    raise TypeError("unknown Phase-B execution type")


@dataclass(frozen=True, slots=True)
class V1SignedWorkCUDAResourceCellBinding:
    """Bind one Phase-A coordinate to one pre-registered resource cell.

    The continuous resource value is provenance.  Decision equality uses the
    cell identity, which is determined only by the frozen partition, ordinal,
    and closed bounds.
    """

    partition_registration_hash: str
    resource_coordinate_hash: str
    coordinate_ordinal: int
    coordinate_resource_value: float
    cell_ordinal: int
    cell_lower_bound: float
    cell_upper_bound: float
    schema: str = SIGNED_WORK_D_W_CUDA_RESOURCE_CELL_SCHEMA

    def __post_init__(self) -> None:
        for name in ("partition_registration_hash", "resource_coordinate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "coordinate_ordinal",
            _positive_int(self.coordinate_ordinal, name="coordinate ordinal"),
        )
        object.__setattr__(
            self,
            "cell_ordinal",
            _positive_int(self.cell_ordinal, name="resource-cell ordinal"),
        )
        value = _finite(self.coordinate_resource_value, name="resource coordinate")
        lower = _finite(self.cell_lower_bound, name="resource-cell lower bound")
        upper = _finite(self.cell_upper_bound, name="resource-cell upper bound")
        if lower > upper or value < lower or value > upper:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_RESOURCE_CELL_REGISTRATION,
                "resource coordinate is outside its closed registered cell",
            )
        object.__setattr__(self, "coordinate_resource_value", value)
        object.__setattr__(self, "cell_lower_bound", lower)
        object.__setattr__(self, "cell_upper_bound", upper)
        if self.schema != SIGNED_WORK_D_W_CUDA_RESOURCE_CELL_SCHEMA:
            raise ValueError("unsupported CUDA resource-cell semantics")

    @property
    def resource_cell_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "partition_registration_hash": self.partition_registration_hash,
                "cell_ordinal": self.cell_ordinal,
                "cell_lower_bound": self.cell_lower_bound,
                "cell_upper_bound": self.cell_upper_bound,
                "closed_interval": True,
            },
            prefix=_PREFIX + "resource-cell-",
        )

    @property
    def binding_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "resource-cell-binding-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkCUDAPhaseBStateBinding:
    """One coordinate-specific Phase-B problem/execution/state binding."""

    resource_coordinate_hash: str
    coordinate_ordinal: int
    coordinate_resource_value: float
    problem: V1SignedWorkLeastWorkCUDAProblem = field(repr=False, compare=False)
    execution: object = field(repr=False, compare=False)
    schema: str = SIGNED_WORK_D_W_CUDA_PHASE_B_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_coordinate_hash",
            _identity(self.resource_coordinate_hash, name="resource coordinate hash"),
        )
        object.__setattr__(
            self,
            "coordinate_ordinal",
            _positive_int(self.coordinate_ordinal, name="coordinate ordinal"),
        )
        resource = _finite(
            self.coordinate_resource_value,
            name="coordinate resource value",
        )
        object.__setattr__(self, "coordinate_resource_value", resource)
        if type(self.problem) is not V1SignedWorkLeastWorkCUDAProblem:
            raise TypeError("Phase-B binding requires an exact least-work problem")
        if type(self.execution) not in _EXECUTION_TYPES:
            raise TypeError("Phase-B binding requires an exact retained execution")
        self.problem.assert_integrity()
        execution = self.execution
        handle = execution.state_handle
        if type(handle) is not V1SignedWorkLeastWorkCUDAStateHandle:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.MISSING_SOLVER_DUAL_CUSTODY,
                "Phase-B execution retained no theorem-state handle",
            )
        if (
            execution.result.problem_hash != self.problem.problem_hash
            or handle.problem_hash != self.problem.problem_hash
            or handle.policy_hash != execution.result.policy_hash
            or tuple(handle.branch_receipt_hashes)
            != _execution_state_binding_hashes(execution)
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "Phase-B execution, result, and theorem state differ",
            )
        tolerance = 64.0 * math.ulp(max(1.0, abs(resource)))
        if abs(self.problem.resource_floor - resource) > tolerance:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "Phase-B problem does not use the registered resource coordinate",
            )
        if self.schema != SIGNED_WORK_D_W_CUDA_PHASE_B_BINDING_SCHEMA:
            raise ValueError("unsupported CUDA Phase-B binding semantics")

    @property
    def state_handle(self) -> V1SignedWorkLeastWorkCUDAStateHandle:
        handle = self.execution.state_handle
        if type(handle) is not V1SignedWorkLeastWorkCUDAStateHandle:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.MISSING_SOLVER_DUAL_CUSTODY,
                "Phase-B theorem state is no longer available",
            )
        return handle

    @property
    def q_branches(self) -> tuple[V1RelaxedKKTBranch, ...]:
        return self.state_handle.branch_order

    @property
    def signed_f_operator_receipt_hash(self) -> str:
        return self.problem.signed_work_operator.receipt_hash

    @property
    def execution_hash(self) -> str:
        return self.execution.execution_hash

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "resource_coordinate_hash": self.resource_coordinate_hash,
                "coordinate_ordinal": self.coordinate_ordinal,
                "coordinate_resource_value": self.coordinate_resource_value,
                "problem_hash": self.problem.problem_hash,
                "execution_hash": self.execution_hash,
                "state_identity_hash": self.state_handle.identity_hash,
                "q_branches": self.q_branches,
                "signed_f_operator_receipt_hash": (self.signed_f_operator_receipt_hash),
            },
            prefix=_PREFIX + "phase-B-binding-",
        )

    @property
    def production_cuda_execution(self) -> bool:
        return bool(self.execution.production_cuda_execution)

    def assert_live_identity(self) -> None:
        self.problem.assert_integrity()
        handle = self.state_handle
        if (
            handle.identity_hash != self.execution.result.state_identity_hash
            or handle.problem_hash != self.problem.problem_hash
            or tuple(handle.branch_receipt_hashes)
            != _execution_state_binding_hashes(self.execution)
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "Phase-B state binding became stale",
            )
        # This property runs the handle's live tensor-version guard.
        _ = handle.route_count

    def __reduce__(self) -> NoReturn:
        raise TypeError("Phase-B CUDA state bindings are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("Phase-B CUDA state bindings are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("Phase-B CUDA state bindings cannot be copied")

    def __deepcopy__(self, _memo: dict[int, Any]) -> NoReturn:
        raise TypeError("Phase-B CUDA state bindings cannot be copied")


@dataclass(frozen=True, slots=True)
class V1SignedWorkCUDATensorCustodyReceipt:
    registered_cell_hash: str
    phase_a_handle_hash: str
    phase_b_state_identity_hashes: tuple[str, ...]
    devices: tuple[str, ...]
    route_count: int
    response_rank: int
    phase_a_branch_modes: tuple[tuple[str, str], ...]
    tensor_metadata_commitment_hash: str
    phase_a_pdhg_dual_or_exact_cap_witness_custody_complete: bool = True
    phase_b_solver_dual_custody_complete: bool = True
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    schema: str = SIGNED_WORK_D_W_CUDA_CUSTODY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registered_cell_hash",
            "phase_a_handle_hash",
            "tensor_metadata_commitment_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "phase_b_state_identity_hashes",
            _identity_axis(
                self.phase_b_state_identity_hashes,
                name="Phase-B state identities",
            ),
        )
        object.__setattr__(
            self, "devices", _identity_axis(self.devices, name="devices")
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        if isinstance(self.response_rank, bool) or not isinstance(
            self.response_rank, Integral
        ):
            raise TypeError("response rank must be one nonnegative integer")
        if int(self.response_rank) < 0:
            raise ValueError("response rank must be nonnegative")
        object.__setattr__(self, "response_rank", int(self.response_rank))
        if (
            type(self.phase_a_pdhg_dual_or_exact_cap_witness_custody_complete)
            is not bool
            or not self.phase_a_pdhg_dual_or_exact_cap_witness_custody_complete
            or type(self.phase_b_solver_dual_custody_complete) is not bool
            or not self.phase_b_solver_dual_custody_complete
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
                "CUDA witness custody crossed its bounded device-native seam",
            )
        if self.schema != SIGNED_WORK_D_W_CUDA_CUSTODY_SCHEMA:
            raise ValueError("unsupported CUDA custody semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "custody-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkCUDARegisteredUncertaintyCell:
    """One source-bound optimizer image at one four-owner uncertainty cell."""

    uncertainty_set_hash: str
    grid_key: tuple[str, str, str, str]
    phase_a_result: V1SignedWorkResourceFrontierCUDAResult = field(
        repr=False,
        compare=False,
    )
    phase_a_state_handle: V1SignedWorkResourceFrontierCUDATheoremStateHandle = field(
        repr=False,
        compare=False,
    )
    phase_b_state_bindings: tuple[V1SignedWorkCUDAPhaseBStateBinding, ...] = field(
        repr=False,
        compare=False,
    )
    resource_cell_bindings: tuple[V1SignedWorkCUDAResourceCellBinding, ...]
    packed_action_family: V1SignedWorkPreCPackedActionFamily = field(
        repr=False,
        compare=False,
    )
    preferred_physical_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    registered_d_w_prefix_hash: str
    signed_f_uncertainty_set_hash: str
    literal_j_uncertainty_set_hash: str
    fd_residual_uncertainty_set_hash: str
    solver_uncertainty_set_hash: str
    _issuer_token: InitVar[object]
    schema: str = SIGNED_WORK_D_W_CUDA_REGISTERED_CELL_SCHEMA
    resource_cell_family: tuple[str, ...] = field(init=False)
    physical_action_family: tuple[tuple[str, str], ...] = field(init=False)
    preferred_physical_decision_family: tuple[tuple[str, str], ...] = field(init=False)
    phase_a_face_family_hash: str = field(init=False)
    phase_b_face_family_hash: str = field(init=False)
    registered_cell_hash: str = field(init=False)
    registered_d_coordinate_hash: str = field(init=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _CELL_ISSUER_TOKEN:
            raise TypeError("registered CUDA uncertainty cells are issuer-only")
        object.__setattr__(
            self,
            "uncertainty_set_hash",
            _identity(self.uncertainty_set_hash, name="uncertainty-set hash"),
        )
        grid = tuple(
            _identity(value, name="uncertainty realization") for value in self.grid_key
        )
        if len(grid) != 4:
            raise ValueError(
                "one CUDA uncertainty cell requires four owner coordinates"
            )
        object.__setattr__(self, "grid_key", grid)
        for name in (
            "registered_d_w_prefix_hash",
            "signed_f_uncertainty_set_hash",
            "literal_j_uncertainty_set_hash",
            "fd_residual_uncertainty_set_hash",
            "solver_uncertainty_set_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.phase_a_result) is not V1SignedWorkResourceFrontierCUDAResult:
            raise TypeError("registered cell requires an exact Phase-A result")
        if type(self.phase_a_state_handle) is not (
            V1SignedWorkResourceFrontierCUDATheoremStateHandle
        ):
            raise TypeError("registered cell requires an exact Phase-A state handle")
        self.phase_a_state_handle.assert_integrity(result=self.phase_a_result)
        phase_b = tuple(self.phase_b_state_bindings)
        if not phase_b or any(
            type(value) is not V1SignedWorkCUDAPhaseBStateBinding for value in phase_b
        ):
            raise TypeError("registered cell requires exact Phase-B state bindings")
        if len({value.binding_hash for value in phase_b}) != len(phase_b):
            raise ValueError("registered cell repeats a Phase-B state binding")
        object.__setattr__(self, "phase_b_state_bindings", phase_b)
        cells = tuple(self.resource_cell_bindings)
        if not cells or any(
            type(value) is not V1SignedWorkCUDAResourceCellBinding for value in cells
        ):
            raise TypeError("registered cell requires exact resource-cell bindings")
        object.__setattr__(self, "resource_cell_bindings", cells)
        if type(self.packed_action_family) is not V1SignedWorkPreCPackedActionFamily:
            raise TypeError("registered cell requires an exact packed-action family")
        family = self.packed_action_family
        handle = self.phase_a_state_handle
        if (
            family.route_axis_hash != handle.route_axis_hash
            or family.route_count != handle.route_count
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "packed actions and Phase-A theorem state use different route axes",
            )

        coordinates = tuple(self.phase_a_result.reference_result.coordinates)
        coordinate_hashes = tuple(value.coordinate_hash for value in coordinates)
        cell_hashes = tuple(value.resource_coordinate_hash for value in cells)
        if (
            len(coordinates) != len(cells)
            or coordinate_hashes != cell_hashes
            or tuple(value.coordinate_ordinal for value in cells)
            != tuple(value.ordinal for value in coordinates)
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_RESOURCE_CELL_REGISTRATION,
                "resource-cell bindings do not cover Phase A exactly in coordinate order",
            )
        for coordinate, cell in zip(coordinates, cells):
            tolerance = 64.0 * math.ulp(max(1.0, abs(coordinate.total_resource_floor)))
            if (
                abs(coordinate.total_resource_floor - cell.coordinate_resource_value)
                > tolerance
            ):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                    "resource-cell numeric provenance differs from Phase A",
                )
        partitions = {value.partition_registration_hash for value in cells}
        if len(partitions) != 1:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_RESOURCE_CELL_REGISTRATION,
                "one optimizer image crossed resource-cell partitions",
            )
        definitions: dict[int, tuple[float, float, str]] = {}
        for cell in cells:
            definition = (
                cell.cell_lower_bound,
                cell.cell_upper_bound,
                cell.resource_cell_hash,
            )
            existing = definitions.get(cell.cell_ordinal)
            if existing is not None and existing != definition:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_RESOURCE_CELL_REGISTRATION,
                    "one resource-cell ordinal has conflicting registered bounds",
                )
            definitions[cell.cell_ordinal] = definition
        ordered_definitions = tuple(definitions[index] for index in sorted(definitions))
        for previous, current in zip(ordered_definitions, ordered_definitions[1:]):
            if previous[1] >= current[0]:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_RESOURCE_CELL_REGISTRATION,
                    "registered closed resource cells overlap",
                )

        phase_b_by_coordinate: dict[
            str,
            list[V1SignedWorkCUDAPhaseBStateBinding],
        ] = {value.coordinate_hash: [] for value in coordinates}
        registered_d_coordinates = {
            value.problem.registered_d_coordinate_hash for value in phase_b
        }
        if len(registered_d_coordinates) != 1:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "Phase-B state bindings crossed registered D coordinates",
            )
        registered_d_coordinate_hash = next(iter(registered_d_coordinates))
        object.__setattr__(
            self,
            "registered_d_coordinate_hash",
            registered_d_coordinate_hash,
        )
        for binding in phase_b:
            if binding.resource_coordinate_hash not in phase_b_by_coordinate:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_PHASE_B_IMAGE,
                    "Phase B contains a coordinate not exposed by Phase A",
                )
            if binding.coordinate_ordinal > len(coordinates):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                    "Phase-B coordinate ordinal is outside Phase A",
                )
            coordinate = coordinates[binding.coordinate_ordinal - 1]
            resource_tolerance = 64.0 * math.ulp(
                max(1.0, abs(coordinate.total_resource_floor))
            )
            if (
                coordinate.coordinate_hash != binding.resource_coordinate_hash
                or abs(
                    coordinate.total_resource_floor - binding.coordinate_resource_value
                )
                > resource_tolerance
                or binding.problem.source_problem.route_axis_hash
                != handle.route_axis_hash
                or binding.problem.source_problem.route_count != handle.route_count
                or binding.problem.source_problem.rank != handle.response_rank
                or binding.problem.source_registration_hash
                != self.registered_d_w_prefix_hash
            ):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                    "Phase-B state binding left its registered D_W chart",
                )
            phase_b_by_coordinate[binding.resource_coordinate_hash].append(binding)
        crosswalk_by_coordinate: dict[str, list[object]] = {
            value.coordinate_hash: [] for value in coordinates
        }
        for crosswalk in family.candidate_crosswalks:
            if crosswalk.resource_coordinate_hash not in crosswalk_by_coordinate:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_PHASE_B_IMAGE,
                    "packed Phase-B alias was not exposed by Phase A",
                )
            crosswalk_by_coordinate[crosswalk.resource_coordinate_hash].append(
                crosswalk
            )
        for coordinate in coordinates:
            key = coordinate.coordinate_hash
            bindings = phase_b_by_coordinate[key]
            aliases = crosswalk_by_coordinate[key]
            expected = _expected_q_branches(coordinate)
            bound_axis = tuple(
                branch for value in bindings for branch in value.q_branches
            )
            bound = tuple(
                branch
                for branch in (
                    V1RelaxedKKTBranch.SPECTRAL,
                    V1RelaxedKKTBranch.DIAGONAL_L1,
                )
                if any(branch in value.q_branches for value in bindings)
            )
            aliased = tuple(
                branch
                for branch in (
                    V1RelaxedKKTBranch.SPECTRAL,
                    V1RelaxedKKTBranch.DIAGONAL_L1,
                )
                if any(getattr(value, "q_branch") is branch for value in aliases)
            )
            if (
                not bindings
                or not aliases
                or len(bound_axis) != len(set(bound_axis))
                or bound != expected
                or aliased != expected
            ):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_PHASE_B_IMAGE,
                    "Phase-B state/action branch image is incomplete for a resource coordinate",
                )

        preferred = tuple(self.preferred_physical_dedup_identity)
        canonical_by_identity = {
            tuple(value.dedup_identity): value for value in family.canonical_actions
        }
        if preferred not in canonical_by_identity:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.PACKED_ACTION_PROVENANCE_MISMATCH,
                "preferred physical action is absent after exact packed-byte dedup",
            )
        object.__setattr__(self, "preferred_physical_dedup_identity", preferred)
        cell_by_coordinate = {
            value.resource_coordinate_hash: value.resource_cell_hash for value in cells
        }
        physical_pairs: set[tuple[str, str]] = set()
        preferred_pairs: set[tuple[str, str]] = set()
        preferred_hash = _physical_identity_hash(preferred)
        for action in family.canonical_actions:
            if not action.actual_packed_byte_equality_verified:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.PACKED_ACTION_PROVENANCE_MISMATCH,
                    "physical action did not verify actual packed-byte equality",
                )
            action_hash = _physical_identity_hash(action.dedup_identity)
            action_cells = {
                cell_by_coordinate[alias.resource_coordinate_hash]
                for alias in action.aliases
            }
            pairs = {(cell_hash, action_hash) for cell_hash in action_cells}
            physical_pairs.update(pairs)
            if action_hash == preferred_hash:
                preferred_pairs.update(pairs)
        if not physical_pairs or not preferred_pairs:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_PHASE_B_IMAGE,
                "Phase B has no complete preferred packed physical decision",
            )
        resource_family = tuple(sorted({value.resource_cell_hash for value in cells}))
        physical_family = tuple(sorted(physical_pairs))
        preferred_family = tuple(sorted(preferred_pairs))
        phase_a_faces = content_hash(
            {
                "branch_receipt_hashes": handle.branch_receipt_hashes,
                "coordinate_branch_membership": tuple(
                    (
                        coordinate.coordinate_hash,
                        tuple(
                            item.value
                            for item in coordinate.witnessed_feasible_branches
                        ),
                    )
                    for coordinate in coordinates
                ),
            },
            prefix=_PREFIX + "phase-A-faces-",
        )
        phase_b_faces = content_hash(
            {
                "state_bindings": tuple(value.binding_hash for value in phase_b),
                "crosswalk_aliases": tuple(
                    value.crosswalk_hash for value in family.candidate_crosswalks
                ),
                "canonical_aliases": tuple(
                    (
                        _physical_identity_hash(value.dedup_identity),
                        tuple(alias.crosswalk_hash for alias in value.aliases),
                    )
                    for value in family.canonical_actions
                ),
            },
            prefix=_PREFIX + "phase-B-faces-",
        )
        object.__setattr__(self, "resource_cell_family", resource_family)
        object.__setattr__(self, "physical_action_family", physical_family)
        object.__setattr__(
            self,
            "preferred_physical_decision_family",
            preferred_family,
        )
        object.__setattr__(self, "phase_a_face_family_hash", phase_a_faces)
        object.__setattr__(self, "phase_b_face_family_hash", phase_b_faces)
        object.__setattr__(
            self,
            "registered_cell_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "uncertainty_set_hash": self.uncertainty_set_hash,
                    "grid_key": grid,
                    "phase_a_result_hash": self.phase_a_result.result_hash,
                    "phase_a_handle_hash": handle.handle_hash,
                    "phase_b_binding_hashes": tuple(
                        value.binding_hash for value in phase_b
                    ),
                    "resource_cell_binding_hashes": tuple(
                        value.binding_hash for value in cells
                    ),
                    "packed_action_family_hash": family.family_hash,
                    "preferred_physical_action_hash": preferred_hash,
                    "resource_cell_family": resource_family,
                    "physical_action_family": physical_family,
                    "preferred_physical_decision_family": preferred_family,
                    "phase_a_face_family_hash": phase_a_faces,
                    "phase_b_face_family_hash": phase_b_faces,
                    "owner_uncertainty_set_hashes": (
                        self.signed_f_uncertainty_set_hash,
                        self.literal_j_uncertainty_set_hash,
                        self.fd_residual_uncertainty_set_hash,
                        self.solver_uncertainty_set_hash,
                    ),
                    "registered_d_w_prefix_hash": self.registered_d_w_prefix_hash,
                    "registered_d_coordinate_hash": registered_d_coordinate_hash,
                },
                prefix=_PREFIX + "registered-cell-",
            ),
        )
        if self.schema != SIGNED_WORK_D_W_CUDA_REGISTERED_CELL_SCHEMA:
            raise ValueError("unsupported registered CUDA cell semantics")

    @property
    def parent_mask_hash(self) -> str:
        return self.packed_action_family.parent_mask_hash

    @property
    def route_axis_hash(self) -> str:
        return self.packed_action_family.route_axis_hash

    @property
    def route_count(self) -> int:
        return self.packed_action_family.route_count

    @property
    def literal_j_response_geometry_hash(self) -> str:
        return self.phase_a_state_handle.response_geometry_hash

    @property
    def signed_f_operator_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.signed_f_operator_receipt_hash
            for value in self.phase_b_state_bindings
        )

    @property
    def production_cuda_sources(self) -> bool:
        phase_a_cuda = (
            self.phase_a_result.production_cuda_execution
            or self.phase_a_result.engineering_h200_qualification_execution
        )
        return phase_a_cuda and all(
            value.production_cuda_execution for value in self.phase_b_state_bindings
        )

    def assert_live_identity(self) -> None:
        self.phase_a_state_handle.assert_integrity(result=self.phase_a_result)
        for binding in self.phase_b_state_bindings:
            binding.assert_live_identity()

    def verify_tensor_custody(
        self,
        *,
        require_cuda: bool,
    ) -> V1SignedWorkCUDATensorCustodyReceipt:
        self.assert_live_identity()
        if require_cuda and not self.production_cuda_sources:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.CUDA_REQUIRED,
                "production qualification requires CUDA/H200 source executions",
            )
        metadata: list[tuple[object, ...]] = []
        devices: set[str] = set()
        branch_modes: list[tuple[str, str]] = []
        with self.phase_a_state_handle.acquire_witness_lease(
            result=self.phase_a_result
        ) as phase_a_lease:
            source = phase_a_lease.source_tensors
            for name in (
                "resource",
                "factor",
                "sqrt_residual",
                "lower",
                "upper",
                "block_capacities",
                "block_scales",
            ):
                item = _tensor_metadata(
                    getattr(source, name),
                    name=f"phase_A.source.{name}",
                )
                metadata.append(item)
                devices.add(str(item[-1]))
            if tuple(getattr(source.resource, "shape")) != (self.route_count,):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                    "Phase-A resource tensor left the route axis",
                )
            if tuple(getattr(source.factor, "shape")) != (
                self.phase_a_state_handle.response_rank,
                self.route_count,
            ):
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                    "Phase-A literal-J factor left its registered rank/route axis",
                )
            for state in phase_a_lease.branch_states:
                branch_modes.append((state.branch.value, state.witness_mode))
                metadata.append(
                    _tensor_metadata(
                        state.primal,
                        name=f"phase_A.{state.branch.value}.primal",
                    )
                )
                if state.witness_mode == "pdhg_resource_dual":
                    for name in ("low_q_dual", "tail_q_dual", "block_dual"):
                        value = getattr(state, name)
                        if value is None:
                            _refuse(
                                V1SignedWorkCUDAWitnessRefusalCode.MISSING_SOLVER_DUAL_CUSTODY,
                                "Phase-A PDHG state lost final solver dual custody",
                            )
                        metadata.append(
                            _tensor_metadata(
                                value,
                                name=f"phase_A.{state.branch.value}.{name}",
                            )
                        )
                elif state.witness_mode != "exact_cap":
                    _refuse(
                        V1SignedWorkCUDAWitnessRefusalCode.MISSING_SOLVER_DUAL_CUSTODY,
                        "Phase-A state has no recognized analytic or dual witness",
                    )

        phase_b_ids: list[str] = []
        for ordinal, binding in enumerate(self.phase_b_state_bindings, start=1):
            handle = binding.state_handle
            if not handle.signed_work_decision_dual_custody_available:
                _refuse(
                    V1SignedWorkCUDAWitnessRefusalCode.MISSING_SOLVER_DUAL_CUSTODY,
                    "Phase-B state lacks final signed-work solver dual custody",
                )
            phase_b_ids.append(handle.identity_hash)
            with handle.borrow_for_signed_work_decision_witness(
                expected_problem_hash=handle.problem_hash,
                expected_policy_hash=handle.policy_hash,
                expected_branch_receipt_hashes=handle.branch_receipt_hashes,
            ) as phase_b_lease:
                for name in (
                    "resource",
                    "singleton_work",
                    "factor",
                    "residual_diagonal",
                    "sqrt_residual",
                    "eligible",
                    "required",
                ):
                    item = _tensor_metadata(
                        phase_b_lease.source_tensor(name),
                        name=f"phase_B.{ordinal}.source.{name}",
                    )
                    metadata.append(item)
                    devices.add(str(item[-1]))
                factor = phase_b_lease.source_tensor("factor")
                if tuple(getattr(factor, "shape")) != (
                    self.phase_a_state_handle.response_rank,
                    self.route_count,
                ):
                    _refuse(
                        V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                        "Phase-B literal-J factor left the Phase-A rank/route axis",
                    )
                for branch in handle.branch_order:
                    metadata.append(
                        _tensor_metadata(
                            phase_b_lease.primal(branch),
                            name=f"phase_B.{ordinal}.{branch.value}.primal",
                        )
                    )
                    metadata.append(
                        _tensor_metadata(
                            phase_b_lease.reduced_advantage(branch),
                            name=(
                                f"phase_B.{ordinal}.{branch.value}.reduced_advantage"
                            ),
                        )
                    )
                    for name in (
                        "work_dual",
                        "low_rank_dual",
                        "q_tail_dual",
                        "resource_dual",
                        "block_dual",
                    ):
                        metadata.append(
                            _tensor_metadata(
                                phase_b_lease.decision_dual_tensor(name, branch),
                                name=f"phase_B.{ordinal}.{branch.value}.{name}",
                            )
                        )
        ordered_devices = tuple(sorted(devices))
        if len(ordered_devices) != 1:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "one registered uncertainty cell crossed tensor devices",
            )
        if require_cuda and not ordered_devices[0].startswith("cuda"):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.CUDA_REQUIRED,
                "production qualification observed non-CUDA theorem state",
            )
        return V1SignedWorkCUDATensorCustodyReceipt(
            registered_cell_hash=self.registered_cell_hash,
            phase_a_handle_hash=self.phase_a_state_handle.handle_hash,
            phase_b_state_identity_hashes=tuple(phase_b_ids),
            devices=ordered_devices,
            route_count=self.route_count,
            response_rank=self.phase_a_state_handle.response_rank,
            phase_a_branch_modes=tuple(branch_modes),
            tensor_metadata_commitment_hash=content_hash(
                tuple(metadata),
                prefix=_PREFIX + "tensor-metadata-",
            ),
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("registered CUDA uncertainty cells are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("registered CUDA uncertainty cells are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("registered CUDA uncertainty cells cannot be copied")

    def __deepcopy__(self, _memo: dict[int, Any]) -> NoReturn:
        raise TypeError("registered CUDA uncertainty cells cannot be copied")


def register_v1_signed_work_cuda_uncertainty_cell(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    grid_key: tuple[str, str, str, str],
    phase_a_result: V1SignedWorkResourceFrontierCUDAResult,
    phase_a_state_handle: V1SignedWorkResourceFrontierCUDATheoremStateHandle,
    phase_b_state_bindings: tuple[V1SignedWorkCUDAPhaseBStateBinding, ...],
    resource_cell_bindings: tuple[V1SignedWorkCUDAResourceCellBinding, ...],
    packed_action_family: V1SignedWorkPreCPackedActionFamily,
    preferred_physical_dedup_identity: V1SignedWorkPackedActionDedupIdentity,
) -> V1SignedWorkCUDARegisteredUncertaintyCell:
    """Register one complete live optimizer image without granting authority."""

    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError(
            "CUDA uncertainty-cell registration requires the exact owner set"
        )
    key = tuple(grid_key)
    if len(key) != 4 or any(
        key[index] not in axis
        for index, axis in enumerate(uncertainty.realization_axes)
    ):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_UNCERTAINTY_PRODUCT,
            "registered CUDA cell is outside the four-owner uncertainty product",
        )
    if phase_a_state_handle.response_rank != uncertainty.fd_residual.current_rank:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "Phase-A literal-J factor rank differs from the registered FD rank",
        )
    if phase_a_state_handle.route_axis_hash != uncertainty.route_axis_hash:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "Phase-A theorem state differs from the uncertainty route axis",
        )
    return V1SignedWorkCUDARegisteredUncertaintyCell(
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        grid_key=key,
        phase_a_result=phase_a_result,
        phase_a_state_handle=phase_a_state_handle,
        phase_b_state_bindings=phase_b_state_bindings,
        resource_cell_bindings=resource_cell_bindings,
        packed_action_family=packed_action_family,
        preferred_physical_dedup_identity=preferred_physical_dedup_identity,
        registered_d_w_prefix_hash=uncertainty.registered_d_w_prefix_hash,
        signed_f_uncertainty_set_hash=uncertainty.signed_f_sampling.set_hash,
        literal_j_uncertainty_set_hash=uncertainty.literal_j_sampling.set_hash,
        fd_residual_uncertainty_set_hash=uncertainty.fd_residual.set_hash,
        solver_uncertainty_set_hash=uncertainty.solver_rounding.set_hash,
        _issuer_token=_CELL_ISSUER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkCUDADecisionWitnessQualification:
    """Stable live-state qualification receipt with every authority closed."""

    uncertainty_set_hash: str
    registered_cell_hashes: tuple[str, ...]
    custody_receipts: tuple[V1SignedWorkCUDATensorCustodyReceipt, ...]
    resource_cell_family: tuple[str, ...]
    physical_action_family: tuple[tuple[str, str], ...]
    preferred_physical_decision_family: tuple[tuple[str, str], ...]
    phase_a_face_family_hashes: tuple[str, ...]
    phase_b_face_family_hashes: tuple[str, ...]
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    registered_d_w_prefix_hash: str
    registered_d_coordinate_hash: str
    signed_f_uncertainty_set_hash: str
    literal_j_uncertainty_set_hash: str
    fd_residual_uncertainty_set_hash: str
    solver_uncertainty_set_hash: str
    source_cells: tuple[V1SignedWorkCUDARegisteredUncertaintyCell, ...] = field(
        repr=False,
        compare=False,
    )
    production_cuda_sources_verified: bool
    _issuer_token: InitVar[object]
    complete_registered_uncertainty_product: bool = True
    resource_cell_physical_action_invariant: bool = True
    exact_face_or_branch_identity_required: bool = False
    byte_deduplicated_physical_actions_verified: bool = True
    final_solver_dual_custody_verified: bool = True
    nonserializable_identity_bound: bool = True
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
    authority_scope: str = SIGNED_WORK_D_W_CUDA_AUTHORITY_SCOPE
    decision_rule: str = SIGNED_WORK_D_W_CUDA_DECISION_RULE
    schema: str = SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _QUALIFICATION_ISSUER_TOKEN:
            raise TypeError("CUDA decision qualifications are issuer-only")
        cells = tuple(self.source_cells)
        if not cells or any(
            type(value) is not V1SignedWorkCUDARegisteredUncertaintyCell
            for value in cells
        ):
            raise TypeError("qualification requires exact live source cells")
        if tuple(value.registered_cell_hash for value in cells) != tuple(
            self.registered_cell_hashes
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "qualification source-cell identity does not replay",
            )
        flags = (
            self.complete_registered_uncertainty_product,
            self.resource_cell_physical_action_invariant,
            self.exact_face_or_branch_identity_required,
            self.byte_deduplicated_physical_actions_verified,
            self.final_solver_dual_custody_verified,
            self.nonserializable_identity_bound,
            self.generic_C_stream_open_authority,
            self.production_C_stream_open_authority,
            self.C_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            False,
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
                V1SignedWorkCUDAWitnessRefusalCode.AUTHORITY_VIOLATION,
                "CUDA qualification crossed its authority-neutral boundary",
            )
        if self.route_scale_d2h_bytes != 0:
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
                "CUDA decision issuer performed a route-scale host transfer",
            )
        if (
            self.authority_scope != SIGNED_WORK_D_W_CUDA_AUTHORITY_SCOPE
            or self.decision_rule != SIGNED_WORK_D_W_CUDA_DECISION_RULE
            or self.schema != SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA
        ):
            raise ValueError("unsupported CUDA decision-qualification semantics")
        object.__setattr__(self, "source_cells", cells)

    @property
    def h200_completeness_qualification_candidate(self) -> bool:
        """True means evidence may enter H200 testing, never that it passed."""

        return self.production_cuda_sources_verified

    @property
    def qualification_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "registered_cell_hashes": self.registered_cell_hashes,
                "custody_receipt_hashes": tuple(
                    value.receipt_hash for value in self.custody_receipts
                ),
                "resource_cell_family": self.resource_cell_family,
                "physical_action_family": self.physical_action_family,
                "preferred_physical_decision_family": (
                    self.preferred_physical_decision_family
                ),
                "phase_a_face_family_hashes": self.phase_a_face_family_hashes,
                "phase_b_face_family_hashes": self.phase_b_face_family_hashes,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "registered_d_w_prefix_hash": self.registered_d_w_prefix_hash,
                "owner_uncertainty_set_hashes": (
                    self.signed_f_uncertainty_set_hash,
                    self.literal_j_uncertainty_set_hash,
                    self.fd_residual_uncertainty_set_hash,
                    self.solver_uncertainty_set_hash,
                ),
                "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
                "production_cuda_sources_verified": (
                    self.production_cuda_sources_verified
                ),
                "route_scale_d2h_bytes": 0,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
                "decision_rule": self.decision_rule,
            },
            prefix=_PREFIX + "qualification-",
        )

    def assert_live_identity(self) -> None:
        for cell in self.source_cells:
            cell.assert_live_identity()

    def __reduce__(self) -> NoReturn:
        raise TypeError("CUDA decision qualifications are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("CUDA decision qualifications are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("CUDA decision qualifications cannot be copied")

    def __deepcopy__(self, _memo: dict[int, Any]) -> NoReturn:
        raise TypeError("CUDA decision qualifications cannot be copied")


def _issue_v1_signed_work_cuda_decision_witness(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    cells: tuple[V1SignedWorkCUDARegisteredUncertaintyCell, ...],
    require_cuda: bool,
) -> V1SignedWorkCUDADecisionWitnessQualification:
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError(
            "CUDA decision issuance requires the exact uncertainty owner set"
        )
    registered = tuple(cells)
    if not registered or any(
        type(value) is not V1SignedWorkCUDARegisteredUncertaintyCell
        for value in registered
    ):
        raise TypeError("CUDA decision issuance requires exact registered cells")
    if len(registered) > uncertainty.maximum_reference_realizations:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.BOUNDEDNESS_VIOLATION,
            "registered uncertainty-cell product exceeds its frozen bound",
        )
    expected_keys = tuple(product(*uncertainty.realization_axes))
    observed_keys = tuple(value.grid_key for value in registered)
    if (
        len(observed_keys) != len(set(observed_keys))
        or set(observed_keys) != set(expected_keys)
        or len(observed_keys) != len(expected_keys)
    ):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.INCOMPLETE_UNCERTAINTY_PRODUCT,
            "CUDA issuer did not receive every registered four-owner cell exactly once",
        )
    by_key = {value.grid_key: value for value in registered}
    ordered = tuple(by_key[key] for key in expected_keys)
    if any(
        value.uncertainty_set_hash != uncertainty.uncertainty_hash for value in ordered
    ):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "registered cells do not share the requested uncertainty owner set",
        )
    first = ordered[0]
    for value in ordered[1:]:
        if (
            value.parent_mask_hash != first.parent_mask_hash
            or value.route_axis_hash != first.route_axis_hash
            or value.route_count != first.route_count
            or value.registered_d_w_prefix_hash != first.registered_d_w_prefix_hash
            or value.registered_d_coordinate_hash != first.registered_d_coordinate_hash
        ):
            _refuse(
                V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
                "uncertainty cells crossed parent/chart/route identities",
            )
    owner_hashes = (
        uncertainty.signed_f_sampling.set_hash,
        uncertainty.literal_j_sampling.set_hash,
        uncertainty.fd_residual.set_hash,
        uncertainty.solver_rounding.set_hash,
    )
    if any(
        (
            value.signed_f_uncertainty_set_hash,
            value.literal_j_uncertainty_set_hash,
            value.fd_residual_uncertainty_set_hash,
            value.solver_uncertainty_set_hash,
        )
        != owner_hashes
        for value in ordered
    ):
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.STALE_IDENTITY,
            "signed-F/literal-J/FD/solver owner identities were conflated or lost",
        )
    resource_images = {value.resource_cell_family for value in ordered}
    physical_images = {value.physical_action_family for value in ordered}
    preferred_images = {value.preferred_physical_decision_family for value in ordered}
    if len(resource_images) != 1:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.DECISION_UNSTABLE,
            "Phase-A registered resource-cell family changes over uncertainty",
        )
    if len(physical_images) != 1 or len(preferred_images) != 1:
        _refuse(
            V1SignedWorkCUDAWitnessRefusalCode.DECISION_UNSTABLE,
            "Phase-B resource-cell x physical-action decision changes over uncertainty",
        )
    custody = tuple(
        value.verify_tensor_custody(require_cuda=require_cuda) for value in ordered
    )
    return V1SignedWorkCUDADecisionWitnessQualification(
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        registered_cell_hashes=tuple(value.registered_cell_hash for value in ordered),
        custody_receipts=custody,
        resource_cell_family=first.resource_cell_family,
        physical_action_family=first.physical_action_family,
        preferred_physical_decision_family=(first.preferred_physical_decision_family),
        phase_a_face_family_hashes=tuple(
            value.phase_a_face_family_hash for value in ordered
        ),
        phase_b_face_family_hashes=tuple(
            value.phase_b_face_family_hash for value in ordered
        ),
        parent_mask_hash=first.parent_mask_hash,
        route_axis_hash=first.route_axis_hash,
        route_count=first.route_count,
        registered_d_w_prefix_hash=first.registered_d_w_prefix_hash,
        registered_d_coordinate_hash=first.registered_d_coordinate_hash,
        signed_f_uncertainty_set_hash=owner_hashes[0],
        literal_j_uncertainty_set_hash=owner_hashes[1],
        fd_residual_uncertainty_set_hash=owner_hashes[2],
        solver_uncertainty_set_hash=owner_hashes[3],
        source_cells=ordered,
        production_cuda_sources_verified=require_cuda,
        _issuer_token=_QUALIFICATION_ISSUER_TOKEN,
    )


def issue_v1_signed_work_cuda_decision_witness_qualification(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    cells: tuple[V1SignedWorkCUDARegisteredUncertaintyCell, ...],
) -> V1SignedWorkCUDADecisionWitnessQualification:
    """Issue a CUDA/H200 qualification candidate with all C authority closed."""

    return _issue_v1_signed_work_cuda_decision_witness(
        uncertainty=uncertainty,
        cells=cells,
        require_cuda=True,
    )


def _issue_v1_signed_work_cuda_decision_witness_private_cpu_reference(
    *,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    cells: tuple[V1SignedWorkCUDARegisteredUncertaintyCell, ...],
) -> V1SignedWorkCUDADecisionWitnessQualification:
    """Tiny CPU-only semantic test seam; it can never become C authority."""

    return _issue_v1_signed_work_cuda_decision_witness(
        uncertainty=uncertainty,
        cells=cells,
        require_cuda=False,
    )


__all__ = [
    "SIGNED_WORK_D_W_CUDA_AUTHORITY_SCOPE",
    "SIGNED_WORK_D_W_CUDA_DECISION_RULE",
    "SIGNED_WORK_D_W_CUDA_QUALIFICATION_SCHEMA",
    "V1SignedWorkCUDADecisionWitnessQualification",
    "V1SignedWorkCUDAPhaseBStateBinding",
    "V1SignedWorkCUDARegisteredUncertaintyCell",
    "V1SignedWorkCUDAResourceCellBinding",
    "V1SignedWorkCUDATensorCustodyReceipt",
    "V1SignedWorkCUDAWitnessRefusal",
    "V1SignedWorkCUDAWitnessRefusalCode",
    "issue_v1_signed_work_cuda_decision_witness_qualification",
    "register_v1_signed_work_cuda_uncertainty_cell",
]
