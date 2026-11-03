"""Fail-closed two-branch union of bounded fixed-primal diagnostic shells.

Each convex Q branch may expose a tiny set of coordinates whose current
fixed-primal normal/rounding classification is ambiguous.  This module forms
their exact union and resolves only those route IDs from the constructive
physical axis.  The result is a *provisional* active-set initializer for the
later variable-z solve.  It does not convert fixed-primal diagnostics into an
action-invariance theorem and cannot open C/K/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import RemovedRouteIndexSet
from mechanistic_systems.core.axis_identity import ConstructiveOrdinalRouteAxis

from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_normal_cone_action_invariance_cuda import (
    V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceRegistration,
)


SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_two_branch_shell_union_v1"
)
SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_AUTHORITY: Final = (
    "provisional_B_le_8_variable_z_active_set_initializer_only_no_sampling_"
    "FD_rank_numerical_solution_sensitivity_action_invariance_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-signed-work-fixed-primal-shell-union-"
_MAXIMUM_CANONICAL_SHELL: Final = 8
_ISSUER_TOKEN: Final = object()


class V1SignedWorkFixedPrimalShellUnionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    BRANCH_AXIS_MISMATCH = "branch_axis_mismatch"
    STALE_IDENTITY = "stale_identity"
    EMPTY_SHELL = "empty_shell"
    SHELL_OVERSIZED = "shell_oversized"
    ROUTE_ID_AXIS_MISMATCH = "route_id_axis_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkFixedPrimalShellUnionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkFixedPrimalShellUnionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkFixedPrimalShellUnionRefusalCode:
            raise TypeError("shell-union refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkFixedPrimalShellUnionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkFixedPrimalShellUnionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalShellUnionReceipt:
    source_result_hashes: tuple[str, str]
    source_diagnostic_hashes: tuple[str, str]
    source_handoff_hashes: tuple[str, str]
    source_shell_index_hashes: tuple[str, str]
    source_branch_shell_indices: tuple[tuple[int, ...], tuple[int, ...]]
    problem_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    branch_axis: tuple[V1RelaxedKKTBranch, V1RelaxedKKTBranch]
    route_count: int
    maximum_shell_routes: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_removed_route_index_hash: str
    provisional_shell_hash: str
    provisional_action_family_hash: str
    complete_two_branch_fixed_primal_scans: bool = True
    exact_set_union_without_truncation: bool = True
    bounded_route_id_resolution_only: bool = True
    provisional_initializer_only: bool = True
    branch_baseline_symmetric_difference_included: bool = False
    complete_cross_branch_action_family_authority: bool = False
    alternative_solution_sensitivity_resolved: bool = False
    action_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_AUTHORITY
    schema: str = SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.INVALID_TYPE,
                "fixed-primal shell-union receipts are source-issued only",
            )
        for axis_name in (
            "source_result_hashes",
            "source_diagnostic_hashes",
            "source_handoff_hashes",
            "source_shell_index_hashes",
        ):
            axis = tuple(
                _identity(value, name=f"{axis_name} item")
                for value in getattr(self, axis_name)
            )
            if len(axis) != 2:
                raise ValueError(f"{axis_name} must contain exactly two branches")
            object.__setattr__(self, axis_name, axis)
        for name in (
            "problem_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "native_fiber_axis_receipt_hash",
            "shell_removed_route_index_hash",
            "provisional_shell_hash",
            "provisional_action_family_hash",
        ):
            _identity(getattr(self, name), name=name)
        if self.branch_axis != (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        ):
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.BRANCH_AXIS_MISMATCH,
                "shell union requires the complete canonical Q-branch axis",
            )
        route_count = _positive_int(self.route_count, name="route count")
        maximum = _positive_int(
            self.maximum_shell_routes,
            name="maximum shell routes",
        )
        if maximum > _MAXIMUM_CANONICAL_SHELL:
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.SHELL_OVERSIZED,
                "canonical variable-shell enumeration is bounded by B=8",
            )
        branch_indices = tuple(tuple(axis) for axis in self.source_branch_shell_indices)
        if len(branch_indices) != 2:
            raise ValueError("source branch shell axis must contain two entries")
        for axis in branch_indices:
            if any(type(index) is not int for index in axis) or any(
                right <= left for left, right in zip(axis, axis[1:])
            ):
                raise ValueError("source branch shell indices are not canonical")
        indices = tuple(self.shell_indices)
        route_ids = tuple(self.shell_route_ids)
        expected_union = tuple(sorted(set(branch_indices[0]).union(branch_indices[1])))
        if not indices:
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.EMPTY_SHELL,
                "the two fixed-primal scans produced no variable-shell initializer",
            )
        if indices != expected_union or len(indices) > maximum:
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.SHELL_OVERSIZED,
                "two-branch shell union was truncated, reordered, or exceeds B=8",
            )
        if len(route_ids) != len(indices) or any(
            route_id != f"route:{index}"
            for index, route_id in zip(indices, route_ids, strict=True)
        ):
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.ROUTE_ID_AXIS_MISMATCH,
                "bounded route IDs do not match the constructive physical axis",
            )
        if any(index < 0 or index >= route_count for index in indices):
            raise ValueError("shell-union index is outside the route axis")
        canonical = RemovedRouteIndexSet.from_indices(
            indices,
            route_count=route_count,
        )
        if canonical.index_hash != self.shell_removed_route_index_hash:
            raise ValueError("shell-union removed-route identity does not replay")
        expected_shell_hash = content_hash(
            {
                "schema": SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA,
                "route_axis_hash": self.route_axis_hash,
                "route_id_axis_hash": self.route_id_axis_hash,
                "shell_indices": indices,
                "shell_route_ids": route_ids,
                "source_handoff_hashes": self.source_handoff_hashes,
            },
            prefix=_PREFIX + "shell-",
        )
        expected_family_hash = content_hash(
            {
                "schema": "fixed-primal-two-branch-provisional-action-family-v1",
                "problem_hash": self.problem_hash,
                "branch_axis": tuple(branch.value for branch in self.branch_axis),
                "source_diagnostic_hashes": self.source_diagnostic_hashes,
                "source_result_hashes": self.source_result_hashes,
                "provisional_shell_hash": expected_shell_hash,
            },
            prefix=_PREFIX + "action-family-",
        )
        if (
            expected_shell_hash != self.provisional_shell_hash
            or expected_family_hash != self.provisional_action_family_hash
        ):
            raise ValueError("shell-union provisional identity does not replay")
        open_flags = (
            self.complete_two_branch_fixed_primal_scans,
            self.exact_set_union_without_truncation,
            self.bounded_route_id_resolution_only,
            self.provisional_initializer_only,
        )
        closed_flags = (
            self.branch_baseline_symmetric_difference_included,
            self.complete_cross_branch_action_family_authority,
            self.alternative_solution_sensitivity_resolved,
            self.action_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(flag is not True for flag in open_flags) or any(
            flag is not False for flag in closed_flags
        ):
            _refuse(
                V1SignedWorkFixedPrimalShellUnionRefusalCode.AUTHORITY_VIOLATION,
                "fixed-primal shell union broadened its provisional claim ceiling",
            )
        if (
            self.authority_scope != SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_AUTHORITY
            or self.schema != SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA
        ):
            raise ValueError("unsupported fixed-primal shell-union semantics")
        object.__setattr__(self, "source_branch_shell_indices", branch_indices)
        object.__setattr__(self, "shell_indices", indices)
        object.__setattr__(self, "shell_route_ids", route_ids)

    @property
    def receipt_hash(self) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "receipt-")


def freeze_v1_signed_work_fixed_primal_shell_union(
    *,
    branch_results: Sequence[V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult],
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    route_ids: ConstructiveOrdinalRouteAxis,
    maximum_shell_routes: int = _MAXIMUM_CANONICAL_SHELL,
) -> V1SignedWorkFixedPrimalShellUnionReceipt:
    """Form the exact B<=8 union without granting theorem authority."""

    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        raise TypeError("shell union requires the exact native-fiber receipt")
    native_fiber_axis_receipt.assert_integrity()
    if type(route_ids) is not ConstructiveOrdinalRouteAxis:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.ROUTE_ID_AXIS_MISMATCH,
            "production shell union requires the constructive route-ID owner",
        )
    results = tuple(branch_results)
    if len(results) != 2 or any(
        type(result) is not V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult
        for result in results
    ):
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.INVALID_TYPE,
            "shell union requires exactly two fixed-primal diagnostic results",
        )
    by_branch = {result.diagnostic.branch: result for result in results}
    branch_axis = (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    )
    if set(by_branch) != set(branch_axis) or len(by_branch) != 2:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.BRANCH_AXIS_MISMATCH,
            "shell union requires one result from each canonical Q branch",
        )
    ordered = tuple(by_branch[branch] for branch in branch_axis)
    handoffs = tuple(result.bounded_shell_handoff for result in ordered)
    shared_axes = (
        "problem_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "route_axis_hash",
        "native_fiber_axis_receipt_hash",
        "route_count",
    )
    if any(
        getattr(handoffs[0], name) != getattr(handoffs[1], name) for name in shared_axes
    ):
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.STALE_IDENTITY,
            "fixed-primal branch results do not share one chart and physical axis",
        )
    first = handoffs[0]
    if (
        native_fiber_axis_receipt.receipt_hash != first.native_fiber_axis_receipt_hash
        or native_fiber_axis_receipt.route_axis_hash != first.route_axis_hash
        or native_fiber_axis_receipt.route_count != first.route_count
        or len(route_ids) != first.route_count
    ):
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.ROUTE_ID_AXIS_MISMATCH,
            "native-fiber or constructive route identity differs from the diagnostics",
        )
    maximum = _positive_int(maximum_shell_routes, name="maximum shell routes")
    if maximum > _MAXIMUM_CANONICAL_SHELL:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.SHELL_OVERSIZED,
            "canonical variable-shell enumeration is bounded by B=8",
        )
    indices = tuple(
        sorted(set(handoffs[0].shell_indices).union(handoffs[1].shell_indices))
    )
    if not indices:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.EMPTY_SHELL,
            "the two fixed-primal scans produced no variable-shell initializer",
        )
    if len(indices) > maximum:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.SHELL_OVERSIZED,
            "the exact two-branch union exceeds the pre-outcome shell cap",
        )
    resolved_ids = tuple(route_ids[index] for index in indices)
    result_hashes = tuple(
        content_hash(
            {
                "diagnostic_hash": result.diagnostic.diagnostic_hash,
                "outside_scan_receipt_hash": result.outside_scan.receipt_hash,
                "shell_enumeration_receipt_hash": (
                    result.shell_enumeration.receipt_hash
                ),
                "hard_replay_receipt_hashes": tuple(
                    replay.receipt_hash for replay in result.hard_replays
                ),
                "bounded_shell_handoff_hash": (
                    result.bounded_shell_handoff.handoff_hash
                ),
            },
            prefix=_PREFIX + "source-result-",
        )
        for result in ordered
    )
    handoff_hashes = tuple(handoff.handoff_hash for handoff in handoffs)
    shell_hash = content_hash(
        {
            "schema": SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA,
            "route_axis_hash": first.route_axis_hash,
            "route_id_axis_hash": native_fiber_axis_receipt.route_id_axis_hash,
            "shell_indices": indices,
            "shell_route_ids": resolved_ids,
            "source_handoff_hashes": handoff_hashes,
        },
        prefix=_PREFIX + "shell-",
    )
    diagnostic_hashes = tuple(result.diagnostic.diagnostic_hash for result in ordered)
    action_family_hash = content_hash(
        {
            "schema": "fixed-primal-two-branch-provisional-action-family-v1",
            "problem_hash": first.problem_hash,
            "branch_axis": tuple(branch.value for branch in branch_axis),
            "source_diagnostic_hashes": diagnostic_hashes,
            "source_result_hashes": result_hashes,
            "provisional_shell_hash": shell_hash,
        },
        prefix=_PREFIX + "action-family-",
    )
    return V1SignedWorkFixedPrimalShellUnionReceipt(
        source_result_hashes=result_hashes,  # type: ignore[arg-type]
        source_diagnostic_hashes=diagnostic_hashes,  # type: ignore[arg-type]
        source_handoff_hashes=handoff_hashes,  # type: ignore[arg-type]
        source_shell_index_hashes=tuple(
            handoff.shell_index_hash for handoff in handoffs
        ),  # type: ignore[arg-type]
        source_branch_shell_indices=tuple(
            handoff.shell_indices for handoff in handoffs
        ),  # type: ignore[arg-type]
        problem_hash=first.problem_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.current_mask_hash,
        route_axis_hash=first.route_axis_hash,
        route_id_axis_hash=native_fiber_axis_receipt.route_id_axis_hash,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        branch_axis=branch_axis,
        route_count=first.route_count,
        maximum_shell_routes=maximum,
        shell_indices=indices,
        shell_route_ids=resolved_ids,
        shell_removed_route_index_hash=RemovedRouteIndexSet.from_indices(
            indices,
            route_count=first.route_count,
        ).index_hash,
        provisional_shell_hash=shell_hash,
        provisional_action_family_hash=action_family_hash,
        _issuance_token=_ISSUER_TOKEN,
    )


def build_v1_signed_work_variable_shell_source_registration_from_fixed_primal_shell_union(
    *,
    shell_union: V1SignedWorkFixedPrimalShellUnionReceipt,
    branch_result: V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    registration_id: str,
    model_identity_hash: str,
    proposal_generation: int,
    registered_d_prefix_hash: str,
    registered_d_w2_prefix_hash: str,
    parent_d_source_receipt_hash: str,
    resource_axis_hash: str,
    resource_source_receipt_hash: str,
    witness_stream_source_hash: str,
    same_capture_vjp_source_hash: str,
    corrected_evidence_stream_source_hash: str,
    literal_j_stream_source_hash: str,
    alpha_ledger_id: str,
    alpha_coordinate_id: str,
    alpha_lineage_hash: str,
    population_count: int,
    population_context_axis_hash: str,
    population_draw_axis_hash: str,
    registered_context_ids: Sequence[str],
    registered_draw_ids: Sequence[str],
    registered_measurement_ids: Sequence[str],
    registered_capture_ids: Sequence[str],
    registered_source_packet_hashes: Sequence[str],
    context_weights: Sequence[float],
) -> V1SignedWorkVariableShellSourceRegistration:
    """Convert a provisional B<=8 shell union into one branch-local D_W2 registration.

    The shell union only chooses the bounded coordinate support.  The returned
    registration still has no source rows, no compact source handle, and no
    action/C/K/O authority; it merely pins the exact shell, chart, branch, and
    registered D_W2 prefix before the same-capture rows are streamed.
    """

    if type(shell_union) is not V1SignedWorkFixedPrimalShellUnionReceipt:
        raise TypeError("shell_union must be the exact fixed-primal shell-union receipt")
    if type(branch_result) is not V1SignedWorkFixedPrimalRoundingDiagnosticCUDAResult:
        raise TypeError("branch_result must be the exact fixed-primal diagnostic result")
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        raise TypeError("native_fiber_axis_receipt must be exact")
    native_fiber_axis_receipt.assert_integrity()
    branch = branch_result.diagnostic.branch
    if branch not in shell_union.branch_axis:
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.BRANCH_AXIS_MISMATCH,
            "branch diagnostic is not part of the shell-union branch axis",
        )
    branch_ordinal = shell_union.branch_axis.index(branch)
    expected_result_hash = content_hash(
        {
            "diagnostic_hash": branch_result.diagnostic.diagnostic_hash,
            "outside_scan_receipt_hash": branch_result.outside_scan.receipt_hash,
            "shell_enumeration_receipt_hash": (
                branch_result.shell_enumeration.receipt_hash
            ),
            "hard_replay_receipt_hashes": tuple(
                replay.receipt_hash for replay in branch_result.hard_replays
            ),
            "bounded_shell_handoff_hash": (
                branch_result.bounded_shell_handoff.handoff_hash
            ),
        },
        prefix=_PREFIX + "source-result-",
    )
    handoff = branch_result.bounded_shell_handoff
    if (
        shell_union.source_result_hashes[branch_ordinal] != expected_result_hash
        or shell_union.source_diagnostic_hashes[branch_ordinal]
        != branch_result.diagnostic.diagnostic_hash
        or shell_union.source_handoff_hashes[branch_ordinal] != handoff.handoff_hash
        or shell_union.source_branch_shell_indices[branch_ordinal]
        != handoff.shell_indices
        or shell_union.problem_hash != handoff.problem_hash
        or shell_union.chart_identity_hash != handoff.chart_identity_hash
        or shell_union.current_mask_hash != handoff.current_mask_hash
        or shell_union.route_axis_hash != handoff.route_axis_hash
        or shell_union.native_fiber_axis_receipt_hash
        != handoff.native_fiber_axis_receipt_hash
        or shell_union.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or shell_union.route_axis_hash != native_fiber_axis_receipt.route_axis_hash
        or shell_union.route_id_axis_hash
        != native_fiber_axis_receipt.route_id_axis_hash
        or shell_union.route_count != native_fiber_axis_receipt.route_count
    ):
        _refuse(
            V1SignedWorkFixedPrimalShellUnionRefusalCode.STALE_IDENTITY,
            "branch diagnostic, shell union, and native-fiber receipt do not close",
        )
    return V1SignedWorkVariableShellSourceRegistration(
        registration_id=_identity(registration_id, name="registration ID"),
        model_identity_hash=_identity(model_identity_hash, name="model identity hash"),
        chart_identity_hash=shell_union.chart_identity_hash,
        current_mask_hash=shell_union.current_mask_hash,
        proposal_generation=_nonnegative_int(
            proposal_generation,
            name="proposal generation",
        ),
        route_axis_hash=shell_union.route_axis_hash,
        route_id_axis_hash=shell_union.route_id_axis_hash,
        registered_d_prefix_hash=_identity(
            registered_d_prefix_hash,
            name="registered D prefix hash",
        ),
        registered_d_w2_prefix_hash=_identity(
            registered_d_w2_prefix_hash,
            name="registered D_W2 prefix hash",
        ),
        parent_d_source_receipt_hash=_identity(
            parent_d_source_receipt_hash,
            name="parent D source receipt hash",
        ),
        shell_freeze_receipt_hash=shell_union.receipt_hash,
        shell_indices=shell_union.shell_indices,
        shell_route_ids=shell_union.shell_route_ids,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
        kkt_branch_id=branch.value,
        kkt_branch_source_receipt_hash=branch_result.diagnostic.branch_receipt_hash,
        resource_axis_hash=_identity(resource_axis_hash, name="resource axis hash"),
        resource_source_receipt_hash=_identity(
            resource_source_receipt_hash,
            name="resource source receipt hash",
        ),
        witness_stream_source_hash=_identity(
            witness_stream_source_hash,
            name="witness stream source hash",
        ),
        same_capture_vjp_source_hash=_identity(
            same_capture_vjp_source_hash,
            name="same-capture VJP source hash",
        ),
        corrected_evidence_stream_source_hash=_identity(
            corrected_evidence_stream_source_hash,
            name="corrected evidence stream source hash",
        ),
        literal_j_stream_source_hash=_identity(
            literal_j_stream_source_hash,
            name="literal-J stream source hash",
        ),
        alpha_ledger_id=_identity(alpha_ledger_id, name="alpha ledger ID"),
        alpha_coordinate_id=_identity(alpha_coordinate_id, name="alpha coordinate ID"),
        alpha_lineage_hash=_identity(alpha_lineage_hash, name="alpha lineage hash"),
        population_count=_positive_int(population_count, name="population count"),
        population_context_axis_hash=_identity(
            population_context_axis_hash,
            name="population context axis hash",
        ),
        population_draw_axis_hash=_identity(
            population_draw_axis_hash,
            name="population draw axis hash",
        ),
        registered_context_ids=tuple(registered_context_ids),
        registered_draw_ids=tuple(registered_draw_ids),
        registered_measurement_ids=tuple(registered_measurement_ids),
        registered_capture_ids=tuple(registered_capture_ids),
        registered_source_packet_hashes=tuple(registered_source_packet_hashes),
        context_weights=tuple(float(value) for value in context_weights),
    )


__all__ = [
    "SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_AUTHORITY",
    "SIGNED_WORK_FIXED_PRIMAL_SHELL_UNION_SCHEMA",
    "V1SignedWorkFixedPrimalShellUnionReceipt",
    "V1SignedWorkFixedPrimalShellUnionRefusal",
    "V1SignedWorkFixedPrimalShellUnionRefusalCode",
    "build_v1_signed_work_variable_shell_source_registration_from_fixed_primal_shell_union",
    "freeze_v1_signed_work_fixed_primal_shell_union",
]
