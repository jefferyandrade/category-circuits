r"""Exact pair cover for the variable-shell outside-action seam.

This module deliberately lands only the first authority-inert joint of the
full-axis theorem.  Let ``P`` be one preferred fixed-cardinality hard action
and ``B`` the bounded decision shell.  Every admissible action ``S`` with

``|S| = |P|`` and ``(S triangle P) intersect B**c != empty``

contains a selected/unselected exchange ``q in S\P``, ``p in P\S`` with
``z_q >= z_p`` in its closed rounding cell.  The complete cover is the
disjoint union of two rectangles:

1. every droppable ``p in P`` crossed with every addable ``q outside B``;
2. every droppable ``p in P outside B`` crossed with every addable ``q in B``.

The second rectangle is needed by the generic theorem even though today's
device-shell construction normally has ``P subset B``.  Each pair cell is a
convex over-cover.  Given a dual lower plane with stationarity coefficients
``s``, minimizing its box term subject to ``z_p <= z_q`` replaces

``min(0,s_p) + min(0,s_q)``

by the exact three-vertex expression ``min(0,s_q,s_p+s_q)``.  This gives a
cheap full-axis pair screen once the stationarity plane has been replayed.
Required prior-rung routes are fixed at one in every nested action; ineligible
and already-removed routes are fixed at zero on the next-action axis.

No production A-row replay provider is landed here.  The CUDA issuer therefore
refuses explicitly rather than pretending that bounded-shell rows, host-side
arrays, or a caller-provided stationarity vector establish same-prefix source
custody.  Reference evaluators consume precomputed binary64 coefficients only.
They minimize the supplied lower plane exactly over the continuous box with
the pair halfspace and fixed required/forbidden coordinates.  Cardinality,
resource, block-capacity, and Q constraints are not retained primal
constraints; any dualization embedded in the caller's plane is unverified.
Consequently the reference value is a lower bound on every retained feasible
domain, never a claim to its exact optimum.  Reference inputs are explicitly
caller-supplied route-scale host coefficients; zero-D2H fields scope only the
operations performed by this module's issuer.  Every receipt in this module
keeps global invariance, C/K/O, commit, terminal, and Pareto authority closed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Final, Iterator, NoReturn, Sequence

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceReceipt,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


OUTSIDE_PAIR_COVER_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_pair_cover_v1"
)
OUTSIDE_PAIR_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_pair_identity_v1"
)
OUTSIDE_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_exclusion_cuda_design_v1"
)
OUTSIDE_CUDA_HBM_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_exclusion_cuda_hbm_v2"
)
OUTSIDE_LOWER_PLANE_BINDING_SCHEMA: Final = (
    "partition_free_v1_variable_shell_full_axis_lower_plane_binding_v2"
)
OUTSIDE_LOWER_PLANE_LEASE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_full_axis_lower_plane_lease_v1"
)
OUTSIDE_PAIR_BOUND_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_pair_dual_bound_v2"
)
OUTSIDE_EXCLUSION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_variable_shell_outside_pair_exclusion_receipt_v2"
)

OUTSIDE_PAIR_COVER_RULE: Final = (
    "rectangle_1_equals_droppable_preferred_cross_addable_outside_shell_and_"
    "rectangle_2_equals_droppable_preferred_outside_shell_cross_addable_"
    "inside_shell_so_every_same_cardinality_outside_membership_change_has_"
    "one_selected_q_unselected_p_closed_cell_zq_ge_zp_v1"
)
OUTSIDE_PAIR_BOX_RULE: Final = (
    "replace_min_0_sp_plus_min_0_sq_by_min_0_sq_sp_plus_sq_for_zp_le_zq_v1"
)
OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE: Final = (
    "exact_continuous_box_plus_required_forbidden_coordinates_plus_pair_"
    "halfspace_only_cardinality_resource_block_and_Q_primal_constraints_"
    "dropped_caller_lower_plane_dualization_unverified_reference_only_v1"
)
OUTSIDE_AUTHORITY_SCOPE: Final = (
    "combinatorial_pair_cover_and_reference_dual_box_evaluation_only_no_"
    "production_A_row_replay_global_action_invariance_C_K_O_commit_terminal_"
    "or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-outside-pair-exclusion-v1-"
_DESIGN_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_PAIR_TOKEN: Final = object()
_COVER_TOKEN: Final = object()
_BINDING_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_BOUND_TOKEN: Final = object()
_EXCLUSION_TOKEN: Final = object()
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8


class V1SignedWorkVariableShellOutsideExclusionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    AXIS_MISMATCH = "axis_mismatch"
    INADMISSIBLE_PREFERRED_ACTION = "inadmissible_preferred_action"
    INADMISSIBLE_ACTION = "inadmissible_action"
    INCOMPLETE_PAIR_AXIS = "incomplete_pair_axis"
    DUPLICATE_PAIR = "duplicate_pair"
    NONFINITE_LOWER_PLANE = "nonfinite_lower_plane"
    NONSTRICT_EXCLUSION = "nonstrict_exclusion"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    CUDA_REQUIRED = "cuda_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    STALE_IDENTITY = "stale_identity"
    SAME_PREFIX_A_ROW_REPLAY_PROVIDER_REQUIRED = (
        "same_prefix_a_row_replay_provider_required"
    )
    RESOURCE_BLOCK_CARDINALITY_DUAL_PROVIDER_REQUIRED = (
        "resource_block_cardinality_dual_provider_required"
    )
    PRODUCTION_PROVIDER_INTEGRATION_PENDING = "production_provider_integration_pending"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkVariableShellOutsideExclusionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkVariableShellOutsideExclusionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkVariableShellOutsideExclusionRefusalCode:
            raise TypeError("outside-exclusion refusal code has another type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkVariableShellOutsideExclusionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkVariableShellOutsideExclusionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.NONFINITE_LOWER_PLANE,
            f"{name} must be finite",
        )
    return result


def _index_axis(
    values: Sequence[int],
    *,
    route_count: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be one integer sequence",
        )
    axis = tuple(_nonnegative_int(value, name=f"{name} item") for value in values)
    if any(value >= route_count for value in axis) or axis != tuple(sorted(set(axis))):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.AXIS_MISMATCH,
            f"{name} must be sorted, unique, and inside the route axis",
        )
    return axis


def _bool_axis(
    values: Sequence[bool],
    *,
    route_count: int,
    name: str,
) -> tuple[bool, ...]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != route_count
        or any(type(value) is not bool for value in values)
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.AXIS_MISMATCH,
            f"{name} must be an exact Boolean route axis",
        )
    return tuple(values)


def _fraction(value: object, *, name: str) -> Fraction:
    result = _finite(value, name=name)
    return Fraction.from_float(result)


def _down_from_fraction(value: Fraction) -> float:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.NONFINITE_LOWER_PLANE,
            "exact lower-plane value is outside finite binary64",
        )
    return math.nextafter(center, -math.inf)


def _closed_authority(values: Sequence[bool], *, owner: str) -> None:
    if any(type(value) is not bool or value is not False for value in values):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} leaked downstream authority",
        )


class V1SignedWorkVariableShellOutsidePairRectangle(str, Enum):
    ADD_OUTSIDE = "add_outside"
    DROP_OUTSIDE_ADD_INSIDE = "drop_outside_add_inside"


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsidePairIdentity:
    preferred_selected_index: int
    preferred_unselected_index: int
    rectangle: V1SignedWorkVariableShellOutsidePairRectangle
    pair_ordinal: int
    schema: str = OUTSIDE_PAIR_IDENTITY_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _PAIR_TOKEN:
            raise TypeError("outside-pair identities are issuer-only")
        p = _nonnegative_int(
            self.preferred_selected_index,
            name="preferred-selected route index",
        )
        q = _nonnegative_int(
            self.preferred_unselected_index,
            name="preferred-unselected route index",
        )
        if p == q:
            raise ValueError("outside-pair coordinates must be distinct")
        object.__setattr__(self, "preferred_selected_index", p)
        object.__setattr__(self, "preferred_unselected_index", q)
        object.__setattr__(
            self,
            "pair_ordinal",
            _nonnegative_int(self.pair_ordinal, name="pair ordinal"),
        )
        if type(self.rectangle) is not V1SignedWorkVariableShellOutsidePairRectangle:
            raise TypeError("outside-pair rectangle has another type")
        if self.schema != OUTSIDE_PAIR_IDENTITY_SCHEMA:
            raise ValueError("unsupported outside-pair schema")

    @property
    def pair_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "pair-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsidePairCoverReceipt:
    route_count: int
    fixed_cardinality: int
    preferred_action_index_hash: str
    shell_index_hash: str
    eligible_mask_hash: str
    required_mask_hash: str
    already_removed_mask_hash: str
    selectable_mask_hash: str
    preferred_outside_shell_count: int
    droppable_preferred_count: int
    addable_outside_shell_count: int
    droppable_preferred_outside_shell_count: int
    addable_inside_shell_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_count: int
    pair_axis_commitment_hash: str
    complete_fixed_cardinality_outside_change_cover: bool = True
    deterministic_physical_index_order: bool = True
    closed_ties_covered: bool = True
    generic_preferred_outside_shell_supported: bool = True
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    cover_rule: str = OUTSIDE_PAIR_COVER_RULE
    authority_scope: str = OUTSIDE_AUTHORITY_SCOPE
    schema: str = OUTSIDE_PAIR_COVER_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _COVER_TOKEN:
            raise TypeError("outside pair-cover receipts are issuer-only")
        route_count = _positive_int(self.route_count, name="route count")
        cardinality = _nonnegative_int(
            self.fixed_cardinality,
            name="fixed cardinality",
        )
        if cardinality > route_count:
            raise ValueError("fixed cardinality exceeds the route axis")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "fixed_cardinality", cardinality)
        for name in (
            "preferred_action_index_hash",
            "shell_index_hash",
            "eligible_mask_hash",
            "required_mask_hash",
            "already_removed_mask_hash",
            "selectable_mask_hash",
            "pair_axis_commitment_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "preferred_outside_shell_count",
            "droppable_preferred_count",
            "addable_outside_shell_count",
            "droppable_preferred_outside_shell_count",
            "addable_inside_shell_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        expected_first = (
            self.droppable_preferred_count * self.addable_outside_shell_count
        )
        expected_second = (
            self.droppable_preferred_outside_shell_count
            * self.addable_inside_shell_count
        )
        if (
            self.add_outside_rectangle_pair_count != expected_first
            or self.drop_outside_add_inside_rectangle_pair_count != expected_second
            or self.complete_pair_count != expected_first + expected_second
            or self.droppable_preferred_outside_shell_count
            > self.preferred_outside_shell_count
        ):
            raise ValueError("outside pair-cover rectangle counts do not close")
        required_true = (
            self.complete_fixed_cardinality_outside_change_cover,
            self.deterministic_physical_index_order,
            self.closed_ties_covered,
            self.generic_preferred_outside_shell_supported,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            raise ValueError("outside pair-cover theorem flags changed")
        _closed_authority(
            (
                self.action_invariance_authority,
                self.c_stream_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="outside pair-cover receipt",
        )
        if (
            self.cover_rule != OUTSIDE_PAIR_COVER_RULE
            or self.authority_scope != OUTSIDE_AUTHORITY_SCOPE
            or self.schema != OUTSIDE_PAIR_COVER_SCHEMA
        ):
            raise ValueError("unsupported outside pair-cover semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "cover-")


class V1SignedWorkVariableShellOutsidePairCoverEnumeration:
    """Process-local exact lazy enumeration of the two pair rectangles."""

    __slots__ = (
        "_route_count",
        "_preferred",
        "_shell",
        "_selectable",
        "_required",
        "_droppable",
        "_addable",
        "_addable_outside",
        "_droppable_outside",
        "_addable_inside",
        "receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        route_count: int,
        preferred: tuple[int, ...],
        shell: tuple[int, ...],
        selectable: tuple[bool, ...],
        required: tuple[int, ...],
        droppable: tuple[int, ...],
        addable: tuple[int, ...],
        addable_outside: tuple[int, ...],
        droppable_outside: tuple[int, ...],
        addable_inside: tuple[int, ...],
        receipt: V1SignedWorkVariableShellOutsidePairCoverReceipt,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _COVER_TOKEN:
            raise TypeError("outside pair-cover enumerations are issuer-only")
        self._route_count = route_count
        self._preferred = preferred
        self._shell = shell
        self._selectable = selectable
        self._required = required
        self._droppable = droppable
        self._addable = addable
        self._addable_outside = addable_outside
        self._droppable_outside = droppable_outside
        self._addable_inside = addable_inside
        self.receipt = receipt

    @property
    def preferred_action_indices(self) -> tuple[int, ...]:
        return self._preferred

    @property
    def shell_indices(self) -> tuple[int, ...]:
        return self._shell

    @property
    def selectable_mask(self) -> tuple[bool, ...]:
        return self._selectable

    @property
    def required_action_indices(self) -> tuple[int, ...]:
        return self._required

    def iter_pairs(self) -> Iterator[V1SignedWorkVariableShellOutsidePairIdentity]:
        ordinal = 0
        for p in self._droppable:
            for q in self._addable_outside:
                yield V1SignedWorkVariableShellOutsidePairIdentity(
                    preferred_selected_index=p,
                    preferred_unselected_index=q,
                    rectangle=V1SignedWorkVariableShellOutsidePairRectangle.ADD_OUTSIDE,
                    pair_ordinal=ordinal,
                    _issuer_token=_PAIR_TOKEN,
                )
                ordinal += 1
        for p in self._droppable_outside:
            for q in self._addable_inside:
                yield V1SignedWorkVariableShellOutsidePairIdentity(
                    preferred_selected_index=p,
                    preferred_unselected_index=q,
                    rectangle=(
                        V1SignedWorkVariableShellOutsidePairRectangle.DROP_OUTSIDE_ADD_INSIDE
                    ),
                    pair_ordinal=ordinal,
                    _issuer_token=_PAIR_TOKEN,
                )
                ordinal += 1
        if ordinal != self.receipt.complete_pair_count:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.INCOMPLETE_PAIR_AXIS,
                "lazy outside pair axis no longer matches its receipt",
            )

    def _validated_action(self, action_indices: Sequence[int]) -> tuple[int, ...]:
        action = _index_axis(
            action_indices,
            route_count=self._route_count,
            name="hard action indices",
        )
        # The exact required axis is retained process-locally through the
        # droppable complement: required preferred coordinates are precisely
        # P \ droppable.  They must remain selected in every nested rung.
        required = set(self._required)
        if (
            len(action) != self.receipt.fixed_cardinality
            or any(not self._selectable[index] for index in action)
            or not required.issubset(action)
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.INADMISSIBLE_ACTION,
                "hard action has another cardinality, drops a required prior rung, or selects a forbidden route",
            )
        return action

    def action_changes_outside_membership(
        self,
        action_indices: Sequence[int],
    ) -> bool:
        action = set(self._validated_action(action_indices))
        preferred = set(self._preferred)
        shell = set(self._shell)
        return any(
            index not in shell for index in action.symmetric_difference(preferred)
        )

    def covering_pairs_for_action(
        self,
        action_indices: Sequence[int],
    ) -> tuple[V1SignedWorkVariableShellOutsidePairIdentity, ...]:
        action = set(self._validated_action(action_indices))
        preferred = set(self._preferred)
        removed = preferred - action
        added = action - preferred
        observed = tuple(
            pair
            for pair in self.iter_pairs()
            if pair.preferred_selected_index in removed
            and pair.preferred_unselected_index in added
        )
        if (
            self.action_changes_outside_membership(tuple(sorted(action)))
            and not observed
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.INCOMPLETE_PAIR_AXIS,
                "one admissible outside-changing action has no pair cell",
            )
        return observed

    def deterministic_top_k_action(
        self,
        route_values: Sequence[Real],
    ) -> tuple[int, ...]:
        if len(route_values) != self._route_count:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.AXIS_MISMATCH,
                "route values do not align with the route axis",
            )
        values = tuple(_finite(value, name="route value") for value in route_values)
        required = set(self._required)
        candidates = [
            index
            for index, admitted in enumerate(self._selectable)
            if admitted and index not in required
        ]
        selected = (
            list(required)
            + sorted(
                candidates,
                key=lambda index: (-values[index], index),
            )[: self.receipt.fixed_cardinality - len(required)]
        )
        return tuple(sorted(selected))

    def __reduce__(self) -> NoReturn:
        raise TypeError("live outside pair-cover enumerations are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live outside pair-cover enumerations are nonserializable")


def build_v1_signed_work_variable_shell_outside_pair_cover(
    *,
    route_count: int,
    preferred_action_indices: Sequence[int],
    shell_indices: Sequence[int],
    eligible_mask: Sequence[bool],
    required_mask: Sequence[bool],
    already_removed_mask: Sequence[bool],
) -> V1SignedWorkVariableShellOutsidePairCoverEnumeration:
    """Build the exact combinatorial cover without claiming source replay."""

    count = _positive_int(route_count, name="route count")
    preferred = _index_axis(
        preferred_action_indices,
        route_count=count,
        name="preferred hard action",
    )
    shell = _index_axis(shell_indices, route_count=count, name="bounded shell")
    eligible = _bool_axis(eligible_mask, route_count=count, name="eligible mask")
    required = _bool_axis(required_mask, route_count=count, name="required mask")
    already = _bool_axis(
        already_removed_mask,
        route_count=count,
        name="already-removed mask",
    )
    if any(required[index] and already[index] for index in range(count)):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.AXIS_MISMATCH,
            "a route cannot be both required-live and already removed",
        )
    selectable = tuple(eligible[index] and not already[index] for index in range(count))
    required_indices = tuple(index for index, value in enumerate(required) if value)
    if any(not selectable[index] for index in preferred) or not set(
        required_indices
    ).issubset(preferred):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INADMISSIBLE_PREFERRED_ACTION,
            "preferred action must contain every required prior-rung route and may not select an ineligible or already-removed route",
        )
    preferred_set = set(preferred)
    shell_set = set(shell)
    droppable = tuple(index for index in preferred if not required[index])
    addable = tuple(
        index
        for index, admitted in enumerate(selectable)
        if admitted and index not in preferred_set
    )
    addable_outside = tuple(index for index in addable if index not in shell_set)
    droppable_outside = tuple(index for index in droppable if index not in shell_set)
    addable_inside = tuple(index for index in addable if index in shell_set)
    eligible_hash = content_hash(eligible, prefix=_PREFIX + "eligible-mask-")
    required_hash = content_hash(required, prefix=_PREFIX + "required-mask-")
    already_hash = content_hash(already, prefix=_PREFIX + "already-removed-mask-")
    selectable_hash = content_hash(selectable, prefix=_PREFIX + "selectable-mask-")
    preferred_hash = content_hash(preferred, prefix=_PREFIX + "preferred-action-")
    shell_hash = content_hash(shell, prefix=_PREFIX + "shell-")
    axis_commitment = content_hash(
        {
            "rule": OUTSIDE_PAIR_COVER_RULE,
            "route_count": count,
            "fixed_cardinality": len(preferred),
            "preferred_action_index_hash": preferred_hash,
            "shell_index_hash": shell_hash,
            "selectable_mask_hash": selectable_hash,
            "droppable_preferred": droppable,
            "addable_outside": addable_outside,
            "droppable_outside": droppable_outside,
            "addable_inside": addable_inside,
        },
        prefix=_PREFIX + "pair-axis-commitment-",
    )
    receipt = V1SignedWorkVariableShellOutsidePairCoverReceipt(
        route_count=count,
        fixed_cardinality=len(preferred),
        preferred_action_index_hash=preferred_hash,
        shell_index_hash=shell_hash,
        eligible_mask_hash=eligible_hash,
        required_mask_hash=required_hash,
        already_removed_mask_hash=already_hash,
        selectable_mask_hash=selectable_hash,
        # This is a descriptive count over the complete preferred action.
        # Required prior-rung routes remain preferred even though they may
        # not be dropped and therefore do not enter ``droppable_outside``.
        preferred_outside_shell_count=sum(
            index not in shell_set for index in preferred
        ),
        droppable_preferred_count=len(droppable),
        addable_outside_shell_count=len(addable_outside),
        droppable_preferred_outside_shell_count=len(droppable_outside),
        addable_inside_shell_count=len(addable_inside),
        add_outside_rectangle_pair_count=len(droppable) * len(addable_outside),
        drop_outside_add_inside_rectangle_pair_count=(
            len(droppable_outside) * len(addable_inside)
        ),
        complete_pair_count=(
            len(droppable) * len(addable_outside)
            + len(droppable_outside) * len(addable_inside)
        ),
        pair_axis_commitment_hash=axis_commitment,
        _issuer_token=_COVER_TOKEN,
    )
    return V1SignedWorkVariableShellOutsidePairCoverEnumeration(
        route_count=count,
        preferred=preferred,
        shell=shell,
        selectable=selectable,
        required=required_indices,
        droppable=droppable,
        addable=addable,
        addable_outside=addable_outside,
        droppable_outside=droppable_outside,
        addable_inside=addable_inside,
        receipt=receipt,
        _issuer_token=_COVER_TOKEN,
    )


class V1SignedWorkVariableShellOutsideExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsideCUDAExecutionDesign:
    route_tile_size: int
    pair_tile_size: int
    hbm_ceiling_fraction: float
    execution_mode: V1SignedWorkVariableShellOutsideExecutionMode
    deterministic_fp64_required: bool = True
    forbid_route_scale_d2h: bool = True
    forbid_route_scale_host_retention: bool = True
    schema: str = OUTSIDE_CUDA_DESIGN_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DESIGN_TOKEN:
            raise TypeError("outside CUDA execution designs are issuer-only")
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        object.__setattr__(
            self,
            "pair_tile_size",
            _positive_int(self.pair_tile_size, name="pair tile size"),
        )
        ceiling = _finite(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if not 0.0 < ceiling <= 0.85:
            raise ValueError("outside CUDA HBM ceiling must lie in (0,0.85]")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if (
            type(self.execution_mode)
            is not V1SignedWorkVariableShellOutsideExecutionMode
        ):
            raise TypeError("outside execution mode has another type")
        if (
            self.deterministic_fp64_required is not True
            or self.forbid_route_scale_d2h is not True
            or self.forbid_route_scale_host_retention is not True
            or self.schema != OUTSIDE_CUDA_DESIGN_SCHEMA
        ):
            raise ValueError("outside CUDA execution invariants changed")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


def build_v1_signed_work_variable_shell_outside_cuda_design(
    *,
    route_tile_size: int = 131_072,
    pair_tile_size: int = 131_072,
    hbm_ceiling_fraction: float = 0.85,
    execution_mode: V1SignedWorkVariableShellOutsideExecutionMode = (
        V1SignedWorkVariableShellOutsideExecutionMode.CUDA_PRODUCTION
    ),
) -> V1SignedWorkVariableShellOutsideCUDAExecutionDesign:
    return V1SignedWorkVariableShellOutsideCUDAExecutionDesign(
        route_tile_size=route_tile_size,
        pair_tile_size=pair_tile_size,
        hbm_ceiling_fraction=hbm_ceiling_fraction,
        execution_mode=execution_mode,
        _issuer_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsideCUDAHBMForecast:
    design_hash: str
    route_count: int
    y_f_count: int
    block_multiplier_count: int
    provider_workspace_bytes: int
    full_axis_stationarity_bytes: int
    full_axis_membership_bytes: int
    live_dual_state_bytes: int
    route_tile_workspace_bytes: int
    pair_tile_workspace_bytes: int
    incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    issuer_route_scale_d2h_bytes: int = 0
    schema: str = OUTSIDE_CUDA_HBM_SCHEMA
    _issuer_token: InitVar[object] = None

    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _issuer_token: object, _admission_memory: tuple[int, int] | None) -> None:
        if _issuer_token is not _FORECAST_TOKEN:
            raise TypeError("outside CUDA HBM forecasts are issuer-only")
        object.__setattr__(
            self, "design_hash", _identity(self.design_hash, name="design hash")
        )
        for name in (
            "route_count",
            "y_f_count",
            "block_multiplier_count",
            "provider_workspace_bytes",
            "full_axis_stationarity_bytes",
            "full_axis_membership_bytes",
            "live_dual_state_bytes",
            "route_tile_workspace_bytes",
            "pair_tile_workspace_bytes",
            "incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_peak_reserved_bytes",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
            "issuer_route_scale_d2h_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if self.route_count <= 0 or self.y_f_count <= 0:
            raise ValueError("outside HBM forecast axes must be nonempty")
        expected_stationarity = _F64_BYTES * self.route_count
        expected_membership = 4 * self.route_count
        expected_dual = _F64_BYTES * (self.y_f_count + 1 + self.block_multiplier_count)
        if (
            self.full_axis_stationarity_bytes != expected_stationarity
            or self.full_axis_membership_bytes != expected_membership
            or self.live_dual_state_bytes != expected_dual
        ):
            raise ValueError("outside HBM persistent formula changed")
        expected_incremental = (
            expected_stationarity
            + expected_membership
            + expected_dual
            + self.provider_workspace_bytes
            + max(self.route_tile_workspace_bytes, self.pair_tile_workspace_bytes)
        )
        if (
            self.incremental_peak_bytes != expected_incremental
            or self.projected_peak_reserved_bytes
            != self.current_reserved_device_bytes + expected_incremental
        ):
            raise ValueError("outside HBM peak formula does not close")
        ceiling = _finite(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        expected_ceiling = math.floor(self.device_total_memory_bytes * ceiling)
        available, allocated = _admission_memory or (
            self.device_free_memory_bytes, self.current_reserved_device_bytes
        )
        expected_admitted = (
            allocated + expected_incremental <= expected_ceiling
            and expected_incremental <= available
        )
        if (
            self.hbm_ceiling_bytes != expected_ceiling
            or type(self.admitted) is not bool
            or self.admitted is not expected_admitted
            or self.issuer_route_scale_d2h_bytes != 0
            or self.schema != OUTSIDE_CUDA_HBM_SCHEMA
        ):
            raise ValueError("outside HBM admission or no-D2H invariant changed")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hbm-")


def build_v1_signed_work_variable_shell_outside_cuda_hbm_forecast(
    *,
    design: V1SignedWorkVariableShellOutsideCUDAExecutionDesign,
    route_count: int,
    y_f_count: int,
    block_multiplier_count: int,
    provider_workspace_bytes: int,
    current_reserved_device_bytes: int,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
) -> V1SignedWorkVariableShellOutsideCUDAHBMForecast:
    admission_memory = None
    if type(design) is not V1SignedWorkVariableShellOutsideCUDAExecutionDesign:
        raise TypeError("outside HBM forecast requires the exact CUDA design")
    count = _positive_int(route_count, name="route count")
    yf = _positive_int(y_f_count, name="y_F count")
    blocks = _nonnegative_int(block_multiplier_count, name="block multiplier count")
    provider = _nonnegative_int(
        provider_workspace_bytes, name="provider workspace bytes"
    )
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    if (
        design.execution_mode
        is V1SignedWorkVariableShellOutsideExecutionMode.CUDA_PRODUCTION
    ):
        if torch is None or not torch.cuda.is_available():
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.CUDA_REQUIRED,
                "production outside HBM forecast requires live CUDA memory facts",
            )
        device = torch.device("cuda", torch.cuda.current_device())
        live_free, live_total = torch.cuda.mem_get_info(device)
        admission_memory = (
            cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(live_free)),
            int(torch.cuda.memory_allocated(device)),
        )
        total = _positive_int(int(live_total), name="live device total memory bytes")
        free = _nonnegative_int(int(live_free), name="live device free memory bytes")
        reserved = _nonnegative_int(
            int(torch.cuda.memory_reserved(device)),
            name="live current reserved device bytes",
        )
    elif free > total or reserved > total:
        raise ValueError("reference HBM facts exceed total device memory")
    stationarity = _F64_BYTES * count
    memberships = 4 * count
    dual = _F64_BYTES * (yf + 1 + blocks)
    route_workspace = _F64_BYTES * design.route_tile_size * 4
    pair_workspace = design.pair_tile_size * (2 * _I64_BYTES + 4 * _F64_BYTES)
    incremental = (
        stationarity
        + memberships
        + dual
        + provider
        + max(route_workspace, pair_workspace)
    )
    available, allocated = admission_memory or (free, reserved)
    peak = reserved + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    return V1SignedWorkVariableShellOutsideCUDAHBMForecast(
        design_hash=design.design_hash,
        route_count=count,
        y_f_count=yf,
        block_multiplier_count=blocks,
        provider_workspace_bytes=provider,
        full_axis_stationarity_bytes=stationarity,
        full_axis_membership_bytes=memberships,
        live_dual_state_bytes=dual,
        route_tile_workspace_bytes=route_workspace,
        pair_tile_workspace_bytes=pair_workspace,
        incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=(allocated + incremental <= ceiling and incremental <= available),
        _issuer_token=_FORECAST_TOKEN,
        _admission_memory=admission_memory,
    )


def _tensor_invariant(value: object) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("lower-plane custody requires exact torch tensors")
    return (
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.data_ptr()),
        int(value.storage_offset()),
        int(getattr(value, "_version", -1)),
        int(id(value)),
    )


def _cpu_tensor_content_commitment(value: object) -> str:
    """Return an exact bitwise commitment for reference-owned CPU storage.

    PyTorch's mutation version is useful metadata but ``tensor.data`` writes
    can bypass it.  The reference seam is CPU-only, so it can cheaply bind the
    complete bytes without any device-to-host transfer.  The future production
    provider must supply an equivalent source-owned seal without materializing
    a route-scale host copy; this helper is intentionally unavailable on CUDA.
    """

    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("tensor content custody requires one exact torch tensor")
    if value.device.type != "cpu" or not value.is_contiguous():
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            "reference content commitment requires contiguous CPU storage",
        )
    raw = value.detach().view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _require_fp64_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or (device is not None and value.device != device)
        or not bool(torch.isfinite(value).all().item())
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous FP64 tensor on the bound device",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellFullAxisLowerPlaneBindingReceipt:
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    source_group_hash: str
    branch_id: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    resource_alias_id: str
    route_count: int
    y_f_count: int
    block_multiplier_count: int
    y_f_mutation_identity_hash: str
    resource_multiplier_mutation_identity_hash: str
    block_multiplier_mutation_identity_hash: str
    stationarity_mutation_identity_hash: str
    y_f_content_commitment_hash: str
    resource_multiplier_content_commitment_hash: str
    block_multiplier_content_commitment_hash: str
    stationarity_content_commitment_hash: str
    design_hash: str
    hbm_forecast_hash: str
    same_prefix_a_row_replay_provider_bound: bool
    reference_precomputed_stationarity_only: bool
    source_owned_exact_bitwise_content_custody: bool
    reference_route_scale_host_input: bool
    caller_supplied_route_scale_host_coefficient_bytes: int
    issuer_route_scale_d2h_bytes: int = 0
    solver_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    commit_authority: bool = False
    schema: str = OUTSIDE_LOWER_PLANE_BINDING_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _BINDING_TOKEN:
            raise TypeError("full-axis lower-plane bindings are issuer-only")
        for name in (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "source_group_hash",
            "branch_id",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "resource_alias_id",
            "y_f_mutation_identity_hash",
            "resource_multiplier_mutation_identity_hash",
            "block_multiplier_mutation_identity_hash",
            "stationarity_mutation_identity_hash",
            "y_f_content_commitment_hash",
            "resource_multiplier_content_commitment_hash",
            "block_multiplier_content_commitment_hash",
            "stationarity_content_commitment_hash",
            "design_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self, "y_f_count", _positive_int(self.y_f_count, name="y_F count")
        )
        object.__setattr__(
            self,
            "block_multiplier_count",
            _nonnegative_int(
                self.block_multiplier_count, name="block multiplier count"
            ),
        )
        if (
            type(self.same_prefix_a_row_replay_provider_bound) is not bool
            or type(self.reference_precomputed_stationarity_only) is not bool
            or type(self.source_owned_exact_bitwise_content_custody) is not bool
            or type(self.reference_route_scale_host_input) is not bool
        ):
            raise TypeError("lower-plane replay flags must be exact bool")
        if (
            self.same_prefix_a_row_replay_provider_bound
            == self.reference_precomputed_stationarity_only
        ):
            raise ValueError(
                "lower-plane binding must be exactly production replay or reference-only"
            )
        host_bytes = _nonnegative_int(
            self.caller_supplied_route_scale_host_coefficient_bytes,
            name="caller-supplied route-scale host coefficient bytes",
        )
        object.__setattr__(
            self,
            "caller_supplied_route_scale_host_coefficient_bytes",
            host_bytes,
        )
        if self.reference_precomputed_stationarity_only:
            if (
                self.source_owned_exact_bitwise_content_custody is not True
                or self.reference_route_scale_host_input is not True
                or host_bytes != self.route_count * _F64_BYTES
            ):
                raise ValueError(
                    "reference lower plane lacks exact CPU content custody"
                )
        elif self.reference_route_scale_host_input or host_bytes != 0:
            raise ValueError(
                "production replay may not import route-scale host coefficients"
            )
        issuer_d2h = _nonnegative_int(
            self.issuer_route_scale_d2h_bytes,
            name="issuer route-scale D2H bytes",
        )
        object.__setattr__(self, "issuer_route_scale_d2h_bytes", issuer_d2h)
        if issuer_d2h != 0:
            raise ValueError("lower-plane binding performed forbidden route-scale D2H")
        _closed_authority(
            (
                self.solver_authority,
                self.action_invariance_authority,
                self.c_stream_authority,
                self.commit_authority,
            ),
            owner="full-axis lower-plane binding",
        )
        if self.schema != OUTSIDE_LOWER_PLANE_BINDING_SCHEMA:
            raise ValueError("unsupported full-axis lower-plane binding schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "lower-plane-binding-")


class V1SignedWorkVariableShellFullAxisLowerPlaneLease:
    """Nonserializable process-local borrow of one exact lower plane."""

    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "binding_receipt_hash",
        "mutation_identity_hash",
        "sample_function_source_only",
        "solver_authority",
        "action_invariance_authority",
        "c_stream_authority",
        "commit_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1SignedWorkVariableShellFullAxisLowerPlaneHandle",
        token: object,
    ) -> None:
        if type(owner) is not V1SignedWorkVariableShellFullAxisLowerPlaneHandle:
            raise TypeError("lower-plane lease requires the exact owner")
        self._owner = owner
        self._token = token
        self._closed = False
        self.binding_receipt_hash = owner.binding.receipt_hash
        self.mutation_identity_hash = owner.mutation_identity_hash
        self.sample_function_source_only = True
        self.solver_authority = False
        self.action_invariance_authority = False
        self.c_stream_authority = False
        self.commit_authority = False
        self.schema = OUTSIDE_LOWER_PLANE_LEASE_SCHEMA

    def live_tensors(self) -> tuple[Any, Any, Any, Any]:
        if self._closed:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.EVICTED,
                "full-axis lower-plane lease is closed",
            )
        if self._owner.mutation_identity_hash != self.mutation_identity_hash:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.MUTATED_CUSTODY,
                "full-axis lower-plane mutation identity changed",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._owner._close_lease(self._token)
        finally:
            # Closing is terminal even when custody validation correctly
            # refuses; the owner's token cleanup runs in its own ``finally``.
            self._closed = True

    def __enter__(self) -> "V1SignedWorkVariableShellFullAxisLowerPlaneLease":
        self.live_tensors()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("full-axis lower-plane leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("full-axis lower-plane leases are nonserializable")


class V1SignedWorkVariableShellFullAxisLowerPlaneHandle:
    __slots__ = (
        "_y_f",
        "_resource_multiplier",
        "_block_multipliers",
        "_stationarity",
        "_invariants",
        "_content_commitments",
        "_active_leases",
        "binding",
        "design",
        "forecast",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        y_f: object,
        resource_multiplier: object,
        block_multipliers: object,
        stationarity_coefficients: object,
        binding: V1SignedWorkVariableShellFullAxisLowerPlaneBindingReceipt,
        design: V1SignedWorkVariableShellOutsideCUDAExecutionDesign,
        forecast: V1SignedWorkVariableShellOutsideCUDAHBMForecast,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _HANDLE_TOKEN:
            raise TypeError("full-axis lower-plane handles are issuer-only")
        if (
            type(binding)
            is not V1SignedWorkVariableShellFullAxisLowerPlaneBindingReceipt
            or type(design) is not V1SignedWorkVariableShellOutsideCUDAExecutionDesign
            or type(forecast) is not V1SignedWorkVariableShellOutsideCUDAHBMForecast
        ):
            raise TypeError(
                "lower-plane handle requires exact binding, design, and forecast"
            )
        stationarity = _require_fp64_tensor(
            stationarity_coefficients,
            name="full-axis stationarity coefficients",
            shape=(binding.route_count,),
        )
        device = stationarity.device
        y_f_tensor = _require_fp64_tensor(
            y_f,
            name="y_F",
            shape=(binding.y_f_count,),
            device=device,
        )
        resource_tensor = _require_fp64_tensor(
            resource_multiplier,
            name="resource multiplier",
            shape=(1,),
            device=device,
        )
        block_tensor = _require_fp64_tensor(
            block_multipliers,
            name="block multipliers",
            shape=(binding.block_multiplier_count,),
            device=device,
        )
        if bool((resource_tensor < 0.0).any().item()) or bool(
            (block_tensor < 0.0).any().item()
        ):
            raise ValueError("resource and block multipliers must be nonnegative")
        if (
            design.execution_mode
            is V1SignedWorkVariableShellOutsideExecutionMode.CUDA_PRODUCTION
        ):
            if device.type != "cuda":
                _refuse(
                    V1SignedWorkVariableShellOutsideExclusionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production full-axis lower-plane custody requires CUDA",
                )
            if not binding.same_prefix_a_row_replay_provider_bound:
                _refuse(
                    V1SignedWorkVariableShellOutsideExclusionRefusalCode.SAME_PREFIX_A_ROW_REPLAY_PROVIDER_REQUIRED,
                    "production lower plane lacks typed same-prefix A-row replay",
                )
        if (
            binding.design_hash != design.design_hash
            or binding.hbm_forecast_hash != forecast.forecast_hash
            or forecast.design_hash != design.design_hash
            or forecast.route_count != binding.route_count
            or not forecast.admitted
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.STALE_IDENTITY,
                "lower-plane binding, design, and HBM forecast do not align",
            )
        tensors = (y_f_tensor, resource_tensor, block_tensor, stationarity)
        invariants = tuple(_tensor_invariant(value) for value in tensors)
        expected_hashes = tuple(
            content_hash(value, prefix=_PREFIX + "tensor-mutation-")
            for value in invariants
        )
        if expected_hashes != (
            binding.y_f_mutation_identity_hash,
            binding.resource_multiplier_mutation_identity_hash,
            binding.block_multiplier_mutation_identity_hash,
            binding.stationarity_mutation_identity_hash,
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.MUTATED_CUSTODY,
                "lower-plane tensors do not match their binding identities",
            )
        if not binding.reference_precomputed_stationarity_only:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.PRODUCTION_PROVIDER_INTEGRATION_PENDING,
                "device-resident source-owned content custody is not landed in this reference handle",
            )
        content_commitments = tuple(
            _cpu_tensor_content_commitment(value) for value in tensors
        )
        if content_commitments != (
            binding.y_f_content_commitment_hash,
            binding.resource_multiplier_content_commitment_hash,
            binding.block_multiplier_content_commitment_hash,
            binding.stationarity_content_commitment_hash,
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.MUTATED_CUSTODY,
                "lower-plane tensor bytes do not match their source-owned seals",
            )
        self._y_f = y_f_tensor
        self._resource_multiplier = resource_tensor
        self._block_multipliers = block_tensor
        self._stationarity = stationarity
        self._invariants = invariants
        self._content_commitments = content_commitments
        self._active_leases: set[object] = set()
        self.binding = binding
        self.design = design
        self.forecast = forecast

    def _require_live(self) -> tuple[Any, Any, Any, Any]:
        tensors = (
            self._y_f,
            self._resource_multiplier,
            self._block_multipliers,
            self._stationarity,
        )
        if any(value is None for value in tensors):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.EVICTED,
                "full-axis lower plane was evicted",
            )
        if tuple(_tensor_invariant(value) for value in tensors) != self._invariants:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.MUTATED_CUSTODY,
                "full-axis lower-plane tensors changed after sealing",
            )
        if (
            tuple(_cpu_tensor_content_commitment(value) for value in tensors)
            != self._content_commitments
        ):
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.MUTATED_CUSTODY,
                "full-axis lower-plane tensor bytes changed after sealing",
            )
        return tensors  # type: ignore[return-value]

    @property
    def mutation_identity_hash(self) -> str:
        self._require_live()
        return content_hash(
            (self._invariants, self._content_commitments),
            prefix=_PREFIX + "lower-plane-content-custody-",
        )

    def borrow(self) -> V1SignedWorkVariableShellFullAxisLowerPlaneLease:
        self._require_live()
        token = object()
        self._active_leases.add(token)
        return V1SignedWorkVariableShellFullAxisLowerPlaneLease(owner=self, token=token)

    def _close_lease(self, token: object) -> None:
        if token not in self._active_leases:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.STALE_IDENTITY,
                "lower-plane lease token is not active",
            )
        try:
            self._require_live()
        finally:
            # A mutation refusal must not leak the process-local borrow token.
            # The refusal still propagates, but the owner can now be evicted.
            self._active_leases.remove(token)

    def evict(self) -> None:
        if self._active_leases:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.ACTIVE_LEASE,
                "cannot evict a full-axis lower plane with an active lease",
            )
        self._y_f = None
        self._resource_multiplier = None
        self._block_multipliers = None
        self._stationarity = None
        self._invariants = ()
        self._content_commitments = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("full-axis lower-plane handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("full-axis lower-plane handles are nonserializable")


def build_v1_signed_work_variable_shell_reference_lower_plane_handle(
    *,
    design: V1SignedWorkVariableShellOutsideCUDAExecutionDesign,
    forecast: V1SignedWorkVariableShellOutsideCUDAHBMForecast,
    model_identity_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    route_id_axis_hash: str,
    registered_d_prefix_hash: str,
    source_group_hash: str,
    branch_id: str,
    resource_axis_hash: str,
    resource_source_receipt_hash: str,
    resource_alias_id: str,
    y_f: object,
    resource_multiplier: object,
    block_multipliers: object,
    stationarity_coefficients: object,
) -> V1SignedWorkVariableShellFullAxisLowerPlaneHandle:
    """Seal a CPU-only precomputed plane for reference tests, never production."""

    if type(design) is not V1SignedWorkVariableShellOutsideCUDAExecutionDesign:
        raise TypeError("reference lower plane requires the exact design")
    if (
        design.execution_mode
        is not V1SignedWorkVariableShellOutsideExecutionMode.CPU_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "precomputed stationarity is allowed only in CPU reference mode",
        )
    if torch is None or any(
        type(value) is not torch.Tensor
        for value in (
            y_f,
            resource_multiplier,
            block_multipliers,
            stationarity_coefficients,
        )
    ):
        raise TypeError("reference lower plane requires torch tensors")
    stationarity_source = _require_fp64_tensor(
        stationarity_coefficients,
        name="reference full-axis stationarity coefficients",
        shape=(int(stationarity_coefficients.numel()),),
    )
    if stationarity_source.device.type != "cpu":
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INVALID_TYPE,
            "reference lower-plane input must already reside on CPU; the issuer performs no D2H",
        )
    device = stationarity_source.device
    y_f_source = _require_fp64_tensor(
        y_f,
        name="reference y_F",
        shape=(int(y_f.numel()),),
        device=device,
    )
    resource_source = _require_fp64_tensor(
        resource_multiplier,
        name="reference resource multiplier",
        shape=(1,),
        device=device,
    )
    block_source = _require_fp64_tensor(
        block_multipliers,
        name="reference block multipliers",
        shape=(int(block_multipliers.numel()),),
        device=device,
    )
    # Own the storage instead of trusting continued caller custody.  A
    # bitwise content commitment below also detects writes through ``.data``
    # to the tensors returned by a live reference lease.
    tensors = tuple(
        value.detach().clone(memory_format=torch.contiguous_format)
        for value in (
            y_f_source,
            resource_source,
            block_source,
            stationarity_source,
        )
    )
    invariants = tuple(_tensor_invariant(value) for value in tensors)
    content_commitments = tuple(
        _cpu_tensor_content_commitment(value) for value in tensors
    )
    binding = V1SignedWorkVariableShellFullAxisLowerPlaneBindingReceipt(
        model_identity_hash=model_identity_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        registered_d_prefix_hash=registered_d_prefix_hash,
        source_group_hash=source_group_hash,
        branch_id=branch_id,
        resource_axis_hash=resource_axis_hash,
        resource_source_receipt_hash=resource_source_receipt_hash,
        resource_alias_id=resource_alias_id,
        route_count=int(tensors[3].numel()),
        y_f_count=int(tensors[0].numel()),
        block_multiplier_count=int(tensors[2].numel()),
        y_f_mutation_identity_hash=content_hash(
            invariants[0], prefix=_PREFIX + "tensor-mutation-"
        ),
        resource_multiplier_mutation_identity_hash=content_hash(
            invariants[1], prefix=_PREFIX + "tensor-mutation-"
        ),
        block_multiplier_mutation_identity_hash=content_hash(
            invariants[2], prefix=_PREFIX + "tensor-mutation-"
        ),
        stationarity_mutation_identity_hash=content_hash(
            invariants[3], prefix=_PREFIX + "tensor-mutation-"
        ),
        y_f_content_commitment_hash=content_commitments[0],
        resource_multiplier_content_commitment_hash=content_commitments[1],
        block_multiplier_content_commitment_hash=content_commitments[2],
        stationarity_content_commitment_hash=content_commitments[3],
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        same_prefix_a_row_replay_provider_bound=False,
        reference_precomputed_stationarity_only=True,
        source_owned_exact_bitwise_content_custody=True,
        reference_route_scale_host_input=True,
        caller_supplied_route_scale_host_coefficient_bytes=(
            int(tensors[3].numel()) * _F64_BYTES
        ),
        _issuer_token=_BINDING_TOKEN,
    )
    return V1SignedWorkVariableShellFullAxisLowerPlaneHandle(
        y_f=tensors[0],
        resource_multiplier=tensors[1],
        block_multipliers=tensors[2],
        stationarity_coefficients=tensors[3],
        binding=binding,
        design=design,
        forecast=forecast,
        _issuer_token=_HANDLE_TOKEN,
    )


class V1SignedWorkVariableShellSamePrefixARowReplayProvider(ABC):
    """Nominal interface for source-bound full-axis corrected-A replay.

    A conforming provider may own an exact empirical F support plane without
    owning the remaining resource, block-capacity, and cardinality duals.
    Those are separate gates because an F plane alone is not a complete pair
    lower bound.
    """

    @property
    @abstractmethod
    def provider_receipt_hash(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def resource_block_duals_source_bound(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def validate_same_prefix_source(
        self,
        source_receipt: V1SignedWorkVariableShellSourceReceipt,
    ) -> str:
        raise NotImplementedError


def issue_v1_signed_work_variable_shell_outside_exclusion_cuda(
    *,
    source_receipt: V1SignedWorkVariableShellSourceReceipt,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    design: V1SignedWorkVariableShellOutsideCUDAExecutionDesign,
    hbm_forecast: V1SignedWorkVariableShellOutsideCUDAHBMForecast,
    same_prefix_a_row_replay_provider: (
        V1SignedWorkVariableShellSamePrefixARowReplayProvider | None
    ) = None,
) -> NoReturn:
    """Refuse until a typed same-prefix full-axis A-row provider is landed."""

    if same_prefix_a_row_replay_provider is None:
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.SAME_PREFIX_A_ROW_REPLAY_PROVIDER_REQUIRED,
            "full-axis exclusion cannot reuse bounded-shell rows or caller-provided coefficients; a typed same-prefix corrected-A row replay provider is required",
        )
    if not isinstance(
        same_prefix_a_row_replay_provider,
        V1SignedWorkVariableShellSamePrefixARowReplayProvider,
    ):
        raise TypeError("same-prefix A-row replay provider has another type")
    if (
        type(source_receipt) is not V1SignedWorkVariableShellSourceReceipt
        or type(pair_cover) is not V1SignedWorkVariableShellOutsidePairCoverEnumeration
        or type(design) is not V1SignedWorkVariableShellOutsideCUDAExecutionDesign
        or type(hbm_forecast) is not V1SignedWorkVariableShellOutsideCUDAHBMForecast
    ):
        raise TypeError("production outside issuer received another typed seam")
    same_prefix_a_row_replay_provider.validate_same_prefix_source(source_receipt)
    if not same_prefix_a_row_replay_provider.resource_block_duals_source_bound:
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.RESOURCE_BLOCK_CARDINALITY_DUAL_PROVIDER_REQUIRED,
            "the corrected-A empirical support plane is source-bound, but resource, block-capacity, and cardinality duals are not yet source-bound",
        )
    _refuse(
        V1SignedWorkVariableShellOutsideExclusionRefusalCode.PRODUCTION_PROVIDER_INTEGRATION_PENDING,
        "all nominal provider gates are present but the complete production pair-cell issuer is not landed",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsidePairDualBoundReference:
    pair: V1SignedWorkVariableShellOutsidePairIdentity
    pair_cover_receipt_hash: str
    route_count: int
    exact_dual_lower_numerator: int
    exact_dual_lower_denominator: int
    dual_lower_center: float
    dual_lower_bound_outward: float
    independent_pair_box_numerator: int
    independent_pair_box_denominator: int
    coupled_pair_box_numerator: int
    coupled_pair_box_denominator: int
    caller_supplied_route_scale_host_coefficient_bytes: int
    exact_binary64_rational_evaluation: bool = True
    continuous_box_relaxation_exact: bool = True
    required_and_forbidden_coordinate_constraints_exact: bool = True
    pair_halfspace_constraint_exact: bool = True
    fixed_cardinality_primal_constraint_enforced: bool = False
    resource_primal_constraint_enforced: bool = False
    block_capacity_primal_constraint_enforced: bool = False
    q_primal_constraint_enforced: bool = False
    caller_lower_plane_constraint_dualization_verified: bool = False
    exact_retained_domain_optimization_claimed: bool = False
    reference_route_scale_host_input: bool = True
    issuer_route_scale_d2h_bytes: int = 0
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    commit_authority: bool = False
    box_rule: str = OUTSIDE_PAIR_BOX_RULE
    reference_domain_rule: str = OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE
    schema: str = OUTSIDE_PAIR_BOUND_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _BOUND_TOKEN:
            raise TypeError("outside pair dual-bound references are issuer-only")
        if type(self.pair) is not V1SignedWorkVariableShellOutsidePairIdentity:
            raise TypeError("pair dual bound requires its exact pair identity")
        object.__setattr__(
            self,
            "pair_cover_receipt_hash",
            _identity(self.pair_cover_receipt_hash, name="pair-cover receipt hash"),
        )
        count = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "route_count", count)
        host_bytes = _nonnegative_int(
            self.caller_supplied_route_scale_host_coefficient_bytes,
            name="caller-supplied route-scale host coefficient bytes",
        )
        object.__setattr__(
            self,
            "caller_supplied_route_scale_host_coefficient_bytes",
            host_bytes,
        )
        issuer_d2h = _nonnegative_int(
            self.issuer_route_scale_d2h_bytes,
            name="issuer route-scale D2H bytes",
        )
        object.__setattr__(self, "issuer_route_scale_d2h_bytes", issuer_d2h)
        for denominator in (
            self.exact_dual_lower_denominator,
            self.independent_pair_box_denominator,
            self.coupled_pair_box_denominator,
        ):
            if _positive_int(denominator, name="exact denominator") != denominator:
                raise ValueError("exact denominator changed")
        center = _finite(self.dual_lower_center, name="dual lower center")
        lower = _finite(self.dual_lower_bound_outward, name="outward dual lower")
        if lower > center:
            raise ValueError("outward dual lower bound is above its center")
        required_true = (
            self.exact_binary64_rational_evaluation,
            self.continuous_box_relaxation_exact,
            self.required_and_forbidden_coordinate_constraints_exact,
            self.pair_halfspace_constraint_exact,
            self.reference_route_scale_host_input,
        )
        required_false = (
            self.fixed_cardinality_primal_constraint_enforced,
            self.resource_primal_constraint_enforced,
            self.block_capacity_primal_constraint_enforced,
            self.q_primal_constraint_enforced,
            self.caller_lower_plane_constraint_dualization_verified,
            self.exact_retained_domain_optimization_claimed,
        )
        if (
            any(type(value) is not bool or value is not True for value in required_true)
            or any(
                type(value) is not bool or value is not False
                for value in required_false
            )
            or host_bytes != count * _F64_BYTES
            or issuer_d2h != 0
            or self.box_rule != OUTSIDE_PAIR_BOX_RULE
            or self.reference_domain_rule != OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE
            or self.schema != OUTSIDE_PAIR_BOUND_SCHEMA
        ):
            raise ValueError("outside pair reference semantics changed")
        _closed_authority(
            (
                self.action_invariance_authority,
                self.c_stream_authority,
                self.commit_authority,
            ),
            owner="outside pair dual-bound reference",
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "pair-bound-")


def evaluate_v1_signed_work_variable_shell_outside_pair_dual_bounds_reference(
    *,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    stationarity_coefficients: Sequence[Real],
    linear_constant: Real,
) -> tuple[V1SignedWorkVariableShellOutsidePairDualBoundReference, ...]:
    """Evaluate the box-relaxed pair plane in exact rational arithmetic.

    The evaluator enforces only the continuous unit box, fixed required and
    forbidden coordinates, and ``z_p <= z_q``.  Fixed cardinality, resource,
    block-capacity, and Q constraints are not primal constraints here.  A
    caller's coefficients may contain their dual multipliers, but this
    reference seam neither observes nor verifies that provenance.  The result
    is therefore a sound lower bound for any retained feasible subset, not a
    claim to its exact optimum.
    """

    if type(pair_cover) is not V1SignedWorkVariableShellOutsidePairCoverEnumeration:
        raise TypeError("pair reference requires the exact live pair cover")
    count = pair_cover.receipt.route_count
    if (
        isinstance(stationarity_coefficients, (str, bytes))
        or len(stationarity_coefficients) != count
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.AXIS_MISMATCH,
            "stationarity coefficients do not span the complete route axis",
        )
    coefficients = tuple(
        _fraction(value, name="stationarity coefficient")
        for value in stationarity_coefficients
    )
    constant = _fraction(linear_constant, name="dual linear constant")
    free = pair_cover.selectable_mask
    required = set(pair_cover.required_action_indices)
    independent_terms = tuple(
        (
            value
            if index in required
            else min(Fraction(0), value) if free[index] else Fraction(0)
        )
        for index, value in enumerate(coefficients)
    )
    unconstrained = constant + sum(independent_terms, Fraction(0))
    bounds: list[V1SignedWorkVariableShellOutsidePairDualBoundReference] = []
    for pair in pair_cover.iter_pairs():
        p = pair.preferred_selected_index
        q = pair.preferred_unselected_index
        if not free[p] or not free[q]:
            _refuse(
                V1SignedWorkVariableShellOutsideExclusionRefusalCode.INCOMPLETE_PAIR_AXIS,
                "pair axis contains a nonfree coordinate",
            )
        independent_pair = independent_terms[p] + independent_terms[q]
        coupled_pair = min(
            Fraction(0),
            coefficients[q],
            coefficients[p] + coefficients[q],
        )
        exact = unconstrained - independent_pair + coupled_pair
        center = float(exact)
        lower = _down_from_fraction(exact)
        bounds.append(
            V1SignedWorkVariableShellOutsidePairDualBoundReference(
                pair=pair,
                pair_cover_receipt_hash=pair_cover.receipt.receipt_hash,
                route_count=count,
                exact_dual_lower_numerator=exact.numerator,
                exact_dual_lower_denominator=exact.denominator,
                dual_lower_center=center,
                dual_lower_bound_outward=lower,
                independent_pair_box_numerator=independent_pair.numerator,
                independent_pair_box_denominator=independent_pair.denominator,
                coupled_pair_box_numerator=coupled_pair.numerator,
                coupled_pair_box_denominator=coupled_pair.denominator,
                caller_supplied_route_scale_host_coefficient_bytes=(count * _F64_BYTES),
                _issuer_token=_BOUND_TOKEN,
            )
        )
    return tuple(bounds)


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsideExclusionReferenceReceipt:
    pair_cover_receipt_hash: str
    route_count: int
    complete_route_stationarity_scan_count: int
    complete_pair_scan_count: int
    add_outside_rectangle_scan_count: int
    drop_outside_add_inside_rectangle_scan_count: int
    worst_pair: V1SignedWorkVariableShellOutsidePairIdentity | None
    minimum_pair_dual_lower_bound: float
    preferred_f_upper_bound: float
    strict_margin_lower_bound: float
    all_pair_cells_strictly_excluded: bool
    vacuous_no_outside_pair: bool
    caller_supplied_route_scale_host_coefficient_bytes: int
    full_pair_axis_complete: bool = True
    production_same_prefix_a_row_replay_complete: bool = False
    reference_precomputed_stationarity_only: bool = True
    reference_route_scale_host_input: bool = True
    continuous_box_relaxation_only: bool = True
    fixed_cardinality_primal_constraint_enforced: bool = False
    resource_primal_constraint_enforced: bool = False
    block_capacity_primal_constraint_enforced: bool = False
    q_primal_constraint_enforced: bool = False
    exact_retained_domain_optimization_claimed: bool = False
    issuer_route_scale_d2h_bytes: int = 0
    outside_shell_exclusion_authority: bool = False
    global_action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = OUTSIDE_AUTHORITY_SCOPE
    reference_domain_rule: str = OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE
    schema: str = OUTSIDE_EXCLUSION_RECEIPT_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _EXCLUSION_TOKEN:
            raise TypeError("outside exclusion reference receipts are issuer-only")
        object.__setattr__(
            self,
            "pair_cover_receipt_hash",
            _identity(self.pair_cover_receipt_hash, name="pair-cover receipt hash"),
        )
        for name in (
            "route_count",
            "complete_route_stationarity_scan_count",
            "complete_pair_scan_count",
            "add_outside_rectangle_scan_count",
            "drop_outside_add_inside_rectangle_scan_count",
            "caller_supplied_route_scale_host_coefficient_bytes",
            "issuer_route_scale_d2h_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if (
            self.route_count <= 0
            or self.complete_route_stationarity_scan_count != self.route_count
        ):
            raise ValueError(
                "reference receipt did not scan the complete stationarity axis"
            )
        if self.complete_pair_scan_count != (
            self.add_outside_rectangle_scan_count
            + self.drop_outside_add_inside_rectangle_scan_count
        ):
            raise ValueError("reference receipt pair scan counts do not close")
        if (
            self.caller_supplied_route_scale_host_coefficient_bytes
            != self.route_count * _F64_BYTES
        ):
            raise ValueError(
                "reference receipt did not disclose its route-scale host input"
            )
        preferred = _finite(self.preferred_f_upper_bound, name="preferred F upper")
        object.__setattr__(self, "preferred_f_upper_bound", preferred)
        if self.vacuous_no_outside_pair:
            if self.worst_pair is not None or self.complete_pair_scan_count != 0:
                raise ValueError("vacuous outside receipt has a worst pair")
            if not (
                math.isinf(self.minimum_pair_dual_lower_bound)
                and self.minimum_pair_dual_lower_bound > 0.0
                and math.isinf(self.strict_margin_lower_bound)
                and self.strict_margin_lower_bound > 0.0
            ):
                raise ValueError(
                    "vacuous outside receipt must use positive infinite bounds"
                )
        else:
            if (
                type(self.worst_pair)
                is not V1SignedWorkVariableShellOutsidePairIdentity
            ):
                raise TypeError(
                    "nonvacuous outside receipt requires its exact worst pair"
                )
            minimum = _finite(
                self.minimum_pair_dual_lower_bound,
                name="minimum pair dual lower",
            )
            margin = _finite(self.strict_margin_lower_bound, name="strict margin lower")
            if minimum <= preferred or margin <= 0.0:
                raise ValueError("outside receipt lacks strict outward separation")
        if (
            self.all_pair_cells_strictly_excluded is not True
            or self.full_pair_axis_complete is not True
            or self.production_same_prefix_a_row_replay_complete is not False
            or self.reference_precomputed_stationarity_only is not True
            or self.reference_route_scale_host_input is not True
            or self.continuous_box_relaxation_only is not True
            or self.fixed_cardinality_primal_constraint_enforced is not False
            or self.resource_primal_constraint_enforced is not False
            or self.block_capacity_primal_constraint_enforced is not False
            or self.q_primal_constraint_enforced is not False
            or self.exact_retained_domain_optimization_claimed is not False
            or self.issuer_route_scale_d2h_bytes != 0
        ):
            raise ValueError("outside reference receipt semantics changed")
        _closed_authority(
            (
                self.outside_shell_exclusion_authority,
                self.global_action_invariance_authority,
                self.c_stream_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="outside exclusion reference receipt",
        )
        if (
            self.authority_scope != OUTSIDE_AUTHORITY_SCOPE
            or self.reference_domain_rule != OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE
            or self.schema != OUTSIDE_EXCLUSION_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported outside exclusion reference receipt")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "outside-exclusion-reference-")


def issue_v1_signed_work_variable_shell_outside_exclusion_reference(
    *,
    pair_cover: V1SignedWorkVariableShellOutsidePairCoverEnumeration,
    pair_bounds: Sequence[V1SignedWorkVariableShellOutsidePairDualBoundReference],
    preferred_f_upper_bound: Real,
) -> V1SignedWorkVariableShellOutsideExclusionReferenceReceipt:
    """Close only a synthetic/reference pair scan; no production authority."""

    if type(pair_cover) is not V1SignedWorkVariableShellOutsidePairCoverEnumeration:
        raise TypeError("outside reference issuance requires the exact pair cover")
    preferred = _finite(preferred_f_upper_bound, name="preferred F upper")
    bounds = tuple(pair_bounds)
    if any(
        type(value) is not V1SignedWorkVariableShellOutsidePairDualBoundReference
        for value in bounds
    ):
        raise TypeError("outside reference bound axis contains another type")
    expected_pairs = tuple(pair_cover.iter_pairs())
    observed_hashes = tuple(value.pair.pair_hash for value in bounds)
    expected_hashes = tuple(value.pair_hash for value in expected_pairs)
    if (
        len(bounds) != pair_cover.receipt.complete_pair_count
        or observed_hashes != expected_hashes
        or any(
            value.pair_cover_receipt_hash != pair_cover.receipt.receipt_hash
            for value in bounds
        )
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INCOMPLETE_PAIR_AXIS,
            "outside reference bounds do not cover the exact pair axis once and in order",
        )
    if not bounds:
        return V1SignedWorkVariableShellOutsideExclusionReferenceReceipt(
            pair_cover_receipt_hash=pair_cover.receipt.receipt_hash,
            route_count=pair_cover.receipt.route_count,
            complete_route_stationarity_scan_count=pair_cover.receipt.route_count,
            complete_pair_scan_count=0,
            add_outside_rectangle_scan_count=0,
            drop_outside_add_inside_rectangle_scan_count=0,
            worst_pair=None,
            minimum_pair_dual_lower_bound=math.inf,
            preferred_f_upper_bound=preferred,
            strict_margin_lower_bound=math.inf,
            all_pair_cells_strictly_excluded=True,
            vacuous_no_outside_pair=True,
            caller_supplied_route_scale_host_coefficient_bytes=(
                pair_cover.receipt.route_count * _F64_BYTES
            ),
            _issuer_token=_EXCLUSION_TOKEN,
        )
    worst = min(
        bounds,
        key=lambda value: (
            value.dual_lower_bound_outward,
            value.pair.preferred_selected_index,
            value.pair.preferred_unselected_index,
        ),
    )
    minimum = worst.dual_lower_bound_outward
    margin_exact = Fraction.from_float(minimum) - Fraction.from_float(preferred)
    margin_lower = _down_from_fraction(margin_exact)
    if minimum <= preferred or margin_lower <= 0.0:
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.NONSTRICT_EXCLUSION,
            "one outside pair lower bound touches or falls below the preferred outward upper bound",
        )
    first_count = sum(
        value.pair.rectangle
        is V1SignedWorkVariableShellOutsidePairRectangle.ADD_OUTSIDE
        for value in bounds
    )
    second_count = len(bounds) - first_count
    if (
        first_count != pair_cover.receipt.add_outside_rectangle_pair_count
        or second_count
        != pair_cover.receipt.drop_outside_add_inside_rectangle_pair_count
    ):
        _refuse(
            V1SignedWorkVariableShellOutsideExclusionRefusalCode.INCOMPLETE_PAIR_AXIS,
            "outside reference rectangle scan counts changed",
        )
    return V1SignedWorkVariableShellOutsideExclusionReferenceReceipt(
        pair_cover_receipt_hash=pair_cover.receipt.receipt_hash,
        route_count=pair_cover.receipt.route_count,
        complete_route_stationarity_scan_count=pair_cover.receipt.route_count,
        complete_pair_scan_count=len(bounds),
        add_outside_rectangle_scan_count=first_count,
        drop_outside_add_inside_rectangle_scan_count=second_count,
        worst_pair=worst.pair,
        minimum_pair_dual_lower_bound=minimum,
        preferred_f_upper_bound=preferred,
        strict_margin_lower_bound=margin_lower,
        all_pair_cells_strictly_excluded=True,
        vacuous_no_outside_pair=False,
        caller_supplied_route_scale_host_coefficient_bytes=(
            pair_cover.receipt.route_count * _F64_BYTES
        ),
        _issuer_token=_EXCLUSION_TOKEN,
    )


__all__ = [
    "OUTSIDE_AUTHORITY_SCOPE",
    "OUTSIDE_PAIR_BOX_RULE",
    "OUTSIDE_PAIR_COVER_RULE",
    "OUTSIDE_PAIR_REFERENCE_DOMAIN_RULE",
    "V1SignedWorkVariableShellFullAxisLowerPlaneBindingReceipt",
    "V1SignedWorkVariableShellFullAxisLowerPlaneHandle",
    "V1SignedWorkVariableShellFullAxisLowerPlaneLease",
    "V1SignedWorkVariableShellOutsideCUDAExecutionDesign",
    "V1SignedWorkVariableShellOutsideCUDAHBMForecast",
    "V1SignedWorkVariableShellOutsideExclusionReferenceReceipt",
    "V1SignedWorkVariableShellOutsideExclusionRefusal",
    "V1SignedWorkVariableShellOutsideExclusionRefusalCode",
    "V1SignedWorkVariableShellOutsideExecutionMode",
    "V1SignedWorkVariableShellOutsidePairCoverEnumeration",
    "V1SignedWorkVariableShellOutsidePairCoverReceipt",
    "V1SignedWorkVariableShellOutsidePairDualBoundReference",
    "V1SignedWorkVariableShellOutsidePairIdentity",
    "V1SignedWorkVariableShellOutsidePairRectangle",
    "V1SignedWorkVariableShellSamePrefixARowReplayProvider",
    "build_v1_signed_work_variable_shell_outside_cuda_design",
    "build_v1_signed_work_variable_shell_outside_cuda_hbm_forecast",
    "build_v1_signed_work_variable_shell_outside_pair_cover",
    "build_v1_signed_work_variable_shell_reference_lower_plane_handle",
    "evaluate_v1_signed_work_variable_shell_outside_pair_dual_bounds_reference",
    "issue_v1_signed_work_variable_shell_outside_exclusion_cuda",
    "issue_v1_signed_work_variable_shell_outside_exclusion_reference",
]
