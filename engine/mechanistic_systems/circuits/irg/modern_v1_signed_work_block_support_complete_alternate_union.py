"""Source-bound pieces of the complete physical alternate-action union.

The within-shell solver deliberately solves its *registered cell* family,
which still contains every branch/alias representation of the preferred hard
action ``P``.  Those cells are useful for the narrow solver diagnostic but are
not alternatives to ``P``.  This module removes all of them using the exact
physical-domain crosswalk and replays the least surviving raw slope.

The replay is intentionally expressed as an exact delta from the already
enclosed family lower bound.  If ``d_0`` is the least registered projected-dual
lower and ``d_A`` is the least P-excluded projected-dual lower, then

    L_A = down(L_0 + (d_A - d_0)).

The preferred-dot and objective-combination radii are common to every cell and
are already present exactly once in ``L_0``.  Transporting only the exact
binary64-rational difference ``d_A-d_0`` therefore preserves those debits; it
does not reconstruct, omit, or duplicate them.  D/residual preferred-plane
deficits are *not* part of this raw receipt.  They belong after the minimum of
the complete within/outside union has been taken.

This is one narrow issuer-only seam.  It grants no complete-union, full-U, C,
K, O, commit, terminal, or Pareto authority by itself.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from typing import Final, NoReturn, Sequence

from .contracts import content_hash
from .modern_v1_signed_work_block_support_alternate_union_solver_cuda import (
    V1BlockSupportWithinShellCUDAExecutionReceipt,
    V1BlockSupportWithinShellCellDualLowerBoundReceipt,
    V1BlockSupportWithinShellLowerBoundFamilyReceipt,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBinding,
    V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
    V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
)


P_EXCLUDED_WITHIN_SCHEMA: Final = (
    "partition_free_v1_block_support_P_excluded_within_raw_slope_v1"
)
P_EXCLUDED_WITHIN_RULE: Final = (
    "exclude_every_structurally_identified_preferred_action_cell_then_recompute_"
    "the_surviving_minimum_by_exact_projected_dual_delta_without_plane_deficit"
)
P_EXCLUDED_WITHIN_AUTHORITY: Final = (
    "raw_P_excluded_within_shell_lower_bound_only_no_complete_union_full_U_FQR_"
    "C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-block-support-complete-alternate-union-"
_P_EXCLUDED_TOKEN: Final = object()


class V1BlockSupportCompleteAlternateUnionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    INCOMPLETE_CELL_AXIS = "incomplete_cell_axis"
    P_EXCLUSION_MISMATCH = "P_exclusion_mismatch"
    EMPTY_ALTERNATE_DOMAIN = "empty_alternate_domain"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"
    RECEIPT_TAMPERING = "receipt_tampering"


class V1BlockSupportCompleteAlternateUnionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportCompleteAlternateUnionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1BlockSupportCompleteAlternateUnionRefusalCode:
            raise TypeError("complete-union refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1BlockSupportCompleteAlternateUnionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1BlockSupportCompleteAlternateUnionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _fraction_down(value: Fraction) -> float:
    """Greatest available binary64 no larger than one exact rational."""

    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
            "exact P-excluded slope does not fit finite binary64",
        )
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result) or Fraction.from_float(result) > value:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
            "P-excluded outward conversion failed",
        )
    return result


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: getattr(value, item.name)
        for item in fields(value)
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} attempted to open downstream authority",
        )


@dataclass(frozen=True, slots=True)
class V1BlockSupportPExcludedWithinShellRawSlopeReceipt:
    block_support_registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    structural_domain_receipt_hash: str
    physical_block_support_binding_receipt_hash: str
    registry_receipt_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    within_shell_execution_receipt_hash: str
    within_shell_family_receipt_hash: str
    complete_cell_receipt_axis: tuple[tuple[str, str], ...]
    excluded_preferred_action_cell_hashes: tuple[str, ...]
    surviving_cell_crosswalk_axis: tuple[tuple[str, str, str, int, str], ...]
    deduplicated_physical_action_axis: tuple[
        tuple[str, str, int, tuple[int, ...]], ...
    ]
    registered_cell_count: int
    excluded_preferred_cell_count: int
    surviving_cell_count: int
    deduplicated_physical_action_count: int
    registered_minimum_cell_hash: str
    registered_minimum_projected_dual_lower: float
    p_excluded_minimum_cell_hash: str
    p_excluded_minimum_projected_dual_lower: float
    projected_dual_delta_numerator: int
    projected_dual_delta_denominator: int
    registered_within_shell_raw_lower: float
    p_excluded_within_shell_raw_lower: float
    objective_combination_l1_radius_upper: float
    preferred_dot_roundoff_radius_upper: float
    receipt_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    complete_registered_cell_axis_replayed: bool = True
    every_preferred_alias_excluded: bool = True
    physical_alternates_deduplicated: bool = True
    exact_binary64_rational_delta: bool = True
    common_preferred_dot_debit_preserved_once: bool = True
    common_objective_combination_debit_preserved_once: bool = True
    d_plane_deficit_debited: bool = False
    residual_plane_deficit_debited: bool = False
    relaxed_cardinality_equality_enforced: bool = False
    cardinality_dual_present: bool = False
    alternate_Q_dropped: bool = True
    complete_physical_alternate_union_authority: bool = False
    canonical_block_observation_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    rule: str = P_EXCLUDED_WITHIN_RULE
    authority_scope: str = P_EXCLUDED_WITHIN_AUTHORITY
    schema: str = P_EXCLUDED_WITHIN_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _P_EXCLUDED_TOKEN:
            raise TypeError("P-excluded within-shell receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and item.name != "receipt_hash"
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        for name in (
            "block_index",
            "registered_cell_count",
            "excluded_preferred_cell_count",
            "surviving_cell_count",
            "deduplicated_physical_action_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "block_context_count",
            _positive_int(self.block_context_count, name="block_context_count"),
        )
        for name in (
            "registered_minimum_projected_dual_lower",
            "p_excluded_minimum_projected_dual_lower",
            "registered_within_shell_raw_lower",
            "p_excluded_within_shell_raw_lower",
            "objective_combination_l1_radius_upper",
            "preferred_dot_roundoff_radius_upper",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if (
            self.objective_combination_l1_radius_upper < 0.0
            or self.preferred_dot_roundoff_radius_upper < 0.0
            or isinstance(self.projected_dual_delta_numerator, bool)
            or type(self.projected_dual_delta_numerator) is not int
            or isinstance(self.projected_dual_delta_denominator, bool)
            or type(self.projected_dual_delta_denominator) is not int
            or self.projected_dual_delta_denominator <= 0
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
                "P-excluded numerical metadata is malformed",
            )
        cells = tuple(
            (
                _identity(cell_hash, name="cell hash"),
                _identity(receipt_hash, name="cell receipt hash"),
            )
            for cell_hash, receipt_hash in self.complete_cell_receipt_axis
        )
        excluded = tuple(
            _identity(value, name="excluded P cell hash")
            for value in self.excluded_preferred_action_cell_hashes
        )
        crosswalk = tuple(self.surviving_cell_crosswalk_axis)
        actions = tuple(self.deduplicated_physical_action_axis)
        cell_hashes = tuple(value[0] for value in cells)
        surviving_hashes = tuple(value[0] for value in crosswalk)
        if (
            not cells
            or len(cell_hashes) != len(set(cell_hashes))
            or len({value[1] for value in cells}) != len(cells)
            or not excluded
            or len(excluded) != len(set(excluded))
            or not crosswalk
            or len(surviving_hashes) != len(set(surviving_hashes))
            or set(excluded).intersection(surviving_hashes)
            or set(excluded).union(surviving_hashes) != set(cell_hashes)
            or self.registered_cell_count != len(cells)
            or self.excluded_preferred_cell_count != len(excluded)
            or self.surviving_cell_count != len(crosswalk)
            or self.deduplicated_physical_action_count != len(actions)
            or self.registered_minimum_cell_hash not in cell_hashes
            or self.p_excluded_minimum_cell_hash not in surviving_hashes
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.INCOMPLETE_CELL_AXIS,
                "P-excluded receipt does not contain one exact cell partition",
            )
        object.__setattr__(self, "complete_cell_receipt_axis", cells)
        object.__setattr__(self, "excluded_preferred_action_cell_hashes", excluded)
        object.__setattr__(self, "surviving_cell_crosswalk_axis", crosswalk)
        object.__setattr__(self, "deduplicated_physical_action_axis", actions)
        delta = Fraction(
            self.projected_dual_delta_numerator,
            self.projected_dual_delta_denominator,
        )
        expected_delta = Fraction.from_float(
            self.p_excluded_minimum_projected_dual_lower
        ) - Fraction.from_float(self.registered_minimum_projected_dual_lower)
        expected_raw = Fraction.from_float(
            self.registered_within_shell_raw_lower
        ) + expected_delta
        if (
            delta != expected_delta
            or delta < 0
            or self.p_excluded_within_shell_raw_lower > _fraction_down(expected_raw)
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
                "P-excluded exact-delta enclosure does not replay",
            )
        required_true = (
            self.complete_registered_cell_axis_replayed,
            self.every_preferred_alias_excluded,
            self.physical_alternates_deduplicated,
            self.exact_binary64_rational_delta,
            self.common_preferred_dot_debit_preserved_once,
            self.common_objective_combination_debit_preserved_once,
            self.alternate_Q_dropped,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.AUTHORITY_VIOLATION,
                "P-excluded proof lost a mandatory premise",
            )
        _closed(
            (
                self.d_plane_deficit_debited,
                self.residual_plane_deficit_debited,
                self.relaxed_cardinality_equality_enforced,
                self.cardinality_dual_present,
                self.complete_physical_alternate_union_authority,
                self.canonical_block_observation_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="P-excluded within-shell receipt",
        )
        if (
            self.rule != P_EXCLUDED_WITHIN_RULE
            or self.authority_scope != P_EXCLUDED_WITHIN_AUTHORITY
            or self.schema != P_EXCLUDED_WITHIN_SCHEMA
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.AUTHORITY_VIOLATION,
                "P-excluded semantics changed",
            )
        seal = content_hash(_payload(self), prefix=_PREFIX + "P-excluded-integrity-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.RECEIPT_TAMPERING,
                "P-excluded receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "P-excluded-"),
        )

    def assert_integrity(self) -> str:
        seal = content_hash(_payload(self), prefix=_PREFIX + "P-excluded-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "P-excluded-")
        if seal != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.RECEIPT_TAMPERING,
                "P-excluded receipt does not replay",
            )
        return expected


def _issue_p_excluded_within_from_receipts(
    *,
    structural_receipt: V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt,
    physical_binding_receipt: V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt,
    within_shell_family: V1BlockSupportWithinShellLowerBoundFamilyReceipt,
    within_shell_execution: V1BlockSupportWithinShellCUDAExecutionReceipt,
    cell_receipts: Sequence[V1BlockSupportWithinShellCellDualLowerBoundReceipt],
) -> V1BlockSupportPExcludedWithinShellRawSlopeReceipt:
    if (
        type(structural_receipt)
        is not V1SignedWorkPhysicalAlternateStructuralCUDADomainReceipt
        or type(physical_binding_receipt)
        is not V1SignedWorkPhysicalAlternateBlockSupportBindingReceipt
        or type(within_shell_family)
        is not V1BlockSupportWithinShellLowerBoundFamilyReceipt
        or type(within_shell_execution)
        is not V1BlockSupportWithinShellCUDAExecutionReceipt
    ):
        raise TypeError("P-excluded replay requires exact issuer-owned receipts")
    try:
        structural_receipt.assert_integrity()
        physical_binding_receipt.assert_integrity()
        within_shell_family.assert_integrity()
        within_shell_execution.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.STALE_SOURCE,
            f"one P-excluded source receipt is stale: {error}",
        )
    cells = tuple(cell_receipts)
    if (
        not cells
        or any(
            type(value) is not V1BlockSupportWithinShellCellDualLowerBoundReceipt
            for value in cells
        )
    ):
        raise TypeError("P-excluded replay requires every exact cell receipt")
    try:
        cell_integrities = tuple(value.assert_integrity() for value in cells)
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.STALE_SOURCE,
            f"one within-shell cell receipt is stale: {error}",
        )
    cell_hashes = tuple(value.action_cell_hash for value in cells)
    cell_axis = tuple(zip(cell_hashes, cell_integrities, strict=True))
    excluded = structural_receipt.excluded_preferred_action_cell_hashes
    crosswalk = structural_receipt.cell_crosswalk_axis
    surviving_hashes = tuple(value[0] for value in crosswalk)
    if (
        physical_binding_receipt.structural_domain_receipt_hash
        != structural_receipt.receipt_hash
        or within_shell_family.execution_receipt_hash
        != within_shell_execution.receipt_hash
        or within_shell_family.registry_receipt_hash
        != structural_receipt.registry_receipt_hash
        or within_shell_execution.registry_receipt_hash
        != structural_receipt.registry_receipt_hash
        or within_shell_family.preferred_action_hash
        != structural_receipt.preferred_action_hash
        or within_shell_family.preferred_action_binding_hash
        != structural_receipt.preferred_action_binding_hash
        or within_shell_execution.preferred_action_hash
        != structural_receipt.preferred_action_hash
        or within_shell_execution.preferred_action_binding_hash
        != structural_receipt.preferred_action_binding_hash
        or within_shell_execution.d_fixed_support_receipt_hash
        != physical_binding_receipt.d_fixed_support_receipt_hash
        or within_shell_execution.residual_support_receipt_hash
        != physical_binding_receipt.residual_support_receipt_hash
        or within_shell_family.block_index
        != physical_binding_receipt.residual_block_ordinal
        or within_shell_execution.block_index != within_shell_family.block_index
        or structural_receipt.registered_cell_count != len(cells)
        or within_shell_execution.registered_cell_count != len(cells)
        or within_shell_execution.solved_cell_count != len(cells)
        or within_shell_family.action_cell_hash_axis != cell_hashes
        or within_shell_execution.action_cell_hash_axis != cell_hashes
        or within_shell_family.cell_receipt_hash_axis != cell_integrities
        or within_shell_execution.cell_receipt_hash_axis != cell_integrities
    ):
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.SOURCE_MISMATCH,
            "physical, support, family, execution, or cell identities differ",
        )
    excluded_set = set(excluded)
    surviving_set = set(surviving_hashes)
    cell_set = set(cell_hashes)
    if (
        not excluded
        or len(excluded_set) != len(excluded)
        or not surviving_hashes
        or len(surviving_set) != len(surviving_hashes)
        or excluded_set.intersection(surviving_set)
        or excluded_set.union(surviving_set) != cell_set
        or structural_receipt.preferred_cell_count_excluded != len(excluded)
        or structural_receipt.alternate_cell_count != len(crosswalk)
    ):
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.P_EXCLUSION_MISMATCH,
            "structural P exclusions and surviving crosswalk do not partition cells",
        )
    aliases = set(within_shell_family.resource_alias_axis)
    cell_by_hash = {value.action_cell_hash: value for value in cells}
    for cell_hash, branch, resource_alias_id, ordinal, _coordinate in crosswalk:
        cell = cell_by_hash[cell_hash]
        if (
            cell.branch.value != branch
            or cell.resource_alias_id != resource_alias_id
            or cell.action_cell_ordinal != ordinal
            or (branch, resource_alias_id) not in aliases
            or cell.registry_receipt_hash != structural_receipt.registry_receipt_hash
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.SOURCE_MISMATCH,
                "one surviving structural cell does not replay its solver receipt",
            )
    for cell in cells:
        if (
            cell.registry_receipt_hash != structural_receipt.registry_receipt_hash
            or (cell.branch.value, cell.resource_alias_id) not in aliases
        ):
            _refuse(
                V1BlockSupportCompleteAlternateUnionRefusalCode.SOURCE_MISMATCH,
                "one registered cell is foreign to the structural registry",
            )
    registered_minimum = min(cells, key=lambda value: value.projected_dual_lower)
    if registered_minimum.action_cell_hash != within_shell_family.minimum_cell_hash:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.INCOMPLETE_CELL_AXIS,
            "family minimum does not replay from the complete cell axis",
        )
    surviving = tuple(cell_by_hash[value] for value in cell_hashes if value in surviving_set)
    if not surviving:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.EMPTY_ALTERNATE_DOMAIN,
            "P exclusion removed every within-shell cell",
        )
    p_excluded_minimum = min(
        surviving, key=lambda value: value.projected_dual_lower
    )
    delta = Fraction.from_float(
        p_excluded_minimum.projected_dual_lower
    ) - Fraction.from_float(registered_minimum.projected_dual_lower)
    if delta < 0:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.NUMERICAL_REFUSAL,
            "P-excluded minimum is below the replayed registered minimum",
        )
    p_excluded_raw = _fraction_down(
        Fraction.from_float(within_shell_family.weighted_within_shell_slope_lower)
        + delta
    )
    return V1BlockSupportPExcludedWithinShellRawSlopeReceipt(
        block_support_registration_hash=(
            within_shell_family.block_support_registration_hash
        ),
        block_index=within_shell_family.block_index,
        block_id=within_shell_family.block_id,
        block_context_axis_hash=within_shell_family.block_context_axis_hash,
        block_context_count=within_shell_family.block_context_count,
        structural_domain_receipt_hash=structural_receipt.receipt_hash,
        physical_block_support_binding_receipt_hash=(
            physical_binding_receipt.receipt_hash
        ),
        registry_receipt_hash=structural_receipt.registry_receipt_hash,
        preferred_action_hash=structural_receipt.preferred_action_hash,
        preferred_action_binding_hash=(
            structural_receipt.preferred_action_binding_hash
        ),
        d_fixed_support_receipt_hash=(
            physical_binding_receipt.d_fixed_support_receipt_hash
        ),
        residual_support_receipt_hash=(
            physical_binding_receipt.residual_support_receipt_hash
        ),
        within_shell_execution_receipt_hash=within_shell_execution.receipt_hash,
        within_shell_family_receipt_hash=within_shell_family.receipt_hash,
        complete_cell_receipt_axis=cell_axis,
        excluded_preferred_action_cell_hashes=excluded,
        surviving_cell_crosswalk_axis=crosswalk,
        deduplicated_physical_action_axis=structural_receipt.physical_action_axis,
        registered_cell_count=len(cells),
        excluded_preferred_cell_count=len(excluded),
        surviving_cell_count=len(crosswalk),
        deduplicated_physical_action_count=(
            structural_receipt.deduplicated_physical_action_count
        ),
        registered_minimum_cell_hash=registered_minimum.action_cell_hash,
        registered_minimum_projected_dual_lower=(
            registered_minimum.projected_dual_lower
        ),
        p_excluded_minimum_cell_hash=p_excluded_minimum.action_cell_hash,
        p_excluded_minimum_projected_dual_lower=(
            p_excluded_minimum.projected_dual_lower
        ),
        projected_dual_delta_numerator=delta.numerator,
        projected_dual_delta_denominator=delta.denominator,
        registered_within_shell_raw_lower=(
            within_shell_family.weighted_within_shell_slope_lower
        ),
        p_excluded_within_shell_raw_lower=p_excluded_raw,
        objective_combination_l1_radius_upper=(
            within_shell_family.objective_combination_l1_radius_upper
        ),
        preferred_dot_roundoff_radius_upper=(
            within_shell_family.preferred_dot_roundoff_radius_upper
        ),
        _issuer_token=_P_EXCLUDED_TOKEN,
    )


def issue_v1_block_support_p_excluded_within_shell_raw_lower(
    *,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    within_shell_family: V1BlockSupportWithinShellLowerBoundFamilyReceipt,
    within_shell_execution: V1BlockSupportWithinShellCUDAExecutionReceipt,
    cell_receipts: Sequence[V1BlockSupportWithinShellCellDualLowerBoundReceipt],
) -> V1BlockSupportPExcludedWithinShellRawSlopeReceipt:
    """Issue the raw P-excluded within-shell lower from one live block binding."""

    if (
        type(physical_block_support_binding)
        is not V1SignedWorkPhysicalAlternateBlockSupportBinding
    ):
        raise TypeError("P-excluded replay requires the live physical block binding")
    try:
        physical_block_support_binding.assert_live()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportCompleteAlternateUnionRefusalCode.STALE_SOURCE,
            f"physical block binding is stale: {error}",
        )
    return _issue_p_excluded_within_from_receipts(
        structural_receipt=(
            physical_block_support_binding.structural_provider.receipt
        ),
        physical_binding_receipt=physical_block_support_binding.receipt,
        within_shell_family=within_shell_family,
        within_shell_execution=within_shell_execution,
        cell_receipts=cell_receipts,
    )


__all__ = [
    "P_EXCLUDED_WITHIN_AUTHORITY",
    "P_EXCLUDED_WITHIN_RULE",
    "P_EXCLUDED_WITHIN_SCHEMA",
    "V1BlockSupportCompleteAlternateUnionRefusal",
    "V1BlockSupportCompleteAlternateUnionRefusalCode",
    "V1BlockSupportPExcludedWithinShellRawSlopeReceipt",
    "issue_v1_block_support_p_excluded_within_shell_raw_lower",
]
