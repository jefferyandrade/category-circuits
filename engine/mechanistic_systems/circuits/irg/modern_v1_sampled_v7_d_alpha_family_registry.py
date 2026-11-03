"""Pre-outcome alpha-family registration for sampled-V7 D inference.

One proposal generation is charged to the trajectory ledger through the exact
base :class:`V1GroupSequentialSerflingDesign`.  This module registers how that
already charged generation allocation is subdivided among the three D-side
statistical roles used by sampled V7:

* corrected-A proposal ordering;
* signed-F action witnessing; and
* literal-J response sampling.

The optional HS/EB hybrid design is a wrapper around the charged base design;
its hash is recorded separately and never appears as another ledger charge.
The receipt proves only this finite, pre-outcome alpha-family registration.
Persisting ``ledger_after`` as the controller's unique durable head and binding
the cells to downstream measurement-owner receipts remain separate premises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_proposal_alpha_ledger import (
    V1ProposalOrderAlphaLedger,
    advance_v1_proposal_order_alpha_ledger,
    validate_v1_proposal_order_design,
)
from .modern_v1_proposal_statistics import (
    V1GroupSequentialEmpiricalBernsteinSerflingDesign,
    V1GroupSequentialSerflingDesign,
)
from .modern_v1_response_exactification import V1MasterDecisionAlphaRegistration


SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_cell_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCOPE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_cell_scope_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_PLAN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_plan_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_LEGACY_REGISTRY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_registry_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_LEGACY_REPLAY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_registry_replay_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_REGISTRY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_registry_v2"
)
SAMPLED_V7_D_ALPHA_FAMILY_REPLAY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_alpha_family_registry_replay_v2"
)
SAMPLED_V7_D_ALPHA_FAMILY_REGISTRATION_AUTHORITY: Final = (
    "preoutcome_sampled_v7_D_alpha_family_registration_only_no_durable_"
    "controller_head_downstream_owner_D_C_K_O_commit_terminal_or_pareto_"
    "authority_v1"
)
SAMPLED_V7_D_ALPHA_FAMILY_MISSING_PREMISES: Final = (
    "durable_controller_head_compare_and_swap",
    "downstream_measurement_owner_binding",
)
SAMPLED_V7_D_ALPHA_SPEND_SEMANTICS: Final = (
    "combined_two_sided_cell_alpha_before_optional_hybrid_method_split_v1"
)
SAMPLED_V7_D_ALPHA_MAXIMIN_POLICY: Final = (
    "equal_preoutcome_role_totals_then_equal_noncensus_cells_with_last_"
    "residual_and_zero_census_v1"
)
SAMPLED_V7_D_ALPHA_PLAN_AUTHORITY: Final = (
    "preoutcome_sampled_v7_D_alpha_family_plan_only_no_ledger_charge_"
    "measurement_decision_or_commit_authority_v1"
)

_CELL_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-cell-"
_CELL_SCOPE_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-cell-scope-"
_CORRECTED_A_SCOPE_PREFIX: Final = (
    "partition-free-v1-sampled-v7-D-corrected-A-look-scope-"
)
_PLAN_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-plan-"
_REGISTRY_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-registry-"
_CELL_AXIS_PREFIX: Final = "partition-free-v1-sampled-v7-D-alpha-cell-axis-"
_ISSUER_TOKEN: Final = object()
_PLAN_ISSUER_TOKEN: Final = object()


class V1SampledV7DAlphaCellRole(str, Enum):
    CORRECTED_A_ORDER = "corrected_A_order"
    SIGNED_F_WITNESS = "signed_F_witness"
    LITERAL_J_SAMPLING = "literal_J_sampling"


SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS: Final = (
    V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER,
    V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS,
    V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING,
)
_ROLE_ORDER: Final = {
    role: index for index, role in enumerate(SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS)
}


class V1SampledV7DAlphaFamilyRegistryRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    CAMPAIGN_MISMATCH = "campaign_mismatch"
    CONTROLLER_MISMATCH = "controller_mismatch"
    GENERATION_MISMATCH = "generation_mismatch"
    ALLOCATION_MISMATCH = "allocation_mismatch"
    LEDGER_CHAIN_MISMATCH = "ledger_chain_mismatch"
    CELL_AXIS_MISSING_ROLE = "cell_axis_missing_role"
    DUPLICATE_CELL = "duplicate_cell"
    CELL_ALPHA_INVALID = "cell_alpha_invalid"
    CENSUS_ALPHA_VIOLATION = "census_alpha_violation"
    NONCENSUS_ALPHA_VIOLATION = "noncensus_alpha_violation"
    CELL_ALPHA_OVERSPEND = "cell_alpha_overspend"
    BASE_CELL_ALPHA_MISMATCH = "base_cell_alpha_mismatch"
    MASTER_REGISTRATION_MISMATCH = "master_registration_mismatch"
    PLAN_MISMATCH = "plan_mismatch"
    CORRECTED_A_LOOK_MISMATCH = "corrected_a_look_mismatch"
    HYBRID_BASE_MISMATCH = "hybrid_base_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"
    REPLAY_MISMATCH = "replay_mismatch"


class V1SampledV7DAlphaFamilyRegistryRefusal(RuntimeError):
    """Typed fail-closed refusal at the D alpha-family boundary."""

    def __init__(
        self,
        code: V1SampledV7DAlphaFamilyRegistryRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DAlphaFamilyRegistryRefusalCode:
            raise TypeError("sampled-V7 D alpha-family refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DAlphaFamilyRegistryRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DAlphaFamilyRegistryRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _nonnegative_alpha(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CELL_ALPHA_INVALID,
            f"{name} must be one finite nonnegative scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CELL_ALPHA_INVALID,
            f"{name} must be one finite nonnegative scalar",
        )
    return result


def _pair_of_names(value: object, *, name: str) -> tuple[str, str]:
    if not isinstance(value, (tuple, list)):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one two-item axis",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != 2 or result[0] == result[1]:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must contain exactly two distinct names",
        )
    return result  # type: ignore[return-value]


def _pair_of_positive_ints(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one two-item axis",
        )
    result = tuple(_positive_int(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must contain exactly two positive counts",
        )
    return result  # type: ignore[return-value]


def _pair_of_nonnegative_ints(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must be one two-item axis",
        )
    result = tuple(_nonnegative_int(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} must contain exactly two counts",
        )
    return result  # type: ignore[return-value]


def _closed_equal_alpha_axis(total: float, count: int) -> tuple[float, ...]:
    """Split binary64 ``total`` evenly and close ``math.fsum`` exactly.

    The final coordinate absorbs only representation-level residual.  The
    represented cells therefore have the maximin equal split up to the
    unavoidable final binary64 ULP, while their registered sum replays the
    parent allocation exactly.
    """

    if count <= 0:
        return ()
    if count == 1:
        return (total,)
    base = total / count
    result = [base] * (count - 1)
    last = total - math.fsum(result)
    result.append(last)
    for _ in range(8):
        observed = math.fsum(result)
        if observed == total:
            break
        result[-1] = math.nextafter(
            result[-1], math.inf if observed < total else -math.inf
        )
    if math.fsum(result) != total or any(value <= 0.0 for value in result):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
            "equal alpha split cannot close in finite positive binary64 cells",
        )
    return tuple(result)


def _alpha_axes_match(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        Fraction.from_float(a) == Fraction.from_float(b)
        for a, b in zip(left, right, strict=True)
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaFamilyCellScope:
    """Outcome-free identity and finite-population extent of one D cell."""

    role: V1SampledV7DAlphaCellRole
    cell_id: str
    cell_scope_hash: str
    population_size: int
    draw_count: int
    schema: str = SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCOPE_SCHEMA
    _scope_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.role) is not V1SampledV7DAlphaCellRole:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "D alpha-family cell-scope role must be exact",
            )
        object.__setattr__(self, "cell_id", _identity(self.cell_id, name="cell id"))
        object.__setattr__(
            self,
            "cell_scope_hash",
            _identity(self.cell_scope_hash, name="cell scope hash"),
        )
        population = _positive_int(self.population_size, name="population size")
        draws = _nonnegative_int(self.draw_count, name="draw count")
        if draws > population:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "cell-scope draw count cannot exceed its population",
            )
        if self.schema != SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCOPE_SCHEMA:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "cell-scope schema changed",
            )
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "draw_count", draws)
        object.__setattr__(self, "_scope_hash", self._compute_hash())

    @property
    def exact_census(self) -> bool:
        return self.draw_count == self.population_size

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "role": self.role.value,
            "cell_id": self.cell_id,
            "cell_scope_hash": self.cell_scope_hash,
            "population_size": self.population_size,
            "draw_count": self.draw_count,
            "exact_census": self.exact_census,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_CELL_SCOPE_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            rebuilt = V1SampledV7DAlphaFamilyCellScope(
                **self._constructor_payload()  # type: ignore[arg-type]
            )
        except V1SampledV7DAlphaFamilyRegistryRefusal:
            raise
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                f"D alpha-family cell scope no longer validates: {error}",
            )
        observed = rebuilt._compute_hash()
        if self._scope_hash is not None and observed != self._scope_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                "D alpha-family cell scope changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "D alpha-family cell scope hash does not replay",
            )
        object.__setattr__(self, "_scope_hash", observed)
        return observed

    @property
    def scope_hash(self) -> str:
        return self.assert_integrity()

    def _constructor_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "cell_id": self.cell_id,
            "cell_scope_hash": self.cell_scope_hash,
            "population_size": self.population_size,
            "draw_count": self.draw_count,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_cell_scope, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaFamilyCell:
    """One pre-outcome statistical cell in the sampled-V7 D family.

    ``draw_count`` is the registered count for this individual inference cell,
    not merely the maximum draw count of a larger optional-stopping design.
    Consequently ``draw_count == population_size`` is an actual census cell
    and spends zero alpha; every non-census inference cell spends positively.
    """

    role: V1SampledV7DAlphaCellRole
    cell_id: str
    cell_scope_hash: str
    population_size: int
    draw_count: int
    alpha_spend: float
    alpha_spend_semantics: str = SAMPLED_V7_D_ALPHA_SPEND_SEMANTICS
    schema: str = SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCHEMA
    _cell_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.role) is not V1SampledV7DAlphaCellRole:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "D alpha-family cell role must be exact",
            )
        object.__setattr__(self, "cell_id", _identity(self.cell_id, name="cell id"))
        object.__setattr__(
            self,
            "cell_scope_hash",
            _identity(self.cell_scope_hash, name="cell scope hash"),
        )
        population = _positive_int(self.population_size, name="population size")
        draws = _nonnegative_int(self.draw_count, name="draw count")
        if draws > population:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "cell draw count cannot exceed its population",
            )
        alpha = _nonnegative_alpha(self.alpha_spend, name="cell alpha spend")
        if draws == population:
            if alpha != 0.0:
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.CENSUS_ALPHA_VIOLATION,
                    "an exact-census cell must spend zero alpha",
                )
        elif alpha <= 0.0:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.NONCENSUS_ALPHA_VIOLATION,
                "a non-census inference cell must spend positive alpha",
            )
        if self.alpha_spend_semantics != SAMPLED_V7_D_ALPHA_SPEND_SEMANTICS:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "cell alpha-spend semantics changed",
            )
        if self.schema != SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCHEMA:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "cell schema changed",
            )
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "draw_count", draws)
        object.__setattr__(self, "alpha_spend", alpha)
        object.__setattr__(self, "_cell_hash", self._compute_hash())

    @property
    def exact_census(self) -> bool:
        return self.draw_count == self.population_size

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "role": self.role.value,
            "cell_id": self.cell_id,
            "cell_scope_hash": self.cell_scope_hash,
            "population_size": self.population_size,
            "draw_count": self.draw_count,
            "exact_census": self.exact_census,
            "alpha_spend": self.alpha_spend,
            "alpha_spend_semantics": self.alpha_spend_semantics,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_CELL_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if type(self) is not V1SampledV7DAlphaFamilyCell:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "D alpha-family cell integrity requires the exact cell type",
            )
        try:
            rebuilt = V1SampledV7DAlphaFamilyCell(**self._constructor_payload())
        except V1SampledV7DAlphaFamilyRegistryRefusal:
            raise
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                f"D alpha-family cell fields no longer validate: "
                f"{type(error).__name__}: {error}",
            )
        observed = rebuilt._compute_hash()
        if self._cell_hash is not None and observed != self._cell_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                "D alpha-family cell changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "D alpha-family cell hash does not replay",
            )
        object.__setattr__(self, "_cell_hash", observed)
        return observed

    @property
    def cell_hash(self) -> str:
        return self.assert_integrity()

    def _constructor_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "cell_id": self.cell_id,
            "cell_scope_hash": self.cell_scope_hash,
            "population_size": self.population_size,
            "draw_count": self.draw_count,
            "alpha_spend": self.alpha_spend,
            "alpha_spend_semantics": self.alpha_spend_semantics,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_cell, (self._constructor_payload(), self.assert_integrity())


def _validate_cell_axis(
    value: object,
    *,
    generation_alpha_allocation: float,
) -> tuple[V1SampledV7DAlphaFamilyCell, ...]:
    if type(value) is not tuple or not value:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "D alpha-family cell axis must be one nonempty immutable tuple",
        )
    cells = tuple(value)
    if any(type(cell) is not V1SampledV7DAlphaFamilyCell for cell in cells):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "D alpha-family cell axis requires exact cell types",
        )
    for cell in cells:
        cell.assert_integrity()
    roles = {cell.role for cell in cells}
    required_roles = set(SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS)
    if roles != required_roles:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CELL_AXIS_MISSING_ROLE,
            "cell axis must cover corrected-A, signed-F, and literal-J roles",
        )
    cell_ids = tuple(cell.cell_id for cell in cells)
    if len(set(cell_ids)) != len(cell_ids):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.DUPLICATE_CELL,
            "a D alpha-family cell id cannot be registered twice",
        )
    role_scopes = tuple((cell.role, cell.cell_scope_hash) for cell in cells)
    if len(set(role_scopes)) != len(role_scopes):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.DUPLICATE_CELL,
            "a role/scope alpha cell cannot be double-spent under another id",
        )
    represented_total = math.fsum(cell.alpha_spend for cell in cells)
    # The registered spend semantics are binary64 coordinates closed by
    # ``math.fsum``.  Summing every split coordinate as an exact rational is
    # over-conservative after nested role/cell splits: it can report a tiny
    # ULP-level overspend even when the represented controller arithmetic is
    # exactly closed to the generation allocation.  The strict budget guard
    # therefore compares the represented fsum value, not the non-operational
    # rational sum of all intermediate binary64 cells.
    if Fraction.from_float(represented_total) > Fraction.from_float(
        generation_alpha_allocation
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CELL_ALPHA_OVERSPEND,
            "registered D alpha-family cells exceed the generation allocation",
        )
    return tuple(
        sorted(cells, key=lambda cell: (_ROLE_ORDER[cell.role], cell.cell_id))
    )


def _cell_axis_hash(cells: tuple[V1SampledV7DAlphaFamilyCell, ...]) -> str:
    return content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCHEMA,
            "cell_hashes": tuple(cell.assert_integrity() for cell in cells),
        },
        prefix=_CELL_AXIS_PREFIX,
    )


def _corrected_a_scope_hash(
    *,
    campaign_contract_hash: str,
    model_identity_hash: str,
    query_contract_hash: str,
    preprocessing_hash: str,
    current_mask_hash: str,
    raw_logit_stream_source_hash: str,
    proposal_generation: int,
    route_count: int,
    route_axis_hash: str,
    stratum_names: tuple[str, str],
    population_stratum_counts: tuple[int, int],
    sealed_population_order_hashes: tuple[str, str],
    sampling_law_hash: str,
    look_index: int,
    registered_look_stratum_counts: tuple[int, int],
) -> str:
    return content_hash(
        {
            "schema": SAMPLED_V7_D_ALPHA_FAMILY_PLAN_SCHEMA,
            "role": V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER.value,
            "campaign_contract_hash": campaign_contract_hash,
            "model_identity_hash": model_identity_hash,
            "query_contract_hash": query_contract_hash,
            "preprocessing_hash": preprocessing_hash,
            "current_mask_hash": current_mask_hash,
            "raw_logit_stream_source_hash": raw_logit_stream_source_hash,
            "proposal_generation": proposal_generation,
            "route_count": route_count,
            "route_axis_hash": route_axis_hash,
            "stratum_names": stratum_names,
            "population_stratum_counts": population_stratum_counts,
            "sealed_population_order_hashes": sealed_population_order_hashes,
            "sampling_law_hash": sampling_law_hash,
            "look_index": look_index,
            "registered_look_stratum_counts": registered_look_stratum_counts,
        },
        prefix=_CORRECTED_A_SCOPE_PREFIX,
    )


def _corrected_a_cell_id(look_index: int) -> str:
    return f"corrected-A-order-look:{look_index:08d}"


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaFamilyPlan:
    """Outcome-free maximin split fixed before the corrected-A base design."""

    master_registration_hash: str
    proposal_order_alpha_schedule_hash: str
    ledger_before_hash: str
    campaign_contract_hash: str
    controller_run_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    registered_look_stratum_counts: tuple[tuple[int, int], ...]
    sampling_law_hash: str
    generation_alpha_allocation: float
    role_alpha_allocations: tuple[float, float, float]
    cell_axis: tuple[V1SampledV7DAlphaFamilyCell, ...]
    cell_axis_hash: str
    corrected_a_alpha_by_look: tuple[float, ...]
    role_registered_alpha_totals: tuple[float, float, float]
    registered_cell_alpha_total: float
    unused_generation_alpha: float
    allocation_policy: str = SAMPLED_V7_D_ALPHA_MAXIMIN_POLICY
    issued_preoutcome: bool = True
    contains_outcomes: bool = False
    ledger_charge_authority: bool = False
    measurement_authority: bool = False
    decision_authority: bool = False
    commit_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_ALPHA_PLAN_AUTHORITY
    schema: str = SAMPLED_V7_D_ALPHA_FAMILY_PLAN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _plan_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PLAN_ISSUER_TOKEN:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "sampled-V7 D alpha-family plans are builder-only",
            )
        for name in (
            "master_registration_hash",
            "proposal_order_alpha_schedule_hash",
            "ledger_before_hash",
            "campaign_contract_hash",
            "controller_run_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "route_axis_hash",
            "sampling_law_hash",
            "cell_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        names = _pair_of_names(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        orders = tuple(
            _identity(value, name="sealed population order hash")
            for value in self.sealed_population_order_hashes
        )
        if len(orders) != 2:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "sealed population order hashes must contain two identities",
            )
        looks = tuple(
            _pair_of_nonnegative_ints(row, name="registered look stratum counts")
            for row in self.registered_look_stratum_counts
        )
        if not looks:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                "the corrected-A role requires a finite nonempty look axis",
            )
        prior = (0, 0)
        for counts in looks:
            if (
                any(value <= 0 for value in counts)
                or any(value > total for value, total in zip(counts, populations))
                or any(value < previous for value, previous in zip(counts, prior))
                or counts == prior
            ):
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                    "corrected-A looks must be live, nested finite-population prefixes",
                )
            prior = counts
        generation_alpha = _nonnegative_alpha(
            self.generation_alpha_allocation,
            name="generation alpha allocation",
        )
        if generation_alpha <= 0.0:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                "generation alpha allocation must be positive",
            )
        role_allocations = tuple(
            _nonnegative_alpha(value, name="role alpha allocation")
            for value in self.role_alpha_allocations
        )
        expected_role_allocations = _closed_equal_alpha_axis(generation_alpha, 3)
        if len(role_allocations) != 3 or not _alpha_axes_match(
            role_allocations, expected_role_allocations
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                "role alpha allocations do not replay the canonical maximin split",
            )
        cells = _validate_cell_axis(
            self.cell_axis,
            generation_alpha_allocation=generation_alpha,
        )
        if self.cell_axis_hash != _cell_axis_hash(cells):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "planned cell-axis identity does not replay",
            )

        corrected_cells = tuple(
            cell
            for cell in cells
            if cell.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
        )
        if len(corrected_cells) != len(looks):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                "corrected-A requires exactly one planned cell per registered look",
            )
        corrected_by_id = {cell.cell_id: cell for cell in corrected_cells}
        expected_corrected: list[V1SampledV7DAlphaFamilyCell] = []
        for look_index, counts in enumerate(looks, start=1):
            cell_id = _corrected_a_cell_id(look_index)
            cell = corrected_by_id.get(cell_id)
            expected_scope = _corrected_a_scope_hash(
                campaign_contract_hash=self.campaign_contract_hash,
                model_identity_hash=self.model_identity_hash,
                query_contract_hash=self.query_contract_hash,
                preprocessing_hash=self.preprocessing_hash,
                current_mask_hash=self.current_mask_hash,
                raw_logit_stream_source_hash=self.raw_logit_stream_source_hash,
                proposal_generation=generation,
                route_count=route_count,
                route_axis_hash=self.route_axis_hash,
                stratum_names=names,
                population_stratum_counts=populations,
                sealed_population_order_hashes=orders,  # type: ignore[arg-type]
                sampling_law_hash=self.sampling_law_hash,
                look_index=look_index,
                registered_look_stratum_counts=counts,
            )
            if (
                cell is None
                or cell.cell_scope_hash != expected_scope
                or cell.population_size != sum(populations)
                or cell.draw_count != sum(counts)
                or cell.exact_census is not (counts == populations)
            ):
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                    "corrected-A cell does not bind its exact registered look scope",
                )
            expected_corrected.append(cell)
        corrected_alpha = tuple(cell.alpha_spend for cell in expected_corrected)
        if not _alpha_axes_match(
            tuple(float(value) for value in self.corrected_a_alpha_by_look),
            corrected_alpha,
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                "corrected-A alpha-by-look does not replay its exact cell axis",
            )
        role_totals = tuple(
            math.fsum(cell.alpha_spend for cell in cells if cell.role is role)
            for role in SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS
        )
        recorded_role_totals = tuple(
            _nonnegative_alpha(value, name="registered role alpha total")
            for value in self.role_registered_alpha_totals
        )
        if not _alpha_axes_match(recorded_role_totals, role_totals):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "registered role alpha totals do not replay",
            )
        for role_index, role in enumerate(SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS):
            role_cells = tuple(cell for cell in cells if cell.role is role)
            noncensus = tuple(cell for cell in role_cells if not cell.exact_census)
            expected_total = role_allocations[role_index] if noncensus else 0.0
            if Fraction.from_float(role_totals[role_index]) != Fraction.from_float(
                expected_total
            ):
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                    "each live D role must close its maximin role allocation exactly",
                )
        total = math.fsum(cell.alpha_spend for cell in cells)
        unused = math.fsum(
            allocation - spent
            for allocation, spent in zip(
                role_allocations, role_totals, strict=True
            )
        )
        if (
            Fraction.from_float(float(self.registered_cell_alpha_total))
            != Fraction.from_float(total)
            or Fraction.from_float(float(self.unused_generation_alpha))
            != Fraction.from_float(unused)
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "planned cell alpha arithmetic does not replay",
            )
        if (
            self.allocation_policy != SAMPLED_V7_D_ALPHA_MAXIMIN_POLICY
            or self.issued_preoutcome is not True
            or self.contains_outcomes is not False
            or self.ledger_charge_authority is not False
            or self.measurement_authority is not False
            or self.decision_authority is not False
            or self.commit_authority is not False
            or self.authority_scope != SAMPLED_V7_D_ALPHA_PLAN_AUTHORITY
            or self.schema != SAMPLED_V7_D_ALPHA_FAMILY_PLAN_SCHEMA
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "D alpha-family plan exceeded its pre-design authority",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", orders)
        object.__setattr__(self, "registered_look_stratum_counts", looks)
        object.__setattr__(self, "generation_alpha_allocation", generation_alpha)
        object.__setattr__(self, "role_alpha_allocations", role_allocations)
        object.__setattr__(self, "cell_axis", cells)
        object.__setattr__(self, "corrected_a_alpha_by_look", corrected_alpha)
        object.__setattr__(self, "role_registered_alpha_totals", role_totals)
        object.__setattr__(self, "registered_cell_alpha_total", total)
        object.__setattr__(self, "unused_generation_alpha", unused)
        object.__setattr__(self, "_plan_hash", self._compute_hash())

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "master_registration_hash": self.master_registration_hash,
            "proposal_order_alpha_schedule_hash": (
                self.proposal_order_alpha_schedule_hash
            ),
            "ledger_before_hash": self.ledger_before_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "controller_run_hash": self.controller_run_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "registered_look_stratum_counts": self.registered_look_stratum_counts,
            "sampling_law_hash": self.sampling_law_hash,
            "generation_alpha_allocation": self.generation_alpha_allocation,
            "role_alpha_allocations": self.role_alpha_allocations,
            "cell_axis": tuple(cell._hash_payload() for cell in self.cell_axis),
            "cell_axis_hash": self.cell_axis_hash,
            "corrected_a_alpha_by_look": self.corrected_a_alpha_by_look,
            "role_registered_alpha_totals": self.role_registered_alpha_totals,
            "registered_cell_alpha_total": self.registered_cell_alpha_total,
            "unused_generation_alpha": self.unused_generation_alpha,
            "allocation_policy": self.allocation_policy,
            "issued_preoutcome": self.issued_preoutcome,
            "contains_outcomes": self.contains_outcomes,
            "ledger_charge_authority": self.ledger_charge_authority,
            "measurement_authority": self.measurement_authority,
            "decision_authority": self.decision_authority,
            "commit_authority": self.commit_authority,
            "authority_scope": self.authority_scope,
        }

    def _compute_hash(self) -> str:
        for cell in self.cell_axis:
            cell.assert_integrity()
        return content_hash(self._hash_payload(), prefix=_PLAN_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            rebuilt = V1SampledV7DAlphaFamilyPlan(
                **self._constructor_payload(),  # type: ignore[arg-type]
                _issuance_token=_PLAN_ISSUER_TOKEN,
            )
        except V1SampledV7DAlphaFamilyRegistryRefusal as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                f"D alpha-family plan fields no longer replay: {error}",
            )
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                f"D alpha-family plan no longer validates: {error}",
            )
        observed = rebuilt._compute_hash()
        if self._plan_hash is not None and observed != self._plan_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                "D alpha-family plan changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "D alpha-family plan hash does not replay",
            )
        object.__setattr__(self, "_plan_hash", observed)
        return observed

    @property
    def plan_hash(self) -> str:
        return self.assert_integrity()

    def _constructor_payload(self) -> dict[str, object]:
        payload = dict(self._hash_payload())
        payload.pop("cell_axis")
        payload["cell_axis"] = self.cell_axis
        return payload

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_plan, (self._constructor_payload(), self.assert_integrity())


def _assert_exact_master_registration(
    value: object,
) -> tuple[V1MasterDecisionAlphaRegistration, str]:
    if type(value) is not V1MasterDecisionAlphaRegistration:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "D alpha-family planning requires the exact master registration type",
        )
    try:
        observed = value.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PARENT_INTEGRITY,
            f"master registration failed integrity replay: {error}",
        )
    return value, observed


def _validate_master_ledger_join(
    master: V1MasterDecisionAlphaRegistration,
    ledger: V1ProposalOrderAlphaLedger,
) -> None:
    schedule = ledger.schedule
    generation = ledger.proposal_generation
    if (
        master.campaign_contract_hash != schedule.campaign_contract_hash
        or master.campaign_contract_hash != ledger.campaign_contract_hash
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CAMPAIGN_MISMATCH,
            "master registration and proposal ledger disagree on campaign",
        )
    if (
        master.controller_run_hash != schedule.controller_run_hash
        or master.controller_run_hash != ledger.controller_run_hash
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CONTROLLER_MISMATCH,
            "master registration and proposal ledger disagree on controller",
        )
    if (
        master.proposal_order_alpha_schedule_hash != schedule.schedule_hash
        or master.maximum_proposal_generations
        != schedule.maximum_proposal_generations
        or master.proposal_order_familywise_alpha
        != schedule.trajectory_familywise_alpha
        or master.proposal_order_generation_allocations
        != schedule.generation_alpha_allocations
        or generation >= master.maximum_proposal_generations
        or master.proposal_order_generation_allocations[generation]
        != ledger.generation_alpha_allocation
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.MASTER_REGISTRATION_MISMATCH,
            "master proposal-order family does not exactly replay the ledger schedule",
        )


def build_v1_sampled_v7_d_alpha_family_plan(
    *,
    master_registration: V1MasterDecisionAlphaRegistration,
    ledger_before: V1ProposalOrderAlphaLedger,
    model_identity_hash: str,
    query_contract_hash: str,
    preprocessing_hash: str,
    current_mask_hash: str,
    raw_logit_stream_source_hash: str,
    route_count: int,
    route_axis_hash: str,
    stratum_names: tuple[str, str],
    population_stratum_counts: tuple[int, int],
    sealed_population_order_hashes: tuple[str, str],
    registered_look_stratum_counts: tuple[tuple[int, int], ...],
    sampling_law_hash: str,
    non_a_cell_scopes: tuple[V1SampledV7DAlphaFamilyCellScope, ...],
) -> V1SampledV7DAlphaFamilyPlan:
    """Build the canonical pre-design maximin A/F/J alpha plan."""

    master, master_hash = _assert_exact_master_registration(master_registration)
    before, before_hash = _assert_exact_ledger(ledger_before, name="ledger before")
    if before.is_exhausted:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
            "proposal alpha schedule is exhausted",
        )
    _validate_master_ledger_join(master, before)
    names = _pair_of_names(stratum_names, name="stratum names")
    populations = _pair_of_positive_ints(
        population_stratum_counts, name="population stratum counts"
    )
    orders = tuple(
        _identity(value, name="sealed population order hash")
        for value in sealed_population_order_hashes
    )
    if len(orders) != 2:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "sealed population order hashes must contain two identities",
        )
    looks = tuple(
        _pair_of_nonnegative_ints(row, name="registered look stratum counts")
        for row in registered_look_stratum_counts
    )
    if type(non_a_cell_scopes) is not tuple or any(
        type(scope) is not V1SampledV7DAlphaFamilyCellScope
        for scope in non_a_cell_scopes
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "non-A cell scopes must be one immutable axis of exact scope types",
        )
    for scope in non_a_cell_scopes:
        scope.assert_integrity()
        if scope.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                "corrected-A scopes are derived only from the registered look axis",
            )
    if {scope.role for scope in non_a_cell_scopes} != {
        V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS,
        V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING,
    }:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CELL_AXIS_MISSING_ROLE,
            "the pre-design scope axis must cover signed-F and literal-J",
        )

    route_count_value = _positive_int(route_count, name="route count")
    generation = before.proposal_generation
    a_scopes: list[V1SampledV7DAlphaFamilyCellScope] = []
    for look_index, counts in enumerate(looks, start=1):
        a_scopes.append(
            V1SampledV7DAlphaFamilyCellScope(
                role=V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER,
                cell_id=_corrected_a_cell_id(look_index),
                cell_scope_hash=_corrected_a_scope_hash(
                    campaign_contract_hash=before.campaign_contract_hash,
                    model_identity_hash=_identity(
                        model_identity_hash, name="model identity hash"
                    ),
                    query_contract_hash=_identity(
                        query_contract_hash, name="query contract hash"
                    ),
                    preprocessing_hash=_identity(
                        preprocessing_hash, name="preprocessing hash"
                    ),
                    current_mask_hash=_identity(
                        current_mask_hash, name="current mask hash"
                    ),
                    raw_logit_stream_source_hash=_identity(
                        raw_logit_stream_source_hash,
                        name="raw logit stream source hash",
                    ),
                    proposal_generation=generation,
                    route_count=route_count_value,
                    route_axis_hash=_identity(route_axis_hash, name="route axis hash"),
                    stratum_names=names,
                    population_stratum_counts=populations,
                    sealed_population_order_hashes=orders,  # type: ignore[arg-type]
                    sampling_law_hash=_identity(
                        sampling_law_hash, name="sampling law hash"
                    ),
                    look_index=look_index,
                    registered_look_stratum_counts=counts,
                ),
                population_size=sum(populations),
                draw_count=sum(counts),
            )
        )
    scopes = tuple(a_scopes) + non_a_cell_scopes
    scope_ids = tuple(scope.cell_id for scope in scopes)
    role_scopes = tuple((scope.role, scope.cell_scope_hash) for scope in scopes)
    if len(scope_ids) != len(set(scope_ids)) or len(role_scopes) != len(
        set(role_scopes)
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.DUPLICATE_CELL,
            "the pre-design scope axis contains a duplicate cell or role/scope",
        )
    role_allocations = _closed_equal_alpha_axis(
        before.generation_alpha_allocation, 3
    )
    cells: list[V1SampledV7DAlphaFamilyCell] = []
    for role_index, role in enumerate(SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS):
        role_scopes = tuple(scope for scope in scopes if scope.role is role)
        live_scopes = tuple(scope for scope in role_scopes if not scope.exact_census)
        live_alpha = _closed_equal_alpha_axis(
            role_allocations[role_index], len(live_scopes)
        )
        alpha_by_scope_hash = {
            scope.scope_hash: alpha
            for scope, alpha in zip(live_scopes, live_alpha, strict=True)
        }
        for scope in role_scopes:
            cells.append(
                V1SampledV7DAlphaFamilyCell(
                    role=scope.role,
                    cell_id=scope.cell_id,
                    cell_scope_hash=scope.cell_scope_hash,
                    population_size=scope.population_size,
                    draw_count=scope.draw_count,
                    alpha_spend=alpha_by_scope_hash.get(scope.scope_hash, 0.0),
                )
            )
    cell_axis = _validate_cell_axis(
        tuple(cells),
        generation_alpha_allocation=before.generation_alpha_allocation,
    )
    corrected_by_id = {
        cell.cell_id: cell
        for cell in cell_axis
        if cell.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
    }
    corrected_alpha_by_look = tuple(
        corrected_by_id[_corrected_a_cell_id(index)].alpha_spend
        for index in range(1, len(looks) + 1)
    )
    role_totals = tuple(
        math.fsum(cell.alpha_spend for cell in cell_axis if cell.role is role)
        for role in SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS
    )
    registered_total = math.fsum(cell.alpha_spend for cell in cell_axis)
    return V1SampledV7DAlphaFamilyPlan(
        master_registration_hash=master_hash,
        proposal_order_alpha_schedule_hash=before.schedule.schedule_hash,
        ledger_before_hash=before_hash,
        campaign_contract_hash=before.campaign_contract_hash,
        controller_run_hash=before.controller_run_hash,
        model_identity_hash=model_identity_hash,
        query_contract_hash=query_contract_hash,
        preprocessing_hash=preprocessing_hash,
        current_mask_hash=current_mask_hash,
        raw_logit_stream_source_hash=raw_logit_stream_source_hash,
        proposal_generation=generation,
        route_count=route_count_value,
        route_axis_hash=route_axis_hash,
        stratum_names=names,
        population_stratum_counts=populations,
        sealed_population_order_hashes=orders,  # type: ignore[arg-type]
        registered_look_stratum_counts=looks,
        sampling_law_hash=sampling_law_hash,
        generation_alpha_allocation=before.generation_alpha_allocation,
        role_alpha_allocations=role_allocations,  # type: ignore[arg-type]
        cell_axis=cell_axis,
        cell_axis_hash=_cell_axis_hash(cell_axis),
        corrected_a_alpha_by_look=corrected_alpha_by_look,
        role_registered_alpha_totals=role_totals,  # type: ignore[arg-type]
        registered_cell_alpha_total=registered_total,
        unused_generation_alpha=math.fsum(
            allocation - spent
            for allocation, spent in zip(
                role_allocations, role_totals, strict=True
            )
        ),
        _issuance_token=_PLAN_ISSUER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DAlphaFamilyRegistryReceipt:
    master_registration_hash: str
    proposal_order_alpha_schedule_hash: str
    alpha_family_plan_hash: str
    campaign_contract_hash: str
    controller_run_hash: str
    proposal_generation: int
    trajectory_familywise_alpha: float
    trajectory_alpha_committed_before_generation: float
    generation_alpha_allocation: float
    ledger_before_hash: str
    charged_base_design_hash: str
    hybrid_wrapper_design_hash: str | None
    ledger_after_hash: str
    ledger_after_generation: int
    cell_axis: tuple[V1SampledV7DAlphaFamilyCell, ...]
    cell_axis_hash: str
    role_alpha_allocations: tuple[float, float, float]
    role_registered_alpha_totals: tuple[float, float, float]
    corrected_a_alpha_by_look: tuple[float, ...]
    base_design_registered_look_alpha_total: float
    corrected_a_cell_alpha_total: float
    registered_cell_alpha_total: float
    unused_generation_alpha: float
    hybrid_wrapper_present: bool
    hybrid_wrapper_alpha_inert: bool = True
    issued_preoutcome: bool = True
    contains_outcomes: bool = False
    alpha_family_registration_authority: bool = True
    durable_controller_head_closed: bool = False
    downstream_owner_binding_closed: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    missing_premises: tuple[str, str] = SAMPLED_V7_D_ALPHA_FAMILY_MISSING_PREMISES
    authority_scope: str = SAMPLED_V7_D_ALPHA_FAMILY_REGISTRATION_AUTHORITY
    schema: str = SAMPLED_V7_D_ALPHA_FAMILY_REGISTRY_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "sampled-V7 D alpha-family registry receipts are issuer-only",
            )
        for name in (
            "master_registration_hash",
            "proposal_order_alpha_schedule_hash",
            "alpha_family_plan_hash",
            "campaign_contract_hash",
            "controller_run_hash",
            "ledger_before_hash",
            "charged_base_design_hash",
            "ledger_after_hash",
            "cell_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        after_generation = _nonnegative_int(
            self.ledger_after_generation, name="ledger-after generation"
        )
        if after_generation != generation + 1:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.GENERATION_MISMATCH,
                "ledger-after generation must directly succeed proposal generation",
            )
        for name in (
            "trajectory_familywise_alpha",
            "trajectory_alpha_committed_before_generation",
            "generation_alpha_allocation",
            "base_design_registered_look_alpha_total",
            "corrected_a_cell_alpha_total",
            "registered_cell_alpha_total",
            "unused_generation_alpha",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                    f"{name} must be finite",
                )
            finite = float(value)
            if not math.isfinite(finite) or finite < 0.0:
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                    f"{name} must be finite and nonnegative",
                )
            object.__setattr__(self, name, finite)
        if not 0.0 < self.trajectory_familywise_alpha < 1.0:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                "trajectory familywise alpha must lie in (0,1)",
            )
        if self.generation_alpha_allocation <= 0.0:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                "generation alpha allocation must be positive",
            )
        cells = _validate_cell_axis(
            self.cell_axis,
            generation_alpha_allocation=self.generation_alpha_allocation,
        )
        role_allocations = tuple(
            _nonnegative_alpha(value, name="role alpha allocation")
            for value in self.role_alpha_allocations
        )
        role_totals = tuple(
            math.fsum(cell.alpha_spend for cell in cells if cell.role is role)
            for role in SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS
        )
        recorded_role_totals = tuple(
            _nonnegative_alpha(value, name="registered role alpha total")
            for value in self.role_registered_alpha_totals
        )
        expected_role_allocations = _closed_equal_alpha_axis(
            self.generation_alpha_allocation, 3
        )
        if (
            len(role_allocations) != 3
            or len(recorded_role_totals) != 3
            or not _alpha_axes_match(role_allocations, expected_role_allocations)
            or not _alpha_axes_match(recorded_role_totals, role_totals)
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
                "registry role allocations do not replay the maximin plan",
            )
        total = math.fsum(cell.alpha_spend for cell in cells)
        corrected_a_total = math.fsum(
            cell.alpha_spend
            for cell in cells
            if cell.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
        )
        unused = math.fsum(
            allocation - spent
            for allocation, spent in zip(
                role_allocations, role_totals, strict=True
            )
        )
        if (
            self.cell_axis_hash != _cell_axis_hash(cells)
            or self.corrected_a_cell_alpha_total != corrected_a_total
            or self.registered_cell_alpha_total != total
            or self.unused_generation_alpha != unused
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "D alpha-family cell-axis arithmetic or identity does not replay",
            )
        corrected_by_id = {
            cell.cell_id: cell
            for cell in cells
            if cell.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
        }
        corrected_alpha = tuple(
            corrected_by_id[_corrected_a_cell_id(index)].alpha_spend
            for index in range(1, len(corrected_by_id) + 1)
            if _corrected_a_cell_id(index) in corrected_by_id
        )
        if (
            len(corrected_alpha) != len(corrected_by_id)
            or not _alpha_axes_match(
                tuple(float(value) for value in self.corrected_a_alpha_by_look),
                corrected_alpha,
            )
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.CORRECTED_A_LOOK_MISMATCH,
                "registry corrected-A alpha-by-look does not replay its cells",
            )
        if (
            Fraction.from_float(self.corrected_a_cell_alpha_total)
            != Fraction.from_float(self.base_design_registered_look_alpha_total)
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.BASE_CELL_ALPHA_MISMATCH,
                "corrected-A cells must exactly account for base-design look alpha",
            )
        if type(self.hybrid_wrapper_present) is not bool:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "hybrid-wrapper presence must be exact bool",
            )
        if self.hybrid_wrapper_present:
            wrapper_hash = _identity(
                self.hybrid_wrapper_design_hash,
                name="hybrid wrapper design hash",
            )
            if wrapper_hash == self.charged_base_design_hash:
                _refuse(
                    V1SampledV7DAlphaFamilyRegistryRefusalCode.HYBRID_BASE_MISMATCH,
                    "hybrid wrapper and charged base hashes must remain distinct",
                )
        elif self.hybrid_wrapper_design_hash is not None:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.HYBRID_BASE_MISMATCH,
                "an absent hybrid wrapper cannot name a wrapper hash",
            )
        positives = (
            self.hybrid_wrapper_alpha_inert,
            self.issued_preoutcome,
            self.alpha_family_registration_authority,
        )
        closed = (
            self.contains_outcomes,
            self.durable_controller_head_closed,
            self.downstream_owner_binding_closed,
            self.d_decision_authority,
            self.c_measurement_authority,
            self.calibration_k_authority,
            self.paired_oracle_o_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool or value is not True for value in positives):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "D alpha-family registry lost a required registration premise",
            )
        if any(type(value) is not bool or value is not False for value in closed):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "D alpha-family registry exceeded registration-only authority",
            )
        if self.missing_premises != SAMPLED_V7_D_ALPHA_FAMILY_MISSING_PREMISES:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "D alpha-family registry omitted an explicit missing premise",
            )
        if (
            self.authority_scope
            != SAMPLED_V7_D_ALPHA_FAMILY_REGISTRATION_AUTHORITY
            or self.schema != SAMPLED_V7_D_ALPHA_FAMILY_REGISTRY_SCHEMA
        ):
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.AUTHORITY_VIOLATION,
                "D alpha-family registry schema or authority scope changed",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "ledger_after_generation", after_generation)
        object.__setattr__(self, "cell_axis", cells)
        object.__setattr__(self, "role_alpha_allocations", role_allocations)
        object.__setattr__(self, "role_registered_alpha_totals", role_totals)
        object.__setattr__(self, "corrected_a_alpha_by_look", corrected_alpha)
        object.__setattr__(self, "_receipt_hash", self._compute_hash())

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "master_registration_hash": self.master_registration_hash,
            "proposal_order_alpha_schedule_hash": (
                self.proposal_order_alpha_schedule_hash
            ),
            "alpha_family_plan_hash": self.alpha_family_plan_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "controller_run_hash": self.controller_run_hash,
            "proposal_generation": self.proposal_generation,
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_generation": (
                self.trajectory_alpha_committed_before_generation
            ),
            "generation_alpha_allocation": self.generation_alpha_allocation,
            "ledger_before_hash": self.ledger_before_hash,
            "charged_base_design_hash": self.charged_base_design_hash,
            "hybrid_wrapper_design_hash": self.hybrid_wrapper_design_hash,
            "ledger_after_hash": self.ledger_after_hash,
            "ledger_after_generation": self.ledger_after_generation,
            "cell_axis": tuple(cell._hash_payload() for cell in self.cell_axis),
            "cell_axis_hash": self.cell_axis_hash,
            "role_alpha_allocations": self.role_alpha_allocations,
            "role_registered_alpha_totals": self.role_registered_alpha_totals,
            "corrected_a_alpha_by_look": self.corrected_a_alpha_by_look,
            "base_design_registered_look_alpha_total": (
                self.base_design_registered_look_alpha_total
            ),
            "corrected_a_cell_alpha_total": self.corrected_a_cell_alpha_total,
            "registered_cell_alpha_total": self.registered_cell_alpha_total,
            "unused_generation_alpha": self.unused_generation_alpha,
            "hybrid_wrapper_present": self.hybrid_wrapper_present,
            "hybrid_wrapper_alpha_inert": self.hybrid_wrapper_alpha_inert,
            "issued_preoutcome": self.issued_preoutcome,
            "contains_outcomes": self.contains_outcomes,
            "alpha_family_registration_authority": (
                self.alpha_family_registration_authority
            ),
            "durable_controller_head_closed": self.durable_controller_head_closed,
            "downstream_owner_binding_closed": self.downstream_owner_binding_closed,
            "d_decision_authority": self.d_decision_authority,
            "c_measurement_authority": self.c_measurement_authority,
            "calibration_k_authority": self.calibration_k_authority,
            "paired_oracle_o_authority": self.paired_oracle_o_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "performance_authority": self.performance_authority,
            "pareto_authority": self.pareto_authority,
            "missing_premises": self.missing_premises,
            "authority_scope": self.authority_scope,
        }

    def _compute_hash(self) -> str:
        for cell in self.cell_axis:
            cell.assert_integrity()
        return content_hash(self._hash_payload(), prefix=_REGISTRY_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if type(self) is not V1SampledV7DAlphaFamilyRegistryReceipt:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "sampled-V7 D alpha-family integrity requires the exact receipt type",
            )
        try:
            rebuilt = V1SampledV7DAlphaFamilyRegistryReceipt(
                **self._constructor_payload(),
                _issuance_token=_ISSUER_TOKEN,
            )
        except V1SampledV7DAlphaFamilyRegistryRefusal:
            raise
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                "sampled-V7 D alpha-family receipt fields no longer validate: "
                f"{type(error).__name__}: {error}",
            )
        observed = rebuilt._compute_hash()
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.MUTATED_RECEIPT,
                "sampled-V7 D alpha-family registry changed after sealing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "sampled-V7 D alpha-family registry hash does not replay",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def require_alpha_family_registration(self) -> str:
        return self.assert_integrity()

    def require_d_decision(self) -> NoReturn:
        raise PermissionError("D alpha-family registration has no D-decision authority")

    def require_c_measurement(self) -> NoReturn:
        raise PermissionError("D alpha-family registration has no C authority")

    def require_calibration_k(self) -> NoReturn:
        raise PermissionError("D alpha-family registration has no K authority")

    def require_paired_oracle_o(self) -> NoReturn:
        raise PermissionError("D alpha-family registration has no O authority")

    def require_mask_commit(self) -> NoReturn:
        raise PermissionError("D alpha-family registration cannot commit masks")

    def _constructor_payload(self) -> dict[str, object]:
        payload = dict(self._hash_payload())
        payload.pop("cell_axis")
        payload["cell_axis"] = self.cell_axis
        return payload

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        return _rebuild_registry_receipt, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


def _assert_exact_ledger(
    value: object,
    *,
    name: str,
) -> tuple[V1ProposalOrderAlphaLedger, str]:
    if type(value) is not V1ProposalOrderAlphaLedger:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            f"{name} requires the exact proposal-order alpha ledger type",
        )
    try:
        observed = value.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PARENT_INTEGRITY,
            f"{name} failed integrity replay: {type(error).__name__}: {error}",
        )
    return value, observed


def _assert_exact_base_design(
    value: object,
) -> tuple[V1GroupSequentialSerflingDesign, str]:
    if type(value) is not V1GroupSequentialSerflingDesign:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "D alpha-family registration requires the exact base Serfling design",
        )
    try:
        observed = value.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PARENT_INTEGRITY,
            f"base design failed integrity replay: {type(error).__name__}: {error}",
        )
    return value, observed


def issue_v1_sampled_v7_d_alpha_family_registry(
    *,
    master_registration: V1MasterDecisionAlphaRegistration,
    alpha_family_plan: V1SampledV7DAlphaFamilyPlan,
    ledger_before: V1ProposalOrderAlphaLedger,
    base_design: V1GroupSequentialSerflingDesign,
    ledger_after: V1ProposalOrderAlphaLedger,
    hybrid_wrapper: V1GroupSequentialEmpiricalBernsteinSerflingDesign | None = None,
) -> V1SampledV7DAlphaFamilyRegistryReceipt:
    """Register one finite D alpha family against an exact ledger transition."""

    master, master_hash = _assert_exact_master_registration(master_registration)
    before, before_hash = _assert_exact_ledger(ledger_before, name="ledger before")
    base, base_hash = _assert_exact_base_design(base_design)
    after, after_hash = _assert_exact_ledger(ledger_after, name="ledger after")
    if type(alpha_family_plan) is not V1SampledV7DAlphaFamilyPlan:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "D alpha-family registration requires the exact pre-design plan type",
        )
    try:
        plan_hash = alpha_family_plan.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PARENT_INTEGRITY,
            f"D alpha-family plan failed integrity replay: {error}",
        )
    _validate_master_ledger_join(master, before)
    if (
        alpha_family_plan.master_registration_hash != master_hash
        or alpha_family_plan.proposal_order_alpha_schedule_hash
        != master.proposal_order_alpha_schedule_hash
        or alpha_family_plan.ledger_before_hash != before_hash
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PLAN_MISMATCH,
            "pre-design plan does not bind the exact master and ledger-before parents",
        )

    if before.is_exhausted:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
            "proposal alpha schedule is exhausted",
        )
    if base.campaign_contract_hash != before.campaign_contract_hash:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CAMPAIGN_MISMATCH,
            "base design does not name the ledger campaign",
        )
    if base.proposal_generation != before.proposal_generation:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.GENERATION_MISMATCH,
            "base design does not name the current proposal generation",
        )
    if (
        base.trajectory_alpha_ledger_hash != before_hash
        or base.trajectory_alpha_ledger_generation != before.proposal_generation
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.LEDGER_CHAIN_MISMATCH,
            "base design does not name the ledger-before head",
        )
    allocation = before.generation_alpha_allocation
    if (
        base.trajectory_familywise_alpha != before.trajectory_familywise_alpha
        or base.trajectory_alpha_committed_before_generation
        != before.trajectory_alpha_committed_before_generation
        or base.generation_alpha_allocation != allocation
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.ALLOCATION_MISMATCH,
            "base design does not replay the current generation allocation",
        )
    plan_base_coordinates = (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "proposal_generation",
        "route_count",
        "route_axis_hash",
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
        "registered_look_stratum_counts",
        "sampling_law_hash",
        "generation_alpha_allocation",
    )
    if any(
        getattr(alpha_family_plan, name) != getattr(base, name)
        for name in plan_base_coordinates
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.PLAN_MISMATCH,
            "base Serfling design drifted from the pre-design source/look axis",
        )
    if not _alpha_axes_match(
        base.alpha_by_look,
        alpha_family_plan.corrected_a_alpha_by_look,
    ):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.BASE_CELL_ALPHA_MISMATCH,
            "base-design alpha-by-look must exactly equal planned corrected-A cells",
        )
    try:
        validate_v1_proposal_order_design(
            before,
            base,
            expected_ledger_hash=before_hash,
            expected_design_hash=base_hash,
        )
        expected_after = advance_v1_proposal_order_alpha_ledger(
            before,
            base,
            expected_ledger_hash=before_hash,
            expected_design_hash=base_hash,
        )
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.LEDGER_CHAIN_MISMATCH,
            f"base design cannot advance ledger-before: {type(error).__name__}: {error}",
        )
    if after.campaign_contract_hash != before.campaign_contract_hash:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CAMPAIGN_MISMATCH,
            "ledger-after changed campaign",
        )
    if after.controller_run_hash != before.controller_run_hash:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.CONTROLLER_MISMATCH,
            "ledger-after changed controller run",
        )
    if after.proposal_generation != before.proposal_generation + 1:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.GENERATION_MISMATCH,
            "ledger-after is not the direct successor generation",
        )
    if after_hash != expected_after.ledger_hash:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.LEDGER_CHAIN_MISMATCH,
            "ledger-after is not the exact base-design successor",
        )

    wrapper_hash: str | None = None
    if hybrid_wrapper is not None:
        if type(hybrid_wrapper) is not V1GroupSequentialEmpiricalBernsteinSerflingDesign:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
                "hybrid wrapper must have the exact HS/EB wrapper type",
            )
        try:
            wrapper_hash = hybrid_wrapper.assert_integrity()
        except Exception as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.PARENT_INTEGRITY,
                f"hybrid wrapper failed integrity replay: {type(error).__name__}: {error}",
            )
        if hybrid_wrapper.base_design_hash != base_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.HYBRID_BASE_MISMATCH,
                "hybrid wrapper does not wrap the charged base design",
            )
        if wrapper_hash == base_hash:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.HYBRID_BASE_MISMATCH,
                "hybrid wrapper hash cannot impersonate the charged base hash",
            )

    cells = _validate_cell_axis(
        alpha_family_plan.cell_axis,
        generation_alpha_allocation=allocation,
    )
    corrected_a_total = math.fsum(
        cell.alpha_spend
        for cell in cells
        if cell.role is V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER
    )
    base_look_total = base.total_allocated_alpha
    if Fraction.from_float(corrected_a_total) != Fraction.from_float(base_look_total):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.BASE_CELL_ALPHA_MISMATCH,
            "corrected-A cells do not exactly account for base-design look alpha",
        )
    total = math.fsum(cell.alpha_spend for cell in cells)
    return V1SampledV7DAlphaFamilyRegistryReceipt(
        master_registration_hash=master_hash,
        proposal_order_alpha_schedule_hash=(
            master.proposal_order_alpha_schedule_hash
        ),
        alpha_family_plan_hash=plan_hash,
        campaign_contract_hash=before.campaign_contract_hash,
        controller_run_hash=before.controller_run_hash,
        proposal_generation=before.proposal_generation,
        trajectory_familywise_alpha=before.trajectory_familywise_alpha,
        trajectory_alpha_committed_before_generation=(
            before.trajectory_alpha_committed_before_generation
        ),
        generation_alpha_allocation=allocation,
        ledger_before_hash=before_hash,
        charged_base_design_hash=base_hash,
        hybrid_wrapper_design_hash=wrapper_hash,
        ledger_after_hash=after_hash,
        ledger_after_generation=after.proposal_generation,
        cell_axis=cells,
        cell_axis_hash=_cell_axis_hash(cells),
        role_alpha_allocations=alpha_family_plan.role_alpha_allocations,
        role_registered_alpha_totals=(
            alpha_family_plan.role_registered_alpha_totals
        ),
        corrected_a_alpha_by_look=alpha_family_plan.corrected_a_alpha_by_look,
        base_design_registered_look_alpha_total=base_look_total,
        corrected_a_cell_alpha_total=corrected_a_total,
        registered_cell_alpha_total=total,
        unused_generation_alpha=math.fsum(
            role_allocation - spent
            for role_allocation, spent in zip(
                alpha_family_plan.role_alpha_allocations,
                alpha_family_plan.role_registered_alpha_totals,
                strict=True,
            )
        ),
        hybrid_wrapper_present=hybrid_wrapper is not None,
        _issuance_token=_ISSUER_TOKEN,
    )


def export_v1_sampled_v7_d_alpha_family_registry_replay(
    receipt: V1SampledV7DAlphaFamilyRegistryReceipt,
) -> dict[str, object]:
    if type(receipt) is not V1SampledV7DAlphaFamilyRegistryReceipt:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "alpha-family replay export requires the exact receipt type",
        )
    return {
        "schema": SAMPLED_V7_D_ALPHA_FAMILY_REPLAY_SCHEMA,
        "receipt_payload": receipt._hash_payload(),
        "receipt_hash": receipt.assert_integrity(),
    }


def restore_v1_sampled_v7_d_alpha_family_registry_replay(
    envelope: Mapping[str, object],
) -> V1SampledV7DAlphaFamilyRegistryReceipt:
    if not isinstance(envelope, Mapping):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.INVALID_TYPE,
            "alpha-family replay envelope must be one mapping",
        )
    payload = dict(envelope)
    if set(payload) != {"schema", "receipt_payload", "receipt_hash"}:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay envelope fields are invalid",
        )
    if payload["schema"] != SAMPLED_V7_D_ALPHA_FAMILY_REPLAY_SCHEMA:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay envelope schema is invalid",
        )
    if not isinstance(payload["receipt_payload"], Mapping):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay receipt payload must be one mapping",
        )
    constructor = dict(payload["receipt_payload"])
    raw_cells = constructor.get("cell_axis")
    if not isinstance(raw_cells, (tuple, list)):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay cell axis is invalid",
        )
    cells: list[V1SampledV7DAlphaFamilyCell] = []
    expected_cell_keys = {
        "schema",
        "role",
        "cell_id",
        "cell_scope_hash",
        "population_size",
        "draw_count",
        "exact_census",
        "alpha_spend",
        "alpha_spend_semantics",
    }
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping) or set(raw_cell) != expected_cell_keys:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "alpha-family replay cell fields are invalid",
            )
        cell_payload = dict(raw_cell)
        exact_census = cell_payload.pop("exact_census")
        try:
            role = V1SampledV7DAlphaCellRole(cell_payload.pop("role"))
        except (TypeError, ValueError) as error:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                f"alpha-family replay cell role is invalid: {error}",
            )
        cell = V1SampledV7DAlphaFamilyCell(role=role, **cell_payload)  # type: ignore[arg-type]
        if exact_census is not cell.exact_census:
            _refuse(
                V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
                "alpha-family replay census coordinate changed",
            )
        cells.append(cell)
    constructor["cell_axis"] = tuple(cells)
    raw_missing = constructor.get("missing_premises")
    if not isinstance(raw_missing, (tuple, list)):
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay missing-premise axis is invalid",
        )
    constructor["missing_premises"] = tuple(raw_missing)
    expected_hash = payload["receipt_hash"]
    if type(expected_hash) is not str or not expected_hash:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            "alpha-family replay receipt hash is invalid",
        )
    try:
        receipt = V1SampledV7DAlphaFamilyRegistryReceipt(
            **constructor,  # type: ignore[arg-type]
            _issuance_token=_ISSUER_TOKEN,
        )
    except V1SampledV7DAlphaFamilyRegistryRefusal as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            f"alpha-family replay payload violates the sealed contract: {error}",
        )
    except Exception as error:
        _refuse(
            V1SampledV7DAlphaFamilyRegistryRefusalCode.REPLAY_MISMATCH,
            f"alpha-family replay payload is invalid: {type(error).__name__}: {error}",
        )
    receipt.assert_integrity(expected_hash)
    return receipt


def _rebuild_cell(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledV7DAlphaFamilyCell:
    result = V1SampledV7DAlphaFamilyCell(**payload)  # type: ignore[arg-type]
    result.assert_integrity(expected_hash)
    return result


def _rebuild_cell_scope(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledV7DAlphaFamilyCellScope:
    result = V1SampledV7DAlphaFamilyCellScope(**payload)  # type: ignore[arg-type]
    result.assert_integrity(expected_hash)
    return result


def _rebuild_plan(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledV7DAlphaFamilyPlan:
    result = V1SampledV7DAlphaFamilyPlan(
        **payload,  # type: ignore[arg-type]
        _issuance_token=_PLAN_ISSUER_TOKEN,
    )
    result.assert_integrity(expected_hash)
    return result


def _rebuild_registry_receipt(
    payload: dict[str, object],
    expected_hash: str,
) -> V1SampledV7DAlphaFamilyRegistryReceipt:
    result = V1SampledV7DAlphaFamilyRegistryReceipt(
        **payload,  # type: ignore[arg-type]
        _issuance_token=_ISSUER_TOKEN,
    )
    result.assert_integrity(expected_hash)
    return result


__all__ = [
    "SAMPLED_V7_D_ALPHA_CELL_ROLE_AXIS",
    "SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_CELL_SCOPE_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_PLAN_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_LEGACY_REGISTRY_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_LEGACY_REPLAY_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_MISSING_PREMISES",
    "SAMPLED_V7_D_ALPHA_FAMILY_REGISTRATION_AUTHORITY",
    "SAMPLED_V7_D_ALPHA_FAMILY_REGISTRY_SCHEMA",
    "SAMPLED_V7_D_ALPHA_FAMILY_REPLAY_SCHEMA",
    "SAMPLED_V7_D_ALPHA_SPEND_SEMANTICS",
    "SAMPLED_V7_D_ALPHA_MAXIMIN_POLICY",
    "SAMPLED_V7_D_ALPHA_PLAN_AUTHORITY",
    "V1SampledV7DAlphaCellRole",
    "V1SampledV7DAlphaFamilyCell",
    "V1SampledV7DAlphaFamilyCellScope",
    "V1SampledV7DAlphaFamilyPlan",
    "V1SampledV7DAlphaFamilyRegistryReceipt",
    "V1SampledV7DAlphaFamilyRegistryRefusal",
    "V1SampledV7DAlphaFamilyRegistryRefusalCode",
    "export_v1_sampled_v7_d_alpha_family_registry_replay",
    "build_v1_sampled_v7_d_alpha_family_plan",
    "issue_v1_sampled_v7_d_alpha_family_registry",
    "restore_v1_sampled_v7_d_alpha_family_registry_replay",
]
