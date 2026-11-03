"""Source-owned Phase-A action quotient and fail-closed reach-margin seam.

The positive sampled-V7 source image closes the complete *registered* relation
from Phase-A resource/action faces to branch/action aliases and packed physical
actions.  This module joins that relation to one exact live Phase-A
decision-margin lease and the legacy four-owner lineage source without
accepting caller-authored numbers, hashes, or completeness claims.  That
legacy scalar axis is provenance only: its entries do not share a norm or
unit.  Complete closed
action-cell coverage is categorical: a tie belongs to every compatible cell,
and exact packed-byte equality is a quotient relation, not a scalar margin.
Consequently, a face whose complete alias image has one physical-action class
is invariant without spending any numerical radius.

What remains continuous is narrower: reaching a resource-cell exit, a Q
feasibility boundary, or a support/action face whose exact quotient contains
more than one physical action.  Corrected-A and signed-F bands are Phase-B-only
and have exactly zero Phase-A causal sensitivity.  Literal-J retains distinct
``M_J z`` and scalar-Q bands; FD retains both spectral ``rho`` and diagonal
``d`` envelopes; solver error retains resource-endpoint and Q intervals in
their native coordinates.  The retained graph does not yet provide the
missing norm-compatible map from those typed J/FD/solver sources to each
decision-changing branch endpoint.
Therefore the registered quotient can be positively receipted, while the
continuum reach margin remains exactly zero.  Equal sampled vertices are never
treated as continuum coverage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, Iterator, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_v7_complete_action_coordinate_packed_crosswalk import (
    V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
    V1SampledV7StructuralActionCrosswalkRow,
)
from .modern_v1_sampled_v7_d_source_image_binding import (
    V1SampledV7DSourceImageBindingReceipt,
)
from .modern_v1_signed_work_phase_a_endogenous_reach_source import (
    PHASE_A_ENDOGENOUS_REACH_API_NAME,
    PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE,
    PHASE_A_ENDOGENOUS_REACH_REASON,
    PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES,
    PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS,
    V1PhaseAEndogenousTypedSourceRequirementSet,
)
from .modern_v1_signed_work_decision_margin_source_cuda import (
    V1SignedWorkLiveOwnerRadiusSourceReceipt,
    issue_v1_signed_work_live_owner_radius_source,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    V1SignedWorkCUDARegisteredUncertaintyCell,
    V1SignedWorkCUDAResourceCellBinding,
)
from .modern_v1_signed_work_least_work_reference import (
    V1SignedWorkLeastWorkBranch,
)
from .modern_v1_signed_work_resource_frontier_cuda import (
    V1SignedWorkResourceFrontierCUDADecisionMarginLease,
    V1SignedWorkResourceFrontierCUDADecisionMarginState,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellIdentity,
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
)


PHASE_A_COMPLETE_ALIAS_MAP_SCHEMA: Final = (
    "partition_free_v1_phase_A_complete_registered_face_action_alias_map_v1"
)
PHASE_A_COMPLETE_ALIAS_MAP_AUTHORITY: Final = (
    "complete_registered_phase_A_closed_action_cell_face_resource_cell_branch_"
    "action_and_exact_physical_dedup_quotient_relation_only_no_continuum_reach_"
    "D_stability_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
PHASE_A_ALIAS_SAFE_MARGIN_MISSING_SCHEMA: Final = (
    "partition_free_v1_phase_A_alias_safe_margin_missing_endogenous_reach_v3"
)
PHASE_A_ALIAS_SAFE_MARGIN_MISSING_AUTHORITY: Final = (
    "registered_categorical_action_quotient_closed_but_continuum_reach_margin_"
    "zero_legacy_owner_scalar_axis_provenance_only_missing_typed_source_to_"
    "branch_endpoint_Q_motion_map_to_only_"
    "decision_changing_boundaries_no_D_stability_C_K_O_commit_terminal_"
    "performance_or_pareto_authority_v3"
)

_PREFIX: Final = "partition-free-v1-phase-A-alias-safe-margin-"
_ALIAS_TOKEN: Final = object()
_MISSING_TOKEN: Final = object()


class V1PhaseAAliasSafeMarginRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_OR_EVICTED_SOURCE = "stale_or_evicted_source"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    INCOMPLETE_ALIAS_IMAGE = "incomplete_alias_image"
    ALIAS_DISAGREEMENT = "alias_disagreement"
    MISSING_ENDOGENOUS_DECISION_BOUNDARY_REACH = (
        "missing_endogenous_decision_boundary_reach"
    )
    AUTHORITY_VIOLATION = "authority_violation"


@dataclass(frozen=True, slots=True)
class V1PhaseAEndogenousDecisionBoundaryReachRequirement:
    """One typed missing theorem, never four interchangeable caller floats."""

    producer_type: str = PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE
    api_name: str = PHASE_A_ENDOGENOUS_REACH_API_NAME
    supplies: tuple[str, ...] = PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES
    reason: str = PHASE_A_ENDOGENOUS_REACH_REASON
    typed_source_requirements: V1PhaseAEndogenousTypedSourceRequirementSet = (
        PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS
    )

    def __post_init__(self) -> None:
        for name in ("producer_type", "api_name", "reason"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be one nonempty string")
        if (
            type(self.supplies) is not tuple
            or not self.supplies
            or any(type(value) is not str or not value for value in self.supplies)
            or len(set(self.supplies)) != len(self.supplies)
        ):
            raise TypeError("operator requirement supplies must be unique strings")
        if (
            type(self.typed_source_requirements)
            is not V1PhaseAEndogenousTypedSourceRequirementSet
            or self.typed_source_requirements
            != PHASE_A_ENDOGENOUS_TYPED_SOURCE_REQUIREMENTS
        ):
            raise TypeError("operator requirement must bind the canonical typed sources")

    @property
    def requirement_id(self) -> str:
        return f"{self.producer_type}.{self.api_name}"

    @property
    def requirement_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "operator-requirement-")


MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE: Final = (
    V1PhaseAEndogenousDecisionBoundaryReachRequirement()
)


class V1PhaseAAliasSafeMarginRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAAliasSafeMarginRefusalCode,
        message: str,
        *,
        alias_map_receipt: V1PhaseACompleteRegisteredAliasMapReceipt | None = None,
        missing_reach_receipt: (
            V1PhaseAAliasSafeMarginMissingReachReceipt | None
        ) = None,
    ) -> None:
        if type(code) is not V1PhaseAAliasSafeMarginRefusalCode:
            raise TypeError("Phase-A alias-safe refusal code must be exact")
        if (
            alias_map_receipt is not None
            and type(alias_map_receipt)
            is not V1PhaseACompleteRegisteredAliasMapReceipt
        ):
            raise TypeError("alias-map refusal premise must be one exact receipt")
        if (
            missing_reach_receipt is not None
            and type(missing_reach_receipt)
            is not V1PhaseAAliasSafeMarginMissingReachReceipt
        ):
            raise TypeError("missing-reach premise must be one exact receipt")
        self.code = code
        self.alias_map_receipt = alias_map_receipt
        self.missing_reach_receipt = missing_reach_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAAliasSafeMarginRefusalCode,
    message: str,
    *,
    alias_map_receipt: V1PhaseACompleteRegisteredAliasMapReceipt | None = None,
    missing_reach_receipt: (
        V1PhaseAAliasSafeMarginMissingReachReceipt | None
    ) = None,
) -> NoReturn:
    raise V1PhaseAAliasSafeMarginRefusal(
        code,
        message,
        alias_map_receipt=alias_map_receipt,
        missing_reach_receipt=missing_reach_receipt,
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1PhaseARegisteredFaceActionAlias:
    """One retained registered face/resource/branch/action alias."""

    registered_cell_hash: str
    uncertainty_grid_key: tuple[str, str, str, str]
    phase_a_handle_hash: str
    phase_a_result_hash: str
    resource_cell_hash: str
    resource_cell_binding_hash: str
    resource_coordinate_hash: str
    phase_a_resource_alias_hash: str
    phase_a_resource_alias_id: str
    phase_a_resource_cell_id: str
    phase_a_route_face_hash: str
    phase_a_action_cell_hash: str
    phase_a_action_coordinate_hash: str
    phase_a_subset_identity_hash: str
    branch: V1RelaxedKKTBranch
    qualification_crosswalk_hash: str
    physical_action_identity_hash: str
    packed_dedup_identity: tuple[str, str, str, str, int]
    resulting_mask_packed_sha256: str
    source_structural_row_hash: str
    alias_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "registered_cell_hash",
            "phase_a_handle_hash",
            "phase_a_result_hash",
            "resource_cell_hash",
            "resource_cell_binding_hash",
            "resource_coordinate_hash",
            "phase_a_resource_alias_hash",
            "phase_a_resource_alias_id",
            "phase_a_resource_cell_id",
            "phase_a_route_face_hash",
            "phase_a_action_cell_hash",
            "phase_a_action_coordinate_hash",
            "phase_a_subset_identity_hash",
            "qualification_crosswalk_hash",
            "physical_action_identity_hash",
            "resulting_mask_packed_sha256",
            "source_structural_row_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.uncertainty_grid_key) is not tuple
            or len(self.uncertainty_grid_key) != 4
            or any(type(value) is not str or not value for value in self.uncertainty_grid_key)
            or type(self.branch) is not V1RelaxedKKTBranch
            or type(self.packed_dedup_identity) is not tuple
            or len(self.packed_dedup_identity) != 5
            or self.packed_dedup_identity[0]
            != self.resulting_mask_packed_sha256
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
                "registered alias lost its grid, branch, or packed identity",
            )
        object.__setattr__(
            self,
            "alias_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "alias_hash"
                },
                prefix=_PREFIX + "registered-alias-",
            ),
        )


def _shell_branch(branch: V1RelaxedKKTBranch) -> V1SignedWorkVariableShellQBranch:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return V1SignedWorkVariableShellQBranch.SPECTRAL
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return V1SignedWorkVariableShellQBranch.DIAGONAL_L1
    _refuse(
        V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
        "registered action alias has a noncanonical Q branch",
    )


@dataclass(frozen=True, slots=True)
class V1PhaseARegisteredActionCellQuotient:
    """Exact physical-action quotient of one complete closed action-cell cover."""

    registered_cell_hash: str
    resource_cell_hash: str
    resource_cell_binding_hash: str
    resource_coordinate_hash: str
    branch: V1RelaxedKKTBranch
    phase_a_resource_alias_hash: str
    phase_a_resource_alias_id: str
    phase_a_resource_cell_id: str
    action_cell_hashes: tuple[str, ...]
    action_alias_hashes: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    packed_dedup_identities: tuple[tuple[str, str, str, str, int], ...]
    complete_closed_action_cell_cover: bool = True
    exact_ties_multicovered: bool = True
    exact_packed_byte_quotient: bool = True
    categorical_single_physical_action: bool = field(init=False)
    quotient_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "registered_cell_hash",
            "resource_cell_hash",
            "resource_cell_binding_hash",
            "resource_coordinate_hash",
            "phase_a_resource_alias_hash",
            "phase_a_resource_alias_id",
            "phase_a_resource_cell_id",
        ):
            _identity(getattr(self, name), name=name)
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("action quotient branch must be exact")
        for name in (
            "action_cell_hashes",
            "action_alias_hashes",
            "physical_action_identity_hashes",
        ):
            axis = getattr(self, name)
            if (
                type(axis) is not tuple
                or not axis
                or any(type(value) is not str or not value for value in axis)
                or len(axis) != len(set(axis))
            ):
                _refuse(
                    V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
                    f"{name} must be one complete unique axis",
                )
        packed = self.packed_dedup_identities
        if (
            type(packed) is not tuple
            or not packed
            or any(type(value) is not tuple or len(value) != 5 for value in packed)
            or len(packed) != len(set(packed))
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
                "packed-action quotient is malformed or duplicated",
            )
        flags = (
            self.complete_closed_action_cell_cover,
            self.exact_ties_multicovered,
            self.exact_packed_byte_quotient,
        )
        if any(type(value) is not bool or value is not True for value in flags):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.AUTHORITY_VIOLATION,
                "action quotient lost its exact cover or packed-byte premise",
            )
        object.__setattr__(
            self,
            "categorical_single_physical_action",
            len(self.physical_action_identity_hashes) == 1,
        )
        object.__setattr__(
            self,
            "quotient_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "quotient_hash"
                },
                prefix=_PREFIX + "action-cell-quotient-",
            ),
        )


@dataclass(frozen=True, slots=True)
class V1PhaseARegisteredSupportFaceQuotient:
    """Categorical image of one registered support face in one resource cell."""

    boundary_kind: str
    native_scalar_unit: str
    resource_cell_hash: str
    branches: tuple[V1RelaxedKKTBranch, ...]
    action_quotient_hashes: tuple[str, ...]
    action_alias_hashes: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    preferred_physical_action_identity_hashes: tuple[str, ...]
    complete_closed_face_image: bool
    categorical_physical_action_invariance: bool
    numerical_reach_required: bool
    face_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("boundary_kind", "native_scalar_unit", "resource_cell_hash"):
            _identity(getattr(self, name), name=name)
        if (
            type(self.branches) is not tuple
            or not self.branches
            or any(type(value) is not V1RelaxedKKTBranch for value in self.branches)
            or len(self.branches) != len(set(self.branches))
        ):
            raise TypeError("support-face quotient requires unique exact branches")
        for name in (
            "action_quotient_hashes",
            "action_alias_hashes",
            "physical_action_identity_hashes",
            "preferred_physical_action_identity_hashes",
        ):
            axis = getattr(self, name)
            if (
                type(axis) is not tuple
                or any(type(value) is not str or not value for value in axis)
                or len(axis) != len(set(axis))
            ):
                raise TypeError(f"{name} must be one unique immutable axis")
        flags = (
            self.complete_closed_face_image,
            self.categorical_physical_action_invariance,
            self.numerical_reach_required,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("support-face quotient flags must be exact")
        expected_categorical = (
            self.complete_closed_face_image
            and len(
                set(self.physical_action_identity_hashes)
                | set(self.preferred_physical_action_identity_hashes)
            )
            == 1
        )
        if (
            self.categorical_physical_action_invariance is not expected_categorical
            or self.numerical_reach_required is expected_categorical
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.AUTHORITY_VIOLATION,
                "support-face categorical/numerical split does not replay",
            )
        object.__setattr__(
            self,
            "face_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "face_hash"
                },
                prefix=_PREFIX + "support-face-quotient-",
            ),
        )


def _binding_for_row(
    *,
    cell: V1SignedWorkCUDARegisteredUncertaintyCell,
    row: V1SampledV7StructuralActionCrosswalkRow,
) -> V1SignedWorkCUDAResourceCellBinding:
    matches = tuple(
        value
        for value in cell.resource_cell_bindings
        if value.resource_coordinate_hash == row.resource_coordinate_hash
        and value.resource_cell_hash == row.resource_cell_hash
    )
    if len(matches) != 1:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
            "one structural row has no unique registered resource-cell binding",
        )
    return matches[0]


def _derive_alias_map(
    *,
    decision_margin_lease: V1SignedWorkResourceFrontierCUDADecisionMarginLease,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt,
) -> dict[str, object]:
    if (
        type(decision_margin_lease)
        is not V1SignedWorkResourceFrontierCUDADecisionMarginLease
        or type(source_image) is not V1SampledV7DSourceImageBindingReceipt
        or type(owner_radius_source) is not V1SignedWorkLiveOwnerRadiusSourceReceipt
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INVALID_TYPE,
            "alias-map derivation requires exact live lease, source image, and owner source",
        )
    try:
        state = decision_margin_lease.state
        source_image_hash = source_image.assert_integrity()
        owner_radius_source.assert_live_identity()
        crosswalk = source_image._complete_crosswalk
        crosswalk_hash = crosswalk.assert_integrity()
        qualification = source_image._cuda_qualification
        qualification.assert_live_identity()
        registry = source_image._phase_a_registry
        registry.assert_live_sources()
        registry_receipt = registry.receipt
    except Exception as error:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
            f"one retained alias-map parent failed replay: {type(error).__name__}: {error}",
        )
    if (
        type(state) is not V1SignedWorkResourceFrontierCUDADecisionMarginState
        or type(crosswalk)
        is not V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt
        or type(registry) is not V1SignedWorkVariableShellActionCellRegistry
        or owner_radius_source.source_image is not source_image
        or owner_radius_source.source_qualification is not qualification
        or owner_radius_source.source_image_receipt_hash != source_image_hash
        or owner_radius_source.finite_cuda_qualification_hash
        != qualification.qualification_hash
        or source_image.cuda_qualification_hash != qualification.qualification_hash
        or crosswalk.cuda_qualification_hash != qualification.qualification_hash
        or crosswalk._phase_a_registry is not registry
        or crosswalk.phase_a_registry_receipt_hash != registry_receipt.receipt_hash
        or source_image.phase_a_registry_receipt_hash != registry_receipt.receipt_hash
        or registry_receipt.complete_action_cell_coverage is not True
        or registry_receipt.exact_ties_registered_in_every_compatible_cell is not True
        or state.route_axis_hash != source_image.route_axis_hash
        or state.route_count != source_image.route_count
        or decision_margin_lease._handle.route_axis_hash != source_image.route_axis_hash
        or decision_margin_lease._handle.route_count != source_image.route_count
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "lease, source image, owner radii, crosswalk, and qualification are not one graph",
        )

    cells = tuple(qualification.source_cells)
    if (
        not cells
        or any(type(value) is not V1SignedWorkCUDARegisteredUncertaintyCell for value in cells)
        or len({value.registered_cell_hash for value in cells}) != len(cells)
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "qualification lost its unique registered-cell family",
        )
    by_cell = {value.registered_cell_hash: value for value in cells}
    structural_rows = tuple(crosswalk.structural_rows)
    aliases: list[V1PhaseARegisteredFaceActionAlias] = []
    for row in structural_rows:
        cell = by_cell.get(row.qualification_registered_cell_hash)
        if cell is None or cell.grid_key != row.qualification_grid_key:
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
                "one structural alias left its registered uncertainty cell",
            )
        binding = _binding_for_row(cell=cell, row=row)
        route_face_hash = content_hash(
            (
                cell.registered_cell_hash,
                row.phase_a_action_cell_hash,
                row.phase_a_action_coordinate_hash,
                row.phase_a_subset_identity_hash,
            ),
            prefix=_PREFIX + "registered-route-face-",
        )
        aliases.append(
            V1PhaseARegisteredFaceActionAlias(
                registered_cell_hash=cell.registered_cell_hash,
                uncertainty_grid_key=cell.grid_key,
                phase_a_handle_hash=cell.phase_a_state_handle.handle_hash,
                phase_a_result_hash=cell.phase_a_result.result_hash,
                resource_cell_hash=binding.resource_cell_hash,
                resource_cell_binding_hash=binding.binding_hash,
                resource_coordinate_hash=binding.resource_coordinate_hash,
                phase_a_resource_alias_hash=row.phase_a_resource_alias_hash,
                phase_a_resource_alias_id=row.phase_a_resource_alias_id,
                phase_a_resource_cell_id=row.phase_a_resource_cell_id,
                phase_a_route_face_hash=route_face_hash,
                phase_a_action_cell_hash=row.phase_a_action_cell_hash,
                phase_a_action_coordinate_hash=row.phase_a_action_coordinate_hash,
                phase_a_subset_identity_hash=row.phase_a_subset_identity_hash,
                branch=row.branch,
                qualification_crosswalk_hash=row.qualification_crosswalk_hash,
                physical_action_identity_hash=row.physical_action_identity_hash,
                packed_dedup_identity=row.packed_dedup_identity,
                resulting_mask_packed_sha256=row.resulting_mask_packed_sha256,
                source_structural_row_hash=row.structural_row_hash,
            )
        )
    alias_rows = tuple(aliases)
    if (
        not alias_rows
        or len(alias_rows) != len(structural_rows)
        or len({value.alias_hash for value in alias_rows}) != len(alias_rows)
        or tuple(value.source_structural_row_hash for value in alias_rows)
        != tuple(value.structural_row_hash for value in structural_rows)
        or {
            (value.resource_cell_hash, value.physical_action_identity_hash)
            for value in alias_rows
        }
        != set(source_image.cuda_physical_action_family)
        or {value.resource_cell_hash for value in alias_rows}
        != set(source_image.cuda_resource_cell_family)
        or {
            (value.branch.value, value.phase_a_action_coordinate_hash)
            for value in alias_rows
        }
        != set(crosswalk.branch_action_projection_domain)
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "derived aliases are not set-equal to the complete registered source image",
        )

    registry_cells = tuple(registry.iter_action_cells())
    if (
        not registry_cells
        or len(registry_cells) != registry_receipt.total_action_cell_count
        or any(
            type(value) is not V1SignedWorkVariableShellActionCellIdentity
            for value in registry_cells
        )
        or len({value.action_cell_hash for value in registry_cells})
        != len(registry_cells)
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "live registry no longer exposes its complete unique action-cell axis",
        )
    registry_cells_by_alias: dict[
        tuple[V1SignedWorkVariableShellQBranch, str],
        tuple[V1SignedWorkVariableShellActionCellIdentity, ...],
    ] = {}
    for branch in (
        V1SignedWorkVariableShellQBranch.SPECTRAL,
        V1SignedWorkVariableShellQBranch.DIAGONAL_L1,
    ):
        alias_ids = tuple(
            sorted(
                {
                    value.resource_alias_id
                    for value in registry_cells
                    if value.branch is branch
                }
            )
        )
        for alias_id in alias_ids:
            registry_cells_by_alias[(branch, alias_id)] = tuple(
                sorted(
                    (
                        value
                        for value in registry_cells
                        if value.branch is branch
                        and value.resource_alias_id == alias_id
                    ),
                    key=lambda value: value.action_cell_ordinal,
                )
            )

    grouped_aliases: dict[
        tuple[str, str, V1RelaxedKKTBranch, str],
        list[V1PhaseARegisteredFaceActionAlias],
    ] = {}
    for alias in alias_rows:
        grouped_aliases.setdefault(
            (
                alias.registered_cell_hash,
                alias.resource_cell_hash,
                alias.branch,
                alias.phase_a_resource_alias_id,
            ),
            [],
        ).append(alias)
    action_quotients: list[V1PhaseARegisteredActionCellQuotient] = []
    for (_registered, _resource, branch, resource_alias_id), values in sorted(
        grouped_aliases.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2].value,
            item[0][3],
        ),
    ):
        group = tuple(values)
        expected_cells = registry_cells_by_alias.get(
            (_shell_branch(branch), resource_alias_id)
        )
        if not expected_cells:
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
                "one source-image alias has no registered closed action-cell cover",
            )
        source = registry.branch_source(_shell_branch(branch))
        resource_alias = source.resource_alias(resource_alias_id)
        expected_hashes = tuple(value.action_cell_hash for value in expected_cells)
        expected_by_hash = {value.action_cell_hash: value for value in expected_cells}
        observed_hashes = tuple(sorted(value.phase_a_action_cell_hash for value in group))
        if (
            len(group) != len(expected_cells)
            or observed_hashes != tuple(sorted(expected_hashes))
            or len({value.phase_a_action_cell_hash for value in group}) != len(group)
            or any(
                value.phase_a_resource_alias_hash != resource_alias.alias_hash
                or value.phase_a_resource_cell_id
                != expected_by_hash[value.phase_a_action_cell_hash].phase_a_resource_cell_id
                for value in group
            )
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
                "a registered branch/resource alias lost one closed action cell",
            )
        physical_to_packed: dict[str, tuple[str, str, str, str, int]] = {}
        packed_to_physical: dict[tuple[str, str, str, str, int], str] = {}
        for value in group:
            previous_packed = physical_to_packed.setdefault(
                value.physical_action_identity_hash,
                value.packed_dedup_identity,
            )
            previous_physical = packed_to_physical.setdefault(
                value.packed_dedup_identity,
                value.physical_action_identity_hash,
            )
            if (
                previous_packed != value.packed_dedup_identity
                or previous_physical != value.physical_action_identity_hash
            ):
                _refuse(
                    V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
                    "packed-byte equality did not define one exact physical-action quotient",
                )
        first = group[0]
        action_quotients.append(
            V1PhaseARegisteredActionCellQuotient(
                registered_cell_hash=first.registered_cell_hash,
                resource_cell_hash=first.resource_cell_hash,
                resource_cell_binding_hash=first.resource_cell_binding_hash,
                resource_coordinate_hash=first.resource_coordinate_hash,
                branch=first.branch,
                phase_a_resource_alias_hash=first.phase_a_resource_alias_hash,
                phase_a_resource_alias_id=first.phase_a_resource_alias_id,
                phase_a_resource_cell_id=first.phase_a_resource_cell_id,
                action_cell_hashes=tuple(sorted(expected_hashes)),
                action_alias_hashes=tuple(sorted(value.alias_hash for value in group)),
                physical_action_identity_hashes=tuple(sorted(physical_to_packed)),
                packed_dedup_identities=tuple(sorted(packed_to_physical)),
            )
        )
    quotient_axis = tuple(action_quotients)
    if (
        not quotient_axis
        or len({value.quotient_hash for value in quotient_axis}) != len(quotient_axis)
        or {alias.alias_hash for alias in alias_rows}
        != {
            alias_hash
            for quotient in quotient_axis
            for alias_hash in quotient.action_alias_hashes
        }
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "closed action-cell quotients do not partition the alias image exactly",
        )

    state_binding_hashes = tuple(state.resource_cell_binding_hashes)
    leased_cells = tuple(
        value
        for value in cells
        if value.phase_a_state_handle is decision_margin_lease._handle
        and value.phase_a_result.result_hash == state.result_hash
        and tuple(item.binding_hash for item in value.resource_cell_bindings)
        == state_binding_hashes
    )
    if not leased_cells:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "the live decision-margin lease is absent from the source-image cell family",
        )
    leased_cell_hashes = tuple(value.registered_cell_hash for value in leased_cells)
    boundary_pairs = {
        (value.resource_cell_binding_hash, value.resource_cell_hash)
        for value in state.resource_cell_boundaries
    }
    leased_binding_pairs = {
        (binding.binding_hash, binding.resource_cell_hash)
        for cell in leased_cells
        for binding in cell.resource_cell_bindings
    }
    if boundary_pairs != leased_binding_pairs:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.ALIAS_DISAGREEMENT,
            "leased resource-cell boundaries differ from their registered source aliases",
        )

    preferred_by_resource: dict[str, tuple[str, ...]] = {}
    for resource_cell_hash in source_image.cuda_resource_cell_family:
        preferred_by_resource[resource_cell_hash] = tuple(
            sorted(
                physical_hash
                for cell_hash, physical_hash in (
                    source_image.cuda_preferred_physical_decision_family
                )
                if cell_hash == resource_cell_hash
            )
        )
        if not preferred_by_resource[resource_cell_hash]:
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
                "one registered resource cell lost its preferred physical action",
            )

    support_faces: list[V1PhaseARegisteredSupportFaceQuotient] = []
    for resource_boundary in state.resource_cell_boundaries:
        resource_cell_hash = resource_boundary.resource_cell_hash
        for boundary in state.branch_support_boundaries:
            branches = tuple(V1RelaxedKKTBranch(value.value) for value in boundary.branches)
            quotients = tuple(
                value
                for value in quotient_axis
                if value.registered_cell_hash in leased_cell_hashes
                and value.resource_cell_hash == resource_cell_hash
                and value.branch in branches
            )
            expected_pairs = {
                (registered_cell_hash, branch)
                for registered_cell_hash in leased_cell_hashes
                for branch in branches
            }
            observed_pairs = {
                (value.registered_cell_hash, value.branch) for value in quotients
            }
            complete_face = expected_pairs == observed_pairs
            action_alias_hashes = tuple(
                sorted(
                    {
                        alias_hash
                        for quotient in quotients
                        for alias_hash in quotient.action_alias_hashes
                    }
                )
            )
            physical_hashes = tuple(
                sorted(
                    {
                        physical_hash
                        for quotient in quotients
                        for physical_hash in quotient.physical_action_identity_hashes
                    }
                )
            )
            preferred = preferred_by_resource[resource_cell_hash]
            categorical = (
                complete_face
                and len(set(physical_hashes) | set(preferred)) == 1
            )
            support_faces.append(
                V1PhaseARegisteredSupportFaceQuotient(
                    boundary_kind=boundary.boundary_kind,
                    native_scalar_unit=boundary.native_scalar_unit,
                    resource_cell_hash=resource_cell_hash,
                    branches=branches,
                    action_quotient_hashes=tuple(
                        sorted(value.quotient_hash for value in quotients)
                    ),
                    action_alias_hashes=action_alias_hashes,
                    physical_action_identity_hashes=physical_hashes,
                    preferred_physical_action_identity_hashes=preferred,
                    complete_closed_face_image=complete_face,
                    categorical_physical_action_invariance=categorical,
                    numerical_reach_required=not categorical,
                )
            )
    support_face_axis = tuple(support_faces)
    if not support_face_axis or len(
        {value.face_hash for value in support_face_axis}
    ) != len(support_face_axis):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "support-face quotient axis is empty or duplicated",
        )

    # A complete closed action-cell cover removes the need for any numerical
    # tie/key/packed-byte margin when its exact quotient is a singleton.
    categorical_face_hashes = tuple(
        sorted(
            {
                *(
                    value.quotient_hash
                    for value in quotient_axis
                    if value.categorical_single_physical_action
                ),
                *(
                    value.face_hash
                    for value in support_face_axis
                    if value.categorical_physical_action_invariance
                ),
            }
        )
    )
    decision_changing_face_hashes = tuple(
        sorted(
            {
                *(
                    value.quotient_hash
                    for value in quotient_axis
                    if not value.categorical_single_physical_action
                ),
                *(
                    value.face_hash
                    for value in support_face_axis
                    if value.numerical_reach_required
                ),
            }
        )
    )
    # The registered resource intervals are disjoint and do not themselves
    # certify an adjacent closed continuum cover.  Leaving any one therefore
    # remains a genuine continuous reach question, irrespective of equality
    # among the actions represented *inside* the registered cells.
    resource_cell_exit_hashes = tuple(
        content_hash(
            (
                value.resource_cell_hash,
                value.resource_cell_binding_hash,
                value.cell_lower_bound,
                value.cell_upper_bound,
            ),
            prefix=_PREFIX + "continuous-resource-cell-exit-",
        )
        for value in state.resource_cell_boundaries
    )

    support_axis = tuple(
        (
            value.boundary_kind,
            ",".join(branch.value for branch in value.branches),
            value.action_alias_hashes,
            value.categorical_physical_action_invariance,
        )
        for value in support_face_axis
    )

    return {
        "source_image_receipt_hash": source_image_hash,
        "complete_crosswalk_receipt_hash": crosswalk_hash,
        "action_cell_registry_receipt_hash": registry_receipt.receipt_hash,
        "owner_radius_source_receipt_hash": owner_radius_source.receipt_hash,
        "qualification_hash": qualification.qualification_hash,
        "phase_a_decision_margin_state_hash": state.state_hash,
        "leased_phase_a_handle_hash": decision_margin_lease._handle.handle_hash,
        "leased_registered_cell_hashes": leased_cell_hashes,
        "alias_rows": alias_rows,
        "action_cell_quotients": quotient_axis,
        "support_face_quotients": support_face_axis,
        "categorical_invariant_face_hashes": categorical_face_hashes,
        "decision_changing_face_hashes": decision_changing_face_hashes,
        "continuous_resource_cell_exit_hashes": resource_cell_exit_hashes,
        "support_face_alias_axis": support_axis,
        "registered_cell_hashes": tuple(value.registered_cell_hash for value in cells),
        "resource_cell_family": tuple(source_image.cuda_resource_cell_family),
        "physical_action_family": tuple(source_image.cuda_physical_action_family),
        "branch_action_projection_domain": crosswalk.branch_action_projection_domain,
        "route_axis_hash": source_image.route_axis_hash,
        "route_count": source_image.route_count,
    }


@dataclass(frozen=True, slots=True)
class V1PhaseACompleteRegisteredAliasMapReceipt:
    source_image_receipt_hash: str
    complete_crosswalk_receipt_hash: str
    action_cell_registry_receipt_hash: str
    owner_radius_source_receipt_hash: str
    qualification_hash: str
    phase_a_decision_margin_state_hash: str
    leased_phase_a_handle_hash: str
    leased_registered_cell_hashes: tuple[str, ...]
    alias_rows: tuple[V1PhaseARegisteredFaceActionAlias, ...]
    action_cell_quotients: tuple[V1PhaseARegisteredActionCellQuotient, ...]
    support_face_quotients: tuple[V1PhaseARegisteredSupportFaceQuotient, ...]
    categorical_invariant_face_hashes: tuple[str, ...]
    decision_changing_face_hashes: tuple[str, ...]
    continuous_resource_cell_exit_hashes: tuple[str, ...]
    support_face_alias_axis: tuple[tuple[str, str, tuple[str, ...], bool], ...]
    registered_cell_hashes: tuple[str, ...]
    resource_cell_family: tuple[str, ...]
    physical_action_family: tuple[tuple[str, str], ...]
    branch_action_projection_domain: tuple[tuple[str, str], ...]
    route_axis_hash: str
    route_count: int
    decision_margin_lease: V1SignedWorkResourceFrontierCUDADecisionMarginLease = field(
        repr=False, compare=False
    )
    source_image: V1SampledV7DSourceImageBindingReceipt = field(
        repr=False, compare=False
    )
    owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt = field(
        repr=False, compare=False
    )
    _issuer_token: InitVar[object]
    _integrity_seal: str = field(init=False, repr=False, compare=False)
    complete_registered_alias_relation: bool = True
    complete_closed_action_cell_coverage: bool = True
    exact_ties_multicovered: bool = True
    categorical_same_action_faces_need_no_numeric_margin: bool = True
    all_branch_action_aliases_preserved: bool = True
    physical_byte_dedup_preserved_without_alias_collapse: bool = True
    sampled_vertices_cover_continuum: bool = False
    registered_resource_cells_form_closed_continuum_cover: bool = False
    continuum_margin_authority: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHASE_A_COMPLETE_ALIAS_MAP_AUTHORITY
    schema: str = PHASE_A_COMPLETE_ALIAS_MAP_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ALIAS_TOKEN:
            raise TypeError("Phase-A complete alias-map receipts are issuer-only")
        observed = _derive_alias_map(
            decision_margin_lease=self.decision_margin_lease,
            source_image=self.source_image,
            owner_radius_source=self.owner_radius_source,
        )
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
                "complete alias-map public image differs from its live parents",
            )
        flags = (
            self.complete_registered_alias_relation,
            self.complete_closed_action_cell_coverage,
            self.exact_ties_multicovered,
            self.categorical_same_action_faces_need_no_numeric_margin,
            self.all_branch_action_aliases_preserved,
            self.physical_byte_dedup_preserved_without_alias_collapse,
            self.sampled_vertices_cover_continuum,
            self.registered_resource_cells_form_closed_continuum_cover,
            self.continuum_margin_authority,
            self.d_stability_authority,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in flags)
            or flags
            != (
                True,
                True,
                True,
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
                False,
                False,
            )
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or self.authority_scope != PHASE_A_COMPLETE_ALIAS_MAP_AUTHORITY
            or self.schema != PHASE_A_COMPLETE_ALIAS_MAP_SCHEMA
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.AUTHORITY_VIOLATION,
                "complete alias map crossed its registered-relation authority",
            )
        object.__setattr__(self, "_integrity_seal", self.receipt_hash)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_image_receipt_hash": self.source_image_receipt_hash,
                "complete_crosswalk_receipt_hash": self.complete_crosswalk_receipt_hash,
                "action_cell_registry_receipt_hash": (
                    self.action_cell_registry_receipt_hash
                ),
                "owner_radius_source_receipt_hash": self.owner_radius_source_receipt_hash,
                "qualification_hash": self.qualification_hash,
                "phase_a_decision_margin_state_hash": self.phase_a_decision_margin_state_hash,
                "leased_phase_a_handle_hash": self.leased_phase_a_handle_hash,
                "leased_registered_cell_hashes": self.leased_registered_cell_hashes,
                "alias_row_hashes": tuple(value.alias_hash for value in self.alias_rows),
                "action_cell_quotient_hashes": tuple(
                    value.quotient_hash for value in self.action_cell_quotients
                ),
                "support_face_quotient_hashes": tuple(
                    value.face_hash for value in self.support_face_quotients
                ),
                "categorical_invariant_face_hashes": (
                    self.categorical_invariant_face_hashes
                ),
                "decision_changing_face_hashes": self.decision_changing_face_hashes,
                "continuous_resource_cell_exit_hashes": (
                    self.continuous_resource_cell_exit_hashes
                ),
                "support_face_alias_axis": self.support_face_alias_axis,
                "registered_cell_hashes": self.registered_cell_hashes,
                "resource_cell_family": self.resource_cell_family,
                "physical_action_family": self.physical_action_family,
                "branch_action_projection_domain": self.branch_action_projection_domain,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "sampled_vertices_cover_continuum": False,
                "registered_resource_cells_form_closed_continuum_cover": False,
                "route_scale_d2h_bytes": 0,
                "route_scale_host_tensor_materializations": 0,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "complete-alias-map-",
        )

    def assert_live_identity(self) -> str:
        observed = _derive_alias_map(
            decision_margin_lease=self.decision_margin_lease,
            source_image=self.source_image,
            owner_radius_source=self.owner_radius_source,
        )
        if (
            any(getattr(self, name) != value for name, value in observed.items())
            or self.receipt_hash != self._integrity_seal
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
                "complete registered alias map no longer replays",
            )
        return self.receipt_hash

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("Phase-A complete alias-map receipts are nonserializable")


def bind_v1_phase_a_complete_registered_alias_map_cuda(
    *,
    decision_margin_lease: V1SignedWorkResourceFrontierCUDADecisionMarginLease,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt,
) -> V1PhaseACompleteRegisteredAliasMapReceipt:
    """Bind the full registered relation; accept no caller claims or scalars."""

    public = _derive_alias_map(
        decision_margin_lease=decision_margin_lease,
        source_image=source_image,
        owner_radius_source=owner_radius_source,
    )
    return V1PhaseACompleteRegisteredAliasMapReceipt(
        **public,
        decision_margin_lease=decision_margin_lease,
        source_image=source_image,
        owner_radius_source=owner_radius_source,
        _issuer_token=_ALIAS_TOKEN,
    )


@contextmanager
def open_v1_phase_a_complete_registered_alias_map_cuda(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
) -> Iterator[V1PhaseACompleteRegisteredAliasMapReceipt]:
    """Open the canonical live alias map under its source-owned Phase-A lease.

    Cell choice, uncertainty ownership, and every numerical boundary are
    derived from the complete source image.  The caller cannot select a cell,
    pass a margin, or substitute an owner receipt.  The qualification's first
    cell is canonical because qualification issuance has already reordered the
    complete Cartesian uncertainty product into its frozen product order.

    The yielded receipt is deliberately valid only inside this context.  Its
    integrity replays the exclusive decision-margin lease; returning it after
    exit is stale custody, not durable D evidence.
    """

    if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INVALID_TYPE,
            "canonical alias-map opening requires the exact source image",
        )
    try:
        source_image.assert_integrity()
        qualification = source_image._cuda_qualification
        qualification.assert_live_identity()
        cells = tuple(qualification.source_cells)
        branches = tuple(source_image._branch_evidence)
    except Exception as error:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
            "canonical alias-map source graph failed replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        len(cells) == 0
        or tuple(value.registered_cell_hash for value in cells)
        != tuple(qualification.registered_cell_hashes)
        or len(branches) != 2
        or getattr(branches[0], "_uncertainty", None) is None
        or getattr(branches[1], "_uncertainty", None)
        is not getattr(branches[0], "_uncertainty", None)
    ):
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INCOMPLETE_ALIAS_IMAGE,
            "source image lacks one canonically ordered cell family or one "
            "shared branch uncertainty owner",
        )
    cell = cells[0]
    if type(cell) is not V1SignedWorkCUDARegisteredUncertaintyCell:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.INVALID_TYPE,
            "canonical source cell lost its exact CUDA registered-cell type",
        )
    uncertainty = branches[0]._uncertainty
    try:
        owner_source = issue_v1_signed_work_live_owner_radius_source(
            source_image=source_image,
            qualification=qualification,
            uncertainty=uncertainty,
        )
        lease = cell.phase_a_state_handle.acquire_decision_margin_lease(
            result=cell.phase_a_result,
            resource_cell_bindings=cell.resource_cell_bindings,
        )
    except V1PhaseAAliasSafeMarginRefusal:
        raise
    except Exception as error:
        _refuse(
            V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
            "canonical alias-map lease could not be acquired: "
            f"{type(error).__name__}: {error}",
        )

    # Do not wrap exceptions raised by the synchronous consumer.  They are
    # downstream execution evidence (for example a physical-C refusal), not
    # proof that alias-map acquisition failed.  The lease still closes through
    # its own context manager, while normal completion replays the map once
    # more before releasing custody.
    with lease:
        try:
            alias_map = bind_v1_phase_a_complete_registered_alias_map_cuda(
                decision_margin_lease=lease,
                source_image=source_image,
                owner_radius_source=owner_source,
            )
            alias_map.assert_live_identity()
        except V1PhaseAAliasSafeMarginRefusal:
            raise
        except Exception as error:
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
                "canonical alias-map lease could not be opened: "
                f"{type(error).__name__}: {error}",
            )
        yield alias_map
        alias_map.assert_live_identity()


@dataclass(frozen=True, slots=True)
class V1PhaseAAliasSafeMarginMissingReachReceipt:
    alias_map_receipt_hash: str
    owner_radius_source_receipt_hash: str
    phase_a_decision_margin_state_hash: str
    typed_source_requirement_hash: str
    branch_endpoint_motion_requirement_hash: str
    owner_radius_lineage_custody_axis: tuple[tuple[str, str, float], ...]
    resource_cell_slack_axis: tuple[tuple[str, float], ...]
    branch_support_slack_axis: tuple[tuple[str, tuple[str, ...], float, float], ...]
    categorical_invariant_face_hashes: tuple[str, ...]
    decision_changing_face_hashes: tuple[str, ...]
    continuous_resource_cell_exit_hashes: tuple[str, ...]
    missing_requirement: V1PhaseAEndogenousDecisionBoundaryReachRequirement
    alias_map_receipt: V1PhaseACompleteRegisteredAliasMapReceipt = field(
        repr=False, compare=False
    )
    _issuer_token: InitVar[object]
    receipt_hash: str = field(init=False)
    registered_alias_map_complete: bool = True
    complete_categorical_action_quotient: bool = True
    categorical_faces_require_no_numeric_margin: bool = True
    owner_radius_lineage_custody_bound: bool = True
    legacy_untyped_owner_radius_axis_provenance_only: bool = True
    corrected_a_f_phase_a_causal_sensitivity_exactly_zero: bool = True
    fd_and_solver_embedded_uncertainty_double_charge_forbidden: bool = True
    universal_owner_radius_authority: bool = False
    typed_source_values_endogenously_materialized: bool = False
    norm_compatible_endpoint_operator_present: bool = False
    sampled_vertices_cover_continuum: bool = False
    registered_resource_cells_form_closed_continuum_cover: bool = False
    robust_alias_safe_margin_lower_bound: float = 0.0
    continuum_margin_authority: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHASE_A_ALIAS_SAFE_MARGIN_MISSING_AUTHORITY
    schema: str = PHASE_A_ALIAS_SAFE_MARGIN_MISSING_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _MISSING_TOKEN:
            raise TypeError("Phase-A missing-reach receipts are issuer-only")
        self.alias_map_receipt.assert_live_identity()
        state = self.alias_map_receipt.decision_margin_lease.state
        if (
            self.alias_map_receipt_hash != self.alias_map_receipt.receipt_hash
            or self.owner_radius_source_receipt_hash
            != self.alias_map_receipt.owner_radius_source.receipt_hash
            or self.phase_a_decision_margin_state_hash != state.state_hash
            or self.typed_source_requirement_hash
            != self.missing_requirement.typed_source_requirements.requirement_hash
            or self.branch_endpoint_motion_requirement_hash
            != self.missing_requirement.typed_source_requirements.branch_endpoint_motion.requirement_hash
            or type(self.missing_requirement)
            is not V1PhaseAEndogenousDecisionBoundaryReachRequirement
            or self.missing_requirement
            != MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE
            or self.categorical_invariant_face_hashes
            != self.alias_map_receipt.categorical_invariant_face_hashes
            or self.decision_changing_face_hashes
            != self.alias_map_receipt.decision_changing_face_hashes
            or self.continuous_resource_cell_exit_hashes
            != self.alias_map_receipt.continuous_resource_cell_exit_hashes
            or self.robust_alias_safe_margin_lower_bound != 0.0
            or self.registered_alias_map_complete is not True
            or self.complete_categorical_action_quotient is not True
            or self.categorical_faces_require_no_numeric_margin is not True
            or self.owner_radius_lineage_custody_bound is not True
            or self.legacy_untyped_owner_radius_axis_provenance_only is not True
            or self.corrected_a_f_phase_a_causal_sensitivity_exactly_zero is not True
            or self.fd_and_solver_embedded_uncertainty_double_charge_forbidden
            is not True
            or self.universal_owner_radius_authority is not False
            or self.typed_source_values_endogenously_materialized is not False
            or self.norm_compatible_endpoint_operator_present is not False
            or self.sampled_vertices_cover_continuum is not False
            or self.registered_resource_cells_form_closed_continuum_cover
            is not False
            or self.continuum_margin_authority is not False
            or self.d_stability_authority is not False
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or self.authority_scope != PHASE_A_ALIAS_SAFE_MARGIN_MISSING_AUTHORITY
            or self.schema != PHASE_A_ALIAS_SAFE_MARGIN_MISSING_SCHEMA
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.AUTHORITY_VIOLATION,
                "missing-reach receipt overstated continuum authority",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "alias_map_receipt_hash": self.alias_map_receipt_hash,
                    "owner_radius_source_receipt_hash": self.owner_radius_source_receipt_hash,
                    "phase_a_decision_margin_state_hash": self.phase_a_decision_margin_state_hash,
                    "typed_source_requirement_hash": self.typed_source_requirement_hash,
                    "branch_endpoint_motion_requirement_hash": (
                        self.branch_endpoint_motion_requirement_hash
                    ),
                    "owner_radius_lineage_custody_axis": (
                        self.owner_radius_lineage_custody_axis
                    ),
                    "resource_cell_slack_axis": self.resource_cell_slack_axis,
                    "branch_support_slack_axis": self.branch_support_slack_axis,
                    "categorical_invariant_face_hashes": (
                        self.categorical_invariant_face_hashes
                    ),
                    "decision_changing_face_hashes": (
                        self.decision_changing_face_hashes
                    ),
                    "continuous_resource_cell_exit_hashes": (
                        self.continuous_resource_cell_exit_hashes
                    ),
                    "missing_requirement_hash": (
                        self.missing_requirement.requirement_hash
                    ),
                    "owner_radius_lineage_custody_bound": True,
                    "legacy_untyped_owner_radius_axis_provenance_only": True,
                    "corrected_a_f_phase_a_causal_sensitivity_exactly_zero": True,
                    "fd_and_solver_embedded_uncertainty_double_charge_forbidden": True,
                    "universal_owner_radius_authority": False,
                    "typed_source_values_endogenously_materialized": False,
                    "robust_alias_safe_margin_lower_bound": 0.0,
                    "sampled_vertices_cover_continuum": False,
                    "registered_resource_cells_form_closed_continuum_cover": False,
                    "route_scale_d2h_bytes": 0,
                    "route_scale_host_tensor_materializations": 0,
                    "all_downstream_authority": False,
                    "authority_scope": self.authority_scope,
                },
                prefix=_PREFIX + "missing-reach-",
            ),
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("Phase-A missing-reach receipts are nonserializable")

    def assert_live_identity(self) -> str:
        observed = _missing_reach_receipt(self.alias_map_receipt)
        names = tuple(
            name
            for name in self.__dataclass_fields__
            if not name.startswith("_")
            and name not in {"receipt_hash", "alias_map_receipt"}
        )
        if (
            any(getattr(self, name) != getattr(observed, name) for name in names)
            or self.receipt_hash != observed.receipt_hash
        ):
            _refuse(
                V1PhaseAAliasSafeMarginRefusalCode.STALE_OR_EVICTED_SOURCE,
                "missing-reach receipt no longer replays its live source graph",
            )
        return self.receipt_hash


def _missing_reach_receipt(
    alias_map: V1PhaseACompleteRegisteredAliasMapReceipt,
) -> V1PhaseAAliasSafeMarginMissingReachReceipt:
    state = alias_map.decision_margin_lease.state
    owner_source = alias_map.owner_radius_source
    return V1PhaseAAliasSafeMarginMissingReachReceipt(
        alias_map_receipt_hash=alias_map.receipt_hash,
        owner_radius_source_receipt_hash=owner_source.receipt_hash,
        phase_a_decision_margin_state_hash=state.state_hash,
        typed_source_requirement_hash=(
            MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE
            .typed_source_requirements.requirement_hash
        ),
        branch_endpoint_motion_requirement_hash=(
            MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE
            .typed_source_requirements.branch_endpoint_motion.requirement_hash
        ),
        owner_radius_lineage_custody_axis=tuple(
            (
                value.owner_role.value,
                value.receipt_hash,
                float(value.outward_radius_upper_bound),
            )
            for value in owner_source.owner_receipts
        ),
        resource_cell_slack_axis=tuple(
            (
                value.resource_cell_hash,
                value.scalar_replay_enclosed_boundary_slack_lower_bound,
            )
            for value in state.resource_cell_boundaries
        ),
        branch_support_slack_axis=tuple(
            (
                value.boundary_kind,
                tuple(branch.value for branch in value.branches),
                value.nominal_slack_lower_bound,
                value.endpoint_linf_sensitivity_upper_bound,
            )
            for value in state.branch_support_boundaries
        ),
        categorical_invariant_face_hashes=(
            alias_map.categorical_invariant_face_hashes
        ),
        decision_changing_face_hashes=alias_map.decision_changing_face_hashes,
        continuous_resource_cell_exit_hashes=(
            alias_map.continuous_resource_cell_exit_hashes
        ),
        missing_requirement=MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE,
        alias_map_receipt=alias_map,
        _issuer_token=_MISSING_TOKEN,
    )


def issue_v1_phase_a_alias_safe_continuum_margin_cuda(
    *,
    decision_margin_lease: V1SignedWorkResourceFrontierCUDADecisionMarginLease,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt,
) -> NoReturn:
    """Close the registered alias map, then refuse the absent operator bound."""

    alias_map = bind_v1_phase_a_complete_registered_alias_map_cuda(
        decision_margin_lease=decision_margin_lease,
        source_image=source_image,
        owner_radius_source=owner_radius_source,
    )
    missing = _missing_reach_receipt(alias_map)
    _refuse(
        V1PhaseAAliasSafeMarginRefusalCode.MISSING_ENDOGENOUS_DECISION_BOUNDARY_REACH,
        "the categorical action quotient is complete and corrected A/F has zero "
        "Phase-A sensitivity, but no self-consistent B<=8 robust variable-shell "
        "re-solve or full KKT strong-regularity producer maps typed J/FD/solver "
        "uncertainty to the remaining branch endpoint/Q boundaries; fixed-z "
        "M_J z alone refuses and continuum margin is zero",
        alias_map_receipt=alias_map,
        missing_reach_receipt=missing,
    )


__all__ = (
    "MISSING_PHASE_A_ENDOGENOUS_REACH_CERTIFICATE",
    "PHASE_A_ALIAS_SAFE_MARGIN_MISSING_AUTHORITY",
    "PHASE_A_ALIAS_SAFE_MARGIN_MISSING_SCHEMA",
    "PHASE_A_COMPLETE_ALIAS_MAP_AUTHORITY",
    "PHASE_A_COMPLETE_ALIAS_MAP_SCHEMA",
    "V1PhaseAAliasSafeMarginMissingReachReceipt",
    "V1PhaseAAliasSafeMarginRefusal",
    "V1PhaseAAliasSafeMarginRefusalCode",
    "V1PhaseACompleteRegisteredAliasMapReceipt",
    "V1PhaseAEndogenousDecisionBoundaryReachRequirement",
    "V1PhaseARegisteredActionCellQuotient",
    "V1PhaseARegisteredFaceActionAlias",
    "V1PhaseARegisteredSupportFaceQuotient",
    "bind_v1_phase_a_complete_registered_alias_map_cuda",
    "open_v1_phase_a_complete_registered_alias_map_cuda",
    "issue_v1_phase_a_alias_safe_continuum_margin_cuda",
)
