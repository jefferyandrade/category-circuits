"""Complete sampled-V7 qualified-action projection onto origin-Q sources.

The common sampled-D source and each Phase-A origin-Q source retain the same
literal-J compact rows.  The older live projection receipt proves that
source-wide fact, but couples it to one preferred hard action.  This module
uses the two live branch projections only as *source-root* witnesses and then
derives the complete qualified action image directly from the live Phase-A
registry and CUDA qualification.

Every Q-safe coordinate/branch alias is retained.  Exact packed-byte physical
actions may be deduplicated as representations, but branch aliases and
Phase-A physical action coordinates are never collapsed.  No Q number or
identity is caller supplied or synthesized here: registered Q remains owned
by the qualification's exact endpoint decisions and the live origin sources.
The receipt grants only complete common-D/origin-Q projection-family custody;
all D-stability, C/K/O, commit, terminal, performance, and Pareto authority is
closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_v7_common_d_origin_q_live_projection import (
    V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    V1SignedWorkCUDADecisionWitnessQualification,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointStatus,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkPackedActionDedupIdentity,
    v1_signed_work_packed_action_dedup_identity,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
)


SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_all_qualified_action_origin_Q_projection_"
    "family_v1"
)
SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_AUTHORITY: Final = (
    "complete_live_common_D_to_origin_Q_projection_for_every_qualified_"
    "branch_phase_A_action_coordinate_with_all_coordinate_and_packed_action_"
    "aliases_retained_only_no_D_stability_C_K_O_commit_terminal_performance_"
    "or_pareto_authority_v1"
)
SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_RULE: Final = (
    "replay_one_source_wide_common_D_literal_J_projection_per_Q_branch_then_"
    "join_every_Q_safe_qualification_alias_to_its_exact_live_phase_A_subset_"
    "coordinate_without_preferred_only_or_cross_branch_collapse_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-all-action-origin-Q-"
_ISSUER: Final = object()
_CANONICAL_RELAXED_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)
_CANONICAL_PHASE_A_BRANCHES: Final = (
    V1SignedWorkVariableShellQBranch.SPECTRAL,
    V1SignedWorkVariableShellQBranch.DIAGONAL_L1,
)


class V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    MISSING_OR_DUPLICATE_BRANCH = "missing_or_duplicate_branch"
    STALE_OR_EVICTED_SOURCE = "stale_or_evicted_source"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    MISSING_EXACT_ORIGIN_Q_PROJECTION = "missing_exact_origin_Q_projection"
    INCOMPLETE_QUALIFIED_ACTION_IMAGE = "incomplete_qualified_action_image"
    DUPLICATE_QUALIFIED_ALIAS = "duplicate_qualified_alias"
    PREFERRED_ONLY_COLLAPSE = "preferred_only_collapse"
    BRANCH_ALIAS_COLLAPSE = "branch_alias_collapse"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7AllQualifiedActionOriginQProjectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode:
            raise TypeError("all-action origin-Q refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7AllQualifiedActionOriginQProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _phase_a_branch(branch: V1RelaxedKKTBranch) -> V1SignedWorkVariableShellQBranch:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return V1SignedWorkVariableShellQBranch.SPECTRAL
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return V1SignedWorkVariableShellQBranch.DIAGONAL_L1
    _refuse(
        V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.MISSING_OR_DUPLICATE_BRANCH,
        "qualified Q alias has a noncanonical branch",
    )


def _physical_identity_hash(identity: V1SignedWorkPackedActionDedupIdentity) -> str:
    return content_hash(
        tuple(identity),
        prefix="partition-free-v1-signed-work-D-W-cuda-witness-physical-action-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7QualifiedActionOriginQAlias:
    """One exact qualification alias attached to a Phase-A action coordinate."""

    branch: V1RelaxedKKTBranch
    registered_cell_hash: str
    grid_key: tuple[str, str, str, str]
    resource_cell_hash: str
    resource_coordinate_hash: str
    qualification_crosswalk_hash: str
    qualification_endpoint_decision_hash: str
    phase_a_action_coordinate_hash: str
    phase_a_subset_identity_hash: str
    packed_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    physical_action_identity_hash: str
    alias_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("qualified origin-Q alias branch must be exact")
        for name in (
            "registered_cell_hash",
            "resource_cell_hash",
            "resource_coordinate_hash",
            "qualification_crosswalk_hash",
            "qualification_endpoint_decision_hash",
            "phase_a_action_coordinate_hash",
            "phase_a_subset_identity_hash",
            "physical_action_identity_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.grid_key) is not tuple
            or len(self.grid_key) != 4
            or any(type(value) is not str or not value for value in self.grid_key)
            or type(self.packed_dedup_identity) is not tuple
            or len(self.packed_dedup_identity) != 5
            or _physical_identity_hash(self.packed_dedup_identity)
            != self.physical_action_identity_hash
        ):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "qualified alias lost its uncertainty or packed-action identity",
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
                prefix=_PREFIX + "qualified-alias-",
            ),
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7QualifiedActionOriginQProjection:
    """One source-bound projection for one ``(Q branch, action coordinate)``."""

    branch: V1RelaxedKKTBranch
    phase_a_branch: V1SignedWorkVariableShellQBranch
    phase_a_action_coordinate_hash: str
    phase_a_subset_identity_hash: str
    source_root_projection_receipt_hash: str
    common_source_receipt_hash: str
    origin_q_source_receipt_hash: str
    origin_q_source_replay_hash: str
    origin_q_source_kkt_receipt_hash: str
    phase_a_branch_registration_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    same_capture_vjp_source_hash: str
    literal_j_stream_source_hash: str
    literal_j_direction_axis_hash: str
    aliases: tuple[V1SampledV7QualifiedActionOriginQAlias, ...]
    physical_action_identity_hashes: tuple[str, ...]
    projection_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.branch) is not V1RelaxedKKTBranch
            or type(self.phase_a_branch) is not V1SignedWorkVariableShellQBranch
            or self.phase_a_branch is not _phase_a_branch(self.branch)
        ):
            raise TypeError("origin-Q projection branch axis must be exact")
        for name in (
            "phase_a_action_coordinate_hash",
            "phase_a_subset_identity_hash",
            "source_root_projection_receipt_hash",
            "common_source_receipt_hash",
            "origin_q_source_receipt_hash",
            "origin_q_source_replay_hash",
            "origin_q_source_kkt_receipt_hash",
            "phase_a_branch_registration_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "same_capture_vjp_source_hash",
            "literal_j_stream_source_hash",
            "literal_j_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        aliases = tuple(self.aliases)
        physical = tuple(self.physical_action_identity_hashes)
        if (
            not aliases
            or any(type(value) is not V1SampledV7QualifiedActionOriginQAlias for value in aliases)
            or len({value.alias_hash for value in aliases}) != len(aliases)
            or any(
                value.branch is not self.branch
                or value.phase_a_action_coordinate_hash
                != self.phase_a_action_coordinate_hash
                or value.phase_a_subset_identity_hash
                != self.phase_a_subset_identity_hash
                for value in aliases
            )
            or not physical
            or len(set(physical)) != len(physical)
            or tuple(sorted({value.physical_action_identity_hash for value in aliases}))
            != physical
        ):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INCOMPLETE_QUALIFIED_ACTION_IMAGE,
                "one action projection lost, duplicated, or crossed a qualified alias",
            )
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "physical_action_identity_hashes", physical)
        object.__setattr__(
            self,
            "projection_hash",
            content_hash(
                {
                    name: (
                        tuple(value.alias_hash for value in aliases)
                        if name == "aliases"
                        else getattr(self, name)
                    )
                    for name in self.__dataclass_fields__
                    if name != "projection_hash"
                },
                prefix=_PREFIX + "action-projection-",
            ),
        )


def _exact_projection_pair(
    values: Sequence[V1SampledV7CommonDOriginQLiveProjectionBindingReceipt],
) -> tuple[
    V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
    V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
]:
    if type(values) is not tuple:
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INVALID_TYPE,
            "source-root projections must be one immutable tuple",
        )
    axis = tuple(values)
    if (
        len(axis) != 2
        or any(type(value) is not V1SampledV7CommonDOriginQLiveProjectionBindingReceipt for value in axis)
        or tuple(value.phase_a_branch for value in axis) != _CANONICAL_PHASE_A_BRANCHES
    ):
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.MISSING_OR_DUPLICATE_BRANCH,
            "source-root projections must contain both canonical Q branches exactly once",
        )
    return axis  # type: ignore[return-value]


def _derive_live_projection_image(
    *,
    live_projections: tuple[
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
    ],
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> tuple[
    tuple[V1SampledV7QualifiedActionOriginQProjection, ...],
    tuple[V1SampledV7QualifiedActionOriginQAlias, ...],
]:
    if (
        type(phase_a_registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(cuda_qualification) is not V1SignedWorkCUDADecisionWitnessQualification
    ):
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INVALID_TYPE,
            "projection-family derivation requires the exact live registry and qualification",
        )
    try:
        projection_hashes = tuple(value.assert_integrity() for value in live_projections)
        phase_a_registry.assert_live_sources()
        cuda_qualification.assert_live_identity()
    except Exception as error:
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.STALE_OR_EVICTED_SOURCE,
            f"one live projection-family parent failed replay: {type(error).__name__}: {error}",
        )

    first, second = live_projections
    registry_receipt = phase_a_registry.receipt
    common_identity = (
        first._d1_fixed_source,
        first._common_source,
        first.v7_d1_fixed_source_receipt_hash,
        first.v7_d1_literal_j_projection_registration_hash,
        first.v7_d1_literal_j_projection_receipt_hash,
        first.common_source_receipt_hash,
        first.common_source_replay_hash,
        first.model_identity_hash,
        first.chart_identity_hash,
        first.current_mask_hash,
        first.proposal_generation,
        first.route_axis_hash,
        first.route_id_axis_hash,
        first.route_count,
        first.shell_freeze_receipt_hash,
        first.registered_d_prefix_hash,
        first.registered_d_w2_prefix_hash,
        first.same_capture_vjp_source_hash,
        first.literal_j_stream_source_hash,
    )
    second_common_identity = (
        second._d1_fixed_source,
        second._common_source,
        second.v7_d1_fixed_source_receipt_hash,
        second.v7_d1_literal_j_projection_registration_hash,
        second.v7_d1_literal_j_projection_receipt_hash,
        second.common_source_receipt_hash,
        second.common_source_replay_hash,
        second.model_identity_hash,
        second.chart_identity_hash,
        second.current_mask_hash,
        second.proposal_generation,
        second.route_axis_hash,
        second.route_id_axis_hash,
        second.route_count,
        second.shell_freeze_receipt_hash,
        second.registered_d_prefix_hash,
        second.registered_d_w2_prefix_hash,
        second.same_capture_vjp_source_hash,
        second.literal_j_stream_source_hash,
    )
    if (
        common_identity != second_common_identity
        or registry_receipt.branch_axis != _CANONICAL_PHASE_A_BRANCHES
        or registry_receipt.complete_branch_axis is not True
        or registry_receipt.complete_action_cell_coverage is not True
        or any(value._phase_a_registry is not phase_a_registry for value in live_projections)
        or tuple(value.receipt_hash for value in live_projections) != projection_hashes
        or cuda_qualification.parent_mask_hash != registry_receipt.current_mask_hash
        or cuda_qualification.route_axis_hash != registry_receipt.route_axis_hash
        or cuda_qualification.route_count != first.route_count
        or cuda_qualification.parent_mask_hash != first.current_mask_hash
        or cuda_qualification.route_axis_hash != first.route_axis_hash
    ):
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "common D, literal-J/FD, Phase-A, and qualification axes are not one live source graph",
        )
    for projection in live_projections:
        branch_source = phase_a_registry.branch_source(projection.phase_a_branch)
        if (
            projection._origin_q_source is not branch_source.source_handle
            or projection.origin_q_source_receipt_hash
            != branch_source.registration.source_receipt_hash
            or projection.origin_q_source_replay_hash
            != branch_source.registration.source_replay_hash
            or projection.phase_a_branch_registration_hash
            != branch_source.registration.registration_hash
        ):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.MISSING_EXACT_ORIGIN_Q_PROJECTION,
                "a source-root projection is not the exact live origin-Q branch source",
            )

    projection_by_branch = {
        V1RelaxedKKTBranch(value.phase_a_branch.value): value
        for value in live_projections
    }
    aliases: list[V1SampledV7QualifiedActionOriginQAlias] = []
    observed_qualification_aliases: set[tuple[str, str]] = set()
    for registered_cell in cuda_qualification.source_cells:
        resource_cells = {
            value.resource_coordinate_hash: value.resource_cell_hash
            for value in registered_cell.resource_cell_bindings
        }
        for candidate in registered_cell.packed_action_family.candidate_crosswalks:
            if candidate.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE:
                continue
            action = candidate.packed_action
            if action is None:
                _refuse(
                    V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INCOMPLETE_QUALIFIED_ACTION_IMAGE,
                    "a Q-safe qualified alias lost its exact packed action",
                )
            branch = candidate.q_branch
            phase_branch = _phase_a_branch(branch)
            branch_source = phase_a_registry.branch_source(phase_branch)
            matches = tuple(
                value
                for value in branch_source.subset_identities
                if value.removed_route_index_hash == action.removed_route_index_hash
                and value.removed_count == action.removed_count
            )
            resource_cell_hash = resource_cells.get(candidate.resource_coordinate_hash)
            if len(matches) != 1 or resource_cell_hash is None:
                _refuse(
                    V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.MISSING_EXACT_ORIGIN_Q_PROJECTION,
                    "a qualified branch/action has no exact Phase-A subset or resource cell",
                )
            subset = matches[0]
            identity = v1_signed_work_packed_action_dedup_identity(action)
            physical_hash = _physical_identity_hash(identity)
            alias_key = (registered_cell.registered_cell_hash, candidate.crosswalk_hash)
            if alias_key in observed_qualification_aliases:
                _refuse(
                    V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.DUPLICATE_QUALIFIED_ALIAS,
                    "one qualified uncertainty-cell alias was repeated",
                )
            observed_qualification_aliases.add(alias_key)
            aliases.append(
                V1SampledV7QualifiedActionOriginQAlias(
                    branch=branch,
                    registered_cell_hash=registered_cell.registered_cell_hash,
                    grid_key=registered_cell.grid_key,
                    resource_cell_hash=resource_cell_hash,
                    resource_coordinate_hash=candidate.resource_coordinate_hash,
                    qualification_crosswalk_hash=candidate.crosswalk_hash,
                    qualification_endpoint_decision_hash=(
                        candidate.endpoint_decision.receipt_hash
                    ),
                    phase_a_action_coordinate_hash=(
                        subset.physical_action_coordinate_hash
                    ),
                    phase_a_subset_identity_hash=subset.subset_hash,
                    packed_dedup_identity=identity,
                    physical_action_identity_hash=physical_hash,
                )
            )
    if not aliases:
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INCOMPLETE_QUALIFIED_ACTION_IMAGE,
            "qualification contains no Q-safe action aliases",
        )

    observed_physical_image = {
        (value.resource_cell_hash, value.physical_action_identity_hash)
        for value in aliases
    }
    expected_physical_image = set(cuda_qualification.physical_action_family)
    if observed_physical_image != expected_physical_image:
        preferred_image = set(cuda_qualification.preferred_physical_decision_family)
        code = (
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.PREFERRED_ONLY_COLLAPSE
            if observed_physical_image == preferred_image
            and preferred_image != expected_physical_image
            else V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INCOMPLETE_QUALIFIED_ACTION_IMAGE
        )
        _refuse(
            code,
            "derived projection aliases do not equal the complete qualified physical image",
        )

    grouped: dict[
        tuple[V1RelaxedKKTBranch, str],
        list[V1SampledV7QualifiedActionOriginQAlias],
    ] = {}
    for alias in aliases:
        grouped.setdefault(
            (alias.branch, alias.phase_a_action_coordinate_hash), []
        ).append(alias)
    projections: list[V1SampledV7QualifiedActionOriginQProjection] = []
    for branch in _CANONICAL_RELAXED_BRANCHES:
        root = projection_by_branch[branch]
        branch_items = sorted(
            (
                (coordinate, values)
                for (item_branch, coordinate), values in grouped.items()
                if item_branch is branch
            ),
            key=lambda value: value[0],
        )
        for coordinate, values in branch_items:
            ordered = tuple(sorted(values, key=lambda value: value.alias_hash))
            subset_hashes = {value.phase_a_subset_identity_hash for value in ordered}
            if len(subset_hashes) != 1:
                _refuse(
                    V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.SOURCE_IDENTITY_MISMATCH,
                    "one physical action coordinate names multiple Phase-A subsets",
                )
            projections.append(
                V1SampledV7QualifiedActionOriginQProjection(
                    branch=branch,
                    phase_a_branch=_phase_a_branch(branch),
                    phase_a_action_coordinate_hash=coordinate,
                    phase_a_subset_identity_hash=next(iter(subset_hashes)),
                    source_root_projection_receipt_hash=root.receipt_hash,
                    common_source_receipt_hash=root.common_source_receipt_hash,
                    origin_q_source_receipt_hash=root.origin_q_source_receipt_hash,
                    origin_q_source_replay_hash=root.origin_q_source_replay_hash,
                    origin_q_source_kkt_receipt_hash=(
                        root.origin_q_source_kkt_receipt_hash
                    ),
                    phase_a_branch_registration_hash=(
                        root.phase_a_branch_registration_hash
                    ),
                    registered_d_prefix_hash=root.registered_d_prefix_hash,
                    registered_d_w2_prefix_hash=root.registered_d_w2_prefix_hash,
                    same_capture_vjp_source_hash=root.same_capture_vjp_source_hash,
                    literal_j_stream_source_hash=root.literal_j_stream_source_hash,
                    literal_j_direction_axis_hash=(
                        root._d1_fixed_source.literal_j_projection.receipt.direction_axis_hash
                    ),
                    aliases=ordered,
                    physical_action_identity_hashes=tuple(
                        sorted(
                            {value.physical_action_identity_hash for value in ordered}
                        )
                    ),
                )
            )
    if {value.alias_hash for value in aliases} != {
        alias.alias_hash for value in projections for alias in value.aliases
    }:
        _refuse(
            V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.BRANCH_ALIAS_COLLAPSE,
            "branch/action grouping lost a qualified coordinate alias",
        )
    return tuple(projections), tuple(sorted(aliases, key=lambda value: value.alias_hash))


@dataclass(frozen=True, slots=True)
class V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt:
    """Issuer-owned complete projection family over the qualified action image."""

    source_root_projection_receipt_hashes: tuple[str, str]
    phase_a_registry_receipt_hash: str
    cuda_qualification_hash: str
    common_source_receipt_hash: str
    v7_d1_fixed_source_receipt_hash: str
    literal_j_projection_registration_hash: str
    literal_j_projection_receipt_hash: str
    literal_j_private_primal_descriptor_axis_hash: str
    literal_j_direction_axis_hash: str
    literal_j_direction_ids: tuple[str, ...]
    literal_j_numeric_source_receipt_hash: str
    literal_j_same_capture_execution_proof_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    literal_j_stream_source_hash: str
    projections: tuple[V1SampledV7QualifiedActionOriginQProjection, ...]
    projection_domain: tuple[tuple[str, str], ...]
    qualification_alias_hashes: tuple[str, ...]
    qualified_physical_action_family: tuple[tuple[str, str], ...]
    receipt_hash: str = field(init=False)
    complete_qualified_action_projection_authority: bool = True
    preferred_only_collapse_used: bool = False
    cross_branch_action_collapse_used: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    rule: str = SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_RULE
    authority_scope: str = SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_AUTHORITY
    schema: str = SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_SCHEMA
    _live_projections: tuple[
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
    ] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _phase_a_registry: V1SignedWorkVariableShellActionCellRegistry = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER:
            raise TypeError("all-qualified-action origin-Q families are issuer-only")
        for name in (
            "phase_a_registry_receipt_hash",
            "cuda_qualification_hash",
            "common_source_receipt_hash",
            "v7_d1_fixed_source_receipt_hash",
            "literal_j_projection_registration_hash",
            "literal_j_projection_receipt_hash",
            "literal_j_private_primal_descriptor_axis_hash",
            "literal_j_direction_axis_hash",
            "literal_j_numeric_source_receipt_hash",
            "literal_j_same_capture_execution_proof_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "literal_j_stream_source_hash",
            "rule",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        projections = tuple(self.projections)
        if (
            len(self.source_root_projection_receipt_hashes) != 2
            or any(type(value) is not str or not value for value in self.source_root_projection_receipt_hashes)
            or not projections
            or any(type(value) is not V1SampledV7QualifiedActionOriginQProjection for value in projections)
            or len({value.projection_hash for value in projections}) != len(projections)
            or self.projection_domain
            != tuple((value.branch.value, value.phase_a_action_coordinate_hash) for value in projections)
            or len(set(self.projection_domain)) != len(self.projection_domain)
            or self.qualification_alias_hashes
            != tuple(
                sorted(
                    alias.alias_hash
                    for value in projections
                    for alias in value.aliases
                )
            )
            or tuple(self.qualified_physical_action_family)
            != self._cuda_qualification.physical_action_family
            or type(self.literal_j_direction_ids) is not tuple
            or not self.literal_j_direction_ids
            or any(
                type(value) is not str or not value
                for value in self.literal_j_direction_ids
            )
        ):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.INCOMPLETE_QUALIFIED_ACTION_IMAGE,
                "projection family is not an exact set image of its qualified aliases",
            )
        flags = (
            self.complete_qualified_action_projection_authority,
            self.preferred_only_collapse_used,
            self.cross_branch_action_collapse_used,
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
            or flags != (True, False, False, False, False, False, False, False, False, False, False)
            or self.rule != SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_RULE
            or self.authority_scope != SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_AUTHORITY
            or self.schema != SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_SCHEMA
        ):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.AUTHORITY_VIOLATION,
                "projection family crossed its source-binding-only authority",
            )
        object.__setattr__(self, "projections", projections)
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.STALE_OR_EVICTED_SOURCE,
                "projection-family receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                tuple(value.projection_hash for value in self.projections)
                if name == "projections"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        projections, aliases = _derive_live_projection_image(
            live_projections=self._live_projections,
            phase_a_registry=self._phase_a_registry,
            cuda_qualification=self._cuda_qualification,
        )
        first = self._live_projections[0]
        literal_j_receipt = first._d1_fixed_source.literal_j_projection.receipt
        d1_receipt = first._d1_fixed_source.receipt
        observed = {
            "source_root_projection_receipt_hashes": tuple(
                value.receipt_hash for value in self._live_projections
            ),
            "phase_a_registry_receipt_hash": self._phase_a_registry.receipt.receipt_hash,
            "cuda_qualification_hash": self._cuda_qualification.qualification_hash,
            "common_source_receipt_hash": first.common_source_receipt_hash,
            "v7_d1_fixed_source_receipt_hash": first.v7_d1_fixed_source_receipt_hash,
            "literal_j_projection_registration_hash": (
                first.v7_d1_literal_j_projection_registration_hash
            ),
            "literal_j_projection_receipt_hash": (
                first.v7_d1_literal_j_projection_receipt_hash
            ),
            "literal_j_private_primal_descriptor_axis_hash": (
                d1_receipt.private_primal_descriptor_axis_hash
            ),
            "literal_j_direction_axis_hash": literal_j_receipt.direction_axis_hash,
            "literal_j_direction_ids": literal_j_receipt.direction_ids,
            "literal_j_numeric_source_receipt_hash": (
                literal_j_receipt.numeric_source_receipt_hash
            ),
            "literal_j_same_capture_execution_proof_hash": (
                literal_j_receipt.same_capture_and_vjp_execution_proof_hash
            ),
            "registered_d_prefix_hash": first.registered_d_prefix_hash,
            "registered_d_w2_prefix_hash": first.registered_d_w2_prefix_hash,
            "literal_j_stream_source_hash": first.literal_j_stream_source_hash,
            "projections": projections,
            "projection_domain": tuple(
                (value.branch.value, value.phase_a_action_coordinate_hash)
                for value in projections
            ),
            "qualification_alias_hashes": tuple(value.alias_hash for value in aliases),
            "qualified_physical_action_family": tuple(
                self._cuda_qualification.physical_action_family
            ),
        }
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.STALE_OR_EVICTED_SOURCE,
                "projection family no longer equals its live producer image",
            )
        return content_hash(self._payload(), prefix=_PREFIX + "family-receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.receipt_hash or observed != self._integrity_seal:
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.STALE_OR_EVICTED_SOURCE,
                "projection-family receipt no longer replays",
            )
        return observed

    def projection_for(
        self,
        *,
        branch: V1RelaxedKKTBranch,
        phase_a_action_coordinate_hash: str,
    ) -> V1SampledV7QualifiedActionOriginQProjection:
        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("projection lookup branch must be exact")
        coordinate = _identity(
            phase_a_action_coordinate_hash,
            name="Phase-A action coordinate hash",
        )
        matches = tuple(
            value
            for value in self.projections
            if value.branch is branch
            and value.phase_a_action_coordinate_hash == coordinate
        )
        if len(matches) != 1:
            _refuse(
                V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode.MISSING_EXACT_ORIGIN_Q_PROJECTION,
                "requested branch/action coordinate lacks one exact projection",
            )
        return matches[0]

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("all-qualified-action origin-Q families are nonserializable")


def issue_v1_sampled_v7_all_qualified_action_origin_q_projection_family(
    *,
    live_projections: Sequence[V1SampledV7CommonDOriginQLiveProjectionBindingReceipt],
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt:
    """Derive the exact complete qualified branch/action projection family."""

    roots = _exact_projection_pair(live_projections)
    projections, aliases = _derive_live_projection_image(
        live_projections=roots,
        phase_a_registry=phase_a_registry,
        cuda_qualification=cuda_qualification,
    )
    first = roots[0]
    literal_j_receipt = first._d1_fixed_source.literal_j_projection.receipt
    d1_receipt = first._d1_fixed_source.receipt
    return V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt(
        source_root_projection_receipt_hashes=tuple(
            value.receipt_hash for value in roots
        ),
        phase_a_registry_receipt_hash=phase_a_registry.receipt.receipt_hash,
        cuda_qualification_hash=cuda_qualification.qualification_hash,
        common_source_receipt_hash=first.common_source_receipt_hash,
        v7_d1_fixed_source_receipt_hash=first.v7_d1_fixed_source_receipt_hash,
        literal_j_projection_registration_hash=(
            first.v7_d1_literal_j_projection_registration_hash
        ),
        literal_j_projection_receipt_hash=(
            first.v7_d1_literal_j_projection_receipt_hash
        ),
        literal_j_private_primal_descriptor_axis_hash=(
            d1_receipt.private_primal_descriptor_axis_hash
        ),
        literal_j_direction_axis_hash=literal_j_receipt.direction_axis_hash,
        literal_j_direction_ids=literal_j_receipt.direction_ids,
        literal_j_numeric_source_receipt_hash=(
            literal_j_receipt.numeric_source_receipt_hash
        ),
        literal_j_same_capture_execution_proof_hash=(
            literal_j_receipt.same_capture_and_vjp_execution_proof_hash
        ),
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        literal_j_stream_source_hash=first.literal_j_stream_source_hash,
        projections=projections,
        projection_domain=tuple(
            (value.branch.value, value.phase_a_action_coordinate_hash)
            for value in projections
        ),
        qualification_alias_hashes=tuple(value.alias_hash for value in aliases),
        qualified_physical_action_family=tuple(cuda_qualification.physical_action_family),
        _live_projections=roots,
        _phase_a_registry=phase_a_registry,
        _cuda_qualification=cuda_qualification,
        _issuer_token=_ISSUER,
    )


__all__ = (
    "SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_AUTHORITY",
    "SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_RULE",
    "SAMPLED_V7_ALL_QUALIFIED_ACTION_ORIGIN_Q_PROJECTION_SCHEMA",
    "V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt",
    "V1SampledV7AllQualifiedActionOriginQProjectionRefusal",
    "V1SampledV7AllQualifiedActionOriginQProjectionRefusalCode",
    "V1SampledV7QualifiedActionOriginQAlias",
    "V1SampledV7QualifiedActionOriginQProjection",
    "issue_v1_sampled_v7_all_qualified_action_origin_q_projection_family",
)
