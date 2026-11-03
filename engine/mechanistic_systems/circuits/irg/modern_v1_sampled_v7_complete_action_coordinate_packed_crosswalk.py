"""Complete sampled-V7 action-coordinate/packed-action crosswalk seam.

The live Phase-A registry and CUDA decision qualification contain enough
producer-owned information to identify a qualified packed action structurally:
the branch/resource alias, exact removed-index identity, packed child digest,
and corrected-F alternate-domain alias all replay from retained parents.  The
all-qualified-action projection family lifts the two source-root
common-D/origin-Q bindings over every qualified branch/action coordinate.

This module joins that family to the complete corrected-F and packed-action
aliases and proves exact set equality.  Equal preferred endpoints, equal
digests, or cross-branch representation deduplication never substitute for
full alias coverage.  The positive receipt grants source-image crosswalk
authority only; all D/C/K/O/commit/terminal/performance/Pareto authority
remains closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_v7_branch_decision_image_production import (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
)
from .modern_v1_sampled_v7_all_qualified_action_origin_q_projection import (
    V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt,
    issue_v1_sampled_v7_all_qualified_action_origin_q_projection_family,
)
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


SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_complete_action_coordinate_packed_crosswalk_"
    "with_all_action_origin_Q_projection_v2"
)
SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_AUTHORITY: Final = (
    "complete_phase_A_corrected_F_alias_to_exact_qualified_packed_action_and_"
    "all_action_common_D_origin_Q_projection_source_image_crosswalk_only_no_D_"
    "stability_C_K_O_commit_terminal_performance_or_pareto_authority_v2"
)
SAMPLED_V7_ALL_ACTION_Q_MISSING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_complete_action_coordinate_packed_crosswalk_"
    "missing_all_action_origin_Q_projection_v1"
)
SAMPLED_V7_ALL_ACTION_Q_MISSING_AUTHORITY: Final = (
    "structural_phase_A_corrected_F_alias_to_exact_qualified_packed_action_"
    "crosswalk_only_missing_all_action_common_D_origin_Q_projection_no_source_"
    "image_D_stability_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_ALL_ACTION_Q_REQUIRED_PRODUCER_API: Final = (
    "V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt",
    "one_live_common_D_to_origin_Q_projection_per_unique_branch_action_coordinate",
    "complete_qualified_branch_action_coordinate_axis",
    "exact_registered_Q_bridge_and_packed_child_identity_per_action",
    "same_D_prefix_capture_literal_J_chart_route_shell_population_lineage_per_action",
    "no_preferred_endpoint_substitution_or_cross_branch_deduplication",
)

_PREFIX: Final = "partition-free-v1-sampled-v7-complete-action-crosswalk-"
_MISSING_TOKEN: Final = object()
_ISSUER_TOKEN: Final = object()
_CANONICAL_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)
_CANONICAL_PHASE_A_BRANCHES: Final = (
    V1SignedWorkVariableShellQBranch.SPECTRAL,
    V1SignedWorkVariableShellQBranch.DIAGONAL_L1,
)


class V1SampledV7CompleteActionCrosswalkRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    MISSING_OR_DUPLICATE_BRANCH = "missing_or_duplicate_branch"
    STALE_OR_EVICTED_SOURCE = "stale_or_evicted_source"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    INCOMPLETE_F_ALIAS_COVERAGE = "incomplete_F_alias_coverage"
    INCOMPLETE_PACKED_ACTION_COVERAGE = "incomplete_packed_action_coverage"
    MISSING_ALL_ACTION_ORIGIN_Q_PROJECTION_PRODUCER = (
        "missing_all_action_origin_Q_projection_producer"
    )


class V1SampledV7CompleteActionCrosswalkRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CompleteActionCrosswalkRefusalCode,
        message: str,
        *,
        missing_premise_receipt: object | None = None,
    ) -> None:
        if type(code) is not V1SampledV7CompleteActionCrosswalkRefusalCode:
            raise TypeError("complete-action crosswalk refusal code must be exact")
        self.code = code
        self.missing_premise_receipt = missing_premise_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CompleteActionCrosswalkRefusalCode,
    message: str,
    *,
    missing_premise_receipt: object | None = None,
) -> NoReturn:
    raise V1SampledV7CompleteActionCrosswalkRefusal(
        code,
        message,
        missing_premise_receipt=missing_premise_receipt,
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _phase_a_branch(branch: V1RelaxedKKTBranch) -> V1SignedWorkVariableShellQBranch:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return V1SignedWorkVariableShellQBranch.SPECTRAL
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return V1SignedWorkVariableShellQBranch.DIAGONAL_L1
    _refuse(
        V1SampledV7CompleteActionCrosswalkRefusalCode.MISSING_OR_DUPLICATE_BRANCH,
        "qualified alias has a noncanonical Q branch",
    )


def _physical_identity_hash(identity: V1SignedWorkPackedActionDedupIdentity) -> str:
    return content_hash(
        tuple(identity),
        prefix="partition-free-v1-signed-work-D-W-cuda-witness-physical-action-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7StructuralActionCrosswalkRow:
    """One derived branch/resource/F-alias/action/packed-child row."""

    branch: V1RelaxedKKTBranch
    qualification_registered_cell_hash: str
    qualification_grid_key: tuple[str, str, str, str]
    resource_cell_hash: str
    resource_coordinate_hash: str
    phase_a_resource_alias_hash: str
    phase_a_resource_alias_id: str
    phase_a_resource_cell_id: str
    phase_a_action_cell_hash: str
    phase_a_action_coordinate_hash: str
    phase_a_subset_identity_hash: str
    corrected_f_alternate_domain_hash: str
    corrected_f_resource_alias_hash: str
    qualification_crosswalk_hash: str
    qualification_endpoint_decision_hash: str
    packed_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    physical_action_identity_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    origin_q_projection_receipt_hash: str | None
    structural_row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("structural crosswalk row branch must be exact")
        for name in (
            "qualification_registered_cell_hash",
            "resource_cell_hash",
            "resource_coordinate_hash",
            "phase_a_resource_alias_hash",
            "phase_a_resource_alias_id",
            "phase_a_resource_cell_id",
            "phase_a_action_cell_hash",
            "phase_a_action_coordinate_hash",
            "phase_a_subset_identity_hash",
            "corrected_f_alternate_domain_hash",
            "corrected_f_resource_alias_hash",
            "qualification_crosswalk_hash",
            "qualification_endpoint_decision_hash",
            "physical_action_identity_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.qualification_grid_key) is not tuple
            or len(self.qualification_grid_key) != 4
            or any(type(value) is not str or not value for value in self.qualification_grid_key)
            or type(self.packed_dedup_identity) is not tuple
            or len(self.packed_dedup_identity) != 5
            or isinstance(self.removed_count, bool)
            or not isinstance(self.removed_count, int)
            or self.removed_count <= 0
            or self.packed_dedup_identity[3] != self.removed_route_index_hash
            or self.packed_dedup_identity[4] != self.removed_count
            or self.packed_dedup_identity[0] != self.resulting_mask_packed_sha256
            or self.packed_dedup_identity[1] != self.packed_action_hash
            or self.packed_dedup_identity[2] != self.resulting_mask_hash
            or _physical_identity_hash(self.packed_dedup_identity)
            != self.physical_action_identity_hash
            or (
                self.origin_q_projection_receipt_hash is not None
                and (
                    type(self.origin_q_projection_receipt_hash) is not str
                    or not self.origin_q_projection_receipt_hash
                )
            )
        ):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                "structural row lost its exact packed-action identity",
            )
        object.__setattr__(
            self,
            "structural_row_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "structural_row_hash"
                },
                prefix=_PREFIX + "structural-row-",
            ),
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7AllActionQProjectionMissingReceipt:
    """Maximal structural crosswalk plus its uncovered origin-Q axis."""

    branch_evidence_receipt_hashes: tuple[str, str]
    phase_a_registry_receipt_hash: str
    cuda_qualification_hash: str
    structural_rows: tuple[V1SampledV7StructuralActionCrosswalkRow, ...]
    uncovered_branch_action_axis: tuple[tuple[str, str], ...]
    supplied_projection_receipt_hashes: tuple[str, str]
    required_producer_api: tuple[str, ...] = SAMPLED_V7_ALL_ACTION_Q_REQUIRED_PRODUCER_API
    receipt_hash: str = field(init=False)
    structural_action_coordinate_packed_crosswalk_complete: bool = True
    all_action_origin_q_projection_complete: bool = False
    source_image_binding_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_ALL_ACTION_Q_MISSING_AUTHORITY
    schema: str = SAMPLED_V7_ALL_ACTION_Q_MISSING_SCHEMA
    _issuer_token: InitVar[object | None] = None

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _MISSING_TOKEN:
            raise TypeError("all-action-Q missing receipts are issuer-only")
        if (
            len(self.branch_evidence_receipt_hashes) != 2
            or len(self.supplied_projection_receipt_hashes) != 2
            or not self.structural_rows
            or any(
                type(value) is not V1SampledV7StructuralActionCrosswalkRow
                for value in self.structural_rows
            )
            or not self.uncovered_branch_action_axis
            or any(
                type(value) is not tuple
                or len(value) != 2
                or any(type(item) is not str or not item for item in value)
                for value in self.uncovered_branch_action_axis
            )
        ):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                "missing-Q receipt requires a complete nonempty structural axis",
            )
        for name in (
            "phase_a_registry_receipt_hash",
            "cuda_qualification_hash",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        closed = (
            self.all_action_origin_q_projection_complete,
            self.source_image_binding_authority,
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
            self.structural_action_coordinate_packed_crosswalk_complete is not True
            or any(type(value) is not bool or value is not False for value in closed)
            or self.required_producer_api != SAMPLED_V7_ALL_ACTION_Q_REQUIRED_PRODUCER_API
            or self.authority_scope != SAMPLED_V7_ALL_ACTION_Q_MISSING_AUTHORITY
            or self.schema != SAMPLED_V7_ALL_ACTION_Q_MISSING_SCHEMA
        ):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "missing-Q receipt crossed its refusal-only boundary",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(
                {
                    "schema": self.schema,
                    "branch_evidence_receipt_hashes": self.branch_evidence_receipt_hashes,
                    "phase_a_registry_receipt_hash": self.phase_a_registry_receipt_hash,
                    "cuda_qualification_hash": self.cuda_qualification_hash,
                    "structural_row_hashes": tuple(
                        value.structural_row_hash for value in self.structural_rows
                    ),
                    "uncovered_branch_action_axis": self.uncovered_branch_action_axis,
                    "supplied_projection_receipt_hashes": self.supplied_projection_receipt_hashes,
                    "required_producer_api": self.required_producer_api,
                    "authority_scope": self.authority_scope,
                },
                prefix=_PREFIX + "missing-all-action-Q-",
            ),
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("all-action-Q missing receipts are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt:
    """Complete corrected-F/action/packed-child/origin-Q source crosswalk."""

    branch_evidence_receipt_hashes: tuple[str, str]
    source_root_projection_receipt_hashes: tuple[str, str]
    all_action_projection_family_receipt_hash: str
    phase_a_registry_receipt_hash: str
    cuda_qualification_hash: str
    structural_rows: tuple[V1SampledV7StructuralActionCrosswalkRow, ...]
    branch_action_projection_domain: tuple[tuple[str, str], ...]
    cuda_physical_action_family: tuple[tuple[str, str], ...]
    receipt_hash: str = field(init=False)
    structural_action_coordinate_packed_crosswalk_complete: bool = True
    all_action_origin_q_projection_complete: bool = True
    source_image_binding_authority: bool = True
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_AUTHORITY
    schema: str = SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_SCHEMA
    _branch_evidence: tuple[
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
    ] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _projection_family: V1SampledV7AllQualifiedActionOriginQProjectionFamilyReceipt = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _phase_a_registry: V1SignedWorkVariableShellActionCellRegistry = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("complete-action crosswalk receipts are issuer-only")
        rows = tuple(self.structural_rows)
        for name in (
            "all_action_projection_family_receipt_hash",
            "phase_a_registry_receipt_hash",
            "cuda_qualification_hash",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        if (
            len(self.branch_evidence_receipt_hashes) != 2
            or len(self.source_root_projection_receipt_hashes) != 2
            or any(
                type(value) is not str or not value
                for value in self.branch_evidence_receipt_hashes
                + self.source_root_projection_receipt_hashes
            )
            or not rows
            or any(type(value) is not V1SampledV7StructuralActionCrosswalkRow for value in rows)
            or len({value.structural_row_hash for value in rows}) != len(rows)
            or any(value.origin_q_projection_receipt_hash is None for value in rows)
            or self.branch_action_projection_domain
            != self._projection_family.projection_domain
            or tuple(self.cuda_physical_action_family)
            != self._cuda_qualification.physical_action_family
        ):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                "positive crosswalk lost a structural row, action projection, or physical image",
            )
        flags = (
            self.structural_action_coordinate_packed_crosswalk_complete,
            self.all_action_origin_q_projection_complete,
            self.source_image_binding_authority,
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
            or flags != (True, True, True, False, False, False, False, False, False, False, False)
            or self.authority_scope != SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_AUTHORITY
            or self.schema != SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_SCHEMA
        ):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "complete crosswalk crossed its source-image-only authority",
            )
        object.__setattr__(self, "structural_rows", rows)
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE,
                "complete crosswalk receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _payload(self) -> dict[str, object]:
        return {
            name: (
                tuple(value.structural_row_hash for value in self.structural_rows)
                if name == "structural_rows"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        try:
            branch_hashes = tuple(
                value.assert_integrity() for value in self._branch_evidence
            )
            family_hash = self._projection_family.assert_integrity()
            self._phase_a_registry.assert_live_sources()
            self._cuda_qualification.assert_live_identity()
            rows = _derive_structural_rows(
                branches=self._branch_evidence,
                projection_family=self._projection_family,
                registry=self._phase_a_registry,
                qualification=self._cuda_qualification,
            )
        except V1SampledV7CompleteActionCrosswalkRefusal:
            raise
        except Exception as error:
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE,
                f"complete crosswalk parent replay failed: {type(error).__name__}: {error}",
            )
        observed = {
            "branch_evidence_receipt_hashes": branch_hashes,
            "source_root_projection_receipt_hashes": (
                self._projection_family.source_root_projection_receipt_hashes
            ),
            "all_action_projection_family_receipt_hash": family_hash,
            "phase_a_registry_receipt_hash": self._phase_a_registry.receipt.receipt_hash,
            "cuda_qualification_hash": self._cuda_qualification.qualification_hash,
            "structural_rows": rows,
            "branch_action_projection_domain": self._projection_family.projection_domain,
            "cuda_physical_action_family": tuple(
                self._cuda_qualification.physical_action_family
            ),
        }
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE,
                "complete crosswalk no longer equals its live producer graph",
            )
        return content_hash(self._payload(), prefix=_PREFIX + "complete-receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.receipt_hash or observed != self._integrity_seal:
            _refuse(
                V1SampledV7CompleteActionCrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE,
                "complete crosswalk receipt no longer replays",
            )
        return observed

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("complete-action crosswalk receipts are nonserializable")


def _exact_pair(values: Sequence[object], *, exact_type: type, branches, getter, name: str):
    if type(values) is not tuple:
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    axis = tuple(values)
    if (
        len(axis) != 2
        or any(type(value) is not exact_type for value in axis)
        or tuple(getter(value) for value in axis) != branches
    ):
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.MISSING_OR_DUPLICATE_BRANCH,
            f"{name} must contain the canonical branch axis exactly once",
        )
    return axis


def _derive_structural_rows(*, branches, projection_family, registry, qualification):
    # Live corrected-F aliases are retained by the branch evidence block cover.
    f_aliases = {}
    for evidence in branches:
        for cover in evidence._blocks[0]._live_covers:
            if cover.branch_id != evidence.branch.value:
                continue
            key = (evidence.branch, cover.resource_coordinate_hash)
            value = (
                evidence.complete_alternate_domain_hash,
                cover.resource_alias_hash,
                cover.resource_alias_id,
                cover.phase_a_resource_cell_id,
            )
            if key in f_aliases and f_aliases[key] != value:
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_F_ALIAS_COVERAGE,
                    "one corrected-F resource coordinate has conflicting aliases",
                )
            f_aliases[key] = value

    phase_aliases = {}
    for source in registry.branch_sources:
        branch = V1RelaxedKKTBranch(source.branch.value)
        for alias in source.resource_aliases:
            phase_aliases[(branch, alias.resource_coordinate_hash)] = (source, alias)

    projection_by_action_coordinate = {
        (value.branch, value.phase_a_action_coordinate_hash): value
        for value in projection_family.projections
    }

    rows = []
    for registered_cell in qualification.source_cells:
        resource_cell_by_coordinate = {
            value.resource_coordinate_hash: value.resource_cell_hash
            for value in registered_cell.resource_cell_bindings
        }
        for crosswalk in registered_cell.packed_action_family.candidate_crosswalks:
            if crosswalk.q_status is not V1SignedWorkHardFQEndpointStatus.Q_SAFE_FEASIBLE:
                continue
            action = crosswalk.packed_action
            if action is None:
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                    "a qualified Q-safe alias lost its packed action",
                )
            branch = crosswalk.q_branch
            phase_entry = phase_aliases.get((branch, crosswalk.resource_coordinate_hash))
            f_entry = f_aliases.get((branch, crosswalk.resource_coordinate_hash))
            resource_cell_hash = resource_cell_by_coordinate.get(
                crosswalk.resource_coordinate_hash
            )
            if phase_entry is None or f_entry is None or resource_cell_hash is None:
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_F_ALIAS_COVERAGE,
                    "qualified resource/branch alias is absent from Phase-A or corrected-F custody",
                )
            source, phase_alias = phase_entry
            f_domain_hash, f_alias_hash, f_alias_id, f_resource_cell = f_entry
            if (
                phase_alias.resource_alias_id != f_alias_id
                or phase_alias.phase_a_resource_cell_id != f_resource_cell
            ):
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.SOURCE_IDENTITY_MISMATCH,
                    "Phase-A, corrected-F, and CUDA resource-cell aliases differ",
                )
            subset_matches = tuple(
                value
                for value in source.subset_identities
                if value.removed_route_index_hash == action.removed_route_index_hash
                and value.removed_count == action.removed_count
            )
            if len(subset_matches) != 1:
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                    "qualified packed action has no unique Phase-A action coordinate",
                )
            subset = subset_matches[0]
            cells = tuple(
                cell
                for cell in source.closed_cells(phase_alias.resource_alias_id)
                if cell.selected_shell_offsets == subset.shell_offsets
            )
            if len(cells) != 1:
                _refuse(
                    V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
                    "qualified Phase-A subset has no unique registered action cell",
                )
            action_cell = source.action_cell(
                registry_receipt_hash=registry.receipt.receipt_hash,
                resource_alias_id=phase_alias.resource_alias_id,
                action_cell_ordinal=cells[0].cell_ordinal,
            )
            identity = v1_signed_work_packed_action_dedup_identity(action)
            projection = projection_by_action_coordinate.get(
                (branch, subset.physical_action_coordinate_hash)
            )
            rows.append(
                V1SampledV7StructuralActionCrosswalkRow(
                    branch=branch,
                    qualification_registered_cell_hash=(
                        registered_cell.registered_cell_hash
                    ),
                    qualification_grid_key=registered_cell.grid_key,
                    resource_cell_hash=resource_cell_hash,
                    resource_coordinate_hash=crosswalk.resource_coordinate_hash,
                    phase_a_resource_alias_hash=phase_alias.alias_hash,
                    phase_a_resource_alias_id=phase_alias.resource_alias_id,
                    phase_a_resource_cell_id=phase_alias.phase_a_resource_cell_id,
                    phase_a_action_cell_hash=action_cell.action_cell_hash,
                    phase_a_action_coordinate_hash=(
                        subset.physical_action_coordinate_hash
                    ),
                    phase_a_subset_identity_hash=subset.subset_hash,
                    corrected_f_alternate_domain_hash=f_domain_hash,
                    corrected_f_resource_alias_hash=f_alias_hash,
                    qualification_crosswalk_hash=crosswalk.crosswalk_hash,
                    qualification_endpoint_decision_hash=(
                        crosswalk.endpoint_decision.receipt_hash
                    ),
                    packed_dedup_identity=identity,
                    physical_action_identity_hash=_physical_identity_hash(identity),
                    packed_action_hash=action.action_hash,
                    resulting_mask_hash=action.resulting_mask.mask_hash,
                    resulting_mask_packed_sha256=hashlib.sha256(
                        action.resulting_mask.packed_bits
                    ).hexdigest(),
                    removed_route_index_hash=action.removed_route_index_hash,
                    removed_count=action.removed_count,
                    origin_q_projection_receipt_hash=(
                        None if projection is None else projection.projection_hash
                    ),
                )
            )
    if not rows:
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
            "qualification contains no Q-safe packed physical actions",
        )
    return tuple(rows)


def issue_v1_sampled_v7_complete_action_coordinate_packed_crosswalk(
    *,
    branch_evidence: Sequence[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt],
    live_projections: Sequence[V1SampledV7CommonDOriginQLiveProjectionBindingReceipt],
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt:
    """Issue the complete structural/F/packed-action/origin-Q crosswalk."""

    branches = _exact_pair(
        branch_evidence,
        exact_type=V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
        branches=_CANONICAL_BRANCHES,
        getter=lambda value: value.branch,
        name="branch evidence",
    )
    projections = _exact_pair(
        live_projections,
        exact_type=V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        branches=_CANONICAL_PHASE_A_BRANCHES,
        getter=lambda value: value.phase_a_branch,
        name="live projection",
    )
    if (
        type(phase_a_registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(cuda_qualification) is not V1SignedWorkCUDADecisionWitnessQualification
    ):
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.INVALID_TYPE,
            "crosswalk issuance requires the exact live Phase-A registry and CUDA qualification",
        )
    try:
        branch_hashes = tuple(value.assert_integrity() for value in branches)
        projection_hashes = tuple(value.assert_integrity() for value in projections)
        phase_a_registry.assert_live_sources()
        cuda_qualification.assert_live_identity()
    except Exception as error:
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE,
            f"one retained crosswalk parent failed replay: {type(error).__name__}: {error}",
        )
    if (
        phase_a_registry.receipt.branch_axis != _CANONICAL_PHASE_A_BRANCHES
        or phase_a_registry.receipt.complete_branch_axis is not True
        or phase_a_registry.receipt.complete_action_cell_coverage is not True
        or any(value._phase_a_registry is not phase_a_registry for value in projections)
        or cuda_qualification.parent_mask_hash != phase_a_registry.receipt.current_mask_hash
        or cuda_qualification.route_axis_hash != phase_a_registry.receipt.route_axis_hash
        or cuda_qualification.route_count != branches[0].route_count
        or any(value.parent_mask_hash != cuda_qualification.parent_mask_hash for value in branches)
        or any(value.route_axis_hash != cuda_qualification.route_axis_hash for value in branches)
    ):
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "Phase-A, corrected-F, projection, and qualification chart axes differ",
        )
    projection_family = (
        issue_v1_sampled_v7_all_qualified_action_origin_q_projection_family(
            live_projections=projections,
            phase_a_registry=phase_a_registry,
            cuda_qualification=cuda_qualification,
        )
    )
    rows = _derive_structural_rows(
        branches=branches,
        projection_family=projection_family,
        registry=phase_a_registry,
        qualification=cuda_qualification,
    )
    observed_physical_image = {
        (value.resource_cell_hash, value.physical_action_identity_hash)
        for value in rows
    }
    if observed_physical_image != set(cuda_qualification.physical_action_family):
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE,
            "derived structural rows do not cover the exact qualified physical image",
        )
    structural_domain = {
        (value.branch.value, value.phase_a_action_coordinate_hash)
        for value in rows
    }
    projection_domain = set(projection_family.projection_domain)
    structural_alias_axis = {
        (
            value.qualification_registered_cell_hash,
            value.qualification_crosswalk_hash,
        )
        for value in rows
    }
    projection_alias_axis = {
        (value.registered_cell_hash, value.qualification_crosswalk_hash)
        for projection in projection_family.projections
        for value in projection.aliases
    }
    if (
        structural_domain != projection_domain
        or structural_alias_axis != projection_alias_axis
        or any(value.origin_q_projection_receipt_hash is None for value in rows)
    ):
        _refuse(
            V1SampledV7CompleteActionCrosswalkRefusalCode.MISSING_ALL_ACTION_ORIGIN_Q_PROJECTION_PRODUCER,
            "all-action origin-Q projection family is not set-equal to the complete structural crosswalk domain",
        )
    return V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt(
        branch_evidence_receipt_hashes=branch_hashes,
        source_root_projection_receipt_hashes=projection_hashes,
        all_action_projection_family_receipt_hash=projection_family.receipt_hash,
        phase_a_registry_receipt_hash=phase_a_registry.receipt.receipt_hash,
        cuda_qualification_hash=cuda_qualification.qualification_hash,
        structural_rows=rows,
        branch_action_projection_domain=projection_family.projection_domain,
        cuda_physical_action_family=tuple(cuda_qualification.physical_action_family),
        _branch_evidence=branches,
        _projection_family=projection_family,
        _phase_a_registry=phase_a_registry,
        _cuda_qualification=cuda_qualification,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = (
    "SAMPLED_V7_ALL_ACTION_Q_REQUIRED_PRODUCER_API",
    "SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_AUTHORITY",
    "SAMPLED_V7_COMPLETE_ACTION_CROSSWALK_SCHEMA",
    "V1SampledV7AllActionQProjectionMissingReceipt",
    "V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt",
    "V1SampledV7CompleteActionCrosswalkRefusal",
    "V1SampledV7CompleteActionCrosswalkRefusalCode",
    "V1SampledV7StructuralActionCrosswalkRow",
    "issue_v1_sampled_v7_complete_action_coordinate_packed_crosswalk",
)
