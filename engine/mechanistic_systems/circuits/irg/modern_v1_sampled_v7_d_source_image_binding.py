"""Positive sampled-V7 D source-image binding.

The complete action-coordinate/packed-action crosswalk closes the source image
which used to be deliberately uninhabited.  No caller-authored crosswalk,
completeness flag, or digest is accepted: the issuer derives it from exact live
parents and retains them for replay.  This grants source-image binding only;
D stability, C/K/O, commit, terminal, performance, and Pareto stay closed.

The former missing-crosswalk receipt remains readable for historical replay.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceHandle,
)
from .modern_v1_sampled_v7_branch_decision_image_production import (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
)
from .modern_v1_sampled_v7_common_d_origin_q_live_projection import (
    V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
)
from .modern_v1_sampled_v7_complete_action_coordinate_packed_crosswalk import (
    V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
    V1SampledV7CompleteActionCrosswalkRefusal,
    V1SampledV7CompleteActionCrosswalkRefusalCode as CrosswalkRefusalCode,
    issue_v1_sampled_v7_complete_action_coordinate_packed_crosswalk,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    V1SignedWorkCUDADecisionWitnessQualification,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
)


SAMPLED_V7_D_SOURCE_IMAGE_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_complete_action_source_image_binding_v2"
)
SAMPLED_V7_D_SOURCE_IMAGE_BINDING_AUTHORITY: Final = (
    "complete_live_common_D_corrected_F_origin_Q_phase_A_and_CUDA_packed_action_"
    "source_image_binding_only_no_D_stability_C_K_O_commit_terminal_"
    "performance_or_pareto_authority_v2"
)
SAMPLED_V7_D_SOURCE_IMAGE_REQUIRED_PRODUCER_API: Final = (
    "V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt",
    "complete_phase_a_branch_resource_action_coordinate_axis",
    "complete_corrected_f_alternate_domain_coordinate_axis_per_branch",
    "resource_cell_to_action_coordinate_alias_axis",
    "action_coordinate_to_packed_dedup_identity_axis",
    "action_coordinate_to_packed_action_hash_and_child_sha256_axis",
    "all_qualified_physical_actions_projected_to_origin_q_sources",
    "exact_live_common_source_phase_a_registry_and_cuda_cell_ancestry",
)
SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_source_image_binding_missing_crosswalk_v1"
)
SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_AUTHORITY: Final = (
    "fail_closed_missing_complete_phase_A_action_coordinate_corrected_F_"
    "alternate_domain_to_CUDA_resource_cell_packed_action_crosswalk_no_source_"
    "image_D_stability_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-source-image-binding-"
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


class V1SampledV7DSourceImageBindingRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    MISSING_OR_DUPLICATE_BRANCH = "missing_or_duplicate_branch"
    STALE_OR_EVICTED_SOURCE = "stale_or_evicted_source"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    PARTIAL_RESOURCE_ACTION_COVERAGE = "partial_resource_action_coverage"
    MISSING_COMPLETE_ACTION_CROSSWALK_PRODUCER = (
        "missing_complete_action_crosswalk_producer"
    )


class V1SampledV7DSourceImageBindingRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7DSourceImageBindingRefusalCode,
        message: str,
        *,
        missing_premise_receipt: object | None = None,
    ) -> None:
        if type(code) is not V1SampledV7DSourceImageBindingRefusalCode:
            raise TypeError("source-image refusal code must be exact")
        self.code = code
        self.missing_premise_receipt = missing_premise_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DSourceImageBindingRefusalCode,
    message: str,
    *,
    missing_premise_receipt: object | None = None,
) -> NoReturn:
    raise V1SampledV7DSourceImageBindingRefusal(
        code, message, missing_premise_receipt=missing_premise_receipt
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1SampledV7DSourceImageMissingCrosswalkReceipt:
    """Historical refusal receipt for replay compatibility only."""

    branch_evidence_receipt_hashes: tuple[str, str]
    projection_receipt_hashes: tuple[str, str]
    common_source_receipt_hash: str
    phase_a_registry_receipt_hash: str
    cuda_qualification_hash: str
    phase_a_branch_axis: tuple[V1SignedWorkVariableShellQBranch, ...]
    corrected_f_alternate_domain_hashes: tuple[str, str]
    cuda_resource_cell_family: tuple[str, ...]
    cuda_physical_action_family: tuple[tuple[str, str], ...]
    cuda_preferred_physical_decision_family: tuple[tuple[str, str], ...]
    required_producer_api: tuple[str, ...] = SAMPLED_V7_D_SOURCE_IMAGE_REQUIRED_PRODUCER_API
    available_parent_graph_hash: str = field(init=False)
    source_image_binding_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_AUTHORITY
    schema: str = SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_SCHEMA
    _issuer_token: InitVar[object | None] = None

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _MISSING_TOKEN:
            raise TypeError("source-image missing-crosswalk receipts are issuer-only")
        axes = (
            self.branch_evidence_receipt_hashes,
            self.projection_receipt_hashes,
            self.corrected_f_alternate_domain_hashes,
            self.cuda_resource_cell_family,
            self.required_producer_api,
        )
        if any(type(axis) is not tuple or not axis for axis in axes):
            raise TypeError("historical missing-crosswalk axes must be nonempty tuples")
        if (
            len(self.branch_evidence_receipt_hashes) != 2
            or len(self.projection_receipt_hashes) != 2
            or len(self.corrected_f_alternate_domain_hashes) != 2
            or len(set(self.branch_evidence_receipt_hashes)) != 2
            or len(set(self.projection_receipt_hashes)) != 2
            or self.phase_a_branch_axis != _CANONICAL_PHASE_A_BRANCHES
            or not self.cuda_physical_action_family
            or not self.cuda_preferred_physical_decision_family
            or any(
                type(value) is not tuple
                or len(value) != 2
                or any(type(item) is not str or not item for item in value)
                for axis in (
                    self.cuda_physical_action_family,
                    self.cuda_preferred_physical_decision_family,
                )
                for value in axis
            )
        ):
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.PARTIAL_RESOURCE_ACTION_COVERAGE,
                "historical source image has an incomplete finite axis",
            )
        for axis in axes:
            tuple(_identity(value, name="historical axis identity") for value in axis)
        for name in (
            "common_source_receipt_hash",
            "phase_a_registry_receipt_hash",
            "cuda_qualification_hash",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        flags = (
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
            any(type(value) is not bool or value is not False for value in flags)
            or self.required_producer_api
            != SAMPLED_V7_D_SOURCE_IMAGE_REQUIRED_PRODUCER_API
            or self.authority_scope
            != SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_AUTHORITY
            or self.schema != SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_SCHEMA
        ):
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "historical refusal receipt crossed its authority boundary",
            )
        object.__setattr__(
            self,
            "available_parent_graph_hash",
            content_hash(
                {
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if not name.startswith("_") and name != "available_parent_graph_hash"
                },
                prefix=_PREFIX + "available-parent-graph-",
            ),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "missing-crosswalk-",
        )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("source-image missing-crosswalk receipts are nonserializable")


def _exact_pair(
    values: Sequence[object],
    *,
    exact_type: type,
    branches: tuple[object, object],
    branch_getter,
    name: str,
) -> tuple[object, object]:
    if type(values) is not tuple:
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    axis = tuple(values)
    if (
        len(axis) != 2
        or any(type(value) is not exact_type for value in axis)
        or tuple(branch_getter(value) for value in axis) != branches
    ):
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.MISSING_OR_DUPLICATE_BRANCH,
            f"{name} must contain each canonical branch exactly once",
        )
    return axis  # type: ignore[return-value]


def _replay_positive_graph(
    *,
    branches: tuple[
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
    ],
    projections: tuple[
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
    ],
    common_source: V1SampledFullPopulationCompactSourceHandle,
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification,
    complete_crosswalk: V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
) -> dict[str, object]:
    if (
        type(common_source) is not V1SampledFullPopulationCompactSourceHandle
        or type(phase_a_registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(cuda_qualification) is not V1SignedWorkCUDADecisionWitnessQualification
        or type(complete_crosswalk)
        is not V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt
    ):
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.INVALID_TYPE,
            "positive replay requires exact live parents and complete crosswalk",
        )
    try:
        branch_hashes = tuple(value.assert_integrity() for value in branches)
        projection_hashes = tuple(value.assert_integrity() for value in projections)
        common_receipt = common_source.receipt
        common_hash = common_receipt.assert_integrity()
        phase_a_registry.assert_live_sources()
        registry_receipt = phase_a_registry.receipt
        cuda_qualification.assert_live_identity()
        qualification_hash = cuda_qualification.qualification_hash
        crosswalk_hash = complete_crosswalk.assert_integrity()
    except V1SampledV7DSourceImageBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
            f"one source-image parent failed replay: {type(error).__name__}: {error}",
        )

    family = complete_crosswalk._projection_family
    if (
        complete_crosswalk._branch_evidence[0] is not branches[0]
        or complete_crosswalk._branch_evidence[1] is not branches[1]
        or family._live_projections[0] is not projections[0]
        or family._live_projections[1] is not projections[1]
        or complete_crosswalk._phase_a_registry is not phase_a_registry
        or complete_crosswalk._cuda_qualification is not cuda_qualification
        or any(value._common_source is not common_source for value in projections)
        or complete_crosswalk.branch_evidence_receipt_hashes != branch_hashes
        or complete_crosswalk.source_root_projection_receipt_hashes != projection_hashes
        or complete_crosswalk.phase_a_registry_receipt_hash != registry_receipt.receipt_hash
        or complete_crosswalk.cuda_qualification_hash != qualification_hash
        or family.common_source_receipt_hash != common_hash
        or any(value.source_receipt_hash != common_hash for value in branches)
        or any(value.source_replay_hash != common_receipt.source_replay_hash for value in branches)
        or any(value.common_source_receipt_hash != common_hash for value in projections)
        or any(
            value.common_source_replay_hash != common_receipt.source_replay_hash
            for value in projections
        )
        or any(value.parent_mask_hash != branches[0].parent_mask_hash for value in branches)
        or any(value.route_axis_hash != branches[0].route_axis_hash for value in branches)
        or any(value.route_count != branches[0].route_count for value in branches)
        or any(value.uncertainty_set_hash != branches[0].uncertainty_set_hash for value in branches)
        or cuda_qualification.parent_mask_hash != branches[0].parent_mask_hash
        or cuda_qualification.route_axis_hash != branches[0].route_axis_hash
        or cuda_qualification.route_count != branches[0].route_count
        or registry_receipt.current_mask_hash != branches[0].parent_mask_hash
        or registry_receipt.route_axis_hash != branches[0].route_axis_hash
        or registry_receipt.branch_axis != _CANONICAL_PHASE_A_BRANCHES
        or registry_receipt.complete_branch_axis is not True
        or registry_receipt.complete_action_cell_coverage is not True
    ):
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "crosswalk is not the exact common-D/branch/Phase-A/CUDA graph",
        )

    resource_family = tuple(cuda_qualification.resource_cell_family)
    physical_family = tuple(cuda_qualification.physical_action_family)
    preferred_family = tuple(cuda_qualification.preferred_physical_decision_family)
    rows = complete_crosswalk.structural_rows
    row_physical_image = {
        (value.resource_cell_hash, value.physical_action_identity_hash) for value in rows
    }
    row_domain = {
        (value.branch.value, value.phase_a_action_coordinate_hash) for value in rows
    }
    if (
        not resource_family
        or len(set(resource_family)) != len(resource_family)
        or not physical_family
        or len(set(physical_family)) != len(physical_family)
        or not preferred_family
        or not set(preferred_family).issubset(set(physical_family))
        or {resource for resource, _ in physical_family} != set(resource_family)
        or tuple(complete_crosswalk.cuda_physical_action_family) != physical_family
        or row_physical_image != set(physical_family)
        or row_domain != set(complete_crosswalk.branch_action_projection_domain)
        or any(value.origin_q_projection_receipt_hash is None for value in rows)
    ):
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.PARTIAL_RESOURCE_ACTION_COVERAGE,
            "positive crosswalk lost a resource, action, alias, or origin-Q projection",
        )

    return {
        "branch_evidence_receipt_hashes": branch_hashes,
        "source_root_projection_receipt_hashes": projection_hashes,
        "common_source_receipt_hash": common_hash,
        "common_source_replay_hash": common_receipt.source_replay_hash,
        "complete_action_crosswalk_receipt_hash": crosswalk_hash,
        "all_action_projection_family_receipt_hash": (
            complete_crosswalk.all_action_projection_family_receipt_hash
        ),
        "phase_a_registry_receipt_hash": registry_receipt.receipt_hash,
        "cuda_qualification_hash": qualification_hash,
        "parent_mask_hash": branches[0].parent_mask_hash,
        "route_axis_hash": branches[0].route_axis_hash,
        "route_count": branches[0].route_count,
        "uncertainty_set_hash": branches[0].uncertainty_set_hash,
        "structural_row_hashes": tuple(value.structural_row_hash for value in rows),
        "branch_action_projection_domain": complete_crosswalk.branch_action_projection_domain,
        "cuda_resource_cell_family": resource_family,
        "cuda_physical_action_family": physical_family,
        "cuda_preferred_physical_decision_family": preferred_family,
    }


@dataclass(frozen=True, slots=True)
class V1SampledV7DSourceImageBindingReceipt:
    """Complete common-D/F/Q/packed-action source image, not a decision."""

    branch_evidence_receipt_hashes: tuple[str, str]
    source_root_projection_receipt_hashes: tuple[str, str]
    common_source_receipt_hash: str
    common_source_replay_hash: str
    complete_action_crosswalk_receipt_hash: str
    all_action_projection_family_receipt_hash: str
    phase_a_registry_receipt_hash: str
    cuda_qualification_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    uncertainty_set_hash: str
    structural_row_hashes: tuple[str, ...]
    branch_action_projection_domain: tuple[tuple[str, str], ...]
    cuda_resource_cell_family: tuple[str, ...]
    cuda_physical_action_family: tuple[tuple[str, str], ...]
    cuda_preferred_physical_decision_family: tuple[tuple[str, str], ...]
    receipt_hash: str = field(init=False)
    source_image_binding_authority: bool = True
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_D_SOURCE_IMAGE_BINDING_AUTHORITY
    schema: str = SAMPLED_V7_D_SOURCE_IMAGE_BINDING_SCHEMA
    _branch_evidence: tuple[
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
        V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
    ] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _live_projections: tuple[
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
    ] = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _common_source: V1SampledFullPopulationCompactSourceHandle = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _phase_a_registry: V1SignedWorkVariableShellActionCellRegistry = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _complete_crosswalk: V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object | None] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER_TOKEN:
            raise TypeError("D source-image binding receipts are issuer-only")
        flags = (
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
            or flags != (True, False, False, False, False, False, False, False, False)
            or self.authority_scope != SAMPLED_V7_D_SOURCE_IMAGE_BINDING_AUTHORITY
            or self.schema != SAMPLED_V7_D_SOURCE_IMAGE_BINDING_SCHEMA
        ):
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "D source image crossed its binding-only authority",
            )
        public = self._replay()
        if any(getattr(self, name) != value for name, value in public.items()):
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
                "D source-image public image differs from its live parent graph",
            )
        observed = self._compute_hash(public)
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
                "D source-image receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _replay(self) -> dict[str, object]:
        return _replay_positive_graph(
            branches=self._branch_evidence,
            projections=self._live_projections,
            common_source=self._common_source,
            phase_a_registry=self._phase_a_registry,
            cuda_qualification=self._cuda_qualification,
            complete_crosswalk=self._complete_crosswalk,
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "receipt_hash"
        }

    def _compute_hash(self, public: dict[str, object] | None = None) -> str:
        observed_public = self._replay() if public is None else public
        if any(getattr(self, name) != value for name, value in observed_public.items()):
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
                "D source image no longer equals its live producer graph",
            )
        return content_hash(self._payload(), prefix=_PREFIX + "positive-receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self.receipt_hash or observed != self._integrity_seal:
            _refuse(
                V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
                "D source-image receipt no longer replays",
            )
        return observed

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("D source-image binding receipts are nonserializable")


def issue_v1_sampled_v7_d_source_image_binding(
    *,
    branch_evidence: Sequence[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt],
    live_projections: Sequence[V1SampledV7CommonDOriginQLiveProjectionBindingReceipt],
    common_source: V1SampledFullPopulationCompactSourceHandle,
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    cuda_qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> V1SampledV7DSourceImageBindingReceipt:
    """Derive and bind the complete source image from exact live parents."""

    branches = _exact_pair(
        branch_evidence,
        exact_type=V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
        branches=_CANONICAL_BRANCHES,
        branch_getter=lambda value: value.branch,
        name="branch evidence",
    )
    projections = _exact_pair(
        live_projections,
        exact_type=V1SampledV7CommonDOriginQLiveProjectionBindingReceipt,
        branches=_CANONICAL_PHASE_A_BRANCHES,
        branch_getter=lambda value: value.phase_a_branch,
        name="live projection",
    )
    if (
        type(common_source) is not V1SampledFullPopulationCompactSourceHandle
        or type(phase_a_registry) is not V1SignedWorkVariableShellActionCellRegistry
        or type(cuda_qualification) is not V1SignedWorkCUDADecisionWitnessQualification
    ):
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.INVALID_TYPE,
            "source-image issuance requires exact live common-D, Phase-A, and CUDA parents",
        )
    try:
        crosswalk = issue_v1_sampled_v7_complete_action_coordinate_packed_crosswalk(
            branch_evidence=branches,
            live_projections=projections,
            phase_a_registry=phase_a_registry,
            cuda_qualification=cuda_qualification,
        )
    except V1SampledV7CompleteActionCrosswalkRefusal as error:
        code = {
            CrosswalkRefusalCode.INVALID_TYPE: (
                V1SampledV7DSourceImageBindingRefusalCode.INVALID_TYPE
            ),
            CrosswalkRefusalCode.MISSING_OR_DUPLICATE_BRANCH: (
                V1SampledV7DSourceImageBindingRefusalCode.MISSING_OR_DUPLICATE_BRANCH
            ),
            CrosswalkRefusalCode.STALE_OR_EVICTED_SOURCE: (
                V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE
            ),
            CrosswalkRefusalCode.INCOMPLETE_F_ALIAS_COVERAGE: (
                V1SampledV7DSourceImageBindingRefusalCode.PARTIAL_RESOURCE_ACTION_COVERAGE
            ),
            CrosswalkRefusalCode.INCOMPLETE_PACKED_ACTION_COVERAGE: (
                V1SampledV7DSourceImageBindingRefusalCode.PARTIAL_RESOURCE_ACTION_COVERAGE
            ),
            CrosswalkRefusalCode.MISSING_ALL_ACTION_ORIGIN_Q_PROJECTION_PRODUCER: (
                V1SampledV7DSourceImageBindingRefusalCode.PARTIAL_RESOURCE_ACTION_COVERAGE
            ),
        }.get(
            error.code,
            V1SampledV7DSourceImageBindingRefusalCode.SOURCE_IDENTITY_MISMATCH,
        )
        _refuse(code, f"complete crosswalk derivation refused: {error}")
    except V1SampledV7DSourceImageBindingRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DSourceImageBindingRefusalCode.STALE_OR_EVICTED_SOURCE,
            f"complete crosswalk derivation failed: {type(error).__name__}: {error}",
        )
    public = _replay_positive_graph(
        branches=branches,  # type: ignore[arg-type]
        projections=projections,  # type: ignore[arg-type]
        common_source=common_source,
        phase_a_registry=phase_a_registry,
        cuda_qualification=cuda_qualification,
        complete_crosswalk=crosswalk,
    )
    return V1SampledV7DSourceImageBindingReceipt(
        **public,
        _branch_evidence=branches,
        _live_projections=projections,
        _common_source=common_source,
        _phase_a_registry=phase_a_registry,
        _cuda_qualification=cuda_qualification,
        _complete_crosswalk=crosswalk,
        _issuer_token=_ISSUER_TOKEN,
    )


__all__ = (
    "SAMPLED_V7_D_SOURCE_IMAGE_BINDING_AUTHORITY",
    "SAMPLED_V7_D_SOURCE_IMAGE_BINDING_SCHEMA",
    "SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_AUTHORITY",
    "SAMPLED_V7_D_SOURCE_IMAGE_MISSING_CROSSWALK_SCHEMA",
    "SAMPLED_V7_D_SOURCE_IMAGE_REQUIRED_PRODUCER_API",
    "V1SampledV7DSourceImageBindingReceipt",
    "V1SampledV7DSourceImageBindingRefusal",
    "V1SampledV7DSourceImageBindingRefusalCode",
    "V1SampledV7DSourceImageMissingCrosswalkReceipt",
    "issue_v1_sampled_v7_d_source_image_binding",
)
