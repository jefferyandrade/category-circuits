"""Live-source authority seam for CUDA D-decision continuum closure.

The older continuum receipt type is intentionally representational: a caller
can construct it from numeric margins and source hashes.  That is useful for
replay, but it cannot establish that those numbers were derived from the live
CUDA theorem state.  This module is the stricter producer-side seam.

The legacy five-margin issuer remains readable and fail-closed.  Canonical v2
retains exact conditional custody of the sampled-V7 source image,
tie-complete closed top-k action-cell registry, aggregate variable-shell
F-separation owner, categorical packed-byte quotient, and production full-U
F/Q hard-action invariance.  Order and byte identity are therefore
set/quotient statements, not artificial scalar distances.  Those finite
objects do not cover continuum reach across Phase-A resource-cell exits or
between registered source images.  The retained four-owner scalar axis is
lineage provenance, not one homogeneous uncertainty ball.  Until an
endogenous typed-source-to-boundary reach producer lands, the public positive issuer
refuses and every D/C/K/O/commit/terminal/performance/Pareto authority remains
closed.  No issuer accepts caller-authored floats, hashes, or completeness
flags.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from itertools import product
import math
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_signed_work_decision_enclosure import (
    SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE,
    V1SignedWorkDecisionContinuumClosureMode,
    V1SignedWorkDecisionMarginReceipt,
    V1SignedWorkDecisionMarginRole,
    V1SignedWorkDecisionOwnerRole,
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
)
from .modern_v1_sampled_v7_complete_action_coordinate_packed_crosswalk import (
    V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
)
from .modern_v1_sampled_v7_branch_decision_image_production import (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
)
from .modern_v1_sampled_v7_d_source_image_binding import (
    V1SampledV7DSourceImageBindingReceipt,
)
from .modern_v1_signed_work_phase_a_endogenous_reach_source import (
    PHASE_A_ENDOGENOUS_REACH_API_NAME,
    PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE,
    PHASE_A_ENDOGENOUS_REACH_REASON,
    PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
)
from .modern_v1_signed_work_decision_witness_cuda import (
    V1SignedWorkCUDADecisionWitnessQualification,
    V1SignedWorkCUDARegisteredUncertaintyCell,
)
from .modern_v1_signed_work_full_u_f_q_action_invariance_production import (
    V1SignedWorkFullUFQProductionActionInvarianceReceiptV3,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellIdentity,
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellActionSeparationCertificate,
    V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt,
    V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt,
)


SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_live_cuda_margin_source_v2"
)
SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_AUTHORITY_SCOPE: Final = (
    "live_source_derived_cuda_D_decision_margin_family_only_no_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v2"
)
SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_POLICY: Final = (
    "exact_positive_sampled_v7_source_image_supplies_identical_branch_retained_"
    "legacy_four_owner_scalar_lineage_as_provenance_only_then_typed_source_"
    "owned_outward_decision_boundary_reductions_"
    "with_no_caller_numeric_hash_or_completeness_inputs_v2"
)
SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_live_four_owner_radius_source_v1"
)
SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_AUTHORITY: Final = (
    "exact_source_image_bound_four_owner_radius_custody_only_no_continuum_"
    "margin_D_stability_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_complete_action_image_source_custody_v2"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_AUTHORITY: Final = (
    "exact_live_complete_closed_action_cell_aggregate_F_and_packed_quotient_"
    "custody_only_conditional_on_frozen_shell_no_global_D_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v2"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_D_W_complete_action_image_global_closure_v2"
)
SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_AUTHORITY: Final = (
    "exact_source_bound_complete_action_image_plus_production_full_U_F_Q_"
    "continuum_D_closure_only_no_C_K_O_commit_terminal_performance_or_pareto_"
    "authority_v2"
)

_PREFIX: Final = "partition-free-v1-signed-work-live-margin-source-"
_RECEIPT_ISSUER_TOKEN: Final = object()
_OWNER_RADIUS_ISSUER_TOKEN: Final = object()
_ACTION_IMAGE_CUSTODY_ISSUER_TOKEN: Final = object()
_ACTION_IMAGE_CERTIFICATE_ISSUER_TOKEN: Final = object()
_CANONICAL_OWNER_ROLES: Final = (
    V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
    V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
    V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
    V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
)
_CANONICAL_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)


class V1SignedWorkDecisionMarginSourceRefusalCode(str, Enum):
    """Fail-closed outcomes at the live margin-source seam."""

    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    LIVE_STATE_EVICTED = "live_state_evicted"
    CUDA_REQUIRED = "cuda_required"
    INCOMPLETE_OWNER_PRODUCT = "incomplete_owner_product"
    INCOMPLETE_ACTION_CELL_COVER = "incomplete_action_cell_cover"
    NONUNIQUE_ACTION_IMAGE_WINNER = "nonunique_action_image_winner"
    INCOMPLETE_PHYSICAL_ACTION_QUOTIENT = "incomplete_physical_action_quotient"
    INCOMPLETE_GLOBAL_ACTION_CLOSURE = "incomplete_global_action_closure"
    MISSING_PRODUCER_API = "missing_live_margin_producer_api"
    AUTHORITY_VIOLATION = "authority_violation"


@dataclass(frozen=True, slots=True)
class V1SignedWorkDecisionMarginProducerRequirement:
    """One exact producer-owned API required for live margin derivation."""

    producer_type: str
    api_name: str
    supplies: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        for name in ("producer_type", "api_name", "reason"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be one nonempty string")
        supplies = tuple(self.supplies)
        if (
            not supplies
            or any(type(value) is not str or not value for value in supplies)
            or len(supplies) != len(set(supplies))
        ):
            raise TypeError("producer requirement supplies must be unique strings")
        object.__setattr__(self, "supplies", supplies)

    @property
    def requirement_id(self) -> str:
        return f"{self.producer_type}.{self.api_name}"

    @property
    def requirement_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "producer-requirement-")


REQUIRED_LIVE_MARGIN_PRODUCER_APIS: Final = (
    V1SignedWorkDecisionMarginProducerRequirement(
        producer_type=PHASE_A_ENDOGENOUS_REACH_PRODUCER_TYPE,
        api_name=PHASE_A_ENDOGENOUS_REACH_API_NAME,
        supplies=PHASE_A_ENDOGENOUS_REACH_TYPED_SUPPLIES,
        reason=PHASE_A_ENDOGENOUS_REACH_REASON,
    ),
)


class V1SignedWorkDecisionMarginSourceRefusal(RuntimeError):
    """Typed refusal carrying the precise missing source producer contract."""

    def __init__(
        self,
        code: V1SignedWorkDecisionMarginSourceRefusalCode,
        message: str,
        *,
        missing_requirements: tuple[
            V1SignedWorkDecisionMarginProducerRequirement, ...
        ] = (),
        live_source_identity_verified: bool = False,
        landed_owner_radius_source: (
            V1SignedWorkLiveOwnerRadiusSourceReceipt | None
        ) = None,
        landed_complete_action_image_custody: (
            V1SignedWorkCompleteActionImageCustodyReceipt | None
        ) = None,
    ) -> None:
        if type(code) is not V1SignedWorkDecisionMarginSourceRefusalCode:
            raise TypeError("margin-source refusal code must be exact")
        requirements = tuple(missing_requirements)
        if any(
            type(value) is not V1SignedWorkDecisionMarginProducerRequirement
            for value in requirements
        ):
            raise TypeError("missing requirements must be exact producer requirements")
        if type(live_source_identity_verified) is not bool:
            raise TypeError("live-source verification flag must be exact")
        if (
            landed_owner_radius_source is not None
            and type(landed_owner_radius_source)
            is not V1SignedWorkLiveOwnerRadiusSourceReceipt
        ):
            raise TypeError("landed owner-radius source must be one exact receipt")
        if (
            landed_complete_action_image_custody is not None
            and type(landed_complete_action_image_custody)
            is not V1SignedWorkCompleteActionImageCustodyReceipt
        ):
            raise TypeError("landed complete-action custody must be one exact receipt")
        self.code = code
        self.missing_requirements = requirements
        self.live_source_identity_verified = live_source_identity_verified
        self.landed_owner_radius_source = landed_owner_radius_source
        self.landed_complete_action_image_custody = landed_complete_action_image_custody
        suffix = ""
        if requirements:
            suffix = "; required APIs: " + ", ".join(
                value.requirement_id for value in requirements
            )
        super().__init__(f"{code.value}: {message}{suffix}")


def _refuse(
    code: V1SignedWorkDecisionMarginSourceRefusalCode,
    message: str,
    *,
    missing_requirements: tuple[
        V1SignedWorkDecisionMarginProducerRequirement, ...
    ] = (),
    live_source_identity_verified: bool = False,
    landed_owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt | None = None,
    landed_complete_action_image_custody: (
        V1SignedWorkCompleteActionImageCustodyReceipt | None
    ) = None,
) -> NoReturn:
    raise V1SignedWorkDecisionMarginSourceRefusal(
        code,
        message,
        missing_requirements=missing_requirements,
        live_source_identity_verified=live_source_identity_verified,
        landed_owner_radius_source=landed_owner_radius_source,
        landed_complete_action_image_custody=(landed_complete_action_image_custody),
    )


def _closed_downstream_authority(value: object) -> None:
    for name in (
        "generic_C_stream_open_authority",
        "production_C_stream_open_authority",
        "C_measurement_authority",
        "c_measurement_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "performance_authority",
        "pareto_authority",
    ):
        if bool(getattr(value, name, False)):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                f"finite qualification already opened downstream authority: {name}",
            )


def _source_state_evicted(
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
) -> bool:
    for cell in qualification.source_cells:
        if cell.phase_a_state_handle.evicted:
            return True
        if any(
            binding.state_handle.is_evicted for binding in cell.phase_b_state_bindings
        ):
            return True
    return False


def _assert_exact_live_product(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]:
    """Replay exact owner/product/source ancestry before capability refusal."""

    if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "live margin derivation requires an exact positive sampled-V7 source image",
        )
    if type(qualification) is not V1SignedWorkCUDADecisionWitnessQualification:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "live margin derivation requires an exact CUDA D qualification",
        )
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "live margin derivation requires the exact four-owner uncertainty set",
        )
    if _source_state_evicted(qualification):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.LIVE_STATE_EVICTED,
            "a retained Phase-A or Phase-B theorem-state handle was evicted",
        )
    try:
        source_image_hash = source_image.assert_integrity()
    except Exception as error:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"positive sampled-V7 source image failed replay: {type(error).__name__}",
        )
    if source_image.source_image_binding_authority is not True:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
            "sampled-V7 source image does not grant positive binding authority",
        )
    _closed_downstream_authority(source_image)
    try:
        qualification.assert_live_identity()
    except Exception as error:
        if _source_state_evicted(qualification):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.LIVE_STATE_EVICTED,
                "live theorem state was evicted during identity replay",
            )
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"live qualification identity replay failed: {type(error).__name__}",
        )

    owner_hashes = (
        uncertainty.signed_f_sampling.set_hash,
        uncertainty.literal_j_sampling.set_hash,
        uncertainty.fd_residual.set_hash,
        uncertainty.solver_rounding.set_hash,
    )
    if (
        qualification.uncertainty_set_hash != uncertainty.uncertainty_hash
        or qualification.route_axis_hash != uncertainty.route_axis_hash
        or qualification.registered_d_w_prefix_hash
        != uncertainty.registered_d_w_prefix_hash
        or (
            qualification.signed_f_uncertainty_set_hash,
            qualification.literal_j_uncertainty_set_hash,
            qualification.fd_residual_uncertainty_set_hash,
            qualification.solver_uncertainty_set_hash,
        )
        != owner_hashes
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            "qualification and requested owner-set ancestry differ",
        )
    _closed_downstream_authority(qualification)
    if (
        not qualification.complete_registered_uncertainty_product
        or not qualification.resource_cell_physical_action_invariant
        or not qualification.byte_deduplicated_physical_actions_verified
        or not qualification.final_solver_dual_custody_verified
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_OWNER_PRODUCT,
            "finite qualification does not close its registered decision image",
        )

    cells = tuple(qualification.source_cells)
    if (
        not cells
        or any(
            type(value) is not V1SignedWorkCUDARegisteredUncertaintyCell
            for value in cells
        )
        or tuple(value.registered_cell_hash for value in cells)
        != qualification.registered_cell_hashes
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            "qualification lost its exact live source-cell family",
        )
    observed_production_cuda_sources = all(
        cell.production_cuda_sources for cell in cells
    )
    if (
        type(qualification.production_cuda_sources_verified) is not bool
        or qualification.production_cuda_sources_verified
        is not observed_production_cuda_sources
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            "qualification CUDA provenance flag differs from its live source cells",
        )
    expected_grid = tuple(product(*uncertainty.realization_axes))
    observed_grid = tuple(value.grid_key for value in cells)
    if len(observed_grid) != len(set(observed_grid)) or set(observed_grid) != set(
        expected_grid
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_OWNER_PRODUCT,
            "live source cells do not cover the exact four-owner product",
        )
    for cell in cells:
        if (
            cell.uncertainty_set_hash != uncertainty.uncertainty_hash
            or cell.registered_d_w_prefix_hash != uncertainty.registered_d_w_prefix_hash
            or (
                cell.signed_f_uncertainty_set_hash,
                cell.literal_j_uncertainty_set_hash,
                cell.fd_residual_uncertainty_set_hash,
                cell.solver_uncertainty_set_hash,
            )
            != owner_hashes
            or cell.registered_d_coordinate_hash
            != qualification.registered_d_coordinate_hash
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "one live source cell left the registered owner/product ancestry",
            )

    branches = tuple(source_image._branch_evidence)
    if (
        len(branches) != 2
        or any(
            type(value) is not V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
            for value in branches
        )
        or tuple(value.branch for value in branches) != _CANONICAL_BRANCHES
        or source_image._cuda_qualification is not qualification
        or source_image.receipt_hash != source_image_hash
        or source_image.cuda_qualification_hash != qualification.qualification_hash
        or source_image.uncertainty_set_hash != uncertainty.uncertainty_hash
        or source_image.route_axis_hash != uncertainty.route_axis_hash
        or source_image.route_axis_hash != qualification.route_axis_hash
        or source_image.route_count != qualification.route_count
        or source_image.parent_mask_hash != qualification.parent_mask_hash
        or source_image.branch_evidence_receipt_hashes
        != tuple(value.receipt_hash for value in branches)
        or any(value._uncertainty is not uncertainty for value in branches)
        or any(
            value.uncertainty_set_hash != uncertainty.uncertainty_hash
            or value.route_axis_hash != uncertainty.route_axis_hash
            or value.route_count != qualification.route_count
            or value.parent_mask_hash != qualification.parent_mask_hash
            for value in branches
        )
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            "source image, branch evidence, qualification, and uncertainty are not one live graph",
        )

    first = tuple(branches[0]._owners)
    second = tuple(branches[1]._owners)
    owner_coordinate_axes = tuple(
        tuple(
            (
                value.owner_role.value,
                value.uncertainty_set_hash,
                value.owner_set_hash,
                tuple(value.realization_hashes),
                float(value.outward_radius_upper_bound),
            )
            for value in owners
        )
        for owners in (first, second)
    )
    if (
        len(first) != 4
        or len(second) != 4
        or any(
            type(value) is not V1SignedWorkDecisionOwnerUncertaintyReceipt
            for value in first + second
        )
        or tuple(value.owner_role for value in first) != _CANONICAL_OWNER_ROLES
        or tuple(value.owner_role for value in second) != _CANONICAL_OWNER_ROLES
        or first != second
        or tuple(value.receipt_hash for value in first)
        != tuple(value.receipt_hash for value in second)
        or any(
            branch.owner_receipt_hashes != tuple(value.receipt_hash for value in owners)
            for branch, owners in zip(branches, (first, second), strict=True)
        )
        or any(
            branch.owner_uncertainty_coordinate_axis != axis
            for branch, axis in zip(branches, owner_coordinate_axes, strict=True)
        )
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_OWNER_PRODUCT,
            "the two source-image branches do not retain one identical "
            "legacy four-owner scalar-lineage family",
        )

    owners_by_role = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: uncertainty.signed_f_sampling,
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: uncertainty.literal_j_sampling,
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: uncertainty.fd_residual,
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: uncertainty.solver_rounding,
    }
    qualification_owner_hashes = dict(
        zip(_CANONICAL_OWNER_ROLES, owner_hashes, strict=True)
    )
    for owner in first:
        registered = owners_by_role[owner.owner_role]
        if (
            owner.uncertainty_set_hash != uncertainty.uncertainty_hash
            or owner.owner_set_hash != registered.set_hash
            or owner.owner_set_hash != qualification_owner_hashes[owner.owner_role]
            or owner.registered_d_w_prefix_hash
            != uncertainty.registered_d_w_prefix_hash
            or owner.registered_d_w_prefix_hash
            != qualification.registered_d_w_prefix_hash
            or owner.route_axis_hash != uncertainty.route_axis_hash
            or owner.route_axis_hash != qualification.route_axis_hash
            or owner.realization_hashes != tuple(registered.realization_hashes)
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "one retained owner radius differs from its registered "
                "role/set/prefix/route source",
            )
    return first


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiveOwnerRadiusSourceReceipt:
    """Exact legacy scalar lineage recovered for provenance, not reach units."""

    source_image_receipt_hash: str
    finite_cuda_qualification_hash: str
    uncertainty_set_hash: str
    branch_evidence_receipt_hashes: tuple[str, str]
    owner_receipts: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]
    source_image: V1SampledV7DSourceImageBindingReceipt = field(
        repr=False, compare=False
    )
    source_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        repr=False, compare=False
    )
    source_uncertainty: V1SignedWorkDecisionUncertaintySet = field(
        repr=False, compare=False
    )
    _issuer_token: InitVar[object]
    _issued_receipt_hash: str = field(init=False, repr=False, compare=False)
    all_owner_radii_live_source_derived: bool = True
    both_branch_owner_copies_identical: bool = True
    source_image_binding_consumed: bool = True
    continuum_margin_authority: bool = False
    d_stability_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_AUTHORITY
    schema: str = SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _OWNER_RADIUS_ISSUER_TOKEN:
            raise TypeError("live owner-radius source receipts are issuer-only")
        for name in (
            "source_image_receipt_hash",
            "finite_cuda_qualification_hash",
            "uncertainty_set_hash",
            "authority_scope",
            "schema",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be one nonempty string")
        owners = tuple(self.owner_receipts)
        replayed = _assert_exact_live_product(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
        )
        if (
            owners != replayed
            or tuple(value.owner_role for value in owners) != _CANONICAL_OWNER_ROLES
            or self.source_image_receipt_hash != self.source_image.receipt_hash
            or self.finite_cuda_qualification_hash
            != self.source_qualification.qualification_hash
            or self.uncertainty_set_hash != self.source_uncertainty.uncertainty_hash
            or self.branch_evidence_receipt_hashes
            != self.source_image.branch_evidence_receipt_hashes
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "owner-radius source does not replay its exact positive source image",
            )
        flags = (
            self.all_owner_radii_live_source_derived,
            self.both_branch_owner_copies_identical,
            self.source_image_binding_consumed,
            self.continuum_margin_authority,
            self.d_stability_authority,
            self.C_measurement_authority,
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
            or self.authority_scope
            != SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_AUTHORITY
            or self.schema != SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_SCHEMA
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                "owner-radius source crossed its custody-only authority",
            )
        object.__setattr__(self, "owner_receipts", owners)
        object.__setattr__(self, "_issued_receipt_hash", self.receipt_hash)

    @property
    def owner_radius_axis(
        self,
    ) -> tuple[tuple[V1SignedWorkDecisionOwnerRole, str, float], ...]:
        return tuple(
            (
                value.owner_role,
                value.receipt_hash,
                float(value.outward_radius_upper_bound),
            )
            for value in self.owner_receipts
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_image_receipt_hash": self.source_image_receipt_hash,
                "finite_cuda_qualification_hash": (self.finite_cuda_qualification_hash),
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "branch_evidence_receipt_hashes": (self.branch_evidence_receipt_hashes),
                "owner_radius_axis": self.owner_radius_axis,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "owner-radius-source-",
        )

    def assert_live_identity(self) -> None:
        owners = _assert_exact_live_product(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
        )
        if (
            owners != self.owner_receipts
            or self.receipt_hash != self._issued_receipt_hash
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "live owner-radius source changed after issuance",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live owner-radius source receipts are nonserializable")


def _issue_live_owner_radius_source(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> V1SignedWorkLiveOwnerRadiusSourceReceipt:
    owners = _assert_exact_live_product(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
    )
    return V1SignedWorkLiveOwnerRadiusSourceReceipt(
        source_image_receipt_hash=source_image.receipt_hash,
        finite_cuda_qualification_hash=qualification.qualification_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        branch_evidence_receipt_hashes=source_image.branch_evidence_receipt_hashes,
        owner_receipts=owners,
        source_image=source_image,
        source_qualification=qualification,
        source_uncertainty=uncertainty,
        _issuer_token=_OWNER_RADIUS_ISSUER_TOKEN,
    )


def issue_v1_signed_work_live_owner_radius_source(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> V1SignedWorkLiveOwnerRadiusSourceReceipt:
    """Issue the authority-inert live owner-lineage custody receipt.

    The complete registered Phase-A alias map needs this exact source parent,
    but obtaining it through a deliberately thrown continuum-refusal object
    made the lawful categorical path depend on exception transport.  This
    public issuer exposes only the already-validated provenance object: it
    does not claim continuum reach, D stability, C authority, or CUDA
    qualification.  The separate continuum issuer remains fail-closed until
    the typed robust variable-shell reach theorem is present.
    """

    return _issue_live_owner_radius_source(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
    )


def _complete_action_image_source_payload(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    action_separation: V1SignedWorkVariableShellActionSeparationCertificate,
) -> dict[str, Any]:
    """Replay the exact conditional action image and categorical quotient.

    The aggregate action-separation owner has already enclosed statistical and
    solver uncertainty in its per-source-group ``L``/``U`` bounds.  This join
    therefore uses its outward ``Gamma_lower`` directly and never subtracts
    the four legacy owner radii a second time.
    """

    owners = _assert_exact_live_product(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
    )
    if (
        type(action_separation)
        is not V1SignedWorkVariableShellActionSeparationCertificate
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "complete-action closure requires the exact aggregate action-cell "
            "separation certificate",
        )
    try:
        action_separation.assert_live_identity()
    except Exception as error:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"aggregate action separation failed live replay: {type(error).__name__}",
        )
    registry = action_separation.registry
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "aggregate action separation lost its exact live registry",
        )
    family = action_separation.source_uncertainty_family
    if type(family) not in {
        V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt,
        V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt,
    }:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "aggregate action separation lost its exact source uncertainty family",
        )
    crosswalk = getattr(source_image, "_complete_crosswalk", None)
    if type(crosswalk) is not V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_PHYSICAL_ACTION_QUOTIENT,
            "sampled-V7 source image does not retain its exact packed-action crosswalk",
        )
    try:
        crosswalk_hash = crosswalk.assert_integrity()
        registry.assert_live_sources()
    except Exception as error:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"action-cell/crosswalk source failed replay: {type(error).__name__}",
        )
    receipt = registry.receipt
    if (
        getattr(source_image, "_phase_a_registry", None) is not registry
        or getattr(crosswalk, "_phase_a_registry", None) is not registry
        or getattr(crosswalk, "_cuda_qualification", None) is not qualification
        or crosswalk.phase_a_registry_receipt_hash != receipt.receipt_hash
        or crosswalk.cuda_qualification_hash != qualification.qualification_hash
        or getattr(source_image, "complete_action_crosswalk_receipt_hash", None)
        != crosswalk_hash
        or getattr(source_image, "phase_a_registry_receipt_hash", None)
        != receipt.receipt_hash
        or getattr(family, "registry_receipt_hash", None) != receipt.receipt_hash
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            "source image, aggregate separation, uncertainty family, crosswalk, "
            "and qualification are not one exact live graph",
        )
    if (
        receipt.complete_branch_axis is not True
        or receipt.complete_action_cell_coverage is not True
        or receipt.exact_ties_registered_in_every_compatible_cell is not True
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_ACTION_CELL_COVER,
            "closed top-k cells do not cover the complete branch/order continuum "
            "with every exact tie in every compatible cell",
        )
    cells = tuple(registry.iter_action_cells())
    if (
        not cells
        or any(
            type(value) is not V1SignedWorkVariableShellActionCellIdentity
            for value in cells
        )
        or len({value.action_cell_hash for value in cells}) != len(cells)
        or len(cells) != receipt.total_action_cell_count
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_ACTION_CELL_COVER,
            "live action-cell image is missing, duplicated, or differs from its registry",
        )
    preferred = action_separation.preferred_physical_action_coordinate_hash
    preferred_group_uppers = tuple(
        action_separation.preferred_source_group_F_upper_bounds
    )
    preferred_upper = float(
        action_separation.robust_preferred_feasible_primal_F_upper_bound
    )
    alternate_lower = float(action_separation.minimum_alternate_dual_F_lower_bound)
    declared_gamma = float(action_separation.strict_separation_margin_lower_bound)
    if (
        not preferred_group_uppers
        or any(
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not str
            or not value[0]
            or not math.isfinite(float(value[1]))
            for value in preferred_group_uppers
        )
        or not all(
            math.isfinite(value)
            for value in (preferred_upper, alternate_lower, declared_gamma)
        )
        or preferred_upper
        != max(float(value) for _group, value in preferred_group_uppers)
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.NONUNIQUE_ACTION_IMAGE_WINNER,
            "aggregate source-group F bounds are absent, nonfinite, or inconsistent",
        )
    outward_gamma = math.nextafter(alternate_lower - preferred_upper, -math.inf)
    if declared_gamma <= 0.0 or declared_gamma > outward_gamma:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.NONUNIQUE_ACTION_IMAGE_WINNER,
            "aggregate Gamma_lower does not strictly separate one physical action",
        )

    cell_by_hash = {value.action_cell_hash: value for value in cells}
    cell_coordinate_axis = {
        (value.action_cell_hash, value.physical_action_coordinate_hash)
        for value in cells
    }
    row_cell_axis = {
        (value.phase_a_action_cell_hash, value.phase_a_action_coordinate_hash)
        for value in crosswalk.structural_rows
    }
    if cell_coordinate_axis != row_cell_axis:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_PHYSICAL_ACTION_QUOTIENT,
            "the exact packed crosswalk is not set-equal to the complete action-cell image",
        )
    coordinate_to_identity: dict[str, set[tuple[object, ...]]] = {}
    coordinate_to_packed: dict[str, set[str]] = {}
    for row in crosswalk.structural_rows:
        cell = cell_by_hash[row.phase_a_action_cell_hash]
        if row.phase_a_action_coordinate_hash != cell.physical_action_coordinate_hash:
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_PHYSICAL_ACTION_QUOTIENT,
                "one packed crosswalk row names another action coordinate",
            )
        coordinate_to_identity.setdefault(
            row.phase_a_action_coordinate_hash, set()
        ).add(tuple(row.packed_dedup_identity))
        coordinate_to_packed.setdefault(row.phase_a_action_coordinate_hash, set()).add(
            row.packed_action_hash
        )
    if (
        any(len(value) != 1 for value in coordinate_to_identity.values())
        or any(len(value) != 1 for value in coordinate_to_packed.values())
        or preferred not in coordinate_to_identity
        or len(coordinate_to_identity) < 2
        or crosswalk.structural_action_coordinate_packed_crosswalk_complete is not True
        or crosswalk.all_action_origin_q_projection_complete is not True
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_PHYSICAL_ACTION_QUOTIENT,
            "action coordinates do not map exactly and totally to packed byte categories",
        )
    identity_to_coordinates: dict[tuple[object, ...], set[str]] = {}
    for coordinate, identities in coordinate_to_identity.items():
        identity = next(iter(identities))
        identity_to_coordinates.setdefault(identity, set()).add(coordinate)
    quotient_axis = tuple(
        sorted(
            (
                content_hash(identity, prefix=_PREFIX + "packed-category-"),
                tuple(sorted(coordinates)),
            )
            for identity, coordinates in identity_to_coordinates.items()
        )
    )
    preferred_packed_actions = tuple(sorted(coordinate_to_packed[preferred]))
    if len(preferred_packed_actions) != 1:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_PHYSICAL_ACTION_QUOTIENT,
            "preferred action coordinate has no unique packed child identity",
        )
    return {
        "owners": owners,
        "registry": registry,
        "uncertainty_family": family,
        "crosswalk": crosswalk,
        "action_separation": action_separation,
        "preferred_physical_action_coordinate_hash": preferred,
        "preferred_packed_action_hash": preferred_packed_actions[0],
        "preferred_source_group_F_upper_bounds": preferred_group_uppers,
        "robust_preferred_F_upper_bound": preferred_upper,
        "minimum_alternate_F_lower_bound": alternate_lower,
        "aggregate_gamma_lower_bound": declared_gamma,
        "action_cell_count": len(cells),
        "action_coordinate_count": len(coordinate_to_identity),
        "physical_action_category_count": len(identity_to_coordinates),
        "categorical_quotient_axis": quotient_axis,
        "categorical_quotient_hash": content_hash(
            quotient_axis, prefix=_PREFIX + "categorical-quotient-"
        ),
    }


@dataclass(frozen=True, slots=True)
class V1SignedWorkCompleteActionImageCustodyReceipt:
    """Exact within-shell action image and packed categorical quotient.

    This is a prerequisite/custody receipt, not global D stability: a separate
    full-U or outside-shell proof must still join before canonical closure.
    """

    source_image_receipt_hash: str
    finite_cuda_qualification_hash: str
    uncertainty_set_hash: str
    action_separation_certificate_hash: str
    source_uncertainty_family_receipt_hash: str
    action_cell_registry_receipt_hash: str
    complete_packed_crosswalk_receipt_hash: str
    preferred_physical_action_coordinate_hash: str
    preferred_packed_action_hash: str
    preferred_source_group_F_upper_bounds: tuple[tuple[str, float], ...]
    robust_preferred_F_upper_bound: float
    minimum_alternate_F_lower_bound: float
    aggregate_gamma_lower_bound: float
    action_cell_count: int
    action_coordinate_count: int
    physical_action_category_count: int
    categorical_quotient_axis: tuple[tuple[str, tuple[str, ...]], ...]
    categorical_quotient_hash: str
    source_image: V1SampledV7DSourceImageBindingReceipt = field(
        repr=False, compare=False
    )
    source_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        repr=False, compare=False
    )
    source_uncertainty: V1SignedWorkDecisionUncertaintySet = field(
        repr=False, compare=False
    )
    source_action_separation: V1SignedWorkVariableShellActionSeparationCertificate = (
        field(repr=False, compare=False)
    )
    _issuer_token: InitVar[object]
    _issued_receipt_hash: str = field(init=False, repr=False, compare=False)
    closure_mode: V1SignedWorkDecisionContinuumClosureMode = (
        V1SignedWorkDecisionContinuumClosureMode.COMPLETE_ACTION_IMAGE
    )
    complete_closed_top_k_action_cell_cover: bool = True
    exact_ties_in_every_compatible_cell: bool = True
    phase_b_winner_from_aggregate_action_cell_separation: bool = True
    hard_rounding_order_closed_without_positive_scalar_margin: bool = True
    packed_bytes_are_exact_categorical_quotient: bool = True
    packed_byte_positive_scalar_margin_required: bool = False
    sampling_and_numerical_uncertainty_already_enclosed_in_L_U: bool = True
    additional_owner_radius_subtraction_applied: bool = False
    within_frozen_shell_conditional_only: bool = True
    global_D_continuum_closure_authority: bool = False
    d_stability_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_AUTHORITY
    closure_rule: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE
    schema: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_IMAGE_CUSTODY_ISSUER_TOKEN:
            raise TypeError("complete-action image custody receipts are issuer-only")
        replay = _complete_action_image_source_payload(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
            action_separation=self.source_action_separation,
        )
        expected = {
            "source_image_receipt_hash": self.source_image.receipt_hash,
            "finite_cuda_qualification_hash": self.source_qualification.qualification_hash,
            "uncertainty_set_hash": self.source_uncertainty.uncertainty_hash,
            "action_separation_certificate_hash": self.source_action_separation.certificate_hash,
            "source_uncertainty_family_receipt_hash": replay[
                "uncertainty_family"
            ].receipt_hash,
            "action_cell_registry_receipt_hash": replay[
                "registry"
            ].receipt.receipt_hash,
            "complete_packed_crosswalk_receipt_hash": replay["crosswalk"].receipt_hash,
            "preferred_physical_action_coordinate_hash": replay[
                "preferred_physical_action_coordinate_hash"
            ],
            "preferred_packed_action_hash": replay["preferred_packed_action_hash"],
            "preferred_source_group_F_upper_bounds": replay[
                "preferred_source_group_F_upper_bounds"
            ],
            "robust_preferred_F_upper_bound": replay["robust_preferred_F_upper_bound"],
            "minimum_alternate_F_lower_bound": replay[
                "minimum_alternate_F_lower_bound"
            ],
            "aggregate_gamma_lower_bound": replay["aggregate_gamma_lower_bound"],
            "action_cell_count": replay["action_cell_count"],
            "action_coordinate_count": replay["action_coordinate_count"],
            "physical_action_category_count": replay["physical_action_category_count"],
            "categorical_quotient_axis": replay["categorical_quotient_axis"],
            "categorical_quotient_hash": replay["categorical_quotient_hash"],
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "complete-action custody public image differs from its exact sources",
            )
        flags = (
            self.complete_closed_top_k_action_cell_cover,
            self.exact_ties_in_every_compatible_cell,
            self.phase_b_winner_from_aggregate_action_cell_separation,
            self.hard_rounding_order_closed_without_positive_scalar_margin,
            self.packed_bytes_are_exact_categorical_quotient,
            self.packed_byte_positive_scalar_margin_required,
            self.sampling_and_numerical_uncertainty_already_enclosed_in_L_U,
            self.additional_owner_radius_subtraction_applied,
            self.within_frozen_shell_conditional_only,
            self.global_D_continuum_closure_authority,
            self.d_stability_authority,
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
            True,
            True,
            True,
            False,
            True,
            False,
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
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                "complete-action custody crossed its conditional authority",
            )
        if (
            self.closure_mode
            is not V1SignedWorkDecisionContinuumClosureMode.COMPLETE_ACTION_IMAGE
            or self.authority_scope
            != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_AUTHORITY
            or self.closure_rule != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE
            or self.schema != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_SCHEMA
        ):
            raise ValueError("unsupported complete-action image custody semantics")
        object.__setattr__(self, "_issued_receipt_hash", self.receipt_hash)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_") and not name.startswith("source_")
            }
            | {
                "source_image_receipt_hash": self.source_image_receipt_hash,
                "source_uncertainty_family_receipt_hash": self.source_uncertainty_family_receipt_hash,
            },
            prefix=_PREFIX + "complete-action-custody-",
        )

    def assert_live_identity(self) -> None:
        replay = _complete_action_image_source_payload(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
            action_separation=self.source_action_separation,
        )
        if (
            replay["categorical_quotient_hash"] != self.categorical_quotient_hash
            or replay["aggregate_gamma_lower_bound"] != self.aggregate_gamma_lower_bound
            or self.receipt_hash != self._issued_receipt_hash
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "complete-action custody changed after issuance",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("complete-action image custody receipts are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("complete-action image custody receipts cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("complete-action image custody receipts cannot be copied")


def issue_v1_signed_work_complete_action_image_custody(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    action_separation: V1SignedWorkVariableShellActionSeparationCertificate,
) -> V1SignedWorkCompleteActionImageCustodyReceipt:
    """Issue exact conditional custody without accepting scalar/hash claims."""

    replay = _complete_action_image_source_payload(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
        action_separation=action_separation,
    )
    return V1SignedWorkCompleteActionImageCustodyReceipt(
        source_image_receipt_hash=source_image.receipt_hash,
        finite_cuda_qualification_hash=qualification.qualification_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        action_separation_certificate_hash=action_separation.certificate_hash,
        source_uncertainty_family_receipt_hash=replay[
            "uncertainty_family"
        ].receipt_hash,
        action_cell_registry_receipt_hash=replay["registry"].receipt.receipt_hash,
        complete_packed_crosswalk_receipt_hash=replay["crosswalk"].receipt_hash,
        preferred_physical_action_coordinate_hash=replay[
            "preferred_physical_action_coordinate_hash"
        ],
        preferred_packed_action_hash=replay["preferred_packed_action_hash"],
        preferred_source_group_F_upper_bounds=replay[
            "preferred_source_group_F_upper_bounds"
        ],
        robust_preferred_F_upper_bound=replay["robust_preferred_F_upper_bound"],
        minimum_alternate_F_lower_bound=replay["minimum_alternate_F_lower_bound"],
        aggregate_gamma_lower_bound=replay["aggregate_gamma_lower_bound"],
        action_cell_count=replay["action_cell_count"],
        action_coordinate_count=replay["action_coordinate_count"],
        physical_action_category_count=replay["physical_action_category_count"],
        categorical_quotient_axis=replay["categorical_quotient_axis"],
        categorical_quotient_hash=replay["categorical_quotient_hash"],
        source_image=source_image,
        source_qualification=qualification,
        source_uncertainty=uncertainty,
        source_action_separation=action_separation,
        _issuer_token=_ACTION_IMAGE_CUSTODY_ISSUER_TOKEN,
    )


def _assert_complete_action_image_full_u_join(
    *,
    custody: V1SignedWorkCompleteActionImageCustodyReceipt,
    full_u_f_q_action_invariance: V1SignedWorkFullUFQProductionActionInvarianceReceiptV3,
) -> None:
    custody.assert_live_identity()
    if (
        type(full_u_f_q_action_invariance)
        is not V1SignedWorkFullUFQProductionActionInvarianceReceiptV3
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INVALID_TYPE,
            "global complete-action closure requires the exact production full-U F/Q receipt",
            landed_complete_action_image_custody=custody,
        )
    try:
        full_u_f_q_action_invariance.assert_integrity()
    except Exception as error:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"production full-U F/Q receipt failed replay: {type(error).__name__}",
            landed_complete_action_image_custody=custody,
        )
    full_u = full_u_f_q_action_invariance
    registry = custody.source_action_separation.registry
    receipt = registry.receipt
    preferred_category = tuple(
        coordinates
        for _category, coordinates in custody.categorical_quotient_axis
        if custody.preferred_physical_action_coordinate_hash in coordinates
    )
    full_u_coordinate_packed_actions = {
        row.packed_action_hash
        for row in custody.source_image._complete_crosswalk.structural_rows
        if row.phase_a_action_coordinate_hash
        == full_u_f_q_action_invariance.physical_action_coordinate_hash
    }
    required_identity = (
        full_u.parent_mask_hash == custody.source_image.parent_mask_hash,
        full_u.parent_mask_hash == receipt.current_mask_hash,
        full_u.chart_identity_hash == receipt.chart_identity_hash,
        full_u.proposal_generation == receipt.proposal_generation,
        full_u.route_axis_hash == custody.source_image.route_axis_hash,
        full_u.route_axis_hash == receipt.route_axis_hash,
        full_u.route_count == custody.source_image.route_count,
        full_u.route_id_axis_hash == receipt.route_id_axis_hash,
        full_u.registered_d_prefix_hash == receipt.registered_d_prefix_hash,
        full_u.registered_d_w2_prefix_hash == receipt.registered_d_w2_prefix_hash,
        len(preferred_category) == 1,
        len(preferred_category) == 1
        and full_u.physical_action_coordinate_hash in preferred_category[0],
        full_u.packed_action_hash in full_u_coordinate_packed_actions,
        full_u.alternate_action_count == custody.physical_action_category_count - 1,
    )
    if not all(required_identity):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_GLOBAL_ACTION_CLOSURE,
            "production full-U F/Q witness names another chart, source prefix, "
            "preferred quotient class, or complete physical-action domain",
            missing_requirements=REQUIRED_LIVE_MARGIN_PRODUCER_APIS,
            live_source_identity_verified=True,
            landed_complete_action_image_custody=custody,
        )
    required_true = (
        full_u.source_bound_full_population_corrected_F_separation,
        full_u.strict_origin_branch_registered_Q_safety,
        full_u.literal_J_same_capture_response_geometry,
        full_u.complete_two_branch_hard_domain_replayed,
        full_u.deterministic_FD_residual_decision_stable,
        full_u.production_D_action_invariance_authority,
        full_u.array_free_receipt,
    )
    required_false = (
        full_u.production_C_stream_open_authority,
        full_u.calibration_K_open_authority,
        full_u.paired_oracle_O_open_authority,
        full_u.mask_commit_authority,
        full_u.terminal_authority,
        full_u.performance_authority,
        full_u.pareto_authority,
    )
    if any(
        type(value) is not bool or value is not True for value in required_true
    ) or any(type(value) is not bool or value is not False for value in required_false):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
            "production full-U F/Q witness lacks D closure or opened downstream authority",
            landed_complete_action_image_custody=custody,
        )
    try:
        handles = registry.assert_live_sources()
        corrected_sources = {
            handle.receipt.corrected_evidence_stream_source_hash for handle in handles
        }
        literal_j_sources = {
            handle.receipt.literal_j_stream_source_hash for handle in handles
        }
    except Exception as error:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
            f"registry stream custody failed replay: {type(error).__name__}",
            landed_complete_action_image_custody=custody,
        )
    if (
        not corrected_sources
        or not literal_j_sources
        or full_u.corrected_a_stream_identity_hash not in corrected_sources
        or full_u.literal_j_stream_identity_hash not in literal_j_sources
    ):
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_GLOBAL_ACTION_CLOSURE,
            "production full-U receipt is not bound to the registry's exact A/J streams",
            missing_requirements=REQUIRED_LIVE_MARGIN_PRODUCER_APIS,
            live_source_identity_verified=True,
            landed_complete_action_image_custody=custody,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiveCUDACompleteActionImageClosureReceipt:
    """Reserved positive shape pending an exact Phase-A reach certificate.

    The current public issuer cannot construct this type.  Complete finite
    action-image custody plus full-U hard-action invariance does not by itself
    prove continuum reach between registered Phase-A source images.
    """

    complete_action_image_custody_receipt_hash: str
    production_full_u_f_q_action_invariance_receipt_hash: str
    source_image_receipt_hash: str
    finite_cuda_qualification_hash: str
    uncertainty_set_hash: str
    source_uncertainty_family_receipt_hash: str
    preferred_physical_action_coordinate_hash: str
    preferred_packed_action_hash: str
    aggregate_gamma_lower_bound: float
    categorical_quotient_hash: str
    complete_action_image_custody: V1SignedWorkCompleteActionImageCustodyReceipt = (
        field(repr=False, compare=False)
    )
    production_full_u_f_q_action_invariance: (
        V1SignedWorkFullUFQProductionActionInvarianceReceiptV3
    ) = field(repr=False, compare=False)
    _issuer_token: InitVar[object]
    _issued_receipt_hash: str = field(init=False, repr=False, compare=False)
    closure_mode: V1SignedWorkDecisionContinuumClosureMode = (
        V1SignedWorkDecisionContinuumClosureMode.COMPLETE_ACTION_IMAGE
    )
    complete_action_cell_image_continuum_closed: bool = True
    aggregate_phase_b_action_separation_closed: bool = True
    exact_categorical_physical_action_quotient_closed: bool = True
    outside_shell_and_full_u_f_q_invariance_closed: bool = True
    no_double_subtraction_of_enclosed_uncertainty: bool = True
    global_D_continuum_closure_authority: bool = True
    d_stability_authority: bool = True
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_AUTHORITY
    closure_rule: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE
    schema: str = SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_IMAGE_CERTIFICATE_ISSUER_TOKEN:
            raise TypeError("complete-action continuum certificates are issuer-only")
        if (
            type(self.complete_action_image_custody)
            is not V1SignedWorkCompleteActionImageCustodyReceipt
            or type(self.production_full_u_f_q_action_invariance)
            is not V1SignedWorkFullUFQProductionActionInvarianceReceiptV3
        ):
            raise TypeError(
                "complete-action certificate requires exact source receipts"
            )
        custody = self.complete_action_image_custody
        full_u = self.production_full_u_f_q_action_invariance
        _assert_complete_action_image_full_u_join(
            custody=custody,
            full_u_f_q_action_invariance=full_u,
        )
        expected = (
            self.complete_action_image_custody_receipt_hash == custody.receipt_hash,
            self.production_full_u_f_q_action_invariance_receipt_hash
            == full_u.receipt_hash,
            self.source_image_receipt_hash == custody.source_image_receipt_hash,
            self.finite_cuda_qualification_hash
            == custody.finite_cuda_qualification_hash,
            self.uncertainty_set_hash == custody.uncertainty_set_hash,
            self.source_uncertainty_family_receipt_hash
            == custody.source_uncertainty_family_receipt_hash,
            self.preferred_physical_action_coordinate_hash
            == custody.preferred_physical_action_coordinate_hash,
            self.preferred_packed_action_hash == full_u.packed_action_hash,
            self.aggregate_gamma_lower_bound == custody.aggregate_gamma_lower_bound,
            self.categorical_quotient_hash == custody.categorical_quotient_hash,
        )
        if not all(expected):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "complete-action certificate public image differs from its sources",
            )
        flags = (
            self.complete_action_cell_image_continuum_closed,
            self.aggregate_phase_b_action_separation_closed,
            self.exact_categorical_physical_action_quotient_closed,
            self.outside_shell_and_full_u_f_q_invariance_closed,
            self.no_double_subtraction_of_enclosed_uncertainty,
            self.global_D_continuum_closure_authority,
            self.d_stability_authority,
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
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                "complete-action certificate crossed its D-only authority",
            )
        if (
            self.closure_mode
            is not V1SignedWorkDecisionContinuumClosureMode.COMPLETE_ACTION_IMAGE
            or self.authority_scope
            != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_AUTHORITY
            or self.closure_rule != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CLOSURE_RULE
            or self.schema != SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported complete-action continuum semantics")
        object.__setattr__(self, "_issued_receipt_hash", self.receipt_hash)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
                and name
                not in {
                    "complete_action_image_custody",
                    "production_full_u_f_q_action_invariance",
                }
            },
            prefix=_PREFIX + "complete-action-certificate-",
        )

    def assert_live_identity(self) -> None:
        _assert_complete_action_image_full_u_join(
            custody=self.complete_action_image_custody,
            full_u_f_q_action_invariance=(self.production_full_u_f_q_action_invariance),
        )
        if self.receipt_hash != self._issued_receipt_hash:
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "complete-action continuum certificate changed after issuance",
            )

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("complete-action continuum certificates are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("complete-action continuum certificates cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        del memo
        raise TypeError("complete-action continuum certificates cannot be copied")


def issue_v1_signed_work_live_cuda_complete_action_image_closure(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    action_separation: V1SignedWorkVariableShellActionSeparationCertificate,
    full_u_f_q_action_invariance: V1SignedWorkFullUFQProductionActionInvarianceReceiptV3,
) -> NoReturn:
    """Retain exact finite-action custody, then refuse absent continuum reach.

    The aggregate Phase-B separation, tie-complete action-cell image, exact
    packed categorical quotient, and production full-U F/Q hard-action replay
    are necessary premises.  They cannot impersonate the missing endogenous
    Phase-A typed-source-to-branch-endpoint reach certificate.
    """

    custody = issue_v1_signed_work_complete_action_image_custody(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
        action_separation=action_separation,
    )
    if qualification.production_cuda_sources_verified is not True:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.CUDA_REQUIRED,
            "canonical complete-action closure requires verified production CUDA sources",
            live_source_identity_verified=True,
            landed_complete_action_image_custody=custody,
        )
    _assert_complete_action_image_full_u_join(
        custody=custody,
        full_u_f_q_action_invariance=full_u_f_q_action_invariance,
    )
    _refuse(
        V1SignedWorkDecisionMarginSourceRefusalCode.INCOMPLETE_GLOBAL_ACTION_CLOSURE,
        "complete registered action-image and full-U F/Q hard-action invariance "
        "do not prove continuum reach across Phase-A resource-cell exits or "
        "between registered source images",
        missing_requirements=REQUIRED_LIVE_MARGIN_PRODUCER_APIS,
        live_source_identity_verified=True,
        landed_complete_action_image_custody=custody,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkLiveCUDADecisionMarginFamilyReceipt:
    """Issuer-only positive receipt reserved for complete live derivation.

    No current public path can construct this receipt: the producer APIs listed
    in :data:`REQUIRED_LIVE_MARGIN_PRODUCER_APIS` must first land.  Keeping the
    positive type here freezes the narrow authority and replay contract without
    pretending that the missing numerical proof exists.
    """

    source_image_receipt_hash: str
    finite_cuda_qualification_hash: str
    uncertainty_set_hash: str
    owner_receipts: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]
    margin_receipts: tuple[V1SignedWorkDecisionMarginReceipt, ...]
    source_derivation_hash: str
    outward_enclosure_policy_hash: str
    source_image: V1SampledV7DSourceImageBindingReceipt = field(
        repr=False,
        compare=False,
    )
    source_qualification: V1SignedWorkCUDADecisionWitnessQualification = field(
        repr=False,
        compare=False,
    )
    source_uncertainty: V1SignedWorkDecisionUncertaintySet = field(
        repr=False,
        compare=False,
    )
    _issuer_token: InitVar[object]
    _issued_receipt_hash: str = field(init=False, repr=False, compare=False)
    all_owner_radii_live_source_derived: bool = True
    all_decision_margins_live_source_derived: bool = True
    bounded_outward_fp64_reductions: bool = True
    route_scale_d2h_bytes: int = 0
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_AUTHORITY_SCOPE
    derivation_policy: str = SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_POLICY
    schema: str = SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_ISSUER_TOKEN:
            raise TypeError("live CUDA decision-margin receipts are issuer-only")
        if type(self.source_image) is not V1SampledV7DSourceImageBindingReceipt:
            raise TypeError("live margin receipt requires its exact source image")
        if (
            type(self.source_qualification)
            is not V1SignedWorkCUDADecisionWitnessQualification
        ):
            raise TypeError("live margin receipt requires its exact qualification")
        if type(self.source_uncertainty) is not V1SignedWorkDecisionUncertaintySet:
            raise TypeError("live margin receipt requires its exact uncertainty set")
        for name in (
            "source_image_receipt_hash",
            "finite_cuda_qualification_hash",
            "uncertainty_set_hash",
            "source_derivation_hash",
            "outward_enclosure_policy_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError(f"{name} must be one nonempty string")
        owners = tuple(self.owner_receipts)
        margins = tuple(self.margin_receipts)
        if len(owners) != 4 or len(margins) != len(V1SignedWorkDecisionMarginRole):
            raise TypeError("live margin receipt requires four owners and five margins")
        if any(
            type(value) is not V1SignedWorkDecisionOwnerUncertaintyReceipt
            for value in owners
        ):
            raise TypeError("live margin receipt owner family is not exact")
        if any(
            type(value) is not V1SignedWorkDecisionMarginReceipt for value in margins
        ):
            raise TypeError("live margin receipt margin family is not exact")
        if tuple(value.margin_role for value in margins) != tuple(
            V1SignedWorkDecisionMarginRole
        ):
            raise ValueError("live margin receipt has a noncanonical margin order")
        object.__setattr__(self, "owner_receipts", owners)
        object.__setattr__(self, "margin_receipts", margins)
        flags = (
            self.all_owner_radii_live_source_derived,
            self.all_decision_margins_live_source_derived,
            self.bounded_outward_fp64_reductions,
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
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                "live margin receipt crossed its narrow authority boundary",
            )
        if self.route_scale_d2h_bytes != 0:
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.AUTHORITY_VIOLATION,
                "live margin receipt crossed the bounded device-native seam",
            )
        if (
            self.finite_cuda_qualification_hash
            != self.source_qualification.qualification_hash
            or self.uncertainty_set_hash != self.source_uncertainty.uncertainty_hash
            or self.authority_scope
            != SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_AUTHORITY_SCOPE
            or self.derivation_policy != SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_POLICY
            or self.schema != SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_SCHEMA
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "live margin receipt source identity does not replay",
            )
        replayed_owners = _assert_exact_live_product(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
        )
        if (
            replayed_owners != owners
            or self.source_image_receipt_hash != self.source_image.receipt_hash
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "live margin owner radii differ from the positive source image",
            )
        object.__setattr__(self, "_issued_receipt_hash", self.receipt_hash)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_image_receipt_hash": self.source_image_receipt_hash,
                "finite_cuda_qualification_hash": self.finite_cuda_qualification_hash,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "owner_receipt_hashes": tuple(
                    value.receipt_hash for value in self.owner_receipts
                ),
                "margin_receipt_hashes": tuple(
                    value.receipt_hash for value in self.margin_receipts
                ),
                "source_derivation_hash": self.source_derivation_hash,
                "outward_enclosure_policy_hash": self.outward_enclosure_policy_hash,
                "bounded_outward_fp64_reductions": True,
                "route_scale_d2h_bytes": 0,
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
                "derivation_policy": self.derivation_policy,
            },
            prefix=_PREFIX + "receipt-",
        )

    def assert_live_identity(self) -> None:
        owners = _assert_exact_live_product(
            source_image=self.source_image,
            qualification=self.source_qualification,
            uncertainty=self.source_uncertainty,
        )
        if (
            owners != self.owner_receipts
            or self.receipt_hash != self._issued_receipt_hash
        ):
            _refuse(
                V1SignedWorkDecisionMarginSourceRefusalCode.STALE_IDENTITY,
                "live CUDA decision-margin receipt changed after issuance",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live CUDA decision-margin receipts are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live CUDA decision-margin receipts are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live CUDA decision-margin receipts cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("live CUDA decision-margin receipts cannot be copied")


def issue_v1_signed_work_live_cuda_decision_margin_family(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> V1SignedWorkLiveCUDADecisionMarginFamilyReceipt:
    """Retain the legacy five-margin live-source reader and refuse promotion.

    New production code uses
    :func:`issue_v1_signed_work_live_cuda_complete_action_image_closure`.
    This deliberately narrow historical seam still verifies exact ancestry,
    then names the missing endogenous Phase-A reach producer instead of asking
    a KKT handle, finite sampled vertices, or packed bytes to manufacture a
    continuum margin.
    """

    owner_radius_source = _issue_live_owner_radius_source(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
    )
    if qualification.production_cuda_sources_verified is not True:
        _refuse(
            V1SignedWorkDecisionMarginSourceRefusalCode.CUDA_REQUIRED,
            "public live margin issuance requires verified production CUDA sources",
            live_source_identity_verified=True,
            landed_owner_radius_source=owner_radius_source,
        )
    _refuse_missing_live_margin_producer_apis(owner_radius_source)


def _probe_v1_signed_work_live_decision_margin_capability_cpu_reference(
    *,
    source_image: V1SampledV7DSourceImageBindingReceipt,
    qualification: V1SignedWorkCUDADecisionWitnessQualification,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
) -> NoReturn:
    """Private refusal-only CPU probe for local source-contract coverage."""

    owner_radius_source = _issue_live_owner_radius_source(
        source_image=source_image,
        qualification=qualification,
        uncertainty=uncertainty,
    )
    _refuse_missing_live_margin_producer_apis(owner_radius_source)


def _refuse_missing_live_margin_producer_apis(
    owner_radius_source: V1SignedWorkLiveOwnerRadiusSourceReceipt,
) -> NoReturn:
    """Emit the one source-audited capability refusal after all live checks."""

    _refuse(
        V1SignedWorkDecisionMarginSourceRefusalCode.MISSING_PRODUCER_API,
        "the legacy four-owner scalar lineage is source-image bound as provenance, "
        "but canonical closure requires the typed A/F, J, FD, and solver sources "
        "and a self-consistent B<=8 robust variable-shell re-solve (or full KKT "
        "strong-regularity theorem) mapping them to every decision-changing "
        "Phase-A boundary; no fixed-z M_J z, naked KKT, finite-vertex equality, "
        "rounding-key, or byte-boundary scalar may substitute",
        missing_requirements=REQUIRED_LIVE_MARGIN_PRODUCER_APIS,
        live_source_identity_verified=True,
        landed_owner_radius_source=owner_radius_source,
    )


__all__ = [
    "REQUIRED_LIVE_MARGIN_PRODUCER_APIS",
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_AUTHORITY",
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CERTIFICATE_SCHEMA",
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_AUTHORITY",
    "SIGNED_WORK_D_W_COMPLETE_ACTION_IMAGE_CUSTODY_SCHEMA",
    "SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_AUTHORITY_SCOPE",
    "SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_POLICY",
    "SIGNED_WORK_D_W_LIVE_MARGIN_SOURCE_SCHEMA",
    "SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_AUTHORITY",
    "SIGNED_WORK_D_W_LIVE_OWNER_RADIUS_SOURCE_SCHEMA",
    "V1SignedWorkDecisionMarginProducerRequirement",
    "V1SignedWorkDecisionMarginSourceRefusal",
    "V1SignedWorkDecisionMarginSourceRefusalCode",
    "V1SignedWorkCompleteActionImageCustodyReceipt",
    "V1SignedWorkLiveCUDACompleteActionImageClosureReceipt",
    "V1SignedWorkLiveCUDADecisionMarginFamilyReceipt",
    "V1SignedWorkLiveOwnerRadiusSourceReceipt",
    "issue_v1_signed_work_live_owner_radius_source",
    "issue_v1_signed_work_live_cuda_decision_margin_family",
    "issue_v1_signed_work_complete_action_image_custody",
    "issue_v1_signed_work_live_cuda_complete_action_image_closure",
]
