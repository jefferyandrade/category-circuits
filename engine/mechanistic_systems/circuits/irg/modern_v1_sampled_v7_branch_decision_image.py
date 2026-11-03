"""Complete, byte-exact decision image for one sampled-V7 Q branch.

This module is a conservative proof join.  It never infers completeness from
one nominal fixed-primal result.  A certificate requires:

* one full-population same-capture corrected-F/literal-J source;
* all four source-derived uncertainty-owner enclosures (F, J, FD/rank, and
  solver/rounding numerics);
* an independently issued strict exclusion of every retained coordinate
  outside the frozen branch shell; and
* every vertex of the registered four-owner uncertainty product.

Within each uncertainty vertex the issuer enumerates the *entire* frozen-shell
powerset itself.  This deliberately conservative structural cover is a proof,
not a claim about which solver face is likely: once outside coordinates are
excluded, any hard endpoint the numerical solver can reach is necessarily in
that powerset.  Actual packed child bytes are retained and compared before
physical deduplication; cell and branch aliases remain explicit.

The current public production seam has no typed device-side outside-coordinate
scan receipt.  It therefore fails closed.  A bounded private reference issuer
exists solely to exercise the theorem on small domains by scanning every
outside-coordinate margin value.  A successful certificate grants only
one-branch action-image completeness.  It grants no cross-branch D stability,
C, K, O, commit, terminal, performance, or Pareto authority.

The reference cells retain one packed mask per assignment and uncertainty
vertex.  That is bounded theorem evidence, not a route-scale storage design.
The production successor must retain compact shell-assignment labels per cell
and one shared packed payload per deduplicated physical action.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
from itertools import product
import math
from numbers import Integral, Real
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceReceipt,
)
from .modern_v1_sampled_variable_shell_packet_registration import (
    SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
)
from .modern_v1_signed_work_decision_enclosure import (
    V1SignedWorkDecisionOwnerRole,
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
)
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkPackedActionDedupIdentity,
    v1_signed_work_packed_action_dedup_identity,
)


SAMPLED_V7_BRANCH_DECISION_IMAGE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_complete_decision_image_v1"
)
SAMPLED_V7_BRANCH_DECISION_CELL_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_structural_decision_cell_v1"
)
SAMPLED_V7_BRANCH_OUTSIDE_EXCLUSION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_outside_coordinate_exclusion_v1"
)
SAMPLED_V7_BRANCH_PHYSICAL_ACTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_decision_image_physical_action_v1"
)
SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_decision_image_action_alias_v1"
)
SAMPLED_V7_BRANCH_DECISION_IMAGE_AUTHORITY: Final = (
    "one_branch_source_bound_complete_uncertainty_product_structural_shell_"
    "powerset_and_exact_packed_action_image_only_no_cross_branch_D_stability_"
    "C_K_O_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_V7_BRANCH_DECISION_IMAGE_REFERENCE_MEMORY_LIMITATION: Final = (
    "private_reference_repeats_packed_masks_per_uncertainty_cell_and_is_not_"
    "a_route_scale_storage_design_v1"
)
SAMPLED_V7_BRANCH_DECISION_IMAGE_PRODUCTION_STORAGE_REQUIREMENT: Final = (
    "compact_shell_assignment_labels_per_cell_plus_one_shared_packed_payload_"
    "per_byte_deduplicated_physical_action_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-branch-decision-image-"
_REFERENCE_OUTSIDE_TOKEN: Final = object()
_CELL_TOKEN: Final = object()
_ALIAS_TOKEN: Final = object()
_PHYSICAL_TOKEN: Final = object()
_CERTIFICATE_TOKEN: Final = object()
_MAXIMUM_REFERENCE_ROUTE_COUNT: Final = 4_096
_MAXIMUM_STRUCTURAL_SHELL_SIZE: Final = 8
_OWNER_ROLES: Final = (
    V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING,
    V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING,
    V1SignedWorkDecisionOwnerRole.FD_RESIDUAL,
    V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING,
)


class V1SampledV7BranchDecisionImageRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    BRANCH_MISMATCH = "branch_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    OWNER_ENCLOSURE_INCOMPLETE = "owner_enclosure_incomplete"
    OUTSIDE_EXCLUSION_INCOMPLETE = "outside_exclusion_incomplete"
    PRODUCTION_OUTSIDE_PROOF_UNAVAILABLE = "production_outside_proof_unavailable"
    UNCERTAINTY_PRODUCT_INCOMPLETE = "uncertainty_product_incomplete"
    CELL_COVERAGE_INCOMPLETE = "cell_coverage_incomplete"
    PACKED_ACTION_MISMATCH = "packed_action_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7BranchDecisionImageRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7BranchDecisionImageRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7BranchDecisionImageRefusalCode:
            raise TypeError("branch-decision-image refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7BranchDecisionImageRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7BranchDecisionImageRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered sequence",
        )
    axis = tuple(_identity(item, name=f"{name} item") for item in value)
    if not axis or len(axis) != len(set(axis)):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonempty and unique",
        )
    return axis


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its D-image boundary",
        )


def _source_supports_branch(
    source: V1SampledFullPopulationCompactSourceReceipt,
    branch: V1RelaxedKKTBranch,
) -> bool:
    return source.kkt_branch_id in {
        branch.value,
        SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
    }


def _mask_bit(mask: HardMask, index: int) -> bool:
    return bool(mask.packed_bits[index // 8] & (1 << (7 - index % 8)))


def _set_mask_bit(payload: bytearray, index: int, *, retained: bool) -> None:
    mask = 1 << (7 - index % 8)
    if retained:
        payload[index // 8] |= mask
    else:
        payload[index // 8] &= ~mask


def _shell_axis_hash(source: V1SampledFullPopulationCompactSourceReceipt) -> str:
    return content_hash(
        {
            "shell_freeze_receipt_hash": source.shell_freeze_receipt_hash,
            "shell_indices": source.shell_indices,
            "shell_route_ids": source.shell_route_ids,
            "shell_native_fiber_indices": source.shell_native_fiber_indices,
        },
        prefix=_PREFIX + "shell-axis-",
    )


def _physical_identity_hash(
    identity: V1SignedWorkPackedActionDedupIdentity,
) -> str:
    return content_hash(identity, prefix=_PREFIX + "physical-action-identity-")


def _owner_set(uncertainty: V1SignedWorkDecisionUncertaintySet, role: object) -> object:
    if role is V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING:
        return uncertainty.signed_f_sampling
    if role is V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING:
        return uncertainty.literal_j_sampling
    if role is V1SignedWorkDecisionOwnerRole.FD_RESIDUAL:
        return uncertainty.fd_residual
    if role is V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING:
        return uncertainty.solver_rounding
    raise AssertionError("unreachable owner role")


def _validate_source_uncertainty(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
) -> None:
    if type(source) is not V1SampledFullPopulationCompactSourceReceipt:
        raise TypeError("branch image requires the exact full-population source receipt")
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("branch image requires the exact four-owner uncertainty set")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("branch image requires one exact relaxed-KKT branch")
    if type(parent_mask) is not HardMask:
        raise TypeError("branch image requires the exact packed parent mask")
    try:
        source.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
            f"full-population source failed integrity: {error}",
        )
    if not _source_supports_branch(source, branch):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.BRANCH_MISMATCH,
            "source KKT coordinate does not contain this branch",
        )
    if (
        source.chart_identity_hash != uncertainty.chart_identity_hash
        or source.route_axis_hash != uncertainty.route_axis_hash
        or source.registered_d_prefix_hash
        != uncertainty.registered_d_w_prefix_hash
        or source.current_mask_hash != parent_mask.mask_hash
        or source.route_axis_hash != parent_mask.route_axis_hash
        or source.route_count != parent_mask.route_count
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.SOURCE_MISMATCH,
            "source, uncertainty, and packed parent do not share one chart",
        )
    f_owner = uncertainty.signed_f_sampling
    j_owner = uncertainty.literal_j_sampling
    if (
        f_owner.context_axis_hash != source.context_axis_hash
        or j_owner.context_axis_hash != source.context_axis_hash
        or f_owner.draw_axis_hash != source.draw_axis_hash
        or j_owner.draw_axis_hash != source.draw_axis_hash
        or f_owner.direction_axis_hash
        != source.corrected_evidence_stream_source_hash
        or j_owner.direction_axis_hash != source.literal_j_stream_source_hash
        or f_owner.alpha_ledger_family_hash != source.alpha_lineage_hash
        or j_owner.alpha_ledger_family_hash != source.alpha_lineage_hash
        or uncertainty.fd_residual.same_capture_vjp_lineage_hash
        != source.same_capture_vjp_source_hash
        or bool(f_owner.full_train_census) is not source.finite_census_complete
        or bool(j_owner.full_train_census) is not source.finite_census_complete
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.SOURCE_MISMATCH,
            "F/J sampling or deterministic FD custody is not source-bound",
        )
    shell = tuple(source.shell_indices)
    if (
        not 1 <= len(shell) <= _MAXIMUM_STRUCTURAL_SHELL_SIZE
        or shell != tuple(sorted(set(shell)))
        or any(index >= parent_mask.route_count for index in shell)
        or any(not _mask_bit(parent_mask, index) for index in shell)
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            "shell is not a bounded unique retained parent-coordinate axis",
        )


def v1_sampled_v7_branch_owner_source_derivation_hash(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    branch: V1RelaxedKKTBranch,
    owner_role: V1SignedWorkDecisionOwnerRole,
) -> str:
    """Replay the source identity required on an existing owner enclosure."""

    if type(source) is not V1SampledFullPopulationCompactSourceReceipt:
        raise TypeError("owner derivation requires the exact source receipt")
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("owner derivation requires the exact uncertainty set")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("owner derivation requires an exact branch")
    if type(owner_role) is not V1SignedWorkDecisionOwnerRole:
        raise TypeError("owner derivation role must be exact")
    source.assert_integrity()
    if (
        source.chart_identity_hash != uncertainty.chart_identity_hash
        or source.route_axis_hash != uncertainty.route_axis_hash
        or source.registered_d_prefix_hash
        != uncertainty.registered_d_w_prefix_hash
        or not _source_supports_branch(source, branch)
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.SOURCE_MISMATCH,
            "owner derivation source differs from branch uncertainty custody",
        )
    owner = _owner_set(uncertainty, owner_role)
    source_coordinate = {
        V1SignedWorkDecisionOwnerRole.SIGNED_F_SAMPLING: (
            source.corrected_evidence_stream_source_hash
        ),
        V1SignedWorkDecisionOwnerRole.LITERAL_J_SAMPLING: (
            source.literal_j_stream_source_hash
        ),
        V1SignedWorkDecisionOwnerRole.FD_RESIDUAL: (
            source.same_capture_vjp_source_hash
        ),
        V1SignedWorkDecisionOwnerRole.SOLVER_ROUNDING: content_hash(
            {
                "solver_hash": uncertainty.solver_rounding.solver_hash,
                "policy_hash": uncertainty.solver_rounding.policy_hash,
            },
            prefix=_PREFIX + "solver-coordinate-",
        ),
    }[owner_role]
    return content_hash(
        {
            "schema": (
                "partition_free_v1_sampled_v7_branch_full_population_"
                "owner_source_derivation_v1"
            ),
            "branch": branch.value,
            "source_receipt_hash": source.receipt_hash,
            "source_replay_hash": source.source_replay_hash,
            "uncertainty_hash": uncertainty.uncertainty_hash,
            "owner_role": owner_role.value,
            "owner_set_hash": owner.set_hash,
            "source_coordinate": source_coordinate,
            "population_context_axis_hash": source.population_context_axis_hash,
            "population_draw_axis_hash": source.population_draw_axis_hash,
            "observed_context_axis_hash": source.context_axis_hash,
            "observed_draw_axis_hash": source.draw_axis_hash,
            "alpha_lineage_hash": source.alpha_lineage_hash,
            "finite_census_complete": source.finite_census_complete,
        },
        prefix=_PREFIX + "owner-source-",
    )


def _validate_owner_receipts(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    branch: V1RelaxedKKTBranch,
    owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
) -> tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...]:
    receipts = tuple(owner_receipts)
    if (
        len(receipts) != 4
        or any(
            type(value) is not V1SignedWorkDecisionOwnerUncertaintyReceipt
            for value in receipts
        )
        or tuple(value.owner_role for value in receipts) != _OWNER_ROLES
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.OWNER_ENCLOSURE_INCOMPLETE,
            "owner enclosures must cover F, J, FD, and solver exactly once in order",
        )
    for receipt in receipts:
        owner = _owner_set(uncertainty, receipt.owner_role)
        expected_derivation = v1_sampled_v7_branch_owner_source_derivation_hash(
            source=source,
            uncertainty=uncertainty,
            branch=branch,
            owner_role=receipt.owner_role,
        )
        if (
            receipt.uncertainty_set_hash != uncertainty.uncertainty_hash
            or receipt.owner_set_hash != owner.set_hash
            or receipt.registered_d_w_prefix_hash
            != uncertainty.registered_d_w_prefix_hash
            or receipt.route_axis_hash != uncertainty.route_axis_hash
            or receipt.realization_hashes != tuple(owner.realization_hashes)
            or receipt.source_derivation_hash != expected_derivation
            or not receipt.source_derived_from_registered_owner
            or not receipt.finite_grid_covered
            or not receipt.continuum_enclosure_provided
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.OWNER_ENCLOSURE_INCOMPLETE,
                f"{receipt.owner_role.value} is stale or not continuum-enclosed",
            )
    return receipts


def _validate_baseline(
    *,
    baseline: V1PackedRemovalAction,
    parent_mask: HardMask,
) -> V1SignedWorkPackedActionDedupIdentity:
    if type(baseline) is not V1PackedRemovalAction:
        raise TypeError("branch decision image requires an exact packed baseline")
    try:
        count, index_hash = baseline.resulting_mask.removed_index_count_and_hash_from(
            parent_mask
        )
    except Exception as error:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.PACKED_ACTION_MISMATCH,
            f"branch baseline is not an irreversible parent action: {error}",
        )
    if count != baseline.removed_count or index_hash != baseline.removed_route_index_hash:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.PACKED_ACTION_MISMATCH,
            "branch baseline delta identity does not replay",
        )
    return v1_signed_work_packed_action_dedup_identity(baseline)


def _baseline_binding_hash(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    branch: V1RelaxedKKTBranch,
    baseline_identity: V1SignedWorkPackedActionDedupIdentity,
) -> str:
    return content_hash(
        {
            "source_receipt_hash": source.receipt_hash,
            "branch": branch.value,
            "shell_axis_hash": _shell_axis_hash(source),
            "baseline_physical_identity": baseline_identity,
        },
        prefix=_PREFIX + "branch-baseline-binding-",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchOutsideCoordinateExclusionReceipt:
    source_receipt_hash: str
    uncertainty_set_hash: str
    branch: V1RelaxedKKTBranch
    branch_baseline_binding_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    shell_axis_hash: str
    shell_size: int
    eligible_outside_coordinate_count: int
    scanned_outside_coordinate_count: int
    outside_coordinate_axis_hash: str
    nominal_margin_vector_hash: str
    owner_receipt_hashes: tuple[str, str, str, str]
    owner_perturbation_upper_bound: float
    strict_minimum_exclusion_margin_lower_bound: float
    complete_retained_outside_coordinate_scan: bool = True
    strict_outside_coordinate_exclusion_authority: bool = True
    private_exact_reference_only: bool = True
    production_device_scan_authority: bool = False
    action_image_authority: bool = False
    D_stability_authority: bool = False
    C_K_O_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_OUTSIDE_EXCLUSION_SCHEMA
    _issuance_token: InitVar[object] = None

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _REFERENCE_OUTSIDE_TOKEN:
            raise TypeError("outside-coordinate receipts are issuer-only")
        for name in (
            "source_receipt_hash",
            "uncertainty_set_hash",
            "branch_baseline_binding_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "shell_freeze_receipt_hash",
            "shell_axis_hash",
            "outside_coordinate_axis_hash",
            "nominal_margin_vector_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("outside-coordinate branch must be exact")
        for name in (
            "route_count",
            "shell_size",
            "eligible_outside_coordinate_count",
            "scanned_outside_coordinate_count",
        ):
            value = _nonnegative_int(getattr(self, name), name=name)
            if name in {"route_count", "shell_size"} and value == 0:
                _refuse(
                    V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
                    f"{name} must be positive",
                )
            object.__setattr__(self, name, value)
        owners = tuple(self.owner_receipt_hashes)
        if len(owners) != 4 or any(type(value) is not str or not value for value in owners):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.OWNER_ENCLOSURE_INCOMPLETE,
                "outside exclusion lost one of the four owner enclosures",
            )
        object.__setattr__(
            self,
            "owner_perturbation_upper_bound",
            _nonnegative(
                self.owner_perturbation_upper_bound,
                name="owner perturbation upper bound",
            ),
        )
        strict = _finite(
            self.strict_minimum_exclusion_margin_lower_bound,
            name="strict minimum exclusion margin",
        )
        object.__setattr__(self, "strict_minimum_exclusion_margin_lower_bound", strict)
        if (
            self.complete_retained_outside_coordinate_scan is not True
            or self.strict_outside_coordinate_exclusion_authority is not True
            or self.private_exact_reference_only is not True
            or self.scanned_outside_coordinate_count
            != self.eligible_outside_coordinate_count
            or strict <= 0.0
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.OUTSIDE_EXCLUSION_INCOMPLETE,
                "outside-coordinate exclusion is not strict and exhaustive",
            )
        _closed_authority(
            (
                self.production_device_scan_authority,
                self.action_image_authority,
                self.D_stability_authority,
                self.C_K_O_or_commit_authority,
            ),
            owner="reference outside-coordinate receipt",
        )
        if self.schema != SAMPLED_V7_BRANCH_OUTSIDE_EXCLUSION_SCHEMA:
            raise ValueError("unsupported outside-coordinate exclusion schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "outside-exclusion-")


def _issue_v1_sampled_v7_branch_outside_coordinate_exclusion_private_reference(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
    branch_baseline: V1PackedRemovalAction,
    outside_coordinate_nominal_margin_lower_bounds: Sequence[Real],
) -> V1SampledV7BranchOutsideCoordinateExclusionReceipt:
    """Exhaustive small-domain theorem fixture; never a production substitute."""

    _validate_source_uncertainty(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        parent_mask=parent_mask,
    )
    if parent_mask.route_count > _MAXIMUM_REFERENCE_ROUTE_COUNT:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.PRODUCTION_OUTSIDE_PROOF_UNAVAILABLE,
            "route-scale outside exclusion requires the missing typed device scan receipt",
        )
    owners = _validate_owner_receipts(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        owner_receipts=owner_receipts,
    )
    baseline_identity = _validate_baseline(
        baseline=branch_baseline,
        parent_mask=parent_mask,
    )
    shell = set(source.shell_indices)
    outside_axis = tuple(
        index
        for index in range(parent_mask.route_count)
        if _mask_bit(parent_mask, index) and index not in shell
    )
    margins = tuple(
        _finite(value, name="outside-coordinate nominal margin")
        for value in outside_coordinate_nominal_margin_lower_bounds
    )
    if len(margins) != len(outside_axis):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.OUTSIDE_EXCLUSION_INCOMPLETE,
            "reference margin vector does not cover every retained outside coordinate",
        )
    owner_radius = math.nextafter(
        sum(value.outward_radius_upper_bound for value in owners),
        math.inf,
    )
    strict = (
        math.inf
        if not margins
        else min(math.nextafter(value - owner_radius, -math.inf) for value in margins)
    )
    if strict <= 0.0:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.OUTSIDE_EXCLUSION_INCOMPLETE,
            "an outside coordinate can enter under registered owner uncertainty",
        )
    return V1SampledV7BranchOutsideCoordinateExclusionReceipt(
        source_receipt_hash=source.receipt_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        branch=branch,
        branch_baseline_binding_hash=_baseline_binding_hash(
            source=source,
            branch=branch,
            baseline_identity=baseline_identity,
        ),
        parent_mask_hash=parent_mask.mask_hash,
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        shell_freeze_receipt_hash=source.shell_freeze_receipt_hash,
        shell_axis_hash=_shell_axis_hash(source),
        shell_size=len(source.shell_indices),
        eligible_outside_coordinate_count=len(outside_axis),
        scanned_outside_coordinate_count=len(margins),
        outside_coordinate_axis_hash=content_hash(
            outside_axis,
            prefix=_PREFIX + "reference-outside-axis-",
        ),
        nominal_margin_vector_hash=content_hash(
            margins,
            prefix=_PREFIX + "reference-outside-margin-vector-",
        ),
        owner_receipt_hashes=tuple(value.receipt_hash for value in owners),
        owner_perturbation_upper_bound=owner_radius,
        strict_minimum_exclusion_margin_lower_bound=strict,
        _issuance_token=_REFERENCE_OUTSIDE_TOKEN,
    )


def require_v1_sampled_v7_branch_production_outside_coordinate_exclusion(
    **_: object,
) -> NoReturn:
    """Typed closure for the current fixed-primal production path."""

    _refuse(
        V1SampledV7BranchDecisionImageRefusalCode.PRODUCTION_OUTSIDE_PROOF_UNAVAILABLE,
        "no source-bound device receipt yet proves the complete outside-coordinate scan",
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchDecisionCellActionAlias:
    branch: V1RelaxedKKTBranch
    grid_key: tuple[str, str, str, str]
    shell_assignment_ordinal: int
    shell_assignment_bits: tuple[bool, ...]
    shell_assignment_hash: str
    physical_action_identity: V1SignedWorkPackedActionDedupIdentity
    physical_action_identity_hash: str
    exact_packed_bytes_verified: bool = True
    downstream_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA
    _issuance_token: InitVar[object] = None

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ALIAS_TOKEN:
            raise TypeError("decision-cell action aliases are issuer-only")
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("decision-cell alias branch must be exact")
        key = tuple(self.grid_key)
        bits = tuple(self.shell_assignment_bits)
        if (
            len(key) != 4
            or any(type(value) is not str or not value for value in key)
            or not bits
            or any(type(value) is not bool for value in bits)
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
                "decision-cell alias axes are malformed",
            )
        object.__setattr__(self, "grid_key", key)
        object.__setattr__(self, "shell_assignment_bits", bits)
        object.__setattr__(
            self,
            "shell_assignment_ordinal",
            _nonnegative_int(
                self.shell_assignment_ordinal,
                name="shell assignment ordinal",
            ),
        )
        object.__setattr__(
            self,
            "shell_assignment_hash",
            _identity(self.shell_assignment_hash, name="shell assignment hash"),
        )
        if (
            self.physical_action_identity_hash
            != _physical_identity_hash(self.physical_action_identity)
            or self.exact_packed_bytes_verified is not True
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.PACKED_ACTION_MISMATCH,
                "cell alias physical identity is not byte-authoritative",
            )
        _closed_authority((self.downstream_authority,), owner="cell action alias")
        if self.schema != SAMPLED_V7_BRANCH_ACTION_ALIAS_SCHEMA:
            raise ValueError("unsupported decision-cell alias schema")

    @property
    def alias_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "cell-action-alias-")


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchDecisionCellWitness:
    source_receipt_hash: str
    uncertainty_set_hash: str
    branch: V1RelaxedKKTBranch
    grid_key: tuple[str, str, str, str]
    branch_baseline_binding_hash: str
    parent_mask_hash: str
    shell_axis_hash: str
    shell_size: int
    structural_assignment_count: int
    hard_action_aliases: tuple[V1SampledV7BranchDecisionCellActionAlias, ...]
    no_action_assignment_hashes: tuple[str, ...]
    solver_uncertainty_set_hash: str
    solver_realization_hash: str
    fd_uncertainty_set_hash: str
    fd_realization_hash: str
    complete_structural_shell_powerset: bool = True
    solver_face_subset_claimed_exact: bool = False
    complete_hard_action_superset_authority: bool = True
    D_stability_authority: bool = False
    C_K_O_or_commit_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_DECISION_CELL_SCHEMA
    _packed_actions: tuple[V1PackedRemovalAction, ...] = field(
        default=(), repr=False, compare=False
    )
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _CELL_TOKEN:
            raise TypeError("branch decision-cell witnesses are issuer-only")
        for name in (
            "source_receipt_hash",
            "uncertainty_set_hash",
            "branch_baseline_binding_hash",
            "parent_mask_hash",
            "shell_axis_hash",
            "solver_uncertainty_set_hash",
            "solver_realization_hash",
            "fd_uncertainty_set_hash",
            "fd_realization_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("decision-cell branch must be exact")
        key = tuple(self.grid_key)
        if len(key) != 4 or any(type(value) is not str or not value for value in key):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
                "decision cell must bind all four uncertainty owners",
            )
        object.__setattr__(self, "grid_key", key)
        shell_size = _positive_int(self.shell_size, name="shell size")
        assignments = _positive_int(
            self.structural_assignment_count,
            name="structural assignment count",
        )
        aliases = tuple(self.hard_action_aliases)
        no_action = tuple(self.no_action_assignment_hashes)
        actions = tuple(self._packed_actions)
        if (
            shell_size > _MAXIMUM_STRUCTURAL_SHELL_SIZE
            or assignments != 1 << shell_size
            or len(actions) != len(aliases)
            or len(aliases) + len(no_action) != assignments
            or any(
                type(value) is not V1SampledV7BranchDecisionCellActionAlias
                for value in aliases
            )
            or any(type(value) is not V1PackedRemovalAction for value in actions)
            or any(value.branch is not self.branch for value in aliases)
            or any(value.grid_key != key for value in aliases)
            or len({value.shell_assignment_ordinal for value in aliases}) != len(aliases)
            or len(set(no_action)) != len(no_action)
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.CELL_COVERAGE_INCOMPLETE,
                "decision cell does not cover the exact shell powerset",
            )
        object.__setattr__(self, "shell_size", shell_size)
        object.__setattr__(self, "structural_assignment_count", assignments)
        object.__setattr__(self, "hard_action_aliases", aliases)
        object.__setattr__(self, "no_action_assignment_hashes", no_action)
        object.__setattr__(self, "_packed_actions", actions)
        if (
            self.complete_structural_shell_powerset is not True
            or self.solver_face_subset_claimed_exact is not False
            or self.complete_hard_action_superset_authority is not True
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.AUTHORITY_VIOLATION,
                "cell confused a conservative complete image with an exact solver face",
            )
        _closed_authority(
            (self.D_stability_authority, self.C_K_O_or_commit_authority),
            owner="branch decision-cell witness",
        )
        if self.schema != SAMPLED_V7_BRANCH_DECISION_CELL_SCHEMA:
            raise ValueError("unsupported branch decision-cell schema")
        object.__setattr__(self, "_sealed_hash", self._payload_hash())

    def _payload_hash(self) -> str:
        for alias, action in zip(
            self.hard_action_aliases, self._packed_actions, strict=True
        ):
            if (
                v1_signed_work_packed_action_dedup_identity(action)
                != alias.physical_action_identity
            ):
                _refuse(
                    V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                    "packed cell endpoint changed after issuance",
                )
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "branch": self.branch,
                "grid_key": self.grid_key,
                "branch_baseline_binding_hash": self.branch_baseline_binding_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "shell_axis_hash": self.shell_axis_hash,
                "shell_size": self.shell_size,
                "structural_assignment_count": self.structural_assignment_count,
                "hard_action_alias_hashes": tuple(
                    value.alias_hash for value in self.hard_action_aliases
                ),
                "no_action_assignment_hashes": self.no_action_assignment_hashes,
                "solver_uncertainty_set_hash": self.solver_uncertainty_set_hash,
                "solver_realization_hash": self.solver_realization_hash,
                "fd_uncertainty_set_hash": self.fd_uncertainty_set_hash,
                "fd_realization_hash": self.fd_realization_hash,
                "complete_structural_shell_powerset": True,
                "solver_face_subset_claimed_exact": False,
                "complete_hard_action_superset_authority": True,
            },
            prefix=_PREFIX + "cell-witness-",
        )

    def assert_integrity(self) -> str:
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                "branch decision-cell witness changed after issuance",
            )
        return observed

    @property
    def witness_hash(self) -> str:
        return self.assert_integrity()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("branch decision-cell witnesses are nonserializable")


def issue_v1_sampled_v7_branch_decision_cell_witness(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
    branch_baseline: V1PackedRemovalAction,
    grid_key: tuple[str, str, str, str],
) -> V1SampledV7BranchDecisionCellWitness:
    """Enumerate all shell assignments for one registered uncertainty vertex."""

    _validate_source_uncertainty(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        parent_mask=parent_mask,
    )
    baseline_identity = _validate_baseline(
        baseline=branch_baseline,
        parent_mask=parent_mask,
    )
    key = tuple(grid_key)
    if len(key) != 4 or any(
        value not in axis for value, axis in zip(key, uncertainty.realization_axes)
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.AXIS_MISMATCH,
            "decision cell lies outside the registered uncertainty product",
        )
    shell = source.shell_indices
    aliases: list[V1SampledV7BranchDecisionCellActionAlias] = []
    actions: list[V1PackedRemovalAction] = []
    no_action: list[str] = []
    for assignment in range(1 << len(shell)):
        bits = tuple(bool(assignment & (1 << offset)) for offset in range(len(shell)))
        payload = bytearray(branch_baseline.resulting_mask.packed_bits)
        for index, retained in zip(shell, bits, strict=True):
            _set_mask_bit(payload, index, retained=retained)
        child = HardMask(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            packed_bits=bytes(payload),
        )
        assignment_hash = content_hash(
            {
                "source_receipt_hash": source.receipt_hash,
                "branch": branch.value,
                "grid_key": key,
                "shell_axis_hash": _shell_axis_hash(source),
                "assignment_ordinal": assignment,
                "assignment_bits": bits,
            },
            prefix=_PREFIX + "shell-assignment-",
        )
        try:
            action = V1PackedRemovalAction.from_resulting_mask(
                parent_mask=parent_mask,
                resulting_mask=child,
            )
        except ValueError as error:
            if "nonempty route delta" not in str(error):
                raise
            no_action.append(assignment_hash)
            continue
        identity = v1_signed_work_packed_action_dedup_identity(action)
        aliases.append(
            V1SampledV7BranchDecisionCellActionAlias(
                branch=branch,
                grid_key=key,
                shell_assignment_ordinal=assignment,
                shell_assignment_bits=bits,
                shell_assignment_hash=assignment_hash,
                physical_action_identity=identity,
                physical_action_identity_hash=_physical_identity_hash(identity),
                _issuance_token=_ALIAS_TOKEN,
            )
        )
        actions.append(action)
    return V1SampledV7BranchDecisionCellWitness(
        source_receipt_hash=source.receipt_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        branch=branch,
        grid_key=key,
        branch_baseline_binding_hash=_baseline_binding_hash(
            source=source,
            branch=branch,
            baseline_identity=baseline_identity,
        ),
        parent_mask_hash=parent_mask.mask_hash,
        shell_axis_hash=_shell_axis_hash(source),
        shell_size=len(shell),
        structural_assignment_count=1 << len(shell),
        hard_action_aliases=tuple(aliases),
        no_action_assignment_hashes=tuple(no_action),
        solver_uncertainty_set_hash=uncertainty.solver_rounding.set_hash,
        solver_realization_hash=key[3],
        fd_uncertainty_set_hash=uncertainty.fd_residual.set_hash,
        fd_realization_hash=key[2],
        _packed_actions=tuple(actions),
        _issuance_token=_CELL_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchPhysicalActionCustody:
    physical_action_ordinal: int
    physical_action_identity: V1SignedWorkPackedActionDedupIdentity
    physical_action_identity_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    packed_payload_bytes: int
    cell_action_alias_hashes: tuple[str, ...]
    exact_packed_byte_deduplication: bool = True
    downstream_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_PHYSICAL_ACTION_SCHEMA
    _packed_action: V1PackedRemovalAction = field(default=None, repr=False, compare=False)
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _PHYSICAL_TOKEN:
            raise TypeError("branch physical-action custody is issuer-only")
        if type(self._packed_action) is not V1PackedRemovalAction:
            raise TypeError("physical-action custody requires exact packed bytes")
        object.__setattr__(
            self,
            "physical_action_ordinal",
            _positive_int(self.physical_action_ordinal, name="physical action ordinal"),
        )
        for name in (
            "physical_action_identity_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "removed_count",
            _positive_int(self.removed_count, name="removed count"),
        )
        object.__setattr__(
            self,
            "packed_payload_bytes",
            _positive_int(self.packed_payload_bytes, name="packed payload bytes"),
        )
        object.__setattr__(
            self,
            "cell_action_alias_hashes",
            _identity_axis(self.cell_action_alias_hashes, name="cell action aliases"),
        )
        if self.exact_packed_byte_deduplication is not True:
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.PACKED_ACTION_MISMATCH,
                "physical deduplication cannot rely on digests alone",
            )
        _closed_authority((self.downstream_authority,), owner="physical-action custody")
        if self.schema != SAMPLED_V7_BRANCH_PHYSICAL_ACTION_SCHEMA:
            raise ValueError("unsupported branch physical-action schema")
        object.__setattr__(self, "_sealed_hash", self._payload_hash())

    def _payload_hash(self) -> str:
        action = self._packed_action
        identity = v1_signed_work_packed_action_dedup_identity(action)
        if (
            identity != self.physical_action_identity
            or _physical_identity_hash(identity) != self.physical_action_identity_hash
            or action.action_hash != self.packed_action_hash
            or action.resulting_mask.mask_hash != self.resulting_mask_hash
            or hashlib.sha256(action.resulting_mask.packed_bits).hexdigest()
            != self.resulting_mask_packed_sha256
            or action.removed_route_index_hash != self.removed_route_index_hash
            or action.removed_count != self.removed_count
            or len(action.resulting_mask.packed_bits) != self.packed_payload_bytes
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                "physical packed action changed after image issuance",
            )
        return content_hash(
            {
                "schema": self.schema,
                "physical_action_ordinal": self.physical_action_ordinal,
                "physical_action_identity": self.physical_action_identity,
                "physical_action_identity_hash": self.physical_action_identity_hash,
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "resulting_mask_packed_sha256": self.resulting_mask_packed_sha256,
                "removed_route_index_hash": self.removed_route_index_hash,
                "removed_count": self.removed_count,
                "packed_payload_bytes": self.packed_payload_bytes,
                "cell_action_alias_hashes": self.cell_action_alias_hashes,
                "exact_packed_byte_deduplication": True,
            },
            prefix=_PREFIX + "physical-action-custody-",
        )

    def assert_integrity(self) -> str:
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                "physical-action custody identity changed",
            )
        return observed

    @property
    def custody_hash(self) -> str:
        return self.assert_integrity()

    @property
    def packed_action(self) -> V1PackedRemovalAction:
        self.assert_integrity()
        return self._packed_action

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("branch physical-action custody is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchDecisionImageCertificate:
    source_receipt_hash: str
    source_replay_hash: str
    uncertainty_set_hash: str
    branch: V1RelaxedKKTBranch
    branch_baseline_binding_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    shell_axis_hash: str
    owner_receipt_hashes: tuple[str, str, str, str]
    outside_coordinate_exclusion_receipt_hash: str
    uncertainty_cell_witness_hashes: tuple[str, ...]
    uncertainty_cell_axis: tuple[tuple[str, str, str, str], ...]
    expected_uncertainty_cell_count: int
    cell_action_alias_axis: tuple[tuple[str, str], ...]
    no_action_cell_assignment_axis: tuple[tuple[str, str], ...]
    physical_actions: tuple[V1SampledV7BranchPhysicalActionCustody, ...]
    physical_action_identity_axis: tuple[str, ...]
    physical_action_count: int
    complete_cartesian_uncertainty_product: bool = True
    complete_structural_shell_powerset_per_cell: bool = True
    strict_outside_coordinate_exclusion_bound: bool = True
    exact_packed_action_union: bool = True
    branch_aliases_preserved_during_physical_dedup: bool = True
    exact_solver_reachable_subset_claimed: bool = False
    complete_branch_action_image_authority: bool = True
    production_branch_action_image_authority: bool = False
    D_stability_authority: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    reference_memory_limitation: str = (
        SAMPLED_V7_BRANCH_DECISION_IMAGE_REFERENCE_MEMORY_LIMITATION
    )
    production_storage_requirement: str = (
        SAMPLED_V7_BRANCH_DECISION_IMAGE_PRODUCTION_STORAGE_REQUIREMENT
    )
    authority_scope: str = SAMPLED_V7_BRANCH_DECISION_IMAGE_AUTHORITY
    schema: str = SAMPLED_V7_BRANCH_DECISION_IMAGE_SCHEMA
    _source: V1SampledFullPopulationCompactSourceReceipt = field(
        default=None, repr=False, compare=False
    )
    _outside: V1SampledV7BranchOutsideCoordinateExclusionReceipt = field(
        default=None, repr=False, compare=False
    )
    _cells: tuple[V1SampledV7BranchDecisionCellWitness, ...] = field(
        default=(), repr=False, compare=False
    )
    _issuance_token: InitVar[object] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _CERTIFICATE_TOKEN:
            raise TypeError("branch decision-image certificates are issuer-only")
        if (
            type(self._source) is not V1SampledFullPopulationCompactSourceReceipt
            or type(self._outside)
            is not V1SampledV7BranchOutsideCoordinateExclusionReceipt
            or any(
                type(value) is not V1SampledV7BranchDecisionCellWitness
                for value in self._cells
            )
        ):
            raise TypeError("branch image lost exact live proof custody")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "uncertainty_set_hash",
            "branch_baseline_binding_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "shell_freeze_receipt_hash",
            "shell_axis_hash",
            "outside_coordinate_exclusion_receipt_hash",
            "reference_memory_limitation",
            "production_storage_requirement",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch image branch must be exact")
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "expected_uncertainty_cell_count",
            _positive_int(
                self.expected_uncertainty_cell_count,
                name="expected uncertainty cell count",
            ),
        )
        object.__setattr__(
            self,
            "physical_action_count",
            _positive_int(self.physical_action_count, name="physical action count"),
        )
        required_true = (
            self.complete_cartesian_uncertainty_product,
            self.complete_structural_shell_powerset_per_cell,
            self.strict_outside_coordinate_exclusion_bound,
            self.exact_packed_action_union,
            self.branch_aliases_preserved_during_physical_dedup,
            self.complete_branch_action_image_authority,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.AUTHORITY_VIOLATION,
                "branch image lost a required completeness proof",
            )
        _closed_authority(
            (
                self.exact_solver_reachable_subset_claimed,
                self.production_branch_action_image_authority,
                self.D_stability_authority,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="complete branch decision image",
        )
        if (
            self.reference_memory_limitation
            != SAMPLED_V7_BRANCH_DECISION_IMAGE_REFERENCE_MEMORY_LIMITATION
            or self.production_storage_requirement
            != SAMPLED_V7_BRANCH_DECISION_IMAGE_PRODUCTION_STORAGE_REQUIREMENT
            or self.authority_scope != SAMPLED_V7_BRANCH_DECISION_IMAGE_AUTHORITY
            or self.schema != SAMPLED_V7_BRANCH_DECISION_IMAGE_SCHEMA
        ):
            raise ValueError("unsupported sampled-V7 branch decision-image semantics")
        object.__setattr__(self, "_sealed_hash", self._payload_hash())

    def _payload_hash(self) -> str:
        try:
            source_hash = self._source.assert_integrity()
        except Exception as error:
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                f"branch image source changed: {error}",
            )
        cell_hashes = tuple(value.assert_integrity() for value in self._cells)
        custody_hashes = tuple(value.assert_integrity() for value in self.physical_actions)
        if (
            source_hash != self.source_receipt_hash
            or self._outside.receipt_hash
            != self.outside_coordinate_exclusion_receipt_hash
            or cell_hashes != self.uncertainty_cell_witness_hashes
            or tuple(value.physical_action_identity_hash for value in self.physical_actions)
            != self.physical_action_identity_axis
            or len(self.physical_actions) != self.physical_action_count
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                "branch decision-image proof graph changed after issuance",
            )
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "source_replay_hash": self.source_replay_hash,
                "uncertainty_set_hash": self.uncertainty_set_hash,
                "branch": self.branch,
                "branch_baseline_binding_hash": self.branch_baseline_binding_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "shell_freeze_receipt_hash": self.shell_freeze_receipt_hash,
                "shell_axis_hash": self.shell_axis_hash,
                "owner_receipt_hashes": self.owner_receipt_hashes,
                "outside_coordinate_exclusion_receipt_hash": (
                    self.outside_coordinate_exclusion_receipt_hash
                ),
                "uncertainty_cell_witness_hashes": cell_hashes,
                "uncertainty_cell_axis": self.uncertainty_cell_axis,
                "expected_uncertainty_cell_count": self.expected_uncertainty_cell_count,
                "cell_action_alias_axis": self.cell_action_alias_axis,
                "no_action_cell_assignment_axis": (
                    self.no_action_cell_assignment_axis
                ),
                "physical_action_custody_hashes": custody_hashes,
                "physical_action_identity_axis": self.physical_action_identity_axis,
                "physical_action_count": self.physical_action_count,
                "complete_cartesian_uncertainty_product": True,
                "complete_structural_shell_powerset_per_cell": True,
                "strict_outside_coordinate_exclusion_bound": True,
                "exact_packed_action_union": True,
                "branch_aliases_preserved_during_physical_dedup": True,
                "exact_solver_reachable_subset_claimed": False,
                "complete_branch_action_image_authority": True,
                "production_branch_action_image_authority": False,
                "D_stability_authority": False,
                "C_measurement_authority": False,
                "calibration_authority": False,
                "oracle_authority": False,
                "mask_commit_authority": False,
                "terminal_authority": False,
                "pareto_authority": False,
                "reference_memory_limitation": self.reference_memory_limitation,
                "production_storage_requirement": (
                    self.production_storage_requirement
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "certificate-",
        )

    def assert_integrity(self) -> str:
        observed = self._payload_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.STALE_SOURCE,
                "branch decision-image certificate changed after issuance",
            )
        return observed

    @property
    def certificate_hash(self) -> str:
        return self.assert_integrity()

    def packed_action(
        self,
        physical_action_identity_hash: str,
    ) -> V1PackedRemovalAction:
        identity = _identity(
            physical_action_identity_hash,
            name="physical action identity hash",
        )
        for custody in self.physical_actions:
            if custody.physical_action_identity_hash == identity:
                return custody.packed_action
        raise KeyError(identity)

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("branch decision-image certificates are nonserializable")


def certify_v1_sampled_v7_branch_complete_decision_image(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
    branch_baseline: V1PackedRemovalAction,
    outside_coordinate_exclusion: (
        V1SampledV7BranchOutsideCoordinateExclusionReceipt | None
    ),
    cell_witnesses: Sequence[V1SampledV7BranchDecisionCellWitness],
) -> V1SampledV7BranchDecisionImageCertificate:
    """Join the complete one-branch proof graph into a packed action image."""

    _validate_source_uncertainty(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        parent_mask=parent_mask,
    )
    owners = _validate_owner_receipts(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        owner_receipts=owner_receipts,
    )
    baseline_identity = _validate_baseline(
        baseline=branch_baseline,
        parent_mask=parent_mask,
    )
    baseline_binding = _baseline_binding_hash(
        source=source,
        branch=branch,
        baseline_identity=baseline_identity,
    )
    if type(outside_coordinate_exclusion) is not (
        V1SampledV7BranchOutsideCoordinateExclusionReceipt
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.PRODUCTION_OUTSIDE_PROOF_UNAVAILABLE,
            "fixed-primal shell has no typed complete outside-coordinate proof",
        )
    outside = outside_coordinate_exclusion
    owner_hashes = tuple(value.receipt_hash for value in owners)
    if (
        outside.source_receipt_hash != source.receipt_hash
        or outside.uncertainty_set_hash != uncertainty.uncertainty_hash
        or outside.branch is not branch
        or outside.branch_baseline_binding_hash != baseline_binding
        or outside.parent_mask_hash != parent_mask.mask_hash
        or outside.route_axis_hash != parent_mask.route_axis_hash
        or outside.route_count != parent_mask.route_count
        or outside.shell_freeze_receipt_hash != source.shell_freeze_receipt_hash
        or outside.shell_axis_hash != _shell_axis_hash(source)
        or outside.owner_receipt_hashes != owner_hashes
        or not outside.strict_outside_coordinate_exclusion_authority
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.OUTSIDE_EXCLUSION_INCOMPLETE,
            "outside exclusion belongs to another source/baseline/owner graph",
        )
    cells = tuple(cell_witnesses)
    if any(type(value) is not V1SampledV7BranchDecisionCellWitness for value in cells):
        raise TypeError("branch image cell axis requires exact cell witnesses")
    expected_axis = tuple(product(*uncertainty.realization_axes))
    observed_axis = tuple(value.grid_key for value in cells)
    if (
        len(cells) != uncertainty.expected_realization_count
        or len(cells) != len(expected_axis)
        or set(observed_axis) != set(expected_axis)
        or len(set(observed_axis)) != len(observed_axis)
    ):
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.UNCERTAINTY_PRODUCT_INCOMPLETE,
            "cells do not cover the complete four-owner Cartesian product",
        )
    order = {key: index for index, key in enumerate(expected_axis)}
    cells = tuple(sorted(cells, key=lambda value: order[value.grid_key]))
    for cell in cells:
        if (
            cell.source_receipt_hash != source.receipt_hash
            or cell.uncertainty_set_hash != uncertainty.uncertainty_hash
            or cell.branch is not branch
            or cell.branch_baseline_binding_hash != baseline_binding
            or cell.parent_mask_hash != parent_mask.mask_hash
            or cell.shell_axis_hash != _shell_axis_hash(source)
            or cell.solver_uncertainty_set_hash
            != uncertainty.solver_rounding.set_hash
            or cell.solver_realization_hash != cell.grid_key[3]
            or cell.fd_uncertainty_set_hash != uncertainty.fd_residual.set_hash
            or cell.fd_realization_hash != cell.grid_key[2]
            or not cell.complete_structural_shell_powerset
            or not cell.complete_hard_action_superset_authority
            or cell.solver_face_subset_claimed_exact
        ):
            _refuse(
                V1SampledV7BranchDecisionImageRefusalCode.CELL_COVERAGE_INCOMPLETE,
                "one uncertainty cell is stale or lacks its structural powerset proof",
            )
        cell.assert_integrity()

    by_identity: dict[
        str,
        tuple[
            V1SignedWorkPackedActionDedupIdentity,
            V1PackedRemovalAction,
            list[str],
        ],
    ] = {}
    alias_axis: list[tuple[str, str]] = []
    no_action_axis: list[tuple[str, str]] = []
    for cell in cells:
        for assignment_hash in cell.no_action_assignment_hashes:
            no_action_axis.append((cell.witness_hash, assignment_hash))
        for alias, action in zip(
            cell.hard_action_aliases, cell._packed_actions, strict=True
        ):
            alias_hash = alias.alias_hash
            alias_axis.append((cell.witness_hash, alias_hash))
            identity_hash = alias.physical_action_identity_hash
            previous = by_identity.get(identity_hash)
            if previous is None:
                by_identity[identity_hash] = (
                    alias.physical_action_identity,
                    action,
                    [alias_hash],
                )
                continue
            previous_identity, previous_action, alias_hashes = previous
            if (
                previous_identity != alias.physical_action_identity
                or previous_action.resulting_mask.packed_bits
                != action.resulting_mask.packed_bits
            ):
                _refuse(
                    V1SampledV7BranchDecisionImageRefusalCode.PACKED_ACTION_MISMATCH,
                    "matching identities did not have byte-equal packed children",
                )
            alias_hashes.append(alias_hash)
    baseline_physical_hash = _physical_identity_hash(baseline_identity)
    if baseline_physical_hash not in by_identity:
        _refuse(
            V1SampledV7BranchDecisionImageRefusalCode.CELL_COVERAGE_INCOMPLETE,
            "registered branch baseline is absent from its structural hard image",
        )
    physical: list[V1SampledV7BranchPhysicalActionCustody] = []
    for ordinal, identity_hash in enumerate(sorted(by_identity), start=1):
        identity, action, alias_hashes = by_identity[identity_hash]
        physical.append(
            V1SampledV7BranchPhysicalActionCustody(
                physical_action_ordinal=ordinal,
                physical_action_identity=identity,
                physical_action_identity_hash=identity_hash,
                packed_action_hash=action.action_hash,
                resulting_mask_hash=action.resulting_mask.mask_hash,
                resulting_mask_packed_sha256=hashlib.sha256(
                    action.resulting_mask.packed_bits
                ).hexdigest(),
                removed_route_index_hash=action.removed_route_index_hash,
                removed_count=action.removed_count,
                packed_payload_bytes=len(action.resulting_mask.packed_bits),
                cell_action_alias_hashes=tuple(alias_hashes),
                _packed_action=action,
                _issuance_token=_PHYSICAL_TOKEN,
            )
        )
    return V1SampledV7BranchDecisionImageCertificate(
        source_receipt_hash=source.receipt_hash,
        source_replay_hash=source.source_replay_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        branch=branch,
        branch_baseline_binding_hash=baseline_binding,
        parent_mask_hash=parent_mask.mask_hash,
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        shell_freeze_receipt_hash=source.shell_freeze_receipt_hash,
        shell_axis_hash=_shell_axis_hash(source),
        owner_receipt_hashes=owner_hashes,
        outside_coordinate_exclusion_receipt_hash=outside.receipt_hash,
        uncertainty_cell_witness_hashes=tuple(value.witness_hash for value in cells),
        uncertainty_cell_axis=tuple(value.grid_key for value in cells),
        expected_uncertainty_cell_count=len(expected_axis),
        cell_action_alias_axis=tuple(alias_axis),
        no_action_cell_assignment_axis=tuple(no_action_axis),
        physical_actions=tuple(physical),
        physical_action_identity_axis=tuple(
            value.physical_action_identity_hash for value in physical
        ),
        physical_action_count=len(physical),
        _source=source,
        _outside=outside,
        _cells=cells,
        _issuance_token=_CERTIFICATE_TOKEN,
    )


__all__ = [
    "SAMPLED_V7_BRANCH_DECISION_IMAGE_AUTHORITY",
    "SAMPLED_V7_BRANCH_DECISION_IMAGE_PRODUCTION_STORAGE_REQUIREMENT",
    "SAMPLED_V7_BRANCH_DECISION_IMAGE_REFERENCE_MEMORY_LIMITATION",
    "SAMPLED_V7_BRANCH_DECISION_IMAGE_SCHEMA",
    "V1SampledV7BranchDecisionCellActionAlias",
    "V1SampledV7BranchDecisionCellWitness",
    "V1SampledV7BranchDecisionImageCertificate",
    "V1SampledV7BranchDecisionImageRefusal",
    "V1SampledV7BranchDecisionImageRefusalCode",
    "V1SampledV7BranchOutsideCoordinateExclusionReceipt",
    "V1SampledV7BranchPhysicalActionCustody",
    "certify_v1_sampled_v7_branch_complete_decision_image",
    "issue_v1_sampled_v7_branch_decision_cell_witness",
    "require_v1_sampled_v7_branch_production_outside_coordinate_exclusion",
    "v1_sampled_v7_branch_owner_source_derivation_hash",
]
