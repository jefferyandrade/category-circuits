"""Production evidence seam and typed refusal for one sampled-V7 branch action.

This module does not solve another optimization problem.  It joins the
already-issued source-native production evidence needed to reduce one V7
branch image to one strictly preferred physical hard action:

* every observed block carries the complete within-shell and both outside
  rectangle families for every registered branch/resource/cardinality alias;
* the append-only production witness proves strict full-population corrected-F
  separation over that same complete physical alternate domain;
* an F-free, issuer-owned Q/J projection proves strict origin-branch hard-Q
  safety for that same fixed action and replays same-capture literal-J and
  adaptive-rank custody; and
* the V7 full-population source, uncertainty owners, shell, parent, and packed
  child all name exactly the same chart and irreversible action.

The positive output is deliberately narrower than a branch decision image.
It retains a source-coordinate strict-F witness and a strict branch-Q witness
for one physical action, but it does *not* pretend that the two-branch compact
conditioned-D1 source is the branch-specific Q source.  Existing parents do
not expose a typed branch-projection binding, do not bind the full owner
realization/radius product into the Q/J projection, do not supply a strict
continuum-margin certificate, and do not close the trajectory-wide
nonrecycling alpha ledger.  The decision-image issuer therefore fails closed
with a typed missing-premise receipt.  No same-source F/Q, cross-branch D, C,
K, O, commit, terminal, performance, or Pareto authority is opened here.

Raw tensors, caller margins, caller completeness flags, and caller-authored
hash stand-ins are absent from the public issuer.  Every proof coordinate is
recovered from an exact live receipt and replayed again at this consumption
seam.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import hashlib
import math
from numbers import Integral
from typing import Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceReceipt,
)
from .modern_v1_sampled_v7_branch_decision_image import (
    _shell_axis_hash,
    _source_supports_branch,
    _validate_baseline,
    _validate_owner_receipts,
)
from .modern_v1_signed_work_block_support_complete_union_production import (
    V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
    V1BlockSupportCompleteUnionRawSlopeReceiptV4,
)
from .modern_v1_signed_work_block_support_outside_pair_live_family import (
    V1BlockSupportOutsidePairLiveFamilyReceiptV3,
)
from .modern_v1_signed_work_block_support_outside_pair_production import (
    V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    V1BlockSupportOutsidePairProductionDualSourceReceiptV2,
)
from .modern_v1_signed_work_block_support_production_witness import (
    V1BlockSupportProductionMarginLowerBoundObservationV2,
    V1BlockSupportProductionWitnessDecisionV2,
    V1BlockSupportProductionWitnessStateV2,
    _EMPTY_CHAIN_HASH,
    require_v1_block_support_production_witness_decision_v2,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportWitnessStatus,
)
from .modern_v1_signed_work_decision_enclosure import (
    V1SignedWorkDecisionOwnerUncertaintyReceipt,
)
from .modern_v1_signed_work_decision_witness import (
    V1SignedWorkDecisionUncertaintySet,
)
from .modern_v1_signed_work_full_u_f_q_action_invariance_production import (
    V1SignedWorkFullUFQProductionQJDomainProjection,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    v1_signed_work_packed_action_dedup_identity,
)


SAMPLED_V7_BRANCH_PRODUCTION_BLOCK_EVIDENCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_production_block_evidence_v1"
)
SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_production_unjoined_F_Q_evidence_v1"
)
SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_AUTHORITY: Final = (
    "one_branch_v7_provenance_plus_complete_alias_cardinality_resource_two_"
    "rectangle_strict_full_population_corrected_F_and_fixed_action_origin_Q_"
    "evidence_only_no_branch_projection_same_source_continuum_or_alpha_ledger_"
    "closure_cross_branch_D_C_K_O_commit_terminal_performance_or_pareto_authority_v1"
)
# Compatibility spellings are exact aliases; they do not restore fixed-source
# or action-image authority.
SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_SCHEMA: Final = (
    SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_SCHEMA
)
SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_AUTHORITY: Final = (
    SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_AUTHORITY
)
SAMPLED_V7_BRANCH_PRODUCTION_MISSING_PREMISE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_branch_production_missing_premise_v1"
)
SAMPLED_V7_BRANCH_PRODUCTION_Q_PREMISE: Final = (
    "issuer_owned_F_free_Q_J_domain_projection_with_strict_fixed_action_origin_"
    "branch_Q_and_replayed_same_capture_literal_J_adaptive_rank_custody_v1"
)
SAMPLED_V7_BRANCH_PRODUCTION_REQUIRED_SUCCESSOR_FIELDS: Final = (
    "branch_projection_source_binding_receipt_hash",
    "composite_conditioned_d1_source_receipt_hash",
    "origin_branch_q_source_receipt_hash",
    "origin_branch_q_source_replay_hash",
    "branch_projection_same_d_rows_prefix_capture_route_lineage_closed",
    "live_cuda_finite_grid_qualification_receipt_hash",
    "qualified_uncertainty_set_hash",
    "ordered_owner_role_set_realization_radius_axis",
    "qj_projection_owner_uncertainty_product_hash",
    "continuum_margin_certificate_hash",
    "continuum_margin_lower_outward",
    "f_witness_registration_alpha_ledger_hash",
    "f_witness_registration_alpha_coordinate_hash",
    "f_witness_alpha_allocation_index",
    "f_witness_family_alpha_budget",
    "f_witness_family_alpha_spent_before",
    "f_witness_attempt_alpha_allocation",
    "trajectory_proposal_order_alpha_ledger_head_hash",
    "trajectory_nonrecycling_alpha_closure_receipt_hash",
    "trajectory_nonrecycling_alpha_closed",
)

# Legacy names remain import-readable inside this not-yet-facaded module, but
# they name the narrowed unjoined-evidence semantics rather than decision authority.
SAMPLED_V7_BRANCH_PRODUCTION_DECISION_IMAGE_SCHEMA: Final = (
    SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_SCHEMA
)
SAMPLED_V7_BRANCH_PRODUCTION_DECISION_IMAGE_AUTHORITY: Final = (
    SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_AUTHORITY
)

_PREFIX: Final = "partition-free-v1-sampled-v7-branch-production-image-"
_BLOCK_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_MISSING_PREMISE_TOKEN: Final = object()
_CANONICAL_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL.value,
    V1RelaxedKKTBranch.DIAGONAL_L1.value,
)


class V1SampledV7BranchProductionImageRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    BRANCH_MISMATCH = "branch_mismatch"
    OWNER_ENCLOSURE_INCOMPLETE = "owner_enclosure_incomplete"
    BLOCK_CHAIN_INCOMPLETE = "block_chain_incomplete"
    ALIAS_COVERAGE_INCOMPLETE = "alias_coverage_incomplete"
    F_SEPARATION_NOT_STRICT = "F_separation_not_strict"
    Q_SAFETY_NOT_STRICT = "Q_safety_not_strict"
    ACTION_MISMATCH = "action_mismatch"
    MISSING_PRODUCTION_PREMISE = "missing_production_premise"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7BranchProductionImageRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7BranchProductionImageRefusalCode,
        message: str,
        *,
        missing_premise_receipt: object | None = None,
    ) -> None:
        if type(code) is not V1SampledV7BranchProductionImageRefusalCode:
            raise TypeError("production-image refusal code must be exact")
        self.code = code
        self.missing_premise_receipt = missing_premise_receipt
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7BranchProductionImageRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7BranchProductionImageRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its pre-C authority boundary",
        )


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)
        if not item.name.startswith("_") and item.name not in {"receipt_hash"}
    }


def _assert_integrity(value: object, *, owner: str) -> str:
    method = getattr(value, "assert_integrity", None)
    if not callable(method):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.INVALID_TYPE,
            f"{owner} lacks an exact integrity replay method",
        )
    try:
        result = method()
    except Exception as error:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
            f"{owner} failed integrity replay: {type(error).__name__}: {error}",
        )
    return _identity(result, name=f"{owner} integrity hash")


def _owner_uncertainty_coordinate_axis(
    owners: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
) -> tuple[tuple[str, str, str, tuple[str, ...], float], ...]:
    """Expose the exact branch-local owner coordinates without closing them."""

    return tuple(
        (
            value.owner_role.value,
            value.uncertainty_set_hash,
            value.owner_set_hash,
            tuple(value.realization_hashes),
            float(value.outward_radius_upper_bound),
        )
        for value in owners
    )


def _validate_v7_provenance_head(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    branch: V1RelaxedKKTBranch,
) -> None:
    """Replay the V7 provenance coordinate without joining an F/Q parent.

    The caller must separately replay the four owner receipts.  This helper is
    intentionally parent-free: the missing branch-projection receipt is the
    only lawful object that can later bind this two-branch source to a
    branch-specific F/Q action parent.
    """

    if type(source) is not V1SampledFullPopulationCompactSourceReceipt:
        raise TypeError("production evidence requires the exact V7 source receipt")
    if type(uncertainty) is not V1SignedWorkDecisionUncertaintySet:
        raise TypeError("production evidence requires the exact uncertainty set")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("production evidence requires one exact KKT branch")
    try:
        source.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
            f"V7 provenance source failed replay: {type(error).__name__}: {error}",
        )
    if not _source_supports_branch(source, branch):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BRANCH_MISMATCH,
            "V7 provenance source does not contain the requested branch",
        )
    f_owner = uncertainty.signed_f_sampling
    j_owner = uncertainty.literal_j_sampling
    if (
        source.chart_identity_hash != uncertainty.chart_identity_hash
        or source.route_axis_hash != uncertainty.route_axis_hash
        or source.registered_d_prefix_hash
        != uncertainty.registered_d_w_prefix_hash
        or f_owner.context_axis_hash != source.context_axis_hash
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
            V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
            "V7 source and uncertainty provenance coordinates differ",
        )
    return None


def _q_projection_values(
    projection: V1SignedWorkFullUFQProductionQJDomainProjection,
) -> dict[str, object]:
    if type(projection) is not V1SignedWorkFullUFQProductionQJDomainProjection:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.MISSING_PRODUCTION_PREMISE,
            "production V7 requires the exact issuer-owned F-free robust Q/J "
            "domain projection; a conditional Q bridge or raw endpoint is insufficient",
        )
    try:
        projection.assert_integrity()
        values = projection.as_dict()
    except Exception as error:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
            f"robust preferred-Q projection failed replay: {type(error).__name__}: {error}",
        )
    return values


def _fresh_parent_hash(parent_mask: HardMask) -> str:
    if type(parent_mask) is not HardMask:
        raise TypeError("production branch image requires the exact HardMask type")
    try:
        snapshot = HardMask(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            packed_bits=bytes(parent_mask.packed_bits),
        )
    except (TypeError, ValueError) as error:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.ACTION_MISMATCH,
            f"packed parent is malformed: {error}",
        )
    return snapshot.mask_hash


def _strict_full_population_lower(
    decision: V1BlockSupportProductionWitnessDecisionV2,
) -> float:
    """Return one strict outward lower; equality and underflow fail closed."""

    if decision.status is not V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.F_SEPARATION_NOT_STRICT,
            "production corrected-F witness is not stable",
        )
    if decision.full_census:
        if (
            decision.exact_census_sum_numerator is None
            or decision.exact_census_sum_denominator is None
            or decision.exact_census_sum_numerator <= 0
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.F_SEPARATION_NOT_STRICT,
                "full-census corrected-F separation is nonpositive",
            )
        exact_mean = Fraction(
            decision.exact_census_sum_numerator,
            decision.exact_census_sum_denominator * decision.block_population_count,
        )
        lower = float(exact_mean)
        if Fraction.from_float(lower) > exact_mean:
            lower = math.nextafter(lower, -math.inf)
    else:
        lower = math.nextafter(float(decision.confidence_lower), -math.inf)
    if not math.isfinite(lower) or lower <= 0.0:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.F_SEPARATION_NOT_STRICT,
            "outward full-population corrected-F lower is not strictly positive",
        )
    return lower


def _production_chain_hash(
    observations: Sequence[V1BlockSupportProductionMarginLowerBoundObservationV2],
) -> str:
    chain = _EMPTY_CHAIN_HASH
    for index, observation in enumerate(observations):
        chain = content_hash(
            {
                "schema": "partition_free_v1_block_support_production_chain_link_v2",
                "prior_receipt_chain_hash": chain,
                "block_index": index,
                "observation_hash": observation.observation_hash,
            },
            prefix="partition-free-v1-block-support-production-witness-chain-",
        )
    return chain


def _strict_origin_q(
    projection: V1SignedWorkFullUFQProductionQJDomainProjection,
) -> tuple[dict[str, object], float, float]:
    """Replay the robust-Q parent and enforce a genuinely open budget."""

    values = _q_projection_values(projection)
    required = (
        "strict_origin_branch_registered_Q_safety",
        "literal_J_same_capture_response_geometry",
        "complete_two_branch_hard_domain_replayed",
        "deterministic_FD_residual_decision_stable",
        "complete_physical_alternate_domain_crosswalk",
        "physical_action_invariance_authority",
        "array_free_receipt",
    )
    if any(values.get(name) is not True for name in required):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.MISSING_PRODUCTION_PREMISE,
            "F-free preferred-Q parent lacks robust J/FD/domain evidence",
        )
    try:
        q_upper = float(values["origin_branch_q_upper"])
        q_budget = float(values["quadratic_budget"])
    except (KeyError, TypeError, ValueError):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.MISSING_PRODUCTION_PREMISE,
            "F-free preferred-Q parent lacks finite branch-Q coordinates",
        )
    if not (
        math.isfinite(q_upper)
        and math.isfinite(q_budget)
        and 0.0 <= q_upper < q_budget
    ):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.Q_SAFETY_NOT_STRICT,
            "preferred origin-branch Q is unsafe, ambiguous, or exactly on budget",
        )
    return values, q_upper, q_budget


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchProductionBlockEvidence:
    """One exact observed block of the production pair-cover chain."""

    block_index: int
    complete_cover_receipt_hash: str
    complete_raw_union_receipt_hash: str
    observation_hash: str
    live_cover_receipt_hashes: tuple[str, ...]
    production_dual_source_receipt_hashes: tuple[str, ...]
    live_family_receipt_hashes: tuple[str, ...]
    ordered_alias_binding_hashes: tuple[str, ...]
    ordered_resource_alias_hashes: tuple[str, ...]
    ordered_alias_cardinality_axis: tuple[tuple[str, str, str, int, int, int, int], ...]
    ordered_source_pair_kkt_axis: tuple[tuple[str, ...], ...]
    ordered_rectangle_count_axis: tuple[tuple[int, int, int, int, int, int, int, int, int], ...]
    receipt_hash: str = field(init=False)
    complete_alias_cardinality_resource_axis: bool = True
    complete_two_rectangle_cover_per_alias: bool = True
    source_bound_production_cuda_chain: bool = True
    population_separation_authority: bool = False
    q_authority: bool = False
    action_image_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_PRODUCTION_BLOCK_EVIDENCE_SCHEMA
    _complete_cover: V1BlockSupportCompleteConstraintCoverBindingReceiptV3 = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _complete_raw: V1BlockSupportCompleteUnionRawSlopeReceiptV4 = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _observation: V1BlockSupportProductionMarginLowerBoundObservationV2 = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _live_covers: tuple[V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2, ...] = field(
        default=(), repr=False, compare=False
    )
    _production_duals: tuple[
        V1BlockSupportOutsidePairProductionDualSourceReceiptV2, ...
    ] = field(default=(), repr=False, compare=False)
    _live_families: tuple[V1BlockSupportOutsidePairLiveFamilyReceiptV3, ...] = field(
        default=(), repr=False, compare=False
    )
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _BLOCK_TOKEN:
            raise TypeError("production V7 block evidence is issuer-only")
        exact = (
            type(self._complete_cover)
            is V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
            type(self._complete_raw) is V1BlockSupportCompleteUnionRawSlopeReceiptV4,
            type(self._observation)
            is V1BlockSupportProductionMarginLowerBoundObservationV2,
            type(self._live_covers) is tuple
            and all(
                type(value) is V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
                for value in self._live_covers
            ),
            type(self._production_duals) is tuple
            and all(
                type(value)
                is V1BlockSupportOutsidePairProductionDualSourceReceiptV2
                for value in self._production_duals
            ),
            type(self._live_families) is tuple
            and all(
                type(value) is V1BlockSupportOutsidePairLiveFamilyReceiptV3
                for value in self._live_families
            ),
        )
        if not all(exact):
            raise TypeError("production block evidence lost exact live receipt custody")
        object.__setattr__(
            self,
            "block_index",
            _nonnegative_int(self.block_index, name="block index"),
        )
        for name in (
            "complete_cover_receipt_hash",
            "complete_raw_union_receipt_hash",
            "observation_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        alias_count = len(self.ordered_alias_binding_hashes)
        if (
            alias_count == 0
            or len(self.live_cover_receipt_hashes) != alias_count
            or len(self.production_dual_source_receipt_hashes) != alias_count
            or len(self.live_family_receipt_hashes) != alias_count
            or len(self.ordered_resource_alias_hashes) != alias_count
            or len(self.ordered_alias_cardinality_axis) != alias_count
            or len(self.ordered_source_pair_kkt_axis) != alias_count
            or len(self.ordered_rectangle_count_axis) != alias_count
            or any(
                type(axis) is not tuple
                or any(type(value) is not str or not value for value in axis)
                for axis in (
                    self.live_cover_receipt_hashes,
                    self.production_dual_source_receipt_hashes,
                    self.live_family_receipt_hashes,
                    self.ordered_alias_binding_hashes,
                    self.ordered_resource_alias_hashes,
                )
            )
            or any(
                len(set(axis)) != len(axis)
                for axis in (
                    self.live_cover_receipt_hashes,
                    self.production_dual_source_receipt_hashes,
                    self.live_family_receipt_hashes,
                    self.ordered_alias_binding_hashes,
                    self.ordered_resource_alias_hashes,
                )
            )
            or any(
                type(axis) is not tuple
                or len(axis) != 7
                or any(type(value) is not str or not value for value in axis[:3])
                or any(
                    isinstance(value, bool) or not isinstance(value, Integral)
                    for value in axis[3:]
                )
                for axis in self.ordered_alias_cardinality_axis
            )
            or any(
                type(axis) is not tuple
                or not axis
                or any(type(value) is not str or not value for value in axis)
                for axis in self.ordered_source_pair_kkt_axis
            )
            or any(
                type(axis) is not tuple
                or len(axis) != 9
                or any(
                    isinstance(value, bool) or not isinstance(value, Integral)
                    for value in axis
                )
                for axis in self.ordered_rectangle_count_axis
            )
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
                "production block evidence lost an exact immutable cover axis",
            )
        required = (
            self.complete_alias_cardinality_resource_axis,
            self.complete_two_rectangle_cover_per_alias,
            self.source_bound_production_cuda_chain,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.AUTHORITY_VIOLATION,
                "production block evidence lost one complete-cover premise",
            )
        _closed(
            (
                self.population_separation_authority,
                self.q_authority,
                self.action_image_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="production block evidence",
        )
        if self.schema != SAMPLED_V7_BRANCH_PRODUCTION_BLOCK_EVIDENCE_SCHEMA:
            raise ValueError("unsupported production block-evidence schema")
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production block evidence was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _compute_hash(self) -> str:
        cover_hash = _assert_integrity(self._complete_cover, owner="complete cover")
        raw_hash = _assert_integrity(self._complete_raw, owner="complete raw union")
        observation_hash = _assert_integrity(
            self._observation, owner="production block observation"
        )
        live_cover_hashes = tuple(
            _assert_integrity(value, owner="outside live-cover authority")
            for value in self._live_covers
        )
        production_dual_hashes = tuple(
            _assert_integrity(value, owner="outside production dual source")
            for value in self._production_duals
        )
        live_family_hashes = tuple(
            _assert_integrity(value, owner="outside live family")
            for value in self._live_families
        )
        if (
            cover_hash != self.complete_cover_receipt_hash
            or raw_hash != self.complete_raw_union_receipt_hash
            or observation_hash != self.observation_hash
            or live_cover_hashes != self.live_cover_receipt_hashes
            or production_dual_hashes
            != self.production_dual_source_receipt_hashes
            or live_family_hashes != self.live_family_receipt_hashes
            or self._complete_cover.ordered_alias_binding_hashes
            != self.ordered_alias_binding_hashes
            or self._complete_cover.ordered_resource_alias_hashes
            != self.ordered_resource_alias_hashes
            or self._complete_cover.ordered_alias_cardinality_axis
            != self.ordered_alias_cardinality_axis
            or self._complete_cover.ordered_source_pair_kkt_axis
            != self.ordered_source_pair_kkt_axis
            or self._complete_cover.ordered_rectangle_count_axis
            != self.ordered_rectangle_count_axis
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production block proof graph changed after issuance",
            )
        return content_hash(
            _payload(self), prefix=_PREFIX + "block-evidence-"
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._integrity_seal or observed != self.receipt_hash:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production block evidence no longer replays",
            )
        return observed


def bind_v1_sampled_v7_branch_production_block_evidence(
    *,
    complete_cover: V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
    complete_raw_union: V1BlockSupportCompleteUnionRawSlopeReceiptV4,
    observation: V1BlockSupportProductionMarginLowerBoundObservationV2,
    outside_live_cover_authorities: Sequence[
        V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
    ],
    outside_production_dual_sources: Sequence[
        V1BlockSupportOutsidePairProductionDualSourceReceiptV2
    ],
    outside_live_families: Sequence[V1BlockSupportOutsidePairLiveFamilyReceiptV3],
) -> V1SampledV7BranchProductionBlockEvidence:
    """Bind one observed block without accepting any proof booleans or margins."""

    if (
        type(complete_cover)
        is not V1BlockSupportCompleteConstraintCoverBindingReceiptV3
        or type(complete_raw_union) is not V1BlockSupportCompleteUnionRawSlopeReceiptV4
        or type(observation)
        is not V1BlockSupportProductionMarginLowerBoundObservationV2
    ):
        raise TypeError("production block binding requires exact cover/raw/observation receipts")
    covers = tuple(outside_live_cover_authorities)
    duals = tuple(outside_production_dual_sources)
    families = tuple(outside_live_families)
    if (
        not covers
        or len(covers) != len(duals)
        or len(covers) != len(families)
        or any(
            type(value) is not V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
            for value in covers
        )
        or any(
            type(value)
            is not V1BlockSupportOutsidePairProductionDualSourceReceiptV2
            for value in duals
        )
        or any(
            type(value) is not V1BlockSupportOutsidePairLiveFamilyReceiptV3
            for value in families
        )
    ):
        raise TypeError(
            "production block binding requires exact nonempty cover/dual/family axes"
        )
    _assert_integrity(complete_cover, owner="complete cover")
    _assert_integrity(complete_raw_union, owner="complete raw union")
    _assert_integrity(observation, owner="production block observation")
    for value in covers:
        _assert_integrity(value, owner="outside live-cover authority")
    for value in duals:
        _assert_integrity(value, owner="outside production dual source")
    for value in families:
        _assert_integrity(value, owner="outside live family")

    cover_hashes = tuple(value.receipt_hash for value in covers)
    dual_hashes = tuple(value.receipt_hash for value in duals)
    family_hashes = tuple(value.receipt_hash for value in families)
    alias_count = len(complete_cover.ordered_alias_binding_hashes)
    if (
        alias_count == 0
        or len(covers) != alias_count
        or complete_cover.ordered_live_cover_authority_receipt_hashes != cover_hashes
        or complete_cover.ordered_live_family_receipt_hashes != family_hashes
        or complete_raw_union.complete_cover_v3_receipt_hash
        != complete_cover.receipt_hash
        or complete_raw_union.ordered_alias_binding_hashes
        != complete_cover.ordered_alias_binding_hashes
        or complete_raw_union.ordered_live_cover_authority_receipt_hashes
        != cover_hashes
        or complete_raw_union.ordered_live_family_receipt_hashes != family_hashes
        or observation.complete_cover_v3_receipt_hash != complete_cover.receipt_hash
        or observation.complete_union_raw_v4_receipt_hash
        != complete_raw_union.receipt_hash
        or observation.block_index != complete_cover.block_index
        or observation.canonical_context_receipt_hash
        != complete_cover.canonical_block_context_receipt_hash
            or observation.block_context_axis_hash
            != complete_cover.block_support_context_axis_hash
            or observation.complete_union_raw_lower_outward
            != complete_raw_union.complete_union_raw_lower_outward
            or observation.d_weight_numerator != complete_cover.d_weight_numerator
            or observation.residual_weight_numerator
            != complete_cover.residual_weight_numerator
            or observation.population_weight_denominator
            != complete_cover.population_weight_denominator
            or complete_raw_union.canonical_block_observation_raw_union_authority
            is not True
            or complete_raw_union.canonical_block_observation_authority is not False
            or complete_raw_union.population_separation_authority is not False
        ):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BLOCK_CHAIN_INCOMPLETE,
            "block cover, raw union, observation, and live family axes do not close",
        )
    common = (
        "bound_registration_receipt_hash",
        "preferred_action_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
    )
    observation_name = {
        "bound_registration_receipt_hash": "bound_registration_receipt_hash",
    }
    for name in common:
        observed_name = observation_name.get(name, name)
        if getattr(observation, observed_name) != getattr(complete_cover, name):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
                f"block observation and complete cover disagree on {name}",
            )
    for ordinal, (cover, dual, family) in enumerate(
        zip(covers, duals, families, strict=True)
    ):
        cardinality = complete_cover.ordered_alias_cardinality_axis[ordinal]
        rectangle = complete_cover.ordered_rectangle_count_axis[ordinal]
        if (
            cover.receipt_hash
            != complete_cover.ordered_live_cover_authority_receipt_hashes[ordinal]
            or family.receipt_hash
            != complete_cover.ordered_live_family_receipt_hashes[ordinal]
            or family.live_cover_authority_receipt_hash != cover.receipt_hash
            or dual.live_cover_authority_receipt_hash != cover.receipt_hash
            or family.production_dual_source_receipt_hash != dual.receipt_hash
            or dual.live_cover_source_native_payload_hash
            != cover.live_cover_source_native_payload_hash
            or dual.source_native_alias_commitment_hash
            != cover.source_native_alias_commitment_hash
            or cover.resource_alias_hash
            != complete_cover.ordered_resource_alias_hashes[ordinal]
            or family.resource_alias_hash != cover.resource_alias_hash
            or dual.resource_alias_hash != cover.resource_alias_hash
            or dual.branch_id != cover.branch_id
            or dual.resource_alias_id != cover.resource_alias_id
            or dual.cardinality_alias_id != cover.cardinality_alias_id
            or dual.registered_fd_rank != cover.registered_fd_rank
            or dual.block_index != cover.block_index
            or dual.block_id != cover.block_id
            or dual.route_axis_hash != cover.route_axis_hash
            or dual.route_count != cover.route_count
            or dual.kkt_state_identity_hash != cover.kkt_state_identity_hash
            or dual.least_work_problem_hash != cover.kkt_problem_hash
            or dual.kkt_policy_hash != cover.kkt_policy_hash
            or dual.kkt_branch_receipt_hash != cover.kkt_branch_receipt_hash
            or dual.kkt_eligible_mask_identity_hash
            != cover.kkt_eligible_mask_identity_hash
            or dual.kkt_required_mask_identity_hash
            != cover.kkt_required_mask_identity_hash
            or dual.source_native_pair_axis_commitment_hash
            != cover.source_native_pair_axis_commitment_hash
            or family.least_work_problem_hash != dual.least_work_problem_hash
            or family.source_problem_hash != dual.source_problem_hash
            or family.signed_work_operator_receipt_hash
            != dual.signed_work_operator_receipt_hash
            or family.resource_axis_hash != dual.resource_axis_hash
            or family.resource_coefficient_source_hash
            != dual.resource_coefficient_source_hash
            or family.capacity_operator_source_hash
            != dual.capacity_operator_source_hash
            or family.resource_multiplier_source_hash
            != dual.resource_multiplier_source_hash
            or family.capacity_multiplier_source_hash
            != dual.capacity_multiplier_source_hash
            or family.operator_scales_hash != dual.operator_scales_hash
            or family.capacity_count != dual.capacity_count
            or family.design_hash != dual.design_hash
            or family.hbm_forecast_hash != dual.hbm_forecast_hash
            or family.resident_device_identity != dual.resident_device_identity
            or (
                cover.branch_id,
                cover.resource_alias_id,
                cover.cardinality_alias_id,
                cover.registered_fd_rank,
                cover.fixed_outside_removed_count,
                cover.shell_top_k_count,
                cover.hard_total_removed_count,
            )
            != cardinality
            or (
                cover.route_count,
                cover.complete_route_scan_count,
                cover.add_outside_droppable_count,
                cover.add_outside_addable_count,
                cover.drop_outside_add_inside_droppable_count,
                cover.drop_outside_add_inside_addable_count,
                cover.add_outside_rectangle_pair_count,
                cover.drop_outside_add_inside_rectangle_pair_count,
                cover.complete_pair_count,
            )
            != rectangle
            or family.complete_pair_cover_count != cover.complete_pair_count
            or family.complete_two_rectangle_pair_cover_scanned is not True
            or family.complete_route_scan is not True
            or cover.all_four_rectangle_axes_bound is not True
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
                f"alias {ordinal} lost its exact cardinality/resource/two-rectangle cover",
            )
    return V1SampledV7BranchProductionBlockEvidence(
        block_index=complete_cover.block_index,
        complete_cover_receipt_hash=complete_cover.receipt_hash,
        complete_raw_union_receipt_hash=complete_raw_union.receipt_hash,
        observation_hash=observation.observation_hash,
        live_cover_receipt_hashes=cover_hashes,
        production_dual_source_receipt_hashes=dual_hashes,
        live_family_receipt_hashes=family_hashes,
        ordered_alias_binding_hashes=complete_cover.ordered_alias_binding_hashes,
        ordered_resource_alias_hashes=complete_cover.ordered_resource_alias_hashes,
        ordered_alias_cardinality_axis=complete_cover.ordered_alias_cardinality_axis,
        ordered_source_pair_kkt_axis=complete_cover.ordered_source_pair_kkt_axis,
        ordered_rectangle_count_axis=complete_cover.ordered_rectangle_count_axis,
        _complete_cover=complete_cover,
        _complete_raw=complete_raw_union,
        _observation=observation,
        _live_covers=covers,
        _production_duals=duals,
        _live_families=families,
        _issuer_token=_BLOCK_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt:
    source_receipt_hash: str
    source_replay_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    uncertainty_set_hash: str
    owner_receipt_hashes: tuple[str, str, str, str]
    owner_uncertainty_coordinate_axis: tuple[
        tuple[str, str, str, tuple[str, ...], float], ...
    ]
    branch: V1RelaxedKKTBranch
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    shell_axis_hash: str
    production_block_evidence_hashes: tuple[str, ...]
    production_witness_decision_hash: str
    production_witness_state_hash: str
    production_witness_receipt_chain_hash: str
    observed_block_count: int
    block_population_count: int
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    alternate_action_count: int
    alias_axis_hash: str
    ordered_alias_binding_hashes: tuple[str, ...]
    ordered_resource_alias_hashes: tuple[str, ...]
    ordered_alias_cardinality_axis: tuple[tuple[str, str, str, int, int, int, int], ...]
    target_branch_alias_ordinals: tuple[int, ...]
    target_branch_phase_a_source_receipt_hashes: tuple[str, ...]
    target_branch_phase_a_source_replay_hashes: tuple[str, ...]
    strict_full_population_F_lower_outward: float
    robust_preferred_q_projection_receipt_hash: str
    composite_conditioned_d1_source_receipt_hash: str
    origin_branch_source_receipt_hash: str
    origin_branch_source_replay_hash: str
    origin_branch_source_identity_receipt_hash: str
    source_pre_c_branch_family_hash: str
    robust_branch_family_hash: str
    frozen_rank: int
    origin_branch_q_upper: float
    quadratic_budget: float
    preferred_packed_action_hash: str
    preferred_resulting_mask_hash: str
    preferred_resulting_mask_packed_sha256: str
    preferred_removed_route_index_hash: str
    preferred_removed_count: int
    preferred_physical_action_identity_hash: str
    receipt_hash: str = field(init=False)
    complete_F_source_bound_alias_axis: bool = True
    complete_cardinality_resource_axis: bool = True
    complete_two_rectangle_outside_cover: bool = True
    strict_full_population_corrected_F_separation: bool = True
    strict_origin_branch_registered_Q_safety: bool = True
    literal_J_same_capture_response_geometry: bool = True
    robust_q_projection_adaptive_rank_custody_replayed: bool = True
    complete_alias_outside_F_evidence: bool = True
    separate_strict_F_and_Q_action_evidence: bool = True
    alpha_identifiers_are_provenance_only: bool = True
    branch_projection_source_binding_joined: bool = False
    same_source_F_Q_witness_authority: bool = False
    fixed_source_single_action_witness: bool = False
    fixed_source_registered_alias_outside_exclusion_authority: bool = False
    complete_source_bound_alias_axis: bool = False
    owner_uncertainty_product_qj_joined: bool = False
    continuum_margin_certificate_joined: bool = False
    trajectory_nonrecycling_alpha_authority: bool = False
    production_outside_coordinate_exclusion_authority: bool = False
    production_branch_action_image_authority: bool = False
    cross_branch_D_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    q_premise: str = SAMPLED_V7_BRANCH_PRODUCTION_Q_PREMISE
    authority_scope: str = SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_AUTHORITY
    schema: str = SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_SCHEMA
    _source: V1SampledFullPopulationCompactSourceReceipt = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _uncertainty: V1SignedWorkDecisionUncertaintySet = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _owners: tuple[V1SignedWorkDecisionOwnerUncertaintyReceipt, ...] = field(
        default=(), repr=False, compare=False
    )
    _parent: HardMask = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _preferred_action: V1PackedRemovalAction = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _blocks: tuple[V1SampledV7BranchProductionBlockEvidence, ...] = field(
        default=(), repr=False, compare=False
    )
    _decision: V1BlockSupportProductionWitnessDecisionV2 = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _robust_q: V1SignedWorkFullUFQProductionQJDomainProjection = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("production V7 decision images are issuer-only")
        if (
            type(self._source) is not V1SampledFullPopulationCompactSourceReceipt
            or type(self._uncertainty) is not V1SignedWorkDecisionUncertaintySet
            or type(self._parent) is not HardMask
            or type(self._preferred_action) is not V1PackedRemovalAction
            or type(self._decision) is not V1BlockSupportProductionWitnessDecisionV2
            or type(self._robust_q)
            is not V1SignedWorkFullUFQProductionQJDomainProjection
            or not self._blocks
            or any(
                type(value) is not V1SampledV7BranchProductionBlockEvidence
                for value in self._blocks
            )
            or len(self._owners) != 4
            or any(
                type(value) is not V1SignedWorkDecisionOwnerUncertaintyReceipt
                for value in self._owners
            )
        ):
            raise TypeError("production V7 image lost exact live proof custody")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "uncertainty_set_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "shell_freeze_receipt_hash",
            "shell_axis_hash",
            "production_witness_decision_hash",
            "production_witness_state_hash",
            "production_witness_receipt_chain_hash",
            "complete_alternate_domain_hash",
            "complete_alternate_domain_receipt_hash",
            "alias_axis_hash",
            "robust_preferred_q_projection_receipt_hash",
            "composite_conditioned_d1_source_receipt_hash",
            "origin_branch_source_receipt_hash",
            "origin_branch_source_replay_hash",
            "origin_branch_source_identity_receipt_hash",
            "source_pre_c_branch_family_hash",
            "robust_branch_family_hash",
            "preferred_packed_action_hash",
            "preferred_resulting_mask_hash",
            "preferred_resulting_mask_packed_sha256",
            "preferred_removed_route_index_hash",
            "preferred_physical_action_identity_hash",
            "q_premise",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("production V7 image branch must be exact")
        for name in (
            "route_count",
            "observed_block_count",
            "block_population_count",
            "alternate_action_count",
            "preferred_removed_count",
            "frozen_rank",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        alias_count = len(self.ordered_alias_binding_hashes)
        owner_axis = self.owner_uncertainty_coordinate_axis
        if (
            type(self.owner_receipt_hashes) is not tuple
            or len(self.owner_receipt_hashes) != 4
            or any(type(value) is not str or not value for value in self.owner_receipt_hashes)
            or len(set(self.owner_receipt_hashes)) != 4
            or type(owner_axis) is not tuple
            or len(owner_axis) != 4
            or len({value[0] for value in owner_axis}) != 4
            or any(
                type(value) is not tuple
                or len(value) != 5
                or any(type(item) is not str or not item for item in value[:3])
                or type(value[3]) is not tuple
                or not value[3]
                or any(type(item) is not str or not item for item in value[3])
                or isinstance(value[4], bool)
                or not isinstance(value[4], (int, float))
                or not math.isfinite(float(value[4]))
                or float(value[4]) < 0.0
                for value in owner_axis
            )
            or type(self.production_block_evidence_hashes) is not tuple
            or not self.production_block_evidence_hashes
            or any(
                type(value) is not str or not value
                for value in self.production_block_evidence_hashes
            )
            or len(set(self.production_block_evidence_hashes))
            != len(self.production_block_evidence_hashes)
            or not self.ordered_alias_binding_hashes
            or not self.ordered_resource_alias_hashes
            or not self.ordered_alias_cardinality_axis
            or not self.target_branch_alias_ordinals
            or not self.target_branch_phase_a_source_receipt_hashes
            or not self.target_branch_phase_a_source_replay_hashes
            or len(self.target_branch_phase_a_source_receipt_hashes)
            != len(self.target_branch_phase_a_source_replay_hashes)
            or any(
                type(value) is not str or not value
                for axis in (
                    self.target_branch_phase_a_source_receipt_hashes,
                    self.target_branch_phase_a_source_replay_hashes,
                )
                for value in axis
            )
            or alias_count != len(self.ordered_resource_alias_hashes)
            or alias_count != len(self.ordered_alias_cardinality_axis)
            or any(
                type(value) is not str or not value
                for axis in (
                    self.ordered_alias_binding_hashes,
                    self.ordered_resource_alias_hashes,
                )
                for value in axis
            )
            or len(set(self.ordered_alias_binding_hashes)) != alias_count
            or len(set(self.ordered_resource_alias_hashes)) != alias_count
            or type(self.target_branch_alias_ordinals) is not tuple
            or tuple(sorted(set(self.target_branch_alias_ordinals)))
            != self.target_branch_alias_ordinals
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) < alias_count
                for value in self.target_branch_alias_ordinals
            )
            or any(
                type(axis) is not tuple
                or len(axis) != 7
                or axis[0] != self.branch.value
                for index, axis in enumerate(self.ordered_alias_cardinality_axis)
                if index in self.target_branch_alias_ordinals
            )
            or self.observed_block_count != len(self.production_block_evidence_hashes)
            or self.observed_block_count > self.block_population_count
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
                "production image lost one owner, block, alias, or branch coordinate",
            )
        positives = (
            self.complete_F_source_bound_alias_axis,
            self.complete_cardinality_resource_axis,
            self.complete_two_rectangle_outside_cover,
            self.strict_full_population_corrected_F_separation,
            self.strict_origin_branch_registered_Q_safety,
            self.literal_J_same_capture_response_geometry,
            self.robust_q_projection_adaptive_rank_custody_replayed,
            self.complete_alias_outside_F_evidence,
            self.separate_strict_F_and_Q_action_evidence,
            self.alpha_identifiers_are_provenance_only,
        )
        if any(type(value) is not bool or value is not True for value in positives):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.AUTHORITY_VIOLATION,
                "production image lost one mandatory proof premise",
            )
        _closed(
            (
                self.cross_branch_D_stability_authority,
                self.branch_projection_source_binding_joined,
                self.same_source_F_Q_witness_authority,
                self.fixed_source_single_action_witness,
                self.fixed_source_registered_alias_outside_exclusion_authority,
                self.complete_source_bound_alias_axis,
                self.owner_uncertainty_product_qj_joined,
                self.continuum_margin_certificate_joined,
                self.trajectory_nonrecycling_alpha_authority,
                self.production_outside_coordinate_exclusion_authority,
                self.production_branch_action_image_authority,
                self.c_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="production branch decision image",
        )
        if (
            not math.isfinite(self.strict_full_population_F_lower_outward)
            or self.strict_full_population_F_lower_outward <= 0.0
            or not math.isfinite(self.origin_branch_q_upper)
            or not math.isfinite(self.quadratic_budget)
            or self.origin_branch_q_upper < 0.0
            or self.quadratic_budget <= 0.0
            or not self.origin_branch_q_upper < self.quadratic_budget
            or self.q_premise != SAMPLED_V7_BRANCH_PRODUCTION_Q_PREMISE
            or self.authority_scope
            != SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_AUTHORITY
            or self.schema
            != SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_SCHEMA
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.AUTHORITY_VIOLATION,
                "production image separation/Q/semantic boundary is invalid",
            )
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production decision image was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _compute_hash(self) -> str:
        try:
            _validate_v7_provenance_head(
                source=self._source,
                uncertainty=self._uncertainty,
                branch=self.branch,
            )
            source_hash = self._source.assert_integrity()
            replayed_owners = _validate_owner_receipts(
                source=self._source,
                uncertainty=self._uncertainty,
                branch=self.branch,
                owner_receipts=self._owners,
            )
            _validate_baseline(
                baseline=self._preferred_action,
                parent_mask=self._parent,
            )
        except Exception as error:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                f"V7 source failed replay: {type(error).__name__}: {error}",
            )
        owner_hashes = tuple(value.receipt_hash for value in replayed_owners)
        owner_axis = _owner_uncertainty_coordinate_axis(replayed_owners)
        block_hashes = tuple(value.assert_integrity() for value in self._blocks)
        target_source_pairs = tuple(
            dict.fromkeys(
                (
                    self._blocks[0]._live_covers[ordinal].phase_a_source_receipt_hash,
                    self._blocks[0]._live_covers[ordinal].phase_a_source_replay_hash,
                )
                for ordinal in self.target_branch_alias_ordinals
            )
        )
        decision_hash = _assert_integrity(
            self._decision, owner="production witness decision"
        )
        q_values, q_upper, q_budget = _strict_origin_q(self._robust_q)
        parent_hash = _fresh_parent_hash(self._parent)
        identity = v1_signed_work_packed_action_dedup_identity(self._preferred_action)
        physical_hash = content_hash(
            identity,
            prefix="partition-free-v1-sampled-v7-branch-decision-image-physical-action-identity-",
        )
        if (
            source_hash != self.source_receipt_hash
            or self._source.source_replay_hash != self.source_replay_hash
            or self._source.alpha_ledger_id != self.alpha_ledger_id
            or self._source.alpha_coordinate_id != self.alpha_coordinate_id
            or self._source.alpha_lineage_hash != self.alpha_lineage_hash
            or self._uncertainty.uncertainty_hash != self.uncertainty_set_hash
            or owner_hashes != self.owner_receipt_hashes
            or owner_axis != self.owner_uncertainty_coordinate_axis
            or block_hashes != self.production_block_evidence_hashes
            or tuple(value[0] for value in target_source_pairs)
            != self.target_branch_phase_a_source_receipt_hashes
            or tuple(value[1] for value in target_source_pairs)
            != self.target_branch_phase_a_source_replay_hashes
            or decision_hash != self.production_witness_decision_hash
            or self._decision.state_hash != self.production_witness_state_hash
            or self._decision.receipt_chain_hash
            != self.production_witness_receipt_chain_hash
            or self._robust_q.receipt_hash
            != self.robust_preferred_q_projection_receipt_hash
            or q_upper != self.origin_branch_q_upper
            or q_budget != self.quadratic_budget
            or q_values["origin_branch_source_receipt_hash"]
            != self.origin_branch_source_receipt_hash
            or q_values["origin_branch_source_replay_hash"]
            != self.origin_branch_source_replay_hash
            or q_values["origin_branch_source_identity_receipt_hash"]
            != self.origin_branch_source_identity_receipt_hash
            or self._source.conditioned_d1_device_source_receipt_hash
            != self.composite_conditioned_d1_source_receipt_hash
            or q_values["source_pre_c_branch_family_hash"]
            != self.source_pre_c_branch_family_hash
            or q_values["robust_branch_family_hash"]
            != self.robust_branch_family_hash
            or q_values["frozen_rank"] != self.frozen_rank
            or parent_hash != self.parent_mask_hash
            or self._preferred_action.action_hash != self.preferred_packed_action_hash
            or self._preferred_action.resulting_mask.mask_hash
            != self.preferred_resulting_mask_hash
            or hashlib.sha256(
                self._preferred_action.resulting_mask.packed_bits
            ).hexdigest()
            != self.preferred_resulting_mask_packed_sha256
            or self._preferred_action.removed_route_index_hash
            != self.preferred_removed_route_index_hash
            or self._preferred_action.removed_count != self.preferred_removed_count
            or physical_hash != self.preferred_physical_action_identity_hash
            or q_values["packed_action_hash"] != self.preferred_packed_action_hash
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production image proof graph changed after issuance",
            )
        return content_hash(_payload(self), prefix=_PREFIX + "receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._integrity_seal or observed != self.receipt_hash:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production decision image no longer replays",
            )
        return observed

    @property
    def packed_action(self) -> V1PackedRemovalAction:
        self.assert_integrity()
        return self._preferred_action

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("production V7 unjoined F/Q evidence is nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7BranchProductionMissingPremiseReceipt:
    """Typed reason unjoined F/Q evidence cannot become a decision image."""

    unjoined_f_q_evidence_receipt_hash: str
    source_receipt_hash: str
    uncertainty_set_hash: str
    branch: V1RelaxedKKTBranch
    composite_conditioned_d1_source_receipt_hash: str
    target_branch_phase_a_source_receipt_hashes: tuple[str, ...]
    target_branch_phase_a_source_replay_hashes: tuple[str, ...]
    origin_branch_q_source_receipt_hash: str
    origin_branch_q_source_replay_hash: str
    alpha_ledger_id_provenance: str
    alpha_coordinate_id_provenance: str
    alpha_lineage_hash_provenance: str
    owner_uncertainty_coordinate_axis: tuple[
        tuple[str, str, str, tuple[str, ...], float], ...
    ]
    missing_premise_codes: tuple[str, str, str, str]
    required_upstream_field_axis: tuple[str, ...]
    receipt_hash: str = field(init=False)
    source_coordinate_F_and_separate_Q_evidence_retained: bool = True
    branch_projection_source_binding_joined: bool = False
    same_source_F_Q_witness_authority: bool = False
    live_cuda_finite_grid_qualification_joined: bool = False
    owner_uncertainty_product_qj_joined: bool = False
    continuum_margin_certificate_joined: bool = False
    f_witness_alpha_coordinate_replayed: bool = False
    trajectory_nonrecycling_alpha_closed: bool = False
    production_outside_coordinate_exclusion_authority: bool = False
    production_branch_action_image_authority: bool = False
    cross_branch_D_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_V7_BRANCH_PRODUCTION_MISSING_PREMISE_SCHEMA
    _witness: V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if (
            _issuer_token is not _MISSING_PREMISE_TOKEN
            or type(self._witness)
            is not V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
        ):
            raise TypeError("production V7 missing-premise receipts are issuer-only")
        for name in (
            "unjoined_f_q_evidence_receipt_hash",
            "source_receipt_hash",
            "uncertainty_set_hash",
            "composite_conditioned_d1_source_receipt_hash",
            "origin_branch_q_source_receipt_hash",
            "origin_branch_q_source_replay_hash",
            "alpha_ledger_id_provenance",
            "alpha_coordinate_id_provenance",
            "alpha_lineage_hash_provenance",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("missing-premise branch must be exact")
        for name in (
            "target_branch_phase_a_source_receipt_hashes",
            "target_branch_phase_a_source_replay_hashes",
        ):
            value = getattr(self, name)
            if (
                type(value) is not tuple
                or not value
                or any(type(item) is not str or not item for item in value)
            ):
                raise TypeError(f"{name} must be one nonempty immutable source axis")
        expected_codes = (
            "branch_specific_q_source_not_bound_to_composite_source",
            "owner_uncertainty_product_not_joined_to_qj_projection",
            "strict_continuum_margin_certificate_absent",
            "f_witness_alpha_coordinate_and_trajectory_ledger_closure_absent",
        )
        if (
            self.missing_premise_codes != expected_codes
            or self.required_upstream_field_axis
            != SAMPLED_V7_BRANCH_PRODUCTION_REQUIRED_SUCCESSOR_FIELDS
            or self.owner_uncertainty_coordinate_axis
            != self._witness.owner_uncertainty_coordinate_axis
            or self.source_coordinate_F_and_separate_Q_evidence_retained is not True
            or self.schema != SAMPLED_V7_BRANCH_PRODUCTION_MISSING_PREMISE_SCHEMA
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.AUTHORITY_VIOLATION,
                "missing-premise receipt changed its exact successor contract",
            )
        _closed(
            (
                self.live_cuda_finite_grid_qualification_joined,
                self.branch_projection_source_binding_joined,
                self.same_source_F_Q_witness_authority,
                self.owner_uncertainty_product_qj_joined,
                self.continuum_margin_certificate_joined,
                self.f_witness_alpha_coordinate_replayed,
                self.trajectory_nonrecycling_alpha_closed,
                self.production_outside_coordinate_exclusion_authority,
                self.production_branch_action_image_authority,
                self.cross_branch_D_stability_authority,
                self.c_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="production V7 missing-premise receipt",
        )
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production V7 missing-premise receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _compute_hash(self) -> str:
        witness_hash = self._witness.assert_integrity()
        if (
            witness_hash != self.unjoined_f_q_evidence_receipt_hash
            or self._witness.source_receipt_hash != self.source_receipt_hash
            or self._witness.uncertainty_set_hash != self.uncertainty_set_hash
            or self._witness.branch is not self.branch
            or self._witness.composite_conditioned_d1_source_receipt_hash
            != self.composite_conditioned_d1_source_receipt_hash
            or self._witness.target_branch_phase_a_source_receipt_hashes
            != self.target_branch_phase_a_source_receipt_hashes
            or self._witness.target_branch_phase_a_source_replay_hashes
            != self.target_branch_phase_a_source_replay_hashes
            or self._witness.origin_branch_source_receipt_hash
            != self.origin_branch_q_source_receipt_hash
            or self._witness.origin_branch_source_replay_hash
            != self.origin_branch_q_source_replay_hash
            or self._witness.alpha_ledger_id != self.alpha_ledger_id_provenance
            or self._witness.alpha_coordinate_id
            != self.alpha_coordinate_id_provenance
            or self._witness.alpha_lineage_hash != self.alpha_lineage_hash_provenance
            or self._witness.owner_uncertainty_coordinate_axis
            != self.owner_uncertainty_coordinate_axis
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "missing-premise receipt no longer names its fixed-source witness",
            )
        return content_hash(_payload(self), prefix=_PREFIX + "missing-premise-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._integrity_seal or observed != self.receipt_hash:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.STALE_SOURCE,
                "production V7 missing-premise receipt no longer replays",
            )
        return observed

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("production V7 missing-premise receipts are nonserializable")


def issue_v1_sampled_v7_branch_production_missing_premise_receipt(
    *,
    unjoined_f_q_evidence: V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
) -> V1SampledV7BranchProductionMissingPremiseReceipt:
    """Name the exact successor evidence absent from the unjoined F/Q seam."""

    if (
        type(unjoined_f_q_evidence)
        is not V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
    ):
        raise TypeError("missing-premise issuance requires exact unjoined F/Q evidence")
    unjoined_f_q_evidence.assert_integrity()
    return V1SampledV7BranchProductionMissingPremiseReceipt(
        unjoined_f_q_evidence_receipt_hash=unjoined_f_q_evidence.receipt_hash,
        source_receipt_hash=unjoined_f_q_evidence.source_receipt_hash,
        uncertainty_set_hash=unjoined_f_q_evidence.uncertainty_set_hash,
        branch=unjoined_f_q_evidence.branch,
        composite_conditioned_d1_source_receipt_hash=(
            unjoined_f_q_evidence.composite_conditioned_d1_source_receipt_hash
        ),
        target_branch_phase_a_source_receipt_hashes=(
            unjoined_f_q_evidence.target_branch_phase_a_source_receipt_hashes
        ),
        target_branch_phase_a_source_replay_hashes=(
            unjoined_f_q_evidence.target_branch_phase_a_source_replay_hashes
        ),
        origin_branch_q_source_receipt_hash=(
            unjoined_f_q_evidence.origin_branch_source_receipt_hash
        ),
        origin_branch_q_source_replay_hash=(
            unjoined_f_q_evidence.origin_branch_source_replay_hash
        ),
        alpha_ledger_id_provenance=unjoined_f_q_evidence.alpha_ledger_id,
        alpha_coordinate_id_provenance=unjoined_f_q_evidence.alpha_coordinate_id,
        alpha_lineage_hash_provenance=unjoined_f_q_evidence.alpha_lineage_hash,
        owner_uncertainty_coordinate_axis=(
            unjoined_f_q_evidence.owner_uncertainty_coordinate_axis
        ),
        missing_premise_codes=(
            "branch_specific_q_source_not_bound_to_composite_source",
            "owner_uncertainty_product_not_joined_to_qj_projection",
            "strict_continuum_margin_certificate_absent",
            "f_witness_alpha_coordinate_and_trajectory_ledger_closure_absent",
        ),
        required_upstream_field_axis=(
            SAMPLED_V7_BRANCH_PRODUCTION_REQUIRED_SUCCESSOR_FIELDS
        ),
        _witness=unjoined_f_q_evidence,
        _issuer_token=_MISSING_PREMISE_TOKEN,
    )


V1SampledV7BranchProductionDecisionImageReceipt = (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
)
V1SampledV7BranchProductionFixedSourceActionWitnessReceipt = (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
)


def issue_v1_sampled_v7_branch_production_unjoined_f_q_evidence(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
    preferred_action: V1PackedRemovalAction,
    block_evidence: Sequence[V1SampledV7BranchProductionBlockEvidence],
    production_witness_decision: V1BlockSupportProductionWitnessDecisionV2,
    robust_preferred_q: V1SignedWorkFullUFQProductionQJDomainProjection,
) -> V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt:
    """Issue strict F/Q action evidence while preserving its unjoined sources."""

    _validate_v7_provenance_head(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
    )
    owners = _validate_owner_receipts(
        source=source,
        uncertainty=uncertainty,
        branch=branch,
        owner_receipts=owner_receipts,
    )
    if not _source_supports_branch(source, branch):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BRANCH_MISMATCH,
            "V7 source does not contain the requested branch",
        )
    if type(preferred_action) is not V1PackedRemovalAction:
        raise TypeError("production branch image requires one exact packed action")
    _validate_baseline(
        baseline=preferred_action,
        parent_mask=parent_mask,
    )
    parent_hash = _fresh_parent_hash(parent_mask)
    packed_digest = hashlib.sha256(
        preferred_action.resulting_mask.packed_bits
    ).hexdigest()

    blocks = tuple(block_evidence)
    if not blocks or any(
        type(value) is not V1SampledV7BranchProductionBlockEvidence
        for value in blocks
    ):
        raise TypeError("production image requires exact nonempty block evidence")
    for block in blocks:
        block.assert_integrity()
    if tuple(value.block_index for value in blocks) != tuple(range(len(blocks))):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BLOCK_CHAIN_INCOMPLETE,
            "production block evidence is omitted, duplicated, or reordered",
        )
    first = blocks[0]
    invariant_axes = (
        "ordered_alias_binding_hashes",
        "ordered_resource_alias_hashes",
        "ordered_alias_cardinality_axis",
        "ordered_source_pair_kkt_axis",
        "ordered_rectangle_count_axis",
    )
    if any(
        any(getattr(value, name) != getattr(first, name) for name in invariant_axes)
        for value in blocks[1:]
    ):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
            "observed blocks do not share one complete alias/resource/rectangle domain",
        )

    if type(production_witness_decision) is not V1BlockSupportProductionWitnessDecisionV2:
        raise TypeError("production image requires the exact production witness decision")
    try:
        require_v1_block_support_production_witness_decision_v2(
            production_witness_decision
        )
    except Exception as error:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.F_SEPARATION_NOT_STRICT,
            f"production corrected-F witness refused: {type(error).__name__}: {error}",
        )
    observations = tuple(value._observation for value in blocks)
    chain_hash = _production_chain_hash(observations)
    if (
        len(blocks) != production_witness_decision.observed_block_count
        or chain_hash != production_witness_decision.receipt_chain_hash
    ):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BLOCK_CHAIN_INCOMPLETE,
            "live block observations do not replay the production decision chain",
        )
    replayed_state = V1BlockSupportProductionWitnessStateV2(
        bound_registration_receipt_hash=(
            production_witness_decision.bound_registration_receipt_hash
        ),
        legacy_registration_hash=production_witness_decision.legacy_registration_hash,
        observed_block_count=len(blocks),
        last_observation_hash=observations[-1].observation_hash,
        receipt_chain_hash=chain_hash,
    )
    if replayed_state.state_hash != production_witness_decision.state_hash:
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.BLOCK_CHAIN_INCOMPLETE,
            "production decision state is not the live observation prefix",
        )
    strict_f = _strict_full_population_lower(production_witness_decision)

    f_action_common = (
        (production_witness_decision.current_mask_hash, parent_hash),
        (production_witness_decision.route_axis_hash, parent_mask.route_axis_hash),
        (production_witness_decision.route_count, parent_mask.route_count),
        (production_witness_decision.preferred_action_hash, preferred_action.action_hash),
    )
    if any(left != right for left, right in f_action_common):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
            "production corrected-F decision names another physical parent/action",
        )
    for block in blocks:
        cover = block._complete_cover
        observation = block._observation
        if (
            cover.bound_registration_receipt_hash
            != production_witness_decision.bound_registration_receipt_hash
            or cover.preoutcome_domain_receipt_hash
            != production_witness_decision.preoutcome_domain_receipt_hash
            or cover.preferred_action_hash != preferred_action.action_hash
            or cover.preferred_removed_route_index_hash
            != preferred_action.removed_route_index_hash
            or cover.preferred_selected_count != preferred_action.removed_count
            or observation.complete_alternate_domain_hash
            != production_witness_decision.complete_alternate_domain_hash
            or observation.alias_axis_hash != production_witness_decision.alias_axis_hash
        ):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
                "one live block names another preferred action or complete domain",
            )
        cover_common = tuple(
            (getattr(cover, name), getattr(production_witness_decision, name))
            for name in (
                "model_identity_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_axis_hash",
                "route_id_axis_hash",
                "route_count",
                "registered_d_prefix_hash",
                "registered_d_w2_prefix_hash",
            )
        )
        if any(left != right for left, right in cover_common):
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
                "one live complete cover names another F chart/mask/prefix source",
            )

    target_ordinals = tuple(
        ordinal
        for ordinal, cardinality in enumerate(first.ordered_alias_cardinality_axis)
        if cardinality[0] == branch.value
    )
    branch_axis = tuple(
        dict.fromkeys(
            cardinality[0] for cardinality in first.ordered_alias_cardinality_axis
        )
    )
    if not target_ordinals or set(branch_axis) != set(_CANONICAL_BRANCHES):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
            "complete alias axis omits the requested canonical branch",
        )
    target_source_pairs = tuple(
        dict.fromkeys(
            (
                first._live_covers[ordinal].phase_a_source_receipt_hash,
                first._live_covers[ordinal].phase_a_source_replay_hash,
            )
            for ordinal in target_ordinals
        )
    )
    for block in blocks:
        block_target_pairs = tuple(
            dict.fromkeys(
                (
                    block._live_covers[ordinal].phase_a_source_receipt_hash,
                    block._live_covers[ordinal].phase_a_source_replay_hash,
                )
                for ordinal in target_ordinals
            )
        )
        if block_target_pairs != target_source_pairs:
            _refuse(
                V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
                "requested-branch aliases changed source across observed blocks",
            )
        for ordinal in target_ordinals:
            cover = block._live_covers[ordinal]
            family = block._live_families[ordinal]
            if (
                cover.branch_id != branch.value
                or family.branch_id != branch.value
                or cover.phase_a_source_receipt_hash
                != family.phase_a_source_receipt_hash
                or cover.phase_a_source_replay_hash
                != family.phase_a_source_replay_hash
                or cover.source_resource_receipt_hash
                != family.source_resource_receipt_hash
            ):
                _refuse(
                    V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
                    "requested-branch aliases lost their exact source/resource custody",
                )

    q_values, q_upper, q_budget = _strict_origin_q(robust_preferred_q)
    q_common = (
        (q_values["packed_action_hash"], preferred_action.action_hash),
        (q_values["resulting_mask_hash"], preferred_action.resulting_mask.mask_hash),
        (q_values["resulting_mask_packed_sha256"], packed_digest),
        (q_values["removed_route_index_hash"], preferred_action.removed_route_index_hash),
        (q_values["removed_count"], preferred_action.removed_count),
        (q_values["parent_mask_hash"], parent_hash),
        (
            q_values["model_identity_hash"],
            production_witness_decision.model_identity_hash,
        ),
        (
            q_values["chart_identity_hash"],
            production_witness_decision.chart_identity_hash,
        ),
        (
            q_values["proposal_generation"],
            production_witness_decision.proposal_generation,
        ),
        (q_values["route_axis_hash"], production_witness_decision.route_axis_hash),
        (
            q_values["route_id_axis_hash"],
            production_witness_decision.route_id_axis_hash,
        ),
        (q_values["route_count"], production_witness_decision.route_count),
        (
            q_values["corrected_a_stream_identity_hash"],
            production_witness_decision.corrected_evidence_source_hash,
        ),
        (
            q_values["registered_d_prefix_hash"],
            production_witness_decision.registered_d_prefix_hash,
        ),
        (
            q_values["registered_d_w2_prefix_hash"],
            production_witness_decision.registered_d_w2_prefix_hash,
        ),
        (q_values["origin_branch"], branch.value),
        (
            q_values["complete_alternate_domain_hash"],
            production_witness_decision.complete_alternate_domain_hash,
        ),
        (
            q_values["complete_alternate_domain_receipt_hash"],
            production_witness_decision.complete_alternate_domain_receipt_hash,
        ),
        (
            q_values["alternate_action_count"],
            production_witness_decision.alternate_action_count,
        ),
        (q_values["alias_axis_hash"], production_witness_decision.alias_axis_hash),
    )
    if any(left != right for left, right in q_common):
        _refuse(
            V1SampledV7BranchProductionImageRefusalCode.SOURCE_MISMATCH,
            "robust preferred-Q and V7 F/domain/action coordinates differ",
        )

    physical_identity = v1_signed_work_packed_action_dedup_identity(preferred_action)
    physical_identity_hash = content_hash(
        physical_identity,
        prefix="partition-free-v1-sampled-v7-branch-decision-image-physical-action-identity-",
    )
    return V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt(
        source_receipt_hash=source.receipt_hash,
        source_replay_hash=source.source_replay_hash,
        alpha_ledger_id=source.alpha_ledger_id,
        alpha_coordinate_id=source.alpha_coordinate_id,
        alpha_lineage_hash=source.alpha_lineage_hash,
        uncertainty_set_hash=uncertainty.uncertainty_hash,
        owner_receipt_hashes=tuple(value.receipt_hash for value in owners),  # type: ignore[arg-type]
        owner_uncertainty_coordinate_axis=_owner_uncertainty_coordinate_axis(owners),
        branch=branch,
        parent_mask_hash=parent_hash,
        route_axis_hash=parent_mask.route_axis_hash,
        route_count=parent_mask.route_count,
        shell_freeze_receipt_hash=source.shell_freeze_receipt_hash,
        shell_axis_hash=_shell_axis_hash(source),
        production_block_evidence_hashes=tuple(value.receipt_hash for value in blocks),
        production_witness_decision_hash=production_witness_decision.decision_hash,
        production_witness_state_hash=production_witness_decision.state_hash,
        production_witness_receipt_chain_hash=chain_hash,
        observed_block_count=len(blocks),
        block_population_count=production_witness_decision.block_population_count,
        complete_alternate_domain_hash=(
            production_witness_decision.complete_alternate_domain_hash
        ),
        complete_alternate_domain_receipt_hash=(
            production_witness_decision.complete_alternate_domain_receipt_hash
        ),
        alternate_action_count=production_witness_decision.alternate_action_count,
        alias_axis_hash=production_witness_decision.alias_axis_hash,
        ordered_alias_binding_hashes=first.ordered_alias_binding_hashes,
        ordered_resource_alias_hashes=first.ordered_resource_alias_hashes,
        ordered_alias_cardinality_axis=first.ordered_alias_cardinality_axis,
        target_branch_alias_ordinals=target_ordinals,
        target_branch_phase_a_source_receipt_hashes=tuple(
            value[0] for value in target_source_pairs
        ),
        target_branch_phase_a_source_replay_hashes=tuple(
            value[1] for value in target_source_pairs
        ),
        strict_full_population_F_lower_outward=strict_f,
        robust_preferred_q_projection_receipt_hash=robust_preferred_q.receipt_hash,
        composite_conditioned_d1_source_receipt_hash=(
            source.conditioned_d1_device_source_receipt_hash
        ),
        origin_branch_source_receipt_hash=_identity(
            q_values["origin_branch_source_receipt_hash"],
            name="origin branch source receipt hash",
        ),
        origin_branch_source_replay_hash=_identity(
            q_values["origin_branch_source_replay_hash"],
            name="origin branch source replay hash",
        ),
        origin_branch_source_identity_receipt_hash=_identity(
            q_values["origin_branch_source_identity_receipt_hash"],
            name="origin branch source identity receipt hash",
        ),
        source_pre_c_branch_family_hash=_identity(
            q_values["source_pre_c_branch_family_hash"],
            name="source pre-C branch family hash",
        ),
        robust_branch_family_hash=_identity(
            q_values["robust_branch_family_hash"],
            name="robust branch family hash",
        ),
        frozen_rank=_positive_int(q_values["frozen_rank"], name="frozen rank"),
        origin_branch_q_upper=q_upper,
        quadratic_budget=q_budget,
        preferred_packed_action_hash=preferred_action.action_hash,
        preferred_resulting_mask_hash=preferred_action.resulting_mask.mask_hash,
        preferred_resulting_mask_packed_sha256=packed_digest,
        preferred_removed_route_index_hash=preferred_action.removed_route_index_hash,
        preferred_removed_count=preferred_action.removed_count,
        preferred_physical_action_identity_hash=physical_identity_hash,
        _source=source,
        _uncertainty=uncertainty,
        _owners=owners,
        _parent=parent_mask,
        _preferred_action=preferred_action,
        _blocks=blocks,
        _decision=production_witness_decision,
        _robust_q=robust_preferred_q,
        _issuer_token=_RECEIPT_TOKEN,
    )


def issue_v1_sampled_v7_branch_production_decision_image(
    *,
    source: V1SampledFullPopulationCompactSourceReceipt,
    uncertainty: V1SignedWorkDecisionUncertaintySet,
    owner_receipts: Sequence[V1SignedWorkDecisionOwnerUncertaintyReceipt],
    branch: V1RelaxedKKTBranch,
    parent_mask: HardMask,
    preferred_action: V1PackedRemovalAction,
    block_evidence: Sequence[V1SampledV7BranchProductionBlockEvidence],
    production_witness_decision: V1BlockSupportProductionWitnessDecisionV2,
    robust_preferred_q: V1SignedWorkFullUFQProductionQJDomainProjection,
) -> NoReturn:
    """Fail closed until every typed source/statistical premise is joined."""

    witness = issue_v1_sampled_v7_branch_production_unjoined_f_q_evidence(
        source=source,
        uncertainty=uncertainty,
        owner_receipts=owner_receipts,
        branch=branch,
        parent_mask=parent_mask,
        preferred_action=preferred_action,
        block_evidence=block_evidence,
        production_witness_decision=production_witness_decision,
        robust_preferred_q=robust_preferred_q,
    )
    missing = issue_v1_sampled_v7_branch_production_missing_premise_receipt(
        unjoined_f_q_evidence=witness
    )
    raise V1SampledV7BranchProductionImageRefusal(
        V1SampledV7BranchProductionImageRefusalCode.MISSING_PRODUCTION_PREMISE,
        "strict F and Q evidence share a physical coordinate, but the branch-"
        "specific Q source is not bound to the composite conditioned-D1 source; "
        "owner-product Q/J closure, a strict continuum margin, F-witness alpha "
        "coordinates, and a live trajectory nonrecycling ledger also remain absent",
        missing_premise_receipt=missing,
    )


# The old fixed-source spelling remains import-readable while downstream
# integrations migrate; it is an exact alias of the deliberately
# non-authorizing unjoined-evidence issuer and exposes no extra switch.
issue_v1_sampled_v7_branch_production_fixed_source_action_witness = (
    issue_v1_sampled_v7_branch_production_unjoined_f_q_evidence
)


__all__ = (
    "SAMPLED_V7_BRANCH_PRODUCTION_BLOCK_EVIDENCE_SCHEMA",
    "SAMPLED_V7_BRANCH_PRODUCTION_DECISION_IMAGE_AUTHORITY",
    "SAMPLED_V7_BRANCH_PRODUCTION_DECISION_IMAGE_SCHEMA",
    "SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_AUTHORITY",
    "SAMPLED_V7_BRANCH_PRODUCTION_FIXED_SOURCE_WITNESS_SCHEMA",
    "SAMPLED_V7_BRANCH_PRODUCTION_MISSING_PREMISE_SCHEMA",
    "SAMPLED_V7_BRANCH_PRODUCTION_Q_PREMISE",
    "SAMPLED_V7_BRANCH_PRODUCTION_REQUIRED_SUCCESSOR_FIELDS",
    "SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_AUTHORITY",
    "SAMPLED_V7_BRANCH_PRODUCTION_UNJOINED_F_Q_EVIDENCE_SCHEMA",
    "V1SampledV7BranchProductionBlockEvidence",
    "V1SampledV7BranchProductionDecisionImageReceipt",
    "V1SampledV7BranchProductionFixedSourceActionWitnessReceipt",
    "V1SampledV7BranchProductionImageRefusal",
    "V1SampledV7BranchProductionImageRefusalCode",
    "V1SampledV7BranchProductionMissingPremiseReceipt",
    "V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt",
    "bind_v1_sampled_v7_branch_production_block_evidence",
    "issue_v1_sampled_v7_branch_production_decision_image",
    "issue_v1_sampled_v7_branch_production_fixed_source_action_witness",
    "issue_v1_sampled_v7_branch_production_missing_premise_receipt",
    "issue_v1_sampled_v7_branch_production_unjoined_f_q_evidence",
)
