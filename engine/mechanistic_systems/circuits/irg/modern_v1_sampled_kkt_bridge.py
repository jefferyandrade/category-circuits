"""Bind one sampled dual chart to the canonical two-branch relaxed KKT family.

The sampled D stream already owns both production-local coordinates required by
the relaxed proposal problem: conserving corrected work for ``F`` and the
literal relaxed-gate Jacobian FD geometry for ``Q``.  This module performs the
small but authority-sensitive conversion from that immutable chart to the two
convex KKT branches.  It observes no C, K, O, or validation outcome and grants
no calibration, oracle, commit, or terminal authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Final, Sequence

import numpy as np

from .contracts import content_hash
from .modern_v1_confidence_box_external_state import V1ConfidenceBoxExternalView
from .modern_v1_fqr_branch_family import (
    V1FQRKKTPreFrozenBranchFamily,
    freeze_v1_relaxed_kkt_branch_family,
)
from .modern_v1_geometric_budgets import (
    V1GeometricTrustBudgets,
    same_geometric_coordinate,
)
from .modern_v1_geometric_selector_production import V1ProductionLegalityBlock
from .modern_v1_kkt_corrected_work_identity import (
    V1KKTCorrectedWorkIdentityReceipt,
    bind_v1_kkt_corrected_work_identity,
)
from .modern_v1_neutral_runtime import DeclaredRouteAxis, FixedAffineBCECalibration
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTBlockCapacity,
    V1RelaxedKKTCUDAExecutionDesign,
    V1RelaxedKKTGeometryUncertaintySet,
    V1RelaxedKKTPolicy,
    V1RelaxedKKTProblem,
    V1RelaxedKKTResult,
    solve_v1_relaxed_kkt,
    solve_v1_relaxed_kkt_with_telemetry,
)
from .modern_v1_sampled_acquisition_adapters import (
    V1SampledPreFreezeGeometrySource,
)
from .modern_v1_sampled_geometric_provenance import (
    V1SampledProposalPositionCertificate,
    V1SampledSnapshotSelectorNumerics,
    certify_v1_sampled_proposal_position_correspondence,
    prepare_v1_sampled_snapshot_selector_numerics,
)


SAMPLED_KKT_BRANCH_FAMILY_GRAPH_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_kkt_branch_family_graph_v2"
)
SAMPLED_KKT_SOURCE_PROBLEM_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_sampled_dual_chart_kkt_source_problem_bundle_v1"
)
SAMPLED_KKT_BRANCH_FAMILY_AUTHORITY_SCOPE: Final = (
    "sampled_D_dual_chart_to_prefrozen_two_branch_hard_ladders_only_"
    "no_C_K_O_calibration_oracle_commit_terminal_or_pareto_authority_v1"
)
SAMPLED_KKT_SOURCE_PROBLEM_AUTHORITY_SCOPE: Final = (
    "sampled_D_dual_chart_to_frozen_relaxed_KKT_problem_only_"
    "no_solve_rounding_C_K_O_calibration_oracle_commit_terminal_or_"
    "pareto_authority_v1"
)
_GRAPH_PREFIX: Final = "partition-free-v1-sampled-kkt-branch-family-graph-"
_SOURCE_PROBLEM_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-source-problem-bundle-"
)
_CHART_PREFIX: Final = "partition-free-v1-sampled-kkt-chart-"
_PROBLEM_PREFIX: Final = "partition-free-v1-sampled-kkt-problem-"
_FREEZE_PREFIX: Final = "partition-free-v1-sampled-kkt-family-freeze-"
_BRANCH_FREEZE_PREFIX: Final = "partition-free-v1-fqr-kkt-branch-freeze-"
_ELIGIBILITY_PREFIX: Final = "partition-free-v1-sampled-kkt-eligibility-"
_RESOURCE_PREFIX: Final = "partition-free-v1-sampled-kkt-resource-"
_LEGALITY_PREFIX: Final = "partition-free-v1-sampled-kkt-legality-"
_REGISTERED_D_COORDINATE_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-registered-D-coordinate-"
)
_CORRECTED_WORK_CHART_BINDING_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-corrected-work-chart-binding-"
)
_CONFIDENCE_POINT_NAME: Final = "class_balanced_absolute_point"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly_f64_vector(value: object, *, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite route vector")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C"),
        dtype=np.float64,
    )


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string or None")
    return value


def _readonly_bool_vector(value: object, *, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype != np.bool_:
        raise ValueError("eligible_mask must be one Boolean route vector")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype=np.bool_).tobytes(order="C"),
        dtype=np.bool_,
    )


def _legality_blocks(
    axis: DeclaredRouteAxis,
    supplied: Sequence[V1ProductionLegalityBlock] | None,
) -> tuple[V1ProductionLegalityBlock, ...]:
    blocks = (
        tuple(
            V1ProductionLegalityBlock(
                block_id=str(block.layer_id),
                start=int(block.offset),
                stop=int(block.stop),
            )
            for block in axis.route_blocks
        )
        if supplied is None
        else tuple(supplied)
    )
    if not blocks or any(
        type(block) is not V1ProductionLegalityBlock for block in blocks
    ):
        raise TypeError("legality_blocks require exact compact legality blocks")
    expected_start = 0
    for block in blocks:
        if block.start != expected_start or block.stop <= block.start:
            raise ValueError("legality blocks must be contiguous and ordered")
        expected_start = block.stop
    if expected_start != axis.route_count:
        raise ValueError("legality blocks must cover the complete route axis")
    if len({block.block_id for block in blocks}) != len(blocks):
        raise ValueError("legality block IDs must be unique")
    return blocks


def _confidence_point_raw_sha256(view: V1ConfidenceBoxExternalView) -> str:
    descriptor = view.descriptor
    identities = {
        identity.logical_name: identity for identity in descriptor.external_state.arrays
    }
    try:
        point = identities[_CONFIDENCE_POINT_NAME]
    except KeyError as error:
        raise ValueError(
            "confidence-box descriptor lacks the corrected-work point"
        ) from error
    return point.sha256


def _corrected_work_chart_binding_hash(
    *,
    confidence_box_external_descriptor_hash: str,
    confidence_box_external_commit_hash: str,
    confidence_box_receipt_hash: str,
    point_raw_sha256: str,
    sampled_geometry_source_hash: str,
    sampled_snapshot_certificate_hash: str,
    sampled_snapshot_numerics_hash: str,
    proposal_position_certificate_hash: str,
    corrected_path_cost_hash: str,
    physical_route_axis_hash: str,
    ordered_route_axis_hash: str,
    route_count: int,
) -> str:
    return content_hash(
        {
            "confidence_box_external_descriptor_hash": (
                confidence_box_external_descriptor_hash
            ),
            "confidence_box_external_commit_hash": (
                confidence_box_external_commit_hash
            ),
            "confidence_box_receipt_hash": confidence_box_receipt_hash,
            "confidence_point_raw_sha256": point_raw_sha256,
            "sampled_geometry_source_hash": sampled_geometry_source_hash,
            "sampled_snapshot_certificate_hash": (sampled_snapshot_certificate_hash),
            "sampled_snapshot_numerics_hash": sampled_snapshot_numerics_hash,
            "proposal_position_certificate_hash": (proposal_position_certificate_hash),
            "corrected_path_cost_hash": corrected_path_cost_hash,
            "physical_route_axis_hash": physical_route_axis_hash,
            "ordered_route_axis_hash": ordered_route_axis_hash,
            "route_count": route_count,
            "confidence_point_coordinate": (
                "class_balanced_absolute_corrected_evidence"
            ),
        },
        prefix=_CORRECTED_WORK_CHART_BINDING_PREFIX,
    )


def _family_freeze_hash(
    *,
    problem_hash: str,
    policy_hash: str,
    result_hash: str,
    chart_identity_hash: str,
    corrected_work_identity_receipt_hash: str,
    discovery_evidence_hash: str,
) -> str:
    return content_hash(
        {
            "problem_hash": problem_hash,
            "policy_hash": policy_hash,
            "result_hash": result_hash,
            "chart_identity_hash": chart_identity_hash,
            "corrected_work_identity_receipt_hash": (
                corrected_work_identity_receipt_hash
            ),
            "discovery_evidence_hash": discovery_evidence_hash,
            "C_outcomes_seen": False,
        },
        prefix=_FREEZE_PREFIX,
    )


def _registered_d_coordinate_hash(
    *,
    source: V1SampledPreFreezeGeometrySource,
    numerics: V1SampledSnapshotSelectorNumerics,
) -> str:
    receipt = source.admission.receipt
    certificate = numerics.certificate
    return content_hash(
        {
            "sampled_geometry_source_hash": source.source_hash,
            "source_receipt_hash": receipt.receipt_hash,
            "observed_stratum_counts": receipt.observed_stratum_counts,
            "observed_draw_axis_hash": receipt.observed_draw_axis_hash,
            "observed_source_context_axis_hash": (
                receipt.observed_source_context_axis_hash
            ),
            "source_measurement_axis_hash": receipt.source_measurement_axis_hash,
            "sample_weight_hash": receipt.sample_weight_hash,
            "corrected_moments_state_hash": receipt.corrected_moments_state_hash,
            "fd_design_hash": receipt.fd_design_hash,
            "fd_snapshot_hash": receipt.fd_snapshot_hash,
            "snapshot_numerical_backend_hash": (
                receipt.snapshot_numerical_backend_hash
            ),
            "selector_numerics_hash": numerics.numerics_hash,
            "low_rank_factor_hash": certificate.low_rank_factor_hash,
            "residual_diagonal_hash": certificate.residual_diagonal_hash,
            "mean_row_count": certificate.mean_row_count,
            "centered_rank": certificate.centered_rank,
            "route_count": receipt.route_count,
            "same_registered_D_prefix_replay": True,
        },
        prefix=_REGISTERED_D_COORDINATE_PREFIX,
    )


def _sampled_chart_identity_hash(
    *,
    source: V1SampledPreFreezeGeometrySource,
    numerics: V1SampledSnapshotSelectorNumerics,
    position: V1SampledProposalPositionCertificate,
    calibration: FixedAffineBCECalibration,
    budgets: V1GeometricTrustBudgets,
    eligibility_hash: str,
    resource_hash: str,
    legality_hash: str,
    registered_d_coordinate_hash: str,
    corrected_work_chart_binding_hash: str,
) -> str:
    return content_hash(
        {
            "sampled_geometry_source_hash": source.source_hash,
            "sampled_snapshot_certificate_hash": (
                numerics.certificate.certificate_hash
            ),
            "proposal_position_certificate_hash": position.certificate_hash,
            "fixed_affine_bce_calibration_hash": calibration.content_hash,
            "geometric_trust_budget_hash": budgets.budget_hash,
            "eligibility_hash": eligibility_hash,
            "resource_hash": resource_hash,
            "legality_hash": legality_hash,
            "registered_d_coordinate_hash": registered_d_coordinate_hash,
            "corrected_work_chart_binding_hash": corrected_work_chart_binding_hash,
            "corrected_work_owner": "conserving_aumann_shapley_evidence",
            "quadratic_geometry_owner": "literal_relaxed_gate_jacobian_fd",
        },
        prefix=_CHART_PREFIX,
    )


def _sampled_problem_id(
    *,
    chart_identity_hash: str,
    policy: V1RelaxedKKTPolicy,
    budgets: V1GeometricTrustBudgets,
) -> str:
    return content_hash(
        {
            "chart_identity_hash": chart_identity_hash,
            "policy_hash": policy.policy_hash,
            "work_budget": budgets.finsler_path_budget,
            "quadratic_budget": budgets.quadratic_trust_energy_budget,
        },
        prefix=_PROBLEM_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTSourceProblemBundle:
    """Frozen sampled-D chart and relaxed problem before any numerical solve.

    This is the reusable joint between acquisition and either the small-domain
    reference solver or the production two-phase CUDA path.  It deliberately
    contains no relaxed primal, rounded endpoint, C-stream observation, or
    downstream authority.
    """

    sampled_geometry_source: V1SampledPreFreezeGeometrySource = field(
        repr=False, compare=False
    )
    selector_numerics: V1SampledSnapshotSelectorNumerics = field(
        repr=False, compare=False
    )
    proposal_position_certificate: V1SampledProposalPositionCertificate
    corrected_work_identity_receipt: V1KKTCorrectedWorkIdentityReceipt
    physical_route_axis: DeclaredRouteAxis = field(repr=False, compare=False)
    fixed_affine_bce_calibration: FixedAffineBCECalibration
    geometric_trust_budgets: V1GeometricTrustBudgets
    problem: V1RelaxedKKTProblem = field(repr=False, compare=False)
    policy: V1RelaxedKKTPolicy
    chart_identity_hash: str
    corrected_work_chart_binding_hash: str
    eligibility_hash: str
    resource_hash: str
    legality_hash: str
    registered_d_coordinate_hash: str
    route_scale_cpu_fallback_forbidden: bool = True
    solve_authority: bool = False
    rounding_authority: bool = False
    c_stream_outcomes_seen: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = SAMPLED_KKT_SOURCE_PROBLEM_AUTHORITY_SCOPE
    schema: str = SAMPLED_KKT_SOURCE_PROBLEM_BUNDLE_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.sampled_geometry_source) is not V1SampledPreFreezeGeometrySource:
            raise TypeError("sampled source problem requires its exact D source")
        if type(self.selector_numerics) is not V1SampledSnapshotSelectorNumerics:
            raise TypeError("sampled source problem requires exact selector numerics")
        if (
            type(self.proposal_position_certificate)
            is not V1SampledProposalPositionCertificate
        ):
            raise TypeError("sampled source problem requires exact position proof")
        if (
            type(self.corrected_work_identity_receipt)
            is not V1KKTCorrectedWorkIdentityReceipt
        ):
            raise TypeError(
                "sampled source problem requires exact corrected-work identity"
            )
        if type(self.physical_route_axis) is not DeclaredRouteAxis:
            raise TypeError("sampled source problem requires exact route axis")
        if (
            type(self.fixed_affine_bce_calibration)
            is not FixedAffineBCECalibration
        ):
            raise TypeError("sampled source problem requires exact calibration")
        if type(self.geometric_trust_budgets) is not V1GeometricTrustBudgets:
            raise TypeError("sampled source problem requires exact analytic budgets")
        if type(self.problem) is not V1RelaxedKKTProblem:
            raise TypeError("sampled source problem requires exact KKT problem")
        if type(self.policy) is not V1RelaxedKKTPolicy:
            raise TypeError("sampled source problem requires exact KKT policy")
        for name in (
            "chart_identity_hash",
            "corrected_work_chart_binding_hash",
            "eligibility_hash",
            "resource_hash",
            "legality_hash",
            "registered_d_coordinate_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        required_true = (self.route_scale_cpu_fallback_forbidden,)
        required_false = (
            self.solve_authority,
            self.rounding_authority,
            self.c_stream_outcomes_seen,
            self.calibration_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("sampled source-problem authority flags must be booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError(
                "sampled source problem crossed its pre-solve authority boundary"
            )
        source = self.sampled_geometry_source
        numerics = self.selector_numerics
        position = self.proposal_position_certificate
        identity = self.corrected_work_identity_receipt
        problem = self.problem
        calibration = self.fixed_affine_bce_calibration
        budgets = self.geometric_trust_budgets
        admission = source.admission
        receipt = admission.receipt
        current_mask = admission.current_mask
        identity.__post_init__()
        expected_work_binding_hash = _corrected_work_chart_binding_hash(
            confidence_box_external_descriptor_hash=(
                identity.confidence_box_external_descriptor_hash
            ),
            confidence_box_external_commit_hash=(
                identity.confidence_box_external_commit_hash
            ),
            confidence_box_receipt_hash=identity.confidence_box_receipt_hash,
            point_raw_sha256=identity.point_raw_sha256,
            sampled_geometry_source_hash=identity.sampled_geometry_source_hash,
            sampled_snapshot_certificate_hash=(
                identity.sampled_snapshot_certificate_hash
            ),
            sampled_snapshot_numerics_hash=(
                identity.sampled_snapshot_numerics_hash
            ),
            proposal_position_certificate_hash=(
                identity.proposal_position_certificate_hash
            ),
            corrected_path_cost_hash=identity.corrected_path_cost_hash,
            physical_route_axis_hash=identity.physical_route_axis_hash,
            ordered_route_axis_hash=identity.ordered_route_axis_hash,
            route_count=identity.route_count,
        )
        expected_eligibility_hash = content_hash(
            problem.eligible_mask, prefix=_ELIGIBILITY_PREFIX
        )
        expected_resource_hash = content_hash(
            problem.resource, prefix=_RESOURCE_PREFIX
        )
        expected_legality_hash = content_hash(
            problem.block_capacities, prefix=_LEGALITY_PREFIX
        )
        expected_registered_d_hash = _registered_d_coordinate_hash(
            source=source,
            numerics=numerics,
        )
        expected_chart_hash = _sampled_chart_identity_hash(
            source=source,
            numerics=numerics,
            position=position,
            calibration=calibration,
            budgets=budgets,
            eligibility_hash=self.eligibility_hash,
            resource_hash=self.resource_hash,
            legality_hash=self.legality_hash,
            registered_d_coordinate_hash=self.registered_d_coordinate_hash,
            corrected_work_chart_binding_hash=(
                self.corrected_work_chart_binding_hash
            ),
        )
        expected_problem_id = _sampled_problem_id(
            chart_identity_hash=self.chart_identity_hash,
            policy=self.policy,
            budgets=budgets,
        )
        if (
            self.schema != SAMPLED_KKT_SOURCE_PROBLEM_BUNDLE_SCHEMA
            or self.authority_scope != SAMPLED_KKT_SOURCE_PROBLEM_AUTHORITY_SCOPE
            or not admission.canonical_q_input_authority
            or source.geometry_state.metadata.get("canonical_q_input_authority")
            is not True
            or numerics.certificate.sampled_geometry_source_hash != source.source_hash
            or position.sampled_geometry_source_hash != source.source_hash
            or position.physical_route_axis_declaration_hash
            != self.physical_route_axis.content_hash
            or position.physical_route_axis_hash
            != self.physical_route_axis.route_axis_hash
            or position.ordered_route_axis_hash
            != self.physical_route_axis.route_id_axis_hash
            or position.route_count != self.physical_route_axis.route_count
            or current_mask.route_axis_hash
            != self.physical_route_axis.route_axis_hash
            or receipt.physical_route_axis_hash
            != self.physical_route_axis.route_axis_hash
            or receipt.ordered_route_axis_hash
            != self.physical_route_axis.route_id_axis_hash
            or identity.sampled_geometry_source_hash != source.source_hash
            or identity.sampled_snapshot_certificate_hash
            != numerics.certificate.certificate_hash
            or identity.sampled_snapshot_numerics_hash != numerics.numerics_hash
            or identity.proposal_position_certificate_hash != position.certificate_hash
            or identity.declared_kkt_problem_hash != problem.problem_hash
            or identity.kkt_problem_id != problem.problem_id
            or identity.kkt_source_geometry_hash != self.chart_identity_hash
            or problem.problem_id != expected_problem_id
            or problem.route_axis_hash != self.physical_route_axis.route_axis_hash
            or problem.source_geometry_hash != self.chart_identity_hash
            or problem.corrected_work is not numerics.corrected_path_cost
            or problem.low_rank_factor is not numerics.low_rank_factor
            or problem.residual_diagonal is not numerics.residual_diagonal
            or problem.work_budget != budgets.finsler_path_budget
            or problem.quadratic_budget
            != budgets.quadratic_trust_energy_budget
            or expected_work_binding_hash != self.corrected_work_chart_binding_hash
            or expected_eligibility_hash != self.eligibility_hash
            or expected_resource_hash != self.resource_hash
            or expected_legality_hash != self.legality_hash
            or expected_registered_d_hash != self.registered_d_coordinate_hash
            or expected_chart_hash != self.chart_identity_hash
        ):
            raise ValueError("sampled D source and frozen KKT problem do not close")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sampled_geometry_source_hash": self.sampled_geometry_source.source_hash,
            "selector_numerics_hash": self.selector_numerics.numerics_hash,
            "proposal_position_certificate_hash": (
                self.proposal_position_certificate.certificate_hash
            ),
            "corrected_work_identity_receipt_hash": (
                self.corrected_work_identity_receipt.receipt_hash
            ),
            "physical_route_axis_declaration_hash": (
                self.physical_route_axis.content_hash
            ),
            "fixed_affine_bce_calibration_hash": (
                self.fixed_affine_bce_calibration.content_hash
            ),
            "geometric_trust_budget_hash": self.geometric_trust_budgets.budget_hash,
            "problem_hash": self.problem.problem_hash,
            "policy_hash": self.policy.policy_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "corrected_work_chart_binding_hash": (
                self.corrected_work_chart_binding_hash
            ),
            "eligibility_hash": self.eligibility_hash,
            "resource_hash": self.resource_hash,
            "legality_hash": self.legality_hash,
            "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
            "route_scale_cpu_fallback_forbidden": (
                self.route_scale_cpu_fallback_forbidden
            ),
            "solve_authority": self.solve_authority,
            "rounding_authority": self.rounding_authority,
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def bundle_hash(self) -> str:
        cached = self._bundle_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_SOURCE_PROBLEM_PREFIX)
        if self._bundle_hash is not None and self._bundle_hash != observed:
            raise ValueError("sampled source problem was tampered after hashing")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("sampled source-problem hash does not match")
        object.__setattr__(self, "_bundle_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledKKTBranchFamilyGraph:
    """Closed, pre-C sampled-chart-to-KKT lineage graph."""

    sampled_geometry_source: V1SampledPreFreezeGeometrySource = field(
        repr=False, compare=False
    )
    selector_numerics: V1SampledSnapshotSelectorNumerics = field(
        repr=False, compare=False
    )
    proposal_position_certificate: V1SampledProposalPositionCertificate
    corrected_work_identity_receipt: V1KKTCorrectedWorkIdentityReceipt
    problem: V1RelaxedKKTProblem = field(repr=False, compare=False)
    policy: V1RelaxedKKTPolicy
    result: V1RelaxedKKTResult = field(repr=False, compare=False)
    branch_family: V1FQRKKTPreFrozenBranchFamily = field(repr=False, compare=False)
    chart_identity_hash: str
    corrected_work_chart_binding_hash: str
    family_freeze_receipt_hash: str
    eligibility_hash: str
    resource_hash: str
    legality_hash: str
    registered_d_coordinate_hash: str
    registered_d_observed_stratum_counts: tuple[int, int]
    registered_d_observed_draw_axis_hash: str
    registered_d_observed_source_context_axis_hash: str
    registered_d_source_measurement_axis_hash: str
    registered_d_sample_weight_hash: str
    registered_d_fd_design_hash: str
    registered_d_fd_snapshot_hash: str
    kkt_fd_rank: int
    cuda_execution_design_hash: str | None = None
    cuda_execution_bundle_hash: str | None = None
    cuda_telemetry_hash: str | None = None
    cuda_hbm_forecast_hash: str | None = None
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    adaptive_rank_witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...] = ()
    adaptive_rank_witnesses_carried: bool = False
    adaptive_rank_witness_missing_reason: str = (
        "missing_source_derived_literal_J_MJz_projection_or_D_census_receipt"
    )
    route_scale_cpu_fallback_forbidden: bool = True
    c_stream_outcomes_seen: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = SAMPLED_KKT_BRANCH_FAMILY_AUTHORITY_SCOPE
    schema: str = SAMPLED_KKT_BRANCH_FAMILY_GRAPH_SCHEMA
    _graph_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.sampled_geometry_source) is not V1SampledPreFreezeGeometrySource:
            raise TypeError("sampled KKT graph requires its exact geometry source")
        if type(self.selector_numerics) is not V1SampledSnapshotSelectorNumerics:
            raise TypeError("sampled KKT graph requires exact selector numerics")
        if (
            type(self.proposal_position_certificate)
            is not V1SampledProposalPositionCertificate
        ):
            raise TypeError("sampled KKT graph requires exact position certificate")
        if (
            type(self.corrected_work_identity_receipt)
            is not V1KKTCorrectedWorkIdentityReceipt
        ):
            raise TypeError(
                "sampled KKT graph requires the exact corrected-work identity receipt"
            )
        if type(self.problem) is not V1RelaxedKKTProblem:
            raise TypeError("sampled KKT graph requires an exact KKT problem")
        if type(self.policy) is not V1RelaxedKKTPolicy:
            raise TypeError("sampled KKT graph requires an exact KKT policy")
        if type(self.result) is not V1RelaxedKKTResult:
            raise TypeError("sampled KKT graph requires an exact KKT result")
        if type(self.branch_family) is not V1FQRKKTPreFrozenBranchFamily:
            raise TypeError("sampled KKT graph requires an exact branch family")
        for name in (
            "registered_d_coordinate_hash",
            "registered_d_observed_draw_axis_hash",
            "registered_d_observed_source_context_axis_hash",
            "registered_d_source_measurement_axis_hash",
            "registered_d_sample_weight_hash",
            "registered_d_fd_design_hash",
            "registered_d_fd_snapshot_hash",
            "adaptive_rank_witness_missing_reason",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        for name in (
            "cuda_execution_design_hash",
            "cuda_execution_bundle_hash",
            "cuda_telemetry_hash",
            "cuda_hbm_forecast_hash",
        ):
            object.__setattr__(
                self,
                name,
                _optional_identity(getattr(self, name), name=name),
            )
        if (
            type(self.registered_d_observed_stratum_counts) is not tuple
            or len(self.registered_d_observed_stratum_counts) != 2
            or any(
                not isinstance(item, Integral) or int(item) <= 0
                for item in self.registered_d_observed_stratum_counts
            )
        ):
            raise ValueError("registered D stratum counts must be one positive pair")
        object.__setattr__(
            self,
            "registered_d_observed_stratum_counts",
            tuple(int(item) for item in self.registered_d_observed_stratum_counts),
        )
        object.__setattr__(self, "kkt_fd_rank", _positive_int(self.kkt_fd_rank, name="kkt_fd_rank"))
        uncertainty = self.geometry_uncertainty
        if uncertainty is not None:
            if type(uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
                raise TypeError("sampled KKT graph requires exact uncertainty")
            if (
                uncertainty.problem_hash != self.problem.problem_hash
                or uncertainty.route_axis_hash != self.problem.route_axis_hash
                or uncertainty.source_geometry_hash != self.problem.source_geometry_hash
                or not uncertainty.source_derived_from_registered_geometry_receipts
            ):
                raise ValueError(
                    "sampled KKT uncertainty does not bind this source chart"
                )
        witnesses = tuple(self.adaptive_rank_witnesses)
        if any(
            type(value) is not V1RelaxedKKTAdaptiveRankBranchWitness
            for value in witnesses
        ):
            raise TypeError("sampled KKT graph requires exact adaptive-rank witnesses")
        if self.adaptive_rank_witnesses_carried is not bool(witnesses):
            raise ValueError("adaptive-rank witness availability flag is stale")
        if witnesses and uncertainty is None:
            raise PermissionError(
                "adaptive-rank witnesses require source-derived uncertainty"
            )
        if witnesses and tuple(value.branch.value for value in witnesses) != (
            "spectral",
            "diagonal_l1",
        ):
            raise ValueError(
                "adaptive-rank witnesses must use the canonical branch axis"
            )
        required_true = (self.route_scale_cpu_fallback_forbidden,)
        required_false = (
            self.c_stream_outcomes_seen,
            self.calibration_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("sampled KKT authority flags must be exact booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError(
                "sampled KKT graph crossed its pre-C proposal boundary"
            )
        source = self.sampled_geometry_source
        numerics = self.selector_numerics
        position = self.proposal_position_certificate
        identity = self.corrected_work_identity_receipt
        family = self.branch_family
        identity.__post_init__()
        expected_work_binding_hash = _corrected_work_chart_binding_hash(
            confidence_box_external_descriptor_hash=(
                identity.confidence_box_external_descriptor_hash
            ),
            confidence_box_external_commit_hash=(
                identity.confidence_box_external_commit_hash
            ),
            confidence_box_receipt_hash=identity.confidence_box_receipt_hash,
            point_raw_sha256=identity.point_raw_sha256,
            sampled_geometry_source_hash=identity.sampled_geometry_source_hash,
            sampled_snapshot_certificate_hash=(
                identity.sampled_snapshot_certificate_hash
            ),
            sampled_snapshot_numerics_hash=(identity.sampled_snapshot_numerics_hash),
            proposal_position_certificate_hash=(
                identity.proposal_position_certificate_hash
            ),
            corrected_path_cost_hash=identity.corrected_path_cost_hash,
            physical_route_axis_hash=identity.physical_route_axis_hash,
            ordered_route_axis_hash=identity.ordered_route_axis_hash,
            route_count=identity.route_count,
        )
        expected_family_freeze_hash = _family_freeze_hash(
            problem_hash=self.problem.problem_hash,
            policy_hash=self.policy.policy_hash,
            result_hash=self.result.result_hash,
            chart_identity_hash=self.chart_identity_hash,
            corrected_work_identity_receipt_hash=identity.receipt_hash,
            discovery_evidence_hash=family.discovery_evidence_hash,
        )
        expected_registered_d_hash = _registered_d_coordinate_hash(
            source=source,
            numerics=numerics,
        )
        if (
            self.schema != SAMPLED_KKT_BRANCH_FAMILY_GRAPH_SCHEMA
            or self.authority_scope != SAMPLED_KKT_BRANCH_FAMILY_AUTHORITY_SCOPE
            or numerics.certificate.sampled_geometry_source_hash != source.source_hash
            or position.sampled_geometry_source_hash != source.source_hash
            or identity.sampled_geometry_source_hash != source.source_hash
            or identity.sampled_snapshot_certificate_hash
            != numerics.certificate.certificate_hash
            or identity.sampled_snapshot_numerics_hash != numerics.numerics_hash
            or identity.proposal_position_certificate_hash != position.certificate_hash
            or identity.corrected_path_cost_hash
            != numerics.certificate.corrected_path_cost_hash
            or identity.physical_route_axis_declaration_hash
            != position.physical_route_axis_declaration_hash
            or identity.physical_route_axis_hash != position.physical_route_axis_hash
            or identity.ordered_route_axis_hash != position.ordered_route_axis_hash
            or identity.route_count != position.route_count
            or identity.declared_kkt_problem_hash != self.problem.problem_hash
            or identity.kkt_problem_id != self.problem.problem_id
            or identity.kkt_source_geometry_hash != self.chart_identity_hash
            or expected_work_binding_hash != self.corrected_work_chart_binding_hash
            or expected_family_freeze_hash != self.family_freeze_receipt_hash
            or expected_registered_d_hash != self.registered_d_coordinate_hash
            or self.registered_d_observed_stratum_counts
            != source.admission.receipt.observed_stratum_counts
            or self.registered_d_observed_draw_axis_hash
            != source.admission.receipt.observed_draw_axis_hash
            or self.registered_d_observed_source_context_axis_hash
            != source.admission.receipt.observed_source_context_axis_hash
            or self.registered_d_source_measurement_axis_hash
            != source.admission.receipt.source_measurement_axis_hash
            or self.registered_d_sample_weight_hash
            != source.admission.receipt.sample_weight_hash
            or self.registered_d_fd_design_hash
            != source.admission.receipt.fd_design_hash
            or self.registered_d_fd_snapshot_hash
            != source.admission.receipt.fd_snapshot_hash
            or self.kkt_fd_rank != self.problem.rank
            or self.problem.source_geometry_hash != self.chart_identity_hash
            or self.result.problem_hash != self.problem.problem_hash
            or self.result.policy_hash != self.policy.policy_hash
            or family.problem_hash != self.problem.problem_hash
            or family.policy_hash != self.policy.policy_hash
            or family.result_hash != self.result.result_hash
            or family.chart_identity_hash != self.chart_identity_hash
            or family.c_stream_outcomes_seen
        ):
            raise ValueError(
                "sampled KKT source, problem, result, and branch family do not close"
            )
        for binding in family.retained_bindings:
            expected_branch_freeze_hash = content_hash(
                {
                    "family_freeze_receipt_hash": self.family_freeze_receipt_hash,
                    "problem_hash": self.problem.problem_hash,
                    "policy_hash": self.policy.policy_hash,
                    "result_hash": self.result.result_hash,
                    "branch": binding.branch,
                    "branch_result_hash": binding.branch_result_hash,
                    "rounded_ladder_hash": binding.rounded_ladder_hash,
                    "physical_prefix_digests": (
                        binding.endpoint_physical_prefix_digests
                    ),
                    "issued_before_c_outcomes": True,
                },
                prefix=_BRANCH_FREEZE_PREFIX,
            )
            if binding.frozen_ladder.freeze_receipt_hash != expected_branch_freeze_hash:
                raise ValueError(
                    "sampled corrected-work identity does not bind a family branch"
                )
        witness_expected = {
            "spectral": (
                self.result.spectral.result_hash,
                self.result.spectral.receipt.receipt_hash,
                self.result.spectral.rounded_ladder.ladder_hash,
            ),
            "diagonal_l1": (
                self.result.diagonal_l1.result_hash,
                self.result.diagonal_l1.receipt.receipt_hash,
                self.result.diagonal_l1.rounded_ladder.ladder_hash,
            ),
        }
        for witness in witnesses:
            result_hash, receipt_hash, ladder_hash = witness_expected[
                witness.branch.value
            ]
            if (
                witness.problem_hash != self.problem.problem_hash
                or witness.policy_hash != self.policy.policy_hash
                or witness.branch_result_hash != result_hash
                or witness.branch_receipt_hash != receipt_hash
                or witness.rounded_ladder_hash != ladder_hash
                or witness.route_axis_hash != self.problem.route_axis_hash
                or witness.route_count != self.problem.route_count
                or (
                    uncertainty is not None
                    and witness.finite_population_geometry_uncertainty_set_hash
                    != uncertainty.uncertainty_hash
                )
                or (
                    uncertainty is not None
                    and witness.literal_j_fixed_primal_direction_axis_hash
                    != uncertainty.literal_j_fixed_primal_direction_axis_hash
                )
            ):
                raise ValueError("adaptive-rank witness does not bind this KKT graph")
        object.__setattr__(self, "adaptive_rank_witnesses", witnesses)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sampled_geometry_source_hash": self.sampled_geometry_source.source_hash,
            "selector_numerics_hash": self.selector_numerics.numerics_hash,
            "proposal_position_certificate_hash": (
                self.proposal_position_certificate.certificate_hash
            ),
            "corrected_work_identity_receipt_hash": (
                self.corrected_work_identity_receipt.receipt_hash
            ),
            "problem_hash": self.problem.problem_hash,
            "policy_hash": self.policy.policy_hash,
            "result_hash": self.result.result_hash,
            "branch_family_hash": self.branch_family.family_hash,
            "adaptive_rank_witness_hashes": tuple(
                value.witness_hash for value in self.adaptive_rank_witnesses
            ),
            "chart_identity_hash": self.chart_identity_hash,
            "corrected_work_chart_binding_hash": (
                self.corrected_work_chart_binding_hash
            ),
            "family_freeze_receipt_hash": self.family_freeze_receipt_hash,
            "eligibility_hash": self.eligibility_hash,
            "resource_hash": self.resource_hash,
            "legality_hash": self.legality_hash,
            "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
            "registered_d_observed_stratum_counts": (
                self.registered_d_observed_stratum_counts
            ),
            "registered_d_observed_draw_axis_hash": (
                self.registered_d_observed_draw_axis_hash
            ),
            "registered_d_observed_source_context_axis_hash": (
                self.registered_d_observed_source_context_axis_hash
            ),
            "registered_d_source_measurement_axis_hash": (
                self.registered_d_source_measurement_axis_hash
            ),
            "registered_d_sample_weight_hash": self.registered_d_sample_weight_hash,
            "registered_d_fd_design_hash": self.registered_d_fd_design_hash,
            "registered_d_fd_snapshot_hash": self.registered_d_fd_snapshot_hash,
            "kkt_fd_rank": self.kkt_fd_rank,
            "cuda_execution_design_hash": self.cuda_execution_design_hash,
            "cuda_execution_bundle_hash": self.cuda_execution_bundle_hash,
            "cuda_telemetry_hash": self.cuda_telemetry_hash,
            "cuda_hbm_forecast_hash": self.cuda_hbm_forecast_hash,
            "geometry_uncertainty_hash": (
                None
                if self.geometry_uncertainty is None
                else self.geometry_uncertainty.uncertainty_hash
            ),
            "route_scale_cpu_fallback_forbidden": self.route_scale_cpu_fallback_forbidden,
            "adaptive_rank_witnesses_carried": (
                self.adaptive_rank_witnesses_carried
            ),
            "adaptive_rank_witness_missing_reason": (
                self.adaptive_rank_witness_missing_reason
            ),
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def graph_hash(self) -> str:
        cached = self._graph_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.branch_family.assert_integrity()
        observed = content_hash(self._payload(), prefix=_GRAPH_PREFIX)
        if self._graph_hash is not None and self._graph_hash != observed:
            raise ValueError("sampled KKT graph was tampered after hashing")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("sampled KKT graph hash does not match")
        object.__setattr__(self, "_graph_hash", observed)
        return observed


def prepare_v1_sampled_kkt_source_problem(
    *,
    sampled_geometry_source: V1SampledPreFreezeGeometrySource,
    confidence_box_view: V1ConfidenceBoxExternalView,
    physical_route_axis: DeclaredRouteAxis,
    fixed_affine_bce_calibration: FixedAffineBCECalibration,
    geometric_trust_budgets: V1GeometricTrustBudgets,
    eligible_mask: Sequence[bool] | np.ndarray,
    resource: Sequence[float] | np.ndarray | float,
    policy: V1RelaxedKKTPolicy,
    legality_blocks: Sequence[V1ProductionLegalityBlock] | None = None,
    minimum_live_per_block: int = 1,
    production_route_scale_threshold: int = 1_000_000,
) -> V1SampledKKTSourceProblemBundle:
    """Freeze one sampled D chart into a reusable, authority-inert problem.

    The returned object is the common source for reference solving, nominal
    CUDA solving, independent literal-J witness acquisition, and an eventual
    authoritative CUDA replay.  It observes no solver state and therefore
    cannot itself authorize a relaxed face or a hard endpoint.
    """

    if type(sampled_geometry_source) is not V1SampledPreFreezeGeometrySource:
        raise TypeError("sampled KKT preparation requires an exact geometry source")
    if type(confidence_box_view) is not V1ConfidenceBoxExternalView:
        raise TypeError(
            "sampled KKT preparation requires an exact open confidence-box view"
        )
    if type(physical_route_axis) is not DeclaredRouteAxis:
        raise TypeError("sampled KKT preparation requires an exact route axis")
    if type(fixed_affine_bce_calibration) is not FixedAffineBCECalibration:
        raise TypeError("sampled KKT preparation requires exact fixed calibration")
    if type(geometric_trust_budgets) is not V1GeometricTrustBudgets:
        raise TypeError("sampled KKT preparation requires exact analytic budgets")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("sampled KKT preparation requires an exact KKT policy")
    minimum = _positive_int(minimum_live_per_block, name="minimum_live_per_block")
    if minimum != 1:
        raise ValueError("canonical v1 requires exactly one live survivor per block")
    route_scale = _positive_int(
        production_route_scale_threshold,
        name="production_route_scale_threshold",
    )

    source = sampled_geometry_source
    admission = source.admission
    receipt = admission.receipt
    current_mask = admission.current_mask
    contract = admission.binding.contract
    if not admission.canonical_q_input_authority or (
        source.geometry_state.metadata.get("canonical_q_input_authority") is not True
    ):
        raise PermissionError(
            "canonical KKT Q requires the same-capture/VJP literal-Jacobian stream"
        )
    if (
        physical_route_axis.route_axis_hash != current_mask.route_axis_hash
        or physical_route_axis.route_axis_hash != receipt.physical_route_axis_hash
        or physical_route_axis.route_id_axis_hash != receipt.ordered_route_axis_hash
        or physical_route_axis.route_count != current_mask.route_count
        or physical_route_axis.route_count != receipt.route_count
    ):
        raise ValueError("sampled KKT physical and ordered route axes do not close")
    calibration = fixed_affine_bce_calibration
    budgets = geometric_trust_budgets
    stale_coordinates: list[str] = []
    if calibration.query_id != contract.query_id:
        stale_coordinates.append("query_id")
    if calibration.calibration_state_id != contract.calibration_identity:
        stale_coordinates.append("calibration_state_id")
    if calibration.terminal_outcomes_opened:
        stale_coordinates.append("terminal_outcomes_opened")
    if budgets.current_mask_hash != current_mask.mask_hash:
        stale_coordinates.append("current_mask_hash")
    if (
        budgets.damage_target_derivation_hash
        != contract.damage_target_derivation_hash
    ):
        stale_coordinates.append("damage_target_derivation_hash")
    if not same_geometric_coordinate(
        budgets.raw_logit_to_calibrated_logit_slope,
        calibration.scale,
    ):
        stale_coordinates.append("raw_logit_to_calibrated_logit_slope")
    if stale_coordinates:
        raise ValueError(
            "sampled KKT calibration or analytic budgets are stale: "
            + ", ".join(stale_coordinates)
        )
    if current_mask.route_count >= route_scale and policy.backend != "torch_cuda":
        raise RuntimeError(
            "production-scale sampled KKT preparation requires torch_cuda; "
            "CPU/NumPy fallback is forbidden"
        )

    numerics = prepare_v1_sampled_snapshot_selector_numerics(source)
    position = certify_v1_sampled_proposal_position_correspondence(
        numerics=numerics,
        physical_route_axis=physical_route_axis,
    )
    confidence_descriptor = confidence_box_view.descriptor
    corrected_work_chart_binding_hash = _corrected_work_chart_binding_hash(
        confidence_box_external_descriptor_hash=confidence_descriptor.descriptor_hash,
        confidence_box_external_commit_hash=confidence_descriptor.commit_hash,
        confidence_box_receipt_hash=confidence_descriptor.receipt_hash,
        point_raw_sha256=_confidence_point_raw_sha256(confidence_box_view),
        sampled_geometry_source_hash=source.source_hash,
        sampled_snapshot_certificate_hash=numerics.certificate.certificate_hash,
        sampled_snapshot_numerics_hash=numerics.numerics_hash,
        proposal_position_certificate_hash=position.certificate_hash,
        corrected_path_cost_hash=numerics.certificate.corrected_path_cost_hash,
        physical_route_axis_hash=position.physical_route_axis_hash,
        ordered_route_axis_hash=position.ordered_route_axis_hash,
        route_count=position.route_count,
    )
    eligible = _readonly_bool_vector(eligible_mask, size=current_mask.route_count)
    live = current_mask.to_bool_array()
    if bool(np.any(eligible & ~live)):
        raise ValueError("eligible_mask may contain only currently live routes")
    if not bool(np.any(eligible)):
        raise ValueError("sampled KKT eligible set is empty")
    if isinstance(resource, Real) and not isinstance(resource, bool):
        scalar = float(resource)
        if not math.isfinite(scalar) or scalar <= 0.0:
            raise ValueError("uniform resource must be finite and positive")
        resource_vector = _readonly_f64_vector(
            np.full(current_mask.route_count, scalar, dtype=np.float64),
            size=current_mask.route_count,
            name="resource",
        )
    else:
        resource_vector = _readonly_f64_vector(
            resource,
            size=current_mask.route_count,
            name="resource",
        )
        if bool(np.any(resource_vector[eligible] <= 0.0)):
            raise ValueError("resource must be positive on every eligible route")
    blocks = _legality_blocks(physical_route_axis, legality_blocks)
    capacities = tuple(
        V1RelaxedKKTBlockCapacity(
            block_id=block.block_id,
            start=block.start,
            stop=block.stop,
            removal_capacity=max(
                0,
                int(np.count_nonzero(live[block.start : block.stop])) - minimum,
            ),
        )
        for block in blocks
    )
    if (
        sum(
            min(
                capacity.removal_capacity,
                int(np.count_nonzero(eligible[capacity.start : capacity.stop])),
            )
            for capacity in capacities
        )
        <= 0
    ):
        raise ValueError("sampled KKT survivor constraints leave no legal removal")

    eligibility_hash = content_hash(eligible, prefix=_ELIGIBILITY_PREFIX)
    resource_hash = content_hash(resource_vector, prefix=_RESOURCE_PREFIX)
    legality_hash = content_hash(capacities, prefix=_LEGALITY_PREFIX)
    registered_d_hash = _registered_d_coordinate_hash(
        source=source,
        numerics=numerics,
    )
    chart_identity_hash = _sampled_chart_identity_hash(
        source=source,
        numerics=numerics,
        position=position,
        calibration=calibration,
        budgets=budgets,
        eligibility_hash=eligibility_hash,
        resource_hash=resource_hash,
        legality_hash=legality_hash,
        registered_d_coordinate_hash=registered_d_hash,
        corrected_work_chart_binding_hash=corrected_work_chart_binding_hash,
    )
    problem = V1RelaxedKKTProblem(
        problem_id=_sampled_problem_id(
            chart_identity_hash=chart_identity_hash,
            policy=policy,
            budgets=budgets,
        ),
        resource=resource_vector,
        corrected_work=numerics.corrected_path_cost,
        low_rank_factor=numerics.low_rank_factor,
        residual_diagonal=numerics.residual_diagonal,
        residual_spectral_bound=numerics.certificate.residual_spectral_bound,
        eligible_mask=eligible,
        work_budget=budgets.finsler_path_budget,
        quadratic_budget=budgets.quadratic_trust_energy_budget,
        route_axis_hash=physical_route_axis.route_axis_hash,
        source_geometry_hash=chart_identity_hash,
        block_capacities=capacities,
        input_array_policy="admit_readonly_c_contiguous",
    )
    corrected_work_identity = bind_v1_kkt_corrected_work_identity(
        confidence_box_view=confidence_box_view,
        sampled_geometry_source=source,
        selector_numerics=numerics,
        proposal_position_certificate=position,
        problem=problem,
    )
    return V1SampledKKTSourceProblemBundle(
        sampled_geometry_source=source,
        selector_numerics=numerics,
        proposal_position_certificate=position,
        corrected_work_identity_receipt=corrected_work_identity,
        physical_route_axis=physical_route_axis,
        fixed_affine_bce_calibration=calibration,
        geometric_trust_budgets=budgets,
        problem=problem,
        policy=policy,
        chart_identity_hash=chart_identity_hash,
        corrected_work_chart_binding_hash=corrected_work_chart_binding_hash,
        eligibility_hash=eligibility_hash,
        resource_hash=resource_hash,
        legality_hash=legality_hash,
        registered_d_coordinate_hash=registered_d_hash,
    )


def solve_and_freeze_v1_sampled_kkt_source_problem(
    *,
    prepared_source_problem: V1SampledKKTSourceProblemBundle,
    cuda_execution_design: V1RelaxedKKTCUDAExecutionDesign | None = None,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
) -> V1SampledKKTBranchFamilyGraph:
    """Solve one already-frozen sampled problem and freeze its ordinary family."""

    if type(prepared_source_problem) is not V1SampledKKTSourceProblemBundle:
        raise TypeError("prepared_source_problem must use the exact bundle type")
    prepared = prepared_source_problem
    prepared.assert_integrity(prepared.bundle_hash)
    if cuda_execution_design is not None and type(cuda_execution_design) is not (
        V1RelaxedKKTCUDAExecutionDesign
    ):
        raise TypeError("cuda_execution_design must be exact")
    if geometry_uncertainty is not None and type(geometry_uncertainty) is not (
        V1RelaxedKKTGeometryUncertaintySet
    ):
        raise TypeError("geometry_uncertainty must be exact")

    policy = prepared.policy
    if policy.backend == "torch_cuda" and cuda_execution_design is None:
        raise RuntimeError(
            "sampled torch_cuda KKT construction requires an explicit fused CUDA "
            "execution design; the historical implicit CUDA solve is forbidden"
        )
    if policy.backend != "torch_cuda" and cuda_execution_design is not None:
        raise ValueError(
            "cuda_execution_design is valid only for the torch_cuda backend"
        )

    source = prepared.sampled_geometry_source
    admission = source.admission
    receipt = admission.receipt
    current_mask = admission.current_mask
    numerics = prepared.selector_numerics
    position = prepared.proposal_position_certificate
    corrected_work_identity = prepared.corrected_work_identity_receipt
    problem = prepared.problem
    chart_identity_hash = prepared.chart_identity_hash
    corrected_work_chart_binding_hash = (
        prepared.corrected_work_chart_binding_hash
    )
    eligibility_hash = prepared.eligibility_hash
    resource_hash = prepared.resource_hash
    legality_hash = prepared.legality_hash
    registered_d_hash = prepared.registered_d_coordinate_hash

    cuda_execution_bundle_hash: str | None = None
    cuda_telemetry_hash: str | None = None
    cuda_hbm_forecast_hash: str | None = None
    adaptive_rank_witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...] = ()
    if policy.backend == "torch_cuda":
        if cuda_execution_design is None:  # pragma: no cover - guarded above.
            raise RuntimeError("explicit fused CUDA design was not retained")
        cuda_bundle = solve_v1_relaxed_kkt_with_telemetry(
            problem=problem,
            policy=policy,
            execution_design=cuda_execution_design,
            geometry_uncertainty=geometry_uncertainty,
        )
        result = cuda_bundle.result
        cuda_execution_bundle_hash = cuda_bundle.bundle_hash
        cuda_telemetry_hash = cuda_bundle.telemetry.telemetry_hash
        cuda_hbm_forecast_hash = cuda_bundle.forecast.forecast_hash
        adaptive_rank_witnesses = cuda_bundle.adaptive_rank_witnesses
    else:
        if geometry_uncertainty is not None:
            raise PermissionError(
                "source-derived adaptive-rank witnesses require fused private "
                "state; numpy_reference can only build the small-domain "
                "proposal family"
            )
        result = solve_v1_relaxed_kkt(problem=problem, policy=policy)
    if not result.both_branches_converged:
        raise RuntimeError(
            "canonical sampled KKT requires both convex branches to converge"
        )
    moments = admission.moments
    discovery_context_ids = tuple(
        context for axis in moments.observed_context_ids_by_stratum for context in axis
    )
    discovery_draw_ids = tuple(
        draw for axis in moments.observed_draw_ids_by_stratum for draw in axis
    )
    family_freeze_receipt_hash = _family_freeze_hash(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        result_hash=result.result_hash,
        chart_identity_hash=chart_identity_hash,
        corrected_work_identity_receipt_hash=(corrected_work_identity.receipt_hash),
        discovery_evidence_hash=receipt.corrected_moments_state_hash,
    )
    family = freeze_v1_relaxed_kkt_branch_family(
        problem=problem,
        policy=policy,
        result=result,
        current_mask=current_mask,
        chart_identity_hash=chart_identity_hash,
        proposal_generation=receipt.proposal_generation,
        campaign_contract_hash=receipt.campaign_contract_hash,
        model_identity_hash=receipt.model_identity_hash,
        query_contract_hash=receipt.query_contract_hash,
        preprocessing_hash=receipt.preprocessing_hash,
        raw_logit_stream_source_hash=receipt.raw_logit_stream_source_hash,
        sampling_law_hash=receipt.sampling_law_hash,
        discovery_context_ids=discovery_context_ids,
        discovery_draw_ids=discovery_draw_ids,
        discovery_evidence_hash=receipt.corrected_moments_state_hash,
        family_freeze_receipt_hash=family_freeze_receipt_hash,
    )
    return V1SampledKKTBranchFamilyGraph(
        sampled_geometry_source=source,
        selector_numerics=numerics,
        proposal_position_certificate=position,
        corrected_work_identity_receipt=corrected_work_identity,
        problem=problem,
        policy=policy,
        result=result,
        branch_family=family,
        chart_identity_hash=chart_identity_hash,
        corrected_work_chart_binding_hash=corrected_work_chart_binding_hash,
        family_freeze_receipt_hash=family_freeze_receipt_hash,
        eligibility_hash=eligibility_hash,
        resource_hash=resource_hash,
        legality_hash=legality_hash,
        registered_d_coordinate_hash=registered_d_hash,
        registered_d_observed_stratum_counts=receipt.observed_stratum_counts,
        registered_d_observed_draw_axis_hash=receipt.observed_draw_axis_hash,
        registered_d_observed_source_context_axis_hash=(
            receipt.observed_source_context_axis_hash
        ),
        registered_d_source_measurement_axis_hash=(
            receipt.source_measurement_axis_hash
        ),
        registered_d_sample_weight_hash=receipt.sample_weight_hash,
        registered_d_fd_design_hash=receipt.fd_design_hash,
        registered_d_fd_snapshot_hash=receipt.fd_snapshot_hash,
        kkt_fd_rank=problem.rank,
        cuda_execution_design_hash=(
            None if cuda_execution_design is None else cuda_execution_design.design_hash
        ),
        cuda_execution_bundle_hash=cuda_execution_bundle_hash,
        cuda_telemetry_hash=cuda_telemetry_hash,
        cuda_hbm_forecast_hash=cuda_hbm_forecast_hash,
        geometry_uncertainty=geometry_uncertainty,
        adaptive_rank_witnesses=adaptive_rank_witnesses,
        adaptive_rank_witnesses_carried=bool(adaptive_rank_witnesses),
        adaptive_rank_witness_missing_reason=(
            "source_derived_literal_J_MJz_projection_or_D_census_receipt_bound"
            if adaptive_rank_witnesses
            else "missing_source_derived_literal_J_MJz_projection_or_D_census_receipt"
        ),
    )


def build_v1_sampled_kkt_branch_family_graph(
    *,
    sampled_geometry_source: V1SampledPreFreezeGeometrySource,
    confidence_box_view: V1ConfidenceBoxExternalView,
    physical_route_axis: DeclaredRouteAxis,
    fixed_affine_bce_calibration: FixedAffineBCECalibration,
    geometric_trust_budgets: V1GeometricTrustBudgets,
    eligible_mask: Sequence[bool] | np.ndarray,
    resource: Sequence[float] | np.ndarray | float,
    policy: V1RelaxedKKTPolicy,
    legality_blocks: Sequence[V1ProductionLegalityBlock] | None = None,
    cuda_execution_design: V1RelaxedKKTCUDAExecutionDesign | None = None,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet | None = None,
    minimum_live_per_block: int = 1,
    production_route_scale_threshold: int = 1_000_000,
) -> V1SampledKKTBranchFamilyGraph:
    """Solve and freeze both KKT branches from one canonical sampled D chart."""

    if cuda_execution_design is not None and type(cuda_execution_design) is not (
        V1RelaxedKKTCUDAExecutionDesign
    ):
        raise TypeError("cuda_execution_design must be exact")
    if geometry_uncertainty is not None and type(geometry_uncertainty) is not (
        V1RelaxedKKTGeometryUncertaintySet
    ):
        raise TypeError("geometry_uncertainty must be exact")
    prepared = prepare_v1_sampled_kkt_source_problem(
        sampled_geometry_source=sampled_geometry_source,
        confidence_box_view=confidence_box_view,
        physical_route_axis=physical_route_axis,
        fixed_affine_bce_calibration=fixed_affine_bce_calibration,
        geometric_trust_budgets=geometric_trust_budgets,
        eligible_mask=eligible_mask,
        resource=resource,
        policy=policy,
        legality_blocks=legality_blocks,
        minimum_live_per_block=minimum_live_per_block,
        production_route_scale_threshold=production_route_scale_threshold,
    )
    return solve_and_freeze_v1_sampled_kkt_source_problem(
        prepared_source_problem=prepared,
        cuda_execution_design=cuda_execution_design,
        geometry_uncertainty=geometry_uncertainty,
    )


__all__ = [
    "SAMPLED_KKT_BRANCH_FAMILY_AUTHORITY_SCOPE",
    "SAMPLED_KKT_BRANCH_FAMILY_GRAPH_SCHEMA",
    "SAMPLED_KKT_SOURCE_PROBLEM_AUTHORITY_SCOPE",
    "SAMPLED_KKT_SOURCE_PROBLEM_BUNDLE_SCHEMA",
    "V1SampledKKTBranchFamilyGraph",
    "V1SampledKKTSourceProblemBundle",
    "build_v1_sampled_kkt_branch_family_graph",
    "prepare_v1_sampled_kkt_source_problem",
    "solve_and_freeze_v1_sampled_kkt_source_problem",
]
