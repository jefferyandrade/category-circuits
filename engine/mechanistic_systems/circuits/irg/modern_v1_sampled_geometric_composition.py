"""Factory for one closed sampled-geometric proposal composition graph.

This module is intentionally orchestration-only.  It composes the landed
sampled source, provenance, selector, freeze, and neutral-scope contracts
without strengthening any of their authority.  Full-axis arrays are made
immutable once and admitted by alias thereafter; the graph never constructs a
second route-sized selector payload.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Final, Literal, NoReturn

import numpy as np

from .contracts import content_hash
from .modern_v1_composition_scope import (
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
    V1CompositionDecisionScopeReceipt,
)
from .modern_v1_geometric_budgets import (
    V1GeometricTrustBudgets,
    same_geometric_coordinate,
)
from .modern_v1_geometric_selector_production import (
    PRODUCTION_SELECTOR_CANONICAL_POLICY_ID,
    PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID,
    PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND,
    V1ProductionGeometricSelectorPolicy,
    V1ProductionGeometricSelectorProblem,
    V1ProductionGeometricSelectorResult,
    V1ProductionLegalityBlock,
    V1ProductionScalarization,
    select_v1_production_geometric_frontier,
)
from .modern_v1_neutral_runtime import (
    DeclaredRouteAxis,
    FixedAffineBCECalibration,
)
from .modern_v1_sampled_acquisition_adapters import (
    V1SampledPreFreezeGeometrySource,
)
from .modern_v1_sampled_geometric_freeze import (
    V1SampledGeometricFreeze,
    V1SampledGeometricFreezeIdentity,
    composition_scope_from_v1_sampled_geometric_freeze,
    freeze_v1_sampled_geometric_selector_ladder,
)
from .modern_v1_sampled_geometric_provenance import (
    V1SampledProposalPositionCertificate,
    V1SampledScoreDiscoveryCoordinates,
    V1SampledSelectorChartProvenance,
    V1SampledSnapshotSelectorNumerics,
    build_v1_sampled_selector_chart_provenance,
    certify_v1_sampled_proposal_position_correspondence,
    prepare_v1_sampled_snapshot_selector_numerics,
)


SAMPLED_GEOMETRIC_COMPOSITION_POLICY_INPUTS_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_composition_policy_inputs_v1"
)
SAMPLED_GEOMETRIC_ELIGIBILITY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_eligibility_receipt_v1"
)
SAMPLED_GEOMETRIC_COMPOSITION_GRAPH_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_composition_graph_v1"
)
SAMPLED_GEOMETRIC_COMPOSITION_AUTHORITY_SCOPE: Final = (
    "sampled_source_to_frozen_competitive_proposal_scope_only_no_score_regret_"
    "fq_bce_mask_commit_production_or_terminal_authority_v1"
)
SAMPLED_GEOMETRIC_EXPLICIT_COUNT_LADDER_SCOPE: Final = (
    "explicit_campaign_step_count_ladder_no_diagnostic_default_v1"
)

_POLICY_INPUTS_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-composition-policy-inputs-"
)
_ELIGIBILITY_MASK_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-eligible-mask-"
)
_ELIGIBILITY_RECEIPT_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-eligibility-receipt-"
)
_LEGALITY_POLICY_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-legality-policy-"
)
_BUILD_REQUEST_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-composition-build-request-"
)
_PROBLEM_ID_PREFIX: Final = (
    "partition-free-v1-sampled-geometric-composition-problem-id-"
)
_GRAPH_PREFIX: Final = "partition-free-v1-sampled-geometric-composition-graph-"
_GRAPH_ISSUANCE_TOKEN = object()

SelectorBackend = Literal["numpy", "torch"]
SelectorTiePolicy = Literal[
    "refuse_threshold_ties",
    "physical_index_deterministic_reference",
]


class V1SampledGeometricCompositionRefusalCode(str, Enum):
    """Typed fail-closed outcomes before a frozen graph can exist."""

    NO_CAMPAIGN_REMOVALS_REMAIN = "no_campaign_removals_remain"
    EMPTY_SELECTOR_FRONTIER = "empty_selector_frontier"


class V1SampledGeometricCompositionRefusal(ValueError):
    """Intentional refusal to misrepresent a partial graph as a frozen one."""

    def __init__(
        self,
        code: V1SampledGeometricCompositionRefusalCode,
        message: str,
        *,
        selector_result: V1ProductionGeometricSelectorResult | None = None,
    ) -> None:
        self.code = code
        self.selector_result = selector_result
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledGeometricCompositionRefusalCode,
    message: str,
    *,
    selector_result: V1ProductionGeometricSelectorResult | None = None,
) -> NoReturn:
    raise V1SampledGeometricCompositionRefusal(
        code,
        message,
        selector_result=selector_result,
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _nonnegative_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _immutable_bool_vector(value: object, *, size: int) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if array.size != size:
        raise ValueError("eligibility mask differs from the physical route axis")
    if array.dtype == np.dtype(np.bool_) and array.flags.c_contiguous:
        if not array.flags.writeable:
            return array
        contiguous = array
    else:
        try:
            numeric = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("eligibility mask must be binary") from error
        if numeric.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("eligibility mask must be binary")
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0, 1)).all():
            raise ValueError("eligibility mask must be binary")
        contiguous = np.ascontiguousarray(numeric.reshape(-1), dtype=np.bool_)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=np.bool_)


def _immutable_f64_vector(value: object, *, size: int) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if array.size != size:
        raise ValueError("resource cost differs from the physical route axis")
    if (
        array.dtype == np.dtype(np.float64)
        and array.flags.c_contiguous
        and not array.flags.writeable
    ):
        result = array
    else:
        try:
            contiguous = np.ascontiguousarray(array, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("resource cost must be float64") from error
        result = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    if not np.isfinite(result).all() or bool(np.any(result < 0.0)):
        raise ValueError("resource cost must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class V1SampledGeometricCompositionPolicyInputs:
    """Canonical selector and bounded freeze execution inputs.

    ``count_ladder`` is deliberately required.  The selector's library default
    stops at 128 and is diagnostic-scale for an ImageNet-sized route axis; a
    production caller must bind its campaign-specific step ladder explicitly.
    """

    count_ladder: Sequence[int]
    backend: SelectorBackend = "numpy"
    torch_device: str = "cpu"
    maximum_removal_count: int | None = None
    chunk_size: int = 1 << 16
    max_frontier_candidates: int = 128
    dual_iterations: int = 48
    dual_initial_step_size: float = 0.75
    tie_tolerance: float = 1e-12
    tie_policy: SelectorTiePolicy = "physical_index_deterministic_reference"
    packed_freeze_byte_chunk_size: int = 1 << 20
    schema: str = SAMPLED_GEOMETRIC_COMPOSITION_POLICY_INPUTS_SCHEMA

    def __post_init__(self) -> None:
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("unsupported sampled composition selector backend")
        object.__setattr__(
            self,
            "torch_device",
            _identity(self.torch_device, name="selector torch device"),
        )
        counts = tuple(
            _positive_int(value, name="selector count-ladder item")
            for value in self.count_ladder
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            raise ValueError("selector count ladder must be strictly increasing")
        object.__setattr__(self, "count_ladder", counts)
        if self.maximum_removal_count is not None:
            object.__setattr__(
                self,
                "maximum_removal_count",
                _positive_int(
                    self.maximum_removal_count,
                    name="maximum removal count",
                ),
            )
        for name in (
            "chunk_size",
            "max_frontier_candidates",
            "dual_iterations",
            "packed_freeze_byte_chunk_size",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "dual_initial_step_size",
            _positive_float(
                self.dual_initial_step_size,
                name="dual initial step size",
            ),
        )
        object.__setattr__(
            self,
            "tie_tolerance",
            _nonnegative_float(self.tie_tolerance, name="tie tolerance"),
        )
        if self.tie_policy not in {
            "refuse_threshold_ties",
            "physical_index_deterministic_reference",
        }:
            raise ValueError("unsupported sampled composition tie policy")
        if self.schema != SAMPLED_GEOMETRIC_COMPOSITION_POLICY_INPUTS_SCHEMA:
            raise ValueError("unsupported sampled composition policy-input schema")

    @property
    def inputs_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_INPUTS_PREFIX)

    def build_selector_policy(
        self,
        *,
        route_count: int,
        remaining_campaign_removal_count: int,
        build_request_hash: str,
    ) -> V1ProductionGeometricSelectorPolicy:
        routes = _positive_int(route_count, name="selector route count")
        if isinstance(remaining_campaign_removal_count, bool) or not isinstance(
            remaining_campaign_removal_count,
            Integral,
        ):
            raise TypeError("remaining campaign removal count must be an integer")
        remaining = int(remaining_campaign_removal_count)
        if remaining <= 0:
            _refuse(
                V1SampledGeometricCompositionRefusalCode.NO_CAMPAIGN_REMOVALS_REMAIN,
                "the campaign contract admits no further irreversible removal",
            )
        if remaining > routes:
            raise ValueError("campaign removal remainder exceeds the route axis")
        requested = self.maximum_removal_count
        if requested is not None and requested > remaining:
            raise ValueError(
                "selector maximum removal count exceeds the campaign remainder"
            )
        maximum = remaining if requested is None else requested
        return V1ProductionGeometricSelectorPolicy(
            policy_id=PRODUCTION_SELECTOR_CANONICAL_POLICY_ID,
            count_ladder=self.count_ladder,
            scalarizations=(
                V1ProductionScalarization(
                    scalarization_id=(
                        PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID
                    ),
                ),
            ),
            backend=self.backend,
            torch_device=self.torch_device,
            chunk_size=self.chunk_size,
            max_frontier_candidates=self.max_frontier_candidates,
            maximum_removal_count=maximum,
            dual_iterations=self.dual_iterations,
            dual_initial_step_size=self.dual_initial_step_size,
            tie_tolerance=self.tie_tolerance,
            tie_policy=self.tie_policy,
            exact_reference_route_limit=0,
            input_array_policy="admit_readonly_c_contiguous",
            diagnostic_only=False,
            metadata={
                "sampled_geometric_composition_build_request_hash": (
                    build_request_hash
                ),
                "sampled_geometric_composition_policy_inputs_hash": (
                    self.inputs_hash
                ),
                "sampled_geometric_explicit_count_ladder_scope": (
                    SAMPLED_GEOMETRIC_EXPLICIT_COUNT_LADDER_SCOPE
                ),
                "sampled_geometric_declared_count_ladder": self.count_ladder,
                "sampled_geometric_route_count": routes,
                "sampled_geometric_remaining_campaign_removal_count": remaining,
            },
        )


@dataclass(frozen=True, slots=True)
class V1SampledGeometricEligibilityReceipt:
    """Current-chart eligibility identity with proposal authority only."""

    sampled_geometry_source_hash: str
    sampled_snapshot_certificate_hash: str
    eligibility_policy_hash: str
    eligible_mask_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    proposal_eligibility_authority: bool = True
    irreversible_order_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = SAMPLED_GEOMETRIC_ELIGIBILITY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "sampled_geometry_source_hash",
            "sampled_snapshot_certificate_hash",
            "eligibility_policy_hash",
            "eligible_mask_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="eligibility route count"),
        )
        if isinstance(self.proposal_generation, bool) or not isinstance(
            self.proposal_generation,
            Integral,
        ):
            raise TypeError("eligibility proposal generation must be nonnegative")
        generation = int(self.proposal_generation)
        if generation < 0:
            raise ValueError("eligibility proposal generation must be nonnegative")
        object.__setattr__(self, "proposal_generation", generation)
        required_false = (
            self.irreversible_order_authority,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(
            type(value) is not bool
            for value in (self.proposal_eligibility_authority, *required_false)
        ):
            raise TypeError("eligibility authority fields must be exact booleans")
        if self.proposal_eligibility_authority is not True or any(required_false):
            raise PermissionError("eligibility receipt crossed its proposal boundary")
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError("eligibility receipt must retain distinct route axes")
        if self.schema != SAMPLED_GEOMETRIC_ELIGIBILITY_RECEIPT_SCHEMA:
            raise ValueError("unsupported sampled eligibility receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_ELIGIBILITY_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True, init=False)
class V1SampledGeometricCompositionGraph:
    """Factory-issued exact source-to-neutral-scope composition graph."""

    build_request_hash: str
    sampled_geometry_source: V1SampledPreFreezeGeometrySource = field(
        repr=False,
        compare=False,
    )
    physical_route_axis: DeclaredRouteAxis = field(repr=False, compare=False)
    score_discovery_coordinates: V1SampledScoreDiscoveryCoordinates
    fixed_affine_bce_calibration: FixedAffineBCECalibration
    geometric_trust_budgets: V1GeometricTrustBudgets
    policy_inputs: V1SampledGeometricCompositionPolicyInputs
    eligibility_receipt: V1SampledGeometricEligibilityReceipt
    selector_numerics: V1SampledSnapshotSelectorNumerics = field(
        repr=False,
        compare=False,
    )
    proposal_position_certificate: V1SampledProposalPositionCertificate
    selector_chart_provenance: V1SampledSelectorChartProvenance
    selector_problem: V1ProductionGeometricSelectorProblem = field(
        repr=False,
        compare=False,
    )
    selector_policy: V1ProductionGeometricSelectorPolicy
    selector_result: V1ProductionGeometricSelectorResult = field(
        repr=False,
        compare=False,
    )
    sampled_geometric_freeze: V1SampledGeometricFreeze = field(
        repr=False,
        compare=False,
    )
    composition_scope: V1CompositionDecisionScopeReceipt
    legality_policy_hash: str
    resource_cost_hash: str
    proposal_only: bool = True
    all_endpoints_composition_frozen: bool = True
    all_endpoints_decision_competitive: bool = True
    score_zero_regret_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_execution_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = SAMPLED_GEOMETRIC_COMPOSITION_AUTHORITY_SCOPE
    schema: str = SAMPLED_GEOMETRIC_COMPOSITION_GRAPH_SCHEMA
    _graph_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(
        self,
        *,
        _issuance_token: object | None = None,
        **values: object,
    ) -> None:
        if _issuance_token is not _GRAPH_ISSUANCE_TOKEN:
            raise TypeError(
                "sampled geometric composition graphs can only be factory issued"
            )
        for item in self.__dataclass_fields__.values():
            if not item.init:
                continue
            default = item.default
            value = values.pop(item.name, default)
            if value is MISSING:
                raise TypeError(f"missing composition graph field {item.name!r}")
            object.__setattr__(self, item.name, value)
        if values:
            raise TypeError(
                "unexpected composition graph fields: "
                f"{tuple(sorted(values))}"
            )
        object.__setattr__(self, "_graph_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        exact_types = (
            (self.sampled_geometry_source, V1SampledPreFreezeGeometrySource),
            (self.physical_route_axis, DeclaredRouteAxis),
            (self.score_discovery_coordinates, V1SampledScoreDiscoveryCoordinates),
            (self.fixed_affine_bce_calibration, FixedAffineBCECalibration),
            (self.geometric_trust_budgets, V1GeometricTrustBudgets),
            (self.policy_inputs, V1SampledGeometricCompositionPolicyInputs),
            (self.eligibility_receipt, V1SampledGeometricEligibilityReceipt),
            (self.selector_numerics, V1SampledSnapshotSelectorNumerics),
            (
                self.proposal_position_certificate,
                V1SampledProposalPositionCertificate,
            ),
            (self.selector_chart_provenance, V1SampledSelectorChartProvenance),
            (self.selector_problem, V1ProductionGeometricSelectorProblem),
            (self.selector_policy, V1ProductionGeometricSelectorPolicy),
            (self.selector_result, V1ProductionGeometricSelectorResult),
            (self.sampled_geometric_freeze, V1SampledGeometricFreeze),
            (self.composition_scope, V1CompositionDecisionScopeReceipt),
        )
        if any(type(value) is not expected for value, expected in exact_types):
            raise TypeError("sampled composition graph contains a noncanonical type")
        for name in (
            "build_request_hash",
            "legality_policy_hash",
            "resource_cost_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        snapshot = self.selector_numerics.certificate
        chart = self.selector_chart_provenance
        problem = self.selector_problem
        policy = self.selector_policy
        result = self.selector_result
        frozen = self.sampled_geometric_freeze
        scope = self.composition_scope
        position = self.proposal_position_certificate
        endpoints = tuple(range(1, scope.endpoint_count + 1))
        closure = (
            (
                snapshot.sampled_geometry_source_hash,
                self.sampled_geometry_source.source_hash,
            ),
            (
                snapshot.geometry_state_hash,
                self.sampled_geometry_source.geometry_state.content_hash,
            ),
            (
                snapshot.geometry_result_hash,
                self.sampled_geometry_source.geometry_result.result_hash,
            ),
            (snapshot.assembly_receipt_hash, self.sampled_geometry_source.receipt.receipt_hash),
            (snapshot.certificate_hash, chart.snapshot_certificate.certificate_hash),
            (position.certificate_hash, chart.proposal_position_certificate.certificate_hash),
            (
                position.certificate_hash,
                problem.sampled_proposal_position_certificate.certificate_hash,
            ),
            (self.eligibility_receipt.receipt_hash, problem.eligibility_receipt_hash),
            (
                content_hash(problem.eligible_mask, prefix=_ELIGIBILITY_MASK_PREFIX),
                self.eligibility_receipt.eligible_mask_hash,
            ),
            (chart.resource_cost_hash, self.resource_cost_hash),
            (problem.legality_policy_hash, self.legality_policy_hash),
            (chart.provenance_hash, problem.chart_provenance.provenance_hash),
            (problem.route_axis_hash, self.physical_route_axis.route_axis_hash),
            (
                problem.fd_ordered_route_axis_hash,
                self.physical_route_axis.route_id_axis_hash,
            ),
            (
                policy.metadata.get(
                    "sampled_geometric_composition_build_request_hash"
                ),
                self.build_request_hash,
            ),
            (
                policy.metadata.get(
                    "sampled_geometric_composition_policy_inputs_hash"
                ),
                self.policy_inputs.inputs_hash,
            ),
            (
                policy.metadata.get(
                    "sampled_geometric_explicit_count_ladder_scope"
                ),
                SAMPLED_GEOMETRIC_EXPLICIT_COUNT_LADDER_SCOPE,
            ),
            (
                problem.metadata.get(
                    "sampled_geometric_composition_build_request_hash"
                ),
                self.build_request_hash,
            ),
            (problem.problem_hash, result.problem_hash),
            (policy.policy_hash, result.policy_hash),
            (problem.problem_hash, frozen.selector_problem.problem_hash),
            (policy.policy_hash, frozen.selector_policy.policy_hash),
            (result.result_hash, frozen.selector_result.result_hash),
            (frozen.freeze_hash, self.sampled_geometric_freeze.freeze_hash),
            (frozen.ladder.ladder_hash, scope.frozen_candidate_ladder_hash),
            (problem.problem_hash, scope.sampled_selector_problem_hash),
            (policy.policy_hash, scope.sampled_selector_policy_hash),
            (result.receipt.receipt_hash, scope.sampled_selector_receipt_hash),
            (result.result_hash, scope.sampled_selector_result_hash),
            (position.certificate_hash, scope.sampled_proposal_position_certificate_hash),
            (
                self.score_discovery_coordinates.coordinates_hash,
                chart.score_discovery_coordinates.coordinates_hash,
            ),
            (self.geometric_trust_budgets.budget_hash, chart.geometric_trust_budgets.budget_hash),
            (
                self.fixed_affine_bce_calibration.content_hash,
                chart.fixed_affine_bce_calibration.content_hash,
            ),
            (scope.composition_frozen_endpoint_indices, endpoints),
            (scope.composition_provisional_endpoint_indices, ()),
            (scope.decision_competitive_endpoint_indices, endpoints),
        )
        if any(left != right for left, right in closure):
            raise ValueError("sampled geometric composition graph does not close")
        if (
            policy.diagnostic_only
            or policy.policy_id != PRODUCTION_SELECTOR_CANONICAL_POLICY_ID
            or policy.input_array_policy != "admit_readonly_c_contiguous"
            or not result.receipt.production_chart_authority
            or scope.composition_kind
            != SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND
        ):
            raise PermissionError("sampled composition graph is not canonical production proposal")
        required_true = (
            self.proposal_only,
            self.all_endpoints_composition_frozen,
            self.all_endpoints_decision_competitive,
        )
        required_false = (
            self.score_zero_regret_authority,
            self.fq_certification_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.production_execution_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("composition graph authority fields must be exact booleans")
        if (
            not all(required_true)
            or any(required_false)
            or scope.score_zero_regret_authority
            or scope.fq_certification_authority
            or scope.bce_oracle_authority
            or scope.mask_commit_authority
            or scope.terminal_authority
        ):
            raise PermissionError("sampled composition graph crossed its proposal boundary")
        if (
            self.authority_scope != SAMPLED_GEOMETRIC_COMPOSITION_AUTHORITY_SCOPE
            or self.schema != SAMPLED_GEOMETRIC_COMPOSITION_GRAPH_SCHEMA
        ):
            raise ValueError("unsupported sampled composition graph scope")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "build_request_hash": self.build_request_hash,
            "sampled_geometry_source_hash": self.sampled_geometry_source.source_hash,
            "physical_route_axis_declaration_hash": self.physical_route_axis.content_hash,
            "score_discovery_coordinates_hash": self.score_discovery_coordinates.coordinates_hash,
            "fixed_affine_bce_calibration_hash": self.fixed_affine_bce_calibration.content_hash,
            "geometric_trust_budget_hash": self.geometric_trust_budgets.budget_hash,
            "policy_inputs_hash": self.policy_inputs.inputs_hash,
            "eligibility_receipt_hash": self.eligibility_receipt.receipt_hash,
            "selector_numerics_hash": self.selector_numerics.numerics_hash,
            "proposal_position_certificate_hash": (
                self.proposal_position_certificate.certificate_hash
            ),
            "selector_chart_provenance_hash": self.selector_chart_provenance.provenance_hash,
            "selector_problem_hash": self.selector_problem.problem_hash,
            "selector_policy_hash": self.selector_policy.policy_hash,
            "selector_receipt_hash": self.selector_result.receipt.receipt_hash,
            "selector_result_hash": self.selector_result.result_hash,
            "sampled_geometric_freeze_hash": self.sampled_geometric_freeze.freeze_hash,
            "sampled_geometric_freeze_identity_hash": (
                self.sampled_geometric_freeze.identity.identity_hash
            ),
            "sampled_geometric_replay_receipt_hash": (
                self.sampled_geometric_freeze.replay_receipt.receipt_hash
            ),
            "frozen_candidate_ladder_hash": (
                self.sampled_geometric_freeze.ladder.ladder_hash
            ),
            "composition_scope_hash": self.composition_scope.receipt_hash,
            "legality_policy_hash": self.legality_policy_hash,
            "resource_cost_hash": self.resource_cost_hash,
            "proposal_only": True,
            "all_endpoints_composition_frozen": True,
            "all_endpoints_decision_competitive": True,
            "score_zero_regret_authority": False,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "production_execution_authority": False,
            "terminal_authority": False,
            "authority_scope": self.authority_scope,
        }

    @property
    def graph_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.sampled_geometric_freeze.assert_integrity()
        self.composition_scope.assert_integrity()
        observed = content_hash(self._payload(), prefix=_GRAPH_PREFIX)
        cached = self._graph_hash
        if cached is not None and observed != cached:
            raise ValueError("sampled composition graph was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("sampled composition graph hash does not match")
        object.__setattr__(self, "_graph_hash", observed)
        return observed


def _legality_blocks(
    physical_route_axis: DeclaredRouteAxis,
    supplied: Sequence[V1ProductionLegalityBlock] | None,
) -> tuple[V1ProductionLegalityBlock, ...]:
    if supplied is None:
        return tuple(
            V1ProductionLegalityBlock(
                block_id=str(block.layer_id),
                start=block.offset,
                stop=block.stop,
            )
            for block in physical_route_axis.route_blocks
        )
    blocks = tuple(supplied)
    if not blocks or any(type(block) is not V1ProductionLegalityBlock for block in blocks):
        raise TypeError("legality blocks must use exact compact selector blocks")
    return blocks


def build_v1_sampled_geometric_composition_graph(
    *,
    sampled_geometry_source: V1SampledPreFreezeGeometrySource,
    physical_route_axis: DeclaredRouteAxis,
    score_discovery_coordinates: V1SampledScoreDiscoveryCoordinates,
    fixed_affine_bce_calibration: FixedAffineBCECalibration,
    geometric_trust_budgets: V1GeometricTrustBudgets,
    train_bank_source_hash: str,
    eligible_mask: Sequence[bool] | np.ndarray,
    eligibility_policy_hash: str,
    resource_plane: str,
    resource_cost: Sequence[float] | np.ndarray | float,
    policy_inputs: V1SampledGeometricCompositionPolicyInputs,
    legality_blocks: Sequence[V1ProductionLegalityBlock] | None = None,
    minimum_live_per_block: int = 1,
) -> V1SampledGeometricCompositionGraph:
    """Construct one exact source-to-neutral-scope proposal graph."""

    if type(sampled_geometry_source) is not V1SampledPreFreezeGeometrySource:
        raise TypeError("composition requires an exact sampled pre-freeze source")
    if type(physical_route_axis) is not DeclaredRouteAxis:
        raise TypeError("composition requires an exact DeclaredRouteAxis")
    if type(score_discovery_coordinates) is not V1SampledScoreDiscoveryCoordinates:
        raise TypeError("composition requires exact score-discovery coordinates")
    if type(fixed_affine_bce_calibration) is not FixedAffineBCECalibration:
        raise TypeError("composition requires one exact fixed affine calibration")
    if type(geometric_trust_budgets) is not V1GeometricTrustBudgets:
        raise TypeError("composition requires exact analytic geometric budgets")
    config = policy_inputs
    if type(config) is not V1SampledGeometricCompositionPolicyInputs:
        raise TypeError("composition policy inputs use a noncanonical type")
    train_bank = _identity(train_bank_source_hash, name="train-bank source hash")
    eligibility_policy = _identity(
        eligibility_policy_hash,
        name="eligibility policy hash",
    )
    plane = _identity(resource_plane, name="resource plane")
    minimum = _positive_int(
        minimum_live_per_block,
        name="minimum live routes per legality block",
    )
    if minimum != 1:
        raise ValueError("canonical sampled composition requires one survivor per block")

    source = sampled_geometry_source
    admission = source.admission
    binding = admission.binding
    contract = binding.contract
    current_mask = admission.current_mask
    moments = admission.moments
    if (
        physical_route_axis.route_axis_hash != contract.route_axis_hash
        or physical_route_axis.route_axis_hash != current_mask.route_axis_hash
        or physical_route_axis.route_id_axis_hash != moments.route_axis_hash
        or physical_route_axis.route_count != current_mask.route_count
        or physical_route_axis.route_count != moments.route_count
    ):
        raise ValueError("composition physical and sampled route axes do not close")
    if (
        fixed_affine_bce_calibration.query_id != contract.query_id
        or fixed_affine_bce_calibration.calibration_state_id
        != contract.calibration_identity
        or fixed_affine_bce_calibration.terminal_outcomes_opened
    ):
        raise ValueError("composition calibration differs from the sampled campaign")
    if (
        geometric_trust_budgets.current_mask_hash != current_mask.mask_hash
        or geometric_trust_budgets.damage_target_derivation_hash
        != contract.damage_target_derivation_hash
        or not same_geometric_coordinate(
            geometric_trust_budgets.raw_logit_to_calibrated_logit_slope,
            fixed_affine_bce_calibration.scale,
        )
    ):
        raise ValueError("composition analytic budgets differ from this chart")
    if (
        score_discovery_coordinates.statistics_hash != moments.state_hash
        or score_discovery_coordinates.look_index != moments.look_index
        or score_discovery_coordinates.score_zero_regret_authority
        or score_discovery_coordinates.mask_commit_authority
        or score_discovery_coordinates.terminal_authority
    ):
        raise ValueError("composition discovery coordinates differ from sampled state")

    already_removed = current_mask.route_count - current_mask.retained_count
    remaining_campaign_removals = contract.maximum_removed_count - already_removed
    if remaining_campaign_removals <= 0:
        _refuse(
            V1SampledGeometricCompositionRefusalCode.NO_CAMPAIGN_REMOVALS_REMAIN,
            "the current mask has exhausted the campaign removal ceiling",
        )

    numerics = prepare_v1_sampled_snapshot_selector_numerics(source)
    snapshot = numerics.certificate
    position = certify_v1_sampled_proposal_position_correspondence(
        numerics=numerics,
        physical_route_axis=physical_route_axis,
    )
    eligible = _immutable_bool_vector(
        eligible_mask,
        size=current_mask.route_count,
    )
    eligible_hash = content_hash(eligible, prefix=_ELIGIBILITY_MASK_PREFIX)
    eligibility = V1SampledGeometricEligibilityReceipt(
        sampled_geometry_source_hash=source.source_hash,
        sampled_snapshot_certificate_hash=snapshot.certificate_hash,
        eligibility_policy_hash=eligibility_policy,
        eligible_mask_hash=eligible_hash,
        current_mask_hash=current_mask.mask_hash,
        physical_route_axis_hash=physical_route_axis.route_axis_hash,
        ordered_route_axis_hash=physical_route_axis.route_id_axis_hash,
        route_count=current_mask.route_count,
        proposal_generation=snapshot.proposal_generation,
    )

    if isinstance(resource_cost, Real) and not isinstance(resource_cost, bool):
        uniform_resource = _positive_float(
            resource_cost,
            name="uniform resource cost",
        )
        resource_vector = None
        problem_resource: float | np.ndarray = uniform_resource
    else:
        uniform_resource = None
        resource_vector = _immutable_f64_vector(
            resource_cost,
            size=current_mask.route_count,
        )
        problem_resource = resource_vector

    blocks = _legality_blocks(physical_route_axis, legality_blocks)
    legality_policy_hash = content_hash(
        {
            "schema": "partition_free_v1_sampled_geometric_legality_policy_v1",
            "sampled_geometry_source_hash": source.source_hash,
            "physical_route_axis_declaration_hash": physical_route_axis.content_hash,
            "candidate_domain_policy_hash": contract.candidate_domain_policy_hash,
            "legality_blocks": blocks,
            "minimum_live_per_block": minimum,
            "layers_used_only_for_survivor_legality": True,
            "no_layer_pruning_quota": True,
            "no_sign_partition": True,
        },
        prefix=_LEGALITY_POLICY_PREFIX,
    )
    chart = build_v1_sampled_selector_chart_provenance(
        numerics=numerics,
        proposal_position_certificate=position,
        score_discovery_coordinates=score_discovery_coordinates,
        geometric_trust_budgets=geometric_trust_budgets,
        fixed_affine_bce_calibration=fixed_affine_bce_calibration,
        train_bank_source_hash=train_bank,
        resource_plane=plane,
        resource_cost_policy_hash=contract.resource_cost_policy_hash,
        eligibility_receipt_hash=eligibility.receipt_hash,
        legality_policy_hash=legality_policy_hash,
        uniform_resource_cost=uniform_resource,
        resource_cost=resource_vector,
    )
    build_request_hash = content_hash(
        {
            "schema": SAMPLED_GEOMETRIC_COMPOSITION_GRAPH_SCHEMA,
            "sampled_geometry_source_hash": source.source_hash,
            "physical_route_axis_declaration_hash": physical_route_axis.content_hash,
            "score_discovery_coordinates_hash": (
                score_discovery_coordinates.coordinates_hash
            ),
            "fixed_affine_bce_calibration_hash": (
                fixed_affine_bce_calibration.content_hash
            ),
            "geometric_trust_budget_hash": geometric_trust_budgets.budget_hash,
            "train_bank_source_hash": train_bank,
            "eligibility_receipt_hash": eligibility.receipt_hash,
            "resource_plane": plane,
            "resource_cost_policy_hash": contract.resource_cost_policy_hash,
            "resource_cost_hash": chart.resource_cost_hash,
            "legality_policy_hash": legality_policy_hash,
            "policy_inputs_hash": config.inputs_hash,
        },
        prefix=_BUILD_REQUEST_PREFIX,
    )
    policy = config.build_selector_policy(
        route_count=current_mask.route_count,
        remaining_campaign_removal_count=remaining_campaign_removals,
        build_request_hash=build_request_hash,
    )
    problem_id = content_hash(
        {
            "build_request_hash": build_request_hash,
            "sampled_snapshot_certificate_hash": snapshot.certificate_hash,
            "proposal_position_certificate_hash": position.certificate_hash,
            "selector_chart_provenance_hash": chart.provenance_hash,
            "selector_policy_hash": policy.policy_hash,
        },
        prefix=_PROBLEM_ID_PREFIX,
    )
    problem = V1ProductionGeometricSelectorProblem(
        problem_id=problem_id,
        parent_mask=current_mask,
        eligible_mask=eligible,
        corrected_path_cost=numerics.corrected_path_cost,
        signed_covector=numerics.signed_covector,
        low_rank_factor=numerics.low_rank_factor,
        residual_diagonal=numerics.residual_diagonal,
        residual_spectral_bound=snapshot.residual_spectral_bound,
        resource_cost=problem_resource,
        uniform_resource_cost=uniform_resource,
        finsler_path_budget=geometric_trust_budgets.finsler_path_budget,
        quadratic_trust_energy_budget=(
            geometric_trust_budgets.quadratic_trust_energy_budget
        ),
        source_receipt_hash=snapshot.corrected_point_receipt_hash,
        eligibility_receipt_hash=eligibility.receipt_hash,
        fd_residual_certificate_hash=snapshot.certificate_hash,
        route_axis_hash=physical_route_axis.route_axis_hash,
        fd_ordered_route_axis_hash=physical_route_axis.route_id_axis_hash,
        resource_plane=plane,
        resource_cost_policy_hash=contract.resource_cost_policy_hash,
        legality_policy_hash=legality_policy_hash,
        legality_blocks=blocks,
        minimum_live_per_block=minimum,
        input_array_policy="admit_readonly_c_contiguous",
        fd_residual_certificate_kind=(
            PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND
        ),
        fd_residual_geometry_state_hash=snapshot.geometry_state_hash,
        fd_residual_sampled_snapshot_certificate=snapshot,
        sampled_proposal_position_certificate=position,
        chart_provenance=chart,
        metadata={
            "sampled_geometric_composition_build_request_hash": build_request_hash,
            "sampled_geometric_eligibility_receipt_hash": eligibility.receipt_hash,
            "sampled_geometric_legality_policy_hash": legality_policy_hash,
        },
    )
    result = select_v1_production_geometric_frontier(problem, policy)
    if not result.frontier:
        reasons = ", ".join(result.receipt.refusal_reasons) or "no candidate emitted"
        _refuse(
            V1SampledGeometricCompositionRefusalCode.EMPTY_SELECTOR_FRONTIER,
            f"the selector emitted no freezable proposal ({reasons})",
            selector_result=result,
        )
    contexts = (
        moments.observed_context_ids_by_stratum[0]
        + moments.observed_context_ids_by_stratum[1]
    )
    draws = (
        moments.observed_draw_ids_by_stratum[0]
        + moments.observed_draw_ids_by_stratum[1]
    )
    identity = V1SampledGeometricFreezeIdentity(
        campaign_contract_hash=snapshot.campaign_contract_hash,
        campaign_binding_hash=snapshot.campaign_binding_hash,
        model_identity_hash=snapshot.model_identity_hash,
        query_contract_hash=snapshot.query_contract_hash,
        preprocessing_hash=snapshot.preprocessing_hash,
        current_mask_hash=snapshot.current_mask_hash,
        raw_logit_stream_source_hash=snapshot.raw_logit_stream_source_hash,
        sampling_law_hash=snapshot.sampling_law_hash,
        context_mixture_hash=snapshot.context_mixture_hash,
        train_bank_source_hash=train_bank,
        calibration_identity=snapshot.calibration_identity,
        proposal_generation=snapshot.proposal_generation,
        physical_route_axis_hash=snapshot.physical_route_axis_hash,
        ordered_route_axis_hash=snapshot.ordered_route_axis_hash,
        discovery_context_ids=contexts,
        discovery_draw_ids=draws,
        score_discovery_registration_hash=(
            score_discovery_coordinates.registration_hash
        ),
        score_discovery_design_hash=score_discovery_coordinates.design_hash,
        score_discovery_look_receipt_hash=(
            score_discovery_coordinates.look_receipt_hash
        ),
        score_discovery_look_index=score_discovery_coordinates.look_index,
        maximum_registered_score_looks=contract.maximum_looks_per_step,
        sampled_proposal_position_certificate_hash=position.certificate_hash,
        score_discovery_result_hash=score_discovery_coordinates.result_hash,
        score_discovery_statistics_hash=score_discovery_coordinates.statistics_hash,
        score_discovery_evidence_hash=score_discovery_coordinates.evidence_hash,
        sampled_geometry_source_hash=snapshot.sampled_geometry_source_hash,
        sampled_geometry_state_hash=snapshot.geometry_state_hash,
        sampled_geometry_result_hash=snapshot.geometry_result_hash,
        sampled_geometry_assembly_hash=snapshot.assembly_receipt_hash,
        source_measurement_axis_hash=snapshot.source_measurement_axis_hash,
        geometric_budget_hash=geometric_trust_budgets.budget_hash,
        selector_source_receipt_hash=problem.source_receipt_hash,
        selector_eligibility_receipt_hash=problem.eligibility_receipt_hash,
        selector_fd_residual_certificate_hash=(
            problem.fd_residual_certificate_hash
        ),
        finsler_path_budget=problem.finsler_path_budget,
        quadratic_trust_energy_budget=problem.quadratic_trust_energy_budget,
    )
    frozen = freeze_v1_sampled_geometric_selector_ladder(
        selector_problem=problem,
        selector_policy=policy,
        selector_result=result,
        current_mask=current_mask,
        identity=identity,
        packed_byte_chunk_size=config.packed_freeze_byte_chunk_size,
    )
    scope = composition_scope_from_v1_sampled_geometric_freeze(frozen)
    return V1SampledGeometricCompositionGraph(
        _issuance_token=_GRAPH_ISSUANCE_TOKEN,
        build_request_hash=build_request_hash,
        sampled_geometry_source=source,
        physical_route_axis=physical_route_axis,
        score_discovery_coordinates=score_discovery_coordinates,
        fixed_affine_bce_calibration=fixed_affine_bce_calibration,
        geometric_trust_budgets=geometric_trust_budgets,
        policy_inputs=config,
        eligibility_receipt=eligibility,
        selector_numerics=numerics,
        proposal_position_certificate=position,
        selector_chart_provenance=chart,
        selector_problem=problem,
        selector_policy=policy,
        selector_result=result,
        sampled_geometric_freeze=frozen,
        composition_scope=scope,
        legality_policy_hash=legality_policy_hash,
        resource_cost_hash=chart.resource_cost_hash,
    )


__all__ = [
    "SAMPLED_GEOMETRIC_COMPOSITION_AUTHORITY_SCOPE",
    "SAMPLED_GEOMETRIC_COMPOSITION_GRAPH_SCHEMA",
    "SAMPLED_GEOMETRIC_COMPOSITION_POLICY_INPUTS_SCHEMA",
    "SAMPLED_GEOMETRIC_EXPLICIT_COUNT_LADDER_SCOPE",
    "SAMPLED_GEOMETRIC_ELIGIBILITY_RECEIPT_SCHEMA",
    "V1SampledGeometricCompositionGraph",
    "V1SampledGeometricCompositionPolicyInputs",
    "V1SampledGeometricCompositionRefusal",
    "V1SampledGeometricCompositionRefusalCode",
    "V1SampledGeometricEligibilityReceipt",
    "build_v1_sampled_geometric_composition_graph",
]
