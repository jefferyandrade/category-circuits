"""Proposal-only freeze bridge for a sampled-geometric selector frontier.

The production geometric selector emits a resource-descending packed frontier.
Certification consumes a strictly nested, shallow-to-deep candidate ladder.
This module is the narrow bridge between those two representations.  It:

* validates the selector problem, policy, result, and every packed endpoint;
* normalizes one nested selector family by increasing cumulative removals;
* recomputes the exact endpoint resource, F, and certified Q coordinates;
* streams a shell-ordered little-endian uint64 prefix identity without retaining
  route tuples or claiming that an external prefix was persisted; and
* freezes a :class:`V1FrozenCandidateLadder` with proposal authority only.

No sampling, F/Q certification, BCE oracle execution, mask transition, or
terminal disclosure occurs here.  In particular, a sampled geometry is not
claimed to be statistically optimal merely because its finite proposal
frontier was replayed exactly.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_geometric_budgets import same_geometric_coordinate
from .modern_v1_geometric_selector_production import (
    PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND,
    V1ProductionGeometricCandidate,
    V1ProductionGeometricSelectorPolicy,
    V1ProductionGeometricSelectorProblem,
    V1ProductionGeometricSelectorResult,
)
from .modern_v1_sampled_geometric_provenance import (
    V1SampledProposalPositionCertificate,
    V1SampledSelectorChartProvenance,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
)
from .modern_v1_composition_scope import (
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
    V1CompositionDecisionScopeReceipt,
)


SAMPLED_GEOMETRIC_FREEZE_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_freeze_identity_v1"
)
SAMPLED_GEOMETRIC_REPLAY_FREEZE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_replay_freeze_receipt_v1"
)
SAMPLED_GEOMETRIC_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_freeze_v1"
)
SAMPLED_GEOMETRIC_FREEZE_AUTHORITY_SCOPE: Final = (
    "sampled_geometry_selector_to_nested_proposal_ladder_only_"
    "no_fq_bce_mask_commit_or_terminal_authority_v1"
)
SAMPLED_GEOMETRIC_SHELL_ORDER_SCHEMA: Final = (
    "partition_free_v1_sampled_geometric_shell_order_u64_stream_v1"
)
SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE: Final = "<u8"

_IDENTITY_PREFIX: Final = "partition-free-v1-sampled-geometric-freeze-input-"
_RECEIPT_PREFIX: Final = "partition-free-v1-sampled-geometric-replay-freeze-"
_FREEZE_ID_PREFIX: Final = "partition-free-v1-sampled-geometric-freeze-id-"
_FREEZE_PREFIX: Final = "partition-free-v1-sampled-geometric-freeze-"
_ACTION_ID_PREFIX: Final = "partition-free-v1-sampled-geometric-action-id-"
_ACTION_AXIS_PREFIX: Final = "partition-free-v1-sampled-geometric-action-axis-"
_COUNT_AXIS_PREFIX: Final = "partition-free-v1-sampled-geometric-count-axis-"
_SHELL_ORDER_PREFIX: Final = "partition-free-v1-sampled-geometric-shell-order-"

_DEFAULT_PACKED_BYTE_CHUNK_SIZE: Final = 1 << 20
_DEFAULT_ROUTE_CHUNK_SIZE: Final = 1 << 20
_FLOAT_RTOL: Final = 2.0e-12
_FLOAT_ATOL: Final = 1.0e-12


class V1SampledGeometricFreezeRefusalCode(str, Enum):
    """Stable fail-closed reasons at the geometric freeze boundary."""

    INVALID_TYPE = "invalid_type"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    EMPTY_FRONTIER = "empty_frontier"
    MULTIPLE_FAMILIES = "multiple_families"
    NON_NESTED_FRONTIER = "non_nested_frontier"
    CANDIDATE_MISMATCH = "candidate_mismatch"
    BUDGET_VIOLATION = "budget_violation"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledGeometricFreezeRefusal(ValueError):
    """Typed refusal raised before a geometric proposal ladder can freeze."""

    def __init__(
        self,
        code: V1SampledGeometricFreezeRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SampledGeometricFreezeRefusalCode, message: str) -> NoReturn:
    raise V1SampledGeometricFreezeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonempty and unique",
        )
    return result


def _same_mask(left: object, right: object) -> bool:
    return (
        type(left) is HardMask
        and type(right) is HardMask
        and left.route_axis_hash == right.route_axis_hash
        and left.route_count == right.route_count
        and left.packed_bits == right.packed_bits
        and left.mask_hash == right.mask_hash
    )


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=_FLOAT_RTOL,
        abs_tol=_FLOAT_ATOL,
    )


def _budget_ratio(value: float, budget: float) -> float:
    if budget == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return value / budget


def _proposal_burden(
    *,
    path: float,
    quadratic: float,
    tau_f: float,
    tau_q: float,
) -> float:
    burden = max(
        _budget_ratio(path, tau_f),
        math.sqrt(_budget_ratio(quadratic, tau_q)),
    )
    if not math.isfinite(burden):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.BUDGET_VIOLATION,
            "selector candidate has undefined burden under a zero trust budget",
        )
    return burden


@dataclass(frozen=True, slots=True)
class V1SampledGeometricFreezeIdentity:
    """Source-bound identities required to freeze one sampled-geometric ladder.

    Geometry result and assembly hashes are retained as supplied source graph
    coordinates.  This object does not promote their finite-sample evidence to
    a statistical optimality claim.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    context_mixture_hash: str
    train_bank_source_hash: str
    calibration_identity: str
    proposal_generation: int
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    discovery_context_ids: tuple[str, ...]
    discovery_draw_ids: tuple[str, ...]
    score_discovery_registration_hash: str
    score_discovery_design_hash: str
    score_discovery_look_receipt_hash: str
    score_discovery_look_index: int
    maximum_registered_score_looks: int
    sampled_proposal_position_certificate_hash: str | None
    score_discovery_result_hash: str
    score_discovery_statistics_hash: str
    score_discovery_evidence_hash: str
    sampled_geometry_source_hash: str
    sampled_geometry_state_hash: str
    sampled_geometry_result_hash: str
    sampled_geometry_assembly_hash: str
    source_measurement_axis_hash: str
    geometric_budget_hash: str
    selector_source_receipt_hash: str
    selector_eligibility_receipt_hash: str
    selector_fd_residual_certificate_hash: str
    finsler_path_budget: float
    quadratic_trust_energy_budget: float
    schema: str = SAMPLED_GEOMETRIC_FREEZE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "context_mixture_hash",
            "train_bank_source_hash",
            "calibration_identity",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "score_discovery_registration_hash",
            "score_discovery_design_hash",
            "score_discovery_look_receipt_hash",
            "score_discovery_result_hash",
            "score_discovery_statistics_hash",
            "score_discovery_evidence_hash",
            "sampled_geometry_source_hash",
            "sampled_geometry_state_hash",
            "sampled_geometry_result_hash",
            "sampled_geometry_assembly_hash",
            "source_measurement_axis_hash",
            "geometric_budget_hash",
            "selector_source_receipt_hash",
            "selector_eligibility_receipt_hash",
            "selector_fd_residual_certificate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        score_look_index = _positive_int(
            self.score_discovery_look_index,
            name="score-discovery look index",
        )
        maximum_score_looks = _positive_int(
            self.maximum_registered_score_looks,
            name="maximum registered score looks",
        )
        if score_look_index > maximum_score_looks:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "score-discovery look index exceeds its preregistered axis",
            )
        position_certificate_hash = _optional_identity(
            self.sampled_proposal_position_certificate_hash,
            name="sampled proposal-position certificate hash",
        )
        contexts = _identity_axis(
            self.discovery_context_ids,
            name="discovery context IDs",
        )
        draws = _identity_axis(
            self.discovery_draw_ids,
            name="discovery draw IDs",
        )
        if len(contexts) != len(draws):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "discovery context and draw ledgers must align",
            )
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "physical and ordered route axes must remain distinct coordinates",
            )
        tau_f = _finite_nonnegative(
            self.finsler_path_budget,
            name="Finsler path budget",
        )
        tau_q = _finite_nonnegative(
            self.quadratic_trust_energy_budget,
            name="quadratic trust-energy budget",
        )
        if self.schema != SAMPLED_GEOMETRIC_FREEZE_IDENTITY_SCHEMA:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "unsupported sampled-geometric freeze identity schema",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "score_discovery_look_index", score_look_index)
        object.__setattr__(
            self,
            "maximum_registered_score_looks",
            maximum_score_looks,
        )
        object.__setattr__(
            self,
            "sampled_proposal_position_certificate_hash",
            position_certificate_hash,
        )
        object.__setattr__(self, "discovery_context_ids", contexts)
        object.__setattr__(self, "discovery_draw_ids", draws)
        object.__setattr__(self, "finsler_path_budget", tau_f)
        object.__setattr__(self, "quadratic_trust_energy_budget", tau_q)

    @property
    def identity_hash(self) -> str:
        return content_hash(self, prefix=_IDENTITY_PREFIX)


def _freeze_identity_payload(
    *,
    identity_hash: str,
    selector_problem_hash: str,
    selector_policy_hash: str,
    selector_result_hash: str,
    selector_receipt_hash: str,
    normalized_source_frontier_indices: tuple[int, ...],
    endpoint_candidate_hashes: tuple[str, ...],
    endpoint_packed_action_hashes: tuple[str, ...],
    endpoint_frozen_action_hashes: tuple[str, ...],
    endpoint_removed_route_index_hashes: tuple[str, ...],
    endpoint_removal_counts: tuple[int, ...],
    endpoint_resulting_mask_hashes: tuple[str, ...],
    shell_order_stream_hash: str,
) -> dict[str, object]:
    return {
        "schema": SAMPLED_GEOMETRIC_FREEZE_SCHEMA,
        "identity_hash": identity_hash,
        "selector_problem_hash": selector_problem_hash,
        "selector_policy_hash": selector_policy_hash,
        "selector_result_hash": selector_result_hash,
        "selector_receipt_hash": selector_receipt_hash,
        "normalized_source_frontier_indices": normalized_source_frontier_indices,
        "endpoint_candidate_hashes": endpoint_candidate_hashes,
        "endpoint_packed_action_hashes": endpoint_packed_action_hashes,
        "endpoint_frozen_action_hashes": endpoint_frozen_action_hashes,
        "endpoint_removed_route_index_hashes": (
            endpoint_removed_route_index_hashes
        ),
        "endpoint_removal_counts": endpoint_removal_counts,
        "endpoint_resulting_mask_hashes": endpoint_resulting_mask_hashes,
        "shell_order_stream_hash": shell_order_stream_hash,
        "proposal_only": True,
        "sampled_geometry_statistical_optimality_claim": False,
        "fq_certification_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
    }


@dataclass(frozen=True, slots=True)
class V1SampledGeometricReplayFreezeReceipt:
    """Exact replay receipt for one normalized packed proposal ladder."""

    identity_hash: str
    selector_problem_hash: str
    selector_policy_hash: str
    selector_result_hash: str
    selector_receipt_hash: str
    selector_source_receipt_hash: str
    selector_eligibility_receipt_hash: str
    selector_fd_residual_certificate_hash: str
    geometry_state_hash: str
    geometry_result_hash: str
    geometry_assembly_hash: str
    score_discovery_registration_hash: str
    score_discovery_design_hash: str
    score_discovery_look_receipt_hash: str
    score_discovery_look_index: int
    maximum_registered_score_looks: int
    sampled_proposal_position_certificate_hash: str | None
    score_discovery_result_hash: str
    score_discovery_statistics_hash: str
    score_discovery_evidence_hash: str
    parent_mask_hash: str
    route_count: int
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    finsler_path_budget: float
    quadratic_trust_energy_budget: float
    geometric_budget_hash: str
    normalized_source_frontier_indices: tuple[int, ...]
    endpoint_candidate_hashes: tuple[str, ...]
    endpoint_packed_action_hashes: tuple[str, ...]
    endpoint_frozen_action_hashes: tuple[str, ...]
    endpoint_removed_route_index_hashes: tuple[str, ...]
    endpoint_removal_counts: tuple[int, ...]
    endpoint_resulting_mask_hashes: tuple[str, ...]
    endpoint_action_axis_hash: str
    endpoint_count_axis_hash: str
    shell_order_stream_sha256: str
    shell_order_stream_hash: str
    shell_order_route_count: int
    shell_order_dtype: str
    packed_scan_byte_chunk_size: int
    freeze_identity_hash: str
    proposal_ladder_freeze_authority: bool = True
    proposal_only: bool = True
    sampled_geometry_statistical_optimality_claim: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    shell_order_materialized: bool = False
    shell_order_external_prefix_persisted: bool = False
    shell_order_reopenable: bool = False
    authority_scope: str = SAMPLED_GEOMETRIC_FREEZE_AUTHORITY_SCOPE
    schema: str = SAMPLED_GEOMETRIC_REPLAY_FREEZE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "identity_hash",
            "selector_problem_hash",
            "selector_policy_hash",
            "selector_result_hash",
            "selector_receipt_hash",
            "selector_source_receipt_hash",
            "selector_eligibility_receipt_hash",
            "selector_fd_residual_certificate_hash",
            "geometry_state_hash",
            "geometry_result_hash",
            "geometry_assembly_hash",
            "score_discovery_registration_hash",
            "score_discovery_design_hash",
            "score_discovery_look_receipt_hash",
            "score_discovery_result_hash",
            "score_discovery_statistics_hash",
            "score_discovery_evidence_hash",
            "parent_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "geometric_budget_hash",
            "endpoint_action_axis_hash",
            "endpoint_count_axis_hash",
            "shell_order_stream_sha256",
            "shell_order_stream_hash",
            "freeze_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        score_look_index = _positive_int(
            self.score_discovery_look_index,
            name="score-discovery look index",
        )
        maximum_score_looks = _positive_int(
            self.maximum_registered_score_looks,
            name="maximum registered score looks",
        )
        if score_look_index > maximum_score_looks:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "score-discovery look index exceeds its preregistered axis",
            )
        position_certificate_hash = _optional_identity(
            self.sampled_proposal_position_certificate_hash,
            name="sampled proposal-position certificate hash",
        )
        stream_count = _positive_int(
            self.shell_order_route_count,
            name="shell-order route count",
        )
        byte_chunk_size = _positive_int(
            self.packed_scan_byte_chunk_size,
            name="packed scan byte chunk size",
        )
        source_indices = tuple(
            _nonnegative_int(value, name="source frontier index")
            for value in self.normalized_source_frontier_indices
        )
        candidates = _identity_axis(
            self.endpoint_candidate_hashes,
            name="endpoint candidate hashes",
        )
        packed_actions = _identity_axis(
            self.endpoint_packed_action_hashes,
            name="endpoint packed-action hashes",
        )
        frozen_actions = _identity_axis(
            self.endpoint_frozen_action_hashes,
            name="endpoint frozen-action hashes",
        )
        removed_hashes = _identity_axis(
            self.endpoint_removed_route_index_hashes,
            name="endpoint removed-route hashes",
        )
        mask_hashes = _identity_axis(
            self.endpoint_resulting_mask_hashes,
            name="endpoint mask hashes",
        )
        counts = tuple(
            _positive_int(value, name="endpoint removal count")
            for value in self.endpoint_removal_counts
        )
        arities = {
            len(source_indices),
            len(candidates),
            len(packed_actions),
            len(frozen_actions),
            len(removed_hashes),
            len(counts),
            len(mask_hashes),
        }
        if len(arities) != 1 or not counts:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "geometric freeze endpoint axes do not align",
            )
        if (
            tuple(sorted(counts)) != counts
            or len(counts) != len(set(counts))
            or len(source_indices) != len(set(source_indices))
            or tuple(sorted(source_indices)) != tuple(range(len(source_indices)))
            or stream_count != counts[-1]
            or counts[-1] > route_count
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.NON_NESTED_FRONTIER,
                "geometric freeze counts are not one strict cumulative ladder",
            )
        expected_action_axis_hash = content_hash(
            {
                "candidate_hashes": candidates,
                "packed_action_hashes": packed_actions,
                "frozen_action_hashes": frozen_actions,
                "removed_route_index_hashes": removed_hashes,
                "resulting_mask_hashes": mask_hashes,
            },
            prefix=_ACTION_AXIS_PREFIX,
        )
        expected_count_axis_hash = content_hash(
            counts,
            prefix=_COUNT_AXIS_PREFIX,
        )
        if (
            self.endpoint_action_axis_hash != expected_action_axis_hash
            or self.endpoint_count_axis_hash != expected_count_axis_hash
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "geometric endpoint action/count axis hash does not close",
            )
        try:
            stream_digest_bytes = bytes.fromhex(self.shell_order_stream_sha256)
        except ValueError:
            stream_digest_bytes = b""
        if len(stream_digest_bytes) != hashlib.sha256().digest_size:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "shell-order stream SHA-256 is malformed",
            )
        expected_stream_hash = content_hash(
            {
                "schema": SAMPLED_GEOMETRIC_SHELL_ORDER_SCHEMA,
                "dtype": self.shell_order_dtype,
                "route_count": stream_count,
                "raw_stream_sha256": self.shell_order_stream_sha256,
                "endpoint_removal_counts": counts,
                "endpoint_shell_counts": tuple(
                    count - (0 if index == 0 else counts[index - 1])
                    for index, count in enumerate(counts)
                ),
            },
            prefix=_SHELL_ORDER_PREFIX,
        )
        if self.shell_order_stream_hash != expected_stream_hash:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "shell-order streaming identity does not close",
            )
        expected_freeze_id = content_hash(
            _freeze_identity_payload(
                identity_hash=self.identity_hash,
                selector_problem_hash=self.selector_problem_hash,
                selector_policy_hash=self.selector_policy_hash,
                selector_result_hash=self.selector_result_hash,
                selector_receipt_hash=self.selector_receipt_hash,
                normalized_source_frontier_indices=source_indices,
                endpoint_candidate_hashes=candidates,
                endpoint_packed_action_hashes=packed_actions,
                endpoint_frozen_action_hashes=frozen_actions,
                endpoint_removed_route_index_hashes=removed_hashes,
                endpoint_removal_counts=counts,
                endpoint_resulting_mask_hashes=mask_hashes,
                shell_order_stream_hash=self.shell_order_stream_hash,
            ),
            prefix=_FREEZE_ID_PREFIX,
        )
        if self.freeze_identity_hash != expected_freeze_id:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "geometric freeze identity does not close",
            )
        authority_flags = (
            self.proposal_ladder_freeze_authority is True,
            self.proposal_only is True,
            self.sampled_geometry_statistical_optimality_claim is False,
            self.fq_certification_authority is False,
            self.bce_oracle_authority is False,
            self.mask_commit_authority is False,
            self.terminal_authority is False,
            self.shell_order_materialized is False,
            self.shell_order_external_prefix_persisted is False,
            self.shell_order_reopenable is False,
        )
        if any(type(flag) is not bool or not flag for flag in authority_flags):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.AUTHORITY_VIOLATION,
                "sampled-geometric freeze crossed its proposal-only boundary",
            )
        if (
            self.shell_order_dtype != SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE
            or self.authority_scope != SAMPLED_GEOMETRIC_FREEZE_AUTHORITY_SCOPE
            or self.schema != SAMPLED_GEOMETRIC_REPLAY_FREEZE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.AUTHORITY_VIOLATION,
                "sampled-geometric freeze scope, schema, or stream dtype drifted",
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "score_discovery_look_index", score_look_index)
        object.__setattr__(
            self,
            "maximum_registered_score_looks",
            maximum_score_looks,
        )
        object.__setattr__(
            self,
            "sampled_proposal_position_certificate_hash",
            position_certificate_hash,
        )
        object.__setattr__(self, "shell_order_route_count", stream_count)
        object.__setattr__(self, "packed_scan_byte_chunk_size", byte_chunk_size)
        object.__setattr__(self, "normalized_source_frontier_indices", source_indices)
        object.__setattr__(self, "endpoint_candidate_hashes", candidates)
        object.__setattr__(self, "endpoint_packed_action_hashes", packed_actions)
        object.__setattr__(self, "endpoint_frozen_action_hashes", frozen_actions)
        object.__setattr__(self, "endpoint_removed_route_index_hashes", removed_hashes)
        object.__setattr__(self, "endpoint_removal_counts", counts)
        object.__setattr__(self, "endpoint_resulting_mask_hashes", mask_hashes)
        object.__setattr__(
            self,
            "finsler_path_budget",
            _finite_nonnegative(self.finsler_path_budget, name="Finsler path budget"),
        )
        object.__setattr__(
            self,
            "quadratic_trust_energy_budget",
            _finite_nonnegative(
                self.quadratic_trust_energy_budget,
                name="quadratic trust-energy budget",
            ),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


def _iter_ladder_shell_order_chunks(
    ladder: V1FrozenCandidateLadder,
    *,
    route_chunk_size: int,
    packed_byte_chunk_size: int,
) -> Iterator[np.ndarray]:
    chunk_size = _positive_int(route_chunk_size, name="route chunk size")
    byte_chunk_size = _positive_int(
        packed_byte_chunk_size,
        name="packed byte chunk size",
    )
    dtype = np.dtype(SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE)
    for action in ladder.actions:
        for raw in ladder.iter_shell_index_chunks(
            endpoint_index=action.endpoint_index,
            packed_byte_chunk_size=byte_chunk_size,
        ):
            source = np.asarray(raw).reshape(-1)
            for start in range(0, int(source.size), chunk_size):
                stop = min(int(source.size), start + chunk_size)
                chunk = np.asarray(source[start:stop], dtype=dtype)
                if chunk.dtype != dtype:
                    raise RuntimeError(
                        "shell-order chunk did not normalize to little-endian u64"
                    )
                chunk.setflags(write=False)
                yield chunk


@dataclass(frozen=True, slots=True)
class V1SampledGeometricFreeze:
    """Typed source graph and frozen nested proposal ladder."""

    identity: V1SampledGeometricFreezeIdentity
    selector_problem: V1ProductionGeometricSelectorProblem = field(
        repr=False,
        compare=False,
    )
    selector_policy: V1ProductionGeometricSelectorPolicy = field(
        repr=False,
        compare=False,
    )
    selector_result: V1ProductionGeometricSelectorResult = field(
        repr=False,
        compare=False,
    )
    ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    replay_receipt: V1SampledGeometricReplayFreezeReceipt
    schema: str = SAMPLED_GEOMETRIC_FREEZE_SCHEMA
    _freeze_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.identity) is not V1SampledGeometricFreezeIdentity:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires its exact identity object",
            )
        if type(self.selector_problem) is not V1ProductionGeometricSelectorProblem:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires its exact selector problem",
            )
        if type(self.selector_policy) is not V1ProductionGeometricSelectorPolicy:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires its exact selector policy",
            )
        if type(self.selector_result) is not V1ProductionGeometricSelectorResult:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires its exact selector result",
            )
        if type(self.ladder) is not V1FrozenCandidateLadder:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires one exact frozen candidate ladder",
            )
        if type(self.replay_receipt) is not V1SampledGeometricReplayFreezeReceipt:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "geometric freeze requires its distinct geometric replay receipt",
            )
        receipt = self.replay_receipt
        position_certificate = (
            self.selector_problem.sampled_proposal_position_certificate
        )
        position_certificate_hash = (
            None
            if position_certificate is None
            else position_certificate.certificate_hash
        )
        normalized = tuple(
            sorted(
                enumerate(self.selector_result.frontier),
                key=lambda item: item[1].removal_count,
            )
        )
        source_values = (
            (receipt.identity_hash, self.identity.identity_hash),
            (receipt.selector_problem_hash, self.selector_problem.problem_hash),
            (receipt.selector_policy_hash, self.selector_policy.policy_hash),
            (receipt.selector_result_hash, self.selector_result.result_hash),
            (receipt.selector_receipt_hash, self.selector_result.receipt.receipt_hash),
            (receipt.parent_mask_hash, self.selector_problem.parent_mask.mask_hash),
            (receipt.route_count, self.selector_problem.route_count),
            (receipt.physical_route_axis_hash, self.ladder.route_axis_hash),
            (receipt.ordered_route_axis_hash, self.identity.ordered_route_axis_hash),
            (
                receipt.selector_source_receipt_hash,
                self.selector_problem.source_receipt_hash,
            ),
            (
                receipt.selector_eligibility_receipt_hash,
                self.selector_problem.eligibility_receipt_hash,
            ),
            (
                receipt.selector_fd_residual_certificate_hash,
                self.selector_problem.fd_residual_certificate_hash,
            ),
            (receipt.geometry_state_hash, self.identity.sampled_geometry_state_hash),
            (receipt.geometry_result_hash, self.identity.sampled_geometry_result_hash),
            (
                receipt.geometry_assembly_hash,
                self.identity.sampled_geometry_assembly_hash,
            ),
            (
                receipt.score_discovery_registration_hash,
                self.identity.score_discovery_registration_hash,
            ),
            (
                receipt.score_discovery_design_hash,
                self.identity.score_discovery_design_hash,
            ),
            (
                receipt.score_discovery_look_receipt_hash,
                self.identity.score_discovery_look_receipt_hash,
            ),
            (
                receipt.score_discovery_look_index,
                self.identity.score_discovery_look_index,
            ),
            (
                receipt.maximum_registered_score_looks,
                self.identity.maximum_registered_score_looks,
            ),
            (
                receipt.sampled_proposal_position_certificate_hash,
                self.identity.sampled_proposal_position_certificate_hash,
            ),
            (
                self.identity.sampled_proposal_position_certificate_hash,
                position_certificate_hash,
            ),
            (
                receipt.score_discovery_result_hash,
                self.identity.score_discovery_result_hash,
            ),
            (
                receipt.score_discovery_statistics_hash,
                self.identity.score_discovery_statistics_hash,
            ),
            (
                receipt.score_discovery_evidence_hash,
                self.identity.score_discovery_evidence_hash,
            ),
            (receipt.geometric_budget_hash, self.identity.geometric_budget_hash),
            (
                receipt.normalized_source_frontier_indices,
                tuple(index for index, _candidate in normalized),
            ),
            (
                receipt.endpoint_candidate_hashes,
                tuple(candidate.candidate_hash for _index, candidate in normalized),
            ),
            (
                receipt.endpoint_packed_action_hashes,
                tuple(candidate.action.action_hash for _index, candidate in normalized),
            ),
            (
                receipt.endpoint_removed_route_index_hashes,
                tuple(
                    candidate.action.removed_route_index_hash
                    for _index, candidate in normalized
                ),
            ),
            (
                receipt.endpoint_resulting_mask_hashes,
                tuple(
                    candidate.action.resulting_mask.mask_hash
                    for _index, candidate in normalized
                ),
            ),
            (self.ladder.freeze_receipt_hash, receipt.receipt_hash),
            (
                self.ladder.discovery_evidence_hash,
                self.identity.score_discovery_evidence_hash,
            ),
            (self.ladder.action_hashes, receipt.endpoint_frozen_action_hashes),
            (self.ladder.endpoint_removed_counts, receipt.endpoint_removal_counts),
            (self.ladder.campaign_contract_hash, self.identity.campaign_contract_hash),
            (self.ladder.model_identity_hash, self.identity.model_identity_hash),
            (self.ladder.query_contract_hash, self.identity.query_contract_hash),
            (self.ladder.preprocessing_hash, self.identity.preprocessing_hash),
            (
                self.ladder.raw_logit_stream_source_hash,
                self.identity.raw_logit_stream_source_hash,
            ),
            (self.ladder.sampling_law_hash, self.identity.sampling_law_hash),
            (self.ladder.proposal_generation, self.identity.proposal_generation),
            (self.ladder.discovery_context_ids, self.identity.discovery_context_ids),
            (self.ladder.discovery_draw_ids, self.identity.discovery_draw_ids),
        )
        if any(left != right for left, right in source_values):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled-geometric freeze source graph does not close",
            )
        if not same_geometric_coordinate(
            receipt.finsler_path_budget,
            self.selector_problem.finsler_path_budget,
        ) or not same_geometric_coordinate(
            receipt.quadratic_trust_energy_budget,
            self.selector_problem.quadratic_trust_energy_budget,
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled-geometric freeze trust budgets do not close",
            )
        if self.schema != SAMPLED_GEOMETRIC_FREEZE_SCHEMA:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "unsupported sampled-geometric freeze schema",
            )
        self.ladder.assert_integrity()
        self.assert_integrity()

    def iter_shell_ordered_prefix_chunks(
        self,
        *,
        route_chunk_size: int = _DEFAULT_ROUTE_CHUNK_SIZE,
        packed_byte_chunk_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Reconstruct the shell-ordered ``<u8`` prefix in bounded chunks.

        This is a deterministic in-memory replay over the packed masks.  The
        receipt deliberately does not claim that these bytes were persisted or
        can be reopened without the retained typed masks.
        """

        yield from _iter_ladder_shell_order_chunks(
            self.ladder,
            route_chunk_size=route_chunk_size,
            packed_byte_chunk_size=(
                self.replay_receipt.packed_scan_byte_chunk_size
                if packed_byte_chunk_size is None
                else packed_byte_chunk_size
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "identity_hash": self.identity.identity_hash,
            "selector_problem_hash": self.selector_problem.problem_hash,
            "selector_policy_hash": self.selector_policy.policy_hash,
            "selector_result_hash": self.selector_result.result_hash,
            "replay_receipt_hash": self.replay_receipt.receipt_hash,
            "ladder_hash": self.ladder.assert_integrity(),
            "freeze_identity_hash": self.replay_receipt.freeze_identity_hash,
        }

    @property
    def freeze_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FREEZE_PREFIX)
        if self._freeze_hash is not None and observed != self._freeze_hash:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled-geometric freeze was tampered after hashing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled-geometric freeze hash does not match",
            )
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


def _validate_problem_result_graph(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
    result: V1ProductionGeometricSelectorResult,
    current_mask: HardMask,
    identity: V1SampledGeometricFreezeIdentity,
) -> None:
    if not _same_mask(problem.parent_mask, current_mask):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            "selector problem and supplied current mask differ",
        )
    if (
        result.problem_hash != problem.problem_hash
        or result.receipt.problem_hash != problem.problem_hash
        or result.policy_hash != policy.policy_hash
        or result.receipt.policy_hash != policy.policy_hash
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            "selector problem/policy/result hashes do not close",
        )
    if (
        identity.current_mask_hash != current_mask.mask_hash
        or identity.physical_route_axis_hash != current_mask.route_axis_hash
        or problem.route_axis_hash != current_mask.route_axis_hash
        or problem.route_count != current_mask.route_count
        or result.receipt.route_count != problem.route_count
        or result.receipt.live_route_count != problem.live_count
        or result.receipt.eligible_route_count != problem.eligible_count
        or result.receipt.selectable_route_count != problem.selectable_count
        or result.receipt.block_count != problem.block_count
        or result.receipt.scalarization_count != len(policy.scalarizations)
        or result.receipt.candidate_count != len(result.frontier)
        or result.receipt.packed_action_count != len(result.frontier)
        or result.receipt.packed_action_payload_bytes
        != len(current_mask.packed_bits) * len(result.frontier)
        or result.receipt.retained_index_tuple_count != 0
        or result.receipt.backend != policy.backend
        or result.receipt.chunk_size != policy.chunk_size
        or result.receipt.device
        != (policy.torch_device if policy.backend == "torch" else "cpu")
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            "selector mask, axis, population, or receipt coordinates differ",
        )
    if (
        identity.selector_source_receipt_hash != problem.source_receipt_hash
        or identity.selector_eligibility_receipt_hash
        != problem.eligibility_receipt_hash
        or identity.selector_fd_residual_certificate_hash
        != problem.fd_residual_certificate_hash
        or identity.sampled_geometry_state_hash
        != problem.fd_residual_geometry_state_hash
        or identity.ordered_route_axis_hash != problem.fd_ordered_route_axis_hash
        or not same_geometric_coordinate(
            identity.finsler_path_budget,
            problem.finsler_path_budget,
        )
        or not same_geometric_coordinate(
            identity.quadratic_trust_energy_budget,
            problem.quadratic_trust_energy_budget,
        )
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            "selector source, geometry, axis, or trust-budget coordinates differ",
        )
    position_certificate = problem.sampled_proposal_position_certificate
    if position_certificate is None:
        if (
            problem.fd_residual_certificate_kind
            == PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND
            or identity.sampled_proposal_position_certificate_hash is not None
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled selector freeze lacks its proposal-position certificate",
            )
    else:
        if type(position_certificate) is not V1SampledProposalPositionCertificate:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "sampled selector freeze requires the exact proposal-position "
                "certificate type",
            )
        if (
            identity.sampled_proposal_position_certificate_hash
            != position_certificate.certificate_hash
            or position_certificate.current_mask_hash != current_mask.mask_hash
            or position_certificate.physical_route_axis_hash
            != identity.physical_route_axis_hash
            or position_certificate.ordered_route_axis_hash
            != identity.ordered_route_axis_hash
            or position_certificate.route_count != problem.route_count
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "sampled proposal-position certificate coordinates do not close",
            )
        if (
            position_certificate.proposal_positional_correspondence_authority
            is not True
            or position_certificate.pre_freeze_ordered_to_physical_application_authority
            is not False
            or position_certificate.irreversible_order_authority is not False
            or position_certificate.irreversible_application_authority is not False
            or position_certificate.fq_certification_authority is not False
            or position_certificate.bce_oracle_authority is not False
            or position_certificate.mask_commit_authority is not False
            or position_certificate.production_execution_authority is not False
            or position_certificate.terminal_authority is not False
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.AUTHORITY_VIOLATION,
                "proposal-position certificate crossed its neutral authority",
            )
    provenance = problem.chart_provenance
    if provenance is not None:
        if (
            identity.geometric_budget_hash
            != provenance.geometric_trust_budgets.budget_hash
            or identity.campaign_binding_hash
            != provenance.corrected_attribution_coordinates.campaign_binding_hash
            or identity.model_identity_hash
            != provenance.corrected_attribution_coordinates.model_seal_hash
            or identity.preprocessing_hash
            != provenance.corrected_attribution_coordinates.preprocessing_hash
            or identity.sampling_law_hash
            != provenance.corrected_attribution_coordinates.sampling_law_hash
            or identity.train_bank_source_hash
            != provenance.corrected_attribution_coordinates.train_bank_source_hash
            or identity.calibration_identity
            != provenance.fixed_affine_bce_calibration.calibration_state_id
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                "typed selector chart and sampled source coordinates differ",
            )
        if type(provenance) is V1SampledSelectorChartProvenance:
            snapshot = provenance.snapshot_certificate
            discovery = provenance.score_discovery_coordinates
            if (
                provenance.proposal_position_certificate.certificate_hash
                != identity.sampled_proposal_position_certificate_hash
                or snapshot.sampled_geometry_source_hash
                != identity.sampled_geometry_source_hash
                or snapshot.geometry_state_hash
                != identity.sampled_geometry_state_hash
                or snapshot.geometry_result_hash
                != identity.sampled_geometry_result_hash
                or snapshot.assembly_receipt_hash
                != identity.sampled_geometry_assembly_hash
                or snapshot.source_measurement_axis_hash
                != identity.source_measurement_axis_hash
                or snapshot.campaign_contract_hash
                != identity.campaign_contract_hash
                or snapshot.campaign_binding_hash
                != identity.campaign_binding_hash
                or snapshot.model_identity_hash != identity.model_identity_hash
                or snapshot.query_contract_hash != identity.query_contract_hash
                or snapshot.preprocessing_hash != identity.preprocessing_hash
                or snapshot.sampling_law_hash != identity.sampling_law_hash
                or snapshot.current_mask_hash != identity.current_mask_hash
                or snapshot.proposal_generation != identity.proposal_generation
                or snapshot.physical_route_axis_hash
                != identity.physical_route_axis_hash
                or snapshot.ordered_route_axis_hash
                != identity.ordered_route_axis_hash
                or snapshot.calibration_identity != identity.calibration_identity
                or snapshot.raw_logit_stream_source_hash
                != identity.raw_logit_stream_source_hash
                or snapshot.context_mixture_hash != identity.context_mixture_hash
                or discovery.registration_hash
                != identity.score_discovery_registration_hash
                or discovery.design_hash != identity.score_discovery_design_hash
                or discovery.look_receipt_hash
                != identity.score_discovery_look_receipt_hash
                or discovery.look_index != identity.score_discovery_look_index
                or discovery.result_hash != identity.score_discovery_result_hash
                or discovery.statistics_hash
                != identity.score_discovery_statistics_hash
                or discovery.evidence_hash != identity.score_discovery_evidence_hash
            ):
                _refuse(
                    V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
                    "sampled selector chart, discovery look, and geometry source "
                    "coordinates do not close",
                )
    elif result.receipt.production_chart_authority:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.AUTHORITY_VIOLATION,
            "selector claimed production chart authority without typed provenance",
        )
    if (
        result.receipt.frontier_exact
        or result.receipt.global_optimality_claim
        or result.receipt.signed_covector_used_for_selection
        or not result.receipt.no_layer_partitions_or_quotas
        or not result.receipt.no_sign_partitions_or_quotas
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.AUTHORITY_VIOLATION,
            "selector result crossed the admitted proposal authority",
        )
    if len(policy.scalarizations) != 1:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.MULTIPLE_FAMILIES,
            "one frozen ladder requires exactly one selector scalarization",
        )


def _validate_legality_endpoint(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    child: HardMask,
    packed_byte_chunk_size: int,
) -> None:
    def retained_between(start: int, stop: int) -> int:
        packed = np.frombuffer(child.packed_bits, dtype=np.uint8)
        first_byte = start // 8
        final_byte = (stop + 7) // 8
        retained = 0
        for byte_start in range(
            first_byte,
            final_byte,
            packed_byte_chunk_size,
        ):
            byte_stop = min(final_byte, byte_start + packed_byte_chunk_size)
            bits = np.unpackbits(packed[byte_start:byte_stop], bitorder="big")
            global_bit_start = byte_start * 8
            local_start = max(0, start - global_bit_start)
            local_stop = min(int(bits.size), stop - global_bit_start)
            if local_stop > local_start:
                retained += int(
                    np.sum(bits[local_start:local_stop], dtype=np.int64)
                )
        return retained

    if problem.legality_blocks is not None:
        if any(
            retained_between(block.start, block.stop)
            < problem.minimum_live_per_block
            for block in problem.legality_blocks
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                "selector endpoint violates compact survivor legality",
            )
    elif problem.legality_block_ids is not None:
        # Per-route legacy block IDs are constructor-limited to reference-size
        # problems; production axes use the bounded compact-block path above.
        live = child.to_bool_array()
        live_by_block: dict[str, int] = {}
        for block_id, retained in zip(problem.legality_block_ids, live, strict=True):
            if bool(retained):
                live_by_block[block_id] = live_by_block.get(block_id, 0) + 1
        if any(
            live_by_block.get(block_id, 0) < problem.minimum_live_per_block
            for block_id in set(problem.legality_block_ids)
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                "selector endpoint violates survivor legality",
            )


def _validate_candidate_metrics(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    candidate: V1ProductionGeometricCandidate,
    shell_chunks: Iterator[np.ndarray],
    running: dict[str, object],
    shell_digest: object,
) -> int:
    factor_sum = running["factor_sum"]
    if type(factor_sum) is not np.ndarray:  # pragma: no cover - internal invariant.
        raise RuntimeError("geometric metric accumulator lost its factor sum")
    shell_count = 0
    try:
        for chunk in shell_chunks:
            indices = np.asarray(chunk, dtype=np.int64).reshape(-1)
            if not bool(np.all(problem.eligible_mask[indices])):
                _refuse(
                    V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                    "selector endpoint contains an ineligible or non-live route",
                )
            normalized_chunk = np.asarray(
                chunk,
                dtype=np.dtype(SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE),
            )
            # ``hashlib`` exposes concrete OpenSSL implementations rather than
            # one stable public digest protocol, so this narrow duck-typed call
            # avoids binding the runtime contract to a private CPython type.
            getattr(shell_digest, "update")(memoryview(normalized_chunk).cast("B"))
            shell_count += int(normalized_chunk.size)
            running["resource"] = (
                float(running["resource"]) + problem.resource_sum(indices)
            )
            running["path"] = float(running["path"]) + float(
                np.sum(problem.corrected_path_cost[indices], dtype=np.float64)
            )
            if problem.signed_covector is not None:
                running["signed"] = float(running["signed"]) + float(
                    np.sum(problem.signed_covector[indices], dtype=np.float64)
                )
            residual = problem.residual_diagonal[indices]
            running["diagonal"] = float(running["diagonal"]) + float(
                np.sum(residual, dtype=np.float64)
            )
            running["residual_sqrt"] = float(running["residual_sqrt"]) + float(
                np.sum(np.sqrt(residual), dtype=np.float64)
            )
            if problem.rank:
                factor_sum += np.sum(
                    problem.low_rank_factor[:, indices],
                    axis=1,
                    dtype=np.float64,
                )
    except V1SampledGeometricFreezeRefusal:
        raise
    except ValueError:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.NON_NESTED_FRONTIER,
            "selector child mask reactivated a route from its preceding endpoint",
        )

    count = candidate.removal_count
    resource = float(running["resource"])
    path = float(running["path"])
    signed = float(running["signed"])
    diagonal = float(running["diagonal"])
    residual_sqrt = float(running["residual_sqrt"])
    certified_residual = min(
        problem.residual_spectral_bound * float(count),
        residual_sqrt * residual_sqrt,
    )
    low_rank = float(np.dot(factor_sum, factor_sum))
    quadratic = certified_residual + low_rank
    burden = _proposal_burden(
        path=path,
        quadratic=quadratic,
        tau_f=problem.finsler_path_budget,
        tau_q=problem.quadratic_trust_energy_budget,
    )
    exact_pairs = (
        (candidate.resource_gain, resource, "resource gain"),
        (candidate.corrected_path_cost, path, "corrected path cost"),
        (candidate.signed_covector_sum, signed, "signed covector sum"),
        (candidate.residual_diagonal_sum, diagonal, "residual diagonal sum"),
        (candidate.diagonal_surrogate_q, diagonal, "diagonal surrogate Q"),
        (
            candidate.residual_diagonal_sqrt_sum,
            residual_sqrt,
            "residual square-root sum",
        ),
        (
            candidate.certified_residual_quadratic,
            certified_residual,
            "certified residual quadratic",
        ),
        (candidate.low_rank_quadratic, low_rank, "low-rank quadratic"),
        (candidate.quadratic_trust_value, quadratic, "quadratic trust value"),
        (candidate.proposal_burden, burden, "proposal burden"),
    )
    for observed, expected, label in exact_pairs:
        if not _close(observed, expected):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                f"selector candidate {label} does not replay against its problem",
            )
    if (
        path > problem.finsler_path_budget
        and not _close(path, problem.finsler_path_budget)
    ) or (
        quadratic > problem.quadratic_trust_energy_budget
        and not _close(quadratic, problem.quadratic_trust_energy_budget)
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.BUDGET_VIOLATION,
            "selector endpoint exceeds its registered F or Q budget",
        )
    return shell_count


def freeze_v1_sampled_geometric_selector_ladder(
    *,
    selector_problem: V1ProductionGeometricSelectorProblem,
    selector_policy: V1ProductionGeometricSelectorPolicy,
    selector_result: V1ProductionGeometricSelectorResult,
    current_mask: HardMask,
    identity: V1SampledGeometricFreezeIdentity,
    packed_byte_chunk_size: int = _DEFAULT_PACKED_BYTE_CHUNK_SIZE,
) -> V1SampledGeometricFreeze:
    """Validate and freeze one sampled-geometric packed proposal frontier."""

    if type(selector_problem) is not V1ProductionGeometricSelectorProblem:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "geometric freeze requires an exact selector problem",
        )
    if type(selector_policy) is not V1ProductionGeometricSelectorPolicy:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "geometric freeze requires an exact selector policy",
        )
    if type(selector_result) is not V1ProductionGeometricSelectorResult:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "geometric freeze requires an exact selector result",
        )
    if type(current_mask) is not HardMask:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "geometric freeze requires an exact current HardMask",
        )
    if type(identity) is not V1SampledGeometricFreezeIdentity:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "geometric freeze requires its exact source identity",
        )
    byte_chunk_size = _positive_int(
        packed_byte_chunk_size,
        name="packed byte chunk size",
    )
    _validate_problem_result_graph(
        problem=selector_problem,
        policy=selector_policy,
        result=selector_result,
        current_mask=current_mask,
        identity=identity,
    )
    if not selector_result.frontier:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.EMPTY_FRONTIER,
            "cannot freeze an empty geometric selector frontier",
        )

    indexed = tuple(enumerate(selector_result.frontier))
    normalized = tuple(sorted(indexed, key=lambda item: item[1].removal_count))
    counts = tuple(candidate.removal_count for _index, candidate in normalized)
    if tuple(sorted(counts)) != counts or len(counts) != len(set(counts)):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.NON_NESTED_FRONTIER,
            "selector frontier does not define unique increasing removal counts",
        )
    scalarization_id = selector_policy.scalarizations[0].scalarization_id
    if any(
        candidate.scalarization_id != scalarization_id
        for _index, candidate in normalized
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.MULTIPLE_FAMILIES,
            "selector frontier mixes scalarization families",
        )

    previous_mask = current_mask
    previous_count = 0
    running: dict[str, object] = {
        "resource": 0.0,
        "path": 0.0,
        "signed": 0.0,
        "diagonal": 0.0,
        "residual_sqrt": 0.0,
        "factor_sum": np.zeros(selector_problem.rank, dtype=np.float64),
    }
    shell_digest = hashlib.sha256()
    frozen_actions: list[V1FrozenLadderAction] = []
    endpoint_candidate_hashes: list[str] = []
    endpoint_packed_action_hashes: list[str] = []
    endpoint_removed_hashes: list[str] = []
    endpoint_mask_hashes: list[str] = []
    shell_total = 0

    for endpoint_index, (source_index, candidate) in enumerate(normalized, start=1):
        if type(candidate) is not V1ProductionGeometricCandidate:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
                "selector frontier contains an unsupported candidate type",
            )
        action = candidate.action
        child = action.resulting_mask
        if (
            candidate.route_count != selector_problem.route_count
            or child.route_count != selector_problem.route_count
            or child.route_axis_hash != selector_problem.route_axis_hash
            or candidate.removal_count != action.removed_count
            or candidate.removal_count <= previous_count
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                "selector candidate/action route or count coordinates differ",
            )
        cumulative_count, cumulative_hash = child.removed_index_count_and_hash_from(
            current_mask
        )
        if (
            cumulative_count != candidate.removal_count
            or cumulative_hash != action.removed_route_index_hash
            or child.retained_count
            != current_mask.retained_count - candidate.removal_count
        ):
            _refuse(
                V1SampledGeometricFreezeRefusalCode.CANDIDATE_MISMATCH,
                "selector packed action delta identity does not replay",
            )
        shell_count = _validate_candidate_metrics(
            problem=selector_problem,
            candidate=candidate,
            shell_chunks=child.iter_removed_index_chunks_from(
                previous_mask,
                packed_byte_chunk_size=byte_chunk_size,
            ),
            running=running,
            shell_digest=shell_digest,
        )
        if shell_count <= 0 or shell_count != candidate.removal_count - previous_count:
            _refuse(
                V1SampledGeometricFreezeRefusalCode.NON_NESTED_FRONTIER,
                "selector endpoints are not strictly nested packed masks",
            )
        shell_total += shell_count
        _validate_legality_endpoint(
            problem=selector_problem,
            child=child,
            packed_byte_chunk_size=byte_chunk_size,
        )
        action_id = content_hash(
            {
                "schema": SAMPLED_GEOMETRIC_FREEZE_SCHEMA,
                "selector_result_hash": selector_result.result_hash,
                "source_frontier_index": source_index,
                "endpoint_index": endpoint_index,
                "candidate_hash": candidate.candidate_hash,
                "packed_action_hash": action.action_hash,
                "removed_count": candidate.removal_count,
                "removed_route_index_hash": action.removed_route_index_hash,
                "resulting_mask_hash": child.mask_hash,
            },
            prefix=_ACTION_ID_PREFIX,
        )
        frozen_actions.append(
            V1FrozenLadderAction(
                endpoint_index=endpoint_index,
                action_id=action_id,
                packed_removal_action=action,
            )
        )
        endpoint_candidate_hashes.append(candidate.candidate_hash)
        endpoint_packed_action_hashes.append(action.action_hash)
        endpoint_removed_hashes.append(action.removed_route_index_hash)
        endpoint_mask_hashes.append(child.mask_hash)
        previous_mask = child
        previous_count = candidate.removal_count

    if shell_total != counts[-1]:
        raise RuntimeError(
            "shell-order stream count diverged from the deepest endpoint"
        )
    frozen_action_axis = tuple(action.action_hash for action in frozen_actions)
    candidate_axis = tuple(endpoint_candidate_hashes)
    packed_action_axis = tuple(endpoint_packed_action_hashes)
    removed_hash_axis = tuple(endpoint_removed_hashes)
    mask_hash_axis = tuple(endpoint_mask_hashes)
    source_index_axis = tuple(
        source_index for source_index, _candidate in normalized
    )
    shell_sha256 = shell_digest.hexdigest()
    shell_stream_hash = content_hash(
        {
            "schema": SAMPLED_GEOMETRIC_SHELL_ORDER_SCHEMA,
            "dtype": SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE,
            "route_count": shell_total,
            "raw_stream_sha256": shell_sha256,
            "endpoint_removal_counts": counts,
            "endpoint_shell_counts": tuple(
                count - (0 if index == 0 else counts[index - 1])
                for index, count in enumerate(counts)
            ),
        },
        prefix=_SHELL_ORDER_PREFIX,
    )
    endpoint_action_axis_hash = content_hash(
        {
            "candidate_hashes": candidate_axis,
            "packed_action_hashes": packed_action_axis,
            "frozen_action_hashes": frozen_action_axis,
            "removed_route_index_hashes": removed_hash_axis,
            "resulting_mask_hashes": mask_hash_axis,
        },
        prefix=_ACTION_AXIS_PREFIX,
    )
    endpoint_count_axis_hash = content_hash(counts, prefix=_COUNT_AXIS_PREFIX)
    freeze_identity_hash = content_hash(
        _freeze_identity_payload(
            identity_hash=identity.identity_hash,
            selector_problem_hash=selector_problem.problem_hash,
            selector_policy_hash=selector_policy.policy_hash,
            selector_result_hash=selector_result.result_hash,
            selector_receipt_hash=selector_result.receipt.receipt_hash,
            normalized_source_frontier_indices=source_index_axis,
            endpoint_candidate_hashes=candidate_axis,
            endpoint_packed_action_hashes=packed_action_axis,
            endpoint_frozen_action_hashes=frozen_action_axis,
            endpoint_removed_route_index_hashes=removed_hash_axis,
            endpoint_removal_counts=counts,
            endpoint_resulting_mask_hashes=mask_hash_axis,
            shell_order_stream_hash=shell_stream_hash,
        ),
        prefix=_FREEZE_ID_PREFIX,
    )
    receipt = V1SampledGeometricReplayFreezeReceipt(
        identity_hash=identity.identity_hash,
        selector_problem_hash=selector_problem.problem_hash,
        selector_policy_hash=selector_policy.policy_hash,
        selector_result_hash=selector_result.result_hash,
        selector_receipt_hash=selector_result.receipt.receipt_hash,
        selector_source_receipt_hash=selector_problem.source_receipt_hash,
        selector_eligibility_receipt_hash=selector_problem.eligibility_receipt_hash,
        selector_fd_residual_certificate_hash=(
            selector_problem.fd_residual_certificate_hash
        ),
        geometry_state_hash=identity.sampled_geometry_state_hash,
        geometry_result_hash=identity.sampled_geometry_result_hash,
        geometry_assembly_hash=identity.sampled_geometry_assembly_hash,
        score_discovery_registration_hash=(
            identity.score_discovery_registration_hash
        ),
        score_discovery_design_hash=identity.score_discovery_design_hash,
        score_discovery_look_receipt_hash=(
            identity.score_discovery_look_receipt_hash
        ),
        score_discovery_look_index=identity.score_discovery_look_index,
        maximum_registered_score_looks=(
            identity.maximum_registered_score_looks
        ),
        sampled_proposal_position_certificate_hash=(
            identity.sampled_proposal_position_certificate_hash
        ),
        score_discovery_result_hash=identity.score_discovery_result_hash,
        score_discovery_statistics_hash=identity.score_discovery_statistics_hash,
        score_discovery_evidence_hash=identity.score_discovery_evidence_hash,
        parent_mask_hash=current_mask.mask_hash,
        route_count=current_mask.route_count,
        physical_route_axis_hash=current_mask.route_axis_hash,
        ordered_route_axis_hash=identity.ordered_route_axis_hash,
        finsler_path_budget=selector_problem.finsler_path_budget,
        quadratic_trust_energy_budget=(
            selector_problem.quadratic_trust_energy_budget
        ),
        geometric_budget_hash=identity.geometric_budget_hash,
        normalized_source_frontier_indices=source_index_axis,
        endpoint_candidate_hashes=candidate_axis,
        endpoint_packed_action_hashes=packed_action_axis,
        endpoint_frozen_action_hashes=frozen_action_axis,
        endpoint_removed_route_index_hashes=removed_hash_axis,
        endpoint_removal_counts=counts,
        endpoint_resulting_mask_hashes=mask_hash_axis,
        endpoint_action_axis_hash=endpoint_action_axis_hash,
        endpoint_count_axis_hash=endpoint_count_axis_hash,
        shell_order_stream_sha256=shell_sha256,
        shell_order_stream_hash=shell_stream_hash,
        shell_order_route_count=shell_total,
        shell_order_dtype=SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE,
        packed_scan_byte_chunk_size=byte_chunk_size,
        freeze_identity_hash=freeze_identity_hash,
    )
    ladder = V1FrozenCandidateLadder(
        campaign_contract_hash=identity.campaign_contract_hash,
        model_identity_hash=identity.model_identity_hash,
        query_contract_hash=identity.query_contract_hash,
        preprocessing_hash=identity.preprocessing_hash,
        current_mask_hash=current_mask.mask_hash,
        raw_logit_stream_source_hash=identity.raw_logit_stream_source_hash,
        sampling_law_hash=identity.sampling_law_hash,
        proposal_generation=identity.proposal_generation,
        route_count=current_mask.route_count,
        route_axis_hash=current_mask.route_axis_hash,
        actions=tuple(frozen_actions),
        discovery_context_ids=identity.discovery_context_ids,
        discovery_draw_ids=identity.discovery_draw_ids,
        discovery_evidence_hash=identity.score_discovery_evidence_hash,
        freeze_receipt_hash=receipt.receipt_hash,
        parent_mask=current_mask,
    )
    return V1SampledGeometricFreeze(
        identity=identity,
        selector_problem=selector_problem,
        selector_policy=selector_policy,
        selector_result=selector_result,
        ladder=ladder,
        replay_receipt=receipt,
    )


def composition_scope_from_v1_sampled_geometric_freeze(
    freeze: V1SampledGeometricFreeze,
) -> V1CompositionDecisionScopeReceipt:
    """Bind a sampled-geometric frozen ladder to the neutral F/Q scope.

    The geometric freeze owns selector, geometry, mask, endpoint, and complete
    score-discovery look coordinates.  No downstream caller may replace the
    discovery design, look receipt, or registered look axis while issuing the
    neutral scope.
    """

    if type(freeze) is not V1SampledGeometricFreeze:
        _refuse(
            V1SampledGeometricFreezeRefusalCode.INVALID_TYPE,
            "composition scope requires an exact sampled-geometric freeze",
        )
    _ = freeze.freeze_hash
    freeze.ladder.assert_integrity()
    identity = freeze.identity
    replay = freeze.replay_receipt
    ladder = freeze.ladder
    endpoint_count = len(ladder.actions)
    endpoint_indices = tuple(range(1, endpoint_count + 1))
    if (
        tuple(ladder.action_hashes) != replay.endpoint_frozen_action_hashes
        or tuple(ladder.endpoint_removed_counts) != replay.endpoint_removal_counts
        or ladder.current_mask_hash != identity.current_mask_hash
        or ladder.route_axis_hash != identity.physical_route_axis_hash
        or replay.ordered_route_axis_hash != identity.ordered_route_axis_hash
    ):
        _refuse(
            V1SampledGeometricFreezeRefusalCode.COORDINATE_MISMATCH,
            "sampled-geometric freeze and ladder coordinates do not close",
        )
    return V1CompositionDecisionScopeReceipt(
        composition_kind=SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
        discovery_score_registration_hash=(
            identity.score_discovery_registration_hash
        ),
        discovery_score_design_hash=identity.score_discovery_design_hash,
        discovery_score_look_receipt_hash=(
            identity.score_discovery_look_receipt_hash
        ),
        discovery_score_look_index=identity.score_discovery_look_index,
        maximum_registered_discovery_score_looks=(
            identity.maximum_registered_score_looks
        ),
        sampled_geometry_state_hash=identity.sampled_geometry_state_hash,
        sampled_geometry_result_hash=identity.sampled_geometry_result_hash,
        sampled_geometry_assembly_hash=identity.sampled_geometry_assembly_hash,
        sampled_selector_problem_hash=freeze.selector_problem.problem_hash,
        sampled_selector_policy_hash=freeze.selector_policy.policy_hash,
        sampled_selector_receipt_hash=(
            freeze.selector_result.receipt.receipt_hash
        ),
        sampled_selector_result_hash=freeze.selector_result.result_hash,
        sampled_proposal_position_certificate_hash=(
            identity.sampled_proposal_position_certificate_hash
        ),
        composition_freeze_identity_hash=identity.identity_hash,
        composition_freeze_replay_receipt_hash=replay.receipt_hash,
        frozen_candidate_ladder_hash=ladder.ladder_hash,
        endpoint_action_hashes=ladder.action_hashes,
        endpoint_new_removal_counts=ladder.endpoint_removed_counts,
        composition_frozen_endpoint_indices=endpoint_indices,
        composition_provisional_endpoint_indices=(),
        decision_competitive_endpoint_indices=endpoint_indices,
        current_mask_hash=ladder.current_mask_hash,
        proposal_generation=ladder.proposal_generation,
        route_count=ladder.route_count,
        physical_route_axis_hash=identity.physical_route_axis_hash,
        ordered_route_axis_hash=identity.ordered_route_axis_hash,
    )


__all__ = [
    "SAMPLED_GEOMETRIC_FREEZE_AUTHORITY_SCOPE",
    "SAMPLED_GEOMETRIC_FREEZE_IDENTITY_SCHEMA",
    "SAMPLED_GEOMETRIC_FREEZE_SCHEMA",
    "SAMPLED_GEOMETRIC_REPLAY_FREEZE_RECEIPT_SCHEMA",
    "SAMPLED_GEOMETRIC_SHELL_ORDER_DTYPE",
    "SAMPLED_GEOMETRIC_SHELL_ORDER_SCHEMA",
    "V1SampledGeometricFreeze",
    "V1SampledGeometricFreezeIdentity",
    "V1SampledGeometricFreezeRefusal",
    "V1SampledGeometricFreezeRefusalCode",
    "V1SampledGeometricReplayFreezeReceipt",
    "composition_scope_from_v1_sampled_geometric_freeze",
    "freeze_v1_sampled_geometric_selector_ladder",
]
