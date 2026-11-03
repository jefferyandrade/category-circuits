"""Close paired C-stream corrected-work and literal-Jacobian projections.

The directional secant provider needs two scalar projection tables on exactly
the same frozen C coordinates.  The generic ladder projector intentionally
closes one row stream at a time, so neither one-stream acquisition can prove
that corrected evidence and the literal relaxed-gate Jacobian came from one
capture and one target-logit VJP.  This module supplies that missing, bounded
joint seam.

The pre-outcome plan registers exactly one request for each dual-chart role.
The post-execution receipt binds their independently closed acquisitions to a
single scorer same-capture/VJP execution proof and copies only the four
``[context, endpoint]`` center/radius matrices.  Route rows are never retained.
The resulting proof is input evidence for directional secants only; it grants
no F/Q/R certification, BCE-oracle, mask-commit, or terminal authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from string import hexdigits
from typing import Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_ladder_projection_gpu import (
    V1DirectionalActionProjectionRequest,
    V1LadderProjectionAcquisition,
    V1LadderProjectionAcquisitionClosure,
    V1LadderProjectionResult,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.modern_v1_directional_secant_provider import (
    directional_projection_hash,
)


DIRECTIONAL_DUAL_PROJECTION_PLAN_SCHEMA: Final = (
    "partition_free_v1_directional_dual_projection_plan_v1"
)
DIRECTIONAL_DUAL_PROJECTION_SCORER_PROOF_SCHEMA: Final = (
    "partition_free_v1_directional_dual_projection_scorer_proof_v1"
)
DIRECTIONAL_DUAL_PROJECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_directional_dual_projection_receipt_v1"
)
DIRECTIONAL_DUAL_PROJECTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_directional_dual_projection_result_v1"
)
DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE: Final = (
    "native_directional_secant_dual_projection_input_only_"
    "no_fqr_or_oracle_or_commit_or_terminal_authority"
)

_PLAN_PREFIX: Final = "partition-free-v1-directional-dual-projection-plan-"
_SCORER_PROOF_PREFIX: Final = (
    "partition-free-v1-directional-dual-projection-scorer-proof-"
)
_RECEIPT_PREFIX: Final = "partition-free-v1-directional-dual-projection-receipt-"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _sha256_identity(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if (
        len(result) != 64
        or result.lower() != result
        or any(character not in hexdigits.lower() for character in result)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 identity")
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in tuple(values))
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _immutable_matrix(
    value: object,
    *,
    shape: tuple[int, int],
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.array(value, dtype="<f8", order="C", copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite float64 matrix of shape {shape}")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    bits = array.reshape(-1).view(np.uint64)
    bits[bits == np.uint64(1 << 63)] = np.uint64(0)
    result = np.frombuffer(array.tobytes(order="C"), dtype="<f8").reshape(shape)
    result.setflags(write=False)
    return result


def _request_pair_payload(
    evidence: V1DirectionalActionProjectionRequest,
    jacobian: V1DirectionalActionProjectionRequest,
) -> dict[str, object]:
    return {
        "design_hash": evidence.design_hash,
        "ladder_hash": evidence.ladder_hash,
        "chart_identity_hash": evidence.design.chart_identity_hash,
        "joint_fqr_family_registration_hash": (
            evidence.design.joint_fqr_family_registration_hash
        ),
        "current_mask_hash": evidence.current_mask_hash,
        "route_count": evidence.route_count,
        "physical_mask_axis_hash": evidence.physical_mask_axis_hash,
        "ordered_score_route_axis_hash": evidence.ordered_score_route_axis_hash,
        "endpoint_action_hashes": evidence.endpoint_action_hashes,
        "shell_hashes": evidence.shell_hashes,
        "shell_removed_counts": evidence.shell_removed_counts,
        "expected_context_ids": evidence.expected_context_ids,
        "expected_draw_ids": evidence.expected_draw_ids,
        "expected_stratum_ids": evidence.expected_stratum_ids,
        "sealed_order_start_indices_by_stratum": (
            evidence.sealed_order_start_indices_by_stratum
        ),
        "csr_hash": evidence.csr.csr_hash,
        "destination_transport_hash": evidence.destination_transport_hash,
        "registered_route_chunk_size": evidence.registered_route_chunk_size,
        "projection_kernel_source_hash": evidence.projection_kernel_source_hash,
        "corrected_evidence_stream_identity_hash": (
            evidence.corrected_evidence_stream_identity_hash
        ),
        "literal_jacobian_stream_identity_hash": (
            evidence.literal_jacobian_stream_identity_hash
        ),
        "same_capture_and_vjp_registration_hash": (
            evidence.same_capture_and_vjp_registration_hash
        ),
        "corrected_request_hash": evidence.request_hash,
        "literal_jacobian_request_hash": jacobian.request_hash,
        "corrected_scorer_request_identity_hash": (
            evidence.scorer_request_identity_hash
        ),
        "literal_jacobian_scorer_request_identity_hash": (
            jacobian.scorer_request_identity_hash
        ),
        "corrected_acquisition_lineage_hash": evidence.acquisition_lineage_hash,
        "literal_jacobian_acquisition_lineage_hash": (
            jacobian.acquisition_lineage_hash
        ),
    }


def _validate_request_pair(
    requests: Sequence[V1DirectionalActionProjectionRequest],
) -> tuple[
    V1DirectionalActionProjectionRequest,
    V1DirectionalActionProjectionRequest,
]:
    values = tuple(requests)
    if len(values) != 2 or any(
        type(value) is not V1DirectionalActionProjectionRequest for value in values
    ):
        raise TypeError(
            "directional dual projection requires exactly two typed requests"
        )
    for request in values:
        request.assert_integrity()
    by_role: dict[str, V1DirectionalActionProjectionRequest] = {}
    for request in values:
        if request.projection_role in by_role:
            raise ValueError(
                "directional dual projection requires exactly one request per role"
            )
        by_role[request.projection_role] = request
    if set(by_role) != {"corrected_evidence", "literal_jacobian"}:
        raise ValueError(
            "directional dual projection requires corrected evidence and literal J"
        )
    evidence = by_role["corrected_evidence"]
    jacobian = by_role["literal_jacobian"]
    paired_names = (
        "design_hash",
        "ladder_hash",
        "current_mask_hash",
        "route_count",
        "physical_mask_axis_hash",
        "ordered_score_route_axis_hash",
        "endpoint_action_hashes",
        "shell_hashes",
        "shell_removed_counts",
        "expected_context_ids",
        "expected_draw_ids",
        "expected_stratum_ids",
        "sealed_order_start_indices_by_stratum",
        "destination_transport_hash",
        "registered_route_chunk_size",
        "projection_kernel_source_hash",
        "corrected_evidence_stream_identity_hash",
        "literal_jacobian_stream_identity_hash",
        "same_capture_and_vjp_registration_hash",
    )
    mismatches = tuple(
        name for name in paired_names if getattr(evidence, name) != getattr(jacobian, name)
    )
    if mismatches:
        raise ValueError(
            "directional dual projection request coordinates differ: "
            + ", ".join(mismatches)
        )
    if evidence.csr.csr_hash != jacobian.csr.csr_hash:
        raise ValueError("directional dual projection CSR coordinates differ")
    if (
        evidence.design.chart_identity_hash != jacobian.design.chart_identity_hash
        or evidence.design.joint_fqr_family_registration_hash
        != jacobian.design.joint_fqr_family_registration_hash
    ):
        raise ValueError("directional dual projection chart registration differs")
    if (
        evidence.raw_logit_stream_identity_hash
        != evidence.corrected_evidence_stream_identity_hash
        or jacobian.raw_logit_stream_identity_hash
        != jacobian.literal_jacobian_stream_identity_hash
        or evidence.raw_logit_stream_identity_hash
        == jacobian.raw_logit_stream_identity_hash
    ):
        raise ValueError("directional dual projection row streams were swapped or collapsed")
    return evidence, jacobian


@dataclass(frozen=True, slots=True)
class V1DirectionalDualProjectionPlan:
    """Pre-outcome registration of the two same-capture C projections."""

    requests: tuple[V1DirectionalActionProjectionRequest, ...] = field(
        repr=False, compare=False
    )
    issued_before_projection_outcomes: bool = True
    same_capture_and_vjp_required: bool = True
    route_rows_may_be_retained: bool = False
    fqr_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE
    schema: str = DIRECTIONAL_DUAL_PROJECTION_PLAN_SCHEMA
    _plan_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        evidence, jacobian = _validate_request_pair(self.requests)
        if (
            self.issued_before_projection_outcomes is not True
            or self.same_capture_and_vjp_required is not True
            or self.route_rows_may_be_retained is not False
            or any(
                (
                    self.fqr_certification_authority,
                    self.bce_oracle_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                )
            )
        ):
            raise PermissionError("directional dual projection plan exceeded its authority")
        if self.authority_scope != DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE:
            raise ValueError("directional dual projection authority scope is invalid")
        if self.schema != DIRECTIONAL_DUAL_PROJECTION_PLAN_SCHEMA:
            raise ValueError("directional dual projection plan schema is invalid")
        object.__setattr__(self, "requests", (evidence, jacobian))
        self.assert_integrity()

    @property
    def corrected_evidence_request(self) -> V1DirectionalActionProjectionRequest:
        return self.requests[0]

    @property
    def literal_jacobian_request(self) -> V1DirectionalActionProjectionRequest:
        return self.requests[1]

    @property
    def design_hash(self) -> str:
        return self.corrected_evidence_request.design_hash

    @property
    def ladder_hash(self) -> str:
        return self.corrected_evidence_request.ladder_hash

    @property
    def chart_identity_hash(self) -> str:
        return self.corrected_evidence_request.design.chart_identity_hash

    @property
    def context_ids(self) -> tuple[str, ...]:
        return self.corrected_evidence_request.expected_context_ids

    @property
    def draw_ids(self) -> tuple[str, ...]:
        return self.corrected_evidence_request.expected_draw_ids

    @property
    def stratum_ids(self) -> tuple[str, ...]:
        return self.corrected_evidence_request.expected_stratum_ids

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.corrected_evidence_request.endpoint_action_hashes

    @property
    def plan_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        evidence, jacobian = _validate_request_pair(self.requests)
        return {
            "schema": self.schema,
            **_request_pair_payload(evidence, jacobian),
            "issued_before_projection_outcomes": self.issued_before_projection_outcomes,
            "same_capture_and_vjp_required": self.same_capture_and_vjp_required,
            "route_rows_may_be_retained": self.route_rows_may_be_retained,
            "fqr_certification_authority": self.fqr_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PLAN_PREFIX)
        if self._plan_hash is not None and observed != self._plan_hash:
            raise ValueError("directional dual projection plan was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional dual projection plan hash does not match")
        object.__setattr__(self, "_plan_hash", observed)
        return observed


def _validate_acquisition(
    *,
    request: V1DirectionalActionProjectionRequest,
    acquisition: V1LadderProjectionAcquisition,
) -> V1LadderProjectionAcquisition:
    if type(acquisition) is not V1LadderProjectionAcquisition:
        raise TypeError("directional dual projection requires exact acquisitions")
    request.assert_integrity()
    if (
        type(acquisition.result) is not V1LadderProjectionResult
        or type(acquisition.closure) is not V1LadderProjectionAcquisitionClosure
    ):
        raise TypeError("directional projection acquisition children are not exact")
    # Reconstruct both typed containers to re-run all packet, matrix, closure,
    # and certifier-record checks even after hostile object-level mutation.
    checked_result = V1LadderProjectionResult(
        receipt=acquisition.result.receipt,
        packet_receipts=acquisition.result.packet_receipts,
        signed_projection_matrix=acquisition.result.signed_projection_matrix,
        observed_absolute_sum_matrix=(
            acquisition.result.observed_absolute_sum_matrix
        ),
        absolute_input_sum_upper_matrix=(
            acquisition.result.absolute_input_sum_upper_matrix
        ),
        numerical_radius_matrix=acquisition.result.numerical_radius_matrix,
        numerical_lower_matrix=acquisition.result.numerical_lower_matrix,
        numerical_upper_matrix=acquisition.result.numerical_upper_matrix,
    )
    checked = V1LadderProjectionAcquisition(
        result=checked_result,
        closure=acquisition.closure,
        projection_records=acquisition.projection_records,
    )
    receipt = checked.result.receipt
    closure = checked.closure
    expected_context_axis_hash = content_hash(
        request.expected_context_ids, prefix="irg-device-panel-contexts-"
    )
    expected_draw_axis_hash = content_hash(
        request.expected_draw_ids,
        prefix="partition-free-v1-projection-draw-axis-",
    )
    expected_stratum_axis_hash = content_hash(
        request.expected_stratum_ids,
        prefix="partition-free-v1-projection-stratum-axis-",
    )
    if (
        closure.request_hash != request.request_hash
        or receipt.request_hash != request.request_hash
        or closure.design_hash != request.design_hash
        or receipt.design_hash != request.design_hash
        or closure.ladder_hash != request.ladder_hash
        or receipt.ladder_hash != request.ladder_hash
        or receipt.csr_hash != request.csr.csr_hash
        or receipt.current_mask_hash != request.current_mask_hash
        or receipt.route_count != request.route_count
        or closure.physical_mask_axis_hash != request.physical_mask_axis_hash
        or receipt.physical_mask_axis_hash != request.physical_mask_axis_hash
        or closure.ordered_score_route_axis_hash
        != request.ordered_score_route_axis_hash
        or receipt.ordered_score_route_axis_hash
        != request.ordered_score_route_axis_hash
        or receipt.raw_logit_stream_identity_hash
        != request.raw_logit_stream_identity_hash
        or receipt.destination_transport_hash != request.destination_transport_hash
        or closure.scorer_request_identity_hash
        != request.scorer_request_identity_hash
        or receipt.scorer_request_identity_hash
        != request.scorer_request_identity_hash
        or closure.acquisition_lineage_hash != request.acquisition_lineage_hash
        or receipt.acquisition_lineage_hash != request.acquisition_lineage_hash
        or receipt.context_ids != request.expected_context_ids
        or receipt.draw_ids != request.expected_draw_ids
        or receipt.stratum_ids != request.expected_stratum_ids
        or receipt.endpoint_action_hashes != request.endpoint_action_hashes
        or receipt.shell_hashes != request.shell_hashes
        or receipt.shell_removed_counts != request.shell_removed_counts
        or closure.context_axis_hash != expected_context_axis_hash
        or closure.draw_axis_hash != expected_draw_axis_hash
        or closure.stratum_axis_hash != expected_stratum_axis_hash
    ):
        raise ValueError(
            "directional projection acquisition is stale or has mismatched child axes"
        )
    center = checked.result.signed_projection_matrix
    radius = checked.result.numerical_radius_matrix
    lower = checked.result.numerical_lower_matrix
    upper = checked.result.numerical_upper_matrix
    center_minus_radius = center - radius
    center_plus_radius = center + radius
    if (
        not np.isfinite(center_minus_radius).all()
        or not np.isfinite(center_plus_radius).all()
        or np.any(lower > center_minus_radius)
        or np.any(upper < center_plus_radius)
    ):
        raise ValueError("directional projection numerical enclosure is not outward")
    return checked


@dataclass(frozen=True, slots=True)
class V1DirectionalDualProjectionScorerExecutionProof:
    """Typed scorer witness that both projection children share one VJP."""

    plan_hash: str
    corrected_request_hash: str
    literal_jacobian_request_hash: str
    corrected_acquisition_closure_hash: str
    literal_jacobian_acquisition_closure_hash: str
    corrected_scorer_receipt_hash: str
    literal_jacobian_scorer_receipt_hash: str
    same_capture_and_vjp_registration_hash: str
    scorer_same_capture_and_vjp_execution_proof_hash: str
    exactly_one_capture: bool = True
    exactly_one_target_logit_vjp: bool = True
    corrected_and_jacobian_children_closed: bool = True
    fqr_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE
    schema: str = DIRECTIONAL_DUAL_PROJECTION_SCORER_PROOF_SCHEMA
    _proof_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_hash",
            "corrected_request_hash",
            "literal_jacobian_request_hash",
            "corrected_acquisition_closure_hash",
            "literal_jacobian_acquisition_closure_hash",
            "corrected_scorer_receipt_hash",
            "literal_jacobian_scorer_receipt_hash",
            "same_capture_and_vjp_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "scorer_same_capture_and_vjp_execution_proof_hash",
            _sha256_identity(
                self.scorer_same_capture_and_vjp_execution_proof_hash,
                name="scorer same-capture/VJP execution proof hash",
            ),
        )
        if (
            self.exactly_one_capture is not True
            or self.exactly_one_target_logit_vjp is not True
            or self.corrected_and_jacobian_children_closed is not True
            or any(
                (
                    self.fqr_certification_authority,
                    self.bce_oracle_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                )
            )
        ):
            raise PermissionError("directional dual scorer proof exceeded its authority")
        if self.authority_scope != DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE:
            raise ValueError("directional dual scorer proof scope is invalid")
        if self.schema != DIRECTIONAL_DUAL_PROJECTION_SCORER_PROOF_SCHEMA:
            raise ValueError("directional dual scorer proof schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "plan_hash",
                "corrected_request_hash",
                "literal_jacobian_request_hash",
                "corrected_acquisition_closure_hash",
                "literal_jacobian_acquisition_closure_hash",
                "corrected_scorer_receipt_hash",
                "literal_jacobian_scorer_receipt_hash",
                "same_capture_and_vjp_registration_hash",
                "scorer_same_capture_and_vjp_execution_proof_hash",
                "exactly_one_capture",
                "exactly_one_target_logit_vjp",
                "corrected_and_jacobian_children_closed",
                "fqr_certification_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def proof_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_SCORER_PROOF_PREFIX)
        if self._proof_hash is not None and observed != self._proof_hash:
            raise ValueError("directional dual scorer proof was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional dual scorer proof hash does not match")
        object.__setattr__(self, "_proof_hash", observed)
        return observed


def bind_v1_directional_dual_projection_scorer_proof(
    *,
    plan: V1DirectionalDualProjectionPlan,
    corrected_evidence_acquisition: V1LadderProjectionAcquisition,
    literal_jacobian_acquisition: V1LadderProjectionAcquisition,
    scorer_same_capture_and_vjp_execution_proof_hash: str,
) -> V1DirectionalDualProjectionScorerExecutionProof:
    """Bind one scorer-native same-capture/VJP witness to both children."""

    if type(plan) is not V1DirectionalDualProjectionPlan:
        raise TypeError("directional dual scorer proof requires the exact plan")
    plan_hash = plan.assert_integrity()
    evidence = _validate_acquisition(
        request=plan.corrected_evidence_request,
        acquisition=corrected_evidence_acquisition,
    )
    jacobian = _validate_acquisition(
        request=plan.literal_jacobian_request,
        acquisition=literal_jacobian_acquisition,
    )
    return V1DirectionalDualProjectionScorerExecutionProof(
        plan_hash=plan_hash,
        corrected_request_hash=plan.corrected_evidence_request.request_hash,
        literal_jacobian_request_hash=plan.literal_jacobian_request.request_hash,
        corrected_acquisition_closure_hash=evidence.closure.closure_hash,
        literal_jacobian_acquisition_closure_hash=jacobian.closure.closure_hash,
        corrected_scorer_receipt_hash=evidence.closure.scorer_receipt_hash,
        literal_jacobian_scorer_receipt_hash=jacobian.closure.scorer_receipt_hash,
        same_capture_and_vjp_registration_hash=(
            plan.corrected_evidence_request.same_capture_and_vjp_registration_hash
        ),
        scorer_same_capture_and_vjp_execution_proof_hash=(
            scorer_same_capture_and_vjp_execution_proof_hash
        ),
    )


@dataclass(frozen=True, slots=True)
class V1DirectionalDualProjectionReceipt:
    """Compact, content-addressed proof for the four provider input tables."""

    plan_hash: str
    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    joint_fqr_family_registration_hash: str
    current_mask_hash: str
    route_count: int
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    csr_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    endpoint_action_hashes: tuple[str, ...]
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    same_capture_and_vjp_registration_hash: str
    corrected_request_hash: str
    literal_jacobian_request_hash: str
    corrected_acquisition_closure_hash: str
    literal_jacobian_acquisition_closure_hash: str
    corrected_projection_receipt_hash: str
    literal_jacobian_projection_receipt_hash: str
    corrected_scorer_receipt_hash: str
    literal_jacobian_scorer_receipt_hash: str
    scorer_same_capture_and_vjp_execution_proof_hash: str
    typed_scorer_proof_hash: str | None
    corrected_projection_hash: str
    jacobian_projection_hash: str
    same_capture_and_vjp: bool = True
    route_rows_retained: bool = False
    fqr_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE
    schema: str = DIRECTIONAL_DUAL_PROJECTION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "plan_hash",
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "joint_fqr_family_registration_hash",
            "current_mask_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "csr_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "corrected_request_hash",
            "literal_jacobian_request_hash",
            "corrected_acquisition_closure_hash",
            "literal_jacobian_acquisition_closure_hash",
            "corrected_projection_receipt_hash",
            "literal_jacobian_projection_receipt_hash",
            "corrected_scorer_receipt_hash",
            "literal_jacobian_scorer_receipt_hash",
            "corrected_projection_hash",
            "jacobian_projection_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "scorer_same_capture_and_vjp_execution_proof_hash",
            _sha256_identity(
                self.scorer_same_capture_and_vjp_execution_proof_hash,
                name="scorer same-capture/VJP execution proof hash",
            ),
        )
        if self.typed_scorer_proof_hash is not None:
            object.__setattr__(
                self,
                "typed_scorer_proof_hash",
                _identity(self.typed_scorer_proof_hash, name="typed scorer proof hash"),
            )
        if isinstance(self.route_count, bool) or int(self.route_count) != self.route_count:
            raise ValueError("directional dual route count must be an integer")
        object.__setattr__(self, "route_count", int(self.route_count))
        if self.route_count <= 0:
            raise ValueError("directional dual route count must be positive")
        contexts = _identity_axis(self.context_ids, name="dual projection context axis")
        draws = _identity_axis(self.draw_ids, name="dual projection draw axis")
        strata = _identity_axis(self.stratum_ids, name="dual projection stratum axis")
        endpoints = _identity_axis(
            self.endpoint_action_hashes, name="dual projection endpoint axis"
        )
        starts_list: list[tuple[str, int]] = []
        for name, index in self.sealed_order_start_indices_by_stratum:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError("directional dual projection starts must be integers")
            starts_list.append(
                (_identity(name, name="dual projection start stratum"), int(index))
            )
        starts = tuple(starts_list)
        if (
            not (len(contexts) == len(draws) == len(strata))
            or len(contexts) != len(set(contexts))
            or len(draws) != len(set(draws))
            or any(index < 0 for _, index in starts)
            or len(starts) != len({name for name, _ in starts})
        ):
            raise ValueError("directional dual projection row/start axes are invalid")
        if (
            self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
        ):
            raise ValueError("directional dual projection streams collapsed")
        if (
            self.same_capture_and_vjp is not True
            or self.route_rows_retained is not False
            or any(
                (
                    self.fqr_certification_authority,
                    self.bce_oracle_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                )
            )
        ):
            raise PermissionError("directional dual projection receipt exceeded authority")
        if self.authority_scope != DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE:
            raise ValueError("directional dual projection receipt scope is invalid")
        if self.schema != DIRECTIONAL_DUAL_PROJECTION_RECEIPT_SCHEMA:
            raise ValueError("directional dual projection receipt schema is invalid")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "sealed_order_start_indices_by_stratum", starts)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "plan_hash",
                "design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "joint_fqr_family_registration_hash",
                "current_mask_hash",
                "route_count",
                "physical_mask_axis_hash",
                "ordered_score_route_axis_hash",
                "csr_hash",
                "context_ids",
                "draw_ids",
                "stratum_ids",
                "sealed_order_start_indices_by_stratum",
                "endpoint_action_hashes",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "same_capture_and_vjp_registration_hash",
                "corrected_request_hash",
                "literal_jacobian_request_hash",
                "corrected_acquisition_closure_hash",
                "literal_jacobian_acquisition_closure_hash",
                "corrected_projection_receipt_hash",
                "literal_jacobian_projection_receipt_hash",
                "corrected_scorer_receipt_hash",
                "literal_jacobian_scorer_receipt_hash",
                "scorer_same_capture_and_vjp_execution_proof_hash",
                "typed_scorer_proof_hash",
                "corrected_projection_hash",
                "jacobian_projection_hash",
                "same_capture_and_vjp",
                "route_rows_retained",
                "fqr_certification_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def dual_chart_projection_proof_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("directional dual projection receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional dual projection receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DirectionalDualProjectionResult:
    """Immutable provider-ready dual projection matrices and their receipt."""

    receipt: V1DirectionalDualProjectionReceipt
    corrected_projection_center_by_context_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    corrected_projection_radius_by_context_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    jacobian_projection_center_by_context_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    jacobian_projection_radius_by_context_endpoint: np.ndarray = field(
        repr=False, compare=False
    )
    schema: str = DIRECTIONAL_DUAL_PROJECTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1DirectionalDualProjectionReceipt:
            raise TypeError("directional dual projection result requires its receipt")
        self.receipt.assert_integrity()
        shape = (len(self.receipt.context_ids), len(self.receipt.endpoint_action_hashes))
        matrices = {
            "corrected_projection_center_by_context_endpoint": _immutable_matrix(
                self.corrected_projection_center_by_context_endpoint,
                shape=shape,
                name="corrected projection centers",
            ),
            "corrected_projection_radius_by_context_endpoint": _immutable_matrix(
                self.corrected_projection_radius_by_context_endpoint,
                shape=shape,
                name="corrected projection radii",
                nonnegative=True,
            ),
            "jacobian_projection_center_by_context_endpoint": _immutable_matrix(
                self.jacobian_projection_center_by_context_endpoint,
                shape=shape,
                name="Jacobian projection centers",
            ),
            "jacobian_projection_radius_by_context_endpoint": _immutable_matrix(
                self.jacobian_projection_radius_by_context_endpoint,
                shape=shape,
                name="Jacobian projection radii",
                nonnegative=True,
            ),
        }
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)
        if self.schema != DIRECTIONAL_DUAL_PROJECTION_RESULT_SCHEMA:
            raise ValueError("directional dual projection result schema is invalid")
        self.assert_integrity()

    @property
    def corrected_projection_hash(self) -> str:
        return self.receipt.corrected_projection_hash

    @property
    def jacobian_projection_hash(self) -> str:
        return self.receipt.jacobian_projection_hash

    @property
    def dual_chart_projection_proof_hash(self) -> str:
        return self.receipt.dual_chart_projection_proof_hash

    def assert_integrity(self, expected_proof_hash: str | None = None) -> str:
        proof_hash = self.receipt.assert_integrity(expected_proof_hash)
        corrected = directional_projection_hash(
            kind="corrected_evidence",
            design_hash=self.receipt.design_hash,
            chart_identity_hash=self.receipt.chart_identity_hash,
            context_ids=self.receipt.context_ids,
            endpoint_action_hashes=self.receipt.endpoint_action_hashes,
            centers=self.corrected_projection_center_by_context_endpoint,
            radii=self.corrected_projection_radius_by_context_endpoint,
        )
        jacobian = directional_projection_hash(
            kind="literal_positive_removal_jacobian",
            design_hash=self.receipt.design_hash,
            chart_identity_hash=self.receipt.chart_identity_hash,
            context_ids=self.receipt.context_ids,
            endpoint_action_hashes=self.receipt.endpoint_action_hashes,
            centers=self.jacobian_projection_center_by_context_endpoint,
            radii=self.jacobian_projection_radius_by_context_endpoint,
        )
        if (
            corrected != self.receipt.corrected_projection_hash
            or jacobian != self.receipt.jacobian_projection_hash
        ):
            raise ValueError("directional dual projection matrices were tampered")
        return proof_hash


def close_v1_directional_dual_projection(
    *,
    plan: V1DirectionalDualProjectionPlan,
    corrected_evidence_acquisition: V1LadderProjectionAcquisition,
    literal_jacobian_acquisition: V1LadderProjectionAcquisition,
    scorer_same_capture_and_vjp_execution_proof: (
        V1DirectionalDualProjectionScorerExecutionProof | str
    ),
) -> V1DirectionalDualProjectionResult:
    """Close the dual projection without retaining either route-row stream."""

    if type(plan) is not V1DirectionalDualProjectionPlan:
        raise TypeError("directional dual projection requires the exact plan")
    plan_hash = plan.assert_integrity()
    evidence = _validate_acquisition(
        request=plan.corrected_evidence_request,
        acquisition=corrected_evidence_acquisition,
    )
    jacobian = _validate_acquisition(
        request=plan.literal_jacobian_request,
        acquisition=literal_jacobian_acquisition,
    )
    proof = scorer_same_capture_and_vjp_execution_proof
    typed_proof_hash: str | None
    if type(proof) is V1DirectionalDualProjectionScorerExecutionProof:
        typed_proof_hash = proof.assert_integrity()
        expected = (
            plan_hash,
            plan.corrected_evidence_request.request_hash,
            plan.literal_jacobian_request.request_hash,
            evidence.closure.closure_hash,
            jacobian.closure.closure_hash,
            evidence.closure.scorer_receipt_hash,
            jacobian.closure.scorer_receipt_hash,
            plan.corrected_evidence_request.same_capture_and_vjp_registration_hash,
        )
        observed = (
            proof.plan_hash,
            proof.corrected_request_hash,
            proof.literal_jacobian_request_hash,
            proof.corrected_acquisition_closure_hash,
            proof.literal_jacobian_acquisition_closure_hash,
            proof.corrected_scorer_receipt_hash,
            proof.literal_jacobian_scorer_receipt_hash,
            proof.same_capture_and_vjp_registration_hash,
        )
        if observed != expected:
            raise ValueError("typed same-capture/VJP proof names other children")
        scorer_proof_hash = proof.scorer_same_capture_and_vjp_execution_proof_hash
    elif type(proof) is str:
        typed_proof_hash = None
        scorer_proof_hash = _sha256_identity(
            proof, name="explicit scorer same-capture/VJP execution proof hash"
        )
    else:
        raise TypeError(
            "same-capture/VJP proof must be typed or an explicit SHA-256 hash"
        )

    evidence_center = evidence.result.signed_projection_matrix
    evidence_radius = evidence.result.numerical_radius_matrix
    jacobian_center = jacobian.result.signed_projection_matrix
    jacobian_radius = jacobian.result.numerical_radius_matrix
    corrected_hash = directional_projection_hash(
        kind="corrected_evidence",
        design_hash=plan.design_hash,
        chart_identity_hash=plan.chart_identity_hash,
        context_ids=plan.context_ids,
        endpoint_action_hashes=plan.endpoint_action_hashes,
        centers=evidence_center,
        radii=evidence_radius,
    )
    jacobian_hash = directional_projection_hash(
        kind="literal_positive_removal_jacobian",
        design_hash=plan.design_hash,
        chart_identity_hash=plan.chart_identity_hash,
        context_ids=plan.context_ids,
        endpoint_action_hashes=plan.endpoint_action_hashes,
        centers=jacobian_center,
        radii=jacobian_radius,
    )
    request = plan.corrected_evidence_request
    receipt = V1DirectionalDualProjectionReceipt(
        plan_hash=plan_hash,
        design_hash=plan.design_hash,
        ladder_hash=plan.ladder_hash,
        chart_identity_hash=plan.chart_identity_hash,
        joint_fqr_family_registration_hash=(
            request.design.joint_fqr_family_registration_hash
        ),
        current_mask_hash=request.current_mask_hash,
        route_count=request.route_count,
        physical_mask_axis_hash=request.physical_mask_axis_hash,
        ordered_score_route_axis_hash=request.ordered_score_route_axis_hash,
        csr_hash=request.csr.csr_hash,
        context_ids=plan.context_ids,
        draw_ids=plan.draw_ids,
        stratum_ids=plan.stratum_ids,
        sealed_order_start_indices_by_stratum=(
            request.sealed_order_start_indices_by_stratum
        ),
        endpoint_action_hashes=plan.endpoint_action_hashes,
        corrected_evidence_stream_identity_hash=(
            request.corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=(
            request.literal_jacobian_stream_identity_hash
        ),
        same_capture_and_vjp_registration_hash=(
            request.same_capture_and_vjp_registration_hash
        ),
        corrected_request_hash=request.request_hash,
        literal_jacobian_request_hash=(
            plan.literal_jacobian_request.request_hash
        ),
        corrected_acquisition_closure_hash=evidence.closure.closure_hash,
        literal_jacobian_acquisition_closure_hash=jacobian.closure.closure_hash,
        corrected_projection_receipt_hash=evidence.result.receipt.receipt_hash,
        literal_jacobian_projection_receipt_hash=(
            jacobian.result.receipt.receipt_hash
        ),
        corrected_scorer_receipt_hash=evidence.closure.scorer_receipt_hash,
        literal_jacobian_scorer_receipt_hash=(
            jacobian.closure.scorer_receipt_hash
        ),
        scorer_same_capture_and_vjp_execution_proof_hash=scorer_proof_hash,
        typed_scorer_proof_hash=typed_proof_hash,
        corrected_projection_hash=corrected_hash,
        jacobian_projection_hash=jacobian_hash,
    )
    return V1DirectionalDualProjectionResult(
        receipt=receipt,
        corrected_projection_center_by_context_endpoint=evidence_center,
        corrected_projection_radius_by_context_endpoint=evidence_radius,
        jacobian_projection_center_by_context_endpoint=jacobian_center,
        jacobian_projection_radius_by_context_endpoint=jacobian_radius,
    )


__all__ = [
    "DIRECTIONAL_DUAL_PROJECTION_AUTHORITY_SCOPE",
    "DIRECTIONAL_DUAL_PROJECTION_PLAN_SCHEMA",
    "DIRECTIONAL_DUAL_PROJECTION_RECEIPT_SCHEMA",
    "DIRECTIONAL_DUAL_PROJECTION_RESULT_SCHEMA",
    "DIRECTIONAL_DUAL_PROJECTION_SCORER_PROOF_SCHEMA",
    "V1DirectionalDualProjectionPlan",
    "V1DirectionalDualProjectionReceipt",
    "V1DirectionalDualProjectionResult",
    "V1DirectionalDualProjectionScorerExecutionProof",
    "bind_v1_directional_dual_projection_scorer_proof",
    "close_v1_directional_dual_projection",
]
