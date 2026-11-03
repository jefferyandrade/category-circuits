"""Bind a closed dual C projection to the packed directional secant executor.

The dual-projection result is the only landed object that proves corrected
evidence and the literal relaxed-gate Jacobian were projected from one capture
and one target-logit VJP.  This bridge derives the exact sealed C slice from
that result and refuses every manual coordinate substitution before creating
the finite-dose executor.  It grants no F/Q/R decision, oracle, commit, or
terminal authority.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.measurement.modern_v1_directional_secant_provider import (
    NativeTorchV1FiberedDirectionalSecantProvider,
    NativeTorchV1DirectionalSecantProvider,
    RawLogitIntervalVerifier,
    V1DirectionalSecantObservationBounds,
    V1FiberedDirectionalSecantObservationBounds,
    fibered_directional_context_slice_axes,
    sealed_directional_context_prefix_hash,
    sealed_directional_context_slice_hash,
)

from .modern_v1_fibered_directional_record import (
    V1FiberedDirectionalProjectionClosure,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_directional_dual_projection import (
    V1DirectionalDualProjectionResult,
)
from .modern_v1_directional_remainder import V1DirectionalSecantDesign


DIRECTIONAL_SECANT_BRIDGE_AUTHORITY_SCOPE: Final = (
    "closed_same_capture_dual_C_projection_to_directional_secant_executor_only_"
    "no_fqr_oracle_commit_or_terminal_authority_v1"
)
FIBERED_DIRECTIONAL_SECANT_BRIDGE_AUTHORITY_SCOPE: Final = (
    "closed_native_fibered_C_projection_rows_to_complete_action_secant_executor_"
    "only_no_fqr_calibration_oracle_commit_or_terminal_authority_v1"
)


def _context_slice_axes(
    *,
    result: V1DirectionalDualProjectionResult,
    design: V1DirectionalSecantDesign,
) -> tuple[
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, int], ...],
]:
    receipt = result.receipt
    registered = design.certification_context_ids_by_stratum
    names = tuple(name for name, _axis in registered)
    starts = tuple(receipt.sealed_order_start_indices_by_stratum)
    if tuple(name for name, _start in starts) != names:
        raise ValueError("dual projection slice-start strata differ from the design")
    if not (
        len(receipt.context_ids)
        == len(receipt.draw_ids)
        == len(receipt.stratum_ids)
    ):
        raise ValueError("dual projection C axes are not aligned")

    rows: list[tuple[str, tuple[str, ...]]] = []
    cursor = 0
    for (name, complete_axis), (_start_name, start) in zip(
        registered,
        starts,
        strict=True,
    ):
        contexts: list[str] = []
        while cursor < len(receipt.stratum_ids) and (
            receipt.stratum_ids[cursor] == name
        ):
            contexts.append(receipt.context_ids[cursor])
            cursor += 1
        stop = start + len(contexts)
        if (
            start < 0
            or start > len(complete_axis)
            or stop > len(complete_axis)
            or tuple(complete_axis[start:stop]) != tuple(contexts)
        ):
            raise ValueError("dual projection contexts are not an exact C slice")
        rows.append((name, tuple(contexts)))
    if cursor != len(receipt.context_ids) or not any(axis for _name, axis in rows):
        raise ValueError("dual projection stratum order is not one nonempty C slice")
    return tuple(rows), starts


def build_v1_native_directional_secant_provider(
    *,
    dual_projection: V1DirectionalDualProjectionResult,
    design: V1DirectionalSecantDesign,
    evaluator: Any,
    batch_factory: Callable[[tuple[str, ...]], Any],
    parent_mask: HardMask,
    relaxed_gate_executor_source_hash: str,
    raw_logit_interval_verifier_source_hash: str,
    raw_logit_interval_verifier: RawLogitIntervalVerifier | None = None,
    observation_bounds: V1DirectionalSecantObservationBounds | None = None,
    maximum_packed_lanes: int = 8,
    maximum_gate_workspace_bytes: int = 8 * 1024**3,
    target_readout_index: int = 0,
) -> NativeTorchV1DirectionalSecantProvider:
    """Create the authoritative finite-dose executor from one closed C slice."""

    if type(dual_projection) is not V1DirectionalDualProjectionResult:
        raise TypeError("directional secant bridge requires an exact dual result")
    if type(design) is not V1DirectionalSecantDesign:
        raise TypeError("directional secant bridge requires an exact secant design")
    if type(parent_mask) is not HardMask:
        raise TypeError("directional secant bridge requires an exact parent mask")
    proof_hash = dual_projection.assert_integrity()
    design_hash = design.assert_integrity()
    receipt = dual_projection.receipt
    if (
        receipt.design_hash != design_hash
        or receipt.ladder_hash != design.ladder_hash
        or receipt.chart_identity_hash != design.chart_identity_hash
        or receipt.joint_fqr_family_registration_hash
        != design.joint_fqr_family_registration_hash
        or receipt.current_mask_hash != design.ladder.current_mask_hash
        or receipt.current_mask_hash != parent_mask.mask_hash
        or receipt.route_count != design.ladder.route_count
        or receipt.route_count != parent_mask.route_count
        or design.ladder.route_axis_hash != parent_mask.route_axis_hash
        or receipt.endpoint_action_hashes != design.endpoint_action_hashes
    ):
        raise ValueError("dual projection and directional secant coordinates differ")
    axes, starts = _context_slice_axes(result=dual_projection, design=design)
    slice_hash = (
        sealed_directional_context_prefix_hash(
            design_hash=design_hash,
            context_ids_by_stratum=axes,
            draw_ids=receipt.draw_ids,
        )
        if all(start == 0 for _name, start in starts)
        else sealed_directional_context_slice_hash(
            design_hash=design_hash,
            context_ids_by_stratum=axes,
            sealed_order_start_indices_by_stratum=starts,
            draw_ids=receipt.draw_ids,
        )
    )
    return NativeTorchV1DirectionalSecantProvider(
        evaluator=evaluator,
        design=design,
        batch_factory=batch_factory,
        parent_mask=parent_mask,
        sealed_context_prefix_ids_by_stratum=axes,
        sealed_order_start_indices_by_stratum=starts,
        draw_ids=receipt.draw_ids,
        sealed_context_prefix_hash=slice_hash,
        corrected_projection_center_by_context_endpoint=(
            dual_projection.corrected_projection_center_by_context_endpoint
        ),
        corrected_projection_radius_by_context_endpoint=(
            dual_projection.corrected_projection_radius_by_context_endpoint
        ),
        corrected_projection_hash=dual_projection.corrected_projection_hash,
        jacobian_projection_center_by_context_endpoint=(
            dual_projection.jacobian_projection_center_by_context_endpoint
        ),
        jacobian_projection_radius_by_context_endpoint=(
            dual_projection.jacobian_projection_radius_by_context_endpoint
        ),
        jacobian_projection_hash=dual_projection.jacobian_projection_hash,
        dual_chart_projection_proof_hash=proof_hash,
        relaxed_gate_executor_source_hash=relaxed_gate_executor_source_hash,
        raw_logit_interval_verifier_source_hash=(
            raw_logit_interval_verifier_source_hash
        ),
        raw_logit_interval_verifier=raw_logit_interval_verifier,
        observation_bounds=observation_bounds,
        maximum_packed_lanes=maximum_packed_lanes,
        maximum_gate_workspace_bytes=maximum_gate_workspace_bytes,
        target_readout_index=target_readout_index,
    )


def build_v1_native_fibered_directional_secant_provider(
    *,
    projection_closures: tuple[V1FiberedDirectionalProjectionClosure, ...],
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...],
    design: V1DirectionalSecantDesign,
    evaluator: Any,
    batch_factory: Callable[[tuple[str, ...]], Any],
    parent_mask: HardMask,
    relaxed_gate_executor_source_hash: str,
    raw_logit_interval_verifier_source_hash: str,
    raw_logit_interval_verifier: RawLogitIntervalVerifier | None = None,
    observation_bounds: V1FiberedDirectionalSecantObservationBounds | None = None,
    maximum_packed_lanes: int = 8,
    maximum_gate_workspace_bytes: int = 8 * 1024**3,
    target_readout_index: int = 0,
) -> NativeTorchV1FiberedDirectionalSecantProvider:
    """Create the finite-dose executor from exact native-fibered C closures."""

    if type(projection_closures) is not tuple or not projection_closures:
        raise TypeError("fibered secant bridge requires a nonempty closure tuple")
    if any(
        type(closure) is not V1FiberedDirectionalProjectionClosure
        for closure in projection_closures
    ):
        raise TypeError("fibered secant bridge requires exact projection closures")
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        raise TypeError("fibered secant bridge requires the exact fiber receipt")
    if type(sealed_order_start_indices_by_stratum) is not tuple:
        raise TypeError("fibered secant bridge requires an exact C-slice start tuple")
    if type(design) is not V1DirectionalSecantDesign:
        raise TypeError("fibered secant bridge requires the exact secant design")
    if type(parent_mask) is not HardMask:
        raise TypeError("fibered secant bridge requires the exact parent mask")
    design_hash = design.assert_integrity()
    native_fiber_axis_receipt.assert_integrity()
    axes, starts, _contexts, draws = fibered_directional_context_slice_axes(
        design=design,
        projection_closures=projection_closures,
        sealed_order_start_indices_by_stratum=(
            sealed_order_start_indices_by_stratum
        ),
    )
    first = projection_closures[0]
    frozen_coordinates = {
        "design_hash": design_hash,
        "ladder_hash": design.ladder_hash,
        "chart_identity_hash": design.chart_identity_hash,
        "parent_mask_hash": design.ladder.current_mask_hash,
        "proposal_generation": design.ladder.proposal_generation,
        "route_count": design.ladder.route_count,
        "endpoint_action_hashes": design.endpoint_action_hashes,
        "certification_stream_commitment_hash": (
            design.certification_stream_commitment_hash
        ),
        "joint_fqr_family_registration_hash": (
            design.joint_fqr_family_registration_hash
        ),
        "raw_logit_stream_identity_hash": (
            design.ladder.raw_logit_stream_source_hash
        ),
        "native_fiber_axis_receipt_hash": native_fiber_axis_receipt.receipt_hash,
        "native_fiber_measure_hash": native_fiber_axis_receipt.measure_hash,
    }
    family_coordinates = {
        "corrected_evidence_stream_identity_hash": (
            first.corrected_evidence_stream_identity_hash
        ),
        "literal_jacobian_stream_identity_hash": (
            first.literal_jacobian_stream_identity_hash
        ),
        "same_capture_and_vjp_execution_proof_hash": (
            first.same_capture_and_vjp_execution_proof_hash
        ),
        "fibered_work_projection_receipt_hash": (
            first.fibered_work_projection_receipt_hash
        ),
        "literal_jacobian_projection_receipt_hash": (
            first.literal_jacobian_projection_receipt_hash
        ),
    }
    for closure in projection_closures:
        closure.assert_integrity()
        if any(
            getattr(closure, name) != value
            for name, value in frozen_coordinates.items()
        ):
            raise ValueError("fibered closure and secant coordinates differ")
        if any(
            getattr(closure, name) != value
            for name, value in family_coordinates.items()
        ):
            raise ValueError("fibered closure rows do not share one VJP family")
        if (
            closure.same_capture_and_vjp_certified is not True
            or closure.global_corrected_scalar_projection_used is not False
            or closure.postselection_occurred is not False
        ):
            raise PermissionError("fibered closure exceeds measurement authority")
    if (
        first.corrected_evidence_stream_identity_hash
        == first.literal_jacobian_stream_identity_hash
    ):
        raise ValueError("corrected-evidence and literal-J streams collapsed")
    if (
        native_fiber_axis_receipt.route_axis_hash != design.ladder.route_axis_hash
        or native_fiber_axis_receipt.route_count != design.ladder.route_count
        or parent_mask.mask_hash != design.ladder.current_mask_hash
        or parent_mask.route_axis_hash != design.ladder.route_axis_hash
        or parent_mask.route_count != design.ladder.route_count
    ):
        raise ValueError("fibered receipt, parent mask, and route axis differ")
    slice_hash = (
        sealed_directional_context_prefix_hash(
            design_hash=design_hash,
            context_ids_by_stratum=axes,
            draw_ids=draws,
        )
        if all(start == 0 for _name, start in starts)
        else sealed_directional_context_slice_hash(
            design_hash=design_hash,
            context_ids_by_stratum=axes,
            sealed_order_start_indices_by_stratum=starts,
            draw_ids=draws,
        )
    )
    return NativeTorchV1FiberedDirectionalSecantProvider(
        evaluator=evaluator,
        design=design,
        batch_factory=batch_factory,
        parent_mask=parent_mask,
        projection_closures=projection_closures,
        native_fiber_axis_receipt=native_fiber_axis_receipt,
        sealed_order_start_indices_by_stratum=starts,
        sealed_context_prefix_hash=slice_hash,
        relaxed_gate_executor_source_hash=relaxed_gate_executor_source_hash,
        raw_logit_interval_verifier_source_hash=(
            raw_logit_interval_verifier_source_hash
        ),
        raw_logit_interval_verifier=raw_logit_interval_verifier,
        observation_bounds=observation_bounds,
        maximum_packed_lanes=maximum_packed_lanes,
        maximum_gate_workspace_bytes=maximum_gate_workspace_bytes,
        target_readout_index=target_readout_index,
    )


__all__ = [
    "DIRECTIONAL_SECANT_BRIDGE_AUTHORITY_SCOPE",
    "FIBERED_DIRECTIONAL_SECANT_BRIDGE_AUTHORITY_SCOPE",
    "build_v1_native_fibered_directional_secant_provider",
    "build_v1_native_directional_secant_provider",
]
