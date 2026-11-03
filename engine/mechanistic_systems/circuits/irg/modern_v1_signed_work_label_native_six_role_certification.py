"""Receipt-only C decision authority for label-native signed-work modern v1.

This is the label-native sibling of :mod:`modern_v1_signed_work_six_role_certification`.
It deliberately reuses that module's representation-neutral decision,
certificate, and authorization records while replacing the materialized-mask
C bridge with the compact device-shell/hard-FQ crosswalk.

The two endpoint coordinates are kept distinct throughout:

* ``endpoint_index`` is the compact, Q-safe C-axis position; and
* ``endpoint_label`` is the original device-shell action label.

C summaries are therefore joined by the exact source action receipt and
source endpoint label, never by assuming that the compact index is a shell
label.  Signed native-fiber ``F`` remains objective evidence only, direct
finite displacement ``D`` owns admission, and ``Q_J + R`` labels chart
transport.  Every public authority function replays the complete pre-C source
graph and returns no mask, label handle, provider, ndarray, or route payload.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import NoReturn

from .modern_v1_fqr_sequential_acquisition import (
    _response_burden,
    _response_burden_lower,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_label_native_c_bridge import (
    V1SignedWorkLabelNativeCActionCrosswalkReceipt,
    V1SignedWorkLabelNativeCBranchReceipt,
    V1SignedWorkLabelNativeCBridgeFamilyBundle,
)
from .modern_v1_signed_work_label_native_six_role_c import (
    V1LabelNativeSixRoleCDesign,
    V1LabelNativeSixRoleEndpointSummary,
    V1LabelNativeSixRoleProgressReceipt,
)
from .modern_v1_signed_work_six_role_certification import (
    SixRoleAdmissionBasis,
    V1SignedWorkChartTransportStatus,
    V1SignedWorkIntervalStatus,
    V1SignedWorkSixRoleCertificationReceipt,
    V1SignedWorkSixRoleDecision,
    V1SignedWorkSixRoleDecisionDesign,
    V1SignedWorkSixRoleDecisionRefusal,
    V1SignedWorkSixRoleDecisionRefusalCode,
    V1SignedWorkSixRoleEndpointCertificate,
    V1SignedWorkSixRoleEndpointCoordinate,
    V1SignedWorkSixRoleFallbackQueueAuthorization,
    V1SignedWorkSixRoleProposalAuthorization,
)


def _refuse(
    code: V1SignedWorkSixRoleDecisionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkSixRoleDecisionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and positive",
        )
    return result


def _interval_status(
    *,
    lower_by_dose: tuple[float, float],
    upper_by_dose: tuple[float, float],
    tau_l: float,
) -> V1SignedWorkIntervalStatus:
    budgets = (0.5 * tau_l, tau_l)
    if all(upper <= budget for upper, budget in zip(upper_by_dose, budgets)):
        return V1SignedWorkIntervalStatus.CERTIFIED
    if any(lower > budget for lower, budget in zip(lower_by_dose, budgets)):
        return V1SignedWorkIntervalStatus.REFUTED
    return V1SignedWorkIntervalStatus.UNRESOLVED


def _endpoint_interval_status(
    *,
    lower: float,
    upper: float,
    tau_l: float,
) -> V1SignedWorkIntervalStatus:
    """Classify the committed hard endpoint, independent of path probes."""

    if upper <= tau_l:
        return V1SignedWorkIntervalStatus.CERTIFIED
    if lower > tau_l:
        return V1SignedWorkIntervalStatus.REFUTED
    return V1SignedWorkIntervalStatus.UNRESOLVED


def _bridge_branch(
    bridge: V1SignedWorkLabelNativeCBridgeFamilyBundle,
    branch: V1RelaxedKKTBranch,
) -> V1SignedWorkLabelNativeCBranchReceipt:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return bridge.spectral
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return bridge.diagonal_l1
    _refuse(
        V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
        "label-native bridge contains an unsupported KKT branch",
    )


def _validate_source_graph(
    *,
    six_role_C_design: V1LabelNativeSixRoleCDesign,
    bridge_family: V1SignedWorkLabelNativeCBridgeFamilyBundle,
) -> None:
    """Replay the complete full-shell-to-filtered-C source graph."""

    design = six_role_C_design
    bridge = bridge_family
    design.assert_integrity()
    bridge.assert_integrity()
    if (
        design.compact_C_bridge_family_bundle.bundle_hash != bridge.bundle_hash
        or design.compact_C_bridge_family_bundle.receipt.receipt_hash
        != bridge.receipt.receipt_hash
        or design.route_axis_hash != bridge.receipt.route_axis_hash
        or design.route_count != bridge.receipt.route_count
        or design.current_mask_hash != bridge.receipt.parent_mask_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "label-native C design and hard-F/Q bridge name another route axis or parent",
        )

    for branch_design in design.branch_designs:
        branch = _bridge_branch(bridge, branch_design.branch)
        source_ladder = branch_design.action_axis_registration.source_shell_ladder
        source_actions = source_ladder.actions
        identities = branch_design.action_identities
        crosswalks = branch.eligible_action_crosswalks
        if (
            branch_design.compact_C_bridge_branch_receipt.receipt_hash
            != branch.receipt_hash
            or branch_design.eligible_action_subset_registration.compact_C_bridge_branch_receipt_hash
            != branch.receipt_hash
            or source_ladder.receipt_hash != branch.shell_ladder_receipt_hash
            or source_ladder.route_axis_hash != branch.route_axis_hash
            or source_ladder.route_count != branch.route_count
            or source_ladder.parent_mask_hash != branch.parent_mask_hash
            or tuple(value.cumulative_removed_count for value in source_actions)
            != branch.complete_source_endpoint_counts
            or tuple(value.receipt_hash for value in source_actions)
            != branch.complete_source_action_receipt_hashes
            or tuple(value.packed_action_hash for value in source_actions)
            != branch.complete_source_action_hashes
            or len(identities) != len(crosswalks)
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
                "label-native C action source differs from the complete pre-C hard-F/Q audit",
            )
        for identity, crosswalk in zip(
            identities,
            crosswalks,
            strict=True,
        ):
            if (
                identity.endpoint_index != crosswalk.compact_C_endpoint_index
                or identity.source_endpoint_label
                != crosswalk.source_endpoint_label
                or identity.source_shell_ladder_receipt_hash
                != source_ladder.receipt_hash
                or identity.compact_C_bridge_branch_receipt_hash
                != branch.receipt_hash
                or identity.bridge_action_crosswalk_receipt_hash
                != crosswalk.receipt_hash
                or identity.canonical_physical_identity_hash
                != crosswalk.canonical_physical_identity_hash
                or identity.source_action_receipt.receipt_hash
                != crosswalk.source_action_receipt_hash
                or identity.packed_action_hash != crosswalk.packed_action_hash
                or identity.cumulative_removed_count
                != crosswalk.cumulative_removed_count
            ):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "label-native C identity no longer bijects to its source shell action",
                )
        for crosswalk in crosswalks:
            label = crosswalk.source_endpoint_label
            if label > len(source_actions):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "filtered C crosswalk left the complete source action axis",
                )
            action = source_actions[label - 1]
            if (
                action.endpoint_label != label
                or action.receipt_hash != crosswalk.source_action_receipt_hash
                or action.packed_action_hash != crosswalk.packed_action_hash
                or action.cumulative_removed_count
                != crosswalk.cumulative_removed_count
            ):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "filtered C crosswalk differs from its exact source action identity",
                )


def register_v1_signed_work_label_native_six_role_decision_design(
    *,
    six_role_C_design: object,
    bridge_family: object,
    tau_l: float,
    tau_l_source_hash: str,
) -> V1SignedWorkSixRoleDecisionDesign:
    """Freeze the receipt-only endpoint order before inspecting C outcomes."""

    if (
        type(six_role_C_design) is not V1LabelNativeSixRoleCDesign
        or type(bridge_family) is not V1SignedWorkLabelNativeCBridgeFamilyBundle
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "label-native decision registration requires exact C and bridge types",
        )
    design = six_role_C_design
    bridge = bridge_family
    _validate_source_graph(six_role_C_design=design, bridge_family=bridge)
    coordinates: list[V1SignedWorkSixRoleEndpointCoordinate] = []
    source_decisions: list[str] = []
    closed = 0
    crosswalk_by_receipt_hash: dict[
        str, V1SignedWorkLabelNativeCActionCrosswalkReceipt
    ] = {}
    for branch_design in design.branch_designs:
        branch = _bridge_branch(bridge, branch_design.branch)
        source_decisions.extend(value.receipt_hash for value in branch.endpoint_decisions)
        closed += sum(not value.eligible_for_C_measurement for value in branch.endpoint_decisions)
        for crosswalk in branch.eligible_action_crosswalks:
            if crosswalk.receipt_hash in crosswalk_by_receipt_hash:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "pre-C hard-F order contains a repeated crosswalk identity",
                )
            crosswalk_by_receipt_hash[crosswalk.receipt_hash] = crosswalk
    frozen_order = bridge.receipt.pre_C_hard_F_order_crosswalk_receipt_hashes
    if set(frozen_order) != set(crosswalk_by_receipt_hash) or len(frozen_order) != len(
        crosswalk_by_receipt_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "family pre-C hard-F order is not a bijection over the filtered C axis",
        )
    for priority, crosswalk_hash in enumerate(frozen_order, start=1):
        crosswalk = crosswalk_by_receipt_hash[crosswalk_hash]
        coordinates.append(
            V1SignedWorkSixRoleEndpointCoordinate(
                pre_C_priority=priority,
                branch=crosswalk.branch,
                endpoint_index=crosswalk.compact_C_endpoint_index,
                endpoint_label=crosswalk.source_endpoint_label,
                action_hash=crosswalk.packed_action_hash,
                canonical_candidate_identity_hash=(
                    crosswalk.canonical_physical_identity_hash
                ),
                cumulative_removed_count=crosswalk.cumulative_removed_count,
                crosswalk_receipt_hash=crosswalk.receipt_hash,
                hard_replay_endpoint_receipt_hash=(
                    crosswalk.hard_fq_endpoint_receipt_hash
                ),
            )
        )
    return V1SignedWorkSixRoleDecisionDesign(
        six_role_C_design_hash=design.design_hash,
        bridge_family_bundle_hash=bridge.bundle_hash,
        chart_identity_hash=design.chart_identity_hash,
        current_mask_hash=design.current_mask_hash,
        route_axis_hash=design.route_axis_hash,
        route_count=design.route_count,
        proposal_generation=design.proposal_generation,
        tau_l=_positive_finite(tau_l, name="tau_L"),
        tau_l_source_hash=_identity(tau_l_source_hash, name="tau_L source hash"),
        endpoint_coordinates=tuple(coordinates),
        complete_source_endpoint_decision_hashes=tuple(source_decisions),
        deterministic_pre_C_closed_endpoint_count=closed,
    )


def _crosswalk_by_coordinate(
    *,
    bridge_family: V1SignedWorkLabelNativeCBridgeFamilyBundle,
    coordinate: V1SignedWorkSixRoleEndpointCoordinate,
) -> V1SignedWorkLabelNativeCActionCrosswalkReceipt:
    branch = _bridge_branch(bridge_family, coordinate.branch)
    matches = tuple(
        value
        for value in branch.eligible_action_crosswalks
        if (
            value.compact_C_endpoint_index == coordinate.endpoint_index
            and value.source_endpoint_label == coordinate.endpoint_label
            and value.packed_action_hash == coordinate.action_hash
            and value.receipt_hash == coordinate.crosswalk_receipt_hash
            and value.hard_fq_endpoint_receipt_hash
            == coordinate.hard_replay_endpoint_receipt_hash
        )
    )
    if len(matches) != 1:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "decision coordinate no longer names exactly one compact bridge crosswalk",
        )
    crosswalk = matches[0]
    if (
        crosswalk.canonical_physical_identity_hash
        != coordinate.canonical_candidate_identity_hash
        or crosswalk.cumulative_removed_count
        != coordinate.cumulative_removed_count
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "decision coordinate changed its physical candidate or resource",
        )
    return crosswalk


def _summary_for_crosswalk(
    *,
    progress_receipt: V1LabelNativeSixRoleProgressReceipt,
    crosswalk: V1SignedWorkLabelNativeCActionCrosswalkReceipt,
) -> V1LabelNativeSixRoleEndpointSummary:
    branch_progress = next(
        (value for value in progress_receipt.branch_receipts if value.branch is crosswalk.branch),
        None,
    )
    if branch_progress is None:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "C progress omitted one registered KKT branch",
        )
    matches = tuple(
        value
        for value in branch_progress.endpoint_summaries
        if (
            value.branch is crosswalk.branch
            and value.action_identity.endpoint_index
            == crosswalk.compact_C_endpoint_index
            and value.action_identity.source_endpoint_label
            == crosswalk.source_endpoint_label
            and value.action_identity.source_action_receipt.receipt_hash
            == crosswalk.source_action_receipt_hash
            and value.action_identity.packed_action_hash
            == crosswalk.packed_action_hash
            and value.action_identity.bridge_action_crosswalk_receipt_hash
            == crosswalk.receipt_hash
        )
    )
    if len(matches) != 1:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "C progress does not contain exactly one summary for the source action receipt",
        )
    return matches[0]


def _certificate(
    *,
    coordinate: V1SignedWorkSixRoleEndpointCoordinate,
    crosswalk: V1SignedWorkLabelNativeCActionCrosswalkReceipt,
    summary: V1LabelNativeSixRoleEndpointSummary,
    tau_l: float,
) -> V1SignedWorkSixRoleEndpointCertificate:
    identity = summary.action_identity
    if (
        summary.branch is not coordinate.branch
        or crosswalk.branch is not coordinate.branch
        or identity.endpoint_index != coordinate.endpoint_index
        or identity.source_endpoint_label != coordinate.endpoint_label
        or identity.source_action_receipt.receipt_hash
        != crosswalk.source_action_receipt_hash
        or identity.packed_action_hash != coordinate.action_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "C endpoint summary left its exact source-action coordinate",
        )
    chart_half = (
        _response_burden_lower(
            eta=0.5,
            q_lower=summary.q_j_lower,
            r_lower=summary.r_half_lower,
        ),
        _response_burden(
            eta=0.5,
            q_upper=summary.q_j_upper,
            r_upper=summary.r_half_upper,
        ),
    )
    chart_one = (
        _response_burden_lower(
            eta=1.0,
            q_lower=summary.q_j_lower,
            r_lower=summary.r_one_lower,
        ),
        _response_burden(
            eta=1.0,
            q_upper=summary.q_j_upper,
            r_upper=summary.r_one_upper,
        ),
    )
    if (
        chart_half
        != (summary.chart_half_burden_lower, summary.chart_half_burden_upper)
        or chart_one
        != (summary.chart_one_burden_lower, summary.chart_one_burden_upper)
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "label-native Q/R chart burden does not replay its registered components",
        )
    direct = _endpoint_interval_status(
        lower=summary.d_one_lower,
        upper=summary.d_one_upper,
        tau_l=tau_l,
    )
    chart_interval = _interval_status(
        lower_by_dose=(chart_half[0], chart_one[0]),
        upper_by_dose=(chart_half[1], chart_one[1]),
        tau_l=tau_l,
    )
    chart = {
        V1SignedWorkIntervalStatus.CERTIFIED: (
            V1SignedWorkChartTransportStatus.CHART_CERTIFIED
        ),
        V1SignedWorkIntervalStatus.REFUTED: (
            V1SignedWorkChartTransportStatus.CHART_REFUTED
        ),
        V1SignedWorkIntervalStatus.UNRESOLVED: (
            V1SignedWorkChartTransportStatus.CHART_UNRESOLVED
        ),
    }[chart_interval]
    basis = None
    if direct is V1SignedWorkIntervalStatus.CERTIFIED:
        basis = (
            "chart_certified"
            if chart is V1SignedWorkChartTransportStatus.CHART_CERTIFIED
            else "direct_D_exactified"
        )
    return V1SignedWorkSixRoleEndpointCertificate(
        coordinate=coordinate,
        f_native_interval=(summary.f_native_lower, summary.f_native_upper),
        q_j_interval=(summary.q_j_lower, summary.q_j_upper),
        r_half_interval=(summary.r_half_lower, summary.r_half_upper),
        r_one_interval=(summary.r_one_lower, summary.r_one_upper),
        d_half_interval=(summary.d_half_lower, summary.d_half_upper),
        d_one_interval=(summary.d_one_lower, summary.d_one_upper),
        chart_half_interval=chart_half,
        chart_one_interval=chart_one,
        tau_l=tau_l,
        direct_D_status=direct,
        chart_transport_status=chart,
        admission_basis=basis,
    )


def certify_v1_signed_work_label_native_six_role_decision(
    *,
    six_role_C_design: object,
    bridge_family: object,
    progress_receipt: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> tuple[V1SignedWorkSixRoleCertificationReceipt, V1SignedWorkSixRoleDecision]:
    """Replay label-native C evidence and decide only a stable frozen action."""

    if (
        type(six_role_C_design) is not V1LabelNativeSixRoleCDesign
        or type(bridge_family) is not V1SignedWorkLabelNativeCBridgeFamilyBundle
        or type(progress_receipt) is not V1LabelNativeSixRoleProgressReceipt
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "label-native certification requires exact design, bridge, progress, and policy",
        )
    design = six_role_C_design
    bridge = bridge_family
    progress = progress_receipt
    policy = decision_design
    _validate_source_graph(six_role_C_design=design, bridge_family=bridge)
    progress.assert_replays(design)
    expected_policy = register_v1_signed_work_label_native_six_role_decision_design(
        six_role_C_design=design,
        bridge_family=bridge,
        tau_l=policy.tau_l,
        tau_l_source_hash=policy.tau_l_source_hash,
    )
    if policy.design_hash != expected_policy.design_hash:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "label-native decision design does not replay from its pre-C sources",
        )
    if (
        progress.design_hash != design.design_hash
        or policy.six_role_C_design_hash != design.design_hash
        or policy.bridge_family_bundle_hash != bridge.bundle_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "label-native C/bridge/decision graph hashes do not close",
        )
    live = (
        _identity(live_chart_identity_hash, name="live chart identity"),
        _identity(live_current_mask_hash, name="live current-mask hash"),
        _identity(live_route_axis_hash, name="live route-axis hash"),
        _nonnegative_int(live_proposal_generation, name="live proposal generation"),
    )
    registered = (
        policy.chart_identity_hash,
        policy.current_mask_hash,
        policy.route_axis_hash,
        policy.proposal_generation,
    )
    if live != registered:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.STALE_CHART,
            "label-native C evidence is stale for the live chart/mask/generation",
        )
    source_decisions = tuple(
        value.receipt_hash
        for branch_design in design.branch_designs
        for value in _bridge_branch(bridge, branch_design.branch).endpoint_decisions
    )
    if source_decisions != policy.complete_source_endpoint_decision_hashes:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "complete nonmonotone hard-Q decision axis changed after registration",
        )

    certificates: list[V1SignedWorkSixRoleEndpointCertificate] = []
    for coordinate in policy.endpoint_coordinates:
        crosswalk = _crosswalk_by_coordinate(
            bridge_family=bridge,
            coordinate=coordinate,
        )
        summary = _summary_for_crosswalk(
            progress_receipt=progress,
            crosswalk=crosswalk,
        )
        certificates.append(
            _certificate(
                coordinate=coordinate,
                crosswalk=crosswalk,
                summary=summary,
                tau_l=policy.tau_l,
            )
        )
    certificate_axis = tuple(certificates)
    receipt = V1SignedWorkSixRoleCertificationReceipt(
        decision_design_hash=policy.design_hash,
        six_role_C_design_hash=design.design_hash,
        six_role_C_progress_receipt_hash=progress.receipt_hash,
        bridge_family_bundle_hash=bridge.bundle_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        tau_l=policy.tau_l,
        endpoint_certificates=certificate_axis,
        sampling_status=progress.sampling_status,
    )

    certified = tuple(
        value
        for value in certificate_axis
        if value.direct_D_status is V1SignedWorkIntervalStatus.CERTIFIED
    )
    unresolved = tuple(
        value
        for value in certificate_axis
        if value.direct_D_status is V1SignedWorkIntervalStatus.UNRESOLVED
    )
    selected: V1SignedWorkSixRoleEndpointCertificate | None = None
    blockers: tuple[V1SignedWorkSixRoleEndpointCertificate, ...] = ()
    if certified:
        maximum_resource = max(
            value.coordinate.cumulative_removed_count for value in certified
        )
        selected = min(
            (
                value
                for value in certified
                if value.coordinate.cumulative_removed_count == maximum_resource
            ),
            key=lambda value: value.coordinate.pre_C_priority,
        )
        blockers = tuple(
            value
            for value in unresolved
            if (
                value.coordinate.cumulative_removed_count > maximum_resource
                or (
                    value.coordinate.cumulative_removed_count == maximum_resource
                    and value.coordinate.pre_C_priority
                    < selected.coordinate.pre_C_priority
                )
            )
        )
    elif unresolved:
        blockers = unresolved

    if blockers:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="acquire_more_C",
            selected_coordinate_hash=None,
            selected_branch=None,
            selected_endpoint_index=None,
            selected_action_hash=None,
            selected_cumulative_removed_count=None,
            admission_basis=None,
            chart_transport_status=None,
            unresolved_coordinate_hashes=tuple(
                value.coordinate.coordinate_hash for value in blockers
            ),
            single_endpoint_proposal_authority=False,
        )
    elif selected is None:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="stable_no_endpoint",
            selected_coordinate_hash=None,
            selected_branch=None,
            selected_endpoint_index=None,
            selected_action_hash=None,
            selected_cumulative_removed_count=None,
            admission_basis=None,
            chart_transport_status=None,
            unresolved_coordinate_hashes=(),
            single_endpoint_proposal_authority=False,
        )
    else:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="stable_endpoint",
            selected_coordinate_hash=selected.coordinate.coordinate_hash,
            selected_branch=selected.coordinate.branch,
            selected_endpoint_index=selected.coordinate.endpoint_index,
            selected_action_hash=selected.coordinate.action_hash,
            selected_cumulative_removed_count=(
                selected.coordinate.cumulative_removed_count
            ),
            admission_basis=selected.admission_basis,
            chart_transport_status=selected.chart_transport_status,
            unresolved_coordinate_hashes=(),
            single_endpoint_proposal_authority=True,
        )
    return receipt, decision


def authorize_v1_signed_work_label_native_six_role_proposal(
    *,
    six_role_C_design: object,
    bridge_family: object,
    progress_receipt: object,
    certification_receipt: object,
    decision: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> V1SignedWorkSixRoleProposalAuthorization:
    """Source-replay and authorize one stable receipt-only action."""

    if (
        type(certification_receipt) is not V1SignedWorkSixRoleCertificationReceipt
        or type(decision) is not V1SignedWorkSixRoleDecision
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "label-native authorization requires exact receipt, decision, and design",
        )
    replayed_receipt, replayed_decision = (
        certify_v1_signed_work_label_native_six_role_decision(
            six_role_C_design=six_role_C_design,
            bridge_family=bridge_family,
            progress_receipt=progress_receipt,
            decision_design=decision_design,
            live_chart_identity_hash=live_chart_identity_hash,
            live_current_mask_hash=live_current_mask_hash,
            live_route_axis_hash=live_route_axis_hash,
            live_proposal_generation=live_proposal_generation,
        )
    )
    if (
        certification_receipt.receipt_hash != replayed_receipt.receipt_hash
        or decision.decision_hash != replayed_decision.decision_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "supplied label-native receipt/decision differs from source replay",
        )
    if replayed_decision.status != "stable_endpoint":
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.NO_CERTIFIED_ENDPOINT,
            "only a stable direct-D endpoint can be authorized",
        )
    policy = decision_design
    coordinate = next(
        (
            value
            for value in policy.endpoint_coordinates
            if value.coordinate_hash == replayed_decision.selected_coordinate_hash
        ),
        None,
    )
    if (
        coordinate is None
        or coordinate.branch is not replayed_decision.selected_branch
        or coordinate.endpoint_index != replayed_decision.selected_endpoint_index
        or coordinate.action_hash != replayed_decision.selected_action_hash
        or coordinate.cumulative_removed_count
        != replayed_decision.selected_cumulative_removed_count
        or replayed_decision.admission_basis is None
        or replayed_decision.chart_transport_status is None
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "selected label-native proposal differs from its pre-C coordinate",
        )
    _crosswalk_by_coordinate(bridge_family=bridge_family, coordinate=coordinate)
    return V1SignedWorkSixRoleProposalAuthorization(
        certification_receipt_hash=replayed_receipt.receipt_hash,
        decision_hash=replayed_decision.decision_hash,
        decision_design_hash=policy.design_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        selected_coordinate=coordinate,
        admission_basis=replayed_decision.admission_basis,
        chart_transport_status=replayed_decision.chart_transport_status,
        old_chart_validity_claim=(
            replayed_decision.admission_basis == "chart_certified"
        ),
    )


def _duplicate_replay_signature(
    certificate: V1SignedWorkSixRoleEndpointCertificate,
) -> tuple[object, ...]:
    """Evidence that must agree before equal hard actions may be quotiented."""

    return (
        certificate.f_native_interval,
        certificate.direct_D_status,
        certificate.d_half_interval,
        certificate.d_one_interval,
        certificate.admission_basis,
        certificate.chart_transport_status,
        certificate.q_j_interval,
        certificate.r_half_interval,
        certificate.r_one_interval,
        certificate.chart_half_interval,
        certificate.chart_one_interval,
    )


def authorize_v1_signed_work_label_native_six_role_pre_O_fallback_queue(
    *,
    six_role_C_design: object,
    bridge_family: object,
    progress_receipt: object,
    certification_receipt: object,
    decision: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> V1SignedWorkSixRoleFallbackQueueAuthorization:
    """Freeze the resolved physical fallback prefix before paired O opens."""

    singleton = authorize_v1_signed_work_label_native_six_role_proposal(
        six_role_C_design=six_role_C_design,
        bridge_family=bridge_family,
        progress_receipt=progress_receipt,
        certification_receipt=certification_receipt,
        decision=decision,
        decision_design=decision_design,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_proposal_generation=live_proposal_generation,
    )
    if (
        type(certification_receipt) is not V1SignedWorkSixRoleCertificationReceipt
        or type(decision) is not V1SignedWorkSixRoleDecision
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "fallback authorization requires exact label-native certification inputs",
        )
    receipt = certification_receipt
    policy = decision_design
    certificate_by_coordinate = {
        value.coordinate.coordinate_hash: value for value in receipt.endpoint_certificates
    }
    if len(certificate_by_coordinate) != len(policy.endpoint_coordinates):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "fallback certificate axis is incomplete or duplicated",
        )
    ordered = tuple(
        sorted(
            policy.endpoint_coordinates,
            key=lambda value: (-value.cumulative_removed_count, value.pre_C_priority),
        )
    )
    duplicate_signature_by_physical: dict[str, tuple[object, ...]] = {}
    for coordinate in ordered:
        _crosswalk_by_coordinate(bridge_family=bridge_family, coordinate=coordinate)
        certificate = certificate_by_coordinate.get(coordinate.coordinate_hash)
        if certificate is None or certificate.coordinate != coordinate:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback coordinate lacks its exact replayed certificate",
            )
        physical = coordinate.canonical_candidate_identity_hash
        signature = _duplicate_replay_signature(certificate)
        existing = duplicate_signature_by_physical.setdefault(physical, signature)
        if existing != signature:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "duplicate physical candidates disagree in D admission or complete Q/R chart replay",
            )

    queued: list[V1SignedWorkSixRoleEndpointCoordinate] = []
    certificate_hashes: list[str] = []
    bases: list[SixRoleAdmissionBasis] = []
    chart_statuses: list[V1SignedWorkChartTransportStatus] = []
    physical_identities: list[str] = []
    seen_physical: set[str] = set()
    blocked_unresolved: str | None = None
    selected_hash = singleton.selected_coordinate.coordinate_hash
    selected_seen = False
    for coordinate in ordered:
        certificate = certificate_by_coordinate[coordinate.coordinate_hash]
        physical = coordinate.canonical_candidate_identity_hash
        if physical in seen_physical:
            continue
        if certificate.direct_D_status is V1SignedWorkIntervalStatus.UNRESOLVED:
            blocked_unresolved = coordinate.coordinate_hash
            break
        if certificate.direct_D_status is V1SignedWorkIntervalStatus.REFUTED:
            seen_physical.add(physical)
            continue
        if (
            certificate.direct_D_status is not V1SignedWorkIntervalStatus.CERTIFIED
            or certificate.admission_basis is None
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "fallback endpoint has an unsupported direct-D admission state",
            )
        queued.append(coordinate)
        certificate_hashes.append(certificate.certificate_hash)
        bases.append(certificate.admission_basis)
        chart_statuses.append(certificate.chart_transport_status)
        physical_identities.append(physical)
        seen_physical.add(physical)
        selected_seen = selected_seen or coordinate.coordinate_hash == selected_hash
    if not selected_seen or not queued:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "fallback resolved prefix does not contain the selected endpoint",
        )
    return V1SignedWorkSixRoleFallbackQueueAuthorization(
        certification_receipt_hash=receipt.receipt_hash,
        decision_hash=decision.decision_hash,
        decision_design_hash=policy.design_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        selected_coordinate=singleton.selected_coordinate,
        fallback_coordinates=tuple(queued),
        fallback_certificate_hashes=tuple(certificate_hashes),
        fallback_admission_bases=tuple(bases),
        fallback_chart_transport_statuses=tuple(chart_statuses),
        blocked_unresolved_coordinate_hash=blocked_unresolved,
        queued_physical_candidate_identity_hashes=tuple(physical_identities),
        fallback_exhaustion_status=(
            "blocked_on_more_C"
            if blocked_unresolved is not None
            else "fully_resolved_no_more_candidate"
        ),
    )


__all__ = [
    "authorize_v1_signed_work_label_native_six_role_pre_O_fallback_queue",
    "authorize_v1_signed_work_label_native_six_role_proposal",
    "certify_v1_signed_work_label_native_six_role_decision",
    "register_v1_signed_work_label_native_six_role_decision_design",
]
