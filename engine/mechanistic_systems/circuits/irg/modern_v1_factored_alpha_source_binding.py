"""Live initial-D factored source / registered A-F-J identity join.

This receipt neither bounds an unobserved population nor authorizes alpha spend
or a pruning decision. It retains the native backend and live factors; an
exported hash cannot reopen them. Initial D uses prefixes starting at zero.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Mapping

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_factored_row_capture import V1FactoredDeviceResidentDLiveHandoff
from .modern_v1_response_exactification import V1OrdinalPhysicalAxisBridgeReceipt
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
)
from .modern_v1_sampled_v7_d_alpha_family_registry import (
    V1SampledV7DAlphaCellRole, V1SampledV7DAlphaFamilyPlan,
    V1SampledV7DAlphaFamilyRegistryReceipt,
)
from .modern_v1_sampled_v7_d_alpha_head_cas import V1SampledV7DAlphaHeadCASReceipt
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    _cell_for, _expected_non_a_scope_hash,
)

_SCHEMA = "partition_free_v1_factored_initial_D_alpha_source_binding_v2"
_OBSERVED_SCHEMA = "partition_free_v1_factored_initial_D_alpha_source_binding_v3"
_PREFIX = "partition-free-v1-factored-initial-D-alpha-source-binding-"
_ISSUER = object()


class V1FactoredDAlphaSourceBindingRefusal(ValueError):
    """A required native source identity did not close."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V1FactoredDAlphaSourceBindingRefusal(message)


def _joined(owner, request, handoff, bridge, current_calibration) -> dict[str, object]:
    # Import here to avoid a backend/module import cycle. Exact native owners
    # prevent a row-backed owner or receipt-shaped object from impersonating D.
    from .modern_v1_alexnet_backend import (
        _AlexNetStagedProposalBackend, _V1SampledScoreOrderOwnerCacheEntry,
        _campaign_policy_calibration_view, _raw_target_logit_stream_identity,
        _raw_target_logit_literal_jacobian_stream_identity,
        FixedAffineBCECalibration, V1RuntimeProposalAcquisitionRequest,
        _SAMPLED_D_EXECUTION_FACTORED_DEVICE_RESIDENT_V8,
    )
    _require(type(owner) is _AlexNetStagedProposalBackend, "exact native backend required")
    _require(owner.sampled_d_execution_mode == _SAMPLED_D_EXECUTION_FACTORED_DEVICE_RESIDENT_V8,
             "backend no longer owns factored D")
    _require(type(request) is V1RuntimeProposalAcquisitionRequest, "exact runtime request required")
    _require(type(handoff) is V1FactoredDeviceResidentDLiveHandoff, "exact live factored handoff required")
    _require(bridge is None or type(bridge) is V1OrdinalPhysicalAxisBridgeReceipt,
             "exact ordinal/physical bridge required")
    _require(type(current_calibration) is FixedAffineBCECalibration, "exact current calibration required")
    current_calibration.__post_init__()
    boundary_hash = handoff.assert_live()
    # Population source identity precedes response action selection. Reopen the
    # executable axis declaration itself; no endpoint hashes are invented when
    # initial D has not produced a response ladder yet.
    from .modern_v1_neutral_runtime import DeclaredRouteAxis
    from mechanistic_systems.core.axis_identity import ordered_string_axis_hash

    axis = owner.route_axis
    _require(type(axis) is DeclaredRouteAxis, "exact declared physical source axis required")
    axis.__post_init__()
    ordered_hash = ordered_string_axis_hash(owner.route_ids, name="factored source route IDs")
    _require(ordered_hash == axis.route_id_axis_hash == owner.ordered_route_axis_hash
             and axis.route_axis_hash == owner.physical_route_axis_hash,
             "native declared source axes differ from backend")
    bridge_hash = None if bridge is None else bridge.assert_integrity()
    boundary, session = handoff.replay_boundary, handoff.session
    design = session.design
    _require(boundary.registered_runtime_request_hashes[-1] == request.request_hash,
             "runtime request does not close the live handoff")
    _require(owner._sampled_factored_device_resident_d_live_handoff_by_chart.get(
        (request.current_mask_hash, request.acquisition_generation_index)) is handoff,
        "backend no longer owns this live handoff")
    registration = request.payload.get("proposal_order_inference_registration")
    _require(isinstance(registration, Mapping), "runtime request lacks score registration")
    registration_hash = registration.get("registration_hash")
    _require(type(registration_hash) is str and bool(registration_hash), "score registration identity absent")
    cached = owner._sampled_score_order_cache.get(registration_hash)
    _require(type(cached) is _V1SampledScoreOrderOwnerCacheEntry, "live score registry cache absent")
    plan, registry, cas = cached.alpha_family_plan, cached.alpha_family_registry, cached.alpha_head_cas
    frame, stream = owner._active_generation_sampling_frame, owner._active_generation_discovery_stream
    for parent, expected in ((plan, V1SampledV7DAlphaFamilyPlan),
                             (registry, V1SampledV7DAlphaFamilyRegistryReceipt),
                             (cas, V1SampledV7DAlphaHeadCASReceipt),
                             (frame, V1AlexNetFQCausalGenerationFrameReceipt),
                             (stream, V1AlexNetFQCausalRoleSamplingStreamReceipt)):
        _require(type(parent) is expected, "exact native alpha/frame source absent")
        parent.assert_integrity()
    _require(registry.alpha_family_plan_hash == plan.plan_hash == cas.alpha_family_plan_hash
             and registry.master_registration_hash == plan.master_registration_hash == cas.master_registration_hash
             and cas.alpha_family_registry_receipt_hash == registry.receipt_hash
             and cas.published_ledger_head_hash == registry.ledger_after_hash
             and cas.expected_predecessor_ledger_hash == registry.ledger_before_hash == plan.ledger_before_hash
             and cas.durable_controller_head_closed is True
             and registry.cell_axis_hash == plan.cell_axis_hash
             and tuple(c.cell_hash for c in registry.cell_axis) == tuple(c.cell_hash for c in plan.cell_axis),
             "alpha plan/registry/durable head chain mismatch")
    cached.__post_init__()
    _require(registry.charged_base_design_hash == cached.base_design.design_hash
             and registry.hybrid_wrapper_design_hash == cached.confidence_design.design_hash,
             "registered alpha family does not own the cached confidence design")
    for key, value in (
        ("sampled_v7_D_alpha_family_plan_hash", plan.plan_hash),
        ("sampled_v7_D_alpha_family_registry_receipt_hash", registry.receipt_hash),
        ("sampled_v7_D_alpha_head_CAS_receipt_hash", cas.receipt_hash),
        ("proposal_order_base_serfling_design_hash", cached.base_design.design_hash),
        ("proposal_order_design_hash", cached.confidence_design.design_hash),
        ("generation_sampling_frame_receipt_hash", frame.receipt_hash),
    ):
        _require(registration.get(key) == value, "runtime score registration source mismatch: " + key)
    owner.binding.__post_init__()
    contract = owner.binding.contract
    _require(request.campaign_binding_hash == owner.binding.binding_hash, "runtime request campaign mismatch")
    _require(plan.campaign_contract_hash == registry.campaign_contract_hash == cas.campaign_contract_hash
             == contract.campaign_contract_hash
             and plan.controller_run_hash == registry.controller_run_hash == cas.controller_run_hash == owner.controller_run_id
             and plan.proposal_generation == registry.proposal_generation == cas.proposal_generation
             == frame.generation_index == stream.generation_index == design.proposal_generation
             == request.acquisition_generation_index,
             "campaign/controller/generation mismatch")
    _require(plan.model_identity_hash == contract.model_seal_hash
             and plan.query_contract_hash == owner.query_contract_hash
             and plan.preprocessing_hash == owner.preprocessing_hash
             and plan.sampling_law_hash == contract.sampling_law_hash
             and plan.current_mask_hash == frame.current_mask_hash == stream.current_mask_hash
             == design.current_mask_hash == request.current_mask_hash
             and plan.raw_logit_stream_source_hash == frame.frozen_transform_hash == stream.frozen_transform_hash,
             "backend model/query/chart source mismatch")
    _require(stream.role == "discovery" and stream.generation_frame_hash == frame.receipt_hash
             and stream.registration_hash == frame.registration_hash
             and stream.causal_parent_receipt_hash == frame.receipt_hash,
             "active discovery stream is not from this frame")
    orders = (stream.target_assignment_receipt, stream.complement_assignment_receipt)
    order_hashes = tuple(p.content_hash for p in orders)
    populations = frame.full_population_stratum_counts
    _require(plan.stratum_names == design.stratum_names == ("target", "complement")
             and plan.population_stratum_counts == design.population_stratum_counts
             == stream.full_population_stratum_counts == populations
             and plan.sealed_population_order_hashes == design.sealed_population_order_hashes == order_hashes
             and frame.target_population_membership_hash == stream.target_population_membership_hash
             and frame.complement_population_membership_hash == stream.complement_population_membership_hash,
             "source population/order mismatch")
    look = len(handoff.look_commits)
    _require(0 < look <= len(plan.registered_look_stratum_counts), "look outside alpha plan")
    counts = plan.registered_look_stratum_counts[look - 1]
    _require(type(request.payload.get("registered_look_index")) is int
             and request.payload["registered_look_index"] == look
             and tuple(request.payload.get("registered_look_cumulative_stratum_counts", ())) == counts,
             "runtime request look/count mismatch")
    _require(counts == session.stratum_counts
             and tuple(r.registered_look_cumulative_stratum_counts for r in handoff.look_registrations)
             == plan.registered_look_stratum_counts[:look], "registered look/count mismatch")
    prefixes = tuple(tuple(p.generated_order[:n]) for p, n in zip(orders, counts, strict=True))
    _require(tuple(tuple(axis) for axis in session._context_ledgers) == prefixes,
             "factored initial-D contexts are not the exact zero-based discovery prefix")
    # Reconcile metadata with actual retained packets, not only mutable ledgers.
    packet_prefixes = tuple(tuple(context for stratum, packet in session.packets if stratum == index
                                  for context in packet.receipt.context_ids) for index in (0, 1))
    _require(packet_prefixes == prefixes, "retained packet context axis differs from discovery prefix")
    _require(plan.route_axis_hash == ordered_hash == design.route_id_axis_hash
             and design.physical_route_axis_hash == axis.route_axis_hash
             and ordered_hash != axis.route_axis_hash
             and axis.route_count == plan.route_count == design.route_count == len(owner.route_ids),
             "native population source axis mismatch")
    if bridge is not None:
        _require(bridge.ordered_route_axis_hash == ordered_hash
                 and bridge.physical_route_axis_hash == axis.route_axis_hash
                 and bridge.current_mask_hash == plan.current_mask_hash
                 and bridge.route_count == axis.route_count,
                 "ordinal/physical route bridge mismatch")
    _require(current_calibration.query_hash == owner.query_contract_hash
             and current_calibration.query_id == contract.query_id,
             "current calibration belongs to another query")
    calibration = _campaign_policy_calibration_view(
        current_calibration, calibration_identity=contract.calibration_identity)
    corrected = _raw_target_logit_stream_identity(calibration)
    literal = _raw_target_logit_literal_jacobian_stream_identity(
        query_id=calibration.query_id, target_class_id=calibration.target_class_id)
    current_mask = owner.mask_registry.get(plan.current_mask_hash)
    _require(current_mask is not None, "backend current mask absent")
    _require(current_mask.route_axis_hash == axis.route_axis_hash
             and current_mask.route_count == axis.route_count
             and current_mask.mask_hash == plan.current_mask_hash,
             "current mask differs from native declared source axis")
    chart = owner._factored_device_resident_d_chart_identity_hash(
        current_mask=current_mask, current_calibration=current_calibration,
        calibration=calibration, corrected_stream=corrected, literal_j_stream=literal,
        generation=plan.proposal_generation)
    _require(chart == design.chart_identity_hash
             and corrected == design.corrected_a_stream_identity_hash == plan.raw_logit_stream_source_hash
             and literal == design.literal_j_stream_identity_hash,
             "same-capture chart/stream mismatch")
    cells = tuple(_cell_for(registry.cell_axis, role=role, look_index=look) for role in (
        V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER,
        V1SampledV7DAlphaCellRole.SIGNED_F_WITNESS,
        V1SampledV7DAlphaCellRole.LITERAL_J_SAMPLING))
    for cell in cells:
        _require(cell.population_size == sum(populations) and cell.draw_count == sum(counts)
                 and cell.exact_census is (counts == populations), "alpha cell population/count mismatch")
        if cell.role is not V1SampledV7DAlphaCellRole.CORRECTED_A_ORDER:
            _require(cell.cell_scope_hash == _expected_non_a_scope_hash(
                plan=plan, frame_hash=frame.receipt_hash, role=cell.role,
                look_index=look, look_counts=counts), "F/J cell scope mismatch")
    result = dict(
        alpha_family_plan_hash=plan.plan_hash, alpha_family_registry_receipt_hash=registry.receipt_hash,
        alpha_head_cas_receipt_hash=cas.receipt_hash, published_ledger_head_hash=cas.published_ledger_head_hash,
        score_order_registration_hash=registration_hash, campaign_binding_hash=owner.binding.binding_hash,
        campaign_contract_hash=plan.campaign_contract_hash, controller_run_hash=plan.controller_run_hash,
        proposal_generation=plan.proposal_generation, model_identity_hash=plan.model_identity_hash,
        query_contract_hash=plan.query_contract_hash, preprocessing_hash=plan.preprocessing_hash,
        raw_logit_stream_source_hash=plan.raw_logit_stream_source_hash, current_mask_hash=plan.current_mask_hash,
        chart_identity_hash=chart, generation_sampling_frame_receipt_hash=frame.receipt_hash,
        discovery_stream_receipt_hash=stream.receipt_hash, sampling_law_hash=plan.sampling_law_hash,
        stratum_names=plan.stratum_names, population_stratum_counts=populations,
        population_membership_hashes=(stream.target_population_membership_hash, stream.complement_population_membership_hash),
        sealed_population_order_hashes=order_hashes, registered_look_index=look,
        registered_look_stratum_counts=counts,
        registered_prefix_axis_hash=content_hash(prefixes, prefix=_PREFIX + "prefix-"),
        ordered_route_axis_hash=plan.route_axis_hash, physical_route_axis_hash=design.physical_route_axis_hash,
        route_id_axis_hash=design.route_id_axis_hash, route_count=plan.route_count,
        ordinal_physical_axis_bridge_receipt_hash=bridge_hash,
        source_axis_kind="native_declared_physical_coordinates",
        declared_source_axis_hash=axis.content_hash,
        bridge_lineage_present=bridge is not None,
        response_endpoint_family_bound=False,
        corrected_a_stream_identity_hash=corrected, literal_j_stream_identity_hash=literal,
        session_design_hash=design.design_hash, replay_boundary_hash=boundary_hash,
        look_registration_hashes=boundary.look_registration_hashes, look_commit_hashes=boundary.look_commit_hashes,
        packet_receipt_hashes=tuple(p.receipt.receipt_hash for _, p in session.packets),
        look_receipt_hashes=tuple(r.receipt_hash for r in session.look_receipts),
        source_measurement_axis_hashes=tuple(r.source_measurement_axis_hash for r in session.look_receipts),
        finalization_receipt_hash=boundary.finalization_receipt_hash,
        kkt_geometry_receipt_hash=boundary.kkt_geometry_receipt_hash,
        exact_geometry_receipt_hash=boundary.exact_geometry_receipt_hash,
        draw_axis_hash=content_hash(tuple(tuple(a) for a in session._draw_ledgers), prefix=_PREFIX + "draws-"),
    )
    for name, cell in zip(("corrected_a", "signed_f", "literal_j"), cells, strict=True):
        for key in ("cell_id", "cell_hash", "cell_scope_hash"):
            result[name + "_" + key] = getattr(cell, key)
        result[name + "_cell_alpha"] = cell.alpha_spend
    if session._capture_admission is not None or session.device.type == "cuda":
        from .modern_v1_alexnet_capture_admission import assert_factored_source_capture_admission
        result["actual_capture_admission"] = assert_factored_source_capture_admission(owner, session)
    return result


@dataclass(frozen=True)
class V1FactoredDAlphaSourceBindingReceipt:
    """Issuer-only, process-local binding; source identity is not evidence coverage."""
    coordinates: Mapping[str, object]
    registered_a_f_j_cells_bound: bool = True
    factored_handoff_bound: bool = True
    source_population_bound: bool = True
    same_capture_streams_bound: bool = True
    alpha_spend_authority: bool = False
    physical_acquisition_authority: bool = False
    population_enclosure_authority: bool = False
    d_decision_authority: bool = False
    c_measurement_authority: bool = False
    calibration_k_authority: bool = False
    paired_oracle_o_authority: bool = False
    kkt_authority: bool = False
    action_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    production_geometry_authority: bool = False
    schema: str = _SCHEMA
    _parents: tuple = field(default=(), repr=False, compare=False)
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self, _issuance_token):
        _require(_issuance_token is _ISSUER, "factored alpha bindings are backend-issuer-only")
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, "_receipt_hash", self._hash())
        self.assert_live()

    def _hash(self):
        return content_hash(self.to_json_payload(), prefix=_PREFIX)

    def to_json_payload(self):
        return {name: (dict(value) if name == "coordinates" else value)
                for name, value in vars(self).items() if not name.startswith("_")}

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_live(self, expected_hash: str | None = None):
        _require(self._hash() == self._receipt_hash, "factored alpha receipt mutated")
        expected_schema = (_OBSERVED_SCHEMA if "actual_capture_admission" in self.coordinates else _SCHEMA)
        _require(self.schema == expected_schema and all(
            value is False for name, value in vars(self).items() if name.endswith("_authority")),
            "identity-only binding cannot grant downstream authority")
        _require(dict(self.coordinates) == _joined(*self._parents), "live factored alpha source changed")
        _require(expected_hash is None or expected_hash == self.receipt_hash, "unexpected binding identity")
        return self.receipt_hash

    def __reduce_ex__(self, _protocol):
        raise TypeError("live factored alpha bindings are nonserializable")


def _issue_factored_d_alpha_source_binding(owner, *, request, handoff, ordinal_physical_axis_bridge,
                                          current_calibration):
    parents = (owner, request, handoff, ordinal_physical_axis_bridge, current_calibration)
    coordinates = _joined(*parents)
    return V1FactoredDAlphaSourceBindingReceipt(
        coordinates=coordinates, _parents=parents, _issuance_token=_ISSUER,
        schema=(_OBSERVED_SCHEMA if "actual_capture_admission" in coordinates else _SCHEMA))
