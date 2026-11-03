"""Mandatory finite actual-capture admission for native AlexNet factored D.

Structural source facts and process-local packet observations are separate.
Neither an exported record nor an opaque parity reference can admit a packet.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path
import threading
from types import MappingProxyType

from contracts.hashing import stable_digest
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.measurement.optimized_irg_providers import OptimizedScoringResponseProvider
from . import modern_v1_alexnet_native_fiber as native
from .modern_v1_alexnet_capture_execution import V1AlexNetFactoredCaptureVerifier
from .modern_v1_native_fiber_axis import (
    CAPTURE_STRUCTURAL_SCOPE, PRIVATE_STRUCTURAL_SCOPE,
    NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA,
)
from .modern_v1_sampled_look_transaction import (
    _provider_plan_identity_hash, _source_capture_identity_hash,
)

_TOKEN = object()
_PREFIX = "native-alexnet-factored-capture-admission-v1-"
_STRUCTURAL_SCHEMA = "native_alexnet_structural_capture_preflight_v1"
_ADMISSION_SCHEMA = "native_alexnet_factored_D_capture_admission_v1"


class V1AlexNetCaptureAdmissionRefusal(ValueError):
    """A source, packet, session, or resource admission did not close."""


def _require(condition, message):
    if not condition:
        raise V1AlexNetCaptureAdmissionRefusal(message)


def _closed_scope():
    return dict(global_evaluator_certification=False, future_execution_authority=False,
                population_confidence_authority=False, production_circuit_ready=False)


def _structural_source(evaluator, axis):
    native._assert_registered_source_objects()
    model, bound, blocks, graph = native._validate_structural_evaluator_and_axis(evaluator, axis)
    from mechanistic_systems.backends import torch_gate_graph
    return dict(
        schema="native_alexnet_structural_source_identity_v1",
        architecture_source=native._architecture_source_bundle_hash(type(model)),
        activation_source=native._activation_source_bundle_hash(type(model)),
        reducer_source=native._reducer_source_bundle_hash(),
        compiler_id=torch_gate_graph.TorchGateGraphCompiler.compiler_id,
        compiler_source_sha256=hashlib.sha256(Path(torch_gate_graph.__file__).read_bytes()).hexdigest(),
        architecture_id=bound.architecture_id, graph=graph,
        route_axis_content_hash=axis.content_hash,
        route_blocks=[block.to_dict() for block in blocks],
        opaque_execution_parity_reference_hash=evaluator.parity_receipt_hash,
    )


@dataclass(frozen=True)
class V1AlexNetNativeStructuralPreflight:
    coordinates: object
    native_fiber_axis: object
    _evaluator: object = field(repr=False, compare=False)
    _axis: object = field(repr=False, compare=False)
    _verifier: object = field(repr=False, compare=False)
    _issuance_token: InitVar[object] = None
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self, _issuance_token):
        _require(_issuance_token is _TOKEN, "structural preflight is issuer-only")
        object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))
        object.__setattr__(self, "_receipt_hash", self._hash())

    def _hash(self):
        return content_hash(self.to_json_payload(), prefix=_PREFIX + "structural-")

    def to_json_payload(self):
        return dict(schema=_STRUCTURAL_SCHEMA, coordinates=dict(self.coordinates),
                    native_fiber_axis_receipt_hash=self.native_fiber_axis.receipt_hash,
                    **_closed_scope())

    @property
    def receipt_hash(self):
        return self._receipt_hash

    def assert_live(self):
        _require(self._hash() == self._receipt_hash, "structural preflight mutated")
        self.native_fiber_axis.assert_integrity()
        _require(self.native_fiber_axis.schema == NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA
                 and self.native_fiber_axis.structural_source_binding_hash
                 == self.coordinates["structural_source_binding_hash"], "structural axis changed")
        native._assert_registered_source_objects()
        native._validate_structural_evaluator_and_axis(self._evaluator, self._axis)
        _require(self._axis.content_hash == self.coordinates["route_axis_content_hash"],
                 "structural route axis replaced")
        if self._verifier is not None:
            _require(type(self._verifier) is V1AlexNetFactoredCaptureVerifier
                     and self._verifier._evaluator is self._evaluator
                     and self._verifier._axis is self._axis, "structural verifier source replaced")
            _require(self._verifier.assert_live() == self.coordinates["verifier_live_binding_hash"],
                     "structural verifier changed")
        return self.receipt_hash

    def __reduce_ex__(self, _protocol):
        raise TypeError("live structural preflight is nonserializable")


def issue_v1_alexnet_native_structural_preflight(*, evaluator, route_axis, verifier):
    _require(type(verifier) is V1AlexNetFactoredCaptureVerifier
             and verifier._evaluator is evaluator and verifier._axis is route_axis,
             "exact live actual-capture verifier required for structural preflight")
    binding = verifier.assert_live()
    return _issue_preflight(evaluator, route_axis, verifier, binding, CAPTURE_STRUCTURAL_SCOPE)


def _issue_private_structural_reference_for_test(evaluator, route_axis):
    """Meta/CPU topology fixture only; this cannot create a capture admission."""
    import torch
    native._validate_structural_evaluator_and_axis(evaluator, route_axis)
    _require(torch.device(evaluator.device).type in {"cpu", "meta"},
             "private structural reference refuses CUDA")
    return _issue_preflight(evaluator, route_axis, None, None, PRIVATE_STRUCTURAL_SCOPE)


def _issue_preflight(evaluator, axis, verifier, binding, scope):
    source = _structural_source(evaluator, axis)
    source_hash = stable_digest(dict(source=source, execution_scope=scope))
    fiber = native._build_alexnet_native_axis(
        evaluator, axis, structural_capture=(scope, source_hash))
    receipt = V1AlexNetNativeStructuralPreflight(
        dict(structural_source_binding_hash=source_hash, source=source,
             route_axis_content_hash=axis.content_hash, verifier_live_binding_hash=binding,
             execution_scope=scope),
        fiber, evaluator, axis, verifier, _issuance_token=_TOKEN)
    receipt.assert_live()
    return receipt


def _provider_custody(provider):
    adapter, scoring, batch = provider.model_adapter, provider.scoring_backend, provider._batch_provider
    return (
        id(provider.evaluator), id(adapter), id(scoring), id(batch),
        id(adapter.gate_graph), id(adapter.source_model) if hasattr(adapter, "source_model") else None,
        id(adapter.batch_provider), id(scoring.model_adapter),
        id(provider.mask_resolver), id(provider.calibration_resolver),
        tuple(query.content_hash for query in provider.queries),
        id(provider.route_ids), len(provider.route_ids),
        tuple(provider._route_blocks), tuple(provider._mask_bindings),
        tuple(provider.conserving_activation_bindings),
        id(getattr(batch, "_batch_factory", None)),
        getattr(getattr(batch, "_batch_factory", None), "__code__", None),
        type(provider).accumulate_device_factored,
        type(adapter).iter_captured_batches, type(scoring).capture_factored_rows,
        type(batch).iter_batches, scoring.autograd_backend,
    )


def _accumulator_coordinates(accumulator):
    return dict(
        session_design_hash=accumulator.session.design.design_hash,
        registered_look_index=accumulator.registered_look_index,
        packet_ordinal=accumulator.packet_ordinal,
        contexts=accumulator.expected_context_ids, draws=accumulator.expected_draw_ids,
        stratum=accumulator.stratum_label, sealed_order_indices=accumulator.sealed_order_indices,
        source_measurement_hashes=accumulator.source_measurement_hashes,
        provider_plan_identity_hash=accumulator.provider_plan_identity_hash,
        source_capture_identity_hash=accumulator.source_capture_identity_hash,
        current_mask_hash=accumulator.gate.mask_hash,
    )


class V1AlexNetFactoredCaptureAdmission:
    """One installed source owner; observations are scoped to exact session draws."""

    def __init__(self, *, preflight, _issuance_token=None):
        _require(_issuance_token is _TOKEN, "capture admission is issuer-only")
        _require(type(preflight) is V1AlexNetNativeStructuralPreflight
                 and type(preflight._verifier) is V1AlexNetFactoredCaptureVerifier
                 and preflight.coordinates["execution_scope"] == CAPTURE_STRUCTURAL_SCOPE,
                 "actual verifier preflight required; structural fixture is insufficient")
        preflight.assert_live()
        self._preflight, self._verifier = preflight, preflight._verifier
        self._provider = None
        self._provider_seal = None
        self._sessions = {}
        self._contexts = {}
        self._records = {}
        self._closed = False
        self._lock = threading.Lock()

    def __reduce_ex__(self, _protocol):
        raise TypeError("capture admission owners are nonserializable")

    def __copy__(self):
        raise TypeError("capture admission owners cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("capture admission owners cannot be copied")

    @contextmanager
    def _query(self, *, poison_session=None):
        # Failure to acquire changes no D state and preserves its refusal.
        _require(self._lock.acquire(blocking=False), "capture admission query/close overlap")
        try:
            self.assert_live()
            yield
            self.assert_live()
        except BaseException:
            if poison_session is not None:
                self._release_session_locked(poison_session)
                poison_session._evict_payload()
            raise
        finally:
            self._lock.release()

    def assert_live(self):
        _require(not self._closed and self._provider is not None, "capture admission is closed or uninstalled")
        self._preflight.assert_live()
        provider = self._provider
        _require(provider._factored_capture_admission is self
                 and provider.evaluator is self._preflight._evaluator
                 and provider.model_adapter.gate_graph is provider.evaluator.gate_graph
                 and _provider_custody(provider) == self._provider_seal,
                 "installed capture provider/source changed")
        return self._preflight.receipt_hash

    def bind_session(self, session):
        from .modern_v1_factored_row_capture import V1FactoredDeviceResidentDSession
        _require(type(session) is V1FactoredDeviceResidentDSession, "exact D session required")
        with self._query():
            d, axis = session.design, self._preflight._axis
            _require(session.device == self._verifier._device
                     and d.route_count == axis.route_count
                     and d.physical_route_axis_hash == axis.route_axis_hash
                     and d.route_id_axis_hash == axis.route_id_axis_hash,
                     "capture admission/session device or physical axes differ")
            _require(d.execution_mode.value == ("cuda_production" if session.device.type == "cuda"
                                                else "cpu_reference_test_only"),
                     "capture admission execution mode differs")
            _require(id(session) not in self._sessions, "session already bound to capture owner")
            # Each retained packet has at least one context; this is a safe,
            # deliberately conservative count cap before exact packet plans exist.
            _require(d.maximum_registered_contexts <= self._verifier._maximum_observations,
                     "registered context bound exceeds live observation cap")
            token = object()
            self._sessions[id(session)] = (session, d.design_hash, token)
            return token

    def assert_session(self, session):
        self.assert_live()
        item = self._sessions.get(id(session))
        _require(item is not None and item[0] is session
                 and item[1] == session.design.design_hash
                 and item[2] is session._capture_session_token
                 and session._capture_admission is self,
                 "capture session owner/design/lease changed")
        return item

    def begin_capture(self, accumulator, *, provider, request, current_mask, current_calibration):
        with self._query():
            self.assert_session(accumulator.session)
            _require(provider is self._provider and type(current_mask) is HardMask
                     and current_mask is accumulator.gate,
                     "capture provider/mask does not own accumulator")
            _require(provider._active_mask is current_mask
                     and provider._active_request_id == request.request_id
                     and provider._active_calibration is current_calibration,
                     "capture context is not the provider's active request")
            _require(_provider_plan_identity_hash(request) == accumulator.provider_plan_identity_hash
                     and _source_capture_identity_hash(request) == accumulator.source_capture_identity_hash
                     and tuple(request.context_selector.context_ids or ()) == accumulator.expected_context_ids,
                     "provider request differs from registered capture axes")
            query = provider.queries[0]
            _require(len(provider.queries) == 1 and query.query_kind == "class_logit",
                     "capture admission requires one registered target-logit query")
            target = query.evaluator.get("class_id", query.evaluator.get("unit", query.evaluator.get("coordinate")))
            _require(type(target) is int and target == current_calibration.target_class_id,
                     "capture target/calibration differs")
            _require(id(accumulator) not in self._contexts, "capture context already activated")
            self._contexts[id(accumulator)] = (
                accumulator, _accumulator_coordinates(accumulator), current_mask, target, request)

    def require_capture(self, accumulator):
        self.assert_session(accumulator.session)
        context = self._contexts.get(id(accumulator))
        _require(context is not None and context[0] is accumulator
                 and context[1] == _accumulator_coordinates(accumulator),
                 "missing or changed registered provider capture context")
        context[4].__post_init__()
        _require(_provider_plan_identity_hash(context[4]) == context[1]["provider_plan_identity_hash"]
                 and _source_capture_identity_hash(context[4]) == context[1]["source_capture_identity_hash"],
                 "registered request changed during capture")
        return context

    def observe(self, accumulator, packet, **actual):
        with self._query():
            context = self.require_capture(accumulator)
            _require(id(packet) not in self._records, "packet capture already admitted")
            observation = self._verifier.observe(
                packet=packet, current_mask=context[2], target_class_id=context[3], **actual)
            self.require_capture(accumulator)
            observation.assert_live()
            self._records[id(packet)] = (packet, accumulator.session, context[1], context[2], observation, False)
            return observation

    def _assert_packet(self, session, packet, gate, *, committed=False):
        self.assert_session(session)
        row = self._records.get(id(packet))
        _require(row is not None and row[0] is packet and row[1] is session
                 and gate is row[3] and type(gate) is HardMask,
                 "packet lacks exact registered capture admission")
        _require(gate.mask_hash == session.design.current_mask_hash
                 and row[2]["session_design_hash"] == session.design.design_hash,
                 "admitted mask/session changed")
        self._verifier.assert_observed(packet, observation=row[4])
        if not committed:
            active = session._active
            _require(active is not None and not row[5]
                     and active.registered_look_index == row[2]["registered_look_index"]
                     and active.packet_ordinal == row[2]["packet_ordinal"]
                     and active.context_ids == row[2]["contexts"] and active.draw_ids == row[2]["draws"]
                     and active.stratum_name == row[2]["stratum"]
                     and active.sealed_order_indices == row[2]["sealed_order_indices"]
                     and active.source_measurement_hashes == row[2]["source_measurement_hashes"]
                     and packet.receipt.parent_packet_plan_hash == row[2]["provider_plan_identity_hash"],
                     "active D draw/plan differs from its observed capture")
        return row

    @contextmanager
    def consuming(self, session, packet, gate):
        with self._query(poison_session=session):
            self._assert_packet(session, packet, gate)
            yield
            self._assert_packet(session, packet, gate)

    @contextmanager
    def committing(self, session, packet, gate):
        with self._query(poison_session=session):
            row = self._assert_packet(session, packet, gate)
            yield
            self._assert_packet(session, packet, gate, committed=True)
            self._records[id(packet)] = (*row[:5], True)

    def session_payload(self, session):
        with self._query():
            return self._session_payload_locked(session)

    def _session_payload_locked(self, session):
        self.assert_session(session)
        rows = []
        for _, packet in session._packets:
            row = self._records.get(id(packet))
            _require(row is not None and row[5], "retained packet lacks a committed observation")
            self._assert_packet(session, packet, row[3], committed=True)
            rows.append(dict(packet_receipt_hash=packet.receipt.receipt_hash,
                             observation_hash=row[4]._hash,
                             registered_capture_coordinates=row[2]))
        expected = [row for row in self._records.values() if row[1] is session and row[5]]
        _require(len(expected) == len(rows), "capture admission has omitted or extra committed packets")
        payload = dict(schema=_ADMISSION_SCHEMA, structural_preflight_hash=self._preflight.receipt_hash,
                       structural_source_binding_hash=self._preflight.coordinates["structural_source_binding_hash"],
                       verifier_live_binding_hash=self._verifier.assert_live(),
                       session_design_hash=session.design.design_hash,
                       execution_mode=session.design.execution_mode.value, device=str(session.device),
                       packets=rows, **_closed_scope())
        return payload

    def export_session_payload(self, session):
        payload = self.session_payload(session)
        return dict(payload, admission_hash=content_hash(payload, prefix=_PREFIX))

    def _release_session_locked(self, session):
        # Also release observed captures rejected before the session retained
        # them. The verifier prunes their evicted observation shells on its
        # next capture; no factor tensor survives through those shells.
        for row in tuple(self._records.values()):
            if row[1] is session:
                row[0].evict()
        self._sessions.pop(id(session), None)
        self._contexts = {key: row for key, row in self._contexts.items()
                          if row[0].session is not session}
        self._records = {key: row for key, row in self._records.items() if row[1] is not session}

    def release_session(self, session):
        _require(self._lock.acquire(blocking=False), "capture admission eviction/query overlap")
        try:
            self._release_session_locked(session)
        finally:
            self._lock.release()

    def evict_session(self, session):
        _require(self._lock.acquire(blocking=False), "capture admission eviction/query overlap")
        try:
            self._release_session_locked(session)
            session._evict_payload()
        finally:
            self._lock.release()

    def close(self):
        _require(self._lock.acquire(blocking=False), "capture admission close/query overlap")
        try:
            _require(not self._sessions, "capture admission has live session leases")
            self._verifier.close()
            self._contexts.clear()
            self._records.clear()
            self._closed = True
        finally:
            self._lock.release()

    def workspace_bound(self, *, maximum_packet_contexts, factor_numbers_per_context):
        self.assert_live()
        _require(type(maximum_packet_contexts) is int and maximum_packet_contexts > 0
                 and type(factor_numbers_per_context) is int and factor_numbers_per_context > 0,
                 "exact registered capture workspace dimensions required")
        p, largest = self._verifier._model_bytes, self._verifier._largest_parameter_bytes
        factors = 4 * maximum_packet_contexts * factor_numbers_per_context
        logits = maximum_packet_contexts * 1000
        required = p + 3 * largest + 6 * factors + 16 * logits
        _require(required <= self._verifier._budget, "registered capture workspace exceeds verifier cap")
        return dict(schema="native_alexnet_capture_workspace_bound_v1",
                    model_bytes=p, largest_parameter_bytes=largest, maximum_packet_factor_bytes=factors,
                    maximum_response_elements=logits, additional_required_bytes=required,
                    resident_reference_also_in_frozen_reserved=True)


def install_v1_alexnet_factored_capture_admission(*, provider, preflight):
    _require(type(provider) is OptimizedScoringResponseProvider, "exact optimized provider required")
    _require(provider._factored_capture_admission is None, "provider capture admission already installed")
    owner = V1AlexNetFactoredCaptureAdmission(preflight=preflight, _issuance_token=_TOKEN)
    _require(provider.evaluator is preflight._evaluator
             and provider.model_adapter.gate_graph is provider.evaluator.gate_graph,
             "provider does not execute preflight graph")
    owner._provider = provider
    owner._provider_seal = _provider_custody(provider)
    provider._factored_capture_admission = owner
    try:
        owner.assert_live()
    except BaseException:
        provider._factored_capture_admission = None
        raise
    return owner


def assert_backend_capture_admission(owner):
    """Backend and cloned backend must retain the same exact installed scorer."""
    from .modern_v1_alexnet_backend import _AlexNetStagedProposalBackend
    _require(type(owner) is _AlexNetStagedProposalBackend, "exact registered backend required")
    admission = getattr(owner, "_factored_capture_admission", None)
    _require(type(admission) is V1AlexNetFactoredCaptureAdmission
             and admission._provider is owner.scorer
             and owner.scorer._factored_capture_admission is admission,
             "backend lacks its exact installed capture admission")
    admission.assert_live()
    _require(owner.native_fiber_axis_receipt is admission._preflight.native_fiber_axis
             and owner.route_axis is admission._preflight._axis
             and owner.fq_model is admission._preflight._evaluator.source_model
             and tuple(owner.scorer.queries) == (owner.fq_query,),
             "backend/preflight model/query/native axes differ")
    return admission


def assert_factored_source_capture_admission(owner, session):
    admission = assert_backend_capture_admission(owner)
    _require(session._capture_admission is admission, "source session belongs to another capture owner")
    return admission.export_session_payload(session)
