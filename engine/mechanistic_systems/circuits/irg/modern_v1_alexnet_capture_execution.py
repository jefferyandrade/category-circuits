"""Finite native/compiled AlexNet capture observations, never global certification.

The verifier owns a plain torchvision reference. It observes actual packet
factors against that reference and retains process-local packet custody. An
exported observation cannot authorize a later capture or reopen its tensors.
Provider and downstream source admission must separately require this owner.
"""
from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

import copy
from dataclasses import InitVar, dataclass, field
import hashlib
import os
from pathlib import Path
import threading
import time
from types import MappingProxyType

import torch

from mechanistic_systems.backends.torch_gate_graph import GatedConv2d, GatedLinear
from mechanistic_systems.circuits.torch_evaluator import TorchCircuitEvaluator
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.numerics import exact_batched_vjp
from mechanistic_systems.measurement.queries import QueryCotangentBatch
from .modern_v1_neutral_runtime import DeclaredRouteAxis, resolve_declared_route_axis
from .modern_v1_factored_row_capture import V1FactoredRowPacket
from . import modern_v1_alexnet_native_fiber as native

_TOKEN = object()
_PREFIX = 'native-alexnet-actual-capture-observation-v1-'
_PATHS = ('features.0', 'features.3', 'features.6', 'features.8', 'features.10',
          'classifier.1', 'classifier.4')
_ACTIVATIONS = ('features.1', 'features.4', 'features.7', 'features.9', 'features.11',
                'classifier.2', 'classifier.5')


class V1AlexNetCaptureExecutionRefusal(ValueError):
    """An exact source, numerical comparison or custody requirement failed."""


def _require(condition, message):
    if not condition:
        raise V1AlexNetCaptureExecutionRefusal(message)


def _custody(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape),
            tuple(value.stride()), value.storage_offset(), str(value.dtype),
            str(value.device), value.requires_grad)


def _model_state(model, *, gate_values_mutable=False):
    modules = tuple((name, id(module), type(module), id(type(module).forward),
                     module.extra_repr(), module.training,
                     tuple(module._forward_hooks.items()), tuple(module._forward_pre_hooks.items()),
                     id(vars(module).get('forward'))) for name, module in model.named_modules())
    tensors = tuple((name, _custody(value)) for name, value in model.named_parameters())
    buffers = tuple((name, (_custody(value)[:2] + _custody(value)[3:])
                     if gate_values_mutable and name.endswith('.route_gate') else _custody(value))
                    for name, value in model.named_buffers())
    return modules, tensors, buffers


def _same_bytes(left, right, *, name):
    _require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
             and left.dtype == right.dtype == torch.float32 and left.device == right.device
             and tuple(left.shape) == tuple(right.shape), name + ': tensor axes/dtype/device differ')
    _require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), name + ': nonfinite tensor')
    _require(torch.equal(left.detach().contiguous().view(torch.int32),
                         right.detach().contiguous().view(torch.int32)), name + ': native/compiled bytes differ')


def _numerical_settings(device):
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                autocast=torch.is_autocast_enabled(device.type),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision())


@dataclass(frozen=True)
class V1AlexNetCaptureObservation:
    coordinates: object
    _verifier: object = field(repr=False, compare=False)
    _packet: object = field(repr=False, compare=False)
    _issuance_token: InitVar[object] = None
    _hash: str = field(init=False, repr=False)

    def __post_init__(self, _issuance_token):
        _require(_issuance_token is _TOKEN, 'capture observation is issuer-only')
        object.__setattr__(self, 'coordinates', MappingProxyType(copy.deepcopy(dict(self.coordinates))))
        object.__setattr__(self, '_hash', content_hash(self.to_json_payload(), prefix=_PREFIX))

    def to_json_payload(self):
        return dict(schema='native_alexnet_actual_capture_observation_v1',
                    coordinates=copy.deepcopy(dict(self.coordinates)),
                    global_evaluator_certification=False, future_execution_authority=False,
                    population_confidence_authority=False, production_circuit_ready=False)

    def assert_live(self):
        _require(content_hash(self.to_json_payload(), prefix=_PREFIX) == self._hash, 'observation changed')
        return self._verifier.assert_observed(self._packet, observation=self)


class V1AlexNetFactoredCaptureVerifier:
    """Own a plain native replay and exact finite observations of actual packets."""

    def __init__(self, *, evaluator, route_axis, additional_memory_budget_bytes=8 << 30,
                 allow_private_cpu_reference=False, maximum_observations=256):
        _require(type(evaluator) is TorchCircuitEvaluator and type(route_axis) is DeclaredRouteAxis,
                 'exact evaluator and declared route axis required')
        _require(type(allow_private_cpu_reference) is bool, 'exact CPU reference policy required')
        _require(type(additional_memory_budget_bytes) is int and additional_memory_budget_bytes > 0,
                 'positive exact additional memory budget required')
        _require(type(maximum_observations) is int and maximum_observations > 0,
                 'positive exact observation cap required')
        self._evaluator, self._axis = evaluator, route_axis
        # The evaluator keeps its source model on its loading device while
        # the compiler places a separate graph on the execution device.
        self._device = next(evaluator.gate_graph.model.parameters()).device
        declared_device = torch.device(evaluator.device)
        _require(self._device.type == declared_device.type and
                 (declared_device.index is None or self._device.index == declared_device.index),
                 'compiled parameter device differs from evaluator declaration')
        _require(self._device.type == 'cuda' or (self._device.type == 'cpu' and allow_private_cpu_reference),
                 'capture verifier requires CUDA or explicit private CPU reference')
        native._assert_registered_source_objects()
        native._live_alexnet_graph_payload(evaluator.source_model)
        _require(resolve_declared_route_axis(evaluator).content_hash == route_axis.content_hash,
                 'capture verifier route axis differs from evaluator')
        _require(tuple(b.module_path for b in evaluator.mask_bindings) == _PATHS,
                 'capture verifier requires the seven native AlexNet affines')
        _require(all(not m.training for m in evaluator.source_model.modules())
                 and all(not m.training for m in evaluator.gate_graph.model.modules()), 'eval mode required')
        self._source = evaluator.source_model
        self._graph = evaluator.gate_graph
        self._blocks = tuple(evaluator.route_blocks)
        self._bindings = tuple(evaluator.mask_bindings)
        self._budget = additional_memory_budget_bytes
        self._maximum_observations = maximum_observations
        self._lock = threading.Lock()
        self._observations = {}
        self._closed = False
        self._reference = None
        self._source_state = _model_state(self._source)
        self._compiled_state = _model_state(self._graph.model, gate_values_mutable=True)
        for model in (self._source, self._graph.model):
            _require(all(not m._forward_hooks and not m._forward_pre_hooks and 'forward' not in vars(m)
                         for m in model.modules()), 'unregistered model hooks/forward overrides refused')
        self._model_bytes = sum(p.numel() * p.element_size() for p in self._source.parameters())
        self._largest_parameter_bytes = max(p.numel() * p.element_size() for p in self._source.parameters())
        self._cold_admission = self._admit(self._model_bytes + 3 * self._largest_parameter_bytes)
        started = time.perf_counter()
        try:
            self._reference = copy.deepcopy(self._source).to(self._device).eval().requires_grad_(False)
            for name, parameter in self._source.named_parameters():
                module_path, local_name = name.rsplit('.', 1)
                compiled = self._graph.model.get_submodule(module_path)
                if module_path in _PATHS:
                    expected = GatedConv2d if module_path.startswith('features.') else GatedLinear
                    _require(type(compiled) is expected, 'compiled affine wrapper differs')
                    compiled = compiled.module
                # Source-to-execution transfer is cold construction only. The
                # immutable compiled originals stay resident for later masks.
                source_value = parameter.detach().to(self._device)
                _same_bytes(source_value, getattr(compiled, local_name), name='compiled parameter ' + name)
                _same_bytes(source_value, self._reference.get_parameter(name), name='reference parameter ' + name)
                del source_value
            self._reference_state = _model_state(self._reference)
            self._binding_hash = content_hash(dict(
                schema='native_alexnet_capture_verifier_live_binding_v1',
                evaluator_process_identity=id(evaluator), graph_process_identity=id(self._graph),
                source_parameter_custody=[(name, _custody(p)) for name, p in self._source.named_parameters()],
                route_axis_content_hash=route_axis.content_hash,
                verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                device=str(self._device), torch_version=torch.__version__,
                private_cpu_reference=self._device.type == 'cpu'), prefix=_PREFIX + 'binding-')
            self.assert_live()
        except BaseException:
            self._reference = None
            self._closed = True
            raise
        self._construction_seconds = time.perf_counter() - started

    def _admit(self, required):
        _require(required <= self._budget, 'capture verifier additional memory budget exceeded')
        result = dict(required_bytes=required, budget_bytes=self._budget, device=str(self._device))
        if self._device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(self._device)
            reserved = torch.cuda.memory_reserved(self._device)
            _require(required <= cuda_reusable_memory_bytes(self._device, framework=torch, free_bytes=free)
                     and torch.cuda.memory_allocated(self._device) + required <= int(.85 * total),
                     'fresh capture verifier CUDA memory admission failed')
            result.update(free_bytes=free, total_bytes=total, reserved_bytes=reserved, hbm_ceiling=.85)
        return result

    def assert_live(self):
        _require(not self._closed and self._reference is not None, 'capture verifier is closed')
        _require(self._evaluator.source_model is self._source and self._evaluator.gate_graph is self._graph,
                 'capture verifier source owner replaced')
        _require(tuple(self._evaluator.route_blocks) == self._blocks
                 and tuple(self._evaluator.mask_bindings) == self._bindings, 'capture verifier source axis changed')
        _require(_model_state(self._source) == self._source_state, 'native source model changed')
        _require(_model_state(self._graph.model, gate_values_mutable=True) == self._compiled_state,
                 'compiled source model changed')
        _require(_model_state(self._reference) == self._reference_state, 'owned native reference changed')
        return self._binding_hash

    def _replay(self, inputs, query_batch, gates, *, backend):
        captures, post, handles = {}, {}, []
        # Plain modules receive effective masked parameters. Packet coefficients
        # still use the original verified parameters, checked separately below.
        with torch.no_grad():
            for path, gate in zip(_PATHS, gates, strict=True):
                original = self._graph.model.get_submodule(path).module.weight
                self._reference.get_submodule(path).weight.copy_(original * gate)
        self._reference_state = _model_state(self._reference)
        def affine_hook(path):
            def hook(_module, args, output):
                _require(path not in captures, 'native affine executed more than once')
                captures[path] = (args[0].detach(), output)
                return output.clone()  # Preserve preactivation before in-place ReLU.
            return hook
        def activation_hook(path):
            def hook(_module, _args, output):
                _require(path not in post, 'native activation executed more than once')
                post[path] = output
            return hook
        try:
            for path, activation in zip(_PATHS, _ACTIVATIONS, strict=True):
                handles.append(self._reference.get_submodule(path).register_forward_hook(affine_hook(path)))
                handles.append(self._reference.get_submodule(activation).register_forward_hook(activation_hook(path)))
            response = self._reference(inputs.detach().clone().requires_grad_(True))
        finally:
            for handle in handles:
                handle.remove()
        _require(tuple(captures) == tuple(post) == _PATHS, 'native capture order differs')
        gradients, telemetry = exact_batched_vjp(response, list(post.values()), query_batch,
                                                backend=backend, retain_graph=False)
        return response, captures, post, gradients, telemetry

    def observe(self, *, packet, response_matrix, query_batch, captured_blocks,
                activation_captures, current_mask, target_class_id, backend='auto'):
        _require(self._lock.acquire(blocking=False), 'capture verifier query/close overlap')
        started = time.perf_counter()
        try:
            binding_hash = self.assert_live()
            _require(type(packet) is V1FactoredRowPacket and type(current_mask) is HardMask,
                     'exact packet and hard mask required')
            packet.assert_integrity()
            _require(packet.receipt.current_mask_hash == current_mask.mask_hash
                     and current_mask.route_axis_hash == self._axis.route_axis_hash
                     and packet.receipt.physical_route_axis_hash == self._axis.route_axis_hash
                     and packet.receipt.route_id_axis_hash == self._axis.route_id_axis_hash,
                     'capture packet/mask/physical axes differ')
            _require(id(packet) not in self._observations, 'packet already observed')
            self._observations = {key: value for key, value in self._observations.items()
                                  if not value._packet._evicted}
            _require(len(self._observations) < self._maximum_observations, 'live capture observation cap exceeded')
            _require(type(query_batch) is QueryCotangentBatch and len(query_batch.query_ids) == 1
                     and query_batch.statuses == ('class_logit_exact',), 'one exact target-logit query required')
            _require(packet.receipt.query_id == query_batch.query_ids[0], 'packet query identity differs')
            _require(type(target_class_id) is int and 0 <= target_class_id < response_matrix.shape[1],
                     'exact target class required')
            _require(response_matrix.dtype == torch.float32 and response_matrix.device == self._device,
                     'actual response must be FP32 on the bound device')
            settings = _numerical_settings(self._device)
            _require(not settings['autocast'], 'autocast capture refused')
            _require(settings['cudnn_deterministic'] and not settings['cudnn_benchmark']
                     and not settings['cudnn_allow_tf32'] and not settings['matmul_allow_tf32']
                     and settings['float32_matmul_precision'] == 'highest',
                     'registered exact FP32 capture policy required')
            if self._device.type == 'cuda':
                # Landed D capture deliberately keeps this global error mode
                # off: adaptive_avg_pool2d_backward_cuda has no deterministic
                # implementation. Observe finite bytes; claim no repeatability.
                _require(not settings['deterministic'],
                         'registered CUDA D capture requires global deterministic error mode off')
                _require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8', ':16:8'),
                         'registered deterministic cuBLAS workspace required')
            admission = self._admit(self._model_bytes + 3 * self._largest_parameter_bytes
                                    + 6 * packet.factor_bytes + 4 * response_matrix.numel() * 4)
            expected = torch.zeros_like(response_matrix).unsqueeze(0)
            expected[0, :, target_class_id] = 1.
            _same_bytes(query_batch.cotangents, expected, name='target-logit cotangent')
            _same_bytes(query_batch.values, response_matrix[:, target_class_id].reshape(1, -1), name='query values')
            blocks, activations = tuple(captured_blocks), tuple(activation_captures)
            _require(len(blocks) == len(activations) == len(packet.layers) == 7, 'seven actual captured affines required')
            gates = tuple(self._graph.gates_by_layer[b.layer_id].route_gate for b in self._blocks)
            gate_custody = tuple(_custody(gate) for gate in gates)
            mask_bits = current_mask.to_bool_array()
            for block, gate in zip(self._blocks, gates, strict=True):
                expected_gate = torch.as_tensor(mask_bits[block.offset:block.offset + block.route_count].copy(),
                                                device=self._device).reshape(block.shape).to(torch.float32)
                _same_bytes(gate, expected_gate, name='actual binary gate ' + block.layer_id)
            del expected_gate, mask_bits, expected
            live_inputs = tuple(_custody(value) for value in (response_matrix, query_batch.cotangents,
                                query_batch.values, *(b.output for b in blocks),
                                *(a.post_activation_output for a in activations)))
            native_response, native_captures, native_post, gradients, vjp_telemetry = self._replay(
                blocks[0].input_source, query_batch, gates, backend=backend)
            _require(packet.receipt.vjp_backend_used == vjp_telemetry.backend_used, 'actual/reference VJP backend differs')
            _same_bytes(response_matrix, native_response, name='actual logits')
            for i, (path, layer, block, activation) in enumerate(zip(_PATHS, packet.layers, blocks, activations, strict=True)):
                _require(layer.route_block == block.route_block == self._blocks[i]
                         and layer.layer_id == activation.layer_id == self._blocks[i].layer_id,
                         'actual capture layer/route axis differs')
                module = self._source.get_submodule(path)
                parameters = dict(stride=module.stride, padding=module.padding,
                                  dilation=module.dilation, groups=module.groups) if path.startswith('features.') else {}
                _require(layer.scale == block.scale == 1. and layer.activation_kind == activation.activation_kind == 'relu'
                         and layer.reducer_kind == block.reducer_kind == self._blocks[i].route_kind
                         and layer.abs_aggregation_mode == block.abs_aggregation_mode == 'exact_abs'
                         and dict(layer.reducer_parameters) == dict(block.reducer_parameters) == parameters,
                         'actual capture factor law differs from native affine')
                native_input, native_pre = native_captures[path]
                _same_bytes(block.input_source, native_input, name=path + ' live input')
                _same_bytes(layer.input, native_input, name=path + ' packet input')
                _same_bytes(block.output, native_pre, name=path + ' live preactivation')
                _same_bytes(layer.preactivation, native_pre, name=path + ' packet preactivation')
                _same_bytes(activation.post_activation_output, native_post[path], name=path + ' postactivation')
                _same_bytes(layer.post_activation_cotangent, gradients[i][0], name=path + ' packet VJP')
                original = self._graph.model.get_submodule(path).module
                _same_bytes(layer.weight, original.weight, name=path + ' packet original weight')
                _same_bytes(block.weight, layer.weight, name=path + ' live original weight')
                bias = original.bias
                reference_bias = bias if layer.bias_baseline.ndim == 1 else bias.reshape((1, -1) + (1,) * (native_pre.ndim - 2))
                _same_bytes(layer.bias_baseline, reference_bias, name=path + ' packet bias')
            _require(gate_custody == tuple(_custody(gate) for gate in gates), 'gate changed during capture verification')
            _require(live_inputs == tuple(_custody(value) for value in (response_matrix, query_batch.cotangents,
                     query_batch.values, *(b.output for b in blocks), *(a.post_activation_output for a in activations))),
                     'live capture changed during verification')
            _require(self.assert_live() == binding_hash, 'source changed during verification')
            _require(_numerical_settings(self._device) == settings, 'numerical settings changed during verification')
            packet.assert_integrity()
            observation = V1AlexNetCaptureObservation(dict(
                verifier_binding_hash=binding_hash, packet_receipt_hash=packet.receipt.receipt_hash,
                current_mask_hash=current_mask.mask_hash, context_ids=packet.receipt.context_ids,
                target_class_id=target_class_id, input_shape=tuple(blocks[0].input_source.shape),
                device=str(self._device), private_cpu_reference=self._device.type == 'cpu',
                actual_logits_and_factors_bitwise_equal=True, reference_uses_plain_native_model=True,
                global_execution_repeatability_claimed=False,
                reference_numerical_settings=settings,
                vjp_backend_requested=backend, vjp_backend_used=vjp_telemetry.backend_used,
                construction_seconds=self._construction_seconds,
                observation_seconds=time.perf_counter() - started, memory_admission=admission),
                self, packet, _issuance_token=_TOKEN)
            self._observations[id(packet)] = observation
            return observation
        finally:
            self._lock.release()

    def assert_observed(self, packet, *, observation=None):
        _require(self._lock.acquire(blocking=False), 'capture verifier query/close overlap')
        try:
            self.assert_live()
            current = self._observations.get(id(packet))
            _require(current is not None and current._packet is packet
                     and (observation is None or observation is current), 'packet lacks exact live capture observation')
            packet.assert_integrity()
            _require(current.coordinates['packet_receipt_hash'] == packet.receipt.receipt_hash,
                     'observed packet identity changed')
            _require(content_hash(current.to_json_payload(), prefix=_PREFIX) == current._hash, 'observation changed')
            return current._hash
        finally:
            self._lock.release()

    def close(self):
        _require(self._lock.acquire(blocking=False), 'capture verifier query/close overlap')
        try:
            self._closed = True
            self._observations.clear()
            self._reference = None
        finally:
            self._lock.release()
