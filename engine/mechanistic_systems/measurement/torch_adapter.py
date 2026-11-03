"""Native PyTorch model adapter for canonical route measurement.

The adapter intentionally accepts already-constructed models and typed data
providers.  Model/checkpoint and dataset construction remain architecture and
project concerns, while exact query/VJP/reduction math remains canonical.  A
claim-bearing adapter requires an external forward/replay parity receipt and
explicit certification that each capture point is not mutated in place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.contracts import MeasurementPlan, QuerySpec
from mechanistic_systems.core.numerics import CapturedRouteBlock
from mechanistic_systems.measurement.production import CapturedMeasurementBatch
from mechanistic_systems.measurement.local_regime import LocalRegimeCaptureSpec
from mechanistic_systems.measurement.queries import DirectionResolver
from mechanistic_systems.circuits.irg.modern_v1_conserving_route_score import (
    V1ConservingActivationCapture,
)
from mechanistic_systems.backends.torch_gate_graph import (
    GatedConv2d,
    GatedLinear,
    GatedResidualBlock,
    TorchGateGraphCompiler,
)
from mechanistic_systems.backends.profiling import PhaseProfiler

try:  # Optional for import smoke; required by native model execution.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


ResponseSelector = Callable[[Any], Any]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("native PyTorch measurement adapter requires PyTorch")


def _resolve_consumer_device(device: Any) -> Any | None:
    """``torch.device(device)`` or None when ``device`` names no device."""

    if torch is None:
        return None
    try:
        return torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return None


@dataclass(frozen=True)
class MeasurementInputBatch:
    context_ids: tuple[str, ...]
    inputs: Any
    labels: Any = None
    input_identity_hash: str | None = None
    ready_event: Any = None
    completion_event: Any = None
    # Optional delivery-integrity probe installed by the
    # data plane.  Called on the consumer's stream twice: from
    # ``wait_until_ready`` (stage ``"first_use"``, after the readiness wait,
    # right before the forward reads the slot) and from ``mark_consumed``
    # (stage ``"after_use"``, after every consumer read was enqueued and
    # BEFORE the completion event releases the slot).  The two digests
    # bracket the forward's window.  None (the
    # default) leaves every consumer's behaviour byte-identical.
    delivery_probe: Any = None
    # Extra device tensors the producer
    # allocated on a side stream that are NOT protected by a
    # completion-event ring (for example a fresh uint8 staging buffer the
    # delivery probe reads on the consumer stream).  ``inputs`` and
    # ``labels`` are always protected; see ``_record_consumer_stream``.
    stream_guarded_tensors: tuple[Any, ...] = ()

    def validate(self) -> None:
        _require_torch()
        if not self.context_ids or len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("measurement input context IDs must be nonempty and unique")
        if not torch.is_tensor(self.inputs):
            raise ValueError("native PyTorch measurement currently requires tensor inputs")
        if int(self.inputs.shape[0]) != len(self.context_ids):
            raise ValueError("measurement input batch axis differs from its context IDs")
        if self.labels is not None and int(torch.as_tensor(self.labels).reshape(-1).numel()) != len(
            self.context_ids
        ):
            raise ValueError("measurement labels differ from the context axis")
        if self.input_identity_hash is not None and not str(
            self.input_identity_hash
        ):
            raise ValueError("measurement input identity cannot be empty")

    def _record_consumer_stream(self, resolved: Any) -> None:
        """CLASS3: tie side-stream tensors to the consumer stream.

        The data plane allocates delivered tensors while its copy stream is
        current, so the CUDA caching allocator files their blocks under the
        COPY stream.  The consumer reads them on its own stream (the
        forward, the per-batch label clone) and then drops the batch.
        Without ``record_stream`` the allocator returns the block to the
        copy stream's pool at once, and the next copy-stream allocation of
        that size class may overwrite it before the consumer's queued reads
        execute: a delivered label or row silently becomes another chunk's
        bytes.  ``record_stream`` makes the allocator defer reuse until the
        consumer stream has passed the point of the free.  Value-neutral;
        ring-backed tensors are never freed while the plane lives,
        so recording them is harmless.  Only CUDA tensors on the consumer's
        device are recorded; everything else is a no-op.
        """

        if resolved is None or resolved.type != "cuda" or not torch.cuda.is_available():
            return
        stream = torch.cuda.current_stream(resolved)
        seen: set[int] = set()
        for tensor in (self.inputs, self.labels, *tuple(self.stream_guarded_tensors or ())):
            if not torch.is_tensor(tensor) or not tensor.is_cuda:
                continue
            if tensor.device != stream.device:
                continue
            key = int(tensor.untyped_storage().data_ptr())
            if key in seen:
                continue
            seen.add(key)
            tensor.record_stream(stream)

    def wait_until_ready(self, device: Any) -> None:
        """Order the consumer stream after asynchronous panel staging."""

        if self.ready_event is not None:
            resolved = torch.device(device)
            if resolved.type != "cuda":
                raise ValueError("CUDA panel readiness event used by a non-CUDA consumer")
            torch.cuda.current_stream(resolved).wait_event(self.ready_event)
        # CLASS3: record the consumer stream on every delivered
        # device tensor as early as possible (covers consumers that never
        # reach ``mark_consumed``; the allocator inserts its events at FREE
        # time, so every consumer read queued before the free is covered).
        self._record_consumer_stream(_resolve_consumer_device(device))
        probe = self.delivery_probe
        if probe is not None:
            # CLASS3: first-use digest on the consumer stream (no sync).
            probe(device)

    def mark_consumed(self, device: Any) -> None:
        """Release a reusable device workspace after its last consumer."""

        probe = self.delivery_probe
        if probe is not None:
            # CLASS3: after-use digest of the same slot, on the
            # consumer stream, BEFORE the completion event releases it.
            probe(device, "after_use")
        # CLASS3 (defense in depth, not gated by the integrity
        # kill switch): the consumer stream as of the LAST consumer read.
        self._record_consumer_stream(_resolve_consumer_device(device))
        if self.completion_event is None:
            return
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("CUDA workspace completion used by a non-CUDA consumer")
        self.completion_event.record(torch.cuda.current_stream(resolved))


class ContextBatchProvider(Protocol):
    provider_id: str

    def iter_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        ...


@dataclass(frozen=True)
class TorchLayerBinding:
    """One stable route block lowered to a concrete PyTorch module path."""

    layer_id: str
    module_path: str
    route_kind: str
    capture_policy: str
    circuit_role: str = "prunable"
    notes: Mapping[str, Any] | None = None

    def validate(self, *, claim_bearing: bool) -> None:
        if not self.layer_id or not self.module_path:
            raise ValueError("torch layer binding requires canonical and module paths")
        if self.route_kind not in {
            "linear_weight",
            "conv_channel_pair",
            "conv_kernel_weight",
            "residual_stream_gate",
        }:
            raise ValueError(f"unsupported torch route kind {self.route_kind!r}")
        if self.capture_policy not in {
            "direct_nonmutating",
            "clone_for_downstream",
            "residual_premerge_decomposition",
            "direct_unverified",
        }:
            raise ValueError(f"unsupported torch capture policy {self.capture_policy!r}")
        if self.route_kind == "residual_stream_gate" and self.capture_policy != "residual_premerge_decomposition":
            raise ValueError("residual stream routes require pre-merge decomposition capture")
        if self.route_kind != "residual_stream_gate" and self.capture_policy == "residual_premerge_decomposition":
            raise ValueError("pre-merge decomposition capture is reserved for residual stream routes")
        if self.circuit_role not in {"prunable", "fixed_readout"}:
            raise ValueError(f"unsupported torch binding circuit role {self.circuit_role!r}")
        if claim_bearing and self.capture_policy == "direct_unverified":
            raise ValueError(
                f"claim-bearing capture for {self.layer_id!r} lacks in-place mutation certification"
            )


@dataclass(frozen=True)
class TorchConservingActivationBinding:
    """Explicit affine-to-immediate-activation binding for modern v1.

    ``identity`` means that the affine output itself is the declared
    post-activation tensor and therefore has no activation module path.
    ``relu`` requires a concrete ``nn.ReLU`` path.  Runtime hook ordering and
    tensor identity additionally prove that the declared ReLU directly
    consumes the affine output; module-name adjacency is never inferred.
    """

    layer_id: str
    affine_module_path: str
    activation_kind: Literal["relu", "identity"]
    activation_module_path: str | None

    def validate_for(
        self,
        *,
        layer_binding: TorchLayerBinding,
        activation_module: Any | None,
    ) -> None:
        if (
            not self.layer_id
            or not self.affine_module_path
            or self.layer_id != layer_binding.layer_id
            or self.affine_module_path != layer_binding.module_path
        ):
            raise ValueError(
                "conserving activation binding differs from its affine route binding"
            )
        if layer_binding.route_kind not in {
            "linear_weight",
            "conv_channel_pair",
            "conv_kernel_weight",
        }:
            raise ValueError(
                "conserving activation capture requires an affine route block"
            )
        if self.activation_kind == "relu":
            if (
                not self.activation_module_path
                or self.activation_module_path == self.affine_module_path
                or not isinstance(activation_module, nn.ReLU)
            ):
                raise ValueError(
                    "ReLU conserving capture requires an explicit nn.ReLU module path"
                )
            if layer_binding.capture_policy != "clone_for_downstream":
                raise ValueError(
                    "ReLU conserving capture requires clone_for_downstream "
                    "preactivation preservation"
                )
        elif self.activation_kind == "identity":
            if self.activation_module_path is not None or activation_module is not None:
                raise ValueError(
                    "identity conserving capture cannot declare an activation module"
                )
        else:
            raise ValueError(
                f"unsupported conserving activation kind {self.activation_kind!r}"
            )


def default_response_selector(output: Any) -> Any:
    _require_torch()
    if torch.is_tensor(output):
        response = output
    elif isinstance(output, Mapping) and torch.is_tensor(output.get("logits")):
        response = output["logits"]
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        response = output[0]
    else:
        raise ValueError("model output has no canonical tensor response")
    if response.ndim != 2:
        raise ValueError("canonical model response must be [context,response_coordinate]")
    return response


class TorchModuleMeasurementAdapter:
    """Capture standard Conv2d/Linear modules and emit exact route blocks."""

    @classmethod
    def from_bound_architecture(
        cls,
        bound_architecture: Any,
        *,
        model: Any,
        batch_provider: ContextBatchProvider,
        device: Any = "cpu",
        response_selector: ResponseSelector = default_response_selector,
        direction_resolver: DirectionResolver | None = None,
        parity_receipt_hash: str = "",
        autocast_dtype: Any = None,
        local_regime_specs_by_layer: Mapping[
            str, LocalRegimeCaptureSpec
        ] | None = None,
        conserving_activation_bindings: Sequence[
            TorchConservingActivationBinding
        ] | None = None,
    ) -> TorchModuleMeasurementAdapter:
        bound_architecture.validate()
        return cls(
            model=model,
            architecture_id=bound_architecture.architecture_id,
            route_profile_id=bound_architecture.spec.route_profile_id,
            topology_hash=bound_architecture.topology_hash,
            route_schema_hash=bound_architecture.route_schema_hash,
            layer_bindings=tuple(
                TorchLayerBinding(
                    binding.layer_id,
                    binding.module_path,
                    binding.route_kind,
                    binding.capture_policy,
                    binding.circuit_role,
                )
                for binding in bound_architecture.bindings
            ),
            batch_provider=batch_provider,
            device=device,
            response_selector=response_selector,
            direction_resolver=direction_resolver,
            parity_receipt_hash=parity_receipt_hash,
            autocast_dtype=autocast_dtype,
            local_regime_specs_by_layer=local_regime_specs_by_layer,
            conserving_activation_bindings=conserving_activation_bindings,
        )

    def __init__(
        self,
        *,
        model: Any,
        architecture_id: str,
        route_profile_id: str = "custom_explicit_v1",
        topology_hash: str,
        route_schema_hash: str,
        layer_bindings: Sequence[TorchLayerBinding],
        batch_provider: ContextBatchProvider,
        device: Any = "cpu",
        response_selector: ResponseSelector = default_response_selector,
        direction_resolver: DirectionResolver | None = None,
        parity_receipt_hash: str = "",
        autocast_dtype: Any = None,
        compiled_gate_graph: Any | None = None,
        local_regime_specs_by_layer: Mapping[
            str, LocalRegimeCaptureSpec
        ] | None = None,
        conserving_activation_bindings: Sequence[
            TorchConservingActivationBinding
        ] | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(model, nn.Module):
            raise ValueError("native torch adapter requires an nn.Module")
        if not architecture_id or not route_profile_id or len(topology_hash) != 64 or len(route_schema_hash) != 64:
            raise ValueError(
                "native torch adapter requires architecture, topology, and route-schema identity"
            )
        self.source_model = model
        self.model = model
        self.architecture_id = str(architecture_id)
        self.route_profile_id = str(route_profile_id)
        self.topology_hash = str(topology_hash)
        self.route_schema_hash = str(route_schema_hash)
        self.layer_bindings = tuple(layer_bindings)
        if not self.layer_bindings or len({item.layer_id for item in self.layer_bindings}) != len(
            self.layer_bindings
        ):
            raise ValueError("torch layer bindings must be nonempty with unique canonical IDs")
        if len({item.module_path for item in self.layer_bindings}) != len(self.layer_bindings):
            raise ValueError("torch layer module paths must be unique")
        self.batch_provider = batch_provider
        self.device = torch.device(device)
        self.response_selector = response_selector
        self._direction_resolver = direction_resolver
        self.parity_receipt_hash = str(parity_receipt_hash)
        self.autocast_dtype = autocast_dtype
        self.local_regime_specs_by_layer = dict(
            local_regime_specs_by_layer or {}
        )
        binding_by_layer = {
            binding.layer_id: binding for binding in self.layer_bindings
        }
        if not set(self.local_regime_specs_by_layer).issubset(binding_by_layer):
            raise ValueError(
                "local regime specifications refer to absent route layers"
            )
        for layer_id, spec in self.local_regime_specs_by_layer.items():
            binding = binding_by_layer[layer_id]
            if (
                spec.layer_id != layer_id
                or binding.route_kind != "conv_channel_pair"
            ):
                raise ValueError(
                    "local regime geometry is defined only for "
                    "conv_channel_pair route blocks (A50: kernel-weight "
                    "routes are a registered exclusion, not a silent skip)"
                )
        self.forward_batches_executed = 0
        self.profiler = PhaseProfiler()
        self.adapter_id = f"torch-module::{self.architecture_id}::v1"
        self._blocks = self._build_route_blocks()
        if compiled_gate_graph is None:
            self.gate_graph = TorchGateGraphCompiler().compile_bindings(
                self.source_model,
                self.layer_bindings,
                architecture_id=self.architecture_id,
                topology_hash=self.topology_hash,
                route_schema_hash=self.route_schema_hash,
                device=self.device,
            )
            self.gate_graph_source = "compiled_for_measurement_adapter"
        else:
            expected_layers = {binding.layer_id for binding in self.layer_bindings}
            observed_layers = set(
                str(layer_id)
                for layer_id in getattr(compiled_gate_graph, "gates_by_layer", {})
            )
            if (
                str(getattr(compiled_gate_graph, "architecture_id", ""))
                != self.architecture_id
                or str(getattr(compiled_gate_graph, "topology_hash", ""))
                != self.topology_hash
                or str(getattr(compiled_gate_graph, "route_schema_hash", ""))
                != self.route_schema_hash
                or observed_layers != expected_layers
            ):
                raise ValueError(
                    "shared gate graph differs from the canonical measurement axis"
                )
            self.gate_graph = compiled_gate_graph
            self.gate_graph_source = "shared_canonical_evaluator_graph"
        self.model = self.gate_graph.model
        self.adapter_id = f"torch-fx-gate-graph::{self.architecture_id}::v1"
        requested_activation_bindings = tuple(
            conserving_activation_bindings or ()
        )
        if requested_activation_bindings:
            by_layer = {
                binding.layer_id: binding
                for binding in requested_activation_bindings
            }
            expected_layers = {
                binding.layer_id for binding in self.layer_bindings
            }
            if (
                len(by_layer) != len(requested_activation_bindings)
                or set(by_layer) != expected_layers
            ):
                raise ValueError(
                    "conserving activation bindings must cover every route "
                    "layer exactly once"
                )
            ordered_activation_bindings = tuple(
                by_layer[binding.layer_id] for binding in self.layer_bindings
            )
            relu_paths = tuple(
                binding.activation_module_path
                for binding in ordered_activation_bindings
                if binding.activation_kind == "relu"
            )
            if len(set(relu_paths)) != len(relu_paths):
                raise ValueError(
                    "conserving activation module paths must be one-to-one"
                )
            affine_paths = {
                binding.module_path for binding in self.layer_bindings
            }
            if any(path in affine_paths for path in relu_paths):
                raise ValueError(
                    "a conserving activation path cannot also be an affine route path"
                )
            for layer_binding, activation_binding in zip(
                self.layer_bindings,
                ordered_activation_bindings,
                strict=True,
            ):
                activation_module = (
                    None
                    if activation_binding.activation_module_path is None
                    else self._module(
                        activation_binding.activation_module_path
                    )
                )
                activation_binding.validate_for(
                    layer_binding=layer_binding,
                    activation_module=activation_module,
                )
            self.conserving_activation_bindings = (
                ordered_activation_bindings
            )
        else:
            self.conserving_activation_bindings = ()

    def _module(self, path: str) -> Any:
        try:
            module = self.model.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unresolved native torch module path {path!r}") from error
        return module

    def _build_route_blocks(self) -> tuple[RouteBlockSpec, ...]:
        offset = 0
        blocks: list[RouteBlockSpec] = []
        for binding in self.layer_bindings:
            module = self._module(binding.module_path)
            if binding.route_kind == "linear_weight":
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"layer {binding.module_path!r} is not nn.Linear")
                shape = (int(module.out_features), int(module.in_features))
            elif binding.route_kind == "conv_channel_pair":
                if not isinstance(module, nn.Conv2d):
                    raise ValueError(f"layer {binding.module_path!r} is not nn.Conv2d")
                shape = (int(module.out_channels), int(module.in_channels // module.groups))
            elif binding.route_kind == "conv_kernel_weight":
                # One route per kernel tap under the versioned
                # (target, source, kh, kw) flattening convention.
                if not isinstance(module, nn.Conv2d):
                    raise ValueError(f"layer {binding.module_path!r} is not nn.Conv2d")
                shape = tuple(int(value) for value in module.weight.shape)
            else:
                required = ("conv1", "bn1", "relu", "conv2", "bn2")
                if any(not hasattr(module, name) for name in required):
                    raise ValueError(
                        f"layer {binding.module_path!r} is not a supported residual block"
                    )
                shape = (2,)
            block = RouteBlockSpec(
                architecture_id=self.architecture_id,
                topology_hash=self.topology_hash,
                layer_id=binding.layer_id,
                route_kind=binding.route_kind,
                shape=shape,
                offset=offset,
                flattening=(
                    "c_order_target_source_taps_v1"
                    if binding.route_kind == "conv_kernel_weight"
                    else "c_order_target_then_source"
                ),
            )
            block.validate()
            blocks.append(block)
            offset += block.route_count
        return tuple(blocks)

    def validate_plan(self, plan: MeasurementPlan) -> None:
        claim_bearing = plan.status == "frozen"
        for binding in self.layer_bindings:
            binding.validate(claim_bearing=claim_bearing)
        if claim_bearing:
            raise ValueError(
                "claim-bearing native torch measurement requires a verified execution "
                "receipt interface; opaque parity references are diagnostic only"
            )
        if plan.route_schema_hash != self.route_schema_hash:
            raise ValueError("measurement plan route schema differs from native torch adapter")

    def route_blocks(self, plan: MeasurementPlan) -> Sequence[RouteBlockSpec]:
        self.validate_plan(plan)
        return self._blocks

    def direction_resolver(self) -> DirectionResolver | None:
        return self._direction_resolver

    def scientific_metadata(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "architecture_id": self.architecture_id,
            "route_profile_id": self.route_profile_id,
            "topology_hash": self.topology_hash,
            "route_schema_hash": self.route_schema_hash,
            "parity_receipt_hash": self.parity_receipt_hash,
            "capture_bindings": [
                {
                    "layer_id": binding.layer_id,
                    "module_path": binding.module_path,
                    "route_kind": binding.route_kind,
                    "capture_policy": binding.capture_policy,
                    "circuit_role": binding.circuit_role,
                }
                for binding in self.layer_bindings
            ],
            "response_selector": getattr(
                self.response_selector,
                "__qualname__",
                type(self.response_selector).__qualname__,
            ),
            "claim_capture_certified": False,
            "execution_certification_status": "unverified_opaque_parity_reference",
            "gate_graph_compiler": self.gate_graph.model.__class__.__qualname__,
            "gate_graph_source": self.gate_graph_source,
            "forward_monkeypatching": False,
            "local_regime_geometry": {
                layer_id: {
                    "geometry_hash": spec.geometry_hash,
                    "activation_kind": spec.activation_kind,
                    "capture_source": spec.capture_source,
                }
                for layer_id, spec in sorted(
                    self.local_regime_specs_by_layer.items()
                )
            },
            "conserving_activation_capture": {
                "enabled": bool(self.conserving_activation_bindings),
                "bindings": [
                    {
                        "layer_id": binding.layer_id,
                        "affine_module_path": binding.affine_module_path,
                        "activation_kind": binding.activation_kind,
                        "activation_module_path": (
                            binding.activation_module_path
                        ),
                    }
                    for binding in self.conserving_activation_bindings
                ],
                "immediacy_proof": (
                    "runtime_module_invocation_and_tensor_identity_v1"
                    if self.conserving_activation_bindings
                    else "not_requested"
                ),
            },
        }

    def runtime_details(self) -> dict[str, Any]:
        peak_hbm = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_hbm = int(torch.cuda.max_memory_allocated(self.device))
        return {
            "adapter_id": self.adapter_id,
            "device_type": self.device.type,
            "forward_batches_executed": int(self.forward_batches_executed),
            "peak_hbm_bytes": peak_hbm,
            "gate_graph_source": self.gate_graph_source,
            **self.profiler.snapshot(),
        }

    @staticmethod
    def _track_module_invocation(
        module_path: str,
        invocation_state: dict[str, Any],
    ) -> Callable[..., None]:
        def hook(_module: Any, _arguments: tuple[Any, ...]) -> None:
            invocation_state["ordinal"] = int(invocation_state["ordinal"]) + 1
            invocation_state["module_path"] = str(module_path)

        return hook

    def _capture_one(
        self,
        binding: TorchLayerBinding,
        module: Any,
        captured: dict[str, tuple[Any, Any]],
        affine_downstream: dict[str, Any],
        affine_invocation_ordinals: dict[str, int],
        invocation_state: dict[str, Any],
    ) -> Any:
        def hook(_module: Any, arguments: tuple[Any, ...], output: Any) -> Any:
            if not arguments or not torch.is_tensor(arguments[0]) or not torch.is_tensor(output):
                raise ValueError(f"layer {binding.module_path!r} did not expose tensor input/output")
            if binding.layer_id in captured:
                raise ValueError(f"layer {binding.module_path!r} executed more than once in one forward")
            captured[binding.layer_id] = (arguments[0].detach(), output)
            downstream = (
                output.clone()
                if binding.capture_policy == "clone_for_downstream"
                else output
            )
            affine_downstream[binding.layer_id] = downstream
            affine_invocation_ordinals[binding.layer_id] = int(
                invocation_state["ordinal"]
            )
            if binding.capture_policy == "clone_for_downstream":
                # The model consumes the clone. Gradients still flow to the
                # unmodified captured preactivation through CloneBackward,
                # while a downstream in-place ReLU cannot overwrite it.
                return downstream
            return None

        return module.register_forward_hook(hook)

    def _capture_immediate_activation(
        self,
        binding: TorchConservingActivationBinding,
        module: Any,
        *,
        affine_downstream: dict[str, Any],
        affine_invocation_ordinals: dict[str, int],
        captured_activations: dict[str, Any],
        invocation_state: dict[str, Any],
    ) -> Any:
        def hook(_module: Any, arguments: tuple[Any, ...], output: Any) -> None:
            if binding.layer_id in captured_activations:
                raise ValueError(
                    f"activation {binding.activation_module_path!r} executed "
                    "more than once in one forward"
                )
            expected = affine_downstream.get(binding.layer_id)
            affine_ordinal = affine_invocation_ordinals.get(binding.layer_id)
            if (
                expected is None
                or affine_ordinal is None
                or not arguments
                or arguments[0] is not expected
                or invocation_state["module_path"]
                != binding.activation_module_path
                or int(invocation_state["ordinal"]) != affine_ordinal + 1
            ):
                raise ValueError(
                    "declared conserving activation is not the immediate "
                    f"consumer of affine layer {binding.affine_module_path!r}"
                )
            if not torch.is_tensor(output) or not output.requires_grad:
                raise ValueError(
                    "declared conserving activation did not emit a "
                    "differentiable tensor"
                )
            captured_activations[binding.layer_id] = output

        return module.register_forward_hook(hook)

    @staticmethod
    def _bias_baseline(module: Any, output: Any) -> Any:
        if isinstance(module, GatedLinear):
            if output.ndim != 2:
                raise ValueError(
                    "conserving linear capture requires [context,output] geometry"
                )
            source = module.module
            shape = (1, int(source.out_features))
        elif isinstance(module, GatedConv2d):
            if output.ndim != 4:
                raise ValueError(
                    "conserving Conv capture requires [context,channel,height,width] geometry"
                )
            source = module.module
            shape = (1, int(source.out_channels), 1, 1)
        else:  # pragma: no cover - guarded by route-block construction.
            raise TypeError("conserving capture requires a gated affine module")
        bias = source.bias
        if bias is None:
            return torch.zeros(shape, device=output.device, dtype=output.dtype)
        return bias.detach().to(
            device=output.device,
            dtype=output.dtype,
        ).reshape(shape)

    def iter_captured_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
        queries: Sequence[QuerySpec],
    ) -> Iterable[CapturedMeasurementBatch]:
        del queries  # Query evaluation belongs to the canonical producer.
        self.validate_plan(plan)
        prior_training = bool(self.model.training)
        self.model.to(self.device)
        self.model.eval()
        try:
            for input_batch in self.batch_provider.iter_batches(
                plan,
                context_block_id=context_block_id,
            ):
                input_batch.validate()
                input_batch.wait_until_ready(self.device)
                with self.profiler.phase("measurement.input_transfer"):
                    inputs = input_batch.inputs.to(self.device, non_blocking=True)
                    if not inputs.is_floating_point():
                        inputs = inputs.to(dtype=torch.float32)
                    inputs = inputs.detach().requires_grad_(True)
                    labels = (
                        None
                        if input_batch.labels is None
                        else torch.as_tensor(input_batch.labels, device=self.device)
                    )
                captured: dict[str, tuple[Any, Any]] = {}
                affine_downstream: dict[str, Any] = {}
                affine_invocation_ordinals: dict[str, int] = {}
                captured_activations: dict[str, Any] = {}
                invocation_state: dict[str, Any] = {
                    "ordinal": 0,
                    "module_path": "",
                }
                handles = []
                armed_residual_paths: set[str] = set()

                if self.conserving_activation_bindings:
                    for module_path, module in self.model.named_modules():
                        if module_path:
                            handles.append(
                                module.register_forward_pre_hook(
                                    self._track_module_invocation(
                                        module_path,
                                        invocation_state,
                                    )
                                )
                            )

                def arm_residual_capture(module_path: str) -> None:
                    path = str(module_path)
                    if path in armed_residual_paths:
                        return
                    residual_module = self._module(path)
                    if not isinstance(residual_module, GatedResidualBlock):
                        raise TypeError(
                            "local regime residual preactivation source is "
                            "not a gated residual block"
                        )
                    residual_module.reset_capture()
                    armed_residual_paths.add(path)

                for binding in self.layer_bindings:
                    module = self._module(binding.module_path)
                    if binding.route_kind == "residual_stream_gate":
                        if not isinstance(module, GatedResidualBlock):
                            raise TypeError("residual route was not lowered to an explicit gate")
                        arm_residual_capture(binding.module_path)
                    else:
                        handles.append(
                            self._capture_one(
                                binding,
                                module,
                                captured,
                                affine_downstream,
                                affine_invocation_ordinals,
                                invocation_state,
                            )
                        )
                for activation_binding in self.conserving_activation_bindings:
                    if activation_binding.activation_kind != "relu":
                        continue
                    assert activation_binding.activation_module_path is not None
                    handles.append(
                        self._capture_immediate_activation(
                            activation_binding,
                            self._module(
                                activation_binding.activation_module_path
                            ),
                            affine_downstream=affine_downstream,
                            affine_invocation_ordinals=(
                                affine_invocation_ordinals
                            ),
                            captured_activations=captured_activations,
                            invocation_state=invocation_state,
                        )
                    )
                for spec in self.local_regime_specs_by_layer.values():
                    if spec.capture_source == "residual_output_preactivation":
                        assert spec.residual_block_path is not None
                        arm_residual_capture(spec.residual_block_path)
                try:
                    autocast_enabled = self.autocast_dtype is not None and self.device.type == "cuda"
                    autocast_context = (
                        torch.autocast(
                            device_type=self.device.type,
                            dtype=self.autocast_dtype,
                            enabled=True,
                        )
                        if autocast_enabled
                        else nullcontext()
                    )
                    with autocast_context:
                        with self.profiler.phase("measurement.model_forward"):
                            self.forward_batches_executed += 1
                            response = self.response_selector(self.model(inputs))
                finally:
                    for handle in handles:
                        handle.remove()
                local_regime_baselines: dict[str, Any] = {}
                for layer_id, spec in self.local_regime_specs_by_layer.items():
                    if spec.capture_source == "affine_bound_output":
                        raw_output = captured[layer_id][1]
                        scale = torch.as_tensor(
                            spec.route_scale,
                            dtype=torch.float32,
                            device=raw_output.device,
                        ).reshape(1, -1, 1, 1)
                        offset = torch.as_tensor(
                            spec.affine_offset,
                            dtype=torch.float32,
                            device=raw_output.device,
                        ).reshape(1, -1, 1, 1)
                        local_regime_baselines[layer_id] = (
                            raw_output.detach().to(torch.float32) * scale + offset
                        )
                    else:
                        assert spec.residual_block_path is not None
                        residual_module = self._module(
                            spec.residual_block_path
                        )
                        baseline = residual_module.last_output_preactivation
                        if baseline is None:
                            raise ValueError(
                                "residual block omitted its armed output "
                                "preactivation capture"
                            )
                        local_regime_baselines[layer_id] = baseline.detach().to(
                            torch.float32
                        )
                for binding in self.layer_bindings:
                    if binding.route_kind != "residual_stream_gate":
                        continue
                    module = self._module(binding.module_path)
                    if module.capture_execution_count != 1 or module.last_branches is None:
                        raise ValueError(
                            f"residual gate {binding.module_path!r} executed "
                            f"{module.capture_execution_count} times"
                        )
                    captured[binding.layer_id] = (None, module.last_branches)
                for path in armed_residual_paths:
                    # Disarm after every consumer has copied its ephemeral
                    # branch/preactivation reference.
                    self._module(path).disarm_capture()
                missing = sorted({binding.layer_id for binding in self.layer_bindings} - set(captured))
                if missing:
                    raise ValueError("model forward omitted captured layers: " + ",".join(missing))
                if self.conserving_activation_bindings:
                    missing_activations = sorted(
                        binding.layer_id
                        for binding in self.conserving_activation_bindings
                        if binding.activation_kind == "relu"
                        and binding.layer_id not in captured_activations
                    )
                    if missing_activations:
                        raise ValueError(
                            "model forward omitted declared immediate activations: "
                            + ",".join(missing_activations)
                        )
                route_packets: list[CapturedRouteBlock] = []
                for binding, route_block in zip(self.layer_bindings, self._blocks, strict=True):
                    module = self._module(binding.module_path)
                    input_source, output = captured[binding.layer_id]
                    if binding.route_kind == "linear_weight":
                        if not isinstance(module, GatedLinear):
                            raise TypeError("linear route was not lowered to an explicit gate")
                        parameters: dict[str, Any] = {}
                        weight = module.module.weight.detach()
                    elif binding.route_kind == "conv_channel_pair":
                        if not isinstance(module, GatedConv2d):
                            raise TypeError("Conv route was not lowered to an explicit gate")
                        parameters = {
                            "stride": module.module.stride,
                            "padding": module.module.padding,
                            "dilation": module.module.dilation,
                            "groups": int(module.module.groups),
                        }
                        regime_spec = self.local_regime_specs_by_layer.get(
                            binding.layer_id
                        )
                        if regime_spec is not None:
                            parameters.update(
                                {
                                    "local_regime_preactivation": (
                                        local_regime_baselines[
                                            binding.layer_id
                                        ]
                                    ),
                                    "local_regime_route_scale": (
                                        regime_spec.route_scale
                                    ),
                                    "local_regime_geometry_hash": (
                                        regime_spec.geometry_hash
                                    ),
                                    "local_regime_activation_kind": (
                                        regime_spec.activation_kind
                                    ),
                                    "local_regime_active_tolerance": (
                                        regime_spec.local_active_tolerance
                                    ),
                                    "local_regime_boundary_epsilon": (
                                        regime_spec.boundary_epsilon
                                    ),
                                    "local_regime_route_chunk_size": (
                                        regime_spec.route_chunk_size
                                    ),
                                }
                            )
                        weight = module.module.weight.detach()
                    elif binding.route_kind == "conv_kernel_weight":
                        # The per-tap capture branch emits the same
                        # geometry parameters as the channel-pair branch;
                        # the per-tap operator keys on them, and per-route
                        # rows live on the versioned 4-D
                        # (target, source, kh, kw) axis. Local-regime
                        # geometry stays a registered exclusion for this
                        # kind (enforced at adapter construction).
                        if (
                            not isinstance(module, GatedConv2d)
                            or module.gate_granularity != "kernel_weight"
                        ):
                            raise TypeError(
                                "Conv kernel route was not lowered to an "
                                "explicit kernel-weight gate"
                            )
                        parameters = {
                            "stride": module.module.stride,
                            "padding": module.module.padding,
                            "dilation": module.module.dilation,
                            "groups": int(module.module.groups),
                        }
                        weight = module.module.weight.detach()
                    else:
                        parameters = {}
                        weight = None
                    route_packets.append(
                        CapturedRouteBlock(
                            route_block=route_block,
                            reducer_kind=binding.route_kind,
                            output=output,
                            input_source=input_source,
                            weight=weight,
                            reducer_parameters=parameters,
                        )
                    )
                conserving_captures: (
                    tuple[V1ConservingActivationCapture, ...] | None
                ) = None
                if self.conserving_activation_bindings:
                    activation_packets: list[
                        V1ConservingActivationCapture
                    ] = []
                    for activation_binding, route_packet in zip(
                        self.conserving_activation_bindings,
                        route_packets,
                        strict=True,
                    ):
                        activation_source = (
                            activation_binding.activation_module_path
                            or "<identity>"
                        )
                        post_activation = (
                            captured_activations[activation_binding.layer_id]
                            if activation_binding.activation_kind == "relu"
                            else affine_downstream[activation_binding.layer_id]
                        )
                        activation_packet = V1ConservingActivationCapture(
                            layer_id=activation_binding.layer_id,
                            activation_kind=activation_binding.activation_kind,
                            post_activation_output=post_activation,
                            bias_baseline=self._bias_baseline(
                                self._module(
                                    activation_binding.affine_module_path
                                ),
                                route_packet.output,
                            ),
                            capture_source_identity=(
                                f"{activation_binding.affine_module_path}->"
                                f"{activation_source}:immediate_v1"
                            ),
                        )
                        activation_packet.validate_for(route_packet)
                        activation_packets.append(activation_packet)
                    conserving_captures = tuple(activation_packets)
                yield CapturedMeasurementBatch(
                    context_block_id=context_block_id,
                    response_matrix=response,
                    labels=labels,
                    route_blocks=tuple(route_packets),
                    context_ids=input_batch.context_ids,
                    conserving_activation_captures=conserving_captures,
                )
                input_batch.mark_consumed(self.device)
        finally:
            self.model.train(prior_training)


__all__ = [
    "ContextBatchProvider",
    "MeasurementInputBatch",
    "ResponseSelector",
    "TorchConservingActivationBinding",
    "TorchLayerBinding",
    "TorchModuleMeasurementAdapter",
    "default_response_selector",
]
