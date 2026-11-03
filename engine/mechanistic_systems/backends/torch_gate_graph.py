"""Explicit, nonmutating Torch gate graph for shared scoring and execution."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

try:  # Keep contract/report imports available in CPU-only environments.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.fx as fx
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    fx = None  # type: ignore[assignment]

from mechanistic_systems.core.architectures import (
    BoundTorchArchitecture,
)


_ModuleBase = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None or F is None or fx is None:
        raise RuntimeError("explicit gate graphs require PyTorch")


FROZEN_AFFINE_BATCH_NORM_MODE = "frozen_affine_v2"
NATIVE_BATCH_NORM_MODE = "native_eval_lane_batch_norm_v1"


def _batch_norm_compiler_id(mode: str) -> str:
    if mode == FROZEN_AFFINE_BATCH_NORM_MODE:
        return "mechanistic_systems.torch_gate_graph.v2_frozen_affine"
    if mode == NATIVE_BATCH_NORM_MODE:
        return "mechanistic_systems.torch_gate_graph.v3_native_eval_lane_batch_norm"
    raise ValueError("unsupported gate-graph BatchNorm execution mode")


class _NativeLaneBatchNorm(torch.autograd.Function if torch is not None else object):
    @staticmethod
    def forward(inputs, running_mean, running_var, weight, bias, epsilon):
        if inputs.device.type != "cpu" or inputs.dtype not in {torch.float32, torch.float64}:
            raise ValueError("native lane BatchNorm is qualified only for CPU float32/float64")
        if inputs.ndim != 4 or not inputs.is_contiguous():
            raise ValueError("native lane BatchNorm requires contiguous NCHW inputs")
        return F.batch_norm(inputs, running_mean, running_var, weight, bias,
                            False, 0.0, float(epsilon))

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, mean, variance, weight, bias, epsilon = inputs
        ctx.save_for_backward(value, mean, variance, weight)
        ctx.epsilon = float(epsilon)

    @staticmethod
    def backward(ctx, output_gradient):
        value, mean, variance, weight = ctx.saved_tensors
        dx, dw, db = torch.ops.aten.native_batch_norm_backward(
            output_gradient, value, weight, mean, variance, None, None,
            False, ctx.epsilon,
            (ctx.needs_input_grad[0], ctx.needs_input_grad[3], ctx.needs_input_grad[4]))
        return dx, None, None, dw, db, None

    @staticmethod
    def vmap(info, in_dims, inputs, running_mean, running_var, weight, bias, epsilon):
        if any(dim is not None for dim in in_dims[1:]):
            raise ValueError("native lane BatchNorm requires shared statistics and affine parameters")
        if in_dims[0] is None:
            return _NativeLaneBatchNorm.apply(inputs, running_mean, running_var, weight, bias, epsilon), None
        lanes = inputs.movedim(in_dims[0], 0)
        return torch.stack([
            _NativeLaneBatchNorm.apply(value.contiguous(), running_mean, running_var, weight, bias, epsilon)
            for value in lanes.unbind(0)
        ], dim=0), 0


@(fx.wrap if fx is not None else (lambda function: function))
def _native_lane_batch_norm(inputs, running_mean, running_var, weight, bias, epsilon):
    return _NativeLaneBatchNorm.apply(inputs, running_mean, running_var, weight, bias, epsilon)


def _eval_batch_norm_2d(module: Any, inputs: Any, *, batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE) -> Any:
    """Apply the explicitly selected frozen normalization arithmetic.

    ``torch.vmap`` represents the candidate axis with ``BatchedTensor``.
    PyTorch's CUDA BatchNorm dispatcher currently queries a non-default
    memory format on that wrapper and raises before executing, even though
    evaluation BatchNorm has no batch coupling: its running statistics are
    frozen constants.  Writing the same affine explicitly keeps every
    channel and spatial coordinate, is differentiable, and makes the
    registered ResNet suffix genuinely vectorizable across candidate lanes.

    ``nn.Identity`` is admitted as the exact no-normalization map.  This keeps
    structurally valid BasicBlock-like modules executable without pretending
    that an arbitrary normalization layer has BatchNorm semantics.  The gate
    graph is permanently in ``eval()``; every other module type refuses, as
    does BatchNorm without frozen running statistics.
    """

    _batch_norm_compiler_id(batch_norm_mode)
    if isinstance(module, nn.Identity):
        return inputs
    if not isinstance(module, nn.BatchNorm2d):
        raise TypeError(
            "residual normalization requires nn.BatchNorm2d or nn.Identity"
        )
    if module.training or module.running_mean is None or module.running_var is None:
        raise RuntimeError("gated residual execution requires frozen eval BatchNorm")
    if batch_norm_mode == NATIVE_BATCH_NORM_MODE:
        return _native_lane_batch_norm(inputs, module.running_mean, module.running_var,
                                       module.weight, module.bias, float(module.eps))
    scale = torch.rsqrt(module.running_var + float(module.eps))
    if module.weight is not None:
        scale = scale * module.weight
    offset = -module.running_mean * scale
    if module.bias is not None:
        offset = offset + module.bias
    # ``scale`` becomes an FX Proxy when the observation-only boundary graph
    # traces through a residual wrapper.  The module declaration is the exact
    # static channel authority and remains a Python integer during tracing.
    channel_shape = (1, int(module.num_features), 1, 1)
    return inputs * scale.reshape(channel_shape) + offset.reshape(channel_shape)


def _eval_downsample(module: Any, inputs: Any, *, batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE) -> Any:
    """Execute a torchvision residual projection with vmap-safe BatchNorm."""

    if isinstance(module, nn.Sequential):
        output = inputs
        for child in module:
            output = (
                _eval_batch_norm_2d(child, output, batch_norm_mode=batch_norm_mode)
                if isinstance(child, nn.BatchNorm2d)
                else child(output)
            )
        return output
    return module(inputs)


class FrozenBatchNorm2d(_ModuleBase):
    """Frozen normalization with affine or native per-lane arithmetic.

    The wrapped normalization retains every parameter and running-statistic
    buffer. Compiler callers supply a module from their independent model
    copy; the original source network is not changed. This real-valued eval
    affine map agrees with BatchNorm in real arithmetic; native kernels can
    round differently. The native mode preserves each lane
    as a separate native evaluation call while other operations remain vmapped.
    """

    def __init__(self, module: Any, *, batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE) -> None:
        _require_torch()
        _batch_norm_compiler_id(batch_norm_mode)
        self.batch_norm_mode = batch_norm_mode
        super().__init__()
        if not isinstance(module, nn.BatchNorm2d):
            raise TypeError("frozen normalization requires nn.BatchNorm2d")
        if module.running_mean is None or module.running_var is None:
            raise ValueError("frozen normalization requires running statistics")
        self.module = module
        # The compiled graph is an eval-only map. Freeze before FX tracing,
        # which occurs before the compiler's final recursive eval call.
        self.eval()

    def forward(self, inputs: Any) -> Any:
        if self.training:
            raise RuntimeError("frozen normalization cannot execute in training mode")
        return _eval_batch_norm_2d(self.module, inputs, batch_norm_mode=self.batch_norm_mode)


class _GateLeafTracer(fx.Tracer if fx is not None else object):
    """Trace the model while preserving executable gate modules as leaves."""

    def is_leaf_module(self, module: Any, qualified_name: str) -> bool:
        if isinstance(
            module, (GatedConv2d, GatedLinear, GatedResidualBlock, FrozenBatchNorm2d)
        ):
            return True
        return super().is_leaf_module(module, qualified_name)


class GatedConv2d(_ModuleBase):
    """Conv2d with one executable gate per output/input-within-group pair.

    ``granularity="kernel_weight"`` carries one gate per
    (output, input-within-group, kh, kw) tap instead; the forward is
    the same elementwise product with the weight tensor.
    """

    def __init__(self, module: Any, *, granularity: str = "channel_pair") -> None:
        _require_torch()
        super().__init__()
        if not isinstance(module, nn.Conv2d):
            raise TypeError("GatedConv2d requires nn.Conv2d")
        if granularity not in {"channel_pair", "kernel_weight"}:
            raise ValueError(
                f"unsupported conv gate granularity {granularity!r}"
            )
        self.module = module
        self.gate_granularity = str(granularity)
        gate_shape = (
            (
                int(module.out_channels),
                int(module.in_channels // module.groups),
            )
            if granularity == "channel_pair"
            else tuple(int(value) for value in module.weight.shape)
        )
        self.register_buffer(
            "route_gate",
            torch.ones(
                gate_shape,
                dtype=module.weight.dtype,
                device=module.weight.device,
            ),
            persistent=False,
        )
        # Empty outside functional execution. Prepared lanes supply these
        # buffers without changing the differentiable gate/scoring graph.
        for name in ("execution_weight", "execution_stem_weight", "execution_stem_bias", "uint8_decode_pad",
                     "execution_stem_weight_hi", "execution_stem_weight_lo", "execution_stem_pad_residual"):
            self.register_buffer(name, module.weight.new_empty(0), persistent=False)

    def set_gate(self, gate: Any) -> None:
        value = torch.as_tensor(gate, dtype=self.route_gate.dtype, device=self.route_gate.device)
        if value.shape != self.route_gate.shape or not torch.isfinite(value).all():
            raise ValueError("Conv route gate has an invalid shape or value")
        self.route_gate.copy_(value)

    def forward(self, inputs: Any) -> Any:
        weight = self.execution_weight if self.execution_weight.numel() else self.module.weight * (
            self.route_gate[:, :, None, None]
            if self.gate_granularity == "channel_pair"
            else self.route_gate
        )
        if inputs.dtype == torch.uint8 and hasattr(self, "uint8_decode_scale"):
            from mechanistic_systems.measurement.native_panel_data_plane import compute_uint8_folded_stem
            from .torch_uint8_stem import (
                pad_stem_weight, prepare_stem_components, uint8_stem_conv, uint8_stem_pad,
            )
            if self.execution_stem_weight.numel():
                stem_weight, stem_bias, pad = (self.execution_stem_weight,
                                              self.execution_stem_bias, self.uint8_decode_pad)
            else:
                folded = compute_uint8_folded_stem(
                    weight, self.module.bias, decode_scale=self.uint8_decode_scale,
                    decode_bias=self.uint8_decode_bias, detach=False,
                )
                stem_weight, stem_bias, pad = (pad_stem_weight(folded["weight"]),
                                              folded["bias"], folded["pad_value"].reshape(-1))
            ph, pw = self.module.padding
            if inputs.device.type == "cuda" and stem_weight.dtype == torch.float32 and self.module.groups == 1:
                if self.execution_stem_weight_hi.numel():
                    components = (self.execution_stem_weight_hi, self.execution_stem_weight_lo,
                                  self.execution_stem_pad_residual)
                else:
                    values = prepare_stem_components(stem_weight, pad)
                    components = (values["execution_stem_weight_hi"], values["execution_stem_weight_lo"],
                                  values["execution_stem_pad_residual"])
                return uint8_stem_conv(inputs, stem_weight, stem_bias, pad, *components,
                                       *self.module.stride, ph, pw, *self.module.dilation)
            raw = uint8_stem_pad(inputs, pad, ph, pw,
                                 stem_weight.shape[1])
            # Fold AFTER applying the lane's mask: the bias shift includes
            # exactly the surviving taps, including at the padded edges.
            return F.conv2d(raw, stem_weight, stem_bias, self.module.stride,
                            0, self.module.dilation, self.module.groups)
        return F.conv2d(
            inputs,
            weight,
            self.module.bias,
            self.module.stride,
            self.module.padding,
            self.module.dilation,
            self.module.groups,
        )


class GatedLinear(_ModuleBase):
    """Linear layer with one executable gate per weight edge."""

    def __init__(self, module: Any) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(module, nn.Linear):
            raise TypeError("GatedLinear requires nn.Linear")
        self.module = module
        self.register_buffer(
            "route_gate",
            torch.ones_like(module.weight, memory_format=torch.preserve_format),
            persistent=False,
        )
        self.register_buffer("execution_weight", module.weight.new_empty(0), persistent=False)

    def set_gate(self, gate: Any) -> None:
        value = torch.as_tensor(gate, dtype=self.route_gate.dtype, device=self.route_gate.device)
        if value.shape != self.route_gate.shape or not torch.isfinite(value).all():
            raise ValueError("Linear route gate has an invalid shape or value")
        self.route_gate.copy_(value)

    def forward(self, inputs: Any) -> Any:
        weight = self.execution_weight if self.execution_weight.numel() else self.module.weight * self.route_gate
        return F.linear(inputs, weight, self.module.bias)


class GatedResidualBlock(_ModuleBase):
    """Torchvision-style BasicBlock/Bottleneck with explicit pre-add gates."""

    def __init__(self, block: Any, *, batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE) -> None:
        _require_torch()
        _batch_norm_compiler_id(batch_norm_mode)
        self.batch_norm_mode = batch_norm_mode
        super().__init__()
        required = ("conv1", "bn1", "relu", "conv2", "bn2")
        if any(not hasattr(block, name) for name in required):
            raise TypeError("GatedResidualBlock requires a BasicBlock/Bottleneck-like module")
        # Preserve the original child paths (for example ``layer2.0.conv1``)
        # when the containing block is replaced. Nesting the source block
        # under ``.block`` would silently invalidate stable route bindings.
        self.conv1 = block.conv1
        self.bn1 = block.bn1
        self.relu = block.relu
        self.conv2 = block.conv2
        self.bn2 = block.bn2
        self.downsample = getattr(block, "downsample", None)
        if hasattr(block, "conv3") and hasattr(block, "bn3"):
            self.conv3 = block.conv3
            self.bn3 = block.bn3
        reference = next(block.parameters(), None)
        dtype = reference.dtype if reference is not None else torch.float32
        device = reference.device if reference is not None else torch.device("cpu")
        self.register_buffer(
            "route_gate",
            torch.ones(2, dtype=dtype, device=device),
            persistent=False,
        )
        self.last_branches: Any | None = None
        self.last_output_preactivation: Any | None = None
        self.capture_execution_count = 0
        # Capture is opt-in: retaining the [B, 2, C, H, W] pre-add branch
        # stack on every forward taxed memory on the masked-forward hot path
        # (where nothing reads it) and mutates module state, which is
        # incompatible with vmap-batched candidate execution. Measurement
        # consumers arm capture around their window via reset_capture().
        self.capture_enabled: bool = False

    def reset_capture(self) -> None:
        """Arm branch capture for the caller's measurement window."""

        self.capture_enabled = True
        self.last_branches = None
        self.last_output_preactivation = None
        self.capture_execution_count = 0

    def disarm_capture(self) -> None:
        """Disarm capture and release retained branch activations."""

        self.capture_enabled = False
        self.last_branches = None
        self.last_output_preactivation = None
        self.capture_execution_count = 0

    def set_gate(self, gate: Any) -> None:
        value = torch.as_tensor(gate, dtype=self.route_gate.dtype, device=self.route_gate.device)
        if value.shape != (2,) or not torch.isfinite(value).all():
            raise ValueError("residual stream gate must contain main and skip scalars")
        self.route_gate.copy_(value)

    def forward(self, inputs: Any) -> Any:
        identity = inputs
        out = self.conv1(inputs)
        out = _eval_batch_norm_2d(self.bn1, out, batch_norm_mode=self.batch_norm_mode)
        out = self.relu(out)
        out = self.conv2(out)
        out = _eval_batch_norm_2d(self.bn2, out, batch_norm_mode=self.batch_norm_mode)
        if hasattr(self, "conv3") and hasattr(self, "bn3"):
            out = self.relu(out)
            out = self.conv3(out)
            out = _eval_batch_norm_2d(self.bn3, out, batch_norm_mode=self.batch_norm_mode)
        if self.downsample is not None:
            identity = _eval_downsample(self.downsample, inputs, batch_norm_mode=self.batch_norm_mode)
        branches = torch.stack((out, identity), dim=1)
        if self.capture_enabled:
            self.last_branches = branches
            self.capture_execution_count += 1
        # Residual wrappers are defined only for ConvNet feature maps, hence
        # branches are always [batch, main/skip, channel, height, width].  The
        # explicit rank also keeps the observation-only boundary-valued FX
        # graph traceable; deriving a Python tuple from Proxy.ndim is illegal.
        out = (branches * self.route_gate.reshape(1, 2, 1, 1, 1)).sum(dim=1)
        if self.capture_enabled:
            # Exact activation-boundary coordinate after branch gating and
            # before the block's terminal ReLU. This is retained only inside
            # an explicitly armed measurement window. Torchvision ResNets use
            # an in-place ReLU, so a detached clone is required; retaining the
            # alias would silently overwrite the preactivation with its
            # post-ReLU value.
            self.last_output_preactivation = out.detach().clone()
        return self.relu(out)


def _replace_submodule(model: Any, module_path: str, replacement: Any) -> None:
    components = str(module_path).split(".")
    parent = model
    for component in components[:-1]:
        parent = parent.get_submodule(component)
    setattr(parent, components[-1], replacement)


@dataclass
class CompiledGateGraph:
    model: Any
    gates_by_layer: dict[str, Any]
    gate_buffer_names: dict[str, str]
    architecture_id: str
    topology_hash: str
    route_schema_hash: str
    batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE
    compiler_id: str = "mechanistic_systems.torch_gate_graph.v2_frozen_affine"

    def validate_execution_mode(self) -> None:
        if self.compiler_id != _batch_norm_compiler_id(self.batch_norm_mode):
            raise ValueError("compiled BatchNorm mode and compiler identity disagree")
        for module in self.model.modules():
            if isinstance(module, (FrozenBatchNorm2d, GatedResidualBlock)):
                if module.batch_norm_mode != self.batch_norm_mode:
                    raise ValueError("compiled BatchNorm module mode changed")

    def set_gate(self, layer_id: str, gate: Any) -> None:
        try:
            module = self.gates_by_layer[str(layer_id)]
        except KeyError as error:
            raise KeyError(f"unknown executable gate layer {layer_id!r}") from error
        module.set_gate(gate)

    def reset(self) -> None:
        for module in self.gates_by_layer.values():
            module.route_gate.fill_(1.0)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


class TorchGateGraphCompiler:
    """Compile shared route bindings into an independent executable model."""

    compiler_id = "mechanistic_systems.torch_gate_graph.v2_frozen_affine"

    def __init__(self, *, batch_norm_mode: str = FROZEN_AFFINE_BATCH_NORM_MODE) -> None:
        self.batch_norm_mode = batch_norm_mode
        self.compiler_id = _batch_norm_compiler_id(batch_norm_mode)

    def compile(
        self,
        model: Any,
        route_bindings: BoundTorchArchitecture,
        *,
        device: Any = None,
    ) -> CompiledGateGraph:
        _require_torch()
        route_bindings.validate()
        return self.compile_bindings(
            model,
            route_bindings.bindings,
            architecture_id=route_bindings.architecture_id,
            topology_hash=route_bindings.topology_hash,
            route_schema_hash=route_bindings.route_schema_hash,
            device=device,
        )

    def compile_bindings(
        self,
        model: Any,
        bindings: Any,
        *,
        architecture_id: str,
        topology_hash: str,
        route_schema_hash: str,
        device: Any = None,
    ) -> CompiledGateGraph:
        """Compile an already validated ordered executable binding axis."""

        if not isinstance(model, nn.Module):
            raise TypeError("TorchGateGraphCompiler requires nn.Module")
        if self.batch_norm_mode == NATIVE_BATCH_NORM_MODE:
            if device is not None and torch.device(device).type != "cpu":
                raise ValueError("native lane BatchNorm compiler currently requires CPU")
            if any(value.device.type != "cpu" for value in model.state_dict().values()):
                raise ValueError("native lane BatchNorm compiler requires a CPU source model")
        compiled = copy.deepcopy(model)
        from mechanistic_systems.backends.deterministic_pooling import install_deterministic_adaptive_avg_pool2d
        install_deterministic_adaptive_avg_pool2d(compiled)
        gate_paths: dict[str, str] = {}

        # Weight gates are inserted first. Residual wrappers then retain those
        # already-gated child modules inside their exact native block order.
        for binding in bindings:
            if binding.route_kind == "residual_stream_gate":
                continue
            module = compiled.get_submodule(binding.module_path)
            if binding.route_kind == "conv_channel_pair":
                wrapper = GatedConv2d(module)
            elif binding.route_kind == "conv_kernel_weight":
                wrapper = GatedConv2d(module, granularity="kernel_weight")
            elif binding.route_kind == "linear_weight":
                wrapper = GatedLinear(module)
            else:
                raise ValueError(
                    "no executable gate wrapper is registered for route "
                    f"kind {binding.route_kind!r}"
                )
            _replace_submodule(compiled, binding.module_path, wrapper)
            gate_paths[str(binding.layer_id)] = str(binding.module_path)
        for binding in bindings:
            if binding.route_kind != "residual_stream_gate":
                continue
            block = compiled.get_submodule(binding.module_path)
            wrapper = GatedResidualBlock(block, batch_norm_mode=self.batch_norm_mode)
            _replace_submodule(compiled, binding.module_path, wrapper)
            gate_paths[str(binding.layer_id)] = str(binding.module_path)
        # Residual wrappers already use _eval_batch_norm_2d directly and must
        # retain native child BatchNorm types for its strict frozen-map checks.
        # Apply the same arithmetic to the remaining normalization call sites
        # (notably ResNet's stem) so lane scheduling does not switch between
        # native BatchNorm and a different batched kernel convention.
        modules = tuple(compiled.named_modules())
        frozen_owner_paths = tuple(
            path
            for path, module in modules
            if isinstance(module, (GatedResidualBlock, FrozenBatchNorm2d))
        )
        for path, module in modules:
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if any(not parent or path.startswith(parent + ".") for parent in frozen_owner_paths):
                continue
            frozen = FrozenBatchNorm2d(module, batch_norm_mode=self.batch_norm_mode)
            if path:
                _replace_submodule(compiled, path, frozen)
            else:
                compiled = frozen
        try:
            graph = _GateLeafTracer().trace(compiled)
            compiled = fx.GraphModule(compiled, graph)
        except Exception as exc:
            raise RuntimeError(
                "FX gate graph compilation failed; canonical execution does not "
                "fall back to forward monkeypatching"
            ) from exc
        if device is not None:
            compiled.to(device)
        compiled.eval()
        gates = {
            layer_id: compiled.get_submodule(module_path)
            for layer_id, module_path in gate_paths.items()
        }
        result = CompiledGateGraph(
            model=compiled,
            gates_by_layer=gates,
            gate_buffer_names={
                layer_id: f"{module_path}.route_gate"
                for layer_id, module_path in gate_paths.items()
            },
            architecture_id=str(architecture_id),
            topology_hash=str(topology_hash),
            route_schema_hash=str(route_schema_hash),
            batch_norm_mode=self.batch_norm_mode, compiler_id=self.compiler_id,
        )
        result.validate_execution_mode()
        result.reset()
        return result


__all__ = [
    "CompiledGateGraph",
    "FrozenBatchNorm2d",
    "GatedConv2d",
    "GatedLinear",
    "GatedResidualBlock",
    "TorchGateGraphCompiler",
]
