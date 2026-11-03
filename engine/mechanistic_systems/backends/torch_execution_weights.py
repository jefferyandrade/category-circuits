"""Prepared effective weights, with in-place resident-parent route updates."""

from collections import OrderedDict
import weakref

import torch

from .torch_gate_graph import GatedConv2d, GatedLinear
from .torch_masks import tensor_mutation_identity
from .torch_uint8_stem import STEM_CHANNEL_MULTIPLE, pad_stem_weight, prepare_stem_components


def _buffers(module, gate):
    gate = gate[:, :, None, None] if isinstance(module, GatedConv2d) and module.gate_granularity == "channel_pair" else gate
    weight = module.module.weight.detach() * gate.detach()
    if isinstance(module, GatedConv2d) and weight.device.type == "cuda":
        weight = weight.contiguous(memory_format=torch.channels_last)
    result = {"execution_weight": weight}
    _fold_stem(module, result)
    return result


def _fold_stem(module, buffers):
    if not hasattr(module, "uint8_decode_scale"):
        return
    from mechanistic_systems.measurement.native_panel_data_plane import compute_uint8_folded_stem

    folded = compute_uint8_folded_stem(
        buffers["execution_weight"], module.module.bias,
        decode_scale=module.uint8_decode_scale, decode_bias=module.uint8_decode_bias,
        detach=False,
    )
    # Decode constants belong to preprocessing, and are shared across lanes.
    # Keeping the edge value on the module avoids batching this constant.
    if not module.uint8_decode_pad.numel():
        module.uint8_decode_pad = folded["pad_value"].reshape(-1)
    values = {"execution_stem_weight": pad_stem_weight(folded["weight"]),
              "execution_stem_bias": folded["bias"].detach()}
    if (values["execution_stem_weight"].device.type == "cuda"
            and values["execution_stem_weight"].dtype == torch.float32
            and module.module.groups == 1):
        values.update(prepare_stem_components(values["execution_stem_weight"],
                                              module.uint8_decode_pad))
    for name, value in values.items():
        if name in buffers:
            buffers[name].copy_(value)
        else:
            buffers[name] = value


def _bytes(buffers):
    return sum(value.numel() * value.element_size() for layer in buffers.values() for value in layer.values())


class ResidentExecutionWeights:
    """Persistent parent/candidate weights, changed only at proposal/resolution.

    The mask owner invokes proposal/resolution after ordering its readers.
    No device gate comparison, weight download or per-forward multiplication
    is needed. Nonresident/historical masks use ordinary prepared weights.
    """

    def __init__(self, owner, modules):
        # The resident owns this state. A strong reverse edge would retain
        # both float lanes and effective weights until cyclic GC at reanchor.
        self.owner, self.modules = weakref.proxy(owner), modules
        # Check the full planned allocation before constructing the buffers.
        planned = 2 * sum(module.module.weight.numel() * module.module.weight.element_size()
                      for module in modules.values())
        for module in modules.values():
            if hasattr(module, "uint8_decode_scale"):
                w = module.module.weight
                padded = ((w.shape[1] + STEM_CHANNEL_MULTIPLE - 1)
                          // STEM_CHANNEL_MULTIPLE) * STEM_CHANNEL_MULTIPLE
                planned += 2 * (w.shape[0] * padded * w.shape[2] * w.shape[3]
                                + w.shape[0]) * 4
                if w.device.type == "cuda" and w.dtype == torch.float32 and module.module.groups == 1:
                    # Two packed BF16 components retain FP32 weight accuracy.
                    # Include both parent/candidate lanes before allocating.
                    planned += 2 * (w.shape[0] * padded * w.shape[2] * w.shape[3] * 4
                                    + w.shape[1] * 2)
        owner._check_memory_budget(planned, "effective_weight_memory_budget")
        self.stacked, self.parent, self.candidate = {}, {}, {}
        with torch.inference_mode(False), torch.no_grad():
            for layer, module in modules.items():
                values = _buffers(module, owner._parent_gates[layer])
                stacks = {}
                for name, value in values.items():
                    if value.ndim == 4 and value.device.type == "cuda":
                        lanes = torch.empty((2 * value.shape[0], *value.shape[1:]),
                                            dtype=value.dtype, device=value.device,
                                            memory_format=torch.channels_last).view(2, *value.shape)
                    else:
                        lanes = value.new_empty((2, *value.shape))
                    stacks[name] = lanes.copy_(value)
                self.stacked[layer] = stacks
                self.parent[layer] = {name: value[0] for name, value in stacks.items()}
                self.candidate[layer] = {name: value[1] for name, value in stacks.items()}
        owner._execution_weight_bytes += _bytes(self.stacked)
        # A candidate may already have been proposed before the first forward.
        self.propose()

    def propose(self):
        self._update(self.candidate, restore=False)

    def buffers(self, layer, role):
        if role == "parent" or layer not in self.owner._indices:
            return self.parent[layer]
        return self.candidate[layer]

    def resolve(self, accept):
        self._update(self.parent if accept else self.candidate, restore=not accept)

    def _update(self, buffers, *, restore):
        with torch.no_grad():
            for layer, index in self.owner._indices.items():
                if layer not in self.modules:
                    continue
                module = self.modules[layer]
                weight = buffers[layer]["execution_weight"]
                source = module.module.weight.detach()
                index = index.long()
                if isinstance(module, GatedLinear):
                    if restore:
                        weight.view(-1).index_copy_(0, index, source.reshape(-1)[index])
                    else:
                        weight.view(-1).index_fill_(0, index, 0.)
                elif module.gate_granularity == "channel_pair":
                    o, c = index // weight.shape[1], index % weight.shape[1]
                    weight[o, c] = source[o, c] if restore else 0.
                else:
                    o = index // (weight.shape[1] * weight.shape[2] * weight.shape[3])
                    c = (index // (weight.shape[2] * weight.shape[3])) % weight.shape[1]
                    h = (index // weight.shape[3]) % weight.shape[2]
                    w = index % weight.shape[3]
                    weight[o, c, h, w] = source[o, c, h, w] if restore else 0.
                _fold_stem(module, buffers[layer])

    def stacked_buffers(self, graph):
        return {f"{graph.gate_buffer_names[layer].removesuffix('.route_gate')}.{name}": value
                for layer, buffers in self.stacked.items() for name, value in buffers.items()}


class ExecutionWeightCache:
    """Small per-layer cache for immutable lowered masks, plus live parent state."""

    def __init__(self):
        self._layers = {}

    def for_lanes(self, graph, lanes, trusted):
        extras = []
        modules = {layer: module for layer, module in graph.gates_by_layer.items()
                   if isinstance(module, (GatedConv2d, GatedLinear))}
        if not trusted or any(getattr(value, "requires_grad", False)
                              for lane in lanes for value in lane.values()):
            return [{} for _ in lanes]
        for lane in lanes:
            extra = {}
            owner = getattr(lane, "owner", None)
            live = owner is not None and lane._live()
            state = None
            if trusted and live:
                key = id(graph.model)
                state = owner._execution_weight_states.get(key)
                if state is None:
                    state = ResidentExecutionWeights(owner, modules)
                    owner._execution_weight_states[key] = state
            for layer, module in modules.items():
                gate = torch.as_tensor(lane.get(layer, module.route_gate),
                                       dtype=module.route_gate.dtype, device=module.route_gate.device)
                if state is not None:
                    values = state.buffers(layer, lane.role)
                else:
                    identity = tensor_mutation_identity(gate)
                    constant_values = (module.module.weight, module.module.bias,
                                       getattr(module, "uint8_decode_scale", None),
                                       getattr(module, "uint8_decode_bias", None))
                    constants = tuple(tensor_mutation_identity(value) for value in constant_values)
                    key = (identity, constants)
                    cache = self._layers.setdefault(id(module), OrderedDict())
                    entry = cache.get(key) if identity is not None else None
                    if entry is not None and entry[0] is gate:
                        values = entry[1]
                        cache.move_to_end(key)
                    else:
                        with torch.no_grad():
                            values = _buffers(module, gate)
                        if identity is not None:
                            cache[key] = (gate, values, constant_values)
                            while len(cache) > 2:
                                cache.popitem(last=False)
                path = graph.gate_buffer_names[layer].removesuffix(".route_gate")
                extra.update({f"{path}.{name}": value for name, value in values.items()})
            extras.append(extra)
        return extras


def stack_execution_buffers(extras):
    """Shared weights use a lane view, rather than two physical copies."""
    if not extras:
        return {}
    result = {}
    for name, first in extras[0].items():
        values = [extra[name] for extra in extras]
        result[name] = (first.unsqueeze(0).expand(len(values), *first.shape)
                        if all(value is first for value in values)
                        else torch.stack(values))
    return result


def remap_execution_buffers(graph, model, extras):
    """FX partitions keep leaf modules; only their qualified names change."""
    paths = {id(module): graph.gate_buffer_names[layer].removesuffix(".route_gate")
             for layer, module in graph.gates_by_layer.items()}
    result = {}
    for path, module in model.named_modules():
        original = paths.get(id(module))
        if original is None:
            continue
        for name in module._buffers:
            key = f"{original}.{name}"
            if name.startswith("execution_") and key in extras:
                result[f"{path}.{name}"] = extras[key]
    return result
