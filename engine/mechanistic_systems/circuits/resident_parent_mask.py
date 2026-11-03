"""Resident float gates, generation leases and O(R) route updates.

The walk owns proposals and actions. Parent and candidate buffers stay in
step without a whole-mask copy at proposal time. Historical generations are
compact replay recipes, materialized only by historical consumers.
"""
from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

import weakref
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from mechanistic_systems.backends.torch_masks import (
    CompiledTorchCircuitMask, HostGateIdentity, tensor_mutation_identity,
)

DEFAULT_MEMORY_BUDGET = 2 * 1024**3


def resident_parent_memory_budget_bytes(device):
    """Available process budget; CPU tests never query a CUDA device."""
    from mechanistic_systems.measurement.gpu_runtime import process_device_memory_budget_bytes

    device = torch.device(device)
    if device.type != "cuda":
        return process_device_memory_budget_bytes(DEFAULT_MEMORY_BUDGET)
    free, total = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    return min(
        cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free),
        max(0, process_device_memory_budget_bytes(int(total)) - allocated),
    )


def _release_storage(ledger, size):
    ledger["bytes"] -= size


def _release_upload(staging, key):
    staging.pop(key, None)


def same_execution_device(actual, requested):
    """An unspecified CUDA index admits the tensor's resolved device index."""
    requested = torch.device(requested)
    return actual.type == requested.type and (
        actual.type == "cpu" or requested.index is None or actual.index == requested.index
    )


class ResidentMaskPrecondition(ValueError):
    """A resident operation's inputs or generation are invalid."""


@dataclass(frozen=True)
class RouteOrderLocator:
    order_key: tuple[str, str]
    start: int
    stop: int
    parent_generation: int


class _GenerationGates(Mapping):
    """A lease over live storage or an independently replayed generation."""

    def __init__(self, owner, role, epoch, accepted, delta):
        self.owner, self.role, self.epoch = owner, role, epoch
        self.accepted, self.delta = accepted, delta
        self._history = {}

    def __iter__(self):
        return iter(self.owner._initial)

    def __len__(self):
        return len(self.owner._initial)

    def _live(self):
        return self.epoch == getattr(self.owner, f"_{self.role}_epoch")

    def __getitem__(self, layer):
        if self._live():
            return getattr(self.owner, f"_{self.role}_gates")[layer]
        if layer not in self._history:
            self._history[layer] = self.owner._replay_layer(layer, self.accepted, self.delta)
        return self._history[layer]

    def execution_gates(self, generation):
        if generation != self.epoch:
            raise ResidentMaskPrecondition("generation_mismatch")
        if self._live():
            return self
        return {layer: self[layer] for layer in self}


class _ResidualGates(Mapping):
    def __init__(self, gates, layers):
        self.gates, self.layers = gates, tuple(layers)

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, layer):
        if layer not in self.layers:
            raise KeyError(layer)
        return self.gates[layer]


class ResidentParentMask:
    """Two float buffers, O(R) scatters, and immutable generation recipes.

    Propose clears scratch; refusal restores scratch; accept clears parent.
    Completion events order buffer reuse after execution, including consumers
    on a different CUDA stream. No gate array is downloaded for transitions.
    """

    def __init__(self, parent_hash, compiled, *, memory_budget_bytes=None,
                 reserved_bytes=0):
        if compiled.parameter_overrides:
            raise ResidentMaskPrecondition("parameter_overrides")
        source = compiled.execution_gates()
        if not source:
            raise ResidentMaskPrecondition("missing_route_gates")
        devices = {value.device for value in source.values()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ResidentMaskPrecondition("gate_device")
        self.device = next(iter(devices))
        if any(not value.is_contiguous() for value in source.values()):
            raise ResidentMaskPrecondition("gate_layout")
        if any(value.numel() > np.iinfo(np.int32).max for value in source.values()):
            raise ResidentMaskPrecondition("route_axis_int32_capacity")
        self.route_count = sum(value.numel() for value in source.values())
        self.memory_budget_bytes = (resident_parent_memory_budget_bytes(self.device)
                                    if memory_budget_bytes is None else int(memory_budget_bytes))
        self.reserved_bytes = int(reserved_bytes)
        # Only the two float lanes are allocated here. The compiler owns the
        # dense anchor; orders and explicit deltas are charged when allocated.
        self.lane_bytes = 8 * self.route_count
        self._order_bytes = 0
        self._index_storage = {"bytes": 0}
        self._replay_storage = {"bytes": 0}
        self._execution_weight_states = {}
        self._execution_weight_bytes = 0
        self._check_memory_budget(0)
        if self.device.type == "cuda":
            from mechanistic_systems.inference.device_bce import defer_device_check
            failed = torch.stack(tuple(((v != 0) & (v != 1)).any() for v in source.values())).any()
            if not defer_device_check(failed,lambda:ResidentMaskPrecondition("nonbinary_gates")):
                torch._assert_async(~failed,"nonbinary_gates")
            identities = compiled.gate_identities
            if identities is not None and any(
                    layer not in identities or not identities[layer].matches(value)
                    for layer,value in source.items()):
                raise ResidentMaskPrecondition("gate_identity")
            self._dense_layers = ({layer for layer in source if identities[layer].dense}
                                  if identities is not None else set())
        else:
            if any(not bool(torch.all((v == 0) | (v == 1))) for v in source.values()):
                raise ResidentMaskPrecondition("nonbinary_gates")
            self._dense_layers = {k for k, v in source.items() if bool(torch.all(v == 1))}
        self._initial = dict(source)  # immutable compiler anchor
        with torch.inference_mode(False):
            self._stacked_gates = {k: torch.empty((2, *v.shape), dtype=torch.float32,
                                                 device=self.device).copy_(v)
                                   for k, v in source.items()}
            self._parent_gates = {k: v[0] for k, v in self._stacked_gates.items()}
            self._candidate_gates = {k: v[1] for k, v in self._stacked_gates.items()}
        self._record_buffer_versions()
        self._residual_layers = tuple(compiled.residual_gates)
        self._has_identities = compiled.gate_identities is not None
        self._parent_epoch = self._candidate_epoch = 0
        self.generation = 0
        self._accepted = []
        self._orders = {}
        self._order_versions = {}
        self._upload_staging = {}
        self._indices = {}
        self._read_events = []
        self.parent_hash = str(parent_hash)
        self.parent = self._lease("parent")
        self.candidate_hash = None
        self.candidate = None
        self.proposals = self.accepts = self.refusals = 0
        self.staged_index_bytes = self.order_upload_bytes = self.order_upload_calls = 0
        self.scatter_routes = self.historical_replays = self.final_exports = 0
        self.lane_view_prepares = 0
        self.historical_generation_prepares = 0
        self._exported = None

    @property
    def allocated_bytes(self):
        return (self.lane_bytes + self._order_bytes + self._index_storage["bytes"]
                + self._replay_storage["bytes"] + self._execution_weight_bytes)

    @property
    def planned_bytes(self):
        return self.reserved_bytes + self.allocated_bytes

    def _check_memory_budget(self, additional_bytes, reason="memory_budget"):
        if (self.reserved_bytes < 0 or self.memory_budget_bytes < 0
                or self.planned_bytes + int(additional_bytes) > self.memory_budget_bytes):
            raise ResidentMaskPrecondition(reason)

    def _charge_storage(self, value, ledger):
        size = value.numel() * value.element_size()
        ledger["bytes"] += size
        # A refused lease may remain in the bounded cache. Charge its tensor
        # until its last lease releases it, rather than counting upload bytes.
        weakref.finalize(value, _release_storage, ledger, size)

    def _lease(self, role, delta=None):
        epoch = getattr(self, f"_{role}_epoch")
        gates = _GenerationGates(self, role, epoch, len(self._accepted), delta or {})
        identities = None
        if self._has_identities:
            identities = {layer: HostGateIdentity(
                dense=layer in self._dense_layers and (role == "parent" or layer not in self._indices),
                tensor_identity=tensor_mutation_identity(value), tensor=value,
            ) for layer, value in getattr(self, f"_{role}_gates").items()}
        return CompiledTorchCircuitMask(
            {}, _ResidualGates(gates, self._residual_layers), gates, gates,
            identities, epoch, gates,
        )

    def _replay_layer(self, layer, accepted, delta):
        initial = self._initial[layer]
        self._check_memory_budget(4 * initial.numel(), "replay_memory_budget")
        self.historical_replays += 1
        value = torch.empty(initial.shape, dtype=torch.float32, device=self.device).copy_(initial)
        self._charge_storage(value, self._replay_storage)
        for removal in self._accepted[:accepted]:
            if layer in removal:
                value.reshape(-1).index_fill_(0, removal[layer].long(), 0.)
        if layer in delta:
            value.reshape(-1).index_fill_(0, delta[layer].long(), 0.)
        return value

    def mark_read_complete(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._read_events.append(event)

    def _wait_for_reads(self):
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for event in self._read_events:
                stream.wait_event(event)
        self._read_events.clear()

    def upload_order(self, key, layer, order):
        if key in self._orders:
            self.check_order_versions()
            return
        if layer not in self._initial:
            raise ResidentMaskPrecondition("order_layer")
        values = np.asarray(order)
        if (values.ndim != 1 or values.dtype.kind not in "iu"
                or (values.size and (np.any(values < 0)
                    or np.any(values >= self._initial[layer].numel())))):
            raise ResidentMaskPrecondition("route_order")
        new_bytes = values.size * 4
        if self.order_upload_bytes + new_bytes > 4 * self.route_count:
            raise ResidentMaskPrecondition("order_memory_budget")
        self._check_memory_budget(new_bytes, "order_memory_budget")
        uploaded = self._upload_indices(values)
        self._orders[key] = uploaded
        self._order_versions[key] = tensor_mutation_identity(uploaded)
        self.order_upload_calls += 1
        self.order_upload_bytes += new_bytes
        self._order_bytes += new_bytes

    def _upload_indices(self, values):
        array = np.array(values, dtype=np.int32, copy=True)
        with torch.inference_mode(False):
            if self.device.type != "cuda":
                return torch.tensor(array, dtype=torch.int32, device=self.device)
            staging = torch.empty(array.shape, dtype=torch.int32, device="cpu", pin_memory=True)
            staging.copy_(torch.from_numpy(array))
            destination = staging.to(device=self.device, non_blocking=True)
        key = id(destination)
        self._upload_staging[key] = staging
        weakref.finalize(destination, _release_upload, self._upload_staging, key)
        return destination

    def _record_buffer_versions(self):
        # Only this object's scatters may write the lanes (O(layers) metadata).
        self._buffer_versions = {k: v._version for k, v in self._stacked_gates.items()}

    def check_order_versions(self):
        if any(v._version != self._buffer_versions[k] for k, v in self._stacked_gates.items()):
            raise ResidentMaskPrecondition("gate_buffer_mutation")
        if any(tensor_mutation_identity(order) != self._order_versions[key]
               for key, order in self._orders.items()):
            raise ResidentMaskPrecondition("route_order_mutation")

    def propose(self, candidate_hash, removed_by_layer, *, expected_counts=None,
                locators=()):
        self.check_order_versions()
        if self.candidate is not None:
            if self.candidate_hash == str(candidate_hash):
                return self.candidate
            raise ResidentMaskPrecondition("pending_candidate_conflict")
        indices = {}
        arrays = {}
        for layer, removed in removed_by_layer.items():
            if layer not in self._parent_gates:
                raise ResidentMaskPrecondition("delta_layer")
            array = np.asarray(removed)
            if (array.ndim != 1 or array.dtype.kind not in "iu"
                    or (array.size and (array[0] < 0
                        or array[-1] >= self._parent_gates[layer].numel()
                        or np.any(array[1:] <= array[:-1])))):
                raise ResidentMaskPrecondition("delta_indices")
            if array.size:
                arrays[layer] = array
        self._check_memory_budget(sum(array.size * 4 for array in arrays.values()),
                                  "delta_memory_budget")
        for layer, array in arrays.items():
            indices[layer] = self._upload_indices(array)
            self._charge_storage(indices[layer], self._index_storage)
        for locator in locators:
            layer = locator.order_key[0]
            order = self._orders.get(locator.order_key)
            if locator.parent_generation != self.generation:
                raise ResidentMaskPrecondition("locator_generation")
            if (order is None or locator.start < 0 or locator.stop < locator.start
                    or locator.stop > order.numel() or layer in indices):
                raise ResidentMaskPrecondition("order_locator")
            if locator.stop > locator.start:
                indices[layer] = order[locator.start:locator.stop]
        if expected_counts is not None and any(
            (indices[k].numel() if k in indices else 0) != int(expected_counts.get(k, 0))
            for k in set(indices) | set(expected_counts)
        ):
            raise ResidentMaskPrecondition("delta_count")
        # The proposal owns its route delta. Scatters stay on the device;
        # transitions do not reconstruct membership from a packed mask.
        self._wait_for_reads()
        for layer, index in indices.items():
            self._candidate_gates[layer].reshape(-1).index_fill_(0, index.long(), 0.)
        self._record_buffer_versions()
        self._candidate_epoch += 1
        self.candidate_hash = str(candidate_hash)
        self._indices = indices
        for weights in self._execution_weight_states.values():
            weights.propose()
        self.candidate = self._lease("candidate", indices)
        self.parent = self._lease("parent")
        self.proposals += 1
        self.scatter_routes += sum(i.numel() for i in indices.values())
        self.staged_index_bytes += sum(i.numel() * 4 for k, i in indices.items()
                                       if k in removed_by_layer)
        return self.candidate

    def resolve(self, accept):
        if self.candidate is None:
            raise ResidentMaskPrecondition("unknown_transition")
        self.check_order_versions()
        self._wait_for_reads()
        target = self._parent_gates if accept else self._candidate_gates
        for layer, index in self._indices.items():
            target[layer].reshape(-1).index_fill_(0, index.long(), 0. if accept else 1.)
        for weights in self._execution_weight_states.values():
            weights.resolve(accept)
        self._record_buffer_versions()
        self.scatter_routes += sum(i.numel() for i in self._indices.values())
        if accept:
            self._accepted.append(self._indices)
            self._dense_layers.difference_update(self._indices)
            self.parent_hash = self.candidate_hash
            self.generation += 1
            self._parent_epoch += 1
            self.parent = self._lease("parent")
            self.accepts += 1
            self._exported = None
        else:
            self.refusals += 1
        self._candidate_epoch += 1
        self.candidate_hash = self.candidate = None
        self._indices = {}
        self.parent = self._lease("parent")

    def export_final(self):
        """Bulk output boundary: one download, cached for repeated consumers."""
        if self.candidate is not None:
            raise ResidentMaskPrecondition("pending_final_export")
        if self._exported is None:
            self._exported = {k: v.detach().to(device="cpu", dtype=torch.bool).numpy().copy()
                              for k, v in self._parent_gates.items()}
            for value in self._exported.values():
                value.setflags(write=False)
            self.final_exports += 1
        return dict(self._exported)

    def rebuild_intermediate_masks(self):
        """Bulk reconstruction from the immutable anchor and accepted deltas."""
        arrays = {k: v.detach().to(device="cpu", dtype=torch.bool).numpy().copy()
                  for k, v in self._initial.items()}
        result = [{k: v.copy() for k, v in arrays.items()}]
        for delta in self._accepted:
            for layer, indices in delta.items():
                arrays[layer].reshape(-1)[indices.to("cpu").numpy()] = False
            result.append({k: v.copy() for k, v in arrays.items()})
        return result
