"""Science-neutral executable Torch mask payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def tensor_mutation_identity(value: Any) -> tuple[int, int] | None:
    """Host metadata only; unversioned inference tensors cannot be reused."""

    try:
        return id(value), int(value._version)
    except (AttributeError, RuntimeError):
        return None


def resident_lane_stack(lanes, overrides, buffer_names):
    """Use two contiguous generation leases directly as fused gate lanes.

    Dtype/device casts, reordered lanes and historical leases use ordinary
    exact lane preparation. Checking storage ownership costs O(layers).
    """
    if len(lanes) != 2:
        return None
    parent, candidate = lanes
    owner = getattr(parent, "owner", None)
    if (owner is None or getattr(candidate, "owner", None) is not owner
            or getattr(parent, "role", None) != "parent"
            or getattr(candidate, "role", None) != "candidate"
            or not parent._live() or not candidate._live()):
        return None
    if any(overrides[i][name] is not lanes[i][layer]
           for layer, name in buffer_names.items() for i in (0, 1)):
        return None
    owner.lane_view_prepares += 1
    return {name: owner._stacked_gates[layer] for layer, name in buffer_names.items()}


@dataclass(frozen=True)
class HostGateIdentity:
    """Dense-mask premise bound to the tensor created by mask lowering."""

    dense: bool
    tensor_identity: tuple[int, int] | None
    tensor: Any = field(repr=False, compare=False)

    def matches(self, value: Any) -> bool:
        return (
            value is self.tensor
            and self.tensor_identity is not None
            and self.tensor_identity == tensor_mutation_identity(value)
        )


@dataclass(frozen=True)
class CompiledTorchCircuitMask:
    """Device-resident executable mask for repeated stimulus batches."""

    parameter_overrides: Mapping[str, Any]
    residual_gates: Mapping[str, Any]
    gate_values: Mapping[str, Any] | None = None
    float_gate_values: Mapping[str, Any] | None = None
    gate_identities: Mapping[str, HostGateIdentity] | None = None
    # Immutable generation lease. The owner supplies current gates or replays
    # an old generation; a reused buffer is never read under an old identity.
    generation: int | None = None
    generation_owner: Any = field(default=None, repr=False, compare=False)

    def execution_gates(self) -> Mapping[str, Any] | None:
        if self.generation_owner is not None:
            return self.generation_owner.execution_gates(self.generation)
        return self.float_gate_values or self.gate_values

    def mark_read_complete(self) -> None:
        if self.generation_owner is not None:
            self.generation_owner.owner.mark_read_complete()

    def drop_float_gates(self) -> None:
        if self.generation_owner is not None:
            self.generation_owner._history.clear()


__all__ = ["CompiledTorchCircuitMask", "HostGateIdentity", "tensor_mutation_identity"]
