"""The layout of AlexNet's connection ("route") axis in the circuit masks, and the layout types of every network.

- ``LayerSpec``: one pruned layer's segment of the axis.
- ``RouteLayout``: the ordered segments; ``offset`` and ``size`` index the flat route vector.
- ``ALEXNET_KERNEL_WEIGHT_LAYOUT``: AlexNet's layout (``alexnet_conv_kernel_weight``: route axis hash a2443eec…473d,
  56,994,496 routes), without the unpruned classifier (fc8). ``architectures.build_layout`` gives the other networks'.

How a circuit's mask file maps onto the weights (the functions named are the extraction engine's, in ``engine/``):

1. Layer order. ``_canonical_route_declarations`` in ``mechanistic_systems/core/architectures.py`` declares AlexNet's
   layers from input to output, ``features.0, .3, .6, .8, .10, classifier.1, classifier.4, classifier.6``, with
   ``classifier.6`` (fc8) as the fixed readout; ``circuit_route_blocks`` keeps the pruned layers and places them on
   one contiguous axis (``rebase_route_blocks`` in ``core/storage.py``). Shapes come from the module in ``bind``:
   convolutions (out, in/groups, kh, kw), linear layers (out, in).
2. The mask is written on that declared axis, not in the order in which the layers are pruned
   (``_declared_route_axis_offsets``, ``_declared_flat_from_per_layer`` and ``_hard_mask_from_layers`` in
   ``mechanistic_systems/circuits/irg/registered_resnet18_pilot.py``).
3. Within a layer, routes are in C order over the weight's shape (``route_coordinate`` in ``core/storage.py``), so
   ``flat[offset:offset + size].reshape(weight.shape)`` is the layer's weight mask.
4. Bits are packed big-endian, NumPy ``packbits``'s default: the first route is the most significant bit of byte 0
   (``HardMask`` in ``mechanistic_systems/contracts/hard_mask_actions.py``). AlexNet's route count is a multiple of 8,
   so its masks have no padding and every layer starts on a byte.

Binding the engine's ``canonical_torch_architecture_spec("alexnet_conv_kernel_weight")`` to
``torchvision.models.alexnet()`` gives exactly the offsets below, and its ``circuit_route_schema_hash`` is the
receipts' ``route_axis_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALEXNET_KERNEL_WEIGHT_ROUTE_AXIS_HASH = (
    "a2443eecaa2df329532437a5ff9b08ac0b4065ec3c8d4ce2117138e36d96473d"
)
ALEXNET_KERNEL_WEIGHT_ROUTE_COUNT = 56_994_496


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str  # the engine's layer name, e.g. "features.stage1.conv"
    module_name: str  # the torchvision module that owns the weight, e.g. "features.0"
    kind: str  # "conv_kernel_weight" | "linear_weight"
    shape: tuple[int, ...]  # (out, in, kh, kw) or (out, in)
    offset: int
    size: int

    @property
    def stop(self) -> int:
        return int(self.offset) + int(self.size)


@dataclass(frozen=True)
class RouteLayout:
    route_axis_hash: str
    route_count: int
    layers: tuple[LayerSpec, ...]

    def layer(self, layer_id: str) -> LayerSpec:
        for spec in self.layers:
            if spec.layer_id == layer_id:
                return spec
        raise KeyError(layer_id)

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return tuple(spec.layer_id for spec in self.layers)

    def validate(self) -> None:
        """Refuse a layout whose segments are not one contiguous, exactly-sized axis."""

        if not self.layers:
            raise ValueError("route layout has no layers")
        ids = self.layer_ids
        if len(set(ids)) != len(ids):
            raise ValueError("route layout repeats a layer id")
        cursor = 0
        for spec in self.layers:
            if int(spec.offset) != cursor:
                raise ValueError(
                    f"layer {spec.layer_id!r} offset {spec.offset} is not contiguous "
                    f"(expected {cursor})"
                )
            if int(np.prod(spec.shape, dtype=np.int64)) != int(spec.size):
                raise ValueError(f"layer {spec.layer_id!r} size differs from its shape")
            cursor = spec.stop
        if cursor != int(self.route_count):
            raise ValueError(
                f"route layout covers {cursor} routes, declares {self.route_count}"
            )


def _alexnet_layers() -> tuple[LayerSpec, ...]:
    rows = (
        ("features.stage1.conv", "features.0", "conv_kernel_weight", (64, 3, 11, 11)),
        ("features.stage2.conv", "features.3", "conv_kernel_weight", (192, 64, 5, 5)),
        ("features.stage3.conv", "features.6", "conv_kernel_weight", (384, 192, 3, 3)),
        ("features.stage4.conv", "features.8", "conv_kernel_weight", (256, 384, 3, 3)),
        ("features.stage5.conv", "features.10", "conv_kernel_weight", (256, 256, 3, 3)),
        ("readout.hidden1", "classifier.1", "linear_weight", (4096, 9216)),
        ("readout.hidden2", "classifier.4", "linear_weight", (4096, 4096)),
    )
    layers: list[LayerSpec] = []
    offset = 0
    for layer_id, module_name, kind, shape in rows:
        size = int(np.prod(shape, dtype=np.int64))
        layers.append(LayerSpec(layer_id, module_name, kind, shape, offset, size))
        offset += size
    return tuple(layers)


# Offsets (declared axis): stage1 0 | stage2 23,232 | stage3 330,432 | stage4 993,984 |
# stage5 1,878,720 | hidden1 2,468,544 | hidden2 40,217,280 | end 56,994,496.
ALEXNET_KERNEL_WEIGHT_LAYOUT: RouteLayout = RouteLayout(
    route_axis_hash=ALEXNET_KERNEL_WEIGHT_ROUTE_AXIS_HASH,
    route_count=ALEXNET_KERNEL_WEIGHT_ROUTE_COUNT,
    layers=_alexnet_layers(),
)
ALEXNET_KERNEL_WEIGHT_LAYOUT.validate()


__all__ = [
    "ALEXNET_KERNEL_WEIGHT_LAYOUT",
    "ALEXNET_KERNEL_WEIGHT_ROUTE_AXIS_HASH",
    "ALEXNET_KERNEL_WEIGHT_ROUTE_COUNT",
    "LayerSpec",
    "RouteLayout",
]
