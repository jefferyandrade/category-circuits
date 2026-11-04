"""The pruned layers of each network on the connection ("route") axis of its circuit masks: AlexNet, VGG-11, ResNet-18
and ResNet-50.

Each layout follows the extraction engine's declaration of the network
(``engine/mechanistic_systems/core/architectures.py``: ``_canonical_route_declarations`` and
``_resnet_declarations``; ``circuit_route_blocks`` keeps the pruned layers in declared order on one contiguous axis).
Shapes come from the torchvision modules: convolution weights are (out, in/groups, kh, kw), linear weights (out, in),
and each ResNet ``residual_stream_gate`` is a block of two routes (on the axis, but without a weight; per-layer
analyses skip them).

The route counts and axis hashes below are those the engine gives for the torchvision models:

| family | routes | pruned blocks | route axis hash |
|---|---|---|---|
| alexnet_conv_kernel_weight | 56,994,496 | 7 | a2443eec... |
| vgg11_conv_kernel_weight | 128,679,936 | 8 (features.6..18, classifier.0 and .3; not features.0, .3) | 6c149ec7... |
| resnet18_conv_kernel_weight | 11,166,928 | 28 (20 convolutions, 3 of them projections, and 8 gates) | 90dc0238... |
| resnet50_conv_kernel_weight | 23,454,944 | 69 (53 convolutions, 4 of them projections, and 16 gates) | 253a2e8b... |

The ResNets' layer offsets are not multiples of 8 (each gate adds 2 routes), so their masks are read by bit offset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from route_layout import ALEXNET_KERNEL_WEIGHT_LAYOUT, LayerSpec, RouteLayout

CHECKPOINT_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

EXPECTED = {
    "alexnet_conv_kernel_weight": (56_994_496, "a2443eecaa2df329532437a5ff9b08ac0b4065ec3c8d4ce2117138e36d96473d"),
    "vgg11_conv_kernel_weight": (128_679_936, "6c149ec7ed47884faea034b965b5284d41c7aea50277d5cce65c4e4f5d5e2b97"),
    "resnet18_conv_kernel_weight": (11_166_928, "90dc0238cb540f50a6d3e36eccd441db9b80a5db2478b8fa8b6e8dc8a6701b8d"),
    "resnet50_conv_kernel_weight": (23_454_944, "253a2e8bb81a9fdf8ac65e4e7fe8a7f6d9bdf13b7058bded66661487ffa4d7d7"),
}

# The torchvision checkpoints, found in CHECKPOINT_DIR (nothing is downloaded).
CHECKPOINTS = {
    "alexnet": "alexnet-owt-7be5be79.pth",
    "vgg11": "vgg11-8a719046.pth",
    "resnet18": "resnet18-f37072fd.pth",
    "resnet50_v1": "resnet50-0676ba61.pth",  # IMAGENET1K_V1, the ResNet-50 of the paper
    "resnet50_v2": "resnet50-11ad3fa6.pth",  # IMAGENET1K_V2, the supplement's backup results
}

_VGG11_ROWS = (
    ("features.stage3.conv1", "features.6", "conv_kernel_weight"),
    ("features.stage3.conv2", "features.8", "conv_kernel_weight"),
    ("features.stage4.conv1", "features.11", "conv_kernel_weight"),
    ("features.stage4.conv2", "features.13", "conv_kernel_weight"),
    ("features.stage5.conv1", "features.16", "conv_kernel_weight"),
    ("features.stage5.conv2", "features.18", "conv_kernel_weight"),
    ("readout.hidden1", "classifier.0", "linear_weight"),
    ("readout.hidden2", "classifier.3", "linear_weight"),
)


def _resnet_rows(block_counts: tuple[int, ...], convs_per_block: int) -> tuple:
    rows = [("stem.conv", "conv1", "conv_kernel_weight")]
    for si, n_blocks in enumerate(block_counts, start=1):
        for bi in range(n_blocks):
            bid = f"stage{si}.block{bi}"
            for ci in range(1, convs_per_block + 1):
                rows.append((f"{bid}.conv{ci}", f"layer{si}.{bi}.conv{ci}", "conv_kernel_weight"))
            if bi == 0 and (si > 1 or convs_per_block == 3):
                rows.append((f"{bid}.projection", f"layer{si}.{bi}.downsample.0", "conv_kernel_weight"))
            rows.append((f"{bid}.residual_stream", f"layer{si}.{bi}", "residual_stream_gate"))
    return tuple(rows)


FAMILIES = {
    "vgg11_conv_kernel_weight": ("vgg11", _VGG11_ROWS),
    "resnet18_conv_kernel_weight": ("resnet18", _resnet_rows((2, 2, 2, 2), 2)),
    "resnet50_conv_kernel_weight": ("resnet50", _resnet_rows((3, 4, 6, 3), 3)),
}


def build_layout(family: str, model=None) -> RouteLayout:
    """The prunable kernel-weight route layout of ``family`` (shapes from torchvision)."""

    if family == "alexnet_conv_kernel_weight":
        return ALEXNET_KERNEL_WEIGHT_LAYOUT
    ctor_name, rows = FAMILIES[family]
    if model is None:
        import torchvision

        model = getattr(torchvision.models, ctor_name)(weights=None)
    specs, off = [], 0
    for layer_id, module_path, kind in rows:
        module = model.get_submodule(module_path)
        if kind == "conv_kernel_weight":
            shape = (int(module.out_channels), int(module.in_channels // module.groups),
                     int(module.kernel_size[0]), int(module.kernel_size[1]))
        elif kind == "linear_weight":
            shape = (int(module.out_features), int(module.in_features))
        else:
            shape = (2,)
        size = int(np.prod(shape))
        specs.append(LayerSpec(layer_id, module_path, kind, shape, off, size))
        off += size
    count, axis_hash = EXPECTED[family]
    layout = RouteLayout(axis_hash, off, tuple(specs))
    layout.validate()
    if off != count:
        raise ValueError(f"{family}: {off} routes, the engine's axis has {count}")
    return layout


def checkpoint_path(name: str) -> Path | None:
    """The local torchvision checkpoint of ``name``, or None if it is absent."""

    p = CHECKPOINT_DIR / CHECKPOINTS[name]
    return p if p.exists() else None


__all__ = [
    "CHECKPOINTS",
    "EXPECTED",
    "FAMILIES",
    "build_layout",
    "checkpoint_path",
]
