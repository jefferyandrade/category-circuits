"""Structural route polarity for matched sign-partition circuit searches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from mechanistic_systems.core.storage import RouteBlockSpec

try:  # Optional for import smoke; required by live model polarity.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("structural route polarity requires PyTorch")


def structural_polarity_array(
    module: Any,
    *,
    route_kind: str,
    expected_shape: tuple[int, ...],
    residual_route_policy: str,
) -> np.ndarray:
    """Return historical structural polarity, assigning exact zeros positive.

    Linear routes use scalar weight sign. Convolutional and downsample
    channel-pair routes use the sign of spatially summed kernel weight. The
    two residual-stream coordinates are unit gates in the historical wrapper,
    so ``unit_gate_nonnegative`` places both in the nonnegative branch.
    """

    _require_torch()
    kind = str(route_kind)
    if kind == "linear_weight":
        if not isinstance(module, nn.Linear):
            raise ValueError("linear structural polarity requires nn.Linear")
        values = module.weight.detach()
    elif kind == "conv_channel_pair":
        if not isinstance(module, nn.Conv2d):
            raise ValueError("convolutional structural polarity requires nn.Conv2d")
        values = module.weight.detach().sum(dim=(2, 3))
    elif kind == "conv_kernel_weight":
        if not isinstance(module, nn.Conv2d):
            raise ValueError("conv kernel-tap structural polarity requires nn.Conv2d")
        values = module.weight.detach()
    elif kind == "residual_stream_gate":
        if residual_route_policy != "unit_gate_nonnegative":
            raise ValueError(
                "residual-stream polarity requires the explicit "
                "unit_gate_nonnegative policy"
            )
        values = torch.ones(expected_shape, dtype=torch.float32)
    else:
        raise ValueError(f"unsupported structural polarity route kind {kind!r}")
    if tuple(values.shape) != tuple(expected_shape):
        raise ValueError("structural polarity shape differs from its route block")
    return (
        torch.where(values >= 0, 0, 1)
        .detach()
        .to(device="cpu", dtype=torch.int8)
        .numpy()
    )


def resolve_residual_route_partition_policy(
    bound_architecture: Any,
    declared_policy: str | None,
) -> str:
    """Resolve residual polarity from the typed route universe."""

    bound_architecture.validate()
    has_residual_routes = any(
        str(block.route_kind) == "residual_stream_gate"
        for block in bound_architecture.circuit_route_blocks
    )
    declared = str(declared_policy or "").strip()
    if has_residual_routes:
        if declared not in {"", "unit_gate_nonnegative"}:
            raise ValueError(
                "residual route coordinates contradict the declared polarity policy"
            )
        return "unit_gate_nonnegative"
    if declared not in {"", "absent"}:
        raise ValueError(
            "a residual polarity policy was declared for a route universe without "
            "residual coordinates"
        )
    return "absent"


class TorchStructuralPolarityProvider:
    """Coordinate-exact partition provider for a bound PyTorch architecture."""

    provider_id = "torch-structural-route-polarity-v1"
    supported_policies = frozenset({"route_coordinate_sign"})
    certified_matching_dimensions = frozenset({"route_coordinate_sign"})

    def __init__(
        self,
        bound_architecture: Any,
        model: Any,
        *,
        residual_route_policy: str,
    ) -> None:
        _require_torch()
        bound_architecture.validate()
        self.bound_architecture = bound_architecture
        self.model = model
        self.residual_route_policy = str(residual_route_policy)
        if self.residual_route_policy not in {"absent", "unit_gate_nonnegative"}:
            raise ValueError("unsupported residual route partition policy")
        self._labels = self._materialize_labels()

    def _module(self, module_path: str) -> Any:
        try:
            return self.model.get_submodule(module_path)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unresolved structural polarity module {module_path!r}") from error

    def _materialize_labels(self) -> np.ndarray:
        blocks = tuple(self.bound_architecture.circuit_route_blocks)
        bindings = tuple(self.bound_architecture.circuit_bindings)
        if len(blocks) != len(bindings):
            raise ValueError("structural polarity bindings and blocks differ")
        labels: list[np.ndarray] = []
        for block, binding in zip(blocks, bindings, strict=True):
            if block.layer_id != binding.layer_id or block.route_kind != binding.route_kind:
                raise ValueError("structural polarity binding semantics differ")
            if block.route_kind == "residual_stream_gate":
                if self.residual_route_policy == "absent":
                    raise ValueError(
                        "the route universe contains residual gates but the plan declares them absent"
                    )
                module = self._module(binding.module_path)
            else:
                module = self._module(binding.module_path)
            labels.append(
                structural_polarity_array(
                    module,
                    route_kind=block.route_kind,
                    expected_shape=block.shape,
                    residual_route_policy=self.residual_route_policy,
                ).reshape(-1)
            )
        return np.concatenate(labels).astype(np.int8, copy=False)

    def labels(
        self,
        *,
        policy_id: str,
        query_id: str,
        query_index: int,
        score_mean: np.ndarray,
        score_magnitude: np.ndarray,
        route_blocks: Sequence[RouteBlockSpec],
        random_seed: int,
    ) -> np.ndarray:
        del query_id, query_index, score_mean, score_magnitude, random_seed
        if policy_id != "route_coordinate_sign":
            raise ValueError("structural polarity provider received another policy")
        if tuple(route_blocks) != tuple(self.bound_architecture.circuit_route_blocks):
            raise ValueError("structural polarity provider route axis differs")
        return self._labels.copy()


__all__ = [
    "TorchStructuralPolarityProvider",
    "resolve_residual_route_partition_policy",
    "structural_polarity_array",
]
