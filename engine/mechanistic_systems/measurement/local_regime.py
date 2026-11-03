"""Ephemeral device-side activation-regime accumulation.

This module is an optional tee for the canonical target-batched convolutional
reducer.  It consumes a route-resolved local contribution chunk while that
chunk is already resident on the accelerator and emits only route-axis
sufficient statistics.  It never changes, reconstructs, or substitutes for
the canonical local/global/loss score heads.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:  # Import smoke remains available without torch.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


REGIME_STATISTIC_NAMES = (
    "observation_mass",
    "local_active_mass",
    "baseline_open_mass",
    "regime_preserving_inhibitory_mass",
    "suppressive_gate_opening_mass",
    "enabling_gate_closing_mass",
    "still_closed_active_mass",
    "crossing_risk_sum",
    "crossing_risk_max",
    "regime_preserving_inhibitory_effect_mass",
    "gate_opening_effect_mass",
    "gate_closing_effect_mass",
)

SMOOTH_TRANSMISSION_STATISTIC_NAMES = (
    "observation_mass",
    "local_active_mass",
    "baseline_gain_sum",
    "counterfactual_gain_sum",
    "absolute_gain_change_sum",
    "absolute_gain_change_max",
    "transmission_modulation_energy",
    "activation_effect_mass",
    "regime_preserving_inhibitory_effect_mass",
)

SUPPORTED_ACTIVATION_KINDS = frozenset(
    {"relu", "gelu", "silu", "sigmoid", "tanh", "softplus"}
)


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("local regime accumulation requires PyTorch")


@dataclass(frozen=True)
class LocalRegimeCaptureSpec:
    """Architecture-bound destination geometry for one route block."""

    layer_id: str
    route_scale: Any
    affine_offset: Any
    capture_source: str
    geometry_hash: str
    residual_block_path: str | None = None
    activation_kind: str = "relu"
    local_active_tolerance: float = 0.0
    boundary_epsilon: float = 1e-8
    route_chunk_size: int = 16

    def __post_init__(self) -> None:
        scale = tuple(float(value) for value in self.route_scale)
        offset = tuple(float(value) for value in self.affine_offset)
        if (
            not self.layer_id
            or not scale
            or len(scale) != len(offset)
            or not all(math.isfinite(value) for value in (*scale, *offset))
            or self.capture_source
            not in {"affine_bound_output", "residual_output_preactivation"}
            or not self.geometry_hash
            or (
                self.capture_source == "residual_output_preactivation"
                and not self.residual_block_path
            )
            or self.activation_kind not in SUPPORTED_ACTIVATION_KINDS
            or not math.isfinite(float(self.local_active_tolerance))
            or float(self.local_active_tolerance) < 0.0
            or not math.isfinite(float(self.boundary_epsilon))
            or float(self.boundary_epsilon) <= 0.0
            or int(self.route_chunk_size) <= 0
        ):
            raise ValueError("local regime capture specification is invalid")
        object.__setattr__(self, "route_scale", scale)
        object.__setattr__(self, "affine_offset", offset)
        object.__setattr__(self, "route_chunk_size", int(self.route_chunk_size))


class TorchReluRegimeTee:
    """Accumulate exact singleton ReLU counterfactuals in route chunks."""

    def __init__(
        self,
        *,
        baseline_preactivation: Any,
        route_scale: Any,
        route_shape: tuple[int, int],
        geometry_hash: str,
        local_active_tolerance: float = 0.0,
        boundary_epsilon: float = 1e-8,
        route_chunk_size: int = 16,
    ) -> None:
        _require_torch()
        baseline = torch.as_tensor(baseline_preactivation)
        scale = torch.as_tensor(
            route_scale, dtype=torch.float32, device=baseline.device
        ).reshape(-1)
        outputs, inputs = map(int, route_shape)
        if (
            baseline.ndim != 4
            or tuple(baseline.shape[1:2]) != (outputs,)
            or scale.shape != (outputs,)
            or not bool(torch.isfinite(baseline).all().item())
            or not bool(torch.isfinite(scale).all().item())
            or not geometry_hash
            or float(local_active_tolerance) < 0.0
            or float(boundary_epsilon) <= 0.0
            or int(route_chunk_size) <= 0
        ):
            raise ValueError("ReLU regime tee axes or identity are invalid")
        self.baseline = baseline.detach().to(torch.float32)
        self.route_scale = scale
        self.route_shape = (outputs, inputs)
        self.geometry_hash = str(geometry_hash)
        self.local_active_tolerance = float(local_active_tolerance)
        self.boundary_epsilon = float(boundary_epsilon)
        self.route_chunk_size = int(route_chunk_size)
        self.statistics = {
            name: torch.zeros(
                self.route_shape,
                dtype=torch.float64,
                device=baseline.device,
            )
            for name in REGIME_STATISTIC_NAMES
        }

    def update(self, batch_start: int, route_local_field: Any) -> None:
        """Consume ``[batch,output,input,height,width]`` raw route drive."""

        _require_torch()
        field = torch.as_tensor(
            route_local_field, dtype=torch.float32, device=self.baseline.device
        )
        if (
            field.ndim != 5
            or tuple(field.shape[1:3]) != self.route_shape
            or tuple(field.shape[-2:]) != tuple(self.baseline.shape[-2:])
            or int(batch_start) < 0
            or int(batch_start) + int(field.shape[0]) > int(self.baseline.shape[0])
            or not bool(torch.isfinite(field).all().item())
        ):
            raise ValueError("local regime tee received an incompatible route field")
        start = int(batch_start)
        stop = start + int(field.shape[0])
        baseline = self.baseline[start:stop].unsqueeze(2)
        tolerance = self.local_active_tolerance
        epsilon = self.boundary_epsilon
        outputs, inputs = self.route_shape
        reduction_axes = (0, 3, 4)
        observation_per_route = float(
            int(field.shape[0]) * int(field.shape[-2]) * int(field.shape[-1])
        )
        for route_start in range(0, inputs, self.route_chunk_size):
            route_stop = min(inputs, route_start + self.route_chunk_size)
            contribution = (
                field[:, :, route_start:route_stop]
                * self.route_scale.reshape(1, outputs, 1, 1, 1)
            )
            counterfactual = baseline - contribution
            active = contribution.abs() > tolerance
            open_before = baseline > 0.0
            open_after = counterfactual > 0.0
            inhibitory = contribution < -tolerance
            regime_preserving_inhibitory = inhibitory & open_before & open_after
            gate_opening = (~open_before) & open_after
            gate_closing = open_before & (~open_after)
            still_closed_active = (~open_before) & (~open_after) & active
            activation_change = torch.relu(counterfactual) - torch.relu(baseline)
            crossing_risk = contribution.abs() / (baseline.abs() + epsilon)
            destination = (slice(None), slice(route_start, route_stop))
            self.statistics["observation_mass"][destination].add_(
                observation_per_route
            )
            self.statistics["local_active_mass"][destination].add_(
                active.sum(dim=reduction_axes, dtype=torch.float64)
            )
            self.statistics["baseline_open_mass"][destination].add_(
                open_before.expand_as(contribution).sum(
                    dim=reduction_axes, dtype=torch.float64
                )
            )
            for name, mask in (
                (
                    "regime_preserving_inhibitory_mass",
                    regime_preserving_inhibitory,
                ),
                ("suppressive_gate_opening_mass", gate_opening),
                ("enabling_gate_closing_mass", gate_closing),
                ("still_closed_active_mass", still_closed_active),
            ):
                self.statistics[name][destination].add_(
                    mask.sum(dim=reduction_axes, dtype=torch.float64)
                )
            self.statistics["crossing_risk_sum"][destination].add_(
                crossing_risk.sum(dim=reduction_axes, dtype=torch.float64)
            )
            self.statistics["crossing_risk_max"][destination] = torch.maximum(
                self.statistics["crossing_risk_max"][destination],
                crossing_risk.amax(dim=reduction_axes).to(torch.float64),
            )
            for name, mask in (
                (
                    "regime_preserving_inhibitory_effect_mass",
                    regime_preserving_inhibitory,
                ),
                ("gate_opening_effect_mass", gate_opening),
                ("gate_closing_effect_mass", gate_closing),
            ):
                self.statistics[name][destination].add_(
                    (activation_change.abs() * mask).sum(
                        dim=reduction_axes, dtype=torch.float64
                    )
                )

    def finalize(self) -> dict[str, Any]:
        return {
            "schema": "torch_relu_local_regime_sufficient_statistics_v1",
            "activation_kind": "relu",
            "geometry_hash": self.geometry_hash,
            "route_shape": self.route_shape,
            "statistics": {
                name: value.detach()
                for name, value in self.statistics.items()
            },
        }


def _activation_and_gain(value: Any, activation_kind: str) -> tuple[Any, Any]:
    """Return activation and its analytic transmission gain."""

    _require_torch()
    kind = str(activation_kind)
    if kind == "relu":
        return torch.relu(value), (value > 0.0).to(value.dtype)
    if kind == "sigmoid":
        activation = torch.sigmoid(value)
        return activation, activation * (1.0 - activation)
    if kind == "tanh":
        activation = torch.tanh(value)
        return activation, 1.0 - activation * activation
    if kind == "silu":
        sigmoid = torch.sigmoid(value)
        activation = value * sigmoid
        return activation, sigmoid * (1.0 + value * (1.0 - sigmoid))
    if kind == "softplus":
        return torch.nn.functional.softplus(value), torch.sigmoid(value)
    if kind == "gelu":
        # PyTorch's default GELU is the exact erf form.
        inv_sqrt_two = 1.0 / math.sqrt(2.0)
        inv_sqrt_two_pi = 1.0 / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + torch.erf(value * inv_sqrt_two))
        density_term = value * torch.exp(-0.5 * value * value) * inv_sqrt_two_pi
        return torch.nn.functional.gelu(value), cdf + density_term
    raise ValueError(f"unsupported activation kind {kind!r}")


class TorchSmoothTransmissionTee:
    """Stream finite transmission-gain diagnostics for smooth activations."""

    def __init__(
        self,
        *,
        baseline_preactivation: Any,
        route_scale: Any,
        route_shape: tuple[int, int],
        geometry_hash: str,
        activation_kind: str,
        local_active_tolerance: float = 0.0,
        route_chunk_size: int = 16,
    ) -> None:
        _require_torch()
        kind = str(activation_kind)
        if kind not in SUPPORTED_ACTIVATION_KINDS - {"relu"}:
            raise ValueError("smooth transmission tee requires a smooth activation")
        baseline = torch.as_tensor(baseline_preactivation)
        scale = torch.as_tensor(
            route_scale, dtype=torch.float32, device=baseline.device
        ).reshape(-1)
        outputs, inputs = map(int, route_shape)
        if (
            baseline.ndim != 4
            or tuple(baseline.shape[1:2]) != (outputs,)
            or scale.shape != (outputs,)
            or not bool(torch.isfinite(baseline).all().item())
            or not bool(torch.isfinite(scale).all().item())
            or not geometry_hash
            or float(local_active_tolerance) < 0.0
            or int(route_chunk_size) <= 0
        ):
            raise ValueError("smooth transmission tee axes or identity are invalid")
        self.baseline = baseline.detach().to(torch.float32)
        self.route_scale = scale
        self.route_shape = (outputs, inputs)
        self.geometry_hash = str(geometry_hash)
        self.activation_kind = kind
        self.local_active_tolerance = float(local_active_tolerance)
        self.route_chunk_size = int(route_chunk_size)
        self.statistics = {
            name: torch.zeros(
                self.route_shape,
                dtype=torch.float64,
                device=baseline.device,
            )
            for name in SMOOTH_TRANSMISSION_STATISTIC_NAMES
        }

    def update(self, batch_start: int, route_local_field: Any) -> None:
        _require_torch()
        field = torch.as_tensor(
            route_local_field, dtype=torch.float32, device=self.baseline.device
        )
        if (
            field.ndim != 5
            or tuple(field.shape[1:3]) != self.route_shape
            or tuple(field.shape[-2:]) != tuple(self.baseline.shape[-2:])
            or int(batch_start) < 0
            or int(batch_start) + int(field.shape[0]) > int(self.baseline.shape[0])
            or not bool(torch.isfinite(field).all().item())
        ):
            raise ValueError("smooth transmission tee received an incompatible field")
        start = int(batch_start)
        stop = start + int(field.shape[0])
        baseline = self.baseline[start:stop].unsqueeze(2)
        outputs, inputs = self.route_shape
        reduction_axes = (0, 3, 4)
        observation_per_route = float(
            int(field.shape[0]) * int(field.shape[-2]) * int(field.shape[-1])
        )
        for route_start in range(0, inputs, self.route_chunk_size):
            route_stop = min(inputs, route_start + self.route_chunk_size)
            contribution = (
                field[:, :, route_start:route_stop]
                * self.route_scale.reshape(1, outputs, 1, 1, 1)
            )
            counterfactual = baseline - contribution
            activation_before, gain_before = _activation_and_gain(
                baseline, self.activation_kind
            )
            activation_after, gain_after = _activation_and_gain(
                counterfactual, self.activation_kind
            )
            gain_change = gain_after - gain_before
            activation_change = activation_after - activation_before
            active = contribution.abs() > self.local_active_tolerance
            inhibitory = contribution < -self.local_active_tolerance
            destination = (slice(None), slice(route_start, route_stop))
            self.statistics["observation_mass"][destination].add_(
                observation_per_route
            )
            self.statistics["local_active_mass"][destination].add_(
                active.sum(dim=reduction_axes, dtype=torch.float64)
            )
            self.statistics["baseline_gain_sum"][destination].add_(
                gain_before.expand_as(contribution).sum(
                    dim=reduction_axes, dtype=torch.float64
                )
            )
            self.statistics["counterfactual_gain_sum"][destination].add_(
                gain_after.sum(dim=reduction_axes, dtype=torch.float64)
            )
            self.statistics["absolute_gain_change_sum"][destination].add_(
                gain_change.abs().sum(dim=reduction_axes, dtype=torch.float64)
            )
            self.statistics["absolute_gain_change_max"][destination] = torch.maximum(
                self.statistics["absolute_gain_change_max"][destination],
                gain_change.abs().amax(dim=reduction_axes).to(torch.float64),
            )
            self.statistics["transmission_modulation_energy"][destination].add_(
                (gain_change * gain_change).sum(
                    dim=reduction_axes, dtype=torch.float64
                )
            )
            self.statistics["activation_effect_mass"][destination].add_(
                activation_change.abs().sum(
                    dim=reduction_axes, dtype=torch.float64
                )
            )
            self.statistics[
                "regime_preserving_inhibitory_effect_mass"
            ][destination].add_(
                (activation_change.abs() * inhibitory).sum(
                    dim=reduction_axes, dtype=torch.float64
                )
            )

    def finalize(self) -> dict[str, Any]:
        return {
            "schema": "torch_smooth_local_transmission_sufficient_statistics_v1",
            "activation_kind": self.activation_kind,
            "geometry_hash": self.geometry_hash,
            "route_shape": self.route_shape,
            "statistics": {
                name: value.detach()
                for name, value in self.statistics.items()
            },
        }


def make_local_regime_tee(
    *,
    baseline_preactivation: Any,
    route_scale: Any,
    route_shape: tuple[int, int],
    geometry_hash: str,
    activation_kind: str,
    local_active_tolerance: float = 0.0,
    boundary_epsilon: float = 1e-8,
    route_chunk_size: int = 16,
) -> TorchReluRegimeTee | TorchSmoothTransmissionTee:
    if str(activation_kind) == "relu":
        return TorchReluRegimeTee(
            baseline_preactivation=baseline_preactivation,
            route_scale=route_scale,
            route_shape=route_shape,
            geometry_hash=geometry_hash,
            local_active_tolerance=local_active_tolerance,
            boundary_epsilon=boundary_epsilon,
            route_chunk_size=route_chunk_size,
        )
    return TorchSmoothTransmissionTee(
        baseline_preactivation=baseline_preactivation,
        route_scale=route_scale,
        route_shape=route_shape,
        geometry_hash=geometry_hash,
        activation_kind=activation_kind,
        local_active_tolerance=local_active_tolerance,
        route_chunk_size=route_chunk_size,
    )


def validate_local_regime_block_statistics(
    value: Mapping[str, Any],
    *,
    route_shape: tuple[int, ...],
) -> None:
    _require_torch()
    activation_kind = str(value.get("activation_kind", ""))
    expected_names = (
        REGIME_STATISTIC_NAMES
        if activation_kind == "relu"
        else SMOOTH_TRANSMISSION_STATISTIC_NAMES
    )
    expected_schema = (
        "torch_relu_local_regime_sufficient_statistics_v1"
        if activation_kind == "relu"
        else "torch_smooth_local_transmission_sufficient_statistics_v1"
    )
    if (
        value.get("schema") != expected_schema
        or activation_kind not in SUPPORTED_ACTIVATION_KINDS
        or tuple(value.get("route_shape", ())) != tuple(route_shape)
        or not value.get("geometry_hash")
    ):
        raise ValueError("local regime block statistics have invalid identity")
    statistics = value.get("statistics")
    if not isinstance(statistics, Mapping) or set(statistics) != set(
        expected_names
    ):
        raise ValueError("local regime block statistics have invalid fields")
    for name in expected_names:
        array = statistics[name]
        if (
            not torch.is_tensor(array)
            or tuple(array.shape) != tuple(route_shape)
            or not bool(torch.isfinite(array).all().item())
            or bool(torch.any(array < 0.0).item())
        ):
            raise ValueError(f"local regime statistic {name!r} is invalid")


def merge_local_regime_block_statistics(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge disjoint context batches without moving route arrays to host."""

    route_shape = tuple(right.get("route_shape", ()))
    validate_local_regime_block_statistics(right, route_shape=route_shape)
    names = (
        REGIME_STATISTIC_NAMES
        if right["activation_kind"] == "relu"
        else SMOOTH_TRANSMISSION_STATISTIC_NAMES
    )
    if left is None:
        return {
            **{key: right[key] for key in ("schema", "activation_kind", "geometry_hash", "route_shape")},
            "statistics": {
                name: right["statistics"][name].clone()
                for name in names
            },
        }
    validate_local_regime_block_statistics(left, route_shape=route_shape)
    if (
        left["geometry_hash"] != right["geometry_hash"]
        or left["activation_kind"] != right["activation_kind"]
    ):
        raise ValueError("local regime destination geometry changed across batches")
    merged: dict[str, Any] = {}
    for name in names:
        if name in {"crossing_risk_max", "absolute_gain_change_max"}:
            merged[name] = torch.maximum(
                left["statistics"][name], right["statistics"][name]
            )
        else:
            merged[name] = (
                left["statistics"][name] + right["statistics"][name]
            )
    return {
        "schema": left["schema"],
        "activation_kind": left["activation_kind"],
        "geometry_hash": left["geometry_hash"],
        "route_shape": route_shape,
        "statistics": merged,
    }


__all__ = [
    "LocalRegimeCaptureSpec",
    "REGIME_STATISTIC_NAMES",
    "SMOOTH_TRANSMISSION_STATISTIC_NAMES",
    "SUPPORTED_ACTIVATION_KINDS",
    "TorchReluRegimeTee",
    "TorchSmoothTransmissionTee",
    "make_local_regime_tee",
    "merge_local_regime_block_statistics",
    "validate_local_regime_block_statistics",
]
