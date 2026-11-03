"""Canonical capability registry for first-class numerical backends."""

from __future__ import annotations

import os

from .calibration_backend import CanonicalCalibrationBackend
from .contracts import BackendRegistry
from .torch_gate_graph import TorchGateGraphCompiler
from .torch_runtime import (
    DirectTorchConvRouteReducerBackend,
    TorchCandidateExecutionBackend,
    TorchContractionBackend,
    TorchConvKernelWeightRouteReducerBackend,
    TorchRouteReducerBackend,
)


def build_default_backend_registry(
    *,
    device_type: str = "all",
    compile_candidate_vmap: bool = False,
) -> BackendRegistry:
    """Build architecture-neutral production dispatch by capability.

    Scientific identity never includes this registry. Backend IDs and runtime
    profiles are recorded separately so a numerically equivalent optimized
    implementation can be promoted without changing a MeasurementPlan.
    """

    registry = BackendRegistry()
    registry.register(TorchGateGraphCompiler(), capability="gate_graph")
    reducer = TorchRouteReducerBackend()
    conv_selector = str(
        os.getenv("MECHANISTIC_CONV_ROUTE_REDUCER", "direct_unfold")
    ).strip().lower()
    direct_selectors = {"direct", "direct_unfold", "direct_torch"}
    reference_selectors = {
        "channel_virtual",
        "channel_virtual_reference",
        "reference",
    }
    if conv_selector not in direct_selectors | reference_selectors:
        raise ValueError(
            "MECHANISTIC_CONV_ROUTE_REDUCER must select direct_unfold or "
            "channel_virtual_reference"
        )
    for family in sorted(reducer.supported_route_families):
        selected = reducer
        if family == "conv_channel_pair" and conv_selector in direct_selectors:
            selected = DirectTorchConvRouteReducerBackend()
        registry.register(
            selected,
            capability="route_reducer",
            route_family=family,
            device_type=str(device_type),
            dtype="float32",
        )
    registry.register(
        TorchConvKernelWeightRouteReducerBackend(),
        capability="route_reducer",
        route_family="conv_kernel_weight",
        device_type=str(device_type),
        dtype="float32",
    )
    registry.register(
        CanonicalCalibrationBackend(),
        capability="calibration",
    )
    registry.register(
        TorchCandidateExecutionBackend(compile_vmap=compile_candidate_vmap),
        capability="candidate_execution",
        device_type=str(device_type),
    )
    registry.register(
        TorchContractionBackend(),
        capability="contraction",
        device_type=str(device_type),
        dtype="float32",
    )
    return registry


__all__ = ["build_default_backend_registry"]
