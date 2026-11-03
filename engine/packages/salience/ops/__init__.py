"""
Ops subpackage exposing salience frontier emission helpers.

For now these modules re-export the functions that still live in
``packages.salience.core.frontier_linear`` /
``packages.salience.core.frontier_conv``.
Subsequent refactors will migrate the implementations here.
"""
from __future__ import annotations

from .linear_emit import (
    linear_compute_frontier_aggregate_score,
    linear_compute_kernel_score,
    linear_compute_kernel_score_aggregated,
)
from .conv_emit import (
    conv_compute_kernel_score_aggregated,
)
from .conv_frontier_aggregate import (
    conv_compute_frontier_aggregate_channel_virtual_fused,
    conv_compute_frontier_aggregate_channel_virtual_preadd,
    fused_frontier_aggregate_channel_virtual_available,
)
from .target_batched import (
    conv_channel_route_measurement_target_batched,
    conv_channel_virtual_frontier_aggregate_target_batched,
    linear_route_measurement_target_batched,
    linear_frontier_aggregate_target_batched,
)

__all__ = [
    "linear_compute_frontier_aggregate_score",
    "linear_compute_kernel_score",
    "linear_compute_kernel_score_aggregated",
    "conv_compute_kernel_score_aggregated",
    "conv_compute_frontier_aggregate_channel_virtual_preadd",
    "conv_compute_frontier_aggregate_channel_virtual_fused",
    "fused_frontier_aggregate_channel_virtual_available",
    "conv_channel_virtual_frontier_aggregate_target_batched",
    "conv_channel_route_measurement_target_batched",
    "linear_frontier_aggregate_target_batched",
    "linear_route_measurement_target_batched",
]
