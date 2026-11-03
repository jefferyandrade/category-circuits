"""Canonical score-only conv frontier aggregate reducers.

This module is the public import surface for the universal conv reducer used by
``fast``/``frontier_direct`` scoring.  The implementation lives in the
compatibility module ``conv_oldfast_fused``.
"""

from __future__ import annotations

from packages.salience.ops.conv_oldfast_fused import (
    _frontier_aggregate_conv_block_size,
    _frontier_aggregate_conv_impl,
    conv_compute_frontier_aggregate_channel_virtual_fused,
    conv_compute_frontier_aggregate_channel_virtual_preadd,
    fused_frontier_aggregate_channel_virtual_available,
)

__all__ = [
    "conv_compute_frontier_aggregate_channel_virtual_preadd",
    "conv_compute_frontier_aggregate_channel_virtual_fused",
    "fused_frontier_aggregate_channel_virtual_available",
    "_frontier_aggregate_conv_impl",
    "_frontier_aggregate_conv_block_size",
]
