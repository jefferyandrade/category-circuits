"""Compatibility facade for neutral response trust geometry.

The implementation owner is :mod:`mechanistic_systems.geometry.trust`.
This module preserves historical IRG import identity without making the
lower-level geometry capability depend on a workflow namespace.
"""

from mechanistic_systems.geometry.trust import (
    FrequentDirectionsTrustAccumulator,
    GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID,
    GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID,
    GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID,
    GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA,
    GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA,
    GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID,
    GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY,
    GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY,
    GlobalTrustGeometryDerivationReceipt,
    GlobalTrustGeometryState,
    GlobalTrustRankGrowthObligation,
    global_trust_ordered_source_rows_hash,
    global_trust_source_row_hash,
    global_trust_stream_weight_hash,
    register_rank_growth_obligation,
)

__all__ = [
    "FrequentDirectionsTrustAccumulator",
    "GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA",
    "GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA",
    "GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY",
    "GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY",
    "GlobalTrustGeometryDerivationReceipt",
    "GlobalTrustGeometryState",
    "GlobalTrustRankGrowthObligation",
    "global_trust_ordered_source_rows_hash",
    "global_trust_source_row_hash",
    "global_trust_stream_weight_hash",
    "register_rank_growth_obligation",
]
