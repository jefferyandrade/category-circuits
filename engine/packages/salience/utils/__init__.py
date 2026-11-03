"""Utility helpers for salience modules."""

from .compile import compile_if_available
from .bitpack import pack_boolean_rows, vectorised_bitpack
from .replay import (
    TensorReplayProvider,
    ReplayTensorProxy,
    replay_batch_size,
    replay_matches_tensor,
    fetch_replay_slice,
    fetch_replay_indices,
)

__all__ = [
    "compile_if_available",
    "pack_boolean_rows",
    "vectorised_bitpack",
    "TensorReplayProvider",
    "ReplayTensorProxy",
    "replay_batch_size",
    "replay_matches_tensor",
    "fetch_replay_slice",
    "fetch_replay_indices",
]
