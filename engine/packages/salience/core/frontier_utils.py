from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Optional

import torch

from packages.salience.stats.stream import GPUStats as _GPUStats
try:
    from configs import scoring_defaults as _scoring_defaults
except ImportError:  # pragma: no cover - fallback when configs unavailable
    _scoring_defaults = None

__all__ = [
    "DEFAULT_TARGET_CHUNK_ELEMENTS",
    "MAX_CHUNK_SAMPLES",
    "COMPUTE_LOSS_DELTA",
    "COMPUTE_BITPACK",
    "ENABLE_ATTENTION_WRAPPER",
    "ENABLE_SWIGLU_WRAPPER",
    "DEBUG_SCORING",
    "VALIDATE_FRONTIER",
    "REPLAY_STORAGE_DEVICE",
    "push_replay_provider",
    "pop_replay_provider",
    "current_replay_provider",
    "GPUStats",
    "debug_print",
    "choose_chunk_size",
]


def _default_from_config(attr: str, fallback):
    if _scoring_defaults is None:
        return fallback
    return getattr(_scoring_defaults, attr, fallback)


def _bool_env(name: str, default: bool) -> bool:
    fallback = "1" if default else "0"
    return bool(int(os.environ.get(name, fallback)))


# Environment-driven frontier configuration flags. Values default to the
# scoring configuration (if available) so jobs inherit consistent behaviour
# without hard-coding the fallback here.
DEFAULT_TARGET_CHUNK_ELEMENTS = int(
    os.environ.get(
        "SCORING_CHUNK_TARGET_ELEMENTS",
        str(_default_from_config("frontier_chunk_target_elements", 262_144)),
    )
)
MAX_CHUNK_SAMPLES = int(
    os.environ.get(
        "SCORING_CHUNK_MAX_SAMPLES",
        str(_default_from_config("frontier_chunk_max_samples", 1_000)),
    )
)
COMPUTE_LOSS_DELTA = _bool_env(
    "SCORING_COMPUTE_LOSS_DELTA",
    _default_from_config("frontier_compute_loss_delta", False),
)
COMPUTE_BITPACK = _bool_env(
    "SCORING_COMPUTE_BITPACK",
    _default_from_config("frontier_compute_bitpack", False),
)
ENABLE_ATTENTION_WRAPPER = _bool_env(
    "SCORING_ENABLE_ATTENTION",
    _default_from_config("frontier_enable_attention_wrapper", False),
)
ENABLE_SWIGLU_WRAPPER = _bool_env(
    "SCORING_ENABLE_SWIGLU",
    _default_from_config("frontier_enable_swiglu_wrapper", False),
)
DEBUG_SCORING = _bool_env(
    "SCORING_DEBUG",
    _default_from_config("frontier_debug", False),
)
VALIDATE_FRONTIER = _bool_env(
    "SCORING_VALIDATE_FRONTIER",
    _default_from_config("frontier_validate", False),
)


def _resolve_replay_storage_device() -> torch.device:
    requested = os.environ.get("REPLAY_CACHE_DEVICE", "cpu")
    try:
        return torch.device(requested)
    except (TypeError, ValueError, RuntimeError):
        print(
            f"[replay] Invalid REPLAY_CACHE_DEVICE='{requested}'. "
            "Defaulting replay cache to CPU."
        )
        return torch.device("cpu")


REPLAY_STORAGE_DEVICE = _resolve_replay_storage_device()

CURRENT_REPLAY_PROVIDER: ContextVar[Optional[object]] = ContextVar(
    "packages.salience.frontier.replay_provider", default=None
)


def push_replay_provider(provider: Optional[object]):
    """
    Register a replay provider for the active batch.

    Returns a context token that should be supplied to `pop_replay_provider`
    once the batch has been consumed.
    """
    if provider is None:
        return None
    return CURRENT_REPLAY_PROVIDER.set(provider)


def pop_replay_provider(token) -> None:
    """
    Restore the previous replay provider registered via `push_replay_provider`.
    """
    if token is None:
        return
    CURRENT_REPLAY_PROVIDER.reset(token)


def current_replay_provider() -> Optional[object]:
    """
    Retrieve the replay provider associated with the current batch, if any.
    """
    return CURRENT_REPLAY_PROVIDER.get()


def debug_print(*args, **kwargs) -> None:
    if DEBUG_SCORING:
        print(*args, **kwargs)


# Re-export GPUStats for downstream consumers
GPUStats = _GPUStats



def choose_chunk_size(
    out_features: int,
    in_features: int,
    spatial_elements: int = 1,
    *,
    target_elements: int = DEFAULT_TARGET_CHUNK_ELEMENTS,
    max_samples: int = 250,
) -> int:
    """Heuristic chunk size selector that caps tensor elements per slice."""
    spatial_elements = max(1, spatial_elements)
    footprint = max(1, out_features * in_features * spatial_elements)
    if footprint == 0:
        return max(1, min(max_samples, target_elements))
    estimated = max(1, target_elements // footprint)
    if max_samples > 0:
        estimated = min(estimated, max_samples)
    return max(1, estimated)
