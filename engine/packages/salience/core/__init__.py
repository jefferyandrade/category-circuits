"""Core modules for salience computations."""

from . import attention_rules  # noqa: F401
from . import frontier_wrappers  # noqa: F401
from .frontier_wrappers import (  # noqa: F401
    HookedConv2d,
    HookedLinear,
    HookedMultiheadAttention,
    HookedSwiGLU,
    convert_conv2d,
    convert_linear,
    convert_multihead_attention,
    convert_swiglu,
    register_frontier_wrappers,
)
from .integrated_engine import IGBackpropEngine, IntegratedEngine, collect_post_frontier_modules

__all__ = [
    "IGBackpropEngine",
    "IntegratedEngine",
    "collect_post_frontier_modules",
    "frontier_wrappers",
    "attention_rules",
    "HookedLinear",
    "HookedConv2d",
    "HookedMultiheadAttention",
    "HookedSwiGLU",
    "convert_linear",
    "convert_conv2d",
    "convert_multihead_attention",
    "convert_swiglu",
    "register_frontier_wrappers",
]
