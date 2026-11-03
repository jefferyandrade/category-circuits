"""Utility wrappers around optional torch.compile."""
from __future__ import annotations

import functools
from typing import Callable, TypeVar

from mechanistic_systems.core.determinism import (
    configure_deterministic_backend,
    deterministic_compile_options,
)
import torch
import torch._dynamo

F = TypeVar("F", bound=Callable)


def compile_if_available(fn: F) -> F:
    """Wrap ``torch.compile`` in a best-effort manner.

    The returned function falls back to ``fn`` unchanged when ``torch.compile``
    is unavailable or raises during compilation. This keeps call sites simple
    while allowing high-performance builds to take advantage of ahead-of-time
    graph compilation.
    """

    configure_deterministic_backend()
    compile_impl = getattr(torch, "compile", None)
    if compile_impl is None:
        return fn

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 4096)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 4096)

    try:
        compiled = compile_impl(
            fn, fullgraph=True,
            options=deterministic_compile_options("max-autotune"),
        )
    except Exception:
        return fn

    return functools.update_wrapper(compiled, fn)  # type: ignore[arg-type]


__all__ = ["compile_if_available"]
