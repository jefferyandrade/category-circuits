"""Always-on deterministic algorithm policy for scientific execution.

Import this module before constructing CUDA models so cuBLAS sees the fixed
workspace configuration when it creates its first handle. Precision remains
the caller's choice; determinism does not change TF32 or autocast settings.
"""

from __future__ import annotations

import os
from typing import Any


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DETERMINISTIC_EXECUTION_CACHE_VERSION = "deterministic_science_v2"
_FIXED_TRITON_RECIPES = {
    name: {"BLOCK_N": 256, "num_warps": 4, "num_stages": 2}
    for name in ("classwise_grad_hess", "linear_lossdiff", "conv_lossdiff")
}
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
_INDUCTOR_AUTOTUNE_SETTINGS = {
    "max_autotune": "TORCHINDUCTOR_MAX_AUTOTUNE",
    "max_autotune_pointwise": "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE",
    "max_autotune_gemm": "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
    "coordinate_descent_tuning": "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING",
    "dynamic_scale_rblock": "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK",
    "search_autotune_cache": "TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE",
    "benchmark_fusion": "TORCHINDUCTOR_BENCHMARK_FUSION",
}
for _environment_name in _INDUCTOR_AUTOTUNE_SETTINGS.values():
    os.environ[_environment_name] = "0"


def configure_deterministic_backend() -> None:
    """Select deterministic kernels without a runtime opt-out or fallback."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    for environment_name in _INDUCTOR_AUTOTUNE_SETTINGS.values():
        os.environ[environment_name] = "0"

    import torch
    from torch._inductor import config as inductor_config

    for setting in _INDUCTOR_AUTOTUNE_SETTINGS:
        setattr(inductor_config, setting, False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def deterministic_compile_options(mode: str | None = None) -> dict[str, Any]:
    """Retain a compile mode's graph policy without its timing-based tuning.

    PyTorch applies mode options inside compilation, overriding process
    configuration. Supplying the expanded options with the mandatory policy
    prevents a requested max-autotune mode from re-enabling algorithm choice.
    """

    configure_deterministic_backend()
    from torch._inductor import list_mode_options

    options = dict(list_mode_options(mode or "default"))
    options.update(dict.fromkeys(_INDUCTOR_AUTOTUNE_SETTINGS, False))
    return options


def deterministic_backend_record() -> dict[str, Any]:
    """Observe the effective policy for receipts without modifying settings."""

    import torch
    from torch._inductor import config as inductor_config

    return {
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **{
            "inductor_" + setting: bool(getattr(inductor_config, setting))
            for setting in _INDUCTOR_AUTOTUNE_SETTINGS
        },
        "fixed_triton_recipes": {
            name: dict(recipe) for name, recipe in _FIXED_TRITON_RECIPES.items()
        },
    }
