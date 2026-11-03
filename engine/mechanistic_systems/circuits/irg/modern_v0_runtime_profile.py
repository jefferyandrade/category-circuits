"""Registered architecture-resolved execution profile for modern-v0."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from collections.abc import Mapping

# Runtime profiles prescribe execution rows, not the scientific tau.
DEFAULT_MODERN_V0_SENSITIVITY_TAU: None = None


@dataclass(frozen=True)
class ModernV0RuntimeProfile:
    architecture_family: str
    model_id: str
    execution_batch_rows: int
    scoring_batch_rows: int
    calibration_batch_rows: int
    v0_paired_oracle_batch_rows: int
    terminal_batch_rows: int
    mps_k: int
    sensitivity_tau: float | None = DEFAULT_MODERN_V0_SENSITIVITY_TAU


_PROFILES = (
    ModernV0RuntimeProfile(
        architecture_family="alexnet_conv_kernel_weight",
        model_id="torchvision.alexnet.conv_kernel_weight.imagenet1k_v1",
        execution_batch_rows=32,
        scoring_batch_rows=32,
        calibration_batch_rows=512,
        v0_paired_oracle_batch_rows=256,
        terminal_batch_rows=512,
        mps_k=4,
    ),
    ModernV0RuntimeProfile(
        architecture_family="vgg11_conv_kernel_weight",
        model_id="torchvision.vgg11.conv_kernel_weight.imagenet1k_v1",
        execution_batch_rows=16,
        scoring_batch_rows=16,
        calibration_batch_rows=256,
        v0_paired_oracle_batch_rows=64,
        terminal_batch_rows=128,
        mps_k=2,
    ),
    ModernV0RuntimeProfile(
        architecture_family="resnet18_conv_kernel_weight",
        model_id="torchvision.resnet18.conv_kernel_weight.imagenet1k_v1",
        execution_batch_rows=32,
        scoring_batch_rows=32,
        calibration_batch_rows=512,
        v0_paired_oracle_batch_rows=256,
        terminal_batch_rows=512,
        mps_k=4,
    ),
    ModernV0RuntimeProfile(
        architecture_family="resnet50_conv_kernel_weight",
        model_id="torchvision.resnet50.conv_kernel_weight.imagenet1k_v2",
        execution_batch_rows=8,
        scoring_batch_rows=8,
        calibration_batch_rows=128,
        v0_paired_oracle_batch_rows=32,
        terminal_batch_rows=128,
        mps_k=2,
    ),
    # The IMAGENET1K_V1 checkpoint on the registered ResNet-50
    # kernel-weight axis runs with exactly the V2 row's execution shape.
    ModernV0RuntimeProfile(
        architecture_family="resnet50_conv_kernel_weight",
        model_id="torchvision.resnet50.conv_kernel_weight.imagenet1k_v1",
        execution_batch_rows=8,
        scoring_batch_rows=8,
        calibration_batch_rows=128,
        v0_paired_oracle_batch_rows=32,
        terminal_batch_rows=128,
        mps_k=2,
    ),
    ModernV0RuntimeProfile(
        architecture_family="robust_resnet50_conv_kernel_weight",
        model_id=("robustbench.salman2020do_r50.conv_kernel_weight.imagenet_linf_4_255_v1"),
        execution_batch_rows=8,
        scoring_batch_rows=8,
        calibration_batch_rows=128,
        v0_paired_oracle_batch_rows=32,
        terminal_batch_rows=128,
        mps_k=2,
    ),
)


def modern_v0_runtime_profile(model_id: str) -> ModernV0RuntimeProfile:
    matches = tuple(profile for profile in _PROFILES if profile.model_id == str(model_id))
    if len(matches) != 1:
        raise ValueError(
            f"modern-v0 has no unique registered runtime profile for model {model_id!r}"
        )
    return matches[0]


def resolve_modern_v0_sensitivity_tau(
    sensitivity_tau: float | None, *, environ: Mapping[str, str] | None = None
) -> float:
    """Resolve the caller's tau declaration without a scientific default.

    ``environ`` is the item's environment (the worker's private next-item
    mapping during host preparation); the process environment otherwise.
    """

    declared = (
        (os.environ if environ is None else environ).get("IRG_SENSITIVITY_TAU")
        if sensitivity_tau is None
        else sensitivity_tau
    )
    if declared is None or (isinstance(declared, str) and not declared.strip()):
        raise ValueError(
            "v0 requires an explicit sensitivity_tau argument or IRG_SENSITIVITY_TAU"
        )
    try:
        tau = float(declared)
    except (TypeError, ValueError):
        raise ValueError("v0 sensitivity_tau must be finite and nonnegative") from None
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("v0 sensitivity_tau must be finite and nonnegative")
    return tau


__all__ = [
    "DEFAULT_MODERN_V0_SENSITIVITY_TAU",
    "ModernV0RuntimeProfile",
    "modern_v0_runtime_profile",
    "resolve_modern_v0_sensitivity_tau",
]
