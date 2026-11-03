"""Neutral contracts for response trust geometry and validity decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from mechanistic_systems.contracts.content_address import canonical_json, content_hash


@dataclass(frozen=True, slots=True)
class GlobalTrustGeometrySpec:
    """Globally coupled full-route relinearization geometry."""

    trust_id: str
    initial_rank: int
    maximum_rank: int
    reuse_threshold: float
    refresh_threshold: float
    residual_policy: Literal["decision_relevant_rank_growth"] = (
        "decision_relevant_rank_growth"
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_global_trust_geometry_spec_v1"

    def __post_init__(self) -> None:
        if not self.trust_id or not self.schema:
            raise ValueError("global trust geometry requires stable identity")
        initial, maximum = int(self.initial_rank), int(self.maximum_rank)
        reuse, refresh = float(self.reuse_threshold), float(self.refresh_threshold)
        if (
            initial <= 0
            or maximum < initial
            or not math.isfinite(reuse)
            or not math.isfinite(refresh)
            or reuse < 0.0
            or refresh <= reuse
        ):
            raise ValueError("global trust geometry controls are invalid")
        object.__setattr__(self, "initial_rank", initial)
        object.__setattr__(self, "maximum_rank", maximum)
        object.__setattr__(self, "reuse_threshold", reuse)
        object.__setattr__(self, "refresh_threshold", refresh)
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-global-trust-spec-")


@dataclass(frozen=True)
class ModelValidityDecision:
    action: Literal["reuse", "shrink", "refresh", "escalate", "fallback", "refuse"]
    reason: str
    recommended_scale: float
    predicted_rms: float = 0.0
    actual_rms: float | None = None
    remainder_rms: float | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {
            "reuse",
            "shrink",
            "refresh",
            "escalate",
            "fallback",
            "refuse",
        }:
            raise ValueError("unknown model validity action")
        if not self.reason:
            raise ValueError("model validity decision requires a reason")
        for name in ("recommended_scale", "predicted_rms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("actual_rms", "remainder_rms"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative when provided")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        canonical_json(self.diagnostics)


__all__ = ["GlobalTrustGeometrySpec", "ModelValidityDecision"]
