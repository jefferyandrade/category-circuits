"""Neutral response-functional and metric contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

from mechanistic_systems.contracts.content_address import canonical_json, content_hash


HistoricalResponseProfileName: TypeAlias = Literal["v0", "v1", "v2", "v3", "v4"]


@dataclass(frozen=True)
class ResponseFunctionalSpec:
    """Frozen scientific definition of the model-side response being preserved.

    v0 may retain the historical raw target-logit functional.  Claim-bearing
    v1-v4 studies should use ``category_contrast`` or an explicitly declared
    exact low-rank realization such as ``target_vs_rest_channels``.  Scalar
    margin, probability, cross-entropy, and decision quantities are derived
    views of that response rather than silently interchangeable targets.
    """

    response_id: str
    kind: Literal[
        "target_logit",
        "category_contrast",
        "target_vs_rest_channels",
        "target_vs_rest_margin",
        "top1_margin",
        "cross_entropy",
        "custom",
    ]
    target_index: int
    class_count: int
    competitor_policy: Literal["all", "explicit", "top_k", "aggregate_rest"] = "all"
    competitor_indices: tuple[int, ...] = ()
    top_k: int | None = None
    calibrated: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "task_response_functional_v1"

    def __post_init__(self) -> None:
        if not self.response_id or not self.schema:
            raise ValueError("response functional requires stable identity")
        if int(self.class_count) < 2:
            raise ValueError("response functional class_count must be at least two")
        if not 0 <= int(self.target_index) < int(self.class_count):
            raise ValueError("response target_index lies outside the class axis")
        competitors = tuple(int(value) for value in self.competitor_indices)
        if any(
            value == self.target_index or not 0 <= value < self.class_count
            for value in competitors
        ):
            raise ValueError("competitor indices must be valid non-target classes")
        if len(set(competitors)) != len(competitors):
            raise ValueError("competitor indices must be unique")
        if self.competitor_policy == "explicit" and not competitors:
            raise ValueError("explicit competitor policy requires competitor_indices")
        if self.competitor_policy == "top_k":
            if self.top_k is None or not 1 <= int(self.top_k) < int(self.class_count):
                raise ValueError(
                    "top_k competitor policy requires 1 <= top_k < class_count"
                )
        elif self.top_k is not None:
            raise ValueError("top_k is only valid for the top_k competitor policy")
        if self.kind == "target_logit" and self.competitor_policy not in {
            "all",
            "aggregate_rest",
        }:
            raise ValueError(
                "target-logit response may not select an arbitrary competitor subset"
            )
        object.__setattr__(self, "competitor_indices", competitors)
        object.__setattr__(self, "parameters", dict(self.parameters))
        canonical_json(self.parameters)

    @property
    def output_dimension(self) -> int:
        if self.kind in {
            "target_logit",
            "target_vs_rest_margin",
            "top1_margin",
            "cross_entropy",
        }:
            return 1
        if self.kind == "target_vs_rest_channels":
            return 2
        if self.kind == "category_contrast":
            if self.competitor_policy == "explicit":
                return len(self.competitor_indices)
            if self.competitor_policy == "top_k":
                return int(self.top_k or 0)
            return self.class_count - 1
        return int(self.parameters.get("output_dimension", 1))

    @property
    def is_contrast_complete(self) -> bool:
        if self.kind == "target_vs_rest_channels":
            return True
        if self.kind == "category_contrast":
            return self.competitor_policy in {"all", "explicit"}
        return False

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="task-response-")

    @classmethod
    def historical_target_logit(
        cls, *, target_index: int, class_count: int, calibrated: bool = True
    ) -> "ResponseFunctionalSpec":
        return cls(
            response_id=f"target-logit:{target_index}",
            kind="target_logit",
            target_index=target_index,
            class_count=class_count,
            competitor_policy="all",
            calibrated=calibrated,
        )

    @classmethod
    def category_contrast(
        cls, *, target_index: int, class_count: int, calibrated: bool = True
    ) -> "ResponseFunctionalSpec":
        return cls(
            response_id=f"category-contrast:{target_index}:all",
            kind="category_contrast",
            target_index=target_index,
            class_count=class_count,
            competitor_policy="all",
            calibrated=calibrated,
        )


@dataclass(frozen=True)
class ResponseMetricSpec:
    metric_id: str
    output_dimension: int
    geometry: Literal[
        "euclidean",
        "fisher",
        "margin",
        "contrast_euclidean",
        "softmax_fisher",
        "competitor_weighted",
        "custom",
    ] = "euclidean"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_id:
            raise ValueError("response metric requires a metric_id")
        if int(self.output_dimension) <= 0:
            raise ValueError("response metric output_dimension must be positive")
        object.__setattr__(self, "parameters", dict(self.parameters))
        canonical_json(self.parameters)


__all__ = [
    "HistoricalResponseProfileName",
    "ResponseFunctionalSpec",
    "ResponseMetricSpec",
]

