from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping

import torch

from .features import FeatureBatch, FeatureSchema


@dataclass(frozen=True)
class FeatureAccumulatorConfig:
    max_buffer_rows: int = 262_144
    device: torch.device | str = torch.device("cpu")


class FeatureAccumulator:
    def __init__(
        self,
        *,
        layer: str,
        unit: str | None,
        schema: FeatureSchema,
        config: FeatureAccumulatorConfig | None = None,
    ) -> None:
        self.layer = layer
        self.unit = unit
        self.schema = schema
        self.config = config or FeatureAccumulatorConfig()
        self._indices: list[torch.Tensor] = []
        self._values: MutableMapping[str, list[torch.Tensor]] = {
            name: [] for name in schema.feature_names
        }
        self._counts: list[torch.Tensor] = []
        self._rows = 0

    def add(self, batch: FeatureBatch) -> FeatureBatch | None:
        if batch.layer != self.layer:
            raise ValueError(f"Feature batch layer {batch.layer!r} does not match {self.layer!r}")
        indices = batch.weight_indices.detach().to("cpu").reshape(-1).clone()
        rows = indices.numel()
        self._indices.append(indices)
        for name in self.schema.feature_names:
            if name not in batch.values:
                raise KeyError(f"Feature batch missing {name!r}")
            tensor = batch.values[name].detach().to("cpu").reshape(-1).clone()
            if tensor.numel() != rows:
                raise ValueError(f"Feature {name!r} has {tensor.numel()} rows, expected {rows}")
            self._values[name].append(tensor)
        if batch.counts is not None:
            counts = batch.counts.detach().to("cpu").reshape(-1).clone()
            if counts.numel() != rows:
                raise ValueError(f"Counts have {counts.numel()} rows, expected {rows}")
            self._counts.append(counts)
        self._rows += rows
        if self._rows >= max(1, int(self.config.max_buffer_rows)):
            return self.flush()
        return None

    def flush(self) -> FeatureBatch | None:
        if self._rows == 0:
            return None
        indices = torch.cat(self._indices)
        values = {name: torch.cat(chunks) for name, chunks in self._values.items()}
        counts = torch.cat(self._counts) if self._counts else None
        self._indices.clear()
        for chunks in self._values.values():
            chunks.clear()
        self._counts.clear()
        self._rows = 0
        return FeatureBatch(
            layer=self.layer,
            unit=self.unit,
            weight_indices=indices,
            values=values,
            counts=counts,
        )


__all__ = ["FeatureAccumulator", "FeatureAccumulatorConfig"]
