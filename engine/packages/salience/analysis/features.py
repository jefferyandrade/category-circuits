from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch


class FeatureReducer(str, Enum):
    SUM = "sum"
    SUM_OF_SQUARES = "sum_of_squares"
    MEAN = "mean"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str = ""
    reducer: FeatureReducer = FeatureReducer.SUM


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    specs: tuple[FeatureSpec, ...]


@dataclass(frozen=True)
class FeatureSchema:
    blocks: tuple[FeatureBlock, ...]
    specs: tuple[FeatureSpec, ...]
    feature_names: tuple[str, ...]
    schema_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blocks": [
                {
                    "name": block.name,
                    "specs": [
                        {
                            **asdict(spec),
                            "reducer": spec.reducer.value,
                        }
                        for spec in block.specs
                    ],
                }
                for block in self.blocks
            ],
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FeatureSchema":
        blocks = []
        for block_payload in payload.get("blocks", ()):
            block_map = dict(block_payload)  # type: ignore[arg-type]
            specs = []
            for spec_payload in block_map.get("specs", ()):
                spec_map = dict(spec_payload)  # type: ignore[arg-type]
                specs.append(
                    FeatureSpec(
                        name=str(spec_map["name"]),
                        description=str(spec_map.get("description", "")),
                        reducer=FeatureReducer(str(spec_map.get("reducer", FeatureReducer.SUM.value))),
                    )
                )
            blocks.append(FeatureBlock(name=str(block_map["name"]), specs=tuple(specs)))
        return build_schema(tuple(blocks))


@dataclass(frozen=True)
class FeatureBatch:
    layer: str
    unit: str | None
    weight_indices: torch.Tensor
    values: Mapping[str, torch.Tensor]
    counts: torch.Tensor | None = None


def build_schema(blocks: Sequence[FeatureBlock]) -> FeatureSchema:
    block_tuple = tuple(blocks)
    specs = tuple(spec for block in block_tuple for spec in block.specs)
    feature_names = tuple(spec.name for spec in specs)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError(f"Feature names must be unique: {feature_names!r}")
    canonical = json.dumps(
        [
            {
                "name": block.name,
                "specs": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "reducer": spec.reducer.value,
                    }
                    for spec in block.specs
                ],
            }
            for block in block_tuple
        ],
        sort_keys=True,
    )
    schema_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return FeatureSchema(
        blocks=block_tuple,
        specs=specs,
        feature_names=feature_names,
        schema_hash=schema_hash,
    )


__all__ = [
    "FeatureBatch",
    "FeatureBlock",
    "FeatureReducer",
    "FeatureSchema",
    "FeatureSpec",
    "build_schema",
]
