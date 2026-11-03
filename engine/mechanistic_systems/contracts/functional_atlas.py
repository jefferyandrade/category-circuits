"""Neutral contract for finite response atlases.

An atlas is a measurement/model representation.  Its tolerances and partition
metadata cannot authorize a physical hard-mask intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from mechanistic_systems.contracts.content_address import canonical_json, content_hash


@dataclass(frozen=True, slots=True)
class FunctionalAtlasSpec:
    """Frozen scientific and operational tolerances for one atlas model."""

    atlas_id: str
    phenotype_model: Literal[
        "rank_one_axial_cover",
        "intrinsic_stable_subspace_cover",
        "stateful_interventional_subspace_cover",
    ]
    scientific_residual_tolerance: float
    action_residual_tolerance: float
    split_fold_count: int = 2
    maximum_module_count: int = 64
    ambiguity_policy: Literal["unresolved_no_prune"] = "unresolved_no_prune"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_functional_atlas_spec_v1"

    def __post_init__(self) -> None:
        if not self.atlas_id or not self.schema:
            raise ValueError("functional atlas requires stable identity")
        science = float(self.scientific_residual_tolerance)
        action = float(self.action_residual_tolerance)
        if (
            not math.isfinite(science)
            or science <= 0.0
            or not math.isfinite(action)
            or action <= 0.0
            or action > science
        ):
            raise ValueError(
                "atlas tolerances must be positive and action tolerance may not "
                "exceed science tolerance"
            )
        if int(self.split_fold_count) < 2 or int(self.maximum_module_count) <= 0:
            raise ValueError("atlas fold and module controls are invalid")
        object.__setattr__(self, "scientific_residual_tolerance", science)
        object.__setattr__(self, "action_residual_tolerance", action)
        object.__setattr__(self, "split_fold_count", int(self.split_fold_count))
        object.__setattr__(self, "maximum_module_count", int(self.maximum_module_count))
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-functional-atlas-spec-")


__all__ = ["FunctionalAtlasSpec"]
