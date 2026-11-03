"""Glocal-v3 extensions to the immutable sampled-path v1 contract.

``sampled_path.py`` is a protected historical runtime surface.  Claim-mode
bounded-risk evidence needs additional request targets, calibration identity,
and resource telemetry, but those extensions must not mutate the v1 source or
its replay hash.  This module therefore owns the additive v3 interface.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .sampled_path import (
    CrossfitPanel,
    SampledComparisonRequest,
    SampledPathEvaluationBackend,
    _digest,
)

GLOCAL_BRIER_CALIBRATION_SCHEMA = "mechanistic_glocal_brier_calibration_v1"
GLOCAL_PAIRED_BRIER_SCHEMA = "mechanistic_glocal_paired_brier_evidence_v1"


def _freeze_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_identity_value(value[key])
                for key in sorted(value, key=str)
            }
        )
    if isinstance(value, np.ndarray):
        return _freeze_identity_value(value.tolist())
    if isinstance(value, np.generic):
        return _freeze_identity_value(value.item())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_identity_value(item) for item in value)
    return value


@dataclass(frozen=True)
class GlocalSampledComparisonRequest(SampledComparisonRequest):
    """Additive exact-prefix request used by continuous-v2 and glocal-v3.

    The subclass remains acceptable anywhere the historical base request is
    expected.  Exact targets are intentionally absent from the v1 request
    identity and live only on this versioned extension.
    """

    cumulative_target_images: int | None = None
    per_stratum_target_images: Mapping[str, int] = field(default_factory=dict)
    maximum_new_contexts: int | None = None
    maximum_cumulative_images: int | None = None
    calibration_target_loss_precision: float | None = None
    runtime_deadline_monotonic: float | None = None

    def __post_init__(self) -> None:
        target = self.cumulative_target_images
        if target is not None and int(target) <= 0:
            raise ValueError("cumulative panel target must be positive")
        targets = {
            str(key): int(value)
            for key, value in self.per_stratum_target_images.items()
        }
        if any(value < 0 for value in targets.values()):
            raise ValueError("per-stratum panel targets must be nonnegative")
        if targets and target is None:
            raise ValueError("per-stratum targets require a cumulative panel target")
        if targets and sum(targets.values()) != int(target):
            raise ValueError(
                "per-stratum panel targets must sum to cumulative target images"
            )
        if self.maximum_new_contexts is not None and int(self.maximum_new_contexts) < 0:
            raise ValueError("new-context budget must be nonnegative")
        if (
            self.maximum_cumulative_images is not None
            and int(self.maximum_cumulative_images) <= 0
        ):
            raise ValueError("cumulative image horizon must be positive")
        if (
            target is not None
            and self.maximum_cumulative_images is not None
            and int(target) > int(self.maximum_cumulative_images)
        ):
            raise ValueError("cumulative target exceeds its image horizon")
        calibration_target = self.calibration_target_loss_precision
        if calibration_target is not None and (
            not math.isfinite(float(calibration_target))
            or float(calibration_target) <= 0.0
        ):
            raise ValueError(
                "calibration loss-precision target must be finite and positive"
            )
        deadline = self.runtime_deadline_monotonic
        if deadline is not None and (
            not math.isfinite(float(deadline)) or float(deadline) <= 0.0
        ):
            raise ValueError("runtime deadline must be finite and positive")
        object.__setattr__(
            self,
            "per_stratum_target_images",
            MappingProxyType(dict(sorted(targets.items()))),
        )


def request_cumulative_target_images(
    request: SampledComparisonRequest,
) -> int | None:
    """Read the additive cumulative target without weakening v1 replay."""

    value = getattr(request, "cumulative_target_images", None)
    return None if value is None else int(value)


def request_per_stratum_target_images(
    request: SampledComparisonRequest,
) -> Mapping[str, int]:
    """Read additive stratum targets from either request generation."""

    return {
        str(key): int(value)
        for key, value in dict(
            getattr(request, "per_stratum_target_images", {}) or {}
        ).items()
    }


def request_maximum_new_contexts(
    request: SampledComparisonRequest,
) -> int | None:
    """Read the additive hard resource cap without changing the v1 protocol."""

    value = getattr(request, "maximum_new_contexts", None)
    return None if value is None else int(value)


class GlocalSampledPathEvaluationBackend(
    SampledPathEvaluationBackend,
    Protocol,
):
    """Resource-accounting extension implemented by the production backend."""

    @property
    def decision_images_consumed(self) -> int: ...

    @property
    def resource_usage(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FrozenBrierCalibration:
    """One state-specific affine fit learned only from calibration-role rows."""

    query_id: str
    role: str
    mask_hash: str
    calibration_context_ids: tuple[str, ...]
    positive_class_id: int
    scale: float
    offset: float
    input_center: float
    covariance: tuple[tuple[float, float], tuple[float, float]]
    converged: bool
    iterations: int
    objective_value: float
    algorithm_id: str
    algorithm_options: Mapping[str, Any] = field(default_factory=dict)
    warm_start_parent_hash: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = GLOCAL_BRIER_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        contexts = tuple(str(value) for value in self.calibration_context_ids)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (2, 2):
            raise ValueError("glocal Brier calibration covariance must be finite 2x2")
        covariance = 0.5 * (covariance + covariance.T)
        frozen_covariance = tuple(
            tuple(float(value) for value in row) for row in covariance
        )
        object.__setattr__(self, "calibration_context_ids", contexts)
        object.__setattr__(self, "positive_class_id", int(self.positive_class_id))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "input_center", float(self.input_center))
        object.__setattr__(self, "covariance", frozen_covariance)
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "objective_value", float(self.objective_value))
        object.__setattr__(
            self,
            "algorithm_options",
            _freeze_identity_value(self.algorithm_options),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_identity_value(self.diagnostics),
        )

    def validate(self) -> None:
        if (
            not self.query_id
            or not self.role
            or len(self.mask_hash) != 64
            or not self.algorithm_id
            or self.schema_name != GLOCAL_BRIER_CALIBRATION_SCHEMA
        ):
            raise ValueError("glocal Brier calibration identity is invalid")
        contexts = tuple(str(value) for value in self.calibration_context_ids)
        if not contexts or len(contexts) != len(set(contexts)):
            raise ValueError("glocal Brier calibration contexts must be unique")
        values = (
            float(self.scale),
            float(self.offset),
            float(self.input_center),
            float(self.objective_value),
        )
        if (
            any(not math.isfinite(value) for value in values)
            or float(self.scale) <= 0.0
            or float(self.objective_value) < 0.0
        ):
            raise ValueError("glocal Brier calibration parameters are invalid")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
            raise ValueError("glocal Brier calibration covariance must be finite 2x2")
        covariance = 0.5 * (covariance + covariance.T)
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10:
            raise ValueError(
                "glocal Brier calibration covariance must be positive semidefinite"
            )
        if not self.converged or int(self.iterations) <= 0:
            raise ValueError("claim-bearing glocal calibration must converge")
        if (
            self.warm_start_parent_hash is not None
            and len(self.warm_start_parent_hash) != 64
        ):
            raise ValueError("glocal calibration warm-start lineage hash is invalid")

    @property
    def output_hash(self) -> str:
        self.validate()
        return _digest(
            {
                "schema_name": self.schema_name,
                "query_id": self.query_id,
                "role": self.role,
                "mask_hash": self.mask_hash,
                "calibration_context_ids": self.calibration_context_ids,
                "positive_class_id": self.positive_class_id,
                "scale": self.scale,
                "offset": self.offset,
                "input_center": self.input_center,
                "covariance": self.covariance,
                "converged": self.converged,
                "iterations": self.iterations,
                "objective_value": self.objective_value,
                "algorithm_id": self.algorithm_id,
                "algorithm_options": dict(self.algorithm_options),
                "warm_start_parent_hash": self.warm_start_parent_hash,
                "diagnostics": dict(self.diagnostics),
            }
        )


@dataclass(frozen=True)
class PairedBrierEvidence:
    """Frozen-calibration paired probabilities on adjudication-only rows."""

    panel: CrossfitPanel
    parent_probabilities: np.ndarray
    candidate_probabilities: np.ndarray
    binary_labels: np.ndarray
    parent_calibration_output_hash: str
    candidate_calibration_output_hash: str
    parent_mask_hash: str
    candidate_mask_hash: str
    common_random_numbers: bool = True
    schema_name: str = GLOCAL_PAIRED_BRIER_SCHEMA

    def __post_init__(self) -> None:
        parent = (
            np.asarray(
                self.parent_probabilities,
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
        )
        candidate = (
            np.asarray(
                self.candidate_probabilities,
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
        )
        raw_labels = np.asarray(self.binary_labels).reshape(-1)
        if not np.all(np.isin(raw_labels, (0, 1, False, True))):
            raise ValueError("paired Brier evidence labels must be binary")
        labels = raw_labels.astype(np.uint8, copy=True)
        parent.setflags(write=False)
        candidate.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "parent_probabilities", parent)
        object.__setattr__(self, "candidate_probabilities", candidate)
        object.__setattr__(self, "binary_labels", labels)

    def validate(self, *, fold_count: int) -> None:
        self.panel.validate(fold_count=fold_count)
        parent = np.asarray(self.parent_probabilities, dtype=np.float64).reshape(-1)
        candidate = np.asarray(self.candidate_probabilities, dtype=np.float64).reshape(
            -1
        )
        labels = np.asarray(self.binary_labels).reshape(-1)
        size = len(self.panel.context_ids)
        if (
            parent.size != size
            or candidate.size != size
            or labels.size != size
            or not np.isfinite(parent).all()
            or not np.isfinite(candidate).all()
            or np.any(parent < 0.0)
            or np.any(parent > 1.0)
            or np.any(candidate < 0.0)
            or np.any(candidate > 1.0)
            or not np.all(np.isin(labels, (0, 1, False, True)))
        ):
            raise ValueError("paired Brier evidence has invalid probability/label rows")
        if not self.common_random_numbers:
            raise ValueError("paired Brier evidence requires common random numbers")
        for value in (
            self.parent_calibration_output_hash,
            self.candidate_calibration_output_hash,
            self.parent_mask_hash,
            self.candidate_mask_hash,
        ):
            if len(str(value)) != 64:
                raise ValueError("paired Brier evidence identity hash is invalid")
        if self.schema_name != GLOCAL_PAIRED_BRIER_SCHEMA:
            raise ValueError("paired Brier evidence schema is invalid")

    @property
    def evidence_hash(self) -> str:
        return _digest(
            {
                "schema_name": self.schema_name,
                "panel_hash": self.panel.panel_hash,
                "parent_probabilities": np.asarray(
                    self.parent_probabilities, dtype=np.float64
                ),
                "candidate_probabilities": np.asarray(
                    self.candidate_probabilities, dtype=np.float64
                ),
                "binary_labels": np.asarray(self.binary_labels, dtype=np.uint8),
                "parent_calibration_output_hash": (self.parent_calibration_output_hash),
                "candidate_calibration_output_hash": (
                    self.candidate_calibration_output_hash
                ),
                "parent_mask_hash": self.parent_mask_hash,
                "candidate_mask_hash": self.candidate_mask_hash,
            }
        )


__all__ = [
    "FrozenBrierCalibration",
    "GLOCAL_BRIER_CALIBRATION_SCHEMA",
    "GLOCAL_PAIRED_BRIER_SCHEMA",
    "GlocalSampledComparisonRequest",
    "GlocalSampledPathEvaluationBackend",
    "PairedBrierEvidence",
    "request_cumulative_target_images",
    "request_maximum_new_contexts",
    "request_per_stratum_target_images",
]
