"""Scientific contracts for readout construction and calibration topology."""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from contracts.hashing import stable_digest


READOUT_PROJECTION_KINDS = frozenset({"identity", "fixed_linear"})
QUERY_OPERATOR_KINDS = frozenset(
    {"coordinate", "linear_combination", "one_vs_rest_log_odds"}
)
CALIBRATION_PARAMETERIZATIONS = frozenset(
    {"post_query_positive_affine", "primitive_diagonal_affine", "shared_temperature"}
)
CALIBRATION_LOSS_FAMILIES = frozenset(
    {"independent_binary_bce", "multiclass_cross_entropy"}
)


def _unique_nonempty(values: tuple[str, ...], *, name: str) -> None:
    if not values or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")


@dataclass(frozen=True)
class ReadoutProjectionSpec:
    """Immutable map from represented state to primitive readout coordinates."""

    projection_id: str
    projection_kind: str
    source_axis_ids: tuple[str, ...]
    primitive_axis_ids: tuple[str, ...]
    source_hash: str
    weight_hash: str = ""
    intercept_hash: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.projection_id:
            raise ValueError("readout projection ID must be nonempty")
        if self.projection_kind not in READOUT_PROJECTION_KINDS:
            raise ValueError(f"unsupported readout projection kind {self.projection_kind!r}")
        _unique_nonempty(self.source_axis_ids, name="readout source axes")
        _unique_nonempty(self.primitive_axis_ids, name="primitive readout axes")
        if len(self.source_hash) != 64:
            raise ValueError("readout projection source hash must be SHA-256")
        if self.projection_kind == "identity":
            if self.source_axis_ids != self.primitive_axis_ids:
                raise ValueError("identity readout projection must preserve its ordered axes")
        elif len(self.weight_hash) != 64:
            raise ValueError("fixed linear readout projection requires a weight hash")
        if self.intercept_hash and len(self.intercept_hash) != 64:
            raise ValueError("readout projection intercept hash must be SHA-256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_readout_projection_spec",
            "schema_version": "1",
            "projection_id": self.projection_id,
            "projection_kind": self.projection_kind,
            "source_axis_ids": list(self.source_axis_ids),
            "primitive_axis_ids": list(self.primitive_axis_ids),
            "source_hash": self.source_hash,
            "weight_hash": self.weight_hash,
            "intercept_hash": self.intercept_hash,
            "notes": dict(self.notes),
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReadoutProjectionSpec":
        return cls(
            projection_id=str(payload["projection_id"]),
            projection_kind=str(payload["projection_kind"]),
            source_axis_ids=tuple(str(value) for value in payload["source_axis_ids"]),
            primitive_axis_ids=tuple(
                str(value) for value in payload["primitive_axis_ids"]
            ),
            source_hash=str(payload["source_hash"]),
            weight_hash=str(payload.get("weight_hash", "")),
            intercept_hash=str(payload.get("intercept_hash", "")),
            notes=dict(payload.get("notes", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReadoutProjectionSpec":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class QueryOperatorSpec:
    """Structural dependence of declared queries on primitive readout axes.

    ``support`` is authoritative structural support. Values close to zero are
    never thresholded to decide whether two calibration problems couple.
    """

    operator_id: str
    operator_kind: str
    primitive_axis_ids: tuple[str, ...]
    query_axis_ids: tuple[str, ...]
    support: tuple[tuple[str, ...], ...]
    source_hash: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.operator_id:
            raise ValueError("query operator ID must be nonempty")
        if self.operator_kind not in QUERY_OPERATOR_KINDS:
            raise ValueError(f"unsupported query operator kind {self.operator_kind!r}")
        _unique_nonempty(self.primitive_axis_ids, name="query primitive axes")
        _unique_nonempty(self.query_axis_ids, name="query axes")
        if len(self.support) != len(self.query_axis_ids):
            raise ValueError("query support must contain one row per query")
        primitive = set(self.primitive_axis_ids)
        for row in self.support:
            if not row or len(set(row)) != len(row) or not set(row).issubset(primitive):
                raise ValueError("query support contains empty, duplicate, or unknown axes")
        if len(self.source_hash) != 64:
            raise ValueError("query operator source hash must be SHA-256")
        if self.operator_kind == "coordinate":
            if any(len(row) != 1 for row in self.support):
                raise ValueError("coordinate query operators require singleton support")
        if self.operator_kind == "one_vs_rest_log_odds":
            if any(len(row) < 2 for row in self.support):
                raise ValueError("one-vs-rest queries require a target and competitors")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_query_operator_spec",
            "schema_version": "1",
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "primitive_axis_ids": list(self.primitive_axis_ids),
            "query_axis_ids": list(self.query_axis_ids),
            "support": [list(row) for row in self.support],
            "source_hash": self.source_hash,
            "parameters": dict(self.parameters),
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def coordinate(
        cls,
        *,
        operator_id: str,
        primitive_axis_ids: tuple[str, ...],
        query_axis_ids: tuple[str, ...] | None = None,
        query_primitive_axis_ids: tuple[str, ...] | None = None,
        source_hash: str,
    ) -> "QueryOperatorSpec":
        selected = (
            primitive_axis_ids
            if query_primitive_axis_ids is None
            else query_primitive_axis_ids
        )
        queries = selected if query_axis_ids is None else query_axis_ids
        if len(queries) != len(selected):
            raise ValueError("coordinate query and selected primitive axes must align")
        if not set(selected).issubset(set(primitive_axis_ids)):
            raise ValueError("coordinate query selects an unknown primitive axis")
        return cls(
            operator_id=operator_id,
            operator_kind="coordinate",
            primitive_axis_ids=primitive_axis_ids,
            query_axis_ids=queries,
            support=tuple((axis,) for axis in selected),
            source_hash=source_hash,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QueryOperatorSpec":
        return cls(
            operator_id=str(payload["operator_id"]),
            operator_kind=str(payload["operator_kind"]),
            primitive_axis_ids=tuple(
                str(value) for value in payload["primitive_axis_ids"]
            ),
            query_axis_ids=tuple(str(value) for value in payload["query_axis_ids"]),
            support=tuple(
                tuple(str(value) for value in row) for row in payload["support"]
            ),
            source_hash=str(payload["source_hash"]),
            parameters=dict(payload.get("parameters", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "QueryOperatorSpec":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CalibrationProblemSpec:
    """Claim-relevant calibration family and loss factorization."""

    problem_id: str
    readout_projection_hash: str
    query_operator_hash: str
    parameterization: str = "post_query_positive_affine"
    loss_family: str = "independent_binary_bce"
    weighting: str = "pos_weighted"
    calibration_axis_ids: tuple[str, ...] = tuple()
    fold_ids: tuple[str, ...] = tuple()
    minimum_scale: float = 1e-8
    notes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.problem_id:
            raise ValueError("calibration problem ID must be nonempty")
        if len(self.readout_projection_hash) != 64 or len(self.query_operator_hash) != 64:
            raise ValueError("calibration problem parents must be SHA-256 hashes")
        if self.parameterization not in CALIBRATION_PARAMETERIZATIONS:
            raise ValueError(f"unsupported calibration parameterization {self.parameterization!r}")
        if self.loss_family not in CALIBRATION_LOSS_FAMILIES:
            raise ValueError(f"unsupported calibration loss family {self.loss_family!r}")
        if self.weighting not in {"unweighted", "pos_weighted", "design_weighted"}:
            raise ValueError(f"unsupported calibration weighting {self.weighting!r}")
        if self.calibration_axis_ids and (
            any(not value for value in self.calibration_axis_ids)
            or len(set(self.calibration_axis_ids)) != len(self.calibration_axis_ids)
        ):
            raise ValueError("calibration coordinate IDs must be nonempty and unique")
        if self.fold_ids and (any(not value for value in self.fold_ids) or len(set(self.fold_ids)) != len(self.fold_ids)):
            raise ValueError("calibration fold IDs must be nonempty and unique")
        if not math.isfinite(float(self.minimum_scale)) or float(self.minimum_scale) < 0.0:
            raise ValueError("calibration minimum scale must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_name": "mechanistic_calibration_problem_spec",
            "schema_version": "1",
            "problem_id": self.problem_id,
            "readout_projection_hash": self.readout_projection_hash,
            "query_operator_hash": self.query_operator_hash,
            "parameterization": self.parameterization,
            "loss_family": self.loss_family,
            "weighting": self.weighting,
            "calibration_axis_ids": list(self.calibration_axis_ids),
            "fold_ids": list(self.fold_ids),
            "minimum_scale": float(self.minimum_scale),
            "notes": dict(self.notes),
        }

    @property
    def content_hash(self) -> str:
        return stable_digest(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationProblemSpec":
        return cls(
            problem_id=str(payload["problem_id"]),
            readout_projection_hash=str(payload["readout_projection_hash"]),
            query_operator_hash=str(payload["query_operator_hash"]),
            parameterization=str(
                payload.get("parameterization", "post_query_positive_affine")
            ),
            loss_family=str(
                payload.get("loss_family", "independent_binary_bce")
            ),
            weighting=str(payload.get("weighting", "pos_weighted")),
            calibration_axis_ids=tuple(
                str(value) for value in payload.get("calibration_axis_ids", ())
            ),
            fold_ids=tuple(str(value) for value in payload.get("fold_ids", ())),
            minimum_scale=float(payload.get("minimum_scale", 1e-8)),
            notes=dict(payload.get("notes", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProblemSpec":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "CALIBRATION_LOSS_FAMILIES",
    "CALIBRATION_PARAMETERIZATIONS",
    "QUERY_OPERATOR_KINDS",
    "READOUT_PROJECTION_KINDS",
    "CalibrationProblemSpec",
    "QueryOperatorSpec",
    "ReadoutProjectionSpec",
]
