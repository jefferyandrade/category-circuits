"""Neutral registered-runtime primitives for a genuine modern-v1 backend.

This module contains no traversal, partition, proposal, acceptance, or
terminal-validation control flow.  It only binds the declared physical route
axis, constructs an outcome-independent class-balanced development population, and
evaluates a parent/candidate hard-mask pair with one fixed affine calibration
over one common native panel stream.
"""

from __future__ import annotations

import hashlib
import math
import operator
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain, repeat
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Iterator, overload

import numpy as np

from contracts.hashing import stable_digest
from mechanistic_systems.circuits.torch_evaluator import TorchCircuitEvaluator
from mechanistic_systems.contracts import QuerySpec
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    compact_ordinal_route_axis,
    ordered_string_axis_hash,
)
from mechanistic_systems.core.storage import (
    RouteBlockSpec,
    ordered_block_route_axis_hash,
)
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.native_panel_data_plane import (
    NativePanelDataPlane,
)


_DECLARED_ROUTE_AXIS_SCHEMA: Final = "registered_declared_route_axis_v1"
_BALANCED_POPULATION_SCHEMA: Final = (
    "registered_class_balanced_sampling_population_v1"
)
_FIXED_CALIBRATION_SCHEMA: Final = "registered_fixed_affine_bce_calibration_v1"
_PAIRED_BCE_SCHEMA: Final = "registered_fixed_calibration_paired_bce_rows_v1"
_PAIRED_ADAPTER_SCHEMA: Final = "registered_fixed_calibration_bce_adapter_v1"
_DEVELOPMENT_SOURCE_ROLES: Final = frozenset({"development", "optimization"})
_NATIVE_EVALUATION_ROLE: Final = "v1_candidate_oracle"


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _nonempty(value, label).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _finite(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _array_payload_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return stable_digest(
        {
            "dtype": array.dtype.str,
            "shape": tuple(int(value) for value in array.shape),
            "payload_sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class DeclaredRouteLayerAxis(Sequence[str]):
    """O(blocks)-storage expansion of layer IDs over the physical route axis."""

    _layer_ids: tuple[str, ...] = field(repr=False)
    _stops: tuple[int, ...] = field(repr=False)
    _route_count: int

    def __init__(self, route_blocks: Sequence[RouteBlockSpec]) -> None:
        blocks = tuple(route_blocks)
        # Besides producing the physical identity, this validates every block
        # and proves that the declarations form one contiguous route axis.
        ordered_block_route_axis_hash(blocks)
        object.__setattr__(
            self,
            "_layer_ids",
            tuple(str(block.layer_id) for block in blocks),
        )
        object.__setattr__(self, "_stops", tuple(block.stop for block in blocks))
        object.__setattr__(self, "_route_count", int(blocks[-1].stop))

    def __len__(self) -> int:
        return self._route_count

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._route_count)
            return tuple(self[ordinal] for ordinal in range(start, stop, step))
        ordinal = operator.index(index)
        if ordinal < 0:
            ordinal += self._route_count
        if ordinal < 0 or ordinal >= self._route_count:
            raise IndexError("declared route-layer index out of range")
        return self._layer_ids[bisect_right(self._stops, ordinal)]

    def __iter__(self) -> Iterator[str]:
        starts = (0, *self._stops[:-1])
        return chain.from_iterable(
            repeat(layer_id, stop - start)
            for layer_id, start, stop in zip(
                self._layer_ids,
                starts,
                self._stops,
                strict=True,
            )
        )

    def __contains__(self, value: object) -> bool:
        return value in self._layer_ids

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeclaredRouteLayerAxis):
            return (
                self._layer_ids == other._layer_ids
                and self._stops == other._stops
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other, strict=True)
            )
        return NotImplemented

    @property
    def block_count(self) -> int:
        """Number of retained run-length records backing this sequence."""

        return len(self._layer_ids)

    def __repr__(self) -> str:
        return (
            "DeclaredRouteLayerAxis("
            f"route_count={self._route_count}, block_count={self.block_count})"
        )


def _route_ids_match_constructive_axis(
    values: Sequence[str],
    expected: ConstructiveOrdinalRouteAxis,
) -> bool:
    """Validate exact ordinal values without building a second route tuple."""

    if isinstance(values, ConstructiveOrdinalRouteAxis):
        return len(values) == len(expected)
    if isinstance(values, (str, bytes)) or len(values) != len(expected):
        return False
    return all(
        value == f"route:{ordinal}" for ordinal, value in enumerate(values)
    )


def _route_layers_match_declared_axis(
    values: Sequence[str],
    expected: DeclaredRouteLayerAxis,
) -> bool:
    """Validate exact layer expansion with O(1) additional route storage."""

    if isinstance(values, DeclaredRouteLayerAxis):
        return values == expected
    if isinstance(values, (str, bytes)) or len(values) != len(expected):
        return False
    return all(
        value == expected_value
        for value, expected_value in zip(values, expected, strict=True)
    )


@dataclass(frozen=True, slots=True)
class DeclaredRouteAxis:
    """Compact logical IDs bound to the executable physical block declaration."""

    route_blocks: tuple[RouteBlockSpec, ...]
    route_ids: Sequence[str]
    route_layer_ids: Sequence[str]
    route_axis_hash: str
    route_id_axis_hash: str
    schema: str = _DECLARED_ROUTE_AXIS_SCHEMA

    def __post_init__(self) -> None:
        blocks = tuple(self.route_blocks)
        declared_hash = ordered_block_route_axis_hash(blocks)
        if self.route_axis_hash != declared_hash:
            raise ValueError("declared route-axis hash differs from its route blocks")
        if len({str(block.layer_id) for block in blocks}) != len(blocks):
            raise ValueError("declared route blocks repeat a layer identity")
        route_count = blocks[-1].stop
        expected_ids = compact_ordinal_route_axis(route_count)
        if not _route_ids_match_constructive_axis(self.route_ids, expected_ids):
            raise ValueError("declared route IDs differ from the physical ordinal axis")
        expected_layers = DeclaredRouteLayerAxis(blocks)
        if not _route_layers_match_declared_axis(
            self.route_layer_ids,
            expected_layers,
        ):
            raise ValueError("route-layer IDs differ from declared block order")
        expected_id_hash = ordered_string_axis_hash(
            expected_ids,
            name="registered declared route IDs",
        )
        if self.route_id_axis_hash != expected_id_hash:
            raise ValueError("declared route-ID axis hash is not reproducible")
        if self.schema != _DECLARED_ROUTE_AXIS_SCHEMA:
            raise ValueError("unsupported declared route-axis schema")
        object.__setattr__(self, "route_blocks", blocks)
        object.__setattr__(self, "route_ids", expected_ids)
        object.__setattr__(self, "route_layer_ids", expected_layers)

    @property
    def route_count(self) -> int:
        return int(len(self.route_ids))

    @property
    def content_hash(self) -> str:
        return stable_digest(
            {
                "schema": self.schema,
                "route_axis_hash": self.route_axis_hash,
                "route_id_axis_hash": self.route_id_axis_hash,
                "route_count": self.route_count,
                "route_blocks": [block.to_dict() for block in self.route_blocks],
            }
        )


def resolve_declared_route_axis(source: Any) -> DeclaredRouteAxis:
    """Resolve the physical route axis from a runtime or real evaluator."""

    evaluator = getattr(source, "evaluator", source)
    blocks = tuple(getattr(evaluator, "route_blocks", ()))
    if not blocks or not all(isinstance(block, RouteBlockSpec) for block in blocks):
        raise TypeError("declared route-axis resolution requires RouteBlockSpec blocks")
    route_axis_hash = _sha256(
        getattr(evaluator, "route_schema_hash", ""),
        "evaluator route_schema_hash",
    )
    declared_hash = ordered_block_route_axis_hash(blocks)
    if route_axis_hash != declared_hash:
        raise ValueError("evaluator route schema differs from its declared route blocks")
    route_ids = compact_ordinal_route_axis(blocks[-1].stop)
    return DeclaredRouteAxis(
        route_blocks=blocks,
        route_ids=route_ids,
        route_layer_ids=DeclaredRouteLayerAxis(blocks),
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=ordered_string_axis_hash(
            route_ids,
            name="registered declared route IDs",
        ),
    )


def hard_mask_from_declared_layers(
    masks_by_layer: Mapping[str, np.ndarray],
    *,
    route_axis: DeclaredRouteAxis,
) -> HardMask:
    """Pack layer masks in declared block order, independent of walk order."""

    expected = {str(block.layer_id) for block in route_axis.route_blocks}
    if set(masks_by_layer) != expected:
        missing = sorted(expected - set(masks_by_layer))
        extra = sorted(set(masks_by_layer) - expected)
        raise ValueError(
            "layer masks differ from the declared route axis: "
            f"missing={missing!r} extra={extra!r}"
        )
    flat = np.empty(route_axis.route_count, dtype=bool)
    for block in route_axis.route_blocks:
        layer_id = str(block.layer_id)
        raw = np.asarray(masks_by_layer[layer_id])
        if raw.shape != block.shape:
            raise ValueError(f"mask for {layer_id!r} has the wrong shape")
        if raw.dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError(f"mask for {layer_id!r} must be a binary array")
        if not np.isfinite(raw).all() or not np.isin(raw, (0, 1)).all():
            raise ValueError(f"mask for {layer_id!r} must contain only zero or one")
        flat[block.offset : block.stop] = raw.reshape(-1, order="C").astype(
            bool,
            copy=False,
        )
    return HardMask.from_bool_array(
        flat,
        route_axis_hash=route_axis.route_axis_hash,
    )


def declared_layers_from_hard_mask(
    mask: HardMask,
    *,
    route_axis: DeclaredRouteAxis,
) -> Mapping[str, np.ndarray]:
    """Return read-only layer views for one physical-axis hard mask."""

    if (
        mask.route_axis_hash != route_axis.route_axis_hash
        or mask.route_count != route_axis.route_count
    ):
        raise ValueError("hard mask differs from the declared physical route axis")
    values = mask.to_bool_array()
    layers: dict[str, np.ndarray] = {}
    for block in route_axis.route_blocks:
        layer = values[block.offset : block.stop].reshape(block.shape, order="C")
        layer.setflags(write=False)
        layers[str(block.layer_id)] = layer
    return MappingProxyType(layers)


@dataclass(frozen=True, slots=True)
class ClassBalancedSamplingPopulation:
    """Outcome-independent 50/50 target-versus-complement development law."""

    source_context_axis_hash: str
    source_label_map_hash: str
    target_class_id: int
    source_role: str
    population_context_ids: tuple[str, ...]
    population_labels: tuple[int, ...]
    on_target_context_ids: tuple[str, ...]
    off_target_context_ids: tuple[str, ...]
    target_population_probabilities: tuple[float, ...]
    sampling_population_probabilities: tuple[float, ...]
    _label_by_context_id: Mapping[str, int] = field(init=False, repr=False, compare=False)
    _population_hash: str = field(init=False, repr=False, compare=False)
    _sampling_law_hash: str = field(init=False, repr=False, compare=False)
    terminal_outcomes_opened: bool = field(default=False, init=False)
    schema: str = _BALANCED_POPULATION_SCHEMA

    def __post_init__(self) -> None:
        axis_hash = _sha256(
            self.source_context_axis_hash,
            "source_context_axis_hash",
        )
        label_hash = _sha256(self.source_label_map_hash, "source_label_map_hash")
        if (
            isinstance(self.target_class_id, bool)
            or not isinstance(self.target_class_id, Integral)
            or int(self.target_class_id) < 0
        ):
            raise ValueError("target_class_id must be a nonnegative integer")
        role = _nonempty(self.source_role, "source_role")
        if role not in _DEVELOPMENT_SOURCE_ROLES:
            raise ValueError("class-balanced sampling is restricted to development data")
        contexts = tuple(
            _nonempty(value, "population context ID")
            for value in self.population_context_ids
        )
        labels = tuple(int(value) for value in self.population_labels)
        if not contexts or len(set(contexts)) != len(contexts) or len(labels) != len(contexts):
            raise ValueError("sampling population requires aligned unique contexts and labels")
        on_target = tuple(self.on_target_context_ids)
        off_target = tuple(self.off_target_context_ids)
        expected_on = tuple(
            context
            for context, label in zip(contexts, labels, strict=True)
            if label == int(self.target_class_id)
        )
        expected_off = tuple(
            context
            for context, label in zip(contexts, labels, strict=True)
            if label != int(self.target_class_id)
        )
        if not expected_on or not expected_off:
            raise ValueError("class-balanced sampling requires both target strata")
        if on_target != expected_on or off_target != expected_off:
            raise ValueError("sampling strata differ from source labels")
        expected_probabilities = tuple(
            (
                0.5 / len(expected_on)
                if label == int(self.target_class_id)
                else 0.5 / len(expected_off)
            )
            for label in labels
        )
        target = tuple(float(value) for value in self.target_population_probabilities)
        sampling = tuple(float(value) for value in self.sampling_population_probabilities)
        if target != expected_probabilities or sampling != expected_probabilities:
            raise ValueError("class-balanced target and sampling laws are not reproducible")
        if not math.isclose(
            math.fsum(target),
            1.0,
            rel_tol=0.0,
            abs_tol=32.0 * np.finfo(np.float64).eps,
        ):
            raise ValueError("class-balanced population probabilities must sum to one")
        if self.schema != _BALANCED_POPULATION_SCHEMA:
            raise ValueError("unsupported class-balanced population schema")
        object.__setattr__(self, "source_context_axis_hash", axis_hash)
        object.__setattr__(self, "source_label_map_hash", label_hash)
        object.__setattr__(self, "source_role", role)
        object.__setattr__(self, "population_context_ids", contexts)
        object.__setattr__(self, "population_labels", labels)
        object.__setattr__(self, "on_target_context_ids", expected_on)
        object.__setattr__(self, "off_target_context_ids", expected_off)
        object.__setattr__(self, "target_population_probabilities", target)
        object.__setattr__(self, "sampling_population_probabilities", sampling)
        object.__setattr__(
            self,
            "_label_by_context_id",
            MappingProxyType(dict(zip(contexts, labels, strict=True))),
        )
        population_hash = stable_digest(
            {
                "schema": self.schema,
                "source_context_axis_hash": axis_hash,
                "source_label_map_hash": label_hash,
                "target_class_id": int(self.target_class_id),
                "population_context_ids": contexts,
                "population_labels": labels,
            }
        )
        sampling_law_hash = stable_digest(
            {
                "population_hash": population_hash,
                "source_role": role,
                "target_population_probabilities": target,
                "sampling_population_probabilities": sampling,
                "terminal_outcomes_opened": False,
            }
        )
        object.__setattr__(self, "_population_hash", population_hash)
        object.__setattr__(self, "_sampling_law_hash", sampling_law_hash)

    @property
    def source_context_ids_by_stratum(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                "on_target": self.on_target_context_ids,
                "off_target": self.off_target_context_ids,
            }
        )

    @property
    def label_by_context_id(self) -> Mapping[str, int]:
        return self._label_by_context_id

    @property
    def population_hash(self) -> str:
        return self._population_hash

    @property
    def sampling_law_hash(self) -> str:
        return self._sampling_law_hash

    @property
    def content_hash(self) -> str:
        return self.sampling_law_hash


def build_class_balanced_sampling_population(
    axis: ContextAxis,
    *,
    target_class_id: int,
    source_role: str = "optimization",
) -> ClassBalancedSamplingPopulation:
    """Construct a class-balanced law using labels only, never responses."""

    if not isinstance(axis, ContextAxis):
        raise TypeError("class-balanced population construction requires ContextAxis")
    axis.validate()
    contexts = tuple(record.context_id for record in axis.records)
    labels = tuple(int(record.label) for record in axis.records)
    on_target = tuple(
        context
        for context, label in zip(contexts, labels, strict=True)
        if label == int(target_class_id)
    )
    off_target = tuple(
        context
        for context, label in zip(contexts, labels, strict=True)
        if label != int(target_class_id)
    )
    if not on_target or not off_target:
        raise ValueError("class-balanced sampling requires target and complement contexts")
    probabilities = tuple(
        0.5 / len(on_target) if label == int(target_class_id) else 0.5 / len(off_target)
        for label in labels
    )
    return ClassBalancedSamplingPopulation(
        source_context_axis_hash=axis.content_hash,
        source_label_map_hash=axis.label_map_hash,
        target_class_id=int(target_class_id),
        source_role=source_role,
        population_context_ids=contexts,
        population_labels=labels,
        on_target_context_ids=on_target,
        off_target_context_ids=off_target,
        target_population_probabilities=probabilities,
        sampling_population_probabilities=probabilities,
    )


@dataclass(frozen=True, slots=True)
class FixedAffineBCECalibration:
    """One immutable affine calibration fitted before candidate evaluation."""

    query_id: str
    query_hash: str
    target_class_id: int
    scale: float
    offset: float
    calibration_state_id: str
    calibration_source_hash: str
    calibration_context_axis_hash: str
    source_label_map_hash: str
    model_state_hash: str
    terminal_outcomes_opened: bool = field(default=False, init=False)
    schema: str = _FIXED_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _nonempty(self.query_id, "query_id"))
        object.__setattr__(self, "query_hash", _sha256(self.query_hash, "query_hash"))
        if (
            isinstance(self.target_class_id, bool)
            or not isinstance(self.target_class_id, Integral)
            or int(self.target_class_id) < 0
        ):
            raise ValueError("target_class_id must be a nonnegative integer")
        scale = _finite(self.scale, "calibration scale")
        if scale <= 0.0:
            raise ValueError("fixed affine BCE calibration requires positive scale")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", _finite(self.offset, "calibration offset"))
        object.__setattr__(
            self,
            "calibration_state_id",
            _nonempty(self.calibration_state_id, "calibration_state_id"),
        )
        for name in (
            "calibration_source_hash",
            "calibration_context_axis_hash",
            "source_label_map_hash",
            "model_state_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.schema != _FIXED_CALIBRATION_SCHEMA:
            raise ValueError("unsupported fixed affine BCE calibration schema")

    @property
    def content_hash(self) -> str:
        return stable_digest(
            {
                "schema": self.schema,
                "query_id": self.query_id,
                "query_hash": self.query_hash,
                "target_class_id": int(self.target_class_id),
                "scale": self.scale,
                "offset": self.offset,
                "calibration_state_id": self.calibration_state_id,
                "calibration_source_hash": self.calibration_source_hash,
                "calibration_context_axis_hash": self.calibration_context_axis_hash,
                "source_label_map_hash": self.source_label_map_hash,
                "model_state_hash": self.model_state_hash,
                "terminal_outcomes_opened": False,
            }
        )


def fixed_calibration_bce_rows(
    logits: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    calibration: FixedAffineBCECalibration,
) -> np.ndarray:
    """Evaluate ordinary per-row Bernoulli log loss under one fixed fit."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels).reshape(-1)
    if values.shape != label_values.shape or not values.size:
        raise ValueError("fixed-calibration logits and labels must align")
    if not np.isfinite(values).all() or not np.isfinite(label_values).all():
        raise ValueError("fixed-calibration logits and labels must be finite")
    is_target = (
        label_values.astype(np.int64, copy=False)
        == int(calibration.target_class_id)
    )
    calibrated = calibration.scale * values + calibration.offset
    # The algebraically equivalent ``softplus(eta) - y * eta`` loses the
    # positive tail to catastrophic cancellation for target rows with large
    # positive eta.  Selecting the label-specific softplus keeps ordinary BCE
    # stable and matches the analytic finite-support transport used by the
    # paired oracle.
    rows = np.where(
        is_target,
        np.logaddexp(0.0, -calibrated),
        np.logaddexp(0.0, calibrated),
    )
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    if not np.isfinite(rows).all():
        raise ValueError("fixed-calibration BCE produced nonfinite rows")
    rows.setflags(write=False)
    return rows


@dataclass(frozen=True, slots=True)
class FixedCalibrationPairedBCERows:
    """Immutable common-context parent/candidate BCE observations."""

    source_context_ids: tuple[str, ...]
    evaluated_context_ids: tuple[str, ...]
    labels: np.ndarray = field(repr=False)
    parent_logits: np.ndarray = field(repr=False)
    candidate_logits: np.ndarray = field(repr=False)
    parent_bce_rows: np.ndarray = field(repr=False)
    candidate_bce_rows: np.ndarray = field(repr=False)
    parent_mask_hash: str
    candidate_mask_hash: str
    route_axis_hash: str
    population_hash: str
    calibration_hash: str
    adapter_hash: str
    common_context_pairing: bool = field(default=True, init=False)
    terminal_outcomes_opened: bool = field(default=False, init=False)
    schema: str = _PAIRED_BCE_SCHEMA

    def __post_init__(self) -> None:
        source_ids = tuple(
            _nonempty(value, "source context ID")
            for value in self.source_context_ids
        )
        evaluated_ids = tuple(
            _nonempty(value, "evaluated context ID") for value in self.evaluated_context_ids
        )
        if not source_ids or not evaluated_ids or len(set(evaluated_ids)) != len(evaluated_ids):
            raise ValueError("paired BCE rows require a nonempty unique evaluated axis")
        if not set(source_ids).issubset(set(evaluated_ids)):
            raise ValueError("paired BCE source rows differ from their evaluated contexts")
        row_count = len(source_ids)
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in (
            ("labels", np.int64),
            ("parent_logits", np.float64),
            ("candidate_logits", np.float64),
            ("parent_bce_rows", np.float64),
            ("candidate_bce_rows", np.float64),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True, order="C").reshape(-1)
            if value.shape != (row_count,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain one finite value per source row")
            value.setflags(write=False)
            arrays[name] = value
        for name in (
            "parent_mask_hash",
            "candidate_mask_hash",
            "route_axis_hash",
            "population_hash",
            "calibration_hash",
            "adapter_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.schema != _PAIRED_BCE_SCHEMA:
            raise ValueError("unsupported fixed-calibration paired-BCE schema")
        object.__setattr__(self, "source_context_ids", source_ids)
        object.__setattr__(self, "evaluated_context_ids", evaluated_ids)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def damage_rows(self) -> np.ndarray:
        result = np.ascontiguousarray(
            self.candidate_bce_rows - self.parent_bce_rows,
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    @property
    def paired_bce_values(self) -> np.ndarray:
        result = np.column_stack((self.parent_bce_rows, self.candidate_bce_rows))
        result.setflags(write=False)
        return result

    @property
    def content_hash(self) -> str:
        return stable_digest(
            {
                "schema": self.schema,
                "source_context_ids": self.source_context_ids,
                "evaluated_context_ids": self.evaluated_context_ids,
                "labels_hash": _array_payload_hash(self.labels),
                "parent_logits_hash": _array_payload_hash(self.parent_logits),
                "candidate_logits_hash": _array_payload_hash(self.candidate_logits),
                "parent_bce_rows_hash": _array_payload_hash(self.parent_bce_rows),
                "candidate_bce_rows_hash": _array_payload_hash(self.candidate_bce_rows),
                "parent_mask_hash": self.parent_mask_hash,
                "candidate_mask_hash": self.candidate_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "population_hash": self.population_hash,
                "calibration_hash": self.calibration_hash,
                "adapter_hash": self.adapter_hash,
                "common_context_pairing": True,
                "terminal_outcomes_opened": False,
            }
        )


class FixedCalibrationPairedBCEAdapter:
    """Real Torch/native-panel adapter for paired fixed-calibration BCE rows."""

    schema = _PAIRED_ADAPTER_SCHEMA
    native_evaluation_role = _NATIVE_EVALUATION_ROLE
    terminal_outcomes_opened = False

    def __init__(
        self,
        *,
        evaluator: TorchCircuitEvaluator,
        data_plane: NativePanelDataPlane,
        route_axis: DeclaredRouteAxis,
        population: ClassBalancedSamplingPopulation,
        calibration: FixedAffineBCECalibration,
    ) -> None:
        if not isinstance(evaluator, TorchCircuitEvaluator):
            raise TypeError("paired BCE adapter requires TorchCircuitEvaluator")
        if not isinstance(data_plane, NativePanelDataPlane):
            raise TypeError("paired BCE adapter requires NativePanelDataPlane")
        if not isinstance(route_axis, DeclaredRouteAxis):
            raise TypeError("paired BCE adapter requires DeclaredRouteAxis")
        if not isinstance(population, ClassBalancedSamplingPopulation):
            raise TypeError("paired BCE adapter requires a class-balanced population")
        if not isinstance(calibration, FixedAffineBCECalibration):
            raise TypeError("paired BCE adapter requires fixed affine calibration")
        if (
            evaluator.route_schema_hash != route_axis.route_axis_hash
            or tuple(evaluator.route_blocks) != route_axis.route_blocks
        ):
            raise ValueError("evaluator differs from the declared route axis")
        axis = data_plane.axis
        if (
            axis.content_hash != population.source_context_axis_hash
            or axis.label_map_hash != population.source_label_map_hash
        ):
            raise ValueError("native data plane differs from the sampling population")
        if data_plane.model_state_hash != calibration.model_state_hash:
            raise ValueError("native data plane and fixed calibration bind different models")
        if population.source_label_map_hash != calibration.source_label_map_hash:
            raise ValueError("sampling population and calibration bind different label maps")
        query_by_id = {query.query_id: query for query in evaluator.queries}
        query = query_by_id.get(calibration.query_id)
        if query is None:
            raise ValueError("fixed calibration names an unknown evaluator query")
        target_class_id = query.evaluator.get(
            "class_id",
            query.evaluator.get("unit", query.evaluator.get("coordinate")),
        )
        if target_class_id is None or int(target_class_id) != int(calibration.target_class_id):
            raise ValueError("fixed calibration differs from the evaluator query coordinate")
        if query.content_hash != calibration.query_hash:
            raise ValueError("fixed calibration differs from the sealed query identity")
        if int(population.target_class_id) != int(calibration.target_class_id):
            raise ValueError("sampling population and fixed calibration target different classes")
        self.evaluator = evaluator
        self.data_plane = data_plane
        self.route_axis = route_axis
        self.population = population
        self.calibration = calibration
        self.query = query
        self._adapter_hash = stable_digest(
            {
                "schema": self.schema,
                "evaluator_id": evaluator.evaluator_id,
                "evaluator_topology_hash": evaluator.topology_hash,
                "evaluator_route_schema_hash": evaluator.route_schema_hash,
                "evaluator_parity_receipt_hash": evaluator.parity_receipt_hash,
                "query_hash": query.content_hash,
                "native_data_plane_source_id": data_plane.source_id,
                "native_context_axis_hash": axis.content_hash,
                "native_preprocessing_hash": axis.preprocessing.content_hash,
                "model_state_hash": data_plane.model_state_hash,
                "route_axis_binding_hash": route_axis.content_hash,
                "population_hash": population.population_hash,
                "sampling_law_hash": population.sampling_law_hash,
                "calibration_hash": calibration.content_hash,
                "native_evaluation_role": self.native_evaluation_role,
                "common_context_pairing": True,
                "terminal_outcomes_opened": False,
            }
        )

    @property
    def adapter_hash(self) -> str:
        return self._adapter_hash

    def evaluate_pair(
        self,
        parent_mask: HardMask,
        candidate_mask: HardMask,
        source_context_ids: Sequence[str],
    ) -> FixedCalibrationPairedBCERows:
        """Evaluate both masks in one fused pass over one common panel stream."""

        requested = tuple(_nonempty(value, "source context ID") for value in source_context_ids)
        if not requested:
            raise ValueError("paired BCE evaluation requires source contexts")
        population_set = set(self.population.population_context_ids)
        if any(value not in population_set for value in requested):
            raise ValueError("paired BCE evaluation includes a context outside its population")
        for name, mask in (("parent", parent_mask), ("candidate", candidate_mask)):
            if not isinstance(mask, HardMask):
                raise TypeError(f"{name} mask must be HardMask")
            if (
                mask.route_axis_hash != self.route_axis.route_axis_hash
                or mask.route_count != self.route_axis.route_count
            ):
                raise ValueError(f"{name} mask differs from the declared route axis")
        # This also refuses route reactivation.  The result need not be
        # materialized here; validating monotone descent is the purpose.
        candidate_mask.removed_index_set_from(parent_mask)
        # ``declared_layers_from_hard_mask`` intentionally exposes immutable
        # evidence views.  Give Torch owned writable arrays at its tensor
        # boundary so ``torch.as_tensor`` never aliases a read-only NumPy
        # allocation.
        parent_layers = declared_layers_from_hard_mask(
            parent_mask,
            route_axis=self.route_axis,
        )
        candidate_layers = declared_layers_from_hard_mask(
            candidate_mask,
            route_axis=self.route_axis,
        )
        parent_compiled = self.evaluator.compile_mask(
            {name: np.array(value, copy=True) for name, value in parent_layers.items()}
        )
        candidate_compiled = self.evaluator.compile_mask(
            {
                name: np.array(value, copy=True)
                for name, value in candidate_layers.items()
            }
        )
        evaluated = tuple(dict.fromkeys(requested))
        batches = self.data_plane.panel_batches(
            evaluated,
            role=self.native_evaluation_role,
        )
        responses = self.evaluator.readout_responses_from_batches(
            batches,
            compiled_masks=(parent_compiled, candidate_compiled),
            lane_keys=(parent_mask.mask_hash, candidate_mask.mask_hash),
        )
        if len(responses) != 2:
            raise RuntimeError("paired evaluator did not return exactly two mask lanes")
        parent_response, candidate_response = responses
        if (
            tuple(parent_response.context_ids) != evaluated
            or tuple(candidate_response.context_ids) != evaluated
        ):
            raise RuntimeError("paired evaluator changed the common context order")
        parent_values = np.asarray(parent_response.values, dtype=np.float64)
        candidate_values = np.asarray(candidate_response.values, dtype=np.float64)
        if parent_values.shape != candidate_values.shape or parent_values.ndim != 2:
            raise RuntimeError("paired evaluator returned incompatible readout matrices")
        target = int(self.calibration.target_class_id)
        if target >= parent_values.shape[1]:
            raise ValueError("target class is outside the model readout axis")
        labels_value = (
            parent_response.labels
            if parent_response.labels is not None
            else parent_response.device_labels.detach().cpu().numpy()
        )
        candidate_labels_value = (
            candidate_response.labels
            if candidate_response.labels is not None
            else candidate_response.device_labels.detach().cpu().numpy()
        )
        labels = np.asarray(labels_value, dtype=np.int64).reshape(-1)
        candidate_labels = np.asarray(candidate_labels_value, dtype=np.int64).reshape(-1)
        if labels.shape != (len(evaluated),) or not np.array_equal(labels, candidate_labels):
            raise RuntimeError("paired evaluator returned misaligned class labels")
        expected_labels = self.population.label_by_context_id
        if any(
            int(label) != expected_labels[context]
            for context, label in zip(evaluated, labels, strict=True)
        ):
            raise ValueError("native panel labels differ from the sealed sampling population")
        parent_logits_unique = parent_values[:, target]
        candidate_logits_unique = candidate_values[:, target]
        parent_bce_unique = fixed_calibration_bce_rows(
            parent_logits_unique,
            labels,
            calibration=self.calibration,
        )
        candidate_bce_unique = fixed_calibration_bce_rows(
            candidate_logits_unique,
            labels,
            calibration=self.calibration,
        )
        index_by_context = {context: index for index, context in enumerate(evaluated)}
        indices = np.fromiter(
            (index_by_context[context] for context in requested),
            dtype=np.int64,
            count=len(requested),
        )
        return FixedCalibrationPairedBCERows(
            source_context_ids=requested,
            evaluated_context_ids=evaluated,
            labels=labels[indices],
            parent_logits=parent_logits_unique[indices],
            candidate_logits=candidate_logits_unique[indices],
            parent_bce_rows=parent_bce_unique[indices],
            candidate_bce_rows=candidate_bce_unique[indices],
            parent_mask_hash=parent_mask.mask_hash,
            candidate_mask_hash=candidate_mask.mask_hash,
            route_axis_hash=self.route_axis.route_axis_hash,
            population_hash=self.population.population_hash,
            calibration_hash=self.calibration.content_hash,
            adapter_hash=self.adapter_hash,
        )

    __call__ = evaluate_pair


__all__ = [
    "ClassBalancedSamplingPopulation",
    "DeclaredRouteAxis",
    "DeclaredRouteLayerAxis",
    "FixedAffineBCECalibration",
    "FixedCalibrationPairedBCEAdapter",
    "FixedCalibrationPairedBCERows",
    "build_class_balanced_sampling_population",
    "declared_layers_from_hard_mask",
    "fixed_calibration_bce_rows",
    "hard_mask_from_declared_layers",
    "resolve_declared_route_axis",
]
