"""Canonical geometric action objects for modern v1 pruning.

This module separates three quantities that are easy to collapse by accident:

* v0-continuous salience, ``A_j = E[|a_corr_j(X)|]``;
* signed corrected first-order damage, ``c_j = E[a_corr_j(X)]``; and
* finite-action metric energy, ``z.T (B.T B + diag(d)) z``.

Layer IDs are accepted only for diagnostics.  They never define selection
queues, quotas, or normalization factors in this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)

from .contracts import content_hash
from .modern_v1_conserving_attribution import (
    CONSERVING_ATTRIBUTION_EQUATION_ID,
    V1ExactGateAttributionFields,
)


GEOMETRIC_ACTION_CHART_SCHEMA: Final = "partition_free_v1_geometric_action_chart_v1"
CORRECTED_ATTRIBUTION_FIELDS_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_fields_v1"
)
ENERGY_DAMAGE_CALIBRATION_SCHEMA: Final = (
    "partition_free_v1_metric_energy_to_bce_damage_calibration_v1"
)
ACTION_EVALUATION_SCHEMA: Final = "partition_free_v1_geometric_action_evaluation_v1"
ACTION_MARGINAL_SCHEMA: Final = "partition_free_v1_geometric_action_marginal_v1"
FIXED_CARDINALITY_SELECTION_SCHEMA: Final = (
    "partition_free_v1_fixed_cardinality_geometric_selection_v1"
)
FIXED_CARDINALITY_POLICY_SCHEMA: Final = (
    "partition_free_v1_fixed_cardinality_dual_policy_v1"
)
DENSE_REFERENCE_MAX_ELEMENTS: Final = 20_000_000
METRIC_EQUATION_ID: Final = "fd_action_metric_BtB_plus_residual_diagonal_v1"
ACTION_EQUATION_ID: Final = (
    "signed_corrected_linear_plus_metric_energy_no_half_cross_terms_v1"
)
MARGINAL_EQUATION_ID: Final = (
    "c_j_plus_d_j_plus_bj_norm_sq_plus_two_bj_dot_current_q_v1"
)
TRIANGLE_BOUND_EQUATION_ID: Final = (
    "sqrt_zMz_le_sum_singleton_metric_lengths_psd_triangle_v1"
)
DUAL_LOWER_BOUND_EQUATION_ID: Final = (
    "fixed_k_sum_smallest_c_plus_d_plus_2lambda_dot_b_minus_lambda_norm_sq_v1"
)

CorrectionGranularity = Literal["route_resolved", "layer_broadcast"]
EnergyDamageMappingKind = Literal["identity", "fitted_linear"]
ActionStorageMode = Literal["dense_reference", "chunked_sufficient_statistics"]


class CorrectionAuthority(str, Enum):
    """Authority attached to a corrected attribution field."""

    ROUTE_RESOLVED_PRODUCTION = "route_resolved_production_v1"
    ROUTE_RESOLVED_SURROGATE_BIAS = "route_resolved_surrogate_bias_v1"
    LAYER_BROADCAST_DIAGNOSTIC = "layer_broadcast_diagnostic_v1"


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        payload = array.tobytes(order="C")
        return np.frombuffer(payload, dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _readonly_f64_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    size: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} does not align with the route axis")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and nonempty")
    if nonnegative and bool(np.any(array < -1e-12)):
        raise ValueError(f"{name} must be nonnegative")
    if nonnegative and bool(np.any(array < 0.0)):
        array = np.maximum(array, 0.0)
    owner: object = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray) and owner.base is not None:
        if id(owner) in seen:  # pragma: no cover - defensive cycle guard
            break
        seen.add(id(owner))
        owner = owner.base
    if (
        not array.flags.writeable
        and array.flags.c_contiguous
        and array.dtype == np.dtype("<f8")
        and (
            isinstance(owner, bytes)
            or (isinstance(owner, memoryview) and owner.readonly)
        )
    ):
        return array
    payload = np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    return np.frombuffer(payload, dtype="<f8")


def _readonly_f64_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    columns: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix on the route axis")
    payload = np.ascontiguousarray(array, dtype="<f8").tobytes(order="C")
    return np.frombuffer(payload, dtype="<f8").reshape(array.shape)


def _normalized_indices(indices: Sequence[int] | np.ndarray, route_count: int) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise TypeError("action indices must be integer route indices")
    raw = tuple(np.asarray(indices, dtype=object).reshape(-1).tolist())
    if not raw:
        raise ValueError("action must contain at least one route")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise TypeError("action indices must be exact integers")
    ordered = tuple(sorted(int(value) for value in raw))
    if len(set(ordered)) != len(ordered) or any(
        value < 0 or value >= route_count for value in ordered
    ):
        raise ValueError("action indices must be unique and on the route axis")
    return ordered


def _weights(
    weights: Sequence[float] | np.ndarray | None,
    *,
    rows: int,
) -> np.ndarray:
    if weights is None:
        return np.full(rows, 1.0 / float(rows), dtype=np.float64)
    array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if array.shape != (rows,) or not np.isfinite(array).all() or bool(np.any(array < 0.0)):
        raise ValueError("sample weights must be finite nonnegative and row-aligned")
    total = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("sample weights must have positive mass")
    return np.asarray(array / total, dtype=np.float64)


def _check_correction_authority(
    *,
    granularity: CorrectionGranularity,
    authority: CorrectionAuthority,
) -> None:
    if granularity not in {"route_resolved", "layer_broadcast"}:
        raise ValueError("unsupported correction granularity")
    if type(authority) is not CorrectionAuthority:
        authority = CorrectionAuthority(authority)
    if granularity == "layer_broadcast" and authority is not (
        CorrectionAuthority.LAYER_BROADCAST_DIAGNOSTIC
    ):
        raise PermissionError(
            "layer-broadcast correction is diagnostic-only and cannot be used as production authority"
        )
    if authority in {
        CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION,
        CorrectionAuthority.ROUTE_RESOLVED_SURROGATE_BIAS,
    } and granularity != "route_resolved":
        raise PermissionError("route-resolved correction authority requires route-resolved correction")


@dataclass(frozen=True, slots=True)
class V1MetricEnergyToBCEDamageCalibration:
    """Typed scalar map from action metric energy into BCE damage units."""

    gamma: float
    source_energy_unit: str
    target_damage_unit: str
    calibration_identity: str
    fit_receipt_hash: str
    mapping_kind: EnergyDamageMappingKind = "identity"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = ENERGY_DAMAGE_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        gamma = _finite(self.gamma, "gamma", positive=True)
        if self.mapping_kind not in {"identity", "fitted_linear"}:
            raise ValueError("unsupported energy-to-damage mapping kind")
        source = _nonempty(self.source_energy_unit, "source_energy_unit")
        target = _nonempty(self.target_damage_unit, "target_damage_unit")
        if self.mapping_kind == "identity":
            if not math.isclose(gamma, 1.0, rel_tol=0.0, abs_tol=0.0):
                raise ValueError("identity energy calibration requires gamma == 1")
            if source != target:
                raise ValueError("identity energy calibration requires matching units")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(
            self,
            "calibration_identity",
            _nonempty(self.calibration_identity, "calibration_identity"),
        )
        object.__setattr__(
            self,
            "fit_receipt_hash",
            _nonempty(self.fit_receipt_hash, "fit_receipt_hash"),
        )
        object.__setattr__(self, "source_energy_unit", source)
        object.__setattr__(self, "target_damage_unit", target)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "calibration metadata"))
        if self.schema != ENERGY_DAMAGE_CALIBRATION_SCHEMA:
            raise ValueError("unsupported energy-damage calibration schema")

    @property
    def calibration_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-energy-damage-calibration-")

    def energy_to_damage(self, energy: float) -> float:
        value = _finite(energy, "metric energy")
        if value < -1e-12:
            raise ValueError("metric energy cannot be negative")
        return float(self.gamma * max(0.0, value))


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionFields:
    """Separated corrected salience and signed corrected damage on one route axis."""

    route_ids: tuple[str, ...]
    v0_corrected_salience: np.ndarray = field(repr=False, compare=False)
    signed_corrected_damage: np.ndarray = field(repr=False, compare=False)
    sample_count: int
    source_attribution_hash: str
    correction_granularity: CorrectionGranularity
    correction_authority: CorrectionAuthority
    correction_applied_before_abs: bool = True
    sample_weight_hash: str | None = None
    storage_mode: ActionStorageMode = "dense_reference"
    peak_copy_mode: str = "bytes_backed_dense_reference"
    target_scale_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = CORRECTED_ATTRIBUTION_FIELDS_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(self.route_ids, name="v1 corrected attribution route axis")
        count = len(routes)
        salience = _readonly_f64_vector(
            self.v0_corrected_salience,
            name="v0 corrected salience",
            size=count,
            nonnegative=True,
        )
        signed = _readonly_f64_vector(
            self.signed_corrected_damage,
            name="signed corrected damage",
            size=count,
        )
        sample_count = _positive_int(self.sample_count, "sample_count")
        authority = CorrectionAuthority(self.correction_authority)
        _check_correction_authority(
            granularity=self.correction_granularity,
            authority=authority,
        )
        if type(self.correction_applied_before_abs) is not bool or not (
            self.correction_applied_before_abs
        ):
            raise ValueError("corrected salience must apply correction before absolute value")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "v0_corrected_salience", salience)
        object.__setattr__(self, "signed_corrected_damage", signed)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(
            self,
            "source_attribution_hash",
            _nonempty(self.source_attribution_hash, "source_attribution_hash"),
        )
        object.__setattr__(self, "correction_authority", authority)
        if self.sample_weight_hash is not None:
            object.__setattr__(
                self,
                "sample_weight_hash",
                _nonempty(self.sample_weight_hash, "sample_weight_hash"),
            )
        if self.storage_mode not in {"dense_reference", "chunked_sufficient_statistics"}:
            raise ValueError("unsupported corrected-attribution storage mode")
        object.__setattr__(self, "peak_copy_mode", _nonempty(self.peak_copy_mode, "peak_copy_mode"))
        if type(self.target_scale_authority) is not bool:
            raise TypeError("target_scale_authority must be bool")
        if self.target_scale_authority and self.storage_mode != "chunked_sufficient_statistics":
            raise PermissionError(
                "dense corrected-attribution helpers cannot carry target-scale authority"
            )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "attribution metadata"))
        if self.schema != CORRECTED_ATTRIBUTION_FIELDS_SCHEMA:
            raise ValueError("unsupported corrected-attribution field schema")

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(self.route_ids, name="v1 corrected attribution route axis")

    @property
    def production_eligible(self) -> bool:
        return (
            self.correction_granularity == "route_resolved"
            and self.correction_authority is CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION
        )

    @property
    def fields_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-corrected-attribution-fields-")


def corrected_attribution_fields_from_samples(
    *,
    route_ids: Sequence[str],
    raw_attribution_samples: Sequence[Sequence[float]] | np.ndarray,
    additive_correction: Sequence[float] | np.ndarray,
    correction_granularity: CorrectionGranularity = "route_resolved",
    correction_authority: CorrectionAuthority = CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    max_dense_elements: int = DENSE_REFERENCE_MAX_ELEMENTS,
    metadata: Mapping[str, Any] | None = None,
) -> V1CorrectedAttributionFields:
    """Dense reference helper: apply correction before ``abs`` and aggregation.

    This helper is bounded and reference-only.  Production-scale AlexNet should
    use chunked sufficient statistics and keep raw context-by-route blocks out
    of durable control-plane objects.
    """

    routes = canonical_unique_string_axis(route_ids, name="v1 corrected attribution route axis")
    samples = np.asarray(raw_attribution_samples, dtype=np.float64)
    limit = _positive_int(max_dense_elements, "max_dense_elements")
    if samples.ndim != 2 or samples.shape[1] != len(routes) or not np.isfinite(samples).all():
        raise ValueError("raw attribution samples must be finite and route-aligned")
    if int(samples.size) > limit:
        raise MemoryError("dense corrected-attribution helper exceeds its reference element limit")
    correction = np.asarray(additive_correction, dtype=np.float64)
    if correction.ndim == 1:
        if correction.shape != (len(routes),):
            raise ValueError("route correction vector does not align with route axis")
        corrected = samples + correction.reshape(1, -1)
    elif correction.ndim == 2:
        if correction.shape != samples.shape:
            raise ValueError("sample correction matrix does not align with samples")
        corrected = samples + correction
    else:
        raise ValueError("additive correction must be a route vector or sample matrix")
    if not np.isfinite(corrected).all():
        raise ValueError("corrected attribution samples are nonfinite")
    weights = _weights(sample_weights, rows=samples.shape[0])
    salience = np.sum(np.abs(corrected) * weights[:, None], axis=0, dtype=np.float64)
    signed = np.sum(corrected * weights[:, None], axis=0, dtype=np.float64)
    source_hash = content_hash(
        {
            "route_ids": routes,
            "raw_attribution_samples": np.asarray(samples, dtype=np.float64),
            "additive_correction": np.asarray(correction, dtype=np.float64),
            "correction_applied_before_abs": True,
        },
        prefix="partition-free-v1-corrected-attribution-source-",
    )
    weight_hash = (
        None
        if sample_weights is None
        else content_hash(weights, prefix="partition-free-v1-corrected-attribution-weights-")
    )
    payload_metadata = {
        "constructor": "dense_reference_samples",
        "dense_element_count": int(samples.size),
        "max_dense_elements": int(limit),
        "target_scale_claim_authority": False,
    }
    if metadata is not None:
        payload_metadata.update(dict(metadata))
    return V1CorrectedAttributionFields(
        route_ids=routes,
        v0_corrected_salience=salience,
        signed_corrected_damage=signed,
        sample_count=int(samples.shape[0]),
        source_attribution_hash=source_hash,
        correction_granularity=correction_granularity,
        correction_authority=CorrectionAuthority(correction_authority),
        sample_weight_hash=weight_hash,
        storage_mode="dense_reference",
        peak_copy_mode="dense_context_route_matrix_plus_correction",
        target_scale_authority=False,
        metadata=payload_metadata,
    )


def corrected_attribution_fields_from_sufficient_statistics(
    *,
    route_ids: Sequence[str],
    corrected_abs_salience_weighted_sum: Sequence[float] | np.ndarray,
    signed_corrected_damage_weighted_sum: Sequence[float] | np.ndarray,
    total_weight: float,
    sample_count: int,
    source_attribution_hash: str,
    correction_granularity: CorrectionGranularity = "route_resolved",
    correction_authority: CorrectionAuthority = CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION,
    sample_weight_hash: str | None = None,
    target_scale_authority: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> V1CorrectedAttributionFields:
    """Build corrected fields from route-chunked sufficient statistics."""

    routes = canonical_unique_string_axis(route_ids, name="v1 corrected attribution route axis")
    weight = _finite(total_weight, "total_weight", positive=True)
    count = len(routes)
    # These are ephemeral finalization inputs, not hash-bearing state.  Check
    # them in place and let ``V1CorrectedAttributionFields`` create the sole
    # immutable normalized snapshot; freezing each sum first would copy two
    # full census-scale route vectors only to divide and copy them again.
    abs_sum = np.asarray(
        corrected_abs_salience_weighted_sum,
        dtype=np.float64,
    ).reshape(-1)
    signed_sum = np.asarray(
        signed_corrected_damage_weighted_sum,
        dtype=np.float64,
    ).reshape(-1)
    if (
        abs_sum.shape != (count,)
        or signed_sum.shape != (count,)
        or not np.isfinite(abs_sum).all()
        or not np.isfinite(signed_sum).all()
        or bool(np.any(abs_sum < -1e-12))
    ):
        raise ValueError(
            "corrected sufficient statistics must be finite and route aligned"
        )
    payload_metadata = {
        "constructor": "chunked_sufficient_statistics",
        "correction_before_abs": True,
        "raw_context_route_matrix_materialized": False,
    }
    if metadata is not None:
        payload_metadata.update(dict(metadata))
    return V1CorrectedAttributionFields(
        route_ids=routes,
        v0_corrected_salience=np.asarray(abs_sum, dtype=np.float64) / weight,
        signed_corrected_damage=np.asarray(signed_sum, dtype=np.float64) / weight,
        sample_count=sample_count,
        source_attribution_hash=source_attribution_hash,
        correction_granularity=correction_granularity,
        correction_authority=CorrectionAuthority(correction_authority),
        sample_weight_hash=sample_weight_hash,
        storage_mode="chunked_sufficient_statistics",
        peak_copy_mode="route_chunked_sufficient_statistics",
        target_scale_authority=target_scale_authority,
        metadata=payload_metadata,
    )


def corrected_attribution_fields_from_exact_gate(
    *,
    gate_fields: V1ExactGateAttributionFields,
    target_scale_authority: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> V1CorrectedAttributionFields:
    """Adapt exact gate/Aumann-Shapley moments into the geometric chart field."""

    if type(gate_fields) is not V1ExactGateAttributionFields:
        raise TypeError("exact gate corrected fields require V1ExactGateAttributionFields")
    if gate_fields.equation_id != CONSERVING_ATTRIBUTION_EQUATION_ID:
        raise ValueError("unsupported exact gate attribution equation")
    if target_scale_authority and not gate_fields.target_scale_eligible:
        raise PermissionError(
            "dense-reference exact gate fields cannot carry target-scale authority"
        )
    payload_metadata: dict[str, Any] = {
        "constructor": "exact_gate_aumann_shapley_sufficient_statistics",
        "correction_before_abs": True,
        "semantic_corrected_default": True,
        "canonical_exact_gate_correction": True,
        "surrogate_bias_layer": False,
        "gate_fields_hash": gate_fields.fields_hash,
        "gate_source_hash": gate_fields.source_hash,
        "gate_unit_axis_hash": gate_fields.unit_axis_hash,
        "g_delta_h_disclosure_column_present": True,
        "owner_score_orientation": "current_contribution",
        "geometric_covector_orientation": "positive_removal",
        "positive_removal_sign_applied": True,
        "g_delta_h_disclosure_hash": content_hash(
            {
                "schema": "partition_free_v1_g_delta_h_disclosure_vector_v1",
                "route_axis_hash": gate_fields.route_axis_hash,
                "signed_g_delta_h": gate_fields.signed_g_delta_h,
                "source_hash": gate_fields.source_hash,
                "sample_weight_hash": gate_fields.sample_weight_hash,
            },
            prefix="partition-free-v1-g-delta-h-disclosure-",
        ),
        "loss_rescaling_law": gate_fields.loss_rescaling_law,
        "raw_context_route_matrix_materialized": bool(
            gate_fields.metadata.get("raw_context_route_matrix_materialized", False)
        ),
        "exact_gate_aggregation_mode": gate_fields.aggregation_mode,
        "exact_gate_target_scale_eligible": gate_fields.target_scale_eligible,
    }
    if metadata is not None:
        payload_metadata.update(dict(metadata))
    return V1CorrectedAttributionFields(
        route_ids=gate_fields.route_ids,
        v0_corrected_salience=gate_fields.aumann_shapley_salience,
        signed_corrected_damage=gate_fields.signed_positive_removal_damage,
        sample_count=gate_fields.sample_count,
        source_attribution_hash=gate_fields.fields_hash,
        correction_granularity="route_resolved",
        correction_authority=CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION,
        sample_weight_hash=gate_fields.sample_weight_hash,
        storage_mode=(
            "chunked_sufficient_statistics"
            if gate_fields.target_scale_eligible
            else "dense_reference"
        ),
        peak_copy_mode=(
            "exact_gate_aumann_shapley_sufficient_statistics"
            if gate_fields.target_scale_eligible
            else "exact_gate_dense_context_route_reference"
        ),
        target_scale_authority=target_scale_authority,
        metadata=payload_metadata,
    )


@dataclass(frozen=True, slots=True)
class V1GeometricActionEvaluation:
    """Exact value of one finite pruning action in the current chart."""

    route_axis_hash: str
    action_indices: tuple[int, ...]
    v0_corrected_salience_sum: float
    signed_corrected_linear_damage: float
    fd_cross_term_energy: float
    residual_diagonal_energy: float
    metric_energy: float
    metric_length: float
    metric_triangle_upper_bound: float
    calibrated_metric_damage: float | None
    calibrated_total_damage: float | None
    energy_calibration_hash: str | None
    metric_equation_id: str = METRIC_EQUATION_ID
    action_equation_id: str = ACTION_EQUATION_ID
    triangle_bound_equation_id: str = TRIANGLE_BOUND_EQUATION_ID
    schema: str = ACTION_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_axis_hash", _nonempty(self.route_axis_hash, "route_axis_hash"))
        indices = tuple(_nonnegative_int(value, "action index") for value in self.action_indices)
        if not indices or tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ValueError("action evaluation indices must be sorted, unique, and nonempty")
        object.__setattr__(self, "action_indices", indices)
        for name in (
            "v0_corrected_salience_sum",
            "signed_corrected_linear_damage",
            "fd_cross_term_energy",
            "residual_diagonal_energy",
            "metric_energy",
            "metric_length",
            "metric_triangle_upper_bound",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.v0_corrected_salience_sum < -1e-12:
            raise ValueError("v0 salience cannot be negative")
        if self.fd_cross_term_energy < -1e-10 or self.residual_diagonal_energy < -1e-10:
            raise ValueError("metric energy components cannot be negative")
        if self.metric_energy < -1e-10 or self.metric_length < -1e-10:
            raise ValueError("metric energy and length cannot be negative")
        if self.calibrated_metric_damage is None:
            if self.calibrated_total_damage is not None or self.energy_calibration_hash is not None:
                raise ValueError("calibrated damage fields must appear together")
        else:
            object.__setattr__(
                self,
                "calibrated_metric_damage",
                _finite(self.calibrated_metric_damage, "calibrated_metric_damage"),
            )
            object.__setattr__(
                self,
                "calibrated_total_damage",
                _finite(self.calibrated_total_damage, "calibrated_total_damage"),
            )
            object.__setattr__(
                self,
                "energy_calibration_hash",
                _nonempty(self.energy_calibration_hash, "energy_calibration_hash"),
            )
        if self.metric_equation_id != METRIC_EQUATION_ID or self.action_equation_id != ACTION_EQUATION_ID:
            raise ValueError("unsupported geometric action equation identity")
        if self.triangle_bound_equation_id != TRIANGLE_BOUND_EQUATION_ID:
            raise ValueError("unsupported triangle-bound equation identity")
        if self.schema != ACTION_EVALUATION_SCHEMA:
            raise ValueError("unsupported action evaluation schema")

    @property
    def evaluation_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-geometric-action-evaluation-")


@dataclass(frozen=True, slots=True)
class V1GeometricActionMarginal:
    """Exact set-conditional marginal for adding one route to an action."""

    route_axis_hash: str
    current_action_indices: tuple[int, ...]
    candidate_index: int
    signed_corrected_linear_increment: float
    singleton_metric_energy: float
    cross_metric_increment: float
    metric_energy_increment: float
    calibrated_metric_damage_increment: float | None
    calibrated_total_damage_increment: float | None
    energy_calibration_hash: str | None
    marginal_equation_id: str = MARGINAL_EQUATION_ID
    schema: str = ACTION_MARGINAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_axis_hash", _nonempty(self.route_axis_hash, "route_axis_hash"))
        current = tuple(_nonnegative_int(value, "current action index") for value in self.current_action_indices)
        if tuple(sorted(current)) != current or len(set(current)) != len(current):
            raise ValueError("current action indices must be sorted and unique")
        candidate = _nonnegative_int(self.candidate_index, "candidate_index")
        if candidate in current:
            raise ValueError("marginal candidate is already in the current action")
        object.__setattr__(self, "current_action_indices", current)
        object.__setattr__(self, "candidate_index", candidate)
        for name in (
            "signed_corrected_linear_increment",
            "singleton_metric_energy",
            "cross_metric_increment",
            "metric_energy_increment",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.singleton_metric_energy < -1e-10:
            raise ValueError("singleton metric energy cannot be negative")
        if self.calibrated_metric_damage_increment is None:
            if (
                self.calibrated_total_damage_increment is not None
                or self.energy_calibration_hash is not None
            ):
                raise ValueError("calibrated marginal fields must appear together")
        else:
            object.__setattr__(
                self,
                "calibrated_metric_damage_increment",
                _finite(
                    self.calibrated_metric_damage_increment,
                    "calibrated_metric_damage_increment",
                ),
            )
            object.__setattr__(
                self,
                "calibrated_total_damage_increment",
                _finite(
                    self.calibrated_total_damage_increment,
                    "calibrated_total_damage_increment",
                ),
            )
            object.__setattr__(
                self,
                "energy_calibration_hash",
                _nonempty(self.energy_calibration_hash, "energy_calibration_hash"),
            )
        if self.marginal_equation_id != MARGINAL_EQUATION_ID:
            raise ValueError("unsupported geometric marginal equation identity")
        if self.schema != ACTION_MARGINAL_SCHEMA:
            raise ValueError("unsupported action marginal schema")

    @property
    def marginal_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-geometric-action-marginal-")


@dataclass(frozen=True, slots=True)
class V1GeometricActionChart:
    """One current-mask chart for partition-free geometric pruning actions."""

    route_ids: tuple[str, ...]
    v0_corrected_salience: np.ndarray = field(repr=False, compare=False)
    signed_corrected_damage: np.ndarray = field(repr=False, compare=False)
    fd_action_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    correction_granularity: CorrectionGranularity
    correction_authority: CorrectionAuthority
    action_metric_unit: str
    signed_damage_unit: str
    current_mask_hash: str
    source_geometry_hash: str
    source_correction_hash: str
    energy_calibration: V1MetricEnergyToBCEDamageCalibration | None = None
    layer_ids: tuple[str, ...] | None = None
    storage_mode: ActionStorageMode = "dense_reference"
    peak_copy_mode: str = "bytes_backed_dense_reference"
    target_scale_authority: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = GEOMETRIC_ACTION_CHART_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(self.route_ids, name="v1 geometric action route axis")
        count = len(routes)
        salience = _readonly_f64_vector(
            self.v0_corrected_salience,
            name="v0 corrected salience",
            size=count,
            nonnegative=True,
        )
        signed = _readonly_f64_vector(
            self.signed_corrected_damage,
            name="signed corrected damage",
            size=count,
        )
        factor = _readonly_f64_matrix(
            self.fd_action_factor,
            name="FD action factor",
            columns=count,
        )
        diagonal = _readonly_f64_vector(
            self.residual_diagonal,
            name="residual diagonal",
            size=count,
            nonnegative=True,
        )
        authority = CorrectionAuthority(self.correction_authority)
        _check_correction_authority(
            granularity=self.correction_granularity,
            authority=authority,
        )
        if self.energy_calibration is not None and type(self.energy_calibration) is not (
            V1MetricEnergyToBCEDamageCalibration
        ):
            raise TypeError("energy_calibration must be an exact calibration object")
        if self.layer_ids is None:
            layers = None
        else:
            layers = tuple(_nonempty(value, "layer_id") for value in self.layer_ids)
            if len(layers) != count:
                raise ValueError("layer diagnostics must align with route axis")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "v0_corrected_salience", salience)
        object.__setattr__(self, "signed_corrected_damage", signed)
        object.__setattr__(self, "fd_action_factor", factor)
        object.__setattr__(self, "residual_diagonal", diagonal)
        object.__setattr__(self, "correction_authority", authority)
        object.__setattr__(self, "action_metric_unit", _nonempty(self.action_metric_unit, "action_metric_unit"))
        object.__setattr__(self, "signed_damage_unit", _nonempty(self.signed_damage_unit, "signed_damage_unit"))
        object.__setattr__(self, "current_mask_hash", _nonempty(self.current_mask_hash, "current_mask_hash"))
        object.__setattr__(self, "source_geometry_hash", _nonempty(self.source_geometry_hash, "source_geometry_hash"))
        object.__setattr__(self, "source_correction_hash", _nonempty(self.source_correction_hash, "source_correction_hash"))
        object.__setattr__(self, "layer_ids", layers)
        if self.storage_mode not in {"dense_reference", "chunked_sufficient_statistics"}:
            raise ValueError("unsupported geometric action storage mode")
        object.__setattr__(self, "peak_copy_mode", _nonempty(self.peak_copy_mode, "peak_copy_mode"))
        if type(self.target_scale_authority) is not bool:
            raise TypeError("target_scale_authority must be bool")
        if self.target_scale_authority and self.storage_mode != "chunked_sufficient_statistics":
            raise PermissionError("dense geometric action charts cannot carry target-scale authority")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "geometric action metadata"))
        if self.schema != GEOMETRIC_ACTION_CHART_SCHEMA:
            raise ValueError("unsupported geometric-action chart schema")

    @classmethod
    def from_corrected_fields(
        cls,
        *,
        fields: V1CorrectedAttributionFields,
        fd_action_factor: Sequence[Sequence[float]] | np.ndarray,
        residual_diagonal: Sequence[float] | np.ndarray,
        action_metric_unit: str,
        signed_damage_unit: str,
        current_mask_hash: str,
        source_geometry_hash: str,
        energy_calibration: V1MetricEnergyToBCEDamageCalibration | None = None,
        layer_ids: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "V1GeometricActionChart":
        if type(fields) is not V1CorrectedAttributionFields:
            raise TypeError("fields must be V1CorrectedAttributionFields")
        return cls(
            route_ids=fields.route_ids,
            v0_corrected_salience=fields.v0_corrected_salience,
            signed_corrected_damage=fields.signed_corrected_damage,
            fd_action_factor=fd_action_factor,
            residual_diagonal=residual_diagonal,
            correction_granularity=fields.correction_granularity,
            correction_authority=fields.correction_authority,
            action_metric_unit=action_metric_unit,
            signed_damage_unit=signed_damage_unit,
            current_mask_hash=current_mask_hash,
            source_geometry_hash=source_geometry_hash,
            source_correction_hash=fields.fields_hash,
            energy_calibration=energy_calibration,
            layer_ids=None if layer_ids is None else tuple(layer_ids),
            storage_mode=fields.storage_mode,
            peak_copy_mode=fields.peak_copy_mode,
            target_scale_authority=fields.target_scale_authority,
            metadata={} if metadata is None else metadata,
        )

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(self.route_ids, name="v1 geometric action route axis")

    @property
    def chart_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-geometric-action-chart-")

    @property
    def production_eligible(self) -> bool:
        return (
            self.correction_granularity == "route_resolved"
            and self.correction_authority is CorrectionAuthority.ROUTE_RESOLVED_PRODUCTION
        )

    @property
    def route_count(self) -> int:
        return len(self.route_ids)

    @property
    def singleton_metric_energy(self) -> np.ndarray:
        values = np.sum(
            np.square(self.fd_action_factor, dtype=np.float64),
            axis=0,
            dtype=np.float64,
        )
        values = np.asarray(values + self.residual_diagonal, dtype=np.float64)
        values.setflags(write=False)
        return values

    def _indices(self, removal_indices: Sequence[int] | np.ndarray) -> tuple[int, ...]:
        return _normalized_indices(removal_indices, self.route_count)

    def _factor_projection(self, indices: tuple[int, ...]) -> np.ndarray:
        index_array = np.asarray(indices, dtype=np.int64)
        return np.sum(self.fd_action_factor[:, index_array], axis=1, dtype=np.float64)

    def metric_energy(self, removal_indices: Sequence[int] | np.ndarray) -> float:
        indices = self._indices(removal_indices)
        projection = self._factor_projection(indices)
        factor_energy = float(np.dot(projection, projection))
        diagonal_energy = float(np.sum(self.residual_diagonal[np.asarray(indices)], dtype=np.float64))
        return max(0.0, math.fsum((factor_energy, diagonal_energy)))

    def metric_length(self, removal_indices: Sequence[int] | np.ndarray) -> float:
        return math.sqrt(self.metric_energy(removal_indices))

    def triangle_upper_bound(self, removal_indices: Sequence[int] | np.ndarray) -> float:
        indices = self._indices(removal_indices)
        singleton = self.singleton_metric_energy[np.asarray(indices, dtype=np.int64)]
        return float(np.sum(np.sqrt(np.maximum(singleton, 0.0)), dtype=np.float64))

    def evaluate_action(
        self,
        removal_indices: Sequence[int] | np.ndarray,
    ) -> V1GeometricActionEvaluation:
        indices = self._indices(removal_indices)
        index_array = np.asarray(indices, dtype=np.int64)
        projection = self._factor_projection(indices)
        fd_cross = float(np.dot(projection, projection))
        diagonal = float(np.sum(self.residual_diagonal[index_array], dtype=np.float64))
        metric_energy = max(0.0, math.fsum((fd_cross, diagonal)))
        metric_length = math.sqrt(metric_energy)
        triangle = self.triangle_upper_bound(indices)
        linear = float(np.sum(self.signed_corrected_damage[index_array], dtype=np.float64))
        salience = float(np.sum(self.v0_corrected_salience[index_array], dtype=np.float64))
        calibrated_metric: float | None
        calibrated_total: float | None
        calibration_hash: str | None
        if self.energy_calibration is None:
            calibrated_metric = None
            calibrated_total = None
            calibration_hash = None
        else:
            calibrated_metric = self.energy_calibration.energy_to_damage(metric_energy)
            calibrated_total = math.fsum((linear, calibrated_metric))
            calibration_hash = self.energy_calibration.calibration_hash
        return V1GeometricActionEvaluation(
            route_axis_hash=self.route_axis_hash,
            action_indices=indices,
            v0_corrected_salience_sum=salience,
            signed_corrected_linear_damage=linear,
            fd_cross_term_energy=fd_cross,
            residual_diagonal_energy=diagonal,
            metric_energy=metric_energy,
            metric_length=metric_length,
            metric_triangle_upper_bound=triangle,
            calibrated_metric_damage=calibrated_metric,
            calibrated_total_damage=calibrated_total,
            energy_calibration_hash=calibration_hash,
        )

    def marginal_addition(
        self,
        current_indices: Sequence[int] | np.ndarray,
        candidate_index: int,
    ) -> V1GeometricActionMarginal:
        current_raw = tuple(current_indices)
        current = () if not current_raw else self._indices(current_raw)
        candidate = _nonnegative_int(candidate_index, "candidate_index")
        if candidate >= self.route_count:
            raise ValueError("candidate index lies outside the route axis")
        if candidate in current:
            raise ValueError("candidate index is already present in current action")
        current_projection = (
            np.zeros(self.fd_action_factor.shape[0], dtype=np.float64)
            if not current
            else self._factor_projection(current)
        )
        column = self.fd_action_factor[:, candidate]
        singleton = math.fsum(
            (
                float(np.dot(column, column)),
                float(self.residual_diagonal[candidate]),
            )
        )
        cross = 2.0 * float(np.dot(column, current_projection))
        metric_increment = math.fsum((singleton, cross))
        linear = float(self.signed_corrected_damage[candidate])
        if self.energy_calibration is None:
            calibrated_metric = None
            calibrated_total = None
            calibration_hash = None
        else:
            calibrated_metric = float(self.energy_calibration.gamma * metric_increment)
            calibrated_total = math.fsum((linear, calibrated_metric))
            calibration_hash = self.energy_calibration.calibration_hash
        return V1GeometricActionMarginal(
            route_axis_hash=self.route_axis_hash,
            current_action_indices=current,
            candidate_index=candidate,
            signed_corrected_linear_increment=linear,
            singleton_metric_energy=singleton,
            cross_metric_increment=cross,
            metric_energy_increment=metric_increment,
            calibrated_metric_damage_increment=calibrated_metric,
            calibrated_total_damage_increment=calibrated_total,
            energy_calibration_hash=calibration_hash,
        )

    def singleton_atom_scores(self, *, require_calibration: bool = True) -> np.ndarray:
        """Return singleton atom damage only; this is not a set objective."""

        if require_calibration and self.energy_calibration is None:
            raise ValueError("singleton atom damage requires an energy-to-damage calibration")
        metric = self.singleton_metric_energy
        if self.energy_calibration is not None:
            metric_damage = self.energy_calibration.gamma * metric
        else:
            metric_damage = metric
        result = np.asarray(self.signed_corrected_damage + metric_damage, dtype=np.float64)
        if not np.isfinite(result).all():
            raise RuntimeError("singleton atom scores are nonfinite")
        result.setflags(write=False)
        return result

    def layer_diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        """Summarize layers without granting layer-selection authority."""

        singleton = self.singleton_metric_energy
        layers: dict[str, list[int]] = {}
        if self.layer_ids is None:
            return ()
        for index, layer_id in enumerate(self.layer_ids):
            layers.setdefault(layer_id, []).append(index)
        rows: list[Mapping[str, Any]] = []
        for layer_id in sorted(layers):
            index_array = np.asarray(layers[layer_id], dtype=np.int64)
            route_count = int(index_array.size)
            salience_sum = float(np.sum(self.v0_corrected_salience[index_array], dtype=np.float64))
            signed_sum = float(np.sum(self.signed_corrected_damage[index_array], dtype=np.float64))
            metric_sum = float(np.sum(singleton[index_array], dtype=np.float64))
            rows.append(
                MappingProxyType(
                    {
                        "layer_id": layer_id,
                        "route_count": route_count,
                        "v0_corrected_salience_sum": salience_sum,
                        "v0_corrected_salience_mean": salience_sum / route_count,
                        "signed_corrected_damage_sum": signed_sum,
                        "signed_corrected_damage_mean": signed_sum / route_count,
                        "singleton_metric_energy_sum": metric_sum,
                        "singleton_metric_energy_mean": metric_sum / route_count,
                        "selection_role": "diagnostic_only",
                    }
                )
            )
        return tuple(rows)

    @property
    def layer_diagnostics_hash(self) -> str:
        return content_hash(
            {
                "schema": "partition_free_v1_layer_diagnostics_only_v1",
                "chart_hash": self.chart_hash,
                "layers": self.layer_diagnostics(),
                "selection_authority": False,
            },
            prefix="partition-free-v1-layer-diagnostics-",
        )

    def assert_loss_rescaling_covariance(
        self,
        other: "V1GeometricActionChart",
        *,
        scale: float,
        probe_actions: Sequence[Sequence[int]] | None = None,
        max_full_gram_routes: int = 2048,
        rtol: float = 1e-9,
        atol: float = 1e-10,
    ) -> None:
        """Check covariance under positive rescaling of raw loss-pullback rows.

        For loss scale ``k``: ``A`` and ``c`` scale by ``k``; raw FD rows
        scale by ``k``; therefore ``B.T B``, ``d``, and ``Q`` scale by ``k^2``;
        ``L`` scales by ``k``; and a fitted ``gamma: Q -> damage`` scales by
        ``1/k`` so calibrated damage still scales by ``k``.
        """

        if type(other) is not V1GeometricActionChart:
            raise TypeError("other must be a V1GeometricActionChart")
        scale_value = _finite(scale, "loss rescaling scale", positive=True)
        if self.route_ids != other.route_ids:
            raise ValueError("loss-rescaling check requires the same route axis")
        if not np.allclose(
            other.v0_corrected_salience,
            scale_value * self.v0_corrected_salience,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError("v0 corrected salience is not loss-rescaling covariant")
        if not np.allclose(
            other.signed_corrected_damage,
            scale_value * self.signed_corrected_damage,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError("signed corrected damage is not loss-rescaling covariant")
        if not np.allclose(
            other.residual_diagonal,
            scale_value * scale_value * self.residual_diagonal,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError("residual diagonal is not loss-rescaling covariant")
        if (self.energy_calibration is None) != (other.energy_calibration is None):
            raise ValueError("energy calibration presence is not loss-rescaling covariant")
        if self.energy_calibration is not None and other.energy_calibration is not None:
            expected_gamma = self.energy_calibration.gamma / scale_value
            if not math.isclose(
                other.energy_calibration.gamma,
                expected_gamma,
                rel_tol=rtol,
                abs_tol=atol,
            ):
                raise ValueError("energy calibration gamma is not loss-rescaling covariant")
        if self.route_count <= int(max_full_gram_routes):
            left = self.fd_action_factor.T @ self.fd_action_factor
            right = other.fd_action_factor.T @ other.fd_action_factor
            if not np.allclose(
                right,
                scale_value * scale_value * left,
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError("FD action Gram is not loss-rescaling covariant")
            actions = tuple((index,) for index in range(self.route_count))
        elif probe_actions is not None:
            actions = tuple(tuple(action) for action in probe_actions)
            if not actions:
                raise ValueError("probe_actions must be nonempty")
        else:
            raise ValueError("large covariance checks require explicit probe actions")
        for action in actions:
            base = self.evaluate_action(action)
            scaled = other.evaluate_action(action)
            if not math.isclose(
                scaled.metric_energy,
                scale_value * scale_value * base.metric_energy,
                rel_tol=rtol,
                abs_tol=atol,
            ):
                raise ValueError("action metric energy is not loss-rescaling covariant")
            if not math.isclose(
                scaled.metric_length,
                scale_value * base.metric_length,
                rel_tol=rtol,
                abs_tol=atol,
            ):
                raise ValueError("action metric length is not loss-rescaling covariant")
            if base.calibrated_total_damage is not None and scaled.calibrated_total_damage is not None:
                if not math.isclose(
                    scaled.calibrated_total_damage,
                    scale_value * base.calibrated_total_damage,
                    rel_tol=rtol,
                    abs_tol=atol,
                ):
                    raise ValueError("calibrated total damage is not loss-rescaling covariant")


@dataclass(frozen=True, slots=True)
class V1FixedCardinalityDualPolicy:
    """Diagnostic dual search for a calibrated signed-plus-quadratic objective."""

    maximum_iterations: int = 32
    damping: float = 0.75
    absolute_gap_tolerance: float = 1e-8
    relative_gap_tolerance: float = 1e-8
    require_certified_gap: bool = True
    selector_role: str = "calibrated_damage_diagnostic"
    schema: str = FIXED_CARDINALITY_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "maximum_iterations", _positive_int(self.maximum_iterations, "maximum_iterations"))
        damping = _finite(self.damping, "damping", positive=True)
        if damping > 1.0:
            raise ValueError("damping must lie in (0, 1]")
        object.__setattr__(self, "damping", damping)
        object.__setattr__(
            self,
            "absolute_gap_tolerance",
            _finite(self.absolute_gap_tolerance, "absolute_gap_tolerance"),
        )
        object.__setattr__(
            self,
            "relative_gap_tolerance",
            _finite(self.relative_gap_tolerance, "relative_gap_tolerance"),
        )
        if self.absolute_gap_tolerance < 0.0 or self.relative_gap_tolerance < 0.0:
            raise ValueError("gap tolerances must be nonnegative")
        if type(self.require_certified_gap) is not bool:
            raise TypeError("require_certified_gap must be bool")
        if self.selector_role not in {
            "calibrated_damage_diagnostic",
            "calibrated_damage_production_candidate",
        }:
            raise ValueError("unsupported fixed-cardinality selector role")
        if self.schema != FIXED_CARDINALITY_POLICY_SCHEMA:
            raise ValueError("unsupported fixed-cardinality policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fixed-k-dual-policy-")


@dataclass(frozen=True, slots=True)
class V1FixedCardinalitySelection:
    """One fixed-K calibrated objective proposal and its duality receipt."""

    route_axis_hash: str
    selected_indices: tuple[int, ...]
    requested_count: int
    objective_value: float
    best_dual_lower_bound: float
    duality_gap: float
    certified: bool
    iteration_count: int
    policy_hash: str
    chart_hash: str
    eligible_index_hash: str
    selector_role: str
    proposal_only: bool = True
    mask_commit_authority: bool = False
    equation_id: str = DUAL_LOWER_BOUND_EQUATION_ID
    schema: str = FIXED_CARDINALITY_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_axis_hash", _nonempty(self.route_axis_hash, "route_axis_hash"))
        indices = tuple(_nonnegative_int(value, "selected index") for value in self.selected_indices)
        if not indices or tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ValueError("selected indices must be sorted, unique, and nonempty")
        requested = _positive_int(self.requested_count, "requested_count")
        if len(indices) != requested:
            raise ValueError("selected index count differs from requested_count")
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "requested_count", requested)
        for name in ("objective_value", "best_dual_lower_bound", "duality_gap"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.duality_gap < -1e-8:
            raise ValueError("duality gap cannot be negative")
        if type(self.certified) is not bool:
            raise TypeError("certified must be bool")
        object.__setattr__(self, "iteration_count", _positive_int(self.iteration_count, "iteration_count"))
        for name in ("policy_hash", "chart_hash", "eligible_index_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.selector_role not in {
            "calibrated_damage_diagnostic",
            "calibrated_damage_production_candidate",
        }:
            raise ValueError("unsupported fixed-cardinality selector role")
        if type(self.proposal_only) is not bool or not self.proposal_only:
            raise PermissionError("fixed-cardinality geometric selections are proposal-only")
        if type(self.mask_commit_authority) is not bool or self.mask_commit_authority:
            raise PermissionError("fixed-cardinality geometric selections cannot commit masks")
        if self.equation_id != DUAL_LOWER_BOUND_EQUATION_ID:
            raise ValueError("unsupported dual lower-bound equation")
        if self.schema != FIXED_CARDINALITY_SELECTION_SCHEMA:
            raise ValueError("unsupported fixed-cardinality selection schema")

    @property
    def selection_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fixed-k-selection-")


def _bottom_k_indices(values: np.ndarray, eligible: np.ndarray, count: int) -> np.ndarray:
    selected_values = np.asarray(values, dtype=np.float64).reshape(-1)[eligible]
    if not np.isfinite(selected_values).all():
        raise RuntimeError("fixed-cardinality costs are nonfinite")
    if count == eligible.size:
        result = np.array(eligible, dtype=np.int64, copy=True)
        result.sort(kind="mergesort")
        return result
    cutoff = float(np.partition(selected_values, count - 1)[count - 1])
    below = np.flatnonzero(selected_values < cutoff)
    ties = np.flatnonzero(selected_values == cutoff)
    needed = count - int(below.size)
    if needed < 0 or needed > int(ties.size):
        raise RuntimeError("fixed-cardinality boundary is inconsistent")
    tie_order = np.lexsort((eligible[ties],))
    chosen = np.concatenate((below, ties[tie_order[:needed]]))
    result = np.array(eligible[chosen], dtype=np.int64, copy=True)
    result.sort(kind="mergesort")
    return result


def select_fixed_cardinality_geometric_action(
    *,
    chart: V1GeometricActionChart,
    eligible_indices: Sequence[int] | np.ndarray,
    count: int,
    policy: V1FixedCardinalityDualPolicy | None = None,
) -> V1FixedCardinalitySelection:
    """Optimize a fixed-size calibrated signed-plus-quadratic diagnostic.

    Canonical production v1 uses corrected salience/length as separate budgets
    plus a direct oracle.  This helper is for calibrated set-objective audits
    and proposal diagnostics; without explicit energy calibration it refuses.
    """

    if type(chart) is not V1GeometricActionChart:
        raise TypeError("chart must be V1GeometricActionChart")
    resolved_policy = V1FixedCardinalityDualPolicy() if policy is None else policy
    if type(resolved_policy) is not V1FixedCardinalityDualPolicy:
        raise TypeError("policy must be V1FixedCardinalityDualPolicy")
    if chart.energy_calibration is None:
        raise PermissionError(
            "fixed-cardinality signed-plus-quadratic selection requires explicit energy calibration"
        )
    if (
        resolved_policy.selector_role == "calibrated_damage_production_candidate"
        and (not chart.production_eligible or not chart.target_scale_authority)
    ):
        raise PermissionError(
            "production-candidate selection requires a production-eligible target-scale chart"
        )
    size = _positive_int(count, "count")
    eligible = np.asarray(eligible_indices, dtype=np.int64).reshape(-1)
    if (
        not eligible.size
        or len(set(int(value) for value in eligible.tolist())) != int(eligible.size)
        or bool(np.any(eligible < 0))
        or bool(np.any(eligible >= chart.route_count))
    ):
        raise ValueError("eligible indices must be unique and on the route axis")
    eligible.sort(kind="mergesort")
    if size > int(eligible.size):
        raise ValueError("requested fixed-cardinality action exceeds eligible routes")
    gamma = chart.energy_calibration.gamma
    factor = math.sqrt(gamma) * chart.fd_action_factor
    linear = np.asarray(
        chart.signed_corrected_damage + gamma * chart.residual_diagonal,
        dtype=np.float64,
    )
    singleton = np.asarray(
        linear + np.sum(np.square(factor, dtype=np.float64), axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    starts: list[np.ndarray] = [np.zeros(factor.shape[0], dtype=np.float64)]
    singleton_selected = _bottom_k_indices(singleton, eligible, size)
    starts.append(np.sum(factor[:, singleton_selected], axis=1, dtype=np.float64))

    best_upper = math.inf
    best_lower = -math.inf
    best_indices: tuple[int, ...] | None = None
    iterations = 0
    for start in starts:
        lambd = np.array(start, dtype=np.float64, copy=True)
        for _ in range(resolved_policy.maximum_iterations):
            iterations += 1
            costs = np.asarray(linear + 2.0 * (lambd @ factor), dtype=np.float64)
            selected = _bottom_k_indices(costs, eligible, size)
            q = np.sum(factor[:, selected], axis=1, dtype=np.float64)
            selected_linear = float(np.sum(linear[selected], dtype=np.float64))
            primal = math.fsum((selected_linear, float(np.dot(q, q))))
            dual = math.fsum(
                (
                    float(np.sum(costs[selected], dtype=np.float64)),
                    -float(np.dot(lambd, lambd)),
                )
            )
            if dual > best_lower:
                best_lower = dual
            if primal < best_upper:
                best_upper = primal
                best_indices = tuple(int(value) for value in selected)
            gap = max(0.0, best_upper - best_lower)
            tolerance = max(
                resolved_policy.absolute_gap_tolerance,
                resolved_policy.relative_gap_tolerance * max(1.0, abs(best_upper)),
            )
            if gap <= tolerance:
                break
            lambd = (1.0 - resolved_policy.damping) * lambd + resolved_policy.damping * q
    if best_indices is None:
        raise RuntimeError("fixed-cardinality selector did not evaluate any candidate")
    gap = max(0.0, best_upper - best_lower)
    tolerance = max(
        resolved_policy.absolute_gap_tolerance,
        resolved_policy.relative_gap_tolerance * max(1.0, abs(best_upper)),
    )
    certified = bool(gap <= tolerance)
    if resolved_policy.require_certified_gap and not certified:
        raise RuntimeError(
            "fixed-cardinality geometric selector could not certify the requested duality gap"
        )
    return V1FixedCardinalitySelection(
        route_axis_hash=chart.route_axis_hash,
        selected_indices=best_indices,
        requested_count=size,
        objective_value=best_upper,
        best_dual_lower_bound=best_lower,
        duality_gap=gap,
        certified=certified,
        iteration_count=max(1, iterations),
        policy_hash=resolved_policy.policy_hash,
        chart_hash=chart.chart_hash,
        eligible_index_hash=content_hash(
            np.asarray(eligible, dtype=np.int64),
            prefix="partition-free-v1-fixed-k-eligible-",
        ),
        selector_role=resolved_policy.selector_role,
        proposal_only=True,
        mask_commit_authority=False,
    )


__all__ = [
    "ACTION_EQUATION_ID",
    "DUAL_LOWER_BOUND_EQUATION_ID",
    "MARGINAL_EQUATION_ID",
    "METRIC_EQUATION_ID",
    "TRIANGLE_BOUND_EQUATION_ID",
    "CorrectionAuthority",
    "V1CorrectedAttributionFields",
    "V1FixedCardinalityDualPolicy",
    "V1FixedCardinalitySelection",
    "V1GeometricActionChart",
    "V1GeometricActionEvaluation",
    "V1GeometricActionMarginal",
    "V1MetricEnergyToBCEDamageCalibration",
    "corrected_attribution_fields_from_exact_gate",
    "corrected_attribution_fields_from_samples",
    "corrected_attribution_fields_from_sufficient_statistics",
    "select_fixed_cardinality_geometric_action",
]
