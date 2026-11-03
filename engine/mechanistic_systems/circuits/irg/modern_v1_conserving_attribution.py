"""Exact conserving first-order attribution for canonical modern v1.

``corrected attribution`` has a specific project meaning.  For a ReLU unit
with preactivation ``a``, bias-only baseline ``b``, post-activation
cotangent ``G``, and route drive ``c_r``, the current-contribution allocation
is

``s_r = G * phi(a, b) * c_r``

where ``phi`` is the fraction of the straight path from ``b`` to ``a`` on
which the ReLU is open.  The allocation is exact and conserving:

``sum_r s_r = G * (relu(a) - relu(b))``.

Positive-removal work has the opposite orientation, ``-s_r``.  The exact
single-route finite difference ``G * (relu(a-c_r)-relu(a))`` is a separate
disclosure quantity; it must not be substituted for the conserving split.

This module contains only the elementwise gate geometry.  Route reducers own
the architecture-specific contraction over linear or convolutional drives.
No learned residual correction, layer intercept, score partition, or
second-order term is part of this definition.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)

from .contracts import content_hash


CONSERVING_ATTRIBUTION_SCHEMA: Final = (
    "partition_free_v1_conserving_relu_attribution_v1"
)
RELU_PATH_FRACTION_EQUATION_ID: Final = (
    "integral_0_1_indicator_b_plus_t_a_minus_b_positive_dt_v1"
)
CONSERVING_ATTRIBUTION_EQUATION_ID: Final = (
    "post_relu_cotangent_times_path_open_fraction_times_route_drive_v1"
)
POSITIVE_REMOVAL_ORIENTATION_ID: Final = (
    "positive_removal_work_is_negative_current_contribution_v1"
)
EXACT_SINGLETON_DISCLOSURE_ID: Final = (
    "post_relu_cotangent_times_relu_a_minus_c_minus_relu_a_v1"
)
EXACT_GATE_ATTRIBUTION_FIELDS_SCHEMA: Final = (
    "partition_free_v1_exact_gate_aumann_shapley_attribution_fields_v1"
)
LOSS_RESCALING_LAW: Final = (
    "under positive loss rescaling k, post-ReLU cotangent G scales by k; "
    "phi, route drive c, and Delta h stay fixed; G*phi*c and G*Delta h "
    "therefore scale by k"
)
ExactGateAggregationMode = Literal[
    "dense_reference",
    "route_chunked_sufficient_statistics",
]


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite nonnegative scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _torch_module() -> Any | None:
    try:
        import torch
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dep
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_module()
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _readonly_numpy(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _freeze_metadata(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze_metadata(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
            array.shape
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_metadata(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
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
        raise ValueError(f"{name} must align to the route axis")
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and nonempty")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    return _readonly_numpy(array)


def _readonly_f64_matrix(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return _readonly_numpy(array)


def _readonly_route_unit_index(
    value: Sequence[int] | np.ndarray,
    *,
    route_count: int,
    unit_count: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if (
        array.shape != (route_count,)
        or bool(np.any(array < 0))
        or bool(np.any(array >= unit_count))
    ):
        raise ValueError(
            "route_unit_index must map every route to a valid unit index"
        )
    payload = np.ascontiguousarray(array, dtype="<i8").tobytes(order="C")
    return np.frombuffer(payload, dtype="<i8", count=route_count)


def relu_path_open_fraction(
    preactivation: Any,
    bias_baseline: Any,
) -> Any:
    """Return the exact ReLU-open fraction along the bias-to-state path.

    The four explicit sign branches avoid an unstable generic quotient near
    equal endpoints.  The strict-gate constant-path convention is one for a
    positive constant path and zero otherwise.  Isolated endpoint values do
    not affect the path integral, so ``b=0<a`` yields one and ``b<0=a``
    yields zero.

    NumPy inputs produce a read-only float64 array.  Torch inputs remain on
    their original device and preserve float32/float64 dtype; mixed NumPy and
    Torch inputs are refused rather than transferred implicitly.
    """

    torch = _torch_module()
    pre_is_torch = bool(torch is not None and isinstance(preactivation, torch.Tensor))
    bias_is_torch = bool(torch is not None and isinstance(bias_baseline, torch.Tensor))
    if pre_is_torch != bias_is_torch:
        raise TypeError("preactivation and bias baseline must use one array backend")

    if pre_is_torch:
        assert torch is not None
        a = preactivation
        b = bias_baseline
        if (
            not a.is_floating_point()
            or not b.is_floating_point()
            or a.dtype != b.dtype
            or a.device != b.device
            or a.dtype not in {torch.float32, torch.float64}
        ):
            raise TypeError(
                "torch ReLU path endpoints must share float32/float64 dtype and device"
            )
        a, b = torch.broadcast_tensors(a, b)
        if not bool(torch.isfinite(a).all().item()) or not bool(
            torch.isfinite(b).all().item()
        ):
            raise ValueError("ReLU path endpoints must be finite")
        result = torch.zeros_like(a)
        both_positive = torch.logical_and(a > 0.0, b > 0.0)
        entering = torch.logical_and(a > 0.0, b <= 0.0)
        leaving = torch.logical_and(a <= 0.0, b > 0.0)
        result = torch.where(both_positive, torch.ones_like(result), result)
        result = torch.where(entering, a / (a - b), result)
        result = torch.where(leaving, b / (b - a), result)
        if not bool(torch.isfinite(result).all().item()) or bool(
            torch.any(torch.logical_or(result < 0.0, result > 1.0)).item()
        ):
            raise RuntimeError("ReLU path fraction left its exact [0,1] range")
        return result

    a, b = np.broadcast_arrays(
        np.asarray(preactivation, dtype=np.float64),
        np.asarray(bias_baseline, dtype=np.float64),
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("ReLU path endpoints must be finite")
    result = np.zeros(a.shape, dtype=np.float64)
    both_positive = np.logical_and(a > 0.0, b > 0.0)
    entering = np.logical_and(a > 0.0, b <= 0.0)
    leaving = np.logical_and(a <= 0.0, b > 0.0)
    result[both_positive] = 1.0
    result[entering] = a[entering] / (a[entering] - b[entering])
    result[leaving] = b[leaving] / (b[leaving] - a[leaving])
    if not np.isfinite(result).all() or bool(
        np.any(np.logical_or(result < 0.0, result > 1.0))
    ):
        raise RuntimeError("ReLU path fraction left its exact [0,1] range")
    return _readonly_numpy(result)


def _path_average_sigmoid(endpoint: Any, change: Any) -> Any:
    """Stable float64 softplus secant, with its continuous zero-change limit."""
    torch = _torch_module()
    assert torch is not None
    start = endpoint - change
    small_change = change.clamp(min=-1.0, max=1.0)
    small_difference = torch.log1p(torch.sigmoid(start) * torch.expm1(small_change))
    large_difference = torch.logaddexp(
        torch.nn.functional.logsigmoid(start) + change,
        torch.nn.functional.logsigmoid(-start),
    )
    difference = torch.where(change.abs() <= 1.0, small_difference, large_difference)
    divisor = torch.where(change == 0.0, torch.ones_like(change), change)
    average = difference / divisor
    # Midpoint is the stable analytic limit: the omitted term is O(change^2)
    # and below float64 rounding at this precision-derived boundary.
    tiny = change.abs() <= math.sqrt(torch.finfo(torch.float64).eps)
    return torch.where(tiny, torch.sigmoid(endpoint - 0.5 * change), average)


def conserving_pruning_loss_cotangent(
    post_activation_cotangent: Any,
    preactivation: Any,
    bias_baseline: Any,
    loss_path: Any,
    *,
    activation_kind: str,
) -> Any:
    """Integrate the calibrated loss on the registered unitwise activation path.

    For each unit, the frozen target functional is
    z(t)=z_dense+G*(h(b+t*(a-b))-h(a)). Other units and dense competitors stay
    fixed. The softplus secant integrates the varying loss probability exactly;
    G and the bias-to-activation path are the registered conserving geometry.
    This function does not define a nonlinear full-network gate path.
    """
    from mechanistic_systems.measurement.pruning_loss import FrozenPruningLossPath

    torch = _torch_module()
    assert torch is not None
    if not isinstance(loss_path, FrozenPruningLossPath):
        raise TypeError("conserving loss integration requires typed frozen path geometry")
    gradient = post_activation_cotangent
    if (
        not isinstance(gradient, torch.Tensor) or gradient.ndim < 3
        or gradient.shape[0] != 1 or gradient.shape[1] != loss_path.dense_eta.numel()
        or gradient.device != loss_path.dense_eta.device
        or activation_kind not in {"relu", "identity"}
    ):
        raise ValueError("conserving loss path differs from its single-target activation axis")
    a = preactivation.detach().to(torch.float64)
    b = bias_baseline.detach().to(torch.float64)
    a, b = torch.broadcast_tensors(a, b)
    g = gradient.detach().to(torch.float64)
    if activation_kind == "relu":
        fraction = relu_path_open_fraction(a, b)
        activation_change = a.relu() - b.relu()
    else:
        fraction = torch.ones_like(a)
        activation_change = a - b
    shape = (1, int(loss_path.dense_eta.numel())) + (1,) * (g.ndim - 2)
    endpoint = loss_path.dense_eta.reshape(shape)
    label = loss_path.binary_label.reshape(shape)
    change = float(loss_path.affine_scale) * g * activation_change
    # Symmetry computes y=1 as -average(sigmoid(-eta)); it never subtracts
    # one from a rounded probability close to one.
    sign = 1.0 - 2.0 * label
    residual = sign * _path_average_sigmoid(sign * endpoint, sign * change)
    integrated = (float(loss_path.affine_scale) * g * fraction * residual).to(gradient.dtype)
    if not bool(torch.isfinite(integrated).all().item()):
        raise FloatingPointError("integrated pruning-loss activation cotangent is nonfinite")
    return integrated


def conserving_relu_cotangent(
    post_activation_cotangent: Any,
    preactivation: Any,
    bias_baseline: Any,
) -> Any:
    """Return ``G * phi`` for architecture-specific route contraction.

    A leading query axis on ``G`` is supported naturally: the context/unit
    axes of ``phi`` align with the trailing axes of the cotangent.
    """

    fraction = relu_path_open_fraction(preactivation, bias_baseline)
    if _is_torch_tensor(fraction):
        torch = _torch_module()
        assert torch is not None
        if not isinstance(post_activation_cotangent, torch.Tensor):
            raise TypeError("post-activation cotangent must use the endpoint backend")
        if (
            post_activation_cotangent.dtype != fraction.dtype
            or post_activation_cotangent.device != fraction.device
            or not bool(torch.isfinite(post_activation_cotangent).all().item())
        ):
            raise ValueError(
                "post-activation cotangent must be finite and share dtype/device"
            )
        try:
            result = post_activation_cotangent * fraction
        except RuntimeError as exc:
            raise ValueError(
                "post-activation cotangent does not broadcast over the ReLU field"
            ) from exc
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("conserving ReLU cotangent is nonfinite")
        return result

    cotangent = np.asarray(post_activation_cotangent, dtype=np.float64)
    if not np.isfinite(cotangent).all():
        raise ValueError("post-activation cotangent must be finite")
    try:
        result = cotangent * fraction
    except ValueError as exc:
        raise ValueError(
            "post-activation cotangent does not broadcast over the ReLU field"
        ) from exc
    if not np.isfinite(result).all():
        raise RuntimeError("conserving ReLU cotangent is nonfinite")
    return _readonly_numpy(result)


def conserving_relu_total_current_attribution(
    post_activation_cotangent: Any,
    preactivation: Any,
    bias_baseline: Any,
) -> Any:
    """Return the exact all-route conserving total ``G(relu(a)-relu(b))``."""

    corrected = conserving_relu_cotangent(
        post_activation_cotangent,
        preactivation,
        bias_baseline,
    )
    if _is_torch_tensor(corrected):
        result = corrected * (preactivation - bias_baseline)
        return result
    result = np.asarray(corrected) * (
        np.asarray(preactivation, dtype=np.float64)
        - np.asarray(bias_baseline, dtype=np.float64)
    )
    return _readonly_numpy(result)


def exact_relu_singleton_removal_disclosure(
    post_activation_cotangent: Any,
    preactivation: Any,
    route_drive: Any,
) -> Any:
    """Return frozen-cotangent ``G*(relu(a-c)-relu(a))``.

    This is an exact one-layer singleton consequence at the supplied state,
    not the conserving allocation and not a finite end-to-end loss oracle.
    """

    if _is_torch_tensor(preactivation):
        torch = _torch_module()
        assert torch is not None
        if not isinstance(route_drive, torch.Tensor) or not isinstance(
            post_activation_cotangent, torch.Tensor
        ):
            raise TypeError("singleton disclosure inputs must share one backend")
        if (
            preactivation.dtype != route_drive.dtype
            or preactivation.device != route_drive.device
            or post_activation_cotangent.dtype != preactivation.dtype
            or post_activation_cotangent.device != preactivation.device
        ):
            raise ValueError("singleton disclosure tensors must share dtype/device")
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (post_activation_cotangent, preactivation, route_drive)
        ):
            raise ValueError("singleton disclosure inputs must be finite")
        delta = torch.relu(preactivation - route_drive) - torch.relu(preactivation)
        try:
            result = post_activation_cotangent * delta
        except RuntimeError as exc:
            raise ValueError("singleton disclosure axes do not broadcast") from exc
        return result

    cotangent = np.asarray(post_activation_cotangent, dtype=np.float64)
    pre = np.asarray(preactivation, dtype=np.float64)
    drive = np.asarray(route_drive, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (cotangent, pre, drive)):
        raise ValueError("singleton disclosure inputs must be finite")
    delta = np.maximum(pre - drive, 0.0) - np.maximum(pre, 0.0)
    try:
        result = cotangent * delta
    except ValueError as exc:
        raise ValueError("singleton disclosure axes do not broadcast") from exc
    return _readonly_numpy(result)


@dataclass(frozen=True, slots=True)
class V1ExactGateAttributionFields:
    """Weighted exact gate moments on one physical route axis.

    ``signed_aumann_shapley`` retains the owner equation's
    current-contribution orientation, ``G*phi*c``.  A pruning action removes
    that contribution, so ``signed_positive_removal_damage`` is its negative.
    ``signed_g_delta_h`` is already a finite-removal disclosure and therefore
    has positive-removal orientation.
    """

    route_ids: tuple[str, ...]
    unit_ids: tuple[str, ...]
    route_unit_index: np.ndarray
    aumann_shapley_salience: np.ndarray
    signed_aumann_shapley: np.ndarray
    signed_g_delta_h: np.ndarray
    sample_count: int
    total_weight: float
    source_hash: str
    sample_weight_hash: str
    aggregation_mode: ExactGateAggregationMode = "dense_reference"
    target_scale_eligible: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    equation_id: str = CONSERVING_ATTRIBUTION_EQUATION_ID
    positive_removal_orientation_id: str = POSITIVE_REMOVAL_ORIENTATION_ID
    loss_rescaling_law: str = LOSS_RESCALING_LAW
    schema: str = EXACT_GATE_ATTRIBUTION_FIELDS_SCHEMA

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="v1 exact gate route axis",
        )
        units = canonical_unique_string_axis(
            self.unit_ids,
            name="v1 exact gate unit axis",
        )
        route_unit = _readonly_route_unit_index(
            self.route_unit_index,
            route_count=len(routes),
            unit_count=len(units),
        )
        salience = _readonly_f64_vector(
            self.aumann_shapley_salience,
            name="Aumann-Shapley salience",
            size=len(routes),
            nonnegative=True,
        )
        signed = _readonly_f64_vector(
            self.signed_aumann_shapley,
            name="signed Aumann-Shapley current contribution",
            size=len(routes),
        )
        sidecar = _readonly_f64_vector(
            self.signed_g_delta_h,
            name="signed G Delta h removal disclosure",
            size=len(routes),
        )
        sample_count = int(self.sample_count)
        total_weight = float(self.total_weight)
        if (
            isinstance(self.sample_count, bool)
            or sample_count <= 0
            or not math.isfinite(total_weight)
            or total_weight <= 0.0
        ):
            raise ValueError(
                "exact gate sample count and total weight must be positive"
            )
        if self.equation_id != CONSERVING_ATTRIBUTION_EQUATION_ID:
            raise ValueError("unsupported conserving attribution equation")
        if self.positive_removal_orientation_id != POSITIVE_REMOVAL_ORIENTATION_ID:
            raise ValueError("unsupported positive-removal orientation")
        if self.loss_rescaling_law != LOSS_RESCALING_LAW:
            raise ValueError("unsupported exact gate loss-rescaling law")
        if self.schema != EXACT_GATE_ATTRIBUTION_FIELDS_SCHEMA:
            raise ValueError("unsupported exact gate attribution field schema")
        if self.aggregation_mode not in {
            "dense_reference",
            "route_chunked_sufficient_statistics",
        }:
            raise ValueError("unsupported exact gate aggregation mode")
        if type(self.target_scale_eligible) is not bool:
            raise TypeError("target_scale_eligible must be bool")
        if self.target_scale_eligible and self.aggregation_mode != (
            "route_chunked_sufficient_statistics"
        ):
            raise PermissionError(
                "dense exact gate fields cannot carry target-scale authority"
            )
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "unit_ids", units)
        object.__setattr__(self, "route_unit_index", route_unit)
        object.__setattr__(self, "aumann_shapley_salience", salience)
        object.__setattr__(self, "signed_aumann_shapley", signed)
        object.__setattr__(self, "signed_g_delta_h", sidecar)
        object.__setattr__(self, "sample_count", sample_count)
        object.__setattr__(self, "total_weight", total_weight)
        object.__setattr__(self, "source_hash", _nonempty(self.source_hash, "source_hash"))
        object.__setattr__(
            self,
            "sample_weight_hash",
            _nonempty(self.sample_weight_hash, "sample_weight_hash"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, "exact gate metadata"),
        )

    @property
    def signed_positive_removal_damage(self) -> np.ndarray:
        """Return ``-E[G*phi*c]`` without changing the owner score column."""

        return _readonly_numpy(-self.signed_aumann_shapley)

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="v1 exact gate route axis",
        )

    @property
    def unit_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.unit_ids,
            name="v1 exact gate unit axis",
        )

    @property
    def fields_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-exact-gate-fields-",
        )


def exact_gate_attribution_fields_from_contexts(
    *,
    route_ids: Sequence[str],
    unit_ids: Sequence[str],
    route_unit_index: Sequence[int] | np.ndarray,
    post_relu_cotangent: Sequence[Sequence[float]] | np.ndarray,
    baseline_preactivation: Sequence[Sequence[float]] | np.ndarray,
    bias_preactivation: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
    route_contribution: Sequence[Sequence[float]] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray,
    source_hash: str,
    sample_weight_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> V1ExactGateAttributionFields:
    """Reference aggregation of exact gate attribution over contexts.

    The dense context-by-route input makes this a numerical reference helper,
    not the ImageNet-scale storage path.  Production capture must aggregate
    the same sufficient statistics route-chunk by route-chunk.
    """

    routes = canonical_unique_string_axis(route_ids, name="v1 exact gate route axis")
    units = canonical_unique_string_axis(unit_ids, name="v1 exact gate unit axis")
    route_unit = _readonly_route_unit_index(
        route_unit_index,
        route_count=len(routes),
        unit_count=len(units),
    )
    cotangent = _readonly_f64_matrix(
        post_relu_cotangent,
        name="post_relu_cotangent",
    )
    if cotangent.shape[1] != len(units):
        raise ValueError("post_relu_cotangent must align to the unit axis")
    preactivation = _readonly_f64_matrix(
        baseline_preactivation,
        name="baseline_preactivation",
        shape=cotangent.shape,
    )
    contribution = _readonly_f64_matrix(
        route_contribution,
        name="route_contribution",
        shape=(cotangent.shape[0], len(routes)),
    )
    weights = _readonly_f64_vector(
        sample_weights,
        name="sample_weights",
        size=cotangent.shape[0],
        nonnegative=True,
    )
    total_weight = float(np.sum(weights, dtype=np.float64))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("sample_weights must have positive finite mass")
    bias = np.asarray(bias_preactivation, dtype=np.float64)
    try:
        broadcast_bias = np.broadcast_to(bias, preactivation.shape)
    except ValueError as exc:
        raise ValueError(
            "bias_preactivation must broadcast over the unit field"
        ) from exc
    if not np.isfinite(broadcast_bias).all():
        raise ValueError("bias_preactivation must be finite")
    normalized = weights / total_weight
    fraction = relu_path_open_fraction(preactivation, broadcast_bias)
    route_fraction = fraction[:, route_unit]
    route_cotangent = cotangent[:, route_unit]
    shares = route_cotangent * route_fraction * contribution
    singleton_delta = (
        np.maximum(preactivation[:, route_unit] - contribution, 0.0)
        - np.maximum(preactivation[:, route_unit], 0.0)
    )
    g_delta_h = route_cotangent * singleton_delta
    return V1ExactGateAttributionFields(
        route_ids=routes,
        unit_ids=units,
        route_unit_index=route_unit,
        aumann_shapley_salience=np.sum(
            np.abs(shares) * normalized[:, None],
            axis=0,
            dtype=np.float64,
        ),
        signed_aumann_shapley=np.sum(
            shares * normalized[:, None],
            axis=0,
            dtype=np.float64,
        ),
        signed_g_delta_h=np.sum(
            g_delta_h * normalized[:, None],
            axis=0,
            dtype=np.float64,
        ),
        sample_count=int(cotangent.shape[0]),
        total_weight=total_weight,
        source_hash=_nonempty(source_hash, "source_hash"),
        sample_weight_hash=_nonempty(sample_weight_hash, "sample_weight_hash"),
        metadata={
            **({} if metadata is None else dict(metadata)),
            "semantic_corrected_default": True,
            "canonical_exact_gate_correction": True,
            "signed_aumann_shapley_orientation": "current_contribution",
            "signed_positive_removal_damage_available": True,
            "g_delta_h_orientation": "positive_removal",
            "g_delta_h_disclosure_column_present": True,
            "dense_context_route_reference_only": True,
            "raw_context_route_matrix_materialized": True,
            "equation_id": CONSERVING_ATTRIBUTION_EQUATION_ID,
        },
    )


def exact_gate_attribution_fields_from_sufficient_statistics(
    *,
    route_ids: Sequence[str],
    unit_ids: Sequence[str],
    route_unit_index: Sequence[int] | np.ndarray,
    aumann_shapley_abs_weighted_sum: Sequence[float] | np.ndarray,
    signed_aumann_shapley_weighted_sum: Sequence[float] | np.ndarray,
    signed_g_delta_h_weighted_sum: Sequence[float] | np.ndarray,
    total_weight: float,
    sample_count: int,
    source_hash: str,
    sample_weight_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> V1ExactGateAttributionFields:
    """Build target-scale fields from route-chunked exact-gate moments."""

    routes = canonical_unique_string_axis(route_ids, name="v1 exact gate route axis")
    units = canonical_unique_string_axis(unit_ids, name="v1 exact gate unit axis")
    route_unit = _readonly_route_unit_index(
        route_unit_index,
        route_count=len(routes),
        unit_count=len(units),
    )
    weight = float(total_weight)
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("total_weight must be finite and positive")
    count = int(sample_count)
    if isinstance(sample_count, bool) or count <= 0:
        raise ValueError("sample_count must be a positive integer")
    abs_sum = _readonly_f64_vector(
        aumann_shapley_abs_weighted_sum,
        name="Aumann-Shapley absolute weighted sum",
        size=len(routes),
        nonnegative=True,
    )
    signed_sum = _readonly_f64_vector(
        signed_aumann_shapley_weighted_sum,
        name="signed Aumann-Shapley weighted sum",
        size=len(routes),
    )
    sidecar_sum = _readonly_f64_vector(
        signed_g_delta_h_weighted_sum,
        name="signed G Delta h weighted sum",
        size=len(routes),
    )
    return V1ExactGateAttributionFields(
        route_ids=routes,
        unit_ids=units,
        route_unit_index=route_unit,
        aumann_shapley_salience=abs_sum / weight,
        signed_aumann_shapley=signed_sum / weight,
        signed_g_delta_h=sidecar_sum / weight,
        sample_count=count,
        total_weight=weight,
        source_hash=_nonempty(source_hash, "source_hash"),
        sample_weight_hash=_nonempty(sample_weight_hash, "sample_weight_hash"),
        aggregation_mode="route_chunked_sufficient_statistics",
        target_scale_eligible=True,
        metadata={
            **({} if metadata is None else dict(metadata)),
            "semantic_corrected_default": True,
            "canonical_exact_gate_correction": True,
            "signed_aumann_shapley_orientation": "current_contribution",
            "signed_positive_removal_damage_available": True,
            "g_delta_h_orientation": "positive_removal",
            "g_delta_h_disclosure_column_present": True,
            "route_chunked_sufficient_statistics": True,
            "raw_context_route_matrix_materialized": False,
            "equation_id": CONSERVING_ATTRIBUTION_EQUATION_ID,
        },
    )


@dataclass(frozen=True, slots=True)
class V1ConservingAttributionCanaryReceipt:
    """Numerical witness for one reducer's all-route conservation identity."""

    route_drive_max_abs_error: float
    attribution_max_abs_error: float
    absolute_tolerance: float
    route_drive_identity_passed: bool
    attribution_identity_passed: bool
    equation_id: str = CONSERVING_ATTRIBUTION_EQUATION_ID
    schema: str = CONSERVING_ATTRIBUTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "route_drive_max_abs_error",
            "attribution_max_abs_error",
            "absolute_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        for name in ("route_drive_identity_passed", "attribution_identity_passed"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if self.equation_id != CONSERVING_ATTRIBUTION_EQUATION_ID:
            raise ValueError("unsupported conserving-attribution equation")
        if self.schema != CONSERVING_ATTRIBUTION_SCHEMA:
            raise ValueError("unsupported conserving-attribution schema")

    @property
    def passed(self) -> bool:
        return self.route_drive_identity_passed and self.attribution_identity_passed

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-conserving-attribution-canary-",
        )


def audit_conserving_relu_reduction(
    *,
    summed_route_drive: Any,
    summed_route_current_attribution: Any,
    post_activation_cotangent: Any,
    preactivation: Any,
    bias_baseline: Any,
    absolute_tolerance: float = 1e-10,
) -> V1ConservingAttributionCanaryReceipt:
    """Audit route-drive completeness and the conserving attribution sum."""

    tolerance = _finite_nonnegative(absolute_tolerance, "absolute_tolerance")

    def as_numpy(value: Any) -> np.ndarray:
        if _is_torch_tensor(value):
            return value.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(value, dtype=np.float64)

    drive = as_numpy(summed_route_drive)
    observed = as_numpy(summed_route_current_attribution)
    pre = as_numpy(preactivation)
    bias = as_numpy(bias_baseline)
    expected_drive = pre - bias
    expected_attribution = as_numpy(
        conserving_relu_total_current_attribution(
            post_activation_cotangent,
            preactivation,
            bias_baseline,
        )
    )
    if not all(
        np.isfinite(value).all()
        for value in (drive, observed, expected_drive, expected_attribution)
    ):
        raise ValueError("conservation audit inputs must be finite")
    try:
        drive_error = float(np.max(np.abs(drive - expected_drive), initial=0.0))
        attribution_error = float(
            np.max(np.abs(observed - expected_attribution), initial=0.0)
        )
    except ValueError as exc:
        raise ValueError("conservation audit axes do not align") from exc
    return V1ConservingAttributionCanaryReceipt(
        route_drive_max_abs_error=drive_error,
        attribution_max_abs_error=attribution_error,
        absolute_tolerance=tolerance,
        route_drive_identity_passed=drive_error <= tolerance,
        attribution_identity_passed=attribution_error <= tolerance,
    )


__all__ = [
    "CONSERVING_ATTRIBUTION_EQUATION_ID",
    "CONSERVING_ATTRIBUTION_SCHEMA",
    "EXACT_GATE_ATTRIBUTION_FIELDS_SCHEMA",
    "EXACT_SINGLETON_DISCLOSURE_ID",
    "LOSS_RESCALING_LAW",
    "POSITIVE_REMOVAL_ORIENTATION_ID",
    "RELU_PATH_FRACTION_EQUATION_ID",
    "V1ConservingAttributionCanaryReceipt",
    "V1ExactGateAttributionFields",
    "audit_conserving_relu_reduction",
    "conserving_relu_cotangent",
    "conserving_relu_total_current_attribution",
    "exact_gate_attribution_fields_from_contexts",
    "exact_gate_attribution_fields_from_sufficient_statistics",
    "exact_relu_singleton_removal_disclosure",
    "relu_path_open_fraction",
]
