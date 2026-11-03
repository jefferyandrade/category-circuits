"""Production-scale packed global selector for canonical modern v1.

This module owns the target-scale proposal step: one global physical route
axis, corrected path budgets, complete-set quadratic trust budgets, and packed
candidate masks.  Layer labels, when supplied, are only a dynamic legality
guard that prevents a block from being emptied.  They never define queues,
quotas, scores, or normalization factors.

The small ``modern_v1_geometric_selector`` module remains the exact reference
oracle.  This module deliberately emits a frozen, packed proposal frontier for
the mask oracle and does not claim a globally optimal powerset solution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from .candidates import HardMask
from .contracts import content_hash
from mechanistic_systems.geometry.trust import (
    GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA,
    GlobalTrustGeometryDerivationReceipt,
)
from .modern_v1_geometric_action import V1CorrectedAttributionFields
from .modern_v1_geometric_budgets import (
    V1GeometricTrustBudgets,
    same_geometric_coordinate,
)
from .modern_v1_neutral_runtime import FixedAffineBCECalibration
from .modern_v1_sampled_geometric_provenance import (
    SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA,
    V1SampledProposalPositionCertificate,
    V1SampledSelectorChartProvenance,
    V1SampledSnapshotSelectorCertificate,
)
from .modern_v1_candidate_composition import (
    FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
    V1FiniteRemovalDomain,
    V1PackedRemovalAction,
)


PRODUCTION_SELECTOR_POLICY_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_policy_v1"
)
PRODUCTION_SELECTOR_PROBLEM_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_problem_v1"
)
PRODUCTION_SELECTOR_SCALARIZATION_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_scalarization_v1"
)
PRODUCTION_SELECTOR_CANDIDATE_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_candidate_v1"
)
PRODUCTION_SELECTOR_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_receipt_v1"
)
PRODUCTION_SELECTOR_RESULT_SCHEMA: Final = (
    "partition_free_v1_production_geometric_selector_result_v1"
)
PRODUCTION_SELECTOR_ID: Final = "packed_global_dimensionless_burden_frontier_v1"
PRODUCTION_SELECTOR_DIAGNOSTIC_ID: Final = (
    "packed_global_dimensionless_burden_frontier_diagnostic_v1"
)
PRODUCTION_SELECTOR_NO_PARTITION_POLICY_ID: Final = (
    "one_global_physical_axis_layers_only_survivor_legality_no_sign_partitions_v1"
)
PRODUCTION_SELECTOR_SCALARIZATION_ID: Final = (
    "conditional_remaining_budget_gauge_registered_prefix_v1"
)
PRODUCTION_SELECTOR_DIAGNOSTIC_SCALARIZATION_ID: Final = (
    "diagnostic_conditional_remaining_budget_gauge_registered_prefix_v1"
)
PRODUCTION_SELECTOR_CHART_PROVENANCE_SCHEMA: Final = (
    "partition_free_v1_production_selector_chart_provenance_v1"
)
PRODUCTION_SELECTOR_ATTRIBUTION_COORDINATES_SCHEMA: Final = (
    "partition_free_v1_corrected_attribution_chart_coordinates_v1"
)
PRODUCTION_SELECTOR_CANONICAL_POLICY_ID: Final = (
    "modern-v1-production-packed-geometric-selector-default"
)
PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID: Final = (
    "conditional_budget_gauge_resource_normalized"
)
PRODUCTION_SELECTOR_RAW_TARGET_LOGIT_COORDINATE: Final = "raw_target_logit"
PRODUCTION_SELECTOR_COMPLETE_Q_ID: Final = (
    "certified_set_Q_norm_Hz_squared_plus_min_rho_cardinality_or_diag_l1_v1"
)
PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_fd_residual_spectral_certificate_v1"
)
PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_STATEMENT: Final = (
    "residual_spectral_bound_upper_bounds_frozen_chart_fd_residual_lambda_max"
)
PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND: Final = (
    "global_trust_geometry_derivation_receipt"
)
PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND: Final = (
    "sampled_stratified_fd_snapshot_numerical_certificate"
)
PRODUCTION_SELECTOR_SYNTHETIC_FD_CERTIFICATE_KIND: Final = (
    "synthetic_selector_diagnostic_fd_residual_certificate"
)
GLOBAL_TRUST_DERIVED_MEAN_PREFIX: Final = "irg-global-trust-derived-mean-"
GLOBAL_TRUST_DERIVED_CENTERED_FACTOR_PREFIX: Final = (
    "irg-global-trust-derived-centered-factor-"
)
GLOBAL_TRUST_DERIVED_RESIDUAL_DIAGONAL_PREFIX: Final = (
    "irg-global-trust-derived-residual-diagonal-"
)
LOSS_RESCALING_LAW: Final = (
    "under a positive coordinate rescaling of the raw-logit displacement "
    "geometry, corrected salience and tau_F scale linearly, metric columns and "
    "tau_L scale linearly, Q and tau_Q scale quadratically, and lexicographic "
    "route identities are invariant when geometry and budgets are transported "
    "together"
)
EXACT_REFERENCE_MAX_SUBSET_EVALUATIONS: Final = 250_000
UNIT_SCALARIZATION_TOLERANCE: Final = 1e-12
ROUNDING_SAFETY_ULPS: Final = 64.0
FLOAT64_EPSILON: Final = float(np.finfo(np.float64).eps)

SelectorBackend = Literal["numpy", "torch"]
InputArrayPolicy = Literal["copy_to_bytes", "admit_readonly_c_contiguous"]
TiePolicy = Literal["refuse_threshold_ties", "physical_index_deterministic_reference"]
CandidateConstruction = Literal[
    "fixed_count_low_rank_lagrangian_dual",
    "single_order_low_rank_lagrangian_dual_prefix",
    "registered_prefix_conditional_budget_gauge",
    "single_order_corrected_salience_primary_prefix",
    "exact_reference_oracle_repacked",
]


@dataclass(frozen=True, slots=True)
class V1CorrectedAttributionChartCoordinates:
    """Typed producer coordinates for one corrected-attribution field.

    A route axis and an opaque source hash are not enough to establish that a
    field belongs to the selector's current chart.  These coordinates bind the
    field to the current mask, campaign/query/model/data preprocessing, and
    the exact raw-logit stream that produced the sufficient statistics.
    """

    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    campaign_binding_hash: str
    query_id: str
    query_hash: str
    target_class_id: int
    model_seal_hash: str
    model_state_hash: str
    preprocessing_hash: str
    source_context_axis_hash: str
    source_label_map_hash: str
    context_population_hash: str
    sampling_law_hash: str
    train_bank_source_hash: str
    source_measurement_hash: str
    raw_stream_identity_hash: str
    raw_stream_receipt_hash: str
    raw_stream_context_ids_hash: str
    destination_transport_hash: str
    schema: str = PRODUCTION_SELECTOR_ATTRIBUTION_COORDINATES_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "campaign_binding_hash",
            "query_id",
            "query_hash",
            "model_seal_hash",
            "model_state_hash",
            "preprocessing_hash",
            "source_context_axis_hash",
            "source_label_map_hash",
            "context_population_hash",
            "sampling_law_hash",
            "train_bank_source_hash",
            "source_measurement_hash",
            "raw_stream_identity_hash",
            "raw_stream_receipt_hash",
            "raw_stream_context_ids_hash",
            "destination_transport_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        target = _nonnegative_int(self.target_class_id, "target_class_id")
        object.__setattr__(self, "target_class_id", target)
        if self.schema != PRODUCTION_SELECTOR_ATTRIBUTION_COORDINATES_SCHEMA:
            raise ValueError("unsupported corrected-attribution chart coordinates schema")

    @property
    def coordinates_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-corrected-attribution-chart-coordinates-",
        )


@dataclass(frozen=True, slots=True)
class V1ProductionSelectorChartProvenance:
    """Typed authority binding for one current-mask production chart.

    The selector never infers scientific authority from untyped hashes.  A
    production FD chart must carry the exact corrected-attribution fields,
    transported analytic trust budgets, and fixed affine BCE calibration that
    jointly define its coordinates.  Resource, eligibility, and legality
    remain separate identities so none can impersonate the attribution or
    budget owner.
    """

    corrected_attribution_fields: V1CorrectedAttributionFields = field(
        repr=False,
        compare=False,
    )
    corrected_attribution_coordinates: V1CorrectedAttributionChartCoordinates
    geometric_trust_budgets: V1GeometricTrustBudgets
    fixed_affine_bce_calibration: FixedAffineBCECalibration
    resource_plane: str
    resource_cost_policy_hash: str
    resource_cost_hash: str
    eligibility_receipt_hash: str
    legality_policy_hash: str
    schema: str = PRODUCTION_SELECTOR_CHART_PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        fields = self.corrected_attribution_fields
        coordinates = self.corrected_attribution_coordinates
        budgets = self.geometric_trust_budgets
        calibration = self.fixed_affine_bce_calibration
        if type(fields) is not V1CorrectedAttributionFields:
            raise TypeError(
                "corrected_attribution_fields must be V1CorrectedAttributionFields"
            )
        if not fields.production_eligible or not fields.target_scale_authority:
            raise PermissionError(
                "production selector requires target-scale-authoritative "
                "route-resolved corrected attribution"
            )
        if fields.metadata.get("attribution_coordinate") != (
            PRODUCTION_SELECTOR_RAW_TARGET_LOGIT_COORDINATE
        ):
            raise ValueError("corrected attribution target-scale coordinate differs")
        if fields.metadata.get("affine_calibration_applied_to_attribution") is not False:
            raise ValueError("corrected attribution must remain in raw-logit units")
        if fields.metadata.get("calibration_offset_applied_to_attribution") is not False:
            raise ValueError("calibration offset cannot enter corrected attribution")
        if type(coordinates) is not V1CorrectedAttributionChartCoordinates:
            raise TypeError(
                "corrected_attribution_coordinates must be "
                "V1CorrectedAttributionChartCoordinates"
            )
        if coordinates.ordered_route_axis_hash != fields.route_axis_hash:
            raise ValueError("corrected-attribution coordinate route axis differs")
        if type(budgets) is not V1GeometricTrustBudgets:
            raise TypeError("geometric_trust_budgets must be V1GeometricTrustBudgets")
        if coordinates.current_mask_hash != budgets.current_mask_hash:
            raise ValueError("corrected-attribution coordinate current mask differs")
        if type(calibration) is not FixedAffineBCECalibration:
            raise TypeError(
                "fixed_affine_bce_calibration must be FixedAffineBCECalibration"
            )
        if not same_geometric_coordinate(
            budgets.raw_logit_to_calibrated_logit_slope,
            calibration.scale,
        ):
            raise ValueError("trust-budget slope differs from fixed affine calibration")
        for observed, expected, label in (
            (coordinates.query_id, calibration.query_id, "query ID"),
            (coordinates.query_hash, calibration.query_hash, "query hash"),
            (
                coordinates.target_class_id,
                calibration.target_class_id,
                "target class",
            ),
            (
                coordinates.model_state_hash,
                calibration.model_state_hash,
                "model state",
            ),
            (
                coordinates.source_context_axis_hash,
                calibration.calibration_context_axis_hash,
                "source context axis",
            ),
            (
                coordinates.source_label_map_hash,
                calibration.source_label_map_hash,
                "source label map",
            ),
        ):
            if observed != expected:
                raise ValueError(f"corrected-attribution coordinate {label} differs")
        field_coordinate_values = {
            "current_mask_hash": coordinates.current_mask_hash,
            "physical_route_axis_hash": coordinates.physical_route_axis_hash,
            "ordered_route_axis_hash": coordinates.ordered_route_axis_hash,
            "campaign_binding_hash": coordinates.campaign_binding_hash,
            "query_id": coordinates.query_id,
            "query_hash": coordinates.query_hash,
            "target_class_id": coordinates.target_class_id,
            "model_seal_hash": coordinates.model_seal_hash,
            "model_state_hash": coordinates.model_state_hash,
            "preprocessing_hash": coordinates.preprocessing_hash,
            "source_context_axis_hash": coordinates.source_context_axis_hash,
            "source_label_map_hash": coordinates.source_label_map_hash,
            "context_population_hash": coordinates.context_population_hash,
            "sampling_law_hash": coordinates.sampling_law_hash,
            "train_bank_source_hash": coordinates.train_bank_source_hash,
            "source_request_id": coordinates.source_measurement_hash,
            "positive_removal_stream_identity_hash": (
                coordinates.raw_stream_identity_hash
            ),
            "streaming_receipt_hash": coordinates.raw_stream_receipt_hash,
            "raw_stream_context_ids_hash": coordinates.raw_stream_context_ids_hash,
            "destination_transport_hash": coordinates.destination_transport_hash,
        }
        for name, expected in field_coordinate_values.items():
            if fields.metadata.get(name) != expected:
                raise ValueError(
                    f"corrected-attribution producer coordinate {name} differs"
                )
        source_measurements = tuple(fields.metadata.get("source_measurement_hashes", ()))
        if source_measurements != (coordinates.source_measurement_hash,):
            raise ValueError("corrected-attribution source measurement axis differs")
        streaming_receipt = fields.metadata.get("streaming_receipt")
        if not isinstance(streaming_receipt, Mapping) or any(
            (
                streaming_receipt.get("schema")
                != "partition_free_v1_raw_target_logit_stream_receipt_v1",
                streaming_receipt.get("receipt_hash")
                != coordinates.raw_stream_receipt_hash,
                streaming_receipt.get("request_id")
                != coordinates.source_measurement_hash,
                streaming_receipt.get("context_ids_hash")
                != coordinates.raw_stream_context_ids_hash,
                streaming_receipt.get("route_count") != len(fields.route_ids),
                streaming_receipt.get("positive_removal_stream_coordinate")
                != PRODUCTION_SELECTOR_RAW_TARGET_LOGIT_COORDINATE,
                streaming_receipt.get("positive_removal_stream_identity_hash")
                != coordinates.raw_stream_identity_hash,
                streaming_receipt.get("raw_target_logit_unit_pullback") is not True,
                streaming_receipt.get("affine_calibration_applied") is not False,
                streaming_receipt.get("context_specific_loss_pullback_applied")
                is not False,
                streaming_receipt.get("one_physical_scorer_invocation") is not True,
                streaming_receipt.get("loss_only_stream_tee") is not True,
                streaming_receipt.get("loss_only_stream_tee_child_count") != 2,
                streaming_receipt.get("shared_positive_removal_rows") is not True,
            )
        ):
            raise ValueError("corrected-attribution raw-stream receipt differs")
        if not same_geometric_coordinate(
            budgets.quadratic_trust_energy_budget,
            budgets.metric_length_budget * budgets.metric_length_budget,
        ):
            raise ValueError("trust-budget tau_Q does not equal tau_L squared")
        if fields.source_attribution_hash == budgets.budget_hash:
            raise ValueError("corrected-attribution and trust-budget hashes must be distinct")
        for name in (
            "resource_plane",
            "resource_cost_policy_hash",
            "resource_cost_hash",
            "eligibility_receipt_hash",
            "legality_policy_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.schema != PRODUCTION_SELECTOR_CHART_PROVENANCE_SCHEMA:
            raise ValueError("unsupported production selector chart provenance schema")

    @property
    def provenance_hash(self) -> str:
        fields = self.corrected_attribution_fields
        budgets = self.geometric_trust_budgets
        calibration = self.fixed_affine_bce_calibration
        return content_hash(
            {
                "schema": self.schema,
                "corrected_attribution_fields_hash": fields.fields_hash,
                "corrected_attribution_coordinates_hash": (
                    self.corrected_attribution_coordinates.coordinates_hash
                ),
                "source_attribution_hash": fields.source_attribution_hash,
                "sample_weight_hash": fields.sample_weight_hash,
                "geometric_trust_budget_hash": budgets.budget_hash,
                "fixed_affine_bce_calibration_hash": calibration.content_hash,
                "resource_plane": self.resource_plane,
                "resource_cost_policy_hash": self.resource_cost_policy_hash,
                "resource_cost_hash": self.resource_cost_hash,
                "eligibility_receipt_hash": self.eligibility_receipt_hash,
                "legality_policy_hash": self.legality_policy_hash,
            },
            prefix="partition-free-v1-production-selector-chart-provenance-",
        )


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


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_positive(value: object, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _roundoff_tolerance(*values: float, ulps: float = ROUNDING_SAFETY_ULPS) -> float:
    """Return a unit-covariant float64 roundoff window with no absolute floor."""

    if not values:
        return 0.0
    scale = max(abs(float(value)) for value in values)
    if not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return float(ulps) * FLOAT64_EPSILON * scale


def _exceeds_upper_bound(value: float, bound: float) -> bool:
    """Safety-directed upper-bound test: real positive gaps cannot hide at 0."""

    lhs = float(value)
    rhs = float(bound)
    if lhs <= rhs:
        return False
    return (lhs - rhs) > _roundoff_tolerance(lhs, rhs)


def _roundoff_close(left: float, right: float) -> bool:
    lhs = float(left)
    rhs = float(right)
    return abs(lhs - rhs) <= _roundoff_tolerance(lhs, rhs)


def _remaining_budget(
    budget: float,
    used: float,
) -> tuple[float, bool, bool]:
    """Return covariantly snapped remainder, snap flag, and genuine deficit.

    A tiny negative subtraction residue at the boundary is the same geometric
    point as zero.  A negative gap outside the scale-relative float64 window
    is a real exhausted budget and remains negative/fail-closed.
    """

    bound = float(budget)
    consumed = float(used)
    remainder = bound - consumed
    if remainder >= 0.0:
        return remainder, False, False
    if abs(remainder) <= _roundoff_tolerance(bound, consumed):
        return 0.0, True, False
    return remainder, False, True


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


def _readonly_empty_f64_vector() -> np.ndarray:
    return np.frombuffer(b"", dtype="<f8")


def _readonly_bool_vector(
    value: Sequence[bool] | np.ndarray,
    *,
    name: str,
    size: int | None = None,
    input_array_policy: InputArrayPolicy,
) -> np.ndarray:
    if input_array_policy == "admit_readonly_c_contiguous":
        array = np.asarray(value).reshape(-1)
        if array.dtype != np.dtype(np.bool_) or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous bool for no-copy admission")
        if array.flags.writeable:
            raise ValueError(f"{name} must be read-only for no-copy admission")
        if not array.size:
            raise ValueError(f"{name} must be nonempty")
        if size is not None and array.size != size:
            raise ValueError(f"{name} must align to the physical route axis")
        return array
    array = np.asarray(value, dtype=np.bool_).reshape(-1)
    if not array.size:
        raise ValueError(f"{name} must be nonempty")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must align to the physical route axis")
    return np.frombuffer(np.ascontiguousarray(array).tobytes(order="C"), dtype=np.bool_)


def _readonly_f64_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    size: int | None = None,
    nonnegative: bool = False,
    input_array_policy: InputArrayPolicy,
) -> np.ndarray:
    if input_array_policy == "admit_readonly_c_contiguous":
        array = np.asarray(value).reshape(-1)
        if array.dtype != np.dtype(np.float64) or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float64 for no-copy admission")
        if array.flags.writeable:
            raise ValueError(f"{name} must be read-only for no-copy admission")
        if not array.size:
            raise ValueError(f"{name} must be nonempty")
        if size is not None and array.size != size:
            raise ValueError(f"{name} must align to the physical route axis")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        if nonnegative and bool(np.any(array < 0.0)):
            raise ValueError(f"{name} must be nonnegative")
        return array
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size:
        raise ValueError(f"{name} must be nonempty")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must align to the physical route axis")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
        count=array.size,
    )


def _readonly_f64_matrix(
    value: np.ndarray,
    *,
    name: str,
    columns: int,
    input_array_policy: InputArrayPolicy,
) -> np.ndarray:
    if input_array_policy == "admit_readonly_c_contiguous":
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float64) or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous float64 for no-copy admission")
        if array.flags.writeable:
            raise ValueError(f"{name} must be read-only for no-copy admission")
        if array.ndim != 2 or array.shape[1] != columns:
            raise ValueError(f"{name} must have shape (rank, route_count)")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        return array
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rank, route_count)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.frombuffer(
        np.ascontiguousarray(array, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
        count=array.size,
    ).reshape(array.shape)


def _dimensionless_scalar(value: float, budget: float) -> float:
    if budget > 0.0:
        return float(value / budget)
    if value <= 0.0:
        return 0.0
    return math.inf


def _count_ladder(
    declared: Sequence[int] | np.ndarray,
    *,
    selectable_count: int,
    maximum_removal_count: int | None = None,
) -> tuple[int, ...]:
    if isinstance(declared, (str, bytes)):
        raise TypeError("count ladder must contain integer counts")
    if maximum_removal_count is None:
        count_cap = selectable_count
    else:
        count_cap = _positive_int(maximum_removal_count, "maximum_removal_count")
        count_cap = min(count_cap, selectable_count)
    counts = []
    for value in tuple(declared):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("count ladder must contain exact integer counts")
        count = int(value)
        if count <= 0:
            raise ValueError("count ladder entries must be positive")
        if count <= count_cap:
            counts.append(count)
    if count_cap > 0:
        counts.extend([1, min(2, count_cap), min(4, count_cap)])
    return tuple(sorted(set(counts)))


@dataclass(frozen=True, slots=True)
class V1ProductionScalarization:
    """Registered dimensionless weights for fixed-count Lagrangian selection."""

    scalarization_id: str
    path_weight: float = 1.0
    quadratic_weight: float = 1.0
    resource_normalized: bool = True
    schema: str = PRODUCTION_SELECTOR_SCALARIZATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scalarization_id",
            _nonempty(self.scalarization_id, "scalarization_id"),
        )
        object.__setattr__(
            self,
            "path_weight",
            _finite_nonnegative(self.path_weight, "path_weight"),
        )
        object.__setattr__(
            self,
            "quadratic_weight",
            _finite_nonnegative(self.quadratic_weight, "quadratic_weight"),
        )
        if self.path_weight == 0.0 and self.quadratic_weight == 0.0:
            raise ValueError("at least one scalarization weight must be positive")
        if type(self.resource_normalized) is not bool:
            raise TypeError("resource_normalized must be bool")
        if self.schema != PRODUCTION_SELECTOR_SCALARIZATION_SCHEMA:
            raise ValueError("unsupported production scalarization schema")

    @property
    def scalarization_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-scalarization-",
        )


@dataclass(frozen=True, slots=True)
class V1ProductionLegalityBlock:
    """Compact contiguous survivor-guard block on the physical route axis."""

    block_id: str
    start: int
    stop: int
    schema: str = "partition_free_v1_production_legality_block_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _nonempty(self.block_id, "block_id"))
        start = _nonnegative_int(self.start, "block start")
        stop = _positive_int(self.stop, "block stop")
        if stop <= start:
            raise ValueError("legality block stop must exceed start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        if self.schema != "partition_free_v1_production_legality_block_v1":
            raise ValueError("unsupported production legality block schema")

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.stop)


@dataclass(frozen=True, slots=True)
class V1ProductionGeometricSelectorPolicy:
    """Production bounds for packed global frontier construction."""

    policy_id: str = PRODUCTION_SELECTOR_CANONICAL_POLICY_ID
    count_ladder: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128)
    scalarizations: tuple[V1ProductionScalarization, ...] = (
        V1ProductionScalarization(
            scalarization_id=PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID
        ),
    )
    backend: SelectorBackend = "numpy"
    torch_device: str = "cpu"
    chunk_size: int = 1 << 16
    max_frontier_candidates: int = 128
    maximum_removal_count: int | None = None
    dual_iterations: int = 48
    dual_initial_step_size: float = 0.75
    tie_tolerance: float = 1e-12
    tie_policy: TiePolicy = "physical_index_deterministic_reference"
    exact_reference_route_limit: int = 0
    input_array_policy: InputArrayPolicy = "copy_to_bytes"
    diagnostic_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PRODUCTION_SELECTOR_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _nonempty(self.policy_id, "policy_id"))
        for name in ("chunk_size", "max_frontier_candidates", "dual_iterations"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.maximum_removal_count is not None:
            object.__setattr__(
                self,
                "maximum_removal_count",
                _positive_int(self.maximum_removal_count, "maximum_removal_count"),
            )
        object.__setattr__(
            self,
            "dual_initial_step_size",
            _finite_positive(self.dual_initial_step_size, "dual_initial_step_size"),
        )
        object.__setattr__(
            self,
            "exact_reference_route_limit",
            _nonnegative_int(self.exact_reference_route_limit, "exact_reference_route_limit"),
        )
        object.__setattr__(
            self,
            "tie_tolerance",
            _finite_nonnegative(self.tie_tolerance, "tie_tolerance"),
        )
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("unsupported production selector backend")
        object.__setattr__(self, "torch_device", _nonempty(self.torch_device, "torch_device"))
        if self.tie_policy not in {
            "refuse_threshold_ties",
            "physical_index_deterministic_reference",
        }:
            raise ValueError("unsupported production selector tie policy")
        if self.input_array_policy not in {"copy_to_bytes", "admit_readonly_c_contiguous"}:
            raise ValueError("unsupported production selector input array policy")
        scalarizations = tuple(self.scalarizations)
        if not scalarizations or any(
            type(item) is not V1ProductionScalarization for item in scalarizations
        ):
            raise TypeError("policy requires registered scalarization objects")
        if len({item.scalarization_id for item in scalarizations}) != len(scalarizations):
            raise ValueError("scalarization IDs must be unique")
        if type(self.diagnostic_only) is not bool:
            raise TypeError("diagnostic_only must be bool")
        if not self.diagnostic_only:
            if self.policy_id != PRODUCTION_SELECTOR_CANONICAL_POLICY_ID:
                raise ValueError(
                    "production selector policy must use the canonical policy ID; "
                    "alternative policy identities are diagnostic-only"
                )
            if len(scalarizations) != 1:
                raise ValueError(
                    "production selector policy requires exactly one scalarization"
                )
            scalarization = scalarizations[0]
            if (
                scalarization.scalarization_id
                != PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID
                or
                scalarization.path_weight != 1.0
                or scalarization.quadratic_weight != 1.0
                or not scalarization.resource_normalized
            ):
                raise ValueError(
                    "production selector policy requires unit path/quadratic weights "
                    "and resource_normalized=True"
                )
        elif self.policy_id == PRODUCTION_SELECTOR_CANONICAL_POLICY_ID or any(
            item.scalarization_id == PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID
            for item in scalarizations
        ):
            raise ValueError(
                "diagnostic selector policies cannot reuse canonical production IDs"
            )
        object.__setattr__(self, "scalarizations", scalarizations)
        object.__setattr__(self, "count_ladder", tuple(self.count_ladder))
        if self.schema != PRODUCTION_SELECTOR_POLICY_SCHEMA:
            raise ValueError("unsupported production selector policy schema")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))

    @property
    def policy_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-selector-policy-",
        )


@dataclass(frozen=True, slots=True)
class V1ProductionGeometricSelectorProblem:
    """One relinearized packed global selection problem."""

    problem_id: str
    parent_mask: HardMask
    eligible_mask: Sequence[bool] | np.ndarray
    corrected_path_cost: Sequence[float] | np.ndarray
    signed_covector: Sequence[float] | np.ndarray | None
    low_rank_factor: np.ndarray
    residual_diagonal: Sequence[float] | np.ndarray
    residual_spectral_bound: float
    resource_cost: Sequence[float] | np.ndarray | float
    finsler_path_budget: float
    quadratic_trust_energy_budget: float
    source_receipt_hash: str
    eligibility_receipt_hash: str
    fd_residual_certificate_hash: str
    route_axis_hash: str | None = None
    fd_ordered_route_axis_hash: str | None = None
    resource_plane: str | None = None
    resource_cost_policy_hash: str | None = None
    legality_policy_hash: str | None = None
    legality_block_ids: Sequence[str] | None = None
    legality_blocks: Sequence[V1ProductionLegalityBlock | tuple[str, int, int]] | None = None
    minimum_live_per_block: int = 1
    input_array_policy: InputArrayPolicy = "copy_to_bytes"
    fd_residual_certificate_schema: str = PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_SCHEMA
    fd_residual_certificate_kind: str = PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND
    fd_residual_geometry_state_hash: str | None = None
    fd_residual_derivation_receipt: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    chart_provenance: (
        V1ProductionSelectorChartProvenance
        | V1SampledSelectorChartProvenance
        | None
    ) = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PRODUCTION_SELECTOR_PROBLEM_SCHEMA
    uniform_resource_cost: float | None = field(default=None, repr=False)
    fd_residual_sampled_snapshot_certificate: (
        V1SampledSnapshotSelectorCertificate | None
    ) = field(default=None, repr=False, compare=False)
    sampled_proposal_position_certificate: (
        V1SampledProposalPositionCertificate | None
    ) = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_id", _nonempty(self.problem_id, "problem_id"))
        if type(self.parent_mask) is not HardMask:
            raise TypeError("production selector problem requires an exact HardMask")
        axis_hash = (
            self.parent_mask.route_axis_hash
            if self.route_axis_hash is None
            else _nonempty(self.route_axis_hash, "route_axis_hash")
        )
        if axis_hash != self.parent_mask.route_axis_hash:
            raise ValueError("problem route axis differs from parent HardMask")
        if self.input_array_policy not in {"copy_to_bytes", "admit_readonly_c_contiguous"}:
            raise ValueError("unsupported production selector input array policy")
        route_count = self.parent_mask.route_count
        eligible = _readonly_bool_vector(
            self.eligible_mask,
            name="eligible_mask",
            size=route_count,
            input_array_policy=self.input_array_policy,
        )
        path = _readonly_f64_vector(
            self.corrected_path_cost,
            name="corrected_path_cost",
            size=route_count,
            nonnegative=True,
            input_array_policy=self.input_array_policy,
        )
        if self.signed_covector is None:
            signed = None
        else:
            signed = _readonly_f64_vector(
                self.signed_covector,
                name="signed_covector",
                size=route_count,
                input_array_policy=self.input_array_policy,
            )
        factor = _readonly_f64_matrix(
            self.low_rank_factor,
            name="low_rank_factor",
            columns=route_count,
            input_array_policy=self.input_array_policy,
        )
        residual = _readonly_f64_vector(
            self.residual_diagonal,
            name="residual_diagonal",
            size=route_count,
            nonnegative=True,
            input_array_policy=self.input_array_policy,
        )
        declared_uniform = self.uniform_resource_cost
        if isinstance(self.resource_cost, Real) and not isinstance(self.resource_cost, bool):
            resource_scalar = _finite_positive(self.resource_cost, "uniform resource_cost")
            if declared_uniform is not None and not _roundoff_close(
                resource_scalar,
                _finite_positive(declared_uniform, "uniform_resource_cost"),
            ):
                raise ValueError("uniform resource-cost coordinates differ")
            resource = _readonly_empty_f64_vector()
        elif declared_uniform is not None:
            resource_scalar = _finite_positive(
                declared_uniform,
                "uniform_resource_cost",
            )
            replay_resource = np.asarray(self.resource_cost).reshape(-1)
            if replay_resource.size != 0:
                raise ValueError(
                    "uniform_resource_cost replay requires the compact empty resource vector"
                )
            resource = _readonly_empty_f64_vector()
        else:
            resource_scalar = None
            resource = _readonly_f64_vector(
                self.resource_cost,  # type: ignore[arg-type]
                name="resource_cost",
                size=route_count,
                nonnegative=True,
                input_array_policy=self.input_array_policy,
            )
        live = self.parent_mask.to_bool_array()
        selectable = np.logical_and(live, eligible)
        if resource_scalar is None and bool(np.any(resource[selectable] <= 0.0)):
            raise ValueError("resource_cost must be positive on selectable routes")
        block_ids = None
        blocks = None
        minimum = _positive_int(self.minimum_live_per_block, "minimum_live_per_block")
        if minimum != 1:
            raise ValueError(
                "production selector currently supports exactly one live survivor per block"
            )
        if self.legality_block_ids is not None and self.legality_blocks is not None:
            raise ValueError("provide either compact legality_blocks or legacy block IDs, not both")
        if self.legality_blocks is not None:
            normalized_blocks: list[V1ProductionLegalityBlock] = []
            previous_stop = 0
            for raw in self.legality_blocks:
                block = (
                    raw
                    if type(raw) is V1ProductionLegalityBlock
                    else V1ProductionLegalityBlock(
                        block_id=str(raw[0]),
                        start=int(raw[1]),
                        stop=int(raw[2]),
                    )
                )
                if block.start != previous_stop:
                    raise ValueError("compact legality blocks must be contiguous and ordered")
                if block.stop > route_count:
                    raise ValueError("compact legality block exceeds the route axis")
                normalized_blocks.append(block)
                previous_stop = block.stop
            if previous_stop != route_count:
                raise ValueError("compact legality blocks must cover the route axis")
            if len({block.block_id for block in normalized_blocks}) != len(normalized_blocks):
                raise ValueError("compact legality block IDs must be unique")
            blocks = tuple(normalized_blocks)
            deficient = tuple(
                block.block_id
                for block in blocks
                if int(np.count_nonzero(live[block.start : block.stop])) < minimum
            )
            if deficient:
                raise ValueError("parent mask already violates block survivor legality")
        elif self.legality_block_ids is not None:
            if route_count > 1_000_000:
                raise ValueError(
                    "per-route legality_block_ids are reference-only; use compact legality_blocks"
                )
            block_ids = tuple(str(value) for value in self.legality_block_ids)
            if len(block_ids) != route_count or any(not value for value in block_ids):
                raise ValueError("legality_block_ids must align to the route axis")
            live_by_block: dict[str, int] = {}
            for block, is_live in zip(block_ids, live, strict=True):
                if bool(is_live):
                    live_by_block[block] = live_by_block.get(block, 0) + 1
            deficient = tuple(block for block, count in live_by_block.items() if count < minimum)
            if deficient:
                raise ValueError("parent mask already violates block survivor legality")
        object.__setattr__(self, "route_axis_hash", axis_hash)
        object.__setattr__(self, "eligible_mask", eligible)
        object.__setattr__(self, "corrected_path_cost", path)
        object.__setattr__(self, "signed_covector", signed)
        spectral_bound = _finite_nonnegative(
            self.residual_spectral_bound,
            "residual_spectral_bound",
        )
        original_spectral_bound = spectral_bound
        max_residual_diagonal = float(np.max(residual)) if residual.size else 0.0
        tolerance = _roundoff_tolerance(spectral_bound, max_residual_diagonal)
        if (
            spectral_bound < max_residual_diagonal
            and (max_residual_diagonal - spectral_bound) > tolerance
        ):
            raise ValueError(
                "residual_spectral_bound must be at least max(residual_diagonal) "
                "for singleton-certified residual coverage"
            )
        spectral_bound_snapped = False
        if spectral_bound < max_residual_diagonal:
            spectral_bound = max_residual_diagonal
            spectral_bound_snapped = True
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_diagonal", residual)
        object.__setattr__(self, "residual_spectral_bound", spectral_bound)
        object.__setattr__(self, "resource_cost", resource)
        object.__setattr__(self, "uniform_resource_cost", resource_scalar)
        object.__setattr__(self, "legality_block_ids", block_ids)
        object.__setattr__(self, "legality_blocks", blocks)
        object.__setattr__(
            self,
            "finsler_path_budget",
            _finite_nonnegative(self.finsler_path_budget, "finsler_path_budget"),
        )
        object.__setattr__(
            self,
            "quadratic_trust_energy_budget",
            _finite_nonnegative(
                self.quadratic_trust_energy_budget,
                "quadratic_trust_energy_budget",
            ),
        )
        for name in (
            "source_receipt_hash",
            "eligibility_receipt_hash",
            "fd_residual_certificate_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.fd_residual_certificate_schema != (
            PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_SCHEMA
        ):
            raise ValueError("unsupported FD residual certificate schema")
        certificate_kind = _nonempty(
            self.fd_residual_certificate_kind,
            "fd_residual_certificate_kind",
        )
        if certificate_kind not in {
            PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND,
            PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND,
            PRODUCTION_SELECTOR_SYNTHETIC_FD_CERTIFICATE_KIND,
        }:
            raise ValueError("unsupported FD residual certificate kind")
        object.__setattr__(self, "fd_residual_certificate_kind", certificate_kind)
        if self.fd_ordered_route_axis_hash is None:
            if certificate_kind in {
                PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND,
                PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND,
            }:
                raise ValueError(
                    "typed FD residual certificates require an ordered route-ID axis hash"
                )
            fd_ordered_axis_hash = axis_hash
        else:
            fd_ordered_axis_hash = _nonempty(
                self.fd_ordered_route_axis_hash,
                "fd_ordered_route_axis_hash",
            )
        object.__setattr__(
            self,
            "fd_ordered_route_axis_hash",
            fd_ordered_axis_hash,
        )
        geometry_state_hash = (
            None
            if self.fd_residual_geometry_state_hash is None
            else _nonempty(
                self.fd_residual_geometry_state_hash,
                "fd_residual_geometry_state_hash",
            )
        )
        object.__setattr__(
            self,
            "fd_residual_geometry_state_hash",
            geometry_state_hash,
        )
        receipt = self.fd_residual_derivation_receipt
        sampled_certificate = self.fd_residual_sampled_snapshot_certificate
        position_certificate = self.sampled_proposal_position_certificate
        if certificate_kind == PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND:
            if sampled_certificate is not None or position_certificate is not None:
                raise ValueError(
                    "raw-row FD certificates cannot carry sampled snapshot provenance"
                )
            if receipt is None:
                raise ValueError(
                    "production FD residual certificates require a derivation receipt"
                )
            if type(receipt) is not GlobalTrustGeometryDerivationReceipt:
                raise TypeError(
                    "fd_residual_derivation_receipt must be a "
                    "GlobalTrustGeometryDerivationReceipt"
                )
            if getattr(receipt, "schema", None) != GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA:
                raise ValueError("unsupported global trust derivation receipt schema")
            receipt_hash = _nonempty(
                getattr(receipt, "receipt_hash", None),
                "fd residual derivation receipt hash",
            )
            if self.fd_residual_certificate_hash != receipt_hash:
                raise ValueError("FD residual certificate hash does not match receipt")
            if geometry_state_hash is None:
                raise ValueError(
                    "production FD residual certificates require a geometry state hash"
                )
            if (
                getattr(receipt, "ordered_route_axis_hash", None)
                != fd_ordered_axis_hash
            ):
                raise ValueError("FD residual certificate route axis differs")
            if getattr(receipt, "measured_mask_hash", None) != self.parent_mask.mask_hash:
                raise ValueError("FD residual certificate measured mask differs")
            receipt_total_weight = float(getattr(receipt, "total_weight"))
            receipt_shrink = float(getattr(receipt, "cumulative_fd_shrink"))
            if not _roundoff_close(spectral_bound, receipt_shrink / receipt_total_weight):
                raise ValueError(
                    "residual_spectral_bound does not match the FD receipt shrink/weight"
                )
            centered_rank = _positive_int(
                getattr(receipt, "sketch_rank"),
                "FD residual certificate sketch_rank",
            )
            mean_rank = int(factor.shape[0]) - centered_rank
            if mean_rank <= 0:
                raise ValueError(
                    "selector low_rank_factor must contain mean rows before centered FD rows"
                )
            mean_hash = content_hash(
                factor[:mean_rank],
                prefix=GLOBAL_TRUST_DERIVED_MEAN_PREFIX,
            )
            centered_hash = content_hash(
                factor[mean_rank:],
                prefix=GLOBAL_TRUST_DERIVED_CENTERED_FACTOR_PREFIX,
            )
            residual_hash = content_hash(
                residual,
                prefix=GLOBAL_TRUST_DERIVED_RESIDUAL_DIAGONAL_PREFIX,
            )
            if mean_hash != getattr(receipt, "mean_component_hash", None):
                raise ValueError("selector mean rows do not match FD receipt")
            if centered_hash != getattr(receipt, "centered_factor_hash", None):
                raise ValueError("selector centered FD rows do not match FD receipt")
            if residual_hash != getattr(receipt, "exact_residual_diagonal_hash", None):
                raise ValueError("selector residual diagonal does not match FD receipt")
        elif (
            certificate_kind
            == PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND
        ):
            if receipt is not None:
                raise ValueError(
                    "sampled snapshot FD certificates cannot carry raw-row receipts"
                )
            if type(sampled_certificate) is not V1SampledSnapshotSelectorCertificate:
                raise TypeError(
                    "sampled FD provenance must be a "
                    "V1SampledSnapshotSelectorCertificate"
                )
            if type(position_certificate) is not V1SampledProposalPositionCertificate:
                raise TypeError(
                    "sampled selector problems require a neutral "
                    "V1SampledProposalPositionCertificate"
                )
            if sampled_certificate.schema != SAMPLED_SNAPSHOT_SELECTOR_CERTIFICATE_SCHEMA:
                raise ValueError("unsupported sampled snapshot selector certificate schema")
            if self.fd_residual_certificate_hash != sampled_certificate.certificate_hash:
                raise ValueError("sampled FD certificate hash differs")
            if geometry_state_hash != sampled_certificate.geometry_state_hash:
                raise ValueError("sampled FD geometry state differs")
            if fd_ordered_axis_hash != sampled_certificate.ordered_route_axis_hash:
                raise ValueError("sampled FD ordered route axis differs")
            if axis_hash != sampled_certificate.physical_route_axis_hash:
                raise ValueError("sampled FD physical route axis differs")
            if self.parent_mask.mask_hash != sampled_certificate.current_mask_hash:
                raise ValueError("sampled FD measured mask differs")
            if route_count != sampled_certificate.route_count:
                raise ValueError("sampled FD route count differs")
            if (
                position_certificate.sampled_snapshot_certificate_hash
                != sampled_certificate.certificate_hash
                or position_certificate.sampled_geometry_source_hash
                != sampled_certificate.sampled_geometry_source_hash
                or position_certificate.current_mask_hash
                != self.parent_mask.mask_hash
                or position_certificate.physical_route_axis_hash != axis_hash
                or position_certificate.ordered_route_axis_hash
                != fd_ordered_axis_hash
                or position_certificate.route_count != route_count
            ):
                raise ValueError("sampled proposal-position coordinates differ")
            sampled_certificate.assert_selector_payload(
                corrected_path_cost=path,
                signed_covector=signed,
                low_rank_factor=factor,
                residual_diagonal=residual,
                residual_spectral_bound=spectral_bound,
            )
            if self.source_receipt_hash != (
                sampled_certificate.corrected_point_receipt_hash
            ):
                raise ValueError("sampled corrected-point source receipt differs")
        else:
            if (
                receipt is not None
                or sampled_certificate is not None
                or position_certificate is not None
            ):
                raise ValueError(
                    "synthetic FD residual certificates cannot carry typed provenance"
                )
        provenance = self.chart_provenance
        if certificate_kind == PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND:
            if type(provenance) is not V1ProductionSelectorChartProvenance:
                raise PermissionError(
                    "production FD selector charts require typed chart provenance"
                )
            fields = provenance.corrected_attribution_fields
            coordinates = provenance.corrected_attribution_coordinates
            budgets = provenance.geometric_trust_budgets
            calibration = provenance.fixed_affine_bce_calibration
            if coordinates.current_mask_hash != self.parent_mask.mask_hash:
                raise ValueError("corrected-attribution parent mask differs")
            if coordinates.physical_route_axis_hash != axis_hash:
                raise ValueError("corrected-attribution physical route axis differs")
            if coordinates.ordered_route_axis_hash != fd_ordered_axis_hash:
                raise ValueError("corrected-attribution ordered route axis differs")
            if fields.route_axis_hash != fd_ordered_axis_hash:
                raise ValueError("corrected-attribution ordered route axis differs")
            if len(fields.route_ids) != route_count:
                raise ValueError("corrected-attribution route count differs")
            if fields.source_attribution_hash != self.source_receipt_hash:
                raise ValueError("corrected-attribution source hash differs")
            if getattr(receipt, "calibration_state_id", None) != (
                calibration.calibration_state_id
            ):
                raise ValueError("FD chart calibration state differs")
            if tuple(getattr(receipt, "source_measurement_hashes", ())) != (
                coordinates.source_measurement_hash,
            ):
                raise ValueError("FD and corrected attribution measurement owners differ")
            if fields.sample_weight_hash != getattr(receipt, "stream_weight_hash", None):
                raise ValueError("FD and corrected attribution stream weights differ")
            if int(getattr(receipt, "effective_context_count", -1)) != (
                fields.sample_count
            ):
                raise ValueError("FD and corrected attribution sample counts differ")
            if not np.array_equal(path, fields.v0_corrected_salience):
                raise ValueError("corrected path cost differs from typed attribution A")
            if signed is not None and not np.array_equal(
                signed,
                fields.signed_corrected_damage,
            ):
                raise ValueError("signed covector differs from typed corrected attribution")
            if budgets.current_mask_hash != self.parent_mask.mask_hash:
                raise ValueError("trust-budget current mask differs")
            if not same_geometric_coordinate(
                self.finsler_path_budget,
                budgets.finsler_path_budget,
            ):
                raise ValueError("selector tau_F differs from typed trust budget")
            if not same_geometric_coordinate(
                self.quadratic_trust_energy_budget,
                budgets.quadratic_trust_energy_budget,
            ):
                raise ValueError("selector tau_Q differs from typed trust budget")
            if not same_geometric_coordinate(
                budgets.quadratic_trust_energy_budget,
                budgets.metric_length_budget * budgets.metric_length_budget,
            ):
                raise ValueError("typed trust-budget tau_Q square law differs")
            if provenance.eligibility_receipt_hash != self.eligibility_receipt_hash:
                raise ValueError("typed chart eligibility identity differs")
            resource_plane = _nonempty(self.resource_plane, "resource_plane")
            resource_cost_policy_hash = _nonempty(
                self.resource_cost_policy_hash,
                "resource_cost_policy_hash",
            )
            legality_policy_hash = _nonempty(
                self.legality_policy_hash,
                "legality_policy_hash",
            )
            if provenance.resource_plane != resource_plane:
                raise ValueError("typed chart resource plane differs")
            if provenance.resource_cost_policy_hash != resource_cost_policy_hash:
                raise ValueError("typed chart resource-cost policy differs")
            if provenance.legality_policy_hash != legality_policy_hash:
                raise ValueError("typed chart legality policy differs")
            resource_cost_hash = content_hash(
                {
                    "route_count": route_count,
                    "uniform_resource_cost": resource_scalar,
                    "resource_cost": None if resource_scalar is not None else resource,
                },
                prefix="partition-free-v1-production-selector-resource-cost-",
            )
            if provenance.resource_cost_hash != resource_cost_hash:
                raise ValueError("typed chart resource-cost payload differs")
        elif (
            certificate_kind
            == PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND
        ):
            if type(provenance) is not V1SampledSelectorChartProvenance:
                raise PermissionError(
                    "sampled production selector charts require typed sampled "
                    "chart provenance"
                )
            if sampled_certificate is None:  # pragma: no cover - narrowed above.
                raise RuntimeError("sampled selector certificate disappeared")
            if position_certificate is None:  # pragma: no cover - narrowed above.
                raise RuntimeError("sampled proposal-position certificate disappeared")
            coordinates = provenance.corrected_attribution_coordinates
            budgets = provenance.geometric_trust_budgets
            calibration = provenance.fixed_affine_bce_calibration
            if (
                provenance.snapshot_certificate.certificate_hash
                != sampled_certificate.certificate_hash
                or provenance.proposal_position_certificate.certificate_hash
                != position_certificate.certificate_hash
                or coordinates.sampled_snapshot_certificate_hash
                != sampled_certificate.certificate_hash
                or coordinates.current_mask_hash != self.parent_mask.mask_hash
                or coordinates.physical_route_axis_hash != axis_hash
                or coordinates.ordered_route_axis_hash != fd_ordered_axis_hash
                or coordinates.proposal_generation
                != sampled_certificate.proposal_generation
                or coordinates.calibration_identity
                != sampled_certificate.calibration_identity
                or provenance.score_discovery_coordinates.statistics_hash
                != sampled_certificate.corrected_moments_state_hash
                or provenance.score_discovery_coordinates.look_index
                != sampled_certificate.sampled_look_index
                or calibration.calibration_state_id
                != sampled_certificate.calibration_identity
            ):
                raise ValueError("sampled selector chart source coordinates differ")
            if budgets.current_mask_hash != self.parent_mask.mask_hash:
                raise ValueError("sampled trust-budget current mask differs")
            if not same_geometric_coordinate(
                self.finsler_path_budget,
                budgets.finsler_path_budget,
            ):
                raise ValueError("sampled selector tau_F differs from typed trust budget")
            if not same_geometric_coordinate(
                self.quadratic_trust_energy_budget,
                budgets.quadratic_trust_energy_budget,
            ):
                raise ValueError("sampled selector tau_Q differs from typed trust budget")
            if not same_geometric_coordinate(
                budgets.quadratic_trust_energy_budget,
                budgets.metric_length_budget * budgets.metric_length_budget,
            ):
                raise ValueError("sampled typed trust-budget tau_Q square law differs")
            if provenance.eligibility_receipt_hash != self.eligibility_receipt_hash:
                raise ValueError("sampled typed chart eligibility identity differs")
            resource_plane = _nonempty(self.resource_plane, "resource_plane")
            resource_cost_policy_hash = _nonempty(
                self.resource_cost_policy_hash,
                "resource_cost_policy_hash",
            )
            legality_policy_hash = _nonempty(
                self.legality_policy_hash,
                "legality_policy_hash",
            )
            if provenance.resource_plane != resource_plane:
                raise ValueError("sampled typed chart resource plane differs")
            if provenance.resource_cost_policy_hash != resource_cost_policy_hash:
                raise ValueError("sampled typed chart resource-cost policy differs")
            if provenance.legality_policy_hash != legality_policy_hash:
                raise ValueError("sampled typed chart legality policy differs")
            resource_cost_hash = content_hash(
                {
                    "route_count": route_count,
                    "uniform_resource_cost": resource_scalar,
                    "resource_cost": None if resource_scalar is not None else resource,
                },
                prefix="partition-free-v1-production-selector-resource-cost-",
            )
            if provenance.resource_cost_hash != resource_cost_hash:
                raise ValueError("sampled typed chart resource-cost payload differs")
        elif provenance is not None:
            raise ValueError(
                "synthetic selector charts cannot carry production chart provenance"
            )
        if self.schema != PRODUCTION_SELECTOR_PROBLEM_SCHEMA:
            raise ValueError("unsupported production selector problem schema")
        metadata: Mapping[str, Any]
        if isinstance(self.metadata, Mapping):
            metadata_payload: dict[str, Any] = dict(self.metadata)
        else:
            raise TypeError("metadata must be a mapping")
        if spectral_bound_snapped:
            metadata_payload.update(
                {
                    "residual_spectral_bound_original": original_spectral_bound,
                    "residual_spectral_bound_effective": spectral_bound,
                    "residual_spectral_bound_snapped_to_max_diagonal": True,
                    "residual_spectral_bound_snap_reason": (
                        "within_tolerance_below_max_residual_diagonal"
                    ),
                    "max_residual_diagonal": max_residual_diagonal,
                    "residual_spectral_bound_tolerance": tolerance,
                }
            )
        metadata = metadata_payload
        object.__setattr__(self, "metadata", _freeze(metadata, "metadata"))

    @property
    def route_count(self) -> int:
        return self.parent_mask.route_count

    @property
    def rank(self) -> int:
        return int(self.low_rank_factor.shape[0])

    @property
    def live_mask(self) -> np.ndarray:
        return self.parent_mask.to_bool_array()

    @property
    def selectable_mask(self) -> np.ndarray:
        return np.logical_and(self.live_mask, self.eligible_mask)

    @property
    def live_count(self) -> int:
        return self.parent_mask.retained_count

    @property
    def eligible_count(self) -> int:
        return int(np.count_nonzero(self.eligible_mask))

    @property
    def selectable_count(self) -> int:
        return int(np.count_nonzero(self.selectable_mask))

    @property
    def block_count(self) -> int:
        if self.legality_blocks is not None:
            return len(self.legality_blocks)
        if self.legality_block_ids is None:
            return 0
        return len(set(self.legality_block_ids))

    @property
    def has_uniform_resource_cost(self) -> bool:
        return self.uniform_resource_cost is not None

    def resource_values(self, indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.uniform_resource_cost is not None:
            return np.full(selected.shape, self.uniform_resource_cost, dtype=np.float64)
        return self.resource_cost[selected]

    def resource_sum(self, indices: np.ndarray) -> float:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.uniform_resource_cost is not None:
            return float(self.uniform_resource_cost * int(selected.size))
        return float(np.sum(self.resource_cost[selected], dtype=np.float64))

    @property
    def resource_scale(self) -> float:
        if self.uniform_resource_cost is not None:
            return self.uniform_resource_cost
        selectable = self.selectable_mask
        positive_resource = self.resource_cost[selectable & (self.resource_cost > 0.0)]
        if positive_resource.size == 0:
            return 1.0
        scale = float(np.median(positive_resource, overwrite_input=False))
        return scale if math.isfinite(scale) and scale > 0.0 else 1.0

    @property
    def has_signed_covector(self) -> bool:
        return self.signed_covector is not None

    @property
    def has_production_chart_provenance(self) -> bool:
        return self.chart_provenance is not None

    @property
    def problem_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_id": self.problem_id,
                "parent_mask_hash": self.parent_mask.mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "fd_ordered_route_axis_hash": self.fd_ordered_route_axis_hash,
                "source_receipt_hash": self.source_receipt_hash,
                "eligibility_receipt_hash": self.eligibility_receipt_hash,
                "fd_residual_certificate_hash": self.fd_residual_certificate_hash,
                "fd_residual_certificate_schema": self.fd_residual_certificate_schema,
                "fd_residual_certificate_kind": self.fd_residual_certificate_kind,
                "fd_residual_geometry_state_hash": self.fd_residual_geometry_state_hash,
                "sampled_snapshot_certificate_hash": (
                    None
                    if self.fd_residual_sampled_snapshot_certificate is None
                    else self.fd_residual_sampled_snapshot_certificate.certificate_hash
                ),
                "sampled_proposal_position_certificate_hash": (
                    None
                    if self.sampled_proposal_position_certificate is None
                    else self.sampled_proposal_position_certificate.certificate_hash
                ),
                "chart_provenance_hash": (
                    None
                    if self.chart_provenance is None
                    else self.chart_provenance.provenance_hash
                ),
                "resource_plane": self.resource_plane,
                "resource_cost_policy_hash": self.resource_cost_policy_hash,
                "legality_policy_hash": self.legality_policy_hash,
                "eligible_mask": self.eligible_mask,
                "corrected_path_cost": self.corrected_path_cost,
                "signed_covector": (
                    None if self.signed_covector is None else self.signed_covector
                ),
                "low_rank_factor": self.low_rank_factor,
                "residual_diagonal": self.residual_diagonal,
                "residual_spectral_bound": self.residual_spectral_bound,
                "resource_cost": self.resource_cost,
                "uniform_resource_cost": self.uniform_resource_cost,
                "finsler_path_budget": self.finsler_path_budget,
                "quadratic_trust_energy_budget": self.quadratic_trust_energy_budget,
                "legality_block_ids": self.legality_block_ids,
                "legality_blocks": self.legality_blocks,
                "minimum_live_per_block": self.minimum_live_per_block,
                "input_array_policy": self.input_array_policy,
                "metadata": self.metadata,
            },
            prefix="partition-free-v1-production-selector-problem-",
        )

    @property
    def admitted_array_payload_bytes(self) -> int:
        block_bytes = 0
        if self.legality_blocks is not None:
            block_bytes = sum(
                len(block.block_id.encode("utf-8")) + 16 for block in self.legality_blocks
            )
        elif self.legality_block_ids is not None:
            block_bytes = sum(len(block.encode("utf-8")) + 8 for block in self.legality_block_ids)
        resource_bytes = 8 if self.uniform_resource_cost is not None else self.resource_cost.nbytes
        signed_bytes = 0 if self.signed_covector is None else self.signed_covector.nbytes
        return int(
            self.eligible_mask.nbytes
            + self.corrected_path_cost.nbytes
            + signed_bytes
            + self.low_rank_factor.nbytes
            + self.residual_diagonal.nbytes
            + resource_bytes
            + len(self.parent_mask.packed_bits)
            + block_bytes
        )


@dataclass(frozen=True, slots=True)
class V1ProductionGeometricCandidate:
    """One packed proposed removal action with certified set-level trust metrics."""

    action: V1PackedRemovalAction
    route_count: int
    scalarization_id: str
    removal_count: int
    resource_gain: float
    corrected_path_cost: float
    signed_covector_sum: float
    residual_diagonal_sum: float
    diagonal_surrogate_q: float
    residual_diagonal_sqrt_sum: float
    certified_residual_quadratic: float
    low_rank_quadratic: float
    quadratic_trust_value: float
    proposal_burden: float
    selection_objective: float
    construction: CandidateConstruction
    scalarized_lower_bound: float | None = None
    scalarized_gap: float | None = None
    fixed_count_scalarized_lower_bound_supported: bool = False
    threshold_tie_refused: bool = False
    exact_candidate_metrics: bool = True
    complete_set_quadratic: bool = True
    retained_index_tuple: bool = False
    proposal_only: bool = True
    mask_commit_authority: bool = False
    schema: str = PRODUCTION_SELECTOR_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.action) is not V1PackedRemovalAction:
            raise TypeError("production candidates require packed removal actions")
        route_count = _positive_int(self.route_count, "route_count")
        removal_count = _positive_int(self.removal_count, "removal_count")
        if self.action.resulting_mask.route_count != route_count:
            raise ValueError("candidate packed mask is on another route count")
        if self.action.removed_count != removal_count:
            raise ValueError("candidate removal count differs from packed action")
        object.__setattr__(
            self,
            "scalarization_id",
            _nonempty(self.scalarization_id, "scalarization_id"),
        )
        for name in (
            "resource_gain",
            "corrected_path_cost",
            "residual_diagonal_sum",
            "diagonal_surrogate_q",
            "residual_diagonal_sqrt_sum",
            "certified_residual_quadratic",
            "low_rank_quadratic",
            "quadratic_trust_value",
            "proposal_burden",
            "selection_objective",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        signed = float(self.signed_covector_sum)
        if not math.isfinite(signed):
            raise ValueError("signed_covector_sum must be finite")
        object.__setattr__(self, "signed_covector_sum", signed)
        if not _roundoff_close(
            self.residual_diagonal_sum,
            self.diagonal_surrogate_q,
        ):
            raise ValueError("candidate residual_diagonal_sum must equal diagonal_surrogate_q")
        closed_quadratic = self.certified_residual_quadratic + self.low_rank_quadratic
        if not _roundoff_close(self.quadratic_trust_value, closed_quadratic):
            raise ValueError("candidate certified Q does not close")
        if self.construction not in {
            "fixed_count_low_rank_lagrangian_dual",
            "single_order_low_rank_lagrangian_dual_prefix",
            "registered_prefix_conditional_budget_gauge",
            "single_order_corrected_salience_primary_prefix",
            "exact_reference_oracle_repacked",
        }:
            raise ValueError("unsupported production candidate construction")
        if self.scalarized_lower_bound is None:
            if self.scalarized_gap is not None:
                raise ValueError("scalarized_gap requires scalarized_lower_bound")
        else:
            lower = float(self.scalarized_lower_bound)
            if not math.isfinite(lower):
                raise ValueError("scalarized_lower_bound must be finite")
            gap = _finite_nonnegative(self.scalarized_gap, "scalarized_gap")
            if lower > self.selection_objective and not _roundoff_close(
                lower,
                self.selection_objective,
            ):
                raise ValueError("scalarized lower bound exceeds selection objective")
            if not _roundoff_close(
                gap,
                self.selection_objective - lower,
            ):
                raise ValueError("scalarized gap does not close")
            object.__setattr__(self, "scalarized_lower_bound", lower)
            object.__setattr__(self, "scalarized_gap", gap)
        for name in (
            "fixed_count_scalarized_lower_bound_supported",
            "threshold_tie_refused",
            "exact_candidate_metrics",
            "complete_set_quadratic",
            "retained_index_tuple",
            "proposal_only",
            "mask_commit_authority",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if (
            not self.exact_candidate_metrics
            or not self.complete_set_quadratic
            or self.retained_index_tuple
            or not self.proposal_only
            or self.mask_commit_authority
        ):
            raise ValueError("production candidate violated its proposal contract")
        if self.schema != PRODUCTION_SELECTOR_CANDIDATE_SCHEMA:
            raise ValueError("unsupported production selector candidate schema")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "removal_count", removal_count)

    @property
    def removed_route_index_hash(self) -> str:
        return self.action.removed_route_index_hash

    @property
    def candidate_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-selector-candidate-",
        )


@dataclass(frozen=True, slots=True)
class V1ProductionGeometricSelectorReceipt:
    """Execution and authority receipt for the packed global frontier."""

    problem_hash: str
    policy_hash: str
    route_count: int
    live_route_count: int
    eligible_route_count: int
    selectable_route_count: int
    block_count: int
    scalarization_count: int
    candidate_count: int
    backend: SelectorBackend
    device: str
    chunk_size: int
    route_score_pass_count: int
    fixed_count_oracle_route_visit_count: int
    packed_action_count: int
    packed_action_payload_bytes: int
    retained_index_tuple_count: int = 0
    algorithm_id: str = PRODUCTION_SELECTOR_ID
    scalarization_equation_id: str = PRODUCTION_SELECTOR_SCALARIZATION_ID
    complete_q_equation_id: str = PRODUCTION_SELECTOR_COMPLETE_Q_ID
    partition_policy_id: str = PRODUCTION_SELECTOR_NO_PARTITION_POLICY_ID
    frozen_composer_id: str = FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID
    exact_candidate_metrics: bool = True
    frontier_exact: bool = False
    global_optimality_claim: bool = False
    lower_bounds_supported: bool = False
    signed_covector_used_for_selection: bool = False
    production_chart_authority: bool = False
    selector_policy_diagnostic_only: bool = True
    no_layer_partitions_or_quotas: bool = True
    layer_blocks_used_only_for_survivor_legality: bool = True
    no_sign_partitions_or_quotas: bool = True
    loss_rescaling_law: str = LOSS_RESCALING_LAW
    refusal_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = PRODUCTION_SELECTOR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in ("problem_hash", "policy_hash", "device"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "live_route_count",
            "eligible_route_count",
            "selectable_route_count",
            "block_count",
            "scalarization_count",
            "candidate_count",
            "chunk_size",
            "route_score_pass_count",
            "fixed_count_oracle_route_visit_count",
            "packed_action_count",
            "packed_action_payload_bytes",
            "retained_index_tuple_count",
        ):
            value = (
                _positive_int(getattr(self, name), name)
                if name in {"route_count", "scalarization_count", "chunk_size"}
                else _nonnegative_int(getattr(self, name), name)
            )
            object.__setattr__(self, name, value)
        if self.backend not in {"numpy", "torch"}:
            raise ValueError("unsupported production selector backend")
        for name in (
            "exact_candidate_metrics",
            "frontier_exact",
            "global_optimality_claim",
            "lower_bounds_supported",
            "signed_covector_used_for_selection",
            "production_chart_authority",
            "selector_policy_diagnostic_only",
            "no_layer_partitions_or_quotas",
            "layer_blocks_used_only_for_survivor_legality",
            "no_sign_partitions_or_quotas",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool")
        if (
            not self.exact_candidate_metrics
            or self.frontier_exact
            or self.global_optimality_claim
            or self.lower_bounds_supported
            or self.signed_covector_used_for_selection
            or not self.no_layer_partitions_or_quotas
            or not self.layer_blocks_used_only_for_survivor_legality
            or not self.no_sign_partitions_or_quotas
        ):
            raise ValueError("production selector receipt overclaims authority")
        if self.production_chart_authority and self.selector_policy_diagnostic_only:
            raise ValueError(
                "diagnostic selector policy cannot carry production chart authority"
            )
        if self.retained_index_tuple_count != 0:
            raise ValueError("production selector may not retain index tuples")
        if self.loss_rescaling_law != LOSS_RESCALING_LAW:
            raise ValueError("unsupported production selector loss-rescaling law")
        for name in (
            "algorithm_id",
            "scalarization_equation_id",
            "complete_q_equation_id",
            "partition_policy_id",
            "frozen_composer_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.schema != PRODUCTION_SELECTOR_RECEIPT_SCHEMA:
            raise ValueError("unsupported production selector receipt schema")
        object.__setattr__(
            self,
            "refusal_reasons",
            tuple(_nonempty(value, "refusal reason") for value in self.refusal_reasons),
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-production-selector-receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1ProductionGeometricSelectorResult:
    """Packed frontier plus receipt for one production selection pass."""

    problem_hash: str
    policy_hash: str
    receipt: V1ProductionGeometricSelectorReceipt
    frontier: tuple[V1ProductionGeometricCandidate, ...]
    schema: str = PRODUCTION_SELECTOR_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_hash", _nonempty(self.problem_hash, "problem_hash"))
        object.__setattr__(self, "policy_hash", _nonempty(self.policy_hash, "policy_hash"))
        if type(self.receipt) is not V1ProductionGeometricSelectorReceipt:
            raise TypeError("production selector result requires a selector receipt")
        if (
            self.problem_hash != self.receipt.problem_hash
            or self.policy_hash != self.receipt.policy_hash
        ):
            raise ValueError("production selector result and receipt identities differ")
        frontier = tuple(self.frontier)
        if len(frontier) != self.receipt.candidate_count:
            raise ValueError("production selector receipt candidate count differs")
        seen: set[str] = set()
        previous_key: tuple[float, float, str] | None = None
        for candidate in frontier:
            if type(candidate) is not V1ProductionGeometricCandidate:
                raise TypeError("frontier entries must be production selector candidates")
            if candidate.removed_route_index_hash in seen:
                raise ValueError("production selector frontier duplicated an action")
            seen.add(candidate.removed_route_index_hash)
            key = (
                -candidate.resource_gain,
                candidate.proposal_burden,
                candidate.removed_route_index_hash,
            )
            if previous_key is not None and key < previous_key:
                raise ValueError("production selector frontier must be resource-descending")
            previous_key = key
        if self.schema != PRODUCTION_SELECTOR_RESULT_SCHEMA:
            raise ValueError("unsupported production selector result schema")
        object.__setattr__(self, "frontier", frontier)

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "receipt": self.receipt,
                "candidate_hashes": tuple(candidate.candidate_hash for candidate in self.frontier),
            },
            prefix="partition-free-v1-production-selector-result-",
        )

    def to_finite_removal_domain(
        self,
        *,
        route_axis_hash: str,
        route_count: int,
        declaration_receipt_hash: str | None = None,
    ) -> V1FiniteRemovalDomain:
        """Return the frozen packed domain expected by the existing composer."""

        if route_count != self.receipt.route_count:
            raise ValueError("finite domain route count differs from selector receipt")
        return V1FiniteRemovalDomain(
            route_axis_hash=route_axis_hash,
            route_count=route_count,
            removal_sets=tuple(candidate.action for candidate in self.frontier),
            declaration_receipt_hash=(
                self.receipt.receipt_hash
                if declaration_receipt_hash is None
                else _nonempty(declaration_receipt_hash, "declaration_receipt_hash")
            ),
        )

def _proposal_burden(
    problem: V1ProductionGeometricSelectorProblem,
    *,
    path: float,
    quadratic: float,
) -> float:
    burden = max(
        _dimensionless_scalar(path, problem.finsler_path_budget),
        math.sqrt(
            _dimensionless_scalar(
                quadratic,
                problem.quadratic_trust_energy_budget,
            )
        ),
    )
    if not math.isfinite(burden):
        raise ValueError("proposal burden is undefined for an infeasible zero-budget action")
    return burden


@dataclass(frozen=True, slots=True)
class _LinearOracleResult:
    selected: np.ndarray
    linear_sum: float
    tie_refused: bool = False
    infeasible: bool = False
    refusal_reason: str | None = None


def _finite_selectable_indices(
    problem: V1ProductionGeometricSelectorProblem,
    values: np.ndarray,
) -> np.ndarray:
    return np.flatnonzero(np.logical_and(problem.selectable_mask, np.isfinite(values))).astype(
        np.int64,
        copy=False,
    )


def _block_removal_capacities(
    problem: V1ProductionGeometricSelectorProblem,
) -> dict[str, int]:
    if problem.legality_blocks is not None:
        live = problem.live_mask
        return {
            block.block_id: max(
                0,
                int(np.count_nonzero(live[block.start : block.stop]))
                - problem.minimum_live_per_block,
            )
            for block in problem.legality_blocks
        }
    blocks = problem.legality_block_ids
    if blocks is None:
        return {}
    live_by_block: dict[str, int] = {}
    for block, is_live in zip(blocks, problem.live_mask, strict=True):
        if bool(is_live):
            live_by_block[block] = live_by_block.get(block, 0) + 1
    return {
        block: max(0, live_count - problem.minimum_live_per_block)
        for block, live_count in live_by_block.items()
    }


def _block_id_for_physical_index(
    problem: V1ProductionGeometricSelectorProblem,
    physical_index: int,
) -> str:
    if problem.legality_blocks is not None:
        # The block list is compact and contiguous; linear scan is fine for the
        # small number of layer blocks, and avoids a 57M per-route string axis.
        for block in problem.legality_blocks:
            if block.start <= physical_index < block.stop:
                return block.block_id
        raise ValueError("physical index is outside compact legality blocks")
    if problem.legality_block_ids is None:
        raise ValueError("no legality block axis is registered")
    return problem.legality_block_ids[physical_index]


def _threshold_tie(
    *,
    values: np.ndarray,
    threshold: float,
    tolerance: float,
) -> np.ndarray:
    scale = max(1.0, abs(float(threshold)))
    return np.flatnonzero(np.abs(values - threshold) <= tolerance * scale)


def _finite_selectable_mask(
    problem: V1ProductionGeometricSelectorProblem,
    adjusted: np.ndarray,
    secondary: np.ndarray | None = None,
) -> np.ndarray:
    finite = np.logical_and(problem.selectable_mask, np.isfinite(adjusted))
    if secondary is not None:
        finite = np.logical_and(finite, np.isfinite(secondary))
    return finite


def _argmax_lexicographic_route(
    *,
    physical: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    tertiary: np.ndarray | None = None,
) -> int:
    """Return the worst route under the deterministic selector tie hierarchy."""

    if physical.size <= 0:
        raise ValueError("cannot choose a protected survivor from an empty block")
    tertiary_values = np.zeros_like(primary) if tertiary is None else tertiary
    best_rel = 0
    best_key = (
        float(primary[0]),
        float(secondary[0]),
        float(tertiary_values[0]),
        int(physical[0]),
    )
    for rel in range(1, int(physical.size)):
        key = (
            float(primary[rel]),
            float(secondary[rel]),
            float(tertiary_values[rel]),
            int(physical[rel]),
        )
        if key > best_key:
            best_rel = rel
            best_key = key
    return int(physical[best_rel])


def _compact_block_protected_survivors(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    adjusted: np.ndarray,
    secondary: np.ndarray | None = None,
    finite_selectable: np.ndarray,
) -> np.ndarray:
    if problem.legality_blocks is None:
        return np.asarray((), dtype=np.int64)
    live = problem.live_mask
    protected: list[int] = []
    for block in problem.legality_blocks:
        live_slice = live[block.start : block.stop]
        finite_slice = finite_selectable[block.start : block.stop]
        live_count = int(np.count_nonzero(live_slice))
        if live_count <= 0:
            continue
        finite_count = int(np.count_nonzero(finite_slice))
        if finite_count < live_count:
            # A live ineligible/nonfinite route is already an unremovable survivor.
            continue
        if finite_count <= 0:
            continue
        finite_positions = np.flatnonzero(finite_slice)
        finite_indices = block.start + finite_positions
        finite_values = adjusted[finite_indices]
        finite_secondary = (
            np.zeros_like(finite_values) if secondary is None else secondary[finite_indices]
        )
        protected.append(
            _argmax_lexicographic_route(
                physical=finite_indices,
                primary=finite_values,
                secondary=finite_secondary,
            )
        )
    result = np.asarray(protected, dtype=np.int64)
    result.setflags(write=False)
    return result


def _legacy_block_protected_survivors(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    adjusted: np.ndarray,
    secondary: np.ndarray | None = None,
    finite_selectable: np.ndarray,
) -> np.ndarray:
    if problem.legality_block_ids is None:
        return np.asarray((), dtype=np.int64)
    live = problem.live_mask
    by_block: dict[str, list[int]] = {}
    finite_by_block: dict[str, list[int]] = {}
    for physical, block in enumerate(problem.legality_block_ids):
        if bool(live[physical]):
            by_block.setdefault(block, []).append(physical)
            if bool(finite_selectable[physical]):
                finite_by_block.setdefault(block, []).append(physical)
    protected: list[int] = []
    for block, live_indices in by_block.items():
        finite_indices = finite_by_block.get(block, ())
        if len(finite_indices) < len(live_indices):
            continue
        if not finite_indices:
            continue
        physical = np.asarray(finite_indices, dtype=np.int64)
        values = adjusted[physical]
        secondary_values = np.zeros_like(values) if secondary is None else secondary[physical]
        protected.append(
            _argmax_lexicographic_route(
                physical=physical,
                primary=values,
                secondary=secondary_values,
            )
        )
    result = np.asarray(protected, dtype=np.int64)
    result.setflags(write=False)
    return result


_CandidateMetricTuple = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


def _metrics_for_indices(
    problem: V1ProductionGeometricSelectorProblem,
    indices: np.ndarray,
) -> _CandidateMetricTuple:
    selected = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
    q = np.sum(problem.low_rank_factor[:, selected], axis=1, dtype=np.float64)
    diagonal_surrogate = float(
        np.sum(problem.residual_diagonal[selected], dtype=np.float64)
    )
    residual_diagonal_sqrt_sum = float(
        np.sum(
            np.sqrt(problem.residual_diagonal[selected]),
            dtype=np.float64,
        )
    )
    certified_residual = min(
        problem.residual_spectral_bound * float(selected.size),
        residual_diagonal_sqrt_sum * residual_diagonal_sqrt_sum,
    )
    low_rank = float(np.dot(q, q))
    return (
        problem.resource_sum(selected),
        float(np.sum(problem.corrected_path_cost[selected], dtype=np.float64)),
        0.0
        if problem.signed_covector is None
        else float(np.sum(problem.signed_covector[selected], dtype=np.float64)),
        diagonal_surrogate,
        residual_diagonal_sqrt_sum,
        certified_residual,
        low_rank,
        certified_residual + low_rank,
    )


def _candidate_from_indices(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    scalarization_id: str,
    construction: CandidateConstruction,
    indices: np.ndarray,
    metrics: _CandidateMetricTuple,
    selection_objective: float,
    scalarized_lower_bound: float | None,
) -> V1ProductionGeometricCandidate:
    action = V1PackedRemovalAction.from_removed_indices(
        parent_mask=problem.parent_mask,
        removed_indices=indices,
    )
    return _candidate_from_action(
        problem=problem,
        scalarization_id=scalarization_id,
        construction=construction,
        action=action,
        metrics=metrics,
        selection_objective=selection_objective,
        scalarized_lower_bound=scalarized_lower_bound,
    )


def _candidate_from_action(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    scalarization_id: str,
    construction: CandidateConstruction,
    action: V1PackedRemovalAction,
    metrics: _CandidateMetricTuple,
    selection_objective: float,
    scalarized_lower_bound: float | None,
) -> V1ProductionGeometricCandidate:
    (
        resource,
        path,
        signed,
        diagonal_surrogate,
        residual_diagonal_sqrt_sum,
        certified_residual,
        low_rank,
        quadratic,
    ) = metrics
    if _exceeds_upper_bound(
        path,
        problem.finsler_path_budget,
    ) or _exceeds_upper_bound(
        quadratic,
        problem.quadratic_trust_energy_budget,
    ):
        raise ValueError("cannot package a budget-infeasible production candidate")
    return V1ProductionGeometricCandidate(
        action=action,
        route_count=problem.route_count,
        scalarization_id=scalarization_id,
        removal_count=action.removed_count,
        resource_gain=resource,
        corrected_path_cost=path,
        signed_covector_sum=signed,
        residual_diagonal_sum=diagonal_surrogate,
        diagonal_surrogate_q=diagonal_surrogate,
        residual_diagonal_sqrt_sum=residual_diagonal_sqrt_sum,
        certified_residual_quadratic=certified_residual,
        low_rank_quadratic=low_rank,
        quadratic_trust_value=quadratic,
        proposal_burden=_proposal_burden(problem, path=path, quadratic=quadratic),
        selection_objective=selection_objective,
        construction=construction,
        scalarized_lower_bound=scalarized_lower_bound,
        scalarized_gap=(
            None
            if scalarized_lower_bound is None
            else selection_objective - scalarized_lower_bound
        ),
        fixed_count_scalarized_lower_bound_supported=scalarized_lower_bound is not None,
    )


def _certified_residual_envelope(
    problem: V1ProductionGeometricSelectorProblem,
    *,
    count: int,
    residual_sqrt_sum: float,
) -> float:
    if count <= 0:
        return 0.0
    spectral = problem.residual_spectral_bound * float(count)
    diagonal_l1 = residual_sqrt_sum * residual_sqrt_sum
    return min(spectral, diagonal_l1)


def _residual_spectral_bound_receipt_metadata(
    problem: V1ProductionGeometricSelectorProblem,
) -> dict[str, Any]:
    sampled_snapshot = (
        problem.fd_residual_certificate_kind
        == PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND
    )
    raw_row_derivation = (
        problem.fd_residual_certificate_kind
        == PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_KIND
    )
    max_residual_diagonal = (
        float(np.max(problem.residual_diagonal))
        if problem.residual_diagonal.size
        else 0.0
    )
    return {
        "fd_residual_certificate_schema": problem.fd_residual_certificate_schema,
        "fd_residual_certificate_kind": problem.fd_residual_certificate_kind,
        "fd_residual_certificate_hash": problem.fd_residual_certificate_hash,
        "fd_residual_geometry_state_hash": problem.fd_residual_geometry_state_hash,
        "fd_ordered_route_axis_hash": problem.fd_ordered_route_axis_hash,
        "physical_route_axis_hash": problem.route_axis_hash,
        "fd_residual_certificate_statement": (
            PRODUCTION_SELECTOR_FD_RESIDUAL_CERTIFICATE_STATEMENT
        ),
        "fd_residual_certificate_nonproduction": (
            problem.fd_residual_certificate_kind
            == PRODUCTION_SELECTOR_SYNTHETIC_FD_CERTIFICATE_KIND
        ),
        "fd_source_row_derivation_authority": raw_row_derivation,
        "fd_sampled_snapshot_numerical_replay_authority": sampled_snapshot,
        "fd_finite_population_geometry_uncertainty_certified": (
            False if sampled_snapshot else None
        ),
        "budget_certificate_scope": (
            "frozen_surrogate_attribution_chart_only_direct_oracle_measures_"
            "nonlinear_child_function"
        ),
        "residual_spectral_bound": problem.residual_spectral_bound,
        "max_residual_diagonal": max_residual_diagonal,
        "residual_spectral_bound_original": problem.metadata.get(
            "residual_spectral_bound_original",
            problem.residual_spectral_bound,
        ),
        "residual_spectral_bound_snapped_to_max_diagonal": bool(
            problem.metadata.get("residual_spectral_bound_snapped_to_max_diagonal", False)
        ),
        "residual_spectral_bound_snap_reason": problem.metadata.get(
            "residual_spectral_bound_snap_reason",
            None,
        ),
    }


def _chart_provenance_receipt_metadata(
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
) -> dict[str, Any]:
    provenance = problem.chart_provenance
    production_authority = provenance is not None and not policy.diagnostic_only
    common = {
        "production_chart_provenance_present": provenance is not None,
        "production_chart_authority": production_authority,
        "selector_policy_diagnostic_only": policy.diagnostic_only,
        "parent_mask_hash": problem.parent_mask.mask_hash,
        "physical_route_axis_hash": problem.route_axis_hash,
        "ordered_route_axis_hash": problem.fd_ordered_route_axis_hash,
        "route_count": problem.route_count,
        "corrected_path_cost_hash": content_hash(
            problem.corrected_path_cost,
            prefix="partition-free-v1-production-selector-corrected-path-",
        ),
        "eligible_mask_hash": content_hash(
            problem.eligible_mask,
            prefix="partition-free-v1-production-selector-eligible-mask-",
        ),
        "low_rank_factor_hash": content_hash(
            problem.low_rank_factor,
            prefix="partition-free-v1-production-selector-low-rank-factor-",
        ),
        "residual_diagonal_hash": content_hash(
            problem.residual_diagonal,
            prefix="partition-free-v1-production-selector-residual-diagonal-",
        ),
        "low_rank_row_count": problem.rank,
        "resource_plane": problem.resource_plane,
        "resource_cost_policy_hash": problem.resource_cost_policy_hash,
        "legality_policy_hash": problem.legality_policy_hash,
        "eligibility_receipt_hash": problem.eligibility_receipt_hash,
    }
    if provenance is None:
        return {
            **common,
            "synthetic_diagnostic_chart_only": True,
            "production_mask_commit_authority": False,
        }
    if type(provenance) is V1SampledSelectorChartProvenance:
        certificate = provenance.snapshot_certificate
        position = provenance.proposal_position_certificate
        coordinates = provenance.corrected_attribution_coordinates
        discovery = provenance.score_discovery_coordinates
        budgets = provenance.geometric_trust_budgets
        calibration = provenance.fixed_affine_bce_calibration
        return {
            **common,
            "synthetic_diagnostic_chart_only": False,
            "production_mask_commit_authority": False,
            "chart_provenance_kind": (
                "sampled_snapshot_numerical_replay_proposal_only"
            ),
            "chart_provenance_hash": provenance.provenance_hash,
            "sampled_snapshot_certificate_hash": certificate.certificate_hash,
            "sampled_geometry_source_hash": certificate.sampled_geometry_source_hash,
            "sampled_geometry_admission_receipt_hash": (
                certificate.admission_receipt_hash
            ),
            "sampled_geometry_assembly_receipt_hash": certificate.assembly_receipt_hash,
            "sampled_geometry_state_hash": certificate.geometry_state_hash,
            "sampled_geometry_result_hash": certificate.geometry_result_hash,
            "sampled_numerical_assembly_receipt_hash": (
                certificate.numerical_assembly_receipt_hash
            ),
            "sampled_corrected_moments_state_hash": (
                certificate.corrected_moments_state_hash
            ),
            "sampled_corrected_point_receipt_hash": (
                certificate.corrected_point_receipt_hash
            ),
            "sampled_fd_design_hash": certificate.fd_design_hash,
            "sampled_fd_snapshot_hash": certificate.fd_snapshot_hash,
            "sampled_snapshot_numerical_backend_hash": (
                certificate.snapshot_numerical_backend_hash
            ),
            "sampled_context_mixture_hash": certificate.context_mixture_hash,
            "sampled_sample_weight_hash": certificate.sample_weight_hash,
            "sampled_source_measurement_axis_hash": (
                certificate.source_measurement_axis_hash
            ),
            "sampled_raw_logit_stream_source_hash": (
                certificate.raw_logit_stream_source_hash
            ),
            "sampled_raw_logit_coordinate": certificate.raw_logit_coordinate,
            "sampled_mean_component_hash": certificate.mean_component_hash,
            "sampled_centered_factor_hash": certificate.centered_factor_hash,
            "sampled_selector_low_rank_factor_hash": certificate.low_rank_factor_hash,
            "sampled_selector_residual_diagonal_hash": (
                certificate.residual_diagonal_hash
            ),
            "sampled_mean_row_count": certificate.mean_row_count,
            "sampled_centered_rank": certificate.centered_rank,
            "sampled_raw_row_derivation_absence_hash": (
                certificate.raw_row_derivation_absence_hash
            ),
            "sampled_snapshot_numerical_replay_authority": True,
            "sampled_source_row_proof_authority": False,
            "sampled_finite_population_geometry_uncertainty_certified": False,
            "sampled_geometry_statistical_optimality_claim": False,
            "sampled_proposal_only": True,
            "sampled_proposal_position_certificate_hash": position.certificate_hash,
            "sampled_physical_route_axis_declaration_hash": (
                position.physical_route_axis_declaration_hash
            ),
            "sampled_position_rule": position.position_rule,
            "sampled_proposal_positional_correspondence_authority": True,
            "sampled_pre_freeze_ordered_to_physical_application_authority": False,
            "sampled_irreversible_order_authority": False,
            "sampled_irreversible_application_authority": False,
            "sampled_fq_certification_authority": False,
            "sampled_bce_oracle_authority": False,
            "sampled_mask_commit_authority": False,
            "sampled_production_execution_authority": False,
            "sampled_terminal_authority": False,
            "sampled_chart_coordinates_hash": coordinates.coordinates_hash,
            "sampled_campaign_binding_hash": coordinates.campaign_binding_hash,
            "sampled_model_seal_hash": coordinates.model_seal_hash,
            "sampled_query_id": coordinates.query_id,
            "sampled_query_contract_hash": coordinates.query_contract_hash,
            "sampled_preprocessing_hash": coordinates.preprocessing_hash,
            "sampled_sampling_law_hash": coordinates.sampling_law_hash,
            "sampled_train_bank_source_hash": coordinates.train_bank_source_hash,
            "sampled_calibration_identity": coordinates.calibration_identity,
            "sampled_proposal_generation": coordinates.proposal_generation,
            "sampled_score_discovery_coordinates_hash": discovery.coordinates_hash,
            "sampled_score_discovery_registration_hash": discovery.registration_hash,
            "sampled_score_discovery_design_hash": discovery.design_hash,
            "sampled_score_discovery_look_receipt_hash": discovery.look_receipt_hash,
            "sampled_score_discovery_result_hash": discovery.result_hash,
            "sampled_score_discovery_statistics_hash": discovery.statistics_hash,
            "sampled_score_discovery_evidence_hash": discovery.evidence_hash,
            "sampled_score_discovery_look_index": discovery.look_index,
            "sampled_score_zero_regret_authority": False,
            "fixed_affine_bce_calibration_hash": calibration.content_hash,
            "fixed_affine_bce_calibration_scale": calibration.scale,
            "fixed_affine_bce_calibration_offset": calibration.offset,
            "fixed_affine_bce_calibration_state_id": calibration.calibration_state_id,
            "dense_baseline_mean_bce_increment_budget": (
                budgets.maximum_dense_baseline_mean_bce_increment
            ),
            "geometric_trust_budget_hash": budgets.budget_hash,
            "geometric_trust_budget_policy_hash": budgets.policy_hash,
            "geometric_trust_budget_damage_target_derivation_hash": (
                budgets.damage_target_derivation_hash
            ),
            "tau_F": budgets.finsler_path_budget,
            "tau_L": budgets.metric_length_budget,
            "tau_Q": budgets.quadratic_trust_energy_budget,
            "tau_Q_equals_tau_L_squared": same_geometric_coordinate(
                budgets.quadratic_trust_energy_budget,
                budgets.metric_length_budget * budgets.metric_length_budget,
            ),
            "raw_logit_to_calibrated_logit_slope": (
                budgets.raw_logit_to_calibrated_logit_slope
            ),
            "resource_cost_hash": provenance.resource_cost_hash,
        }
    fields = provenance.corrected_attribution_fields
    coordinates = provenance.corrected_attribution_coordinates
    budgets = provenance.geometric_trust_budgets
    calibration = provenance.fixed_affine_bce_calibration
    fd_receipt = problem.fd_residual_derivation_receipt
    return {
        **common,
        "synthetic_diagnostic_chart_only": False,
        "production_mask_commit_authority": False,
        "chart_provenance_kind": "global_raw_row_derivation",
        "chart_provenance_hash": provenance.provenance_hash,
        "corrected_attribution_fields_hash": fields.fields_hash,
        "corrected_attribution_coordinates_hash": coordinates.coordinates_hash,
        "corrected_attribution_current_mask_hash": coordinates.current_mask_hash,
        "corrected_attribution_campaign_binding_hash": (
            coordinates.campaign_binding_hash
        ),
        "corrected_attribution_query_id": coordinates.query_id,
        "corrected_attribution_query_hash": coordinates.query_hash,
        "corrected_attribution_target_class_id": coordinates.target_class_id,
        "corrected_attribution_model_seal_hash": coordinates.model_seal_hash,
        "corrected_attribution_model_state_hash": coordinates.model_state_hash,
        "corrected_attribution_preprocessing_hash": coordinates.preprocessing_hash,
        "corrected_attribution_source_context_axis_hash": (
            coordinates.source_context_axis_hash
        ),
        "corrected_attribution_source_label_map_hash": (
            coordinates.source_label_map_hash
        ),
        "corrected_attribution_context_population_hash": (
            coordinates.context_population_hash
        ),
        "corrected_attribution_sampling_law_hash": coordinates.sampling_law_hash,
        "corrected_attribution_train_bank_source_hash": (
            coordinates.train_bank_source_hash
        ),
        "corrected_attribution_source_measurement_hash": (
            coordinates.source_measurement_hash
        ),
        "corrected_attribution_raw_stream_identity_hash": (
            coordinates.raw_stream_identity_hash
        ),
        "corrected_attribution_raw_stream_receipt_hash": (
            coordinates.raw_stream_receipt_hash
        ),
        "corrected_attribution_raw_stream_context_ids_hash": (
            coordinates.raw_stream_context_ids_hash
        ),
        "corrected_attribution_destination_transport_hash": (
            coordinates.destination_transport_hash
        ),
        "corrected_attribution_source_hash": fields.source_attribution_hash,
        "corrected_attribution_sample_count": fields.sample_count,
        "corrected_attribution_sample_weight_hash": fields.sample_weight_hash,
        "corrected_attribution_target_scale_authority": fields.target_scale_authority,
        "corrected_attribution_coordinate": fields.metadata.get(
            "attribution_coordinate"
        ),
        "fixed_affine_bce_calibration_hash": calibration.content_hash,
        "fixed_affine_bce_calibration_scale": calibration.scale,
        "fixed_affine_bce_calibration_offset": calibration.offset,
        "fixed_affine_bce_calibration_state_id": calibration.calibration_state_id,
        "dense_baseline_mean_bce_increment_budget": (
            budgets.maximum_dense_baseline_mean_bce_increment
        ),
        "geometric_trust_budget_hash": budgets.budget_hash,
        "geometric_trust_budget_policy_hash": budgets.policy_hash,
        "geometric_trust_budget_damage_target_derivation_hash": (
            budgets.damage_target_derivation_hash
        ),
        "tau_F": budgets.finsler_path_budget,
        "tau_L": budgets.metric_length_budget,
        "tau_Q": budgets.quadratic_trust_energy_budget,
        "tau_Q_equals_tau_L_squared": same_geometric_coordinate(
            budgets.quadratic_trust_energy_budget,
            budgets.metric_length_budget * budgets.metric_length_budget,
        ),
        "raw_logit_to_calibrated_logit_slope": (
            budgets.raw_logit_to_calibrated_logit_slope
        ),
        "fd_derivation_receipt_hash": (
            None if fd_receipt is None else fd_receipt.receipt_hash
        ),
        "fd_sketch_rank": None if fd_receipt is None else fd_receipt.sketch_rank,
        "fd_mean_component_hash": (
            None if fd_receipt is None else fd_receipt.mean_component_hash
        ),
        "fd_centered_factor_hash": (
            None if fd_receipt is None else fd_receipt.centered_factor_hash
        ),
        "fd_exact_residual_diagonal_hash": (
            None if fd_receipt is None else fd_receipt.exact_residual_diagonal_hash
        ),
        "fd_stream_weight_hash": (
            None if fd_receipt is None else fd_receipt.stream_weight_hash
        ),
        "resource_cost_hash": provenance.resource_cost_hash,
    }


def _is_unit_conditional_scalarization(
    scalarization: V1ProductionScalarization,
) -> bool:
    return (
        abs(float(scalarization.path_weight) - 1.0) <= UNIT_SCALARIZATION_TOLERANCE
        and abs(float(scalarization.quadratic_weight) - 1.0)
        <= UNIT_SCALARIZATION_TOLERANCE
    )


def _remaining_linear_pressure(
    values: np.ndarray,
    *,
    remaining_budget: float,
    remaining_slots: int,
) -> np.ndarray:
    if remaining_slots <= 0:
        raise ValueError("remaining_slots must be positive")
    if remaining_budget > 0.0:
        return values * (float(remaining_slots) / remaining_budget)
    return np.where(values <= 0.0, 0.0, math.inf)


def _remaining_quadratic_pressure(
    delta_quadratic: np.ndarray,
    *,
    remaining_budget: float,
    remaining_slots: int,
) -> np.ndarray:
    if remaining_slots <= 0:
        raise ValueError("remaining_slots must be positive")
    positive_delta = np.maximum(delta_quadratic, 0.0)
    if remaining_budget > 0.0:
        return np.sqrt(positive_delta * (float(remaining_slots) / remaining_budget))
    return np.where(delta_quadratic <= 0.0, 0.0, math.inf)


def _conditional_legality_mask(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    selected_mask: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    tertiary: np.ndarray,
) -> np.ndarray:
    finite = (
        problem.selectable_mask
        & ~selected_mask
        & np.isfinite(primary)
        & np.isfinite(secondary)
        & np.isfinite(tertiary)
    )
    if problem.legality_blocks is None and problem.legality_block_ids is None:
        return finite
    live_remaining = problem.live_mask & ~selected_mask
    result = np.asarray(finite, dtype=np.bool_, copy=True)
    minimum = problem.minimum_live_per_block
    if problem.legality_blocks is not None:
        for block in problem.legality_blocks:
            live_slice = live_remaining[block.start : block.stop]
            remaining_live = int(np.count_nonzero(live_slice))
            if remaining_live <= minimum:
                result[block.start : block.stop] = False
                continue
            finite_slice = result[block.start : block.stop]
            finite_count = int(np.count_nonzero(finite_slice))
            if finite_count <= 0:
                continue
            unremovable_remaining = remaining_live - finite_count
            if unremovable_remaining >= minimum:
                continue
            protect_count = minimum - unremovable_remaining
            finite_positions = np.flatnonzero(finite_slice)
            if protect_count >= int(finite_positions.size):
                result[block.start + finite_positions] = False
                continue
            physical = block.start + finite_positions
            for _ in range(protect_count):
                protected = _argmax_lexicographic_route(
                    physical=physical,
                    primary=primary[physical],
                    secondary=secondary[physical],
                    tertiary=tertiary[physical],
                )
                result[protected] = False
                keep = physical != protected
                physical = physical[keep]
                if not physical.size:
                    break
        return result
    assert problem.legality_block_ids is not None
    by_block: dict[str, list[int]] = {}
    finite_by_block: dict[str, list[int]] = {}
    for physical_index, block in enumerate(problem.legality_block_ids):
        if bool(live_remaining[physical_index]):
            by_block.setdefault(block, []).append(physical_index)
            if bool(result[physical_index]):
                finite_by_block.setdefault(block, []).append(physical_index)
    for block, live_indices in by_block.items():
        remaining_live = len(live_indices)
        if remaining_live <= minimum:
            result[np.asarray(live_indices, dtype=np.int64)] = False
            continue
        finite_indices = finite_by_block.get(block, ())
        finite_count = len(finite_indices)
        if finite_count <= 0:
            continue
        unremovable_remaining = remaining_live - finite_count
        if unremovable_remaining >= minimum:
            continue
        protect_count = minimum - unremovable_remaining
        physical = np.asarray(finite_indices, dtype=np.int64)
        for _ in range(protect_count):
            if not physical.size:
                break
            protected = _argmax_lexicographic_route(
                physical=physical,
                primary=primary[physical],
                secondary=secondary[physical],
                tertiary=tertiary[physical],
            )
            result[protected] = False
            physical = physical[physical != protected]
    return result


def _conditional_bottom_k(
    *,
    selectable_mask: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    tertiary: np.ndarray,
    count: int,
    policy: V1ProductionGeometricSelectorPolicy,
) -> _LinearOracleResult:
    selectable = np.flatnonzero(selectable_mask).astype(np.int64, copy=False)
    if selectable.size < count:
        return _LinearOracleResult(
            selected=np.asarray((), dtype=np.int64),
            linear_sum=math.inf,
            infeasible=True,
            refusal_reason="conditional_budget_legality_cannot_supply_registered_expansion",
        )
    values = primary[selectable]
    if count == int(selectable.size):
        pool = np.arange(int(selectable.size), dtype=np.int64)
    else:
        partition = np.argpartition(values, count - 1)[:count]
        threshold = float(np.max(values[partition]))
        tied = _threshold_tie(
            values=values,
            threshold=threshold,
            tolerance=policy.tie_tolerance,
        )
        strict = np.flatnonzero(
            values < threshold - policy.tie_tolerance * max(1.0, abs(threshold))
        ).astype(np.int64, copy=False)
        needed_from_tie = count - int(strict.size)
        pool = (
            np.concatenate((strict, tied)).astype(np.int64, copy=False)
            if 0 < needed_from_tie < int(tied.size)
            else partition
        )
    pool_indices = selectable[pool]
    order = np.lexsort(
        (
            pool_indices,
            tertiary[pool_indices],
            secondary[pool_indices],
            primary[pool_indices],
        )
    )
    selected = np.sort(pool_indices[order][:count]).astype(np.int64, copy=False)
    selected.setflags(write=False)
    return _LinearOracleResult(
        selected=selected,
        linear_sum=float(np.sum(primary[selected], dtype=np.float64)),
    )


@dataclass(frozen=True, slots=True)
class _ConditionalPrefixOutcome:
    candidates: tuple[V1ProductionGeometricCandidate, ...]
    route_visits: int
    oracle_passes: int
    refusals: tuple[str, ...]
    nonpositive_delta_q_repair_count: int = 0
    nonpositive_delta_q_repair_success_count: int = 0
    nonpositive_delta_q_repair_unavailable_count: int = 0
    nonpositive_delta_q_repair_unsuccessful_count: int = 0
    canonical_prefix_recomputation_count: int = 0
    remaining_f_budget_snap_count: int = 0
    remaining_q_budget_snap_count: int = 0


def _conditional_budget_prefix_candidates(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
    scalarization: V1ProductionScalarization,
    counts: tuple[int, ...],
) -> _ConditionalPrefixOutcome:
    if not counts:
        return _ConditionalPrefixOutcome(
            candidates=(),
            route_visits=0,
            oracle_passes=0,
            refusals=(),
        )
    if not _is_unit_conditional_scalarization(scalarization):
        return _ConditionalPrefixOutcome(
            candidates=(),
            route_visits=0,
            oracle_passes=0,
            refusals=("conditional_budget_requires_unit_scalarization_weights",),
        )
    selected_mask = np.zeros(problem.route_count, dtype=np.bool_)
    selected_indices = np.asarray((), dtype=np.int64)
    h_sum = np.zeros(problem.rank, dtype=np.float64)
    path_sum = 0.0
    residual_sqrt_sum = 0.0
    objective_sum = 0.0
    candidates: list[V1ProductionGeometricCandidate] = []
    refusals: list[str] = []
    route_visits = 0
    oracle_passes = 0
    nonpositive_delta_q_repair_count = 0
    nonpositive_delta_q_repair_success_count = 0
    nonpositive_delta_q_repair_unavailable_count = 0
    nonpositive_delta_q_repair_unsuccessful_count = 0
    canonical_prefix_recomputation_count = 0
    remaining_f_budget_snap_count = 0
    remaining_q_budget_snap_count = 0
    singleton_low_rank = np.sum(
        problem.low_rank_factor * problem.low_rank_factor,
        axis=0,
        dtype=np.float64,
    )
    residual_sqrt = np.sqrt(problem.residual_diagonal)
    resource_scale = problem.resource_scale
    resource_normalizer = (
        np.ones(problem.route_count, dtype=np.float64)
        if (not scalarization.resource_normalized or problem.uniform_resource_cost is not None)
        else np.maximum(problem.resource_cost / resource_scale, 1e-300)
    )
    for target_count in counts:
        increment = target_count - int(selected_indices.size)
        if increment <= 0:
            continue
        current_residual = _certified_residual_envelope(
            problem,
            count=int(selected_indices.size),
            residual_sqrt_sum=residual_sqrt_sum,
        )
        current_low_rank = float(np.dot(h_sum, h_sum))
        remaining_f, snapped_f, exhausted_f = _remaining_budget(
            problem.finsler_path_budget,
            path_sum,
        )
        remaining_q, snapped_q, exhausted_q = _remaining_budget(
            problem.quadratic_trust_energy_budget,
            current_low_rank + current_residual,
        )
        remaining_f_budget_snap_count += int(snapped_f)
        remaining_q_budget_snap_count += int(snapped_q)
        if exhausted_f:
            refusals.append("conditional_budget_finsler_budget_exhausted")
            break
        repair_required = exhausted_q
        if repair_required:
            nonpositive_delta_q_repair_count += 1
        low_rank_cross = (
            np.zeros(problem.route_count, dtype=np.float64)
            if problem.rank == 0
            else np.matmul(h_sum, problem.low_rank_factor)
        )
        delta_low_rank = singleton_low_rank + 2.0 * low_rank_cross
        next_residual = np.minimum(
            problem.residual_spectral_bound * float(int(selected_indices.size) + 1),
            np.square(residual_sqrt_sum + residual_sqrt),
        )
        delta_residual = next_residual - current_residual
        delta_q = delta_low_rank + delta_residual
        path_pressure = _remaining_linear_pressure(
            problem.corrected_path_cost,
            remaining_budget=remaining_f,
            remaining_slots=increment,
        )
        metric_pressure = _remaining_quadratic_pressure(
            delta_q,
            remaining_budget=remaining_q,
            remaining_slots=increment,
        )
        primary = np.maximum(path_pressure, metric_pressure) / resource_normalizer
        secondary = path_pressure / resource_normalizer
        tertiary = metric_pressure / resource_normalizer
        selectable = _conditional_legality_mask(
            problem=problem,
            selected_mask=selected_mask,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
        )
        oracle = _conditional_bottom_k(
            selectable_mask=selectable,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            count=increment,
            policy=policy,
        )
        route_visits += problem.route_count
        oracle_passes += 1
        if oracle.infeasible or oracle.tie_refused:
            if repair_required:
                refusals.append("conditional_budget_nonpositive_delta_q_repair_unavailable")
                nonpositive_delta_q_repair_unavailable_count += 1
            elif oracle.refusal_reason is not None:
                refusals.append(oracle.refusal_reason)
            continue
        chosen = np.sort(oracle.selected).astype(np.int64, copy=False)
        selected_indices = np.sort(
            np.concatenate((selected_indices, chosen)).astype(np.int64, copy=False)
        )
        selected_mask.fill(False)
        selected_mask[selected_indices] = True
        path_sum = float(
            np.sum(problem.corrected_path_cost[selected_indices], dtype=np.float64)
        )
        residual_sqrt_sum = float(
            np.sum(residual_sqrt[selected_indices], dtype=np.float64)
        )
        h_sum = (
            np.zeros(problem.rank, dtype=np.float64)
            if problem.rank == 0
            else np.sum(
                problem.low_rank_factor[:, selected_indices],
                axis=1,
                dtype=np.float64,
            )
        )
        canonical_prefix_recomputation_count += 1
        objective_sum += float(np.sum(primary[chosen], dtype=np.float64))
        if int(selected_indices.size) != target_count:
            refusals.append("conditional_budget_expansion_returned_wrong_cardinality")
            continue
        metrics = _metrics_for_indices(problem, selected_indices)
        path = metrics[1]
        quadratic = metrics[7]
        path_feasible = not _exceeds_upper_bound(path, problem.finsler_path_budget)
        quadratic_feasible = not _exceeds_upper_bound(
            quadratic,
            problem.quadratic_trust_energy_budget,
        )
        if repair_required:
            if path_feasible and quadratic_feasible:
                nonpositive_delta_q_repair_success_count += 1
            else:
                nonpositive_delta_q_repair_unsuccessful_count += 1
                if not quadratic_feasible:
                    refusals.append(
                        "conditional_budget_nonpositive_delta_q_repair_endpoint_infeasible"
                    )
        if not path_feasible:
            refusals.append("conditional_budget_finsler_budget_exhausted")
            break
        if quadratic_feasible:
            candidates.append(
                _candidate_from_indices(
                    problem=problem,
                    scalarization_id=scalarization.scalarization_id,
                    construction="registered_prefix_conditional_budget_gauge",
                    indices=selected_indices,
                    metrics=metrics,
                    selection_objective=objective_sum,
                    scalarized_lower_bound=None,
                )
            )
    return _ConditionalPrefixOutcome(
        candidates=tuple(candidates),
        route_visits=route_visits,
        oracle_passes=oracle_passes,
        refusals=tuple(dict.fromkeys(refusals)),
        nonpositive_delta_q_repair_count=nonpositive_delta_q_repair_count,
        nonpositive_delta_q_repair_success_count=(
            nonpositive_delta_q_repair_success_count
        ),
        nonpositive_delta_q_repair_unavailable_count=(
            nonpositive_delta_q_repair_unavailable_count
        ),
        nonpositive_delta_q_repair_unsuccessful_count=(
            nonpositive_delta_q_repair_unsuccessful_count
        ),
        canonical_prefix_recomputation_count=canonical_prefix_recomputation_count,
        remaining_f_budget_snap_count=remaining_f_budget_snap_count,
        remaining_q_budget_snap_count=remaining_q_budget_snap_count,
    )


@dataclass(frozen=True, slots=True)
class _ExactReferenceOutcome:
    candidates: tuple[V1ProductionGeometricCandidate, ...]
    subset_evaluation_count: int = 0
    refusals: tuple[str, ...] = ()


def _removal_preserves_survivor_legality(
    problem: V1ProductionGeometricSelectorProblem,
    selected_indices: np.ndarray,
) -> bool:
    if problem.legality_blocks is None and problem.legality_block_ids is None:
        return True
    child_live = np.asarray(problem.live_mask, dtype=np.bool_, copy=True)
    child_live[np.asarray(selected_indices, dtype=np.int64)] = False
    minimum = problem.minimum_live_per_block
    if problem.legality_blocks is not None:
        for block in problem.legality_blocks:
            if int(np.count_nonzero(child_live[block.start : block.stop])) < minimum:
                return False
        return True
    assert problem.legality_block_ids is not None
    live_by_block: dict[str, int] = {}
    for block_id, is_live in zip(problem.legality_block_ids, child_live, strict=True):
        if bool(is_live):
            live_by_block[block_id] = live_by_block.get(block_id, 0) + 1
    for block_id in set(problem.legality_block_ids):
        if live_by_block.get(block_id, 0) < minimum:
            return False
    return True


def _exact_reference_oracle_repacked_candidates(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
    scalarization: V1ProductionScalarization,
    counts: tuple[int, ...],
) -> _ExactReferenceOutcome:
    if policy.exact_reference_route_limit <= 0:
        return _ExactReferenceOutcome(candidates=())
    if problem.route_count > policy.exact_reference_route_limit:
        return _ExactReferenceOutcome(
            candidates=(),
            refusals=("exact_reference_route_limit_exceeded",),
        )
    if not _is_unit_conditional_scalarization(scalarization):
        return _ExactReferenceOutcome(
            candidates=(),
            refusals=("conditional_budget_requires_unit_scalarization_weights",),
        )
    selectable = np.flatnonzero(problem.selectable_mask).astype(np.int64, copy=False)
    candidates: list[V1ProductionGeometricCandidate] = []
    refusals: list[str] = []
    subset_evaluations = 0
    for count in counts:
        if count > int(selectable.size):
            refusals.append("not_enough_selectable_routes_for_exact_reference_count")
            continue
        subset_count = math.comb(int(selectable.size), count)
        if subset_count > EXACT_REFERENCE_MAX_SUBSET_EVALUATIONS:
            refusals.append("exact_reference_subset_evaluation_cap_exceeded")
            continue
        best_key: tuple[Any, ...] | None = None
        best_indices: np.ndarray | None = None
        best_metrics: _CandidateMetricTuple | None = None
        for combo in combinations(selectable, count):
            subset_evaluations += 1
            selected = np.asarray(combo, dtype=np.int64)
            if not _removal_preserves_survivor_legality(problem, selected):
                continue
            metrics = _metrics_for_indices(problem, selected)
            path = metrics[1]
            quadratic = metrics[7]
            if _exceeds_upper_bound(
                path,
                problem.finsler_path_budget,
            ) or _exceeds_upper_bound(
                quadratic,
                problem.quadratic_trust_energy_budget,
            ):
                continue
            burden = _proposal_burden(problem, path=path, quadratic=quadratic)
            key = (
                -metrics[0],
                burden,
                path,
                quadratic,
                tuple(int(index) for index in selected),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_indices = np.asarray(selected, dtype=np.int64)
                best_metrics = metrics
        if best_indices is None or best_metrics is None:
            refusals.append("exact_reference_no_feasible_candidate_for_count")
            continue
        best_indices.sort()
        best_indices.setflags(write=False)
        candidates.append(
            _candidate_from_indices(
                problem=problem,
                scalarization_id=scalarization.scalarization_id,
                construction="exact_reference_oracle_repacked",
                indices=best_indices,
                metrics=best_metrics,
                selection_objective=_proposal_burden(
                    problem,
                    path=best_metrics[1],
                    quadratic=best_metrics[7],
                ),
                scalarized_lower_bound=None,
            )
        )
    return _ExactReferenceOutcome(
        candidates=tuple(candidates),
        subset_evaluation_count=subset_evaluations,
        refusals=tuple(dict.fromkeys(refusals)),
    )


def _exact_reference_comparator_metadata(
    *,
    exact_candidates: Sequence[V1ProductionGeometricCandidate],
    scalable_candidates: Sequence[V1ProductionGeometricCandidate],
    subset_evaluation_count: int,
    refusals: Sequence[str],
    route_limit: int,
    route_count: int,
) -> dict[str, Any]:
    scalable_by_key: dict[tuple[str, int], V1ProductionGeometricCandidate] = {}
    for candidate in scalable_candidates:
        key = (candidate.scalarization_id, candidate.removal_count)
        existing = scalable_by_key.get(key)
        if existing is None or (
            -candidate.resource_gain,
            candidate.proposal_burden,
            candidate.removed_route_index_hash,
        ) < (
            -existing.resource_gain,
            existing.proposal_burden,
            existing.removed_route_index_hash,
        ):
            scalable_by_key[key] = candidate
    summaries: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for exact in exact_candidates:
        key = (exact.scalarization_id, exact.removal_count)
        scalable = scalable_by_key.get(key)
        summaries.append(
            {
                "scalarization_id": exact.scalarization_id,
                "removal_count": exact.removal_count,
                "candidate_hash": exact.candidate_hash,
                "removed_route_index_hash": exact.removed_route_index_hash,
                "resource_gain": exact.resource_gain,
                "proposal_burden": exact.proposal_burden,
                "corrected_path_cost": exact.corrected_path_cost,
                "quadratic_trust_value": exact.quadratic_trust_value,
            }
        )
        gaps.append(
            {
                "scalarization_id": exact.scalarization_id,
                "removal_count": exact.removal_count,
                "scalable_candidate_available": scalable is not None,
                "exact_resource_gain": exact.resource_gain,
                "scalable_resource_gain": (
                    None if scalable is None else scalable.resource_gain
                ),
                "resource_gap_exact_minus_scalable": (
                    None
                    if scalable is None
                    else exact.resource_gain - scalable.resource_gain
                ),
                "exact_proposal_burden": exact.proposal_burden,
                "scalable_proposal_burden": (
                    None if scalable is None else scalable.proposal_burden
                ),
                "proposal_burden_gap_scalable_minus_exact": (
                    None
                    if scalable is None
                    else scalable.proposal_burden - exact.proposal_burden
                ),
                "same_action": (
                    False
                    if scalable is None
                    else exact.removed_route_index_hash
                    == scalable.removed_route_index_hash
                ),
            }
        )
    return {
        "exact_reference_diagnostic_only": True,
        "exact_reference_not_merged_into_frontier": True,
        "exact_reference_route_limit": route_limit,
        "exact_reference_evaluated": subset_evaluation_count > 0,
        "exact_reference_route_limit_satisfied": (
            route_limit > 0 and route_count <= route_limit
        ),
        "exact_reference_candidate_count": len(tuple(exact_candidates)),
        "exact_reference_closed_count_count": len(tuple(exact_candidates)),
        "exact_reference_closed_counts": tuple(
            candidate.removal_count for candidate in exact_candidates
        ),
        "exact_reference_candidate_hashes": tuple(
            candidate.candidate_hash for candidate in exact_candidates
        ),
        "exact_reference_candidate_summaries": tuple(summaries),
        "exact_reference_comparator_gaps": tuple(gaps),
        "exact_reference_subset_evaluation_count": subset_evaluation_count,
        "exact_reference_refusal_reasons": tuple(dict.fromkeys(refusals)),
    }


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional torch install.
        raise RuntimeError(
            "backend='torch' requires PyTorch to be installed"
        ) from exc
    return torch


def _torch_tensor_nbytes(tensor: Any) -> int:
    return int(tensor.nelement() * tensor.element_size())


@dataclass(frozen=True, slots=True)
class _TorchSelectorTensors:
    torch: Any
    device: str
    dtype_name: str
    live: Any
    eligible: Any
    selectable: Any
    path: Any
    signed: Any | None
    factor: Any
    residual: Any
    resource: Any | None
    tensor_payload_bytes: int


@dataclass(frozen=True, slots=True)
class _TorchLadderPrefixResult:
    candidates: tuple[V1ProductionGeometricCandidate, ...]
    route_visits: int
    oracle_passes: int
    refusals: tuple[str, ...]
    complement_action_count: int = 0
    materialized_removed_index_count: int = 0
    materialized_keep_index_count: int = 0
    materialized_selectable_mask_count: int = 0
    legacy_legality_cpu_sync_count: int = 0
    prefix_reuse_count: int = 0
    prefix_tie_check_pass_count: int = 0
    prefix_shared_score_pass_count: int = 0
    fallback_required: bool = False
    nonpositive_delta_q_repair_count: int = 0
    nonpositive_delta_q_repair_success_count: int = 0
    nonpositive_delta_q_repair_unavailable_count: int = 0
    nonpositive_delta_q_repair_unsuccessful_count: int = 0
    canonical_prefix_recomputation_count: int = 0
    remaining_f_budget_snap_count: int = 0
    remaining_q_budget_snap_count: int = 0


def _torch_selector_tensors(
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
) -> _TorchSelectorTensors:
    torch = _torch_module()
    device = torch.device(policy.torch_device)
    dtype = torch.float64
    try:
        live = torch.tensor(problem.live_mask, device=device, dtype=torch.bool)
        eligible = torch.tensor(problem.eligible_mask, device=device, dtype=torch.bool)
        path = torch.tensor(problem.corrected_path_cost, device=device, dtype=dtype)
        signed = (
            None
            if problem.signed_covector is None
            else torch.tensor(problem.signed_covector, device=device, dtype=dtype)
        )
        factor = torch.tensor(problem.low_rank_factor, device=device, dtype=dtype)
        residual = torch.tensor(problem.residual_diagonal, device=device, dtype=dtype)
        resource = (
            None
            if problem.uniform_resource_cost is not None
            else torch.tensor(problem.resource_cost, device=device, dtype=dtype)
        )
    except Exception as exc:  # pragma: no cover - device availability is host-dependent.
        raise RuntimeError(
            f"could not place production selector tensors on torch device {device!s}"
        ) from exc
    selectable = live & eligible
    tensors = (live, eligible, selectable, path, factor, residual)
    payload = sum(_torch_tensor_nbytes(tensor) for tensor in tensors)
    if signed is not None:
        payload += _torch_tensor_nbytes(signed)
    if resource is not None:
        payload += _torch_tensor_nbytes(resource)
    return _TorchSelectorTensors(
        torch=torch,
        device=str(device),
        dtype_name=str(dtype).replace("torch.", ""),
        live=live,
        eligible=eligible,
        selectable=selectable,
        path=path,
        signed=signed,
        factor=factor,
        residual=residual,
        resource=resource,
        tensor_payload_bytes=int(payload),
    )


def _torch_threshold_tie_refusal(
    *,
    policy: V1ProductionGeometricSelectorPolicy,
    tensors: _TorchSelectorTensors,
    adjusted: Any,
    secondary: Any | None = None,
    tertiary: Any | None = None,
    actual_selectable: Any,
    target_count: int,
    threshold: float,
    complement: bool,
) -> tuple[bool, Any | None]:
    torch = tensors.torch
    scale = max(1.0, abs(threshold))
    tolerance = policy.tie_tolerance * scale
    tie_mask = actual_selectable & (torch.abs(adjusted - threshold) <= tolerance)
    if complement:
        strict_mask = actual_selectable & (adjusted > threshold + tolerance)
        strict_count = int(torch.count_nonzero(strict_mask).item())
        needed_from_tie = target_count - strict_count
    else:
        strict_mask = actual_selectable & (adjusted < threshold - tolerance)
        strict_count = int(torch.count_nonzero(strict_mask).item())
        needed_from_tie = target_count - strict_count
    tied = int(torch.count_nonzero(tie_mask).item())
    if 0 < needed_from_tie < tied:
        if (
            secondary is None
            and tertiary is None
            and policy.tie_policy == "refuse_threshold_ties"
        ):
            return True, None
        tie_indices = torch.nonzero(tie_mask, as_tuple=False).reshape(-1)
        strict_indices = torch.nonzero(strict_mask, as_tuple=False).reshape(-1)
        if secondary is not None or tertiary is not None:
            tie_physical = np.asarray(tie_indices.detach().cpu().numpy(), dtype=np.int64)
            tie_secondary = (
                np.zeros_like(tie_physical, dtype=np.float64)
                if secondary is None
                else np.asarray(
                    secondary.index_select(0, tie_indices).detach().cpu().numpy(),
                    dtype=np.float64,
                )
            )
            tie_tertiary = (
                np.zeros_like(tie_physical, dtype=np.float64)
                if tertiary is None
                else np.asarray(
                    tertiary.index_select(0, tie_indices).detach().cpu().numpy(),
                    dtype=np.float64,
                )
            )
            tie_primary = np.asarray(
                adjusted.index_select(0, tie_indices).detach().cpu().numpy(),
                dtype=np.float64,
            )
            order_cpu = np.lexsort(
                (tie_physical, tie_tertiary, tie_secondary, tie_primary)
            )
            order = torch.as_tensor(order_cpu, device=adjusted.device, dtype=torch.long)
            if complement:
                chosen_ties = tie_indices.index_select(0, order[-needed_from_tie:])
            else:
                chosen_ties = tie_indices.index_select(0, order[:needed_from_tie])
        elif complement:
            chosen_ties = tie_indices[-needed_from_tie:]
        else:
            chosen_ties = tie_indices[:needed_from_tie]
        return False, torch.cat((strict_indices, chosen_ties), dim=0)
    if needed_from_tie < 0 or needed_from_tie > tied:
        return True, None
    return False, None


def _metrics_from_accumulators(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    count: int,
    resource: float,
    path: float,
    signed: float,
    diagonal_surrogate: float,
    residual_diagonal_sqrt_sum: float,
    low_rank: float,
) -> _CandidateMetricTuple:
    certified_residual = _certified_residual_envelope(
        problem,
        count=count,
        residual_sqrt_sum=residual_diagonal_sqrt_sum,
    )
    return (
        resource,
        path,
        signed,
        diagonal_surrogate,
        residual_diagonal_sqrt_sum,
        certified_residual,
        low_rank,
        certified_residual + low_rank,
    )


def _torch_conditional_legality_mask(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    tensors: _TorchSelectorTensors,
    selected_mask: Any,
    primary: Any,
    secondary: Any,
    tertiary: Any,
) -> Any:
    torch = tensors.torch
    finite = (
        tensors.selectable
        & (~selected_mask)
        & torch.isfinite(primary)
        & torch.isfinite(secondary)
        & torch.isfinite(tertiary)
    )
    if problem.legality_blocks is None and problem.legality_block_ids is None:
        return finite
    result = finite.clone()
    live_remaining = tensors.live & (~selected_mask)
    minimum = problem.minimum_live_per_block
    if problem.legality_blocks is not None:
        for block in problem.legality_blocks:
            live_slice = live_remaining[block.start : block.stop]
            remaining_live = int(torch.count_nonzero(live_slice).item())
            if remaining_live <= minimum:
                result[block.start : block.stop] = False
                continue
            finite_slice = result[block.start : block.stop]
            finite_count = int(torch.count_nonzero(finite_slice).item())
            if finite_count <= 0:
                continue
            unremovable_remaining = remaining_live - finite_count
            if unremovable_remaining >= minimum:
                continue
            protect_count = minimum - unremovable_remaining
            for _ in range(protect_count):
                finite_slice = result[block.start : block.stop]
                finite_positions = torch.nonzero(finite_slice, as_tuple=False).reshape(-1)
                if int(finite_positions.numel()) <= 0:
                    break
                local_primary = primary[block.start : block.stop].index_select(
                    0,
                    finite_positions,
                )
                max_primary = torch.max(local_primary)
                tied_positions = finite_positions[local_primary == max_primary]
                local_secondary = secondary[block.start : block.stop].index_select(
                    0,
                    tied_positions,
                )
                max_secondary = torch.max(local_secondary)
                tied_positions = tied_positions[local_secondary == max_secondary]
                local_tertiary = tertiary[block.start : block.stop].index_select(
                    0,
                    tied_positions,
                )
                max_tertiary = torch.max(local_tertiary)
                tied_positions = tied_positions[local_tertiary == max_tertiary]
                protected = block.start + int(torch.max(tied_positions).item())
                result[protected] = False
        return result
    adjusted_cpu = np.asarray(primary.detach().cpu().numpy(), dtype=np.float64)
    secondary_cpu = np.asarray(secondary.detach().cpu().numpy(), dtype=np.float64)
    tertiary_cpu = np.asarray(tertiary.detach().cpu().numpy(), dtype=np.float64)
    selected_cpu = np.asarray(selected_mask.detach().cpu().numpy(), dtype=np.bool_)
    finite_cpu = np.asarray(result.detach().cpu().numpy(), dtype=np.bool_)
    live_remaining_cpu = np.logical_and(problem.live_mask, ~selected_cpu)
    by_block: dict[str, list[int]] = {}
    finite_by_block: dict[str, list[int]] = {}
    assert problem.legality_block_ids is not None
    for physical, block in enumerate(problem.legality_block_ids):
        if bool(live_remaining_cpu[physical]):
            by_block.setdefault(block, []).append(physical)
            if bool(finite_cpu[physical]):
                finite_by_block.setdefault(block, []).append(physical)
    protected: list[int] = []
    for block, live_indices in by_block.items():
        remaining_live = len(live_indices)
        if remaining_live <= minimum:
            protected.extend(live_indices)
            continue
        finite_indices = finite_by_block.get(block, ())
        finite_count = len(finite_indices)
        if finite_count <= 0:
            continue
        unremovable_remaining = remaining_live - finite_count
        if unremovable_remaining >= minimum:
            continue
        physical = np.asarray(finite_indices, dtype=np.int64)
        for _ in range(minimum - unremovable_remaining):
            if not physical.size:
                break
            chosen = _argmax_lexicographic_route(
                physical=physical,
                primary=adjusted_cpu[physical],
                secondary=secondary_cpu[physical],
                tertiary=tertiary_cpu[physical],
            )
            protected.append(chosen)
            physical = physical[physical != chosen]
    if protected:
        result[
            torch.as_tensor(protected, device=primary.device, dtype=torch.long)
        ] = False
    return result


def _torch_action_from_prefix(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    tensors: _TorchSelectorTensors,
    selected_indices: Any,
    selected_mask: Any,
) -> tuple[V1PackedRemovalAction, int, int, int, bool]:
    selected_count = int(selected_indices.numel())
    if selected_count > problem.selectable_count // 2:
        child = np.asarray(problem.parent_mask.to_bool_array(), dtype=np.bool_).copy()
        removed = np.asarray(selected_mask.detach().cpu().numpy(), dtype=np.bool_)
        child[removed] = False
        action = V1PackedRemovalAction.from_resulting_mask(
            parent_mask=problem.parent_mask,
            resulting_mask=HardMask.from_bool_array(
                child,
                route_axis_hash=problem.parent_mask.route_axis_hash,
            ),
        )
        keep_count = int(problem.selectable_count - selected_count)
        return action, 0, keep_count, 1, True
    selected = _torch_indices_to_sorted_numpy(selected_indices)
    action = V1PackedRemovalAction.from_removed_indices(
        parent_mask=problem.parent_mask,
        removed_indices=selected,
    )
    return action, int(selected.size), 0, 0, False


def _torch_conditional_budget_prefix_candidates(
    *,
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
    scalarization: V1ProductionScalarization,
    tensors: _TorchSelectorTensors,
    counts: tuple[int, ...],
) -> _TorchLadderPrefixResult:
    if not counts:
        return _TorchLadderPrefixResult(
            candidates=(),
            route_visits=0,
            oracle_passes=0,
            refusals=(),
        )
    if not _is_unit_conditional_scalarization(scalarization):
        return _TorchLadderPrefixResult(
            candidates=(),
            route_visits=0,
            oracle_passes=0,
            refusals=("conditional_budget_requires_unit_scalarization_weights",),
        )
    torch = tensors.torch
    selected_mask = torch.zeros(problem.route_count, device=tensors.path.device, dtype=torch.bool)
    selected_indices = torch.empty(0, device=tensors.path.device, dtype=torch.long)
    h_sum = torch.zeros(problem.rank, dtype=tensors.path.dtype, device=tensors.path.device)
    zero = tensors.path.new_tensor(0.0)
    path_sum = zero.clone()
    diagonal_sum = zero.clone()
    residual_sqrt_sum = zero.clone()
    signed_sum = zero.clone()
    resource_sum = zero.clone()
    objective_sum = zero.clone()
    singleton_low_rank = torch.sum(tensors.factor * tensors.factor, dim=0)
    residual_sqrt = torch.sqrt(tensors.residual)
    resource_scale = problem.resource_scale
    if scalarization.resource_normalized and tensors.resource is not None:
        resource_normalizer = torch.clamp(
            tensors.resource / float(resource_scale),
            min=1e-300,
        )
    else:
        resource_normalizer = torch.ones_like(tensors.path)
    candidates: list[V1ProductionGeometricCandidate] = []
    refusals: list[str] = []
    route_visits = 0
    oracle_passes = 0
    complement_actions = 0
    materialized_removed = 0
    materialized_kept = 0
    materialized_selectable_masks = 0
    legacy_syncs = 0
    tie_check_passes = 0
    nonpositive_delta_q_repair_count = 0
    nonpositive_delta_q_repair_success_count = 0
    nonpositive_delta_q_repair_unavailable_count = 0
    nonpositive_delta_q_repair_unsuccessful_count = 0
    canonical_prefix_recomputation_count = 0
    remaining_f_budget_snap_count = 0
    remaining_q_budget_snap_count = 0
    for target_count in counts:
        selected_count = int(selected_indices.numel())
        increment = target_count - selected_count
        if increment <= 0:
            continue
        current_residual = _certified_residual_envelope(
            problem,
            count=selected_count,
            residual_sqrt_sum=float(residual_sqrt_sum.item()),
        )
        current_low_rank = float(h_sum.dot(h_sum).item()) if problem.rank else 0.0
        remaining_f, snapped_f, exhausted_f = _remaining_budget(
            problem.finsler_path_budget,
            float(path_sum.item()),
        )
        remaining_q, snapped_q, exhausted_q = _remaining_budget(
            problem.quadratic_trust_energy_budget,
            current_low_rank + current_residual,
        )
        remaining_f_budget_snap_count += int(snapped_f)
        remaining_q_budget_snap_count += int(snapped_q)
        if exhausted_f:
            refusals.append("conditional_budget_finsler_budget_exhausted")
            break
        repair_required = exhausted_q
        if repair_required:
            nonpositive_delta_q_repair_count += 1
        low_rank_cross = (
            torch.zeros_like(tensors.path)
            if problem.rank == 0
            else torch.matmul(h_sum, tensors.factor)
        )
        delta_low_rank = singleton_low_rank + 2.0 * low_rank_cross
        next_residual = torch.minimum(
            torch.full_like(
                tensors.path,
                problem.residual_spectral_bound * float(selected_count + 1),
            ),
            torch.square(residual_sqrt_sum + residual_sqrt),
        )
        delta_residual = next_residual - float(current_residual)
        delta_q = delta_low_rank + delta_residual
        if remaining_f > 0.0:
            path_pressure = tensors.path * (float(increment) / remaining_f)
        else:
            path_pressure = torch.where(
                tensors.path <= 0.0,
                torch.zeros_like(tensors.path),
                torch.full_like(tensors.path, math.inf),
            )
        positive_delta = torch.clamp(delta_q, min=0.0)
        if remaining_q > 0.0:
            metric_pressure = torch.sqrt(positive_delta * (float(increment) / remaining_q))
        else:
            metric_pressure = torch.where(
                delta_q <= 0.0,
                torch.zeros_like(tensors.path),
                torch.full_like(tensors.path, math.inf),
            )
        primary = torch.maximum(path_pressure, metric_pressure) / resource_normalizer
        secondary = path_pressure / resource_normalizer
        tertiary = metric_pressure / resource_normalizer
        selectable = _torch_conditional_legality_mask(
            problem=problem,
            tensors=tensors,
            selected_mask=selected_mask,
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
        )
        selectable_count = int(torch.count_nonzero(selectable).item())
        route_visits += problem.route_count
        oracle_passes += 1
        if selectable_count < increment:
            if repair_required:
                refusals.append("conditional_budget_nonpositive_delta_q_repair_unavailable")
                nonpositive_delta_q_repair_unavailable_count += 1
            else:
                refusals.append(
                    "conditional_budget_legality_cannot_supply_registered_expansion"
                )
            continue
        masked = torch.where(selectable, primary, torch.full_like(primary, math.inf))
        if increment > selectable_count // 2:
            keep_count = selectable_count - increment
            if keep_count <= 0:
                chosen_mask = selectable
            else:
                topk_count = min(selectable_count, keep_count + 1)
                keep_values, keep_pool = torch.topk(
                    torch.where(selectable, primary, torch.full_like(primary, -math.inf)),
                    topk_count,
                    largest=True,
                    sorted=True,
                )
                keep = keep_pool[:keep_count]
                if topk_count > keep_count:
                    threshold = float(keep_values[keep_count - 1].item())
                    next_value = float(keep_values[keep_count].item())
                    tied_boundary = next_value >= threshold - policy.tie_tolerance * max(
                        1.0,
                        abs(threshold),
                    )
                    if tied_boundary:
                        tie_check_passes += 1
                        refused, deterministic = _torch_threshold_tie_refusal(
                            policy=policy,
                            tensors=tensors,
                            adjusted=primary,
                            secondary=secondary,
                            tertiary=tertiary,
                            actual_selectable=selectable,
                            target_count=keep_count,
                            threshold=threshold,
                            complement=True,
                        )
                        if refused:
                            refusals.append("linear_oracle_threshold_tie_refused")
                            continue
                        if deterministic is not None:
                            keep = deterministic
                chosen_mask = selectable.clone()
                chosen_mask[keep] = False
            chosen = torch.nonzero(chosen_mask, as_tuple=False).reshape(-1)
        else:
            topk_count = min(selectable_count, increment + 1)
            values, chosen_pool = torch.topk(
                masked,
                topk_count,
                largest=False,
                sorted=True,
            )
            chosen = chosen_pool[:increment]
            if topk_count > increment:
                threshold = float(values[increment - 1].item())
                next_value = float(values[increment].item())
                tied_boundary = next_value <= threshold + policy.tie_tolerance * max(
                    1.0,
                    abs(threshold),
                )
                if tied_boundary:
                    tie_check_passes += 1
                    refused, deterministic = _torch_threshold_tie_refusal(
                        policy=policy,
                        tensors=tensors,
                        adjusted=primary,
                        secondary=secondary,
                        tertiary=tertiary,
                        actual_selectable=selectable,
                        target_count=increment,
                        threshold=threshold,
                        complement=False,
                    )
                    if refused:
                        refusals.append("linear_oracle_threshold_tie_refused")
                        continue
                    if deterministic is not None:
                        chosen = deterministic
            del values
        if int(chosen.numel()) != increment:
            refusals.append("conditional_budget_expansion_returned_wrong_cardinality")
            continue
        chosen = torch.sort(chosen).values
        objective_sum = objective_sum + primary.index_select(0, chosen).sum()
        selected_indices = torch.sort(torch.cat((selected_indices, chosen), dim=0)).values
        selected_mask.zero_()
        selected_mask[selected_indices] = True
        path_sum = tensors.path.index_select(0, selected_indices).sum()
        diagonal_sum = tensors.residual.index_select(0, selected_indices).sum()
        residual_sqrt_sum = residual_sqrt.index_select(0, selected_indices).sum()
        signed_sum = (
            zero.clone()
            if tensors.signed is None
            else tensors.signed.index_select(0, selected_indices).sum()
        )
        if tensors.resource is None:
            resource_sum = tensors.path.new_tensor(
                problem.uniform_resource_cost * int(selected_indices.numel())
            )
        else:
            resource_sum = tensors.resource.index_select(0, selected_indices).sum()
        h_sum = (
            torch.zeros(
                problem.rank,
                dtype=tensors.path.dtype,
                device=tensors.path.device,
            )
            if problem.rank == 0
            else tensors.factor.index_select(1, selected_indices).sum(dim=1)
        )
        canonical_prefix_recomputation_count += 1
        selected_count = int(selected_indices.numel())
        if selected_count != target_count:
            refusals.append("conditional_budget_expansion_returned_wrong_cardinality")
            continue
        low_rank = float(h_sum.dot(h_sum).item()) if problem.rank else 0.0
        metrics = _metrics_from_accumulators(
            problem=problem,
            count=selected_count,
            resource=float(resource_sum.item()),
            path=float(path_sum.item()),
            signed=float(signed_sum.item()),
            diagonal_surrogate=float(diagonal_sum.item()),
            residual_diagonal_sqrt_sum=float(residual_sqrt_sum.item()),
            low_rank=low_rank,
        )
        path = metrics[1]
        quadratic = metrics[7]
        path_feasible = not _exceeds_upper_bound(path, problem.finsler_path_budget)
        quadratic_feasible = not _exceeds_upper_bound(
            quadratic,
            problem.quadratic_trust_energy_budget,
        )
        if repair_required:
            if path_feasible and quadratic_feasible:
                nonpositive_delta_q_repair_success_count += 1
            else:
                nonpositive_delta_q_repair_unsuccessful_count += 1
                if not quadratic_feasible:
                    refusals.append(
                        "conditional_budget_nonpositive_delta_q_repair_endpoint_infeasible"
                    )
        if not path_feasible:
            refusals.append("conditional_budget_finsler_budget_exhausted")
            break
        if not quadratic_feasible:
            continue
        action, removed_count, keep_count, mask_count, complement = (
            _torch_action_from_prefix(
                problem=problem,
                tensors=tensors,
                selected_indices=selected_indices,
                selected_mask=selected_mask,
            )
        )
        materialized_removed += removed_count
        materialized_kept += keep_count
        materialized_selectable_masks += mask_count
        complement_actions += 1 if complement else 0
        candidates.append(
            _candidate_from_action(
                problem=problem,
                scalarization_id=scalarization.scalarization_id,
                construction="registered_prefix_conditional_budget_gauge",
                action=action,
                metrics=metrics,
                selection_objective=float(objective_sum.item()),
                scalarized_lower_bound=None,
            )
        )
    return _TorchLadderPrefixResult(
        candidates=tuple(candidates),
        route_visits=route_visits,
        oracle_passes=oracle_passes,
        refusals=tuple(dict.fromkeys(refusals)),
        complement_action_count=complement_actions,
        materialized_removed_index_count=materialized_removed,
        materialized_keep_index_count=materialized_kept,
        materialized_selectable_mask_count=materialized_selectable_masks,
        legacy_legality_cpu_sync_count=legacy_syncs,
        prefix_reuse_count=len(counts),
        prefix_tie_check_pass_count=tie_check_passes,
        prefix_shared_score_pass_count=oracle_passes,
        fallback_required=False,
        nonpositive_delta_q_repair_count=nonpositive_delta_q_repair_count,
        nonpositive_delta_q_repair_success_count=(
            nonpositive_delta_q_repair_success_count
        ),
        nonpositive_delta_q_repair_unavailable_count=(
            nonpositive_delta_q_repair_unavailable_count
        ),
        nonpositive_delta_q_repair_unsuccessful_count=(
            nonpositive_delta_q_repair_unsuccessful_count
        ),
        canonical_prefix_recomputation_count=canonical_prefix_recomputation_count,
        remaining_f_budget_snap_count=remaining_f_budget_snap_count,
        remaining_q_budget_snap_count=remaining_q_budget_snap_count,
    )


def _torch_indices_to_sorted_numpy(indices: Any | None) -> np.ndarray:
    if indices is None or int(indices.numel()) == 0:
        return np.asarray((), dtype=np.int64)
    ordered = indices.detach().to(device="cpu", dtype=indices.dtype)
    ordered = ordered.sort().values
    return np.asarray(ordered.numpy(), dtype=np.int64)


def _empty_result(
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
    *,
    refusal_reasons: tuple[str, ...],
    metadata: Mapping[str, Any] | None = None,
) -> V1ProductionGeometricSelectorResult:
    receipt_metadata = {
        **_residual_spectral_bound_receipt_metadata(problem),
        **_chart_provenance_receipt_metadata(problem, policy),
    }
    if metadata is not None:
        receipt_metadata.update(dict(metadata))
    receipt = V1ProductionGeometricSelectorReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_count=problem.route_count,
        live_route_count=problem.live_count,
        eligible_route_count=problem.eligible_count,
        selectable_route_count=problem.selectable_count,
        block_count=problem.block_count,
        scalarization_count=len(policy.scalarizations),
        candidate_count=0,
        backend=policy.backend,
        device=policy.torch_device if policy.backend == "torch" else "cpu",
        chunk_size=policy.chunk_size,
        route_score_pass_count=0,
        fixed_count_oracle_route_visit_count=0,
        packed_action_count=0,
        packed_action_payload_bytes=0,
        algorithm_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_ID
        ),
        scalarization_equation_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_SCALARIZATION_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_SCALARIZATION_ID
        ),
        lower_bounds_supported=False,
        production_chart_authority=(
            problem.has_production_chart_provenance and not policy.diagnostic_only
        ),
        selector_policy_diagnostic_only=policy.diagnostic_only,
        refusal_reasons=refusal_reasons,
        metadata=receipt_metadata,
    )
    return V1ProductionGeometricSelectorResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        receipt=receipt,
        frontier=(),
    )


def _select_v1_production_geometric_frontier_torch(
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy,
) -> V1ProductionGeometricSelectorResult:
    tensors = _torch_selector_tensors(problem, policy)
    candidates_by_action: dict[str, V1ProductionGeometricCandidate] = {}
    tie_refusal_count = 0
    budget_refusal_count = 0
    legality_skip_count = 0
    linear_oracle_route_visits = 0
    linear_oracle_pass_count = 0
    counts_considered = 0
    complement_action_count = 0
    materialized_removed_index_count = 0
    materialized_keep_index_count = 0
    materialized_selectable_mask_count = 0
    legacy_legality_cpu_sync_count = 0
    prefix_reuse_count = 0
    prefix_shared_score_pass_count = 0
    prefix_tie_check_pass_count = 0
    canonical_conditional_candidate_count = 0
    nonpositive_delta_q_repair_count = 0
    nonpositive_delta_q_repair_success_count = 0
    nonpositive_delta_q_repair_unavailable_count = 0
    nonpositive_delta_q_repair_unsuccessful_count = 0
    canonical_prefix_recomputation_count = 0
    remaining_f_budget_snap_count = 0
    remaining_q_budget_snap_count = 0
    exact_reference_subset_evaluation_count = 0
    exact_reference_candidates: list[V1ProductionGeometricCandidate] = []
    exact_reference_refusals: list[str] = []
    trajectory_refusals: list[str] = []

    for scalarization in policy.scalarizations:
        counts = _count_ladder(
            policy.count_ladder,
            selectable_count=problem.selectable_count,
            maximum_removal_count=policy.maximum_removal_count,
        )
        if not counts:
            continue
        counts_considered += len(counts)
        ladder_outcome = _torch_conditional_budget_prefix_candidates(
            problem=problem,
            policy=policy,
            scalarization=scalarization,
            tensors=tensors,
            counts=counts,
        )
        canonical_conditional_candidate_count += len(ladder_outcome.candidates)
        nonpositive_delta_q_repair_count += (
            ladder_outcome.nonpositive_delta_q_repair_count
        )
        nonpositive_delta_q_repair_success_count += (
            ladder_outcome.nonpositive_delta_q_repair_success_count
        )
        nonpositive_delta_q_repair_unavailable_count += (
            ladder_outcome.nonpositive_delta_q_repair_unavailable_count
        )
        nonpositive_delta_q_repair_unsuccessful_count += (
            ladder_outcome.nonpositive_delta_q_repair_unsuccessful_count
        )
        canonical_prefix_recomputation_count += (
            ladder_outcome.canonical_prefix_recomputation_count
        )
        remaining_f_budget_snap_count += ladder_outcome.remaining_f_budget_snap_count
        remaining_q_budget_snap_count += ladder_outcome.remaining_q_budget_snap_count
        linear_oracle_route_visits += ladder_outcome.route_visits
        linear_oracle_pass_count += ladder_outcome.oracle_passes
        complement_action_count += ladder_outcome.complement_action_count
        materialized_removed_index_count += (
            ladder_outcome.materialized_removed_index_count
        )
        materialized_keep_index_count += ladder_outcome.materialized_keep_index_count
        materialized_selectable_mask_count += (
            ladder_outcome.materialized_selectable_mask_count
        )
        legacy_legality_cpu_sync_count += ladder_outcome.legacy_legality_cpu_sync_count
        prefix_shared_score_pass_count += ladder_outcome.prefix_shared_score_pass_count
        prefix_reuse_count += ladder_outcome.prefix_reuse_count
        prefix_tie_check_pass_count += ladder_outcome.prefix_tie_check_pass_count
        trajectory_refusals.extend(ladder_outcome.refusals)
        if "linear_oracle_threshold_tie_refused" in ladder_outcome.refusals:
            tie_refusal_count += 1
        if (
            "block_survivor_legality_cannot_supply_fixed_count" in ladder_outcome.refusals
            or (
                "conditional_budget_legality_cannot_supply_registered_expansion"
                in ladder_outcome.refusals
                and (
                    problem.legality_blocks is not None
                    or problem.legality_block_ids is not None
                )
            )
        ):
            legality_skip_count += 1
        if not ladder_outcome.candidates and not ladder_outcome.refusals:
            budget_refusal_count += 1
        for candidate in ladder_outcome.candidates:
            existing = candidates_by_action.get(candidate.removed_route_index_hash)
            if existing is None or (
                -candidate.resource_gain,
                candidate.selection_objective,
                candidate.scalarization_id,
            ) < (
                -existing.resource_gain,
                existing.selection_objective,
                existing.scalarization_id,
            ):
                candidates_by_action[candidate.removed_route_index_hash] = candidate
        exact_outcome = _exact_reference_oracle_repacked_candidates(
            problem=problem,
            policy=policy,
            scalarization=scalarization,
            counts=counts,
        )
        exact_reference_subset_evaluation_count += exact_outcome.subset_evaluation_count
        exact_reference_candidates.extend(exact_outcome.candidates)
        exact_reference_refusals.extend(exact_outcome.refusals)

    frontier = tuple(
        sorted(
            candidates_by_action.values(),
            key=lambda item: (
                -item.resource_gain,
                item.proposal_burden,
                item.removed_route_index_hash,
            ),
        )[: policy.max_frontier_candidates]
    )
    refusal_reasons: list[str] = []
    if not frontier:
        refusal_reasons.append("no_declared_fixed_count_candidate_satisfied_path_and_metric_budgets")
    if tie_refusal_count:
        refusal_reasons.append("threshold_ties_refused_under_registered_tie_policy")
    if budget_refusal_count:
        refusal_reasons.append("some_declared_count_ladder_candidates_exceeded_path_or_metric_budget")
    if legality_skip_count:
        refusal_reasons.append("dynamic_block_survivor_legality_removed_routes_from_global_order")
    for refusal in tuple(dict.fromkeys(trajectory_refusals)):
        if refusal not in {
            "linear_oracle_threshold_tie_refused",
            "block_survivor_legality_cannot_supply_fixed_count",
        }:
            refusal_reasons.append(refusal)
    refusal_reasons.extend(tuple(dict.fromkeys(exact_reference_refusals)))
    if len(candidates_by_action) > policy.max_frontier_candidates:
        refusal_reasons.append("frontier_truncated_to_registered_candidate_bound")
    packed_payload_bytes = sum(
        len(candidate.action.resulting_mask.packed_bits) for candidate in frontier
    )
    receipt = V1ProductionGeometricSelectorReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_count=problem.route_count,
        live_route_count=problem.live_count,
        eligible_route_count=problem.eligible_count,
        selectable_route_count=problem.selectable_count,
        block_count=problem.block_count,
        scalarization_count=len(policy.scalarizations),
        candidate_count=len(frontier),
        backend=policy.backend,
        device=tensors.device,
        chunk_size=policy.chunk_size,
        route_score_pass_count=linear_oracle_pass_count,
        fixed_count_oracle_route_visit_count=linear_oracle_route_visits,
        packed_action_count=len(frontier),
        packed_action_payload_bytes=packed_payload_bytes,
        algorithm_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_ID
        ),
        scalarization_equation_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_SCALARIZATION_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_SCALARIZATION_ID
        ),
        lower_bounds_supported=False,
        production_chart_authority=(
            problem.has_production_chart_provenance and not policy.diagnostic_only
        ),
        selector_policy_diagnostic_only=policy.diagnostic_only,
        refusal_reasons=tuple(refusal_reasons),
        metadata={
            **_residual_spectral_bound_receipt_metadata(problem),
            **_chart_provenance_receipt_metadata(problem, policy),
            "counts_considered": counts_considered,
            "linear_oracle_pass_count": linear_oracle_pass_count,
            "linear_oracle_route_visit_count": linear_oracle_route_visits,
            "tie_refusal_count": tie_refusal_count,
            "budget_refusal_count": budget_refusal_count,
            "legality_skip_count": legality_skip_count,
            "registered_count_ladder": _count_ladder(
                policy.count_ladder,
                selectable_count=problem.selectable_count,
                maximum_removal_count=policy.maximum_removal_count,
            ),
            "maximum_removal_count": policy.maximum_removal_count,
            "legacy_dual_iterations_policy_field": policy.dual_iterations,
            "legacy_dual_initial_step_size_policy_field": policy.dual_initial_step_size,
            "legacy_dual_policy_fields_inert": True,
            "canonical_dual_iterations_used": 0,
            "canonical_selection_construction": (
                "registered_prefix_conditional_budget_gauge"
            ),
            "canonical_order_score": (
                "registered-prefix KKT-motivated remaining-budget stress: "
                "max(A_j/(remaining_F/slots), "
                "sqrt(max(0, ΔQ_cert(j|S))/(remaining_Q/slots)))/resource"
            ),
            "canonical_conditional_delta_q": (
                "ΔQ_cert(j|S)=||h_j||^2+2 h_j·h_S + "
                "min(rho*(|S|+1),(sum_sqrt_d_S+sqrt(d_j))^2) - "
                "min(rho*|S|,(sum_sqrt_d_S)^2)"
            ),
            "canonical_registered_prefix_expansions": True,
            "canonical_conditional_candidate_count": canonical_conditional_candidate_count,
            "canonical_conditional_solver_claim": (
                "approximate_registered_prefix_greedy_no_global_optimality_claim"
            ),
            "nonpositive_delta_q_repair_policy": (
                "registered_nonpositive_delta_q_repair_path_stress_no_negative_bonus_v1"
            ),
            "nonpositive_delta_q_repair_count": nonpositive_delta_q_repair_count,
            "nonpositive_delta_q_repair_success_count": (
                nonpositive_delta_q_repair_success_count
            ),
            "nonpositive_delta_q_repair_unavailable_count": (
                nonpositive_delta_q_repair_unavailable_count
            ),
            "nonpositive_delta_q_repair_unsuccessful_count": (
                nonpositive_delta_q_repair_unsuccessful_count
            ),
            "canonical_prefix_recomputation_count": (
                canonical_prefix_recomputation_count
            ),
            "remaining_f_budget_snap_count": remaining_f_budget_snap_count,
            "remaining_q_budget_snap_count": remaining_q_budget_snap_count,
            "diagnostic_a_only_prefix_archived": True,
            "diagnostic_a_only_prefix_construction": (
                "single_order_corrected_salience_primary_prefix"
            ),
            "diagnostic_a_only_prefix_evaluated": False,
            "canonical_fallback_to_a_only_prefix": False,
            "exact_reference_semantics": (
                "max_resource_subject_to_survivor_legality_F_budget_and_"
                "certified_Q_budget"
            ),
            "exact_reference_q_formula": (
                "Q_cert(S)=||sum_{j in S} h_j||^2 + "
                "min(rho*|S|,(sum_{j in S} sqrt(d_j))^2)"
            ),
            **_exact_reference_comparator_metadata(
                exact_candidates=tuple(exact_reference_candidates),
                scalable_candidates=frontier,
                subset_evaluation_count=exact_reference_subset_evaluation_count,
                refusals=exact_reference_refusals,
                route_limit=policy.exact_reference_route_limit,
                route_count=problem.route_count,
            ),
            "per_count_dual_solver_used": False,
            "input_array_policy": problem.input_array_policy,
            "signed_covector_present": problem.has_signed_covector,
            "admitted_array_payload_bytes": problem.admitted_array_payload_bytes,
            "candidate_mask_bytes_each": len(problem.parent_mask.packed_bits),
            "retains_removed_index_tuples": False,
            "selection_claim": (
                "exact_metrics_for_frozen_packed_frontier_no_global_powerset_"
                "or_fixed_count_optimality_or_lower_bound_claim"
            ),
            "torch_device": tensors.device,
            "torch_dtype": tensors.dtype_name,
            "torch_tensor_payload_bytes": tensors.tensor_payload_bytes,
            "torch_tensors_placed_once_per_selector_call": True,
            "torch_full_axis_score_passes": linear_oracle_pass_count,
            "torch_full_axis_route_visit_count": linear_oracle_route_visits,
            "torch_prefix_reused_count": prefix_reuse_count,
            "torch_prefix_shared_score_pass_count": prefix_shared_score_pass_count,
            "torch_prefix_tie_check_pass_count": prefix_tie_check_pass_count,
            "complement_action_count": complement_action_count,
            "materialized_removed_index_count": materialized_removed_index_count,
            "materialized_keep_index_count": materialized_keep_index_count,
            "materialized_selectable_mask_count": materialized_selectable_mask_count,
            "legacy_legality_cpu_sync_count": legacy_legality_cpu_sync_count,
        },
    )
    return V1ProductionGeometricSelectorResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        receipt=receipt,
        frontier=frontier,
    )


def select_v1_production_geometric_frontier(
    problem: V1ProductionGeometricSelectorProblem,
    policy: V1ProductionGeometricSelectorPolicy | None = None,
) -> V1ProductionGeometricSelectorResult:
    """Build a frozen packed frontier for the direct mask oracle.

    The selector extends a global registered-prefix conditional
    remaining-budget stress trajectory.  It computes exact ``F`` and certified
    complete-set ``Q`` for every emitted prefix, but it does not claim a
    fixed-count optimum, powerset optimum, lower bound, or commit authority.
    """

    if type(problem) is not V1ProductionGeometricSelectorProblem:
        raise TypeError("production geometric selection requires a typed problem")
    if policy is None:
        policy = V1ProductionGeometricSelectorPolicy()
    elif type(policy) is not V1ProductionGeometricSelectorPolicy:
        raise TypeError("policy must be V1ProductionGeometricSelectorPolicy")
    if not policy.diagnostic_only and not problem.has_production_chart_provenance:
        raise PermissionError(
            "production selector policy requires typed production chart provenance; "
            "synthetic compatibility must be explicitly diagnostic"
        )
    if problem.selectable_count <= 0:
        return _empty_result(
            problem,
            policy,
            refusal_reasons=("no_live_route_is_eligible_under_supplied_legality_mask",),
        )
    if policy.backend == "torch":
        return _select_v1_production_geometric_frontier_torch(problem, policy)

    candidates_by_action: dict[str, V1ProductionGeometricCandidate] = {}
    tie_refusal_count = 0
    budget_refusal_count = 0
    legality_skip_count = 0
    linear_oracle_route_visits = 0
    linear_oracle_pass_count = 0
    counts_considered = 0
    canonical_conditional_candidate_count = 0
    nonpositive_delta_q_repair_count = 0
    nonpositive_delta_q_repair_success_count = 0
    nonpositive_delta_q_repair_unavailable_count = 0
    nonpositive_delta_q_repair_unsuccessful_count = 0
    canonical_prefix_recomputation_count = 0
    remaining_f_budget_snap_count = 0
    remaining_q_budget_snap_count = 0
    exact_reference_subset_evaluation_count = 0
    exact_reference_candidates: list[V1ProductionGeometricCandidate] = []
    exact_reference_refusals: list[str] = []
    trajectory_refusals: list[str] = []

    for scalarization in policy.scalarizations:
        counts = _count_ladder(
            policy.count_ladder,
            selectable_count=problem.selectable_count,
            maximum_removal_count=policy.maximum_removal_count,
        )
        if not counts:
            continue
        counts_considered += len(counts)
        ladder_outcome = _conditional_budget_prefix_candidates(
            problem=problem,
            policy=policy,
            scalarization=scalarization,
            counts=counts,
        )
        candidates = ladder_outcome.candidates
        canonical_conditional_candidate_count += len(candidates)
        nonpositive_delta_q_repair_count += (
            ladder_outcome.nonpositive_delta_q_repair_count
        )
        nonpositive_delta_q_repair_success_count += (
            ladder_outcome.nonpositive_delta_q_repair_success_count
        )
        nonpositive_delta_q_repair_unavailable_count += (
            ladder_outcome.nonpositive_delta_q_repair_unavailable_count
        )
        nonpositive_delta_q_repair_unsuccessful_count += (
            ladder_outcome.nonpositive_delta_q_repair_unsuccessful_count
        )
        canonical_prefix_recomputation_count += (
            ladder_outcome.canonical_prefix_recomputation_count
        )
        remaining_f_budget_snap_count += ladder_outcome.remaining_f_budget_snap_count
        remaining_q_budget_snap_count += ladder_outcome.remaining_q_budget_snap_count
        linear_oracle_route_visits += ladder_outcome.route_visits
        linear_oracle_pass_count += ladder_outcome.oracle_passes
        trajectory_refusals.extend(ladder_outcome.refusals)
        if "linear_oracle_threshold_tie_refused" in ladder_outcome.refusals:
            tie_refusal_count += 1
        if (
            "block_survivor_legality_cannot_supply_fixed_count"
            in ladder_outcome.refusals
            or (
                "conditional_budget_legality_cannot_supply_registered_expansion"
                in ladder_outcome.refusals
                and (
                    problem.legality_blocks is not None
                    or problem.legality_block_ids is not None
                )
            )
        ):
            legality_skip_count += 1
        if not candidates and not ladder_outcome.refusals:
            budget_refusal_count += 1
            continue
        for candidate in candidates:
            existing = candidates_by_action.get(candidate.removed_route_index_hash)
            if existing is None or (
                -candidate.resource_gain,
                candidate.selection_objective,
                candidate.scalarization_id,
            ) < (
                -existing.resource_gain,
                existing.selection_objective,
                existing.scalarization_id,
            ):
                candidates_by_action[candidate.removed_route_index_hash] = candidate
        exact_outcome = _exact_reference_oracle_repacked_candidates(
            problem=problem,
            policy=policy,
            scalarization=scalarization,
            counts=counts,
        )
        exact_reference_subset_evaluation_count += exact_outcome.subset_evaluation_count
        exact_reference_candidates.extend(exact_outcome.candidates)
        exact_reference_refusals.extend(exact_outcome.refusals)

    frontier = tuple(
        sorted(
            candidates_by_action.values(),
            key=lambda item: (
                -item.resource_gain,
                item.proposal_burden,
                item.removed_route_index_hash,
            ),
        )[: policy.max_frontier_candidates]
    )
    refusal_reasons: list[str] = []
    if not frontier:
        refusal_reasons.append("no_declared_fixed_count_candidate_satisfied_path_and_metric_budgets")
    if tie_refusal_count:
        refusal_reasons.append("threshold_ties_refused_under_registered_tie_policy")
    if budget_refusal_count:
        refusal_reasons.append("some_declared_count_ladder_candidates_exceeded_path_or_metric_budget")
    if legality_skip_count:
        refusal_reasons.append("dynamic_block_survivor_legality_removed_routes_from_global_order")
    for refusal in tuple(dict.fromkeys(trajectory_refusals)):
        if refusal not in {
            "linear_oracle_threshold_tie_refused",
            "block_survivor_legality_cannot_supply_fixed_count",
        }:
            refusal_reasons.append(refusal)
    refusal_reasons.extend(tuple(dict.fromkeys(exact_reference_refusals)))
    if len(candidates_by_action) > policy.max_frontier_candidates:
        refusal_reasons.append("frontier_truncated_to_registered_candidate_bound")
    packed_payload_bytes = sum(
        len(candidate.action.resulting_mask.packed_bits) for candidate in frontier
    )

    receipt = V1ProductionGeometricSelectorReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_count=problem.route_count,
        live_route_count=problem.live_count,
        eligible_route_count=problem.eligible_count,
        selectable_route_count=problem.selectable_count,
        block_count=problem.block_count,
        scalarization_count=len(policy.scalarizations),
        candidate_count=len(frontier),
        backend=policy.backend,
        device="cpu",
        chunk_size=policy.chunk_size,
        route_score_pass_count=linear_oracle_pass_count,
        fixed_count_oracle_route_visit_count=linear_oracle_route_visits,
        packed_action_count=len(frontier),
        packed_action_payload_bytes=packed_payload_bytes,
        algorithm_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_ID
        ),
        scalarization_equation_id=(
            PRODUCTION_SELECTOR_DIAGNOSTIC_SCALARIZATION_ID
            if policy.diagnostic_only
            else PRODUCTION_SELECTOR_SCALARIZATION_ID
        ),
        lower_bounds_supported=False,
        production_chart_authority=(
            problem.has_production_chart_provenance and not policy.diagnostic_only
        ),
        selector_policy_diagnostic_only=policy.diagnostic_only,
        refusal_reasons=tuple(refusal_reasons),
        metadata={
            **_residual_spectral_bound_receipt_metadata(problem),
            **_chart_provenance_receipt_metadata(problem, policy),
            "counts_considered": counts_considered,
            "linear_oracle_pass_count": linear_oracle_pass_count,
            "linear_oracle_route_visit_count": linear_oracle_route_visits,
            "tie_refusal_count": tie_refusal_count,
            "budget_refusal_count": budget_refusal_count,
            "legality_skip_count": legality_skip_count,
            "registered_count_ladder": _count_ladder(
                policy.count_ladder,
                selectable_count=problem.selectable_count,
                maximum_removal_count=policy.maximum_removal_count,
            ),
            "maximum_removal_count": policy.maximum_removal_count,
            "legacy_dual_iterations_policy_field": policy.dual_iterations,
            "legacy_dual_initial_step_size_policy_field": policy.dual_initial_step_size,
            "legacy_dual_policy_fields_inert": True,
            "canonical_dual_iterations_used": 0,
            "canonical_selection_construction": (
                "registered_prefix_conditional_budget_gauge"
            ),
            "canonical_order_score": (
                "registered-prefix KKT-motivated remaining-budget stress: "
                "max(A_j/(remaining_F/slots), "
                "sqrt(max(0, ΔQ_cert(j|S))/(remaining_Q/slots)))/resource"
            ),
            "canonical_conditional_delta_q": (
                "ΔQ_cert(j|S)=||h_j||^2+2 h_j·h_S + "
                "min(rho*(|S|+1),(sum_sqrt_d_S+sqrt(d_j))^2) - "
                "min(rho*|S|,(sum_sqrt_d_S)^2)"
            ),
            "canonical_registered_prefix_expansions": True,
            "canonical_conditional_candidate_count": canonical_conditional_candidate_count,
            "canonical_conditional_solver_claim": (
                "approximate_registered_prefix_greedy_no_global_optimality_claim"
            ),
            "nonpositive_delta_q_repair_policy": (
                "registered_nonpositive_delta_q_repair_path_stress_no_negative_bonus_v1"
            ),
            "nonpositive_delta_q_repair_count": nonpositive_delta_q_repair_count,
            "nonpositive_delta_q_repair_success_count": (
                nonpositive_delta_q_repair_success_count
            ),
            "nonpositive_delta_q_repair_unavailable_count": (
                nonpositive_delta_q_repair_unavailable_count
            ),
            "nonpositive_delta_q_repair_unsuccessful_count": (
                nonpositive_delta_q_repair_unsuccessful_count
            ),
            "canonical_prefix_recomputation_count": (
                canonical_prefix_recomputation_count
            ),
            "remaining_f_budget_snap_count": remaining_f_budget_snap_count,
            "remaining_q_budget_snap_count": remaining_q_budget_snap_count,
            "diagnostic_a_only_prefix_archived": True,
            "diagnostic_a_only_prefix_construction": (
                "single_order_corrected_salience_primary_prefix"
            ),
            "diagnostic_a_only_prefix_evaluated": False,
            "canonical_fallback_to_a_only_prefix": False,
            "exact_reference_semantics": (
                "max_resource_subject_to_survivor_legality_F_budget_and_"
                "certified_Q_budget"
            ),
            "exact_reference_q_formula": (
                "Q_cert(S)=||sum_{j in S} h_j||^2 + "
                "min(rho*|S|,(sum_{j in S} sqrt(d_j))^2)"
            ),
            **_exact_reference_comparator_metadata(
                exact_candidates=tuple(exact_reference_candidates),
                scalable_candidates=frontier,
                subset_evaluation_count=exact_reference_subset_evaluation_count,
                refusals=exact_reference_refusals,
                route_limit=policy.exact_reference_route_limit,
                route_count=problem.route_count,
            ),
            "per_count_dual_solver_used": False,
            "input_array_policy": problem.input_array_policy,
            "signed_covector_present": problem.has_signed_covector,
            "admitted_array_payload_bytes": problem.admitted_array_payload_bytes,
            "candidate_mask_bytes_each": len(problem.parent_mask.packed_bits),
            "retains_removed_index_tuples": False,
            "selection_claim": (
                "exact_metrics_for_frozen_packed_frontier_no_global_powerset_"
                "or_fixed_count_optimality_or_lower_bound_claim"
            ),
        },
    )
    return V1ProductionGeometricSelectorResult(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        receipt=receipt,
        frontier=frontier,
    )


__all__ = [
    "LOSS_RESCALING_LAW",
    "PRODUCTION_SELECTOR_ATTRIBUTION_COORDINATES_SCHEMA",
    "PRODUCTION_SELECTOR_CANONICAL_POLICY_ID",
    "PRODUCTION_SELECTOR_CANONICAL_SCALARIZATION_ID",
    "PRODUCTION_SELECTOR_CHART_PROVENANCE_SCHEMA",
    "PRODUCTION_SELECTOR_COMPLETE_Q_ID",
    "PRODUCTION_SELECTOR_DIAGNOSTIC_ID",
    "PRODUCTION_SELECTOR_DIAGNOSTIC_SCALARIZATION_ID",
    "PRODUCTION_SELECTOR_ID",
    "PRODUCTION_SELECTOR_NO_PARTITION_POLICY_ID",
    "PRODUCTION_SELECTOR_SAMPLED_SNAPSHOT_FD_CERTIFICATE_KIND",
    "PRODUCTION_SELECTOR_SCALARIZATION_ID",
    "V1CorrectedAttributionChartCoordinates",
    "V1ProductionGeometricCandidate",
    "V1ProductionGeometricSelectorPolicy",
    "V1ProductionGeometricSelectorProblem",
    "V1ProductionGeometricSelectorReceipt",
    "V1ProductionGeometricSelectorResult",
    "V1ProductionLegalityBlock",
    "V1ProductionSelectorChartProvenance",
    "V1ProductionScalarization",
    "select_v1_production_geometric_frontier",
]
