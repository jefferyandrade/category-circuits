"""Scientific identity contracts and measurement compatibility exports for IRG.

The measurement transport objects re-exported here are the exact canonical
classes owned by ``mechanistic_systems.contracts.measurement``.  Their
historical ``irg-*`` identity prefixes grant no IRG workflow authority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.contracts.functional_atlas import FunctionalAtlasSpec
from mechanistic_systems.contracts.measurement import (
    ArtifactRef,
    ContextSelector,
    MeasurementKind,
    MeasurementOutcome,
    MeasurementProvenance,
    MeasurementRequest,
    replace_request_value,
)
from mechanistic_systems.contracts.measurement_costs import CostEstimate, RealizedCost
from mechanistic_systems.contracts.response_geometry import (
    ResponseFunctionalSpec,
    ResponseMetricSpec,
)
from mechanistic_systems.contracts.trust_geometry import (
    GlobalTrustGeometrySpec,
    ModelValidityDecision,
)


HistoricalResponseProfileName: TypeAlias = Literal["v0", "v1", "v2", "v3", "v4"]
IRG_VERSION_NAMESPACE = "irg"


def qualify_response_version(version: HistoricalResponseProfileName) -> str:
    """Return the repository-unambiguous qualified IRG version label."""

    return f"{IRG_VERSION_NAMESPACE}/{version}"


@dataclass(frozen=True, slots=True)
class ResponseBoundaryDirectionSpec:
    """One oriented facet of a frozen response-safe region.

    ``vector`` points toward increasing safety.  For a pruning-induced response
    displacement ``delta``, negative ``vector @ delta`` is adverse and the
    declared ``lower_tolerance`` is the maximum robust adverse displacement
    permitted along the normalized direction.
    """

    direction_id: str
    vector: tuple[float, ...]
    lower_tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.direction_id:
            raise ValueError("response boundary direction requires direction_id")
        vector = tuple(float(value) for value in self.vector)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("response boundary vector must be finite and nonempty")
        if float(np.linalg.norm(np.asarray(vector, dtype=np.float64))) <= 0.0:
            raise ValueError("response boundary vector must be nonzero")
        tolerance = float(self.lower_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                "response boundary lower_tolerance must be finite and positive"
            )
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "lower_tolerance", tolerance)
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="response-boundary-")


@dataclass(frozen=True, slots=True)
class ResponseConstraintSpec:
    """Frozen matrix-valued safe region for v1--v4 response geometry.

    The safe region is the intersection of oriented directional slabs and an
    optional quadratic response-energy tube.  Contextual variability, local
    model error, and estimator uncertainty receive separate frozen multipliers.
    A risk of one corresponds to the declared boundary.
    """

    constraint_id: str
    output_dimension: int
    directions: tuple[ResponseBoundaryDirectionSpec, ...] = ()
    contextual_robustness_multiplier: float = 0.0
    model_uncertainty_multiplier: float = 0.0
    estimation_uncertainty_multiplier: float = 0.0
    energy_metric: tuple[tuple[float, ...], ...] = ()
    energy_tolerance: float | None = None
    maximum_accepted_risk: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "matrix_valued_response_safe_region_v1"

    def __post_init__(self) -> None:
        if not self.constraint_id or not self.schema:
            raise ValueError("response constraint requires stable identity")
        dimension = int(self.output_dimension)
        if dimension <= 0:
            raise ValueError("response constraint output_dimension must be positive")
        directions = tuple(self.directions)
        if not directions and not self.energy_metric:
            raise ValueError(
                "response constraint requires at least one directional or energy boundary"
            )
        if len({item.direction_id for item in directions}) != len(directions):
            raise ValueError("response boundary direction IDs must be unique")
        if any(len(item.vector) != dimension for item in directions):
            raise ValueError("response boundary vectors must match output_dimension")
        multipliers = (
            float(self.contextual_robustness_multiplier),
            float(self.model_uncertainty_multiplier),
            float(self.estimation_uncertainty_multiplier),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in multipliers):
            raise ValueError(
                "response uncertainty multipliers must be finite and nonnegative"
            )
        maximum = float(self.maximum_accepted_risk)
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_accepted_risk must be finite and positive")
        metric_tuple: tuple[tuple[float, ...], ...]
        if self.energy_metric:
            metric = np.asarray(self.energy_metric, dtype=np.float64)
            if metric.shape != (dimension, dimension) or not np.isfinite(metric).all():
                raise ValueError(
                    "response energy metric must be a finite output_dimension square matrix"
                )
            metric = 0.5 * (metric + metric.T)
            eigenvalues = np.linalg.eigvalsh(metric)
            scale = (
                max(1.0, float(np.max(np.abs(eigenvalues))))
                if eigenvalues.size
                else 1.0
            )
            if eigenvalues.size and float(np.min(eigenvalues)) < -1e-8 * scale:
                raise ValueError("response energy metric must be positive semidefinite")
            tolerance = (
                float(self.energy_tolerance)
                if self.energy_tolerance is not None
                else float("nan")
            )
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    "response energy metric requires a finite positive energy_tolerance"
                )
            metric_tuple = tuple(tuple(float(value) for value in row) for row in metric)
            object.__setattr__(self, "energy_tolerance", tolerance)
        else:
            if self.energy_tolerance is not None:
                raise ValueError("energy_tolerance is only valid with an energy_metric")
            metric_tuple = ()
        object.__setattr__(self, "output_dimension", dimension)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "contextual_robustness_multiplier", multipliers[0])
        object.__setattr__(self, "model_uncertainty_multiplier", multipliers[1])
        object.__setattr__(self, "estimation_uncertainty_multiplier", multipliers[2])
        object.__setattr__(self, "energy_metric", metric_tuple)
        object.__setattr__(self, "maximum_accepted_risk", maximum)
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="response-safe-region-")

    @classmethod
    def target_rest_margin(
        cls,
        *,
        margin_tolerance: float,
        contextual_robustness_multiplier: float = 0.0,
        model_uncertainty_multiplier: float = 0.0,
        estimation_uncertainty_multiplier: float = 0.0,
        energy_tolerance: float | None = None,
        maximum_accepted_risk: float = 1.0,
        constraint_id: str = "target-vs-rest-margin-safe",
    ) -> "ResponseConstraintSpec":
        return cls(
            constraint_id=constraint_id,
            output_dimension=2,
            directions=(
                ResponseBoundaryDirectionSpec(
                    direction_id="target_over_aggregate_competitor",
                    vector=(1.0, -1.0),
                    lower_tolerance=float(margin_tolerance),
                ),
            ),
            contextual_robustness_multiplier=contextual_robustness_multiplier,
            model_uncertainty_multiplier=model_uncertainty_multiplier,
            estimation_uncertainty_multiplier=estimation_uncertainty_multiplier,
            energy_metric=(
                ((1.0, -1.0), (-1.0, 1.0)) if energy_tolerance is not None else ()
            ),
            energy_tolerance=energy_tolerance,
            maximum_accepted_risk=maximum_accepted_risk,
            metadata={"response_chart": "target_writing_plus_competitor_integration"},
        )

    @classmethod
    def competitor_margin_cell(
        cls,
        *,
        competitor_ids: Sequence[str],
        margin_tolerances: Sequence[float] | float,
        contextual_robustness_multiplier: float = 0.0,
        model_uncertainty_multiplier: float = 0.0,
        estimation_uncertainty_multiplier: float = 0.0,
        maximum_accepted_risk: float = 1.0,
        constraint_id: str = "competitor-resolved-margin-cell",
    ) -> "ResponseConstraintSpec":
        competitors = tuple(map(str, competitor_ids))
        if not competitors or len(set(competitors)) != len(competitors):
            raise ValueError("competitor margin cell requires unique competitor IDs")
        if isinstance(margin_tolerances, (int, float, np.number)):
            tolerances = (float(margin_tolerances),) * len(competitors)
        else:
            tolerances = tuple(float(value) for value in margin_tolerances)
        if len(tolerances) != len(competitors):
            raise ValueError("competitor margin tolerances must align with competitors")
        directions = []
        for index, (competitor, tolerance) in enumerate(
            zip(competitors, tolerances, strict=True)
        ):
            vector = [0.0] * len(competitors)
            vector[index] = 1.0
            directions.append(
                ResponseBoundaryDirectionSpec(
                    direction_id=f"target_over:{competitor}",
                    vector=tuple(vector),
                    lower_tolerance=tolerance,
                )
            )
        return cls(
            constraint_id=constraint_id,
            output_dimension=len(competitors),
            directions=tuple(directions),
            contextual_robustness_multiplier=contextual_robustness_multiplier,
            model_uncertainty_multiplier=model_uncertainty_multiplier,
            estimation_uncertainty_multiplier=estimation_uncertainty_multiplier,
            maximum_accepted_risk=maximum_accepted_risk,
            metadata={
                "competitor_ids": competitors,
                "response_chart": "competitor_resolved_contrasts",
            },
        )


@dataclass(frozen=True)
class CalibrationPolicySpec:
    policy_id: str
    mode: Literal["fixed", "state_specific", "profiled"]
    parameter_dimension: int = 0
    allow_warm_start: bool = False
    allow_exact_refit: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("calibration policy requires a policy_id")
        if int(self.parameter_dimension) < 0:
            raise ValueError("calibration parameter_dimension must be nonnegative")
        if self.mode == "fixed" and self.parameter_dimension < 0:
            raise ValueError("fixed calibration has invalid parameter dimension")
        object.__setattr__(self, "parameters", dict(self.parameters))
        canonical_json(self.parameters)


@dataclass(frozen=True)
class CostFunctionalSpec:
    cost_id: str
    primary_coordinate: str = "route_count"
    scalarization_weights: Mapping[str, float] = field(
        default_factory=lambda: {"route_count": 1.0}
    )

    def __post_init__(self) -> None:
        if not self.cost_id or not self.primary_coordinate:
            raise ValueError("cost functional requires cost_id and primary_coordinate")
        weights = {
            str(key): float(value) for key, value in self.scalarization_weights.items()
        }
        if not weights or any(
            not math.isfinite(value) or value < 0.0 for value in weights.values()
        ):
            raise ValueError(
                "cost scalarization weights must be finite, nonnegative, and nonempty"
            )
        object.__setattr__(self, "scalarization_weights", weights)


@dataclass(frozen=True)
class ContextDistributionSpec:
    context_axis_hash: str
    selector_id: str
    population_mode: Literal["fixed_benchmark", "sampled_population"] = (
        "fixed_benchmark"
    )
    strata: tuple[str, ...] = ()
    preprocessing_hash: str = ""

    def __post_init__(self) -> None:
        if not self.context_axis_hash or not self.selector_id:
            raise ValueError("context distribution requires axis hash and selector_id")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("context strata must be unique")


@dataclass(frozen=True)
class GeometricPartitionSpec:
    """One frozen route block with a common local response geometry.

    A geometric partition may span several physical layers.  Route scores and
    route-space covariance are comparable only inside this block.  The layer
    labels are retained separately so a covariance-small action cannot remove
    an excessive fraction of one layer before relinearization.
    """

    partition_id: str
    route_indices: tuple[int, ...]
    layer_ids: tuple[str, ...]
    score_functional_id: str
    maximum_covariance_leverage: float = 1.0
    maximum_layer_covariance_leverage: float | None = None
    maximum_partition_action_fraction: float = 1.0
    maximum_layer_action_fraction: float = 0.25
    minimum_layer_action_count: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_geometric_partition_v2"

    def __post_init__(self) -> None:
        if not self.partition_id or not self.score_functional_id or not self.schema:
            raise ValueError("geometric partition requires stable identity")
        indices = tuple(int(value) for value in self.route_indices)
        layers = tuple(str(value) for value in self.layer_ids)
        if (
            not indices
            or len(indices) != len(layers)
            or len(set(indices)) != len(indices)
            or any(value < 0 for value in indices)
            or any(not value for value in layers)
        ):
            raise ValueError(
                "geometric partition requires unique nonnegative routes and one layer label per route"
            )
        if indices != tuple(sorted(indices)):
            raise ValueError(
                "geometric partition route indices must use physical-axis order"
            )
        leverage = float(self.maximum_covariance_leverage)
        layer_leverage = (
            leverage
            if self.maximum_layer_covariance_leverage is None
            else float(self.maximum_layer_covariance_leverage)
        )
        partition_fraction = float(self.maximum_partition_action_fraction)
        layer_fraction = float(self.maximum_layer_action_fraction)
        if not math.isfinite(leverage) or leverage <= 0.0:
            raise ValueError("maximum_covariance_leverage must be finite and positive")
        if not math.isfinite(layer_leverage) or layer_leverage <= 0.0:
            raise ValueError(
                "maximum_layer_covariance_leverage must be finite and positive"
            )
        if (
            not math.isfinite(partition_fraction)
            or not 0.0 < partition_fraction <= 1.0
            or not math.isfinite(layer_fraction)
            or not 0.0 < layer_fraction <= 1.0
        ):
            raise ValueError("geometric action fractions must lie in (0,1]")
        minimum = int(self.minimum_layer_action_count)
        if minimum < 0:
            raise ValueError("minimum_layer_action_count must be nonnegative")
        object.__setattr__(self, "route_indices", indices)
        object.__setattr__(self, "layer_ids", layers)
        object.__setattr__(self, "maximum_covariance_leverage", leverage)
        object.__setattr__(
            self,
            "maximum_layer_covariance_leverage",
            layer_leverage,
        )
        object.__setattr__(
            self, "maximum_partition_action_fraction", partition_fraction
        )
        object.__setattr__(self, "maximum_layer_action_fraction", layer_fraction)
        object.__setattr__(self, "minimum_layer_action_count", minimum)
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-geometric-partition-")

    @property
    def layer_by_route_index(self) -> Mapping[int, str]:
        return dict(zip(self.route_indices, self.layer_ids, strict=True))


@dataclass(frozen=True)
class GeometricPartitioningSpec:
    """Disjoint complete partitioning used for v1--v5 proposal geometry.

    This is deliberately not v0's historical sign/retention partition.  It
    declares where a common route score and route-covariance chart is valid.
    Candidate comparisons across blocks must therefore occur only after
    mapping complete actions into the common response-risk and cost units.
    """

    partitioning_id: str
    route_axis_hash: str
    route_count: int
    partitions: tuple[GeometricPartitionSpec, ...]
    raw_score_comparison_policy: Literal["within_partition_only"] = (
        "within_partition_only"
    )
    cross_partition_covariance_policy: Literal["block_diagonal"] = "block_diagonal"
    candidate_scope: Literal["single_partition"] = "single_partition"
    coverage: Literal["complete_disjoint"] = "complete_disjoint"
    schema: str = "irg_geometric_partitioning_v1"

    def __post_init__(self) -> None:
        if not self.partitioning_id or not self.route_axis_hash or not self.schema:
            raise ValueError("geometric partitioning requires stable identity")
        route_count = int(self.route_count)
        partitions = tuple(self.partitions)
        if route_count <= 0 or not partitions:
            raise ValueError(
                "geometric partitioning requires routes and at least one partition"
            )
        if len({item.partition_id for item in partitions}) != len(partitions):
            raise ValueError("geometric partition IDs must be unique")
        if self.raw_score_comparison_policy != "within_partition_only":
            raise ValueError(
                "raw route scores may only be compared within a geometric partition"
            )
        if self.cross_partition_covariance_policy != "block_diagonal":
            raise ValueError(
                "cross-partition covariance is undefined and must remain block diagonal"
            )
        if (
            self.candidate_scope != "single_partition"
            or self.coverage != "complete_disjoint"
        ):
            raise ValueError(
                "current geometric execution requires complete disjoint single-partition actions"
            )
        flattened = tuple(index for item in partitions for index in item.route_indices)
        if len(flattened) != route_count or set(flattened) != set(range(route_count)):
            raise ValueError(
                "geometric partitions must cover every physical route exactly once"
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "partitions", partitions)

    @property
    def content_hash(self) -> str:
        # Runtime only, bitwise-identical: this spec is frozen, so its
        # content hash is immutable — but canonicalizing it walks all
        # ~1.17M route indices (profiled: 8.2M _canonicalize calls, ~25s
        # per proposal, the largest cost once membership is memoized). Memoize
        # per interned instance; the spec is retained beside its digest so
        # an id() collision after collection cannot return another spec's
        # hash.
        cached = _CONTENT_HASH_CACHE.get(id(self))
        if cached is not None and cached[0] is self:
            return cached[1]
        digest = content_hash(self, prefix="irg-geometric-partitioning-")
        if len(_CONTENT_HASH_CACHE) >= _ROUTE_MEMBERSHIP_CACHE_LIMIT:
            _CONTENT_HASH_CACHE.clear()
        _CONTENT_HASH_CACHE[id(self)] = (self, digest)
        return digest

    def partition_for_route(self, route_index: int) -> GeometricPartitionSpec:
        index = int(route_index)
        for partition in self._route_membership():
            if index in partition[1]:
                return partition[0]
        raise ValueError(
            f"route index {index} is absent from the geometric partitioning"
        )

    def _route_membership(
        self,
    ) -> tuple[tuple[GeometricPartitionSpec, frozenset[int]], ...]:
        """Per-partition O(1) membership sets, built once per spec.

        Runtime only, bitwise-identical: ``route_indices`` is a tuple, so
        ``index in partition.route_indices`` is a LINEAR scan.  Callers ask
        per removed route, which made partition lookup O(|removed| x n) —
        profiled at 380s of a 411s proposal (300,000 calls at ~1.27ms) and
        the single largest cost in the whole arm.  Frozensets make each
        lookup O(1); the sets are memoized per interned spec instance, so
        the build cost is paid once per chart rather than once per route.
        """

        cached = _ROUTE_MEMBERSHIP_CACHE.get(id(self))
        if cached is not None and cached[0] is self:
            return cached[1]
        membership = tuple(
            (partition, frozenset(int(v) for v in partition.route_indices))
            for partition in self.partitions
        )
        if len(_ROUTE_MEMBERSHIP_CACHE) >= _ROUTE_MEMBERSHIP_CACHE_LIMIT:
            _ROUTE_MEMBERSHIP_CACHE.clear()
        # The spec is retained alongside its sets so an id() collision after
        # garbage collection can never return another spec's membership.
        _ROUTE_MEMBERSHIP_CACHE[id(self)] = (self, membership)
        return membership


_ROUTE_MEMBERSHIP_CACHE: dict[int, tuple[Any, tuple]] = {}
_CONTENT_HASH_CACHE: dict[int, tuple[Any, str]] = {}
_ROUTE_MEMBERSHIP_CACHE_LIMIT = 32


@dataclass(frozen=True, slots=True)
class HardFunctionalPartitionSpec:
    """Hash-bound hard functional labels for modern v0/v1 execution.

    This contract is deliberately distinct from ``GeometricPartitioningSpec``.
    It identifies how labels are measured; it does not declare that covariance
    vanishes across the resulting blocks.
    """

    partitioning_id: str
    family: Literal[
        "historical_spatial_weight_sum_sign",
        "local_attribution_sign",
        "global_attribution_sign",
        "randomized_count_matched",
    ]
    label_functional_id: str
    measurement_state_hash: str
    refresh_scope: Literal["once", "relinearization_chart"]
    spatial_reducer: Literal["signed_sum_before_orientation"] = (
        "signed_sum_before_orientation"
    )
    zero_tie_rule: Literal["nonnegative"] = "nonnegative"
    zero_activity_fc_tie_rule: Literal["historical_weight_sign"] = (
        "historical_weight_sign"
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_hard_functional_partition_v1"

    def __post_init__(self) -> None:
        if (
            not self.partitioning_id
            or not self.label_functional_id
            or not self.measurement_state_hash
            or not self.schema
        ):
            raise ValueError("hard functional partition requires complete identity")
        if self.spatial_reducer != "signed_sum_before_orientation":
            raise ValueError(
                "functional partitioning must sum spatial contributions before sign"
            )
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-hard-functional-partition-")


@dataclass(frozen=True, slots=True)
class InteractionScheduleSpec:
    """Frozen evidence rules for causal interaction scheduling."""

    schedule_id: str
    probe_family_ids: tuple[str, ...]
    maximum_probe_count: int
    finite_dose_levels: tuple[float, ...] = ()
    interaction_threshold: float = 0.0
    uncertainty_threshold: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_interaction_schedule_spec_v1"

    def __post_init__(self) -> None:
        probes = tuple(map(str, self.probe_family_ids))
        doses = tuple(float(value) for value in self.finite_dose_levels)
        if (
            not self.schedule_id
            or not probes
            or len(set(probes)) != len(probes)
            or any(not value for value in probes)
            or int(self.maximum_probe_count) <= 0
        ):
            raise ValueError("interaction schedule requires valid probe identity")
        if any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 for value in doses
        ) or doses != tuple(sorted(set(doses))):
            raise ValueError(
                "finite interaction doses must be unique, sorted, and in (0,1]"
            )
        thresholds = (
            float(self.interaction_threshold),
            float(self.uncertainty_threshold),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
            raise ValueError("interaction thresholds must be finite and nonnegative")
        object.__setattr__(self, "probe_family_ids", probes)
        object.__setattr__(self, "maximum_probe_count", int(self.maximum_probe_count))
        object.__setattr__(self, "finite_dose_levels", doses)
        object.__setattr__(self, "interaction_threshold", thresholds[0])
        object.__setattr__(self, "uncertainty_threshold", thresholds[1])
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-interaction-schedule-spec-")


@dataclass(frozen=True)
class ScientificIdentity:
    """Immutable identity of the scientific pruning problem."""

    model_id: str
    model_state_hash: str
    route_axis_hash: str
    response_id: str
    metric: ResponseMetricSpec
    calibration: CalibrationPolicySpec
    cost: CostFunctionalSpec
    context: ContextDistributionSpec
    protected_route_ids: tuple[str, ...] = ()
    response_spec: ResponseFunctionalSpec | None = None
    response_constraint: ResponseConstraintSpec | None = None
    geometric_partitioning: GeometricPartitioningSpec | None = None
    hard_functional_partition: HardFunctionalPartitionSpec | None = None
    functional_atlas: FunctionalAtlasSpec | None = None
    global_trust_geometry: GlobalTrustGeometrySpec | None = None
    interaction_schedule: InteractionScheduleSpec | None = None
    schema: str = "interventional_response_geometry_identity_v3"

    def __post_init__(self) -> None:
        required = (
            self.model_id,
            self.model_state_hash,
            self.route_axis_hash,
            self.response_id,
            self.schema,
        )
        if any(not value for value in required):
            raise ValueError("scientific identity has an empty required field")
        if len(set(self.protected_route_ids)) != len(self.protected_route_ids):
            raise ValueError("protected route IDs must be unique")
        if self.response_spec is not None:
            if self.response_spec.response_id != self.response_id:
                raise ValueError("response_id must equal response_spec.response_id")
            if self.response_spec.output_dimension != self.metric.output_dimension:
                raise ValueError(
                    "response metric dimension differs from the response functional"
                )
        if self.response_constraint is not None:
            if (
                self.response_constraint.output_dimension
                != self.metric.output_dimension
            ):
                raise ValueError(
                    "response safe-region dimension differs from the response metric"
                )
        if self.geometric_partitioning is not None:
            if self.geometric_partitioning.route_axis_hash != self.route_axis_hash:
                raise ValueError(
                    "geometric partitioning belongs to a different physical route axis"
                )
        if (
            self.hard_functional_partition is not None
            and self.functional_atlas is not None
        ):
            raise ValueError(
                "one scientific identity may not bind both hard proxy partitions and a functional atlas"
            )
        if self.functional_atlas is not None and (
            self.global_trust_geometry is None or self.interaction_schedule is None
        ):
            raise ValueError(
                "v2-v4 functional atlases require separate global trust and interaction contracts"
            )

    @property
    def response_contract_status(self) -> str:
        if self.response_spec is None:
            return "legacy_untyped_response"
        if self.response_spec.is_contrast_complete:
            return "contrast_complete"
        return "derived_scalar_response"

    def assert_canonical_response_for_version(
        self, version: HistoricalResponseProfileName
    ) -> None:
        if version == "v0":
            if self.geometric_partitioning is not None:
                raise ValueError(
                    "v0 uses its immutable historical policy partitions, not v1-v5 geometric partitions"
                )
            if (
                self.response_spec is not None
                and self.response_spec.kind != "target_logit"
            ):
                raise ValueError(
                    "v0 replay is bound to the historical target-logit response"
                )
            return
        if self.response_spec is None or not self.response_spec.is_contrast_complete:
            raise ValueError(
                "claim-bearing v1-v4 pruning requires a frozen category-contrast or "
                "target-versus-rest channel response specification"
            )

    @property
    def identity_hash(self) -> str:
        if (
            self.hard_functional_partition is None
            and self.functional_atlas is None
            and self.global_trust_geometry is None
            and self.interaction_schedule is None
        ):
            # Preserve hashes for historical/legacy identities created before
            # the commensurability fields existed.
            payload = {
                name: getattr(self, name)
                for name in (
                    "model_id",
                    "model_state_hash",
                    "route_axis_hash",
                    "response_id",
                    "metric",
                    "calibration",
                    "cost",
                    "context",
                    "protected_route_ids",
                    "response_spec",
                    "response_constraint",
                    "geometric_partitioning",
                    "schema",
                )
            }
            return content_hash(payload, prefix="irg-id-")
        return content_hash(self, prefix="irg-id-")

    def assert_commensurability_identity_for_version(
        self, version: HistoricalResponseProfileName
    ) -> None:
        if version == "v0":
            return
        if version == "v1":
            if self.hard_functional_partition is None:
                raise ValueError(
                    "canonical v1 requires a hard functional partition contract"
                )
            if self.functional_atlas is not None:
                raise ValueError("v1 may not bind a v2-v4 functional atlas")
            return
        if self.functional_atlas is None:
            raise ValueError(f"canonical {version} requires a functional atlas")
        if self.hard_functional_partition is not None:
            raise ValueError("v2-v4 may not use hard proxy partitions as identity")

    def assert_compatible(self, other: "ScientificIdentity") -> None:
        if self.identity_hash != other.identity_hash:
            raise ValueError(
                "scientific identity mismatch: response, metric, calibration, route axis, "
                "context distribution, or cost functional changed"
            )


@dataclass(frozen=True)
class ImplementationFidelity:
    """Numerical controls that do not define the scientific version."""

    solver: str
    screen_size: int | None = None
    rank: int | None = None
    beam_width: int | None = None
    horizon: int | None = None
    tolerances: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.solver:
            raise ValueError("implementation fidelity requires a solver label")
        for name in ("screen_size", "rank", "beam_width", "horizon"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")
        tolerances = {str(key): float(value) for key, value in self.tolerances.items()}
        if any(
            not math.isfinite(value) or value < 0.0 for value in tolerances.values()
        ):
            raise ValueError("implementation tolerances must be finite and nonnegative")
        object.__setattr__(self, "tolerances", tolerances)


@dataclass(frozen=True)
class ResponseVersionSpec:
    version: HistoricalResponseProfileName
    information_rank: int
    retained_information: tuple[str, ...]
    lower_version: HistoricalResponseProfileName | None
    fidelity: ImplementationFidelity
    schema: str = "interventional_response_geometry_version_v1"

    def __post_init__(self) -> None:
        expected = {"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4}
        if (
            self.version not in expected
            or int(self.information_rank) != expected[self.version]
        ):
            raise ValueError(
                "version information_rank must match the canonical v0-v4 order"
            )
        # v0 is an immutable external control, not the first member of the
        # scientific information filtration.  v1 is therefore also minimal.
        if self.version in {"v0", "v1"}:
            if self.lower_version is not None:
                raise ValueError(f"{self.version} has no lower scientific version")
        else:
            required_lower = f"v{int(self.information_rank) - 1}"
            if self.lower_version != required_lower:
                raise ValueError(
                    f"{self.version} must declare lower_version={required_lower!r}"
                )
        if not self.retained_information:
            raise ValueError("response version must name its retained information")

    @property
    def qualified_version(self) -> str:
        return qualify_response_version(self.version)

    @property
    def spec_hash(self) -> str:
        return content_hash(self, prefix="irg-version-")


@dataclass(frozen=True)
class ProjectionReceipt:
    source_version: HistoricalResponseProfileName
    target_version: HistoricalResponseProfileName
    source_state_hash: str
    target_state_hash: str
    rule_id: str
    exact: bool
    numerical_tolerance: float = 0.0

    def __post_init__(self) -> None:
        order = {"v0": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4}
        if self.target_version == "v0":
            raise ValueError(
                "irg/v0 is an immutable historical control, not a scientific "
                "projection target"
            )
        if order[self.target_version] >= order[self.source_version]:
            raise ValueError("projection target must be a strictly lower version")
        if not self.rule_id or not self.source_state_hash or not self.target_state_hash:
            raise ValueError("projection receipt has an empty identity field")
        if (
            not math.isfinite(self.numerical_tolerance)
            or self.numerical_tolerance < 0.0
        ):
            raise ValueError(
                "projection numerical_tolerance must be finite and nonnegative"
            )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-projection-")


@dataclass(frozen=True)
class ActionPrediction:
    candidate_id: str
    version: HistoricalResponseProfileName
    predicted_quality_loss: float
    standard_error: float
    model_component: str
    state_hash: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.model_component or not self.state_hash:
            raise ValueError("action prediction has an empty identity field")
        if not math.isfinite(self.predicted_quality_loss):
            raise ValueError("predicted_quality_loss must be finite")
        if not math.isfinite(self.standard_error) or self.standard_error < 0.0:
            raise ValueError("prediction standard_error must be finite and nonnegative")
        object.__setattr__(self, "details", dict(self.details))
        canonical_json(self.details)


__all__ = [
    "ActionPrediction",
    "ArtifactRef",
    "CalibrationPolicySpec",
    "ContextDistributionSpec",
    "ContextSelector",
    "CostFunctionalSpec",
    "FunctionalAtlasSpec",
    "GeometricPartitionSpec",
    "GeometricPartitioningSpec",
    "GlobalTrustGeometrySpec",
    "HardFunctionalPartitionSpec",
    "ImplementationFidelity",
    "InteractionScheduleSpec",
    "MeasurementKind",
    "MeasurementOutcome",
    "MeasurementProvenance",
    "MeasurementRequest",
    "ModelValidityDecision",
    "ProjectionReceipt",
    "ResponseFunctionalSpec",
    "ResponseMetricSpec",
    "HistoricalResponseProfileName",
    "ResponseVersionSpec",
    "ScientificIdentity",
    "canonical_json",
    "content_hash",
    "replace_request_value",
]
