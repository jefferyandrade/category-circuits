"""Partition-local route geometry and relinearization safeguards.

The programme uses three unrelated notions that must not be conflated:

* v0 historical sign/retention partitions (immutable policy);
* context/class strata (statistical sampling);
* v0/v1 hard functional partitions (route-score comparability only).

This module is the legacy hard-chart compatibility implementation.  Its
partition-local covariance classes are not the canonical v1--v4 trust
geometry: the latter is a separate full-route factor retaining cross-partition
terms.  Layer-local removal limits are leverage guards, not covariance
partitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .candidates import ActionCandidate, HardMask
from .contracts import (
    GeometricPartitionSpec,
    GeometricPartitioningSpec,
    content_hash,
)


GEOMETRIC_PARTITION_FILTER_CACHE_KEY = "irg_active_geometric_partition_filter_v1"
COVARIANCE_RANK_EXPANSION_CACHE_KEY = "irg_covariance_rank_expansion_obligation_v1"

CanonicalSignPartitionFamily = Literal[
    "local_attribution_sign",
    "global_attribution_sign",
    "randomized_count_matched",
    "structural_sign",
    "attributional_sign",
    "randomized_2_partition",
]
CANONICAL_GLOBAL_SIGN_PARTITION_IDS = (
    "global_negative",
    "global_nonnegative",
)


@dataclass(frozen=True, slots=True)
class CanonicalTwoPartitionTrustPolicy:
    """Frozen relinearization limits shared by the two global sign blocks."""

    maximum_covariance_leverage: float = 1.0
    maximum_layer_covariance_leverage: float = 1.0
    maximum_partition_action_fraction: float = 1.0
    maximum_layer_action_fraction: float = 0.25
    minimum_layer_action_count: int = 1

    def __post_init__(self) -> None:
        # Reuse the authoritative partition contract for numerical validation.
        GeometricPartitionSpec(
            partition_id="__validation__",
            route_indices=(0,),
            layer_ids=("__validation__",),
            score_functional_id="__validation__",
            maximum_covariance_leverage=self.maximum_covariance_leverage,
            maximum_layer_covariance_leverage=(
                self.maximum_layer_covariance_leverage
            ),
            maximum_partition_action_fraction=(
                self.maximum_partition_action_fraction
            ),
            maximum_layer_action_fraction=self.maximum_layer_action_fraction,
            minimum_layer_action_count=self.minimum_layer_action_count,
        )


def canonical_global_two_sign_partitioning(
    *,
    route_axis_hash: str,
    layer_ids: Sequence[str],
    sign_values: Sequence[float | int],
    partition_family: CanonicalSignPartitionFamily = "structural_sign",
    trust: CanonicalTwoPartitionTrustPolicy = CanonicalTwoPartitionTrustPolicy(),
    random_seed: int | None = None,
    source_geometry_hash: str | None = None,
) -> GeometricPartitioningSpec:
    """Build the canonical two cross-layer, within-sign geometry blocks.

    Zero belongs deterministically to the nonnegative block, matching the
    historical attribution-sign tie convention.  A missing sign block is a
    refusal: the canonical path never invents an arbitrary split merely to
    satisfy the two-partition interface.  Randomized labels are accepted only
    under the explicitly exploratory ablation identity.
    """

    layers = tuple(str(value) for value in layer_ids)
    signs = np.asarray(tuple(sign_values), dtype=np.float64).reshape(-1)
    if (
        not str(route_axis_hash)
        or not layers
        or len(layers) != signs.size
        or any(not value for value in layers)
        or not np.isfinite(signs).all()
    ):
        raise ValueError(
            "canonical two-partition geometry requires one finite sign and "
            "physical layer identity per route"
        )
    aliases = {
        "structural_sign": "local_attribution_sign",
        "attributional_sign": "global_attribution_sign",
        "randomized_2_partition": "randomized_count_matched",
    }
    canonical_family = aliases.get(str(partition_family), str(partition_family))
    if canonical_family not in {
        "local_attribution_sign",
        "global_attribution_sign",
        "randomized_count_matched",
    }:
        raise ValueError("unsupported canonical sign partition family")
    if canonical_family == "randomized_count_matched" and random_seed is None:
        raise ValueError(
            "randomized two-partition ablations require an explicit frozen seed"
        )
    if canonical_family != "randomized_count_matched" and random_seed is not None:
        raise ValueError("nonrandom sign partitions may not carry a random seed")

    sign_groups = (
        tuple(int(index) for index in np.flatnonzero(signs < 0.0)),
        tuple(int(index) for index in np.flatnonzero(signs >= 0.0)),
    )
    if any(not indices for indices in sign_groups):
        raise ValueError(
            "canonical global sign geometry requires both negative and "
            "nonnegative routes; no arbitrary fallback split is permitted"
        )
    if canonical_family == "randomized_count_matched":
        # The control changes membership while preserving the exact two-block
        # cardinalities of the sign partition.  Merely relabelling the original
        # sign groups would not be a randomized-partition ablation.
        permutation = np.random.default_rng(int(random_seed)).permutation(signs.size)
        split = len(sign_groups[0])
        route_groups = (
            tuple(sorted(int(index) for index in permutation[:split])),
            tuple(sorted(int(index) for index in permutation[split:])),
        )
    else:
        route_groups = sign_groups
    score_functional = {
        "local_attribution_sign": (
            "signed_spatial_sum_immediate_destination_context_mean_v2"
        ),
        "global_attribution_sign": (
            "signed_spatial_sum_global_response_context_mean_v2"
        ),
        "randomized_count_matched": (
            "exploratory_count_matched_random_block_score_v2"
        ),
    }[canonical_family]
    partitions = tuple(
        GeometricPartitionSpec(
            partition_id=partition_id,
            route_indices=indices,
            layer_ids=tuple(layers[index] for index in indices),
            score_functional_id=score_functional,
            maximum_covariance_leverage=trust.maximum_covariance_leverage,
            maximum_layer_covariance_leverage=(
                trust.maximum_layer_covariance_leverage
            ),
            maximum_partition_action_fraction=(
                trust.maximum_partition_action_fraction
            ),
            maximum_layer_action_fraction=trust.maximum_layer_action_fraction,
            minimum_layer_action_count=trust.minimum_layer_action_count,
            metadata={
                "canonical_two_partition": True,
                "partition_family": canonical_family,
                "legacy_partition_family_alias": (
                    None
                    if canonical_family == partition_family
                    else partition_family
                ),
                "sign_class": (
                    "negative"
                    if partition_id == "global_negative"
                    else "nonnegative"
                ),
                "zero_sign_policy": "nonnegative",
                "cross_layer_within_partition": True,
                "cross_partition_score_comparison": False,
                "random_seed": random_seed,
                "count_matched_to_sign_partition": (
                    canonical_family == "randomized_count_matched"
                ),
                "randomization_algorithm": (
                    "numpy_pcg64_permutation_v1"
                    if canonical_family == "randomized_count_matched"
                    else None
                ),
                "claim_status": (
                    "exploratory_ablation"
                    if canonical_family == "randomized_count_matched"
                    else "canonical"
                ),
                "source_geometry_hash": source_geometry_hash,
                "structural_sign_semantics": (
                    (
                        "local_attribution_not_weight_sign"
                        if source_geometry_hash is not None
                        else "unmeasured_bootstrap_placeholder"
                    )
                    if canonical_family == "local_attribution_sign"
                    else None
                ),
                "partition_membership_status": (
                    "measured_geometry"
                    if source_geometry_hash is not None
                    else (
                        "bootstrap_not_decision_eligible"
                        if canonical_family
                        in {"local_attribution_sign", "randomized_count_matched"}
                        else "frozen_supplied_geometry"
                    )
                ),
            },
        )
        for partition_id, indices in zip(
            CANONICAL_GLOBAL_SIGN_PARTITION_IDS,
            route_groups,
            strict=True,
        )
    )
    return GeometricPartitioningSpec(
        partitioning_id=(
            f"irg_global_two_partition_{canonical_family}_v2"
            + (
                ""
                if random_seed is None
                else f"_seed_{int(random_seed)}"
            )
        ),
        route_axis_hash=str(route_axis_hash),
        route_count=len(layers),
        partitions=partitions,
    )


def assert_canonical_global_two_partitioning(
    partitioning: GeometricPartitioningSpec,
) -> None:
    """Fail closed unless a partitioning is the canonical two-block form."""

    if (
        len(partitioning.partitions) != 2
        or tuple(item.partition_id for item in partitioning.partitions)
        != CANONICAL_GLOBAL_SIGN_PARTITION_IDS
        or any(
            item.metadata.get("canonical_two_partition") is not True
            for item in partitioning.partitions
        )
    ):
        raise ValueError(
            "canonical v1/v2 execution requires exactly the global negative "
            "and global nonnegative sign partitions"
        )


@dataclass(frozen=True)
class CovarianceRankExpansionObligation:
    """Decision-relevant omitted-route obligation for one local chart."""

    partitioning_hash: str
    partition_id: str
    measured_mask_hash: str
    source_state_hash: str
    current_factor_indices: tuple[int, ...]
    missing_route_indices: tuple[int, ...]
    required_route_indices: tuple[int, ...]
    source_factor_hash: str
    response_projection_residual_bound_per_unit_route_coefficient: float
    sketch_residual_spectral_bound: float
    reason: str
    frontier_coverage_prerequisite: bool = False
    schema: str = "irg_covariance_rank_expansion_obligation_v2"

    def __post_init__(self) -> None:
        current = tuple(int(value) for value in self.current_factor_indices)
        missing = tuple(int(value) for value in self.missing_route_indices)
        required = tuple(int(value) for value in self.required_route_indices)
        if (
            not self.partitioning_hash
            or not self.partition_id
            or not self.measured_mask_hash
            or not self.source_state_hash
            or not self.source_factor_hash
            or not self.reason
            or not self.schema
        ):
            raise ValueError("covariance expansion obligation lacks identity")
        if (
            not missing
            or len(set(current)) != len(current)
            or len(set(missing)) != len(missing)
            or len(set(required)) != len(required)
            or not set(current).issubset(required)
            or not set(missing).issubset(required)
        ):
            raise ValueError("covariance expansion route axes are invalid")
        projection_residual = float(
            self.response_projection_residual_bound_per_unit_route_coefficient
        )
        sketch_residual = float(self.sketch_residual_spectral_bound)
        if (
            not math.isfinite(projection_residual)
            or projection_residual < 0.0
            or not math.isfinite(sketch_residual)
            or sketch_residual < 0.0
        ):
            raise ValueError("covariance expansion residual bounds are invalid")
        object.__setattr__(self, "current_factor_indices", current)
        object.__setattr__(self, "missing_route_indices", missing)
        object.__setattr__(self, "required_route_indices", required)
        object.__setattr__(
            self,
            "response_projection_residual_bound_per_unit_route_coefficient",
            projection_residual,
        )
        object.__setattr__(
            self,
            "sketch_residual_spectral_bound",
            sketch_residual,
        )
        object.__setattr__(
            self,
            "frontier_coverage_prerequisite",
            bool(self.frontier_coverage_prerequisite),
        )

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-covariance-expansion-")


class UnresolvedCovarianceObligation(RuntimeError):
    def __init__(self, obligation: CovarianceRankExpansionObligation) -> None:
        self.obligation = obligation
        super().__init__(obligation.reason)


def _readonly_psd(matrix: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    value = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0)
    if eigenvalues.size and float(np.min(eigenvalues)) < -1e-8 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    result = np.ascontiguousarray(0.5 * (result + result.T))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PartitionCovarianceGeometry:
    """Measured route-space geometry for one declared partition.

    ``route_covariance`` is either a diagonal vector or a small dense base
    matrix.  ``covariance_factor`` optionally represents the full within-block
    off-diagonal term as ``F.T @ F``.  This avoids an O(R^2) materialization for
    the two cross-layer ResNet partitions while preserving exact empirical
    quadratic forms.  No field exists for cross-partition entries.  Spectral
    uncertainty and sketch residual bounds conservatively inflate the
    quadratic action leverage.  Response projection error has different
    units: it is added to the resulting response norm after scaling by the
    action's route-coefficient L1 norm.
    """

    partitioning_hash: str
    partition_id: str
    route_indices: tuple[int, ...]
    layer_ids: tuple[str, ...]
    route_covariance: np.ndarray = field(repr=False, compare=False)
    covariance_uncertainty_spectral_bound: float
    sketch_residual_spectral_bound: float
    sample_count: int
    measured_mask_hash: str
    source_state_hash: str
    approximation: str
    response_projection_residual_bound_per_unit_route_coefficient: float = 0.0
    covariance_factor: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_partition_covariance_geometry_v3"

    def __post_init__(self) -> None:
        indices = tuple(int(value) for value in self.route_indices)
        layers = tuple(str(value) for value in self.layer_ids)
        if (
            not self.partitioning_hash
            or not self.partition_id
            or not self.measured_mask_hash
            or not self.source_state_hash
            or not self.approximation
            or not self.schema
        ):
            raise ValueError("partition covariance geometry requires complete identity")
        if (
            not indices
            or len(indices) != len(layers)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                "partition covariance geometry has an invalid route/layer axis"
            )
        raw_covariance = np.asarray(self.route_covariance, dtype=np.float64)
        if raw_covariance.ndim == 1:
            if (
                raw_covariance.shape != (len(indices),)
                or not np.isfinite(raw_covariance).all()
                or np.any(raw_covariance < 0.0)
            ):
                raise ValueError(
                    "partition diagonal covariance does not align with its route axis"
                )
            covariance = np.ascontiguousarray(raw_covariance)
            covariance.setflags(write=False)
        else:
            covariance = _readonly_psd(
                raw_covariance,
                name=f"partition {self.partition_id} route covariance",
            )
            if covariance.shape != (len(indices), len(indices)):
                raise ValueError(
                    "partition covariance does not align with its route axis"
                )
        factor: np.ndarray | None = None
        if self.covariance_factor is not None:
            factor_value = np.asarray(self.covariance_factor, dtype=np.float64)
            if (
                factor_value.ndim != 2
                or factor_value.shape[0] <= 0
                or factor_value.shape[1] != len(indices)
                or not np.isfinite(factor_value).all()
            ):
                raise ValueError(
                    "partition covariance factor must be finite [rank, route]"
                )
            factor = np.ascontiguousarray(factor_value)
            factor.setflags(write=False)
        uncertainty = float(self.covariance_uncertainty_spectral_bound)
        residual = float(self.sketch_residual_spectral_bound)
        projection_residual = float(
            self.response_projection_residual_bound_per_unit_route_coefficient
        )
        if (
            not math.isfinite(uncertainty)
            or uncertainty < 0.0
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(projection_residual)
            or projection_residual < 0.0
            or int(self.sample_count) <= 0
        ):
            raise ValueError("partition covariance uncertainty metadata is invalid")
        object.__setattr__(self, "route_indices", indices)
        object.__setattr__(self, "layer_ids", layers)
        object.__setattr__(self, "route_covariance", covariance)
        object.__setattr__(self, "covariance_factor", factor)
        object.__setattr__(self, "covariance_uncertainty_spectral_bound", uncertainty)
        object.__setattr__(self, "sketch_residual_spectral_bound", residual)
        object.__setattr__(
            self,
            "response_projection_residual_bound_per_unit_route_coefficient",
            projection_residual,
        )
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "partitioning_hash": self.partitioning_hash,
                "partition_id": self.partition_id,
                "route_indices": self.route_indices,
                "layer_ids": self.layer_ids,
                "route_covariance": self.route_covariance,
                "covariance_factor": self.covariance_factor,
                "covariance_uncertainty_spectral_bound": self.covariance_uncertainty_spectral_bound,
                "sketch_residual_spectral_bound": self.sketch_residual_spectral_bound,
                "response_projection_residual_bound_per_unit_route_coefficient": (
                    self.response_projection_residual_bound_per_unit_route_coefficient
                ),
                "sample_count": self.sample_count,
                "measured_mask_hash": self.measured_mask_hash,
                "source_state_hash": self.source_state_hash,
                "approximation": self.approximation,
                "metadata": self.metadata,
            },
            prefix="irg-partition-covariance-",
        )

    @property
    def inflated_covariance(self) -> np.ndarray:
        if len(self.route_indices) > 4096:
            raise RuntimeError(
                "refusing to materialize a large partition covariance; use "
                "quadratic_form for implicit trust-region evaluation"
            )
        inflation = (
            self.covariance_uncertainty_spectral_bound
            + self.sketch_residual_spectral_bound
        )
        if self.route_covariance.ndim == 1:
            covariance = np.diag(self.route_covariance)
        else:
            covariance = self.route_covariance
        if self.covariance_factor is not None:
            covariance = covariance + self.covariance_factor.T @ self.covariance_factor
        return covariance + inflation * np.eye(len(self.route_indices))

    @property
    def diagonal_only(self) -> bool:
        return self.route_covariance.ndim == 1 and self.covariance_factor is None

    def covariance_entry(self, row: int, column: int) -> float:
        """Return one empirical covariance entry without dense materialization."""

        i = int(row)
        j = int(column)
        if not 0 <= i < len(self.route_indices) or not 0 <= j < len(
            self.route_indices
        ):
            raise IndexError("partition covariance entry is outside the route axis")
        if self.route_covariance.ndim == 1:
            value = float(self.route_covariance[i]) if i == j else 0.0
        else:
            value = float(self.route_covariance[i, j])
        if self.covariance_factor is not None:
            value += float(
                self.covariance_factor[:, i] @ self.covariance_factor[:, j]
            )
        return value

    def quadratic_form(
        self,
        coefficients: Sequence[float] | np.ndarray,
        *,
        local_offsets: Sequence[int] | np.ndarray | None = None,
    ) -> float:
        """Evaluate the uncertainty-inflated covariance action energy."""

        if local_offsets is None:
            offsets = np.arange(len(self.route_indices), dtype=np.int64)
        else:
            offsets = np.asarray(local_offsets, dtype=np.int64).reshape(-1)
            if (
                np.any(offsets < 0)
                or np.any(offsets >= len(self.route_indices))
                or np.unique(offsets).size != offsets.size
            ):
                raise ValueError("partition covariance offsets are invalid")
        vector = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if vector.shape != (offsets.size,) or not np.isfinite(vector).all():
            raise ValueError("partition covariance coefficients are invalid")
        if self.route_covariance.ndim == 1:
            value = float(
                np.sum(self.route_covariance[offsets] * vector * vector)
            )
        else:
            local_covariance = self.route_covariance[np.ix_(offsets, offsets)]
            value = float(vector @ local_covariance @ vector)
        if self.covariance_factor is not None:
            projected = self.covariance_factor[:, offsets] @ vector
            value += float(projected @ projected)
        inflation = (
            self.covariance_uncertainty_spectral_bound
            + self.sketch_residual_spectral_bound
        )
        value += float(inflation * (vector @ vector))
        return max(0.0, value)


@dataclass(frozen=True)
class PartitionActionAssessment:
    action: str
    reason: str
    partition_id: str | None
    covariance_leverage: float
    covariance_limit: float
    partition_removed: int
    partition_removed_limit: int
    layer_removed: Mapping[str, int]
    layer_removed_limits: Mapping[str, int]
    layer_covariance_leverage: Mapping[str, float] = field(default_factory=dict)
    layer_covariance_limits: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action == "reuse"


def candidate_partition(
    candidate: ActionCandidate,
    *,
    parent_mask: HardMask,
    partitioning: GeometricPartitioningSpec,
) -> GeometricPartitionSpec:
    removed = tuple(
        int(value) for value in candidate.resolve_removed_indices(parent_mask)
    )
    if not removed:
        raise ValueError("a pruning candidate must remove at least one route")
    partition_ids = {
        partitioning.partition_for_route(index).partition_id for index in removed
    }
    if len(partition_ids) != 1:
        raise ValueError(
            "one candidate may not compare or combine raw route scores across geometric partitions"
        )
    partition_id = next(iter(partition_ids))
    return next(
        item for item in partitioning.partitions if item.partition_id == partition_id
    )


def _context_gram_uncertainty(rows: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a small dense route Gram and a guidance uncertainty bound."""

    value = np.asarray(rows, dtype=np.float64)
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[0] <= 0 or not np.isfinite(value).all():
        raise ValueError(
            "partition factor rows must be finite [context,response,route]"
        )
    per_context = np.einsum("ndr,nds->nrs", value, value, optimize=True)
    gram = np.mean(per_context, axis=0)
    if value.shape[0] < 2:
        uncertainty = float("inf")
    else:
        centered = per_context - gram[None, :, :]
        # The Frobenius standard error is an upper bound on the spectral
        # standard error.  It is deliberately allocation/trust guidance, not a
        # claim-bearing confidence bound.
        uncertainty = float(
            np.sqrt(np.sum(centered * centered) / (value.shape[0] - 1))
            / math.sqrt(value.shape[0])
        )
    return _readonly_psd(gram, name="empirical partition route Gram"), uncertainty


def _context_energy_factor(rows: np.ndarray) -> tuple[np.ndarray, float]:
    """Return an implicit full covariance factor and Frobenius-SE upper bound.

    If ``A_i`` is the response-by-route row for context ``i``, the returned
    factor is the vertical stack of ``A_i / sqrt(n)``.  Its Gram is therefore
    exactly ``mean_i A_i.T @ A_i``, including every within-partition
    off-diagonal entry.  The uncertainty calculation drops the nonnegative
    centering term from the empirical Frobenius variance, yielding a
    conservative O(n d^2 + n d R) upper bound without an O(R^2) matrix.
    """

    value = np.asarray(rows, dtype=np.float64)
    if value.ndim == 2:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[0] <= 0 or not np.isfinite(value).all():
        raise ValueError(
            "partition factor rows must be finite [context,response,route]"
        )
    contexts, response_dimension, route_count = value.shape
    factor = np.ascontiguousarray(
        value.reshape(contexts * response_dimension, route_count)
        / math.sqrt(float(contexts))
    )
    factor.setflags(write=False)
    if contexts < 2:
        return factor, float("inf")
    per_context_frobenius_squared = np.empty(contexts, dtype=np.float64)
    for index in range(contexts):
        response_gram = value[index] @ value[index].T
        per_context_frobenius_squared[index] = float(
            np.sum(response_gram * response_gram)
        )
    uncertainty = float(
        math.sqrt(
            float(np.sum(per_context_frobenius_squared))
            / float(contexts - 1)
        )
        / math.sqrt(float(contexts))
    )
    return factor, uncertainty


def derive_partition_covariance_geometry(
    *,
    partitioning: GeometricPartitioningSpec,
    partition: GeometricPartitionSpec,
    version_state: object,
    current_mask_hash: str,
) -> PartitionCovarianceGeometry:
    """Project the active v1--v4 response state into one partition-local chart."""

    if partitioning.content_hash == "" or partition not in partitioning.partitions:
        raise ValueError("partition is not part of the declared partitioning")
    state = getattr(version_state, "intrinsic_state", version_state)
    measured_mask_hash = str(getattr(state, "measured_mask_hash", ""))
    if measured_mask_hash != str(current_mask_hash):
        raise RuntimeError(
            "partition covariance geometry is stale for the current mask"
        )
    indices = np.asarray(partition.route_indices, dtype=np.int64)
    sample_count = int(getattr(state, "sample_count", 0))
    covariance: np.ndarray
    covariance_factor: np.ndarray | None = None
    uncertainty: float
    response_projection_residual = 0.0
    sketch_residual = 0.0
    approximation: str

    intrinsic_factor = getattr(state, "intrinsic_response_factor", None)
    if callable(intrinsic_factor):
        intrinsic_factor = intrinsic_factor()
    response_factor = (
        intrinsic_factor
        if intrinsic_factor is not None
        else getattr(state, "response_factor", None)
    )
    factor_indices = tuple(
        int(value) for value in getattr(state, "response_factor_indices", ())
    )
    if response_factor is not None:
        lookup = {
            global_index: local_index
            for local_index, global_index in enumerate(factor_indices)
        }
        missing = [int(value) for value in indices if int(value) not in lookup]
        if missing:
            factor_hash = str(
                getattr(response_factor, "content_hash", "")
                or content_hash(repr(response_factor), prefix="opaque-factor-")
            )
            factor_metadata = dict(getattr(response_factor, "metadata", {}))
            response_projection_residual = max(
                float(
                    getattr(
                        response_factor,
                        "projection_residual_bound",
                        0.0,
                    )
                ),
                float(factor_metadata.get("projection_residual_bound", 0.0)),
            )
            sketch_residual = max(
                float(factor_metadata.get("sketch_residual_spectral_bound", 0.0)),
                float(factor_metadata.get("omitted_route_residual_bound", 0.0)),
            )
            raise UnresolvedCovarianceObligation(
                CovarianceRankExpansionObligation(
                    partitioning_hash=partitioning.content_hash,
                    partition_id=partition.partition_id,
                    measured_mask_hash=measured_mask_hash,
                    source_state_hash=str(
                        getattr(version_state, "state_hash", "")
                        or content_hash(
                            repr(version_state),
                            prefix="opaque-version-state-",
                        )
                    ),
                    current_factor_indices=factor_indices,
                    missing_route_indices=tuple(missing),
                    required_route_indices=tuple(
                        sorted(set(factor_indices) | set(partition.route_indices))
                    ),
                    source_factor_hash=factor_hash,
                    response_projection_residual_bound_per_unit_route_coefficient=(
                        response_projection_residual
                    ),
                    sketch_residual_spectral_bound=sketch_residual,
                    reason=(
                        "partition covariance has an unresolved omitted-route "
                        "obligation: " + ",".join(map(str, missing[:8]))
                    ),
                )
            )
        local = np.asarray([lookup[int(value)] for value in indices], dtype=np.int64)
        rows = np.asarray(response_factor.rows, dtype=np.float64)[:, :, local]
        covariance_factor, uncertainty = _context_energy_factor(rows)
        covariance = np.zeros(indices.size, dtype=np.float64)
        sample_count = int(response_factor.sample_count)
        metadata = dict(getattr(response_factor, "metadata", {}))
        response_projection_residual = max(
            float(getattr(response_factor, "projection_residual_bound", 0.0)),
            float(metadata.get("projection_residual_bound", 0.0)),
        )
        sketch_residual = max(
            float(metadata.get("sketch_residual_spectral_bound", 0.0)),
            float(metadata.get("omitted_route_residual_bound", 0.0)),
        )
        approximation = (
            "energy_factored_full_cross_layer_within_partition_response_covariance"
        )
    elif hasattr(state, "intrinsic_submatrix"):
        covariance = _readonly_psd(
            state.intrinsic_submatrix(indices),
            name="intrinsic partition route covariance",
        )
        rows = np.asarray(getattr(state, "route_factor"), dtype=np.float64)[:, indices]
        _unused, uncertainty = _context_gram_uncertainty(rows)
        approximation = "cross_layer_within_partition_intrinsic_factor"
    elif hasattr(state, "metric_for_indices"):
        rows = np.asarray(getattr(state, "jacobian_factor"), dtype=np.float64)[
            :, indices
        ]
        covariance_factor, uncertainty = _context_energy_factor(rows)
        covariance = np.zeros(indices.size, dtype=np.float64)
        approximation = (
            "energy_factored_full_cross_layer_within_partition_coupled_covariance"
        )
    elif hasattr(state, "diagonal_energy"):
        diagonal = np.asarray(getattr(state, "diagonal_energy"), dtype=np.float64)[
            indices
        ]
        covariance = np.ascontiguousarray(np.maximum(diagonal, 0.0))
        covariance.setflags(write=False)
        errors = np.asarray(
            getattr(state, "energy_standard_error", np.zeros_like(diagonal)),
            dtype=np.float64,
        )[indices]
        uncertainty = float(np.max(np.maximum(errors, 0.0), initial=0.0))
        approximation = "v1_diagonal_projection_of_partition_geometry"
    else:
        raise RuntimeError(
            "active response version cannot expose route covariance geometry"
        )

    if not math.isfinite(uncertainty):
        raise RuntimeError(
            "partition covariance is unresolved: at least two response rows are required"
        )
    if sample_count <= 0:
        sample_count = int(np.asarray(getattr(state, "route_factor", ())).shape[0])
    if sample_count <= 0:
        raise RuntimeError("partition covariance has no physical response rows")
    state_hash = getattr(version_state, "state_hash", None)
    if callable(state_hash):
        state_hash = state_hash()
    if not state_hash:
        state_hash = content_hash(repr(version_state), prefix="opaque-version-state-")
    return PartitionCovarianceGeometry(
        partitioning_hash=partitioning.content_hash,
        partition_id=partition.partition_id,
        route_indices=partition.route_indices,
        layer_ids=partition.layer_ids,
        route_covariance=covariance,
        covariance_factor=covariance_factor,
        covariance_uncertainty_spectral_bound=uncertainty,
        sketch_residual_spectral_bound=sketch_residual,
        sample_count=sample_count,
        measured_mask_hash=measured_mask_hash,
        source_state_hash=str(state_hash),
        approximation=approximation,
        response_projection_residual_bound_per_unit_route_coefficient=(
            response_projection_residual
        ),
        metadata={
            "cross_layer_covariance_retained": approximation.startswith(
                ("dense_cross_layer", "cross_layer", "energy_factored_full")
            ),
            "cross_partition_covariance_retained": False,
            "dense_route_covariance_materialized": covariance.ndim == 2,
            "implicit_covariance_factor_rank": (
                0
                if covariance_factor is None
                else int(covariance_factor.shape[0])
            ),
            "energy_based_full_covariance": covariance_factor is not None,
            "guidance_only": True,
            "exact_hard_mask_evidence_required_for_commit": True,
        },
    )


@dataclass(frozen=True)
class PartitionedCovarianceTrustRegion:
    """Candidate-scale and physical-layer gate for partition-local geometry.

    Covariance displacement is accumulated only inside the candidate's
    declared geometric partition.  The action-fraction guard is different:
    it counts every route removed from each affected physical layer since the
    shared relinearization anchor, including removals made through other
    geometric partitions.
    """

    def assess(
        self,
        candidate: ActionCandidate,
        *,
        parent_mask: HardMask,
        partitioning: GeometricPartitioningSpec,
        geometry: PartitionCovarianceGeometry,
        action_scale: float = 1.0,
        geometry_anchor_mask_hash: str | None = None,
        cumulative_removed_indices: Sequence[int] = (),
    ) -> PartitionActionAssessment:
        scale = float(action_scale)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("partition trust action_scale must lie in (0,1]")
        partition = candidate_partition(
            candidate,
            parent_mask=parent_mask,
            partitioning=partitioning,
        )
        if (
            geometry.partitioning_hash != partitioning.content_hash
            or geometry.partition_id != partition.partition_id
            or geometry.route_indices != partition.route_indices
            or geometry.layer_ids != partition.layer_ids
        ):
            raise ValueError(
                "candidate and covariance geometry use different partitions"
            )
        anchor_hash = str(geometry_anchor_mask_hash or parent_mask.mask_hash)
        if geometry.measured_mask_hash != anchor_hash:
            return PartitionActionAssessment(
                action="refresh",
                reason="partition covariance geometry is stale for the declared chart anchor",
                partition_id=partition.partition_id,
                covariance_leverage=float("inf"),
                covariance_limit=partition.maximum_covariance_leverage * scale,
                partition_removed=candidate.removed_count,
                partition_removed_limit=0,
                layer_removed={},
                layer_removed_limits={},
                layer_covariance_leverage={},
                layer_covariance_limits={},
                diagnostics={"geometry_hash": geometry.content_hash},
            )

        new_removed = tuple(
            int(value) for value in candidate.resolve_removed_indices(parent_mask)
        )
        cumulative = tuple(sorted({int(value) for value in cumulative_removed_indices}))
        if any(value < 0 or value >= parent_mask.route_count for value in cumulative):
            raise ValueError(
                "cumulative geometry action contains an invalid route index"
            )
        partition_routes = set(partition.route_indices)
        cumulative_partition = tuple(
            value for value in cumulative if value in partition_routes
        )
        if set(new_removed).intersection(cumulative):
            raise ValueError(
                "candidate repeats a route already removed since relinearization"
            )
        removed = tuple(sorted((*cumulative_partition, *new_removed)))
        local = {index: offset for offset, index in enumerate(partition.route_indices)}
        coefficients = np.zeros(len(partition.route_indices), dtype=np.float64)
        for index in removed:
            coefficients[local[index]] = -1.0
        quadratic_leverage_squared = geometry.quadratic_form(coefficients)
        quadratic_leverage = math.sqrt(max(0.0, quadratic_leverage_squared))
        response_projection_allowance = float(
            geometry.response_projection_residual_bound_per_unit_route_coefficient
            * np.sum(np.abs(coefficients))
        )
        leverage = quadratic_leverage + response_projection_allowance
        covariance_limit = float(partition.maximum_covariance_leverage) * scale

        retained = parent_mask.to_bool_array()
        # Reconstruct the chart-anchor availability: routes in the cumulative
        # action were retained when the geometry was measured and have since
        # been removed under exact transition receipts.
        if cumulative:
            retained[np.asarray(cumulative, dtype=np.int64)] = True
        retained_partition = int(
            np.sum(retained[np.asarray(partition.route_indices, dtype=np.int64)])
        )
        partition_limit = min(
            retained_partition,
            max(
                1,
                int(
                    math.floor(
                        partition.maximum_partition_action_fraction
                        * scale
                        * retained_partition
                    )
                ),
            ),
        )
        global_layer_by_route: dict[int, str] = {}
        for declared_partition in partitioning.partitions:
            for route_index, layer_id in zip(
                declared_partition.route_indices,
                declared_partition.layer_ids,
                strict=True,
            ):
                global_layer_by_route[int(route_index)] = str(layer_id)
        candidate_layers = {global_layer_by_route[int(index)] for index in new_removed}
        global_action = tuple(sorted((*cumulative, *new_removed)))
        layer_removed: dict[str, int] = {}
        for layer in sorted(candidate_layers):
            layer_removed[layer] = sum(
                1
                for index in global_action
                if global_layer_by_route[int(index)] == layer
            )
        layer_limits: dict[str, int] = {}
        layer_covariance_leverage: dict[str, float] = {}
        layer_covariance_limits: dict[str, float] = {}
        for layer in sorted(candidate_layers):
            global_layer_indices = np.asarray(
                [
                    index
                    for index, route_layer in global_layer_by_route.items()
                    if route_layer == layer
                ],
                dtype=np.int64,
            )
            retained_layer = int(np.sum(retained[global_layer_indices]))
            raw_limit = int(
                math.floor(
                    partition.maximum_layer_action_fraction * scale * retained_layer
                )
            )
            layer_limits[layer] = min(
                retained_layer,
                max(partition.minimum_layer_action_count, raw_limit),
            )
        for layer in sorted(set(partition.layer_ids)):
            partition_layer_indices = [
                index
                for index, route_layer in zip(
                    partition.route_indices,
                    partition.layer_ids,
                    strict=True,
                )
                if route_layer == layer
            ]
            if not partition_layer_indices:
                continue
            global_layer_indices = np.asarray(partition_layer_indices, dtype=np.int64)
            local_layer_indices = np.asarray(
                [local[int(index)] for index in global_layer_indices],
                dtype=np.int64,
            )
            layer_coefficients = coefficients[local_layer_indices]
            layer_squared = geometry.quadratic_form(
                layer_coefficients,
                local_offsets=local_layer_indices,
            )
            layer_quadratic_leverage = math.sqrt(max(0.0, layer_squared))
            layer_projection_allowance = float(
                geometry.response_projection_residual_bound_per_unit_route_coefficient
                * np.sum(np.abs(layer_coefficients))
            )
            layer_covariance_leverage[layer] = (
                layer_quadratic_leverage + layer_projection_allowance
            )
            layer_covariance_limits[layer] = (
                float(partition.maximum_layer_covariance_leverage) * scale
            )

        violations: list[str] = []
        if leverage > covariance_limit + 1e-12:
            violations.append("covariance leverage")
        if len(removed) > partition_limit:
            violations.append("partition action fraction")
        for layer, count in layer_removed.items():
            if count > layer_limits[layer]:
                violations.append(
                    f"layer {layer} relinearization leverage (action fraction)"
                )
        for layer, value in layer_covariance_leverage.items():
            if value > layer_covariance_limits[layer] + 1e-12:
                violations.append(
                    f"layer {layer} relinearization leverage (covariance)"
                )
        action = "reuse" if not violations else "shrink"
        reason = (
            "candidate lies inside its partition covariance and layer-local trust region"
            if not violations
            else "candidate exceeds " + ", ".join(violations)
        )
        return PartitionActionAssessment(
            action=action,
            reason=reason,
            partition_id=partition.partition_id,
            covariance_leverage=leverage,
            covariance_limit=covariance_limit,
            partition_removed=len(removed),
            partition_removed_limit=partition_limit,
            layer_removed=layer_removed,
            layer_removed_limits=layer_limits,
            layer_covariance_leverage=layer_covariance_leverage,
            layer_covariance_limits=layer_covariance_limits,
            diagnostics={
                "geometry_hash": geometry.content_hash,
                "partitioning_hash": partitioning.content_hash,
                "covariance_uncertainty_spectral_bound": geometry.covariance_uncertainty_spectral_bound,
                "sketch_residual_spectral_bound": geometry.sketch_residual_spectral_bound,
                "response_projection_residual_bound_per_unit_route_coefficient": (
                    geometry.response_projection_residual_bound_per_unit_route_coefficient
                ),
                "quadratic_covariance_leverage": quadratic_leverage,
                "response_projection_allowance": response_projection_allowance,
                "cross_layer_within_partition": len(set(partition.layer_ids)) > 1,
                "cross_partition_score_comparison": False,
                "cumulative_since_relinearization": True,
                "new_candidate_removed_count": len(new_removed),
                "prior_cumulative_removed_count": len(cumulative),
                "prior_cumulative_partition_removed_count": len(cumulative_partition),
                "physical_layer_guard_cross_partition": True,
                "exact_hard_mask_commit_authority": True,
            },
        )


__all__ = [
    "CANONICAL_GLOBAL_SIGN_PARTITION_IDS",
    "COVARIANCE_RANK_EXPANSION_CACHE_KEY",
    "CanonicalSignPartitionFamily",
    "CanonicalTwoPartitionTrustPolicy",
    "CovarianceRankExpansionObligation",
    "GEOMETRIC_PARTITION_FILTER_CACHE_KEY",
    "PartitionActionAssessment",
    "PartitionCovarianceGeometry",
    "PartitionedCovarianceTrustRegion",
    "UnresolvedCovarianceObligation",
    "assert_canonical_global_two_partitioning",
    "canonical_global_two_sign_partitioning",
    "candidate_partition",
    "derive_partition_covariance_geometry",
]
