"""Workflow-neutral nuisance-quotiented response measurement field.

The field carries fixed and nuisance-profiled response geometry on one complete
physical route axis. It defines no workflow selector, candidate policy,
stopping rule, or hard-mask commit authority. state_hash deliberately
preserves the historical IRG v3 receipt identity; new lineage should bind
capability_hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.geometry.projectors import (
    ProjectorMatch,
    match_projectors,
    stable_spectral_projectors,
)
from mechanistic_systems.geometry.quotient import (
    IntrinsicMetricOperator,
    QuotientDiagnostics,
)

from .contracts import readonly_matrix, validate_route_axis
from .coupled import CoupledResponseField
from .quotient import (
    IntrinsicResponseFactorOperator,
    ResponseQuotientDiagnostics,
)
from .tensors import ResponseFactorTensor, ResponseMomentTensor


@dataclass(frozen=True)
class IntrinsicResponseField:
    route_ids: tuple[str, ...]
    route_costs: np.ndarray
    protected: np.ndarray
    route_factor: np.ndarray
    nuisance_factor: np.ndarray
    measured_mask_hash: str
    calibration_state_id: str
    regularization: float = 1e-8
    rcond: float = 1e-10
    maximum_condition_number: float = 1e12
    source_outcome_hashes: tuple[str, ...] = ()
    response_moments: ResponseMomentTensor | None = None
    intrinsic_response_moments: ResponseMomentTensor | None = None
    response_factor: ResponseFactorTensor | None = None
    response_factor_indices: tuple[int, ...] = ()
    nuisance_response_rows: np.ndarray | None = None
    response_metric_whitener: np.ndarray | None = None
    metric_route_rows: np.ndarray | None = None
    metric_nuisance_rows: np.ndarray | None = None
    pullback_phenotype_moments: Any | None = None
    global_trust_geometry: Any | None = None
    intrinsic_atlas_geometry: Any | None = None
    functional_atlas: Any | None = None
    interaction_schedule: Any | None = None
    local_regime_summary: Any | None = None
    variational_atlas_posterior: Any | None = None

    def __post_init__(self) -> None:
        route_ids, costs, protected = validate_route_axis(
            route_ids=self.route_ids,
            route_costs=self.route_costs,
            protected=self.protected,
        )
        route = readonly_matrix(self.route_factor, name="route_factor")
        nuisance = readonly_matrix(self.nuisance_factor, name="nuisance_factor")
        if route.shape[0] != nuisance.shape[0] or route.shape[1] != len(route_ids):
            raise ValueError(
                "route and nuisance factors must share contexts and route factor must align"
            )
        if not self.measured_mask_hash or not self.calibration_state_id:
            raise ValueError("intrinsic geometry state has an empty identity field")
        if (
            self.regularization < 0.0
            or self.rcond < 0.0
            or self.maximum_condition_number <= 0.0
        ):
            raise ValueError("intrinsic quotient controls are invalid")
        for label, moments in (
            ("fixed", self.response_moments),
            ("intrinsic", self.intrinsic_response_moments),
        ):
            if moments is not None and moments.route_ids != route_ids:
                raise ValueError(
                    f"{label} response moments use a different physical route axis"
                )
        if (
            self.response_moments is not None
            and self.intrinsic_response_moments is not None
        ):
            if (
                self.response_moments.channel_ids
                != self.intrinsic_response_moments.channel_ids
            ):
                raise ValueError(
                    "fixed and intrinsic response moments use different response axes"
                )
            if (
                self.response_moments.context_axis_hash
                != self.intrinsic_response_moments.context_axis_hash
            ):
                raise ValueError(
                    "fixed and intrinsic response moments use different context axes"
                )

        factor_indices = tuple(int(value) for value in self.response_factor_indices)
        if self.response_factor is not None:
            if not factor_indices:
                lookup = {route_id: index for index, route_id in enumerate(route_ids)}
                try:
                    factor_indices = tuple(
                        lookup[route_id] for route_id in self.response_factor.route_ids
                    )
                except KeyError as error:
                    raise ValueError(
                        "response factor contains a route absent from the physical route axis"
                    ) from error
            if (
                len(factor_indices) != self.response_factor.route_count
                or len(set(factor_indices)) != len(factor_indices)
                or any(index < 0 or index >= len(route_ids) for index in factor_indices)
            ):
                raise ValueError(
                    "response_factor_indices must map the screened factor to unique physical routes"
                )
            expected_ids = tuple(route_ids[index] for index in factor_indices)
            if self.response_factor.route_ids != expected_ids:
                raise ValueError(
                    "screened response factor IDs disagree with response_factor_indices"
                )
            if self.nuisance_response_rows is None:
                raise ValueError(
                    "matrix-valued intrinsic geometry requires nuisance response rows"
                )
            nuisance_response = np.asarray(
                self.nuisance_response_rows, dtype=np.float64
            )
            if (
                nuisance_response.ndim != 3
                or nuisance_response.shape[:2] != self.response_factor.rows.shape[:2]
            ):
                raise ValueError(
                    "nuisance response rows must share context and response axes"
                )
            nuisance_response = np.array(nuisance_response, copy=True, order="C")
            nuisance_response.setflags(write=False)
            object.__setattr__(self, "nuisance_response_rows", nuisance_response)
            if len(factor_indices) < len(route_ids) and (
                self.response_moments is None or self.intrinsic_response_moments is None
            ):
                raise ValueError(
                    "a screened v3 factor requires full-route fixed and intrinsic response moments"
                )
            if self.response_metric_whitener is not None:
                whitener = np.asarray(self.response_metric_whitener, dtype=np.float64)
                if (
                    whitener.ndim != 3
                    or whitener.shape[0] != self.response_factor.sample_count
                    or whitener.shape[2] != self.response_factor.response_dimension
                ):
                    raise ValueError("response metric whitener has incompatible axes")
                whitener = np.array(whitener, copy=True, order="C")
                whitener.setflags(write=False)
                object.__setattr__(self, "response_metric_whitener", whitener)
            direct_route = self.metric_route_rows
            direct_nuisance = self.metric_nuisance_rows
            if (direct_route is None) != (direct_nuisance is None):
                raise ValueError(
                    "v3 direct task-metric quotient requires both route and nuisance factors"
                )
            if direct_route is not None and direct_nuisance is not None:
                route_metric = np.asarray(direct_route, dtype=np.float64)
                nuisance_metric = np.asarray(direct_nuisance, dtype=np.float64)
                if (
                    route_metric.ndim != 3
                    or nuisance_metric.ndim != 3
                    or route_metric.shape[0] != self.response_factor.sample_count
                    or nuisance_metric.shape[0] != self.response_factor.sample_count
                    or route_metric.shape[1] != nuisance_metric.shape[1]
                    or route_metric.shape[2] != self.response_factor.route_count
                    or nuisance_metric.shape[2] != nuisance_response.shape[2]
                    or not np.isfinite(route_metric).all()
                    or not np.isfinite(nuisance_metric).all()
                ):
                    raise ValueError(
                        "v3 direct task-metric factors require finite "
                        "[context,metric_factor,route/nuisance] axes"
                    )
                route_metric = np.array(route_metric, copy=True, order="C")
                nuisance_metric = np.array(nuisance_metric, copy=True, order="C")
                route_metric.setflags(write=False)
                nuisance_metric.setflags(write=False)
                object.__setattr__(self, "metric_route_rows", route_metric)
                object.__setattr__(self, "metric_nuisance_rows", nuisance_metric)
            if self.response_metric_whitener is not None and direct_route is not None:
                raise ValueError(
                    "v3 quotient may use either a whitener or direct task-metric factors, not both"
                )
        elif factor_indices:
            raise ValueError("response_factor_indices require a response_factor")
        for label, geometry in (
            ("global trust", self.global_trust_geometry),
            ("intrinsic atlas", self.intrinsic_atlas_geometry),
        ):
            if geometry is not None and (
                geometry.route_ids != route_ids
                or geometry.measured_mask_hash != self.measured_mask_hash
                or geometry.calibration_state_id != self.calibration_state_id
            ):
                raise ValueError(f"v3 {label} geometry uses a different physical state")
        if (
            self.intrinsic_atlas_geometry is not None
            and self.intrinsic_atlas_geometry.metadata.get("geometry_role")
            != "nuisance_quotiented_atlas_embedding"
        ):
            raise ValueError(
                "v3 atlas identity requires a nuisance-quotiented geometry"
            )
        if self.pullback_phenotype_moments is not None and (
            self.pullback_phenotype_moments.route_ids != route_ids
        ):
            raise ValueError("v3 pullback phenotype uses a different route axis")
        if self.functional_atlas is not None and (
            self.functional_atlas.spec.phenotype_model
            not in {
                "intrinsic_stable_subspace_cover",
                "stateful_interventional_subspace_cover",
            }
            or self.functional_atlas.route_ids != route_ids
            or self.functional_atlas.measured_mask_hash != self.measured_mask_hash
            or self.functional_atlas.calibration_state_id != self.calibration_state_id
        ):
            raise ValueError(
                "intrinsic functional atlas uses a different state or model"
            )
        if self.interaction_schedule is not None and (
            self.functional_atlas is None
            or self.interaction_schedule.atlas_hash
            != self.functional_atlas.content_hash
        ):
            raise ValueError("v3 interaction schedule is not bound to its atlas")
        if self.local_regime_summary is not None and (
            self.local_regime_summary.route_ids != route_ids
            or self.local_regime_summary.measured_mask_hash != self.measured_mask_hash
        ):
            raise ValueError("v3 local regime summary differs from its response chart")
        if self.variational_atlas_posterior is not None and (
            self.functional_atlas is None
            or self.variational_atlas_posterior.route_ids != route_ids
            or self.functional_atlas.metadata.get("variational_posterior_hash")
            != self.variational_atlas_posterior.content_hash
        ):
            raise ValueError(
                "v3 variational atlas posterior is not bound to its functional atlas"
            )

        object.__setattr__(self, "route_ids", route_ids)
        object.__setattr__(self, "route_costs", costs)
        object.__setattr__(self, "protected", protected)
        object.__setattr__(self, "route_factor", route)
        object.__setattr__(self, "nuisance_factor", nuisance)
        object.__setattr__(self, "response_factor_indices", factor_indices)
        _ = self.operator
        if self.response_factor is not None:
            _ = self.response_quotient

    @property
    def sample_count(self) -> int:
        if self.response_factor is not None:
            return self.response_factor.sample_count
        if self.intrinsic_response_moments is not None:
            return self.intrinsic_response_moments.sample_count
        if self.response_moments is not None:
            return self.response_moments.sample_count
        return int(self.route_factor.shape[0])

    @cached_property
    def operator(self) -> IntrinsicMetricOperator:
        return IntrinsicMetricOperator.from_factors(
            self.route_factor,
            self.nuisance_factor,
            regularization=self.regularization,
            rcond=self.rcond,
            maximum_condition_number=self.maximum_condition_number,
        )

    @property
    def diagnostics(self) -> QuotientDiagnostics:
        return self.operator.diagnostics

    @cached_property
    def _global_to_factor(self) -> dict[int, int]:
        return {
            global_index: local_index
            for local_index, global_index in enumerate(self.response_factor_indices)
        }

    def factor_local_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        selected = np.asarray(tuple(int(value) for value in indices), dtype=np.int64)
        if selected.size == 0:
            return selected
        if (
            np.any(selected < 0)
            or np.any(selected >= len(self.route_ids))
            or np.unique(selected).size != selected.size
        ):
            raise ValueError("physical route indices are invalid")
        missing = [
            int(value) for value in selected if int(value) not in self._global_to_factor
        ]
        if missing:
            raise ValueError(
                "intrinsic coupled geometry is unavailable for routes outside the screened factor: "
                + ",".join(map(str, missing[:8]))
            )
        return np.asarray(
            [self._global_to_factor[int(value)] for value in selected], dtype=np.int64
        )

    @cached_property
    def response_quotient(self) -> IntrinsicResponseFactorOperator:
        if self.response_factor is None or self.nuisance_response_rows is None:
            raise RuntimeError("matrix-valued response quotient is unavailable")
        return IntrinsicResponseFactorOperator(
            self.response_factor,
            self.nuisance_response_rows,
            metric_whitener=self.response_metric_whitener,
            whitened_route_rows=self.metric_route_rows,
            whitened_nuisance_rows=self.metric_nuisance_rows,
            regularization=self.regularization,
            rcond=self.rcond,
            maximum_condition_number=self.maximum_condition_number,
        )

    @cached_property
    def intrinsic_response_factor(self) -> ResponseFactorTensor | None:
        if self.response_factor is None:
            return None
        return self.response_quotient.residualized_factor()

    @property
    def response_diagnostics(self) -> ResponseQuotientDiagnostics | None:
        return (
            None if self.response_factor is None else self.response_quotient.diagnostics
        )

    def fixed_action_moments(self, indices: Sequence[int], coefficients=None):
        if self.response_factor is None:
            raise RuntimeError("fixed coupled response factor is unavailable")
        return self.response_factor.action_moments(
            self.factor_local_indices(indices), coefficients
        )

    def intrinsic_action_moments(self, indices: Sequence[int], coefficients=None):
        factor = self.intrinsic_response_factor
        if factor is None:
            raise RuntimeError("intrinsic coupled response factor is unavailable")
        return factor.action_moments(self.factor_local_indices(indices), coefficients)

    @cached_property
    def fixed_diagonal(self) -> np.ndarray:
        if self.response_moments is not None:
            return np.trace(self.response_moments.route_second_moment, axis1=1, axis2=2)
        return np.mean(self.route_factor * self.route_factor, axis=0)

    @cached_property
    def intrinsic_diagonal(self) -> np.ndarray:
        if self.intrinsic_response_moments is not None:
            return np.trace(
                self.intrinsic_response_moments.route_second_moment, axis1=1, axis2=2
            )
        if self.intrinsic_response_factor is not None and len(
            self.response_factor_indices
        ) == len(self.route_ids):
            return np.trace(
                self.intrinsic_response_factor.marginal_moments.route_second_moment,
                axis1=1,
                axis2=2,
            )
        return self.operator.diagonal()

    def intrinsic_submatrix(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        if self.intrinsic_response_factor is not None:
            local = self.factor_local_indices(indices)
            factor = self.intrinsic_response_factor.rows[:, :, local]
            flattened = factor.reshape(-1, local.size) * np.sqrt(float(factor.shape[1]))
            metric = flattened.T @ flattened / flattened.shape[0]
            return 0.5 * (metric + metric.T)
        return self.operator.submatrix(indices)

    @property
    def fixed_metric(self) -> np.ndarray:
        return self.operator.fixed_submatrix(
            np.arange(len(self.route_ids), dtype=np.int64)
        )

    @property
    def intrinsic_metric(self) -> np.ndarray:
        if self.response_factor is not None and len(self.response_factor_indices) < len(
            self.route_ids
        ):
            raise RuntimeError(
                "refusing a native-route dense intrinsic metric from a screened v3 factor"
            )
        return self.operator.dense()

    @cached_property
    def state_hash(self) -> str:
        return content_hash(
            {
                "version": "v3",
                "route_ids": self.route_ids,
                "route_costs": self.route_costs,
                "protected": self.protected.astype(np.uint8),
                "route_factor": self.route_factor,
                "nuisance_factor": self.nuisance_factor,
                "response_moment_hash": (
                    None
                    if self.response_moments is None
                    else self.response_moments.content_hash
                ),
                "intrinsic_response_moment_hash": (
                    None
                    if self.intrinsic_response_moments is None
                    else self.intrinsic_response_moments.content_hash
                ),
                "response_factor_hash": (
                    None
                    if self.response_factor is None
                    else self.response_factor.content_hash
                ),
                "intrinsic_response_factor_hash": (
                    None
                    if self.response_factor is None
                    else self.response_quotient.legacy_residualized_factor().content_hash
                ),
                "response_factor_indices": self.response_factor_indices,
                "response_metric_whitener": self.response_metric_whitener,
                "metric_route_rows": self.metric_route_rows,
                "metric_nuisance_rows": self.metric_nuisance_rows,
                "task_metric_factor_mode": (
                    "prewhitened_task_metric_factor"
                    if self.metric_route_rows is not None
                    else (
                        "explicit_task_metric_whitener"
                        if self.response_metric_whitener is not None
                        else "legacy_euclidean_fallback"
                    )
                ),
                "measured_mask_hash": self.measured_mask_hash,
                "calibration_state_id": self.calibration_state_id,
                "regularization": self.regularization,
                "rcond": self.rcond,
                "maximum_condition_number": self.maximum_condition_number,
                "source_outcome_hashes": self.source_outcome_hashes,
                "pullback_phenotype_hash": (
                    None
                    if self.pullback_phenotype_moments is None
                    else self.pullback_phenotype_moments.content_hash
                ),
                "global_trust_geometry_hash": (
                    None
                    if self.global_trust_geometry is None
                    else self.global_trust_geometry.content_hash
                ),
                "intrinsic_atlas_geometry_hash": (
                    None
                    if self.intrinsic_atlas_geometry is None
                    else self.intrinsic_atlas_geometry.content_hash
                ),
                "functional_atlas_hash": (
                    None
                    if self.functional_atlas is None
                    else self.functional_atlas.content_hash
                ),
                "interaction_schedule_hash": (
                    None
                    if self.interaction_schedule is None
                    else self.interaction_schedule.content_hash
                ),
                "local_regime_summary_hash": (
                    None
                    if self.local_regime_summary is None
                    else self.local_regime_summary.content_hash
                ),
                "variational_atlas_posterior_hash": (
                    None
                    if self.variational_atlas_posterior is None
                    else self.variational_atlas_posterior.content_hash
                ),
            },
            prefix="irg-v3-state-",
        )

    @cached_property
    def capability_hash(self) -> str:
        """Workflow-neutral identity for this nuisance-quotient capability."""

        return content_hash(
            {
                "capability": "intrinsic_response_field_v1",
                "legacy_state_hash": self.state_hash,
                "intrinsic_response_factor_hash": (
                    None
                    if self.intrinsic_response_factor is None
                    else self.intrinsic_response_factor.content_hash
                ),
                "response_quotient_diagnostics": (
                    None
                    if self.response_diagnostics is None
                    else self.response_diagnostics.to_dict()
                ),
            },
            prefix="intrinsic-response-field-",
        )

    def forget_nuisance_quotient(self) -> CoupledResponseField:
        """Forget nuisance-quotient structure without choosing a workflow.

        The resulting coupled-response field preserves the historical v2 state
        hash solely for replay identity.  It is a measurement capability, not
        an executable or promotable ``irg/v2`` workflow.
        """

        if self.response_factor is not None:
            return CoupledResponseField.from_response_factor(
                route_ids=self.route_ids,
                route_costs=self.route_costs,
                protected=self.protected,
                response_moments=self.response_moments,
                response_factor=self.response_factor,
                response_factor_indices=self.response_factor_indices,
                jacobian_factor=self.route_factor,
                measured_mask_hash=self.measured_mask_hash,
                calibration_state_id=self.calibration_state_id,
                source_outcome_hashes=self.source_outcome_hashes,
                pullback_phenotype_moments=self.pullback_phenotype_moments,
                global_trust_geometry=self.global_trust_geometry,
                local_regime_summary=self.local_regime_summary,
            )
        if self.response_moments is not None:
            return CoupledResponseField.from_response_moments(
                route_costs=self.route_costs,
                protected=self.protected,
                response_moments=self.response_moments,
                jacobian_factor=self.route_factor,
                measured_mask_hash=self.measured_mask_hash,
                calibration_state_id=self.calibration_state_id,
                source_outcome_hashes=self.source_outcome_hashes,
                pullback_phenotype_moments=self.pullback_phenotype_moments,
                global_trust_geometry=self.global_trust_geometry,
                local_regime_summary=self.local_regime_summary,
            )
        return CoupledResponseField.from_jacobian_rows(
            route_ids=self.route_ids,
            route_costs=self.route_costs,
            protected=self.protected,
            rows=self.route_factor,
            measured_mask_hash=self.measured_mask_hash,
            calibration_state_id=self.calibration_state_id,
            source_outcome_hashes=self.source_outcome_hashes,
        )

    def transport_matches(
        self,
        newer: "IntrinsicResponseField",
        *,
        route_indices=None,
        maximum_angle: float = 0.35,
        relative_gap: float = 1e-3,
        noise_floor: float = 1e-6,
    ) -> tuple[ProjectorMatch, ...]:
        if self.route_ids != newer.route_ids:
            raise ValueError(
                "projector transport requires a common physical route axis"
            )
        if route_indices is None:
            common = tuple(
                index
                for index in self.response_factor_indices
                if index in set(newer.response_factor_indices)
            )
            if not common:
                if len(self.route_ids) > 4096:
                    raise RuntimeError(
                        "large-route projector transport requires a common screened route set"
                    )
                common = tuple(range(len(self.route_ids)))
            indices = np.asarray(common, dtype=np.int64)
        else:
            indices = np.asarray(route_indices, dtype=np.int64).reshape(-1)
        old = stable_spectral_projectors(
            self.intrinsic_submatrix(indices),
            relative_gap=relative_gap,
            noise_floor=noise_floor,
        )
        new = stable_spectral_projectors(
            newer.intrinsic_submatrix(indices),
            relative_gap=relative_gap,
            noise_floor=noise_floor,
        )
        return match_projectors(old, new, maximum_angle=maximum_angle)


__all__ = ["IntrinsicResponseField"]
