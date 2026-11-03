"""Workflow-neutral screened coupled-response measurement field.

The field retains complete route-local response moments and a screened
cross-route response factor on an explicitly mapped subset of the same physical
route axis. It is a measurement/representation capability: it defines no
workflow selector, candidate policy, stopping rule, or hard-mask commit
authority.

state_hash deliberately preserves the historical v2 serialization identity so
archived receipts and transitional v3 projections remain byte-comparable. New
lineage should use capability_hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.geometry.metrics import validate_metric
from mechanistic_systems.geometry.projectors import (
    StableProjector,
    residualize_route_direction,
    stable_spectral_projectors,
)

from .contracts import readonly_matrix, validate_route_axis
from .tensors import ResponseFactorTensor, ResponseMomentTensor


@dataclass(frozen=True)
class CoupledResponseField:
    """Screened coupled-response state on a complete physical route axis.

    ``response_moments`` is O(R d^2) and covers every route.  ``response_factor``
    is O(n d K) and covers only the screened global indices in
    ``response_factor_indices``.  Every action evaluated by the coupled model is
    mapped explicitly between these axes; a missing route is refused rather than
    silently assigned zero interaction.
    """

    route_ids: tuple[str, ...]
    route_costs: np.ndarray
    protected: np.ndarray
    jacobian_factor: np.ndarray
    measured_mask_hash: str
    calibration_state_id: str
    driver_residualized: bool = False
    driver_hash: str = "none"
    source_outcome_hashes: tuple[str, ...] = ()
    response_moments: ResponseMomentTensor | None = None
    response_factor: ResponseFactorTensor | None = None
    response_factor_indices: tuple[int, ...] = ()
    pullback_phenotype_moments: Any | None = None
    global_trust_geometry: Any | None = None
    functional_atlas: Any | None = None
    interaction_schedule: Any | None = None
    local_regime_summary: Any | None = None
    bootstrap_unmeasured: bool = False

    def __post_init__(self) -> None:
        route_ids, costs, protected = validate_route_axis(
            route_ids=self.route_ids,
            route_costs=self.route_costs,
            protected=self.protected,
        )
        factor = readonly_matrix(self.jacobian_factor, name="jacobian_factor")
        if factor.shape[1] != len(route_ids):
            raise ValueError("jacobian_factor route dimension does not match route_ids")
        if (
            self.response_moments is not None
            and self.response_moments.route_ids != route_ids
        ):
            raise ValueError(
                "matrix-valued response moments use a different physical route axis"
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
                    "response_factor_indices must uniquely map the screened factor to physical routes"
                )
            expected_ids = tuple(route_ids[index] for index in factor_indices)
            if self.response_factor.route_ids != expected_ids:
                raise ValueError(
                    "response factor route IDs do not match response_factor_indices"
                )
            if len(factor_indices) < len(route_ids) and self.response_moments is None:
                raise ValueError(
                    "a screened response factor requires full-route response moments"
                )
        elif factor_indices:
            raise ValueError("response_factor_indices require a response_factor")
        if self.pullback_phenotype_moments is not None:
            if self.pullback_phenotype_moments.route_ids != route_ids:
                raise ValueError(
                    "coupled-response pullback phenotype uses a different physical route axis"
                )
        if self.global_trust_geometry is not None:
            if (
                self.global_trust_geometry.route_ids != route_ids
                or self.global_trust_geometry.measured_mask_hash
                != self.measured_mask_hash
                or self.global_trust_geometry.calibration_state_id
                != self.calibration_state_id
            ):
                raise ValueError(
                    "coupled-response global trust geometry differs from its response chart"
                )
        if self.functional_atlas is not None:
            if (
                self.functional_atlas.route_ids != route_ids
                or self.functional_atlas.measured_mask_hash != self.measured_mask_hash
                or self.functional_atlas.calibration_state_id
                != self.calibration_state_id
            ):
                raise ValueError(
                    "coupled-response atlas differs from its response chart"
                )
        if self.interaction_schedule is not None:
            if (
                self.functional_atlas is None
                or self.interaction_schedule.atlas_hash
                != self.functional_atlas.content_hash
            ):
                raise ValueError(
                    "coupled-response interaction schedule requires the current functional atlas"
                )
        if self.local_regime_summary is not None:
            if (
                self.local_regime_summary.route_ids != route_ids
                or self.local_regime_summary.measured_mask_hash
                != self.measured_mask_hash
            ):
                raise ValueError(
                    "coupled-response local regime summary differs from its response chart"
                )
        if self.bootstrap_unmeasured:
            if (
                self.response_moments is not None
                or self.response_factor is not None
                or self.source_outcome_hashes
                or factor.shape[0] != 1
                or np.any(factor != 0.0)
            ):
                raise ValueError(
                    "an unmeasured coupled-response bootstrap must be the single zero "
                    "placeholder row with no physical outcome lineage"
                )

        if (
            not self.measured_mask_hash
            or not self.calibration_state_id
            or not self.driver_hash
        ):
            raise ValueError("coupled response field has an empty identity field")
        object.__setattr__(self, "route_ids", route_ids)
        object.__setattr__(self, "route_costs", costs)
        object.__setattr__(self, "protected", protected)
        object.__setattr__(self, "jacobian_factor", factor)
        object.__setattr__(self, "response_factor_indices", factor_indices)

    @classmethod
    def from_jacobian_rows(
        cls,
        *,
        route_ids: Sequence[str],
        route_costs: Sequence[float],
        protected: Sequence[bool],
        rows: np.ndarray,
        measured_mask_hash: str,
        calibration_state_id: str,
        residualize_direction: Sequence[float] | np.ndarray | None = None,
        source_outcome_hashes: Sequence[str] = (),
    ) -> "CoupledResponseField":
        factor = np.asarray(rows, dtype=np.float64)
        driver_residualized = residualize_direction is not None
        driver_hash = "none"
        if residualize_direction is not None:
            direction = np.asarray(residualize_direction, dtype=np.float64).reshape(-1)
            factor = residualize_route_direction(factor, direction)
            driver_hash = content_hash(direction, prefix="irg-driver-")
        return cls(
            route_ids=tuple(route_ids),
            route_costs=np.asarray(route_costs, dtype=np.float64),
            protected=np.asarray(protected, dtype=bool),
            jacobian_factor=factor,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            driver_residualized=driver_residualized,
            driver_hash=driver_hash,
            source_outcome_hashes=tuple(source_outcome_hashes),
        )

    @classmethod
    def from_response_moments(
        cls,
        *,
        route_costs: Sequence[float],
        protected: Sequence[bool],
        response_moments: ResponseMomentTensor,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_outcome_hashes: Sequence[str] = (),
        jacobian_factor: np.ndarray | None = None,
        pullback_phenotype_moments: Any | None = None,
        global_trust_geometry: Any | None = None,
        functional_atlas: Any | None = None,
        interaction_schedule: Any | None = None,
        local_regime_summary: Any | None = None,
    ) -> "CoupledResponseField":
        legacy = (
            np.zeros((1, response_moments.route_count), dtype=np.float64)
            if jacobian_factor is None
            else np.asarray(jacobian_factor, dtype=np.float64)
        )
        return cls(
            route_ids=response_moments.route_ids,
            route_costs=np.asarray(route_costs, dtype=np.float64),
            protected=np.asarray(protected, dtype=bool),
            jacobian_factor=legacy,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            source_outcome_hashes=tuple(source_outcome_hashes),
            response_moments=response_moments,
            pullback_phenotype_moments=pullback_phenotype_moments,
            global_trust_geometry=global_trust_geometry,
            functional_atlas=functional_atlas,
            interaction_schedule=interaction_schedule,
            local_regime_summary=local_regime_summary,
        )

    @classmethod
    def from_response_factor(
        cls,
        *,
        route_costs: Sequence[float],
        protected: Sequence[bool],
        response_factor: ResponseFactorTensor,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_outcome_hashes: Sequence[str] = (),
        route_ids: Sequence[str] | None = None,
        response_moments: ResponseMomentTensor | None = None,
        response_factor_indices: Sequence[int] | None = None,
        jacobian_factor: np.ndarray | None = None,
        pullback_phenotype_moments: Any | None = None,
        global_trust_geometry: Any | None = None,
        functional_atlas: Any | None = None,
        interaction_schedule: Any | None = None,
        local_regime_summary: Any | None = None,
    ) -> "CoupledResponseField":
        full_ids = (
            tuple(route_ids) if route_ids is not None else response_factor.route_ids
        )
        if response_factor_indices is None:
            lookup = {route_id: index for index, route_id in enumerate(full_ids)}
            try:
                factor_indices = tuple(
                    lookup[route_id] for route_id in response_factor.route_ids
                )
            except KeyError as error:
                raise ValueError(
                    "response factor route IDs are absent from the declared full route axis"
                ) from error
        else:
            factor_indices = tuple(int(value) for value in response_factor_indices)
        if jacobian_factor is None:
            if response_factor.route_ids == full_ids:
                n, d, r = response_factor.rows.shape
                legacy = response_factor.rows.reshape(n * d, r) * np.sqrt(float(d))
            else:
                legacy = np.zeros((1, len(full_ids)), dtype=np.float64)
        else:
            legacy = np.asarray(jacobian_factor, dtype=np.float64)
        return cls(
            route_ids=full_ids,
            route_costs=np.asarray(route_costs, dtype=np.float64),
            protected=np.asarray(protected, dtype=bool),
            jacobian_factor=legacy,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            source_outcome_hashes=tuple(source_outcome_hashes),
            response_moments=response_moments,
            response_factor=response_factor,
            response_factor_indices=factor_indices,
            pullback_phenotype_moments=pullback_phenotype_moments,
            global_trust_geometry=global_trust_geometry,
            functional_atlas=functional_atlas,
            interaction_schedule=interaction_schedule,
            local_regime_summary=local_regime_summary,
        )

    @property
    def sample_count(self) -> int:
        if self.bootstrap_unmeasured:
            return 0
        if self.response_factor is not None:
            return self.response_factor.sample_count
        if self.response_moments is not None:
            return self.response_moments.sample_count
        return int(self.jacobian_factor.shape[0])

    @property
    def factor_is_full_axis(self) -> bool:
        return bool(
            self.response_factor is not None
            and len(self.response_factor_indices) == len(self.route_ids)
        )

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
                "coupled response geometry is unavailable for routes outside the screened factor: "
                + ",".join(map(str, missing[:8]))
            )
        return np.asarray(
            [self._global_to_factor[int(value)] for value in selected], dtype=np.int64
        )

    def action_moments(
        self,
        indices: Sequence[int] | np.ndarray,
        coefficients: Sequence[float] | np.ndarray | None = None,
    ):
        if self.response_factor is None:
            raise RuntimeError("matrix-valued coupled response factor is unavailable")
        return self.response_factor.action_moments(
            self.factor_local_indices(indices), coefficients
        )

    def diagonal_action_moments(
        self,
        indices: Sequence[int] | np.ndarray,
        coefficients: Sequence[float] | np.ndarray | None = None,
    ):
        if self.response_moments is not None:
            return self.response_moments.action_moments(indices, coefficients)
        if self.response_factor is None:
            raise RuntimeError("response moments are unavailable")
        return self.response_factor.marginal_moments.action_moments(
            self.factor_local_indices(indices), coefficients
        )

    @cached_property
    def mean_response(self) -> np.ndarray:
        if self.response_moments is not None:
            return np.linalg.norm(self.response_moments.mean, axis=0)
        if self.response_factor is not None and self.factor_is_full_axis:
            return np.linalg.norm(self.response_factor.marginal_moments.mean, axis=0)
        return np.mean(self.jacobian_factor, axis=0)

    @cached_property
    def diagonal_energy(self) -> np.ndarray:
        if self.response_moments is not None:
            return np.trace(self.response_moments.route_second_moment, axis1=1, axis2=2)
        if self.response_factor is not None and self.factor_is_full_axis:
            return np.trace(
                self.response_factor.marginal_moments.route_second_moment,
                axis1=1,
                axis2=2,
            )
        return np.mean(self.jacobian_factor * self.jacobian_factor, axis=0)

    def metric_for_indices(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if self.response_factor is not None:
            local = self.factor_local_indices(selected)
            rows = self.response_factor.rows[:, :, local]
            flattened = rows.reshape(-1, local.size) * np.sqrt(float(rows.shape[1]))
            gram = flattened.T @ flattened / flattened.shape[0]
            return validate_metric(gram)
        if (
            np.any(selected < 0)
            or np.any(selected >= len(self.route_ids))
            or np.unique(selected).size != selected.size
        ):
            raise ValueError("metric route indices are invalid")
        factor = self.jacobian_factor[:, selected]
        return validate_metric(factor.T @ factor / self.jacobian_factor.shape[0])

    def damage_for_indices(
        self, indices: Sequence[int] | np.ndarray
    ) -> tuple[float, float]:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return 0.0, 0.0
        if self.response_factor is not None:
            moments = self.action_moments(selected)
            diagonal = self.diagonal_action_moments(selected)
            return float(np.trace(moments.second_moment)), float(
                np.trace(diagonal.second_moment)
            )
        factor = self.jacobian_factor[:, selected]
        rowwise = np.sum(factor, axis=1)
        joint = float(np.mean(rowwise * rowwise))
        diagonal = float(np.sum(np.mean(factor * factor, axis=0)))
        return max(0.0, joint), max(0.0, diagonal)

    @property
    def metric(self) -> np.ndarray:
        if self.response_factor is not None and not self.factor_is_full_axis:
            raise RuntimeError(
                "refusing a native-route dense metric from a screened coupled response factor"
            )
        return self.metric_for_indices(np.arange(len(self.route_ids), dtype=np.int64))

    @cached_property
    def state_hash(self) -> str:
        return content_hash(
            {
                "version": "v2",
                "route_ids": self.route_ids,
                "route_costs": self.route_costs,
                "protected": self.protected.astype(np.uint8),
                "jacobian_factor": self.jacobian_factor,
                "response_moment_hash": (
                    None
                    if self.response_moments is None
                    else self.response_moments.content_hash
                ),
                "response_factor_hash": (
                    None
                    if self.response_factor is None
                    else self.response_factor.content_hash
                ),
                "response_factor_indices": self.response_factor_indices,
                "pullback_phenotype_moment_hash": (
                    None
                    if self.pullback_phenotype_moments is None
                    else self.pullback_phenotype_moments.content_hash
                ),
                "global_trust_geometry_hash": (
                    None
                    if self.global_trust_geometry is None
                    else self.global_trust_geometry.content_hash
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
                "measured_mask_hash": self.measured_mask_hash,
                "calibration_state_id": self.calibration_state_id,
                "driver_residualized": self.driver_residualized,
                "driver_hash": self.driver_hash,
                "source_outcome_hashes": self.source_outcome_hashes,
                "bootstrap_unmeasured": self.bootstrap_unmeasured,
            },
            prefix="irg-v2-state-",
        )

    def stable_projectors(
        self,
        *,
        route_indices=None,
        relative_gap: float = 1e-3,
        noise_floor: float = 1e-6,
    ) -> tuple[StableProjector, ...]:
        if route_indices is None:
            if self.response_factor is not None:
                indices = np.asarray(self.response_factor_indices, dtype=np.int64)
            elif len(self.route_ids) <= 4096:
                indices = np.arange(len(self.route_ids), dtype=np.int64)
            else:
                raise RuntimeError(
                    "refusing a full-route dense spectral decomposition; pass screened route_indices"
                )
        else:
            indices = np.asarray(route_indices, dtype=np.int64).reshape(-1)
        return stable_spectral_projectors(
            self.metric_for_indices(indices),
            relative_gap=relative_gap,
            noise_floor=noise_floor,
        )

    @cached_property
    def capability_hash(self) -> str:
        """Workflow-neutral capability identity.

        state_hash remains the byte-compatible historical v2 schema identity
        needed to read and compare old receipts. New capability lineage should
        bind this hash, which labels the neutral object without reinterpreting
        the old receipt.
        """

        return content_hash(
            {
                "capability": "screened_coupled_response_field_v1",
                "legacy_state_hash": self.state_hash,
            },
            prefix="coupled-response-field-",
        )


__all__ = ["CoupledResponseField"]
