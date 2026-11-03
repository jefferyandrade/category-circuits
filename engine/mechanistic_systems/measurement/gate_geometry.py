"""Gate-aware local-regime diagnostics on the physical route axis.

The statistics in this module are *screening and characterization evidence*.
They distinguish local drive, regime-preserving evidence, and singleton
activation-boundary crossings without claiming that a local crossing is a
task-relevant gate.  Control probes or exact finite-mask evidence are required
for that stronger claim.

Dense ``context × route × spatial`` tensors are deliberately absent from the
public contract.  Producers accumulate the sufficient statistics while an
ephemeral route-resolved local contribution chunk is resident on the device,
then retain only the route-axis arrays below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash


_MASS_FIELDS = (
    "observation_mass",
    "local_active_mass",
    "baseline_open_mass",
    "regime_preserving_inhibitory_mass",
    "suppressive_gate_opening_mass",
    "enabling_gate_closing_mass",
    "still_closed_active_mass",
    "crossing_risk_sum",
    "crossing_risk_max",
    "regime_preserving_inhibitory_effect_mass",
    "gate_opening_effect_mass",
    "gate_closing_effect_mass",
)

_SMOOTH_FIELDS = (
    "observation_mass",
    "local_active_mass",
    "baseline_gain_sum",
    "counterfactual_gain_sum",
    "absolute_gain_change_sum",
    "absolute_gain_change_max",
    "transmission_modulation_energy",
    "activation_effect_mass",
    "regime_preserving_inhibitory_effect_mass",
)


def _readonly_route_array(
    value: Sequence[float] | np.ndarray,
    *,
    route_count: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if (
        array.shape != (route_count,)
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
    ):
        raise ValueError(f"{name} must be finite and nonnegative on the route axis")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class LocalRegimeRouteSummary:
    """Compact ReLU-regime sufficient statistics for physical routes.

    ``observation_mass`` is the number (or frozen-design weight) of aligned
    destination sites inspected for each route.  A zero value means that the
    route had no eligible activation-boundary geometry; it does *not* mean the
    route was observed and never crossed.

    Effect masses are in local activation units:

    ``|relu(a-c)-relu(a)|``

    where ``a`` is the destination preactivation and ``c`` is the route or
    declared population contribution being removed.  They are not global task
    response effects.
    """

    route_ids: tuple[str, ...]
    observation_mass: np.ndarray
    local_active_mass: np.ndarray
    baseline_open_mass: np.ndarray
    regime_preserving_inhibitory_mass: np.ndarray
    suppressive_gate_opening_mass: np.ndarray
    enabling_gate_closing_mass: np.ndarray
    still_closed_active_mass: np.ndarray
    crossing_risk_sum: np.ndarray
    crossing_risk_max: np.ndarray
    regime_preserving_inhibitory_effect_mass: np.ndarray
    gate_opening_effect_mass: np.ndarray
    gate_closing_effect_mass: np.ndarray
    source_context_count: int
    context_axis_hash: str
    measured_mask_hash: str
    source_measurement_hash: str
    geometry_blocks: tuple[tuple[int, int, str], ...]
    observation_measure_id: str = (
        "uniform_delivered_context_x_destination_site_v1"
    )
    scope: str = "singleton_route_ablation"
    activation_kind: str = "relu"
    schema: str = "irg_local_regime_route_summary_v1"

    def __post_init__(self) -> None:
        routes = tuple(map(str, self.route_ids))
        if (
            not routes
            or len(set(routes)) != len(routes)
            or int(self.source_context_count) <= 0
            or not self.context_axis_hash
            or not self.measured_mask_hash
            or not self.source_measurement_hash
            or not self.observation_measure_id
            or self.scope not in {
                "singleton_route_ablation",
                "declared_route_population_ablation",
            }
            or self.activation_kind != "relu"
        ):
            raise ValueError("local regime summary has incomplete identity")
        arrays: dict[str, np.ndarray] = {}
        for name in _MASS_FIELDS:
            arrays[name] = _readonly_route_array(
                getattr(self, name),
                route_count=len(routes),
                name=name,
            )
        observed = arrays["observation_mass"]
        for name in (
            "local_active_mass",
            "baseline_open_mass",
            "regime_preserving_inhibitory_mass",
            "suppressive_gate_opening_mass",
            "enabling_gate_closing_mass",
            "still_closed_active_mass",
        ):
            if np.any(arrays[name] > observed + 1e-9):
                raise ValueError(f"{name} may not exceed observation mass")
        blocks = tuple(
            (int(start), int(stop), str(identity))
            for start, stop, identity in self.geometry_blocks
        )
        prior_stop = 0
        for start, stop, identity in blocks:
            if (
                start < prior_stop
                or stop <= start
                or stop > len(routes)
                or not identity
            ):
                raise ValueError("local regime geometry blocks are invalid")
            prior_stop = stop
        covered = arrays["observation_mass"] > 0.0
        covered_by_block = np.zeros(len(routes), dtype=bool)
        for start, stop, _identity in blocks:
            covered_by_block[start:stop] = True
        if not np.array_equal(covered, covered_by_block):
            raise ValueError(
                "local regime geometry blocks must exactly identify covered routes"
            )
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "source_context_count", int(self.source_context_count))
        object.__setattr__(self, "geometry_blocks", blocks)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def coverage_mask(self) -> np.ndarray:
        return self.observation_mass > 0.0

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.coverage_mask))

    def _frequency(self, numerator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            self.observation_mass,
            out=np.zeros_like(numerator),
            where=self.observation_mass > 0.0,
        )

    @property
    def local_active_frequency(self) -> np.ndarray:
        return self._frequency(self.local_active_mass)

    @property
    def baseline_open_frequency(self) -> np.ndarray:
        return self._frequency(self.baseline_open_mass)

    @property
    def regime_preserving_inhibitory_frequency(self) -> np.ndarray:
        return self._frequency(self.regime_preserving_inhibitory_mass)

    @property
    def suppressive_gate_opening_frequency(self) -> np.ndarray:
        return self._frequency(self.suppressive_gate_opening_mass)

    @property
    def enabling_gate_closing_frequency(self) -> np.ndarray:
        return self._frequency(self.enabling_gate_closing_mass)

    @property
    def mean_crossing_risk(self) -> np.ndarray:
        return self._frequency(self.crossing_risk_sum)

    @property
    def mean_regime_preserving_inhibitory_effect(self) -> np.ndarray:
        return self._frequency(
            self.regime_preserving_inhibitory_effect_mass
        )

    @property
    def mean_gate_opening_effect(self) -> np.ndarray:
        return self._frequency(self.gate_opening_effect_mass)

    @property
    def mean_gate_closing_effect(self) -> np.ndarray:
        return self._frequency(self.gate_closing_effect_mass)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-local-regime-summary-")

    def pack(self) -> tuple[np.ndarray, dict[str, object]]:
        values = np.stack(
            [np.asarray(getattr(self, name), dtype=np.float64) for name in _MASS_FIELDS],
            axis=0,
        )
        return np.ascontiguousarray(values), {
            "schema": self.schema,
            "route_ids": self.route_ids,
            "source_context_count": self.source_context_count,
            "context_axis_hash": self.context_axis_hash,
            "measured_mask_hash": self.measured_mask_hash,
            "source_measurement_hash": self.source_measurement_hash,
            "geometry_blocks": self.geometry_blocks,
            "observation_measure_id": self.observation_measure_id,
            "scope": self.scope,
            "activation_kind": self.activation_kind,
            "row_names": _MASS_FIELDS,
        }

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, object],
    ) -> "LocalRegimeRouteSummary":
        array = np.asarray(values, dtype=np.float64)
        names = tuple(map(str, layout.get("row_names", ())))
        if (
            layout.get("schema") != "irg_local_regime_route_summary_v1"
            or names != _MASS_FIELDS
            or array.ndim != 2
            or array.shape[0] != len(_MASS_FIELDS)
        ):
            raise ValueError("measurement is not a local regime route summary")
        fields = {
            name: np.asarray(array[index], dtype=np.float64)
            for index, name in enumerate(_MASS_FIELDS)
        }
        return cls(
            route_ids=tuple(layout["route_ids"]),  # type: ignore[arg-type]
            source_context_count=int(layout["source_context_count"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            measured_mask_hash=str(layout["measured_mask_hash"]),
            source_measurement_hash=str(layout["source_measurement_hash"]),
            geometry_blocks=tuple(
                tuple(item) for item in layout["geometry_blocks"]  # type: ignore[arg-type]
            ),  # type: ignore[arg-type]
            observation_measure_id=str(
                layout.get(
                    "observation_measure_id",
                    "uniform_delivered_context_x_destination_site_v1",
                )
            ),
            scope=str(layout["scope"]),
            activation_kind=str(layout["activation_kind"]),
            **fields,
        )


@dataclass(frozen=True, slots=True)
class SmoothTransmissionRouteSummary:
    """Compact finite-gain diagnostics for a smooth destination activation."""

    route_ids: tuple[str, ...]
    observation_mass: np.ndarray
    local_active_mass: np.ndarray
    baseline_gain_sum: np.ndarray
    counterfactual_gain_sum: np.ndarray
    absolute_gain_change_sum: np.ndarray
    absolute_gain_change_max: np.ndarray
    transmission_modulation_energy: np.ndarray
    activation_effect_mass: np.ndarray
    regime_preserving_inhibitory_effect_mass: np.ndarray
    source_context_count: int
    context_axis_hash: str
    measured_mask_hash: str
    source_measurement_hash: str
    geometry_blocks: tuple[tuple[int, int, str], ...]
    activation_kind: str
    observation_measure_id: str = (
        "uniform_delivered_context_x_destination_site_v1"
    )
    scope: str = "singleton_route_ablation"
    schema: str = "irg_smooth_transmission_route_summary_v1"

    def __post_init__(self) -> None:
        routes = tuple(map(str, self.route_ids))
        if (
            not routes
            or len(set(routes)) != len(routes)
            or int(self.source_context_count) <= 0
            or not self.context_axis_hash
            or not self.measured_mask_hash
            or not self.source_measurement_hash
            or self.activation_kind
            not in {"gelu", "silu", "sigmoid", "tanh", "softplus"}
        ):
            raise ValueError("smooth transmission summary has incomplete identity")
        arrays = {
            name: _readonly_route_array(
                getattr(self, name), route_count=len(routes), name=name
            )
            for name in _SMOOTH_FIELDS
        }
        if np.any(arrays["local_active_mass"] > arrays["observation_mass"] + 1e-9):
            raise ValueError("smooth local activity may not exceed observation mass")
        blocks = tuple(
            (int(start), int(stop), str(identity))
            for start, stop, identity in self.geometry_blocks
        )
        covered = arrays["observation_mass"] > 0.0
        covered_by_block = np.zeros(len(routes), dtype=bool)
        prior_stop = 0
        for start, stop, identity in blocks:
            if start < prior_stop or stop <= start or stop > len(routes) or not identity:
                raise ValueError("smooth transmission geometry blocks are invalid")
            covered_by_block[start:stop] = True
            prior_stop = stop
        if not np.array_equal(covered, covered_by_block):
            raise ValueError("smooth geometry blocks must exactly identify coverage")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "source_context_count", int(self.source_context_count))
        object.__setattr__(self, "geometry_blocks", blocks)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def coverage_mask(self) -> np.ndarray:
        return self.observation_mass > 0.0

    @property
    def coverage_fraction(self) -> float:
        return float(np.mean(self.coverage_mask))

    def _mean(self, numerator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            self.observation_mass,
            out=np.zeros_like(numerator),
            where=self.observation_mass > 0.0,
        )

    @property
    def mean_baseline_gain(self) -> np.ndarray:
        return self._mean(self.baseline_gain_sum)

    @property
    def mean_counterfactual_gain(self) -> np.ndarray:
        return self._mean(self.counterfactual_gain_sum)

    @property
    def mean_absolute_gain_change(self) -> np.ndarray:
        return self._mean(self.absolute_gain_change_sum)

    @property
    def rms_gain_change(self) -> np.ndarray:
        return np.sqrt(self._mean(self.transmission_modulation_energy))

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-smooth-transmission-summary-")

    def pack(self) -> tuple[np.ndarray, dict[str, object]]:
        values = np.stack(
            [np.asarray(getattr(self, name), dtype=np.float64) for name in _SMOOTH_FIELDS],
            axis=0,
        )
        return np.ascontiguousarray(values), {
            "schema": self.schema,
            "route_ids": self.route_ids,
            "source_context_count": self.source_context_count,
            "context_axis_hash": self.context_axis_hash,
            "measured_mask_hash": self.measured_mask_hash,
            "source_measurement_hash": self.source_measurement_hash,
            "geometry_blocks": self.geometry_blocks,
            "observation_measure_id": self.observation_measure_id,
            "scope": self.scope,
            "activation_kind": self.activation_kind,
            "row_names": _SMOOTH_FIELDS,
        }

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, object],
    ) -> "SmoothTransmissionRouteSummary":
        array = np.asarray(values, dtype=np.float64)
        names = tuple(map(str, layout.get("row_names", ())))
        if (
            layout.get("schema")
            != "irg_smooth_transmission_route_summary_v1"
            or names != _SMOOTH_FIELDS
            or array.ndim != 2
            or array.shape[0] != len(_SMOOTH_FIELDS)
        ):
            raise ValueError(
                "measurement is not a smooth transmission route summary"
            )
        fields = {
            name: np.asarray(array[index], dtype=np.float64)
            for index, name in enumerate(_SMOOTH_FIELDS)
        }
        return cls(
            route_ids=tuple(layout["route_ids"]),  # type: ignore[arg-type]
            source_context_count=int(layout["source_context_count"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            measured_mask_hash=str(layout["measured_mask_hash"]),
            source_measurement_hash=str(layout["source_measurement_hash"]),
            geometry_blocks=tuple(
                tuple(item)
                for item in layout["geometry_blocks"]  # type: ignore[arg-type]
            ),  # type: ignore[arg-type]
            observation_measure_id=str(
                layout.get(
                    "observation_measure_id",
                    "uniform_delivered_context_x_destination_site_v1",
                )
            ),
            scope=str(layout["scope"]),
            activation_kind=str(layout["activation_kind"]),
            **fields,
        )


__all__ = ["LocalRegimeRouteSummary", "SmoothTransmissionRouteSummary"]

