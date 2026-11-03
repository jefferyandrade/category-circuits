"""Adaptive fixed-direction literal-J projection inference for v1.

This module is the bounded-memory statistical reference sibling of the exact
literal-J census.  For directions frozen before the witness stream, it estimates

``M_J z = E[j (j.T z)]``

from sealed finite-population prefixes and emits simultaneous conservative
Hoeffding-Serfling radii.  It does not retain raw literal-J rows and it grants
no C, F/Q/R, calibration, oracle, commit, production, Pareto, or KKT solution
sensitivity authority.  Its role is narrower: provide a typed finite-population
uncertainty set for fixed projected directions when a full census is too large.
The landed accumulator is NumPy-only and is not the route-scale H200 execution
owner; production still requires a tiled device/external-state implementation
of the same registered sufficient statistics and receipt contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, localcontext
from enum import Enum
from fractions import Fraction
import hashlib
import math
from numbers import Integral, Real
from typing import Final, Mapping, NoReturn, Sequence

import numpy as np

from .contracts import content_hash


LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_inference_v2"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_LEGACY_SCHEMA: Final = (
    "partition_free_v1_literal_j_fixed_direction_projection_inference_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE: Final = (
    "fixed_primal_literal_J_MJz_finite_population_projection_uncertainty_only_"
    "no_kkt_solution_sensitivity_C_FQR_calibration_oracle_commit_production_"
    "or_pareto_authority_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM: Final = (
    "class_balanced_prefix_mean_with_two_stratum_simultaneous_"
    "hoeffding_serfling_bounds_v2"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_LEGACY_ALGORITHM: Final = (
    "class_balanced_prefix_mean_with_simultaneous_hoeffding_serfling_bounds_v1"
)
LITERAL_J_FIXED_DIRECTION_CAUSAL_PROTOCOL: Final = (
    "D_G_selects_frozen_primal_directions_then_independent_or_predictable_D_W_"
    "prefix_estimates_literal_J_MJz_no_oracle_conditioning_v1"
)
LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION: Final = (
    "M_Jz=E[j*(j^Tz)]_for_frozen_primal_directions_v1"
)
_RECEIPT_PREFIX: Final = "partition-free-v1-literal-j-fixed-direction-projection-"
_ARRAY_PREFIX: Final = "partition-free-v1-literal-j-fixed-direction-array-"
_AXIS_PREFIX: Final = "partition-free-v1-literal-j-fixed-direction-axis-"


class V1LiteralJFixedDirectionProjectionRefusalCode(str, Enum):
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    DUPLICATE_IDENTITY = "duplicate_identity"
    INVALID_ROW = "invalid_row"
    EMPTY_PREFIX = "empty_prefix"
    FINALIZED = "finalized"


class V1LiteralJFixedDirectionProjectionRefusal(RuntimeError):
    """Typed fail-closed outcome for fixed-direction projection inference."""

    def __init__(
        self,
        code: V1LiteralJFixedDirectionProjectionRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJFixedDirectionProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJFixedDirectionProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be a sequence of identities",
        )
    try:
        result = tuple(values)
    except TypeError as error:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be finite: {error}",
        )
    if not result:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty",
        )
    for index, value in enumerate(result):
        _identity(value, name=f"{name}[{index}]")
    if len(set(result)) != len(result):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} contains duplicate identities",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_float(value: object, *, name: str) -> float:
    if not isinstance(value, Real):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be a real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be finite and positive",
        )
    return result


def _canonical_fp64_array(
    value: object,
    *,
    name: str,
    ndim: int,
    nonnegative: bool = False,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be an exact NumPy array",
        )
    source = value
    if source.dtype != np.dtype(np.float64) or source.ndim != ndim:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be a float64 array with ndim={ndim}",
        )
    result = np.array(source, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.all(np.isfinite(result)):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must contain only finite values",
        )
    if nonnegative and bool(np.any(result < 0.0)):
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    result.setflags(write=False)
    return result


def _fp64_array_hash(value: np.ndarray, *, role: str) -> str:
    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    payload = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
    return content_hash(
        {
            "schema": "literal_j_fixed_direction_fp64_array_bytes_v1",
            "role": role,
            "dtype": "<f8",
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": payload,
        },
        prefix=_ARRAY_PREFIX,
    )


def _axis_hash(role: str, payload: object) -> str:
    return content_hash(
        {"schema": "literal_j_fixed_direction_axis_v1", "role": role, "payload": payload},
        prefix=_AXIS_PREFIX,
    )


def _direction_hashes(directions: np.ndarray) -> tuple[str, ...]:
    return tuple(
        _fp64_array_hash(directions[index], role="fixed_primal_direction")
        for index in range(int(directions.shape[0]))
    )


def _output_hashes(output: np.ndarray) -> tuple[str, ...]:
    return tuple(
        _fp64_array_hash(output[index], role="literal_j_MJz_projection_point")
        for index in range(int(output.shape[0]))
    )


def _serfling_radius(
    *,
    value_range: float,
    observed: int,
    population: int,
    alpha: float,
) -> float:
    if population <= 0 or observed <= 0 or observed > population:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.EMPTY_PREFIX,
            "finite-population counts require 0 < observed <= population",
        )
    if observed >= population:
        return 0.0
    if not math.isfinite(value_range) or value_range < 0.0:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            "Serfling range must be finite and nonnegative",
        )
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _refuse(
            V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
            "Serfling alpha must lie strictly between zero and one",
        )
    if value_range == 0.0:
        return 0.0
    correction = Fraction(population - observed + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        span = Decimal.from_float(value_range)
        correction_d = Decimal(correction.numerator) / Decimal(
            correction.denominator
        )
        alpha_d = Decimal.from_float(alpha)
        log_factor = (Decimal(2) / alpha_d).ln().next_plus()
        radius = (
            span
            * (correction_d * log_factor / Decimal(2 * observed)).sqrt()
        ).next_plus()
    result = float(radius)
    if not math.isfinite(result):
        return math.inf
    if Decimal.from_float(result) < radius:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


@dataclass(frozen=True, slots=True)
class V1LiteralJFixedDirectionProjectionReceipt:
    """Array-free finite-population receipt for fixed literal-J projections."""

    inference_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    discovery_geometry_source_hash: str
    witness_stream_source_hash: str
    literal_j_stream_source_hash: str
    route_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    direction_hashes: tuple[str, ...]
    direction_axis_hash: str
    stratum_ids: tuple[str, str]
    population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    observed_context_axis_hashes: tuple[str, str]
    observed_draw_axis_hashes: tuple[str, str]
    observed_source_measurement_axis_hashes: tuple[str, str]
    output_vector_hashes: tuple[str, ...]
    quadratic_values: tuple[float, ...]
    output_axis_hash: str
    contribution_bound_vector_hashes: tuple[str, ...]
    quadratic_bound_values: tuple[float, ...]
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    alpha_spent: float
    sampling_radius: float
    quadratic_sampling_radius: float
    private_primal_direction_descriptor_axis_hash: str | None = None
    private_primal_direction_lineage_available: bool = False
    algorithm_id: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM
    causal_protocol: str = LITERAL_J_FIXED_DIRECTION_CAUSAL_PROTOCOL
    literal_j_equation: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION
    class_weights: tuple[float, float] = (0.5, 0.5)
    exact_finite_population_census: bool = False
    literal_j_rows_retained: bool = False
    directions_frozen_before_projection: bool = True
    independent_or_predictable_witness_stream: bool = True
    source_derived_for_fixed_primal_projection: bool = True
    fixed_z_only_no_kkt_solution_sensitivity: bool = True
    face_authority: bool = False
    shell_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA
    authority_scope: str = LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_receipt_hash", self._compute_hash())
        self.assert_integrity()

    @property
    def direction_count(self) -> int:
        return len(self.direction_ids)

    @property
    def receipt_hash(self) -> str:
        return self._receipt_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "inference_id": self.inference_id,
            "model_identity_hash": self.model_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_geometry_source_hash": self.discovery_geometry_source_hash,
            "witness_stream_source_hash": self.witness_stream_source_hash,
            "literal_j_stream_source_hash": self.literal_j_stream_source_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "direction_ids": self.direction_ids,
            "direction_hashes": self.direction_hashes,
            "direction_axis_hash": self.direction_axis_hash,
            "stratum_ids": self.stratum_ids,
            "population_counts": self.population_counts,
            "observed_counts": self.observed_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "observed_context_axis_hashes": self.observed_context_axis_hashes,
            "observed_draw_axis_hashes": self.observed_draw_axis_hashes,
            "observed_source_measurement_axis_hashes": (
                self.observed_source_measurement_axis_hashes
            ),
            "output_vector_hashes": self.output_vector_hashes,
            "quadratic_values": self.quadratic_values,
            "output_axis_hash": self.output_axis_hash,
            "contribution_bound_vector_hashes": self.contribution_bound_vector_hashes,
            "quadratic_bound_values": self.quadratic_bound_values,
            "simultaneous_statement_count": self.simultaneous_statement_count,
            "per_statement_two_sided_alpha": self.per_statement_two_sided_alpha,
            "alpha_spent": self.alpha_spent,
            "sampling_radius": self.sampling_radius,
            "quadratic_sampling_radius": self.quadratic_sampling_radius,
            "private_primal_direction_descriptor_axis_hash": (
                self.private_primal_direction_descriptor_axis_hash
            ),
            "private_primal_direction_lineage_available": (
                self.private_primal_direction_lineage_available
            ),
            "algorithm_id": self.algorithm_id,
            "causal_protocol": self.causal_protocol,
            "literal_j_equation": self.literal_j_equation,
            "class_weights": self.class_weights,
            "exact_finite_population_census": self.exact_finite_population_census,
            "literal_j_rows_retained": self.literal_j_rows_retained,
            "directions_frozen_before_projection": self.directions_frozen_before_projection,
            "independent_or_predictable_witness_stream": (
                self.independent_or_predictable_witness_stream
            ),
            "source_derived_for_fixed_primal_projection": (
                self.source_derived_for_fixed_primal_projection
            ),
            "fixed_z_only_no_kkt_solution_sensitivity": (
                self.fixed_z_only_no_kkt_solution_sensitivity
            ),
            "face_authority": self.face_authority,
            "shell_authority": self.shell_authority,
            "c_stream_authority": self.c_stream_authority,
            "fqr_authority": self.fqr_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "commit_authority": self.commit_authority,
            "production_authority": self.production_authority,
            "pareto_authority": self.pareto_authority,
        }

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_RECEIPT_PREFIX)

    def to_payload(self) -> dict[str, object]:
        return dict(self._payload())

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1LiteralJFixedDirectionProjectionReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("literal-J projection payload must be a mapping")
        values = dict(payload)
        for name in (
            "direction_ids",
            "direction_hashes",
            "stratum_ids",
            "population_counts",
            "observed_counts",
            "sealed_population_order_hashes",
            "observed_context_axis_hashes",
            "observed_draw_axis_hashes",
            "observed_source_measurement_axis_hashes",
            "output_vector_hashes",
            "quadratic_values",
            "contribution_bound_vector_hashes",
            "quadratic_bound_values",
            "class_weights",
        ):
            if name in values:
                values[name] = tuple(values[name])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def assert_integrity(self) -> None:
        for name in (
            "inference_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "discovery_geometry_source_hash",
            "witness_stream_source_hash",
            "literal_j_stream_source_hash",
            "route_axis_hash",
            "direction_axis_hash",
            "output_axis_hash",
            "algorithm_id",
            "causal_protocol",
            "literal_j_equation",
            "schema",
            "authority_scope",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be one nonempty string")
        supported_schema_algorithm = {
            LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA: (
                LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM
            ),
            LITERAL_J_FIXED_DIRECTION_PROJECTION_LEGACY_SCHEMA: (
                LITERAL_J_FIXED_DIRECTION_PROJECTION_LEGACY_ALGORITHM
            ),
        }
        if self.schema not in supported_schema_algorithm:
            raise ValueError("literal-J projection schema mismatch")
        if self.authority_scope != LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE:
            raise ValueError("literal-J projection authority scope mismatch")
        if self.algorithm_id != supported_schema_algorithm[self.schema]:
            raise ValueError("literal-J projection algorithm mismatch")
        if self.causal_protocol != LITERAL_J_FIXED_DIRECTION_CAUSAL_PROTOCOL:
            raise ValueError("literal-J projection causal protocol mismatch")
        if self.literal_j_equation != LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION:
            raise ValueError("literal-J projection equation mismatch")
        if type(self.proposal_generation) is not int or self.proposal_generation < 0:
            raise ValueError("proposal_generation must be nonnegative")
        if type(self.route_count) is not int or self.route_count <= 0:
            raise ValueError("route_count must be positive")
        tuple_fields = (
            "direction_ids",
            "direction_hashes",
            "stratum_ids",
            "population_counts",
            "observed_counts",
            "sealed_population_order_hashes",
            "observed_context_axis_hashes",
            "observed_draw_axis_hashes",
            "observed_source_measurement_axis_hashes",
            "output_vector_hashes",
            "quadratic_values",
            "contribution_bound_vector_hashes",
            "quadratic_bound_values",
            "class_weights",
        )
        if any(type(getattr(self, name)) is not tuple for name in tuple_fields):
            raise TypeError("literal-J projection axes must be exact tuples")
        if len(self.direction_ids) == 0 or len(set(self.direction_ids)) != len(
            self.direction_ids
        ):
            raise ValueError("direction IDs must be nonempty and unique")
        if len(self.direction_hashes) != self.direction_count:
            raise ValueError("direction hashes do not cover the direction axis")
        expected_direction_axis = _axis_hash(
            "fixed_primal_directions",
            tuple(zip(self.direction_ids, self.direction_hashes, strict=True)),
        )
        if self.direction_axis_hash != expected_direction_axis:
            raise ValueError("direction axis hash mismatch")
        if len(self.stratum_ids) != 2 or len(set(self.stratum_ids)) != 2:
            raise ValueError("literal-J projection requires exactly two strata")
        for population, observed in zip(
            self.population_counts, self.observed_counts, strict=True
        ):
            if type(population) is not int or population <= 0:
                raise ValueError("stratum populations must be positive")
            if type(observed) is not int or not 0 < observed <= population:
                raise ValueError("observed stratum counts must be nonempty prefixes")
        for name in (
            "sealed_population_order_hashes",
            "observed_context_axis_hashes",
            "observed_draw_axis_hashes",
            "observed_source_measurement_axis_hashes",
        ):
            values = getattr(self, name)
            if len(values) != 2 or any(type(item) is not str or not item for item in values):
                raise ValueError(f"{name} must bind both strata")
        if len(self.output_vector_hashes) != self.direction_count:
            raise ValueError("output hashes do not cover the direction axis")
        if len(self.quadratic_values) != self.direction_count:
            raise ValueError("quadratic values do not cover the direction axis")
        if len(self.contribution_bound_vector_hashes) != self.direction_count:
            raise ValueError("contribution bound hashes do not cover directions")
        if len(self.quadratic_bound_values) != self.direction_count:
            raise ValueError("quadratic bounds do not cover directions")
        for value in (*self.quadratic_values, *self.quadratic_bound_values):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("quadratic values and bounds must be nonnegative")
        expected_output_axis = _axis_hash(
            "literal_j_fixed_direction_outputs",
            tuple(
                zip(
                    self.direction_ids,
                    self.output_vector_hashes,
                    self.quadratic_values,
                    strict=True,
                )
            ),
        )
        if self.output_axis_hash != expected_output_axis:
            raise ValueError("output axis hash mismatch")
        stratum_multiplier = (
            2 if self.schema == LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA else 1
        )
        if type(self.simultaneous_statement_count) is not int or (
            self.simultaneous_statement_count
            != stratum_multiplier * self.direction_count * (self.route_count + 1)
        ):
            raise ValueError("simultaneous statement count is not canonical")
        for name in (
            "per_statement_two_sided_alpha",
            "alpha_spent",
            "sampling_radius",
            "quadratic_sampling_radius",
        ):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.per_statement_two_sided_alpha < 1.0:
            raise ValueError("per-statement alpha must lie in (0, 1)")
        if (
            not 0.0 < self.alpha_spent < 1.0
            or self.per_statement_two_sided_alpha
            != self.alpha_spent / float(self.simultaneous_statement_count)
        ):
            raise ValueError(
                "per-statement alpha does not replay the complete registered family"
            )
        if self.private_primal_direction_descriptor_axis_hash is not None:
            if (
                type(self.private_primal_direction_descriptor_axis_hash) is not str
                or not self.private_primal_direction_descriptor_axis_hash
            ):
                raise ValueError(
                    "private-primal direction descriptor axis hash must be nonempty"
                )
            if self.private_primal_direction_lineage_available is not True:
                raise ValueError(
                    "private-primal descriptor hash requires available lineage"
                )
        elif self.private_primal_direction_lineage_available is not False:
            raise ValueError(
                "private-primal lineage cannot be available without its descriptor axis"
            )
        if self.class_weights != (0.5, 0.5):
            raise ValueError("canonical literal-J projection uses equal strata")
        required_false = (
            self.exact_finite_population_census,
            self.literal_j_rows_retained,
            self.face_authority,
            self.shell_authority,
            self.c_stream_authority,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        required_true = (
            self.directions_frozen_before_projection,
            self.independent_or_predictable_witness_stream,
            self.source_derived_for_fixed_primal_projection,
            self.fixed_z_only_no_kkt_solution_sensitivity,
        )
        if any(value is not False for value in required_false) or any(
            value is not True for value in required_true
        ):
            raise PermissionError("literal-J projection crossed its authority boundary")
        if self._receipt_hash != self._compute_hash():
            raise ValueError("literal-J projection receipt hash mismatch")


class V1LiteralJFixedDirectionProjectionAccumulator:
    """Streaming prefix accumulator for fixed-direction ``M_J z`` inference."""

    __slots__ = (
        "_inference_id",
        "_model_identity_hash",
        "_chart_identity_hash",
        "_current_mask_hash",
        "_proposal_generation",
        "_discovery_geometry_source_hash",
        "_witness_stream_source_hash",
        "_literal_j_stream_source_hash",
        "_route_axis_hash",
        "_route_count",
        "_direction_ids",
        "_directions",
        "_direction_hashes",
        "_direction_axis_hash",
        "_stratum_ids",
        "_sealed_context_ids",
        "_sealed_order_hashes",
        "_literal_j_abs_bound",
        "_projection_abs_bounds",
        "_contribution_abs_bounds",
        "_quadratic_abs_bounds",
        "_familywise_alpha",
        "_private_primal_direction_descriptor_axis_hash",
        "_next_indices",
        "_seen_context_ids",
        "_seen_draw_ids",
        "_seen_source_measurement_ids",
        "_stratum_draw_ids",
        "_stratum_source_ids",
        "_sum_operator",
        "_sum_quadratic",
        "_finalized",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        inference_id: str,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        discovery_geometry_source_hash: str,
        witness_stream_source_hash: str,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
        route_count: int,
        direction_ids: Sequence[str],
        directions: np.ndarray,
        literal_j_abs_bound: np.ndarray,
        projection_abs_bounds: np.ndarray,
        stratum_ids: Sequence[str],
        sealed_context_ids_by_stratum: Sequence[Sequence[str]],
        familywise_alpha: float,
        private_primal_direction_descriptor_axis_hash: str | None = None,
    ) -> None:
        self._inference_id = _identity(inference_id, name="inference_id")
        self._model_identity_hash = _identity(model_identity_hash, name="model_identity_hash")
        self._chart_identity_hash = _identity(chart_identity_hash, name="chart_identity_hash")
        self._current_mask_hash = _identity(current_mask_hash, name="current_mask_hash")
        self._proposal_generation = _nonnegative_int(
            proposal_generation, name="proposal_generation"
        )
        self._discovery_geometry_source_hash = _identity(
            discovery_geometry_source_hash, name="discovery_geometry_source_hash"
        )
        self._witness_stream_source_hash = _identity(
            witness_stream_source_hash, name="witness_stream_source_hash"
        )
        self._literal_j_stream_source_hash = _identity(
            literal_j_stream_source_hash, name="literal_j_stream_source_hash"
        )
        self._route_axis_hash = _identity(route_axis_hash, name="route_axis_hash")
        self._route_count = _nonnegative_int(route_count, name="route_count")
        if self._route_count <= 0:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "route_count must be positive",
            )
        self._direction_ids = _identity_tuple(direction_ids, name="direction_ids")
        directions_array = _canonical_fp64_array(directions, name="directions", ndim=2)
        if directions_array.shape != (len(self._direction_ids), self._route_count):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "directions must have shape (direction_count, route_count)",
            )
        self._directions = directions_array
        self._direction_hashes = _direction_hashes(directions_array)
        self._direction_axis_hash = _axis_hash(
            "fixed_primal_directions",
            tuple(zip(self._direction_ids, self._direction_hashes, strict=True)),
        )
        stratum_tuple = _identity_tuple(stratum_ids, name="stratum_ids")
        if len(stratum_tuple) != 2:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "projection inference requires exactly two strata",
            )
        self._stratum_ids = (stratum_tuple[0], stratum_tuple[1])
        if isinstance(sealed_context_ids_by_stratum, (str, bytes)):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "sealed contexts must contain two ordered strata",
            )
        sealed = tuple(tuple(row) for row in sealed_context_ids_by_stratum)
        if len(sealed) != 2:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "sealed contexts must contain exactly two strata",
            )
        first = _identity_tuple(sealed[0], name="sealed_contexts[0]")
        second = _identity_tuple(sealed[1], name="sealed_contexts[1]")
        if set(first).intersection(second):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "context identities must be unique across strata",
            )
        self._sealed_context_ids = (first, second)
        self._sealed_order_hashes = (
            _axis_hash(
                "sealed_witness_population_order",
                {"stratum_id": self._stratum_ids[0], "context_ids": first},
            ),
            _axis_hash(
                "sealed_witness_population_order",
                {"stratum_id": self._stratum_ids[1], "context_ids": second},
            ),
        )
        self._literal_j_abs_bound = _canonical_fp64_array(
            literal_j_abs_bound,
            name="literal_j_abs_bound",
            ndim=1,
            nonnegative=True,
        )
        if self._literal_j_abs_bound.shape != (self._route_count,):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "literal_j_abs_bound must cover the route axis",
            )
        self._projection_abs_bounds = _canonical_fp64_array(
            projection_abs_bounds,
            name="projection_abs_bounds",
            ndim=1,
            nonnegative=True,
        )
        if self._projection_abs_bounds.shape != (len(self._direction_ids),):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "projection_abs_bounds must cover the direction axis",
            )
        self._contribution_abs_bounds = (
            self._projection_abs_bounds[:, None] * self._literal_j_abs_bound[None, :]
        )
        self._contribution_abs_bounds.setflags(write=False)
        self._quadratic_abs_bounds = self._projection_abs_bounds * self._projection_abs_bounds
        self._quadratic_abs_bounds.setflags(write=False)
        self._familywise_alpha = _positive_float(familywise_alpha, name="familywise_alpha")
        if self._familywise_alpha >= 1.0:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_DESIGN,
                "familywise_alpha must be smaller than one",
            )
        self._private_primal_direction_descriptor_axis_hash = (
            None
            if private_primal_direction_descriptor_axis_hash is None
            else _identity(
                private_primal_direction_descriptor_axis_hash,
                name="private_primal_direction_descriptor_axis_hash",
            )
        )
        direction_count = len(self._direction_ids)
        self._next_indices = [0, 0]
        self._seen_context_ids: set[str] = set()
        self._seen_draw_ids: set[str] = set()
        self._seen_source_measurement_ids: set[str] = set()
        self._stratum_draw_ids: list[list[str]] = [[], []]
        self._stratum_source_ids: list[list[str]] = [[], []]
        self._sum_operator = np.zeros(
            (2, direction_count, self._route_count), dtype=np.dtype("<f8")
        )
        self._sum_quadratic = np.zeros((2, direction_count), dtype=np.dtype("<f8"))
        self._finalized = False

    @property
    def observed_counts(self) -> tuple[int, int]:
        return (int(self._next_indices[0]), int(self._next_indices[1]))

    def update(
        self,
        *,
        stratum_id: str,
        context_id: str,
        draw_id: str,
        source_measurement_id: str,
        literal_j_row: np.ndarray,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
    ) -> None:
        if self._finalized:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.FINALIZED,
                "projection accumulator already finalized",
            )
        if (
            _identity(model_identity_hash, name="model_identity_hash")
            != self._model_identity_hash
            or _identity(chart_identity_hash, name="chart_identity_hash")
            != self._chart_identity_hash
            or _identity(current_mask_hash, name="current_mask_hash")
            != self._current_mask_hash
            or _nonnegative_int(proposal_generation, name="proposal_generation")
            != self._proposal_generation
            or _identity(literal_j_stream_source_hash, name="literal_j_stream_source_hash")
            != self._literal_j_stream_source_hash
            or _identity(route_axis_hash, name="route_axis_hash")
            != self._route_axis_hash
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.STALE_IDENTITY,
                "literal-J row left the frozen projection chart",
            )
        if stratum_id not in self._stratum_ids:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.ORDER_MISMATCH,
                "unknown projection stratum",
            )
        stratum_index = self._stratum_ids.index(stratum_id)
        next_index = self._next_indices[stratum_index]
        expected_order = self._sealed_context_ids[stratum_index]
        if next_index >= len(expected_order) or context_id != expected_order[next_index]:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.ORDER_MISMATCH,
                "literal-J witness contexts must be consumed in sealed prefix order",
            )
        context = _identity(context_id, name="context_id")
        draw = _identity(draw_id, name="draw_id")
        source = _identity(source_measurement_id, name="source_measurement_id")
        if (
            context in self._seen_context_ids
            or draw in self._seen_draw_ids
            or source in self._seen_source_measurement_ids
        ):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.DUPLICATE_IDENTITY,
                "literal-J projection witness identities must be unique",
            )
        row = _canonical_fp64_array(literal_j_row, name="literal_j_row", ndim=1)
        if row.shape != (self._route_count,):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_ROW,
                "literal-J row shape differs from route axis",
            )
        if bool(np.any(np.abs(row) > self._literal_j_abs_bound)):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_ROW,
                "literal-J row exceeds its pre-outcome absolute support bound",
            )
        projections = self._directions @ row
        if bool(np.any(np.abs(projections) > self._projection_abs_bounds)):
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.INVALID_ROW,
                "literal-J projection exceeds its pre-outcome absolute support bound",
            )
        self._sum_operator[stratum_index] += projections[:, None] * row[None, :]
        self._sum_quadratic[stratum_index] += projections * projections
        self._next_indices[stratum_index] += 1
        self._seen_context_ids.add(context)
        self._seen_draw_ids.add(draw)
        self._seen_source_measurement_ids.add(source)
        self._stratum_draw_ids[stratum_index].append(draw)
        self._stratum_source_ids[stratum_index].append(source)

    def finalize_prefix(self) -> V1LiteralJFixedDirectionProjectionReceipt:
        if self._finalized:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.FINALIZED,
                "projection accumulator already finalized",
            )
        observed = self.observed_counts
        if observed[0] <= 0 or observed[1] <= 0:
            _refuse(
                V1LiteralJFixedDirectionProjectionRefusalCode.EMPTY_PREFIX,
                "both strata require nonempty witness prefixes",
            )
        self._finalized = True
        means = np.empty(
            (2, len(self._direction_ids), self._route_count), dtype=np.dtype("<f8")
        )
        quad_means = np.empty((2, len(self._direction_ids)), dtype=np.dtype("<f8"))
        for stratum_index, count in enumerate(observed):
            means[stratum_index] = self._sum_operator[stratum_index] / float(count)
            quad_means[stratum_index] = self._sum_quadratic[stratum_index] / float(count)
        output = 0.5 * (means[0] + means[1])
        quadratic = 0.5 * (quad_means[0] + quad_means[1])
        output.setflags(write=False)
        quadratic.setflags(write=False)
        # Every route/operator and quadratic statement is made separately in
        # each of the two strata before their registered 1/2 mixture.  The
        # Bonferroni family therefore includes the stratum axis explicitly.
        statement_count = 2 * len(self._direction_ids) * (self._route_count + 1)
        per_statement_alpha = self._familywise_alpha / float(statement_count)
        route_radii = np.zeros_like(output)
        quadratic_radii = np.zeros_like(quadratic)
        populations = (len(self._sealed_context_ids[0]), len(self._sealed_context_ids[1]))
        for stratum_index, (count, population) in enumerate(zip(observed, populations, strict=True)):
            del stratum_index
            for direction_index in range(len(self._direction_ids)):
                quadratic_radii[direction_index] += 0.5 * _serfling_radius(
                    value_range=float(self._quadratic_abs_bounds[direction_index]),
                    observed=count,
                    population=population,
                    alpha=per_statement_alpha,
                )
                for route_index in range(self._route_count):
                    route_radii[direction_index, route_index] += 0.5 * _serfling_radius(
                        value_range=float(
                            2.0
                            * self._contribution_abs_bounds[
                                direction_index, route_index
                            ]
                        ),
                        observed=count,
                        population=population,
                        alpha=per_statement_alpha,
                    )
        sampling_radius = float(np.max(route_radii)) if route_radii.size else 0.0
        quadratic_radius = (
            float(np.max(quadratic_radii)) if quadratic_radii.size else 0.0
        )
        output_hashes = _output_hashes(output)
        output_axis = _axis_hash(
            "literal_j_fixed_direction_outputs",
            tuple(
                zip(
                    self._direction_ids,
                    output_hashes,
                    tuple(float(item) for item in quadratic.tolist()),
                    strict=True,
                )
            ),
        )
        return V1LiteralJFixedDirectionProjectionReceipt(
            inference_id=self._inference_id,
            model_identity_hash=self._model_identity_hash,
            chart_identity_hash=self._chart_identity_hash,
            current_mask_hash=self._current_mask_hash,
            proposal_generation=self._proposal_generation,
            discovery_geometry_source_hash=self._discovery_geometry_source_hash,
            witness_stream_source_hash=self._witness_stream_source_hash,
            literal_j_stream_source_hash=self._literal_j_stream_source_hash,
            route_axis_hash=self._route_axis_hash,
            route_count=self._route_count,
            direction_ids=self._direction_ids,
            direction_hashes=self._direction_hashes,
            direction_axis_hash=self._direction_axis_hash,
            stratum_ids=self._stratum_ids,
            population_counts=populations,
            observed_counts=observed,
            sealed_population_order_hashes=self._sealed_order_hashes,
            observed_context_axis_hashes=(
                _axis_hash(
                    "observed_witness_context_prefix",
                    {
                        "stratum_id": self._stratum_ids[0],
                        "context_ids": self._sealed_context_ids[0][: observed[0]],
                    },
                ),
                _axis_hash(
                    "observed_witness_context_prefix",
                    {
                        "stratum_id": self._stratum_ids[1],
                        "context_ids": self._sealed_context_ids[1][: observed[1]],
                    },
                ),
            ),
            observed_draw_axis_hashes=(
                _axis_hash(
                    "observed_witness_draw_prefix",
                    {"stratum_id": self._stratum_ids[0], "draw_ids": tuple(self._stratum_draw_ids[0])},
                ),
                _axis_hash(
                    "observed_witness_draw_prefix",
                    {"stratum_id": self._stratum_ids[1], "draw_ids": tuple(self._stratum_draw_ids[1])},
                ),
            ),
            observed_source_measurement_axis_hashes=(
                _axis_hash(
                    "observed_witness_source_prefix",
                    {
                        "stratum_id": self._stratum_ids[0],
                        "source_measurement_ids": tuple(self._stratum_source_ids[0]),
                    },
                ),
                _axis_hash(
                    "observed_witness_source_prefix",
                    {
                        "stratum_id": self._stratum_ids[1],
                        "source_measurement_ids": tuple(self._stratum_source_ids[1]),
                    },
                ),
            ),
            output_vector_hashes=output_hashes,
            quadratic_values=tuple(float(item) for item in quadratic.tolist()),
            output_axis_hash=output_axis,
            contribution_bound_vector_hashes=_output_hashes(
                self._contribution_abs_bounds
            ),
            quadratic_bound_values=tuple(
                float(item) for item in self._quadratic_abs_bounds.tolist()
            ),
            simultaneous_statement_count=statement_count,
            per_statement_two_sided_alpha=float(per_statement_alpha),
            alpha_spent=float(self._familywise_alpha),
            sampling_radius=sampling_radius,
            quadratic_sampling_radius=quadratic_radius,
            private_primal_direction_descriptor_axis_hash=(
                self._private_primal_direction_descriptor_axis_hash
            ),
            private_primal_direction_lineage_available=(
                self._private_primal_direction_descriptor_axis_hash is not None
            ),
        )


__all__ = [
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_ALGORITHM",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_AUTHORITY_SCOPE",
    "LITERAL_J_FIXED_DIRECTION_CAUSAL_PROTOCOL",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_EQUATION",
    "LITERAL_J_FIXED_DIRECTION_PROJECTION_SCHEMA",
    "V1LiteralJFixedDirectionProjectionAccumulator",
    "V1LiteralJFixedDirectionProjectionReceipt",
    "V1LiteralJFixedDirectionProjectionRefusal",
    "V1LiteralJFixedDirectionProjectionRefusalCode",
]
