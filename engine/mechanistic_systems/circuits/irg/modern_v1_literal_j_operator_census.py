"""Exact two-stratum census owner for the literal-J response operator.

This module owns one deliberately narrow numerical seam.  For fixed directions
``z`` on one frozen route axis it streams signed literal relaxed-gate Jacobian
rows ``j`` and accumulates

``M_J z = E[j (j.T z)]`` and ``Q(z) = E[(j.T z)**2]``.

The expectation is the equal-weight average of two exact finite-population
stratum means.  Raw Jacobian rows are never retained.  The array-backed result
is an ephemeral, nonserializable action state; its durable receipt is array
free and has no mask-selection or commit authority.  The NumPy backend is a
small-domain reference.  The optional torch backend is an explicitly selected
device execution path, not a CPU fallback and not a production claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from numbers import Integral
from typing import Any, Final, Mapping, NoReturn, Sequence

import numpy as np

from .contracts import content_hash


LITERAL_J_OPERATOR_CENSUS_SCHEMA: Final = (
    "partition_free_v1_literal_j_operator_exact_census_v1"
)
LITERAL_J_OPERATOR_EQUATION: Final = "j=-partial_u_r_raw_at_frozen_chart_v1"
LITERAL_J_OPERATOR_CENSUS_AUTHORITY_SCOPE: Final = (
    "exact_two_stratum_literal_J_operator_sufficient_statistics_only_"
    "no_face_shell_C_FQR_calibration_oracle_commit_production_or_pareto_"
    "authority_v1"
)
_RECEIPT_PREFIX: Final = "partition-free-v1-literal-j-operator-census-"
_ARRAY_PREFIX: Final = "partition-free-v1-literal-j-fp64-array-"
_AXIS_PREFIX: Final = "partition-free-v1-literal-j-axis-"
_NUMPY_REFERENCE_ROUTE_LIMIT: Final = 65_536
_NUMPY_REFERENCE_DIRECTION_LIMIT: Final = 256
_ACTION_STATE_ISSUANCE_TOKEN: Final = object()


class V1LiteralJOperatorCensusRefusalCode(str, Enum):
    """Typed fail-closed outcomes for census construction and lifecycle."""

    INVALID_DESIGN = "invalid_design"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    REFERENCE_SCALE_EXCEEDED = "reference_scale_exceeded"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    DUPLICATE_IDENTITY = "duplicate_identity"
    INVALID_ROW = "invalid_row"
    CENSUS_INCOMPLETE = "census_incomplete"
    FINALIZED = "finalized"
    STATE_EVICTED = "state_evicted"
    STATE_TAMPERED = "state_tampered"


class V1LiteralJOperatorCensusRefusal(RuntimeError):
    """An exact literal-J census precondition was not established."""

    def __init__(
        self,
        code: V1LiteralJOperatorCensusRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJOperatorCensusRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJOperatorCensusRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_tuple(
    values: Sequence[str],
    *,
    name: str,
    nonempty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be a sequence of identities",
        )
    try:
        result = tuple(values)
    except TypeError as error:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be a finite sequence: {error}",
        )
    if nonempty and not result:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty",
        )
    for index, value in enumerate(result):
        _identity(value, name=f"{name}[{index}]")
    if len(set(result)) != len(result):
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} contains duplicate identities",
        )
    return result


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _canonical_fp64_array(
    value: object,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be an exact NumPy array",
        )
    source = value
    if source.dtype != np.dtype(np.float64):
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must have dtype float64",
        )
    if source.ndim != ndim:
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must be {ndim}-dimensional",
        )
    result = np.array(source, dtype=np.dtype("<f8"), order="C", copy=True)
    if not np.all(np.isfinite(result)):
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
            f"{name} must contain only finite values",
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
            "schema": "literal_j_fp64_array_bytes_v1",
            "role": role,
            "dtype": "<f8",
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": payload,
        },
        prefix=_ARRAY_PREFIX,
    )


def _direction_hashes(directions: np.ndarray) -> tuple[str, ...]:
    return tuple(
        _fp64_array_hash(directions[index], role="fixed_direction")
        for index in range(int(directions.shape[0]))
    )


def _output_hashes(output: np.ndarray) -> tuple[str, ...]:
    return tuple(
        _fp64_array_hash(output[index], role="literal_j_operator_output")
        for index in range(int(output.shape[0]))
    )


def _axis_hash(role: str, payload: object) -> str:
    return content_hash(
        {"schema": "literal_j_operator_axis_v1", "role": role, "payload": payload},
        prefix=_AXIS_PREFIX,
    )


def _torch_host_fp64(torch_module: Any, value: Any) -> np.ndarray:
    result = value.detach().to(device="cpu", dtype=torch_module.float64).numpy()
    result = np.array(result, dtype=np.dtype("<f8"), order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class V1LiteralJOperatorCensusReceipt:
    """Array-free receipt for one exact class-balanced literal-J census."""

    census_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
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
    context_event_axis_hash: str
    draw_event_axis_hash: str
    source_measurement_event_axis_hash: str
    ingestion_event_axis_hash: str
    output_vector_hashes: tuple[str, ...]
    quadratic_values: tuple[float, ...]
    output_axis_hash: str
    backend: str
    device: str
    accumulation_algorithm: str
    literal_j_equation: str = LITERAL_J_OPERATOR_EQUATION
    class_weights: tuple[float, float] = (0.5, 0.5)
    sampling_radius: float = 0.0
    alpha_spent: float = 0.0
    exact_finite_population_census: bool = True
    literal_j_rows_retained: bool = False
    directions_frozen_before_census: bool = True
    cpu_fallback_used: bool = False
    route_scale_external_state_allocator_integrated: bool = False
    face_authority: bool = False
    shell_authority: bool = False
    c_stream_authority: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_OPERATOR_CENSUS_SCHEMA
    authority_scope: str = LITERAL_J_OPERATOR_CENSUS_AUTHORITY_SCOPE
    _receipt_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_receipt_hash", self._compute_hash())
        self.assert_integrity()

    @property
    def direction_count(self) -> int:
        return len(self.direction_ids)

    @property
    def population_count(self) -> int:
        return int(self.population_counts[0] + self.population_counts[1])

    @property
    def receipt_hash(self) -> str:
        return self._receipt_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "authority_scope": self.authority_scope,
            "census_id": self.census_id,
            "model_identity_hash": self.model_identity_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
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
            "context_event_axis_hash": self.context_event_axis_hash,
            "draw_event_axis_hash": self.draw_event_axis_hash,
            "source_measurement_event_axis_hash": (
                self.source_measurement_event_axis_hash
            ),
            "ingestion_event_axis_hash": self.ingestion_event_axis_hash,
            "output_vector_hashes": self.output_vector_hashes,
            "quadratic_values": self.quadratic_values,
            "output_axis_hash": self.output_axis_hash,
            "backend": self.backend,
            "device": self.device,
            "accumulation_algorithm": self.accumulation_algorithm,
            "literal_j_equation": self.literal_j_equation,
            "class_weights": self.class_weights,
            "sampling_radius": self.sampling_radius,
            "alpha_spent": self.alpha_spent,
            "exact_finite_population_census": self.exact_finite_population_census,
            "literal_j_rows_retained": self.literal_j_rows_retained,
            "directions_frozen_before_census": self.directions_frozen_before_census,
            "cpu_fallback_used": self.cpu_fallback_used,
            "route_scale_external_state_allocator_integrated": (
                self.route_scale_external_state_allocator_integrated
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
    ) -> "V1LiteralJOperatorCensusReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("literal-J census payload must be a mapping")
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
            "class_weights",
        ):
            if name in values:
                values[name] = tuple(values[name])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def assert_integrity(self) -> None:
        """Replay all array-free invariants and the cached content identity."""

        for name in (
            "census_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "literal_j_stream_source_hash",
            "route_axis_hash",
            "direction_axis_hash",
            "context_event_axis_hash",
            "draw_event_axis_hash",
            "source_measurement_event_axis_hash",
            "ingestion_event_axis_hash",
            "output_axis_hash",
            "backend",
            "device",
            "accumulation_algorithm",
            "literal_j_equation",
            "schema",
            "authority_scope",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be one nonempty string")
        if self.schema != LITERAL_J_OPERATOR_CENSUS_SCHEMA:
            raise ValueError("literal-J census schema mismatch")
        if self.literal_j_equation != LITERAL_J_OPERATOR_EQUATION:
            raise ValueError("literal-J census equation/orientation mismatch")
        if self.authority_scope != LITERAL_J_OPERATOR_CENSUS_AUTHORITY_SCOPE:
            raise ValueError("literal-J census authority scope mismatch")
        if type(self.proposal_generation) is not int or self.proposal_generation < 0:
            raise ValueError("proposal_generation must be one nonnegative integer")
        if type(self.route_count) is not int or self.route_count <= 0:
            raise ValueError("route_count must be one positive integer")
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
            "class_weights",
        )
        if any(type(getattr(self, name)) is not tuple for name in tuple_fields):
            raise TypeError("durable census axes and scalar vectors must be tuples")
        if len(self.direction_ids) == 0 or len(set(self.direction_ids)) != len(
            self.direction_ids
        ):
            raise ValueError("direction_ids must be nonempty and unique")
        if any(type(item) is not str or not item for item in self.direction_ids):
            raise ValueError("direction_ids must contain identities")
        if len(self.direction_hashes) != self.direction_count or any(
            type(item) is not str or not item for item in self.direction_hashes
        ):
            raise ValueError("direction hashes do not cover the direction axis")
        expected_direction_axis = _axis_hash(
            "fixed_directions",
            tuple(zip(self.direction_ids, self.direction_hashes, strict=True)),
        )
        if self.direction_axis_hash != expected_direction_axis:
            raise ValueError("direction axis hash mismatch")
        if (
            len(self.stratum_ids) != 2
            or len(set(self.stratum_ids)) != 2
            or any(type(item) is not str or not item for item in self.stratum_ids)
        ):
            raise ValueError("exact census requires two distinct stratum identities")
        if len(self.population_counts) != 2 or len(self.observed_counts) != 2:
            raise ValueError("exact census requires two population counts")
        for population, observed in zip(
            self.population_counts, self.observed_counts, strict=True
        ):
            if type(population) is not int or population <= 0:
                raise ValueError("each stratum population must be positive")
            if type(observed) is not int or observed != population:
                raise ValueError("observed counts must equal declared populations")
        for name in (
            "sealed_population_order_hashes",
            "observed_context_axis_hashes",
            "observed_draw_axis_hashes",
            "observed_source_measurement_axis_hashes",
        ):
            values = getattr(self, name)
            if len(values) != 2 or any(
                type(item) is not str or not item for item in values
            ):
                raise ValueError(f"{name} must bind both strata")
        if len(self.output_vector_hashes) != self.direction_count or any(
            type(item) is not str or not item for item in self.output_vector_hashes
        ):
            raise ValueError("output hashes do not cover the direction axis")
        if len(self.quadratic_values) != self.direction_count:
            raise ValueError("quadratic values do not cover the direction axis")
        for value in self.quadratic_values:
            if type(value) is not float or not np.isfinite(value) or value < 0.0:
                raise ValueError("quadratic values must be finite nonnegative floats")
        expected_output_axis = _axis_hash(
            "literal_j_operator_outputs",
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
        if self.backend not in {"numpy_reference", "torch_device"}:
            raise ValueError("unregistered literal-J census backend")
        if self.backend == "numpy_reference" and self.device != "cpu":
            raise ValueError("NumPy reference census must bind the CPU device")
        if self.class_weights != (0.5, 0.5) or any(
            type(value) is not float for value in self.class_weights
        ):
            raise ValueError("canonical two-stratum census weights must be equal")
        if (
            type(self.sampling_radius) is not float
            or type(self.alpha_spent) is not float
            or self.sampling_radius != 0.0
            or self.alpha_spent != 0.0
        ):
            raise ValueError("exact census has zero sampling radius and zero alpha")
        if self.exact_finite_population_census is not True:
            raise ValueError("receipt must represent an exact finite-population census")
        if self.literal_j_rows_retained is not False:
            raise ValueError("literal-J raw rows may not be retained")
        if self.directions_frozen_before_census is not True:
            raise ValueError("directions must be frozen before the census")
        if self.cpu_fallback_used is not False:
            raise ValueError("this seam makes no CPU fallback claim")
        if self.route_scale_external_state_allocator_integrated is not False:
            raise ValueError("route-scale external-state integration is not landed")
        authority_values = (
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
        if any(value is not False for value in authority_values):
            raise PermissionError("literal-J operator census is authority-inert")
        if self._receipt_hash != self._compute_hash():
            raise ValueError("literal-J census receipt hash mismatch")


class V1LiteralJOperatorCensusActionState:
    """Ephemeral owner of array-backed exact-census sufficient statistics."""

    __slots__ = ("_receipt", "_operator_outputs", "_quadratic_values", "_torch")
    __hash__ = None

    def __init__(
        self,
        *,
        token: object,
        receipt: V1LiteralJOperatorCensusReceipt,
        operator_outputs: Any,
        quadratic_values: Any,
        torch_module: Any | None,
    ) -> None:
        if token is not _ACTION_STATE_ISSUANCE_TOKEN:
            raise PermissionError(
                "literal-J census action states are internally issued"
            )
        self._receipt = receipt
        self._operator_outputs = operator_outputs
        self._quadratic_values = quadratic_values
        self._torch = torch_module
        if self._torch is None:
            self._operator_outputs.setflags(write=False)
            self._quadratic_values.setflags(write=False)
        self.assert_integrity()

    @property
    def receipt(self) -> V1LiteralJOperatorCensusReceipt:
        return self._receipt

    @property
    def backend(self) -> str:
        return self._receipt.backend

    @property
    def device(self) -> str:
        return self._receipt.device

    @property
    def is_evicted(self) -> bool:
        return self._operator_outputs is None

    def __copy__(self) -> NoReturn:
        raise TypeError("literal-J census action state is noncopyable")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("literal-J census action state is noncopyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J census action state is nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J census action state is nonserializable")

    def copy_operator_outputs(self) -> Any:
        """Return a defensive copy without transferring lifecycle authority."""

        self.assert_integrity()
        if self._torch is None:
            return self._operator_outputs.copy()
        return self._operator_outputs.detach().clone()

    def copy_quadratic_values(self) -> Any:
        """Return a defensive copy of Q for inspection or downstream compute."""

        self.assert_integrity()
        if self._torch is None:
            return self._quadratic_values.copy()
        return self._quadratic_values.detach().clone()

    def evict(self) -> None:
        """Release the live arrays; the durable authority-inert receipt remains."""

        self._operator_outputs = None
        self._quadratic_values = None

    def _host_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_evicted:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STATE_EVICTED,
                "literal-J census action arrays have been evicted",
            )
        if self._torch is None:
            outputs = np.asarray(self._operator_outputs, dtype=np.dtype("<f8"))
            quadratic = np.asarray(self._quadratic_values, dtype=np.dtype("<f8"))
            return outputs, quadratic
        return (
            _torch_host_fp64(self._torch, self._operator_outputs),
            _torch_host_fp64(self._torch, self._quadratic_values),
        )

    def assert_integrity(self) -> None:
        """Rehash live outputs and refuse mutation of the ephemeral state."""

        self._receipt.assert_integrity()
        outputs, quadratic = self._host_arrays()
        expected_shape = (self._receipt.direction_count, self._receipt.route_count)
        if outputs.shape != expected_shape or quadratic.shape != (
            self._receipt.direction_count,
        ):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STATE_TAMPERED,
                "literal-J census action-state shapes left their receipt",
            )
        if not np.all(np.isfinite(outputs)) or not np.all(np.isfinite(quadratic)):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STATE_TAMPERED,
                "literal-J census action state contains nonfinite values",
            )
        if _output_hashes(outputs) != self._receipt.output_vector_hashes:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STATE_TAMPERED,
                "literal-J response-operator output hashes changed",
            )
        observed_quadratic = tuple(float(item) for item in quadratic.tolist())
        if observed_quadratic != self._receipt.quadratic_values:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STATE_TAMPERED,
                "literal-J quadratic values changed",
            )


class V1LiteralJOperatorCensusAccumulator:
    """Streaming exact-census sufficient-statistics accumulator."""

    __slots__ = (
        "_census_id",
        "_model_identity_hash",
        "_chart_identity_hash",
        "_current_mask_hash",
        "_proposal_generation",
        "_literal_j_stream_source_hash",
        "_route_axis_hash",
        "_route_count",
        "_direction_ids",
        "_direction_hashes",
        "_direction_axis_hash",
        "_stratum_ids",
        "_sealed_context_ids",
        "_sealed_order_hashes",
        "_next_indices",
        "_seen_context_ids",
        "_seen_draw_ids",
        "_seen_source_measurement_ids",
        "_context_events",
        "_draw_events",
        "_source_events",
        "_stratum_draw_ids",
        "_stratum_source_ids",
        "_backend",
        "_device",
        "_directions",
        "_sum_operator",
        "_sum_quadratic",
        "_torch",
        "_finalized",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        census_id: str,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
        route_count: int,
        direction_ids: Sequence[str],
        directions: np.ndarray,
        stratum_ids: Sequence[str],
        sealed_context_ids_by_stratum: Sequence[Sequence[str]],
        backend: str = "numpy_reference",
        torch_device: str | None = None,
    ) -> None:
        self._census_id = _identity(census_id, name="census_id")
        self._model_identity_hash = _identity(
            model_identity_hash, name="model_identity_hash"
        )
        self._chart_identity_hash = _identity(
            chart_identity_hash, name="chart_identity_hash"
        )
        self._current_mask_hash = _identity(current_mask_hash, name="current_mask_hash")
        self._proposal_generation = _exact_nonnegative_int(
            proposal_generation, name="proposal_generation"
        )
        self._literal_j_stream_source_hash = _identity(
            literal_j_stream_source_hash, name="literal_j_stream_source_hash"
        )
        self._route_axis_hash = _identity(route_axis_hash, name="route_axis_hash")
        self._route_count = _exact_nonnegative_int(route_count, name="route_count")
        if self._route_count <= 0:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "route_count must be positive",
            )
        self._direction_ids = _identity_tuple(direction_ids, name="direction_ids")
        direction_array = _canonical_fp64_array(directions, name="directions", ndim=2)
        if direction_array.shape != (len(self._direction_ids), self._route_count):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "directions must have shape (direction_count, route_count)",
            )
        self._direction_hashes = _direction_hashes(direction_array)
        self._direction_axis_hash = _axis_hash(
            "fixed_directions",
            tuple(zip(self._direction_ids, self._direction_hashes, strict=True)),
        )
        stratum_tuple = _identity_tuple(stratum_ids, name="stratum_ids")
        if len(stratum_tuple) != 2:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "exact class-balanced census requires exactly two strata",
            )
        self._stratum_ids = (stratum_tuple[0], stratum_tuple[1])
        if isinstance(sealed_context_ids_by_stratum, (str, bytes)):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "sealed context orders must contain exactly two sequences",
            )
        try:
            sealed = tuple(sealed_context_ids_by_stratum)
        except TypeError as error:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                f"sealed context orders must be finite: {error}",
            )
        if len(sealed) != 2:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "sealed context orders must contain exactly two sequences",
            )
        order_zero = _identity_tuple(sealed[0], name="sealed_context_ids[0]")
        order_one = _identity_tuple(sealed[1], name="sealed_context_ids[1]")
        if set(order_zero).intersection(order_one):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "context identities must be unique across both strata",
            )
        self._sealed_context_ids = (order_zero, order_one)
        self._sealed_order_hashes = (
            _axis_hash(
                "sealed_population_order",
                {"stratum_id": self._stratum_ids[0], "context_ids": order_zero},
            ),
            _axis_hash(
                "sealed_population_order",
                {"stratum_id": self._stratum_ids[1], "context_ids": order_one},
            ),
        )
        if backend not in {"numpy_reference", "torch_device"}:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                "backend must be numpy_reference or torch_device",
            )
        self._backend = backend
        self._torch = None
        if backend == "numpy_reference":
            if torch_device is not None:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                    "NumPy reference backend does not accept a torch device",
                )
            if (
                self._route_count > _NUMPY_REFERENCE_ROUTE_LIMIT
                or len(self._direction_ids) > _NUMPY_REFERENCE_DIRECTION_LIMIT
            ):
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.REFERENCE_SCALE_EXCEEDED,
                    "NumPy exact-census reference is limited to small domains",
                )
            self._device = "cpu"
            self._directions = direction_array
            self._sum_operator = np.zeros(
                (2, len(self._direction_ids), self._route_count), dtype=np.float64
            )
            self._sum_quadratic = np.zeros(
                (2, len(self._direction_ids)), dtype=np.float64
            )
        else:
            if type(torch_device) is not str or not torch_device:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.INVALID_DESIGN,
                    "torch_device backend requires one explicit device",
                )
            try:
                import torch  # type: ignore
            except ImportError as error:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.BACKEND_UNAVAILABLE,
                    f"torch device backend is unavailable: {error}",
                )
            try:
                device = torch.device(torch_device)
            except (TypeError, ValueError, RuntimeError) as error:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.BACKEND_UNAVAILABLE,
                    f"invalid explicit torch device: {error}",
                )
            if device.type not in {"cpu", "cuda"}:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.BACKEND_UNAVAILABLE,
                    "only explicit CPU and CUDA torch devices are registered",
                )
            if device.type == "cuda" and not torch.cuda.is_available():
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.BACKEND_UNAVAILABLE,
                    "requested CUDA device is unavailable; no CPU fallback is used",
                )
            try:
                direction_device = torch.as_tensor(
                    np.array(direction_array, copy=True),
                    dtype=torch.float64,
                    device=device,
                ).contiguous()
                sum_operator = torch.zeros(
                    (2, len(self._direction_ids), self._route_count),
                    dtype=torch.float64,
                    device=device,
                )
                sum_quadratic = torch.zeros(
                    (2, len(self._direction_ids)),
                    dtype=torch.float64,
                    device=device,
                )
            except (RuntimeError, TypeError) as error:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.BACKEND_UNAVAILABLE,
                    f"torch device allocation failed closed: {error}",
                )
            self._torch = torch
            self._device = str(direction_device.device)
            self._directions = direction_device
            self._sum_operator = sum_operator
            self._sum_quadratic = sum_quadratic
        self._next_indices = [0, 0]
        self._seen_context_ids: set[str] = set()
        self._seen_draw_ids: set[str] = set()
        self._seen_source_measurement_ids: set[str] = set()
        self._context_events: list[tuple[str, str]] = []
        self._draw_events: list[tuple[str, str]] = []
        self._source_events: list[tuple[str, str]] = []
        self._stratum_draw_ids: list[list[str]] = [[], []]
        self._stratum_source_ids: list[list[str]] = [[], []]
        self._finalized = False

    def __copy__(self) -> NoReturn:
        raise TypeError("literal-J census accumulator is noncopyable")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("literal-J census accumulator is noncopyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J census accumulator is nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J census accumulator is nonserializable")

    @property
    def observed_counts(self) -> tuple[int, int]:
        return (int(self._next_indices[0]), int(self._next_indices[1]))

    @property
    def population_counts(self) -> tuple[int, int]:
        return (
            len(self._sealed_context_ids[0]),
            len(self._sealed_context_ids[1]),
        )

    def _stratum_index(self, stratum_id: object) -> int:
        if stratum_id == self._stratum_ids[0]:
            return 0
        if stratum_id == self._stratum_ids[1]:
            return 1
        _refuse(
            V1LiteralJOperatorCensusRefusalCode.STALE_IDENTITY,
            "row stratum identity is not registered by this census",
        )

    def _validate_bound_identities(
        self,
        *,
        model_identity_hash: object,
        chart_identity_hash: object,
        current_mask_hash: object,
        proposal_generation: object,
        literal_j_stream_source_hash: object,
        route_axis_hash: object,
    ) -> None:
        observed = (
            model_identity_hash,
            chart_identity_hash,
            current_mask_hash,
            proposal_generation,
            literal_j_stream_source_hash,
            route_axis_hash,
        )
        expected = (
            self._model_identity_hash,
            self._chart_identity_hash,
            self._current_mask_hash,
            self._proposal_generation,
            self._literal_j_stream_source_hash,
            self._route_axis_hash,
        )
        if observed != expected:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.STALE_IDENTITY,
                "literal-J row left the frozen model/chart/mask/generation/stream/axis",
            )

    def _numpy_contribution(
        self, row: object, *, stratum_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            row_array = _canonical_fp64_array(row, name="literal_j_row", ndim=1)
        except V1LiteralJOperatorCensusRefusal as error:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                str(error),
            )
        if row_array.shape != (self._route_count,):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "literal-J row does not match the frozen route axis",
            )
        with np.errstate(over="ignore", invalid="ignore"):
            projection = self._directions @ row_array
            contribution = projection[:, None] * row_array[None, :]
            quadratic = projection * projection
            candidate_operator = self._sum_operator[stratum_index] + contribution
            candidate_quadratic = self._sum_quadratic[stratum_index] + quadratic
        if not np.all(np.isfinite(candidate_operator)) or not np.all(
            np.isfinite(candidate_quadratic)
        ):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "literal-J FP64 sufficient statistics overflowed or became nonfinite",
            )
        return candidate_operator, candidate_quadratic

    def _torch_contribution(
        self, row: object, *, stratum_index: int
    ) -> tuple[Any, Any]:
        torch = self._torch
        if torch is None or not isinstance(row, torch.Tensor):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "torch-device census requires an exact torch Tensor row",
            )
        if (
            row.dtype != torch.float64
            or row.device != self._directions.device
            or row.ndim != 1
            or int(row.numel()) != self._route_count
            or not row.is_contiguous()
            or bool(row.requires_grad)
        ):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "torch literal-J row must be detached contiguous FP64 on the "
                "registered device",
            )
        if not bool(torch.isfinite(row).all().item()):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "literal-J row must contain only finite values",
            )
        with torch.no_grad():
            projection = self._directions.mv(row)
            contribution = projection.unsqueeze(1) * row.unsqueeze(0)
            quadratic = projection.square()
            candidate_operator = self._sum_operator[stratum_index] + contribution
            candidate_quadratic = self._sum_quadratic[stratum_index] + quadratic
        if not bool(torch.isfinite(candidate_operator).all().item()) or not bool(
            torch.isfinite(candidate_quadratic).all().item()
        ):
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.INVALID_ROW,
                "literal-J FP64 sufficient statistics overflowed or became nonfinite",
            )
        return candidate_operator, candidate_quadratic

    def update(
        self,
        *,
        stratum_id: str,
        context_id: str,
        draw_id: str,
        source_measurement_id: str,
        literal_j_row: object,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        literal_j_stream_source_hash: str,
        route_axis_hash: str,
    ) -> None:
        """Consume one registered row atomically, retaining no raw row."""

        if self._finalized:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.FINALIZED,
                "literal-J census has already finalized",
            )
        self._validate_bound_identities(
            model_identity_hash=model_identity_hash,
            chart_identity_hash=chart_identity_hash,
            current_mask_hash=current_mask_hash,
            proposal_generation=proposal_generation,
            literal_j_stream_source_hash=literal_j_stream_source_hash,
            route_axis_hash=route_axis_hash,
        )
        stratum_index = self._stratum_index(stratum_id)
        for value, name in (
            (context_id, "context_id"),
            (draw_id, "draw_id"),
            (source_measurement_id, "source_measurement_id"),
        ):
            if type(value) is not str or not value:
                _refuse(
                    V1LiteralJOperatorCensusRefusalCode.STALE_IDENTITY,
                    f"{name} must be one nonempty registered identity",
                )
        if context_id in self._seen_context_ids:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.DUPLICATE_IDENTITY,
                "context identity was already consumed",
            )
        if draw_id in self._seen_draw_ids:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.DUPLICATE_IDENTITY,
                "draw identity was already consumed",
            )
        if source_measurement_id in self._seen_source_measurement_ids:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.DUPLICATE_IDENTITY,
                "source-measurement identity was already consumed",
            )
        next_index = self._next_indices[stratum_index]
        order = self._sealed_context_ids[stratum_index]
        if next_index >= len(order) or context_id != order[next_index]:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.ORDER_MISMATCH,
                "context identity is not the next member of its sealed stratum order",
            )
        if self._backend == "numpy_reference":
            candidate_operator, candidate_quadratic = self._numpy_contribution(
                literal_j_row, stratum_index=stratum_index
            )
            self._sum_operator[stratum_index] = candidate_operator
            self._sum_quadratic[stratum_index] = candidate_quadratic
        else:
            candidate_operator, candidate_quadratic = self._torch_contribution(
                literal_j_row, stratum_index=stratum_index
            )
            self._sum_operator[stratum_index].copy_(candidate_operator)
            self._sum_quadratic[stratum_index].copy_(candidate_quadratic)
        self._next_indices[stratum_index] += 1
        self._seen_context_ids.add(context_id)
        self._seen_draw_ids.add(draw_id)
        self._seen_source_measurement_ids.add(source_measurement_id)
        self._context_events.append((stratum_id, context_id))
        self._draw_events.append((stratum_id, draw_id))
        self._source_events.append((stratum_id, source_measurement_id))
        self._stratum_draw_ids[stratum_index].append(draw_id)
        self._stratum_source_ids[stratum_index].append(source_measurement_id)

    def _final_arrays(self) -> tuple[Any, Any, np.ndarray, np.ndarray]:
        populations = self.population_counts
        if self._backend == "numpy_reference":
            stratum_operator_zero = self._sum_operator[0] / float(populations[0])
            stratum_operator_one = self._sum_operator[1] / float(populations[1])
            stratum_q_zero = self._sum_quadratic[0] / float(populations[0])
            stratum_q_one = self._sum_quadratic[1] / float(populations[1])
            outputs = np.array(
                0.5 * (stratum_operator_zero + stratum_operator_one),
                dtype=np.dtype("<f8"),
                order="C",
                copy=True,
            )
            quadratic = np.array(
                0.5 * (stratum_q_zero + stratum_q_one),
                dtype=np.dtype("<f8"),
                order="C",
                copy=True,
            )
            outputs.setflags(write=False)
            quadratic.setflags(write=False)
            return outputs, quadratic, outputs, quadratic
        torch = self._torch
        outputs = 0.5 * (
            self._sum_operator[0] / float(populations[0])
            + self._sum_operator[1] / float(populations[1])
        )
        quadratic = 0.5 * (
            self._sum_quadratic[0] / float(populations[0])
            + self._sum_quadratic[1] / float(populations[1])
        )
        outputs = outputs.detach().clone()
        quadratic = quadratic.detach().clone()
        return (
            outputs,
            quadratic,
            _torch_host_fp64(torch, outputs),
            _torch_host_fp64(torch, quadratic),
        )

    def finalize(self) -> V1LiteralJOperatorCensusActionState:
        """Freeze the exact census; incomplete populations fail closed."""

        if self._finalized:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.FINALIZED,
                "literal-J census has already finalized",
            )
        if self.observed_counts != self.population_counts:
            _refuse(
                V1LiteralJOperatorCensusRefusalCode.CENSUS_INCOMPLETE,
                "both registered stratum populations must be observed exactly",
            )
        outputs, quadratic, host_outputs, host_quadratic = self._final_arrays()
        output_hashes = _output_hashes(host_outputs)
        quadratic_values = tuple(float(item) for item in host_quadratic.tolist())
        observed_context_axis_hashes = (
            _axis_hash(
                "observed_context_axis",
                {
                    "stratum_id": self._stratum_ids[0],
                    "context_ids": self._sealed_context_ids[0],
                },
            ),
            _axis_hash(
                "observed_context_axis",
                {
                    "stratum_id": self._stratum_ids[1],
                    "context_ids": self._sealed_context_ids[1],
                },
            ),
        )
        observed_draw_axis_hashes = (
            _axis_hash(
                "observed_draw_axis",
                {
                    "stratum_id": self._stratum_ids[0],
                    "draw_ids": tuple(self._stratum_draw_ids[0]),
                },
            ),
            _axis_hash(
                "observed_draw_axis",
                {
                    "stratum_id": self._stratum_ids[1],
                    "draw_ids": tuple(self._stratum_draw_ids[1]),
                },
            ),
        )
        observed_source_axis_hashes = (
            _axis_hash(
                "observed_source_measurement_axis",
                {
                    "stratum_id": self._stratum_ids[0],
                    "source_measurement_ids": tuple(self._stratum_source_ids[0]),
                },
            ),
            _axis_hash(
                "observed_source_measurement_axis",
                {
                    "stratum_id": self._stratum_ids[1],
                    "source_measurement_ids": tuple(self._stratum_source_ids[1]),
                },
            ),
        )
        output_axis_hash = _axis_hash(
            "literal_j_operator_outputs",
            tuple(
                zip(
                    self._direction_ids,
                    output_hashes,
                    quadratic_values,
                    strict=True,
                )
            ),
        )
        receipt = V1LiteralJOperatorCensusReceipt(
            census_id=self._census_id,
            model_identity_hash=self._model_identity_hash,
            chart_identity_hash=self._chart_identity_hash,
            current_mask_hash=self._current_mask_hash,
            proposal_generation=self._proposal_generation,
            literal_j_stream_source_hash=self._literal_j_stream_source_hash,
            route_axis_hash=self._route_axis_hash,
            route_count=self._route_count,
            direction_ids=self._direction_ids,
            direction_hashes=self._direction_hashes,
            direction_axis_hash=self._direction_axis_hash,
            stratum_ids=self._stratum_ids,
            population_counts=self.population_counts,
            observed_counts=self.observed_counts,
            sealed_population_order_hashes=self._sealed_order_hashes,
            observed_context_axis_hashes=observed_context_axis_hashes,
            observed_draw_axis_hashes=observed_draw_axis_hashes,
            observed_source_measurement_axis_hashes=observed_source_axis_hashes,
            context_event_axis_hash=_axis_hash(
                "context_event_axis", tuple(self._context_events)
            ),
            draw_event_axis_hash=_axis_hash(
                "draw_event_axis", tuple(self._draw_events)
            ),
            source_measurement_event_axis_hash=_axis_hash(
                "source_measurement_event_axis", tuple(self._source_events)
            ),
            ingestion_event_axis_hash=_axis_hash(
                "ingestion_event_axis",
                tuple(
                    zip(
                        self._context_events,
                        self._draw_events,
                        self._source_events,
                        strict=True,
                    )
                ),
            ),
            output_vector_hashes=output_hashes,
            quadratic_values=quadratic_values,
            output_axis_hash=output_axis_hash,
            backend=self._backend,
            device=self._device,
            accumulation_algorithm=(
                "sealed_order_fp64_two_stratum_equal_mean_numpy_reference_v1"
                if self._backend == "numpy_reference"
                else "sealed_order_fp64_two_stratum_equal_mean_torch_device_v1"
            ),
        )
        action_state = V1LiteralJOperatorCensusActionState(
            token=_ACTION_STATE_ISSUANCE_TOKEN,
            receipt=receipt,
            operator_outputs=outputs,
            quadratic_values=quadratic,
            torch_module=self._torch,
        )
        self._finalized = True
        self._directions = None
        self._sum_operator = None
        self._sum_quadratic = None
        return action_state


__all__ = [
    "LITERAL_J_OPERATOR_CENSUS_AUTHORITY_SCOPE",
    "LITERAL_J_OPERATOR_CENSUS_SCHEMA",
    "LITERAL_J_OPERATOR_EQUATION",
    "V1LiteralJOperatorCensusAccumulator",
    "V1LiteralJOperatorCensusActionState",
    "V1LiteralJOperatorCensusReceipt",
    "V1LiteralJOperatorCensusRefusal",
    "V1LiteralJOperatorCensusRefusalCode",
]
