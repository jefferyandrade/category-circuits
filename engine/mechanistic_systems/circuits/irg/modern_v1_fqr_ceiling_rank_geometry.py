"""Ceiling-rank FD geometry with nested conservative rank views.

This module is a typed mathematical joint, not a CUDA executor.  It records
the production-tractable rank contract for canonical v1 D:

* one HBM-admitted ceiling-rank FD state is produced from the registered D
  prefix;
* lower ranks are nested prefix views of that same state, never independent
  sketches and never zero-padded states;
* the discarded PSD tail plus the deterministic FD residual is an outward
  envelope on every lower-rank Q decision;
* C may open only after the minimum nested rank whose D/FD/KKT decision image
  is invariant has been frozen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from enum import Enum
import math
from numbers import Integral
from typing import Final

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import (
    V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT,
)
from .modern_v1_stratified_fd_gpu import forecast_stratified_fd_gpu_memory


FQR_CEILING_RANK_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_memory_forecast_v1"
)
FQR_CEILING_RANK_NESTED_VIEW_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_nested_view_v1"
)
FQR_CEILING_RANK_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_geometry_v1"
)
FQR_CEILING_RANK_DECISION_IMAGE_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_decision_image_v1"
)
FQR_CEILING_RANK_FREEZE_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_freeze_for_C_v1"
)
FQR_CEILING_RANK_NO_C_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_fqr_ceiling_rank_no_C_refusal_v1"
)

_MEMORY_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-memory-"
_VIEW_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-view-"
_GEOMETRY_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-geometry-"
_IMAGE_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-decision-image-"
_FREEZE_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-freeze-"
_REFUSAL_PREFIX: Final = "partition-free-v1-fqr-ceiling-rank-refusal-"

_GEOMETRY_TOKEN: Final = object()
_VIEW_TOKEN: Final = object()
_IMAGE_TOKEN: Final = object()
_FREEZE_TOKEN: Final = object()

_FLOAT64_BYTES: Final = 8


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty exact string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def canonical_fqr_ceiling_rank_lattice_for_route_count(
    route_count: int,
) -> tuple[int, ...]:
    """Return the natural route-clipped dyadic rank lattice."""

    routes = _positive_int(route_count, name="route count")
    ranks: list[int] = []
    rank = 1
    while rank < routes:
        ranks.append(rank)
        rank *= 2
    ranks.append(routes)
    return tuple(ranks)


def _rank_prefix(route_count: int, ceiling_rank: int) -> tuple[int, ...]:
    lattice = canonical_fqr_ceiling_rank_lattice_for_route_count(route_count)
    rank = _positive_int(ceiling_rank, name="ceiling rank")
    if rank not in lattice:
        raise ValueError("ceiling rank must belong to the dyadic execution lattice")
    return lattice[: lattice.index(rank) + 1]


def _kkt_workspace_bytes(
    *,
    route_count: int,
    rank: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
) -> tuple[int, int, int, int]:
    routes = _positive_int(route_count, name="route count")
    current_rank = _positive_int(rank, name="rank")
    blocks = _positive_int(legality_block_count, name="legality block count")
    tile = _positive_int(rounding_tile_size, name="rounding tile size")
    hard_actions = _positive_int(
        maximum_hard_action_count,
        name="maximum hard action count",
    )
    winner_count = min(routes, hard_actions)
    problem = (current_rank + 5) * routes * _FLOAT64_BYTES + routes + blocks * 4 * 8
    solver = (
        V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT * routes * _FLOAT64_BYTES
        + (
            current_rank
            + 4 * blocks
            + V1_RELAXED_KKT_FUSED_ROUTE_VECTOR_FORECAST_COUNT
        )
        * _FLOAT64_BYTES
    )
    rounding = 12 * (winner_count + min(routes, tile)) * _FLOAT64_BYTES
    return problem, solver, rounding, problem + solver + rounding


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankMemoryForecast:
    """HBM ledger for one ceiling FD state plus nested rank readout."""

    route_count: int
    positive_weight_stratum_count: int
    response_rows: int
    ceiling_rank: int
    exposed_rank_lattice: tuple[int, ...]
    physical_workspace_ceiling_bytes: int
    kkt_legality_block_count: int
    kkt_rounding_tile_size: int
    maximum_hard_action_count: int
    D_packet_staging_device_bytes: int
    ceiling_fd_peak_device_bytes: int
    ceiling_retained_geometry_device_bytes: int
    fused_kkt_problem_device_bytes: int
    fused_kkt_solver_workspace_device_bytes: int
    deterministic_rounding_workspace_device_bytes: int
    nested_view_route_scale_device_bytes: int
    post_D_kkt_peak_device_bytes: int
    registered_phase_peak_device_bytes: int
    admitted: bool
    one_ceiling_fd_state_required: bool = True
    nested_lower_rank_views_are_metadata_only: bool = True
    simultaneous_independent_rank_prefix_required: bool = False
    lower_rank_zero_padding_authority: bool = False
    packet_size_affects_runtime_only: bool = True
    C_stream_open_authority: bool = False
    schema: str = FQR_CEILING_RANK_MEMORY_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(
            self.positive_weight_stratum_count,
            name="positive weight stratum count",
        )
        responses = _positive_int(self.response_rows, name="response rows")
        ceiling_rank = _positive_int(self.ceiling_rank, name="ceiling rank")
        ceiling = _positive_int(
            self.physical_workspace_ceiling_bytes,
            name="physical workspace ceiling bytes",
        )
        blocks = _positive_int(
            self.kkt_legality_block_count,
            name="legality block count",
        )
        tile = _positive_int(
            self.kkt_rounding_tile_size,
            name="rounding tile size",
        )
        hard_actions = _positive_int(
            self.maximum_hard_action_count,
            name="maximum hard action count",
        )
        staging = _nonnegative_int(
            self.D_packet_staging_device_bytes,
            name="D packet staging device bytes",
        )
        expected_prefix = _rank_prefix(routes, ceiling_rank)
        if tuple(self.exposed_rank_lattice) != expected_prefix:
            raise ValueError("exposed rank lattice must be the ceiling-rank prefix")
        base = forecast_stratified_fd_gpu_memory(
            route_count=routes,
            stratum_count=strata,
            response_rows=responses,
            per_stratum_sketch_rank=ceiling_rank,
            combined_maximum_rank=ceiling_rank,
        )
        fd_peak = base.peak_device_bytes + staging
        retained = base.combined_output_state_device_bytes
        if self.ceiling_fd_peak_device_bytes != fd_peak:
            raise ValueError("ceiling FD peak bytes do not replay")
        if self.ceiling_retained_geometry_device_bytes != retained:
            raise ValueError("ceiling retained-geometry bytes do not replay")
        problem, solver, rounding, _total = _kkt_workspace_bytes(
            route_count=routes,
            rank=ceiling_rank,
            legality_block_count=blocks,
            rounding_tile_size=tile,
            maximum_hard_action_count=hard_actions,
        )
        if self.fused_kkt_problem_device_bytes != problem:
            raise ValueError("fused KKT problem bytes do not replay")
        if self.fused_kkt_solver_workspace_device_bytes != solver:
            raise ValueError("fused KKT solver workspace bytes do not replay")
        if self.deterministic_rounding_workspace_device_bytes != rounding:
            raise ValueError("deterministic rounding workspace bytes do not replay")
        for name in (
            "nested_view_route_scale_device_bytes",
            "post_D_kkt_peak_device_bytes",
            "registered_phase_peak_device_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.nested_view_route_scale_device_bytes != 0:
            raise ValueError("nested lower-rank views cannot allocate route-scale HBM")
        expected_post = (
            retained
            + self.fused_kkt_problem_device_bytes
            + self.fused_kkt_solver_workspace_device_bytes
            + self.deterministic_rounding_workspace_device_bytes
        )
        if self.post_D_kkt_peak_device_bytes != expected_post:
            raise ValueError("post-D KKT peak bytes do not replay")
        expected_phase = max(fd_peak, expected_post)
        if self.registered_phase_peak_device_bytes != expected_phase:
            raise ValueError("registered phase peak bytes do not replay")
        if type(self.admitted) is not bool or self.admitted != (expected_phase <= ceiling):
            raise ValueError("ceiling-rank physical admission does not replay")
        flags = (
            self.one_ceiling_fd_state_required,
            self.nested_lower_rank_views_are_metadata_only,
            self.simultaneous_independent_rank_prefix_required,
            self.lower_rank_zero_padding_authority,
            self.packet_size_affects_runtime_only,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank memory flags must be exact booleans")
        if flags != (True, True, False, False, True, False):
            raise PermissionError("ceiling-rank memory forecast crossed authority")
        if self.schema != FQR_CEILING_RANK_MEMORY_FORECAST_SCHEMA:
            raise ValueError("ceiling-rank memory schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "positive_weight_stratum_count", strata)
        object.__setattr__(self, "response_rows", responses)
        object.__setattr__(self, "ceiling_rank", ceiling_rank)
        object.__setattr__(self, "exposed_rank_lattice", expected_prefix)
        object.__setattr__(self, "kkt_legality_block_count", blocks)
        object.__setattr__(self, "kkt_rounding_tile_size", tile)
        object.__setattr__(self, "maximum_hard_action_count", hard_actions)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_MEMORY_PREFIX)


def forecast_v1_fqr_ceiling_rank_memory(
    *,
    route_count: int,
    positive_weight_stratum_count: int,
    response_rows: int,
    ceiling_rank: int,
    physical_workspace_ceiling_bytes: int,
    legality_block_count: int,
    rounding_tile_size: int,
    maximum_hard_action_count: int,
    D_packet_staging_device_bytes: int = 0,
) -> V1FQRCeilingRankMemoryForecast:
    """Forecast one ceiling state; lower ranks add no route-scale HBM."""

    routes = _positive_int(route_count, name="route count")
    strata = _positive_int(
        positive_weight_stratum_count,
        name="positive weight stratum count",
    )
    responses = _positive_int(response_rows, name="response rows")
    rank = _positive_int(ceiling_rank, name="ceiling rank")
    prefix = _rank_prefix(routes, rank)
    staging = _nonnegative_int(
        D_packet_staging_device_bytes,
        name="D packet staging device bytes",
    )
    base = forecast_stratified_fd_gpu_memory(
        route_count=routes,
        stratum_count=strata,
        response_rows=responses,
        per_stratum_sketch_rank=rank,
        combined_maximum_rank=rank,
    )
    problem, solver, rounding, _total = _kkt_workspace_bytes(
        route_count=routes,
        rank=rank,
        legality_block_count=legality_block_count,
        rounding_tile_size=rounding_tile_size,
        maximum_hard_action_count=maximum_hard_action_count,
    )
    fd_peak = base.peak_device_bytes + staging
    post = base.combined_output_state_device_bytes + problem + solver + rounding
    peak = max(fd_peak, post)
    return V1FQRCeilingRankMemoryForecast(
        route_count=routes,
        positive_weight_stratum_count=strata,
        response_rows=responses,
        ceiling_rank=rank,
        exposed_rank_lattice=prefix,
        physical_workspace_ceiling_bytes=physical_workspace_ceiling_bytes,
        kkt_legality_block_count=legality_block_count,
        kkt_rounding_tile_size=rounding_tile_size,
        maximum_hard_action_count=maximum_hard_action_count,
        D_packet_staging_device_bytes=staging,
        ceiling_fd_peak_device_bytes=fd_peak,
        ceiling_retained_geometry_device_bytes=(
            base.combined_output_state_device_bytes
        ),
        fused_kkt_problem_device_bytes=problem,
        fused_kkt_solver_workspace_device_bytes=solver,
        deterministic_rounding_workspace_device_bytes=rounding,
        nested_view_route_scale_device_bytes=0,
        post_D_kkt_peak_device_bytes=post,
        registered_phase_peak_device_bytes=peak,
        admitted=peak <= physical_workspace_ceiling_bytes,
    )


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankNestedResidualEnvelope:
    """Outward Q-error bound for one nested lower-rank prefix view."""

    nested_rank: int
    ceiling_rank: int
    tail_trace_upper_bound: float
    tail_spectral_upper_bound: float
    deterministic_fd_residual_spectral_upper_bound: float
    direction_l2_square_upper_bound: float
    q_error_upper_bound: float
    envelope_is_outward: bool = True
    schema: str = "partition_free_v1_fqr_ceiling_rank_residual_envelope_v1"

    def __post_init__(self) -> None:
        rank = _positive_int(self.nested_rank, name="nested rank")
        ceiling = _positive_int(self.ceiling_rank, name="ceiling rank")
        if rank > ceiling:
            raise ValueError("nested rank cannot exceed ceiling rank")
        trace = _finite_nonnegative(
            self.tail_trace_upper_bound,
            name="tail trace upper bound",
        )
        spectral = _finite_nonnegative(
            self.tail_spectral_upper_bound,
            name="tail spectral upper bound",
        )
        residual = _finite_nonnegative(
            self.deterministic_fd_residual_spectral_upper_bound,
            name="deterministic FD residual spectral upper bound",
        )
        direction = _finite_nonnegative(
            self.direction_l2_square_upper_bound,
            name="direction l2-square upper bound",
        )
        expected = (spectral + residual) * direction
        if spectral > trace and trace != 0.0:
            raise ValueError("spectral tail bound cannot exceed trace tail bound")
        if not math.isclose(
            self.q_error_upper_bound,
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Q-error bound must replay from tail and FD residual")
        if type(self.envelope_is_outward) is not bool or not self.envelope_is_outward:
            raise PermissionError("nested residual envelope must be outward")


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankNestedView:
    """Array-free lower-rank view of one ceiling-rank FD state."""

    geometry_hash: str
    ceiling_fd_state_hash: str
    rank: int
    ceiling_rank: int
    retained_eigenvalue_square_hash: str
    residual_envelope: V1FQRCeilingRankNestedResidualEnvelope
    _issuance_token: InitVar[object]
    lower_rank_zero_padding_used: bool = False
    independent_lower_rank_fd_state_used: bool = False
    nested_prefix_view_of_ceiling_state: bool = True
    route_scale_hbm_increment_bytes: int = 0
    C_stream_open_authority: bool = False
    schema: str = FQR_CEILING_RANK_NESTED_VIEW_SCHEMA
    _view_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _VIEW_TOKEN:
            raise PermissionError("ceiling-rank nested views are issuer-only")
        for name in (
            "geometry_hash",
            "ceiling_fd_state_hash",
            "retained_eigenvalue_square_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _positive_int(self.rank, name="view rank")
        ceiling = _positive_int(self.ceiling_rank, name="ceiling rank")
        if rank > ceiling:
            raise ValueError("view rank cannot exceed ceiling rank")
        if type(self.residual_envelope) is not V1FQRCeilingRankNestedResidualEnvelope:
            raise TypeError("nested view requires the exact residual envelope")
        if (
            self.residual_envelope.nested_rank != rank
            or self.residual_envelope.ceiling_rank != ceiling
        ):
            raise ValueError("nested view residual envelope has stale rank coordinates")
        if _nonnegative_int(
            self.route_scale_hbm_increment_bytes,
            name="route-scale HBM increment bytes",
        ) != 0:
            raise PermissionError("nested views cannot allocate route-scale HBM")
        flags = (
            self.lower_rank_zero_padding_used,
            self.independent_lower_rank_fd_state_used,
            self.nested_prefix_view_of_ceiling_state,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("nested-view flags must be exact booleans")
        if flags != (False, False, True, False):
            raise PermissionError("nested view crossed its authority")
        if self.schema != FQR_CEILING_RANK_NESTED_VIEW_SCHEMA:
            raise ValueError("nested-view schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "ceiling_rank", ceiling)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "geometry_hash": self.geometry_hash,
            "ceiling_fd_state_hash": self.ceiling_fd_state_hash,
            "rank": self.rank,
            "ceiling_rank": self.ceiling_rank,
            "retained_eigenvalue_square_hash": self.retained_eigenvalue_square_hash,
            "residual_envelope": self.residual_envelope,
            "lower_rank_zero_padding_used": self.lower_rank_zero_padding_used,
            "independent_lower_rank_fd_state_used": (
                self.independent_lower_rank_fd_state_used
            ),
            "nested_prefix_view_of_ceiling_state": (
                self.nested_prefix_view_of_ceiling_state
            ),
            "route_scale_hbm_increment_bytes": self.route_scale_hbm_increment_bytes,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def view_hash(self) -> str:
        cached = self._view_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_VIEW_PREFIX)
        cached = self._view_hash
        if cached is not None and cached != observed:
            raise ValueError("nested ceiling-rank view was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("nested ceiling-rank view hash does not match")
        object.__setattr__(self, "_view_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankGeometry:
    """One ceiling FD state and all nested rank views from one D prefix."""

    policy_hash: str
    registered_d_prefix_hash: str
    source_capture_vjp_lineage_hash: str
    ceiling_fd_state_hash: str
    route_axis_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    ceiling_rank: int
    exposed_rank_lattice: tuple[int, ...]
    eigenvalue_square_axis: tuple[float, ...]
    deterministic_fd_residual_spectral_upper_bound: float
    direction_l2_square_upper_bound: float
    memory_forecast: V1FQRCeilingRankMemoryForecast
    _issuance_token: InitVar[object]
    one_ceiling_fd_state: bool = True
    lower_rank_zero_padding_authority: bool = False
    independent_lower_rank_fd_states: bool = False
    same_D_prefix_for_all_views: bool = True
    C_stream_open_authority: bool = False
    schema: str = FQR_CEILING_RANK_GEOMETRY_SCHEMA
    _geometry_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _GEOMETRY_TOKEN:
            raise PermissionError("ceiling-rank geometry is issuer-only")
        for name in (
            "policy_hash",
            "registered_d_prefix_hash",
            "source_capture_vjp_lineage_hash",
            "ceiling_fd_state_hash",
            "route_axis_hash",
            "current_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        rank = _positive_int(self.ceiling_rank, name="ceiling rank")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        exposed = tuple(_positive_int(value, name="exposed rank") for value in self.exposed_rank_lattice)
        if exposed != _rank_prefix(routes, rank):
            raise ValueError("ceiling-rank geometry must expose the full prefix")
        values = tuple(
            _finite_nonnegative(value, name="eigenvalue-square")
            for value in self.eigenvalue_square_axis
        )
        if len(values) != rank:
            raise ValueError("ceiling-rank geometry requires one eigenvalue per row")
        if any(values[index] < values[index + 1] for index in range(len(values) - 1)):
            raise ValueError("ceiling eigenvalue squares must be nonincreasing")
        residual = _finite_nonnegative(
            self.deterministic_fd_residual_spectral_upper_bound,
            name="deterministic FD residual spectral upper bound",
        )
        direction = _finite_nonnegative(
            self.direction_l2_square_upper_bound,
            name="direction l2-square upper bound",
        )
        if type(self.memory_forecast) is not V1FQRCeilingRankMemoryForecast:
            raise TypeError("ceiling geometry requires the exact memory forecast")
        if (
            self.memory_forecast.route_count != routes
            or self.memory_forecast.ceiling_rank != rank
            or self.memory_forecast.exposed_rank_lattice != exposed
        ):
            raise ValueError("ceiling geometry memory forecast has stale coordinates")
        flags = (
            self.one_ceiling_fd_state,
            self.lower_rank_zero_padding_authority,
            self.independent_lower_rank_fd_states,
            self.same_D_prefix_for_all_views,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank geometry flags must be exact booleans")
        if flags != (True, False, False, True, False):
            raise PermissionError("ceiling-rank geometry crossed its authority")
        if self.schema != FQR_CEILING_RANK_GEOMETRY_SCHEMA:
            raise ValueError("ceiling-rank geometry schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "ceiling_rank", rank)
        object.__setattr__(self, "exposed_rank_lattice", exposed)
        object.__setattr__(self, "eigenvalue_square_axis", values)
        object.__setattr__(
            self,
            "deterministic_fd_residual_spectral_upper_bound",
            residual,
        )
        object.__setattr__(self, "direction_l2_square_upper_bound", direction)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "source_capture_vjp_lineage_hash": self.source_capture_vjp_lineage_hash,
            "ceiling_fd_state_hash": self.ceiling_fd_state_hash,
            "route_axis_hash": self.route_axis_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "ceiling_rank": self.ceiling_rank,
            "exposed_rank_lattice": self.exposed_rank_lattice,
            "eigenvalue_square_axis": self.eigenvalue_square_axis,
            "deterministic_fd_residual_spectral_upper_bound": (
                self.deterministic_fd_residual_spectral_upper_bound
            ),
            "direction_l2_square_upper_bound": self.direction_l2_square_upper_bound,
            "memory_forecast_hash": self.memory_forecast.forecast_hash,
            "one_ceiling_fd_state": self.one_ceiling_fd_state,
            "lower_rank_zero_padding_authority": (
                self.lower_rank_zero_padding_authority
            ),
            "independent_lower_rank_fd_states": self.independent_lower_rank_fd_states,
            "same_D_prefix_for_all_views": self.same_D_prefix_for_all_views,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def geometry_hash(self) -> str:
        cached = self._geometry_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GEOMETRY_PREFIX)
        cached = self._geometry_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank geometry was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("ceiling-rank geometry hash does not match")
        object.__setattr__(self, "_geometry_hash", observed)
        return observed

    def nested_views(self) -> tuple[V1FQRCeilingRankNestedView, ...]:
        """Return every nested view with a monotone conservative envelope."""

        geometry_hash = self.geometry_hash
        views: list[V1FQRCeilingRankNestedView] = []
        previous_tail = math.inf
        for rank in self.exposed_rank_lattice:
            tail_values = self.eigenvalue_square_axis[rank:]
            tail_trace = math.fsum(tail_values)
            if tail_trace > previous_tail + 1e-12:
                raise ValueError("nested tail envelope is not monotone")
            previous_tail = tail_trace
            spectral = max(tail_values) if tail_values else 0.0
            envelope = V1FQRCeilingRankNestedResidualEnvelope(
                nested_rank=rank,
                ceiling_rank=self.ceiling_rank,
                tail_trace_upper_bound=tail_trace,
                tail_spectral_upper_bound=spectral,
                deterministic_fd_residual_spectral_upper_bound=(
                    self.deterministic_fd_residual_spectral_upper_bound
                ),
                direction_l2_square_upper_bound=(
                    self.direction_l2_square_upper_bound
                ),
                q_error_upper_bound=(
                    (
                        spectral
                        + self.deterministic_fd_residual_spectral_upper_bound
                    )
                    * self.direction_l2_square_upper_bound
                ),
            )
            views.append(
                V1FQRCeilingRankNestedView(
                    geometry_hash=geometry_hash,
                    ceiling_fd_state_hash=self.ceiling_fd_state_hash,
                    rank=rank,
                    ceiling_rank=self.ceiling_rank,
                    retained_eigenvalue_square_hash=content_hash(
                        self.eigenvalue_square_axis[:rank],
                        prefix="partition-free-v1-fqr-ceiling-rank-retained-eigen-",
                    ),
                    residual_envelope=envelope,
                    _issuance_token=_VIEW_TOKEN,
                )
            )
        return tuple(views)


def issue_v1_fqr_ceiling_rank_geometry(
    *,
    policy_hash: str,
    registered_d_prefix_hash: str,
    source_capture_vjp_lineage_hash: str,
    ceiling_fd_state_hash: str,
    route_axis_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    route_count: int,
    ceiling_rank: int,
    eigenvalue_square_axis: Sequence[float],
    deterministic_fd_residual_spectral_upper_bound: float,
    direction_l2_square_upper_bound: float,
    memory_forecast: V1FQRCeilingRankMemoryForecast,
) -> V1FQRCeilingRankGeometry:
    """Issue one source-native ceiling-rank geometry for nested D decisions."""

    return V1FQRCeilingRankGeometry(
        policy_hash=policy_hash,
        registered_d_prefix_hash=registered_d_prefix_hash,
        source_capture_vjp_lineage_hash=source_capture_vjp_lineage_hash,
        ceiling_fd_state_hash=ceiling_fd_state_hash,
        route_axis_hash=route_axis_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        route_count=route_count,
        ceiling_rank=ceiling_rank,
        exposed_rank_lattice=_rank_prefix(route_count, ceiling_rank),
        eigenvalue_square_axis=tuple(eigenvalue_square_axis),
        deterministic_fd_residual_spectral_upper_bound=(
            deterministic_fd_residual_spectral_upper_bound
        ),
        direction_l2_square_upper_bound=direction_l2_square_upper_bound,
        memory_forecast=memory_forecast,
        _issuance_token=_GEOMETRY_TOKEN,
    )


class V1FQRCeilingRankNoCCode(str, Enum):
    """Typed reasons a nested ceiling-rank decision cannot open C."""

    SOURCE_DECISION_IMAGE_INCOMPLETE = "source_decision_image_incomplete"
    D_SAMPLING_AMBIGUOUS = "D_sampling_ambiguous"
    FD_TAIL_AMBIGUOUS = "FD_tail_ambiguous"
    KKT_NUMERICAL_AMBIGUOUS = "KKT_numerical_ambiguous"
    NESTED_RANK_REFINEMENT_AVAILABLE = "nested_rank_refinement_available"
    MAXIMUM_CEILING_RANK_AMBIGUOUS = "maximum_ceiling_rank_ambiguous"


class V1FQRCeilingRankNextAction(str, Enum):
    """Only lawful computations after a ceiling-rank no-C refusal."""

    ACQUIRE_MORE_D_FOR_SAMPLING = "acquire_more_D_for_sampling"
    EVALUATE_NEXT_NESTED_RANK_FROM_CEILING_STATE = (
        "evaluate_next_nested_rank_from_ceiling_state"
    )
    CONTINUE_REGISTERED_KKT_SOLVER = "continue_registered_KKT_solver"
    PHYSICAL_NO_C_REFUSAL = "physical_no_C_refusal"


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankNoCRefusal:
    """Fail-closed refusal; unresolved rank uncertainty never opens C."""

    code: V1FQRCeilingRankNoCCode
    reason: str
    required_action: V1FQRCeilingRankNextAction
    geometry_hash: str | None = None
    current_rank: int | None = None
    next_rank: int | None = None
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FQR_CEILING_RANK_NO_C_REFUSAL_SCHEMA

    def __post_init__(self) -> None:
        if type(self.code) is not V1FQRCeilingRankNoCCode:
            raise TypeError("ceiling-rank refusal code must be exact")
        if type(self.required_action) is not V1FQRCeilingRankNextAction:
            raise TypeError("ceiling-rank next action must be exact")
        object.__setattr__(self, "reason", _identity(self.reason, name="reason"))
        if self.geometry_hash is not None:
            object.__setattr__(
                self,
                "geometry_hash",
                _identity(self.geometry_hash, name="geometry hash"),
            )
        for name in ("current_rank", "next_rank"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_int(value, name=name))
        flags = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank refusal flags must be exact")
        if any(flags):
            raise PermissionError("ceiling-rank refusal cannot open C or later stages")
        if self.schema != FQR_CEILING_RANK_NO_C_REFUSAL_SCHEMA:
            raise ValueError("ceiling-rank refusal schema is invalid")

    @property
    def refusal_hash(self) -> str:
        return content_hash(self, prefix=_REFUSAL_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankDecisionImage:
    """D/FD/KKT invariance image for one nested ceiling-rank view."""

    geometry_hash: str
    nested_view_hash: str
    rank: int
    selected_packed_action_hash: str
    decision_receipt_hash: str
    D_sampling_decision_invariant: bool
    FD_tail_decision_invariant: bool
    KKT_numerical_decision_invariant: bool
    complete_decision_image_invariant: bool
    _issuance_token: InitVar[object]
    source_issued: bool = True
    C_stream_open_authority: bool = False
    schema: str = FQR_CEILING_RANK_DECISION_IMAGE_SCHEMA
    _image_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _IMAGE_TOKEN:
            raise PermissionError("ceiling-rank decision images are issuer-only")
        for name in (
            "geometry_hash",
            "nested_view_hash",
            "selected_packed_action_hash",
            "decision_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _positive_int(self.rank, name="decision rank")
        flags = (
            self.D_sampling_decision_invariant,
            self.FD_tail_decision_invariant,
            self.KKT_numerical_decision_invariant,
            self.complete_decision_image_invariant,
            self.source_issued,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank decision-image flags must be exact")
        if self.complete_decision_image_invariant is not all(flags[:3]):
            raise ValueError("complete decision-image invariance does not replay")
        if flags[4:] != (True, False):
            raise PermissionError("decision image cannot itself open C")
        if self.schema != FQR_CEILING_RANK_DECISION_IMAGE_SCHEMA:
            raise ValueError("ceiling-rank decision-image schema is invalid")
        object.__setattr__(self, "rank", rank)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "geometry_hash": self.geometry_hash,
            "nested_view_hash": self.nested_view_hash,
            "rank": self.rank,
            "selected_packed_action_hash": self.selected_packed_action_hash,
            "decision_receipt_hash": self.decision_receipt_hash,
            "D_sampling_decision_invariant": self.D_sampling_decision_invariant,
            "FD_tail_decision_invariant": self.FD_tail_decision_invariant,
            "KKT_numerical_decision_invariant": self.KKT_numerical_decision_invariant,
            "complete_decision_image_invariant": (
                self.complete_decision_image_invariant
            ),
            "source_issued": self.source_issued,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def image_hash(self) -> str:
        cached = self._image_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_IMAGE_PREFIX)
        cached = self._image_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank decision image was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("ceiling-rank decision image hash does not match")
        object.__setattr__(self, "_image_hash", observed)
        return observed


def issue_v1_fqr_ceiling_rank_decision_image(
    *,
    geometry: V1FQRCeilingRankGeometry,
    nested_view: V1FQRCeilingRankNestedView,
    selected_packed_action_hash: str,
    decision_receipt_hash: str,
    D_sampling_decision_invariant: bool,
    FD_tail_decision_invariant: bool,
    KKT_numerical_decision_invariant: bool,
) -> V1FQRCeilingRankDecisionImage:
    """Bind one nested view to its source-side decision image."""

    if type(geometry) is not V1FQRCeilingRankGeometry:
        raise TypeError("decision image requires exact ceiling-rank geometry")
    if type(nested_view) is not V1FQRCeilingRankNestedView:
        raise TypeError("decision image requires exact nested view")
    geometry_hash = geometry.assert_integrity()
    if nested_view.geometry_hash != geometry_hash:
        raise ValueError("nested view belongs to another ceiling geometry")
    return V1FQRCeilingRankDecisionImage(
        geometry_hash=geometry_hash,
        nested_view_hash=nested_view.view_hash,
        rank=nested_view.rank,
        selected_packed_action_hash=selected_packed_action_hash,
        decision_receipt_hash=decision_receipt_hash,
        D_sampling_decision_invariant=D_sampling_decision_invariant,
        FD_tail_decision_invariant=FD_tail_decision_invariant,
        KKT_numerical_decision_invariant=KKT_numerical_decision_invariant,
        complete_decision_image_invariant=(
            D_sampling_decision_invariant
            and FD_tail_decision_invariant
            and KKT_numerical_decision_invariant
        ),
        _issuance_token=_IMAGE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1FQRCeilingRankFreezeForC:
    """C-only freeze for the first invariant nested rank."""

    geometry: V1FQRCeilingRankGeometry
    evaluated_decision_images: tuple[V1FQRCeilingRankDecisionImage, ...]
    frozen_rank: int
    frozen_decision_image_hash: str
    _issuance_token: InitVar[object]
    minimum_invariant_rank_selected: bool = True
    one_ceiling_fd_state_used: bool = True
    lower_rank_zero_padding_used: bool = False
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FQR_CEILING_RANK_FREEZE_SCHEMA
    _freeze_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _FREEZE_TOKEN:
            raise PermissionError("ceiling-rank C freeze is issuer-only")
        if type(self.geometry) is not V1FQRCeilingRankGeometry:
            raise TypeError("ceiling-rank freeze requires exact geometry")
        geometry_hash = self.geometry.assert_integrity()
        images = tuple(self.evaluated_decision_images)
        if not images or any(
            type(image) is not V1FQRCeilingRankDecisionImage for image in images
        ):
            raise TypeError("ceiling-rank freeze requires exact decision images")
        for image in images:
            image.assert_integrity()
        ranks = tuple(image.rank for image in images)
        expected_prefix = self.geometry.exposed_rank_lattice[: len(images)]
        invariant_indices = tuple(
            index
            for index, image in enumerate(images)
            if image.complete_decision_image_invariant
        )
        frozen = _positive_int(self.frozen_rank, name="frozen rank")
        if (
            ranks != expected_prefix
            or not invariant_indices
            or invariant_indices[0] != len(images) - 1
            or frozen != images[-1].rank
            or self.frozen_decision_image_hash != images[-1].image_hash
            or any(image.geometry_hash != geometry_hash for image in images)
        ):
            raise ValueError("ceiling-rank freeze is not the minimum invariant rank")
        flags = (
            self.minimum_invariant_rank_selected,
            self.one_ceiling_fd_state_used,
            self.lower_rank_zero_padding_used,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("ceiling-rank freeze flags must be exact")
        if flags != (True, True, False, True, False, False, False, False, False):
            raise PermissionError("ceiling-rank freeze crossed C-only authority")
        if self.schema != FQR_CEILING_RANK_FREEZE_SCHEMA:
            raise ValueError("ceiling-rank freeze schema is invalid")
        object.__setattr__(self, "evaluated_decision_images", images)
        object.__setattr__(self, "frozen_rank", frozen)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "geometry_hash": self.geometry.geometry_hash,
            "evaluated_decision_image_hashes": tuple(
                image.image_hash for image in self.evaluated_decision_images
            ),
            "evaluated_rank_prefix": tuple(
                image.rank for image in self.evaluated_decision_images
            ),
            "frozen_rank": self.frozen_rank,
            "frozen_decision_image_hash": self.frozen_decision_image_hash,
            "minimum_invariant_rank_selected": self.minimum_invariant_rank_selected,
            "one_ceiling_fd_state_used": self.one_ceiling_fd_state_used,
            "lower_rank_zero_padding_used": self.lower_rank_zero_padding_used,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def freeze_hash(self) -> str:
        cached = self._freeze_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FREEZE_PREFIX)
        cached = self._freeze_hash
        if cached is not None and cached != observed:
            raise ValueError("ceiling-rank freeze was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("ceiling-rank freeze hash does not match")
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


def freeze_or_refuse_v1_fqr_ceiling_rank_for_C(
    *,
    geometry: V1FQRCeilingRankGeometry,
    evaluated_decision_images: Sequence[V1FQRCeilingRankDecisionImage],
) -> V1FQRCeilingRankFreezeForC | V1FQRCeilingRankNoCRefusal:
    """Freeze the minimum invariant nested rank or return the next lawful action."""

    if type(geometry) is not V1FQRCeilingRankGeometry:
        raise TypeError("ceiling-rank freeze requires exact geometry")
    geometry_hash = geometry.assert_integrity()
    views = {view.rank: view for view in geometry.nested_views()}
    images = tuple(evaluated_decision_images)
    if any(type(image) is not V1FQRCeilingRankDecisionImage for image in images):
        raise TypeError("ceiling-rank freeze requires exact decision images")
    for image in images:
        image.assert_integrity()
    ranks = tuple(image.rank for image in images)
    expected_prefix = geometry.exposed_rank_lattice[: len(images)]
    if (
        ranks != expected_prefix
        or any(image.geometry_hash != geometry_hash for image in images)
        or any(image.nested_view_hash != views[image.rank].view_hash for image in images)
    ):
        return V1FQRCeilingRankNoCRefusal(
            code=V1FQRCeilingRankNoCCode.SOURCE_DECISION_IMAGE_INCOMPLETE,
            reason="decision images are stale, post-selected, or non-prefix",
            required_action=V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL,
            geometry_hash=geometry_hash,
        )
    invariant_indices = tuple(
        index
        for index, image in enumerate(images)
        if image.complete_decision_image_invariant
    )
    if invariant_indices:
        first = invariant_indices[0]
        if first != len(images) - 1:
            return V1FQRCeilingRankNoCRefusal(
                code=V1FQRCeilingRankNoCCode.SOURCE_DECISION_IMAGE_INCOMPLETE,
                reason="rank evaluation continued after the first invariant image",
                required_action=V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL,
                geometry_hash=geometry_hash,
                current_rank=images[first].rank,
            )
        frozen = images[first]
        return V1FQRCeilingRankFreezeForC(
            geometry=geometry,
            evaluated_decision_images=images,
            frozen_rank=frozen.rank,
            frozen_decision_image_hash=frozen.image_hash,
            _issuance_token=_FREEZE_TOKEN,
        )
    current = images[-1] if images else None
    current_rank = None if current is None else current.rank
    next_rank = (
        geometry.exposed_rank_lattice[0]
        if current is None
        else (
            geometry.exposed_rank_lattice[len(images)]
            if len(images) < len(geometry.exposed_rank_lattice)
            else None
        )
    )
    if current is not None and not current.D_sampling_decision_invariant:
        return V1FQRCeilingRankNoCRefusal(
            code=V1FQRCeilingRankNoCCode.D_SAMPLING_AMBIGUOUS,
            reason="nested decision image is not invariant over D sampling",
            required_action=V1FQRCeilingRankNextAction.ACQUIRE_MORE_D_FOR_SAMPLING,
            geometry_hash=geometry_hash,
            current_rank=current_rank,
        )
    if current is not None and not current.FD_tail_decision_invariant:
        if next_rank is None:
            return V1FQRCeilingRankNoCRefusal(
                code=V1FQRCeilingRankNoCCode.MAXIMUM_CEILING_RANK_AMBIGUOUS,
                reason="maximum admitted ceiling rank remains FD-tail ambiguous",
                required_action=V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL,
                geometry_hash=geometry_hash,
                current_rank=current_rank,
            )
        return V1FQRCeilingRankNoCRefusal(
            code=V1FQRCeilingRankNoCCode.NESTED_RANK_REFINEMENT_AVAILABLE,
            reason="evaluate the next nested rank from the same ceiling FD state",
            required_action=(
                V1FQRCeilingRankNextAction.EVALUATE_NEXT_NESTED_RANK_FROM_CEILING_STATE
            ),
            geometry_hash=geometry_hash,
            current_rank=current_rank,
            next_rank=next_rank,
        )
    if current is not None and not current.KKT_numerical_decision_invariant:
        return V1FQRCeilingRankNoCRefusal(
            code=V1FQRCeilingRankNoCCode.KKT_NUMERICAL_AMBIGUOUS,
            reason="nested decision image is not invariant over solver uncertainty",
            required_action=V1FQRCeilingRankNextAction.CONTINUE_REGISTERED_KKT_SOLVER,
            geometry_hash=geometry_hash,
            current_rank=current_rank,
        )
    if next_rank is not None:
        return V1FQRCeilingRankNoCRefusal(
            code=(
                V1FQRCeilingRankNoCCode.SOURCE_DECISION_IMAGE_INCOMPLETE
                if current is None
                else V1FQRCeilingRankNoCCode.NESTED_RANK_REFINEMENT_AVAILABLE
            ),
            reason=(
                "evaluate the next nested rank from the same ceiling FD state"
            ),
            required_action=(
                V1FQRCeilingRankNextAction.EVALUATE_NEXT_NESTED_RANK_FROM_CEILING_STATE
            ),
            geometry_hash=geometry_hash,
            current_rank=current_rank,
            next_rank=next_rank,
        )
    return V1FQRCeilingRankNoCRefusal(
        code=V1FQRCeilingRankNoCCode.MAXIMUM_CEILING_RANK_AMBIGUOUS,
        reason="maximum admitted ceiling rank remains FD-tail ambiguous",
        required_action=V1FQRCeilingRankNextAction.PHYSICAL_NO_C_REFUSAL,
        geometry_hash=geometry_hash,
        current_rank=current_rank,
    )


def psd_quadratic_form_from_factor_rows(
    factor_rows: Sequence[Sequence[float]],
    direction: Sequence[float],
) -> float:
    """Small CPU reference for Q(z)=||Hz||² used by mathematical tests."""

    rows = tuple(tuple(float(value) for value in row) for row in factor_rows)
    z = tuple(float(value) for value in direction)
    if not rows:
        return 0.0
    width = len(z)
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("factor rows and direction have inconsistent shapes")
    result = 0.0
    for row in rows:
        dot = math.fsum(a * b for a, b in zip(row, z, strict=True))
        result += dot * dot
    return result


__all__ = [
    "FQR_CEILING_RANK_DECISION_IMAGE_SCHEMA",
    "FQR_CEILING_RANK_FREEZE_SCHEMA",
    "FQR_CEILING_RANK_GEOMETRY_SCHEMA",
    "FQR_CEILING_RANK_MEMORY_FORECAST_SCHEMA",
    "FQR_CEILING_RANK_NESTED_VIEW_SCHEMA",
    "FQR_CEILING_RANK_NO_C_REFUSAL_SCHEMA",
    "V1FQRCeilingRankDecisionImage",
    "V1FQRCeilingRankFreezeForC",
    "V1FQRCeilingRankGeometry",
    "V1FQRCeilingRankMemoryForecast",
    "V1FQRCeilingRankNestedResidualEnvelope",
    "V1FQRCeilingRankNestedView",
    "V1FQRCeilingRankNextAction",
    "V1FQRCeilingRankNoCCode",
    "V1FQRCeilingRankNoCRefusal",
    "canonical_fqr_ceiling_rank_lattice_for_route_count",
    "forecast_v1_fqr_ceiling_rank_memory",
    "freeze_or_refuse_v1_fqr_ceiling_rank_for_C",
    "issue_v1_fqr_ceiling_rank_decision_image",
    "issue_v1_fqr_ceiling_rank_geometry",
    "psd_quadratic_form_from_factor_rows",
]
