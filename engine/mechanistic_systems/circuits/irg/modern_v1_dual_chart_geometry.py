"""Bounded-memory dual-stream chart contracts for canonical modern v1.

The conserving Aumann--Shapley row and the literal relaxed-gate Jacobian row
are generated from one activation capture/VJP but have different owners:

* ``-G phi c`` feeds corrected work ``F``;
* ``-G 1[a>0] c`` feeds the response geometry ``Q``.

Production rows are route-chunked, consumed ephemerally, and discarded after
updating the corrected-work accumulator, the Jacobian FD geometry, and an
aggregate mismatch witness.  :class:`V1DualChartGeometry` binds those reduced
artifacts and intentionally contains no raw route arrays.  A separate
diagnostic observation type may own two immutable rows only below a strict
small-domain cap and has no production authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Final

import numpy as np

from .contracts import content_hash


DUAL_CHART_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_streaming_dual_corrected_work_jacobian_chart_v1"
)
DUAL_CHART_MISMATCH_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_streaming_dual_chart_mismatch_receipt_v1"
)
DUAL_CHART_DIAGNOSTIC_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_dual_chart_diagnostic_row_v1"
)
POSITIVE_REMOVAL_SIGN: Final = "z_j_positive_means_gate_removal_v1"
CORRECTED_EVIDENCE_EQUATION: Final = (
    "positive_removal_corrected_evidence_equals_minus_G_times_phi_times_c_v1"
)
LITERAL_JACOBIAN_EQUATION: Final = (
    "positive_removal_response_jacobian_equals_minus_G_times_"
    "indicator_preactivation_positive_times_c_v1"
)
RELU_KINK_POLICY: Final = "relu_gate_derivative_at_zero_is_zero_v1"
DUAL_CHART_STREAMING_POLICY: Final = (
    "route_chunks_consumed_once_into_F_moments_Jacobian_FD_and_mismatch_"
    "accumulator_then_discarded_v1"
)
DUAL_CHART_AUTHORITY_SCOPE: Final = (
    "streaming_chart_reduced_numeric_sources_only_no_candidate_fqr_oracle_"
    "mask_commit_or_terminal_authority_v1"
)
DUAL_CHART_DIAGNOSTIC_AUTHORITY_SCOPE: Final = (
    "small_domain_diagnostic_only_no_production_candidate_fqr_oracle_"
    "mask_commit_or_terminal_authority_v1"
)
MAX_DIAGNOSTIC_DUAL_CHART_ROUTES: Final = 65_536

_MISMATCH_PREFIX: Final = "partition-free-v1-dual-chart-mismatch-"
_GEOMETRY_PREFIX: Final = "partition-free-v1-dual-chart-geometry-"
_ROW_PREFIX: Final = "partition-free-v1-dual-chart-diagnostic-row-array-"
_DIAGNOSTIC_PREFIX: Final = "partition-free-v1-dual-chart-diagnostic-row-"
_LE_F64 = np.dtype("<f8")


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _immutable_row(value: object, *, route_count: int, name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float64 route row") from error
    if source.ndim != 1 or source.shape != (route_count,):
        raise ValueError(f"{name} must have shape ({route_count},)")
    try:
        canonical = np.array(source, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite float64 route row") from error
    if not np.isfinite(canonical).all():
        raise ValueError(f"{name} must be finite")
    canonical[canonical == 0.0] = 0.0
    result = np.frombuffer(canonical.tobytes(order="C"), dtype=_LE_F64)
    result.setflags(write=False)
    return result


def _scaled_row_mismatch(
    corrected: np.ndarray,
    jacobian: np.ndarray,
) -> tuple[int, float, float, float, float]:
    difference = np.subtract(corrected, jacobian, dtype=np.float64)
    if not np.isfinite(difference).all():
        raise ValueError("corrected/Jacobian mismatch is not finite")
    nonzero = int(np.count_nonzero(difference))
    if nonzero == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    absolute = np.abs(difference)
    scale = float(np.max(absolute))
    normalized = absolute / scale
    mean_absolute = scale * float(np.mean(normalized, dtype=np.float64))
    root_mean_square = scale * math.sqrt(
        float(np.mean(np.square(normalized), dtype=np.float64))
    )
    signed_mean = scale * float(np.mean(difference / scale, dtype=np.float64))
    values = (mean_absolute, root_mean_square, scale, signed_mean)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("corrected/Jacobian mismatch diagnostics are not finite")
    return nonzero, mean_absolute, root_mean_square, scale, signed_mean


@dataclass(frozen=True, slots=True)
class V1DualChartMismatchReceipt:
    """Streaming aggregate of ``corrected evidence - literal Jacobian``.

    All nonnegative sums are outward upper enclosures from the registered
    chunk accumulator.  The signed sum retains an explicit radius.  The
    receipt carries only scalar sufficient statistics and source identities.
    """

    campaign_contract_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    observed_context_count: int
    observed_context_axis_hash: str
    observed_draw_axis_hash: str
    corrected_evidence_stream_source_hash: str
    literal_jacobian_stream_source_hash: str
    shared_activation_capture_stream_hash: str
    shared_vjp_stream_hash: str
    accumulator_source_hash: str
    mismatch_nonzero_coordinate_count: int
    absolute_difference_sum_upper: float
    squared_difference_sum_upper: float
    absolute_difference_max_upper: float
    signed_difference_sum_center: float
    signed_difference_sum_radius: float
    raw_route_rows_retained: bool = False
    diagnostic_only: bool = True
    candidate_authority: bool = False
    fqr_certification_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = DUAL_CHART_MISMATCH_RECEIPT_SCHEMA
    scalar_coordinate_count: int = field(init=False)
    mismatch_mean_absolute_upper: float = field(init=False)
    mismatch_root_mean_square_upper: float = field(init=False)
    mismatch_signed_mean_center: float = field(init=False)
    mismatch_signed_mean_radius: float = field(init=False)
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "current_mask_hash",
            "route_axis_hash",
            "observed_context_axis_hash",
            "observed_draw_axis_hash",
            "corrected_evidence_stream_source_hash",
            "literal_jacobian_stream_source_hash",
            "shared_activation_capture_stream_hash",
            "shared_vjp_stream_hash",
            "accumulator_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        route_count = _positive_int(self.route_count, name="route count")
        contexts = _positive_int(
            self.observed_context_count,
            name="observed context count",
        )
        scalar_count = route_count * contexts
        nonzero = _nonnegative_int(
            self.mismatch_nonzero_coordinate_count,
            name="mismatch nonzero coordinate count",
        )
        if nonzero > scalar_count:
            raise ValueError("mismatch nonzero count exceeds streamed coordinates")
        absolute_sum = _finite_nonnegative(
            self.absolute_difference_sum_upper,
            name="absolute difference sum upper",
        )
        squared_sum = _finite_nonnegative(
            self.squared_difference_sum_upper,
            name="squared difference sum upper",
        )
        maximum = _finite_nonnegative(
            self.absolute_difference_max_upper,
            name="absolute difference maximum upper",
        )
        signed_center = _finite(
            self.signed_difference_sum_center,
            name="signed difference sum center",
        )
        signed_radius = _finite_nonnegative(
            self.signed_difference_sum_radius,
            name="signed difference sum radius",
        )
        if nonzero == 0 and any(
            value != 0.0
            for value in (
                absolute_sum,
                squared_sum,
                maximum,
                signed_center,
                signed_radius,
            )
        ):
            raise ValueError("an exact-zero mismatch count requires zero aggregates")
        if absolute_sum + signed_radius < abs(signed_center):
            raise ValueError("signed mismatch exceeds the absolute mismatch enclosure")
        mean_absolute = absolute_sum / scalar_count
        rms = math.sqrt(squared_sum / scalar_count)
        signed_mean = signed_center / scalar_count
        signed_mean_radius = signed_radius / scalar_count
        if any(
            not math.isfinite(value)
            for value in (mean_absolute, rms, signed_mean, signed_mean_radius)
        ):
            raise ValueError("streaming mismatch diagnostics are nonfinite")
        if (
            self.raw_route_rows_retained is not False
            or self.diagnostic_only is not True
            or self.candidate_authority is not False
            or self.fqr_certification_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("mismatch receipt is diagnostic reduced state only")
        if self.schema != DUAL_CHART_MISMATCH_RECEIPT_SCHEMA:
            raise ValueError("dual-chart mismatch receipt schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "observed_context_count", contexts)
        object.__setattr__(self, "scalar_coordinate_count", scalar_count)
        object.__setattr__(self, "mismatch_nonzero_coordinate_count", nonzero)
        object.__setattr__(self, "absolute_difference_sum_upper", absolute_sum)
        object.__setattr__(self, "squared_difference_sum_upper", squared_sum)
        object.__setattr__(self, "absolute_difference_max_upper", maximum)
        object.__setattr__(self, "signed_difference_sum_center", signed_center)
        object.__setattr__(self, "signed_difference_sum_radius", signed_radius)
        object.__setattr__(self, "mismatch_mean_absolute_upper", mean_absolute)
        object.__setattr__(self, "mismatch_root_mean_square_upper", rms)
        object.__setattr__(self, "mismatch_signed_mean_center", signed_mean)
        object.__setattr__(self, "mismatch_signed_mean_radius", signed_mean_radius)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "campaign_contract_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "observed_context_count",
                "observed_context_axis_hash",
                "observed_draw_axis_hash",
                "corrected_evidence_stream_source_hash",
                "literal_jacobian_stream_source_hash",
                "shared_activation_capture_stream_hash",
                "shared_vjp_stream_hash",
                "accumulator_source_hash",
                "mismatch_nonzero_coordinate_count",
                "absolute_difference_sum_upper",
                "squared_difference_sum_upper",
                "absolute_difference_max_upper",
                "signed_difference_sum_center",
                "signed_difference_sum_radius",
                "scalar_coordinate_count",
                "mismatch_mean_absolute_upper",
                "mismatch_root_mean_square_upper",
                "mismatch_signed_mean_center",
                "mismatch_signed_mean_radius",
                "raw_route_rows_retained",
                "diagnostic_only",
                "candidate_authority",
                "fqr_certification_authority",
                "mask_commit_authority",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_MISMATCH_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("dual-chart mismatch receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("dual-chart mismatch receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DualChartGeometry:
    """Production descriptor for streamed corrected-work/Jacobian geometry.

    It binds only reduced state.  Route rows never become dataclass fields and
    therefore cannot be checkpointed accidentally through this contract.
    """

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    context_population_hash: str
    sampling_law_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    observed_context_count: int
    observed_context_axis_hash: str
    observed_draw_axis_hash: str
    shared_activation_capture_stream_hash: str
    shared_vjp_stream_hash: str
    corrected_evidence_stream_source_hash: str
    literal_jacobian_stream_source_hash: str
    corrected_work_accumulator_hash: str
    corrected_work_accumulator_schema: str
    literal_jacobian_fd_design_hash: str
    literal_jacobian_fd_geometry_hash: str
    literal_jacobian_fd_geometry_schema: str
    mismatch_receipt: V1DualChartMismatchReceipt = field(repr=False, compare=False)
    ephemeral_route_chunk_size: int = 1 << 20
    maximum_simultaneous_route_chunks: int = 2
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    corrected_evidence_equation: str = CORRECTED_EVIDENCE_EQUATION
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION
    relu_kink_policy: str = RELU_KINK_POLICY
    streaming_policy: str = DUAL_CHART_STREAMING_POLICY
    same_capture_and_vjp: bool = True
    f_uses_corrected_evidence_only: bool = True
    q_uses_literal_jacobian_only: bool = True
    raw_context_route_rows_retained: bool = False
    chunks_discarded_after_reduced_updates: bool = True
    candidate_authority: bool = False
    fqr_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = DUAL_CHART_AUTHORITY_SCOPE
    schema: str = DUAL_CHART_GEOMETRY_SCHEMA
    _geometry_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "context_population_hash",
            "sampling_law_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "route_axis_hash",
            "observed_context_axis_hash",
            "observed_draw_axis_hash",
            "shared_activation_capture_stream_hash",
            "shared_vjp_stream_hash",
            "corrected_evidence_stream_source_hash",
            "literal_jacobian_stream_source_hash",
            "corrected_work_accumulator_hash",
            "corrected_work_accumulator_schema",
            "literal_jacobian_fd_design_hash",
            "literal_jacobian_fd_geometry_hash",
            "literal_jacobian_fd_geometry_schema",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        route_count = _positive_int(self.route_count, name="route count")
        contexts = _positive_int(
            self.observed_context_count,
            name="observed context count",
        )
        chunk_size = _positive_int(
            self.ephemeral_route_chunk_size,
            name="ephemeral route chunk size",
        )
        simultaneous = _positive_int(
            self.maximum_simultaneous_route_chunks,
            name="maximum simultaneous route chunks",
        )
        if simultaneous > 2:
            raise ValueError("dual chart may retain at most the current pair of chunks")
        if type(self.mismatch_receipt) is not V1DualChartMismatchReceipt:
            raise TypeError("dual chart requires the typed streaming mismatch receipt")
        self.mismatch_receipt.assert_integrity()
        mismatch = self.mismatch_receipt
        coordinate_graph = (
            (mismatch.campaign_contract_hash, self.campaign_contract_hash),
            (mismatch.current_mask_hash, self.current_mask_hash),
            (mismatch.proposal_generation, generation),
            (mismatch.route_count, route_count),
            (mismatch.route_axis_hash, self.route_axis_hash),
            (mismatch.observed_context_count, contexts),
            (mismatch.observed_context_axis_hash, self.observed_context_axis_hash),
            (mismatch.observed_draw_axis_hash, self.observed_draw_axis_hash),
            (
                mismatch.corrected_evidence_stream_source_hash,
                self.corrected_evidence_stream_source_hash,
            ),
            (
                mismatch.literal_jacobian_stream_source_hash,
                self.literal_jacobian_stream_source_hash,
            ),
            (
                mismatch.shared_activation_capture_stream_hash,
                self.shared_activation_capture_stream_hash,
            ),
            (mismatch.shared_vjp_stream_hash, self.shared_vjp_stream_hash),
        )
        if any(left != right for left, right in coordinate_graph):
            raise ValueError("dual-chart mismatch receipt names another streamed chart")
        expected_semantics = (
            self.positive_removal_sign == POSITIVE_REMOVAL_SIGN
            and self.corrected_evidence_equation == CORRECTED_EVIDENCE_EQUATION
            and self.literal_jacobian_equation == LITERAL_JACOBIAN_EQUATION
            and self.relu_kink_policy == RELU_KINK_POLICY
            and self.streaming_policy == DUAL_CHART_STREAMING_POLICY
            and self.same_capture_and_vjp is True
            and self.f_uses_corrected_evidence_only is True
            and self.q_uses_literal_jacobian_only is True
            and self.raw_context_route_rows_retained is False
            and self.chunks_discarded_after_reduced_updates is True
        )
        if not expected_semantics:
            raise ValueError("dual chart semantics differ from canonical streaming v1")
        if any(
            getattr(self, name) is not False
            for name in (
                "candidate_authority",
                "fqr_certification_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            raise PermissionError(
                "dual chart reduced state cannot grant action authority"
            )
        if self.authority_scope != DUAL_CHART_AUTHORITY_SCOPE:
            raise ValueError("dual chart authority scope is invalid")
        if self.schema != DUAL_CHART_GEOMETRY_SCHEMA:
            raise ValueError("dual chart schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "observed_context_count", contexts)
        object.__setattr__(self, "ephemeral_route_chunk_size", chunk_size)
        object.__setattr__(self, "maximum_simultaneous_route_chunks", simultaneous)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "context_population_hash": self.context_population_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "observed_context_count": self.observed_context_count,
            "observed_context_axis_hash": self.observed_context_axis_hash,
            "observed_draw_axis_hash": self.observed_draw_axis_hash,
            "shared_activation_capture_stream_hash": self.shared_activation_capture_stream_hash,
            "shared_vjp_stream_hash": self.shared_vjp_stream_hash,
            "corrected_evidence_stream_source_hash": self.corrected_evidence_stream_source_hash,
            "literal_jacobian_stream_source_hash": self.literal_jacobian_stream_source_hash,
            "corrected_work_accumulator_hash": self.corrected_work_accumulator_hash,
            "corrected_work_accumulator_schema": self.corrected_work_accumulator_schema,
            "literal_jacobian_fd_design_hash": self.literal_jacobian_fd_design_hash,
            "literal_jacobian_fd_geometry_hash": self.literal_jacobian_fd_geometry_hash,
            "literal_jacobian_fd_geometry_schema": self.literal_jacobian_fd_geometry_schema,
            "mismatch_receipt_hash": self.mismatch_receipt.receipt_hash,
            "ephemeral_route_chunk_size": self.ephemeral_route_chunk_size,
            "maximum_simultaneous_route_chunks": self.maximum_simultaneous_route_chunks,
            "positive_removal_sign": self.positive_removal_sign,
            "corrected_evidence_equation": self.corrected_evidence_equation,
            "literal_jacobian_equation": self.literal_jacobian_equation,
            "relu_kink_policy": self.relu_kink_policy,
            "streaming_policy": self.streaming_policy,
            "same_capture_and_vjp": self.same_capture_and_vjp,
            "f_uses_corrected_evidence_only": self.f_uses_corrected_evidence_only,
            "q_uses_literal_jacobian_only": self.q_uses_literal_jacobian_only,
            "raw_context_route_rows_retained": self.raw_context_route_rows_retained,
            "chunks_discarded_after_reduced_updates": self.chunks_discarded_after_reduced_updates,
            "candidate_authority": self.candidate_authority,
            "fqr_certification_authority": self.fqr_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def geometry_hash(self) -> str:
        cached = self._geometry_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.mismatch_receipt.assert_integrity()
        observed = content_hash(self._payload(), prefix=_GEOMETRY_PREFIX)
        cached = self._geometry_hash
        if cached is not None and cached != observed:
            raise ValueError("dual chart geometry was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("dual chart geometry hash does not match")
        object.__setattr__(self, "_geometry_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DualChartDiagnosticObservation:
    """Small-domain row pair for unit tests and diagnostic parity only."""

    route_count: int
    route_axis_hash: str
    context_id: str
    draw_id: str
    shared_activation_capture_hash: str
    shared_vjp_hash: str
    corrected_evidence_row: np.ndarray = field(repr=False, compare=False)
    literal_jacobian_row: np.ndarray = field(repr=False, compare=False)
    positive_removal_sign: str = POSITIVE_REMOVAL_SIGN
    corrected_evidence_equation: str = CORRECTED_EVIDENCE_EQUATION
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION
    relu_kink_policy: str = RELU_KINK_POLICY
    diagnostic_only: bool = True
    production_authority: bool = False
    raw_route_rows_retained: bool = True
    authority_scope: str = DUAL_CHART_DIAGNOSTIC_AUTHORITY_SCOPE
    schema: str = DUAL_CHART_DIAGNOSTIC_OBSERVATION_SCHEMA
    mismatch_route_count: int = field(init=False)
    mismatch_mean_absolute: float = field(init=False)
    mismatch_root_mean_square: float = field(init=False)
    mismatch_linf: float = field(init=False)
    mismatch_signed_mean: float = field(init=False)
    _corrected_row_hash: str = field(init=False, repr=False, compare=False)
    _jacobian_row_hash: str = field(init=False, repr=False, compare=False)
    _observation_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        route_count = _positive_int(self.route_count, name="diagnostic route count")
        if route_count > MAX_DIAGNOSTIC_DUAL_CHART_ROUTES:
            raise MemoryError(
                "row-owning dual-chart observations are diagnostic-only and "
                f"refuse more than {MAX_DIAGNOSTIC_DUAL_CHART_ROUTES} routes"
            )
        for name in (
            "route_axis_hash",
            "context_id",
            "draw_id",
            "shared_activation_capture_hash",
            "shared_vjp_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        corrected = _immutable_row(
            self.corrected_evidence_row,
            route_count=route_count,
            name="diagnostic corrected evidence row",
        )
        jacobian = _immutable_row(
            self.literal_jacobian_row,
            route_count=route_count,
            name="diagnostic literal Jacobian row",
        )
        diagnostics = _scaled_row_mismatch(corrected, jacobian)
        if (
            self.positive_removal_sign != POSITIVE_REMOVAL_SIGN
            or self.corrected_evidence_equation != CORRECTED_EVIDENCE_EQUATION
            or self.literal_jacobian_equation != LITERAL_JACOBIAN_EQUATION
            or self.relu_kink_policy != RELU_KINK_POLICY
        ):
            raise ValueError("diagnostic dual-chart semantics are invalid")
        if (
            self.diagnostic_only is not True
            or self.production_authority is not False
            or self.raw_route_rows_retained is not True
        ):
            raise PermissionError(
                "a row-owning dual chart can never gain production authority"
            )
        if self.authority_scope != DUAL_CHART_DIAGNOSTIC_AUTHORITY_SCOPE:
            raise ValueError("diagnostic dual-chart authority scope is invalid")
        if self.schema != DUAL_CHART_DIAGNOSTIC_OBSERVATION_SCHEMA:
            raise ValueError("diagnostic dual-chart schema is invalid")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "corrected_evidence_row", corrected)
        object.__setattr__(self, "literal_jacobian_row", jacobian)
        object.__setattr__(self, "mismatch_route_count", diagnostics[0])
        object.__setattr__(self, "mismatch_mean_absolute", diagnostics[1])
        object.__setattr__(self, "mismatch_root_mean_square", diagnostics[2])
        object.__setattr__(self, "mismatch_linf", diagnostics[3])
        object.__setattr__(self, "mismatch_signed_mean", diagnostics[4])
        object.__setattr__(
            self,
            "_corrected_row_hash",
            content_hash(corrected, prefix=_ROW_PREFIX),
        )
        object.__setattr__(
            self,
            "_jacobian_row_hash",
            content_hash(jacobian, prefix=_ROW_PREFIX),
        )
        self.assert_integrity()

    @property
    def rows_identical(self) -> bool:
        return self.mismatch_route_count == 0

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "context_id": self.context_id,
            "draw_id": self.draw_id,
            "shared_activation_capture_hash": self.shared_activation_capture_hash,
            "shared_vjp_hash": self.shared_vjp_hash,
            "corrected_evidence_row_hash": self._corrected_row_hash,
            "literal_jacobian_row_hash": self._jacobian_row_hash,
            "mismatch_route_count": self.mismatch_route_count,
            "mismatch_mean_absolute": self.mismatch_mean_absolute,
            "mismatch_root_mean_square": self.mismatch_root_mean_square,
            "mismatch_linf": self.mismatch_linf,
            "mismatch_signed_mean": self.mismatch_signed_mean,
            "positive_removal_sign": self.positive_removal_sign,
            "corrected_evidence_equation": self.corrected_evidence_equation,
            "literal_jacobian_equation": self.literal_jacobian_equation,
            "relu_kink_policy": self.relu_kink_policy,
            "diagnostic_only": self.diagnostic_only,
            "production_authority": self.production_authority,
            "raw_route_rows_retained": self.raw_route_rows_retained,
            "authority_scope": self.authority_scope,
        }

    @property
    def observation_hash(self) -> str:
        cached = self._observation_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if content_hash(self.corrected_evidence_row, prefix=_ROW_PREFIX) != (
            self._corrected_row_hash
        ):
            raise ValueError("diagnostic corrected row was tampered")
        if content_hash(self.literal_jacobian_row, prefix=_ROW_PREFIX) != (
            self._jacobian_row_hash
        ):
            raise ValueError("diagnostic Jacobian row was tampered")
        observed = content_hash(self._payload(), prefix=_DIAGNOSTIC_PREFIX)
        cached = self._observation_hash
        if cached is not None and cached != observed:
            raise ValueError("diagnostic dual-chart observation was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("diagnostic dual-chart observation hash does not match")
        object.__setattr__(self, "_observation_hash", observed)
        return observed


__all__ = [
    "CORRECTED_EVIDENCE_EQUATION",
    "DUAL_CHART_AUTHORITY_SCOPE",
    "DUAL_CHART_DIAGNOSTIC_AUTHORITY_SCOPE",
    "DUAL_CHART_DIAGNOSTIC_OBSERVATION_SCHEMA",
    "DUAL_CHART_GEOMETRY_SCHEMA",
    "DUAL_CHART_MISMATCH_RECEIPT_SCHEMA",
    "DUAL_CHART_STREAMING_POLICY",
    "LITERAL_JACOBIAN_EQUATION",
    "MAX_DIAGNOSTIC_DUAL_CHART_ROUTES",
    "POSITIVE_REMOVAL_SIGN",
    "RELU_KINK_POLICY",
    "V1DualChartDiagnosticObservation",
    "V1DualChartGeometry",
    "V1DualChartMismatchReceipt",
]
