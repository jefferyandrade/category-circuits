"""Historical CUDA source issuer for nominal decision-level shell replay.

This module is the first production joint after the small high-precision
reference theorem in :mod:`modern_v1_relaxed_kkt`.  It deliberately emits the
same five compact source receipts consumed by the existing theorem certificate:

* complete outside-shell interval scan,
* active-shell restricted operator,
* robust primal/dual bound,
* complete worst-case hard F/Q replay,
* exhaustive bounded all-admissible-shell-face enumeration.

The implementation remains useful as provenance for CUDA custody, bounded
tiling, HBM admission, and nominal fixed-shell replay.  Its strong-convexity and dual receipts are not used to
bound the movement of every alternative admissible optimizer.  Consequently
the public issuer now fails closed before emitting
``V1RelaxedKKTDecisionLevelRobustInvarianceCertificate``.  A successor must
add a source-certified variable-solution movement bound and exhaustive hard
action replay over the enlarged shell intervals.  This module does not add a
CPU production fallback and does not materialize a route-scale key/index axis
on the host.
"""

from __future__ import annotations

import math
from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Final, Literal, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import RemovedRouteIndexSet
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_certified_linalg import (
    certified_gram_operator_min_eigenvalue_lower_bound,
)
from .modern_v1_literal_j_projection_inference import (
    V1LiteralJFixedDirectionProjectionReceipt,
)
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTActiveShellRestrictedOperatorReceipt,
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTAllAdmissibleShellFacesReceipt,
    V1RelaxedKKTBranch,
    V1RelaxedKKTBranchResult,
    V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest,
    V1RelaxedKKTDecisionLevelTheoremStateLease,
    V1RelaxedKKTDecisionLevelRobustInvarianceCertificate,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTDeviceShellExecutionBundle,
    V1RelaxedKKTGeometryUncertaintySet,
    V1RelaxedKKTOutsideShellIntervalScanReceipt,
    V1RelaxedKKTPolicy,
    V1RelaxedKKTProblem,
    V1RelaxedKKTRobustDualBoundReceipt,
    V1RelaxedKKTWorstCaseHardFQReplayReceipt,
    _DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    _ROBUST_SENSITIVITY_UNBOUNDED,
    _decision_level_block_axis,
    _decision_level_interval_square,
    _decision_level_outward_sum,
    _decision_level_topological_orders,
    issue_v1_relaxed_kkt_decision_level_robust_invariance_from_source_receipts,
)


DEFAULT_H200_HBM_CEILING_FRACTION: Final = 0.85
DEFAULT_ROUTE_TILE_SIZE: Final = 1 << 20
DEFAULT_MAXIMUM_SHELL_ROUTES: Final = 8
DEFAULT_MAXIMUM_ADMISSIBLE_FACES: Final = 6_561
DEFAULT_MAXIMUM_ROUNDING_OUTCOMES: Final = 100_000
SCHEMA_DESIGN: Final = "partition_free_v1_decision_level_cuda_issuer_design_v1"
SCHEMA_FORECAST: Final = "partition_free_v1_decision_level_cuda_hbm_forecast_v2"
SCHEMA_TELEMETRY: Final = "partition_free_v1_decision_level_cuda_issuer_telemetry_v1"
SCHEMA_HASH_CUSTODY: Final = (
    "partition_free_v1_decision_level_cuda_shell_hash_custody_v1"
)
SCHEMA_PRIVATE_PRIMAL_BINDING: Final = (
    "partition_free_v1_decision_level_cuda_private_primal_binding_v1"
)
SCHEMA_PRIVATE_THEOREM_HOOK_REQUIREMENT: Final = (
    "partition_free_v1_decision_level_cuda_private_theorem_hook_requirement_v1"
)


class V1DecisionLevelCudaIssuerRefusalCode(str, Enum):
    """Typed fail-closed outcomes for the production CUDA issuer."""

    CPU_FALLBACK_FORBIDDEN = "cpu_fallback_forbidden"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    FULL_STATE_LEASE_REQUIRED = "full_state_lease_required"
    HASH_CUSTODY_REQUIRED = "hash_custody_required"
    TENSOR_CONTRACT = "tensor_contract"
    ROUTE_SCAN_INCOMPLETE = "route_scan_incomplete"
    ROUTE_SCALE_D2H_FORBIDDEN = "route_scale_d2h_forbidden"
    NONMONOTONE_Q_EARLY_STOP_FORBIDDEN = "nonmonotone_q_early_stop_forbidden"
    RANK_PADDING_FORBIDDEN = "rank_padding_forbidden"
    DECISION_BOUNDARY_TOUCHED = "decision_boundary_touched"
    SHELL_LIMIT_EXCEEDED = "shell_limit_exceeded"
    ROUNDING_OUTCOME_LIMIT_EXCEEDED = "rounding_outcome_limit_exceeded"
    PRIVATE_PRIMAL_CUSTODY_REQUIRED = "private_primal_custody_required"
    PRIVATE_THEOREM_STATE_HOOK_REQUIRED = "private_theorem_state_hook_required"
    LITERAL_J_LINEAGE_MISMATCH = "literal_j_lineage_mismatch"
    ALTERNATIVE_SOLUTION_SENSITIVITY_UNRESOLVED = (
        "alternative_solution_sensitivity_unresolved"
    )


class V1DecisionLevelCudaIssuerRefusal(RuntimeError):
    """Intentional refusal to substitute an unsafe source-issuer path."""

    def __init__(
        self,
        code: V1DecisionLevelCudaIssuerRefusalCode,
        message: str,
        **payload: object,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.payload = dict(payload)


def _refuse(
    code: V1DecisionLevelCudaIssuerRefusalCode,
    message: str,
    **payload: object,
) -> NoReturn:
    raise V1DecisionLevelCudaIssuerRefusal(code, message, **payload)


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return int(value)


def _finite_nonnegative(value: float, name: str) -> float:
    raw = float(value)
    if not math.isfinite(raw) or raw < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return raw


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaRouteTile:
    """One deterministic half-open physical-axis tile."""

    start: int
    stop: int
    route_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        start = _nonnegative_int(self.start, "tile start")
        stop = _positive_int(self.stop, "tile stop")
        if not start < stop <= self.route_count:
            raise ValueError("route tile is outside the physical route axis")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def length(self) -> int:
        return self.stop - self.start


def plan_v1_decision_level_cuda_route_tiles(
    *,
    route_count: int,
    route_tile_size: int,
) -> tuple[V1DecisionLevelCudaRouteTile, ...]:
    """Return the complete deterministic physical-axis tile plan."""

    total = _positive_int(route_count, "route_count")
    tile = _positive_int(route_tile_size, "route_tile_size")
    return tuple(
        V1DecisionLevelCudaRouteTile(
            start=start,
            stop=min(total, start + tile),
            route_count=total,
        )
        for start in range(0, total, tile)
    )


def assert_complete_v1_decision_level_cuda_route_scan(
    *,
    route_count: int,
    tiles: Sequence[V1DecisionLevelCudaRouteTile],
    scanned_route_count: int,
) -> None:
    """Fail closed unless every physical route was visited exactly once."""

    total = _positive_int(route_count, "route_count")
    scanned = _nonnegative_int(scanned_route_count, "scanned_route_count")
    normalized = tuple(tiles)
    if not normalized:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
            "decision-level CUDA issuer produced no route tiles",
        )
    cursor = 0
    for tile in normalized:
        if type(tile) is not V1DecisionLevelCudaRouteTile:
            raise TypeError("tiles must be exact V1DecisionLevelCudaRouteTile")
        if tile.route_count != total or tile.start != cursor:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "decision-level CUDA route tiles are not a complete ordered partition",
            )
        cursor = tile.stop
    if cursor != total or scanned != total:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
            "decision-level CUDA issuer did not scan the complete route axis",
            planned_route_count=total,
            scanned_route_count=scanned,
        )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaIssuerDesign:
    """Safety policy for source-issued CUDA decision-level receipts."""

    route_tile_size: int = DEFAULT_ROUTE_TILE_SIZE
    hbm_ceiling_fraction: float = DEFAULT_H200_HBM_CEILING_FRACTION
    maximum_shell_routes: int = DEFAULT_MAXIMUM_SHELL_ROUTES
    maximum_admissible_faces: int = DEFAULT_MAXIMUM_ADMISSIBLE_FACES
    maximum_rounding_outcomes: int = DEFAULT_MAXIMUM_ROUNDING_OUTCOMES
    require_complete_route_scan: bool = True
    allow_cpu_fallback: bool = False
    allow_full_key_index_d2h: bool = False
    allow_route_scale_row_retention: bool = False
    allow_nonmonotone_q_early_stop: bool = False
    allow_rank_padding: bool = False
    schema: str = SCHEMA_DESIGN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, "route_tile_size"),
        )
        fraction = float(self.hbm_ceiling_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise ValueError("hbm_ceiling_fraction must be in (0, 1]")
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        for name in (
            "maximum_shell_routes",
            "maximum_admissible_faces",
            "maximum_rounding_outcomes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        flags = (
            "require_complete_route_scan",
            "allow_cpu_fallback",
            "allow_full_key_index_d2h",
            "allow_route_scale_row_retention",
            "allow_nonmonotone_q_early_stop",
            "allow_rank_padding",
        )
        if any(type(getattr(self, name)) is not bool for name in flags):
            raise TypeError("issuer design flags must be exact booleans")
        if not self.require_complete_route_scan:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "production CUDA issuer must require a complete route scan",
            )
        if self.allow_cpu_fallback:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.CPU_FALLBACK_FORBIDDEN,
                "production decision-level issuer has no CPU fallback",
            )
        if self.allow_full_key_index_d2h or self.allow_route_scale_row_retention:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCALE_D2H_FORBIDDEN,
                "production CUDA issuer may not materialize route-scale key/index rows",
            )
        if self.allow_nonmonotone_q_early_stop:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.NONMONOTONE_Q_EARLY_STOP_FORBIDDEN,
                "hard F/Q replay must visit every endpoint because Q is nonmonotone",
            )
        if self.allow_rank_padding:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.RANK_PADDING_FORBIDDEN,
                "adaptive-rank replay may not pad or change the source population",
            )
        if self.schema != SCHEMA_DESIGN:
            raise ValueError("unsupported decision-level CUDA issuer design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-decision-cuda-design-")


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaIssuerHBMForecast:
    """Pre-outcome HBM forecast for the source issuer."""

    route_count: int
    rank: int
    shell_count: int
    endpoint_count: int
    route_tile_size: int
    hbm_ceiling_fraction: float
    device_total_memory_bytes: int
    current_reserved_bytes: int
    hbm_ceiling_bytes: int
    route_tensor_state_bytes: int
    key_interval_tensor_bytes: int
    label_and_mask_tensor_bytes: int
    source_state_already_reserved: bool
    source_state_incremental_bytes: int
    tile_workspace_bytes: int
    shell_workspace_bytes: int
    replay_workspace_bytes: int
    forecast_peak_reserved_bytes: int
    admitted: bool
    schema: str = SCHEMA_FORECAST
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in (
            "route_count",
            "route_tile_size",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "rank",
            "shell_count",
            "endpoint_count",
            "current_reserved_bytes",
            "route_tensor_state_bytes",
            "key_interval_tensor_bytes",
            "label_and_mask_tensor_bytes",
            "source_state_incremental_bytes",
            "tile_workspace_bytes",
            "shell_workspace_bytes",
            "replay_workspace_bytes",
            "forecast_peak_reserved_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _finite_nonnegative(self.hbm_ceiling_fraction, "hbm_ceiling_fraction"),
        )
        if type(self.admitted) is not bool:
            raise TypeError("forecast admitted flag must be an exact bool")
        if type(self.source_state_already_reserved) is not bool:
            raise TypeError("source-state residency flag must be an exact bool")
        expected_source_incremental = (
            0
            if self.source_state_already_reserved
            else (
                self.route_tensor_state_bytes
                + self.key_interval_tensor_bytes
                + self.label_and_mask_tensor_bytes
            )
        )
        if self.source_state_incremental_bytes != expected_source_incremental:
            raise ValueError(
                "source-state incremental bytes disagree with residency scope"
            )
        expected_peak = (
            self.current_reserved_bytes
            + self.source_state_incremental_bytes
            + self.tile_workspace_bytes
            + self.shell_workspace_bytes
            + self.replay_workspace_bytes
        )
        if self.forecast_peak_reserved_bytes != expected_peak:
            raise ValueError("forecast peak does not equal the registered components")
        if _admission_memory is None:
            admitted = self.forecast_peak_reserved_bytes <= self.hbm_ceiling_bytes
        else:
            available, allocated = _admission_memory
            incremental = expected_peak - self.current_reserved_bytes
            admitted = (
                incremental <= available
                and allocated + incremental <= self.hbm_ceiling_bytes
            )
        if self.admitted != admitted:
            raise ValueError("forecast admission flag disagrees with peak/ceiling")
        if self.schema != SCHEMA_FORECAST:
            raise ValueError("unsupported decision-level CUDA HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-decision-cuda-hbm-")


def estimate_v1_decision_level_cuda_issuer_hbm(
    *,
    route_count: int,
    rank: int,
    shell_count: int,
    endpoint_count: int,
    route_tile_size: int = DEFAULT_ROUTE_TILE_SIZE,
    device_total_memory_bytes: int,
    current_reserved_bytes: int = 0,
    source_state_already_reserved: bool = False,
    hbm_ceiling_fraction: float = DEFAULT_H200_HBM_CEILING_FRACTION,
    _admission_memory: tuple[int, int] | None = None,
) -> V1DecisionLevelCudaIssuerHBMForecast:
    """Conservative pre-outcome HBM forecast for the CUDA issuer.

    The forecast reports resident route tensors/state, the two key-interval
    tensors, label/shell masks, per-tile comparison workspace, bounded shell
    operator workspace, and complete multi-endpoint hard-F/Q replay workspace.
    Before source construction, resident state is incremental.  During a live
    KKT theorem lease, ``current_reserved_bytes`` already contains it, so
    ``source_state_already_reserved=True`` prevents double counting while the
    complete resident byte scope remains explicit in the receipt.
    """

    n = _positive_int(route_count, "route_count")
    r = _nonnegative_int(rank, "rank")
    b = _nonnegative_int(shell_count, "shell_count")
    endpoints = _nonnegative_int(endpoint_count, "endpoint_count")
    tile = min(_positive_int(route_tile_size, "route_tile_size"), n)
    total = _positive_int(device_total_memory_bytes, "device_total_memory_bytes")
    reserved = _nonnegative_int(current_reserved_bytes, "current_reserved_bytes")
    if type(source_state_already_reserved) is not bool:
        raise TypeError("source_state_already_reserved must be an exact bool")
    fraction = float(hbm_ceiling_fraction)
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError("hbm_ceiling_fraction must be in (0, 1]")
    f64 = 8
    i64 = 8
    u8 = 1
    # Full route resident state: resource, corrected work, residual diagonal,
    # eligible, low-rank H, and branch/custody scratch vectors.
    route_tensor_state = n * (3 * f64 + u8) + r * n * f64 + n * 5 * f64
    # Two finite key-interval tensors, each shape (route_count, 3), FP64.
    key_interval = 2 * n * 3 * f64
    # Two device-shell label vectors plus one active-shell mask.
    label_and_mask = 3 * n * u8
    # Tile-vs-shell comparison matrices and route-local interval temporaries.
    tile_workspace = tile * max(1, b) * (2 * i64 + 2 * f64) + tile * 12 * f64
    # Restricted H_B intervals, residual direction, shell keys, and bounded
    # dense shell operator used only when B is small enough to enumerate.
    shell_workspace = max(1, b) * (2 * r * f64 + 8 * f64 + i64) + b * b * 3 * f64
    # Complete endpoint replay; this is intentionally endpoint-major rather
    # than a binary-search frontier because Q is nonmonotone.
    replay_workspace = max(1, endpoints) * max(1, b) * 6 * f64
    source_state_incremental = (
        0
        if source_state_already_reserved
        else route_tensor_state + key_interval + label_and_mask
    )
    ceiling = int(math.floor(fraction * total))
    peak = (
        reserved
        + source_state_incremental
        + tile_workspace
        + shell_workspace
        + replay_workspace
    )
    return V1DecisionLevelCudaIssuerHBMForecast(
        route_count=n,
        rank=r,
        shell_count=b,
        endpoint_count=endpoints,
        route_tile_size=tile,
        hbm_ceiling_fraction=fraction,
        device_total_memory_bytes=total,
        current_reserved_bytes=reserved,
        hbm_ceiling_bytes=ceiling,
        route_tensor_state_bytes=route_tensor_state,
        key_interval_tensor_bytes=key_interval,
        label_and_mask_tensor_bytes=label_and_mask,
        source_state_already_reserved=source_state_already_reserved,
        source_state_incremental_bytes=source_state_incremental,
        tile_workspace_bytes=tile_workspace,
        shell_workspace_bytes=shell_workspace,
        replay_workspace_bytes=replay_workspace,
        forecast_peak_reserved_bytes=peak,
        admitted=(
            peak <= ceiling
            if _admission_memory is None
            else peak - reserved <= _admission_memory[0]
            and _admission_memory[1] + peak - reserved <= ceiling
        ),
        _admission_memory=_admission_memory,
    )


def _forecast_hbm_from_device(
    *,
    torch: Any,
    device: Any,
    route_count: int,
    rank: int,
    shell_count: int,
    endpoint_count: int,
    design: V1DecisionLevelCudaIssuerDesign,
) -> V1DecisionLevelCudaIssuerHBMForecast:
    properties = torch.cuda.get_device_properties(device)
    return estimate_v1_decision_level_cuda_issuer_hbm(
        route_count=route_count,
        rank=rank,
        shell_count=shell_count,
        endpoint_count=endpoint_count,
        route_tile_size=design.route_tile_size,
        device_total_memory_bytes=int(properties.total_memory),
        current_reserved_bytes=int(torch.cuda.memory_reserved(device)),
        source_state_already_reserved=True,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        _admission_memory=(
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        ),
    )


def assert_v1_decision_level_cuda_hbm_admitted(
    forecast: V1DecisionLevelCudaIssuerHBMForecast,
) -> None:
    if type(forecast) is not V1DecisionLevelCudaIssuerHBMForecast:
        raise TypeError("forecast must be exact V1DecisionLevelCudaIssuerHBMForecast")
    if not forecast.admitted:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HBM_FORECAST_EXCEEDED,
            "decision-level CUDA issuer forecast exceeds the registered HBM ceiling",
            forecast_hash=forecast.forecast_hash,
            forecast_peak_reserved_bytes=forecast.forecast_peak_reserved_bytes,
            hbm_ceiling_bytes=forecast.hbm_ceiling_bytes,
        )


def classify_v1_decision_level_lex_interval_pair(
    *,
    left_lower: Sequence[float],
    left_upper: Sequence[float],
    right_lower: Sequence[float],
    right_upper: Sequence[float],
) -> tuple[int, float]:
    """Pure-Python parity helper for the theorem's lex interval relation."""

    if not (
        len(left_lower)
        == len(left_upper)
        == len(right_lower)
        == len(right_upper)
        == 3
    ):
        raise ValueError("lex interval keys must have exactly three components")
    for component in range(3):
        left_lo = float(left_lower[component])
        left_hi = float(left_upper[component])
        right_lo = float(right_lower[component])
        right_hi = float(right_upper[component])
        if not all(math.isfinite(v) for v in (left_lo, left_hi, right_lo, right_hi)):
            raise ValueError("lex interval keys must be finite")
        if left_lo > left_hi or right_lo > right_hi:
            raise ValueError("lex interval component is reversed")
        if left_hi < right_lo:
            return -1, math.nextafter(right_lo - left_hi, -math.inf)
        if right_hi < left_lo:
            return 1, math.nextafter(left_lo - right_hi, -math.inf)
        if left_lo == left_hi == right_lo == right_hi:
            continue
        return 0, 0.0
    return 2, 0.0


class V1DecisionLevelCudaShellHashCustody:
    """Opaque compact hash custody for bounded shell outcomes.

    The CUDA scan can count and classify route membership without copying a
    full route-index axis to the host.  Reproducing ``RemovedRouteIndexSet``
    hashes for large fixed prefixes still requires a streaming encoded-index
    digest that is owned outside this first joint.  This exact object is the
    temporary handoff contract: without it the issuer refuses rather than
    fabricating prefix, suffix, endpoint, or all-face outcome hashes.

    ``outcome_ladder_axis_hash`` is retained as the v1 wire name for the
    independent complete shell-subset mapping commitment.  The CUDA theorem
    issuer recomputes the exact face/order outcome axis itself; custody cannot
    dictate that theorem value.
    """

    __slots__ = (
        "endpoint_hash_by_shell_offsets",
        "fixed_prefix_count",
        "fixed_prefix_route_index_hash",
        "fixed_suffix_count",
        "fixed_suffix_route_index_hash",
        "outcome_ladder_axis_hash",
        "shell_route_index_hash",
        "source_evidence_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        source_evidence_hash: str,
        shell_route_index_hash: str,
        fixed_prefix_route_index_hash: str,
        fixed_prefix_count: int,
        fixed_suffix_route_index_hash: str,
        fixed_suffix_count: int,
        endpoint_hash_by_shell_offsets: Mapping[tuple[int, ...], str],
        outcome_ladder_axis_hash: str,
    ) -> None:
        self.source_evidence_hash = _nonempty(source_evidence_hash, "source_evidence_hash")
        self.shell_route_index_hash = _nonempty(shell_route_index_hash, "shell_route_index_hash")
        self.fixed_prefix_route_index_hash = _nonempty(
            fixed_prefix_route_index_hash,
            "fixed_prefix_route_index_hash",
        )
        self.fixed_prefix_count = _nonnegative_int(
            fixed_prefix_count,
            "fixed_prefix_count",
        )
        self.fixed_suffix_route_index_hash = _nonempty(
            fixed_suffix_route_index_hash,
            "fixed_suffix_route_index_hash",
        )
        self.fixed_suffix_count = _nonnegative_int(
            fixed_suffix_count,
            "fixed_suffix_count",
        )
        normalized: dict[tuple[int, ...], str] = {}
        for offsets, digest in endpoint_hash_by_shell_offsets.items():
            key = tuple(sorted(_nonnegative_int(int(value), "shell offset") for value in offsets))
            if len(set(key)) != len(key):
                raise ValueError("endpoint shell offsets must be unique")
            normalized[key] = _nonempty(digest, "endpoint hash")
        if not normalized:
            raise ValueError("hash custody requires at least one endpoint hash")
        self.endpoint_hash_by_shell_offsets = normalized
        self.outcome_ladder_axis_hash = _nonempty(
            outcome_ladder_axis_hash,
            "outcome_ladder_axis_hash",
        )

    @property
    def custody_hash(self) -> str:
        return content_hash(
            {
                "schema": SCHEMA_HASH_CUSTODY,
                "source_evidence_hash": self.source_evidence_hash,
                "shell_route_index_hash": self.shell_route_index_hash,
                "fixed_prefix_route_index_hash": self.fixed_prefix_route_index_hash,
                "fixed_prefix_count": self.fixed_prefix_count,
                "fixed_suffix_route_index_hash": self.fixed_suffix_route_index_hash,
                "fixed_suffix_count": self.fixed_suffix_count,
                "endpoint_hash_by_shell_offsets": tuple(
                    sorted(self.endpoint_hash_by_shell_offsets.items())
                ),
                "outcome_ladder_axis_hash": self.outcome_ladder_axis_hash,
            },
            prefix="partition-free-v1-decision-cuda-hash-custody-",
        )

    @property
    def complete_subset_mapping_hash(self) -> str:
        """Return the independent digest owner's complete subset commitment."""

        return self.outcome_ladder_axis_hash

    def endpoint_hash_for_shell_offsets(self, offsets: Sequence[int]) -> str:
        key = tuple(sorted(int(value) for value in offsets))
        try:
            return self.endpoint_hash_by_shell_offsets[key]
        except KeyError:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "bounded shell hash custody lacks one admissible endpoint outcome",
                shell_offsets=key,
                custody_hash=self.custody_hash,
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("decision-level CUDA hash custody is nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("decision-level CUDA hash custody is nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("decision-level CUDA hash custody is nonserializable")


def _nonempty(value: str, name: str) -> str:
    raw = str(value)
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    return raw


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaPrivateTheoremStateHookRequirement:
    """Minimal future lease seam when branch theorem state is still opaque.

    This object is intentionally array-free.  It is emitted only in refusal
    payloads to name the narrow hook the source issuer needs; it is not a
    substitute for custody and it grants no source-issuance authority.
    """

    reason: str
    required_hook_signature: str
    required_return_contract: tuple[str, ...]
    whole_private_state_exposure_forbidden: bool = True
    route_scale_cpu_materialization_forbidden: bool = True
    schema: str = SCHEMA_PRIVATE_THEOREM_HOOK_REQUIREMENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))
        object.__setattr__(
            self,
            "required_hook_signature",
            _nonempty(self.required_hook_signature, "required_hook_signature"),
        )
        contract = tuple(
            _nonempty(str(item), "required_return_contract item")
            for item in self.required_return_contract
        )
        if not contract:
            raise ValueError("private theorem hook requirement needs return fields")
        object.__setattr__(self, "required_return_contract", contract)
        if (
            self.whole_private_state_exposure_forbidden is not True
            or self.route_scale_cpu_materialization_forbidden is not True
        ):
            raise PermissionError("private theorem hook requirement weakened custody")
        if self.schema != SCHEMA_PRIVATE_THEOREM_HOOK_REQUIREMENT:
            raise ValueError("unsupported private theorem hook requirement schema")

    @property
    def requirement_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-decision-cuda-private-theorem-hook-",
        )


def v1_decision_level_cuda_private_theorem_state_hook_requirement(
    *,
    reason: str = (
        "the CUDA issuer needs a narrow source-theorem lease rather than whole "
        "private branch-state access"
    ),
) -> V1DecisionLevelCudaPrivateTheoremStateHookRequirement:
    """Return the exact minimal relaxed-KKT hook needed by this issuer joint."""

    return V1DecisionLevelCudaPrivateTheoremStateHookRequirement(
        reason=reason,
        required_hook_signature=(
            "borrow_decision_level_theorem_state_for_cuda_issuer("
            "*, request: V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest, "
            "geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet, "
            "literal_j_projection_receipt: "
            "V1LiteralJFixedDirectionProjectionReceipt"
            ") -> V1RelaxedKKTDecisionLevelTheoremStateLease"
        ),
        required_return_contract=(
            "theorem_state_axis_hash",
            "shell_route_index_hash",
            "canonical_key_lower_cuda_f64_route_by_3",
            "canonical_key_upper_cuda_f64_route_by_3",
            "low_rank_factor_cuda_f64_rank_major",
            "resource_cuda_f64",
            "corrected_work_cuda_f64",
            "residual_diagonal_cuda_f64",
            "eligible_mask_cuda_bool",
            "bounded_shell_route_indices",
        ),
    )


def build_v1_decision_level_cuda_theorem_state_borrow_request(
    *,
    binding: V1DecisionLevelCudaPrivatePrimalBinding,
    maximum_shell_routes: int = DEFAULT_MAXIMUM_SHELL_ROUTES,
) -> V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
    """Translate the array-free issuer lineage into the KKT borrow request.

    The request deliberately binds, but does not ask KKT custody to attest,
    ``source_evidence_hash``.  Prefix/suffix and alternate-endpoint hashes stay
    with the independent streaming-digest owner.  This separation prevents the
    numerical solver from silently becoming the owner of mask identity.
    """

    if type(binding) is not V1DecisionLevelCudaPrivatePrimalBinding:
        raise TypeError(
            "binding must be exact V1DecisionLevelCudaPrivatePrimalBinding"
        )
    branch = V1RelaxedKKTBranch(binding.branch)
    return V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest(
        problem_hash=binding.problem_hash,
        policy_hash=binding.policy_hash,
        result_hash=binding.base_result_hash,
        branch=branch,
        branch_result_hash=binding.branch_result_hash,
        branch_receipt_hash=binding.branch_receipt_hash,
        device_shell_execution_receipt_hash=(
            binding.device_shell_execution_receipt_hash
        ),
        device_shell_execution_design_hash=(
            binding.device_shell_execution_design_hash
        ),
        source_rounding_hash=binding.source_rounding_hash,
        endpoint_action_axis_hash=binding.endpoint_action_axis_hash,
        geometry_uncertainty_hash=binding.geometry_uncertainty_hash,
        private_primal_descriptor_axis_hash=(
            binding.private_primal_descriptor_axis_hash
        ),
        private_primal_branch_descriptor_hash=(
            binding.private_primal_branch_descriptor_hash
        ),
        literal_j_projection_receipt_hash=(
            binding.literal_j_projection_receipt_hash
        ),
        literal_j_projection_direction_axis_hash=(
            binding.literal_j_projection_direction_axis_hash
        ),
        literal_j_projection_output_axis_hash=(
            binding.literal_j_projection_output_axis_hash
        ),
        literal_j_private_primal_descriptor_axis_hash=(
            binding.literal_j_private_primal_descriptor_axis_hash
        ),
        source_evidence_hash=binding.source_evidence_hash,
        maximum_shell_routes=_positive_int(
            maximum_shell_routes,
            "maximum_shell_routes",
        ),
    )


def build_v1_decision_level_cuda_theorem_state_borrow_request_from_sources(
    *,
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    literal_j_fixed_projection: V1LiteralJFixedDirectionProjectionReceipt,
    source_evidence_hash: str,
    maximum_shell_routes: int = DEFAULT_MAXIMUM_SHELL_ROUTES,
) -> V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest:
    """Build the pre-hash-custody KKT request from exact source owners.

    This is the noncircular production entry point: KKT must first reveal the
    bounded theorem shell before the independent digest owner can attest that
    shell's prefix/suffix and endpoint masks.  The KKT borrow itself performs
    the authoritative lineage validation; this helper merely names the same
    array-free coordinates without granting any downstream authority.
    """

    for value, expected, name in (
        (execution_bundle, V1RelaxedKKTDeviceShellExecutionBundle, "execution_bundle"),
        (problem, V1RelaxedKKTProblem, "problem"),
        (policy, V1RelaxedKKTPolicy, "policy"),
        (branch_result, V1RelaxedKKTBranchResult, "branch_result"),
        (
            branch_identity,
            V1RelaxedKKTDeviceShellBranchIdentityReceipt,
            "branch_identity",
        ),
        (
            geometry_uncertainty,
            V1RelaxedKKTGeometryUncertaintySet,
            "geometry_uncertainty",
        ),
        (
            literal_j_fixed_projection,
            V1LiteralJFixedDirectionProjectionReceipt,
            "literal_j_fixed_projection",
        ),
    ):
        if type(value) is not expected:
            raise TypeError(f"{name} must be exact {expected.__name__}")
    literal_j_fixed_projection.assert_integrity()
    if (
        branch_result != _branch_result_for_branch(
            execution_bundle,
            branch_result.branch,
        )
        or branch_identity != _branch_identity_for_result(
            execution_bundle,
            branch_result.branch,
        )
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "pre-hash theorem request names a non-owned branch result or ladder",
        )
    descriptor_axis = execution_bundle.private_primal_direction_descriptor_axis
    branch_descriptor = next(
        (
            descriptor
            for descriptor in descriptor_axis.descriptors
            if descriptor.branch == branch_result.branch.value
        ),
        None,
    )
    if branch_descriptor is None:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_PRIMAL_CUSTODY_REQUIRED,
            "private-primal descriptor axis lacks the selected branch",
        )
    return V1RelaxedKKTDecisionLevelTheoremStateBorrowRequest(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        result_hash=execution_bundle.base_execution.result.result_hash,
        branch=branch_result.branch,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        device_shell_execution_receipt_hash=execution_bundle.receipt.receipt_hash,
        device_shell_execution_design_hash=execution_bundle.execution_design.design_hash,
        source_rounding_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        geometry_uncertainty_hash=geometry_uncertainty.uncertainty_hash,
        private_primal_descriptor_axis_hash=descriptor_axis.descriptor_axis_hash,
        private_primal_branch_descriptor_hash=branch_descriptor.descriptor_hash,
        literal_j_projection_receipt_hash=literal_j_fixed_projection.receipt_hash,
        literal_j_projection_direction_axis_hash=(
            literal_j_fixed_projection.direction_axis_hash
        ),
        literal_j_projection_output_axis_hash=(
            literal_j_fixed_projection.output_axis_hash
        ),
        literal_j_private_primal_descriptor_axis_hash=(
            literal_j_fixed_projection.private_primal_direction_descriptor_axis_hash
        ),
        source_evidence_hash=_nonempty(
            source_evidence_hash,
            "source_evidence_hash",
        ),
        maximum_shell_routes=_positive_int(
            maximum_shell_routes,
            "maximum_shell_routes",
        ),
    )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaPrivatePrimalBinding:
    """Array-free proof that issuer inputs descend from post-D_W custody."""

    problem_hash: str
    policy_hash: str
    base_result_hash: str
    branch: str
    branch_result_hash: str
    branch_receipt_hash: str
    device_shell_execution_receipt_hash: str
    device_shell_execution_design_hash: str
    cuda_execution_design_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    source_rounding_hash: str
    endpoint_action_axis_hash: str
    frozen_ladder_hash: str
    hash_custody_hash: str
    source_evidence_hash: str
    shell_route_index_hash: str
    geometry_uncertainty_hash: str
    geometry_literal_j_direction_axis_hash: str
    private_primal_descriptor_axis_hash: str
    private_primal_direction_ids: tuple[str, str]
    private_primal_branch_descriptor_hash: str
    literal_j_projection_receipt_hash: str
    literal_j_projection_direction_axis_hash: str
    literal_j_projection_output_axis_hash: str
    literal_j_private_primal_descriptor_axis_hash: str
    private_branch_state_custody_live: bool
    private_branch_state_retained_tensor_bytes: int
    private_primal_lease_probe_descriptor_axis_hash: str
    whole_private_states_exposed: bool = False
    route_scale_primal_d2h_used: bool = False
    schema: str = SCHEMA_PRIVATE_PRIMAL_BINDING

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "base_result_hash",
            "branch",
            "branch_result_hash",
            "branch_receipt_hash",
            "device_shell_execution_receipt_hash",
            "device_shell_execution_design_hash",
            "cuda_execution_design_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "source_rounding_hash",
            "endpoint_action_axis_hash",
            "frozen_ladder_hash",
            "hash_custody_hash",
            "source_evidence_hash",
            "shell_route_index_hash",
            "geometry_uncertainty_hash",
            "geometry_literal_j_direction_axis_hash",
            "private_primal_descriptor_axis_hash",
            "private_primal_branch_descriptor_hash",
            "literal_j_projection_receipt_hash",
            "literal_j_projection_direction_axis_hash",
            "literal_j_projection_output_axis_hash",
            "literal_j_private_primal_descriptor_axis_hash",
            "private_primal_lease_probe_descriptor_axis_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, "route_count"),
        )
        object.__setattr__(
            self,
            "private_branch_state_retained_tensor_bytes",
            _nonnegative_int(
                self.private_branch_state_retained_tensor_bytes,
                "private_branch_state_retained_tensor_bytes",
            ),
        )
        direction_ids = tuple(
            _nonempty(str(item), "private_primal_direction_id")
            for item in self.private_primal_direction_ids
        )
        if len(direction_ids) != 2 or len(set(direction_ids)) != 2:
            raise ValueError("private-primal binding requires the canonical two directions")
        object.__setattr__(self, "private_primal_direction_ids", direction_ids)
        if self.branch not in {"spectral", "diagonal_l1"}:
            raise ValueError("private-primal binding branch is not canonical")
        if self.private_branch_state_custody_live is not True:
            raise PermissionError("private theorem-state custody is not live")
        if self.whole_private_states_exposed or self.route_scale_primal_d2h_used:
            raise PermissionError("private-primal binding crossed custody boundaries")
        if (
            self.literal_j_private_primal_descriptor_axis_hash
            != self.private_primal_descriptor_axis_hash
            or self.private_primal_lease_probe_descriptor_axis_hash
            != self.private_primal_descriptor_axis_hash
        ):
            raise ValueError("private-primal descriptor hashes do not close")
        if self.schema != SCHEMA_PRIVATE_PRIMAL_BINDING:
            raise ValueError("unsupported private-primal binding schema")

    @property
    def binding_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-decision-cuda-private-primal-binding-",
        )


def _branch_identity_for_result(
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTDeviceShellBranchIdentityReceipt:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return execution_bundle.receipt.spectral
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return execution_bundle.receipt.diagonal_l1
    raise TypeError("unsupported relaxed KKT branch")


def _branch_result_for_branch(
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTBranchResult:
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return execution_bundle.base_execution.result.spectral
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return execution_bundle.base_execution.result.diagonal_l1
    raise TypeError("unsupported relaxed KKT branch")


def bind_v1_decision_level_cuda_private_primal_custody(
    *,
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    literal_j_fixed_projection: V1LiteralJFixedDirectionProjectionReceipt,
    hash_custody: V1DecisionLevelCudaShellHashCustody,
) -> V1DecisionLevelCudaPrivatePrimalBinding:
    """Bind post-D_W private primals, shell lineage, and literal-J projection.

    The function performs only structural checks plus a liveness probe of the
    private-primal handle.  It never borrows whole branch theorem state.  The
    resulting array-free binding is the caller-owned half of the narrow KKT
    theorem-state borrow; streaming mask-hash custody remains a separate half.
    """

    for value, expected, name in (
        (execution_bundle, V1RelaxedKKTDeviceShellExecutionBundle, "execution_bundle"),
        (problem, V1RelaxedKKTProblem, "problem"),
        (policy, V1RelaxedKKTPolicy, "policy"),
        (branch_result, V1RelaxedKKTBranchResult, "branch_result"),
        (
            branch_identity,
            V1RelaxedKKTDeviceShellBranchIdentityReceipt,
            "branch_identity",
        ),
        (geometry_uncertainty, V1RelaxedKKTGeometryUncertaintySet, "geometry_uncertainty"),
        (
            literal_j_fixed_projection,
            V1LiteralJFixedDirectionProjectionReceipt,
            "literal_j_fixed_projection",
        ),
        (hash_custody, V1DecisionLevelCudaShellHashCustody, "hash_custody"),
    ):
        if type(value) is not expected:
            raise TypeError(f"{name} must be exact {expected.__name__}")

    literal_j_fixed_projection.assert_integrity()
    if not execution_bundle.has_private_branch_state_custody:
        requirement = v1_decision_level_cuda_private_theorem_state_hook_requirement(
            reason="device-shell bundle has no live private theorem-state custody"
        )
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_THEOREM_STATE_HOOK_REQUIRED,
            "decision-level CUDA issuer needs live source-theorem custody",
            requirement_hash=requirement.requirement_hash,
            required_hook_signature=requirement.required_hook_signature,
            required_return_contract=requirement.required_return_contract,
        )
    if not execution_bundle.has_private_primal_direction_custody:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_PRIMAL_CUSTODY_REQUIRED,
            "decision-level CUDA issuer requires post-D_W private-primal custody",
        )
    if (
        execution_bundle.base_execution.result.problem_hash != problem.problem_hash
        or execution_bundle.base_execution.result.policy_hash != policy.policy_hash
        or execution_bundle.receipt.problem_hash != problem.problem_hash
        or execution_bundle.receipt.policy_hash != policy.policy_hash
        or execution_bundle.execution_design.cuda_execution_design.design_hash
        != execution_bundle.receipt.cuda_execution_design_hash
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "device-shell bundle does not bind the supplied problem/policy",
        )

    expected_branch_result = _branch_result_for_branch(
        execution_bundle,
        branch_result.branch,
    )
    expected_branch_identity = _branch_identity_for_result(
        execution_bundle,
        branch_result.branch,
    )
    if branch_result != expected_branch_result or branch_identity != expected_branch_identity:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "branch result or shell identity is not the bundle-owned branch",
        )

    descriptor_axis = execution_bundle.private_primal_direction_descriptor_axis
    branch_descriptor = next(
        (
            descriptor
            for descriptor in descriptor_axis.descriptors
            if descriptor.branch == branch_result.branch.value
        ),
        None,
    )
    if branch_descriptor is None:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_PRIMAL_CUSTODY_REQUIRED,
            "private-primal descriptor axis lacks the selected branch",
        )
    if (
        descriptor_axis.source_problem_hash != problem.problem_hash
        or descriptor_axis.source_policy_hash != policy.policy_hash
        or descriptor_axis.source_result_hash
        != execution_bundle.base_execution.result.result_hash
        or descriptor_axis.source_execution_receipt_hash
        != execution_bundle.receipt.receipt_hash
        or descriptor_axis.source_execution_design_hash
        != execution_bundle.execution_design.design_hash
        or descriptor_axis.route_axis_hash != problem.route_axis_hash
        or descriptor_axis.route_count != problem.route_count
        or branch_descriptor.source_branch_result_hash != branch_result.result_hash
        or branch_descriptor.source_branch_receipt_hash
        != branch_result.receipt.receipt_hash
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "private-primal descriptor axis does not close against this branch",
        )
    if (
        geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
        or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
        or geometry_uncertainty.literal_j_fixed_primal_direction_axis_hash
        != literal_j_fixed_projection.direction_axis_hash
        or geometry_uncertainty.literal_j_fixed_primal_direction_count
        != literal_j_fixed_projection.direction_count
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "geometry uncertainty does not bind the literal-J projection axis",
        )
    if (
        literal_j_fixed_projection.private_primal_direction_lineage_available
        is not True
        or literal_j_fixed_projection.private_primal_direction_descriptor_axis_hash
        != descriptor_axis.descriptor_axis_hash
        or literal_j_fixed_projection.route_axis_hash != problem.route_axis_hash
        or literal_j_fixed_projection.route_count != problem.route_count
        or literal_j_fixed_projection.chart_identity_hash != problem.source_geometry_hash
        or literal_j_fixed_projection.current_mask_hash
        != branch_identity.frozen_ladder.parent_mask_hash
        or literal_j_fixed_projection.direction_ids != descriptor_axis.direction_ids
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "literal-J fixed projection is not the exact post-D_W primal projection",
        )
    with execution_bundle.borrow_private_primals_for_literal_j_projection() as lease:
        lease_descriptor_axis_hash = lease.descriptor_axis_hash

    return V1DecisionLevelCudaPrivatePrimalBinding(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        base_result_hash=execution_bundle.base_execution.result.result_hash,
        branch=branch_result.branch.value,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        device_shell_execution_receipt_hash=execution_bundle.receipt.receipt_hash,
        device_shell_execution_design_hash=execution_bundle.execution_design.design_hash,
        cuda_execution_design_hash=(
            execution_bundle.execution_design.cuda_execution_design.design_hash
        ),
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        parent_mask_hash=branch_identity.frozen_ladder.parent_mask_hash,
        source_rounding_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        frozen_ladder_hash=branch_identity.frozen_ladder.receipt_hash,
        hash_custody_hash=hash_custody.custody_hash,
        source_evidence_hash=hash_custody.source_evidence_hash,
        shell_route_index_hash=hash_custody.shell_route_index_hash,
        geometry_uncertainty_hash=geometry_uncertainty.uncertainty_hash,
        geometry_literal_j_direction_axis_hash=(
            geometry_uncertainty.literal_j_fixed_primal_direction_axis_hash
        ),
        private_primal_descriptor_axis_hash=descriptor_axis.descriptor_axis_hash,
        private_primal_direction_ids=descriptor_axis.direction_ids,
        private_primal_branch_descriptor_hash=branch_descriptor.descriptor_hash,
        literal_j_projection_receipt_hash=literal_j_fixed_projection.receipt_hash,
        literal_j_projection_direction_axis_hash=(
            literal_j_fixed_projection.direction_axis_hash
        ),
        literal_j_projection_output_axis_hash=literal_j_fixed_projection.output_axis_hash,
        literal_j_private_primal_descriptor_axis_hash=(
            literal_j_fixed_projection.private_primal_direction_descriptor_axis_hash
        ),
        private_branch_state_custody_live=(
            execution_bundle.has_private_branch_state_custody
        ),
        private_branch_state_retained_tensor_bytes=(
            execution_bundle.private_branch_state_retained_tensor_bytes
        ),
        private_primal_lease_probe_descriptor_axis_hash=lease_descriptor_axis_hash,
    )


class V1DecisionLevelCudaFullStateLease(AbstractContextManager["V1DecisionLevelCudaFullStateLease"]):
    """Ephemeral CUDA/FP64 custody for route-scale interval/state tensors."""

    __slots__ = (
        "_closed",
        "canonical_key_lower",
        "canonical_key_upper",
        "corrected_work",
        "eligible_mask",
        "hash_custody",
        "low_rank_factor",
        "residual_diagonal",
        "resource",
        "shell_route_indices",
        "private_primal_binding_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        canonical_key_lower: Any,
        canonical_key_upper: Any,
        low_rank_factor: Any,
        resource: Any,
        corrected_work: Any,
        residual_diagonal: Any,
        eligible_mask: Any,
        shell_route_indices: Sequence[int],
        hash_custody: V1DecisionLevelCudaShellHashCustody | None,
        private_primal_binding_hash: str | None = None,
    ) -> None:
        torch = _require_torch()
        self.canonical_key_lower = _require_cuda_tensor(
            torch,
            canonical_key_lower,
            name="canonical_key_lower",
            dtype=torch.float64,
            ndim=2,
        )
        self.canonical_key_upper = _require_cuda_tensor(
            torch,
            canonical_key_upper,
            name="canonical_key_upper",
            dtype=torch.float64,
            ndim=2,
        )
        if (
            tuple(int(v) for v in self.canonical_key_lower.shape)
            != tuple(int(v) for v in self.canonical_key_upper.shape)
            or int(self.canonical_key_lower.shape[1]) != 3
        ):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
                "canonical key interval tensors must both have shape (route_count, 3)",
            )
        n = int(self.canonical_key_lower.shape[0])
        self.low_rank_factor = _require_cuda_tensor(
            torch,
            low_rank_factor,
            name="low_rank_factor",
            dtype=torch.float64,
            ndim=2,
        )
        if int(self.low_rank_factor.shape[1]) != n:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
                "low_rank_factor must have shape (rank, route_count)",
            )
        self.resource = _require_cuda_tensor(
            torch,
            resource,
            name="resource",
            dtype=torch.float64,
            ndim=1,
        )
        self.corrected_work = _require_cuda_tensor(
            torch,
            corrected_work,
            name="corrected_work",
            dtype=torch.float64,
            ndim=1,
        )
        self.residual_diagonal = _require_cuda_tensor(
            torch,
            residual_diagonal,
            name="residual_diagonal",
            dtype=torch.float64,
            ndim=1,
        )
        self.eligible_mask = _require_cuda_tensor(
            torch,
            eligible_mask,
            name="eligible_mask",
            dtype=torch.bool,
            ndim=1,
        )
        for tensor_name in ("resource", "corrected_work", "residual_diagonal", "eligible_mask"):
            if int(getattr(self, tensor_name).numel()) != n:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
                    f"{tensor_name} must have length route_count",
                )
        shell = tuple(int(value) for value in shell_route_indices)
        if (
            not shell
            or len(set(shell)) != len(shell)
            or min(shell) < 0
            or max(shell) >= n
        ):
            raise ValueError("shell_route_indices must be nonempty, unique, and in range")
        self.shell_route_indices = tuple(sorted(shell))
        self.hash_custody = hash_custody
        self.private_primal_binding_hash = (
            None
            if private_primal_binding_hash is None
            else _nonempty(
                private_primal_binding_hash,
                "private_primal_binding_hash",
            )
        )
        self._closed = False

    @property
    def route_count(self) -> int:
        self._require_live()
        return int(self.canonical_key_lower.shape[0])

    @property
    def rank(self) -> int:
        self._require_live()
        return int(self.low_rank_factor.shape[0])

    @property
    def shell_count(self) -> int:
        self._require_live()
        return len(self.shell_route_indices)

    @property
    def device(self) -> str:
        self._require_live()
        return str(self.canonical_key_lower.device)

    def _require_live(self) -> None:
        if self._closed:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.FULL_STATE_LEASE_REQUIRED,
                "decision-level CUDA full-state lease is closed",
            )

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "V1DecisionLevelCudaFullStateLease":
        self._require_live()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("decision-level CUDA full-state leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("decision-level CUDA full-state leases are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("decision-level CUDA full-state leases are nonserializable")


class V1DecisionLevelCudaKKTLeaseAdapter(
    AbstractContextManager["V1DecisionLevelCudaKKTLeaseAdapter"]
):
    """Nested zero-copy adapter from KKT custody to the CUDA source issuer.

    KKT owns numerical tensors and their mutation-checked theorem lease.  The
    independent hash-custody object owns mask identities.  This adapter joins
    those authorities only for the dynamic extent of one issuer call and
    closes both leases together.  It cannot serialize, clone, or outlive the
    source execution bundle.
    """

    __slots__ = (
        "_branch_identity",
        "_branch_result",
        "_closed",
        "_design",
        "_entered",
        "_execution_bundle",
        "_geometry_uncertainty",
        "_hash_custody",
        "_hash_custody_receipt",
        "_literal_j_fixed_projection",
        "_policy",
        "_problem",
        "_source_lease",
        "_source_evidence_hash",
        "_theorem_lease",
        "_private_primal_binding",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
        problem: V1RelaxedKKTProblem,
        policy: V1RelaxedKKTPolicy,
        branch_result: V1RelaxedKKTBranchResult,
        branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
        geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
        literal_j_fixed_projection: V1LiteralJFixedDirectionProjectionReceipt,
        hash_custody: V1DecisionLevelCudaShellHashCustody | None = None,
        source_evidence_hash: str | None = None,
        design: V1DecisionLevelCudaIssuerDesign = V1DecisionLevelCudaIssuerDesign(),
    ) -> None:
        if type(design) is not V1DecisionLevelCudaIssuerDesign:
            raise TypeError("design must be exact V1DecisionLevelCudaIssuerDesign")
        self._execution_bundle = execution_bundle
        self._problem = problem
        self._policy = policy
        self._branch_result = branch_result
        self._branch_identity = branch_identity
        self._geometry_uncertainty = geometry_uncertainty
        self._literal_j_fixed_projection = literal_j_fixed_projection
        if hash_custody is not None and type(hash_custody) is not (
            V1DecisionLevelCudaShellHashCustody
        ):
            raise TypeError(
                "hash_custody must be exact "
                "V1DecisionLevelCudaShellHashCustody or None"
            )
        if hash_custody is None:
            if source_evidence_hash is None:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                    "canonical KKT adaptation must either receive independent "
                    "hash custody or the exact source-evidence identity needed "
                    "to stream it from the live theorem lease",
                )
            normalized_source_evidence_hash = _nonempty(
                source_evidence_hash,
                "source_evidence_hash",
            )
        else:
            normalized_source_evidence_hash = hash_custody.source_evidence_hash
            if (
                source_evidence_hash is not None
                and _nonempty(source_evidence_hash, "source_evidence_hash")
                != normalized_source_evidence_hash
            ):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
                    "supplied source evidence differs from independent hash custody",
                )
        self._hash_custody = hash_custody
        self._hash_custody_receipt: object | None = None
        self._source_evidence_hash = normalized_source_evidence_hash
        self._design = design
        self._theorem_lease: V1RelaxedKKTDecisionLevelTheoremStateLease | None = None
        self._source_lease: V1DecisionLevelCudaFullStateLease | None = None
        self._private_primal_binding: V1DecisionLevelCudaPrivatePrimalBinding | None = None
        self._entered = False
        self._closed = False

    def _require_entered(self) -> None:
        if (
            not self._entered
            or self._closed
            or self._source_lease is None
            or self._theorem_lease is None
            or self._private_primal_binding is None
        ):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.FULL_STATE_LEASE_REQUIRED,
                "KKT-to-issuer adapter is not inside its live context",
            )
        self._source_lease._require_live()

    @property
    def source_lease(self) -> V1DecisionLevelCudaFullStateLease:
        self._require_entered()
        assert self._source_lease is not None
        return self._source_lease

    @property
    def private_primal_binding(self) -> V1DecisionLevelCudaPrivatePrimalBinding:
        self._require_entered()
        assert self._private_primal_binding is not None
        return self._private_primal_binding

    @property
    def theorem_state_axis_hash(self) -> str:
        self._require_entered()
        assert self._theorem_lease is not None
        return self._theorem_lease.theorem_state_axis_hash

    @property
    def hash_custody(self) -> V1DecisionLevelCudaShellHashCustody:
        self._require_entered()
        assert self._hash_custody is not None
        return self._hash_custody

    @property
    def hash_custody_receipt(self) -> object | None:
        """Return the durable streaming receipt when custody was built here.

        Legacy callers may inject independently built custody, in which case
        this value is ``None``.  Canonical production callers omit custody and
        retain the exact typed receipt issued from the live KKT theorem lease.
        """

        self._require_entered()
        return self._hash_custody_receipt

    def __enter__(self) -> "V1DecisionLevelCudaKKTLeaseAdapter":
        if self._entered or self._closed:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.FULL_STATE_LEASE_REQUIRED,
                "KKT-to-issuer adapter is single-use",
            )
        request = (
            build_v1_decision_level_cuda_theorem_state_borrow_request_from_sources(
                execution_bundle=self._execution_bundle,
                problem=self._problem,
                policy=self._policy,
                branch_result=self._branch_result,
                branch_identity=self._branch_identity,
                geometry_uncertainty=self._geometry_uncertainty,
                literal_j_fixed_projection=self._literal_j_fixed_projection,
                source_evidence_hash=self._source_evidence_hash,
                maximum_shell_routes=self._design.maximum_shell_routes,
            )
        )
        theorem_lease = (
            self._execution_bundle.borrow_decision_level_theorem_state_for_cuda_issuer(
                request=request,
                geometry_uncertainty=self._geometry_uncertainty,
                literal_j_projection_receipt=self._literal_j_fixed_projection,
            )
        )
        if type(theorem_lease) is not V1RelaxedKKTDecisionLevelTheoremStateLease:
            theorem_lease.close()
            raise TypeError("KKT theorem-state borrow returned an inexact lease")
        try:
            hash_custody = self._hash_custody
            if hash_custody is None:
                # Delayed import is intentional: the independent custody
                # owner consumes issuer contracts but the issuer must not own
                # its implementation.  At this point this module is fully
                # initialized and the import cannot weaken that ownership
                # boundary.
                from .modern_v1_decision_level_cuda_hash_custody import (
                    issue_v1_decision_level_cuda_shell_hash_custody,
                )

                hash_issuance = (
                    issue_v1_decision_level_cuda_shell_hash_custody(
                        theorem_lease=theorem_lease,
                        problem=self._problem,
                        policy=self._policy,
                        branch_result=self._branch_result,
                        branch_identity=self._branch_identity,
                        geometry_uncertainty=self._geometry_uncertainty,
                        source_evidence_hash=self._source_evidence_hash,
                        design=self._design,
                    )
                )
                hash_custody = hash_issuance.custody
                self._hash_custody = hash_custody
                self._hash_custody_receipt = hash_issuance.receipt
            if (
                theorem_lease.branch is not self._branch_result.branch
                or theorem_lease.route_axis_hash != self._problem.route_axis_hash
                or theorem_lease.route_count != self._problem.route_count
                or theorem_lease.rank != self._problem.rank
                or theorem_lease.shell_route_index_hash
                != hash_custody.shell_route_index_hash
                or theorem_lease.source_evidence_hash
                != hash_custody.source_evidence_hash
            ):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
                    "KKT theorem lease and independent hash custody do not close",
                )
            binding = bind_v1_decision_level_cuda_private_primal_custody(
                execution_bundle=self._execution_bundle,
                problem=self._problem,
                policy=self._policy,
                branch_result=self._branch_result,
                branch_identity=self._branch_identity,
                geometry_uncertainty=self._geometry_uncertainty,
                literal_j_fixed_projection=self._literal_j_fixed_projection,
                hash_custody=hash_custody,
            )
            if (
                build_v1_decision_level_cuda_theorem_state_borrow_request(
                    binding=binding,
                    maximum_shell_routes=self._design.maximum_shell_routes,
                ).request_hash
                != request.request_hash
            ):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
                    "post-hash private-primal binding changed the KKT request",
                )
            source_lease = V1DecisionLevelCudaFullStateLease(
                canonical_key_lower=theorem_lease.canonical_key_lower,
                canonical_key_upper=theorem_lease.canonical_key_upper,
                low_rank_factor=theorem_lease.low_rank_factor,
                resource=theorem_lease.resource,
                corrected_work=theorem_lease.corrected_work,
                residual_diagonal=theorem_lease.residual_diagonal,
                eligible_mask=theorem_lease.eligible_mask,
                shell_route_indices=theorem_lease.shell_route_indices,
                hash_custody=hash_custody,
                private_primal_binding_hash=binding.binding_hash,
            )
        except BaseException:
            theorem_lease.close()
            raise
        self._theorem_lease = theorem_lease
        self._source_lease = source_lease
        self._private_primal_binding = binding
        self._entered = True
        return self

    def close(self) -> None:
        if self._closed:
            return
        source_lease = self._source_lease
        theorem_lease = self._theorem_lease
        try:
            if source_lease is not None:
                source_lease.close()
        finally:
            try:
                if theorem_lease is not None:
                    theorem_lease.close()
            finally:
                self._source_lease = None
                self._theorem_lease = None
                self._private_primal_binding = None
                self._closed = True

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("KKT-to-issuer adapters are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("KKT-to-issuer adapters are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("KKT-to-issuer adapters are nonserializable")


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as error:  # pragma: no cover - environment dependent.
        _refuse(V1DecisionLevelCudaIssuerRefusalCode.CUDA_UNAVAILABLE, str(error))
    return torch


def _require_cuda_tensor(
    torch: Any,
    value: Any,
    *,
    name: str,
    dtype: Any,
    ndim: int,
) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch tensor")
    if value.device.type != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.CPU_FALLBACK_FORBIDDEN,
            f"{name} must be a CUDA tensor; production CPU fallback is forbidden",
        )
    if value.dtype != dtype or value.ndim != ndim or not value.is_contiguous():
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
            f"{name} must be contiguous {dtype} with ndim={ndim}",
        )
    if dtype != torch.bool and not bool(torch.isfinite(value).all().detach().cpu().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
            f"{name} contains non-finite values",
        )
    return value


@dataclass(frozen=True, slots=True)
class _CudaOutsideScan:
    receipt: V1RelaxedKKTOutsideShellIntervalScanReceipt
    scanned_route_count: int
    scanned_eligible_outside_route_count: int
    fixed_prefix_count: int
    fixed_suffix_count: int
    prefix_factor_offset: np.ndarray
    corrected_work_offset: float
    residual_offset: float
    prefix_block_counts: tuple[int, ...]


def _cuda_lex_relation_to_shell(
    torch: Any,
    *,
    lower_tile: Any,
    upper_tile: Any,
    shell_lower: Any,
    shell_upper: Any,
) -> tuple[Any, Any]:
    """Vectorized CUDA parity for three-component lex interval relation."""

    rows = int(lower_tile.shape[0])
    shell = int(shell_lower.shape[0])
    relation = torch.zeros((rows, shell), device=lower_tile.device, dtype=torch.int8)
    margin = torch.zeros((rows, shell), device=lower_tile.device, dtype=torch.float64)
    active = torch.ones((rows, shell), device=lower_tile.device, dtype=torch.bool)
    for component in range(3):
        left_lo = lower_tile[:, component].view(rows, 1)
        left_hi = upper_tile[:, component].view(rows, 1)
        right_lo = shell_lower[:, component].view(1, shell)
        right_hi = shell_upper[:, component].view(1, shell)
        left_precedes = active & (left_hi < right_lo)
        right_precedes = active & (right_hi < left_lo)
        exact_equal = active & (left_lo == left_hi) & (left_hi == right_lo) & (right_lo == right_hi)
        ambiguous = active & ~(left_precedes | right_precedes | exact_equal)
        relation = torch.where(left_precedes, torch.full_like(relation, -1), relation)
        relation = torch.where(right_precedes, torch.full_like(relation, 1), relation)
        # Outward lower margin: nextafter(diff, -inf).  A one-ULP separation
        # may therefore become zero, which correctly prevents authority at a
        # numerical boundary.
        left_margin = torch.nextafter(right_lo - left_hi, torch.full_like(right_lo, -math.inf))
        right_margin = torch.nextafter(left_lo - right_hi, torch.full_like(left_lo, -math.inf))
        margin = torch.where(left_precedes, left_margin, margin)
        margin = torch.where(right_precedes, right_margin, margin)
        relation = torch.where(ambiguous, torch.zeros_like(relation), relation)
        active = exact_equal
    relation = torch.where(active, torch.full_like(relation, 2), relation)
    return relation, margin


def _issue_outside_shell_scan_cuda(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    lease: V1DecisionLevelCudaFullStateLease,
    design: V1DecisionLevelCudaIssuerDesign,
) -> _CudaOutsideScan:
    torch = _require_torch()
    custody = lease.hash_custody
    if type(custody) is not V1DecisionLevelCudaShellHashCustody:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "outside-shell scan requires exact fixed-prefix/suffix hash custody",
        )
    if lease.rank != problem.rank:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.RANK_PADDING_FORBIDDEN,
            "CUDA issuer source rank differs from the registered KKT problem rank",
        )
    if lease.route_count != problem.route_count:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
            "CUDA issuer source route count differs from the registered problem",
        )
    shell = tuple(lease.shell_route_indices)
    shell_count = len(shell)
    if shell_count > design.maximum_shell_routes:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.SHELL_LIMIT_EXCEEDED,
            "active shell exceeds the registered production enumeration limit",
        )
    shell_hash = RemovedRouteIndexSet.from_indices(
        np.asarray(shell, dtype=np.int64),
        route_count=problem.route_count,
    ).index_hash
    if shell_hash != custody.shell_route_index_hash:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "shell hash custody does not match the bounded shell indices",
        )
    numerical_key_radius = max(
        geometry_uncertainty.solver_reduced_advantage_numerical_linf_radius,
        geometry_uncertainty.relaxed_gate_numerical_linf_radius,
        geometry_uncertainty.corrected_work_ratio_deterministic_linf_radius,
    )
    shell_tensor = torch.tensor(shell, device=lease.canonical_key_lower.device, dtype=torch.long)
    shell_lower = lease.canonical_key_lower.index_select(0, shell_tensor)
    shell_upper = lease.canonical_key_upper.index_select(0, shell_tensor)
    tiles = plan_v1_decision_level_cuda_route_tiles(
        route_count=problem.route_count,
        route_tile_size=design.route_tile_size,
    )
    route_to_block, block_capacities = _decision_level_block_axis(problem)
    prefix_block_counts = [0 for _ in block_capacities]
    prefix_factor = torch.zeros(
        (problem.rank,),
        device=lease.canonical_key_lower.device,
        dtype=torch.float64,
    )
    work_offset = torch.zeros((), device=lease.canonical_key_lower.device, dtype=torch.float64)
    residual_offset = torch.zeros((), device=lease.canonical_key_lower.device, dtype=torch.float64)
    sqrt_residual = torch.sqrt(lease.residual_diagonal)
    scanned = 0
    outside_count = 0
    prefix_count = 0
    suffix_count = 0
    min_reduced = math.inf
    min_key = math.inf
    for tile in tiles:
        lo = tile.start
        hi = tile.stop
        scanned += tile.length
        indices = torch.arange(lo, hi, device=lease.canonical_key_lower.device, dtype=torch.long)
        lower = lease.canonical_key_lower[lo:hi]
        upper = lease.canonical_key_upper[lo:hi]
        eligible = lease.eligible_mask[lo:hi]
        is_shell = (indices.view(-1, 1) == shell_tensor.view(1, -1)).any(dim=1)
        outside = eligible & ~is_shell
        if bool(outside.any().detach().cpu().item()):
            resource = lease.resource[lo:hi]
            ratio = torch.where(
                resource > 0.0,
                lease.corrected_work[lo:hi] / resource,
                torch.full_like(resource, math.inf),
            )
            ratio_ok = (~eligible) | ((ratio >= lower[:, 2]) & (ratio <= upper[:, 2]))
            if not bool(ratio_ok.all().detach().cpu().item()):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "canonical work/resource intervals do not enclose the source tensors",
                )
            reduced_lower = -upper[:, 1]
            reduced_upper = -lower[:, 1]
            reduced_touches = outside & (reduced_lower <= 0.0) & (reduced_upper >= 0.0)
            if bool(reduced_touches.any().detach().cpu().item()):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "outside reduced-advantage interval touches the active-face boundary",
                )
            reduced_margin = torch.minimum(torch.abs(reduced_lower), torch.abs(reduced_upper))
            relation, relation_margin = _cuda_lex_relation_to_shell(
                torch,
                lower_tile=lower,
                upper_tile=upper,
                shell_lower=shell_lower,
                shell_upper=shell_upper,
            )
            all_prefix = outside & (relation == -1).all(dim=1)
            all_suffix = outside & (relation == 1).all(dim=1)
            bad = outside & ~(all_prefix | all_suffix)
            if bool(bad.any().detach().cpu().item()):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "outside real-key interval touches or crosses the active shell",
                )
            route_margins = relation_margin.min(dim=1).values
            weak = outside & (route_margins <= 2.0 * numerical_key_radius)
            if bool(weak.any().detach().cpu().item()):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "outside real-key margin is not strict over numerical uncertainty",
                )
            outside_count += int(outside.sum().detach().cpu().item())
            prefix_count += int(all_prefix.sum().detach().cpu().item())
            suffix_count += int(all_suffix.sum().detach().cpu().item())
            if bool(outside.any().detach().cpu().item()):
                min_reduced = min(
                    min_reduced,
                    float(reduced_margin[outside].min().detach().cpu().item()),
                )
                min_key = min(
                    min_key,
                    float(route_margins[outside].min().detach().cpu().item()),
                )
            if bool(all_prefix.any().detach().cpu().item()):
                mask = all_prefix.to(dtype=torch.float64)
                prefix_factor = prefix_factor + lease.low_rank_factor[:, lo:hi].matmul(mask)
                work_offset = work_offset + torch.sum(lease.corrected_work[lo:hi] * mask)
                if branch_result.branch is V1RelaxedKKTBranch.DIAGONAL_L1:
                    residual_offset = residual_offset + torch.sum(sqrt_residual[lo:hi] * mask)
                # Prefix block counts are small metadata, not route-scale D2H.
                prefix_flags = all_prefix.detach().cpu().numpy().astype(bool)
                if prefix_flags.any():
                    physical = np.arange(lo, hi, dtype=np.int64)[prefix_flags]
                    blocks = route_to_block[physical]
                    for block in range(len(prefix_block_counts)):
                        prefix_block_counts[block] += int(np.sum(blocks == block))
    assert_complete_v1_decision_level_cuda_route_scan(
        route_count=problem.route_count,
        tiles=tiles,
        scanned_route_count=scanned,
    )
    if prefix_count != custody.fixed_prefix_count or suffix_count != custody.fixed_suffix_count:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "fixed prefix/suffix hash custody counts do not match the complete CUDA scan",
            scanned_prefix_count=prefix_count,
            custody_prefix_count=custody.fixed_prefix_count,
            scanned_suffix_count=suffix_count,
            custody_suffix_count=custody.fixed_suffix_count,
        )
    receipt = V1RelaxedKKTOutsideShellIntervalScanReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        source_evidence_hash=custody.source_evidence_hash,
        shell_route_index_hash=custody.shell_route_index_hash,
        shell_count=shell_count,
        scanned_eligible_outside_route_count=outside_count,
        fixed_prefix_route_index_hash=custody.fixed_prefix_route_index_hash,
        fixed_prefix_count=prefix_count,
        fixed_suffix_route_index_hash=custody.fixed_suffix_route_index_hash,
        fixed_suffix_count=suffix_count,
        minimum_outside_reduced_advantage_margin_lower_bound=(
            min_reduced if outside_count else _ROBUST_SENSITIVITY_UNBOUNDED
        ),
        minimum_outside_real_key_margin_lower_bound=(
            min_key if outside_count else _ROBUST_SENSITIVITY_UNBOUNDED
        ),
        maximum_numerical_key_radius_upper_bound=numerical_key_radius,
        scan_backend="torch_cuda",
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    return _CudaOutsideScan(
        receipt=receipt,
        scanned_route_count=scanned,
        scanned_eligible_outside_route_count=outside_count,
        fixed_prefix_count=prefix_count,
        fixed_suffix_count=suffix_count,
        prefix_factor_offset=prefix_factor.detach().cpu().numpy().astype(np.float64, copy=True),
        corrected_work_offset=float(work_offset.detach().cpu().item()),
        residual_offset=float(residual_offset.detach().cpu().item()),
        prefix_block_counts=tuple(prefix_block_counts),
    )


@dataclass(frozen=True, slots=True)
class _HardFQView:
    restricted_factor_lower: np.ndarray
    restricted_factor_upper: np.ndarray
    factor_offset_lower: np.ndarray
    factor_offset_upper: np.ndarray
    residual_direction_lower: np.ndarray
    residual_direction_upper: np.ndarray
    residual_offset_lower: float
    residual_offset_upper: float
    spectral_rho_lower: float
    spectral_rho_upper: float
    hard_q_additive_radius_upper_bound: float


def _q_interval(
    *,
    branch: V1RelaxedKKTBranch,
    view: _HardFQView,
    selected_shell_offsets: tuple[int, ...],
    fixed_prefix_count: int,
) -> tuple[float, float]:
    low_terms: list[float] = []
    high_terms: list[float] = []
    for rank_offset in range(view.restricted_factor_lower.shape[0]):
        lower = _decision_level_outward_sum(
            (
                float(view.factor_offset_lower[rank_offset]),
                *(float(view.restricted_factor_lower[rank_offset, value]) for value in selected_shell_offsets),
            ),
            upper=False,
        )
        upper = _decision_level_outward_sum(
            (
                float(view.factor_offset_upper[rank_offset]),
                *(float(view.restricted_factor_upper[rank_offset, value]) for value in selected_shell_offsets),
            ),
            upper=True,
        )
        lo, hi = _decision_level_interval_square(lower, upper)
        low_terms.append(lo)
        high_terms.append(hi)
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        cardinality = fixed_prefix_count + len(selected_shell_offsets)
        low_terms.append(
            math.nextafter(view.spectral_rho_lower * cardinality, -math.inf)
            if cardinality
            else 0.0
        )
        high_terms.append(
            math.nextafter(view.spectral_rho_upper * cardinality, math.inf)
            if cardinality
            else 0.0
        )
    else:
        residual_lower = _decision_level_outward_sum(
            (
                view.residual_offset_lower,
                *(float(view.residual_direction_lower[value]) for value in selected_shell_offsets),
            ),
            upper=False,
        )
        residual_upper = _decision_level_outward_sum(
            (
                view.residual_offset_upper,
                *(float(view.residual_direction_upper[value]) for value in selected_shell_offsets),
            ),
            upper=True,
        )
        lo, hi = _decision_level_interval_square(residual_lower, residual_upper)
        low_terms.append(lo)
        high_terms.append(hi)
    nominal_lower = _decision_level_outward_sum(low_terms, upper=False)
    nominal_upper = _decision_level_outward_sum(high_terms, upper=True)
    radius = view.hard_q_additive_radius_upper_bound
    return (
        max(0.0, math.nextafter(nominal_lower - radius, -math.inf)),
        max(0.0, math.nextafter(nominal_upper + radius, math.inf)),
    )


def _issue_remaining_cuda_receipts(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    lease: V1DecisionLevelCudaFullStateLease,
    design: V1DecisionLevelCudaIssuerDesign,
    outside: _CudaOutsideScan,
) -> tuple[
    V1RelaxedKKTActiveShellRestrictedOperatorReceipt,
    V1RelaxedKKTRobustDualBoundReceipt,
    V1RelaxedKKTAllAdmissibleShellFacesReceipt,
    V1RelaxedKKTWorstCaseHardFQReplayReceipt,
]:
    torch = _require_torch()
    custody = lease.hash_custody
    if type(custody) is not V1DecisionLevelCudaShellHashCustody:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "decision-level CUDA receipt issuance requires exact hash custody",
        )
    shell = tuple(lease.shell_route_indices)
    shell_tensor = torch.tensor(shell, device=lease.canonical_key_lower.device, dtype=torch.long)
    key_lower_shell = lease.canonical_key_lower.index_select(0, shell_tensor).detach().cpu().numpy()
    key_upper_shell = lease.canonical_key_upper.index_select(0, shell_tensor).detach().cpu().numpy()
    factor = lease.low_rank_factor.index_select(1, shell_tensor)
    factor_np = factor.detach().cpu().numpy().astype(np.float64, copy=True)
    work_shell = lease.corrected_work.index_select(0, shell_tensor).detach().cpu().numpy().astype(np.float64, copy=True)
    residual_diag_shell = lease.residual_diagonal.index_select(0, shell_tensor)
    sqrt_residual_shell = torch.sqrt(residual_diag_shell).detach().cpu().numpy().astype(np.float64, copy=True)
    work_radius = _decision_level_outward_sum(
        (
            geometry_uncertainty.corrected_work_A_sampling_linf_radius,
            geometry_uncertainty.corrected_work_A_deterministic_linf_radius,
        ),
        upper=True,
    )
    operator_radius = _decision_level_outward_sum(
        (
            geometry_uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius,
            geometry_uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius,
            geometry_uncertainty.fd_q_residual_deterministic_linf_adjoint_radius,
        ),
        upper=True,
    )
    work_lower = np.maximum(0.0, np.nextafter(work_shell - work_radius, -math.inf))
    work_upper = np.nextafter(work_shell + work_radius, math.inf)
    factor_lower = factor_np
    factor_upper = factor_np
    factor_offset = outside.prefix_factor_offset
    if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL:
        residual_direction = np.zeros(len(shell), dtype=np.float64)
        residual_offset = 0.0
        rho = problem.residual_spectral_bound
    else:
        residual_direction = sqrt_residual_shell
        residual_offset = outside.residual_offset
        rho = 0.0
    hard_q_radius = math.nextafter(float(problem.legal_hard_capacity) * operator_radius, math.inf)
    factor_mid = factor_np
    operator_mid = factor_mid.T @ factor_mid
    if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL:
        operator_mid = operator_mid + rho * np.eye(len(shell), dtype=np.float64)
        residual_interval_hash = content_hash(
            {
                "branch": branch_result.branch,
                "rho_lower": rho,
                "rho_upper": rho,
            },
            prefix="partition-free-v1-relaxed-kkt-shell-residual-",
        )
    else:
        operator_mid = operator_mid + np.outer(residual_direction, residual_direction)
        residual_interval_hash = content_hash(
            {
                "branch": branch_result.branch,
                "direction_lower": residual_direction,
                "direction_upper": residual_direction,
                "offset_lower": residual_offset,
                "offset_upper": residual_offset,
            },
            prefix="partition-free-v1-relaxed-kkt-shell-residual-",
        )
    strong_convexity = certified_gram_operator_min_eigenvalue_lower_bound(
        factor_mid,
        residual_direction=(
            None
            if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL
            else residual_direction
        ),
        diagonal_shift=(
            rho if branch_result.branch is V1RelaxedKKTBranch.SPECTRAL else 0.0
        ),
        maximum_dimension=design.maximum_shell_routes,
    )
    operator_receipt = V1RelaxedKKTActiveShellRestrictedOperatorReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        route_axis_hash=problem.route_axis_hash,
        source_evidence_hash=custody.source_evidence_hash,
        shell_route_index_hash=custody.shell_route_index_hash,
        shell_count=len(shell),
        factor_rank=problem.rank,
        restricted_factor_interval_hash=content_hash(
            (factor_lower, factor_upper),
            prefix="partition-free-v1-relaxed-kkt-shell-factor-interval-",
        ),
        factor_offset_interval_hash=content_hash(
            (factor_offset, factor_offset),
            prefix="partition-free-v1-relaxed-kkt-shell-factor-offset-",
        ),
        residual_branch_interval_hash=residual_interval_hash,
        corrected_work_offset_interval_hash=content_hash(
            (
                outside.corrected_work_offset,
                outside.corrected_work_offset,
                work_lower,
                work_upper,
            ),
            prefix="partition-free-v1-relaxed-kkt-shell-work-offset-",
        ),
        hard_q_offset_interval_hash=content_hash(
            (factor_offset, factor_offset, residual_offset, residual_offset),
            prefix="partition-free-v1-relaxed-kkt-shell-q-offset-",
        ),
        shell_operator_norm_sampling_radius_upper_bound=(
            geometry_uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius
        ),
        shell_operator_norm_deterministic_radius_upper_bound=(
            geometry_uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius
            + geometry_uncertainty.fd_q_residual_deterministic_linf_adjoint_radius
        ),
        strong_convexity_lower_bound=strong_convexity,
        dense_reference_operator_interval_hash=None,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    gap = branch_result.receipt.gap
    primal_lower = max(0.0, math.nextafter(gap.primal_resource_lower_bound, -math.inf))
    dual_upper = math.nextafter(gap.conic_dual_resource_upper_bound, math.inf)
    gap_upper = math.nextafter(dual_upper - primal_lower, math.inf)
    strict_gap_tolerance = max(
        0.0,
        _decision_level_outward_sum(
            (
                policy.absolute_gap_tolerance,
                policy.relative_gap_tolerance * max(1.0, dual_upper),
            ),
            upper=False,
        ),
    )
    if gap_upper >= strict_gap_tolerance:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "robust primal/dual interval does not have a strict registered gap",
        )
    dual_receipt = V1RelaxedKKTRobustDualBoundReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        source_evidence_hash=custody.source_evidence_hash,
        restricted_operator_receipt_hash=operator_receipt.receipt_hash,
        robust_primal_resource_lower_bound=primal_lower,
        robust_conic_dual_resource_upper_bound=dual_upper,
        robust_duality_gap_upper_bound=gap_upper,
        registered_strict_gap_tolerance=strict_gap_tolerance,
        strict_gap_margin_lower_bound=math.nextafter(strict_gap_tolerance - gap_upper, -math.inf),
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    route_to_block, block_capacities = _decision_level_block_axis(problem)
    if any(
        count > capacity
        for count, capacity in zip(outside.prefix_block_counts, block_capacities, strict=True)
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "outside fixed prefix saturates a block before membership is invariant",
        )
    if any(count < outside.fixed_prefix_count for count in branch_identity.frozen_ladder.cumulative_counts):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "registered endpoint cuts through the outside fixed prefix",
        )
    reduced_options: list[tuple[int, ...]] = []
    for offset in range(len(shell)):
        reduced_lower = -float(key_upper_shell[offset, 1])
        reduced_upper = -float(key_lower_shell[offset, 1])
        if reduced_lower > 0.0:
            reduced_options.append((1,))
        elif reduced_upper < 0.0:
            reduced_options.append((0,))
        else:
            reduced_options.append((0, 1, 2))
    face_count = math.prod(len(value) for value in reduced_options)
    if face_count > design.maximum_admissible_faces:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.ROUNDING_OUTCOME_LIMIT_EXCEEDED,
            "active shell has more admissible faces than the registered limit",
        )
    physical_by_offset = {route: offset for offset, route in enumerate(shell)}
    registered_axis: list[str] = []
    registered_endpoint_offsets: list[tuple[int, ...]] = []
    for action, count in zip(
        branch_identity.frozen_ladder.actions,
        branch_identity.frozen_ladder.cumulative_counts,
        strict=True,
    ):
        shell_take = count - outside.fixed_prefix_count
        if shell_take < 0:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                "registered endpoint cuts through the fixed prefix",
            )
        # The hash custody defines the registered endpoint by shell offsets.
        matches = [
            offsets
            for offsets, digest in custody.endpoint_hash_by_shell_offsets.items()
            if len(offsets) == shell_take and digest == action.removed_route_index_hash
        ]
        if len(matches) != 1:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "hash custody does not bind one frozen endpoint action",
            )
        registered_endpoint_offsets.append(matches[0])
        registered_axis.append(action.removed_route_index_hash)
    outcome_count = 0
    outcome_hashes: list[str] = []
    physical_ties = 0
    semidefinite = face_count if strong_convexity <= 0.0 else max(0, face_count - 1)
    numerical_key_radius = outside.receipt.maximum_numerical_key_radius_upper_bound
    for face in __import__("itertools").product(*reduced_options):
        face_lower = np.array(key_lower_shell, copy=True)
        face_upper = np.array(key_upper_shell, copy=True)
        for offset, state in enumerate(face):
            if state == 0:
                face_lower[offset, 0] = face_upper[offset, 0] = 0.0
            elif state == 1:
                face_lower[offset, 0] = face_upper[offset, 0] = -1.0
        precedence = {route: set() for route in shell}
        for left_index, left in enumerate(shell):
            for right in shell[left_index + 1 :]:
                left_offset = physical_by_offset[left]
                right_offset = physical_by_offset[right]
                relation, margin = classify_v1_decision_level_lex_interval_pair(
                    left_lower=face_lower[left_offset],
                    left_upper=face_upper[left_offset],
                    right_lower=face_lower[right_offset],
                    right_upper=face_upper[right_offset],
                )
                if relation == 2:
                    physical_ties += 1
                    _refuse(
                        V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                        "active-shell order contains a physical-index-only tie",
                    )
                if relation == -1 and margin > 2.0 * numerical_key_radius:
                    precedence[right].add(left)
                elif relation == 1 and margin > 2.0 * numerical_key_radius:
                    precedence[left].add(right)
        for order in _decision_level_topological_orders(
            shell,
            precedence,
            maximum_outcomes=design.maximum_rounding_outcomes - outcome_count,
        ):
            block_counts = list(outside.prefix_block_counts)
            selected_offsets: list[int] = []
            target_offset = 0
            axis: list[str] = []
            while (
                target_offset < len(branch_identity.frozen_ladder.cumulative_counts)
                and branch_identity.frozen_ladder.cumulative_counts[target_offset]
                == outside.fixed_prefix_count
            ):
                axis.append(custody.endpoint_hash_for_shell_offsets(()))
                target_offset += 1
            for route in order:
                block = int(route_to_block[route])
                if block_counts[block] >= block_capacities[block]:
                    continue
                block_counts[block] += 1
                selected_offsets.append(physical_by_offset[route])
                while (
                    target_offset < len(branch_identity.frozen_ladder.cumulative_counts)
                    and branch_identity.frozen_ladder.cumulative_counts[target_offset]
                    == outside.fixed_prefix_count + len(selected_offsets)
                ):
                    axis.append(custody.endpoint_hash_for_shell_offsets(tuple(selected_offsets)))
                    target_offset += 1
            if target_offset != len(branch_identity.frozen_ladder.cumulative_counts):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "one admissible shell ordering cannot reach every frozen endpoint",
                )
            outcome_count += 1
            if outcome_count > design.maximum_rounding_outcomes:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.ROUNDING_OUTCOME_LIMIT_EXCEEDED,
                    "active shell exceeds the exhaustive rounding-outcome limit",
                )
            if tuple(axis) != tuple(registered_axis):
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                    "admissible nonunique face or key order rounds to a different ladder",
                )
            outcome_hashes.append(
                content_hash(
                    {
                        "face": tuple(face),
                        "order": tuple(order),
                        "ladder_axis": tuple(axis),
                    },
                    prefix=(
                        "partition-free-v1-relaxed-kkt-shell-face-outcome-"
                    ),
                )
            )
    if outcome_count <= 0 or len(outcome_hashes) != outcome_count:
        raise RuntimeError("decision-level CUDA shell enumeration produced no outcomes")
    exact_outcome_ladder_axis_hash = content_hash(
        tuple(outcome_hashes),
        prefix="partition-free-v1-relaxed-kkt-all-shell-outcomes-",
    )
    faces_receipt = V1RelaxedKKTAllAdmissibleShellFacesReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        source_evidence_hash=custody.source_evidence_hash,
        shell_route_index_hash=custody.shell_route_index_hash,
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        admissible_face_count=face_count,
        admissible_rounding_outcome_count=outcome_count,
        outcome_ladder_axis_hash=exact_outcome_ladder_axis_hash,
        physical_index_only_tie_count=physical_ties,
        semidefinite_or_nonunique_face_count=semidefinite,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    view = _HardFQView(
        restricted_factor_lower=factor_lower,
        restricted_factor_upper=factor_upper,
        factor_offset_lower=factor_offset,
        factor_offset_upper=factor_offset,
        residual_direction_lower=residual_direction,
        residual_direction_upper=residual_direction,
        residual_offset_lower=residual_offset,
        residual_offset_upper=residual_offset,
        spectral_rho_lower=rho,
        spectral_rho_upper=rho,
        hard_q_additive_radius_upper_bound=hard_q_radius,
    )
    f_intervals: list[tuple[float, float]] = []
    q_intervals: list[tuple[float, float]] = []
    statuses: list[str] = []
    endpoint_hashes: list[str] = []
    for offsets in registered_endpoint_offsets:
        f_lower = max(
            0.0,
            _decision_level_outward_sum(
                (outside.corrected_work_offset, *(float(work_lower[value]) for value in offsets)),
                upper=False,
            ),
        )
        f_upper = max(
            0.0,
            _decision_level_outward_sum(
                (outside.corrected_work_offset, *(float(work_upper[value]) for value in offsets)),
                upper=True,
            ),
        )
        q_lower, q_upper = _q_interval(
            branch=branch_result.branch,
            view=view,
            selected_shell_offsets=offsets,
            fixed_prefix_count=outside.fixed_prefix_count,
        )
        safe = f_upper < problem.work_budget and q_upper < problem.quadratic_budget
        impossible = f_lower > problem.work_budget or q_lower > problem.quadratic_budget
        if safe == impossible or (not safe and not impossible):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                "worst-case hard F/Q interval touches a feasibility boundary",
            )
        endpoint_hash = custody.endpoint_hash_for_shell_offsets(offsets)
        endpoint_hashes.append(endpoint_hash)
        f_intervals.append((f_lower, f_upper))
        q_intervals.append((q_lower, q_upper))
        statuses.append("safe_feasible" if safe else "robust_impossible")
    hard_status_axis_hash = content_hash(
        tuple(zip(endpoint_hashes, statuses, strict=True)),
        prefix="partition-free-v1-relaxed-kkt-decision-hard-fq-status-axis-",
    )
    hard_fq_receipt = V1RelaxedKKTWorstCaseHardFQReplayReceipt(
        branch=branch_result.branch,
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        source_evidence_hash=custody.source_evidence_hash,
        rounded_ladder_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        endpoint_removed_index_hashes=tuple(endpoint_hashes),
        endpoint_f_intervals=tuple(f_intervals),
        endpoint_q_intervals=tuple(q_intervals),
        endpoint_statuses=tuple(statuses),
        hard_fq_status_axis_hash=hard_status_axis_hash,
        _issuance_token=_DECISION_LEVEL_SOURCE_RECEIPT_ISSUANCE_TOKEN,
    )
    return operator_receipt, dual_receipt, faces_receipt, hard_fq_receipt


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaIssuerTelemetry:
    design_hash: str
    forecast_hash: str
    private_primal_binding_hash: str
    route_count: int
    route_tile_count: int
    scanned_route_count: int
    scanned_eligible_outside_route_count: int
    scan_backend: Literal["torch_cuda"] = "torch_cuda"
    full_key_index_d2h_rows: int = 0
    route_scale_row_retention: bool = False
    nonmonotone_q_early_stop_used: bool = False
    rank_padding_used: bool = False
    schema: str = SCHEMA_TELEMETRY

    def __post_init__(self) -> None:
        for name in ("design_hash", "forecast_hash", "private_primal_binding_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "route_count",
            "route_tile_count",
            "scanned_route_count",
            "scanned_eligible_outside_route_count",
            "full_key_index_d2h_rows",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.scan_backend != "torch_cuda":
            raise ValueError("decision-level CUDA issuer telemetry cannot name a CPU backend")
        if (
            self.full_key_index_d2h_rows != 0
            or self.route_scale_row_retention
            or self.nonmonotone_q_early_stop_used
            or self.rank_padding_used
        ):
            raise PermissionError("unsafe decision-level CUDA issuer behavior was observed")
        if self.scanned_route_count != self.route_count:
            raise PermissionError("decision-level CUDA issuer telemetry is not a complete scan")
        if self.schema != SCHEMA_TELEMETRY:
            raise ValueError("unsupported decision-level CUDA telemetry schema")

    @property
    def telemetry_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-decision-cuda-telemetry-")


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaIssuerResult:
    certificate: V1RelaxedKKTDecisionLevelRobustInvarianceCertificate
    outside_shell_interval_scan: V1RelaxedKKTOutsideShellIntervalScanReceipt
    active_shell_restricted_operator: V1RelaxedKKTActiveShellRestrictedOperatorReceipt
    robust_dual_bound: V1RelaxedKKTRobustDualBoundReceipt
    worst_case_hard_fq_replay: V1RelaxedKKTWorstCaseHardFQReplayReceipt
    all_admissible_shell_faces: V1RelaxedKKTAllAdmissibleShellFacesReceipt
    private_primal_binding: V1DecisionLevelCudaPrivatePrimalBinding
    forecast: V1DecisionLevelCudaIssuerHBMForecast
    telemetry: V1DecisionLevelCudaIssuerTelemetry


def issue_v1_decision_level_cuda_certificate(
    *,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    nominal_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    source_lease: V1DecisionLevelCudaFullStateLease | None,
    private_primal_binding: V1DecisionLevelCudaPrivatePrimalBinding | None = None,
    design: V1DecisionLevelCudaIssuerDesign = V1DecisionLevelCudaIssuerDesign(),
) -> V1DecisionLevelCudaIssuerResult:
    """Refuse the superseded fixed-shell authority path.

    The retained implementation below is intentionally unreachable production
    provenance until a variable-solution theorem bounds every admissible
    alternative optimum or epsilon-near optimum.  Fixed-primal normal-cone,
    key-order, and hard-endpoint replay evidence alone cannot authorize C.
    """

    if type(design) is not V1DecisionLevelCudaIssuerDesign:
        raise TypeError("design must be exact V1DecisionLevelCudaIssuerDesign")
    _refuse(
        V1DecisionLevelCudaIssuerRefusalCode.ALTERNATIVE_SOLUTION_SENSITIVITY_UNRESOLVED,
        "nominal shell replay does not bound alternative optimizer motion; "
        "canonical C requires a source-certified variable-solution action-"
        "invariance theorem",
    )
    if source_lease is None:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.FULL_STATE_LEASE_REQUIRED,
            "decision-level CUDA issuer requires an exact live full-state lease",
        )
    if type(source_lease) is not V1DecisionLevelCudaFullStateLease:
        raise TypeError("source_lease must be exact V1DecisionLevelCudaFullStateLease")
    if type(source_lease.hash_custody) is not V1DecisionLevelCudaShellHashCustody:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "decision-level CUDA issuer requires exact shell hash custody",
        )
    if private_primal_binding is None:
        requirement = v1_decision_level_cuda_private_theorem_state_hook_requirement(
            reason=(
                "route-scale source lease has not been bound to post-D_W "
                "private-primal/literal-J custody"
            )
        )
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_PRIMAL_CUSTODY_REQUIRED,
            "decision-level CUDA issuer requires a private-primal binding receipt",
            requirement_hash=requirement.requirement_hash,
            required_hook_signature=requirement.required_hook_signature,
            required_return_contract=requirement.required_return_contract,
        )
    if type(private_primal_binding) is not V1DecisionLevelCudaPrivatePrimalBinding:
        raise TypeError(
            "private_primal_binding must be exact "
            "V1DecisionLevelCudaPrivatePrimalBinding"
        )
    if source_lease.private_primal_binding_hash != private_primal_binding.binding_hash:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.PRIVATE_PRIMAL_CUSTODY_REQUIRED,
            "full-state lease is not bound to the supplied private-primal receipt",
            lease_private_primal_binding_hash=source_lease.private_primal_binding_hash,
            private_primal_binding_hash=private_primal_binding.binding_hash,
        )
    for value, expected, name in (
        (problem, V1RelaxedKKTProblem, "problem"),
        (policy, V1RelaxedKKTPolicy, "policy"),
        (branch_result, V1RelaxedKKTBranchResult, "branch_result"),
        (branch_identity, V1RelaxedKKTDeviceShellBranchIdentityReceipt, "branch_identity"),
        (geometry_uncertainty, V1RelaxedKKTGeometryUncertaintySet, "geometry_uncertainty"),
        (nominal_witness, V1RelaxedKKTAdaptiveRankBranchWitness, "nominal_witness"),
    ):
        if type(value) is not expected:
            raise TypeError(f"{name} must be exact {expected.__name__}")
    if (
        private_primal_binding.problem_hash != problem.problem_hash
        or private_primal_binding.policy_hash != policy.policy_hash
        or private_primal_binding.branch != branch_result.branch.value
        or private_primal_binding.branch_result_hash != branch_result.result_hash
        or private_primal_binding.branch_receipt_hash
        != branch_result.receipt.receipt_hash
        or private_primal_binding.source_rounding_hash
        != branch_identity.source_rounding_hash
        or private_primal_binding.endpoint_action_axis_hash
        != branch_identity.endpoint_action_axis_hash
        or private_primal_binding.hash_custody_hash
        != source_lease.hash_custody.custody_hash
        or private_primal_binding.source_evidence_hash
        != source_lease.hash_custody.source_evidence_hash
        or private_primal_binding.shell_route_index_hash
        != source_lease.hash_custody.shell_route_index_hash
        or private_primal_binding.geometry_uncertainty_hash
        != geometry_uncertainty.uncertainty_hash
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "private-primal binding does not close against issuer inputs",
        )
    torch = _require_torch()
    if not torch.cuda.is_available():
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.CUDA_UNAVAILABLE,
            "decision-level production source issuer requires CUDA",
        )
    source_lease._require_live()
    try:
        forecast = _forecast_hbm_from_device(
            torch=torch,
            device=source_lease.canonical_key_lower.device,
            route_count=problem.route_count,
            rank=problem.rank,
            shell_count=source_lease.shell_count,
            endpoint_count=branch_identity.frozen_ladder.endpoint_count,
            design=design,
        )
    except Exception as error:  # pragma: no cover - host/device dependent.
        _refuse(V1DecisionLevelCudaIssuerRefusalCode.CUDA_UNAVAILABLE, str(error))
    assert_v1_decision_level_cuda_hbm_admitted(forecast)
    outside = _issue_outside_shell_scan_cuda(
        problem=problem,
        policy=policy,
        branch_result=branch_result,
        geometry_uncertainty=geometry_uncertainty,
        lease=source_lease,
        design=design,
    )
    operator_receipt, dual_receipt, faces_receipt, hard_fq_receipt = (
        _issue_remaining_cuda_receipts(
            problem=problem,
            policy=policy,
            branch_result=branch_result,
            branch_identity=branch_identity,
            geometry_uncertainty=geometry_uncertainty,
            lease=source_lease,
            design=design,
            outside=outside,
        )
    )
    certificate = issue_v1_relaxed_kkt_decision_level_robust_invariance_from_source_receipts(
        problem=problem,
        policy=policy,
        branch_result=branch_result,
        branch_identity=branch_identity,
        geometry_uncertainty=geometry_uncertainty,
        nominal_witness=nominal_witness,
        outside_shell_interval_scan=outside.receipt,
        active_shell_restricted_operator=operator_receipt,
        robust_dual_bound=dual_receipt,
        worst_case_hard_fq_replay=hard_fq_receipt,
        all_admissible_shell_faces=faces_receipt,
        issuer_backend="torch_cuda",
    )
    telemetry = V1DecisionLevelCudaIssuerTelemetry(
        design_hash=design.design_hash,
        forecast_hash=forecast.forecast_hash,
        private_primal_binding_hash=private_primal_binding.binding_hash,
        route_count=problem.route_count,
        route_tile_count=len(
            plan_v1_decision_level_cuda_route_tiles(
                route_count=problem.route_count,
                route_tile_size=design.route_tile_size,
            )
        ),
        scanned_route_count=outside.scanned_route_count,
        scanned_eligible_outside_route_count=outside.scanned_eligible_outside_route_count,
    )
    return V1DecisionLevelCudaIssuerResult(
        certificate=certificate,
        outside_shell_interval_scan=outside.receipt,
        active_shell_restricted_operator=operator_receipt,
        robust_dual_bound=dual_receipt,
        worst_case_hard_fq_replay=hard_fq_receipt,
        all_admissible_shell_faces=faces_receipt,
        private_primal_binding=private_primal_binding,
        forecast=forecast,
        telemetry=telemetry,
    )


__all__ = [
    "V1DecisionLevelCudaFullStateLease",
    "V1DecisionLevelCudaIssuerDesign",
    "V1DecisionLevelCudaIssuerHBMForecast",
    "V1DecisionLevelCudaIssuerRefusal",
    "V1DecisionLevelCudaIssuerRefusalCode",
    "V1DecisionLevelCudaIssuerResult",
    "V1DecisionLevelCudaIssuerTelemetry",
    "V1DecisionLevelCudaKKTLeaseAdapter",
    "V1DecisionLevelCudaPrivatePrimalBinding",
    "V1DecisionLevelCudaPrivateTheoremStateHookRequirement",
    "V1DecisionLevelCudaRouteTile",
    "V1DecisionLevelCudaShellHashCustody",
    "assert_complete_v1_decision_level_cuda_route_scan",
    "assert_v1_decision_level_cuda_hbm_admitted",
    "bind_v1_decision_level_cuda_private_primal_custody",
    "build_v1_decision_level_cuda_theorem_state_borrow_request",
    "build_v1_decision_level_cuda_theorem_state_borrow_request_from_sources",
    "classify_v1_decision_level_lex_interval_pair",
    "estimate_v1_decision_level_cuda_issuer_hbm",
    "issue_v1_decision_level_cuda_certificate",
    "plan_v1_decision_level_cuda_route_tiles",
    "v1_decision_level_cuda_private_theorem_state_hook_requirement",
]
