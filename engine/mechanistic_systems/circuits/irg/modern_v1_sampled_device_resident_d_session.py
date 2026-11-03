"""Device-resident sampled-D session for modern-v1 discovery.

This module is a production-shaped joint for a systems bottleneck: the
registered-look transaction restores and snapshots route-scale
corrected/FD state at every look boundary.  A live
``V1SampledDeviceResidentDSession`` instead owns one nonserializable,
identity-bound state across registered look packets:

* corrected Aumann--Shapley signed/absolute Welford statistics are retained on
  the declared device;
* canonical production keeps one HBM-admitted ceiling-rank literal-J FD
  accumulator from the registered D prefix; lower ranks are nested metadata
  views of that same ceiling state, never independent sketches or zero-padded
  replays;
* legacy CPU/reference mode may still update every HBM-admitted literal-J FD
  rank for historical replay and exact small-domain comparator tests;
* intermediate receipts are scalar/axis/hash-only and record zero route-scale
  D2H;
* route-scale portable materialization is explicit, at-most-once, and authority
  inert with respect to rank stability.

The class deliberately grants no C/K/O, commit, terminal, performance, or
Pareto authority.  CPU execution exists only as an exact tiny reference and
contract test shape.  Production mode requires an actual CUDA device and fails
closed rather than promoting a CPU fallback.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Integral, Real
from typing import Any, Final, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from mechanistic_systems.circuits.irg.modern_v1_stratified_fd_gpu import (
    V1StratifiedDeviceFDTrustAccumulator,
    V1StratifiedFDAccumulatorSnapshot,
    V1StratifiedFDDesign,
    V1StratifiedFDDeviceGeometryState,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import ordered_string_axis_hash
from mechanistic_systems.measurement.device_route_moments import (
    DeviceGramFrequentDirectionsTrustAccumulator,
)

try:  # pragma: no cover - import availability is environment dependent.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V1: Final = (
    "partition_free_v1_sampled_device_resident_D_session_design_v1"
)
DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_device_resident_D_session_design_v2"
)
DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA: Final = (
    DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V2
)
DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V2: Final = (
    "partition_free_v1_sampled_device_resident_D_session_hbm_forecast_v2"
)
DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V3: Final = (
    "partition_free_v1_sampled_device_resident_D_session_hbm_forecast_v3"
)
DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA: Final = (
    DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V3
)
_ROW_FREE_FINALIZATION_TOKEN: Final = object()
DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_look_packet_receipt_v1"
)
DEVICE_RESIDENT_D_MATERIALIZATION_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_materialization_v1"
)
DEVICE_RESIDENT_D_RANK_SELECTION_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_rank_selection_v1"
)
DEVICE_RESIDENT_D_MATERIALIZATION_RANK_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_materialization_rank_v1"
)
DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_replay_coordinate_v1"
)
DEVICE_RESIDENT_D_ROW_FREE_FINALIZATION_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_row_free_finalization_v1"
)
DEVICE_RESIDENT_D_KKT_GEOMETRY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_device_resident_D_KKT_geometry_receipt_v1"
)
DEVICE_RESIDENT_D_SINGLETON_WORK_SEMANTICS: Final = (
    "stratum_weighted_mean_absolute_corrected_A_routewise_singleton_only_"
    "not_native_fiber_signed_F_v1"
)
DEVICE_RESIDENT_D_KKT_FACTOR_SEMANTICS: Final = (
    "literal_J_global_mean_rows_then_centered_FD_rows_v1"
)
_KKT_GEOMETRY_ISSUANCE_TOKEN: Final = object()
DEVICE_RESIDENT_D_SELECTED_MATERIALIZATION_MODE: Final = (
    "selected_rank_authorized_materialization_v1"
)
DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE: Final = (
    "source_registered_rank_authority_inert_materialization_v1"
)
DEVICE_RESIDENT_D_DIAGNOSTIC_MATERIALIZATION_MODE: Final = (
    "diagnostic_all_hbm_admitted_ranks_materialization_v1"
)
DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE: Final = (
    "sampled_D_device_resident_sufficient_statistics_only_no_C_K_O_commit_"
    "terminal_performance_or_pareto_authority_v1"
)
DEVICE_RESIDENT_D_SELECTION_RULE: Final = (
    "choose_smallest_preoutcome_hbm_admitted_rank_marked_decision_stable_v1"
)
DEVICE_RESIDENT_D_MATERIALIZATION_RANK_RULE: Final = (
    "materialize_source_registered_highest_hbm_admitted_proposal_rank_without_"
    "rank_stability_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-device-resident-D-"
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8


class V1SampledDeviceResidentDSessionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1SampledDeviceResidentDRankStorageMode(str, Enum):
    CANONICAL_CEILING_ONLY = "canonical_ceiling_only_v1"
    LEGACY_INDEPENDENT_PREFIX = "legacy_independent_prefix_v1"


class V1SampledDeviceResidentDSessionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    ACTIVE_LOOK = "active_look"
    NO_ACTIVE_LOOK = "no_active_look"
    DUPLICATE_CONTEXT = "duplicate_context"
    NONFINITE_SOURCE = "nonfinite_source"
    EVICTED = "evicted"
    MATERIALIZED = "materialized"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_RECEIPT = "mutated_receipt"
    RANK_NOT_ADMITTED = "rank_not_admitted"
    ACTIVE_LEASE = "active_lease"
    MUTATED_CUSTODY = "mutated_custody"
    DUPLICATE_FREEZE = "duplicate_freeze"
    DEVICE_MISMATCH = "device_mismatch"


class V1SampledDeviceResidentDSessionRefusal(RuntimeError):
    """Typed fail-closed refusal for the live sampled-D session."""

    def __init__(
        self,
        code: V1SampledDeviceResidentDSessionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledDeviceResidentDSessionRefusalCode:
            raise TypeError("sampled-D session refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledDeviceResidentDSessionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledDeviceResidentDSessionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(
    values: Sequence[str],
    *,
    name: str,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be an ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in values)
    if not result:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    if unique and len(result) != len(set(result)):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
            f"{name} must be unique",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, np.integer)):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _positive_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be one integer pair",
        )
    raw = tuple(value)
    if len(raw) != 2:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
            f"{name} must have two entries",
        )
    return (
        _positive_int(raw[0], name=f"{name} first"),
        _positive_int(raw[1], name=f"{name} second"),
    )


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be one finite positive real",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
            f"{name} must be finite and positive",
        )
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
            f"{name} must be exact bool",
        )
    return value


def _torch_device(value: Any) -> Any:
    if torch is None:
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.DEVICE_REQUIRED,
            "sampled-D device resident session requires PyTorch",
        )
    return torch.device(value)


def _tensor_storage_invariant(value: Any) -> tuple[object, ...]:
    if torch is None or not torch.is_tensor(value):
        _refuse(
            V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
            "KKT geometry storage is no longer a torch tensor",
        )
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


def _fd_snapshot_bytes(snapshot: V1StratifiedFDAccumulatorSnapshot) -> int:
    total = 0
    for _name, child in snapshot.stratum_snapshots:
        arrays = (
            child.active_buffer_rows,
            child.mean,
            child.weighted_second_moment_diagonal,
        )
        total += sum(0 if value is None else int(np.asarray(value).nbytes) for value in arrays)
    return total


def _readonly_le_f64(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value), dtype=np.dtype("<f8"))
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype("<f8")).reshape(
        array.shape
    )
    frozen.setflags(write=False)
    return frozen


def _axis_hash(values: Sequence[str], *, prefix: str) -> str:
    return content_hash(tuple(values), prefix=prefix)


def _fd_packet_boundary_index(
    accumulator: V1StratifiedDeviceFDTrustAccumulator,
) -> int:
    """Return the scalar packet boundary without route-scale materialization."""

    return _nonnegative_int(
        accumulator._packet_boundary_index,
        name="FD packet boundary index",
    )


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDSessionDesign:
    """Frozen source and resource identity for one live sampled-D session."""

    session_id: str
    fd_design: V1StratifiedFDDesign
    corrected_a_stream_identity_hash: str
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    fd_rank_ladder: tuple[int, ...]
    maximum_contexts_per_look: int
    registered_route_chunk_size: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    current_reserved_device_bytes: int = 0
    provider_peak_workspace_bytes: int = 0
    hbm_ceiling_fraction: float = 0.85
    execution_mode: V1SampledDeviceResidentDSessionMode = (
        V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION
    )
    deterministic_fp64_required: bool = True
    forbid_intermediate_route_scale_d2h: bool = True
    final_materialization_at_most_once: bool = True
    rank_storage_mode: V1SampledDeviceResidentDRankStorageMode | None = None
    schema: str = DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _identity(self.session_id, name="session id"))
        if type(self.fd_design) is not V1StratifiedFDDesign:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                "sampled-D session requires exact stratified FD design",
            )
        self.fd_design.assert_integrity()
        if len(self.fd_design.stratum_names) != 2:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "first device-resident D session supports the canonical two-stratum D split",
            )
        corrected = _identity(
            self.corrected_a_stream_identity_hash,
            name="corrected-A stream identity hash",
        )
        if corrected == self.fd_design.raw_logit_stream_source_hash:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "corrected-A and literal-J streams must remain distinct",
            )
        object.__setattr__(self, "corrected_a_stream_identity_hash", corrected)
        object.__setattr__(
            self,
            "population_stratum_counts",
            _positive_pair(self.population_stratum_counts, name="population stratum counts"),
        )
        object.__setattr__(
            self,
            "sealed_population_order_hashes",
            tuple(
                _identity(value, name="sealed population order hash")
                for value in self.sealed_population_order_hashes
            ),
        )
        if len(self.sealed_population_order_hashes) != 2:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "sealed population order hashes must align with the two strata",
            )
        ladder = tuple(_positive_int(value, name="FD rank") for value in self.fd_rank_ladder)
        if (
            not ladder
            or ladder != tuple(sorted(set(ladder)))
            or any(value > self.fd_design.spec.maximum_rank for value in ladder)
            or any(value > len(self.fd_design.route_ids) for value in ladder)
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "FD rank ladder must be increasing, unique, and within the physical route/spec limits",
            )
        object.__setattr__(self, "fd_rank_ladder", ladder)
        object.__setattr__(
            self,
            "maximum_contexts_per_look",
            _positive_int(self.maximum_contexts_per_look, name="maximum contexts per look"),
        )
        chunk = _positive_int(self.registered_route_chunk_size, name="registered route chunk size")
        if chunk > len(self.fd_design.route_ids):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "registered route chunk exceeds route axis",
            )
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        for name in (
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "current_reserved_device_bytes",
            "provider_peak_workspace_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if self.device_total_memory_bytes <= 0:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "device total memory must be positive",
            )
        ceiling = _finite_fraction(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling > 0.85:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "canonical HBM ceiling cannot exceed 0.85",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if type(self.execution_mode) is not V1SampledDeviceResidentDSessionMode:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                "execution mode must have exact enum type",
            )
        if self.schema not in (
            DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V1,
            DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V2,
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "unsupported sampled-D session design schema",
            )
        storage = self.rank_storage_mode
        if storage is None:
            storage = (
                V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
                if self.execution_mode
                is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION
                else V1SampledDeviceResidentDRankStorageMode.LEGACY_INDEPENDENT_PREFIX
            )
        elif type(storage) is not V1SampledDeviceResidentDRankStorageMode:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                "rank storage mode must have exact enum type",
            )
        if (
            storage
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
            and self.schema == DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA_V1
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "canonical ceiling-only D rank storage requires the v2 session design schema",
            )
        if (
            self.execution_mode is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION
            and storage
            is not V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "production sampled-D sessions must use canonical ceiling-only rank storage",
            )
        object.__setattr__(self, "rank_storage_mode", storage)
        flags = (
            self.deterministic_fp64_required,
            self.forbid_intermediate_route_scale_d2h,
            self.final_materialization_at_most_once,
        )
        if any(type(item) is not bool for item in flags) or any(item is not True for item in flags):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "deterministic fp64, no intermediate D2H, and at-most-once final materialization are mandatory",
            )

    @property
    def route_count(self) -> int:
        return len(self.fd_design.route_ids)

    @property
    def route_axis_hash(self) -> str:
        return self.fd_design.route_axis_hash

    @property
    def literal_j_stream_identity_hash(self) -> str:
        return self.fd_design.raw_logit_stream_source_hash

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDSessionHBMForecast:
    design_hash: str
    route_count: int
    stratum_count: int
    maximum_contexts_per_look: int
    registered_route_chunk_size: int
    fd_rank_ladder: tuple[int, ...]
    hbm_admitted_fd_ranks: tuple[int, ...]
    corrected_state_bytes: int
    active_literal_j_packet_bytes: int
    colive_dual_chunk_bytes: int
    corrected_chunk_reducer_workspace_bytes: int
    provider_peak_workspace_bytes: int
    fd_state_bytes_by_rank: tuple[tuple[int, int], ...]
    fd_update_workspace_bytes_by_rank: tuple[tuple[int, int], ...]
    maximum_fd_update_workspace_bytes: int
    total_fd_state_bytes: int
    incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    rank_storage_mode: V1SampledDeviceResidentDRankStorageMode | None = None
    ceiling_fd_rank: int | None = None
    live_fd_accumulator_ranks: tuple[int, ...] = ()
    exposed_nested_fd_ranks: tuple[int, ...] = ()
    lower_rank_views_are_metadata_only: bool = False
    independent_rank_accumulators_required: bool = True
    lower_rank_zero_padding_authority: bool = False
    rank_stability_authority: bool = False
    schema: str = DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.design_hash, name="D-session design hash")
        if self.schema not in (
            DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V2,
            DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V3,
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "unsupported D-session HBM forecast schema",
            )
        storage = self.rank_storage_mode
        if storage is None:
            if self.schema != DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V2:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                    "v3 D-session HBM forecasts must name their rank storage mode",
                )
            storage = V1SampledDeviceResidentDRankStorageMode.LEGACY_INDEPENDENT_PREFIX
        elif type(storage) is not V1SampledDeviceResidentDRankStorageMode:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                "D-session HBM forecast rank storage mode must have exact enum type",
            )
        if (
            storage
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
            and self.schema != DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V3
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "canonical ceiling-only HBM forecasts require the v3 schema",
            )
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(self.stratum_count, name="stratum count")
        maximum_contexts = _positive_int(
            self.maximum_contexts_per_look,
            name="maximum contexts per look",
        )
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        ladder = tuple(_positive_int(value, name="FD rank") for value in self.fd_rank_ladder)
        admitted = tuple(
            _positive_int(value, name="HBM-admitted FD rank")
            for value in self.hbm_admitted_fd_ranks
        )
        if not ladder or any(rank not in ladder for rank in admitted):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "HBM-admitted ranks must be drawn from the registered rank ladder",
            )
        fd_bytes = tuple(
            (
                _positive_int(rank, name="FD state rank"),
                _nonnegative_int(byte_count, name="FD state bytes"),
            )
            for rank, byte_count in self.fd_state_bytes_by_rank
        )
        if tuple(rank for rank, _bytes in fd_bytes) != ladder:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "FD byte forecast must cover the rank ladder in order",
            )
        expected_corrected = 4 * strata * routes * _F64_BYTES
        expected_packet = maximum_contexts * routes * _F64_BYTES
        expected_chunk = 2 * maximum_contexts * min(chunk, routes) * _F64_BYTES
        expected_corrected_reducer = (
            3 * maximum_contexts * min(chunk, routes) * _F64_BYTES
        )
        expected_provider = _nonnegative_int(
            self.provider_peak_workspace_bytes,
            name="provider peak workspace bytes",
        )
        expected_fd = tuple(
            (rank, strata * routes * _F64_BYTES * (2 * rank + 2))
            for rank in ladder
        )
        expected_fd_workspace = tuple(
            (
                rank,
                2 * rank * routes * _F64_BYTES
                + (2 * rank) ** 2 * _F64_BYTES,
            )
            for rank in ladder
        )
        observed_fd_workspace = tuple(
            (
                _positive_int(rank, name="FD update-workspace rank"),
                _nonnegative_int(byte_count, name="FD update-workspace bytes"),
            )
            for rank, byte_count in self.fd_update_workspace_bytes_by_rank
        )
        if tuple(rank for rank, _bytes in observed_fd_workspace) != ladder:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "FD update-workspace forecast must cover the rank ladder in order",
            )
        expected_admitted: list[int] = []
        total_fd = 0
        total = _positive_int(self.device_total_memory_bytes, name="device total memory")
        free = _nonnegative_int(self.device_free_memory_bytes, name="device free memory")
        reserved = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="current reserved device bytes",
        )
        ceiling_fraction = _finite_fraction(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        ceiling = math.floor(total * ceiling_fraction)
        expected_live_ranks: tuple[int, ...]
        expected_exposed_ranks: tuple[int, ...]
        expected_ceiling_rank: int | None
        expected_max_fd_workspace: int
        fd_bytes_map = dict(expected_fd)
        fd_workspace_map = dict(expected_fd_workspace)
        if storage is V1SampledDeviceResidentDRankStorageMode.LEGACY_INDEPENDENT_PREFIX:
            for rank, bytes_for_rank in expected_fd:
                trial_fd = total_fd + bytes_for_rank
                trial_incremental = (
                    expected_corrected
                    + expected_packet
                    + expected_provider
                    + trial_fd
                    + max(
                        expected_chunk + expected_corrected_reducer,
                        max(
                            byte_count
                            for workspace_rank, byte_count in expected_fd_workspace
                            if workspace_rank in (*expected_admitted, rank)
                        ),
                    )
                )
                if reserved + trial_incremental <= ceiling and trial_incremental <= free:
                    expected_admitted.append(rank)
                    total_fd = trial_fd
                else:
                    # Legacy/reference mode preserves the historical co-live
                    # prefix semantics for replay comparison.
                    break
            expected_live_ranks = tuple(expected_admitted)
            expected_exposed_ranks = tuple(expected_admitted)
            expected_ceiling_rank = (
                max(expected_admitted) if expected_admitted else None
            )
            expected_max_fd_workspace = max(
                (byte_count for _rank, byte_count in expected_fd_workspace),
                default=0,
            )
            actual_fd_workspace = max(
                (
                    byte_count
                    for rank, byte_count in expected_fd_workspace
                    if rank in expected_admitted
                ),
                default=0,
            )
            expected_lower_metadata = False
            expected_independent = True
        else:
            for rank, bytes_for_rank in expected_fd:
                workspace = fd_workspace_map[rank]
                trial_incremental = (
                    expected_corrected
                    + expected_packet
                    + expected_provider
                    + bytes_for_rank
                    + max(expected_chunk + expected_corrected_reducer, workspace)
                )
                if reserved + trial_incremental <= ceiling and trial_incremental <= free:
                    expected_admitted.append(rank)
                else:
                    # Canonical mode admits a nested prefix as *metadata* but
                    # keeps only the largest admitted ceiling state live.
                    break
            expected_exposed_ranks = tuple(expected_admitted)
            expected_ceiling_rank = (
                max(expected_exposed_ranks) if expected_exposed_ranks else None
            )
            expected_live_ranks = (
                () if expected_ceiling_rank is None else (expected_ceiling_rank,)
            )
            total_fd = 0 if expected_ceiling_rank is None else fd_bytes_map[expected_ceiling_rank]
            actual_fd_workspace = (
                0
                if expected_ceiling_rank is None
                else fd_workspace_map[expected_ceiling_rank]
            )
            expected_max_fd_workspace = actual_fd_workspace
            expected_lower_metadata = True
            expected_independent = False
        expected_incremental = (
            expected_corrected
            + expected_packet
            + expected_provider
            + total_fd
            + max(expected_chunk + expected_corrected_reducer, actual_fd_workspace)
        )
        expected_projected = reserved + expected_incremental
        if self.schema == DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA_V2:
            observed_ceiling_rank = expected_ceiling_rank
            observed_live_ranks = expected_live_ranks
            observed_exposed_ranks = expected_exposed_ranks
            observed_lower_metadata = expected_lower_metadata
            observed_independent = expected_independent
            observed_zero_padding = False
            observed_rank_stability = False
        else:
            observed_ceiling_rank = (
                None
                if self.ceiling_fd_rank is None
                else _positive_int(self.ceiling_fd_rank, name="ceiling FD rank")
            )
            observed_live_ranks = tuple(
                _positive_int(rank, name="live FD accumulator rank")
                for rank in self.live_fd_accumulator_ranks
            )
            observed_exposed_ranks = tuple(
                _positive_int(rank, name="exposed nested FD rank")
                for rank in self.exposed_nested_fd_ranks
            )
            observed_lower_metadata = _exact_bool(
                self.lower_rank_views_are_metadata_only,
                name="lower-rank metadata-only flag",
            )
            observed_independent = _exact_bool(
                self.independent_rank_accumulators_required,
                name="independent-rank-accumulator flag",
            )
            observed_zero_padding = _exact_bool(
                self.lower_rank_zero_padding_authority,
                name="lower-rank zero-padding authority flag",
            )
            observed_rank_stability = _exact_bool(
                self.rank_stability_authority,
                name="HBM forecast rank-stability authority flag",
            )
        expected = (
            expected_corrected,
            expected_packet,
            expected_chunk,
            expected_corrected_reducer,
            expected_provider,
            expected_fd,
            expected_fd_workspace,
            expected_max_fd_workspace,
            tuple(expected_admitted),
            total_fd,
            expected_incremental,
            expected_projected,
            ceiling,
            bool(expected_admitted),
            storage,
            expected_ceiling_rank,
            expected_live_ranks,
            expected_exposed_ranks,
            expected_lower_metadata,
            expected_independent,
            False,
            False,
        )
        observed = (
            self.corrected_state_bytes,
            self.active_literal_j_packet_bytes,
            self.colive_dual_chunk_bytes,
            self.corrected_chunk_reducer_workspace_bytes,
            self.provider_peak_workspace_bytes,
            fd_bytes,
            observed_fd_workspace,
            self.maximum_fd_update_workspace_bytes,
            admitted,
            self.total_fd_state_bytes,
            self.incremental_peak_bytes,
            self.projected_peak_reserved_bytes,
            self.hbm_ceiling_bytes,
            self.admitted,
            storage,
            observed_ceiling_rank,
            observed_live_ranks,
            observed_exposed_ranks,
            observed_lower_metadata,
            observed_independent,
            observed_zero_padding,
            observed_rank_stability,
        )
        if observed != expected:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "D-session HBM forecast does not replay its registered formula",
            )
        object.__setattr__(self, "fd_rank_ladder", ladder)
        object.__setattr__(self, "hbm_admitted_fd_ranks", tuple(expected_admitted))
        object.__setattr__(self, "rank_storage_mode", storage)
        object.__setattr__(self, "ceiling_fd_rank", expected_ceiling_rank)
        object.__setattr__(self, "live_fd_accumulator_ranks", expected_live_ranks)
        object.__setattr__(self, "exposed_nested_fd_ranks", expected_exposed_ranks)
        object.__setattr__(
            self,
            "lower_rank_views_are_metadata_only",
            expected_lower_metadata,
        )
        object.__setattr__(
            self,
            "independent_rank_accumulators_required",
            expected_independent,
        )
        object.__setattr__(self, "lower_rank_zero_padding_authority", False)
        object.__setattr__(self, "rank_stability_authority", False)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hbm-")


def forecast_v1_sampled_device_resident_d_session_hbm(
    design: V1SampledDeviceResidentDSessionDesign,
) -> V1SampledDeviceResidentDSessionHBMForecast:
    if type(design) is not V1SampledDeviceResidentDSessionDesign:
        raise TypeError("D-session HBM forecast requires exact design")
    routes = design.route_count
    strata = len(design.fd_design.stratum_names)
    corrected = 4 * strata * routes * _F64_BYTES
    packet = design.maximum_contexts_per_look * routes * _F64_BYTES
    chunk = 2 * design.maximum_contexts_per_look * min(
        design.registered_route_chunk_size,
        routes,
    ) * _F64_BYTES
    corrected_reducer = 3 * design.maximum_contexts_per_look * min(
        design.registered_route_chunk_size,
        routes,
    ) * _F64_BYTES
    provider_workspace = design.provider_peak_workspace_bytes
    fd_by_rank = tuple(
        (rank, strata * routes * _F64_BYTES * (2 * rank + 2))
        for rank in design.fd_rank_ladder
    )
    fd_workspace_by_rank = tuple(
        (
            rank,
            2 * rank * routes * _F64_BYTES
            + (2 * rank) ** 2 * _F64_BYTES,
        )
        for rank in design.fd_rank_ladder
    )
    admitted: list[int] = []
    total_fd = 0
    ceiling = math.floor(design.device_total_memory_bytes * design.hbm_ceiling_fraction)
    if (
        design.rank_storage_mode
        is V1SampledDeviceResidentDRankStorageMode.LEGACY_INDEPENDENT_PREFIX
    ):
        for rank, bytes_for_rank in fd_by_rank:
            trial_fd = total_fd + bytes_for_rank
            trial_workspace = max(
                chunk + corrected_reducer,
                max(
                    byte_count
                    for workspace_rank, byte_count in fd_workspace_by_rank
                    if workspace_rank in (*admitted, rank)
                ),
            )
            trial_incremental = (
                corrected + packet + provider_workspace + trial_fd + trial_workspace
            )
            if (
                design.current_reserved_device_bytes + trial_incremental <= ceiling
                and trial_incremental <= design.device_free_memory_bytes
            ):
                admitted.append(rank)
                total_fd = trial_fd
            else:
                # Preserve a contiguous pre-outcome rank prefix.  Do not silently
                # skip an infeasible rung and admit a later one under a different
                # co-live state set.
                break
        live_fd_accumulator_ranks = tuple(admitted)
        exposed_nested_fd_ranks = tuple(admitted)
        ceiling_fd_rank = max(admitted) if admitted else None
        actual_fd_workspace = max(
            (
                byte_count
                for rank, byte_count in fd_workspace_by_rank
                if rank in admitted
            ),
            default=0,
        )
        maximum_fd_update_workspace_bytes = max(
            (byte_count for _rank, byte_count in fd_workspace_by_rank),
            default=0,
        )
        lower_rank_views_are_metadata_only = False
        independent_rank_accumulators_required = True
    else:
        fd_bytes_map = dict(fd_by_rank)
        fd_workspace_map = dict(fd_workspace_by_rank)
        for rank, bytes_for_rank in fd_by_rank:
            trial_workspace = max(
                chunk + corrected_reducer,
                fd_workspace_map[rank],
            )
            trial_incremental = (
                corrected
                + packet
                + provider_workspace
                + bytes_for_rank
                + trial_workspace
            )
            if (
                design.current_reserved_device_bytes + trial_incremental <= ceiling
                and trial_incremental <= design.device_free_memory_bytes
            ):
                admitted.append(rank)
            else:
                # Lower ranks are metadata views of one ceiling-rank state.
                # Failure of the next dyadic rung fixes the natural ceiling.
                break
        exposed_nested_fd_ranks = tuple(admitted)
        ceiling_fd_rank = max(admitted) if admitted else None
        live_fd_accumulator_ranks = () if ceiling_fd_rank is None else (ceiling_fd_rank,)
        total_fd = 0 if ceiling_fd_rank is None else fd_bytes_map[ceiling_fd_rank]
        actual_fd_workspace = (
            0 if ceiling_fd_rank is None else fd_workspace_map[ceiling_fd_rank]
        )
        maximum_fd_update_workspace_bytes = actual_fd_workspace
        lower_rank_views_are_metadata_only = True
        independent_rank_accumulators_required = False
    incremental = (
        corrected
        + packet
        + provider_workspace
        + total_fd
        + max(chunk + corrected_reducer, actual_fd_workspace)
    )
    return V1SampledDeviceResidentDSessionHBMForecast(
        design_hash=design.design_hash,
        route_count=routes,
        stratum_count=strata,
        maximum_contexts_per_look=design.maximum_contexts_per_look,
        registered_route_chunk_size=design.registered_route_chunk_size,
        fd_rank_ladder=design.fd_rank_ladder,
        hbm_admitted_fd_ranks=tuple(admitted),
        corrected_state_bytes=corrected,
        active_literal_j_packet_bytes=packet,
        colive_dual_chunk_bytes=chunk,
        corrected_chunk_reducer_workspace_bytes=corrected_reducer,
        provider_peak_workspace_bytes=provider_workspace,
        fd_state_bytes_by_rank=fd_by_rank,
        fd_update_workspace_bytes_by_rank=fd_workspace_by_rank,
        maximum_fd_update_workspace_bytes=maximum_fd_update_workspace_bytes,
        total_fd_state_bytes=total_fd,
        incremental_peak_bytes=incremental,
        current_reserved_device_bytes=design.current_reserved_device_bytes,
        projected_peak_reserved_bytes=design.current_reserved_device_bytes + incremental,
        device_total_memory_bytes=design.device_total_memory_bytes,
        device_free_memory_bytes=design.device_free_memory_bytes,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=bool(admitted),
        rank_storage_mode=design.rank_storage_mode,
        ceiling_fd_rank=ceiling_fd_rank,
        live_fd_accumulator_ranks=live_fd_accumulator_ranks,
        exposed_nested_fd_ranks=exposed_nested_fd_ranks,
        lower_rank_views_are_metadata_only=lower_rank_views_are_metadata_only,
        independent_rank_accumulators_required=(
            independent_rank_accumulators_required
        ),
        lower_rank_zero_padding_authority=False,
        rank_stability_authority=False,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDLookReceipt:
    session_design_hash: str
    hbm_forecast_hash: str
    registered_look_index: int
    packet_ordinal: int
    route_axis_hash: str
    route_count: int
    stratum_name: str
    context_axis_hash: str
    draw_axis_hash: str
    source_measurement_axis_hash: str
    same_capture_chunk_source_axis_hash: str
    stratum_counts_before: tuple[int, int]
    stratum_counts_after: tuple[int, int]
    route_windows: tuple[tuple[int, int], ...]
    hbm_admitted_fd_ranks: tuple[int, ...]
    corrected_state_scalar_hash: str
    fd_state_scalar_hash: str
    avoided_legacy_intermediate_route_scale_d2h_bytes: int
    avoided_legacy_intermediate_fd_snapshot_bytes: int
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
    schema: str = DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "session_design_hash",
            "hbm_forecast_hash",
            "route_axis_hash",
            "stratum_name",
            "context_axis_hash",
            "draw_axis_hash",
            "source_measurement_axis_hash",
            "same_capture_chunk_source_axis_hash",
            "corrected_state_scalar_hash",
            "fd_state_scalar_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_look_index",
            _positive_int(self.registered_look_index, name="registered look index"),
        )
        object.__setattr__(
            self,
            "packet_ordinal",
            _nonnegative_int(self.packet_ordinal, name="packet ordinal"),
        )
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        before = tuple(_nonnegative_int(value, name="stratum count before") for value in self.stratum_counts_before)
        after = tuple(_nonnegative_int(value, name="stratum count after") for value in self.stratum_counts_after)
        if len(before) != 2 or len(after) != 2:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "look receipts currently support exactly two stratum counts",
            )
        windows: list[tuple[int, int]] = []
        cursor = 0
        for raw_start, raw_stop in self.route_windows:
            start = _nonnegative_int(raw_start, name="route window start")
            stop = _positive_int(raw_stop, name="route window stop")
            if start != cursor or not start < stop <= self.route_count:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                    "route windows must cover the route axis in order",
                )
            windows.append((start, stop))
            cursor = stop
        if cursor != self.route_count:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "route windows must cover the full route axis",
            )
        object.__setattr__(self, "route_windows", tuple(windows))
        object.__setattr__(self, "stratum_counts_before", before)
        object.__setattr__(self, "stratum_counts_after", after)
        object.__setattr__(
            self,
            "hbm_admitted_fd_ranks",
            tuple(
                _positive_int(rank, name="HBM-admitted FD rank")
                for rank in self.hbm_admitted_fd_ranks
            ),
        )
        for name in (
            "avoided_legacy_intermediate_route_scale_d2h_bytes",
            "avoided_legacy_intermediate_fd_snapshot_bytes",
            "route_scale_d2h_bytes",
            "route_scale_host_tensor_materializations",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        flags = (
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags) or any(item is not False for item in flags):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "device-resident D look receipt crossed its authority boundary",
            )
        if self.route_scale_d2h_bytes != 0 or self.route_scale_host_tensor_materializations != 0:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "intermediate device-resident D look receipt cannot perform route-scale D2H",
            )
        if (
            self.authority_scope != DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
            or self.schema != DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                "unsupported device-resident D look receipt semantics",
            )
        object.__setattr__(self, "_receipt_hash", self.receipt_hash)

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_PREFIX + "look-")

    def assert_integrity(self) -> str:
        observed = self.receipt_hash
        if self._receipt_hash != observed:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_RECEIPT,
                "device-resident D look receipt was mutated after issuance",
            )
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDRankSelectionReceipt:
    session_design_hash: str
    hbm_forecast_hash: str
    hbm_admitted_fd_ranks: tuple[int, ...]
    decision_stable_fd_ranks: tuple[int, ...]
    selected_physical_fd_rank: int
    selection_rule: str = DEVICE_RESIDENT_D_SELECTION_RULE
    c_stream_authority: bool = False
    commit_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_RESIDENT_D_RANK_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.session_design_hash, name="session design hash")
        _identity(self.hbm_forecast_hash, name="HBM forecast hash")
        admitted = tuple(_positive_int(rank, name="admitted FD rank") for rank in self.hbm_admitted_fd_ranks)
        stable = tuple(_positive_int(rank, name="decision-stable FD rank") for rank in self.decision_stable_fd_ranks)
        selected = _positive_int(self.selected_physical_fd_rank, name="selected physical FD rank")
        if not stable or selected != min(stable) or any(rank not in admitted for rank in stable):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "rank selection must choose the smallest stable pre-admitted physical rank",
            )
        if (
            self.selection_rule != DEVICE_RESIDENT_D_SELECTION_RULE
            or self.schema != DEVICE_RESIDENT_D_RANK_SELECTION_SCHEMA
            or any(
                _exact_bool(flag, name="rank-selection authority flag")
                for flag in (
                    self.c_stream_authority,
                    self.commit_authority,
                    self.pareto_authority,
                )
            )
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "rank selection receipt crossed its authority boundary",
            )
        object.__setattr__(self, "hbm_admitted_fd_ranks", admitted)
        object.__setattr__(self, "decision_stable_fd_ranks", stable)
        object.__setattr__(self, "selected_physical_fd_rank", selected)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "rank-selection-")


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDMaterializationRankReceipt:
    """Authority-inert choice of the one portable FD snapshot to expose.

    This receipt is deliberately *not* a D decision-stability witness.  It
    merely binds the source-registered proposal rank to the highest rank that
    the frozen HBM forecast admitted.  The portable V7 F/Q/KKT construction is
    still responsible for issuing or refusing the actual adaptive-rank face
    stability witness downstream.
    """

    session_design_hash: str
    hbm_forecast_hash: str
    hbm_admitted_fd_ranks: tuple[int, ...]
    registered_proposal_fd_rank: int
    source_rank_registration_hash: str
    materialization_rank_rule: str = DEVICE_RESIDENT_D_MATERIALIZATION_RANK_RULE
    materialization_only: bool = True
    rank_stability_authority: bool = False
    c_stream_authority: bool = False
    commit_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_RESIDENT_D_MATERIALIZATION_RANK_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.session_design_hash, name="session design hash")
        _identity(self.hbm_forecast_hash, name="HBM forecast hash")
        _identity(
            self.source_rank_registration_hash,
            name="source rank registration hash",
        )
        admitted = tuple(
            _positive_int(rank, name="admitted FD rank")
            for rank in self.hbm_admitted_fd_ranks
        )
        registered = _positive_int(
            self.registered_proposal_fd_rank,
            name="registered proposal FD rank",
        )
        if not admitted or registered != max(admitted):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "portable compatibility materialization must use the source-"
                "registered highest HBM-admitted proposal rank",
            )
        flags = (
            self.materialization_only,
            self.rank_stability_authority,
            self.c_stream_authority,
            self.commit_authority,
            self.pareto_authority,
        )
        if (
            self.materialization_rank_rule
            != DEVICE_RESIDENT_D_MATERIALIZATION_RANK_RULE
            or self.schema != DEVICE_RESIDENT_D_MATERIALIZATION_RANK_SCHEMA
            or any(type(flag) is not bool for flag in flags)
            or flags != (True, False, False, False, False)
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "materialization-rank receipt cannot claim rank-stability or "
                "downstream authority",
            )
        object.__setattr__(self, "hbm_admitted_fd_ranks", admitted)
        object.__setattr__(self, "registered_proposal_fd_rank", registered)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "materialization-rank-")


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDReplayCoordinateReceipt:
    session_design_hash: str
    hbm_forecast_hash: str
    sealed_model_context_ledger_hash: str
    route_axis_hash: str
    consumed_look_receipt_hashes: tuple[str, ...]
    source_packet_axis_hashes: tuple[str, ...]
    next_registered_look_index: int
    replay_coordinate_only: bool = True
    restart_identity_authority: bool = False
    schema: str = DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "session_design_hash",
            "hbm_forecast_hash",
            "sealed_model_context_ledger_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "consumed_look_receipt_hashes",
            _identity_axis(
                self.consumed_look_receipt_hashes,
                name="consumed look receipt hashes",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "source_packet_axis_hashes",
            _identity_axis(
                self.source_packet_axis_hashes,
                name="source packet axis hashes",
                unique=False,
            ),
        )
        object.__setattr__(
            self,
            "next_registered_look_index",
            _positive_int(self.next_registered_look_index, name="next registered look index"),
        )
        if (
            self.replay_coordinate_only is not True
            or self.restart_identity_authority is not False
            or self.schema != DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "replay coordinate cannot claim restart identity",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "replay-coordinate-")


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDRowFreeFinalizationReceipt:
    """Scalar endpoint identity for a live rank with no route-scale D2H."""

    session_design_hash: str
    hbm_forecast_hash: str
    materialization_rank_receipt_hash: str
    registered_proposal_fd_rank: int
    final_logical_look_index: int
    route_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    stratum_counts: tuple[int, ...]
    context_axis_hashes_by_stratum: tuple[str, ...]
    draw_axis_hashes_by_stratum: tuple[str, ...]
    source_measurement_axis_hashes_by_stratum: tuple[str, ...]
    look_receipt_axis_hash: str
    source_packet_axis_hash: str
    corrected_reacquisition_source_hash: str
    live_fd_source_hash: str
    row_free_finalized_once: bool = True
    portable_corrected_moments_present: bool = False
    portable_fd_snapshot_present: bool = False
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    live_device_owner_required: bool = True
    rank_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
    schema: str = DEVICE_RESIDENT_D_ROW_FREE_FINALIZATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ROW_FREE_FINALIZATION_TOKEN:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "row-free finalization receipts are live-session issued",
            )
        for name in (
            "session_design_hash",
            "hbm_forecast_hash",
            "materialization_rank_receipt_hash",
            "route_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "look_receipt_axis_hash",
            "source_packet_axis_hash",
            "corrected_reacquisition_source_hash",
            "live_fd_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_proposal_fd_rank",
            _positive_int(self.registered_proposal_fd_rank, name="registered FD rank"),
        )
        object.__setattr__(
            self,
            "final_logical_look_index",
            _positive_int(self.final_logical_look_index, name="final logical look index"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        counts = tuple(
            _positive_int(value, name="final stratum count")
            for value in self.stratum_counts
        )
        axes = tuple(
            _identity_axis(axis, name=name, unique=True)
            for name, axis in (
                ("context axes by stratum", self.context_axis_hashes_by_stratum),
                ("draw axes by stratum", self.draw_axis_hashes_by_stratum),
                (
                    "source-measurement axes by stratum",
                    self.source_measurement_axis_hashes_by_stratum,
                ),
            )
        )
        flags = (
            self.row_free_finalized_once,
            self.portable_corrected_moments_present,
            self.portable_fd_snapshot_present,
            self.live_device_owner_required,
            self.rank_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            not counts
            or any(len(axis) != len(counts) for axis in axes)
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or flags[:4] != (True, False, False, True)
            or any(type(value) is not bool for value in flags)
            or any(flags[4:])
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_host_tensor_materializations != 0
            or self.authority_scope != DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
            or self.schema != DEVICE_RESIDENT_D_ROW_FREE_FINALIZATION_SCHEMA
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "row-free finalization retained arrays, copied routes, or gained authority",
            )
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "context_axis_hashes_by_stratum", axes[0])
        object.__setattr__(self, "draw_axis_hashes_by_stratum", axes[1])
        object.__setattr__(self, "source_measurement_axis_hashes_by_stratum", axes[2])
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "row-free-finalization-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "row-free finalization receipt changed after live-session issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDKKTGeometryReceipt:
    """Compact source binding for the live device proposal geometry."""

    session_design_hash: str
    hbm_forecast_hash: str
    row_free_finalization_receipt_hash: str
    materialization_rank_receipt_hash: str
    stratified_fd_device_receipt_hash: str
    registered_proposal_fd_rank: int
    final_logical_look_index: int
    route_axis_hash: str
    route_count: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    stratum_counts: tuple[int, ...]
    context_axis_hashes_by_stratum: tuple[str, ...]
    draw_axis_hashes_by_stratum: tuple[str, ...]
    source_measurement_axis_hashes_by_stratum: tuple[str, ...]
    device_identity: str
    singleton_work_shape: tuple[int, ...]
    factor_shape: tuple[int, int]
    residual_diagonal_shape: tuple[int, ...]
    response_rows: int
    retained_centered_rank: int
    residual_spectral_bound: float
    factor_frobenius_norm: float
    residual_sqrt_l2_norm: float
    singleton_work_semantics: str = DEVICE_RESIDENT_D_SINGLETON_WORK_SEMANTICS
    factor_semantics: str = DEVICE_RESIDENT_D_KKT_FACTOR_SEMANTICS
    dtype_identity: str = "torch.float64"
    route_scale_d2h_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    source_d_route_tensors_released: bool = True
    device_tensor_roles: tuple[str, ...] = (
        "singleton_work",
        "factor",
        "residual_diagonal",
    )
    kkt_proposal_input_authority: bool = True
    native_fiber_signed_f_authority: bool = False
    rank_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
    schema: str = DEVICE_RESIDENT_D_KKT_GEOMETRY_RECEIPT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str = field(init=False, default="", repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _KKT_GEOMETRY_ISSUANCE_TOKEN:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "KKT geometry receipts are live-session issued only",
            )
        for name in (
            "session_design_hash",
            "hbm_forecast_hash",
            "row_free_finalization_receipt_hash",
            "materialization_rank_receipt_hash",
            "stratified_fd_device_receipt_hash",
            "route_axis_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "device_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _positive_int(
            self.registered_proposal_fd_rank,
            name="registered proposal FD rank",
        )
        look = _positive_int(self.final_logical_look_index, name="final logical look")
        routes = _positive_int(self.route_count, name="route count")
        names = _identity_axis(self.stratum_names, name="KKT geometry strata")
        weights = tuple(float(value) for value in self.stratum_weights)
        counts = tuple(
            _positive_int(value, name="KKT geometry stratum count")
            for value in self.stratum_counts
        )
        context_axes = _identity_axis(
            self.context_axis_hashes_by_stratum,
            name="KKT geometry context-axis hashes",
        )
        draw_axes = _identity_axis(
            self.draw_axis_hashes_by_stratum,
            name="KKT geometry draw-axis hashes",
        )
        measurement_axes = _identity_axis(
            self.source_measurement_axis_hashes_by_stratum,
            name="KKT geometry source-measurement-axis hashes",
        )
        if (
            not (len(names) == len(weights) == len(counts))
            or len(context_axes) != len(names)
            or len(draw_axes) != len(names)
            or len(measurement_axes) != len(names)
            or any(not math.isfinite(value) or value <= 0.0 for value in weights)
            or not math.isclose(
                math.fsum(weights),
                1.0,
                rel_tol=0.0,
                abs_tol=128.0 * math.ulp(1.0),
            )
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "KKT geometry stratum identities do not align",
            )
        response_rows = _positive_int(self.response_rows, name="response rows")
        retained = _nonnegative_int(
            self.retained_centered_rank,
            name="retained centered rank",
        )
        residual = float(self.residual_spectral_bound)
        factor_norm = float(self.factor_frobenius_norm)
        residual_sqrt_norm = float(self.residual_sqrt_l2_norm)
        flags = (
            self.kkt_proposal_input_authority,
            self.native_fiber_signed_f_authority,
            self.rank_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            tuple(self.singleton_work_shape) != (routes,)
            or tuple(self.factor_shape) != (response_rows + retained, routes)
            or tuple(self.residual_diagonal_shape) != (routes,)
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(factor_norm)
            or factor_norm < 0.0
            or not math.isfinite(residual_sqrt_norm)
            or residual_sqrt_norm < 0.0
            or self.corrected_a_stream_identity_hash
            == self.literal_j_stream_identity_hash
            or self.singleton_work_semantics
            != DEVICE_RESIDENT_D_SINGLETON_WORK_SEMANTICS
            or self.factor_semantics != DEVICE_RESIDENT_D_KKT_FACTOR_SEMANTICS
            or self.dtype_identity != "torch.float64"
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.source_d_route_tensors_released is not True
            or self.device_tensor_roles
            != ("singleton_work", "factor", "residual_diagonal")
            or any(type(flag) is not bool for flag in flags)
            or flags != (True, False, False, False, False, False, False, False, False, False)
            or self.authority_scope != DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
            or self.schema != DEVICE_RESIDENT_D_KKT_GEOMETRY_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "KKT geometry receipt crossed its tensor or authority boundary",
            )
        object.__setattr__(self, "registered_proposal_fd_rank", rank)
        object.__setattr__(self, "final_logical_look_index", look)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "stratum_counts", counts)
        object.__setattr__(self, "context_axis_hashes_by_stratum", context_axes)
        object.__setattr__(self, "draw_axis_hashes_by_stratum", draw_axes)
        object.__setattr__(self, "source_measurement_axis_hashes_by_stratum", measurement_axes)
        object.__setattr__(self, "response_rows", response_rows)
        object.__setattr__(self, "retained_centered_rank", retained)
        object.__setattr__(self, "residual_spectral_bound", residual)
        object.__setattr__(self, "factor_frobenius_norm", factor_norm)
        object.__setattr__(self, "residual_sqrt_l2_norm", residual_sqrt_norm)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if not item.name.startswith("_")
        }
        return content_hash(payload, prefix=_PREFIX + "KKT-geometry-receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_RECEIPT,
                "KKT geometry receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1SampledDeviceResidentDKKTGeometryLease:
    """Narrow zero-copy borrow of the three KKT route tensors."""

    __slots__ = ("_handle", "_invariants", "_closed")

    def __init__(
        self,
        handle: "V1SampledDeviceResidentDKKTGeometryDeviceHandle",
    ) -> None:
        self._handle = handle
        self._invariants = handle._acquire_lease()
        self._closed = False

    def _tensor(self, role: str) -> Any:
        if self._closed:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.EVICTED,
                "KKT geometry lease is closed",
            )
        return self._handle._leased_tensor(role, self._invariants)

    @property
    def singleton_work(self) -> Any:
        return self._tensor("singleton_work")

    @property
    def factor(self) -> Any:
        return self._tensor("factor")

    @property
    def residual_diagonal(self) -> Any:
        return self._tensor("residual_diagonal")

    @property
    def residual_spectral_bound(self) -> float:
        if self._closed:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.EVICTED,
                "KKT geometry lease is closed",
            )
        self._handle.assert_live_identity()
        return self._handle.receipt.residual_spectral_bound

    @property
    def receipt(self) -> V1SampledDeviceResidentDKKTGeometryReceipt:
        if self._closed:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.EVICTED,
                "KKT geometry lease is closed",
            )
        return self._handle.receipt

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle._release_lease(self._invariants)
        finally:
            self._closed = True

    def __enter__(self) -> "V1SampledDeviceResidentDKKTGeometryLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("KKT geometry leases are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("KKT geometry leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("KKT geometry leases cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("KKT geometry leases cannot be copied")


class V1SampledDeviceResidentDKKTGeometryDeviceHandle:
    """Issuer-only process-local owner of the frozen D-to-KKT geometry."""

    __slots__ = (
        "_receipt",
        "_tensors",
        "_invariants",
        "_active_lease_count",
        "_evicted",
        "_integrity_failed",
    )

    def __init__(
        self,
        *,
        receipt: V1SampledDeviceResidentDKKTGeometryReceipt,
        tensors: Mapping[str, Any],
        _issuance_token: object,
    ) -> None:
        if (
            _issuance_token is not _KKT_GEOMETRY_ISSUANCE_TOKEN
            or type(receipt) is not V1SampledDeviceResidentDKKTGeometryReceipt
        ):
            raise PermissionError("KKT geometry handles are live-session issued only")
        expected_roles = {"singleton_work", "factor", "residual_diagonal"}
        values = dict(tensors)
        if set(values) != expected_roles or torch is None or any(
            not torch.is_tensor(value) for value in values.values()
        ):
            raise TypeError("KKT geometry handle tensor roles are incomplete")
        receipt.assert_integrity()
        singleton = values["singleton_work"]
        factor = values["factor"]
        residual = values["residual_diagonal"]
        if (
            any(value.dtype != torch.float64 for value in values.values())
            or any(value.requires_grad for value in values.values())
            or any(not value.is_contiguous() for value in values.values())
            or len({str(value.device) for value in values.values()}) != 1
            or str(singleton.device) != receipt.device_identity
            or tuple(singleton.shape) != receipt.singleton_work_shape
            or tuple(factor.shape) != receipt.factor_shape
            or tuple(residual.shape) != receipt.residual_diagonal_shape
            or not bool(torch.isfinite(singleton).all().item())
            or not bool(torch.isfinite(factor).all().item())
            or not bool(torch.isfinite(residual).all().item())
            or bool(torch.any(singleton < 0.0).item())
            or bool(torch.any(residual < 0.0).item())
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                "KKT geometry tensors differ from their receipt",
            )
        factor_norm = float(torch.linalg.vector_norm(factor).item())
        residual_sqrt_norm = math.sqrt(float(torch.sum(residual).item()))
        if (
            not math.isfinite(factor_norm)
            or not math.isfinite(residual_sqrt_norm)
            or factor_norm != receipt.factor_frobenius_norm
            or residual_sqrt_norm != receipt.residual_sqrt_l2_norm
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                "KKT geometry intrinsic norm commitments do not replay",
            )
        self._receipt = receipt
        self._tensors = values
        self._invariants = {
            role: _tensor_storage_invariant(value)
            for role, value in values.items()
        }
        self._active_lease_count = 0
        self._evicted = False
        self._integrity_failed = False
        self.assert_live_identity()

    @property
    def receipt(self) -> V1SampledDeviceResidentDKKTGeometryReceipt:
        self.assert_live_identity()
        return self._receipt

    @property
    def active_lease_count(self) -> int:
        return self._active_lease_count

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live_identity(self) -> str:
        if self._evicted:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.EVICTED,
                "KKT geometry handle was evicted",
            )
        if self._integrity_failed:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                "KKT geometry handle previously detected mutation",
            )
        self._receipt.assert_integrity()
        for role, tensor in self._tensors.items():
            if _tensor_storage_invariant(tensor) != self._invariants[role]:
                self._integrity_failed = True
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                    f"KKT geometry {role} storage or version changed",
                )
        return content_hash(
            {
                "receipt_hash": self._receipt.receipt_hash,
                "storage_invariants": tuple(
                    (role, self._invariants[role]) for role in sorted(self._invariants)
                ),
                "process_local_nonserializable": True,
            },
            prefix=_PREFIX + "KKT-geometry-live-identity-",
        )

    def borrow_for_kkt(
        self,
        *,
        expected_geometry_receipt_hash: str,
        expected_session_design_hash: str,
        expected_row_free_finalization_receipt_hash: str,
        expected_materialization_rank_receipt_hash: str,
        expected_registered_proposal_fd_rank: int,
        expected_route_axis_hash: str,
        expected_device: Any,
    ) -> V1SampledDeviceResidentDKKTGeometryLease:
        receipt = self.receipt
        expected_pairs = (
            (
                _identity(expected_geometry_receipt_hash, name="expected geometry receipt hash"),
                receipt.receipt_hash,
            ),
            (
                _identity(expected_session_design_hash, name="expected session design hash"),
                receipt.session_design_hash,
            ),
            (
                _identity(
                    expected_row_free_finalization_receipt_hash,
                    name="expected row-free finalization receipt hash",
                ),
                receipt.row_free_finalization_receipt_hash,
            ),
            (
                _identity(
                    expected_materialization_rank_receipt_hash,
                    name="expected materialization-rank receipt hash",
                ),
                receipt.materialization_rank_receipt_hash,
            ),
        )
        if any(observed != registered for observed, registered in expected_pairs):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "KKT geometry borrow requested stale source identity",
            )
        if (
            _positive_int(
                expected_registered_proposal_fd_rank,
                name="expected registered proposal FD rank",
            )
            != receipt.registered_proposal_fd_rank
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "KKT geometry borrow requested a stale FD rank",
            )
        if (
            _identity(expected_route_axis_hash, name="expected route axis hash")
            != receipt.route_axis_hash
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "KKT geometry borrow requested a stale route axis",
            )
        if str(_torch_device(expected_device)) != receipt.device_identity:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.DEVICE_MISMATCH,
                "KKT geometry borrow requested a different device",
            )
        return V1SampledDeviceResidentDKKTGeometryLease(self)

    def _acquire_lease(self) -> dict[str, tuple[object, ...]]:
        self.assert_live_identity()
        self._active_lease_count += 1
        return dict(self._invariants)

    def _leased_tensor(
        self,
        role: str,
        invariants: Mapping[str, tuple[object, ...]],
    ) -> Any:
        self.assert_live_identity()
        if dict(invariants) != self._invariants or role not in self._tensors:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "KKT geometry lease has stale storage identity",
            )
        return self._tensors[role]

    def _release_lease(
        self,
        invariants: Mapping[str, tuple[object, ...]],
    ) -> None:
        if self._active_lease_count <= 0:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LEASE,
                "KKT geometry lease accounting underflow",
            )
        try:
            if dict(invariants) != self._invariants:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                    "KKT geometry lease identity changed",
                )
            self.assert_live_identity()
        finally:
            self._active_lease_count -= 1

    def evict(self) -> None:
        if self._evicted:
            return
        if self._active_lease_count:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LEASE,
                "KKT geometry cannot be evicted while leased",
            )
        error: V1SampledDeviceResidentDSessionRefusal | None = None
        try:
            self.assert_live_identity()
        except V1SampledDeviceResidentDSessionRefusal as exc:
            # Eviction still closes custody whose source/receipt mutation was
            # already detected; unrelated control-flow exceptions propagate.
            error = exc
        self._tensors.clear()
        self._invariants.clear()
        self._evicted = True
        if error is not None:
            raise error

    def __reduce__(self) -> NoReturn:
        raise TypeError("KKT geometry handles are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("KKT geometry handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("KKT geometry handles cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("KKT geometry handles cannot be copied")


@dataclass(frozen=True, slots=True)
class V1SampledDeviceResidentDSessionMaterialization:
    session_design_hash: str
    hbm_forecast_hash: str
    final_look_count: int
    route_axis_hash: str
    corrected_attribution_moments: V1StratifiedCorrectedAttributionMoments
    fd_snapshots_by_rank: tuple[tuple[int, V1StratifiedFDAccumulatorSnapshot], ...]
    final_route_scale_d2h_bytes: int
    materialization_mode: str = DEVICE_RESIDENT_D_SELECTED_MATERIALIZATION_MODE
    selected_physical_fd_rank: int | None = None
    selected_rank_selection_receipt_hash: str | None = None
    materialization_rank_receipt_hash: str | None = None
    rank_stability_authority: bool = False
    unselected_rank_snapshot_count: int = 0
    unselected_rank_route_scale_d2h_bytes: int = 0
    materialized_once: bool = True
    route_scale_d2h_only_at_final: bool = True
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
    schema: str = DEVICE_RESIDENT_D_MATERIALIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("session_design_hash", "hbm_forecast_hash", "route_axis_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "final_look_count", _positive_int(self.final_look_count, name="final look count"))
        if type(self.corrected_attribution_moments) is not V1StratifiedCorrectedAttributionMoments:
            raise TypeError("materialization requires exact corrected-attribution moments")
        self.corrected_attribution_moments.assert_integrity()
        snapshots = tuple(self.fd_snapshots_by_rank)
        if (
            not snapshots
            or tuple(rank for rank, _snapshot in snapshots)
            != tuple(sorted({rank for rank, _snapshot in snapshots}))
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "FD snapshots must cover a unique increasing rank axis",
            )
        for rank, snapshot in snapshots:
            _positive_int(rank, name="FD snapshot rank")
            if type(snapshot) is not V1StratifiedFDAccumulatorSnapshot:
                raise TypeError("materialization requires exact FD snapshots")
            snapshot.assert_integrity()
        object.__setattr__(self, "fd_snapshots_by_rank", snapshots)
        object.__setattr__(
            self,
            "final_route_scale_d2h_bytes",
            _nonnegative_int(self.final_route_scale_d2h_bytes, name="final route-scale D2H bytes"),
        )
        object.__setattr__(
            self,
            "unselected_rank_snapshot_count",
            _nonnegative_int(
                self.unselected_rank_snapshot_count,
                name="unselected-rank snapshot count",
            ),
        )
        object.__setattr__(
            self,
            "unselected_rank_route_scale_d2h_bytes",
            _nonnegative_int(
                self.unselected_rank_route_scale_d2h_bytes,
                name="unselected-rank route-scale D2H bytes",
            ),
        )
        if self.materialization_mode == DEVICE_RESIDENT_D_SELECTED_MATERIALIZATION_MODE:
            if self.selected_physical_fd_rank is None:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "selected-rank materialization requires a rank-selection receipt",
                )
            selected = _positive_int(
                self.selected_physical_fd_rank,
                name="selected physical FD rank",
            )
            if tuple(rank for rank, _snapshot in snapshots) != (selected,):
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "selected-rank materialization may snapshot only the selected FD rank",
                )
            object.__setattr__(self, "selected_physical_fd_rank", selected)
            object.__setattr__(
                self,
                "selected_rank_selection_receipt_hash",
                _identity(
                    self.selected_rank_selection_receipt_hash,
                    name="selected rank-selection receipt hash",
                ),
            )
            if self.materialization_rank_receipt_hash is not None:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "decision-stable selected-rank materialization cannot carry "
                    "an authority-inert materialization-rank receipt",
                )
            if (
                self.unselected_rank_snapshot_count != 0
                or self.unselected_rank_route_scale_d2h_bytes != 0
            ):
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "selected-rank production materialization must not snapshot unselected ranks",
                )
        elif (
            self.materialization_mode
            == DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE
        ):
            if self.selected_physical_fd_rank is None:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "registered-rank materialization requires one physical FD rank",
                )
            selected = _positive_int(
                self.selected_physical_fd_rank,
                name="registered materialization physical FD rank",
            )
            if tuple(rank for rank, _snapshot in snapshots) != (selected,):
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "registered-rank materialization may snapshot only its named rank",
                )
            object.__setattr__(self, "selected_physical_fd_rank", selected)
            if self.selected_rank_selection_receipt_hash is not None:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "authority-inert registered-rank materialization cannot carry a "
                    "decision-stability receipt",
                )
            object.__setattr__(
                self,
                "materialization_rank_receipt_hash",
                _identity(
                    self.materialization_rank_receipt_hash,
                    name="materialization-rank receipt hash",
                ),
            )
            if (
                self.unselected_rank_snapshot_count != 0
                or self.unselected_rank_route_scale_d2h_bytes != 0
            ):
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "registered-rank materialization must not snapshot other ranks",
                )
        elif self.materialization_mode == DEVICE_RESIDENT_D_DIAGNOSTIC_MATERIALIZATION_MODE:
            if (
                self.selected_physical_fd_rank is not None
                or self.selected_rank_selection_receipt_hash is not None
                or self.materialization_rank_receipt_hash is not None
            ):
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                    "diagnostic all-rank materialization cannot carry a selected-rank receipt",
                )
        else:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported device-resident D materialization mode",
            )
        flags = (
            self.materialized_once,
            self.route_scale_d2h_only_at_final,
            self.rank_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            flags[:2] != (True, True)
            or any(type(item) is not bool for item in flags)
            or any(item is not False for item in flags[2:])
            or self.authority_scope != DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE
            or self.schema != DEVICE_RESIDENT_D_MATERIALIZATION_SCHEMA
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "D-session materialization crossed its authority boundary",
            )

    @property
    def materialization_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "session_design_hash": self.session_design_hash,
                "hbm_forecast_hash": self.hbm_forecast_hash,
                "final_look_count": self.final_look_count,
                "route_axis_hash": self.route_axis_hash,
                "corrected_attribution_moments_hash": (
                    self.corrected_attribution_moments.state_hash
                ),
                "fd_snapshot_hashes_by_rank": tuple(
                    (rank, snapshot.snapshot_hash)
                    for rank, snapshot in self.fd_snapshots_by_rank
                ),
                "final_route_scale_d2h_bytes": self.final_route_scale_d2h_bytes,
                "materialization_mode": self.materialization_mode,
                "selected_physical_fd_rank": self.selected_physical_fd_rank,
                "selected_rank_selection_receipt_hash": (
                    self.selected_rank_selection_receipt_hash
                ),
                "materialization_rank_receipt_hash": (
                    self.materialization_rank_receipt_hash
                ),
                "rank_stability_authority": self.rank_stability_authority,
                "unselected_rank_snapshot_count": self.unselected_rank_snapshot_count,
                "unselected_rank_route_scale_d2h_bytes": (
                    self.unselected_rank_route_scale_d2h_bytes
                ),
                "all_downstream_authority": False,
                "authority_scope": self.authority_scope,
            },
            prefix=_PREFIX + "materialization-",
        )


@dataclass(slots=True)
class _ActiveLookPacket:
    registered_look_index: int
    packet_ordinal: int
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_name: str
    stratum_index: int
    source_measurement_hashes: tuple[str, ...]
    sealed_order_indices: tuple[int, ...]
    stratum_counts_before: tuple[int, int]
    literal_j_full: Any
    route_cursor: int = 0
    route_windows: list[tuple[int, int]] = field(default_factory=list)
    chunk_source_hashes: list[str] = field(default_factory=list)


class V1SampledDeviceResidentDSession:
    """One nonserializable live sampled-D owner spanning registered look packets."""

    def __init__(
        self,
        design: V1SampledDeviceResidentDSessionDesign,
        *,
        device: Any | None = None,
    ) -> None:
        if type(design) is not V1SampledDeviceResidentDSessionDesign:
            raise TypeError("device-resident D session requires exact design")
        if torch is None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.DEVICE_REQUIRED,
                "device-resident D session requires PyTorch",
            )
        runtime_device = _torch_device("cpu" if device is None else device)
        if design.execution_mode is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION:
            if runtime_device.type != "cuda":
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production sampled-D session refuses CPU fallback",
                )
            if not torch.cuda.is_available():
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.DEVICE_REQUIRED,
                    "production sampled-D session requires available CUDA",
                )
        elif design.execution_mode is V1SampledDeviceResidentDSessionMode.CPU_REFERENCE_TEST_ONLY:
            if runtime_device.type != "cpu":
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.INVALID_DESIGN,
                    "CPU reference sampled-D session must run on CPU",
                )
        else:  # pragma: no cover - exact enum validation is in the dataclass.
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.INVALID_TYPE,
                "invalid sampled-D execution mode",
            )

        forecast = forecast_v1_sampled_device_resident_d_session_hbm(design)
        if not forecast.admitted:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.HBM_FORECAST_EXCEEDED,
                "no FD rank is pre-outcome HBM-admitted for this D session",
            )
        self.design = design
        self.device = runtime_device
        self.hbm_forecast = forecast
        strata = len(design.fd_design.stratum_names)
        routes = design.route_count
        self._signed_mean = torch.zeros((strata, routes), dtype=torch.float64, device=runtime_device)
        self._signed_m2 = torch.zeros_like(self._signed_mean)
        self._absolute_mean = torch.zeros_like(self._signed_mean)
        self._absolute_m2 = torch.zeros_like(self._signed_mean)
        self._stratum_counts = [0 for _ in range(strata)]
        self._context_ledgers = [[] for _ in range(strata)]
        self._draw_ledgers = [[] for _ in range(strata)]
        self._source_measurements = [[] for _ in range(strata)]
        self._fd_accumulators = {
            rank: V1StratifiedDeviceFDTrustAccumulator(
                design.fd_design,
                sketch_rank=rank,
                device=runtime_device,
                emit_derivation_receipt=False,
            )
            for rank in forecast.live_fd_accumulator_ranks
        }
        self._look_receipts: list[V1SampledDeviceResidentDLookReceipt] = []
        self._source_packet_axis_hashes: list[str] = []
        self._active: _ActiveLookPacket | None = None
        self._evicted = False
        self._materialized = False
        self._row_free_finalized = False
        self._row_free_source_invariants: tuple[tuple[str, object], ...] | None = None
        self._kkt_geometry_freeze_attempted = False
        self._kkt_geometry_handle: (
            V1SampledDeviceResidentDKKTGeometryDeviceHandle | None
        ) = None

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("device-resident sampled-D sessions are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device-resident sampled-D sessions cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("device-resident sampled-D sessions cannot be copied")

    @property
    def look_receipts(self) -> tuple[V1SampledDeviceResidentDLookReceipt, ...]:
        return tuple(self._look_receipts)

    @property
    def hbm_admitted_fd_ranks(self) -> tuple[int, ...]:
        return self.hbm_forecast.hbm_admitted_fd_ranks

    @property
    def live_fd_accumulator_ranks(self) -> tuple[int, ...]:
        return self.hbm_forecast.live_fd_accumulator_ranks

    @property
    def exposed_nested_fd_ranks(self) -> tuple[int, ...]:
        return self.hbm_forecast.exposed_nested_fd_ranks

    @property
    def ceiling_fd_rank(self) -> int | None:
        return self.hbm_forecast.ceiling_fd_rank

    @property
    def rank_storage_mode(self) -> V1SampledDeviceResidentDRankStorageMode:
        return self.hbm_forecast.rank_storage_mode

    @property
    def stratum_counts(self) -> tuple[int, int]:
        return tuple(self._stratum_counts)  # type: ignore[return-value]

    def _assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.EVICTED,
                "device-resident D session has been evicted",
            )
        if self._materialized:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MATERIALIZED,
                "device-resident D session has already materialized portable state",
            )
        self.design.fd_design.assert_integrity()

    def _registered_rank_source_invariants(
        self,
        rank: int,
    ) -> tuple[tuple[str, object], ...]:
        """Return process-local mutation guards without hashing route values."""

        entries: list[tuple[str, object]] = []
        for role in (
            "_signed_mean",
            "_signed_m2",
            "_absolute_mean",
            "_absolute_m2",
        ):
            entries.append((role, _tensor_storage_invariant(getattr(self, role))))
        owner = self._fd_accumulators.get(rank)
        if type(owner) is not V1StratifiedDeviceFDTrustAccumulator:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "registered FD rank has no live stratified owner",
            )
        entries.append(
            (
                "fd-owner-scalars",
                (
                    owner.sketch_rank,
                    str(owner.device),
                    owner._packet_boundary_index,
                    tuple(sorted(owner._all_context_ids)),
                    tuple(
                        (name, tuple(owner._source_measurements[name]))
                        for name in owner.design.stratum_names
                    ),
                    bool(owner._consumed),
                ),
            )
        )
        for name in owner.design.stratum_names:
            child = owner._accumulators.get(name)
            if type(child) is not DeviceGramFrequentDirectionsTrustAccumulator:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                    "registered FD stratum child is missing",
                )
            for role, tensor in (
                ("buffer", child._buffer),
                ("mean", child._mean),
                ("weighted-diagonal", child._weighted_second_moment_diagonal),
            ):
                entries.append(
                    (f"fd-{name}-{role}", _tensor_storage_invariant(tensor))
                )
            entries.append(
                (
                    f"fd-{name}-scalars",
                    (
                        child._filled,
                        child._total_shrinkage,
                        child.total_weight,
                        child.context_count,
                        child.context_ids,
                        child.last_update_source_row_hashes,
                        bool(child._consumed),
                    ),
                )
            )
        return tuple(entries)

    def begin_registered_look_packet(
        self,
        *,
        registered_look_index: int,
        packet_ordinal: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        stratum_labels: Sequence[str],
        sealed_order_indices: Sequence[int],
        source_measurement_hashes: Sequence[str],
    ) -> None:
        self._assert_live()
        if self._row_free_finalized:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MATERIALIZED,
                "row-free finalized D sessions cannot acquire another look",
            )
        if self._active is not None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LOOK,
                "a device-resident D look packet is already active",
            )
        look = _positive_int(registered_look_index, name="registered look index")
        packet = _nonnegative_int(packet_ordinal, name="packet ordinal")
        contexts = _identity_axis(context_ids, name="look context ids", unique=True)
        draws = _identity_axis(draw_ids, name="look draw ids", unique=True)
        labels = _identity_axis(stratum_labels, name="look stratum labels", unique=False)
        sources = _identity_axis(
            source_measurement_hashes,
            name="look source measurement hashes",
            unique=False,
        )
        orders = tuple(
            _nonnegative_int(value, name="sealed order index")
            for value in sealed_order_indices
        )
        if not (len(contexts) == len(draws) == len(labels) == len(sources) == len(orders)):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "look context/draw/stratum/source/order ledgers must align",
            )
        if len(contexts) > self.design.maximum_contexts_per_look:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.HBM_FORECAST_EXCEEDED,
                "look exceeds its pre-outcome active-packet HBM forecast",
            )
        if len(set(labels)) != 1:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "first device-resident D session accepts one same-stratum packet at a time",
            )
        stratum_name = labels[0]
        if stratum_name not in self.design.fd_design.stratum_names:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "look packet names an unregistered stratum",
            )
        stratum = self.design.fd_design.stratum_names.index(stratum_name)
        if set(contexts).intersection(
            context for ledger in self._context_ledgers for context in ledger
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.DUPLICATE_CONTEXT,
                "look packet repeats an already consumed context",
            )
        if set(draws).intersection(draw for ledger in self._draw_ledgers for draw in ledger):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.DUPLICATE_CONTEXT,
                "look packet repeats an already consumed draw",
            )
        start = self._stratum_counts[stratum]
        if tuple(range(start, start + len(contexts))) != orders:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "look packet is not the next sealed stratum prefix",
            )
        if start + len(contexts) > self.design.population_stratum_counts[stratum]:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "look packet exceeds its finite-population stratum",
            )
        literal_j_full = torch.empty(
            (len(contexts), self.design.route_count),
            dtype=torch.float64,
            device=self.device,
        )
        self._active = _ActiveLookPacket(
            registered_look_index=look,
            packet_ordinal=packet,
            context_ids=contexts,
            draw_ids=draws,
            stratum_name=stratum_name,
            stratum_index=stratum,
            source_measurement_hashes=sources,
            sealed_order_indices=orders,
            stratum_counts_before=tuple(self._stratum_counts),  # type: ignore[arg-type]
            literal_j_full=literal_j_full,
        )

    def _rows_tensor(self, value: Any, *, name: str, shape: tuple[int, int]) -> Any:
        if self.design.execution_mode is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION:
            if type(value) is not torch.Tensor or value.device != self.device:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    f"{name} must arrive as a tensor on the registered CUDA device",
                )
            tensor = value.detach()
        else:
            tensor = torch.as_tensor(value, dtype=torch.float64, device=self.device).detach()
        if (
            tensor.dtype != torch.float64
            or tuple(int(axis) for axis in tensor.shape) != shape
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all().item())
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.NONFINITE_SOURCE,
                f"{name} must be contiguous finite fp64 with shape {shape}",
            )
        return tensor

    def consume_same_capture_route_chunk(
        self,
        *,
        route_start: int,
        corrected_a_rows: Any,
        literal_j_rows: Any,
        route_axis_hash: str,
        corrected_a_stream_identity_hash: str,
        literal_j_stream_identity_hash: str,
        same_capture_chunk_source_hash: str,
    ) -> None:
        self._assert_live()
        active = self._active
        if active is None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.NO_ACTIVE_LOOK,
                "no active D look packet is ready for route chunks",
            )
        start = _nonnegative_int(route_start, name="route chunk start")
        if (
            start != active.route_cursor
            or _identity(route_axis_hash, name="route axis hash") != self.design.route_axis_hash
            or _identity(corrected_a_stream_identity_hash, name="corrected-A stream")
            != self.design.corrected_a_stream_identity_hash
            or _identity(literal_j_stream_identity_hash, name="literal-J stream")
            != self.design.literal_j_stream_identity_hash
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "route chunk is stale or out of order for this D session",
            )
        chunk_hash = _identity(
            same_capture_chunk_source_hash,
            name="same-capture chunk source hash",
        )
        rows = len(active.context_ids)
        if self.design.execution_mode is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION:
            if type(corrected_a_rows) is not torch.Tensor:
                _refuse(
                    V1SampledDeviceResidentDSessionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production corrected-A route chunks must arrive device-resident",
                )
            corrected_shape = tuple(int(axis) for axis in corrected_a_rows.shape)
        else:
            corrected_shape = tuple(int(axis) for axis in np.asarray(corrected_a_rows).shape)
        if len(corrected_shape) != 2:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "corrected-A route chunk must be a matrix",
            )
        width = corrected_shape[1]
        if (
            width <= 0
            or width > self.design.registered_route_chunk_size
            or start + width > self.design.route_count
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "route chunk width exceeds registered coverage",
            )
        corrected = self._rows_tensor(
            corrected_a_rows,
            name="corrected-A route chunk",
            shape=(rows, width),
        )
        literal_j = self._rows_tensor(
            literal_j_rows,
            name="literal-J route chunk",
            shape=(rows, width),
        )

        stratum = active.stratum_index
        count0 = self._stratum_counts[stratum]
        count1 = rows
        total = count0 + count1
        if count1 <= 0:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AXIS_MISMATCH,
                "route chunk has no contexts",
            )
        stop = start + width
        for source, mean_bank, m2_bank in (
            (corrected, self._signed_mean, self._signed_m2),
            (torch.abs(corrected), self._absolute_mean, self._absolute_m2),
        ):
            packet_mean = torch.mean(source, dim=0)
            packet_m2 = torch.sum((source - packet_mean) ** 2, dim=0)
            mean0 = mean_bank[stratum, start:stop]
            m20 = m2_bank[stratum, start:stop]
            if count0 == 0:
                mean0.copy_(packet_mean)
                m20.copy_(packet_m2)
            else:
                delta = packet_mean - mean0
                mean0.add_(delta, alpha=count1 / total)
                m20.add_(packet_m2 + delta * delta * (count0 * count1 / total))
        active.literal_j_full[:, start:stop].copy_(literal_j)
        active.route_windows.append((start, stop))
        active.chunk_source_hashes.append(chunk_hash)
        active.route_cursor = stop

    def end_registered_look_packet(self) -> V1SampledDeviceResidentDLookReceipt:
        self._assert_live()
        active = self._active
        if active is None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.NO_ACTIVE_LOOK,
                "no active D look packet can be closed",
            )
        if active.route_cursor != self.design.route_count:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "D look packet route chunks did not cover the full route axis",
            )
        labels = (active.stratum_name,) * len(active.context_ids)
        rows = active.literal_j_full.unsqueeze(1).contiguous()
        for accumulator in self._fd_accumulators.values():
            accumulator.append_labeled_rows(
                rows=rows,
                context_ids=active.context_ids,
                stratum_labels=labels,
                source_measurement_hashes=active.source_measurement_hashes,
            )
        stratum = active.stratum_index
        self._stratum_counts[stratum] += len(active.context_ids)
        self._context_ledgers[stratum].extend(active.context_ids)
        self._draw_ledgers[stratum].extend(active.draw_ids)
        self._source_measurements[stratum].extend(active.source_measurement_hashes)
        counts_after = tuple(self._stratum_counts)  # type: ignore[assignment]
        context_hash = _axis_hash(active.context_ids, prefix=_PREFIX + "look-context-axis-")
        draw_hash = _axis_hash(active.draw_ids, prefix=_PREFIX + "look-draw-axis-")
        source_hash = _axis_hash(
            active.source_measurement_hashes,
            prefix=_PREFIX + "look-source-measurement-axis-",
        )
        chunk_axis_hash = _axis_hash(
            active.chunk_source_hashes,
            prefix=_PREFIX + "same-capture-chunk-axis-",
        )
        self._source_packet_axis_hashes.append(chunk_axis_hash)
        corrected_scalar_hash = content_hash(
            {
                "session_design_hash": self.design.design_hash,
                "stratum_counts": counts_after,
                "look_receipts_before": tuple(receipt.receipt_hash for receipt in self._look_receipts),
                "latest_context_axis_hash": context_hash,
                "latest_chunk_source_axis_hash": chunk_axis_hash,
                "route_scale_values_not_hashed": True,
            },
            prefix=_PREFIX + "corrected-state-scalar-",
        )
        fd_scalar_hash = content_hash(
            {
                "session_design_hash": self.design.design_hash,
                "rank_storage_mode": self.rank_storage_mode.value,
                "hbm_admitted_fd_ranks": self.hbm_forecast.hbm_admitted_fd_ranks,
                "exposed_nested_fd_ranks": self.hbm_forecast.exposed_nested_fd_ranks,
                "live_fd_accumulator_ranks": self.hbm_forecast.live_fd_accumulator_ranks,
                "ceiling_fd_rank": self.hbm_forecast.ceiling_fd_rank,
                "packet_boundary_indices_by_rank": tuple(
                    (rank, _fd_packet_boundary_index(acc))
                    for rank, acc in self._fd_accumulators.items()
                ),
                "route_scale_values_not_hashed": True,
            },
            prefix=_PREFIX + "fd-state-scalar-",
        )
        avoided_score_d2h = (
            2
            * len(active.context_ids)
            * self.design.route_count
            * _F64_BYTES
        )
        avoided_fd_snapshot = sum(
            bytes_for_rank
            for rank, bytes_for_rank in self.hbm_forecast.fd_state_bytes_by_rank
            if rank in self.hbm_forecast.hbm_admitted_fd_ranks
        )
        receipt = V1SampledDeviceResidentDLookReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            registered_look_index=active.registered_look_index,
            packet_ordinal=active.packet_ordinal,
            route_axis_hash=self.design.route_axis_hash,
            route_count=self.design.route_count,
            stratum_name=active.stratum_name,
            context_axis_hash=context_hash,
            draw_axis_hash=draw_hash,
            source_measurement_axis_hash=source_hash,
            same_capture_chunk_source_axis_hash=chunk_axis_hash,
            stratum_counts_before=active.stratum_counts_before,
            stratum_counts_after=counts_after,
            route_windows=tuple(active.route_windows),
            hbm_admitted_fd_ranks=self.hbm_forecast.hbm_admitted_fd_ranks,
            corrected_state_scalar_hash=corrected_scalar_hash,
            fd_state_scalar_hash=fd_scalar_hash,
            avoided_legacy_intermediate_route_scale_d2h_bytes=avoided_score_d2h,
            avoided_legacy_intermediate_fd_snapshot_bytes=avoided_fd_snapshot,
        )
        self._look_receipts.append(receipt)
        self._active = None
        return receipt

    def select_smallest_decision_stable_rank(
        self,
        decision_stable_by_rank: Mapping[int, bool],
    ) -> V1SampledDeviceResidentDRankSelectionReceipt:
        self._assert_live()
        if (
            self.rank_storage_mode
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "canonical ceiling-only D sessions do not accept caller-supplied "
                "rank-stability booleans; the source-bound D/FD/KKT stability "
                "closure must freeze a nested ceiling-state view",
            )
        if not isinstance(decision_stable_by_rank, Mapping):
            raise TypeError("rank stability input must be a mapping")
        admitted = self.hbm_forecast.hbm_admitted_fd_ranks
        stable = tuple(
            rank
            for rank in admitted
            if _exact_bool(
                decision_stable_by_rank.get(rank, False),
                name=f"rank {rank} decision-stable flag",
            )
        )
        if not stable:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "no pre-outcome HBM-admitted rank is decision stable",
            )
        return V1SampledDeviceResidentDRankSelectionReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            hbm_admitted_fd_ranks=admitted,
            decision_stable_fd_ranks=stable,
            selected_physical_fd_rank=min(stable),
        )

    def register_authority_inert_materialization_rank(
        self,
        *,
        registered_proposal_fd_rank: int,
        source_rank_registration_hash: str,
    ) -> V1SampledDeviceResidentDMaterializationRankReceipt:
        """Bind one portable snapshot rank without certifying rank stability.

        The registered proposal rank must be the highest rank admitted by the
        frozen pre-outcome HBM forecast.  This compatibility receipt only lets
        the portable V7 graph consume one FD snapshot; it carries no claim
        about whether a lower-rank KKT face would have been decision-stable.
        """

        self._assert_live()
        if self._active is not None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LOOK,
                "cannot register a materialization rank during an active look",
            )
        rank = _positive_int(
            registered_proposal_fd_rank,
            name="registered proposal FD rank",
        )
        admitted = self.hbm_forecast.hbm_admitted_fd_ranks
        if not admitted or rank != max(admitted) or rank not in self._fd_accumulators:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "source-registered proposal rank is not the highest live HBM-"
                "admitted rank",
            )
        return V1SampledDeviceResidentDMaterializationRankReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            hbm_admitted_fd_ranks=admitted,
            registered_proposal_fd_rank=rank,
            source_rank_registration_hash=source_rank_registration_hash,
        )

    def replay_coordinate_receipt(
        self,
        *,
        sealed_model_context_ledger_hash: str,
    ) -> V1SampledDeviceResidentDReplayCoordinateReceipt:
        self._assert_live()
        if self._active is not None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LOOK,
                "cannot emit a replay coordinate with an active unclosed packet",
            )
        return V1SampledDeviceResidentDReplayCoordinateReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            sealed_model_context_ledger_hash=sealed_model_context_ledger_hash,
            route_axis_hash=self.design.route_axis_hash,
            consumed_look_receipt_hashes=tuple(
                receipt.assert_integrity() for receipt in self._look_receipts
            ),
            source_packet_axis_hashes=tuple(self._source_packet_axis_hashes),
            # ``_look_receipts`` is packet-granular: one registered
            # statistical look may contain several bounded transport packets.
            # The controller coordinate is logical-look granular, so packet
            # count must never impersonate the next registered look index.
            next_registered_look_index=(
                1
                if not self._look_receipts
                else self._look_receipts[-1].registered_look_index + 1
            ),
        )

    def finalize_registered_rank_row_free_once(
        self,
        materialization_rank_receipt: (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ),
    ) -> V1SampledDeviceResidentDRowFreeFinalizationReceipt:
        """Freeze a live registered rank without moments or FD snapshot D2H."""

        self._materialize_preflight()
        if self._row_free_finalized:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MATERIALIZED,
                "device-resident D session already has a row-free finalization",
            )
        if type(materialization_rank_receipt) is not (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ):
            raise TypeError("row-free finalization requires exact registered-rank receipt")
        if (
            materialization_rank_receipt.session_design_hash
            != self.design.design_hash
            or materialization_rank_receipt.hbm_forecast_hash
            != self.hbm_forecast.forecast_hash
            or materialization_rank_receipt.hbm_admitted_fd_ranks
            != self.hbm_forecast.hbm_admitted_fd_ranks
            or materialization_rank_receipt.registered_proposal_fd_rank
            not in self._fd_accumulators
            or materialization_rank_receipt.rank_stability_authority is not False
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "row-free finalization rank differs from the live session",
            )
        final_look = self._look_receipts[-1].registered_look_index
        context_axes = tuple(
            _axis_hash(values, prefix=_PREFIX + "final-context-axis-")
            for values in self._context_ledgers
        )
        draw_axes = tuple(
            _axis_hash(values, prefix=_PREFIX + "final-draw-axis-")
            for values in self._draw_ledgers
        )
        measurement_axes = tuple(
            _axis_hash(values, prefix=_PREFIX + "final-source-measurement-axis-")
            for values in self._source_measurements
        )
        look_axis = _axis_hash(
            tuple(receipt.receipt_hash for receipt in self._look_receipts),
            prefix=_PREFIX + "final-look-receipt-axis-",
        )
        packet_axis = _axis_hash(
            tuple(self._source_packet_axis_hashes),
            prefix=_PREFIX + "final-source-packet-axis-",
        )
        corrected_source = content_hash(
            {
                "session_design_hash": self.design.design_hash,
                "hbm_forecast_hash": self.hbm_forecast.forecast_hash,
                "final_corrected_state_scalar_hash": (
                    self._look_receipts[-1].corrected_state_scalar_hash
                ),
                "look_receipt_axis_hash": look_axis,
                "source_packet_axis_hash": packet_axis,
                "context_axis_hashes_by_stratum": context_axes,
                "draw_axis_hashes_by_stratum": draw_axes,
                "source_measurement_axis_hashes_by_stratum": measurement_axes,
                "corrected_A_must_be_reacquired_action_wise": True,
                "route_scale_values_not_hashed_or_materialized": True,
            },
            prefix=_PREFIX + "row-free-corrected-reacquisition-source-",
        )
        live_fd_source = content_hash(
            {
                "session_design_hash": self.design.design_hash,
                "hbm_forecast_hash": self.hbm_forecast.forecast_hash,
                "materialization_rank_receipt_hash": (
                    materialization_rank_receipt.receipt_hash
                ),
                "registered_proposal_fd_rank": (
                    materialization_rank_receipt.registered_proposal_fd_rank
                ),
                "final_fd_state_scalar_hash": (
                    self._look_receipts[-1].fd_state_scalar_hash
                ),
                "look_receipt_axis_hash": look_axis,
                "source_packet_axis_hash": packet_axis,
                "live_device_factor_not_snapshotted": True,
            },
            prefix=_PREFIX + "row-free-live-fd-source-",
        )
        receipt = V1SampledDeviceResidentDRowFreeFinalizationReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            materialization_rank_receipt_hash=(
                materialization_rank_receipt.receipt_hash
            ),
            registered_proposal_fd_rank=(
                materialization_rank_receipt.registered_proposal_fd_rank
            ),
            final_logical_look_index=final_look,
            route_axis_hash=self.design.route_axis_hash,
            route_count=self.design.route_count,
            corrected_a_stream_identity_hash=(
                self.design.corrected_a_stream_identity_hash
            ),
            literal_j_stream_identity_hash=(
                self.design.literal_j_stream_identity_hash
            ),
            stratum_counts=tuple(self._stratum_counts),
            context_axis_hashes_by_stratum=context_axes,
            draw_axis_hashes_by_stratum=draw_axes,
            source_measurement_axis_hashes_by_stratum=measurement_axes,
            look_receipt_axis_hash=look_axis,
            source_packet_axis_hash=packet_axis,
            corrected_reacquisition_source_hash=corrected_source,
            live_fd_source_hash=live_fd_source,
            _issuance_token=_ROW_FREE_FINALIZATION_TOKEN,
        )
        self._row_free_source_invariants = self._registered_rank_source_invariants(
            materialization_rank_receipt.registered_proposal_fd_rank
        )
        self._row_free_finalized = True
        return receipt

    def freeze_registered_rank_kkt_geometry_device_once(
        self,
        *,
        row_free_finalization_receipt: (
            V1SampledDeviceResidentDRowFreeFinalizationReceipt
        ),
        materialization_rank_receipt: (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ),
    ) -> V1SampledDeviceResidentDKKTGeometryDeviceHandle:
        """Consume the registered live D endpoint into a guarded KKT lease.

        The only route-scale outputs are singleton corrected work, the literal-J
        mean-plus-centered factor, and the FD residual diagonal.  No tensor is
        copied to host or content-hashed.  Native-fiber signed work is outside
        this contract and must come from its separately registered corrected-row
        bank.
        """

        self._assert_live()
        if self._kkt_geometry_freeze_attempted or self._kkt_geometry_handle is not None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.DUPLICATE_FREEZE,
                "device-resident D session already attempted its KKT geometry freeze",
            )
        if not self._row_free_finalized:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "KKT geometry requires the row-free endpoint to be frozen first",
            )
        if type(row_free_finalization_receipt) is not (
            V1SampledDeviceResidentDRowFreeFinalizationReceipt
        ):
            raise TypeError("KKT geometry freeze requires the exact row-free receipt")
        if type(materialization_rank_receipt) is not (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ):
            raise TypeError("KKT geometry freeze requires the exact rank receipt")
        row_free_hash = row_free_finalization_receipt.assert_integrity()
        rank_hash = materialization_rank_receipt.receipt_hash
        rank = materialization_rank_receipt.registered_proposal_fd_rank
        if (
            row_free_finalization_receipt.session_design_hash
            != self.design.design_hash
            or row_free_finalization_receipt.hbm_forecast_hash
            != self.hbm_forecast.forecast_hash
            or row_free_finalization_receipt.materialization_rank_receipt_hash
            != rank_hash
            or row_free_finalization_receipt.registered_proposal_fd_rank != rank
            or row_free_finalization_receipt.route_axis_hash
            != self.design.route_axis_hash
            or row_free_finalization_receipt.route_count != self.design.route_count
            or row_free_finalization_receipt.stratum_counts
            != tuple(self._stratum_counts)
            or materialization_rank_receipt.session_design_hash
            != self.design.design_hash
            or materialization_rank_receipt.hbm_forecast_hash
            != self.hbm_forecast.forecast_hash
            or rank not in self._fd_accumulators
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "KKT geometry receipts differ from the live row-free source",
            )
        if rank != max(self.hbm_forecast.hbm_admitted_fd_ranks):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "KKT geometry must consume the source-registered ceiling FD rank",
            )
        expected_invariants = self._row_free_source_invariants
        if expected_invariants is None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "row-free source has no process-local mutation guard",
            )
        if self._registered_rank_source_invariants(rank) != expected_invariants:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MUTATED_CUSTODY,
                "row-free route statistics changed before KKT geometry freeze",
            )

        # Crossing this point is consuming.  A numerical failure must replay
        # the sealed D prefix; it cannot retry against a partially compressed
        # live chart.
        self._kkt_geometry_freeze_attempted = True
        fd_owner = self._fd_accumulators.pop(rank)
        fd_geometry: V1StratifiedFDDeviceGeometryState = (
            fd_owner.finalize_combined_device_consuming()
        )
        fd_receipt = fd_geometry.receipt
        if (
            fd_receipt.design_hash != self.design.fd_design.design_hash
            or fd_receipt.route_axis_hash != self.design.route_axis_hash
            or fd_receipt.route_count != self.design.route_count
            or fd_receipt.device_identity != str(self.device)
            or fd_receipt.stratum_names != self.design.fd_design.stratum_names
            or fd_receipt.stratum_weights != self.design.fd_design.stratum_weights
            or fd_receipt.stratum_context_counts != tuple(self._stratum_counts)
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "consumed literal-J geometry differs from the registered D source",
            )
        mean_component, centered_factor, residual_diagonal = (
            fd_geometry.take_tensors_once()
        )
        singleton_work = torch.zeros(
            self.design.route_count,
            dtype=torch.float64,
            device=self.device,
        )
        for weight, absolute_mean in zip(
            self.design.fd_design.stratum_weights,
            self._absolute_mean,
            strict=True,
        ):
            singleton_work.add_(absolute_mean, alpha=weight)
        singleton_work = singleton_work.detach().contiguous()
        factor = torch.empty(
            (
                int(mean_component.shape[0]) + int(centered_factor.shape[0]),
                self.design.route_count,
            ),
            dtype=torch.float64,
            device=self.device,
        )
        response_rows = int(mean_component.shape[0])
        factor[:response_rows].copy_(mean_component)
        factor[response_rows:].copy_(centered_factor)
        factor = factor.detach().contiguous()
        residual_diagonal = residual_diagonal.detach().contiguous()
        factor_frobenius_norm = float(torch.linalg.vector_norm(factor).item())
        residual_sqrt_l2_norm = math.sqrt(
            float(torch.sum(residual_diagonal).item())
        )
        receipt = V1SampledDeviceResidentDKKTGeometryReceipt(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            row_free_finalization_receipt_hash=row_free_hash,
            materialization_rank_receipt_hash=rank_hash,
            stratified_fd_device_receipt_hash=fd_receipt.receipt_hash,
            registered_proposal_fd_rank=rank,
            final_logical_look_index=(
                row_free_finalization_receipt.final_logical_look_index
            ),
            route_axis_hash=self.design.route_axis_hash,
            route_count=self.design.route_count,
            corrected_a_stream_identity_hash=(
                self.design.corrected_a_stream_identity_hash
            ),
            literal_j_stream_identity_hash=(
                self.design.literal_j_stream_identity_hash
            ),
            stratum_names=self.design.fd_design.stratum_names,
            stratum_weights=self.design.fd_design.stratum_weights,
            stratum_counts=tuple(self._stratum_counts),
            context_axis_hashes_by_stratum=(
                row_free_finalization_receipt.context_axis_hashes_by_stratum
            ),
            draw_axis_hashes_by_stratum=(
                row_free_finalization_receipt.draw_axis_hashes_by_stratum
            ),
            source_measurement_axis_hashes_by_stratum=(
                row_free_finalization_receipt.source_measurement_axis_hashes_by_stratum
            ),
            device_identity=str(self.device),
            singleton_work_shape=tuple(int(value) for value in singleton_work.shape),
            factor_shape=tuple(int(value) for value in factor.shape),
            residual_diagonal_shape=tuple(
                int(value) for value in residual_diagonal.shape
            ),
            response_rows=response_rows,
            retained_centered_rank=int(centered_factor.shape[0]),
            residual_spectral_bound=fd_receipt.combined_residual_spectral_bound,
            factor_frobenius_norm=factor_frobenius_norm,
            residual_sqrt_l2_norm=residual_sqrt_l2_norm,
            _issuance_token=_KKT_GEOMETRY_ISSUANCE_TOKEN,
        )
        handle = V1SampledDeviceResidentDKKTGeometryDeviceHandle(
            receipt=receipt,
            tensors={
                "singleton_work": singleton_work,
                "factor": factor,
                "residual_diagonal": residual_diagonal,
            },
            _issuance_token=_KKT_GEOMETRY_ISSUANCE_TOKEN,
        )
        self._fd_accumulators.clear()
        # KKT geometry is now the sole route-scale D owner.  The routewise
        # corrected moments were needed to form singleton work and to guard the
        # one-shot source freeze; native-fiber signed F is reacquired through
        # its separately registered per-context row bank, so none of these four
        # tensors has a post-freeze scientific consumer.
        self._signed_mean = None
        self._signed_m2 = None
        self._absolute_mean = None
        self._absolute_m2 = None
        self._row_free_source_invariants = None
        self._kkt_geometry_handle = handle
        del mean_component, centered_factor, fd_geometry, fd_owner
        return handle

    def _materialize_corrected_attribution_moments(
        self,
    ) -> tuple[V1StratifiedCorrectedAttributionMoments, int]:
        if not self._look_receipts:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "corrected moments require at least one closed registered look",
            )
        logical_look_index = self._look_receipts[-1].registered_look_index
        prior_logical_receipts = tuple(
            receipt.receipt_hash
            for receipt in self._look_receipts
            if receipt.registered_look_index < logical_look_index
        )
        signed_mean = _readonly_le_f64(self._signed_mean.detach().cpu().numpy())
        signed_m2 = _readonly_le_f64(self._signed_m2.detach().cpu().numpy())
        absolute_mean = _readonly_le_f64(self._absolute_mean.detach().cpu().numpy())
        absolute_m2 = _readonly_le_f64(self._absolute_m2.detach().cpu().numpy())
        corrected_bytes = int(
            signed_mean.nbytes
            + signed_m2.nbytes
            + absolute_mean.nbytes
            + absolute_m2.nbytes
        )
        prior = (
            None
            if logical_look_index == 1
            else content_hash(
                prior_logical_receipts,
                prefix=_PREFIX + "device-resident-prior-corrected-state-",
            )
        )
        return (
            V1StratifiedCorrectedAttributionMoments(
                campaign_contract_hash=self.design.fd_design.campaign_contract_hash,
                model_identity_hash=self.design.fd_design.model_identity_hash,
                query_contract_hash=self.design.fd_design.query_contract_hash,
                preprocessing_hash=self.design.fd_design.preprocessing_hash,
                current_mask_hash=self.design.fd_design.current_mask_hash,
                raw_logit_stream_source_hash=self.design.corrected_a_stream_identity_hash,
                sampling_law_hash=self.design.fd_design.sampling_law_hash,
                proposal_generation=self.design.fd_design.proposal_generation,
                route_count=self.design.route_count,
                route_axis_hash=ordered_string_axis_hash(
                    self.design.fd_design.route_ids,
                    name="device-resident corrected-A route axis",
                ),
                stratum_names=self.design.fd_design.stratum_names,  # type: ignore[arg-type]
                population_stratum_counts=self.design.population_stratum_counts,
                sealed_population_order_hashes=self.design.sealed_population_order_hashes,
                stratum_counts=tuple(self._stratum_counts),  # type: ignore[arg-type]
                observed_context_ids_by_stratum=tuple(
                    tuple(values) for values in self._context_ledgers
                ),  # type: ignore[arg-type]
                observed_draw_ids_by_stratum=tuple(
                    tuple(values) for values in self._draw_ledgers
                ),  # type: ignore[arg-type]
                signed_mean_by_stratum=signed_mean,
                signed_m2_by_stratum=signed_m2,
                absolute_mean_by_stratum=absolute_mean,
                absolute_m2_by_stratum=absolute_m2,
                # Registered looks are statistical decision boundaries;
                # bounded transport packets are an execution detail.  Using
                # packet count here made a multi-packet final look disagree
                # with its preregistered score boundary on H200.
                look_index=logical_look_index,
                prior_state_hash=prior,
            ),
            corrected_bytes,
        )

    def _materialize_preflight(self) -> None:
        self._assert_live()
        if self._row_free_finalized:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.MATERIALIZED,
                "row-free finalized D sessions cannot materialize portable arrays",
            )
        if self._active is not None:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ACTIVE_LOOK,
                "cannot materialize portable D state with an active look packet",
            )
        if any(count <= 0 for count in self._stratum_counts):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.ORDER_MISMATCH,
                "portable D materialization requires every stratum to be observed",
            )

    def materialize_selected_rank_portable_state_once(
        self,
        rank_selection_receipt: V1SampledDeviceResidentDRankSelectionReceipt,
    ) -> V1SampledDeviceResidentDSessionMaterialization:
        """Materialize only the decision-stable selected FD rank.

        This is the normal production path.  A rank-selection receipt is required
        so the final route-scale D2H cannot silently snapshot every HBM-admitted
        rank after the outcome is known.
        """

        self._materialize_preflight()
        if (
            self.rank_storage_mode
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "canonical ceiling-only D sessions materialize their single "
                "source-registered ceiling state; lower-rank decision images are "
                "array-free nested views issued by the D-stability closure",
            )
        if type(rank_selection_receipt) is not V1SampledDeviceResidentDRankSelectionReceipt:
            raise TypeError("selected-rank materialization requires exact rank-selection receipt")
        if (
            rank_selection_receipt.session_design_hash != self.design.design_hash
            or rank_selection_receipt.hbm_forecast_hash != self.hbm_forecast.forecast_hash
            or rank_selection_receipt.hbm_admitted_fd_ranks
            != self.hbm_forecast.hbm_admitted_fd_ranks
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "rank-selection receipt does not bind this live D session",
            )
        rank = rank_selection_receipt.selected_physical_fd_rank
        if rank not in self._fd_accumulators:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "selected FD rank is not live in this D session",
            )
        self._materialized = True
        moments, corrected_bytes = self._materialize_corrected_attribution_moments()
        snapshot = self._fd_accumulators[rank].snapshot()
        fd_bytes = _fd_snapshot_bytes(snapshot)
        return V1SampledDeviceResidentDSessionMaterialization(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            final_look_count=moments.look_index,
            route_axis_hash=self.design.route_axis_hash,
            corrected_attribution_moments=moments,
            fd_snapshots_by_rank=((rank, snapshot),),
            final_route_scale_d2h_bytes=corrected_bytes + fd_bytes,
            materialization_mode=DEVICE_RESIDENT_D_SELECTED_MATERIALIZATION_MODE,
            selected_physical_fd_rank=rank,
            selected_rank_selection_receipt_hash=rank_selection_receipt.receipt_hash,
            unselected_rank_snapshot_count=0,
            unselected_rank_route_scale_d2h_bytes=0,
        )

    def materialize_registered_rank_portable_state_once(
        self,
        materialization_rank_receipt: (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ),
    ) -> V1SampledDeviceResidentDSessionMaterialization:
        """Materialize one source-registered rank without stability authority."""

        self._materialize_preflight()
        if type(materialization_rank_receipt) is not (
            V1SampledDeviceResidentDMaterializationRankReceipt
        ):
            raise TypeError(
                "registered-rank materialization requires exact authority-inert receipt"
            )
        if (
            materialization_rank_receipt.session_design_hash
            != self.design.design_hash
            or materialization_rank_receipt.hbm_forecast_hash
            != self.hbm_forecast.forecast_hash
            or materialization_rank_receipt.hbm_admitted_fd_ranks
            != self.hbm_forecast.hbm_admitted_fd_ranks
            or materialization_rank_receipt.rank_stability_authority is not False
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "materialization-rank receipt does not bind this live D session",
            )
        rank = materialization_rank_receipt.registered_proposal_fd_rank
        if rank not in self._fd_accumulators:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "registered materialization rank is not live in this D session",
            )
        self._materialized = True
        moments, corrected_bytes = self._materialize_corrected_attribution_moments()
        snapshot = self._fd_accumulators[rank].snapshot()
        fd_bytes = _fd_snapshot_bytes(snapshot)
        return V1SampledDeviceResidentDSessionMaterialization(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            final_look_count=moments.look_index,
            route_axis_hash=self.design.route_axis_hash,
            corrected_attribution_moments=moments,
            fd_snapshots_by_rank=((rank, snapshot),),
            final_route_scale_d2h_bytes=corrected_bytes + fd_bytes,
            materialization_mode=(
                DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE
            ),
            selected_physical_fd_rank=rank,
            selected_rank_selection_receipt_hash=None,
            materialization_rank_receipt_hash=(
                materialization_rank_receipt.receipt_hash
            ),
            rank_stability_authority=False,
            unselected_rank_snapshot_count=0,
            unselected_rank_route_scale_d2h_bytes=0,
        )

    def materialize_all_ranks_diagnostic_once(
        self,
    ) -> V1SampledDeviceResidentDSessionMaterialization:
        """Explicit diagnostic/reference snapshot of every HBM-admitted rank."""

        self._materialize_preflight()
        if (
            self.rank_storage_mode
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "canonical ceiling-only D sessions have no independent all-rank "
                "state to materialize; use the authority-inert registered ceiling "
                "materialization and nested metadata views",
            )
        self._materialized = True
        moments, corrected_bytes = self._materialize_corrected_attribution_moments()
        fd_snapshots: list[tuple[int, V1StratifiedFDAccumulatorSnapshot]] = []
        fd_bytes = 0
        for rank, accumulator in self._fd_accumulators.items():
            snapshot = accumulator.snapshot()
            fd_bytes += _fd_snapshot_bytes(snapshot)
            fd_snapshots.append((rank, snapshot))
        return V1SampledDeviceResidentDSessionMaterialization(
            session_design_hash=self.design.design_hash,
            hbm_forecast_hash=self.hbm_forecast.forecast_hash,
            final_look_count=moments.look_index,
            route_axis_hash=self.design.route_axis_hash,
            corrected_attribution_moments=moments,
            fd_snapshots_by_rank=tuple(fd_snapshots),
            final_route_scale_d2h_bytes=corrected_bytes + fd_bytes,
            materialization_mode=DEVICE_RESIDENT_D_DIAGNOSTIC_MATERIALIZATION_MODE,
        )

    def materialize_portable_state_once(
        self,
    ) -> V1SampledDeviceResidentDSessionMaterialization:
        """Compatibility alias for the explicit all-rank diagnostic path.

        Production callers must use
        :meth:`materialize_registered_rank_portable_state_once` with the
        authority-inert source-registered ceiling-rank receipt.  Rank/face
        stability is certified later from source-bound D/FD/KKT evidence.
        """

        if self.design.execution_mode is V1SampledDeviceResidentDSessionMode.CUDA_PRODUCTION:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "production D materialization must name the selected decision-stable rank explicitly",
            )
        return self.materialize_all_ranks_diagnostic_once()

    def selected_rank_fd_accumulator(
        self,
        rank_selection_receipt: V1SampledDeviceResidentDRankSelectionReceipt,
    ) -> V1StratifiedDeviceFDTrustAccumulator:
        """Return the live selected-rank FD owner without materializing D2H state."""

        self._assert_live()
        if (
            self.rank_storage_mode
            is V1SampledDeviceResidentDRankStorageMode.CANONICAL_CEILING_ONLY
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.AUTHORITY_VIOLATION,
                "canonical ceiling-only D sessions expose the one ceiling owner "
                "through its source-bound final-rank receipt, not a caller-selected "
                "rank handle",
            )
        if type(rank_selection_receipt) is not V1SampledDeviceResidentDRankSelectionReceipt:
            raise TypeError("selected-rank FD handle requires exact rank-selection receipt")
        if (
            self._active is not None
            or rank_selection_receipt.session_design_hash != self.design.design_hash
            or rank_selection_receipt.hbm_forecast_hash != self.hbm_forecast.forecast_hash
        ):
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.STALE_IDENTITY,
                "selected-rank FD handle does not bind a closed live D session",
            )
        rank = rank_selection_receipt.selected_physical_fd_rank
        if rank not in self._fd_accumulators:
            _refuse(
                V1SampledDeviceResidentDSessionRefusalCode.RANK_NOT_ADMITTED,
                "selected FD rank is not live in this D session",
            )
        return self._fd_accumulators[rank]

    def evict(self) -> None:
        if self._evicted:
            return
        if self._kkt_geometry_handle is not None:
            # The handle owns the transferred proposal tensors.  Its active
            # lease check must run before any source state is discarded.
            self._kkt_geometry_handle.evict()
        self._active = None
        self._fd_accumulators.clear()
        self._signed_mean = None
        self._signed_m2 = None
        self._absolute_mean = None
        self._absolute_m2 = None
        self._row_free_source_invariants = None
        self._evicted = True
        if torch is not None and self.device.type == "cuda":  # pragma: no cover - CUDA only.
            torch.cuda.empty_cache()


__all__ = [
    "DEVICE_RESIDENT_D_SESSION_AUTHORITY_SCOPE",
    "DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA",
    "DEVICE_RESIDENT_D_SESSION_HBM_FORECAST_SCHEMA",
    "DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA",
    "DEVICE_RESIDENT_D_MATERIALIZATION_SCHEMA",
    "DEVICE_RESIDENT_D_MATERIALIZATION_RANK_SCHEMA",
    "DEVICE_RESIDENT_D_MATERIALIZATION_RANK_RULE",
    "DEVICE_RESIDENT_D_SELECTED_MATERIALIZATION_MODE",
    "DEVICE_RESIDENT_D_REGISTERED_RANK_MATERIALIZATION_MODE",
    "DEVICE_RESIDENT_D_DIAGNOSTIC_MATERIALIZATION_MODE",
    "DEVICE_RESIDENT_D_RANK_SELECTION_SCHEMA",
    "DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA",
    "DEVICE_RESIDENT_D_ROW_FREE_FINALIZATION_SCHEMA",
    "DEVICE_RESIDENT_D_KKT_GEOMETRY_RECEIPT_SCHEMA",
    "DEVICE_RESIDENT_D_SINGLETON_WORK_SEMANTICS",
    "DEVICE_RESIDENT_D_KKT_FACTOR_SEMANTICS",
    "V1SampledDeviceResidentDLookReceipt",
    "V1SampledDeviceResidentDRankSelectionReceipt",
    "V1SampledDeviceResidentDMaterializationRankReceipt",
    "V1SampledDeviceResidentDReplayCoordinateReceipt",
    "V1SampledDeviceResidentDRowFreeFinalizationReceipt",
    "V1SampledDeviceResidentDKKTGeometryReceipt",
    "V1SampledDeviceResidentDKKTGeometryLease",
    "V1SampledDeviceResidentDKKTGeometryDeviceHandle",
    "V1SampledDeviceResidentDSession",
    "V1SampledDeviceResidentDSessionDesign",
    "V1SampledDeviceResidentDSessionHBMForecast",
    "V1SampledDeviceResidentDSessionMaterialization",
    "V1SampledDeviceResidentDSessionMode",
    "V1SampledDeviceResidentDRankStorageMode",
    "V1SampledDeviceResidentDSessionRefusal",
    "V1SampledDeviceResidentDSessionRefusalCode",
    "forecast_v1_sampled_device_resident_d_session_hbm",
]
