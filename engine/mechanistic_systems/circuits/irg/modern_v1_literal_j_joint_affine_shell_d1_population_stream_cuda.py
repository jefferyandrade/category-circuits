"""Bounded D1-only joint affine-shell literal-J projection stream.

This owner replays the already captured canonical D1 literal-J attachments
once for the complete action-plus-shared-shell direction family.  The logical
replay has two route-major source passes: pass one retains only the
context-by-direction projections, their numerical enclosures, ``Q``, and the
compact shell columns of ``M``; pass two exposes ``M v`` through an exclusive
one-live route-tile cursor.  It never constructs a direction-by-route bank.

The estimand here is deliberately only the conditional D1 empirical mean,
with exact rational weight ``1 / d``.  The D1 population mass ``d / N`` is
recorded but is not applied.  Residual rows and full-population authority
belong to a downstream typed join.  No provider, forward pass, VJP, confidence
allocation, decision, or commit is performed by this module.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellFamilyDeviceHandle,
    V1LiteralJJointAffineShellFamilyExecutionMode,
    V1LiteralJJointAffineShellFamilyLease,
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
    V1LiteralJConditionedD1ShardReplaySource,
    _d1_axis_hash,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_D1_population_stream_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_AUTHORITY: Final = (
    "conditional_D1_empirical_literal_J_projection_numerics_only_no_residual_"
    "or_full_population_D_C_K_O_commit_terminal_performance_or_pareto_"
    "authority_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_ALGORITHM: Final = (
    "one_joint_action_plus_shared_shell_D1_replay_two_route_major_passes_"
    "exact_uniform_one_over_d_Q_compact_shell_M_and_one_live_Mv_tile_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-joint-affine-shell-D1-stream-"
_DESIGN_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_Q_TOKEN: Final = object()
_SHELL_TOKEN: Final = object()
_TILE_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_F64_BYTES: Final = 8
_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096


class V1LiteralJJointAffineShellD1PopulationStreamExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1LiteralJJointAffineShellD1PopulationStreamRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    SOURCE_MISMATCH = "source_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    DIRECTION_CUSTODY_MISMATCH = "direction_custody_mismatch"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DEVICE_REQUIRED = "device_required"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONFINITE_SOURCE = "nonfinite_source"
    NUMERICAL_REFUSAL = "numerical_refusal"
    ACTIVE_LEASE = "active_lease"
    ACTIVE_TILE = "active_tile"
    STREAM_NOT_EXHAUSTED = "stream_not_exhausted"
    ROUTE_STREAM_CLAIMED = "route_stream_claimed"
    ROUTE_STREAM_CONSUMED = "route_stream_consumed"
    MUTATED_CUSTODY = "mutated_custody"
    STALE_IDENTITY = "stale_identity"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJJointAffineShellD1PopulationStreamRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LiteralJJointAffineShellD1PopulationStreamRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointAffineShellD1PopulationStreamRefusalCode:
            raise TypeError("D1 joint-stream refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointAffineShellD1PopulationStreamRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointAffineShellD1PopulationStreamRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _identity_axis(value: object, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or (unique and len(result) != len(set(result))):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonempty and correctly unique",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value._version),
        bool(value.requires_grad),
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
    nonnegative: bool = False,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
        or (device is not None and value.device != device)
        or not bool(torch.isfinite(value).all().item())
    ):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            f"{name} must be finite contiguous detached FP64 with shape {shape}",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a negative radius",
        )
    return value


def _gamma(operation_count: int) -> float:
    count = _positive_int(operation_count, name="rounding operation count")
    eps = 2.0**-52
    denominator = 1.0 - count * eps
    if denominator <= 0.0:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.NUMERICAL_REFUSAL,
            "FP64 gamma denominator is nonpositive",
        )
    return count * eps / denominator


def _outward_nonnegative(value: Any) -> Any:
    return torch.nextafter(
        torch.clamp(value, min=0.0) + torch.full_like(value, _MIN_SUBNORMAL),
        torch.full_like(value, math.inf),
    )


def _sum_radius(abs_mass: Any, operation_count: int) -> Any:
    relative = _gamma(operation_count)
    recovery = relative / (1.0 - relative)
    return _outward_nonnegative(
        torch.abs(abs_mass) * recovery
        + torch.full_like(abs_mass, operation_count * _MIN_SUBNORMAL)
    )


def _division_radius(center: Any) -> Any:
    return _outward_nonnegative(torch.abs(center) * _gamma(2))


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamDesign:
    route_tile_size: int
    action_tile_size: int
    execution_mode: V1LiteralJJointAffineShellD1PopulationStreamExecutionMode
    hbm_ceiling_fraction: float = 0.85
    deterministic_fp64_required: bool = True
    one_live_tile_required: bool = True
    forbid_direction_route_bank: bool = True
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_stream_design_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("D1 joint-stream designs are issuer-sealed")
        object.__setattr__(self, "route_tile_size", _positive_int(self.route_tile_size, name="route_tile_size"))
        object.__setattr__(self, "action_tile_size", _positive_int(self.action_tile_size, name="action_tile_size"))
        if (
            type(self.execution_mode)
            is not V1LiteralJJointAffineShellD1PopulationStreamExecutionMode
            or type(self.hbm_ceiling_fraction) is not float
            or not math.isfinite(self.hbm_ceiling_fraction)
            or self.hbm_ceiling_fraction != 0.85
            or any(
                type(item) is not bool or item is not True
                for item in (
                    self.deterministic_fp64_required,
                    self.one_live_tile_required,
                    self.forbid_direction_route_bank,
                )
            )
            or self.schema
            != "partition_free_v1_literal_J_joint_affine_shell_D1_stream_design_v1"
        ):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_DESIGN,
                "D1 joint-stream design changed its bounded deterministic policy",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(
            MappingProxyType(
                {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
            ),
            role="design",
        )

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY,
                "D1 joint-stream design mutated",
            )
        return observed

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()


def register_v1_literal_j_joint_affine_shell_d1_population_stream_cuda_design(
    *, route_tile_size: int = 131_072, action_tile_size: int = 16
) -> V1LiteralJJointAffineShellD1PopulationStreamDesign:
    return V1LiteralJJointAffineShellD1PopulationStreamDesign(
        route_tile_size=route_tile_size,
        action_tile_size=action_tile_size,
        execution_mode=V1LiteralJJointAffineShellD1PopulationStreamExecutionMode.CUDA_PRODUCTION,
        _issuance_token=_DESIGN_TOKEN,
    )


def _register_v1_literal_j_joint_affine_shell_d1_population_stream_cpu_reference_design(
    *, route_tile_size: int = 64, action_tile_size: int = 4
) -> V1LiteralJJointAffineShellD1PopulationStreamDesign:
    return V1LiteralJJointAffineShellD1PopulationStreamDesign(
        route_tile_size=route_tile_size,
        action_tile_size=action_tile_size,
        execution_mode=V1LiteralJJointAffineShellD1PopulationStreamExecutionMode.CPU_REFERENCE_TEST_ONLY,
        _issuance_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class _D1ShardSlice:
    attachment: V1SampledDualChartRowShardAttachment = field(repr=False, compare=False)
    source_rows: tuple[int, ...]
    destination_rows: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _D1SourcePlan:
    slices: tuple[_D1ShardSlice, ...]
    route_windows: tuple[tuple[int, int], ...]
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    strata: tuple[str, ...]
    source_axis_hash: str
    maximum_registered_window_width: int


def _source_axis_hash(
    *,
    plan_slices: tuple[_D1ShardSlice, ...],
    route_windows: tuple[tuple[int, int], ...],
    replay: Any,
    compact_receipt_hash: str,
) -> str:
    return _payload_hash(
        MappingProxyType(
            {
                "conditioned_d1_replay_receipt_hash": replay.receipt_hash,
                "compact_source_receipt_hash": compact_receipt_hash,
                "literal_j_shards": tuple(
                    (
                        item.attachment.receipt.receipt_hash,
                        item.attachment.receipt.logical_identity_hash,
                        item.attachment.receipt.content_sha256,
                        item.source_rows,
                        item.destination_rows,
                    )
                    for item in plan_slices
                ),
                "canonical_context_axis_hash": replay.canonical_context_axis_hash,
                "canonical_draw_axis_hash": replay.canonical_draw_axis_hash,
                "canonical_measurement_axis_hash": replay.canonical_measurement_axis_hash,
                "canonical_capture_axis_hash": replay.canonical_capture_axis_hash,
                "canonical_source_packet_axis_hash": replay.canonical_source_packet_axis_hash,
                "route_windows": route_windows,
            }
        ),
        role="source-axis",
    )


def _build_source_plan(
    source: V1LiteralJConditionedD1ShardReplaySource,
    family: V1LiteralJJointAffineShellFamilyDeviceHandle,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
) -> _D1SourcePlan:
    if (
        type(source) is not V1LiteralJConditionedD1ShardReplaySource
        or type(family) is not V1LiteralJJointAffineShellFamilyDeviceHandle
        or type(preregistration)
        is not V1LiteralJJointAffineShellPreoutcomeRegistration
    ):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
            "D1 stream requires the exact conditioned source and joint family",
        )
    replay = source.replay_receipt
    replay.assert_integrity()
    compact = source.compact_source_receipt
    joint = family.receipt
    try:
        preregistration.assert_integrity()
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "the independently frozen preoutcome action family failed replay: "
            f"{type(error).__name__}: {error}",
        )
    if (
        joint.joint_preoutcome_registration_hash
        != preregistration.registration_hash
        or joint.preoutcome_joint_direction_axis_hash
        != preregistration.joint_direction_axis_hash
        or joint.joint_direction_ids != preregistration.joint_direction_ids
        or joint.physical_action_identity_hashes
        != preregistration.physical_action_identity_hashes
        or joint.physical_action_hashes != preregistration.physical_action_hashes
        or joint.shell_indices != preregistration.shell_indices
        or joint.shell_route_ids != preregistration.shell_route_ids
    ):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH,
            "postsource joint custody does not close the exact frozen preoutcome "
            "action-plus-shell axis",
        )
    if (
        compact.receipt_hash != replay.compact_source_receipt_hash
        or replay.route_axis_hash != joint.route_axis_hash
        or replay.route_id_axis_hash != joint.route_id_axis_hash
        or replay.route_count != joint.route_count
        or replay.chart_identity_hash != joint.chart_identity_hash
        or replay.current_mask_hash != joint.current_mask_hash
        or replay.proposal_generation != joint.proposal_generation
    ):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
            "canonical D1 source and joint family left the same chart/route axis",
        )
    if joint.registered_d_prefix_hash != replay.d1_pair_prefix_axis_hash:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.PREFIX_MISMATCH,
            "joint family is not rooted at the canonical D1 pair-prefix boundary",
        )
    if replay.compact_internal_context_weight_fraction != (1, replay.context_count):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION,
            "conditioned D1 source no longer exposes the exact internal 1/d measure",
        )

    locations: list[tuple[int, int, str, str, str]] = []
    route_windows: tuple[tuple[int, int], ...] | None = None
    for shard_index, attachment in enumerate(source.literal_j_shards):
        receipt = attachment.receipt
        attachment.verify_live_witnesses(
            expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
            expected_chart_identity_hash=replay.chart_identity_hash,
            expected_current_mask_hash=replay.current_mask_hash,
            expected_proposal_generation=replay.proposal_generation,
        )
        if (
            receipt.registered_d_prefix_hash != replay.registered_d_prefix_hash
            or receipt.route_axis_hash != replay.route_axis_hash
            or receipt.route_id_axis_hash != replay.route_id_axis_hash
            or receipt.route_count != replay.route_count
            or receipt.stream_identity_hash != replay.literal_j_stream_source_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.SOURCE_MISMATCH,
                "one raw D1 literal-J attachment left its captured stream",
            )
        if route_windows is None:
            route_windows = receipt.route_windows
        elif route_windows != receipt.route_windows:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
                "D1 attachments do not expose one congruent route-window cover",
            )
        for row, (context, draw, stratum) in enumerate(
            zip(receipt.context_ids, receipt.draw_ids, receipt.stratum_labels, strict=True)
        ):
            locations.append((shard_index, row, context, draw, stratum))
    if route_windows is None:
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.SOURCE_MISMATCH,
            "D1 source has no live route windows",
        )
    ordered = tuple(
        item for stratum in ("target", "complement") for item in locations if item[4] == stratum
    )
    contexts = tuple(item[2] for item in ordered)
    draws = tuple(item[3] for item in ordered)
    strata = tuple(item[4] for item in ordered)
    expected_strata = ("target",) * replay.d1_stratum_counts[0] + (
        "complement",
    ) * replay.d1_stratum_counts[1]
    if (
        len(locations) != replay.context_count
        or
        len(ordered) != replay.context_count
        or len(set(contexts)) != replay.context_count
        or len(set(draws)) != replay.context_count
        or strata != expected_strata
        or _d1_axis_hash("context", contexts) != replay.canonical_context_axis_hash
        or _d1_axis_hash("draw", draws) != replay.canonical_draw_axis_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
            "raw D1 custody does not replay canonical target-then-complement order",
        )
    slices: list[_D1ShardSlice] = []
    for shard_index, attachment in enumerate(source.literal_j_shards):
        selected = tuple((destination, item) for destination, item in enumerate(ordered) if item[0] == shard_index)
        if selected:
            slices.append(
                _D1ShardSlice(
                    attachment=attachment,
                    source_rows=tuple(item[1][1] for item in selected),
                    destination_rows=tuple(item[0] for item in selected),
                )
            )
    source_axis_hash = _source_axis_hash(
        plan_slices=tuple(slices),
        route_windows=route_windows,
        replay=replay,
        compact_receipt_hash=compact.receipt_hash,
    )
    return _D1SourcePlan(
        slices=tuple(slices),
        route_windows=route_windows,
        context_ids=contexts,
        draw_ids=draws,
        strata=strata,
        source_axis_hash=source_axis_hash,
        maximum_registered_window_width=max(stop - start for start, stop in route_windows),
    )


class _D1TileCursor:
    """One sequential pass over every immutable D1 attachment."""

    __slots__ = (
        "_plan", "_tile", "_iterators", "_window", "_offset", "_done",
        "route_window_read_count", "route_tile_read_count", "payload_read_bytes",
    )

    def __init__(self, plan: _D1SourcePlan, route_tile_size: int) -> None:
        self._plan = plan
        self._tile = route_tile_size
        self._iterators = tuple(
            iter(item.attachment.iter_route_windows(verify=True)) for item in plan.slices
        )
        self._window: tuple[int, int, tuple[np.ndarray, ...]] | None = None
        self._offset = 0
        self._done = False
        self.route_window_read_count = 0
        self.route_tile_read_count = 0
        self.payload_read_bytes = 0

    def _read_window(self) -> None:
        if self._window is not None or self._done:
            return
        values: list[tuple[int, int, np.ndarray]] = []
        ended = 0
        for iterator in self._iterators:
            try:
                values.append(next(iterator))
            except StopIteration:
                ended += 1
        if ended:
            if ended != len(self._iterators):
                _refuse(
                    V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
                    "D1 attachment route-window iterators ended asymmetrically",
                )
            self._done = True
            return
        starts = {item[0] for item in values}
        stops = {item[1] for item in values}
        if len(starts) != 1 or len(stops) != 1:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH,
                "D1 attachment route windows lost congruence during replay",
            )
        start, stop = values[0][0], values[0][1]
        self._window = (start, stop, tuple(item[2] for item in values))
        self._offset = 0
        self.route_window_read_count += len(values)
        self.payload_read_bytes += sum(int(item[2].nbytes) for item in values)

    def next(self) -> tuple[int, int, np.ndarray] | None:
        self._read_window()
        if self._done:
            return None
        assert self._window is not None
        start, stop, windows = self._window
        width = min(self._tile, stop - start - self._offset)
        route_start = start + self._offset
        route_stop = route_start + width
        rows = np.empty((len(self._plan.context_ids), width), dtype=np.float64)
        for item, window in zip(self._plan.slices, windows, strict=True):
            rows[np.asarray(item.destination_rows), :] = window[
                np.asarray(item.source_rows), self._offset : self._offset + width
            ]
        if not bool(np.isfinite(rows).all()):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.NONFINITE_SOURCE,
                "one raw D1 literal-J tile contains nonfinite values",
            )
        self._offset += width
        self.route_tile_read_count += 1
        if self._offset == stop - start:
            self._window = None
            self._offset = 0
        return route_start, route_stop, rows


def _stage_rows(rows: np.ndarray, device: object, production: bool) -> Any:
    tensor = torch.from_numpy(np.ascontiguousarray(rows))
    if production:
        return tensor.to(device=device, dtype=torch.float64, non_blocking=False).contiguous()
    return tensor.contiguous()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamHBMForecast:
    design_hash: str
    joint_family_receipt_hash: str
    conditioned_d1_replay_receipt_hash: str
    route_count: int
    context_count: int
    direction_count: int
    shell_size: int
    projected_incremental_peak_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    current_reserved_device_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    hbm_ceiling_fraction: float = 0.85
    context_direction_projection_only: bool = True
    one_direction_route_tile_only: bool = True
    direction_route_bank_forecasted: bool = False
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_stream_hbm_v1"
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError("D1 stream forecasts require their issuer")
        for name in ("design_hash", "joint_family_receipt_hash", "conditioned_d1_replay_receipt_hash"):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count", "context_count", "direction_count", "shell_size",
            "projected_incremental_peak_bytes", "device_total_memory_bytes",
            "device_free_memory_bytes", "hbm_ceiling_bytes",
        ):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.current_reserved_device_bytes, name="current_reserved_device_bytes")
        if (
            self.production_cuda_execution == self.private_cpu_reference_execution
            or type(self.admitted) is not bool
            or self.hbm_ceiling_fraction != 0.85
            or any(
                type(item) is not bool or item is not True
                for item in (self.context_direction_projection_only, self.one_direction_route_tile_only)
            )
            or self.direction_route_bank_forecasted is not False
            or self.admitted
            != (
                allocated + self.projected_incremental_peak_bytes
                <= self.hbm_ceiling_bytes
                and self.projected_incremental_peak_bytes <= available
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_DESIGN,
                "D1 stream HBM forecast does not replay",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="forecast")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 stream forecast mutated")
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()


def _forecast(
    *, design: V1LiteralJJointAffineShellD1PopulationStreamDesign,
    family_receipt: Any, replay: Any, device: object, production: bool,
    device_total_memory_bytes: int | None, device_free_memory_bytes: int | None,
    current_reserved_device_bytes: int | None,
) -> V1LiteralJJointAffineShellD1PopulationStreamHBMForecast:
    routes = replay.route_count
    contexts = replay.context_count
    directions = family_receipt.direction_count
    shell = family_receipt.shell_size
    tile = min(routes, design.route_tile_size)
    actions = min(family_receipt.action_count, design.action_tile_size)
    retained = (4 * contexts * directions + 4 * directions + 4 * directions * shell) * _F64_BYTES
    pass1 = retained + (contexts * tile + actions * tile) * _F64_BYTES
    pass2 = retained + (contexts * tile + 4 * directions * tile) * _F64_BYTES
    peak = max(pass1, pass2)
    if production:
        if any(value is not None for value in (device_total_memory_bytes, device_free_memory_bytes, current_reserved_device_bytes)):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "production HBM facts must come from the live CUDA device")
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        admission_memory = (cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(device)))
    else:
        total = 1_000_000_000 if device_total_memory_bytes is None else _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
        free = total if device_free_memory_bytes is None else _positive_int(device_free_memory_bytes, name="device_free_memory_bytes")
        reserved = 0 if current_reserved_device_bytes is None else _nonnegative_int(current_reserved_device_bytes, name="current_reserved_device_bytes")
    if not production:
        admission_memory = (int(free), int(reserved))
    available, allocated = admission_memory
    ceiling = math.floor(int(total) * 0.85)
    return V1LiteralJJointAffineShellD1PopulationStreamHBMForecast(
        design_hash=design.design_hash,
        joint_family_receipt_hash=family_receipt.receipt_hash,
        conditioned_d1_replay_receipt_hash=replay.receipt_hash,
        route_count=routes, context_count=contexts, direction_count=directions,
        shell_size=shell, projected_incremental_peak_bytes=peak,
        device_total_memory_bytes=int(total), device_free_memory_bytes=int(free),
        current_reserved_device_bytes=int(reserved), hbm_ceiling_bytes=ceiling,
        admitted=allocated + peak <= ceiling and peak <= available,
        production_cuda_execution=production, private_cpu_reference_execution=not production,
        _admission_memory=admission_memory,
        _issuance_token=_FORECAST_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamReceipt:
    design_hash: str
    forecast_hash: str
    joint_family_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_postsource_custody_axis_hash: str
    conditioned_d1_replay_receipt_hash: str
    d1_materialization_hash: str
    d1_row_prefix_manifest_hash: str
    d1_original_capture_binding_hash: str
    compact_source_receipt_hash: str
    source_axis_hash: str
    literal_j_shard_receipt_hashes: tuple[str, ...]
    literal_j_content_sha256s: tuple[str, ...]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d1_capture_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    joint_registered_d_prefix_hash: str
    d1_literal_j_stream_source_hash: str
    joint_literal_j_stream_source_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    preoutcome_registration_hashes: tuple[str, ...]
    destination_binding_hashes: tuple[str, ...]
    action_count: int
    shell_size: int
    direction_count: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    d1_stratum_counts: tuple[int, int]
    context_count: int
    population_count: int
    exact_internal_d1_weight_fraction: tuple[int, int]
    unapplied_d1_population_mass_fraction: tuple[int, int]
    pass1_route_window_read_count: int
    pass1_route_tile_read_count: int
    pass1_action_tile_submission_count: int
    pass1_shared_shell_direct_gather_count: int
    pass1_source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    logical_joint_d1_replay_count: int = 1
    source_pass_count: int = 2
    per_action_d1_replay_count: int = 0
    provider_call_count: int = 0
    additional_model_forward_count: int = 0
    additional_vjp_count: int = 0
    alpha_spent: float = 0.0
    d1_empirical_mean_authority: bool = True
    exact_preoutcome_action_axis_bound: bool = True
    full_population_geometry_authority: bool = False
    population_mass_applied: bool = False
    residual_rows_combined: bool = False
    context_route_bank_retained: bool = False
    direction_route_bank_retained: bool = False
    route_scale_d2h: bool = False
    d_decision_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("D1 stream receipts require their issuer")
        for name in (
            "design_hash", "forecast_hash", "joint_family_receipt_hash",
            "joint_preoutcome_registration_hash", "joint_postsource_custody_axis_hash",
            "conditioned_d1_replay_receipt_hash", "compact_source_receipt_hash",
            "d1_materialization_hash", "d1_row_prefix_manifest_hash",
            "d1_original_capture_binding_hash",
            "source_axis_hash", "model_identity_hash", "chart_identity_hash",
            "current_mask_hash", "route_axis_hash", "route_id_axis_hash",
            "registered_d1_capture_prefix_hash", "d1_pair_prefix_axis_hash",
            "joint_registered_d_prefix_hash", "d1_literal_j_stream_source_hash",
            "joint_literal_j_stream_source_hash", "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        action_axes = (
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
            self.preoutcome_registration_hashes,
            self.destination_binding_hashes,
        )
        shards = _identity_axis(self.literal_j_shard_receipt_hashes, name="literal_j_shard_receipt_hashes")
        contents = _identity_axis(self.literal_j_content_sha256s, name="literal_j_content_sha256s")
        if (
            len(shards) != len(contents)
            or any(
                len(_identity_axis(axis, name="physical_action_axis"))
                != self.action_count
                for axis in action_axes
            )
            or self.direction_count != self.action_count + self.shell_size
            or len(ids) != self.direction_count
            or len(self.shell_indices) != self.shell_size
            or len(self.shell_route_ids) != self.shell_size
            or self.context_count != sum(self.d1_stratum_counts)
            or self.exact_internal_d1_weight_fraction != (1, self.context_count)
            or Fraction(*self.unapplied_d1_population_mass_fraction)
            != Fraction(self.context_count, self.population_count)
            or self.joint_registered_d_prefix_hash != self.d1_pair_prefix_axis_hash
            or self.logical_joint_d1_replay_count != 1
            or self.source_pass_count != 2
            or self.per_action_d1_replay_count != 0
            or any(value != 0 for value in (self.provider_call_count, self.additional_model_forward_count, self.additional_vjp_count))
            or self.alpha_spent != 0.0
            or self.pass1_source_payload_read_bytes
            != self.context_count * self.route_count * _F64_BYTES
            or self.production_cuda_execution
            == self.private_cpu_reference_execution
            or self.host_to_device_payload_bytes
            != (
                self.pass1_source_payload_read_bytes
                if self.production_cuda_execution
                else 0
            )
            or any(
                type(item) is not bool or item is not True
                for item in (
                    self.d1_empirical_mean_authority,
                    self.exact_preoutcome_action_axis_bound,
                )
            )
            or any(
                type(item) is not bool or item is not False
                for item in (
                    self.full_population_geometry_authority, self.population_mass_applied,
                    self.residual_rows_combined, self.context_route_bank_retained,
                    self.direction_route_bank_retained, self.route_scale_d2h,
                    self.d_decision_authority, self.c_stream_authority,
                    self.calibration_authority, self.oracle_authority,
                    self.commit_authority, self.terminal_authority,
                    self.performance_authority, self.pareto_authority,
                )
            )
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_SCHEMA
        ):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "D1 stream receipt crossed its conditional-only authority")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 stream receipt no longer replays")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamQReceipt:
    stream_receipt_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    context_count: int
    exact_internal_d1_weight_fraction: tuple[int, int]
    direction_count: int
    d1_only: bool = True
    full_population_authority: bool = False
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_Q_receipt_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _Q_TOKEN:
            raise PermissionError("D1 Q receipts require their issuer")
        _identity(self.stream_receipt_hash, name="stream_receipt_hash")
        _identity(self.joint_direction_axis_hash, name="joint_direction_axis_hash")
        ids = _identity_axis(self.direction_ids, name="direction_ids")
        if len(ids) != self.direction_count or self.exact_internal_d1_weight_fraction != (1, self.context_count) or self.d1_only is not True or self.full_population_authority is not False:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "D1 Q receipt changed authority or axis")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="q-receipt")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 Q receipt mutated")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class _IntervalLease:
    __slots__ = ("_owner", "_token", "_closed")

    def __init__(self, owner: "_IntervalHandle", token: object) -> None:
        self._owner = owner
        self._token = token
        self._closed = False

    def intervals(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.EVICTED, "interval lease is closed")
        return self._owner._require_live()

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __enter__(self) -> "_IntervalLease":
        self.intervals()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live interval leases are nonserializable")


class _IntervalHandle:
    __slots__ = ("receipt", "_center", "_radius", "_shape", "_guards", "_invariants", "_tokens")

    def __init__(self, *, receipt: Any, center: Any, radius: Any, shape: tuple[int, ...], token: object) -> None:
        if token not in (_Q_TOKEN, _SHELL_TOKEN):
            raise PermissionError("interval handles require their issuer")
        receipt.assert_integrity()
        center = _require_tensor(center, name="interval_center", shape=shape)
        radius = _require_tensor(radius, name="interval_radius", shape=shape, device=center.device, nonnegative=True)
        self.receipt = receipt
        self._center = center
        self._radius = radius
        self._shape = shape
        self._guards = (center.detach().clone(), radius.detach().clone())
        self._invariants = (_tensor_invariant(center), _tensor_invariant(radius))
        self._tokens: set[object] = set()
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        if self._center is None:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.EVICTED, "interval tensors were evicted")
        values = (self._center, self._radius)
        if any(_tensor_invariant(value) != invariant or not bool(torch.equal(value, guard)) for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "interval tensors mutated")
        self.receipt.assert_integrity()
        return values

    def borrow(self) -> _IntervalLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return _IntervalLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY, "interval lease token is stale")
        self._require_live()
        self._tokens.remove(token)

    @property
    def is_evicted(self) -> bool:
        return self._center is None

    def evict(self) -> None:
        self._require_live()
        if self._tokens:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_LEASE, "cannot evict interval tensors while leased")
        self._center = self._radius = None
        self._guards = self._invariants = ()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live interval handles are nonserializable")


class V1LiteralJJointAffineShellD1PopulationStreamQLease(_IntervalLease):
    pass


class V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle(_IntervalHandle):
    def borrow(self) -> V1LiteralJJointAffineShellD1PopulationStreamQLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellD1PopulationStreamQLease(self, token)


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamCompactShellMReceipt:
    stream_receipt_hash: str
    q_receipt_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    context_count: int
    direction_count: int
    shell_size: int
    exact_internal_d1_weight_fraction: tuple[int, int]
    d1_only: bool = True
    full_population_authority: bool = False
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_compact_shell_M_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SHELL_TOKEN:
            raise PermissionError("D1 compact shell-M receipts require their issuer")
        for name in ("stream_receipt_hash", "q_receipt_hash", "joint_direction_axis_hash"):
            _identity(getattr(self, name), name=name)
        if len(_identity_axis(self.direction_ids, name="direction_ids")) != self.direction_count or len(self.shell_indices) != self.shell_size or len(_identity_axis(self.shell_route_ids, name="shell_route_ids")) != self.shell_size or self.exact_internal_d1_weight_fraction != (1, self.context_count) or self.d1_only is not True or self.full_population_authority is not False:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "D1 compact shell-M receipt changed axis or authority")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="compact-shell-m-receipt")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 compact shell-M receipt mutated")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellD1PopulationStreamCompactShellMLease(_IntervalLease):
    pass


class V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle(_IntervalHandle):
    def borrow(
        self,
    ) -> V1LiteralJJointAffineShellD1PopulationStreamCompactShellMLease:
        self._require_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellD1PopulationStreamCompactShellMLease(
            self, token
        )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamTileReceipt:
    stream_receipt_hash: str
    q_receipt_hash: str
    compact_shell_m_receipt_hash: str
    source_axis_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    tile_ordinal: int
    route_start: int
    route_stop: int
    route_count: int
    direction_count: int
    context_count: int
    exact_internal_d1_weight_fraction: tuple[int, int]
    one_live_tile_only: bool = True
    d1_only: bool = True
    full_population_authority: bool = False
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_stream_tile_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("D1 stream tile receipts require their issuer")
        for name in ("stream_receipt_hash", "q_receipt_hash", "compact_shell_m_receipt_hash", "source_axis_hash", "joint_direction_axis_hash"):
            _identity(getattr(self, name), name=name)
        if len(_identity_axis(self.direction_ids, name="direction_ids")) != self.direction_count or not 0 <= self.route_start < self.route_stop <= self.route_count or self.exact_internal_d1_weight_fraction != (1, self.context_count) or self.one_live_tile_only is not True or self.d1_only is not True or self.full_population_authority is not False:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "D1 tile receipt changed its route or authority")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="tile-receipt")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 tile receipt mutated")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellD1PopulationStreamTileLease:
    __slots__ = ("_owner", "_token", "_closed")

    def __init__(self, owner: "V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle", token: object) -> None:
        self._owner = owner
        self._token = token
        self._closed = False

    def intervals(self) -> tuple[Any, Any]:
        if self._closed:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.EVICTED, "D1 tile lease is closed")
        return self._owner._require_live()

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __enter__(self) -> "V1LiteralJJointAffineShellD1PopulationStreamTileLease":
        self.intervals()
        return self

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False


class V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle:
    __slots__ = ("receipt", "_center", "_radius", "_guards", "_invariants", "_tokens", "_release")

    def __init__(self, *, receipt: V1LiteralJJointAffineShellD1PopulationStreamTileReceipt, center: Any, radius: Any, release: Any, issuance_token: object | None = None) -> None:
        if issuance_token is not _TILE_TOKEN:
            raise PermissionError("D1 tile handles require their issuer")
        shape = (receipt.direction_count, receipt.route_stop - receipt.route_start)
        self.receipt = receipt
        self._center = _require_tensor(center, name="mjv_center_tile", shape=shape)
        self._radius = _require_tensor(radius, name="mjv_radius_tile", shape=shape, device=self._center.device, nonnegative=True)
        self._guards = (self._center.detach().clone(), self._radius.detach().clone())
        self._invariants = (_tensor_invariant(self._center), _tensor_invariant(self._radius))
        self._tokens: set[object] = set()
        self._release = release
        self._require_live()

    def _require_live(self) -> tuple[Any, Any]:
        if self._center is None:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.EVICTED, "D1 Mv tile was evicted")
        values = (self._center, self._radius)
        if any(_tensor_invariant(value) != invariant or not bool(torch.equal(value, guard)) for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 Mv tile mutated")
        self.receipt.assert_integrity()
        return values

    def borrow(self) -> V1LiteralJJointAffineShellD1PopulationStreamTileLease:
        self._require_live()
        token = object(); self._tokens.add(token)
        return V1LiteralJJointAffineShellD1PopulationStreamTileLease(self, token)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY, "D1 tile lease token is stale")
        self._require_live(); self._tokens.remove(token)

    @property
    def is_evicted(self) -> bool:
        return self._center is None

    def evict(self) -> None:
        self._require_live()
        if self._tokens:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_LEASE, "cannot evict D1 tile while leased")
        receipt_hash = self.receipt.receipt_hash
        self._center = self._radius = None
        self._guards = self._invariants = ()
        self._release(receipt_hash)


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamExecutionReceipt:
    stream_receipt_hash: str
    source_axis_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    route_count: int
    context_count: int
    direction_count: int
    exact_internal_d1_weight_fraction: tuple[int, int]
    tile_receipt_hashes: tuple[str, ...]
    route_ranges: tuple[tuple[int, int], ...]
    pass2_route_window_read_count: int
    pass2_route_tile_read_count: int
    pass2_source_payload_read_bytes: int
    total_source_payload_read_bytes: int
    source_pass_count: int = 2
    logical_joint_d1_replay_count: int = 1
    one_live_tile_enforced: bool = True
    full_route_cover: bool = True
    d1_only: bool = True
    full_population_authority: bool = False
    schema: str = "partition_free_v1_literal_J_joint_affine_shell_D1_stream_execution_v1"
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("D1 execution receipts require their issuer")
        for name in ("stream_receipt_hash", "source_axis_hash", "joint_direction_axis_hash"):
            _identity(getattr(self, name), name=name)
        tiles = tuple(_identity(item, name="tile_receipt_hash") for item in self.tile_receipt_hashes)
        if len(_identity_axis(self.direction_ids, name="direction_ids")) != self.direction_count or len(tiles) != len(self.route_ranges) or len(set(tiles)) != len(tiles) or not self.route_ranges or self.route_ranges[0][0] != 0 or self.route_ranges[-1][1] != self.route_count or any(self.route_ranges[i][1] != self.route_ranges[i + 1][0] for i in range(len(self.route_ranges) - 1)) or self.exact_internal_d1_weight_fraction != (1, self.context_count) or self.source_pass_count != 2 or self.logical_joint_d1_replay_count != 1 or self.total_source_payload_read_bytes != 2 * self.context_count * self.route_count * _F64_BYTES or self.one_live_tile_enforced is not True or self.full_route_cover is not True or self.d1_only is not True or self.full_population_authority is not False:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AUTHORITY_VIOLATION, "D1 execution receipt changed cover or authority")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return _payload_hash(MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}), role="execution-receipt")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "D1 execution receipt mutated")
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle:
    __slots__ = (
        "receipt", "forecast", "q_handle", "compact_shell_m_handle", "_source",
        "_plan", "_design", "_joint_lease", "_projection", "_projection_radius",
        "_preregistration",
        "_projection_guards", "_projection_invariants", "_cursor", "_active_tile_hash",
        "_tile_hashes", "_route_ranges", "_exhausted", "_final_receipt", "_evicted",
        "_population_route_claim_token", "_population_route_claim_consumed",
        "_public_route_stream_started",
    )

    def __init__(
        self, *, receipt: V1LiteralJJointAffineShellD1PopulationStreamReceipt,
        forecast: V1LiteralJJointAffineShellD1PopulationStreamHBMForecast,
        q_handle: V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle,
        compact_shell_m_handle: V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle,
        source: V1LiteralJConditionedD1ShardReplaySource, plan: _D1SourcePlan,
        design: V1LiteralJJointAffineShellD1PopulationStreamDesign,
        preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
        joint_lease: V1LiteralJJointAffineShellFamilyLease,
        projection: Any, projection_radius: Any, issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("D1 stream handles require their issuer")
        self.receipt = receipt; self.forecast = forecast; self.q_handle = q_handle
        self.compact_shell_m_handle = compact_shell_m_handle
        self._source = source; self._plan = plan; self._design = design
        self._preregistration = preregistration; self._joint_lease = joint_lease
        self._projection = projection; self._projection_radius = projection_radius
        self._projection_guards = (projection.detach().clone(), projection_radius.detach().clone())
        self._projection_invariants = (_tensor_invariant(projection), _tensor_invariant(projection_radius))
        self._cursor = _D1TileCursor(plan, design.route_tile_size)
        self._active_tile_hash: str | None = None
        self._tile_hashes: list[str] = []; self._route_ranges: list[tuple[int, int]] = []
        self._exhausted = False; self._final_receipt: V1LiteralJJointAffineShellD1PopulationStreamExecutionReceipt | None = None
        self._population_route_claim_token: object | None = None
        self._population_route_claim_consumed = False
        self._public_route_stream_started = False
        self._evicted = False; self.assert_live()

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.EVICTED, "D1 stream was evicted")
        try:
            self.receipt.assert_integrity(); self.forecast.assert_integrity()
            self._preregistration.assert_integrity(
                self.receipt.joint_preoutcome_registration_hash
            )
            self.q_handle._require_live(); self.compact_shell_m_handle._require_live()
            self._joint_lease.projection_custody()
            if (
                self._joint_lease.receipt_hash
                != self.receipt.joint_family_receipt_hash
                or self._preregistration.joint_direction_axis_hash
                != self.receipt.joint_direction_axis_hash
                or self._preregistration.joint_direction_ids
                != self.receipt.direction_ids
                or self._preregistration.physical_action_identity_hashes
                != self.receipt.physical_action_identity_hashes
                or self._preregistration.physical_action_hashes
                != self.receipt.physical_action_hashes
                or self._preregistration.packed_action_hashes
                != self.receipt.packed_action_hashes
                or self._preregistration.preoutcome_registration_hashes
                != self.receipt.preoutcome_registration_hashes
                or self._preregistration.destination_binding_hashes
                != self.receipt.destination_binding_hashes
            ):
                _refuse(
                    V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH,
                    "retained preoutcome and postsource action custody changed",
                )
            replay = self._source.replay_receipt
            replay.assert_integrity(
                self.receipt.conditioned_d1_replay_receipt_hash
            )
            if (
                self._source.compact_source_receipt.receipt_hash
                != self.receipt.compact_source_receipt_hash
                or replay.chart_identity_hash != self.receipt.chart_identity_hash
                or replay.current_mask_hash != self.receipt.current_mask_hash
                or replay.proposal_generation != self.receipt.proposal_generation
                or replay.route_axis_hash != self.receipt.route_axis_hash
                or replay.route_id_axis_hash != self.receipt.route_id_axis_hash
                or replay.route_count != self.receipt.route_count
                or replay.registered_d_prefix_hash
                != self.receipt.registered_d1_capture_prefix_hash
                or replay.d1_pair_prefix_axis_hash
                != self.receipt.d1_pair_prefix_axis_hash
                or replay.literal_j_stream_source_hash
                != self.receipt.d1_literal_j_stream_source_hash
                or replay.materialization_hash != self.receipt.d1_materialization_hash
                or replay.row_prefix_manifest_hash
                != self.receipt.d1_row_prefix_manifest_hash
                or replay.original_capture_binding_hash
                != self.receipt.d1_original_capture_binding_hash
                or tuple(
                    item.receipt.receipt_hash
                    for item in self._source.literal_j_shards
                )
                != self.receipt.literal_j_shard_receipt_hashes
                or tuple(
                    item.receipt.content_sha256
                    for item in self._source.literal_j_shards
                )
                != self.receipt.literal_j_content_sha256s
            ):
                _refuse(
                    V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                    "retained D1 source left the frozen receipt axes",
                )
            for item in self._source.literal_j_shards:
                shard = item.receipt
                item.verify_live_witnesses(
                    expected_role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
                    expected_chart_identity_hash=self.receipt.chart_identity_hash,
                    expected_current_mask_hash=self.receipt.current_mask_hash,
                    expected_proposal_generation=self.receipt.proposal_generation,
                )
                if (
                    shard.registered_d_prefix_hash
                    != self.receipt.registered_d1_capture_prefix_hash
                    or shard.route_axis_hash != self.receipt.route_axis_hash
                    or shard.route_id_axis_hash != self.receipt.route_id_axis_hash
                    or shard.route_count != self.receipt.route_count
                    or shard.stream_identity_hash
                    != self.receipt.d1_literal_j_stream_source_hash
                    or shard.route_windows != self._plan.route_windows
                ):
                    _refuse(
                        V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                        "one retained D1 shard left the frozen chart/prefix/J axes",
                    )
            if (
                _source_axis_hash(
                    plan_slices=self._plan.slices,
                    route_windows=self._plan.route_windows,
                    replay=replay,
                    compact_receipt_hash=(
                        self._source.compact_source_receipt.receipt_hash
                    ),
                )
                != self.receipt.source_axis_hash
            ):
                _refuse(
                    V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                    "retained D1 source-plan hash no longer replays",
                )
        except V1LiteralJJointAffineShellD1PopulationStreamRefusal:
            raise
        except Exception as error:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, f"one retained D1 stream parent no longer replays: {type(error).__name__}: {error}")
        values = (self._projection, self._projection_radius)
        if any(_tensor_invariant(value) != invariant or not bool(torch.equal(value, guard)) for value, invariant, guard in zip(values, self._projection_invariants, self._projection_guards, strict=True)):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.MUTATED_CUSTODY, "retained D1 projections mutated")

    @property
    def exhausted(self) -> bool:
        self.assert_live(); return self._exhausted

    def _next_tile_unchecked(
        self,
    ) -> V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle | None:
        self.assert_live()
        if self._exhausted:
            return None
        try:
            raw = self._cursor.next()
        except V1LiteralJJointAffineShellD1PopulationStreamRefusal:
            raise
        except Exception as error:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.SOURCE_MISMATCH, f"D1 pass-2 source replay failed: {type(error).__name__}: {error}")
        if raw is None:
            self._exhausted = True
            return None
        start, stop, host_rows = raw
        device = self._projection.device
        rows = _stage_rows(host_rows, device, self.receipt.host_to_device_payload_bytes > 0)
        count = self.receipt.context_count
        center = (self._projection.T @ rows / float(count)).contiguous()
        abs_mass = self._projection.abs().T @ rows.abs()
        propagated = self._projection_radius.T @ rows.abs()
        radius = _outward_nonnegative(
            (propagated + _sum_radius(abs_mass, max(1, 2 * count + 1))) / float(count)
            + _division_radius(center)
        ).contiguous()
        receipt = V1LiteralJJointAffineShellD1PopulationStreamTileReceipt(
            stream_receipt_hash=self.receipt.receipt_hash,
            q_receipt_hash=self.q_handle.receipt.receipt_hash,
            compact_shell_m_receipt_hash=self.compact_shell_m_handle.receipt.receipt_hash,
            source_axis_hash=self._plan.source_axis_hash,
            joint_direction_axis_hash=self.receipt.joint_direction_axis_hash,
            direction_ids=self.receipt.direction_ids,
            tile_ordinal=len(self._tile_hashes), route_start=start, route_stop=stop,
            route_count=self.receipt.route_count, direction_count=self.receipt.direction_count,
            context_count=count, exact_internal_d1_weight_fraction=(1, count),
            _issuance_token=_TILE_TOKEN,
        )
        receipt_hash = receipt.receipt_hash
        self._active_tile_hash = receipt_hash; self._tile_hashes.append(receipt_hash); self._route_ranges.append((start, stop))
        return V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle(
            receipt=receipt, center=center, radius=radius, release=self._release_tile,
            issuance_token=_TILE_TOKEN,
        )

    def next_tile(
        self,
    ) -> V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle | None:
        self.assert_live()
        if self._population_route_claim_token is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ROUTE_STREAM_CLAIMED,
                "the D1 route stream is exclusively claimed by population-Mv",
            )
        if self._population_route_claim_consumed:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ROUTE_STREAM_CONSUMED,
                "the D1 route stream was already consumed by population-Mv",
            )
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE,
                "evict the active D1 tile before advancing",
            )
        self._public_route_stream_started = True
        return self._next_tile_unchecked()

    def _claim_unopened_route_stream_for_population_mjv(self) -> object:
        """Exclusively reserve the still-unopened pass-2 source stream."""

        self.assert_live()
        if (
            self._population_route_claim_token is not None
            or self._population_route_claim_consumed
            or self._public_route_stream_started
            or self._active_tile_hash is not None
            or self._cursor.route_tile_read_count != 0
            or self._exhausted
        ):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ROUTE_STREAM_CONSUMED,
                "D1 pass 2 is not one unopened exclusively claimable stream",
            )
        token = object()
        self._population_route_claim_token = token
        return token

    def _assert_population_mjv_route_claim(self, token: object) -> None:
        self.assert_live()
        if token is not self._population_route_claim_token:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                "population-Mv D1 route-stream claim token is stale",
            )

    def _next_claimed_population_mjv_tile(
        self,
        token: object,
    ) -> V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle | None:
        self._assert_population_mjv_route_claim(token)
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE,
                "evict the claimed D1 tile before advancing",
            )
        return self._next_tile_unchecked()

    def _release_population_mjv_route_claim(
        self,
        token: object,
        *,
        source_tiles_consumed: bool,
    ) -> None:
        self._assert_population_mjv_route_claim(token)
        if type(source_tiles_consumed) is not bool:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
                "source_tiles_consumed must be one exact boolean",
            )
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE,
                "cannot release a population-Mv claim with one live D1 tile",
            )
        observed_consumption = self._cursor.route_tile_read_count > 0
        if observed_consumption != source_tiles_consumed:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                "population-Mv consumption declaration differs from source custody",
            )
        self._population_route_claim_token = None
        if source_tiles_consumed:
            self._population_route_claim_consumed = True

    def _abort_population_mjv_route_claim(
        self,
        token: object,
        *,
        source_tiles_consumed: bool,
    ) -> None:
        """Burn a partially consumed claimed route stream during cleanup.

        This is intentionally weaker than successful release: it clears the
        live exclusive claim so callers cannot strand this producer, while
        preserving irreversibility by marking any observed consumption as
        non-retryable.  It never grants finalization or replay authority.
        """

        self._assert_population_mjv_route_claim(token)
        if type(source_tiles_consumed) is not bool:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE,
                "source_tiles_consumed must be one exact boolean",
            )
        if self._active_tile_hash is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE,
                "cannot abort a population-Mv claim with one live D1 tile",
            )
        observed_consumption = self._cursor.route_tile_read_count > 0
        if source_tiles_consumed != observed_consumption:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY,
                "population-Mv abort declaration differs from source custody",
            )
        self._population_route_claim_token = None
        if source_tiles_consumed:
            self._population_route_claim_consumed = True

    def _release_tile(self, receipt_hash: str) -> None:
        if self._active_tile_hash != receipt_hash:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STALE_IDENTITY, "released D1 tile is not active")
        self._active_tile_hash = None

    def finalize(self) -> V1LiteralJJointAffineShellD1PopulationStreamExecutionReceipt:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE, "cannot finalize with an active D1 tile")
        if not self._exhausted:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.STREAM_NOT_EXHAUSTED, "consume pass 2 through its terminal None before finalization")
        if self._final_receipt is None:
            self._final_receipt = V1LiteralJJointAffineShellD1PopulationStreamExecutionReceipt(
                stream_receipt_hash=self.receipt.receipt_hash,
                source_axis_hash=self._plan.source_axis_hash,
                joint_direction_axis_hash=self.receipt.joint_direction_axis_hash,
                direction_ids=self.receipt.direction_ids,
                route_count=self.receipt.route_count, context_count=self.receipt.context_count,
                direction_count=self.receipt.direction_count,
                exact_internal_d1_weight_fraction=(1, self.receipt.context_count),
                tile_receipt_hashes=tuple(self._tile_hashes), route_ranges=tuple(self._route_ranges),
                pass2_route_window_read_count=self._cursor.route_window_read_count,
                pass2_route_tile_read_count=self._cursor.route_tile_read_count,
                pass2_source_payload_read_bytes=self._cursor.payload_read_bytes,
                total_source_payload_read_bytes=self.receipt.pass1_source_payload_read_bytes + self._cursor.payload_read_bytes,
                _issuance_token=_EXECUTION_TOKEN,
            )
        return self._final_receipt

    def evict(self) -> None:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_TILE, "cannot evict D1 stream with active tile")
        if self._population_route_claim_token is not None:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ROUTE_STREAM_CLAIMED,
                "cannot evict D1 stream while population-Mv holds its route claim",
            )
        if self.q_handle._tokens or self.compact_shell_m_handle._tokens:
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.ACTIVE_LEASE,
                "cannot evict D1 stream while Q or compact shell-M is leased",
            )
        try:
            self.compact_shell_m_handle.evict(); self.q_handle.evict(); self._joint_lease.close()
        finally:
            self._projection = self._projection_radius = None
            self._projection_guards = self._projection_invariants = ()
            self._source = self._plan = self._cursor = self._joint_lease = None
            self._preregistration = None
            self._evicted = True

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live D1 stream handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1PopulationStreamResult:
    receipt: V1LiteralJJointAffineShellD1PopulationStreamReceipt
    stream_handle: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle = field(repr=False, compare=False)

    @property
    def q_handle(self) -> V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle:
        return self.stream_handle.q_handle

    @property
    def compact_shell_m_handle(self) -> V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle:
        return self.stream_handle.compact_shell_m_handle


def _execute(
    *, conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    design: V1LiteralJJointAffineShellD1PopulationStreamDesign,
    production: bool,
    device_total_memory_bytes: int | None,
    device_free_memory_bytes: int | None,
    current_reserved_device_bytes: int | None,
) -> V1LiteralJJointAffineShellD1PopulationStreamResult:
    if torch is None or type(design) is not V1LiteralJJointAffineShellD1PopulationStreamDesign:
        _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.INVALID_TYPE, "D1 stream requires torch and its exact design")
    design.assert_integrity()
    expected_mode = V1LiteralJJointAffineShellD1PopulationStreamExecutionMode.CUDA_PRODUCTION if production else V1LiteralJJointAffineShellD1PopulationStreamExecutionMode.CPU_REFERENCE_TEST_ONLY
    if design.execution_mode is not expected_mode:
        _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.CPU_PRODUCTION_FALLBACK, "D1 stream execution mode differs from the entry point")
    plan = _build_source_plan(
        conditioned_d1,
        joint_family_handle,
        joint_preoutcome_registration,
    )
    replay = conditioned_d1.replay_receipt
    family_receipt = joint_family_handle.receipt
    joint_lease: V1LiteralJJointAffineShellFamilyLease | None = None
    q_handle: V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle | None = None
    shell_handle: V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle | None = None
    try:
        joint_lease = joint_family_handle.borrow()
        outside, shell_tensor = joint_lease.projection_custody()
        device = outside[0].device
        if production:
            if family_receipt.execution_mode != V1LiteralJJointAffineShellFamilyExecutionMode.CUDA_PRODUCTION.value or device.type != "cuda":
                _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.CPU_PRODUCTION_FALLBACK, "production D1 stream requires a CUDA joint family")
            if not torch.are_deterministic_algorithms_enabled():
                _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DETERMINISM_DISABLED, "production D1 stream requires deterministic algorithms")
        elif family_receipt.execution_mode != V1LiteralJJointAffineShellFamilyExecutionMode.CPU_REFERENCE_TEST_ONLY.value or device.type != "cpu" or replay.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.CPU_PRODUCTION_FALLBACK, "private D1 reference requires a bounded CPU joint family")
        if any(value.device != device or value.dtype != torch.float64 or tuple(value.shape) != (replay.route_count,) for value in outside):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH, "outside directions left their common FP64 route axis")
        if shell_tensor.device != device or tuple(int(item) for item in shell_tensor.shape) != (family_receipt.shell_size,):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH, "shared shell tensor left its compact axis")
        expected_shell = torch.tensor(
            family_receipt.shell_indices,
            dtype=shell_tensor.dtype,
            device=device,
        )
        if not bool(torch.equal(shell_tensor, expected_shell)):
            _refuse(
                V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.DIRECTION_CUSTODY_MISMATCH,
                "live shared-shell tensor differs from the frozen shell axis",
            )
        forecast = _forecast(
            design=design, family_receipt=family_receipt, replay=replay, device=device,
            production=production, device_total_memory_bytes=device_total_memory_bytes,
            device_free_memory_bytes=device_free_memory_bytes,
            current_reserved_device_bytes=current_reserved_device_bytes,
        )
        if not forecast.admitted:
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.HBM_FORECAST_EXCEEDED, "D1 pass-1/pass-2 live set exceeds the 85% memory ceiling")
        d = replay.context_count; directions = family_receipt.direction_count
        projection = torch.zeros((d, directions), dtype=torch.float64, device=device)
        abs_mass = torch.zeros_like(projection)
        cursor = _D1TileCursor(plan, design.route_tile_size)
        shell_seen = [False] * family_receipt.shell_size
        action_submissions = 0; shell_gathers = 0
        while True:
            raw = cursor.next()
            if raw is None:
                break
            start, stop, host_rows = raw
            rows = _stage_rows(host_rows, device, production)
            for action_start in range(0, family_receipt.action_count, design.action_tile_size):
                action_stop = min(family_receipt.action_count, action_start + design.action_tile_size)
                direction_tile = torch.stack(tuple(outside[index][start:stop] for index in range(action_start, action_stop)), dim=0).contiguous()
                projection[:, action_start:action_stop].add_(rows @ direction_tile.T)
                abs_mass[:, action_start:action_stop].add_(rows.abs() @ direction_tile.abs().T)
                action_submissions += 1
            gathered = False
            for shell_ordinal, route in enumerate(family_receipt.shell_indices):
                if start <= route < stop:
                    projection[:, family_receipt.action_count + shell_ordinal].copy_(rows[:, route - start])
                    abs_mass[:, family_receipt.action_count + shell_ordinal].copy_(rows[:, route - start].abs())
                    shell_seen[shell_ordinal] = True; gathered = True
            if gathered:
                shell_gathers += 1
        if not all(shell_seen):
            _refuse(V1LiteralJJointAffineShellD1PopulationStreamRefusalCode.AXIS_MISMATCH, "pass 1 did not encounter every registered shell coordinate")
        projection_radius = _sum_radius(abs_mass, max(1, 2 * replay.route_count + 2 * cursor.route_tile_read_count + 1))
        projection_radius[:, family_receipt.action_count:].zero_()
        q_center = torch.mean(projection * projection, dim=0).contiguous()
        q_propagated = torch.sum(2.0 * projection.abs() * projection_radius + projection_radius * projection_radius, dim=0)
        q_rounding = _sum_radius(torch.sum(projection.abs() * projection.abs(), dim=0), max(1, 2 * d + 1))
        q_radius = _outward_nonnegative((q_propagated + q_rounding) / float(d) + _division_radius(q_center)).contiguous()
        shell_projection = projection[:, family_receipt.action_count:]
        shell_center = (projection.T @ shell_projection / float(d)).contiguous()
        shell_abs_mass = projection.abs().T @ shell_projection.abs()
        shell_propagated = projection_radius.T @ shell_projection.abs()
        shell_radius = _outward_nonnegative((shell_propagated + _sum_radius(shell_abs_mass, max(1, 2 * d + 1))) / float(d) + _division_radius(shell_center)).contiguous()
        mass = Fraction(d, replay.population_count)
        receipt = V1LiteralJJointAffineShellD1PopulationStreamReceipt(
            design_hash=design.design_hash, forecast_hash=forecast.forecast_hash,
            joint_family_receipt_hash=family_receipt.receipt_hash,
            joint_preoutcome_registration_hash=family_receipt.joint_preoutcome_registration_hash,
            joint_postsource_custody_axis_hash=family_receipt.postsource_custody_axis_hash,
            conditioned_d1_replay_receipt_hash=replay.receipt_hash,
            d1_materialization_hash=replay.materialization_hash,
            d1_row_prefix_manifest_hash=replay.row_prefix_manifest_hash,
            d1_original_capture_binding_hash=replay.original_capture_binding_hash,
            compact_source_receipt_hash=conditioned_d1.compact_source_receipt.receipt_hash,
            source_axis_hash=plan.source_axis_hash,
            literal_j_shard_receipt_hashes=tuple(item.receipt.receipt_hash for item in conditioned_d1.literal_j_shards),
            literal_j_content_sha256s=tuple(item.receipt.content_sha256 for item in conditioned_d1.literal_j_shards),
            model_identity_hash=family_receipt.model_identity_hash,
            chart_identity_hash=replay.chart_identity_hash, current_mask_hash=replay.current_mask_hash,
            proposal_generation=replay.proposal_generation, route_axis_hash=replay.route_axis_hash,
            route_id_axis_hash=replay.route_id_axis_hash, route_count=replay.route_count,
            registered_d1_capture_prefix_hash=replay.registered_d_prefix_hash,
            d1_pair_prefix_axis_hash=replay.d1_pair_prefix_axis_hash,
            joint_registered_d_prefix_hash=family_receipt.registered_d_prefix_hash,
            d1_literal_j_stream_source_hash=replay.literal_j_stream_source_hash,
            joint_literal_j_stream_source_hash=family_receipt.literal_j_stream_source_hash,
            joint_direction_axis_hash=family_receipt.preoutcome_joint_direction_axis_hash,
            direction_ids=family_receipt.joint_direction_ids,
            physical_action_identity_hashes=(
                joint_preoutcome_registration.physical_action_identity_hashes
            ),
            physical_action_hashes=(
                joint_preoutcome_registration.physical_action_hashes
            ),
            packed_action_hashes=joint_preoutcome_registration.packed_action_hashes,
            preoutcome_registration_hashes=(
                joint_preoutcome_registration.preoutcome_registration_hashes
            ),
            destination_binding_hashes=(
                joint_preoutcome_registration.destination_binding_hashes
            ),
            action_count=family_receipt.action_count, shell_size=family_receipt.shell_size,
            direction_count=directions, shell_indices=family_receipt.shell_indices,
            shell_route_ids=family_receipt.shell_route_ids, d1_stratum_counts=replay.d1_stratum_counts,
            context_count=d, population_count=replay.population_count,
            exact_internal_d1_weight_fraction=(1, d),
            unapplied_d1_population_mass_fraction=(mass.numerator, mass.denominator),
            pass1_route_window_read_count=cursor.route_window_read_count,
            pass1_route_tile_read_count=cursor.route_tile_read_count,
            pass1_action_tile_submission_count=action_submissions,
            pass1_shared_shell_direct_gather_count=shell_gathers,
            pass1_source_payload_read_bytes=cursor.payload_read_bytes,
            host_to_device_payload_bytes=cursor.payload_read_bytes if production else 0,
            production_cuda_execution=production,
            private_cpu_reference_execution=not production,
            _issuance_token=_RECEIPT_TOKEN,
        )
        q_receipt = V1LiteralJJointAffineShellD1PopulationStreamQReceipt(
            stream_receipt_hash=receipt.receipt_hash,
            joint_direction_axis_hash=receipt.joint_direction_axis_hash,
            direction_ids=receipt.direction_ids, context_count=d,
            exact_internal_d1_weight_fraction=(1, d), direction_count=directions,
            _issuance_token=_Q_TOKEN,
        )
        q_handle = V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle(receipt=q_receipt, center=q_center, radius=q_radius, shape=(directions,), token=_Q_TOKEN)
        shell_receipt = V1LiteralJJointAffineShellD1PopulationStreamCompactShellMReceipt(
            stream_receipt_hash=receipt.receipt_hash, q_receipt_hash=q_receipt.receipt_hash,
            joint_direction_axis_hash=receipt.joint_direction_axis_hash,
            direction_ids=receipt.direction_ids, shell_indices=receipt.shell_indices,
            shell_route_ids=receipt.shell_route_ids, context_count=d,
            direction_count=directions, shell_size=receipt.shell_size,
            exact_internal_d1_weight_fraction=(1, d), _issuance_token=_SHELL_TOKEN,
        )
        shell_handle = V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle(receipt=shell_receipt, center=shell_center, radius=shell_radius, shape=(directions, receipt.shell_size), token=_SHELL_TOKEN)
        handle = V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle(
            receipt=receipt, forecast=forecast, q_handle=q_handle,
            compact_shell_m_handle=shell_handle, source=conditioned_d1, plan=plan,
            design=design, joint_lease=joint_lease, projection=projection,
            preregistration=joint_preoutcome_registration,
            projection_radius=projection_radius, issuance_token=_HANDLE_TOKEN,
        )
        joint_lease = None; q_handle = None; shell_handle = None
        return V1LiteralJJointAffineShellD1PopulationStreamResult(receipt=receipt, stream_handle=handle)
    except Exception:
        for handle in (shell_handle, q_handle):
            if handle is not None and not handle.is_evicted:
                try: handle.evict()
                except Exception: pass
        if joint_lease is not None:
            try: joint_lease.close()
            except Exception: pass
        raise


def execute_v1_literal_j_joint_affine_shell_d1_population_stream_cuda(
    *, conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    design: V1LiteralJJointAffineShellD1PopulationStreamDesign,
) -> V1LiteralJJointAffineShellD1PopulationStreamResult:
    return _execute(
        conditioned_d1=conditioned_d1,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        design=design, production=True, device_total_memory_bytes=None,
        device_free_memory_bytes=None, current_reserved_device_bytes=None,
    )


def _execute_v1_literal_j_joint_affine_shell_d1_population_stream_cpu_reference(
    *, conditioned_d1: V1LiteralJConditionedD1ShardReplaySource,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    design: V1LiteralJJointAffineShellD1PopulationStreamDesign,
    device_total_memory_bytes: int = 1_000_000_000,
    device_free_memory_bytes: int = 1_000_000_000,
    current_reserved_device_bytes: int = 0,
) -> V1LiteralJJointAffineShellD1PopulationStreamResult:
    return _execute(
        conditioned_d1=conditioned_d1,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        design=design, production=False,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
    )


__all__ = (
    "LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_AUTHORITY",
    "LITERAL_J_JOINT_AFFINE_SHELL_D1_POPULATION_STREAM_ALGORITHM",
    "V1LiteralJJointAffineShellD1PopulationStreamExecutionMode",
    "V1LiteralJJointAffineShellD1PopulationStreamRefusalCode",
    "V1LiteralJJointAffineShellD1PopulationStreamRefusal",
    "V1LiteralJJointAffineShellD1PopulationStreamDesign",
    "V1LiteralJJointAffineShellD1PopulationStreamHBMForecast",
    "V1LiteralJJointAffineShellD1PopulationStreamReceipt",
    "V1LiteralJJointAffineShellD1PopulationStreamQReceipt",
    "V1LiteralJJointAffineShellD1PopulationStreamQLease",
    "V1LiteralJJointAffineShellD1PopulationStreamQDeviceHandle",
    "V1LiteralJJointAffineShellD1PopulationStreamCompactShellMReceipt",
    "V1LiteralJJointAffineShellD1PopulationStreamCompactShellMLease",
    "V1LiteralJJointAffineShellD1PopulationStreamCompactShellMDeviceHandle",
    "V1LiteralJJointAffineShellD1PopulationStreamTileReceipt",
    "V1LiteralJJointAffineShellD1PopulationStreamTileLease",
    "V1LiteralJJointAffineShellD1PopulationStreamTileDeviceHandle",
    "V1LiteralJJointAffineShellD1PopulationStreamExecutionReceipt",
    "V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle",
    "V1LiteralJJointAffineShellD1PopulationStreamResult",
    "register_v1_literal_j_joint_affine_shell_d1_population_stream_cuda_design",
    "execute_v1_literal_j_joint_affine_shell_d1_population_stream_cuda",
)
