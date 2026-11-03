"""Source-bound support-plane coefficient scan for global true-``Q``.

The support-family owner freezes compact anchor right-hand sides before the
routewise ``M_J v`` population stream opens.  This module is the matching
route-major bridge: it consumes each one-live ``M_J v`` tile, applies the
support-plane factor two, and exposes one guarded coefficient tile at a time.

It is intentionally not the no-entry theorem.  It produces source-bound
support-plane scan input for the later outside-face solver, while leaving
D-stability, C/K/O, commit, runtime, terminal, and Pareto authority closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_joint_population_mjv_stream_cuda import (
    V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt,
    V1LiteralJJointD1PopulationMvStreamCUDAHandle,
    V1LiteralJJointD1PopulationMvStreamTileReceipt,
    V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt,
    V1LiteralJJointPopulationMvStreamCUDAHandle,
    V1LiteralJJointPopulationMvStreamCUDARefusal,
    V1LiteralJJointPopulationMvStreamCUDARefusalCode,
    V1LiteralJJointPopulationMvStreamTileReceipt,
    V1LiteralJJointPopulationMvStreamTileDeviceHandle,
)
from .modern_v1_phase_a_global_q_support_family import (
    V1PhaseAGlobalQD1SupportFamilyResult,
    V1PhaseAGlobalQSupportFamilyResult,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_TILE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_support_plane_scan_tile_v2"
)
PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_true_q_support_plane_scan_v2"
)
PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_ALGORITHM: Final = (
    "route_major_population_Mv_exact_outward_endpoint_factor_two_support_"
    "coefficient_interval_scan_v2"
)
PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_AUTHORITY: Final = (
    "source_bound_support_plane_coefficient_scan_input_only_no_solver_D_C_K_O_"
    "commit_runtime_terminal_or_pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-phase-a-global-q-support-plane-scan-"
_TILE_TOKEN: Final = object()
_SCAN_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_F64_BYTES: Final = 8

__all__ = (
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_TILE_SCHEMA",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_SCHEMA",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_ALGORITHM",
    "PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_AUTHORITY",
    "V1PhaseAGlobalQSupportPlaneScanRefusalCode",
    "V1PhaseAGlobalQSupportPlaneScanRefusal",
    "V1PhaseAGlobalQSupportPlaneScanTileReceipt",
    "V1PhaseAGlobalQSupportPlaneScanTileLease",
    "V1PhaseAGlobalQSupportPlaneScanTileHandle",
    "V1PhaseAGlobalQSupportPlaneScanReceipt",
    "V1PhaseAGlobalQSupportPlaneScanHandle",
    "open_v1_phase_a_global_q_support_plane_scan",
    "open_v1_phase_a_global_q_d1_support_plane_scan",
)


class V1PhaseAGlobalQSupportPlaneScanRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_MISMATCH = "source_mismatch"
    ACTIVE_TILE = "active_tile"
    STREAM_NOT_EXHAUSTED = "stream_not_exhausted"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQSupportPlaneScanRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQSupportPlaneScanRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQSupportPlaneScanRefusalCode:
            raise TypeError("support-plane scan refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAGlobalQSupportPlaneScanRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAGlobalQSupportPlaneScanRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one finite nonnegative binary64 scalar",
        )
    return value


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one finite binary64 scalar",
        )
    return value


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


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
        or not bool(torch.isfinite(value).all().item())
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous detached FP64 tensor with shape {shape}",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a negative interval radius",
        )
    return value


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _center_radius_from_bounds(lower: Any, upper: Any) -> tuple[Any, Any]:
    # Halving before addition avoids overflow when a finite interval straddles
    # a large magnitude.  The radius is rounded outward from both endpoints.
    center = lower * 0.5 + upper * 0.5
    center = torch.minimum(torch.maximum(center, lower), upper)
    radius = _up(torch.maximum(center - lower, upper - center))
    return center.contiguous(), radius.contiguous()


def _scale_factor_two(center: Any, radius: Any) -> tuple[Any, Any]:
    """Return the canonical coefficient endpoints around ``2 Mv``.

    Endpoint formation deliberately occurs *before* factor-two scaling.  This
    makes the exact replay identity

    ``c_L = down(2 * down(center - radius))`` and
    ``c_U = up(2 * up(center + radius))``

    available to every later outer-LP consumer; a center/radius rescaling is
    sound but is not algebraically the same registered outer half-space.
    """

    source_lower = _down(center - radius)
    source_upper = _up(center + radius)
    support_lower = _down(source_lower * 2.0)
    support_upper = _up(source_upper * 2.0)
    if (
        not bool(torch.isfinite(support_lower).all().item())
        or not bool(torch.isfinite(support_upper).all().item())
        or bool((support_lower > support_upper).any().item())
    ):
        _refuse(
            V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
            "support-plane factor-two interval conversion overflowed binary64",
        )
    return support_lower.contiguous(), support_upper.contiguous()


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlaneScanTileReceipt:
    support_family_receipt_hash: str
    population_mjv_tile_receipt_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    route_start: int
    route_stop: int
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    coefficient_min_lower: float
    coefficient_max_upper: float
    coefficient_radius_max: float
    exact_outward_endpoint_factor_two: bool = True
    rhs_not_repeated_per_route: bool = True
    route_scale_d2h: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_authority: bool = False
    runtime_terminal_or_pareto_authority: bool = False
    schema: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_TILE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("support-plane scan tile receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "support_family_receipt_hash",
            "population_mjv_tile_receipt_hash",
            "joint_direction_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.direction_ids) is not tuple
            or len(self.direction_ids) != self.direction_count
            or len(set(self.direction_ids)) != len(self.direction_ids)
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
                "direction_ids must be one immutable unique direction axis",
            )
        for value in self.direction_ids:
            _identity(value, name="direction_id")
        for name in ("route_count", "action_count", "shell_size", "direction_count"):
            _positive_int(getattr(self, name), name=name)
        if (
            self.direction_count != self.action_count + self.shell_size
            or not 0 <= self.route_start < self.route_stop <= self.route_count
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "support-plane tile axes do not match the registered action-shell family",
            )
        _finite(self.coefficient_min_lower, name="coefficient_min_lower")
        _finite(self.coefficient_max_upper, name="coefficient_max_upper")
        _finite_nonnegative(self.coefficient_radius_max, name="coefficient_radius_max")
        if self.coefficient_min_lower > self.coefficient_max_upper:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
                "support-plane coefficient tile has a reversed scalar enclosure",
            )
        positive = (
            self.exact_outward_endpoint_factor_two,
            self.rhs_not_repeated_per_route,
        )
        closed = (
            self.route_scale_d2h,
            self.solver_authority,
            self.d_stability_authority,
            self.c_k_o_commit_authority,
            self.runtime_terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.schema != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_TILE_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.AUTHORITY_VIOLATION,
                "support-plane tile receipt crossed its scan-input authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="tile")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "support-plane scan tile receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQSupportPlaneScanTileLease:
    __slots__ = ("_owner", "_token", "_closed", "_lower", "_upper", "_rhs_upper")

    def __init__(
        self,
        owner: "V1PhaseAGlobalQSupportPlaneScanTileHandle",
        token: object,
        lower: Any,
        upper: Any,
        rhs_upper: Any,
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self._lower = lower
        self._upper = upper
        self._rhs_upper = rhs_upper

    def __enter__(self) -> "V1PhaseAGlobalQSupportPlaneScanTileLease":
        self._owner.assert_live()
        return self

    def intervals(self) -> tuple[Any, Any, Any]:
        self._owner.assert_live()
        center, radius = _center_radius_from_bounds(self._lower, self._upper)
        return center, radius, self._rhs_upper

    def coefficient_bounds(self) -> tuple[Any, Any, Any]:
        """Return the registered direct coefficient endpoints and compact RHS."""

        self._owner.assert_live()
        return self._lower, self._upper, self._rhs_upper

    def close(self) -> None:
        if not self._closed:
            try:
                self._owner._close_lease(self._token)
            finally:
                self._closed = True

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("support-plane scan tile leases are nonserializable")


class V1PhaseAGlobalQSupportPlaneScanTileHandle:
    __slots__ = (
        "receipt",
        "_lower",
        "_upper",
        "_rhs_upper",
        "_guards",
        "_invariants",
        "_tokens",
        "_upstream_tile",
        "_release",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQSupportPlaneScanTileReceipt,
        lower: Any,
        upper: Any,
        rhs_upper: Any,
        upstream_tile: V1LiteralJJointPopulationMvStreamTileDeviceHandle,
        release: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _TILE_TOKEN:
            raise PermissionError("support-plane scan tile handles require their issuer")
        if type(receipt) is not V1PhaseAGlobalQSupportPlaneScanTileReceipt:
            raise TypeError("support-plane scan tile handle requires its exact receipt")
        self.receipt = receipt
        self._lower = _require_tensor(
            lower,
            name="support coefficient lower",
            shape=(receipt.direction_count, receipt.route_stop - receipt.route_start),
        )
        self._upper = _require_tensor(
            upper,
            name="support coefficient upper",
            shape=(receipt.direction_count, receipt.route_stop - receipt.route_start),
            device=self._lower.device,
        )
        if bool((self._lower > self._upper).any().item()):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.NUMERICAL_REFUSAL,
                "support coefficient endpoint tensors are reversed",
            )
        self._rhs_upper = _require_tensor(
            rhs_upper,
            name="support RHS upper",
            shape=(receipt.direction_count,),
            device=self._lower.device,
            nonnegative=True,
        )
        self._guards = tuple(
            item.detach().clone()
            for item in (self._lower, self._upper, self._rhs_upper)
        )
        self._invariants = tuple(
            _tensor_invariant(item)
            for item in (self._lower, self._upper, self._rhs_upper)
        )
        self._tokens: set[object] = set()
        self._upstream_tile = upstream_tile
        self._release = release
        self._evicted = False
        self.assert_live()

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.EVICTED,
                "support-plane scan tile was evicted",
            )
        self.receipt.assert_integrity()
        self._upstream_tile.assert_live()
        tensors = (self._lower, self._upper, self._rhs_upper)
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(
                tensors, self._invariants, self._guards, strict=True
            )
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "support-plane scan tile tensors mutated",
            )

    def borrow(self) -> V1PhaseAGlobalQSupportPlaneScanTileLease:
        self.assert_live()
        token = object()
        self._tokens.add(token)
        return V1PhaseAGlobalQSupportPlaneScanTileLease(
            self,
            token,
            self._lower,
            self._upper,
            self._rhs_upper,
        )

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "support-plane scan tile lease token is stale",
            )
        try:
            self.assert_live()
        finally:
            self._tokens.remove(token)

    def evict(self) -> None:
        self.assert_live()
        if self._tokens:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.ACTIVE_TILE,
                "cannot evict support-plane tile under an active lease",
            )
        receipt_hash = self.receipt.receipt_hash
        self._lower = None
        self._upper = None
        self._rhs_upper = None
        self._guards = ()
        self._invariants = ()
        self._upstream_tile.evict()
        self._evicted = True
        self._release(receipt_hash)

    def __reduce__(self) -> NoReturn:
        raise TypeError("support-plane scan tile handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQSupportPlaneScanReceipt:
    support_family_receipt_hash: str
    population_mjv_execution_receipt_hash: str
    population_mjv_forecast_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    route_axis_hash: str
    route_id_axis_hash: str
    current_mask_hash: str
    route_count: int
    action_count: int
    shell_size: int
    direction_count: int
    q_budget: float
    scanned_route_count: int
    scanned_tile_count: int
    tile_receipt_hashes: tuple[str, ...]
    population_mjv_tile_receipt_hashes: tuple[str, ...]
    route_ranges: tuple[tuple[int, int], ...]
    coefficient_min_lower: float
    coefficient_max_upper: float
    coefficient_radius_max: float
    support_rhs_upper_max: float
    production_source_authority: bool
    private_cpu_reference: bool
    support_plane_scan_authority: bool
    source_center_radius_factor_two: bool = True
    complete_route_axis_scanned: bool = True
    one_live_tile_enforced: bool = True
    no_route_scale_d2h: bool = True
    no_direction_route_bank_retained: bool = True
    rhs_compact_not_repeated_per_route: bool = True
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_commit_authority: bool = False
    runtime_terminal_or_pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SCAN_TOKEN:
            raise PermissionError("support-plane scan receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "support_family_receipt_hash",
            "population_mjv_execution_receipt_hash",
            "population_mjv_forecast_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "current_mask_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.direction_ids) is not tuple
            or len(self.direction_ids) != self.direction_count
            or len(set(self.direction_ids)) != len(self.direction_ids)
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.INVALID_TYPE,
                "direction_ids must be one immutable unique direction axis",
            )
        for value in self.direction_ids:
            _identity(value, name="direction_id")
        for name in (
            "route_count",
            "action_count",
            "shell_size",
            "direction_count",
            "scanned_route_count",
            "scanned_tile_count",
        ):
            _positive_int(getattr(self, name), name=name)
        _finite_nonnegative(self.q_budget, name="q_budget")
        _finite(self.coefficient_min_lower, name="coefficient_min_lower")
        _finite(self.coefficient_max_upper, name="coefficient_max_upper")
        _finite_nonnegative(self.coefficient_radius_max, name="coefficient_radius_max")
        _finite_nonnegative(self.support_rhs_upper_max, name="support_rhs_upper_max")
        tile_hashes = tuple(
            _identity(item, name="tile_receipt_hash")
            for item in self.tile_receipt_hashes
        )
        upstream_hashes = tuple(
            _identity(item, name="population_mjv_tile_receipt_hash")
            for item in self.population_mjv_tile_receipt_hashes
        )
        if (
            self.direction_count != self.action_count + self.shell_size
            or self.scanned_route_count != self.route_count
            or self.scanned_tile_count != len(tile_hashes)
            or self.scanned_tile_count != len(upstream_hashes)
            or len(set(tile_hashes)) != len(tile_hashes)
            or len(set(upstream_hashes)) != len(upstream_hashes)
            or type(self.route_ranges) is not tuple
            or len(self.route_ranges) != self.scanned_tile_count
            or not self.route_ranges
            or self.route_ranges[0][0] != 0
            or self.route_ranges[-1][1] != self.route_count
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not int for value in item)
                or not 0 <= item[0] < item[1] <= self.route_count
                for item in self.route_ranges
            )
            or any(
                self.route_ranges[index][1] != self.route_ranges[index + 1][0]
                for index in range(len(self.route_ranges) - 1)
            )
            or self.coefficient_min_lower > self.coefficient_max_upper
            or self.production_source_authority == self.private_cpu_reference
            or self.support_plane_scan_authority != self.production_source_authority
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "support-plane scan receipt axes or authority mode diverged",
            )
        positive = (
            self.source_center_radius_factor_two,
            self.complete_route_axis_scanned,
            self.one_live_tile_enforced,
            self.no_route_scale_d2h,
            self.no_direction_route_bank_retained,
            self.rhs_compact_not_repeated_per_route,
        )
        closed = (
            self.solver_authority,
            self.d_stability_authority,
            self.c_k_o_commit_authority,
            self.runtime_terminal_or_pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.algorithm != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_ALGORITHM
            or self.authority_scope != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_AUTHORITY
            or self.schema != PHASE_A_GLOBAL_Q_SUPPORT_PLANE_SCAN_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.AUTHORITY_VIOLATION,
                "support-plane scan receipt crossed its input-only authority",
            )
        object.__setattr__(self, "tile_receipt_hashes", tile_hashes)
        object.__setattr__(self, "population_mjv_tile_receipt_hashes", upstream_hashes)

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "support-plane scan receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQSupportPlaneScanHandle:
    __slots__ = (
        "_support",
        "_stream",
        "_source_mode",
        "_parent_identity",
        "_active_tile_hash",
        "_tile_receipt_hashes",
        "_upstream_tile_receipt_hashes",
        "_route_ranges",
        "_coefficient_min_lower",
        "_coefficient_max_upper",
        "_coefficient_radius_max",
        "_support_rhs_upper_max",
        "_exhausted",
        "_final_receipt",
        "_evicted",
    )

    def __init__(
        self,
        *,
        support_family: (
            V1PhaseAGlobalQSupportFamilyResult
            | V1PhaseAGlobalQD1SupportFamilyResult
        ),
        population_mjv_stream: (
            V1LiteralJJointPopulationMvStreamCUDAHandle
            | V1LiteralJJointD1PopulationMvStreamCUDAHandle
        ),
        source_mode: str,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("support-plane scan handles require their issuer")
        if source_mode not in {"legacy_projection_v1", "full_population_d1_v1"}:
            raise TypeError("support-plane scan requires one registered source mode")
        if source_mode == "legacy_projection_v1" and (
            type(support_family) is not V1PhaseAGlobalQSupportFamilyResult
            or type(population_mjv_stream)
            is not V1LiteralJJointPopulationMvStreamCUDAHandle
        ):
            raise TypeError("legacy support-plane scan requires exact legacy parents")
        if source_mode == "full_population_d1_v1" and (
            type(support_family) is not V1PhaseAGlobalQD1SupportFamilyResult
            or type(population_mjv_stream)
            is not V1LiteralJJointD1PopulationMvStreamCUDAHandle
        ):
            raise TypeError("D1 support-plane scan requires exact D1 parents")
        self._support = support_family
        self._stream = population_mjv_stream
        self._source_mode = source_mode
        receipt = support_family.receipt
        forecast = population_mjv_stream.forecast
        self._parent_identity = self._observed_parent_identity(receipt, forecast)
        self._active_tile_hash: str | None = None
        self._tile_receipt_hashes: list[str] = []
        self._upstream_tile_receipt_hashes: list[str] = []
        self._route_ranges: list[tuple[int, int]] = []
        self._coefficient_min_lower = math.inf
        self._coefficient_max_upper = -math.inf
        self._coefficient_radius_max = 0.0
        self._support_rhs_upper_max = 0.0
        self._exhausted = False
        self._final_receipt: V1PhaseAGlobalQSupportPlaneScanReceipt | None = None
        self._evicted = False
        self.assert_live()

    def _observed_parent_identity(self, receipt: Any, forecast: Any) -> tuple[object, ...]:
        common = (
            self._source_mode,
            receipt.receipt_hash,
            forecast.forecast_hash,
            receipt.joint_preoutcome_registration_hash,
            receipt.joint_direction_axis_hash,
            receipt.anchor_direction_ids,
            receipt.route_count,
            receipt.action_count,
            receipt.shell_size,
            receipt.production_source_authority,
            receipt.private_cpu_reference,
            forecast.production_cuda_execution,
            forecast.private_cpu_reference_execution,
        )
        if self._source_mode == "legacy_projection_v1":
            return common + (receipt.population_projection_registration_hash,)
        return common + (
            receipt.population_measure_registration_hash,
            receipt.population_measure_receipt_hash,
            receipt.action_axis_preoutcome_receipt_hash,
            receipt.action_axis_source_receipt_hash,
            receipt.joint_family_receipt_hash,
            receipt.d1_stream_receipt_hash,
            receipt.d1_q_receipt_hash,
            receipt.d1_compact_shell_m_receipt_hash,
            receipt.d1_source_axis_hash,
            receipt.residual_stream_registration_hash,
            receipt.residual_q_receipt_hash,
            receipt.residual_compact_shell_m_receipt_hash,
            receipt.population_count,
            receipt.d1_count,
            receipt.d1_mass_fraction,
            receipt.residual_population_counts,
            receipt.residual_observed_counts,
            receipt.residual_mass_fractions,
        )

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.EVICTED,
                "support-plane scan handle was evicted",
            )
        try:
            self._support.receipt.assert_integrity()
            self._support.device_handle.assert_live()
            self._stream.assert_live()
        except V1PhaseAGlobalQSupportPlaneScanRefusal:
            raise
        except Exception as error:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "one support-plane scan parent no longer replays: "
                f"{type(error).__name__}: {error}",
            )
        receipt = self._support.receipt
        forecast = self._stream.forecast
        observed = self._observed_parent_identity(receipt, forecast)
        if observed != self._parent_identity:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "support-plane scan parent source graph changed after opening",
            )
        common_mismatch = (
            forecast.joint_preoutcome_registration_hash
            != receipt.joint_preoutcome_registration_hash
            or forecast.route_count != receipt.route_count
            or forecast.action_count != receipt.action_count
            or forecast.shell_size != receipt.shell_size
            or forecast.direction_count != receipt.anchor_count
            or forecast.production_cuda_execution
            != receipt.production_source_authority
            or forecast.private_cpu_reference_execution
            != receipt.private_cpu_reference
        )
        if self._source_mode == "legacy_projection_v1":
            source_mismatch = (
                forecast.population_projection_registration_hash
                != receipt.population_projection_registration_hash
            )
        else:
            source_mismatch = (
                forecast.action_axis_preoutcome_receipt_hash
                != receipt.action_axis_preoutcome_receipt_hash
                or forecast.action_axis_source_receipt_hash
                != receipt.action_axis_source_receipt_hash
                or forecast.joint_family_receipt_hash
                != receipt.joint_family_receipt_hash
                or forecast.d1_stream_receipt_hash
                != receipt.d1_stream_receipt_hash
                or forecast.residual_stream_registration_hash
                != receipt.residual_stream_registration_hash
            )
        if common_mismatch or source_mismatch:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "support-family and population-Mv stream do not share one axis and source mode",
            )

    def next_tile(self) -> V1PhaseAGlobalQSupportPlaneScanTileHandle | None:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.ACTIVE_TILE,
                "the previous support-plane tile must be evicted before advancement",
            )
        if self._exhausted:
            return None
        upstream = self._stream.next_tile()
        if upstream is None:
            self._exhausted = True
            return None
        expected_upstream_receipt = (
            V1LiteralJJointPopulationMvStreamTileReceipt
            if self._source_mode == "legacy_projection_v1"
            else V1LiteralJJointD1PopulationMvStreamTileReceipt
        )
        if type(upstream.receipt) is not expected_upstream_receipt:
            try:
                upstream.evict()
            except Exception:
                pass
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "population-Mv tile receipt crossed the registered legacy/D1 source mode",
            )
        try:
            with upstream.borrow() as upstream_lease:
                center, radius = upstream_lease.intervals()
                support_lower, support_upper = _scale_factor_two(center, radius)
            with self._support.device_handle.borrow() as support_lease:
                rhs_upper = support_lease.rhs_upper.clone().contiguous()
        except V1LiteralJJointPopulationMvStreamCUDARefusal as error:
            try:
                upstream.evict()
            except Exception:
                pass
            code = (
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.ACTIVE_TILE
                if error.code
                is V1LiteralJJointPopulationMvStreamCUDARefusalCode.ACTIVE_TILE
                else V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY
            )
            _refuse(code, f"population-Mv source failed during support scan: {error}")
        except Exception as error:
            try:
                upstream.evict()
            except Exception:
                pass
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "support-family source failed during support scan: "
                f"{type(error).__name__}: {error}",
            )
        receipt = V1PhaseAGlobalQSupportPlaneScanTileReceipt(
            support_family_receipt_hash=self._support.receipt.receipt_hash,
            population_mjv_tile_receipt_hash=upstream.receipt.receipt_hash,
            joint_direction_axis_hash=self._support.receipt.joint_direction_axis_hash,
            direction_ids=self._support.receipt.anchor_direction_ids,
            route_start=upstream.receipt.route_start,
            route_stop=upstream.receipt.route_stop,
            route_count=self._support.receipt.route_count,
            action_count=self._support.receipt.action_count,
            shell_size=self._support.receipt.shell_size,
            direction_count=self._support.receipt.anchor_count,
            coefficient_min_lower=float(
                torch.amin(support_lower).item()
            ),
            coefficient_max_upper=float(
                torch.amax(support_upper).item()
            ),
            coefficient_radius_max=float(
                torch.amax(_center_radius_from_bounds(support_lower, support_upper)[1]).item()
            ),
            _issuance_token=_TILE_TOKEN,
        )
        receipt_hash = receipt.receipt_hash
        self._active_tile_hash = receipt_hash
        self._tile_receipt_hashes.append(receipt_hash)
        self._upstream_tile_receipt_hashes.append(upstream.receipt.receipt_hash)
        self._route_ranges.append((receipt.route_start, receipt.route_stop))
        self._coefficient_min_lower = min(
            self._coefficient_min_lower,
            receipt.coefficient_min_lower,
        )
        self._coefficient_max_upper = max(
            self._coefficient_max_upper,
            receipt.coefficient_max_upper,
        )
        self._coefficient_radius_max = max(
            self._coefficient_radius_max,
            receipt.coefficient_radius_max,
        )
        self._support_rhs_upper_max = max(
            self._support_rhs_upper_max,
            float(torch.amax(rhs_upper).item()),
        )
        return V1PhaseAGlobalQSupportPlaneScanTileHandle(
            receipt=receipt,
            lower=support_lower,
            upper=support_upper,
            rhs_upper=rhs_upper,
            upstream_tile=upstream,
            release=self._release_tile,
            _issuance_token=_TILE_TOKEN,
        )

    def _release_tile(self, receipt_hash: str) -> None:
        if self._active_tile_hash != receipt_hash:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.MUTATED_CUSTODY,
                "released support-plane tile is not the active tile",
            )
        self._active_tile_hash = None

    def _assert_final_stream_source_closure(self, stream_receipt: Any) -> None:
        support = self._support.receipt
        if self._source_mode == "legacy_projection_v1":
            if type(stream_receipt) is not V1LiteralJJointPopulationMvStreamCUDAExecutionReceipt:
                _refuse(
                    V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                    "legacy support scan finalized an inexact population-Mv receipt",
                )
            if (
                stream_receipt.forecast_hash != self._stream.forecast.forecast_hash
                or stream_receipt.joint_preoutcome_registration_hash
                != support.joint_preoutcome_registration_hash
                or stream_receipt.population_projection_registration_hash
                != support.population_projection_registration_hash
                or stream_receipt.joint_direction_axis_hash
                != support.joint_direction_axis_hash
                or stream_receipt.direction_ids != support.anchor_direction_ids
                or stream_receipt.route_count != support.route_count
                or stream_receipt.action_count != support.action_count
                or stream_receipt.shell_size != support.shell_size
                or stream_receipt.direction_count != support.anchor_count
                or stream_receipt.tile_receipt_hashes
                != tuple(self._upstream_tile_receipt_hashes)
                or stream_receipt.production_cuda_execution
                != support.production_source_authority
                or stream_receipt.private_cpu_reference_execution
                != support.private_cpu_reference
            ):
                _refuse(
                    V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                    "legacy support scan and population-Mv execution receipts diverged",
                )
            return
        if type(stream_receipt) is not V1LiteralJJointD1PopulationMvStreamCUDAExecutionReceipt:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "D1 support scan finalized an inexact D1 population-Mv receipt",
            )
        common = (
            (stream_receipt.forecast_hash, self._stream.forecast.forecast_hash),
            (
                stream_receipt.action_axis_preoutcome_receipt_hash,
                support.action_axis_preoutcome_receipt_hash,
            ),
            (
                stream_receipt.action_axis_source_receipt_hash,
                support.action_axis_source_receipt_hash,
            ),
            (
                stream_receipt.joint_preoutcome_registration_hash,
                support.joint_preoutcome_registration_hash,
            ),
            (stream_receipt.joint_family_receipt_hash, support.joint_family_receipt_hash),
            (stream_receipt.d1_stream_receipt_hash, support.d1_stream_receipt_hash),
            (stream_receipt.d1_q_receipt_hash, support.d1_q_receipt_hash),
            (
                stream_receipt.d1_compact_shell_m_receipt_hash,
                support.d1_compact_shell_m_receipt_hash,
            ),
            (
                stream_receipt.residual_stream_registration_hash,
                support.residual_stream_registration_hash,
            ),
            (stream_receipt.residual_q_receipt_hash, support.residual_q_receipt_hash),
            (
                stream_receipt.residual_compact_shell_m_receipt_hash,
                support.residual_compact_shell_m_receipt_hash,
            ),
            (
                stream_receipt.atomic_source_receipt_hashes,
                support.atomic_source_receipt_hashes,
            ),
            (
                stream_receipt.atomic_preoutcome_receipt_hashes,
                support.atomic_preoutcome_receipt_hashes,
            ),
            (
                stream_receipt.d1_literal_j_shard_receipt_hashes,
                support.d1_literal_j_shard_receipt_hashes,
            ),
            (
                stream_receipt.d1_literal_j_content_sha256s,
                support.d1_literal_j_content_sha256s,
            ),
            (
                stream_receipt.residual_source_proof_closure_hash_axes,
                support.residual_source_proof_closure_hash_axes,
            ),
            (
                stream_receipt.residual_source_shard_receipt_hash_axes,
                support.residual_source_shard_receipt_hash_axes,
            ),
            (
                stream_receipt.residual_source_content_sha256_axes,
                support.residual_source_content_sha256_axes,
            ),
            (stream_receipt.joint_direction_axis_hash, support.joint_direction_axis_hash),
            (stream_receipt.direction_ids, support.anchor_direction_ids),
            (stream_receipt.route_axis_hash, support.route_axis_hash),
            (stream_receipt.route_id_axis_hash, support.route_id_axis_hash),
            (stream_receipt.route_count, support.route_count),
            (stream_receipt.action_count, support.action_count),
            (stream_receipt.shell_size, support.shell_size),
            (stream_receipt.direction_count, support.anchor_count),
            (stream_receipt.population_count, support.population_count),
            (stream_receipt.d1_context_count, support.d1_count),
            (stream_receipt.d1_mass_fraction, support.d1_mass_fraction),
            (
                stream_receipt.residual_population_counts,
                support.residual_population_counts,
            ),
            (
                stream_receipt.residual_observed_counts,
                support.residual_observed_counts,
            ),
            (stream_receipt.residual_mass_fractions, support.residual_mass_fractions),
            (
                stream_receipt.output_tile_receipt_hashes,
                tuple(self._upstream_tile_receipt_hashes),
            ),
            (
                stream_receipt.production_cuda_execution,
                support.production_source_authority,
            ),
            (
                stream_receipt.private_cpu_reference_execution,
                support.private_cpu_reference,
            ),
            (
                stream_receipt.action_complete_source_binding_present,
                support.action_complete_source_binding_present,
            ),
            (
                stream_receipt.atomic_source_tuple_nonauthoritative,
                support.private_atomic_reference_nonauthoritative,
            ),
        )
        if any(left != right for left, right in common):
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.SOURCE_MISMATCH,
                "D1 support family and population-Mv execution do not share one "
                "action-complete D1/residual source graph",
            )

    def finalize(self) -> V1PhaseAGlobalQSupportPlaneScanReceipt:
        self.assert_live()
        if self._active_tile_hash is not None:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.ACTIVE_TILE,
                "cannot finalize support-plane scan with a live tile",
            )
        if not self._exhausted:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.STREAM_NOT_EXHAUSTED,
                "call next_tile until it returns None before finalization",
            )
        if self._final_receipt is None:
            stream_receipt = self._stream.finalize()
            self._assert_final_stream_source_closure(stream_receipt)
            support = self._support.receipt
            self._final_receipt = V1PhaseAGlobalQSupportPlaneScanReceipt(
                support_family_receipt_hash=support.receipt_hash,
                population_mjv_execution_receipt_hash=stream_receipt.receipt_hash,
                population_mjv_forecast_hash=self._stream.forecast.forecast_hash,
                joint_preoutcome_registration_hash=(
                    support.joint_preoutcome_registration_hash
                ),
                joint_direction_axis_hash=support.joint_direction_axis_hash,
                direction_ids=support.anchor_direction_ids,
                route_axis_hash=support.route_axis_hash,
                route_id_axis_hash=support.route_id_axis_hash,
                current_mask_hash=support.current_mask_hash,
                route_count=support.route_count,
                action_count=support.action_count,
                shell_size=support.shell_size,
                direction_count=support.anchor_count,
                q_budget=support.q_budget,
                scanned_route_count=(
                    0 if not self._route_ranges else self._route_ranges[-1][1]
                ),
                scanned_tile_count=len(self._tile_receipt_hashes),
                tile_receipt_hashes=tuple(self._tile_receipt_hashes),
                population_mjv_tile_receipt_hashes=tuple(
                    self._upstream_tile_receipt_hashes
                ),
                route_ranges=tuple(self._route_ranges),
                coefficient_min_lower=float(self._coefficient_min_lower),
                coefficient_max_upper=float(self._coefficient_max_upper),
                coefficient_radius_max=float(self._coefficient_radius_max),
                support_rhs_upper_max=float(self._support_rhs_upper_max),
                production_source_authority=support.production_source_authority,
                private_cpu_reference=support.private_cpu_reference,
                support_plane_scan_authority=support.production_source_authority,
                _issuance_token=_SCAN_TOKEN,
            )
        return self._final_receipt

    def evict(self) -> None:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.EVICTED,
                "support-plane scan handle was already evicted",
            )
        try:
            self.assert_live()
        except V1PhaseAGlobalQSupportPlaneScanRefusal as error:
            if error.code is V1PhaseAGlobalQSupportPlaneScanRefusalCode.EVICTED:
                raise
        if self._active_tile_hash is not None:
            _refuse(
                V1PhaseAGlobalQSupportPlaneScanRefusalCode.ACTIVE_TILE,
                "cannot evict support-plane scan with a live tile",
            )
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("support-plane scan handles are nonserializable")


def open_v1_phase_a_global_q_support_plane_scan(
    *,
    support_family: V1PhaseAGlobalQSupportFamilyResult,
    population_mjv_stream: V1LiteralJJointPopulationMvStreamCUDAHandle,
) -> V1PhaseAGlobalQSupportPlaneScanHandle:
    if (
        type(support_family) is not V1PhaseAGlobalQSupportFamilyResult
        or type(population_mjv_stream) is not V1LiteralJJointPopulationMvStreamCUDAHandle
    ):
        raise TypeError("support-plane scan requires exact support-family and Mv stream owners")
    return V1PhaseAGlobalQSupportPlaneScanHandle(
        support_family=support_family,
        population_mjv_stream=population_mjv_stream,
        source_mode="legacy_projection_v1",
        _issuance_token=_HANDLE_TOKEN,
    )


def open_v1_phase_a_global_q_d1_support_plane_scan(
    *,
    support_family: V1PhaseAGlobalQD1SupportFamilyResult,
    population_mjv_stream: V1LiteralJJointD1PopulationMvStreamCUDAHandle,
) -> V1PhaseAGlobalQSupportPlaneScanHandle:
    """Open the exact full-population D1/residual support-plane scan sibling."""

    if (
        type(support_family) is not V1PhaseAGlobalQD1SupportFamilyResult
        or type(population_mjv_stream)
        is not V1LiteralJJointD1PopulationMvStreamCUDAHandle
    ):
        raise TypeError("D1 support-plane scan requires exact D1 support and Mv owners")
    return V1PhaseAGlobalQSupportPlaneScanHandle(
        support_family=support_family,
        population_mjv_stream=population_mjv_stream,
        source_mode="full_population_d1_v1",
        _issuance_token=_HANDLE_TOKEN,
    )
