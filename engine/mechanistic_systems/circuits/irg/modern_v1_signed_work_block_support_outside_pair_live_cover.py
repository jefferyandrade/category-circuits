"""Source-native live CUDA cover for block-support outside pairs.

The legacy outside-pair receipt hashes caller-owned host tuples.  Those hashes
cannot prove equality to the registered KKT device masks or to the structural
physical-domain owner.  This module therefore derives the two exact rectangle
axes directly from live, mutation-checked custody:

``(P_alpha \\ required) x (selectable \\ (P_alpha union shell))`` and
``((P_alpha \\ required) \\ shell) x ((selectable \\ P_alpha) intersect shell)``.

Only compact scalar counts cross the device boundary.  No route tuple, route
membership vector, pair matrix, or route-scale content digest is constructed
by the production path.  Legacy cover identities may be attached as
diagnostics, but never contribute authority.
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from threading import RLock
from typing import Any, NoReturn, Sequence

try:  # pragma: no cover - exercised through the explicit CUDA refusal locally
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .contracts import content_hash
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportAliasBinding,
    V1BlockSupportWitnessSource,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1RelaxedKKTBranch,
    V1SignedWorkLeastWorkCUDAStateHandle,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBinding,
)
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellQBranch,
)


OUTSIDE_PAIR_LIVE_COVER_RULE = (
    "source_native_add_outside_and_drop_outside_add_inside_rectangles_v1"
)
OUTSIDE_PAIR_LIVE_COVER_SCHEMA = (
    "partition_free_v1_block_support_outside_pair_live_cover_receipt_v1"
)
OUTSIDE_PAIR_LIVE_COVER_DESIGN_SCHEMA = (
    "partition_free_v1_block_support_outside_pair_live_cover_design_v1"
)
OUTSIDE_PAIR_LIVE_COVER_FORECAST_SCHEMA = (
    "partition_free_v1_block_support_outside_pair_live_cover_hbm_forecast_v1"
)
OUTSIDE_PAIR_LIVE_COVER_LEASE_SCHEMA = (
    "partition_free_v1_block_support_outside_pair_live_cover_lease_v1"
)
OUTSIDE_PAIR_LIVE_COVER_LEGACY_BRIDGE_REQUIREMENT_SCHEMA = (
    "partition_free_v1_block_support_outside_pair_legacy_bridge_requirement_v1"
)
OUTSIDE_PAIR_LIVE_COVER_AUTHORITY = (
    "exact_live_rectangle_source_identity_and_counts_only_v1"
)
OUTSIDE_PAIR_LIVE_COVER_LEGACY_MISSING_ISSUER = (
    "v1_outside_pair_legacy_tuple_to_kkt_structural_equivalence_issuer_v1"
)

_PREFIX = "partition-free-v1-block-support-outside-pair-live-cover-"
_DESIGN_TOKEN = object()
_FORECAST_TOKEN = object()
_RECEIPT_TOKEN = object()
_PROVIDER_TOKEN = object()
_LEASE_TOKEN = object()
_SCAN_TOKEN = object()

_COUNTER_NAMES = (
    "scanned_route_count",
    "eligible_count",
    "required_count",
    "parent_live_count",
    "selectable_count",
    "rounding_anchor_count",
    "shell_count",
    "fixed_shell_overlap_count",
    "required_outside_anchor_count",
    "anchor_outside_selectable_count",
    "droppable_anchor_count",
    "addable_outside_shell_count",
    "droppable_anchor_outside_shell_count",
    "addable_inside_shell_count",
)
_COUNTER_COUNT = len(_COUNTER_NAMES)


class V1BlockSupportOutsidePairLiveCoverRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    HBM_FORECAST_REFUSED = "hbm_forecast_refused"
    SOURCE_MISMATCH = "source_mismatch"
    ALIAS_MISMATCH = "alias_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    MISSING_MASK_IDENTITY = "missing_mask_identity"
    INADMISSIBLE_ANCHOR = "inadmissible_anchor"
    INCOMPLETE_ROUTE_SCAN = "incomplete_route_scan"
    STALE_SOURCE = "stale_source"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    LEGACY_EQUIVALENCE_UNAVAILABLE = "legacy_equivalence_unavailable"
    AUTHORITY_VIOLATION = "authority_violation"


class V1BlockSupportOutsidePairLiveCoverRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportOutsidePairLiveCoverRefusalCode,
        detail: str,
    ) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _refuse(
    code: V1BlockSupportOutsidePairLiveCoverRefusalCode, detail: str
) -> NoReturn:
    raise V1BlockSupportOutsidePairLiveCoverRefusal(code, detail)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            f"{name} must be a nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or type(value) is not int:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            f"{name} must be an exact integer",
        )
    if value < (1 if positive else 0):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            f"{name} is outside its nonnegative domain",
        )
    return value


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its authority boundary",
        )


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)  # type: ignore[arg-type]
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _source_context_axis_hash(contexts: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": "block_support_context",
            "values": tuple(contexts),
        },
        prefix="partition-free-v1-variable-shell-source-axis-",
    )


def _canonical_context_axis_hash(contexts: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_canonical_sealed_context_tuple_v1",
            "context_ids": tuple(contexts),
        },
        prefix="partition-free-v1-block-support-domain-binding-canonical-context-tuple-",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveCoverDesign:
    route_tile_size: int = 131_072
    hbm_ceiling_fraction: float = 0.85
    require_deterministic_algorithms: bool = True
    cuda_production_only: bool = True
    route_scale_host_membership_forbidden: bool = True
    route_scale_d2h_forbidden: bool = True
    route_scale_content_digest_forbidden: bool = True
    schema: str = OUTSIDE_PAIR_LIVE_COVER_DESIGN_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DESIGN_TOKEN:
            raise TypeError("live-cover designs are factory-issued")
        tile = _integer(self.route_tile_size, name="route tile size", positive=True)
        if tile % 8:
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_DESIGN,
                "route tile size must preserve packed-parent byte boundaries",
            )
        object.__setattr__(self, "route_tile_size", tile)
        ceiling = float(self.hbm_ceiling_fraction)
        if not math.isfinite(ceiling) or not 0.0 < ceiling <= 0.85:
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_DESIGN,
                "live-cover HBM ceiling must be finite and at most 0.85",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        required = (
            self.require_deterministic_algorithms,
            self.cuda_production_only,
            self.route_scale_host_membership_forbidden,
            self.route_scale_d2h_forbidden,
            self.route_scale_content_digest_forbidden,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_DESIGN,
                "production live-cover design weakened a mandatory invariant",
            )
        if self.schema != OUTSIDE_PAIR_LIVE_COVER_DESIGN_SCHEMA:
            raise ValueError("unsupported live-cover design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "design-")


def build_v1_block_support_outside_pair_live_cover_design(
    *,
    route_tile_size: int = 131_072,
    hbm_ceiling_fraction: float = 0.85,
) -> V1BlockSupportOutsidePairLiveCoverDesign:
    return V1BlockSupportOutsidePairLiveCoverDesign(
        route_tile_size=route_tile_size,
        hbm_ceiling_fraction=hbm_ceiling_fraction,
        _issuer_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveCoverHBMForecast:
    design_hash: str
    route_count: int
    route_tile_size: int
    borrowed_fixed_membership_bytes: int
    borrowed_parent_packed_bytes: int
    borrowed_kkt_resource_bytes: int
    borrowed_kkt_eligible_bytes: int
    borrowed_kkt_required_bytes: int
    borrowed_live_source_bytes: int
    owned_route_scale_bytes: int
    owned_pair_matrix_bytes: int
    tile_membership_workspace_bytes: int
    compact_counter_bytes: int
    incremental_device_bytes: int
    current_reserved_device_bytes: int
    device_free_memory_bytes: int
    device_total_memory_bytes: int
    forecast_peak_reserved_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    receipt_hash: str = field(init=False)
    borrowed_sources_not_double_counted_as_owned: bool = True
    bounded_by_one_route_tile: bool = True
    schema: str = OUTSIDE_PAIR_LIVE_COVER_FORECAST_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _FORECAST_TOKEN:
            raise TypeError("live-cover HBM forecasts are factory-issued")
        object.__setattr__(
            self, "design_hash", _identity(self.design_hash, name="design hash")
        )
        integer_names = (
            "route_count",
            "route_tile_size",
            "borrowed_fixed_membership_bytes",
            "borrowed_parent_packed_bytes",
            "borrowed_kkt_resource_bytes",
            "borrowed_kkt_eligible_bytes",
            "borrowed_kkt_required_bytes",
            "borrowed_live_source_bytes",
            "owned_route_scale_bytes",
            "owned_pair_matrix_bytes",
            "tile_membership_workspace_bytes",
            "compact_counter_bytes",
            "incremental_device_bytes",
            "current_reserved_device_bytes",
            "device_free_memory_bytes",
            "device_total_memory_bytes",
            "forecast_peak_reserved_bytes",
            "hbm_ceiling_bytes",
        )
        for name in integer_names:
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {"route_count", "route_tile_size", "device_total_memory_bytes"},
                ),
            )
        expected_borrowed = (
            self.borrowed_fixed_membership_bytes
            + self.borrowed_parent_packed_bytes
            + self.borrowed_kkt_resource_bytes
            + self.borrowed_kkt_eligible_bytes
            + self.borrowed_kkt_required_bytes
        )
        if (
            self.borrowed_live_source_bytes != expected_borrowed
            or self.owned_route_scale_bytes != 0
            or self.owned_pair_matrix_bytes != 0
            or self.incremental_device_bytes
            != self.tile_membership_workspace_bytes + self.compact_counter_bytes
            or self.forecast_peak_reserved_bytes
            != self.current_reserved_device_bytes + self.incremental_device_bytes
            or type(self.admitted) is not bool
            or self.borrowed_sources_not_double_counted_as_owned is not True
            or self.bounded_by_one_route_tile is not True
            or self.schema != OUTSIDE_PAIR_LIVE_COVER_FORECAST_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.HBM_FORECAST_REFUSED,
                "live-cover HBM accounting does not replay",
            )
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "hbm-forecast-"),
        )


def build_v1_block_support_outside_pair_live_cover_hbm_forecast(
    *,
    design: V1BlockSupportOutsidePairLiveCoverDesign,
    route_count: int,
    current_reserved_device_bytes: int,
    device_free_memory_bytes: int,
    device_total_memory_bytes: int,
) -> V1BlockSupportOutsidePairLiveCoverHBMForecast:
    if type(design) is not V1BlockSupportOutsidePairLiveCoverDesign:
        raise TypeError("live-cover HBM forecast requires the exact design")
    routes = _integer(route_count, name="route count", positive=True)
    current = _integer(
        current_reserved_device_bytes, name="current reserved device bytes"
    )
    free = _integer(device_free_memory_bytes, name="device free memory bytes")
    total = _integer(
        device_total_memory_bytes, name="device total memory bytes", positive=True
    )
    tile = min(routes, design.route_tile_size)
    # Six retained bool tiles (anchor, shell, unpacked parent, selectable,
    # droppable, addable), plus conservative allowance for unpack/reduction
    # expression temporaries.  Compact B<=8 indices/weights/scalars fit in the
    # fixed tail.  This deliberately over-accounts allocator overlap.
    tile_workspace = 10 * tile + 512
    counters = _COUNTER_COUNT * 8
    incremental = tile_workspace + counters
    ceiling = int(math.floor(total * design.hbm_ceiling_fraction))
    admitted = (
        current <= total
        and free <= total
        and incremental <= free
        and current + incremental <= ceiling
    )
    parent_bytes = (routes + 7) // 8
    return V1BlockSupportOutsidePairLiveCoverHBMForecast(
        design_hash=design.design_hash,
        route_count=routes,
        route_tile_size=design.route_tile_size,
        borrowed_fixed_membership_bytes=routes,
        borrowed_parent_packed_bytes=parent_bytes,
        borrowed_kkt_resource_bytes=8 * routes,
        borrowed_kkt_eligible_bytes=routes,
        borrowed_kkt_required_bytes=routes,
        borrowed_live_source_bytes=11 * routes + parent_bytes,
        owned_route_scale_bytes=0,
        owned_pair_matrix_bytes=0,
        tile_membership_workspace_bytes=tile_workspace,
        compact_counter_bytes=counters,
        incremental_device_bytes=incremental,
        current_reserved_device_bytes=current,
        device_free_memory_bytes=free,
        device_total_memory_bytes=total,
        forecast_peak_reserved_bytes=current + incremental,
        hbm_ceiling_bytes=ceiling,
        admitted=admitted,
        _issuer_token=_FORECAST_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class _LiveCoverScan:
    scanned_route_count: int
    eligible_count: int
    required_count: int
    parent_live_count: int
    selectable_count: int
    rounding_anchor_count: int
    shell_count: int
    fixed_shell_overlap_count: int
    required_outside_anchor_count: int
    anchor_outside_selectable_count: int
    droppable_anchor_count: int
    addable_outside_shell_count: int
    droppable_anchor_outside_shell_count: int
    addable_inside_shell_count: int
    route_tile_scan_count: int
    scalar_d2h_bytes: int
    d2h_synchronization_count: int
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SCAN_TOKEN:
            raise TypeError("live-cover scan results are issuer-only")
        for item in fields(self):
            if not item.name.startswith("_"):
                object.__setattr__(
                    self,
                    item.name,
                    _integer(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )

    @property
    def add_outside_rectangle_pair_count(self) -> int:
        return self.droppable_anchor_count * self.addable_outside_shell_count

    @property
    def drop_outside_add_inside_rectangle_pair_count(self) -> int:
        return (
            self.droppable_anchor_outside_shell_count * self.addable_inside_shell_count
        )

    @property
    def complete_pair_count(self) -> int:
        return (
            self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
        )


def _require_membership_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: object,
    device: object,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype is not dtype
        or tuple(value.shape) != shape
        or value.device != device
        or not value.is_contiguous()
        or value.requires_grad
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.MUTATED_CUSTODY,
            f"{name} is not canonical live custody",
        )
    return value


def _tile_compact_routes(
    routes: tuple[int, ...], *, start: int, stop: int, device: object
) -> Any:
    left = bisect_left(routes, start)
    right = bisect_left(routes, stop, lo=left)
    if left == right:
        return None
    return torch.tensor(
        tuple(value - start for value in routes[left:right]),
        dtype=torch.int64,
        device=device,
    )


def _unpack_parent_tile(
    parent_packed: Any, *, start: int, stop: int, device: object
) -> Any:
    byte_start = start // 8
    byte_stop = (stop + 7) // 8
    packed = parent_packed[byte_start:byte_stop]
    weights = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1),
        dtype=torch.uint8,
        device=device,
    )
    return (
        torch.bitwise_and(packed[:, None], weights[None, :])
        .ne(0)
        .reshape(-1)[: stop - start]
    )


def _scan_live_cover_membership(
    *,
    fixed_membership: object,
    resident_parent_packed: object,
    eligible_membership: object,
    required_membership: object,
    shell_indices: Sequence[int],
    rounding_anchor_shell_indices: Sequence[int],
    route_count: int,
    route_tile_size: int,
    require_cuda: bool,
    _scan_route_limit_for_test: int | None = None,
) -> _LiveCoverScan:
    """Perform one deterministic tiled O(N) source scan.

    ``_scan_route_limit_for_test`` exists solely to exercise the incomplete
    scan refusal.  It is never accepted by the production builder.
    """

    if torch is None:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.CUDA_REQUIRED,
            "torch is unavailable",
        )
    routes = _integer(route_count, name="route count", positive=True)
    tile = _integer(route_tile_size, name="route tile size", positive=True)
    if tile % 8:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_DESIGN,
            "route tile size must be byte aligned",
        )
    if type(require_cuda) is not bool:
        raise TypeError("require_cuda must be an exact bool")
    if type(fixed_membership) is not torch.Tensor:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "fixed membership is not a tensor",
        )
    device = fixed_membership.device
    if require_cuda and device.type != "cuda":
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.CUDA_REQUIRED,
            "production live-cover scan requires CUDA custody",
        )
    if not require_cuda and device.type != "cpu":
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "reference live-cover scan requires CPU tensors",
        )
    fixed = _require_membership_tensor(
        fixed_membership,
        name="fixed membership",
        shape=(routes,),
        dtype=torch.bool,
        device=device,
    )
    parent = _require_membership_tensor(
        resident_parent_packed,
        name="resident packed parent",
        shape=((routes + 7) // 8,),
        dtype=torch.uint8,
        device=device,
    )
    eligible = _require_membership_tensor(
        eligible_membership,
        name="eligible membership",
        shape=(routes,),
        dtype=torch.bool,
        device=device,
    )
    required = _require_membership_tensor(
        required_membership,
        name="required membership",
        shape=(routes,),
        dtype=torch.bool,
        device=device,
    )
    shell = tuple(_integer(value, name="shell route index") for value in shell_indices)
    anchor_shell = tuple(
        _integer(value, name="rounding-anchor shell route index")
        for value in rounding_anchor_shell_indices
    )
    if (
        not shell
        or len(shell) > 8
        or shell != tuple(sorted(set(shell)))
        or any(value >= routes for value in shell)
        or anchor_shell != tuple(sorted(set(anchor_shell)))
        or not set(anchor_shell).issubset(shell)
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "shell or rounding-anchor shell coordinate is not canonical",
        )
    scan_stop = routes
    if _scan_route_limit_for_test is not None:
        scan_stop = min(
            routes,
            _integer(
                _scan_route_limit_for_test,
                name="test scan route limit",
                positive=True,
            ),
        )
    totals = torch.zeros((_COUNTER_COUNT,), dtype=torch.int64, device=device)
    tile_count = 0
    for start in range(0, scan_stop, tile):
        stop = min(scan_stop, start + tile)
        length = stop - start
        anchor = fixed[start:stop].clone()
        anchor_local = _tile_compact_routes(
            anchor_shell, start=start, stop=stop, device=device
        )
        if anchor_local is not None:
            anchor.index_fill_(0, anchor_local, True)
        shell_mask = torch.zeros((length,), dtype=torch.bool, device=device)
        shell_local = _tile_compact_routes(shell, start=start, stop=stop, device=device)
        if shell_local is not None:
            shell_mask.index_fill_(0, shell_local, True)
        parent_live = _unpack_parent_tile(parent, start=start, stop=stop, device=device)
        eligible_tile = eligible[start:stop]
        required_tile = required[start:stop]
        fixed_tile = fixed[start:stop]
        selectable = eligible_tile & parent_live
        droppable = anchor & ~required_tile
        addable = selectable & ~anchor
        local = torch.stack(
            (
                torch.tensor(length, dtype=torch.int64, device=device),
                eligible_tile.sum(dtype=torch.int64),
                required_tile.sum(dtype=torch.int64),
                parent_live.sum(dtype=torch.int64),
                selectable.sum(dtype=torch.int64),
                anchor.sum(dtype=torch.int64),
                shell_mask.sum(dtype=torch.int64),
                (fixed_tile & shell_mask).sum(dtype=torch.int64),
                (required_tile & ~anchor).sum(dtype=torch.int64),
                (anchor & ~selectable).sum(dtype=torch.int64),
                droppable.sum(dtype=torch.int64),
                (addable & ~shell_mask).sum(dtype=torch.int64),
                (droppable & ~shell_mask).sum(dtype=torch.int64),
                (addable & shell_mask).sum(dtype=torch.int64),
            )
        )
        totals.add_(local)
        tile_count += 1
    if device.type == "cuda":
        # Exactly one compact device-to-host copy and synchronization.
        host_totals = totals.detach().cpu()
        scalar_d2h = _COUNTER_COUNT * 8
        synchronizations = 1
    else:
        host_totals = totals
        scalar_d2h = 0
        synchronizations = 0
    values = tuple(int(value) for value in host_totals.tolist())
    result = _LiveCoverScan(
        **dict(zip(_COUNTER_NAMES, values, strict=True)),
        route_tile_scan_count=tile_count,
        scalar_d2h_bytes=scalar_d2h,
        d2h_synchronization_count=synchronizations,
        _issuer_token=_SCAN_TOKEN,
    )
    if result.scanned_route_count != routes:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INCOMPLETE_ROUTE_SCAN,
            "live-cover source scan did not visit every physical route",
        )
    if result.shell_count != len(shell) or result.fixed_shell_overlap_count:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "structural fixed prefix overlaps or loses the bounded shell",
        )
    if result.required_outside_anchor_count or result.anchor_outside_selectable_count:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "rounding anchor does not contain required routes or is not selectable",
        )
    return result


def _derived_membership_identity(
    role: str, *, route_count: int, sources: Sequence[object]
) -> str:
    return content_hash(
        {
            "rule": OUTSIDE_PAIR_LIVE_COVER_RULE,
            "role": role,
            "route_count": route_count,
            "sources": tuple(sources),
        },
        prefix=_PREFIX + "derived-membership-",
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveCoverReceipt:
    """Array-free exact identity/count receipt for one live alias cover."""

    design_hash: str
    hbm_forecast_hash: str
    block_support_registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    residual_source_context_axis_hash: str
    physical_block_support_binding_receipt_hash: str
    structural_domain_receipt_hash: str
    registry_receipt_hash: str
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    source_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    resident_device_identity: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    preferred_action_binding_hash: str
    comparison_preferred_action_hash: str
    comparison_preferred_removed_route_index_hash: str
    comparison_preferred_selected_count: int
    alias_hash: str
    branch_id: str
    fd_rank: int
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    cardinality_alias_id: str
    alias_fixed_outside_removed_count: int
    alias_shell_prefix_count: int
    alias_total_removed_count: int
    fixed_membership_binding_hash: str
    fixed_outside_removed_route_index_hash: str
    parent_mask_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_membership_identity_hash: str
    branch_execution_hash: str
    source_rounding_hash: str
    frozen_ladder_receipt_hash: str
    frozen_label_commitment: str
    frozen_eligibility_commitment: str
    frozen_endpoint_action_receipt_hash: str
    frozen_endpoint_packed_action_hash: str
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    rounding_anchor_selected_shell_offsets: tuple[int, ...]
    rounding_anchor_shell_route_indices: tuple[int, ...]
    rounding_anchor_membership_identity_hash: str
    kkt_problem_hash: str
    kkt_policy_hash: str
    kkt_state_identity_hash: str
    kkt_branch_receipt_hash: str
    kkt_eligible_mask_identity_hash: str
    kkt_required_mask_identity_hash: str
    parent_live_membership_identity_hash: str
    selectable_membership_identity_hash: str
    add_outside_drop_axis_identity_hash: str
    add_outside_add_axis_identity_hash: str
    drop_outside_add_inside_drop_axis_identity_hash: str
    drop_outside_add_inside_add_axis_identity_hash: str
    source_native_pair_axis_commitment_hash: str
    registered_legacy_pair_cover_hash_diagnostic: str
    legacy_pair_cover_receipt_hash_diagnostic: str | None
    legacy_pair_axis_commitment_hash_diagnostic: str | None
    route_count: int
    eligible_count: int
    required_count: int
    parent_live_count: int
    selectable_count: int
    fixed_outside_count: int
    shell_count: int
    rounding_anchor_count: int
    add_outside_droppable_count: int
    add_outside_addable_count: int
    drop_outside_add_inside_droppable_count: int
    drop_outside_add_inside_addable_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_count: int
    route_tile_size: int
    route_tile_scan_count: int
    complete_route_scan_count: int
    scalar_device_to_host_transfer_bytes: int
    device_to_host_synchronization_count: int
    route_scale_device_to_host_transfer_bytes: int
    route_scale_content_digest_bytes: int
    owned_route_scale_device_bytes: int
    owned_pair_matrix_bytes: int
    receipt_hash: str = field(init=False)
    source_native_pair_axis_complete: bool = True
    exact_registered_device_source_join: bool = True
    exact_frozen_rounding_anchor_join: bool = True
    deterministic_physical_index_ties: bool = True
    array_free_receipt: bool = True
    production_cuda_execution: bool = True
    borrowed_membership_not_copied: bool = True
    legacy_hashes_diagnostic_only: bool = True
    legacy_cover_equivalence_proved: bool = False
    downstream_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    pareto_authority: bool = False
    theorem_rule: str = OUTSIDE_PAIR_LIVE_COVER_RULE
    authority_scope: str = OUTSIDE_PAIR_LIVE_COVER_AUTHORITY
    schema: str = OUTSIDE_PAIR_LIVE_COVER_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("live-cover receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name != "receipt_hash"
                and (
                    item.name.endswith("_hash")
                    or item.name.endswith("_identity")
                    or item.name.endswith("_commitment")
                )
                and getattr(self, item.name) is not None
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        for name in (
            "block_id",
            "branch_id",
            "resource_alias_id",
            "cardinality_alias_id",
            "resident_device_identity",
            "authority_scope",
            "theorem_rule",
            "registered_legacy_pair_cover_hash_diagnostic",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        integer_names = (
            "block_index",
            "proposal_generation",
            "comparison_preferred_selected_count",
            "fd_rank",
            "alias_fixed_outside_removed_count",
            "alias_shell_prefix_count",
            "alias_total_removed_count",
            "route_count",
            "eligible_count",
            "required_count",
            "parent_live_count",
            "selectable_count",
            "fixed_outside_count",
            "shell_count",
            "rounding_anchor_count",
            "add_outside_droppable_count",
            "add_outside_addable_count",
            "drop_outside_add_inside_droppable_count",
            "drop_outside_add_inside_addable_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_count",
            "route_tile_size",
            "route_tile_scan_count",
            "complete_route_scan_count",
            "scalar_device_to_host_transfer_bytes",
            "device_to_host_synchronization_count",
            "route_scale_device_to_host_transfer_bytes",
            "route_scale_content_digest_bytes",
            "owned_route_scale_device_bytes",
            "owned_pair_matrix_bytes",
        )
        for name in integer_names:
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name
                    in {
                        "fd_rank",
                        "route_count",
                        "route_tile_size",
                        "route_tile_scan_count",
                    },
                ),
            )
        shell = tuple(
            _integer(value, name="shell route index") for value in self.shell_indices
        )
        offsets = tuple(
            _integer(value, name="rounding-anchor shell offset")
            for value in self.rounding_anchor_selected_shell_offsets
        )
        anchor_routes = tuple(
            _integer(value, name="rounding-anchor shell route")
            for value in self.rounding_anchor_shell_route_indices
        )
        if (
            not shell
            or len(shell) > 8
            or shell != tuple(sorted(set(shell)))
            or offsets != tuple(sorted(set(offsets)))
            or any(value >= len(shell) for value in offsets)
            or anchor_routes != tuple(shell[value] for value in offsets)
            or self.alias_shell_prefix_count != len(offsets)
            or self.alias_total_removed_count
            != self.alias_fixed_outside_removed_count + self.alias_shell_prefix_count
            or self.alias_total_removed_count != self.rounding_anchor_count
            or self.alias_total_removed_count > self.route_count
            or self.fixed_outside_count != self.alias_fixed_outside_removed_count
            or self.shell_count != len(shell)
            or self.complete_route_scan_count != self.route_count
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
                "live-cover anchor/shell/cardinality axes do not close",
            )
        object.__setattr__(self, "shell_indices", shell)
        object.__setattr__(self, "rounding_anchor_selected_shell_offsets", offsets)
        object.__setattr__(self, "rounding_anchor_shell_route_indices", anchor_routes)
        if (
            self.add_outside_rectangle_pair_count
            != self.add_outside_droppable_count * self.add_outside_addable_count
            or self.drop_outside_add_inside_rectangle_pair_count
            != self.drop_outside_add_inside_droppable_count
            * self.drop_outside_add_inside_addable_count
            or self.complete_pair_count
            != self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.INCOMPLETE_ROUTE_SCAN,
                "live rectangle counts do not replay their Cartesian products",
            )
        expected_shell = _derived_membership_identity(
            "bounded_shell",
            route_count=self.route_count,
            sources=(self.shell_freeze_receipt_hash, shell),
        )
        expected_anchor = _derived_membership_identity(
            "rounding_anchor_P_alpha",
            route_count=self.route_count,
            sources=(
                self.fixed_membership_binding_hash,
                self.fixed_outside_removed_route_index_hash,
                self.rounding_anchor_removed_route_index_hash,
                self.rounding_anchor_subset_identity_hash,
                offsets,
            ),
        )
        expected_parent = _derived_membership_identity(
            "parent_live",
            route_count=self.route_count,
            sources=(self.structural_domain_receipt_hash, self.parent_mask_hash),
        )
        expected_selectable = _derived_membership_identity(
            "selectable_equals_eligible_and_parent_live",
            route_count=self.route_count,
            sources=(self.kkt_eligible_mask_identity_hash, expected_parent),
        )
        p1 = _derived_membership_identity(
            "add_outside_drop_P_alpha_minus_required",
            route_count=self.route_count,
            sources=(expected_anchor, self.kkt_required_mask_identity_hash),
        )
        q1 = _derived_membership_identity(
            "add_outside_add_selectable_minus_P_alpha_union_shell",
            route_count=self.route_count,
            sources=(expected_selectable, expected_anchor, expected_shell),
        )
        p2 = _derived_membership_identity(
            "drop_outside_add_inside_drop_P_alpha_minus_required_minus_shell",
            route_count=self.route_count,
            sources=(p1, expected_shell),
        )
        q2 = _derived_membership_identity(
            "drop_outside_add_inside_add_selectable_minus_P_alpha_intersect_shell",
            route_count=self.route_count,
            sources=(expected_selectable, expected_anchor, expected_shell),
        )
        expected_pair = content_hash(
            {
                "rule": OUTSIDE_PAIR_LIVE_COVER_RULE,
                "route_count": self.route_count,
                "rounding_anchor_membership_identity_hash": expected_anchor,
                "rectangle_axes": (
                    (p1, q1, self.add_outside_rectangle_pair_count),
                    (p2, q2, self.drop_outside_add_inside_rectangle_pair_count),
                ),
                "complete_pair_count": self.complete_pair_count,
            },
            prefix=_PREFIX + "pair-axis-commitment-",
        )
        if (
            self.shell_membership_identity_hash != expected_shell
            or self.rounding_anchor_membership_identity_hash != expected_anchor
            or self.parent_live_membership_identity_hash != expected_parent
            or self.selectable_membership_identity_hash != expected_selectable
            or self.add_outside_drop_axis_identity_hash != p1
            or self.add_outside_add_axis_identity_hash != q1
            or self.drop_outside_add_inside_drop_axis_identity_hash != p2
            or self.drop_outside_add_inside_add_axis_identity_hash != q2
            or self.source_native_pair_axis_commitment_hash != expected_pair
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
                "source-native rectangle identity derivation does not replay",
            )
        required_true = (
            self.source_native_pair_axis_complete,
            self.exact_registered_device_source_join,
            self.exact_frozen_rounding_anchor_join,
            self.deterministic_physical_index_ties,
            self.array_free_receipt,
            self.production_cuda_execution,
            self.borrowed_membership_not_copied,
            self.legacy_hashes_diagnostic_only,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.AUTHORITY_VIOLATION,
                "live-cover receipt lost a mandatory source invariant",
            )
        _closed(
            (
                self.legacy_cover_equivalence_proved,
                self.downstream_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.pareto_authority,
            ),
            owner="source-native outside-pair cover",
        )
        if (
            (self.legacy_pair_cover_receipt_hash_diagnostic is None)
            != (self.legacy_pair_axis_commitment_hash_diagnostic is None)
            or self.route_scale_device_to_host_transfer_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.owned_route_scale_device_bytes != 0
            or self.owned_pair_matrix_bytes != 0
            or self.scalar_device_to_host_transfer_bytes != _COUNTER_COUNT * 8
            or self.device_to_host_synchronization_count != 1
            or self.theorem_rule != OUTSIDE_PAIR_LIVE_COVER_RULE
            or self.authority_scope != OUTSIDE_PAIR_LIVE_COVER_AUTHORITY
            or self.schema != OUTSIDE_PAIR_LIVE_COVER_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.AUTHORITY_VIOLATION,
                "live-cover transfer, legacy, or authority semantics changed",
            )
        seal = content_hash(_payload(self), prefix=_PREFIX + "receipt-integrity-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.MUTATED_CUSTODY,
                "live-cover receipt integrity changed",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(
            self,
            "receipt_hash",
            content_hash(_payload(self), prefix=_PREFIX + "receipt-"),
        )

    def assert_integrity(self) -> str:
        seal = content_hash(_payload(self), prefix=_PREFIX + "receipt-integrity-")
        expected = content_hash(_payload(self), prefix=_PREFIX + "receipt-")
        if seal != self._integrity_seal or expected != self.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.MUTATED_CUSTODY,
                "live-cover receipt no longer replays",
            )
        return self.receipt_hash


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLegacyBridgeRequirement:
    live_cover_receipt_hash: str
    missing_issuer_id: str = OUTSIDE_PAIR_LIVE_COVER_LEGACY_MISSING_ISSUER
    must_replay_legacy_eligible_from_kkt_identity: bool = True
    must_replay_legacy_required_from_kkt_identity: bool = True
    must_replay_already_removed_from_parent: bool = True
    must_replay_selectable_from_eligible_and_parent: bool = True
    must_replay_anchor_index_hash_from_frozen_subset: bool = True
    must_replay_shell_hash_from_structural_freeze: bool = True
    counts_alone_are_equivalence_evidence: bool = False
    route_scale_phase_b_d2h_allowed: bool = False
    downstream_authority: bool = False
    schema: str = OUTSIDE_PAIR_LIVE_COVER_LEGACY_BRIDGE_REQUIREMENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "live_cover_receipt_hash",
            _identity(self.live_cover_receipt_hash, name="live-cover receipt hash"),
        )
        object.__setattr__(
            self,
            "missing_issuer_id",
            _identity(self.missing_issuer_id, name="missing issuer id"),
        )
        required = (
            self.must_replay_legacy_eligible_from_kkt_identity,
            self.must_replay_legacy_required_from_kkt_identity,
            self.must_replay_already_removed_from_parent,
            self.must_replay_selectable_from_eligible_and_parent,
            self.must_replay_anchor_index_hash_from_frozen_subset,
            self.must_replay_shell_hash_from_structural_freeze,
        )
        if any(type(value) is not bool or value is not True for value in required):
            raise ValueError("legacy bridge requirement lost an equivalence obligation")
        _closed(
            (
                self.counts_alone_are_equivalence_evidence,
                self.route_scale_phase_b_d2h_allowed,
                self.downstream_authority,
            ),
            owner="legacy cover bridge requirement",
        )
        if self.schema != OUTSIDE_PAIR_LIVE_COVER_LEGACY_BRIDGE_REQUIREMENT_SCHEMA:
            raise ValueError("unsupported legacy bridge requirement schema")

    @property
    def requirement_hash(self) -> str:
        return content_hash(_payload(self), prefix=_PREFIX + "legacy-requirement-")


@dataclass(frozen=True, slots=True)
class _ResolvedAnchor:
    q_branch: Any
    kkt_branch: Any
    branch_source: Any
    resource_alias: Any
    bundle: Any
    endpoint: Any
    subset: Any
    shell_routes: tuple[int, ...]
    selected_kkt_branch_receipt_hash: str


@dataclass(frozen=True, slots=True)
class _SourceJoin:
    registration: Any
    structural_receipt: Any
    binding_receipt: Any
    d_receipt: Any
    residual_receipt: Any
    contexts: tuple[str, ...]
    anchor: _ResolvedAnchor
    eligible_mask_identity_hash: str
    required_mask_identity_hash: str


def _resolve_rounding_anchor(
    *,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    alias_binding: V1BlockSupportAliasBinding,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
) -> _ResolvedAnchor:
    provider = physical_block_support_binding.structural_provider
    structural = provider.receipt
    try:
        q_branch = V1SignedWorkVariableShellQBranch(alias_binding.branch_id)
        kkt_branch = V1RelaxedKKTBranch(alias_binding.branch_id)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.ALIAS_MISMATCH,
            "alias branch is not one registered Q/KKT branch",
        )
    branch_source = provider.registry.branch_source(q_branch)
    resource_alias = branch_source.resource_alias(alias_binding.resource_alias_id)
    bundles = tuple(
        value
        for value in provider.branch_bundles
        if value.execution_receipt.selected_branch is kkt_branch
    )
    if len(bundles) != 1:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
            "structural custody lacks one exact frozen alias branch",
        )
    bundle = bundles[0]
    frozen = bundle.frozen_ladder
    endpoints = tuple(
        action
        for action in frozen.actions
        if action.cumulative_removed_count == resource_alias.total_removed_count
    )
    if len(endpoints) != 1:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "frozen branch ladder has no unique alias-cardinality endpoint",
        )
    endpoint = endpoints[0]
    subsets = tuple(
        value
        for value in branch_source.subset_identities
        if value.removed_count == endpoint.cumulative_removed_count
        and value.removed_route_index_hash == endpoint.removed_route_index_hash
        and len(value.shell_offsets) == resource_alias.shell_prefix_count
    )
    if len(subsets) != 1:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "frozen endpoint does not replay one registered live subset",
        )
    subset = subsets[0]
    shell_routes = tuple(
        structural.shell_indices[offset] for offset in subset.shell_offsets
    )
    try:
        branch_index = kkt_state.branch_order.index(kkt_branch)
    except ValueError:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
            "KKT state does not retain the alias branch",
        )
    selected_receipt = kkt_state.branch_receipt_hashes[branch_index]
    execution = bundle.execution_receipt
    branch_identity = bundle.branch_identity
    if (
        alias_binding.source_receipt_hash
        != branch_source.registration.source_receipt_hash
        or alias_binding.source_replay_hash
        != branch_source.registration.source_replay_hash
        or alias_binding.cardinality != resource_alias.total_removed_count
        or resource_alias.fixed_outside_removed_count
        != structural.fixed_outside_removed_count
        or resource_alias.total_removed_count
        != resource_alias.fixed_outside_removed_count
        + resource_alias.shell_prefix_count
        or tuple(sorted(shell_routes)) != shell_routes
        or structural.fixed_outside_removed_count + len(shell_routes)
        != endpoint.cumulative_removed_count
        or subset.parent_mask_hash != structural.current_mask_hash
        or frozen.parent_mask_hash != structural.current_mask_hash
        or frozen.route_axis_hash != structural.route_axis_hash
        or frozen.route_count != structural.route_count
        or bundle.production_cuda_execution is not True
        or bundle.private_cpu_reference_execution is not False
        or frozen.handle_kind != "torch_cuda"
        or execution.state_identity_hash != kkt_state.identity_hash
        or execution.problem_hash != kkt_state.problem_hash
        or execution.policy_hash != kkt_state.policy_hash
        or execution.parent_mask_hash != structural.current_mask_hash
        or execution.source_registration_hash != structural.source_registration_hash
        or branch_identity.signed_work_branch_receipt_hash != selected_receipt
        or bundle.execution_hash not in structural.branch_execution_hashes
        or frozen.receipt_hash not in structural.branch_frozen_ladder_hashes
        or frozen.label_commitment not in structural.branch_label_commitments
        or subset.streamed_hash_custody_family_hash
        not in structural.branch_streamed_hash_custody_family_hashes
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
            "frozen rounding anchor differs from structural/KKT source custody",
        )
    return _ResolvedAnchor(
        q_branch=q_branch,
        kkt_branch=kkt_branch,
        branch_source=branch_source,
        resource_alias=resource_alias,
        bundle=bundle,
        endpoint=endpoint,
        subset=subset,
        shell_routes=shell_routes,
        selected_kkt_branch_receipt_hash=selected_receipt,
    )


def _validate_source_join(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
    block_index: int,
) -> _SourceJoin:
    if (
        type(block_witness_source) is not V1BlockSupportWitnessSource
        or type(alias_binding) is not V1BlockSupportAliasBinding
        or type(physical_block_support_binding)
        is not V1SignedWorkPhysicalAlternateBlockSupportBinding
        or type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "live cover requires exact witness/alias/physical/KKT owners",
        )
    index = _integer(block_index, name="block index")
    try:
        physical_block_support_binding.assert_live()
        registration = block_witness_source.registration
        registration.assert_integrity()
        contexts = block_witness_source.block_context_ids(index)
        block_witness_source.block_ids
        # These properties both prove liveness and fail closed when the
        # reference-only state lacks durable source-mask identities.
        eligible_identity = kkt_state.eligible_mask_identity_hash
        required_identity = kkt_state.required_mask_identity_hash
        kkt_route_count = kkt_state.route_count
        kkt_device = kkt_state.device
    except V1BlockSupportOutsidePairLiveCoverRefusal:
        raise
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        detail = str(error)
        code = (
            V1BlockSupportOutsidePairLiveCoverRefusalCode.MISSING_MASK_IDENTITY
            if "mask identit" in detail
            else V1BlockSupportOutsidePairLiveCoverRefusalCode.STALE_SOURCE
        )
        _refuse(code, f"one registered live-cover source is unavailable: {error}")
    structural_provider = physical_block_support_binding.structural_provider
    structural = structural_provider.receipt
    binding = physical_block_support_binding.receipt
    d_receipt = physical_block_support_binding.d_fixed_support.receipt
    residual_receipt = physical_block_support_binding.residual_support.receipt
    aliases = {value.alias_hash for value in registration.alias_bindings}
    expected_source_context = _source_context_axis_hash(contexts)
    registry_receipt = structural_provider.registry.receipt
    if alias_binding.alias_hash not in aliases:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.ALIAS_MISMATCH,
            "alias is absent from the sealed block-support axis",
        )
    if (
        binding.residual_block_ordinal != index
        or residual_receipt.block_ordinal != index
        or residual_receipt.context_axis_hash != expected_source_context
        or residual_receipt.context_count != len(contexts)
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.CONTEXT_MISMATCH,
            "per-block residual source differs from the sealed context block",
        )
    common_pairs = (
        (registration.chart_hash, structural.chart_identity_hash),
        (registration.current_mask_hash, structural.current_mask_hash),
        (registration.generation_index, structural.proposal_generation),
        (registration.route_axis_hash, structural.route_axis_hash),
        (registration.registered_d_prefix_hash, structural.registered_d_prefix_hash),
        (
            registration.registered_d_w2_source_hash,
            structural.registered_d_w2_prefix_hash,
        ),
        (registration.preferred_action_hash, structural.preferred_action_hash),
        (
            registration.preferred_action_cardinality,
            structural.preferred_selected_count,
        ),
        (
            registration.complete_alternate_domain_receipt_hash,
            structural.registry_receipt_hash,
        ),
        (
            registration.complete_alternate_domain_hash,
            registry_receipt.action_cell_axis_hash,
        ),
        (binding.structural_domain_receipt_hash, structural.receipt_hash),
        (binding.d_fixed_support_receipt_hash, d_receipt.receipt_hash),
        (
            binding.residual_support_receipt_hash,
            residual_receipt.receipt_hash,
        ),
        (binding.source_registration_hash, structural.source_registration_hash),
        (binding.chart_identity_hash, structural.chart_identity_hash),
        (binding.current_mask_hash, structural.current_mask_hash),
        (binding.proposal_generation, structural.proposal_generation),
        (binding.route_axis_hash, structural.route_axis_hash),
        (binding.route_id_axis_hash, structural.route_id_axis_hash),
        (
            binding.fixed_membership_binding_hash,
            structural.fixed_membership_binding_hash,
        ),
        (kkt_route_count, structural.route_count),
        (kkt_device, structural.resident_device_identity),
    )
    if any(left != right for left, right in common_pairs):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
            "witness/domain/block/KKT chart, mask, generation, or prefix differs",
        )
    anchor = _resolve_rounding_anchor(
        physical_block_support_binding=physical_block_support_binding,
        alias_binding=alias_binding,
        kkt_state=kkt_state,
    )
    return _SourceJoin(
        registration=registration,
        structural_receipt=structural,
        binding_receipt=binding,
        d_receipt=d_receipt,
        residual_receipt=residual_receipt,
        contexts=tuple(contexts),
        anchor=anchor,
        eligible_mask_identity_hash=eligible_identity,
        required_mask_identity_hash=required_identity,
    )


def _axis_identities(*, join: _SourceJoin, scan: _LiveCoverScan) -> dict[str, str]:
    structural = join.structural_receipt
    anchor = join.anchor
    shell_hash = _derived_membership_identity(
        "bounded_shell",
        route_count=structural.route_count,
        sources=(structural.shell_freeze_receipt_hash, structural.shell_indices),
    )
    anchor_hash = _derived_membership_identity(
        "rounding_anchor_P_alpha",
        route_count=structural.route_count,
        sources=(
            structural.fixed_membership_binding_hash,
            structural.fixed_outside_removed_route_index_hash,
            anchor.subset.removed_route_index_hash,
            anchor.subset.subset_hash,
            anchor.subset.shell_offsets,
        ),
    )
    parent_hash = _derived_membership_identity(
        "parent_live",
        route_count=structural.route_count,
        sources=(structural.receipt_hash, structural.current_mask_hash),
    )
    selectable_hash = _derived_membership_identity(
        "selectable_equals_eligible_and_parent_live",
        route_count=structural.route_count,
        sources=(join.eligible_mask_identity_hash, parent_hash),
    )
    p1 = _derived_membership_identity(
        "add_outside_drop_P_alpha_minus_required",
        route_count=structural.route_count,
        sources=(anchor_hash, join.required_mask_identity_hash),
    )
    q1 = _derived_membership_identity(
        "add_outside_add_selectable_minus_P_alpha_union_shell",
        route_count=structural.route_count,
        sources=(selectable_hash, anchor_hash, shell_hash),
    )
    p2 = _derived_membership_identity(
        "drop_outside_add_inside_drop_P_alpha_minus_required_minus_shell",
        route_count=structural.route_count,
        sources=(p1, shell_hash),
    )
    q2 = _derived_membership_identity(
        "drop_outside_add_inside_add_selectable_minus_P_alpha_intersect_shell",
        route_count=structural.route_count,
        sources=(selectable_hash, anchor_hash, shell_hash),
    )
    pair = content_hash(
        {
            "rule": OUTSIDE_PAIR_LIVE_COVER_RULE,
            "route_count": structural.route_count,
            "rounding_anchor_membership_identity_hash": anchor_hash,
            "rectangle_axes": (
                (p1, q1, scan.add_outside_rectangle_pair_count),
                (p2, q2, scan.drop_outside_add_inside_rectangle_pair_count),
            ),
            "complete_pair_count": scan.complete_pair_count,
        },
        prefix=_PREFIX + "pair-axis-commitment-",
    )
    return {
        "shell": shell_hash,
        "anchor": anchor_hash,
        "parent": parent_hash,
        "selectable": selectable_hash,
        "p1": p1,
        "q1": q1,
        "p2": p2,
        "q2": q2,
        "pair": pair,
    }


class V1BlockSupportOutsidePairLiveCoverLease(AbstractContextManager):
    """Zero-copy joint borrow of structural and KKT membership sources."""

    __slots__ = (
        "_owner",
        "_token",
        "_structural_lease",
        "_kkt_lease",
        "_closed",
        "live_cover_receipt_hash",
        "source_native_pair_axis_commitment_hash",
        "rounding_anchor_shell_route_indices",
        "route_count",
        "device_identity",
        "route_scale_d2h_bytes",
        "route_scale_content_digest_bytes",
        "downstream_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1BlockSupportOutsidePairLiveCoverProvider",
        token: object,
        structural_lease: Any,
        kkt_lease: Any,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _LEASE_TOKEN:
            raise TypeError("live-cover leases are provider-issued")
        self._owner = owner
        self._token = token
        self._structural_lease = structural_lease
        self._kkt_lease = kkt_lease
        self._closed = False
        receipt = owner.receipt
        self.live_cover_receipt_hash = receipt.receipt_hash
        self.source_native_pair_axis_commitment_hash = (
            receipt.source_native_pair_axis_commitment_hash
        )
        self.rounding_anchor_shell_route_indices = (
            receipt.rounding_anchor_shell_route_indices
        )
        self.route_count = receipt.route_count
        self.device_identity = receipt.resident_device_identity
        self.route_scale_d2h_bytes = 0
        self.route_scale_content_digest_bytes = 0
        self.downstream_authority = False
        self.schema = OUTSIDE_PAIR_LIVE_COVER_LEASE_SCHEMA

    def _require_open(self) -> None:
        if self._closed or self._owner is None:
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.EVICTED,
                "live-cover lease is closed",
            )
        self._owner._assert_lease_token(self._token)

    def live_membership_sources(self) -> tuple[Any, Any, Any, Any]:
        """Return ``fixed, parent_packed, eligible, required`` without copies."""

        self._require_open()
        fixed = self._structural_lease.fixed_membership()
        parent = self._structural_lease.resident_parent_packed()
        _resource, eligible, required = self._kkt_lease.live_source_tensors()
        receipt = self._owner.receipt
        if (
            self._structural_lease.physical_domain_receipt_hash
            != receipt.structural_domain_receipt_hash
            or self._structural_lease.fixed_membership_binding_hash
            != receipt.fixed_membership_binding_hash
            or self._structural_lease.parent_mask_hash != receipt.parent_mask_hash
            or self._structural_lease.route_count != receipt.route_count
            or self._structural_lease.route_axis_hash != receipt.route_axis_hash
            or tuple(self._structural_lease.shell_indices) != receipt.shell_indices
            or self._kkt_lease.state_identity_hash != receipt.kkt_state_identity_hash
            or self._kkt_lease.problem_hash != receipt.kkt_problem_hash
            or self._kkt_lease.policy_hash != receipt.kkt_policy_hash
            or self._kkt_lease.eligible_mask_identity_hash
            != receipt.kkt_eligible_mask_identity_hash
            or self._kkt_lease.required_mask_identity_hash
            != receipt.kkt_required_mask_identity_hash
            or self._kkt_lease.route_count != receipt.route_count
            or self._kkt_lease.device != receipt.resident_device_identity
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.MUTATED_CUSTODY,
                "live-cover lease sources changed identity",
            )
        return fixed, parent, eligible, required

    def close(self) -> None:
        if self._closed:
            return
        owner = self._owner
        token = self._token
        first_error: BaseException | None = None
        try:
            try:
                self._kkt_lease.close()
            except BaseException as error:  # preserve mutation refusals on release
                first_error = error
            try:
                self._structural_lease.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
            owner._close_lease_token(token)
        finally:
            self._owner = None
            self._token = None
            self._structural_lease = None
            self._kkt_lease = None
            self._closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "V1BlockSupportOutsidePairLiveCoverLease":
        self.live_membership_sources()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live-cover leases are nonserializable")


class V1BlockSupportOutsidePairLiveCoverProvider:
    """Live owner of a compact source-native cover receipt and source leases."""

    __slots__ = (
        "receipt",
        "forecast",
        "design",
        "block_witness_source",
        "alias_binding",
        "physical_block_support_binding",
        "kkt_state",
        "_live_identity_hash",
        "_leases",
        "_lock",
        "_evicted",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1BlockSupportOutsidePairLiveCoverReceipt,
        forecast: V1BlockSupportOutsidePairLiveCoverHBMForecast,
        design: V1BlockSupportOutsidePairLiveCoverDesign,
        block_witness_source: V1BlockSupportWitnessSource,
        alias_binding: V1BlockSupportAliasBinding,
        physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
        kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
        _issuer_token: object,
    ) -> None:
        if _issuer_token is not _PROVIDER_TOKEN:
            raise TypeError("live-cover providers are factory-issued")
        if (
            type(receipt) is not V1BlockSupportOutsidePairLiveCoverReceipt
            or type(forecast) is not V1BlockSupportOutsidePairLiveCoverHBMForecast
            or type(design) is not V1BlockSupportOutsidePairLiveCoverDesign
            or type(block_witness_source) is not V1BlockSupportWitnessSource
            or type(alias_binding) is not V1BlockSupportAliasBinding
            or type(physical_block_support_binding)
            is not V1SignedWorkPhysicalAlternateBlockSupportBinding
            or type(kkt_state) is not V1SignedWorkLeastWorkCUDAStateHandle
        ):
            raise TypeError("live-cover provider source types changed")
        receipt.assert_integrity()
        if (
            receipt.design_hash != design.design_hash
            or receipt.hbm_forecast_hash != forecast.receipt_hash
            or receipt.alias_hash != alias_binding.alias_hash
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
                "live-cover provider inputs do not replay its receipt",
            )
        self.receipt = receipt
        self.forecast = forecast
        self.design = design
        self.block_witness_source = block_witness_source
        self.alias_binding = alias_binding
        self.physical_block_support_binding = physical_block_support_binding
        self.kkt_state = kkt_state
        self._leases: set[object] = set()
        self._lock = RLock()
        self._evicted = False
        self._live_identity_hash = self._current_live_identity_hash()
        self.assert_live()

    def _current_live_identity_hash(self) -> str:
        return content_hash(
            {
                "receipt_hash": self.receipt.receipt_hash,
                "block_support_registration_hash": (
                    self.block_witness_source.registration.registration_hash
                ),
                "alias_hash": self.alias_binding.alias_hash,
                "physical_block_binding_receipt_hash": (
                    self.physical_block_support_binding.receipt.receipt_hash
                ),
                "kkt_state_identity_hash": self.kkt_state.identity_hash,
            },
            prefix=_PREFIX + "live-provider-identity-",
        )

    def assert_live(self) -> "V1BlockSupportOutsidePairLiveCoverProvider":
        with self._lock:
            if self._evicted:
                _refuse(
                    V1BlockSupportOutsidePairLiveCoverRefusalCode.EVICTED,
                    "live-cover provider was evicted",
                )
            self.receipt.assert_integrity()
            join = _validate_source_join(
                block_witness_source=self.block_witness_source,
                alias_binding=self.alias_binding,
                physical_block_support_binding=self.physical_block_support_binding,
                kkt_state=self.kkt_state,
                block_index=self.receipt.block_index,
            )
            anchor = join.anchor
            if (
                self._current_live_identity_hash() != self._live_identity_hash
                or join.structural_receipt.receipt_hash
                != self.receipt.structural_domain_receipt_hash
                or join.eligible_mask_identity_hash
                != self.receipt.kkt_eligible_mask_identity_hash
                or join.required_mask_identity_hash
                != self.receipt.kkt_required_mask_identity_hash
                or anchor.subset.subset_hash
                != self.receipt.rounding_anchor_subset_identity_hash
                or anchor.subset.removed_route_index_hash
                != self.receipt.rounding_anchor_removed_route_index_hash
                or anchor.shell_routes
                != self.receipt.rounding_anchor_shell_route_indices
            ):
                _refuse(
                    V1BlockSupportOutsidePairLiveCoverRefusalCode.STALE_SOURCE,
                    "live-cover source identity changed",
                )
            return self

    def open_live_cover_lease(self) -> V1BlockSupportOutsidePairLiveCoverLease:
        with self._lock:
            self.assert_live()
            token = object()
            purpose = content_hash(
                {
                    "operation": "block_support_outside_pair_live_cover",
                    "live_cover_receipt_hash": self.receipt.receipt_hash,
                    "lease_ordinal": len(self._leases),
                },
                prefix=_PREFIX + "borrow-purpose-",
            )
            structural_lease = (
                self.physical_block_support_binding.open_fixed_membership_lease(
                    borrow_purpose_hash=purpose
                )
            )
            try:
                kkt_lease = self.kkt_state.borrow_for_block_support_outside_pair(
                    expected_problem_hash=self.receipt.kkt_problem_hash,
                    expected_policy_hash=self.receipt.kkt_policy_hash,
                    expected_branch_receipt_hashes=self.kkt_state.branch_receipt_hashes,
                )
            except BaseException:
                structural_lease.close()
                raise
            self._leases.add(token)
            return V1BlockSupportOutsidePairLiveCoverLease(
                owner=self,
                token=token,
                structural_lease=structural_lease,
                kkt_lease=kkt_lease,
                _issuer_token=_LEASE_TOKEN,
            )

    def _assert_lease_token(self, token: object) -> None:
        with self._lock:
            if self._evicted or token not in self._leases:
                _refuse(
                    V1BlockSupportOutsidePairLiveCoverRefusalCode.EVICTED,
                    "live-cover lease token is no longer active",
                )
            self.assert_live()

    def _close_lease_token(self, token: object) -> None:
        with self._lock:
            if token not in self._leases:
                _refuse(
                    V1BlockSupportOutsidePairLiveCoverRefusalCode.EVICTED,
                    "live-cover lease token was already released",
                )
            self._leases.remove(token)

    def describe_legacy_bridge_requirement(
        self,
    ) -> V1BlockSupportOutsidePairLegacyBridgeRequirement:
        self.assert_live()
        return V1BlockSupportOutsidePairLegacyBridgeRequirement(
            live_cover_receipt_hash=self.receipt.receipt_hash
        )

    def require_legacy_cover_equivalence(self) -> NoReturn:
        requirement = self.describe_legacy_bridge_requirement()
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.LEGACY_EQUIVALENCE_UNAVAILABLE,
            "legacy tuple/hash equivalence is not source-proven; missing issuer "
            f"{requirement.missing_issuer_id}",
        )

    def evict(self) -> None:
        with self._lock:
            self.assert_live()
            if self._leases:
                _refuse(
                    V1BlockSupportOutsidePairLiveCoverRefusalCode.EVICTED,
                    "cannot evict a live-cover provider with active leases",
                )
            self.block_witness_source = None
            self.alias_binding = None
            self.physical_block_support_binding = None
            self.kkt_state = None
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live-cover providers are nonserializable")


def _issue_receipt(
    *,
    join: _SourceJoin,
    scan: _LiveCoverScan,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
    block_index: int,
    design: V1BlockSupportOutsidePairLiveCoverDesign,
    forecast: V1BlockSupportOutsidePairLiveCoverHBMForecast,
    legacy_pair_cover_receipt_hash_diagnostic: str | None,
    legacy_pair_axis_commitment_hash_diagnostic: str | None,
) -> V1BlockSupportOutsidePairLiveCoverReceipt:
    registration = join.registration
    structural = join.structural_receipt
    binding = join.binding_receipt
    anchor = join.anchor
    resource_alias = anchor.resource_alias
    subset = anchor.subset
    bundle = anchor.bundle
    endpoint = anchor.endpoint
    axes = _axis_identities(join=join, scan=scan)
    return V1BlockSupportOutsidePairLiveCoverReceipt(
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.receipt_hash,
        block_support_registration_hash=registration.registration_hash,
        block_index=block_index,
        block_id=block_witness_source.block_ids[block_index],
        block_context_axis_hash=_canonical_context_axis_hash(join.contexts),
        residual_source_context_axis_hash=_source_context_axis_hash(join.contexts),
        physical_block_support_binding_receipt_hash=binding.receipt_hash,
        structural_domain_receipt_hash=structural.receipt_hash,
        registry_receipt_hash=structural.registry_receipt_hash,
        complete_alternate_domain_hash=registration.complete_alternate_domain_hash,
        complete_alternate_domain_receipt_hash=(
            registration.complete_alternate_domain_receipt_hash
        ),
        source_registration_hash=structural.source_registration_hash,
        model_identity_hash=structural.model_identity_hash,
        chart_identity_hash=structural.chart_identity_hash,
        current_mask_hash=structural.current_mask_hash,
        proposal_generation=structural.proposal_generation,
        route_axis_hash=structural.route_axis_hash,
        route_id_axis_hash=structural.route_id_axis_hash,
        resident_device_identity=structural.resident_device_identity,
        native_fiber_axis_receipt_hash=structural.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=structural.native_fiber_measure_hash,
        registered_d_prefix_hash=structural.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=structural.registered_d_w2_prefix_hash,
        preferred_action_binding_hash=structural.preferred_action_binding_hash,
        comparison_preferred_action_hash=structural.preferred_action_hash,
        comparison_preferred_removed_route_index_hash=(
            structural.preferred_removed_route_index_hash
        ),
        comparison_preferred_selected_count=structural.preferred_selected_count,
        alias_hash=alias_binding.alias_hash,
        branch_id=alias_binding.branch_id,
        fd_rank=alias_binding.fd_rank,
        resource_alias_id=alias_binding.resource_alias_id,
        source_receipt_hash=alias_binding.source_receipt_hash,
        source_replay_hash=alias_binding.source_replay_hash,
        cardinality_alias_id=alias_binding.cardinality_alias_id,
        alias_fixed_outside_removed_count=resource_alias.fixed_outside_removed_count,
        alias_shell_prefix_count=resource_alias.shell_prefix_count,
        alias_total_removed_count=resource_alias.total_removed_count,
        fixed_membership_binding_hash=structural.fixed_membership_binding_hash,
        fixed_outside_removed_route_index_hash=(
            structural.fixed_outside_removed_route_index_hash
        ),
        parent_mask_hash=structural.current_mask_hash,
        shell_freeze_receipt_hash=structural.shell_freeze_receipt_hash,
        shell_indices=structural.shell_indices,
        shell_membership_identity_hash=axes["shell"],
        branch_execution_hash=bundle.execution_hash,
        source_rounding_hash=bundle.branch_identity.source_rounding_hash,
        frozen_ladder_receipt_hash=bundle.frozen_ladder.receipt_hash,
        frozen_label_commitment=bundle.frozen_ladder.label_commitment,
        frozen_eligibility_commitment=bundle.frozen_ladder.eligibility_commitment,
        frozen_endpoint_action_receipt_hash=endpoint.receipt_hash,
        frozen_endpoint_packed_action_hash=endpoint.packed_action_hash,
        rounding_anchor_action_coordinate_hash=(subset.physical_action_coordinate_hash),
        rounding_anchor_removed_route_index_hash=subset.removed_route_index_hash,
        rounding_anchor_subset_identity_hash=subset.subset_hash,
        rounding_anchor_selected_shell_offsets=subset.shell_offsets,
        rounding_anchor_shell_route_indices=anchor.shell_routes,
        rounding_anchor_membership_identity_hash=axes["anchor"],
        kkt_problem_hash=kkt_state.problem_hash,
        kkt_policy_hash=kkt_state.policy_hash,
        kkt_state_identity_hash=kkt_state.identity_hash,
        kkt_branch_receipt_hash=anchor.selected_kkt_branch_receipt_hash,
        kkt_eligible_mask_identity_hash=join.eligible_mask_identity_hash,
        kkt_required_mask_identity_hash=join.required_mask_identity_hash,
        parent_live_membership_identity_hash=axes["parent"],
        selectable_membership_identity_hash=axes["selectable"],
        add_outside_drop_axis_identity_hash=axes["p1"],
        add_outside_add_axis_identity_hash=axes["q1"],
        drop_outside_add_inside_drop_axis_identity_hash=axes["p2"],
        drop_outside_add_inside_add_axis_identity_hash=axes["q2"],
        source_native_pair_axis_commitment_hash=axes["pair"],
        registered_legacy_pair_cover_hash_diagnostic=(
            alias_binding.outside_pair_cover_hash
        ),
        legacy_pair_cover_receipt_hash_diagnostic=(
            legacy_pair_cover_receipt_hash_diagnostic
        ),
        legacy_pair_axis_commitment_hash_diagnostic=(
            legacy_pair_axis_commitment_hash_diagnostic
        ),
        route_count=structural.route_count,
        eligible_count=scan.eligible_count,
        required_count=scan.required_count,
        parent_live_count=scan.parent_live_count,
        selectable_count=scan.selectable_count,
        fixed_outside_count=structural.fixed_outside_removed_count,
        shell_count=scan.shell_count,
        rounding_anchor_count=scan.rounding_anchor_count,
        add_outside_droppable_count=scan.droppable_anchor_count,
        add_outside_addable_count=scan.addable_outside_shell_count,
        drop_outside_add_inside_droppable_count=(
            scan.droppable_anchor_outside_shell_count
        ),
        drop_outside_add_inside_addable_count=scan.addable_inside_shell_count,
        add_outside_rectangle_pair_count=(scan.add_outside_rectangle_pair_count),
        drop_outside_add_inside_rectangle_pair_count=(
            scan.drop_outside_add_inside_rectangle_pair_count
        ),
        complete_pair_count=scan.complete_pair_count,
        route_tile_size=design.route_tile_size,
        route_tile_scan_count=scan.route_tile_scan_count,
        complete_route_scan_count=scan.scanned_route_count,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
        device_to_host_synchronization_count=scan.d2h_synchronization_count,
        route_scale_device_to_host_transfer_bytes=0,
        route_scale_content_digest_bytes=0,
        owned_route_scale_device_bytes=0,
        owned_pair_matrix_bytes=0,
        _issuer_token=_RECEIPT_TOKEN,
    )


def build_v1_block_support_outside_pair_live_cover_cuda(
    *,
    block_witness_source: V1BlockSupportWitnessSource,
    alias_binding: V1BlockSupportAliasBinding,
    physical_block_support_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    kkt_state: V1SignedWorkLeastWorkCUDAStateHandle,
    block_index: int,
    design: V1BlockSupportOutsidePairLiveCoverDesign,
    hbm_forecast: V1BlockSupportOutsidePairLiveCoverHBMForecast,
    legacy_pair_cover_receipt_hash_diagnostic: str | None = None,
    legacy_pair_axis_commitment_hash_diagnostic: str | None = None,
) -> V1BlockSupportOutsidePairLiveCoverProvider:
    """Build one production source-native live cover.

    Caller membership is intentionally absent from this API.  The complete
    route scan consumes only structural and KKT leases plus the frozen compact
    shell subset resolved from the registered alias.
    """

    if (
        type(design) is not V1BlockSupportOutsidePairLiveCoverDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairLiveCoverHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "production live cover requires exact design and HBM forecast",
        )
    if hbm_forecast.design_hash != design.design_hash or not hbm_forecast.admitted:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.HBM_FORECAST_REFUSED,
            "production live-cover HBM forecast was not admitted",
        )
    if (legacy_pair_cover_receipt_hash_diagnostic is None) != (
        legacy_pair_axis_commitment_hash_diagnostic is None
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "legacy diagnostic hashes must be supplied as a pair",
        )
    if legacy_pair_cover_receipt_hash_diagnostic is not None:
        _identity(
            legacy_pair_cover_receipt_hash_diagnostic,
            name="legacy pair-cover receipt diagnostic",
        )
        _identity(
            legacy_pair_axis_commitment_hash_diagnostic,
            name="legacy pair-axis commitment diagnostic",
        )
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.CUDA_REQUIRED,
            "production live-cover provider fails closed without CUDA",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.DETERMINISM_REQUIRED,
            "production live-cover scan requires deterministic algorithms",
        )
    join = _validate_source_join(
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        physical_block_support_binding=physical_block_support_binding,
        kkt_state=kkt_state,
        block_index=block_index,
    )
    structural = join.structural_receipt
    if (
        hbm_forecast.route_count != structural.route_count
        or hbm_forecast.route_tile_size != design.route_tile_size
        or structural.production_cuda_execution is not True
        or structural.private_cpu_reference_execution is not False
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
            "HBM or structural execution identity differs from production design",
        )
    purpose = content_hash(
        {
            "operation": "build_block_support_outside_pair_live_cover",
            "physical_block_binding_receipt_hash": join.binding_receipt.receipt_hash,
            "alias_hash": alias_binding.alias_hash,
            "kkt_state_identity_hash": kkt_state.identity_hash,
            "design_hash": design.design_hash,
        },
        prefix=_PREFIX + "builder-borrow-purpose-",
    )
    with (
        physical_block_support_binding.open_fixed_membership_lease(
            borrow_purpose_hash=purpose
        ) as structural_lease,
        kkt_state.borrow_for_block_support_outside_pair(
            expected_problem_hash=kkt_state.problem_hash,
            expected_policy_hash=kkt_state.policy_hash,
            expected_branch_receipt_hashes=kkt_state.branch_receipt_hashes,
        ) as kkt_lease,
    ):
        resource, eligible, required = kkt_lease.live_source_tensors()
        fixed = structural_lease.fixed_membership()
        parent = structural_lease.resident_parent_packed()
        device = fixed.device
        if (
            device.type != "cuda"
            or str(device) != structural.resident_device_identity
            or str(device) != kkt_lease.device
            or any(
                value.device != device
                for value in (parent, resource, eligible, required)
            )
            or resource.dtype is not torch.float64
            or tuple(resource.shape) != (structural.route_count,)
            or not resource.is_contiguous()
            or resource.requires_grad
            or structural_lease.physical_domain_receipt_hash != structural.receipt_hash
            or structural_lease.fixed_membership_binding_hash
            != structural.fixed_membership_binding_hash
            or structural_lease.parent_mask_hash != structural.current_mask_hash
            or structural_lease.route_axis_hash != structural.route_axis_hash
            or tuple(structural_lease.shell_indices) != structural.shell_indices
            or kkt_lease.state_identity_hash != kkt_state.identity_hash
            or kkt_lease.problem_hash != kkt_state.problem_hash
            or kkt_lease.policy_hash != kkt_state.policy_hash
            or kkt_lease.eligible_mask_identity_hash != join.eligible_mask_identity_hash
            or kkt_lease.required_mask_identity_hash != join.required_mask_identity_hash
            or kkt_lease.route_scale_d2h_bytes != 0
            or kkt_lease.route_scale_content_digest_bytes != 0
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveCoverRefusalCode.SOURCE_MISMATCH,
                "structural/KKT leases differ from registered live-cover custody",
            )
        scan = _scan_live_cover_membership(
            fixed_membership=fixed,
            resident_parent_packed=parent,
            eligible_membership=eligible,
            required_membership=required,
            shell_indices=structural.shell_indices,
            rounding_anchor_shell_indices=join.anchor.shell_routes,
            route_count=structural.route_count,
            route_tile_size=design.route_tile_size,
            require_cuda=True,
        )
    if (
        scan.rounding_anchor_count != join.anchor.resource_alias.total_removed_count
        or scan.rounding_anchor_count - len(join.anchor.shell_routes)
        != structural.fixed_outside_removed_count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INADMISSIBLE_ANCHOR,
            "scanned anchor count differs from the frozen alias cardinality",
        )
    receipt = _issue_receipt(
        join=join,
        scan=scan,
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        kkt_state=kkt_state,
        block_index=block_index,
        design=design,
        forecast=hbm_forecast,
        legacy_pair_cover_receipt_hash_diagnostic=(
            legacy_pair_cover_receipt_hash_diagnostic
        ),
        legacy_pair_axis_commitment_hash_diagnostic=(
            legacy_pair_axis_commitment_hash_diagnostic
        ),
    )
    return V1BlockSupportOutsidePairLiveCoverProvider(
        receipt=receipt,
        forecast=hbm_forecast,
        design=design,
        block_witness_source=block_witness_source,
        alias_binding=alias_binding,
        physical_block_support_binding=physical_block_support_binding,
        kkt_state=kkt_state,
        _issuer_token=_PROVIDER_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveCoverReference:
    """Small-domain host enumeration; never production or source authority."""

    route_count: int
    rounding_anchor_indices: tuple[int, ...]
    required_indices: tuple[int, ...]
    selectable_indices: tuple[int, ...]
    shell_indices: tuple[int, ...]
    add_outside_drop_indices: tuple[int, ...]
    add_outside_add_indices: tuple[int, ...]
    drop_outside_add_inside_drop_indices: tuple[int, ...]
    drop_outside_add_inside_add_indices: tuple[int, ...]
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_count: int
    route_tile_scan_count: int
    cpu_reference_test_only: bool = True
    production_authority: bool = False
    downstream_authority: bool = False
    schema: str = "partition_free_v1_outside_pair_live_cover_reference_v1"

    def __post_init__(self) -> None:
        if self.cpu_reference_test_only is not True:
            raise ValueError("outside-pair reference must remain test-only")
        _closed(
            (self.production_authority, self.downstream_authority),
            owner="outside-pair live-cover reference",
        )


def evaluate_v1_block_support_outside_pair_live_cover_reference(
    *,
    fixed_membership: Sequence[bool],
    parent_live_membership: Sequence[bool],
    eligible_membership: Sequence[bool],
    required_membership: Sequence[bool],
    shell_indices: Sequence[int],
    rounding_anchor_shell_indices: Sequence[int],
    route_tile_size: int = 8,
    maximum_route_count: int = 4096,
) -> V1BlockSupportOutsidePairLiveCoverReference:
    """Enumerate and independently replay a tiny exact live-cover domain."""

    if torch is None:
        raise RuntimeError("torch is required for the live-cover reference")
    fixed_axis = tuple(bool(value) for value in fixed_membership)
    parent_axis = tuple(bool(value) for value in parent_live_membership)
    eligible_axis = tuple(bool(value) for value in eligible_membership)
    required_axis = tuple(bool(value) for value in required_membership)
    count = len(fixed_axis)
    maximum = _integer(maximum_route_count, name="maximum route count", positive=True)
    if (
        not count
        or count > maximum
        or len(parent_axis) != count
        or len(eligible_axis) != count
        or len(required_axis) != count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INVALID_TYPE,
            "reference memberships differ in size or exceed the small-domain limit",
        )
    fixed = torch.tensor(fixed_axis, dtype=torch.bool)
    parent = torch.tensor(parent_axis, dtype=torch.bool)
    eligible = torch.tensor(eligible_axis, dtype=torch.bool)
    required = torch.tensor(required_axis, dtype=torch.bool)
    padding = (-count) % 8
    parent_padded = (
        torch.cat((parent, torch.zeros((padding,), dtype=torch.bool)))
        if padding
        else parent
    )
    weights = torch.tensor((128, 64, 32, 16, 8, 4, 2, 1), dtype=torch.uint8)
    parent_packed = (
        (parent_padded.reshape(-1, 8).to(torch.uint8) * weights.reshape(1, 8))
        .sum(dim=1)
        .to(torch.uint8)
        .contiguous()
    )
    scan = _scan_live_cover_membership(
        fixed_membership=fixed,
        resident_parent_packed=parent_packed,
        eligible_membership=eligible,
        required_membership=required,
        shell_indices=shell_indices,
        rounding_anchor_shell_indices=rounding_anchor_shell_indices,
        route_count=count,
        route_tile_size=route_tile_size,
        require_cuda=False,
    )
    shell = tuple(int(value) for value in shell_indices)
    anchor_shell = set(int(value) for value in rounding_anchor_shell_indices)
    shell_set = set(shell)
    anchor = tuple(
        index for index in range(count) if fixed_axis[index] or index in anchor_shell
    )
    anchor_set = set(anchor)
    required_indices = tuple(
        index for index, value in enumerate(required_axis) if value
    )
    selectable = tuple(
        index for index in range(count) if eligible_axis[index] and parent_axis[index]
    )
    selectable_set = set(selectable)
    p1 = tuple(index for index in anchor if not required_axis[index])
    q1 = tuple(
        index
        for index in selectable
        if index not in anchor_set and index not in shell_set
    )
    p2 = tuple(index for index in p1 if index not in shell_set)
    q2 = tuple(
        index
        for index in selectable_set
        if index not in anchor_set and index in shell_set
    )
    q2 = tuple(sorted(q2))
    expected = (
        len(anchor),
        len(required_indices),
        len(selectable),
        len(p1),
        len(q1),
        len(p2),
        len(q2),
    )
    observed = (
        scan.rounding_anchor_count,
        scan.required_count,
        scan.selectable_count,
        scan.droppable_anchor_count,
        scan.addable_outside_shell_count,
        scan.droppable_anchor_outside_shell_count,
        scan.addable_inside_shell_count,
    )
    if observed != expected:
        _refuse(
            V1BlockSupportOutsidePairLiveCoverRefusalCode.INCOMPLETE_ROUTE_SCAN,
            "tiled reference scan differs from direct rectangle enumeration",
        )
    return V1BlockSupportOutsidePairLiveCoverReference(
        route_count=count,
        rounding_anchor_indices=anchor,
        required_indices=required_indices,
        selectable_indices=selectable,
        shell_indices=shell,
        add_outside_drop_indices=p1,
        add_outside_add_indices=q1,
        drop_outside_add_inside_drop_indices=p2,
        drop_outside_add_inside_add_indices=q2,
        add_outside_rectangle_pair_count=scan.add_outside_rectangle_pair_count,
        drop_outside_add_inside_rectangle_pair_count=(
            scan.drop_outside_add_inside_rectangle_pair_count
        ),
        complete_pair_count=scan.complete_pair_count,
        route_tile_scan_count=scan.route_tile_scan_count,
    )


def describe_v1_block_support_outside_pair_legacy_bridge_requirement(
    *, live_cover_receipt_hash: str
) -> V1BlockSupportOutsidePairLegacyBridgeRequirement:
    return V1BlockSupportOutsidePairLegacyBridgeRequirement(
        live_cover_receipt_hash=live_cover_receipt_hash
    )


def require_v1_block_support_outside_pair_legacy_cover_equivalence(
    requirement: V1BlockSupportOutsidePairLegacyBridgeRequirement,
) -> NoReturn:
    if type(requirement) is not V1BlockSupportOutsidePairLegacyBridgeRequirement:
        raise TypeError("legacy equivalence gate requires its exact requirement")
    _refuse(
        V1BlockSupportOutsidePairLiveCoverRefusalCode.LEGACY_EQUIVALENCE_UNAVAILABLE,
        "legacy tuple/hash equivalence is not source-proven; missing issuer "
        f"{requirement.missing_issuer_id}",
    )


__all__ = (
    "OUTSIDE_PAIR_LIVE_COVER_AUTHORITY",
    "OUTSIDE_PAIR_LIVE_COVER_DESIGN_SCHEMA",
    "OUTSIDE_PAIR_LIVE_COVER_FORECAST_SCHEMA",
    "OUTSIDE_PAIR_LIVE_COVER_LEASE_SCHEMA",
    "OUTSIDE_PAIR_LIVE_COVER_LEGACY_BRIDGE_REQUIREMENT_SCHEMA",
    "OUTSIDE_PAIR_LIVE_COVER_LEGACY_MISSING_ISSUER",
    "OUTSIDE_PAIR_LIVE_COVER_RULE",
    "OUTSIDE_PAIR_LIVE_COVER_SCHEMA",
    "V1BlockSupportOutsidePairLegacyBridgeRequirement",
    "V1BlockSupportOutsidePairLiveCoverDesign",
    "V1BlockSupportOutsidePairLiveCoverHBMForecast",
    "V1BlockSupportOutsidePairLiveCoverLease",
    "V1BlockSupportOutsidePairLiveCoverProvider",
    "V1BlockSupportOutsidePairLiveCoverReceipt",
    "V1BlockSupportOutsidePairLiveCoverReference",
    "V1BlockSupportOutsidePairLiveCoverRefusal",
    "V1BlockSupportOutsidePairLiveCoverRefusalCode",
    "build_v1_block_support_outside_pair_live_cover_cuda",
    "build_v1_block_support_outside_pair_live_cover_design",
    "build_v1_block_support_outside_pair_live_cover_hbm_forecast",
    "describe_v1_block_support_outside_pair_legacy_bridge_requirement",
    "evaluate_v1_block_support_outside_pair_live_cover_reference",
    "require_v1_block_support_outside_pair_legacy_cover_equivalence",
)
