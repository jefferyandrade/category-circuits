"""Accelerator-resident sufficient statistics for whole-route pullback fields.

The canonical scorer owns the ephemeral context × head × route rows.  This
module reduces those rows before they cross the device boundary.  It is
deliberately a sufficient-statistic adapter—not another attribution
implementation—and therefore preserves the scorer's physical U→G→D values
exactly.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.determinism import DETERMINISTIC_EXECUTION_CACHE_VERSION
from mechanistic_systems.contracts.trust_geometry import GlobalTrustGeometrySpec
from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)
from mechanistic_systems.geometry.trust import (
    _ACCUMULATOR_DERIVATION_TOKEN,
    _SOURCE_ROW_CHAIN_SCHEMA,
    _STREAM_WEIGHT_CHAIN_SCHEMA,
    _advance_ordered_chain,
    _empty_ordered_chain,
    _finalize_ordered_chain,
    GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID,
    GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID,
    GLOBAL_TRUST_SOURCE_ROW_HASH_PREFIX,
    GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
    GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
    GlobalTrustGeometryDerivationReceipt,
    GlobalTrustGeometryState,
    global_trust_source_row_hash,
)
from mechanistic_systems.measurement.streaming_response_provider import (
    LocalAttributionMomentTensor,
    TripartitePullbackMomentTensor,
)
from mechanistic_systems.measurement.gpu_runtime import (
    cuda_reusable_memory_bytes,
    process_device_memory_budget_bytes,
)

try:  # Optional for import smoke.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


HOST_PRIMARY_ACCUMULATORS_ENVIRONMENT = "IRG_HOST_PRIMARY_ACCUMULATORS"
HOST_PRIMARY_WINDOW_ROUTES_ENVIRONMENT = "IRG_HOST_PRIMARY_WINDOW_ROUTES"
HOST_PRIMARY_WINDOW_BUDGET_ENVIRONMENT = "IRG_HOST_PRIMARY_WINDOW_BUDGET_BYTES"
_HOST_PRIMARY_DEFAULT_WINDOW_BUDGET_BYTES = 2 << 30
_FALSE_ENVIRONMENT_TOKENS = frozenset({"0", "false", "no", "off"})
_DIRECT_FD_SVD_MAX_ELEMENTS = 16_777_216
_GRAM_FD_PSD_TOLERANCE_MULTIPLIER = 64.0
_FD_ACCUMULATOR_SNAPSHOT_SCHEMA = "irg_device_fd_accumulator_snapshot_v1"
_FD_SOURCE_PROOF_SNAPSHOT_SCHEMA = "irg_bounded_ordered_fd_source_proof_snapshot_v1"
_FD_SNAPSHOT_DTYPE_IDENTITY = "ieee754_binary64_little_endian_v1"
_FD_ACCUMULATOR_SNAPSHOT_HASH_PREFIX = "irg-device-fd-accumulator-snapshot-"
_FD_SOURCE_PROOF_SNAPSHOT_HASH_PREFIX = "irg-bounded-fd-source-proof-snapshot-"
_FD_SNAPSHOT_RESTORE_HOST_STAGE_BYTES = 64 << 20
_DEVICE_FD_GEOMETRY_ISSUER = object()

# Census fields are finalized and combined on the compute device in float64.
# Device route chunk of the v6 finalize and look snapshot.  Chunking bounds the
# lifetime of device intermediates only; every expression is route-separable.
CENSUS_V6_ROUTE_CHUNK_ENVIRONMENT = "IRG_V6_C_ROUTE_CHUNK"
_CENSUS_V6_DEFAULT_ROUTE_CHUNK = 1 << 22
# Device residency budget of the census accumulators (both strata).  Above it
# the census keeps host-primary storage and the v6 finalize streams route
# chunks through the device instead.
CENSUS_V6_DEVICE_BUDGET_FRACTION_ENVIRONMENT = "IRG_V6_C_DEVICE_BUDGET_FRACTION"
_CENSUS_V6_DEFAULT_DEVICE_BUDGET_FRACTION = 0.5


def census_v6_route_chunk() -> int:
    raw = str(os.environ.get(CENSUS_V6_ROUTE_CHUNK_ENVIRONMENT, "")).strip()
    value = int(raw) if raw else _CENSUS_V6_DEFAULT_ROUTE_CHUNK
    if value <= 0:
        raise ValueError(f"{CENSUS_V6_ROUTE_CHUNK_ENVIRONMENT} must be positive")
    return value


def census_v6_device_budget_fraction() -> float:
    raw = str(os.environ.get(CENSUS_V6_DEVICE_BUDGET_FRACTION_ENVIRONMENT, "")).strip()
    value = float(raw) if raw else _CENSUS_V6_DEFAULT_DEVICE_BUDGET_FRACTION
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{CENSUS_V6_DEVICE_BUDGET_FRACTION_ENVIRONMENT} must lie in [0,1]"
        )
    return value


def _v6_outer(mean: Any) -> Any:
    """``einsum("hr,jr->rhj", mean, mean)`` as explicit single products."""

    transposed = mean.transpose(0, 1)
    return transposed[:, :, None] * transposed[:, None, :]


def host_primary_accumulators_enabled() -> bool:
    """Whether route-scale moment slabs should remain host-primary."""

    import os

    return (
        str(os.environ.get(HOST_PRIMARY_ACCUMULATORS_ENVIRONMENT, "")).strip().lower()
        not in _FALSE_ENVIRONMENT_TOKENS
    )


def host_primary_bytes_per_route(*, head_count: int, sketch_rank: int) -> int:
    """Return fp64 routed-slab bytes per logical route."""

    heads = int(head_count)
    rank = int(sketch_rank)
    return 8 * (heads + 2 * heads * heads + 6 + rank * heads)


def default_host_primary_window_width(*, head_count: int, sketch_rank: int, slots: int = 2) -> int:
    """Derive a bounded route window from the configured device budget."""

    import os

    routes_raw = str(os.environ.get(HOST_PRIMARY_WINDOW_ROUTES_ENVIRONMENT, "")).strip()
    if routes_raw:
        width = int(routes_raw)
        if width <= 0:
            raise ValueError("host-primary window route override is invalid")
        return width
    budget_raw = str(os.environ.get(HOST_PRIMARY_WINDOW_BUDGET_ENVIRONMENT, "")).strip()
    budget = int(budget_raw) if budget_raw else _HOST_PRIMARY_DEFAULT_WINDOW_BUDGET_BYTES
    if budget <= 0:
        raise ValueError("host-primary window byte budget is invalid")
    budget = process_device_memory_budget_bytes(budget)
    per_route = host_primary_bytes_per_route(
        head_count=head_count,
        sketch_rank=sketch_rank,
    )
    return max(1, budget // (max(1, int(slots)) * per_route))


def _check_torch_all_finite(name: str, tensor: Any) -> None:
    finite = torch.isfinite(tensor).all()
    if tensor.device.type == "cuda":
        torch._assert_async(finite, f"{name} contains non-finite values")
    elif not bool(finite):
        raise ValueError(f"{name} contains non-finite values")


def _immutable_fd_snapshot_array(value: Any, *, name: str) -> np.ndarray:
    """Return a truly read-only, canonical little-endian fp64 array."""

    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have the device-independent fp64 dtype")
    contiguous = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.dtype("<f8"))
    return frozen.reshape(contiguous.shape)


def _require_exact_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an exact nonnegative integer")
    return value


def _require_prefixed_sha256(value: Any, *, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ValueError(f"{name} has the wrong content-address prefix")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError(f"{name} is not a canonical SHA-256 content address")
    return value


def _copy_immutable_fp64_host_array_into_tensor(
    destination: Any,
    source: np.ndarray,
    *,
    name: str,
) -> None:
    """Copy one immutable fp64 snapshot array without a device-sized temporary.

    Portable FD snapshots deliberately own read-only NumPy arrays. Constructing
    ``torch.tensor(source, device="cuda")`` first materializes a second complete
    route tensor and only then copies it into the already allocated accumulator.
    At the production route axis that hidden staging tensor is tens of GiB.

    CPU restoration can copy directly into the destination NumPy view. CUDA
    restoration uses one fixed-size pinned host window and synchronous H2D
    slices, so the only route-scale device allocation is the destination state
    itself. The synchronous copy is intentional: one window is reused and no
    unregistered transfer queue may retain older slices.
    """

    if torch is None:  # pragma: no cover - guarded by accumulator construction.
        raise RuntimeError("device FD snapshot restoration requires PyTorch")
    array = np.asarray(source)
    if array.dtype != np.dtype("<f8"):
        raise TypeError(f"{name} must use canonical little-endian fp64 storage")
    if tuple(destination.shape) != tuple(array.shape):
        raise ValueError(f"{name} shape does not match its restoration target")
    if destination.dtype != torch.float64:
        raise RuntimeError(f"{name} restoration target is not fp64")
    if not destination.is_contiguous():
        raise RuntimeError(f"{name} restoration target must be contiguous")
    if array.size == 0:
        return

    if destination.device.type == "cpu":
        np.copyto(destination.detach().numpy(), array, casting="no")
        return

    flat_source = array.reshape(-1)
    flat_destination = destination.reshape(-1)
    stage_elements = min(
        int(flat_source.size),
        _FD_SNAPSHOT_RESTORE_HOST_STAGE_BYTES // np.dtype(np.float64).itemsize,
    )
    stage = torch.empty(
        stage_elements,
        dtype=torch.float64,
        device="cpu",
        pin_memory=destination.device.type == "cuda",
    )
    stage_array = stage.numpy()
    for start in range(0, int(flat_source.size), stage_elements):
        stop = min(start + stage_elements, int(flat_source.size))
        count = stop - start
        np.copyto(stage_array[:count], flat_source[start:stop], casting="no")
        flat_destination[start:stop].copy_(stage[:count], non_blocking=False)


@dataclass(frozen=True, slots=True)
class RouteDecisionMomentSnapshot:
    """Compact host snapshot used by sequential partition decisions."""

    head_ids: tuple[str, ...]
    mean: np.ndarray
    contextual_variance: np.ndarray
    mean_standard_error: np.ndarray
    activity_mean: np.ndarray
    activity_mean_standard_error: np.ndarray
    sample_count: int
    context_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StratifiedRouteDecisionSnapshot:
    """Already-fused host fields required by a sequential census look."""

    head_indices: tuple[int, ...]
    activity_indices: tuple[int, ...]
    mean: np.ndarray
    mean_standard_error: np.ndarray
    activity_mean: np.ndarray
    activity_mean_standard_error: np.ndarray
    sample_count_by_stratum: Mapping[str, int]
    context_ids_by_stratum: Mapping[str, tuple[str, ...]]
    # Fused fields retained on the compute device and their look receipt.
    device_fields: Mapping[str, Any] | None = None
    v6_receipt: Mapping[str, Any] | None = None


class DeviceTripartiteMomentAccumulator:
    """O(R(H²+KH)) U/G/D accumulator retained on one accelerator."""

    _PERSISTENT_TENSOR_ATTRIBUTES = (
        "sum_rows",
        "sum_outer",
        "sum_outer_squared",
        "sum_view_activity",
        "sum_view_activity_squared",
        "sketch_sum",
        "sketch_constant",
    )

    def __init__(
        self,
        *,
        head_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str,
        loss_pullback_hash: str,
        sketch_rank: int,
        sketch_seed: int,
        device: Any,
        host_primary: bool | None = None,
        window_width: int | None = None,
        staged_windows: bool | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("device route moments require PyTorch")
        self.head_ids = tuple(map(str, head_ids))
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="device tripartite route axis",
        )
        self.destination_transport_hash = str(destination_transport_hash)
        self.loss_pullback_hash = str(loss_pullback_hash)
        self.sketch_rank = int(sketch_rank)
        self.sketch_seed = int(sketch_seed)
        self.device = torch.device(device)
        h, r = len(self.head_ids), len(self.route_ids)
        if (
            h < 3
            or r <= 0
            or self.sketch_rank < 0
            or not self.destination_transport_hash
            or not self.loss_pullback_hash
        ):
            raise ValueError("device tripartite accumulator identity is invalid")
        self.host_primary = (
            host_primary_accumulators_enabled() if host_primary is None else bool(host_primary)
        )
        self._window_width = (
            int(window_width)
            if window_width is not None
            else default_host_primary_window_width(
                head_count=h,
                sketch_rank=self.sketch_rank,
            )
        )
        if self._window_width <= 0:
            raise ValueError("host-primary window width is invalid")
        self._staged_windows = (
            self.device.type == "cuda" if staged_windows is None else bool(staged_windows)
        )
        self._window_manager: HostPrimaryWindowManager | None = None
        self._host_parked = bool(self.host_primary)
        self.total_weight = 0.0
        self.sum_weight_squared = 0.0
        self.sample_count = 0
        # The production route axis is tens of millions wide. Eager ``zeros``
        # touched every page in seven multi-GiB buffers before the first packet,
        # then the residency guard copied the still-empty inactive stratum to
        # host. Allocate nothing until the first write: that packet overwrites
        # every element exactly once, so zero-filling is pure bandwidth work.
        self.sum_rows: Any | None = None
        self.sum_outer: Any | None = None
        self.sum_outer_squared: Any | None = None
        self.sum_view_activity: Any | None = None
        self.sum_view_activity_squared: Any | None = None
        self.sketch_sum: Any | None = None
        self.sketch_constant: Any | None = None
        self._context_ids: list[str] = []
        # A pinned accumulator stays on its device; the census
        # residency guard's offload/restore become no-ops (see
        # ``pin_device_residency``).
        self._device_residency_pinned = False


    def pin_device_residency(self) -> None:
        """Keep this device-resident accumulator on its device.

        The census residency guard parks the inactive stratum on the host
        whenever one stratum exceeds 8 GiB, so every stratum switch moves the
        whole state across PCIe.  The v6 census admits both strata
        against a registered device budget first; a pinned accumulator then
        ignores ``offload_to_host``/``restore_to_device`` requests.  Values
        never change: only where the bytes live.
        """

        if self.host_primary:
            raise RuntimeError("a host-primary accumulator cannot be device pinned")
        self._device_residency_pinned = True

    @property
    def device_residency_pinned(self) -> bool:
        return bool(self._device_residency_pinned)

    @property
    def storage_allocated(self) -> bool:
        """Whether persistent route storage has been materialized."""

        states = tuple(
            getattr(self, attribute) is not None for attribute in self._PERSISTENT_TENSOR_ATTRIBUTES
        )
        if any(states) and not all(states):
            raise RuntimeError("tripartite accumulator storage is partial")
        return bool(states[0])

    def _ensure_storage(self) -> None:
        """Allocate storage whose first complete packet must overwrite it."""

        if self.storage_allocated:
            return
        h, r = len(self.head_ids), len(self.route_ids)
        self.sum_rows = self._allocate_routed_slab((h, r))
        self.sum_outer = self._allocate_routed_slab((r, h, h))
        self.sum_outer_squared = self._allocate_routed_slab((r, h, h))
        self.sum_view_activity = self._allocate_routed_slab((3, r))
        self.sum_view_activity_squared = self._allocate_routed_slab((3, r))
        self.sketch_sum = self._allocate_routed_slab((self.sketch_rank, h, r))
        self.sketch_constant = torch.empty(
            self.sketch_rank, dtype=torch.float64, device=self.device
        )
        if self.host_primary:
            self._window_manager = HostPrimaryWindowManager(
                self,
                staged=self._staged_windows,
            )

    def _allocate_routed_slab(self, shape: tuple[int, ...]) -> Any:
        if not self.host_primary:
            return torch.empty(shape, dtype=torch.float64, device=self.device)
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:  # pragma: no cover - CUDA allocator dependent.
                return torch.empty(
                    shape,
                    dtype=torch.float64,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:  # pragma: no cover - allocator fallback.
                pass
        return torch.empty(shape, dtype=torch.float64, device="cpu")

    def _release_storage(self) -> None:
        """Release persistent tensors after an explicitly consuming close."""

        if self._window_manager is not None:
            self._window_manager.drain()
        for attribute in self._PERSISTENT_TENSOR_ATTRIBUTES:
            setattr(self, attribute, None)
        self._window_manager = None

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(self._context_ids)

    def offload_to_host(self) -> None:
        """Move the persistent tensor state to CPU without changing values."""

        if self.host_primary or self._device_residency_pinned:
            return
        if self.device == torch.device("cpu"):
            return
        if not self.storage_allocated:
            self.device = torch.device("cpu")
            return
        for attribute in self._PERSISTENT_TENSOR_ATTRIBUTES:
            setattr(
                self,
                attribute,
                getattr(self, attribute).to("cpu", non_blocking=False),
            )
        self.device = torch.device("cpu")
        self._host_parked = True

    def restore_to_device(self, device: Any) -> None:
        """Move the persistent tensor state to ``device`` exactly once."""

        if self.host_primary:
            return
        device = torch.device(device)
        if self.device == device:
            return
        if self._device_residency_pinned:
            raise RuntimeError(
                "a device-pinned census accumulator cannot move to another device"
            )
        if not self.storage_allocated:
            self.device = device
            return
        for attribute in self._PERSISTENT_TENSOR_ATTRIBUTES:
            setattr(
                self,
                attribute,
                getattr(self, attribute).to(device, non_blocking=False),
            )
        self.device = device
        self._host_parked = False

    _ROUTED_SLAB_AXES = {
        "sum_rows": 1,
        "sum_outer": 0,
        "sum_outer_squared": 0,
        "sum_view_activity": 1,
        "sum_view_activity_squared": 1,
        "sketch_sum": 2,
    }

    def windowed_slabs(
        self,
        start: int,
        stop: int,
        *,
        stream: Any | None = None,
    ) -> dict[str, Any]:
        """Return routed device windows, or direct slices when unarmed."""

        start = int(start)
        stop = int(stop)
        if not 0 <= start < stop <= len(self.route_ids):
            raise ValueError("windowed slab range is invalid")
        if self._window_manager is not None:
            return self._window_manager.acquire_window(
                start,
                stop,
                stream=stream,
            )
        views: dict[str, Any] = {}
        for field, axis in self._ROUTED_SLAB_AXES.items():
            if field == "sketch_sum" and not self.sketch_rank:
                continue
            tensor = getattr(self, field)
            index = [slice(None)] * tensor.ndim
            index[axis] = slice(start, stop)
            views[field] = tensor[tuple(index)]
        return views

    def commit_window(
        self,
        start: int,
        stop: int,
        *,
        stream: Any | None = None,
    ) -> None:
        if self._window_manager is not None:
            self._window_manager.commit_window(
                int(start),
                int(stop),
                stream=stream,
            )

    def drain_windows(self) -> None:
        if self._window_manager is not None:
            self._window_manager.drain()

    def host_primary_receipts(self) -> dict[str, Any]:
        receipt: dict[str, Any] = {
            "environment": HOST_PRIMARY_ACCUMULATORS_ENVIRONMENT,
            "mode": "host_primary" if self.host_primary else "device_resident",
        }
        if self._window_manager is not None:
            receipt.update(
                {
                    "compute_device": str(self.device),
                    "window_width_default": int(self._window_width),
                    **self._window_manager.receipts(),
                }
            )
        return receipt

    def _signs(self, context_ids: Sequence[str]) -> Any:
        signs = np.empty((self.sketch_rank, len(tuple(context_ids))), dtype=np.float64)
        for probe in range(self.sketch_rank):
            for column, context_id in enumerate(context_ids):
                digest = hashlib.sha256(
                    (f"{self.sketch_seed}:tripartite:{probe}:{context_id}").encode()
                ).digest()
                signs[probe, column] = 1.0 if digest[0] & 1 else -1.0
        if self.sketch_rank:
            signs /= math.sqrt(float(self.sketch_rank))
        # Detached for uniformity with the other ingestion points. These signs
        # are built from numpy here and cannot carry a graph today, but the
        # invariant is "no tensor entering a persistent accumulator carries
        # autograd history" -- an invariant with an exception is one nobody can
        # check at a glance.
        return torch.as_tensor(signs, dtype=torch.float64, device=self.device).detach()

    def update(
        self,
        rows: Any,
        *,
        context_ids: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        ids = tuple(map(str, context_ids))
        # DETACH AT THE MEASUREMENT SINK. `torch.as_tensor` PRESERVES autograd
        # history, and the census computes attributions, so `rows` arrives with a
        # grad_fn. Without this the einsums below inherit it and `add_` writes it
        # into `sum_rows` -- a buffer held across EVERY look -- so the graph for
        # every context ever accumulated stays alive, holding its saved
        # activations. Deleting the activations frees nothing while the
        # accumulator still references them through the graph.
        #
        # That is the shape the OOM has: residency linear in PANEL, at
        # ~9 MiB/context (activation-scale: ~1.2M float64 values), varying ~2x
        # across families as graph depth varies, and host-resident when the
        # captured rows originate on CPU.
        #
        # Accumulating sufficient statistics never needs gradient history: this
        # is a measurement SINK, and no value below is differentiated. Detaching
        # is therefore free and cannot change a single number.
        values = torch.as_tensor(rows, dtype=torch.float64, device=self.device).detach()
        if (
            tuple(values.shape) != (len(ids), len(self.head_ids), len(self.route_ids))
            or not ids
            or len(set(ids)) != len(ids)
            or set(ids).intersection(self._context_ids)
            or not bool(torch.isfinite(values).all().item())
        ):
            raise ValueError("device tripartite rows/context axis is invalid")
        raw_weights = (
            torch.ones(len(ids), dtype=torch.float64, device=self.device)
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float64, device=self.device)
            .detach()
            .reshape(-1)
        )
        if (
            tuple(raw_weights.shape) != (len(ids),)
            or bool(torch.any(raw_weights < 0.0).item())
            or not bool(torch.isfinite(raw_weights).all().item())
            or float(raw_weights.sum().item()) <= 0.0
        ):
            raise ValueError("device tripartite weights are invalid")
        first_packet = self.sample_count == 0
        self._ensure_storage()
        targets = self.windowed_slabs(0, len(self.route_ids))
        weight_total = float(raw_weights.sum().item())
        self.total_weight += weight_total
        self.sum_weight_squared += float(torch.sum(raw_weights * raw_weights).item())
        self.sample_count += len(ids)
        write = "copy_" if first_packet else "add_"
        getattr(targets["sum_rows"], write)(torch.einsum("n,nhr->hr", raw_weights, values))
        getattr(targets["sum_outer"], write)(
            torch.einsum("n,nhr,njr->rhj", raw_weights, values, values)
        )
        squared = values * values
        getattr(targets["sum_outer_squared"], write)(
            torch.einsum("n,nhr,njr->rhj", raw_weights, squared, squared)
        )
        activity = torch.stack(
            (
                torch.abs(values[:, 0, :]),
                torch.sqrt(torch.sum(values[:, 1:-1, :] ** 2, dim=1)),
                # Census ordering reads this imagewise absolute loss gradient,
                # rather than the absolute value of its signed image mean.
                torch.abs(values[:, -1, :]),
            ),
            dim=1,
        )
        getattr(targets["sum_view_activity"], write)(
            torch.einsum("n,nvr->vr", raw_weights, activity)
        )
        getattr(targets["sum_view_activity_squared"], write)(
            torch.einsum("n,nvr->vr", raw_weights, activity * activity)
        )
        if self.sketch_rank:
            signs = self._signs(ids)
            root = torch.sqrt(raw_weights)
            getattr(targets["sketch_sum"], write)(
                torch.einsum("kn,n,nhr->khr", signs, root, values)
            )
            getattr(self.sketch_constant, write)(torch.einsum("kn,n->k", signs, root))
        elif first_packet:
            self.sketch_sum.zero_()
            self.sketch_constant.zero_()
        if self.host_primary:
            self.commit_window(0, len(self.route_ids))
        self._context_ids.extend(ids)

    def decision_snapshot(
        self,
        *,
        population_size: int | None = None,
        route_chunk_size: int = 1_048_576,
    ) -> RouteDecisionMomentSnapshot:
        """Transfer only route-level quantities needed by the next look.

        The route axis is tens of millions wide in the registered workflow.
        Materializing ``mean``, ``second``, ``contextual``, and both standard
        errors over that whole axis at once used several additional full-width
        CUDA buffers while the persistent accumulator was still resident.  A
        decision snapshot is route-separable, so compute and transfer the same
        expressions in bounded route chunks.  This changes neither arithmetic
        nor output layout; it only bounds the lifetime of device intermediates.
        """

        if self.sample_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot snapshot an empty device accumulator")
        if not self.storage_allocated:
            raise RuntimeError("nonempty device accumulator has no storage")
        self.drain_windows()
        chunk_size = int(route_chunk_size)
        if chunk_size <= 0:
            raise ValueError("decision snapshot route chunk size must be positive")
        if population_size is not None:
            population = int(population_size)
            if not 0 < self.sample_count <= population:
                raise ValueError("finite-population snapshot capacity is invalid")
        head_count = len(self.head_ids)
        route_count = len(self.route_ids)
        mean_host = np.empty((head_count, route_count), dtype=np.float64)
        contextual_host = np.empty_like(mean_host)
        mean_se_host = np.empty_like(mean_host)
        activity_host = np.empty((3, route_count), dtype=np.float64)
        activity_se_host = np.empty_like(activity_host)
        for start in range(0, route_count, chunk_size):
            stop = min(start + chunk_size, route_count)
            mean, contextual, mean_se, activity, activity_se = self._decision_chunk(
                start=start,
                stop=stop,
                population_size=population_size,
                head_indices=tuple(range(head_count)),
                activity_indices=(0, 1, 2),
            )
            mean_host[:, start:stop] = mean.detach().cpu().numpy()
            contextual_host[:, start:stop] = contextual.detach().cpu().numpy()
            mean_se_host[:, start:stop] = mean_se.detach().cpu().numpy()
            activity_host[:, start:stop] = activity.detach().cpu().numpy()
            activity_se_host[:, start:stop] = activity_se.detach().cpu().numpy()
        return RouteDecisionMomentSnapshot(
            head_ids=self.head_ids,
            mean=mean_host,
            contextual_variance=contextual_host,
            mean_standard_error=mean_se_host,
            activity_mean=activity_host,
            activity_mean_standard_error=activity_se_host,
            sample_count=self.sample_count,
            context_ids=self.context_ids,
        )


    # -- Exact device evaluation of the finalize expressions ---------

    def _compute_device(self) -> Any:
        """Where v6 arithmetic runs: the producer device (CPU off CUDA)."""

        return self.device

    def _v6_slab(self, attribute: str, index: tuple[Any, ...]) -> Any:
        """One routed slab slice on the compute device (host-primary: H2D)."""

        value = getattr(self, attribute)[index]
        device = self._compute_device()
        if value.device != device:
            value = value.to(device, non_blocking=False)
        return value

    def _decision_chunk(
        self,
        *,
        start: int,
        stop: int,
        population_size: int | None,
        head_indices: tuple[int, ...],
        activity_indices: tuple[int, ...],
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Derive one route slice in float64 on the compute device."""

        weight = self.total_weight
        normalized_w2 = self.sum_weight_squared / (weight * weight)
        denominator = max(1e-15, 1.0 - normalized_w2)
        fpc = 1.0
        if population_size is not None:
            population = int(population_size)
            if not 0 < self.sample_count <= population:
                raise ValueError("finite-population snapshot capacity is invalid")
            fpc = max(
                0.0,
                (population - self.sample_count) / max(population - 1, 1),
            )
        mean_all = (self._v6_slab("sum_rows", (slice(None), slice(start, stop))) / weight)
        second = (self._v6_slab("sum_outer", (slice(start, stop),)) / weight)
        contextual_all = torch.clamp(
            torch.diagonal(second, dim1=1, dim2=2).T - mean_all * mean_all,
            min=0.0,
        )
        del second
        mean_se_all = torch.sqrt(
            torch.clamp(
                (contextual_all / denominator) * normalized_w2 * float(fpc),
                min=0.0,
            )
        )
        activity_all = (self._v6_slab("sum_view_activity", (slice(None), slice(start, stop))) / weight)
        activity_second = (self._v6_slab("sum_view_activity_squared", (slice(None), slice(start, stop))) / weight)
        activity_se_all = torch.sqrt(
            (torch.clamp(
                    activity_second - activity_all * activity_all,
                    min=0.0,
                ) / denominator)
            * normalized_w2
            * float(fpc)
        )
        return (
            mean_all[list(head_indices)],
            contextual_all[list(head_indices)],
            mean_se_all[list(head_indices)],
            activity_all[list(activity_indices)],
            activity_se_all[list(activity_indices)],
        )

    def _finalize_fields_chunk_v6(self, start: int, stop: int) -> dict[str, Any]:
        """The ``_finalize_host_arrays`` fields of one route chunk.

        Same expressions, same order, float64; quotients are exact device
        divisions and the route outer product is spelled as its single
        products (``_v6_outer``).  Returns compute-device tensors.
        """

        weight = self.total_weight
        w2 = self.sum_weight_squared / (weight * weight)
        denominator = max(1e-15, 1.0 - w2)
        routes = (slice(None), slice(start, stop))
        mean = (self._v6_slab("sum_rows", routes) / weight)
        second = (self._v6_slab("sum_outer", (slice(start, stop),)) / weight)
        contextual = second - _v6_outer(mean)
        estimator = (0.5 * (contextual + contextual.transpose(1, 2)) / denominator) * w2
        del contextual
        outer_second = (self._v6_slab("sum_outer_squared", (slice(start, stop),)) / weight)
        second_se = torch.sqrt(
            (torch.clamp(outer_second - second * second, min=0.0) / denominator) * w2
        )
        del outer_second
        if self.sketch_rank:
            sketch_constant = self.sketch_constant.to(device=mean.device)
            factor = (
                self._v6_slab("sketch_sum", (slice(None), slice(None), slice(start, stop)))
                - sketch_constant[:, None, None] * mean[None, :, :]
            ) / math.sqrt(weight)
        else:
            factor = torch.empty(
                (0, len(self.head_ids), stop - start),
                dtype=torch.float64,
                device=mean.device,
            )
        activity_mean = (self._v6_slab("sum_view_activity", routes) / weight)
        activity_second = (self._v6_slab("sum_view_activity_squared", routes) / weight)
        activity_se = torch.sqrt(
            (torch.clamp(
                    activity_second - activity_mean * activity_mean,
                    min=0.0,
                ) / denominator)
            * w2
        )
        return {
            "mean": mean,
            "route_head_second_moment": second,
            "mean_estimator_covariance": estimator,
            "second_moment_standard_error": second_se,
            "covariance_factor": factor,
            "view_activity_mean": activity_mean,
            "view_activity_mean_standard_error": activity_se,
            "view_activity_second_moment": activity_second,
        }

    def _finalize_host_arrays(self) -> dict[str, np.ndarray]:
        """v6: device fields per route chunk, streamed into host arrays."""

        if self.sample_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty device accumulator")
        self.drain_windows()
        h, r, k = len(self.head_ids), len(self.route_ids), int(self.sketch_rank)
        host = _allocate_tripartite_host_fields(head_count=h, route_count=r, factor_rows=k)
        chunk = census_v6_route_chunk()
        for start in range(0, r, chunk):
            stop = min(start + chunk, r)
            fields = self._finalize_fields_chunk_v6(start, stop)
            _stream_tripartite_chunk_to_host(host, fields, start=start, stop=stop)
            del fields
        return host


    def finalize(self) -> TripartitePullbackMomentTensor:
        arrays = self._finalize_host_arrays()
        return TripartitePullbackMomentTensor(
            head_ids=self.head_ids,
            route_ids=self.route_ids,
            mean=arrays["mean"],
            route_head_second_moment=arrays["route_head_second_moment"],
            mean_estimator_covariance=arrays["mean_estimator_covariance"],
            second_moment_standard_error=arrays["second_moment_standard_error"],
            covariance_factor=arrays["covariance_factor"],
            sample_count=self.sample_count,
            context_axis_hash=content_hash(
                self.context_ids,
                prefix="tripartite-pullback-context-axis-",
            ),
            destination_transport_hash=self.destination_transport_hash,
            loss_pullback_hash=self.loss_pullback_hash,
            sketch_seed=self.sketch_seed,
            view_activity_mean=arrays["view_activity_mean"],
            view_activity_mean_standard_error=arrays["view_activity_mean_standard_error"],
            view_activity_second_moment=arrays["view_activity_second_moment"],
        )


@dataclass(frozen=True, slots=True)
class BoundedOrderedFDSourceProofSnapshot:
    """Portable continuation state for one bounded ordered source proof."""

    schema: str
    entry_count: int
    source_row_chain_hash: str
    stream_weight_chain_hash: str
    peak_host_row_bytes: int
    total_host_transfer_bytes: int
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        self.validate_integrity(allow_unsealed=True)
        expected = content_hash(
            self._identity_payload(),
            prefix=_FD_SOURCE_PROOF_SNAPSHOT_HASH_PREFIX,
        )
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("bounded FD source-proof snapshot content hash is invalid")
        object.__setattr__(self, "snapshot_hash", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "entry_count": self.entry_count,
            "source_row_chain_hash": self.source_row_chain_hash,
            "stream_weight_chain_hash": self.stream_weight_chain_hash,
            "peak_host_row_bytes": self.peak_host_row_bytes,
            "total_host_transfer_bytes": self.total_host_transfer_bytes,
        }

    def validate_integrity(self, *, allow_unsealed: bool = False) -> None:
        if self.schema != _FD_SOURCE_PROOF_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported bounded FD source-proof snapshot schema")
        entry_count = _require_exact_nonnegative_int(
            self.entry_count,
            name="bounded FD source-proof entry count",
        )
        peak_bytes = _require_exact_nonnegative_int(
            self.peak_host_row_bytes,
            name="bounded FD source-proof peak host row bytes",
        )
        total_bytes = _require_exact_nonnegative_int(
            self.total_host_transfer_bytes,
            name="bounded FD source-proof total host transfer bytes",
        )
        source_chain = _require_prefixed_sha256(
            self.source_row_chain_hash,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
            name="bounded FD source-row continuation",
        )
        weight_chain = _require_prefixed_sha256(
            self.stream_weight_chain_hash,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
            name="bounded FD weight continuation",
        )
        if peak_bytes > total_bytes:
            raise ValueError("bounded FD source-proof byte counters are inconsistent")
        if entry_count == 0:
            if peak_bytes != 0 or total_bytes != 0:
                raise ValueError("empty bounded FD source proof carries byte counters")
            if source_chain != _empty_ordered_chain(
                schema=_SOURCE_ROW_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
            ) or weight_chain != _empty_ordered_chain(
                schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
            ):
                raise ValueError("empty bounded FD source proof has a nonempty continuation")
        if not allow_unsealed:
            expected = content_hash(
                self._identity_payload(),
                prefix=_FD_SOURCE_PROOF_SNAPSHOT_HASH_PREFIX,
            )
            if self.snapshot_hash != expected:
                raise ValueError("bounded FD source-proof snapshot content hash is invalid")


@dataclass(frozen=True, slots=True)
class DeviceFrequentDirectionsTrustAccumulatorSnapshot:
    """Immutable host sufficient state for exact FD stream continuation.

    Only live sketch rows are represented.  Inactive buffer capacity is an
    implementation detail reconstructed as zeros on the explicitly selected
    destination device.
    """

    schema: str
    producer_algorithm_id: str
    producer_implementation: str
    route_ids: Sequence[str]
    ordered_route_axis_hash: str
    sketch_rank: int
    dtype_identity: str
    active_buffer_rows: np.ndarray
    filled_count: int
    mean: np.ndarray | None
    weighted_second_moment_diagonal: np.ndarray
    cumulative_shrinkage: float
    total_weight: float
    context_count: int
    context_ids: tuple[str, ...]
    bounded_source_proof: BoundedOrderedFDSourceProofSnapshot | None
    latest_source_row_hashes: tuple[str, ...]
    snapshot_hash: str = ""

    def __post_init__(self) -> None:
        axis = canonical_unique_string_axis(
            self.route_ids,
            name="device FD snapshot route axis",
        )
        object.__setattr__(self, "route_ids", axis)
        object.__setattr__(self, "context_ids", tuple(self.context_ids))
        object.__setattr__(
            self,
            "latest_source_row_hashes",
            tuple(self.latest_source_row_hashes),
        )
        object.__setattr__(
            self,
            "active_buffer_rows",
            _immutable_fd_snapshot_array(
                self.active_buffer_rows,
                name="device FD snapshot active buffer rows",
            ),
        )
        if self.mean is not None:
            object.__setattr__(
                self,
                "mean",
                _immutable_fd_snapshot_array(
                    self.mean,
                    name="device FD snapshot mean",
                ),
            )
        object.__setattr__(
            self,
            "weighted_second_moment_diagonal",
            _immutable_fd_snapshot_array(
                self.weighted_second_moment_diagonal,
                name="device FD snapshot weighted second-moment diagonal",
            ),
        )
        self.validate_integrity(allow_unsealed=True)
        expected = content_hash(
            self._identity_payload(),
            prefix=_FD_ACCUMULATOR_SNAPSHOT_HASH_PREFIX,
        )
        if self.snapshot_hash and self.snapshot_hash != expected:
            raise ValueError("device FD accumulator snapshot content hash is invalid")
        object.__setattr__(self, "snapshot_hash", expected)

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Re-run immutability and integrity validation after unpickling."""

        return (
            type(self),
            (
                self.schema,
                self.producer_algorithm_id,
                self.producer_implementation,
                self.route_ids,
                self.ordered_route_axis_hash,
                self.sketch_rank,
                self.dtype_identity,
                self.active_buffer_rows,
                self.filled_count,
                self.mean,
                self.weighted_second_moment_diagonal,
                self.cumulative_shrinkage,
                self.total_weight,
                self.context_count,
                self.context_ids,
                self.bounded_source_proof,
                self.latest_source_row_hashes,
                self.snapshot_hash,
            ),
        )

    def _identity_payload(self) -> dict[str, Any]:
        proof = self.bounded_source_proof
        return {
            "schema": self.schema,
            "producer_algorithm_id": self.producer_algorithm_id,
            "producer_implementation": self.producer_implementation,
            "route_axis_count": len(self.route_ids),
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "sketch_rank": self.sketch_rank,
            "dtype_identity": self.dtype_identity,
            "active_buffer_rows": self.active_buffer_rows,
            "filled_count": self.filled_count,
            "mean": self.mean,
            "weighted_second_moment_diagonal": self.weighted_second_moment_diagonal,
            "cumulative_shrinkage": self.cumulative_shrinkage,
            "total_weight": self.total_weight,
            "context_count": self.context_count,
            "context_ids": self.context_ids,
            "bounded_source_proof_snapshot_hash": (
                None if proof is None else proof.snapshot_hash
            ),
            "latest_source_row_hashes": self.latest_source_row_hashes,
        }

    def validate_integrity(self, *, allow_unsealed: bool = False) -> None:
        if self.schema != _FD_ACCUMULATOR_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported device FD accumulator snapshot schema")
        if not self.producer_algorithm_id or not self.producer_implementation:
            raise ValueError("device FD snapshot producer identity is invalid")
        if self.dtype_identity != _FD_SNAPSHOT_DTYPE_IDENTITY:
            raise ValueError("device FD snapshot has an unsupported dtype identity")
        axis_hash = ordered_string_axis_hash(
            self.route_ids,
            name="device FD snapshot route axis",
        )
        if self.ordered_route_axis_hash != axis_hash:
            raise ValueError("device FD snapshot ordered route axis is inconsistent")
        rank = _require_exact_nonnegative_int(
            self.sketch_rank,
            name="device FD snapshot sketch rank",
        )
        if rank <= 0 or rank > len(self.route_ids):
            raise ValueError("device FD snapshot sketch rank is invalid")
        filled = _require_exact_nonnegative_int(
            self.filled_count,
            name="device FD snapshot filled count",
        )
        if filled > 2 * rank:
            raise ValueError("device FD snapshot filled count exceeds its buffer capacity")
        active = np.asarray(self.active_buffer_rows)
        diagonal = np.asarray(self.weighted_second_moment_diagonal)
        if active.dtype != np.dtype("<f8") or active.shape != (filled, len(self.route_ids)):
            raise ValueError("device FD snapshot active buffer shape or dtype is invalid")
        if diagonal.dtype != np.dtype("<f8") or diagonal.shape != (len(self.route_ids),):
            raise ValueError("device FD snapshot second-moment diagonal is invalid")
        if not np.isfinite(active).all() or not np.isfinite(diagonal).all():
            raise ValueError("device FD snapshot arrays contain non-finite values")
        if np.any(diagonal < 0.0):
            raise ValueError("device FD snapshot second-moment diagonal is negative")
        shrinkage = float(self.cumulative_shrinkage)
        weight = float(self.total_weight)
        if (
            type(self.cumulative_shrinkage) is not float
            or type(self.total_weight) is not float
            or not math.isfinite(shrinkage)
            or shrinkage < 0.0
            or not math.isfinite(weight)
            or weight < 0.0
        ):
            raise ValueError("device FD snapshot scalar moments are invalid")
        context_count = _require_exact_nonnegative_int(
            self.context_count,
            name="device FD snapshot effective context count",
        )
        context_ids = self.context_ids
        if (
            any(type(identifier) is not str for identifier in context_ids)
            or len(set(context_ids)) != len(context_ids)
            or context_count > len(context_ids)
        ):
            raise ValueError("device FD snapshot ordered context ledger is invalid")
        mean = self.mean
        if weight == 0.0:
            if (
                mean is not None
                or context_count != 0
                or context_ids
                or filled != 0
                or shrinkage != 0.0
                or np.any(diagonal != 0.0)
            ):
                raise ValueError("empty device FD snapshot carries nonempty moment state")
        else:
            if context_count <= 0 or not context_ids or mean is None:
                raise ValueError("nonempty device FD snapshot lacks its stream state")
            mean_array = np.asarray(mean)
            if (
                mean_array.dtype != np.dtype("<f8")
                or mean_array.ndim != 2
                or mean_array.shape[0] <= 0
                or mean_array.shape[1] != len(self.route_ids)
                or not np.isfinite(mean_array).all()
            ):
                raise ValueError("device FD snapshot mean is invalid")
        proof = self.bounded_source_proof
        latest = self.latest_source_row_hashes
        if proof is None:
            if latest:
                raise ValueError("device FD snapshot has source hashes without a proof stream")
        else:
            if type(proof) is not BoundedOrderedFDSourceProofSnapshot:
                raise TypeError("device FD snapshot source proof has the wrong type")
            proof.validate_integrity()
            if proof.entry_count != len(context_ids):
                raise ValueError("device FD snapshot proof and context ledgers disagree")
            if proof.entry_count == 0 and latest:
                raise ValueError("empty device FD snapshot has latest source-row hashes")
            if proof.entry_count > 0 and not latest:
                raise ValueError("nonempty device FD proof lacks latest source-row hashes")
            if len(latest) > proof.entry_count:
                raise ValueError("device FD snapshot latest source-row ledger is too long")
            for row_hash in latest:
                _require_prefixed_sha256(
                    row_hash,
                    prefix=GLOBAL_TRUST_SOURCE_ROW_HASH_PREFIX,
                    name="device FD snapshot latest source-row hash",
                )
            if mean is not None:
                expected_row_bytes = int(np.asarray(mean).nbytes)
                if (
                    proof.peak_host_row_bytes != expected_row_bytes
                    or proof.total_host_transfer_bytes
                    != proof.entry_count * expected_row_bytes
                ):
                    raise ValueError("device FD snapshot proof byte ledger is inconsistent")
        if not allow_unsealed:
            expected = content_hash(
                self._identity_payload(),
                prefix=_FD_ACCUMULATOR_SNAPSHOT_HASH_PREFIX,
            )
            if self.snapshot_hash != expected:
                raise ValueError("device FD accumulator snapshot content hash is invalid")


class BoundedOrderedFDSourceProof:
    """O(1)-state exact ordered row/weight proof for a streamed FD source.

    The caller presents one response-by-route host row at a time.  The proof
    immediately hashes that row into the canonical ordered chains and retains
    no row payload or per-context hash list.  Therefore peak proof-side host
    payload is one row, independent of context count.  This is an integrity
    construction, not a decision certificate.
    """

    __slots__ = (
        "_entry_count",
        "_source_row_chain_hash",
        "_stream_weight_chain_hash",
        "peak_host_row_bytes",
        "total_host_transfer_bytes",
    )

    def __init__(self) -> None:
        self._entry_count = 0
        self._source_row_chain_hash = _empty_ordered_chain(
            schema=_SOURCE_ROW_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
        )
        self._stream_weight_chain_hash = _empty_ordered_chain(
            schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
        )
        self.peak_host_row_bytes = 0
        self.total_host_transfer_bytes = 0

    @property
    def entry_count(self) -> int:
        return int(self._entry_count)

    def snapshot(self) -> BoundedOrderedFDSourceProofSnapshot:
        """Return immutable continuation state without retaining source rows."""

        return BoundedOrderedFDSourceProofSnapshot(
            schema=_FD_SOURCE_PROOF_SNAPSHOT_SCHEMA,
            entry_count=int(self._entry_count),
            source_row_chain_hash=str(self._source_row_chain_hash),
            stream_weight_chain_hash=str(self._stream_weight_chain_hash),
            peak_host_row_bytes=int(self.peak_host_row_bytes),
            total_host_transfer_bytes=int(self.total_host_transfer_bytes),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BoundedOrderedFDSourceProofSnapshot,
    ) -> "BoundedOrderedFDSourceProof":
        """Construct a fresh proof stream from a verified continuation."""

        if type(snapshot) is not BoundedOrderedFDSourceProofSnapshot:
            raise TypeError("bounded FD source-proof restoration requires its snapshot type")
        snapshot.validate_integrity()
        restored = cls()
        restored._entry_count = int(snapshot.entry_count)
        restored._source_row_chain_hash = str(snapshot.source_row_chain_hash)
        restored._stream_weight_chain_hash = str(snapshot.stream_weight_chain_hash)
        restored.peak_host_row_bytes = int(snapshot.peak_host_row_bytes)
        restored.total_host_transfer_bytes = int(snapshot.total_host_transfer_bytes)
        return restored

    def update(self, row: np.ndarray, *, weight: float) -> str:
        value = np.asarray(row, dtype=np.float64)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError("bounded FD source proof requires one finite response-by-route row")
        scalar = float(weight)
        if not math.isfinite(scalar) or scalar < 0.0:
            raise ValueError("bounded FD source proof requires a finite nonnegative weight")
        contiguous = np.ascontiguousarray(value, dtype=np.float64)
        row_bytes = int(contiguous.nbytes)
        self.peak_host_row_bytes = max(self.peak_host_row_bytes, row_bytes)
        self.total_host_transfer_bytes += row_bytes
        index = self._entry_count
        row_hash = global_trust_source_row_hash(contiguous)
        self._source_row_chain_hash = _advance_ordered_chain(
            self._source_row_chain_hash,
            index=index,
            value=row_hash,
            schema=_SOURCE_ROW_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
        )
        self._stream_weight_chain_hash = _advance_ordered_chain(
            self._stream_weight_chain_hash,
            index=index,
            value=scalar,
            schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
        )
        self._entry_count += 1
        return row_hash

    @property
    def ordered_source_rows_hash(self) -> str:
        return _finalize_ordered_chain(
            self._source_row_chain_hash,
            entry_count=self._entry_count,
            schema=_SOURCE_ROW_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
        )

    @property
    def stream_weight_hash(self) -> str:
        return _finalize_ordered_chain(
            self._stream_weight_chain_hash,
            entry_count=self._entry_count,
            schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
        )


class DeviceFrequentDirectionsTrustGeometryDeviceState:
    """One-shot device-local output of a consuming FD finalization.

    This is an in-process tensor custody joint, not a portable geometry state
    and not a row-level derivation receipt.  In particular, it deliberately
    carries no route-tensor content hashes: the issuing accumulator binds the
    ordered route axis and transfers its tensors without a route-scale D2H.
    """

    __slots__ = (
        "route_axis_hash",
        "route_count",
        "sketch_rank",
        "retained_rank",
        "producer_algorithm_id",
        "producer_implementation",
        "device_identity",
        "residual_spectral_bound",
        "total_weight",
        "context_count",
        "context_ids",
        "_mean_component",
        "_centered_factor",
        "_residual_diagonal",
        "_consumed",
    )

    def __init__(
        self,
        *,
        route_axis_hash: str,
        route_count: int,
        sketch_rank: int,
        producer_algorithm_id: str,
        producer_implementation: str,
        mean_component: Any,
        centered_factor: Any,
        residual_diagonal: Any,
        residual_spectral_bound: float,
        total_weight: float,
        context_count: int,
        context_ids: Sequence[str],
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _DEVICE_FD_GEOMETRY_ISSUER:
            raise PermissionError("device FD geometry states are accumulator-issued only")
        if torch is None or any(
            not torch.is_tensor(value)
            for value in (mean_component, centered_factor, residual_diagonal)
        ):
            raise TypeError("device FD geometry requires three torch tensors")
        route_count = int(route_count)
        sketch_rank = int(sketch_rank)
        if route_count <= 0 or sketch_rank <= 0:
            raise ValueError("device FD geometry axis and rank must be positive")
        tensors = (mean_component, centered_factor, residual_diagonal)
        devices = {str(value.device) for value in tensors}
        if (
            len(devices) != 1
            or any(value.dtype != torch.float64 for value in tensors)
            or any(value.requires_grad for value in tensors)
            or any(not value.is_contiguous() for value in tensors)
            or mean_component.ndim != 2
            or centered_factor.ndim != 2
            or residual_diagonal.ndim != 1
            or int(mean_component.shape[1]) != route_count
            or int(centered_factor.shape[1]) != route_count
            or tuple(residual_diagonal.shape) != (route_count,)
            or int(centered_factor.shape[0]) > sketch_rank
        ):
            raise ValueError("device FD geometry tensor identity is invalid")
        for name, value in (
            ("device FD mean component", mean_component),
            ("device FD centered factor", centered_factor),
            ("device FD residual diagonal", residual_diagonal),
        ):
            _check_torch_all_finite(name, value)
        if bool(torch.any(residual_diagonal < 0.0).item()):
            raise ValueError("device FD residual diagonal must be nonnegative")
        residual = float(residual_spectral_bound)
        weight = float(total_weight)
        contexts = int(context_count)
        ids = tuple(map(str, context_ids))
        if (
            not route_axis_hash
            or not producer_algorithm_id
            or not producer_implementation
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(weight)
            or weight <= 0.0
            or contexts <= 0
            or len(ids) != contexts
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("device FD geometry scalar identity is invalid")
        self.route_axis_hash = str(route_axis_hash)
        self.route_count = route_count
        self.sketch_rank = sketch_rank
        self.retained_rank = int(centered_factor.shape[0])
        self.producer_algorithm_id = str(producer_algorithm_id)
        self.producer_implementation = str(producer_implementation)
        self.device_identity = devices.pop()
        self.residual_spectral_bound = residual
        self.total_weight = weight
        self.context_count = contexts
        self.context_ids = ids
        self._mean_component = mean_component
        self._centered_factor = centered_factor
        self._residual_diagonal = residual_diagonal
        self._consumed = False

    def take_tensors_once(self) -> tuple[Any, Any, Any]:
        """Transfer all route tensors exactly once to the stratified combiner."""

        if self._consumed:
            raise RuntimeError("device FD geometry tensors were already consumed")
        self._consumed = True
        tensors = (
            self._mean_component,
            self._centered_factor,
            self._residual_diagonal,
        )
        self._mean_component = None
        self._centered_factor = None
        self._residual_diagonal = None
        return tensors

    def __reduce__(self) -> NoReturn:
        raise TypeError("device FD geometry states are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("device FD geometry states are nonserializable")


class DeviceFrequentDirectionsTrustAccumulator:
    """Accelerator-resident weighted Welford + Frequent-Directions sketch.

    Only the retained mean/factor, exact residual diagonal, and scalar
    residual envelope are retained after finalization.  The implementation is
    algebraically identical to the canonical NumPy accumulator.  Normal
    operation avoids transferring source rows merely to update the trust
    sketch.  The opt-in modern-v1 source-proof mode transfers and hashes one
    response-by-route row at a time, retaining no context-by-response-by-route
    host tensor and no per-context row payload.  Hardware-specific memory and
    throughput promotion still requires the separate H200 evidence gate.
    """

    _SNAPSHOT_PRODUCER_ALGORITHM_ID = GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID
    _SNAPSHOT_PRODUCER_IMPLEMENTATION = "torch_cuda"

    def __init__(
        self,
        *,
        route_ids: Sequence[str],
        sketch_rank: int,
        device: Any,
        emit_derivation_receipt: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("device trust geometry requires PyTorch")
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="device Frequent-Directions route axis",
        )
        self.sketch_rank = min(int(sketch_rank), len(self.route_ids))
        self.device = torch.device(device)
        if type(emit_derivation_receipt) is not bool:
            raise TypeError("emit_derivation_receipt must be bool")
        if emit_derivation_receipt and self.device.type != "cuda":
            raise ValueError("CUDA derivation receipts require an actual CUDA device")
        self.emit_derivation_receipt = emit_derivation_receipt
        if not self.route_ids or self.sketch_rank <= 0:
            raise ValueError("device trust accumulator identity is invalid")
        self._buffer = torch.zeros(
            (2 * self.sketch_rank, len(self.route_ids)),
            dtype=torch.float64,
            device=self.device,
        )
        self._filled = 0
        self._total_shrinkage = 0.0
        self._mean: Any | None = None
        self._weighted_second_moment_diagonal = torch.zeros(
            len(self.route_ids),
            dtype=torch.float64,
            device=self.device,
        )
        self.total_weight = 0.0
        self.context_count = 0
        self._context_ids: list[str] = []
        self._producer_algorithm_id = GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID
        self._producer_implementation = "torch_cuda"
        self._bounded_source_proof = (
            BoundedOrderedFDSourceProof() if self.emit_derivation_receipt else None
        )
        self._last_update_source_row_hashes: tuple[str, ...] = ()
        self._consumed = False

    def _assert_not_consumed(self) -> None:
        if self._consumed:
            raise RuntimeError("consumed device FD accumulator cannot be reused")

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(self._context_ids)

    @property
    def last_update_source_row_hashes(self) -> tuple[str, ...]:
        """Hashes already paid for by the latest bounded proof transfer."""

        return self._last_update_source_row_hashes

    @property
    def source_proof_host_transfer_bytes(self) -> int:
        proof = self._bounded_source_proof
        return 0 if proof is None else int(proof.total_host_transfer_bytes)

    def snapshot(self) -> DeviceFrequentDirectionsTrustAccumulatorSnapshot:
        """Return an immutable, device-independent exact continuation state."""

        self._assert_not_consumed()
        active_rows = self._buffer[: self._filled].detach().cpu().numpy()
        mean = None if self._mean is None else self._mean.detach().cpu().numpy()
        diagonal = self._weighted_second_moment_diagonal.detach().cpu().numpy()
        proof = self._bounded_source_proof
        return DeviceFrequentDirectionsTrustAccumulatorSnapshot(
            schema=_FD_ACCUMULATOR_SNAPSHOT_SCHEMA,
            producer_algorithm_id=str(self._producer_algorithm_id),
            producer_implementation=str(self._producer_implementation),
            route_ids=self.route_ids,
            ordered_route_axis_hash=ordered_string_axis_hash(
                self.route_ids,
                name="device Frequent-Directions route axis",
            ),
            sketch_rank=int(self.sketch_rank),
            dtype_identity=_FD_SNAPSHOT_DTYPE_IDENTITY,
            active_buffer_rows=active_rows,
            filled_count=int(self._filled),
            mean=mean,
            weighted_second_moment_diagonal=diagonal,
            cumulative_shrinkage=float(self._total_shrinkage),
            total_weight=float(self.total_weight),
            context_count=int(self.context_count),
            context_ids=self.context_ids,
            bounded_source_proof=None if proof is None else proof.snapshot(),
            latest_source_row_hashes=self.last_update_source_row_hashes,
        )

    @classmethod
    def _validate_snapshot_runtime_identity(
        cls,
        snapshot: DeviceFrequentDirectionsTrustAccumulatorSnapshot,
        *,
        route_ids: Sequence[str],
        sketch_rank: int,
    ) -> tuple[str, ...]:
        """Validate a portable snapshot against one explicit runtime identity."""

        if type(snapshot) is not DeviceFrequentDirectionsTrustAccumulatorSnapshot:
            raise TypeError("device FD restoration requires its exact snapshot type")
        snapshot.validate_integrity()
        if (
            snapshot.producer_algorithm_id != cls._SNAPSHOT_PRODUCER_ALGORITHM_ID
            or snapshot.producer_implementation
            != cls._SNAPSHOT_PRODUCER_IMPLEMENTATION
        ):
            raise ValueError(
                "device FD snapshot producer identity does not match the restorer"
            )
        axis = canonical_unique_string_axis(
            route_ids,
            name="device FD restoration route axis",
        )
        axis_hash = ordered_string_axis_hash(
            axis,
            name="device FD restoration route axis",
        )
        if (
            len(axis) != len(snapshot.route_ids)
            or axis_hash != snapshot.ordered_route_axis_hash
        ):
            raise ValueError("device FD snapshot route axis does not match the runtime")
        if type(sketch_rank) is not int or sketch_rank != snapshot.sketch_rank:
            raise ValueError("device FD snapshot sketch rank does not match the runtime")
        return axis

    def _restore_validated_snapshot_into_empty(
        self,
        snapshot: DeviceFrequentDirectionsTrustAccumulatorSnapshot,
    ) -> None:
        """Load a validated snapshot into this one fresh, already allocated state."""

        proof = self._bounded_source_proof
        if (
            self._filled != 0
            or self._total_shrinkage != 0.0
            or self._mean is not None
            or self.total_weight != 0.0
            or self.context_count != 0
            or self._context_ids
            or self._last_update_source_row_hashes
            or (proof is not None and proof.entry_count != 0)
        ):
            raise RuntimeError(
                "device FD snapshots may only load into a fresh empty accumulator"
            )
        if (
            self._producer_algorithm_id != snapshot.producer_algorithm_id
            or self._producer_implementation != snapshot.producer_implementation
        ):
            raise RuntimeError(
                "fresh device FD accumulator has an inconsistent producer identity"
            )
        if self._buffer.dtype != torch.float64:
            raise RuntimeError("fresh device FD accumulator has the wrong storage dtype")
        proof_snapshot = snapshot.bounded_source_proof
        if (proof is not None) is not (proof_snapshot is not None):
            raise ValueError(
                "device FD snapshot proof policy does not match the restoration target"
            )

        if snapshot.filled_count:
            _copy_immutable_fp64_host_array_into_tensor(
                self._buffer[: snapshot.filled_count],
                snapshot.active_buffer_rows,
                name="device FD active snapshot rows",
            )
        restored_mean = None
        if snapshot.mean is not None:
            restored_mean = torch.empty(
                tuple(int(axis) for axis in snapshot.mean.shape),
                dtype=torch.float64,
                device=self.device,
            )
            _copy_immutable_fp64_host_array_into_tensor(
                restored_mean,
                snapshot.mean,
                name="device FD snapshot mean",
            )
        _copy_immutable_fp64_host_array_into_tensor(
            self._weighted_second_moment_diagonal,
            snapshot.weighted_second_moment_diagonal,
            name="device FD snapshot weighted diagonal",
        )

        self._filled = int(snapshot.filled_count)
        self._total_shrinkage = float(snapshot.cumulative_shrinkage)
        self._mean = restored_mean
        self.total_weight = float(snapshot.total_weight)
        self.context_count = int(snapshot.context_count)
        self._context_ids = list(snapshot.context_ids)
        self._bounded_source_proof = (
            None
            if proof_snapshot is None
            else BoundedOrderedFDSourceProof.from_snapshot(proof_snapshot)
        )
        self._last_update_source_row_hashes = tuple(
            snapshot.latest_source_row_hashes
        )

    def restore_snapshot_into_empty(
        self,
        snapshot: DeviceFrequentDirectionsTrustAccumulatorSnapshot,
    ) -> None:
        """Restore exactly into this fresh private state without replacing it.

        This mutating entry point is intentionally fresh-state-only. It exists
        for stratified transaction restoration, where constructing replacement
        child accumulators would transiently duplicate the complete route-scale
        FD state. A live or committed accumulator cannot be overwritten.
        """

        self._assert_not_consumed()
        axis = type(self)._validate_snapshot_runtime_identity(
            snapshot,
            route_ids=self.route_ids,
            sketch_rank=self.sketch_rank,
        )
        if axis != self.route_ids:
            raise ValueError("device FD snapshot route axis does not match the target")
        self._restore_validated_snapshot_into_empty(snapshot)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: DeviceFrequentDirectionsTrustAccumulatorSnapshot,
        *,
        route_ids: Sequence[str],
        sketch_rank: int,
        device: Any,
    ) -> "DeviceFrequentDirectionsTrustAccumulator":
        """Restore exact stream state against an explicit runtime identity.

        The caller must restate the route axis and rank from the current
        campaign configuration.  The checkpoint is evidence for state, not
        authority to select a different scientific identity.
        """

        axis = cls._validate_snapshot_runtime_identity(
            snapshot,
            route_ids=route_ids,
            sketch_rank=sketch_rank,
        )
        proof_snapshot = snapshot.bounded_source_proof
        restored = cls(
            route_ids=axis,
            sketch_rank=sketch_rank,
            device=device,
            emit_derivation_receipt=proof_snapshot is not None,
        )
        restored._restore_validated_snapshot_into_empty(snapshot)
        return restored

    def _compress(self, *, force: bool = False) -> None:
        if self._filled <= self.sketch_rank and not force:
            return
        if self._filled == 0:
            return
        rows = self._buffer[: self._filled]
        _check_torch_all_finite("device trust FD compression buffer", rows)
        if int(rows.numel()) > _DIRECT_FD_SVD_MAX_ELEMENTS:
            raise RuntimeError(
                "direct CUDA FD compression refuses the wide SVD surface "
                f"{tuple(int(axis) for axis in rows.shape)}; use "
                "DeviceGramFrequentDirectionsTrustAccumulator for route-wide "
                "production sketches"
            )
        _left, singular, right = torch.linalg.svd(rows, full_matrices=False)
        _check_torch_all_finite("device trust FD singular values", singular)
        _check_torch_all_finite("device trust FD right singular vectors", right)
        retain = min(self.sketch_rank, int(singular.numel()))
        delta = (
            float((singular[self.sketch_rank] ** 2).item())
            if int(singular.numel()) > self.sketch_rank
            else 0.0
        )
        shrunk = torch.sqrt(torch.clamp(singular[:retain] ** 2 - delta, min=0.0))
        _check_torch_all_finite("device trust FD shrunk singular values", shrunk)
        self._buffer.zero_()
        self._buffer[:retain].copy_(shrunk[:, None] * right[:retain])
        self._filled = int(torch.count_nonzero(shrunk > 0.0).item())
        self._total_shrinkage += delta

    def _insert(self, rows: Any) -> None:
        """Fill the accelerator buffer in slices at unchanged FD boundaries."""

        values = torch.as_tensor(rows, dtype=torch.float64, device=self.device).detach()
        if values.ndim != 2 or int(values.shape[1]) != int(self._buffer.shape[1]):
            raise ValueError("device trust FD insertion requires a [row,route] matrix")
        _check_torch_all_finite("device trust FD insertion rows", values)
        offset = 0
        row_count = int(values.shape[0])
        capacity = int(self._buffer.shape[0])
        while offset < row_count:
            if self._filled >= capacity:
                self._compress()
            room = capacity - self._filled
            take = min(room, row_count - offset)
            self._buffer[self._filled : self._filled + take].copy_(
                values[offset : offset + take]
            )
            self._filled += take
            offset += take

    def update(
        self,
        rows: Any,
        *,
        context_ids: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self._assert_not_consumed()
        ids = tuple(map(str, context_ids))
        # Detached for the same reason as the tripartite sink above: this feeds a
        # PERSISTENT sketch, so any grad_fn on `rows` would anchor the graph --
        # and its saved activations -- for the life of the accumulator.
        values = torch.as_tensor(rows, dtype=torch.float64, device=self.device).detach()
        if (
            values.ndim != 3
            or int(values.shape[0]) != len(ids)
            or int(values.shape[2]) != len(self.route_ids)
            or not ids
            or len(set(ids)) != len(ids)
            or set(ids).intersection(self._context_ids)
            or not bool(torch.isfinite(values).all().item())
        ):
            raise ValueError(
                "device trust rows require unique contexts and [context,response,route] axes"
            )
        raw_weights = (
            torch.ones(len(ids), dtype=torch.float64, device=self.device)
            if weights is None
            else torch.as_tensor(weights, dtype=torch.float64, device=self.device)
            .detach()
            .reshape(-1)
        )
        if (
            tuple(raw_weights.shape) != (len(ids),)
            or bool(torch.any(raw_weights < 0.0).item())
            or not bool(torch.isfinite(raw_weights).all().item())
            or float(raw_weights.sum().item()) <= 0.0
        ):
            raise ValueError("device trust weights are invalid")
        proof_row_hashes: list[str] = []
        for observation, raw_weight in zip(values, raw_weights, strict=True):
            weight = float(raw_weight.item())
            if self._bounded_source_proof is not None:
                # Transfer and hash exactly one response-by-route row.  No
                # context×response×route host tensor or per-context hash list
                # is retained.
                proof_row_hashes.append(
                    self._bounded_source_proof.update(
                        observation.detach().cpu().numpy(),
                        weight=weight,
                    )
                )
            if weight <= 0.0:
                continue
            self._weighted_second_moment_diagonal.add_(
                torch.sum(observation * observation, dim=0),
                alpha=weight,
            )
            if self._mean is None:
                self._mean = observation.clone()
                self.total_weight = weight
            else:
                delta = observation - self._mean
                updated_weight = self.total_weight + weight
                correction = math.sqrt(self.total_weight * weight / updated_weight) * delta
                self._insert(correction)
                self._mean.add_(delta, alpha=weight / updated_weight)
                self.total_weight = updated_weight
            self.context_count += 1
        self._context_ids.extend(ids)
        self._last_update_source_row_hashes = tuple(proof_row_hashes)

    def finalize_non_mutating(
        self,
        *,
        spec: GlobalTrustGeometrySpec,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_measurement_hashes: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> GlobalTrustGeometryState:
        """Finalize an exact device-local clone, preserving this live stream.

        Portable snapshots remain the crash/restart ABI.  An in-process
        freeze evaluation must not pay a route-scale device-to-host snapshot,
        content-hash scan, and host-to-device restore merely to protect the
        live accumulator from ``finalize``'s forced compression.  Clone the
        sufficient tensors directly on their current device and copy only the
        small ordered-proof continuation through its typed snapshot.
        """

        self._assert_not_consumed()
        proof = self._bounded_source_proof
        clone = type(self)(
            route_ids=self.route_ids,
            sketch_rank=self.sketch_rank,
            device=self.device,
            emit_derivation_receipt=proof is not None,
        )
        if (
            clone._producer_algorithm_id != self._producer_algorithm_id
            or clone._producer_implementation != self._producer_implementation
        ):
            raise RuntimeError(
                "device-local FD finalization clone changed producer identity"
            )
        if self._filled:
            clone._buffer[: self._filled].copy_(self._buffer[: self._filled])
        clone._filled = int(self._filled)
        clone._total_shrinkage = float(self._total_shrinkage)
        clone._mean = None if self._mean is None else self._mean.clone()
        clone._weighted_second_moment_diagonal.copy_(
            self._weighted_second_moment_diagonal
        )
        clone.total_weight = float(self.total_weight)
        clone.context_count = int(self.context_count)
        clone._context_ids = list(self._context_ids)
        clone._bounded_source_proof = (
            None
            if proof is None
            else BoundedOrderedFDSourceProof.from_snapshot(proof.snapshot())
        )
        clone._last_update_source_row_hashes = tuple(
            self._last_update_source_row_hashes
        )
        return clone.finalize(
            spec=spec,
            measured_mask_hash=measured_mask_hash,
            calibration_state_id=calibration_state_id,
            source_measurement_hashes=source_measurement_hashes,
            metadata=metadata,
        )

    def _finalized_residual_diagonal(self, factor: Any) -> Any:
        # With no FD shrinkage the retained factor contains all centered
        # energy. Its coordinate remainder is mathematically zero; subtracting
        # independently accumulated fp64 diagonals would create cancellation
        # residue inconsistent with the zero spectral residual.
        if self._total_shrinkage == 0.0:
            return torch.zeros_like(self._weighted_second_moment_diagonal)
        centered_diagonal = torch.clamp(
            self._weighted_second_moment_diagonal / self.total_weight
            - torch.sum(self._mean * self._mean, dim=0),
            min=0.0,
        )
        return torch.clamp(
            centered_diagonal - torch.sum(factor * factor, dim=0),
            min=0.0,
        )

    def finalize(
        self,
        *,
        spec: GlobalTrustGeometrySpec,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_measurement_hashes: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> GlobalTrustGeometryState:
        self._assert_not_consumed()
        if self._mean is None or self.context_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty device trust geometry")
        self._compress(force=True)
        factor = self._buffer[: self._filled] / math.sqrt(self.total_weight)
        residual_diagonal = self._finalized_residual_diagonal(factor)
        mean_host = self._mean.detach().cpu().numpy()
        factor_host = factor.detach().cpu().numpy()
        residual_diagonal_host = residual_diagonal.detach().cpu().numpy()
        measurement_hashes = tuple(map(str, source_measurement_hashes))
        context_axis_hash = content_hash(self.context_ids, prefix="irg-trust-context-axis-")
        caller_metadata = dict(metadata or {})
        protected_metadata = {
            "sketch_algorithm",
            "canonical_numpy_algorithm_equivalent",
            "cross_module_terms_retained",
            "spectral_residual_is_pathwise_upper_envelope",
            "host_context_route_rows_transferred",
            "residual_diagonal_is_exact_streamed_fd_remainder",
            "derivation_proof_mode",
            "derivation_proof_full_tensor_host_snapshot",
            "derivation_proof_streaming_window_contexts",
            "derivation_proof_peak_host_row_bytes",
            "derivation_proof_total_host_transfer_bytes",
            "aggregate_route_summary_host_transfer_required",
            "aggregate_route_summary_host_transfer_bytes",
            "aggregate_route_summary_host_transfer_role",
        }
        overlap = protected_metadata.intersection(caller_metadata)
        if overlap:
            raise ValueError(
                "device trust metadata cannot override protected producer fields: "
                + ", ".join(sorted(overlap))
            )
        derivation_receipt = None
        if self.emit_derivation_receipt:
            proof = self._bounded_source_proof
            if proof is None or proof.entry_count != len(self.context_ids):
                raise RuntimeError("CUDA FD bounded source-proof stream is incomplete")
            derivation_receipt = GlobalTrustGeometryDerivationReceipt(
                producer_algorithm_id=self._producer_algorithm_id,
                producer_implementation=self._producer_implementation,
                trust_spec_hash=spec.content_hash,
                ordered_route_axis_hash=ordered_string_axis_hash(
                    self.route_ids,
                    name="device Frequent-Directions route axis",
                ),
                context_axis_hash=context_axis_hash,
                stream_entry_count=proof.entry_count,
                effective_context_count=self.context_count,
                ordered_source_rows_hash=proof.ordered_source_rows_hash,
                stream_weight_hash=proof.stream_weight_hash,
                mean_component_hash=content_hash(
                    mean_host,
                    prefix="irg-global-trust-derived-mean-",
                ),
                centered_factor_hash=content_hash(
                    factor_host,
                    prefix="irg-global-trust-derived-centered-factor-",
                ),
                exact_residual_diagonal_hash=content_hash(
                    residual_diagonal_host,
                    prefix="irg-global-trust-derived-residual-diagonal-",
                ),
                total_weight=self.total_weight,
                cumulative_fd_shrink=self._total_shrinkage,
                normalized_stream_squared_frobenius=float(
                    torch.sum(self._weighted_second_moment_diagonal).item() / self.total_weight
                ),
                sketch_rank=self.sketch_rank,
                measured_mask_hash=str(measured_mask_hash),
                calibration_state_id=str(calibration_state_id),
                source_measurement_hashes=measurement_hashes,
                _accumulator_token=_ACCUMULATOR_DERIVATION_TOKEN,
            )
        return GlobalTrustGeometryState(
            spec=spec,
            route_ids=self.route_ids,
            mean_component=mean_host,
            centered_factor=factor_host,
            residual_spectral_bound=(self._total_shrinkage / self.total_weight),
            residual_diagonal_bound=residual_diagonal_host,
            total_weight=self.total_weight,
            context_count=self.context_count,
            context_axis_hash=context_axis_hash,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            source_measurement_hashes=measurement_hashes,
            derivation_receipt=derivation_receipt,
            metadata={
                **caller_metadata,
                "sketch_algorithm": self._producer_algorithm_id,
                "canonical_numpy_algorithm_equivalent": True,
                "cross_module_terms_retained": True,
                "spectral_residual_is_pathwise_upper_envelope": True,
                "host_context_route_rows_transferred": bool(self.emit_derivation_receipt),
                "residual_diagonal_is_exact_streamed_fd_remainder": True,
                "derivation_proof_mode": (
                    "bounded_ordered_row_stream_exact_v1"
                    if self.emit_derivation_receipt
                    else "absent"
                ),
                "derivation_proof_full_tensor_host_snapshot": False,
                "derivation_proof_streaming_window_contexts": (
                    1 if self.emit_derivation_receipt else 0
                ),
                "derivation_proof_peak_host_row_bytes": (
                    0
                    if self._bounded_source_proof is None
                    else self._bounded_source_proof.peak_host_row_bytes
                ),
                "derivation_proof_total_host_transfer_bytes": (
                    0
                    if self._bounded_source_proof is None
                    else self._bounded_source_proof.total_host_transfer_bytes
                ),
                "aggregate_route_summary_host_transfer_required": True,
                "aggregate_route_summary_host_transfer_bytes": int(
                    mean_host.nbytes
                    + factor_host.nbytes
                    + residual_diagonal_host.nbytes
                ),
                "aggregate_route_summary_host_transfer_role": (
                    "final_fd_geometry_state"
                ),
            },
        )

    def finalize_device_consuming(
        self,
    ) -> DeviceFrequentDirectionsTrustGeometryDeviceState:
        """Consume this FD stream into three device-resident sufficient tensors.

        Unlike :meth:`finalize`, this joint performs no route-scale D2H and no
        route-tensor cryptographic hash.  It is intended only for immediate
        in-process law-of-total-covariance assembly under the already-bound
        route/source identity.  Portable replay continues to use ``snapshot``.
        """

        self._assert_not_consumed()
        if self._mean is None or self.context_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty device trust geometry")
        self._compress(force=True)
        factor = (
            self._buffer[: self._filled] / math.sqrt(self.total_weight)
        ).detach().contiguous()
        residual_diagonal = self._finalized_residual_diagonal(factor).detach().contiguous()
        mean_component = self._mean.detach().contiguous()
        state = DeviceFrequentDirectionsTrustGeometryDeviceState(
            route_axis_hash=ordered_string_axis_hash(
                self.route_ids,
                name="device Frequent-Directions route axis",
            ),
            route_count=len(self.route_ids),
            sketch_rank=self.sketch_rank,
            producer_algorithm_id=self._producer_algorithm_id,
            producer_implementation=self._producer_implementation,
            mean_component=mean_component,
            centered_factor=factor,
            residual_diagonal=residual_diagonal,
            residual_spectral_bound=(self._total_shrinkage / self.total_weight),
            total_weight=self.total_weight,
            context_count=self.context_count,
            context_ids=self.context_ids,
            _issuance_token=_DEVICE_FD_GEOMETRY_ISSUER,
        )
        # Transfer complete ownership only after all numerical validation has
        # succeeded.  The accumulator then has no route tensor left to reuse or
        # snapshot accidentally.
        self._consumed = True
        self._buffer = None
        self._mean = None
        self._weighted_second_moment_diagonal = None
        self._bounded_source_proof = None
        self._last_update_source_row_hashes = ()
        self._context_ids = []
        return state


class DeviceGramFrequentDirectionsTrustAccumulator(
    DeviceFrequentDirectionsTrustAccumulator
):
    """Device FD with block insertion and a small Gram eigendecomposition.

    This is an additive v2 producer.  It does not relabel the existing CUDA
    implementation and emits a distinct derivation algorithm identity.
    """

    _SNAPSHOT_PRODUCER_ALGORITHM_ID = GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID
    _SNAPSHOT_PRODUCER_IMPLEMENTATION = "gram_torch_cuda"

    def __init__(
        self,
        *,
        route_ids: Sequence[str],
        sketch_rank: int,
        device: Any,
        emit_derivation_receipt: bool = False,
    ) -> None:
        super().__init__(
            route_ids=route_ids,
            sketch_rank=sketch_rank,
            device=device,
            emit_derivation_receipt=emit_derivation_receipt,
        )
        self._producer_algorithm_id = GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID
        self._producer_implementation = "gram_torch_cuda"

    def _compress(self, *, force: bool = False) -> None:
        if self._filled <= self.sketch_rank and not force:
            return
        if self._filled == 0:
            return
        rows = self._buffer[: self._filled]
        _check_torch_all_finite("device Gram FD compression buffer", rows)
        scale = float(torch.max(torch.abs(rows)).item())
        if scale == 0.0:
            self._buffer.zero_()
            self._filled = 0
            return
        scaled_rows = rows / scale
        gram = scaled_rows @ scaled_rows.T
        gram = 0.5 * (gram + gram.T)
        # Gram construction is eager on the registered stream. Release its
        # filled-by-route scaling workspace before the retained projection;
        # this changes lifetime only, not the registered fp64 operation graph.
        del scaled_rows
        _check_torch_all_finite("device Gram FD row Gram matrix", gram)
        eigenvalues, left = torch.linalg.eigh(gram)
        _check_torch_all_finite("device Gram FD eigenvalues", eigenvalues)
        _check_torch_all_finite("device Gram FD eigenvectors", left)
        max_eigenvalue = (
            float(torch.max(torch.abs(eigenvalues)).item()) if eigenvalues.numel() else 0.0
        )
        dot_length = int(rows.shape[1])
        psd_tolerance = (
            _GRAM_FD_PSD_TOLERANCE_MULTIPLIER
            * float(torch.finfo(eigenvalues.dtype).eps)
            * max(1, int(self._filled), dot_length)
            * max(1.0, max_eigenvalue)
        )
        if bool(torch.any(eigenvalues < -psd_tolerance).item()):
            raise RuntimeError("device Gram FD row Gram matrix is not PSD within tolerance")
        eigenvalues = torch.clamp(torch.flip(eigenvalues, dims=(0,)), min=0.0)
        left = torch.flip(left, dims=(1,))
        pivots = torch.argmax(torch.abs(left), dim=0)
        signs = torch.sign(left[pivots, torch.arange(left.shape[1], device=left.device)])
        signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
        left = left * signs
        singular = torch.sqrt(eigenvalues) * scale
        retain = min(self.sketch_rank, int(singular.numel()))
        delta_scaled = (
            float(eigenvalues[self.sketch_rank].item())
            if int(singular.numel()) > self.sketch_rank
            else 0.0
        )
        scale_squared = scale * scale
        if not math.isfinite(scale_squared):
            raise ValueError("device Gram FD compression scale overflows fp64")
        delta = delta_scaled * scale_squared
        if not math.isfinite(delta):
            raise ValueError("device Gram FD shrinkage is non-finite")
        shrunk = (
            torch.sqrt(
                torch.clamp(eigenvalues[:retain] - delta_scaled, min=0.0)
            )
            * scale
        )
        projected = left[:, :retain].T @ rows
        gains = torch.where(
            singular[:retain] > 0.0,
            shrunk / singular[:retain],
            torch.zeros_like(shrunk),
        )
        projected.mul_(gains[:, None])
        _check_torch_all_finite("device Gram FD replacement rows", projected)
        self._buffer.zero_()
        self._buffer[:retain].copy_(projected)
        self._filled = int(torch.count_nonzero(shrunk > 0.0).item())
        self._total_shrinkage += delta

    def _insert(self, rows: Any) -> None:
        values = torch.as_tensor(rows, dtype=torch.float64, device=self.device).detach()
        if values.ndim != 2 or int(values.shape[1]) != len(self.route_ids):
            raise ValueError("device Gram FD insertion rows have the wrong route axis")
        _check_torch_all_finite("device Gram FD insertion rows", values)
        cursor = 0
        row_count = int(values.shape[0])
        while cursor < row_count:
            if self._filled >= int(self._buffer.shape[0]):
                self._compress()
            capacity = int(self._buffer.shape[0]) - self._filled
            take = min(capacity, row_count - cursor)
            self._buffer[self._filled : self._filled + take].copy_(
                values[cursor : cursor + take]
            )
            self._filled += take
            cursor += take


def combine_stratified_tripartite_moments(
    moments_by_stratum: Mapping[str, TripartitePullbackMomentTensor],
    *,
    weight_by_stratum: Mapping[str, float],
) -> tuple[LocalAttributionMomentTensor, TripartitePullbackMomentTensor]:
    """Project independent stratum moments through frozen estimand weights."""

    identifiers = tuple(sorted(map(str, moments_by_stratum)))
    if not identifiers or set(identifiers) != set(map(str, weight_by_stratum)):
        raise ValueError("stratified pullback moments require matching weights")
    moments = tuple(moments_by_stratum[key] for key in identifiers)
    reference = moments[0]
    if any(
        item.head_ids != reference.head_ids
        or item.route_ids != reference.route_ids
        or item.destination_transport_hash != reference.destination_transport_hash
        or item.loss_pullback_hash != reference.loss_pullback_hash
        for item in moments[1:]
    ):
        raise ValueError("stratified pullback moments use incompatible identities")
    weights = np.asarray(
        [float(weight_by_stratum[key]) for key in identifiers],
        dtype=np.float64,
    )
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError("stratified pullback weights must be positive and normalized")
    mean = np.einsum(
        "s,shr->hr",
        weights,
        np.stack([item.mean for item in moments], axis=0),
        optimize=True,
    )
    second = np.einsum(
        "s,srhj->rhj",
        weights,
        np.stack([item.route_head_second_moment for item in moments], axis=0),
        optimize=True,
    )
    estimator = np.einsum(
        "s,srhj->rhj",
        weights * weights,
        np.stack([item.mean_estimator_covariance for item in moments], axis=0),
        optimize=True,
    )
    second_se = np.sqrt(
        np.einsum(
            "s,srhj->rhj",
            weights * weights,
            np.stack(
                [item.second_moment_standard_error**2 for item in moments],
                axis=0,
            ),
            optimize=True,
        )
    )
    activity_mean = np.einsum(
        "s,svr->vr",
        weights,
        np.stack([item.view_activity_mean for item in moments], axis=0),
        optimize=True,
    )
    activity_second = np.einsum(
        "s,svr->vr",
        weights,
        np.stack([item.view_activity_second_moment for item in moments], axis=0),
        optimize=True,
    )
    activity_se = np.sqrt(
        np.einsum(
            "s,svr->vr",
            weights * weights,
            np.stack(
                [item.view_activity_mean_standard_error**2 for item in moments],
                axis=0,
            ),
            optimize=True,
        )
    )
    within = [
        math.sqrt(float(weight)) * item.covariance_factor
        for weight, item in zip(weights, moments, strict=True)
    ]
    between = [
        math.sqrt(float(weight)) * (item.mean - mean)[None, :, :]
        for weight, item in zip(weights, moments, strict=True)
    ]
    factor = np.concatenate((*within, *between), axis=0)
    context_axis_hash = content_hash(
        tuple(
            (
                key,
                float(weight),
                moments_by_stratum[key].context_axis_hash,
            )
            for key, weight in zip(identifiers, weights, strict=True)
        ),
        prefix="stratified-tripartite-context-axis-",
    )
    combined = TripartitePullbackMomentTensor(
        head_ids=reference.head_ids,
        route_ids=reference.route_ids,
        mean=mean,
        route_head_second_moment=second,
        mean_estimator_covariance=estimator,
        second_moment_standard_error=second_se,
        covariance_factor=factor,
        sample_count=sum(item.sample_count for item in moments),
        context_axis_hash=context_axis_hash,
        destination_transport_hash=reference.destination_transport_hash,
        loss_pullback_hash=reference.loss_pullback_hash,
        sketch_seed=reference.sketch_seed,
        view_activity_mean=activity_mean,
        view_activity_mean_standard_error=activity_se,
        view_activity_second_moment=activity_second,
    )
    local = LocalAttributionMomentTensor(
        route_ids=combined.route_ids,
        mean=combined.mean[0],
        second_moment=combined.route_head_second_moment[:, 0, 0],
        mean_standard_error=np.sqrt(np.maximum(combined.mean_estimator_covariance[:, 0, 0], 0.0)),
        second_moment_standard_error=combined.second_moment_standard_error[:, 0, 0],
        covariance_factor=combined.covariance_factor[:, 0, :],
        sample_count=combined.sample_count,
        context_axis_hash=combined.context_axis_hash,
        destination_transport_hash=combined.destination_transport_hash,
        sketch_seed=combined.sketch_seed,
    )
    return local, combined


# ---------------------------------------------------------------------------
# v6: the device look snapshot and the device census close.
# ---------------------------------------------------------------------------

_TRIPARTITE_FIELD_ROUTE_AXIS: dict[str, int] = {
    "mean": 1,
    "route_head_second_moment": 0,
    "mean_estimator_covariance": 0,
    "second_moment_standard_error": 0,
    "covariance_factor": 2,
    "view_activity_mean": 1,
    "view_activity_mean_standard_error": 1,
    "view_activity_second_moment": 1,
}
_TRIPARTITE_ORDINARY_FIELDS = (
    "mean",
    "route_head_second_moment",
    "view_activity_mean",
    "view_activity_second_moment",
)
_TRIPARTITE_QUADRATIC_WEIGHT_FIELDS = ("mean_estimator_covariance",)
_TRIPARTITE_STANDARD_ERROR_FIELDS = (
    "second_moment_standard_error",
    "view_activity_mean_standard_error",
)
_DECISION_FIELDS = (
    "mean",
    "mean_standard_error",
    "activity_mean",
    "activity_mean_standard_error",
)


def _route_index(axis: int, ndim: int, start: int, stop: int) -> tuple[Any, ...]:
    index: list[Any] = [slice(None)] * ndim
    index[axis] = slice(int(start), int(stop))
    return tuple(index)


def _allocate_tripartite_host_fields(
    *, head_count: int, route_count: int, factor_rows: int
) -> dict[str, np.ndarray]:
    h, r = int(head_count), int(route_count)
    return {
        "mean": np.empty((h, r), dtype=np.float64),
        "route_head_second_moment": np.empty((r, h, h), dtype=np.float64),
        "mean_estimator_covariance": np.empty((r, h, h), dtype=np.float64),
        "second_moment_standard_error": np.empty((r, h, h), dtype=np.float64),
        "covariance_factor": np.empty((int(factor_rows), h, r), dtype=np.float64),
        "view_activity_mean": np.empty((3, r), dtype=np.float64),
        "view_activity_mean_standard_error": np.empty((3, r), dtype=np.float64),
        "view_activity_second_moment": np.empty((3, r), dtype=np.float64),
    }


def _stream_tripartite_chunk_to_host(
    host: Mapping[str, np.ndarray],
    fields: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    factor_rows: slice | None = None,
) -> None:
    """Exact D2H (or CPU copy) of one route chunk into the final host fields."""

    for name, value in fields.items():
        target = host[name]
        if name == "covariance_factor" and factor_rows is not None:
            target = target[factor_rows]
        if target.size == 0 or int(value.numel()) == 0:
            continue
        view = torch.from_numpy(target)[
            _route_index(_TRIPARTITE_FIELD_ROUTE_AXIS[name], target.ndim, start, stop)
        ]
        view.copy_(value, non_blocking=False)


def _validated_weights(
    identifiers: tuple[str, ...], weight_by_stratum: Mapping[str, float], *, message: str
) -> np.ndarray:
    weights = np.asarray(
        [float(weight_by_stratum[key]) for key in identifiers],
        dtype=np.float64,
    )
    if (
        not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-12)
    ):
        raise ValueError(message)
    return weights


def _nonfinite_route_count(fields: Sequence[Any], route_axes: Sequence[int]) -> Any:
    """Device count of routes with a nonfinite value in any field (no sync)."""

    bad = None
    for value, axis in zip(fields, route_axes, strict=True):
        flags = ~torch.isfinite(value)
        other = tuple(index for index in range(value.ndim) if index != axis)
        if other:
            flags = flags.any(dim=other)
        bad = flags if bad is None else bad | flags
    return torch.count_nonzero(bad)


def combine_stratified_route_decision_snapshots(
    accumulators_by_stratum: Mapping[str, DeviceTripartiteMomentAccumulator],
    *,
    weight_by_stratum: Mapping[str, float],
    population_size_by_stratum: Mapping[str, int],
    head_indices: Sequence[int],
    activity_indices: Sequence[int],
    prepare_stratum: Callable[[str], None] | None = None,
    route_chunk_size: int | None = None,
) -> StratifiedRouteDecisionSnapshot:
    """v6 look snapshot: the stratum fusion, evaluated on the compute device.

    Per route chunk every stratum's ``_decision_chunk`` is folded in a
    fixed order (stratum 0 adopted and scaled, later strata scaled and added;
    squared standard errors weighted by ``w*w``; one final square root).  The
    fused fields stay on the device for the device look; one copy of each comes
    to the host for the look's host consumers.  The count of routes with a
    nonfinite derived field is a receipt fault detector.
    """

    started = time.perf_counter()
    identifiers = tuple(sorted(map(str, accumulators_by_stratum)))
    if (
        not identifiers
        or set(identifiers) != set(map(str, weight_by_stratum))
        or set(identifiers) != set(map(str, population_size_by_stratum))
    ):
        raise ValueError("stratified decision snapshots require matching inputs")
    accumulators = tuple(accumulators_by_stratum[key] for key in identifiers)
    reference = accumulators[0]
    if any(
        item.head_ids != reference.head_ids
        or item.route_ids != reference.route_ids
        or item.destination_transport_hash != reference.destination_transport_hash
        or item.loss_pullback_hash != reference.loss_pullback_hash
        for item in accumulators[1:]
    ):
        raise ValueError("stratified decision snapshots use incompatible identities")
    weights = _validated_weights(
        identifiers,
        weight_by_stratum,
        message="stratified decision weights must be positive and normalized",
    )
    selected_heads = tuple(map(int, head_indices))
    selected_activities = tuple(map(int, activity_indices))
    if (
        not selected_heads
        or len(set(selected_heads)) != len(selected_heads)
        or any(not 0 <= index < len(reference.head_ids) for index in selected_heads)
        or not selected_activities
        or len(set(selected_activities)) != len(selected_activities)
        or any(not 0 <= index < 3 for index in selected_activities)
    ):
        raise ValueError("stratified decision field selection is invalid")
    route_count = len(reference.route_ids)
    sample_counts: dict[str, int] = {}
    context_ids: dict[str, tuple[str, ...]] = {}
    for identifier, accumulator in zip(identifiers, accumulators, strict=True):
        if prepare_stratum is not None:
            prepare_stratum(identifier)
        if accumulator.sample_count <= 0 or accumulator.total_weight <= 0.0:
            raise ValueError("cannot snapshot an empty device accumulator")
        if not accumulator.storage_allocated:
            raise RuntimeError("nonempty device accumulator has no storage")
        accumulator.drain_windows()
        sample_counts[identifier] = int(accumulator.sample_count)
        context_ids[identifier] = accumulator.context_ids
    device = reference._compute_device()
    if any(item._compute_device() != device for item in accumulators[1:]):
        raise RuntimeError("v6 census strata must share one compute device")
    shape_heads = (len(selected_heads), route_count)
    shape_activities = (len(selected_activities), route_count)
    fused = {
        "mean": torch.empty(shape_heads, dtype=torch.float64, device=device),
        "mean_standard_error": torch.empty(shape_heads, dtype=torch.float64, device=device),
        "activity_mean": torch.empty(shape_activities, dtype=torch.float64, device=device),
        "activity_mean_standard_error": torch.empty(
            shape_activities, dtype=torch.float64, device=device
        ),
    }
    nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    chunk = census_v6_route_chunk() if route_chunk_size is None else int(route_chunk_size)
    if chunk <= 0:
        raise ValueError("stratified decision chunk size must be positive")
    for start in range(0, route_count, chunk):
        stop = min(start + chunk, route_count)
        mean = mean_se = activity = activity_se = None
        for stratum_index, (identifier, weight, accumulator) in enumerate(
            zip(identifiers, weights, accumulators, strict=True)
        ):
            chunk_mean, _, chunk_mean_se, chunk_activity, chunk_activity_se = (
                accumulator._decision_chunk(
                    start=start,
                    stop=stop,
                    population_size=int(population_size_by_stratum[identifier]),
                    head_indices=selected_heads,
                    activity_indices=selected_activities,
                )
            )
            w = float(weight)
            ww = float(weight * weight)
            if stratum_index == 0:
                mean = chunk_mean * w
                mean_se = (chunk_mean_se * chunk_mean_se) * ww
                activity = chunk_activity * w
                activity_se = (chunk_activity_se * chunk_activity_se) * ww
            else:
                mean = mean + chunk_mean * w
                mean_se = mean_se + (chunk_mean_se * chunk_mean_se) * ww
                activity = activity + chunk_activity * w
                activity_se = activity_se + (chunk_activity_se * chunk_activity_se) * ww
            del chunk_mean, chunk_mean_se, chunk_activity, chunk_activity_se
        mean_se = torch.sqrt(mean_se)
        activity_se = torch.sqrt(activity_se)
        fused["mean"][:, start:stop] = mean
        fused["mean_standard_error"][:, start:stop] = mean_se
        fused["activity_mean"][:, start:stop] = activity
        fused["activity_mean_standard_error"][:, start:stop] = activity_se
        nonfinite += _nonfinite_route_count((mean, mean_se, activity, activity_se), (1, 1, 1, 1))
        del mean, mean_se, activity, activity_se
    host = {name: fused[name].cpu().numpy() for name in _DECISION_FIELDS}
    receipt = {
        "schema": "irg_v6_census_device_look_snapshot_v3",
        "mode": "v6",
        "compute_device": str(device),
        "route_chunk": int(chunk),
        "fields_to_host": list(_DECISION_FIELDS),
        "head_indices": list(selected_heads),
        "activity_indices": list(selected_activities),
        "order_field": "loss_activity" if 2 in selected_activities else None,
        "device_to_host_bytes": int(sum(value.nbytes for value in host.values())),
        "nonfinite_route_count": int(nonfinite.item()),
        "seconds": float(time.perf_counter() - started),
    }
    return StratifiedRouteDecisionSnapshot(
        head_indices=selected_heads,
        activity_indices=selected_activities,
        mean=host["mean"],
        mean_standard_error=host["mean_standard_error"],
        activity_mean=host["activity_mean"],
        activity_mean_standard_error=host["activity_mean_standard_error"],
        sample_count_by_stratum=sample_counts,
        context_ids_by_stratum=context_ids,
        device_fields=fused,
        v6_receipt=receipt,
    )


# The census close retains the raw global diagnostics alongside the loss
# field that the walk reads. Each head's diagonal statistics are slices of the
# same fused tripartite moments.
CENSUS_GLOBAL_HEAD_STATISTIC_NAMES = (
    "global_contextual_variance",
    "global_second_moment",
    "global_second_moment_standard_error",
    "global_mean_estimator_variance",
)
CENSUS_LOSS_HEAD_STATISTIC_NAMES = (
    "loss_contextual_variance",
    "loss_second_moment",
    "loss_second_moment_standard_error",
    "loss_mean_estimator_variance",
)
# Receipt key of the pending background moment hashes until they are joined
# (``join_census_moment_hashes``).
CENSUS_MOMENT_HASHES_PENDING_KEY = "census_moment_hashes_pending"
# Device memory the close leaves free beyond its own footprint estimate
# (allocator rounding, fragmentation).  The strata are released as soon as the
# close ends, so whatever runs next starts with their memory back.
_CENSUS_CLOSE_DEVICE_RESERVE_BYTES = 4 << 30
# Set by a caller that carries the census receipt to a later join (the
# pilot's campaign path joins after the walk): the census then returns its
# moment hashes pending instead of waiting for them.
_CENSUS_MOMENT_HASHES_DEFERRED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "census_moment_hashes_deferred", default=False
)


@contextlib.contextmanager
def deferred_census_moment_hashes() -> Iterator[None]:
    """Within this block a v6 census returns its moment hashes pending."""

    token = _CENSUS_MOMENT_HASHES_DEFERRED.set(True)
    try:
        yield
    finally:
        _CENSUS_MOMENT_HASHES_DEFERRED.reset(token)


def census_moment_hashes_deferred() -> bool:
    return bool(_CENSUS_MOMENT_HASHES_DEFERRED.get())


@dataclass(frozen=True, slots=True)
class CensusFieldClose:
    """The fused census field as the walk and the cache entry read it (host)."""

    local_mean: np.ndarray
    global_mean: np.ndarray
    global_activity: np.ndarray
    global_statistics: Mapping[str, np.ndarray]
    loss_mean: np.ndarray
    loss_activity: np.ndarray
    loss_mean_standard_error: np.ndarray
    loss_activity_standard_error: np.ndarray
    loss_statistics: Mapping[str, np.ndarray]
    moment_hashes: "CensusMomentHashes"
    receipt: Mapping[str, Any]


class CensusMomentHashes:
    """The census's two moment content hashes, computed on a background thread.

    ``result()`` waits for them and returns ``{"local_attribution_moment_hash",
    "tripartite_pullback_moment_hash", "census_moment_hash_seconds"}``; the
    hashes are exactly ``LocalAttributionMomentTensor.content_hash`` and
    ``TripartitePullbackMomentTensor.content_hash`` of the fused fields.
    """

    __slots__ = ("_future",)

    def __init__(self, future: Any) -> None:
        self._future = future

    def done(self) -> bool:
        return bool(self._future.done())

    def result(self) -> dict[str, Any]:
        return dict(self._future.result())

    # Receipt copies (shallow or deep) share the one pending computation.
    def __copy__(self) -> "CensusMomentHashes":
        return self

    def __deepcopy__(self, memo: Any) -> "CensusMomentHashes":
        return self

    def __repr__(self) -> str:
        return f"<census moment hashes: {'ready' if self.done() else 'pending'}>"


def join_census_moment_hashes(receipt: Any) -> Any:
    """Wait for a census receipt's background moment hashes and write them in.

    Idempotent; a receipt without pending hashes is returned unchanged.  Call
    it before a census receipt is stored, hashed or serialized; a copy of the
    receipt carries the same pending object and can be joined on its own.
    """

    if isinstance(receipt, dict):
        pending = receipt.pop(CENSUS_MOMENT_HASHES_PENDING_KEY, None)
        if pending is not None:
            receipt.update(pending.result())
    return receipt


_CENSUS_HASH_EXECUTOR: Any | None = None
_CENSUS_HASH_EXECUTOR_LOCK = threading.Lock()


def _census_hash_executor() -> Any:
    global _CENSUS_HASH_EXECUTOR
    with _CENSUS_HASH_EXECUTOR_LOCK:
        if _CENSUS_HASH_EXECUTOR is None:
            _CENSUS_HASH_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="census-moment-hash"
            )
        return _CENSUS_HASH_EXECUTOR


def _reset_census_hash_executor_after_fork() -> None:  # pragma: no cover - fork hook
    global _CENSUS_HASH_EXECUTOR, _CENSUS_HASH_EXECUTOR_LOCK
    _CENSUS_HASH_EXECUTOR = None
    _CENSUS_HASH_EXECUTOR_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_census_hash_executor_after_fork)


def _census_moment_hash_job(
    fused: dict[str, Any],
    *,
    ready: Any | None,
    device: Any,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the fused fields to the host (side stream) and hash them.

    Each device field is released as soon as its host copy exists.  The
    tensors are the dataclasses the host finalize builds, so the hashes
    are their ``content_hash`` by construction.
    """

    started = time.perf_counter()
    stream = None
    if ready is not None:
        stream = torch.cuda.Stream(device=device)
        stream.wait_event(ready)
    host: dict[str, np.ndarray] = {}
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        for name in tuple(fused):
            tensor = fused.pop(name)
            host[name] = tensor.numpy() if tensor.device.type == "cpu" else tensor.cpu().numpy()
            del tensor
    copied = time.perf_counter()
    tripartite = TripartitePullbackMomentTensor(
        head_ids=identity["head_ids"],
        route_ids=identity["route_ids"],
        mean=host["mean"],
        route_head_second_moment=host["route_head_second_moment"],
        mean_estimator_covariance=host["mean_estimator_covariance"],
        second_moment_standard_error=host["second_moment_standard_error"],
        covariance_factor=host["covariance_factor"],
        sample_count=identity["sample_count"],
        context_axis_hash=identity["context_axis_hash"],
        destination_transport_hash=identity["destination_transport_hash"],
        loss_pullback_hash=identity["loss_pullback_hash"],
        sketch_seed=identity["sketch_seed"],
        view_activity_mean=host["view_activity_mean"],
        view_activity_mean_standard_error=host["view_activity_mean_standard_error"],
        view_activity_second_moment=host["view_activity_second_moment"],
    )
    local = LocalAttributionMomentTensor(
        route_ids=tripartite.route_ids,
        mean=tripartite.mean[0],
        second_moment=tripartite.route_head_second_moment[:, 0, 0],
        mean_standard_error=np.sqrt(
            np.maximum(tripartite.mean_estimator_covariance[:, 0, 0], 0.0)
        ),
        second_moment_standard_error=tripartite.second_moment_standard_error[:, 0, 0],
        covariance_factor=tripartite.covariance_factor[:, 0, :],
        sample_count=tripartite.sample_count,
        context_axis_hash=tripartite.context_axis_hash,
        destination_transport_hash=tripartite.destination_transport_hash,
        sketch_seed=tripartite.sketch_seed,
    )
    result = {
        "local_attribution_moment_hash": local.content_hash,
        "tripartite_pullback_moment_hash": tripartite.content_hash,
    }
    finished = time.perf_counter()
    result["census_moment_hash_seconds"] = {
        "device_to_host": float(copied - started),
        "hash": float(finished - copied),
        "thread": "background",
    }
    return result


def _census_close_residency(device: Any, needed_bytes: int) -> dict[str, Any]:
    """Keep the fused fields on the device when they fit.

    Headroom is the free device memory plus the caching allocator's free
    blocks, capped by this process's share of the device, less a small fixed
    reserve.  Otherwise the fields stream to host memory chunk by chunk.
    """

    receipt: dict[str, Any] = {"needed_bytes": int(needed_bytes)}
    if device.type != "cuda" or not torch.cuda.is_available():
        return {**receipt, "fields": "compute_device", "device_type": device.type}
    free, total = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    budget = int(process_device_memory_budget_bytes(int(total)))
    headroom = min(cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(free)), budget - allocated)
    reserve = int(_CENSUS_CLOSE_DEVICE_RESERVE_BYTES)
    fits = int(needed_bytes) + reserve <= headroom
    return {
        **receipt,
        "fields": "device" if fits else "host",
        "device_type": "cuda",
        "headroom_bytes": int(headroom),
        "reserve_bytes": reserve,
        "process_budget_bytes": budget,
    }


def close_census_fields_v6(
    accumulators_by_stratum: Mapping[str, DeviceTripartiteMomentAccumulator],
    *,
    weight_by_stratum: Mapping[str, float],
    prepare_stratum: Callable[[str], None] | None = None,
) -> CensusFieldClose:
    """Finalize and fuse the census strata on the device; consume them.

    The stratified finalize, evaluated per route chunk on the compute device
    in a fixed order
    (ordinary fields ``x*w`` summed; standard errors ``(x*x)*(w*w)`` summed then
    square-rooted; the estimator covariance ``x*(w*w)`` summed; within factor
    rows ``x*sqrt(w)``; between rows ``(mean_s - mean)*sqrt(w)``).

    The loss-gradient route fields and retained local/global diagnostics stream
    into their final host buffers per chunk. The complete fused tensor stays on the device
    when it fits (``_census_close_residency``), else it streams to host memory
    per chunk; either way its two content hashes are computed on a background
    thread (``CensusFieldClose.moment_hashes``).  A nonfinite fused field
    raises here.
    """

    started = time.perf_counter()
    identifiers = tuple(sorted(map(str, accumulators_by_stratum)))
    if not identifiers or set(identifiers) != set(map(str, weight_by_stratum)):
        raise ValueError("stratified pullback accumulators require matching weights")
    accumulators = tuple(accumulators_by_stratum[key] for key in identifiers)
    reference = accumulators[0]
    if any(
        item.head_ids != reference.head_ids
        or item.route_ids != reference.route_ids
        or item.destination_transport_hash != reference.destination_transport_hash
        or item.loss_pullback_hash != reference.loss_pullback_hash
        or item.sketch_seed != reference.sketch_seed
        for item in accumulators[1:]
    ):
        raise ValueError("stratified pullback accumulators use incompatible identities")
    weights = _validated_weights(
        identifiers,
        weight_by_stratum,
        message="stratified pullback weights must be positive and normalized",
    )
    h, r = len(reference.head_ids), len(reference.route_ids)
    within_rows = sum(int(item.sketch_rank) for item in accumulators)
    factor_rows = within_rows + len(accumulators)
    context_hashes: list[tuple[str, float, str]] = []
    sample_count = 0
    for identifier, weight, accumulator in zip(identifiers, weights, accumulators, strict=True):
        if prepare_stratum is not None:
            prepare_stratum(identifier)
        if accumulator.sample_count <= 0 or accumulator.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty device accumulator")
        if not accumulator.storage_allocated:
            raise RuntimeError("nonempty device accumulator has no storage")
        accumulator.drain_windows()
        context_hashes.append(
            (
                identifier,
                float(weight),
                content_hash(accumulator.context_ids, prefix="tripartite-pullback-context-axis-"),
            )
        )
        sample_count += int(accumulator.sample_count)
    device = reference._compute_device()
    if any(item._compute_device() != device for item in accumulators[1:]):
        raise RuntimeError("v6 census strata must share one compute device")
    global_head = reference.head_ids.index(
        next(head for head in reference.head_ids if head.startswith("global:"))
    )
    loss_head = len(reference.head_ids) - 1
    if not reference.head_ids[loss_head].startswith("loss:"):
        raise ValueError("the census close requires a final loss-gradient head")
    shapes = {
        "mean": (h, r),
        "route_head_second_moment": (r, h, h),
        "mean_estimator_covariance": (r, h, h),
        "second_moment_standard_error": (r, h, h),
        "covariance_factor": (factor_rows, h, r),
        "view_activity_mean": (3, r),
        "view_activity_mean_standard_error": (3, r),
        "view_activity_second_moment": (3, r),
    }
    walk_names = (
        "local_mean", "global_mean", "global_activity", *CENSUS_GLOBAL_HEAD_STATISTIC_NAMES,
        "loss_mean", "loss_activity", "loss_mean_standard_error", "loss_activity_standard_error",
        *CENSUS_LOSS_HEAD_STATISTIC_NAMES,
    )
    chunk = census_v6_route_chunk()
    fused_bytes = 8 * sum(math.prod(shape) for shape in shapes.values())
    # Per-chunk intermediates: each stratum's fields and the running fusion.
    working_bytes = 4 * fused_bytes // max(r, 1) * min(chunk, r)
    residency = _census_close_residency(device, fused_bytes + working_bytes)
    target = torch.device("cpu") if residency["fields"] == "host" else device
    fused = {
        name: torch.empty(shape, dtype=torch.float64, device=target)
        for name, shape in shapes.items()
    }
    # These fields are only read on the host. A full device slab for each
    # diagnostic duplicated the eventual host output alongside the fused
    # moments and both strata (14.38 GiB for VGG-11's 128,679,936 routes).
    # Keep their final destination on the host while all arithmetic remains
    # on the compute device; each assignment below copies just this chunk.
    walk = {name: torch.empty(r, dtype=torch.float64, device="cpu") for name in walk_names}
    nonfinite = torch.zeros((), dtype=torch.int64, device=device)
    for start in range(0, r, chunk):
        stop = min(start + chunk, r)
        combined: dict[str, Any] = {}
        stratum_means: list[Any] = []
        within_cursor = 0
        for stratum_index, (weight, accumulator) in enumerate(
            zip(weights, accumulators, strict=True)
        ):
            fields = accumulator._finalize_fields_chunk_v6(start, stop)
            rank = int(fields["covariance_factor"].shape[0])
            if rank:
                fused["covariance_factor"][within_cursor : within_cursor + rank, :, start:stop] = (
                    fields["covariance_factor"].mul_(math.sqrt(float(weight)))
                )
            within_cursor += rank
            stratum_means.append(fields["mean"])
            w = float(weight)
            ww = float(weight * weight)
            for name in _TRIPARTITE_ORDINARY_FIELDS:
                # Retain the unscaled stratum means for the between-stratum
                # factor rows. Other chunk fields have no later reader.
                value = fields[name] * w if name == "mean" else fields[name].mul_(w)
                if stratum_index == 0:
                    combined[name] = value
                else:
                    combined[name].add_(value)
            for name in _TRIPARTITE_STANDARD_ERROR_FIELDS:
                value = fields[name].square_().mul_(ww)
                if stratum_index == 0:
                    combined[name] = value
                else:
                    combined[name].add_(value)
            for name in _TRIPARTITE_QUADRATIC_WEIGHT_FIELDS:
                value = fields[name].mul_(ww)
                if stratum_index == 0:
                    combined[name] = value
                else:
                    combined[name].add_(value)
            del fields, value
        mean = combined["mean"]
        for stratum_index, weight in enumerate(weights):
            fused["covariance_factor"][within_rows + stratum_index, :, start:stop] = (
                stratum_means[stratum_index] - mean
            ) * math.sqrt(float(weight))
        for name in _TRIPARTITE_STANDARD_ERROR_FIELDS:
            combined[name].sqrt_()
        for name, value in combined.items():
            target_view = fused[name][
                _route_index(_TRIPARTITE_FIELD_ROUTE_AXIS[name], value.ndim, start, stop)
            ]
            target_view.copy_(value)
        del value, target_view
        global_second = torch.clamp(
            combined["route_head_second_moment"][:, global_head, global_head], min=0.0
        )
        global_mean = mean[global_head]
        walk["local_mean"][start:stop] = mean[0]
        walk["global_mean"][start:stop] = mean[1]
        walk["global_activity"][start:stop] = combined["view_activity_mean"][1]
        walk["global_second_moment"][start:stop] = global_second
        walk["global_contextual_variance"][start:stop] = torch.clamp(
            global_second - global_mean * global_mean, min=0.0
        )
        walk["global_second_moment_standard_error"][start:stop] = combined[
            "second_moment_standard_error"
        ][:, global_head, global_head]
        walk["global_mean_estimator_variance"][start:stop] = torch.clamp(
            combined["mean_estimator_covariance"][:, global_head, global_head], min=0.0
        )
        loss_mean = mean[loss_head]
        loss_second = torch.clamp(
            combined["route_head_second_moment"][:, loss_head, loss_head], min=0.0
        )
        loss_mean_estimator_variance = torch.clamp(
            combined["mean_estimator_covariance"][:, loss_head, loss_head], min=0.0
        )
        walk["loss_mean"][start:stop] = loss_mean
        walk["loss_activity"][start:stop] = combined["view_activity_mean"][2]
        walk["loss_mean_standard_error"][start:stop] = torch.sqrt(loss_mean_estimator_variance)
        walk["loss_activity_standard_error"][start:stop] = combined["view_activity_mean_standard_error"][2]
        walk["loss_second_moment"][start:stop] = loss_second
        walk["loss_contextual_variance"][start:stop] = torch.clamp(
            loss_second - loss_mean * loss_mean, min=0.0
        )
        walk["loss_second_moment_standard_error"][start:stop] = combined[
            "second_moment_standard_error"
        ][:, loss_head, loss_head]
        walk["loss_mean_estimator_variance"][start:stop] = loss_mean_estimator_variance
        names = tuple(combined)
        nonfinite += _nonfinite_route_count(
            tuple(combined[name] for name in names),
            tuple(_TRIPARTITE_FIELD_ROUTE_AXIS[name] for name in names),
        )
        del combined, stratum_means, mean, global_second, global_mean
        del loss_mean, loss_second, loss_mean_estimator_variance
    for accumulator in accumulators:
        accumulator._release_storage()
    ready = None
    if device.type == "cuda":
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
    host = {name: walk[name].numpy() for name in walk_names}
    del walk
    nonfinite_count = int(nonfinite.item())
    if nonfinite_count:
        raise ValueError(
            f"the fused census field has nonfinite values on {nonfinite_count} routes"
        )
    device_seconds = time.perf_counter() - started
    identity = {
        "head_ids": reference.head_ids,
        "route_ids": reference.route_ids,
        "sample_count": int(sample_count),
        "context_axis_hash": content_hash(
            tuple(context_hashes), prefix="stratified-tripartite-context-axis-"
        ),
        "destination_transport_hash": reference.destination_transport_hash,
        "loss_pullback_hash": reference.loss_pullback_hash,
        "sketch_seed": reference.sketch_seed,
    }
    moment_hashes = CensusMomentHashes(
        _census_hash_executor().submit(
            _census_moment_hash_job, fused, ready=ready, device=device, identity=identity
        )
    )
    del fused
    receipt = {
        "schema": "irg_v6_census_device_close_v2",
        "mode": "v6",
        "compute_device": str(device),
        "route_chunk": int(chunk),
        "fused_field_bytes": int(fused_bytes),
        "covariance_factor_rows": int(factor_rows),
        "fused_field_residency": residency,
        "walk_field_residency": "host",
        "fields_to_host": list(walk_names),
        "order_field": "loss_activity",
        "device_to_host_bytes": int(sum(value.nbytes for value in host.values())),
        "moment_hashes": "background_thread",
        "seconds": float(device_seconds),
    }
    return CensusFieldClose(
        local_mean=host["local_mean"],
        global_mean=host["global_mean"],
        global_activity=host["global_activity"],
        global_statistics={name: host[name] for name in CENSUS_GLOBAL_HEAD_STATISTIC_NAMES},
        loss_mean=host["loss_mean"],
        loss_activity=host["loss_activity"],
        loss_mean_standard_error=host["loss_mean_standard_error"],
        loss_activity_standard_error=host["loss_activity_standard_error"],
        loss_statistics={name: host[name] for name in CENSUS_LOSS_HEAD_STATISTIC_NAMES},
        moment_hashes=moment_hashes,
        receipt=receipt,
    )


def census_v6_prepare_accumulators(
    accumulators_by_stratum: Mapping[str, DeviceTripartiteMomentAccumulator],
) -> dict[str, Any]:
    """Pin census accumulators when their combined state fits the device budget.

    The strata stay device resident for the whole census (no per-switch
    offload) whenever their combined persistent bytes fit the registered
    fraction of the device's memory.  Off CUDA (tests, CPU stand-in) the
    storage is already where it computes.  Host-primary accumulators keep
    their host storage; the v6 finalize and snapshot then stream route chunks
    through the device.
    """

    items = tuple(accumulators_by_stratum.values())
    persistent = sum(
        8
        * (
            len(item.head_ids) * len(item.route_ids)
            + 2 * len(item.route_ids) * len(item.head_ids) ** 2
            + 6 * len(item.route_ids)
            + int(item.sketch_rank) * len(item.head_ids) * len(item.route_ids)
            + int(item.sketch_rank)
        )
        for item in items
    )
    device = items[0].device if items else None
    capacity = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        capacity = int(torch.cuda.get_device_properties(device).total_memory)
    budget = (
        None if capacity is None else int(
            process_device_memory_budget_bytes(capacity)
            * census_v6_device_budget_fraction()
        )
    )
    host_primary = any(item.host_primary for item in items)
    pinned = (not host_primary) and (budget is None or persistent <= budget)
    if pinned:
        for item in items:
            item.pin_device_residency()
    return {
        "schema": "irg_v6_census_residency_v1",
        "mode": "v6",
        "storage": "host_primary" if host_primary else "device_resident",
        "device_residency_pinned": bool(pinned),
        "persistent_bytes": int(persistent),
        "device_capacity_bytes": capacity,
        "device_budget_bytes": budget,
        "device_budget_fraction": census_v6_device_budget_fraction(),
    }


def census_v6_accumulator_host_primary(
    *, head_count: int, route_count: int, sketch_rank: int, strata: int, device: Any
) -> bool:
    """Construct census strata device-primary when the budget admits."""

    if torch is None:
        return True
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return False
    persistent = strata * 8 * (
        head_count * route_count
        + 2 * route_count * head_count * head_count
        + 6 * route_count
        + sketch_rank * head_count * route_count
        + sketch_rank
    )
    capacity = int(torch.cuda.get_device_properties(resolved).total_memory)
    return persistent > int(
        process_device_memory_budget_bytes(capacity)
        * census_v6_device_budget_fraction()
    )


# The versioned sibling cache key of a v6 fresh census.
CENSUS_V6_CACHE_NAMESPACE = "irg_v6_census_device_" + DETERMINISTIC_EXECUTION_CACHE_VERSION
CENSUS_V6_SIBLING_CACHE_KEY_PREFIX = "irg-dense-partition-field-cache-v6c-"


def census_v6_sibling_cache_key(base_cache_key: str) -> str:
    """Bind a dense-field key to the current device-census cache namespace."""

    base = str(base_cache_key)
    if not base:
        raise ValueError("the base dense-field cache key is empty")
    return content_hash(
        {"base_cache_key": base, "namespace": CENSUS_V6_CACHE_NAMESPACE},
        prefix=CENSUS_V6_SIBLING_CACHE_KEY_PREFIX,
    )


def census_v6_cache_keys(base_cache_key: str) -> tuple[tuple[str, ...], str]:
    """Return the current census cache key for both reads and writes."""

    base = str(base_cache_key)
    sibling = census_v6_sibling_cache_key(base)
    return (sibling,), sibling


__all__ = [
    "BoundedOrderedFDSourceProof",
    "BoundedOrderedFDSourceProofSnapshot",
    "CENSUS_V6_CACHE_NAMESPACE",
    "CENSUS_GLOBAL_HEAD_STATISTIC_NAMES",
    "CENSUS_LOSS_HEAD_STATISTIC_NAMES",
    "CENSUS_MOMENT_HASHES_PENDING_KEY",
    "DeviceFrequentDirectionsTrustAccumulator",
    "DeviceFrequentDirectionsTrustAccumulatorSnapshot",
    "DeviceGramFrequentDirectionsTrustAccumulator",
    "DeviceTripartiteMomentAccumulator",
    "HostPrimarySlabWindow",
    "HostPrimaryWindowManager",
    "CensusFieldClose",
    "CensusMomentHashes",
    "RouteDecisionMomentSnapshot",
    "StratifiedRouteDecisionSnapshot",
    "census_v6_accumulator_host_primary",
    "census_v6_cache_keys",
    "census_v6_prepare_accumulators",
    "census_v6_sibling_cache_key",
    "close_census_fields_v6",
    "combine_stratified_route_decision_snapshots",
    "combine_stratified_tripartite_moments",
    "census_moment_hashes_deferred",
    "default_host_primary_window_width",
    "deferred_census_moment_hashes",
    "host_primary_accumulators_enabled",
    "host_primary_bytes_per_route",
    "join_census_moment_hashes",
]


class HostPrimarySlabWindow:
    """Host-pinned backing slab with double-buffered device route windows.

    The reusable step-slab primitive behind host-primary accumulators
    (and, in the wiring phase, the walk's ~6.9 GiB readout slab): the
    slab's authoritative bytes live on host (pinned when the compute
    device is CUDA, plain CPU otherwise) and consumers see O(window)
    device views of ``[start:stop]`` along one route axis.

    Slot/event discipline (two slots by default, so window N+1 can
    prefetch while window N computes):

    - ``acquire(start, stop)`` — H2D of the host slice into a free slot
      on the SIDE stream; the compute stream waits only on that slot's
      H2D event (never the host).  A range overlapping a live
      (acquired-but-uncommitted) window is refused — that would alias
      un-written-back bytes.
    - ``commit(start, stop)`` — records a compute-done event on the
      compute stream, the side stream waits on it, then copies the
      window back to the host slice (D2H) and records the slot's commit
      event.  The slot recycles on that event; because every transfer
      runs on the one side stream, a committed range's write-back is
      stream-ordered before any later re-acquisition's H2D — a window's
      write-back always completes before its range is re-acquired.
    - ``prefetch(start, stop)`` — stages a free slot early; a matching
      ``acquire`` is then a hit and costs only the event wait.
    - ``drain()`` — host-synchronizes all outstanding transfers; host
      readers call it before touching the backing slab.

    CPU degenerate mode: with ``copy_windows=False`` (the default off
    CUDA) windows are direct views of the host slab — zero copies,
    identical bytes — while the acquire/commit bookkeeping (overlap
    refusal, slot budget, commit pairing) stays enforced.
    ``copy_windows=True`` on CPU exercises the full staged copy
    discipline with exact copies, which is how the CPU test surface
    certifies the CUDA path's value semantics.
    """

    def __init__(
        self,
        backing: Any | None = None,
        *,
        shape: Sequence[int] | None = None,
        dtype: Any | None = None,
        route_axis: int = 0,
        device: Any = "cpu",
        slots: int = 2,
        copy_windows: bool | None = None,
        transfer_stream: Any | None = None,
        name: str = "slab",
    ) -> None:
        if torch is None:
            raise RuntimeError("host-primary slab windows require PyTorch")
        self.name = str(name)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("host-primary slab window requires an available CUDA device")
        if backing is None:
            if shape is None or dtype is None:
                raise ValueError("host-primary slab window requires backing or shape+dtype")
            backing = self._allocate_host(tuple(int(extent) for extent in shape), dtype)
        if backing.device.type != "cpu":
            raise ValueError("host-primary slab backing must be a CPU tensor")
        self.host = backing
        self.route_axis = int(route_axis)
        if not 0 <= self.route_axis < backing.ndim:
            raise ValueError("host-primary slab route axis is invalid")
        self.route_count = int(backing.shape[self.route_axis])
        self._cuda = self.device.type == "cuda"
        self.copy_windows = self._cuda if copy_windows is None else bool(copy_windows)
        if self._cuda and not self.copy_windows:
            raise ValueError("CUDA host-primary windows require staged copies")
        if int(slots) < 1:
            raise ValueError("host-primary slab window needs >= 1 slot")
        self._elements_per_route = 1
        for axis, extent in enumerate(backing.shape):
            if axis != self.route_axis:
                self._elements_per_route *= int(extent)
        self._stream = None
        if self._cuda:  # pragma: no cover - GPU only
            self._stream = (
                transfer_stream
                if transfer_stream is not None
                else torch.cuda.Stream(device=self.device)
            )
        self._slots: list[dict[str, Any]] = [
            {
                "flat": None,
                "range": None,
                "live": False,
                "prefetched": False,
                "h2d_event": torch.cuda.Event() if self._cuda else None,
                "commit_event": torch.cuda.Event() if self._cuda else None,
                "compute_event": torch.cuda.Event() if self._cuda else None,
                "compute_recorded": False,
            }
            for _ in range(int(slots))
        ]
        self.acquire_count = 0
        self.commit_count = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.bytes_to_device = 0
        self.bytes_to_host = 0

    def _allocate_host(self, shape: tuple[int, ...], dtype: Any) -> Any:
        # Pin only under CUDA: on MPS builds ``pin_memory=True`` silently
        # misallocates (measured, torch 2.10 — see _ChunkOutputStager),
        # and CPU-only builds expose no pinned allocator.
        if self.device.type == "cuda" and torch.cuda.is_available():
            return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)
        return torch.zeros(shape, dtype=dtype, device="cpu")

    @property
    def host_pinned(self) -> bool:
        try:
            return bool(self.host.is_pinned())
        except RuntimeError:  # pragma: no cover - allocator-less builds
            return False

    def _validate_range(self, start: int, stop: int) -> tuple[int, int]:
        start = int(start)
        stop = int(stop)
        if not 0 <= start < stop <= self.route_count:
            raise ValueError(f"host-primary window range is invalid for {self.name}")
        return start, stop

    def _host_slice(self, start: int, stop: int) -> Any:
        index = [slice(None)] * self.host.ndim
        index[self.route_axis] = slice(start, stop)
        return self.host[tuple(index)]

    def _window_shape(self, width: int) -> tuple[int, ...]:
        shape = list(self.host.shape)
        shape[self.route_axis] = int(width)
        return tuple(shape)

    def _overlaps_pending(self, start: int, stop: int) -> bool:
        for slot in self._slots:
            if (slot["live"] or slot["prefetched"]) and slot["range"]:
                slot_start, slot_stop = slot["range"]
                if start < slot_stop and slot_start < stop:
                    return True
        return False

    def _free_slot(self) -> dict[str, Any] | None:
        for slot in self._slots:
            if not slot["live"] and not slot["prefetched"]:
                return slot
        return None

    def _slot_window(self, slot: dict[str, Any]) -> Any:
        start, stop = slot["range"]
        numel = self._elements_per_route * (stop - start)
        return slot["flat"][:numel].view(self._window_shape(stop - start))

    def _stage(self, slot: dict[str, Any], start: int, stop: int) -> Any:
        width = stop - start
        numel = self._elements_per_route * width
        flat = slot["flat"]
        if flat is None or int(flat.numel()) < numel:
            if flat is not None and self._cuda:  # pragma: no cover - GPU
                # The retired buffer may still be in flight on either
                # stream; hand it back stream-ordered.
                flat.record_stream(self._stream)
                flat.record_stream(torch.cuda.current_stream(device=self.device))
            flat = torch.empty(numel, dtype=self.host.dtype, device=self.device)
            slot["flat"] = flat
        slot["range"] = (start, stop)
        window = self._slot_window(slot)
        source = self._host_slice(start, stop)
        if self._cuda:  # pragma: no cover - GPU only
            compute_stream = torch.cuda.current_stream(device=self.device)
            with torch.cuda.stream(self._stream):
                # A new or recycled allocation may still have earlier work
                # queued on the compute stream that owned its storage.
                self._stream.wait_stream(compute_stream)
                if slot["compute_recorded"]:
                    # Never overwrite a slot the compute stream may
                    # still be touching from its previous window.
                    self._stream.wait_event(slot["compute_event"])
                window.copy_(source, non_blocking=True)
                slot["h2d_event"].record(self._stream)
        else:
            window.copy_(source)
        self.bytes_to_device += numel * self.host.element_size()
        return window

    def prefetch(self, start: int, stop: int) -> None:
        """Stage [start:stop] into a free slot ahead of its acquire."""

        start, stop = self._validate_range(start, stop)
        if self._overlaps_pending(start, stop):
            raise RuntimeError(f"host-primary prefetch overlaps a pending window for {self.name}")
        slot = self._free_slot()
        if slot is None:
            raise RuntimeError(f"no free host-primary window slot to prefetch {self.name}")
        if self.copy_windows:
            self._stage(slot, start, stop)
        else:
            slot["range"] = (start, stop)
        slot["prefetched"] = True

    def acquire(self, start: int, stop: int, *, stream: Any | None = None) -> Any:
        """Return the device window for [start:stop]; pair with commit."""

        start, stop = self._validate_range(start, stop)
        self.acquire_count += 1
        matched = next(
            (slot for slot in self._slots if slot["prefetched"] and slot["range"] == (start, stop)),
            None,
        )
        if matched is not None:
            matched["prefetched"] = False
            matched["live"] = True
            self.prefetch_hits += 1
            window = (
                self._slot_window(matched) if self.copy_windows else self._host_slice(start, stop)
            )
        else:
            if self._overlaps_pending(start, stop):
                raise RuntimeError(f"host-primary acquire overlaps a live window for {self.name}")
            slot = self._free_slot()
            if slot is None:
                raise RuntimeError(f"all host-primary window slots of {self.name} are live")
            self.prefetch_misses += 1
            if self.copy_windows:
                window = self._stage(slot, start, stop)
            else:
                slot["range"] = (start, stop)
                window = self._host_slice(start, stop)
            slot["live"] = True
            matched = slot
        if self._cuda:  # pragma: no cover - GPU only
            compute_stream = (
                stream if stream is not None else torch.cuda.current_stream(device=self.device)
            )
            # The compute stream waits ONLY on the incoming window's H2D.
            matched["h2d_event"].wait(compute_stream)
        return window

    def commit(
        self,
        start: int,
        stop: int,
        *,
        writeback: bool = True,
        stream: Any | None = None,
    ) -> None:
        """Write the live [start:stop] window back and recycle its slot."""

        start, stop = self._validate_range(start, stop)
        slot = next(
            (
                candidate
                for candidate in self._slots
                if candidate["live"] and candidate["range"] == (start, stop)
            ),
            None,
        )
        if slot is None:
            raise RuntimeError(f"commit without a live window: {self.name}[{start}:{stop}]")
        self.commit_count += 1
        if self.copy_windows:
            window = self._slot_window(slot)
            destination = self._host_slice(start, stop)
            if self._cuda:  # pragma: no cover - GPU only
                compute_stream = (
                    stream if stream is not None else torch.cuda.current_stream(device=self.device)
                )
                slot["compute_event"].record(compute_stream)
                slot["compute_recorded"] = True
                with torch.cuda.stream(self._stream):
                    self._stream.wait_event(slot["compute_event"])
                    if writeback:
                        destination.copy_(window, non_blocking=True)
                        window.record_stream(self._stream)
                    slot["commit_event"].record(self._stream)
            elif writeback:
                destination.copy_(window)
            if writeback:
                self.bytes_to_host += int(window.numel()) * window.element_size()
        slot["live"] = False
        slot["range"] = None

    def drain(self) -> None:
        """Host-synchronize every outstanding transfer (host readers)."""

        if not self._cuda:
            return
        event = torch.cuda.Event()  # pragma: no cover - GPU only
        event.record(self._stream)  # pragma: no cover - GPU only
        event.synchronize()  # pragma: no cover - GPU only

    def receipts(self) -> dict[str, Any]:
        served = self.prefetch_hits + self.prefetch_misses
        return {
            "name": self.name,
            "mode": "staged" if self.copy_windows else "host_view",
            "route_count": int(self.route_count),
            "acquire_count": int(self.acquire_count),
            "commit_count": int(self.commit_count),
            "bytes_to_device": int(self.bytes_to_device),
            "bytes_to_host": int(self.bytes_to_host),
            "prefetch_hit_fraction": (float(self.prefetch_hits) / float(served) if served else 0.0),
            "pinned_host": self.host_pinned,
        }


class HostPrimaryWindowManager:
    """Acquire and commit all routed accumulator slabs as one window set."""

    def __init__(
        self,
        accumulator: DeviceTripartiteMomentAccumulator,
        *,
        staged: bool,
        slots: int = 2,
    ) -> None:
        device = accumulator.device
        cuda = device.type == "cuda" and torch.cuda.is_available()
        transfer_stream = torch.cuda.Stream(device=device) if cuda else None
        axes = DeviceTripartiteMomentAccumulator._ROUTED_SLAB_AXES
        self._fields = tuple(
            field for field in axes if field != "sketch_sum" or accumulator.sketch_rank > 0
        )
        self._windows = {
            field: HostPrimarySlabWindow(
                getattr(accumulator, field),
                route_axis=axes[field],
                device=device,
                slots=int(slots),
                copy_windows=bool(staged),
                transfer_stream=transfer_stream,
                name=field,
            )
            for field in self._fields
        }
        self._staged = bool(staged)
        self.acquire_count = 0
        self.commit_count = 0

    def acquire_window(
        self,
        start: int,
        stop: int,
        *,
        stream: Any | None = None,
    ) -> dict[str, Any]:
        self.acquire_count += 1
        return {
            field: window.acquire(start, stop, stream=stream)
            for field, window in self._windows.items()
        }

    def commit_window(
        self,
        start: int,
        stop: int,
        *,
        stream: Any | None = None,
    ) -> None:
        self.commit_count += 1
        for window in self._windows.values():
            window.commit(start, stop, stream=stream)

    def drain(self) -> None:
        for window in self._windows.values():
            window.drain()

    def receipts(self) -> dict[str, Any]:
        windows = tuple(self._windows.values())
        hits = sum(window.prefetch_hits for window in windows)
        misses = sum(window.prefetch_misses for window in windows)
        served = hits + misses
        return {
            "staged": self._staged,
            "fields": self._fields,
            "window_acquire_count": int(self.acquire_count),
            "window_commit_count": int(self.commit_count),
            "bytes_to_device": int(sum(window.bytes_to_device for window in windows)),
            "bytes_to_host": int(sum(window.bytes_to_host for window in windows)),
            "prefetch_hit_fraction": (float(hits) / float(served) if served else 0.0),
            "pinned_host": all(window.host_pinned for window in windows),
        }
