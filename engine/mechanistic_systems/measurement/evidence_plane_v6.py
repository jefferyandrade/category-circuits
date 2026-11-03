"""Array moment, allocation and row-identity helpers for the v6 evidence plane.

Counts, draw identities and allocation rules retain their algorithmic meaning.
Float64 reductions may differ within numerical noise.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping as _AbstractMapping
from json.encoder import encode_basestring_ascii as _json_str
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - torch exists wherever the provider runs
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]



EVIDENCE_PLANE_SCHEMA = "irg_v6_evidence_plane_v1"

# ---------------------------------------------------------------------------
# Registered look quanta.
#
# A telemetry quantum (the smallest batch within 95% of the measured peak
# throughput, from ``AdaptiveExecutionTuner``) is timing dependent; a cold
# CUDA tuner falls back to the runtime profile's configured batch rows.  The
# quantum is pinned to the registered profile rows on a CUDA device, so it
# never depends on timing.  On the registered AlexNet profile those rows are
# exactly the two constants below.
# ---------------------------------------------------------------------------
REGISTERED_ADJUDICATION_ALIGNMENT_QUANTUM = 256
REGISTERED_CALIBRATION_ALIGNMENT_QUANTUM = 512

# ---------------------------------------------------------------------------
# Allocation boundary tolerance for explicit host allocation callers.
# ---------------------------------------------------------------------------
# Per-stratum mean and M2 agree with the two-pass ``_device_running_moment``
# within this relative tolerance (registered).
V6_MOMENT_RTOL = 1e-12
V6_ENDPOINT_RTOL = 1e-10

# The fixed-calibration adapter's selector: the only request family whose
# outcome is consumed exclusively by ``_assemble_pair_evidence`` and can
# therefore travel as a lightweight record without an integrity hash.
FIXED_PAIR_SELECTOR_ID = "sealed_validation_neyman_suffix_v1"
V6_MOMENT_PACKET_METADATA_KEY = "irg_v6_moment_packet"
V6_ROW_CODES_METADATA_KEY = "irg_v6_oracle_stratum_codes"
V6_UNHASHED_OUTCOME_PREFIX = "irg-outcome-v6-unhashed:"






# ---------------------------------------------------------------------------
# Host-sync accounting (tests and the benchmark read it; zero cost otherwise)
# ---------------------------------------------------------------------------
class HostSyncCounter:
    """Counts the lane's own device->host transfers per look.

    Only the v6 kernels in this module increment it; it is the lane's
    declared sync budget (one moments D2H per look), not a global profiler.
    """

    __slots__ = ("moment_d2h",)

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.moment_d2h = 0

    def snapshot(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__slots__}


HOST_SYNCS = HostSyncCounter()


# ---------------------------------------------------------------------------
# Segmented per-stratum moments, one device reduction per look
# ---------------------------------------------------------------------------
class SegmentedMomentPlan:
    """Host layout of one packet's rows by stratum (integers only).

    Rows keep their packet order inside each stratum (a stable sort by code).
    Each present stratum owns a contiguous block whose width is the next power
    of two of its count; strata of equal width form one ``[members, width]``
    block, so the padded buffer holds fewer than ``2 n`` slots whatever the
    count skew.  Every reduction is then a fixed-shape ``sum`` over a dense
    axis, which is deterministic on every backend: reruns reproduce the same
    float64 bits.
    """

    __slots__ = (
        "stratum_count",
        "row_count",
        "counts",
        "present",
        "slot_of_row",
        "slot_stratum",
        "groups",
        "total_slots",
        "packed",
        "group_offsets",
    )

    def __init__(self, codes: np.ndarray, stratum_count: int) -> None:
        codes = np.asarray(codes, dtype=np.int64).reshape(-1)
        S = int(stratum_count)
        if S <= 0:
            raise ValueError("segmented moments require at least one stratum")
        if codes.size and (int(codes.min()) < 0 or int(codes.max()) >= S):
            raise ValueError("segmented moments received an out-of-axis stratum code")
        n = int(codes.size)
        counts = np.bincount(codes, minlength=S).astype(np.int64)
        present = np.flatnonzero(counts > 0)
        widths = np.zeros(S, dtype=np.int64)
        if present.size:
            c = counts[present]
            width = np.ones(c.size, dtype=np.int64)
            # next power of two >= count, by integer doubling (no float log)
            while True:
                short = width < c
                if not short.any():
                    break
                width[short] <<= 1
            widths[present] = width
        order_strata = present[np.lexsort((present, widths[present]))]
        offsets = np.zeros(S, dtype=np.int64)
        groups: list[tuple[int, int, int]] = []
        group_members: list[np.ndarray] = []
        cursor = 0
        if order_strata.size:
            w_sorted = widths[order_strata]
            boundaries = np.flatnonzero(np.diff(w_sorted)) + 1
            starts = np.concatenate(([0], boundaries)).astype(np.int64)
            stops = np.concatenate((boundaries, [order_strata.size])).astype(np.int64)
            for a, b in zip(starts.tolist(), stops.tolist()):
                members = order_strata[a:b]
                width_value = int(w_sorted[a])
                offsets[members] = cursor + width_value * np.arange(
                    members.size, dtype=np.int64
                )
                groups.append((cursor, int(members.size), width_value))
                group_members.append(members)
                cursor += width_value * int(members.size)
        total = int(cursor)
        row_order = np.argsort(codes, kind="stable")
        sorted_codes = codes[row_order]
        segment_start = np.zeros(S, dtype=np.int64)
        segment_start[1:] = np.cumsum(counts)[:-1]
        position = np.arange(n, dtype=np.int64) - segment_start[sorted_codes]
        slot_sorted = offsets[sorted_codes] + position
        slot_of_row = np.empty(n, dtype=np.int64)
        slot_of_row[row_order] = slot_sorted
        slot_stratum = np.full(total, -1, dtype=np.int64)
        slot_stratum[slot_of_row] = codes
        # One packed int64 H2D per look: [slot_of_row | slot_stratum |
        # group member ids ...].
        member_block = (
            np.concatenate(group_members) if group_members else np.zeros(0, np.int64)
        )
        group_offsets: list[int] = []
        running = n + total
        for members in group_members:
            group_offsets.append(running)
            running += int(members.size)
        self.stratum_count = S
        self.row_count = n
        self.counts = counts
        self.present = present
        self.slot_of_row = slot_of_row
        self.slot_stratum = slot_stratum
        self.groups = tuple(groups)
        self.total_slots = total
        self.packed = np.concatenate((slot_of_row, slot_stratum, member_block))
        self.group_offsets = tuple(group_offsets)


def _host_to_device(array: np.ndarray, device: Any) -> Any:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if getattr(device, "type", str(device)) == "cuda":
        # Pinned staging: the copy is queued without a stream synchronization.
        return tensor.pin_memory().to(device=device, non_blocking=True)
    return tensor.to(device=device)


def segmented_moments_device(
    values: Any,
    plan: SegmentedMomentPlan,
) -> tuple[Any, Any]:
    """Device-resident per-stratum (mean, M2) for every column: [S, C] each.

    The estimator is the two-pass ``_device_running_moment``: the segment
    mean (sum / count), then the sum of squared deviations from it.  Strata
    with no row hold mean 0 and M2 0.  No host synchronization happens here.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("segmented moments require torch")
    rows = values
    if rows.dim() == 1:
        rows = rows.reshape(-1, 1)
    rows = rows.to(dtype=torch.float64)
    n, C = int(rows.shape[0]), int(rows.shape[1])
    if n != plan.row_count:
        raise ValueError("segmented moments: rows and codes differ in length")
    S = plan.stratum_count
    device = rows.device
    counts_host = plan.counts.astype(np.float64)
    safe_counts = np.where(counts_host > 0, counts_host, 1.0)
    packed_device = _host_to_device(plan.packed, device)
    counts_device = _host_to_device(safe_counts.reshape(S, 1), device)
    total = plan.total_slots
    slot_of_row = packed_device[:n]
    slot_stratum = packed_device[n : n + total]
    flat = torch.zeros((total, C), dtype=torch.float64, device=device)
    flat.index_copy_(0, slot_of_row, rows)
    sums = torch.zeros((S, C), dtype=torch.float64, device=device)
    member_index = []
    for (start, members, width), offset in zip(plan.groups, plan.group_offsets):
        index = packed_device[offset : offset + members]
        member_index.append(index)
        block = flat[start : start + members * width].reshape(members, width, C)
        sums.index_copy_(0, index, block.sum(dim=1))
    means = sums / counts_device
    valid = (slot_stratum >= 0).reshape(total, 1)
    slot_mean = means.index_select(0, torch.clamp(slot_stratum, min=0))
    deviation = torch.where(valid, flat - slot_mean, torch.zeros_like(flat))
    squares = deviation * deviation
    m2 = torch.zeros((S, C), dtype=torch.float64, device=device)
    for (start, members, width), index in zip(plan.groups, member_index):
        block = squares[start : start + members * width].reshape(members, width, C)
        m2.index_copy_(0, index, block.sum(dim=1))
    return means, m2


def merge_moment_tensors(
    prior: tuple[np.ndarray, np.ndarray, np.ndarray],
    new_counts: np.ndarray,
    new_means: Any,
    new_m2: Any,
) -> tuple[Any, Any]:
    """Chan merge in float64 on the same device as the new packet."""

    device = new_means.device
    S = int(new_means.shape[0])
    a = torch.as_tensor(prior[0], dtype=torch.float64, device=device).reshape(S, 1)
    b = torch.as_tensor(new_counts, dtype=torch.float64, device=device).reshape(S, 1)
    mean_a = torch.as_tensor(prior[1], dtype=torch.float64, device=device)
    m2_a = torch.as_tensor(prior[2], dtype=torch.float64, device=device)
    total = a + b
    safe_total = torch.where(total > 0, total, torch.ones_like(total))
    delta = new_means - mean_a
    mean = mean_a + delta * b / safe_total
    m2 = m2_a + new_m2 + delta * delta * a * b / safe_total
    mean = torch.where(a == 0, new_means, torch.where(b == 0, mean_a, mean))
    m2 = torch.where(a == 0, new_m2, torch.where(b == 0, m2_a, m2))
    mean = torch.where(total == 0, torch.zeros_like(mean), mean)
    m2 = torch.where(total == 0, torch.zeros_like(m2), m2)
    return mean, torch.clamp_min(m2, 0.0)


def segmented_moments(
    values: Any,
    codes: np.ndarray,
    stratum_count: int,
    *,
    plan: SegmentedMomentPlan | None = None,
    prior: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    before_host_read: Any = None,
) -> tuple[np.ndarray, Any, Any] | tuple[
    np.ndarray, Any, Any, tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Per-stratum (count, mean, M2) of every column, with ONE device->host copy.

    ``values`` is an ``[n]`` or ``[n, C]`` tensor on any device; ``codes``
    holds each row's stratum ordinal on the host.  Counts are exact integers
    from the codes. With a CUDA ``prior``, packet means and M2 remain on the
    device and the last two return values are the cumulative and packet host
    summaries from the same transfer;
    otherwise the packet arrays are host arrays. Strata with no row hold zero.
    Raises the ``RunningMoment`` error for nonfinite moments.
    """

    layout = plan if plan is not None else SegmentedMomentPlan(codes, stratum_count)
    rows = values
    if rows.dim() == 1:
        rows = rows.reshape(-1, 1)
    C = int(rows.shape[1])
    S = layout.stratum_count
    if layout.row_count == 0:
        zeros = np.zeros((S, C), dtype=np.float64)
        packet = (layout.counts.copy(), zeros, zeros.copy())
        return packet if prior is None else (
            *packet, tuple(np.array(x, copy=True) for x in prior), packet
        )
    means, m2 = segmented_moments_device(rows, layout)
    merged_device = None
    if prior is not None and means.device.type == "cuda":
        merged_device = merge_moment_tensors(prior, layout.counts, means, m2)
        # Both histories' inputs cross in the same packed copy.
        packed = torch.stack((*merged_device, means, m2), dim=0)
    else:
        packed = torch.stack((means, m2), dim=0)
    if packed.device.type == "cuda":
        # One queued D2H transfer into pinned storage.  The event is the
        # sole completion point before the host consumes float64 moments.
        pinned = torch.empty_like(packed, device="cpu", pin_memory=True)
        pinned.copy_(packed, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device=packed.device))
        if before_host_read is not None:
            before_host_read(ready)
        ready.synchronize()
        host = pinned.numpy()
    else:
        if before_host_read is not None:
            before_host_read(None)
        host = packed.to(device="cpu").numpy()
    HOST_SYNCS.moment_d2h += 1
    if merged_device is not None:
        merged = (
            np.asarray(prior[0], dtype=np.int64) + layout.counts,
            host[0].astype(np.float64, copy=False),
            np.maximum(host[1], 0.0).astype(np.float64, copy=False),
        )
        if not (np.isfinite(merged[1]).all() and np.isfinite(merged[2]).all()):
            raise ValueError("running moments must be finite")
        packet_host = (
            layout.counts.copy(), host[2].astype(np.float64, copy=False),
            np.maximum(host[3], 0.0).astype(np.float64, copy=False),
        )
        if not (np.isfinite(packet_host[1]).all() and np.isfinite(packet_host[2]).all()):
            raise ValueError("running moments must be finite")
        return layout.counts.copy(), means, m2, merged, packet_host
    present = (layout.counts > 0).reshape(S, 1)
    means_host = np.where(present, host[0], 0.0)
    m2_host = np.where(present, np.maximum(host[1], 0.0), 0.0)
    if not (np.isfinite(means_host).all() and np.isfinite(m2_host).all()):
        raise ValueError("running moments must be finite")
    packet = (layout.counts.copy(), means_host, m2_host)
    if prior is None:
        return packet
    merged = merge_moment_arrays(prior, packet)
    return (*packet, merged, packet)


def merge_moment_arrays(
    left: tuple[np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge count, mean and M2 arrays with Chan's formulas.

    Empty entries retain the populated side. Counts broadcast against
    ``[S, C]`` float arrays; tiny negative M2 values are clamped to zero.
    """

    c1, m1, s1 = left
    c2, m2_, s2 = right
    c1 = np.asarray(c1, dtype=np.int64)
    c2 = np.asarray(c2, dtype=np.int64)
    m1 = np.asarray(m1, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    m2_ = np.asarray(m2_, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    total = c1 + c2
    shape = np.broadcast_shapes(m1.shape, m2_.shape)
    extra = (1,) * (len(shape) - c1.ndim)
    c1b = c1.reshape(c1.shape + extra)
    c2b = c2.reshape(c2.shape + extra)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        total_f = (c1b + c2b).astype(np.float64)
        c1_f = c1b.astype(np.float64)
        c2_f = c2b.astype(np.float64)
        delta = m2_ - m1
        mean = m1 + delta * c2_f / total_f
        m2 = s1 + s2 + delta * delta * c1_f * c2_f / total_f
    left_empty = np.broadcast_to(c1b == 0, shape)
    right_empty = np.broadcast_to(c2b == 0, shape)
    mean = np.where(left_empty, m2_, np.where(right_empty, m1, mean))
    m2 = np.where(left_empty, s2, np.where(right_empty, s1, m2))
    both_empty = left_empty & right_empty
    mean = np.where(both_empty, 0.0, mean)
    m2 = np.where(both_empty, 0.0, m2)
    if not (np.isfinite(mean).all() and np.isfinite(m2).all()):
        raise ValueError("running moments must be finite")
    return total, mean.astype(np.float64), np.maximum(m2, 0.0)


# ---------------------------------------------------------------------------
# Moment packets and histories with an exact resolver
# ---------------------------------------------------------------------------
class V6MomentPacket:
    """One look's segmented per-stratum packet moments."""

    __slots__ = (
        "names",
        "candidate_ids",
        "codes",
        "counts",
        "_means",
        "_m2",
    )

    def __init__(
        self,
        *,
        names: Sequence[str],
        candidate_ids: Sequence[str],
        codes: np.ndarray,
        counts: np.ndarray,
        means: Any,
        m2: Any,
        values: Any,
        moment_fn: Callable[[Any], Any],
    ) -> None:
        self.names = tuple(names)
        self.candidate_ids = tuple(candidate_ids)
        self.codes = codes
        self.counts = counts
        self._means = means
        self._m2 = m2

    @property
    def means(self) -> np.ndarray:
        if torch is not None and isinstance(self._means, torch.Tensor):
            self._means = self._means.to(device="cpu").numpy()
            HOST_SYNCS.moment_d2h += 1
        return self._means

    @property
    def m2(self) -> np.ndarray:
        if torch is not None and isinstance(self._m2, torch.Tensor):
            self._m2 = self._m2.to(device="cpu").numpy()
            HOST_SYNCS.moment_d2h += 1
        return self._m2

def _running_moment_type() -> Any:
    from mechanistic_systems.inference.sequential_bce import RunningMoment

    return RunningMoment


class V6MomentHistory:
    """Cumulative per-stratum moments of one or more columns."""

    __slots__ = ("names", "column_count", "counts", "means", "m2", "segments", "generation")

    def __init__(self, names: Sequence[str], column_count: int) -> None:
        self.names = tuple(names)
        self.column_count = int(column_count)
        S = len(self.names)
        self.counts = np.zeros(S, dtype=np.int64)
        self.means = np.zeros((S, self.column_count), dtype=np.float64)
        self.m2 = np.zeros((S, self.column_count), dtype=np.float64)
        self.segments: list[tuple[Any, ...]] = []
        self.generation: tuple[str, ...] | None = None

    def merge_packet(
        self,
        packet: V6MomentPacket,
        columns: Sequence[int],
        *,
        premerged: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        cols = list(int(value) for value in columns)
        counts, means, m2 = (
            premerged
            if premerged is not None
            else merge_moment_arrays(
                (self.counts, self.means, self.m2),
                (packet.counts, packet.means[:, cols], packet.m2[:, cols]),
            )
        )
        self.counts, self.means, self.m2 = counts, means, m2
        self.segments.append(None)

    def reset_exact(self, exact: Mapping[str, Sequence[Any]]) -> None:
        """Seed the state from supplied per-column moments."""

        S = len(self.names)
        counts = np.zeros(S, dtype=np.int64)
        means = np.zeros((S, self.column_count), dtype=np.float64)
        m2 = np.zeros((S, self.column_count), dtype=np.float64)
        for index, stratum in enumerate(self.names):
            values = tuple(exact[stratum])
            counts[index] = int(values[0].count)
            for column, moment in enumerate(values):
                if int(moment.count) != int(counts[index]):
                    raise ValueError("exact reset columns differ in count")
                means[index, column] = float(moment.mean)
                m2[index, column] = float(moment.m2)
        self.counts, self.means, self.m2 = counts, means, m2
        self.generation = None
        self.segments.append(None)

class V6StratumMoments(_AbstractMapping):
    """Read-only per-stratum count, mean and M2 view over arrays."""

    __slots__ = ("names", "counts", "means", "m2", "_index")

    def __init__(
        self,
        *,
        names: Sequence[str],
        counts: np.ndarray,
        means: np.ndarray,
        m2: np.ndarray,
        exact: Callable[[], Mapping[str, Any]] | None,
    ) -> None:
        self.names = tuple(names)
        self.counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        self.means = np.asarray(means, dtype=np.float64).reshape(-1)
        self.m2 = np.asarray(m2, dtype=np.float64).reshape(-1)
        self._index = {name: index for index, name in enumerate(self.names)}

    def __getitem__(self, key: str) -> dict[str, float | int]:
        index = self._index[key]
        count = int(self.counts[index])
        mean = float(self.means[index])
        m2 = max(0.0, float(self.m2[index]))
        return {
            "count": count,
            "mean": mean,
            "m2": m2,
            "sample_variance": m2 / float(count - 1) if count > 1 else 0.0,
        }

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

class DrawnCount:
    """``len()``-only stand-in for a stratum's drawn-context set (v6).

    Every reader of the provider's per-stratum history sets uses only
    ``len``; sealed and counter-addressed draws are unique by construction, so
    the count equals the set's size.  ``update`` keeps the set API.
    """

    __slots__ = ("n",)

    def __init__(self, n: int = 0) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def update(self, values: Iterable[Any]) -> None:
        self.n += sum(1 for _ in values)


class V6OracleMomentDict(dict):
    """The provider's per-decision moment entry in v6 mode.

    The arrays in ``history`` own the moments. Pending walk-loop callers
    can materialize ``{stratum: (RunningMoment, ...)}`` when they need the
    mapping interface.
    """

    __slots__ = ("history", "evidence_history")

    def __init__(self, history: V6MomentHistory) -> None:
        super().__init__()
        self.history = history
        self.evidence_history = history

    def sync_from_dict(self) -> "V6OracleMomentDict":
        """Adopt moments written through the pending walk-loop mapping API."""

        if len(self):
            self.history.reset_exact(dict(self))
            self.evidence_history = self.history
            self.clear()
        return self

    def materialize(self) -> "V6OracleMomentDict":
        RunningMoment = _running_moment_type()
        history = self.history
        for index, stratum in enumerate(history.names):
            self[stratum] = tuple(
                RunningMoment(
                    count=int(history.counts[index]),
                    mean=float(history.means[index, column]),
                    m2=float(history.m2[index, column]),
                )
                for column in range(history.column_count)
            )
        return self


class V6BackendState:
    """The fixed-calibration adapter's per-decision v6 state.

    ``history`` holds the candidate column's cumulative moments; ``codes`` the
    per-row stratum ordinals in panel order; ``digest`` the incremental panel
    ``draw_id``; ``sources`` the per-row physical source ids; ``count`` how
    many rows all four describe.
    """

    __slots__ = ("history", "codes", "digest", "sources", "count", "seen", "unique")

    def __init__(self, names: Sequence[str]) -> None:
        self.history = V6MomentHistory(names, 1)
        self.codes = GrowingIntArray()
        self.digest = IncrementalContextsDigest()
        self.sources: list[str] = []
        self.count = 0
        # the panel's context ids, for the O(new rows) uniqueness check
        self.seen: set[str] = set()
        self.unique = True

    def admit(self, context_ids: Sequence[str]) -> None:
        before = len(self.seen)
        self.seen.update(context_ids)
        if len(self.seen) != before + len(context_ids):
            self.unique = False


class V6BackendMomentDict(dict):
    """The adapter's ``moment_history`` slot in v6 mode (dict part empty)."""

    __slots__ = ("state",)

    def __init__(self, state: V6BackendState) -> None:
        super().__init__()
        self.state = state

    def materialize(self) -> "V6BackendMomentDict":
        RunningMoment = _running_moment_type()
        history = self.state.history
        for index, stratum in enumerate(history.names):
            self[stratum] = RunningMoment(
                count=int(history.counts[index]),
                mean=float(history.means[index, 0]),
                m2=float(history.m2[index, 0]),
            )
        return self


# ---------------------------------------------------------------------------
# Exact vectorized Neyman count-target allocation
# ---------------------------------------------------------------------------
def neyman_count_target_allocation(
    request: Any,
    *,
    tie_band_relative: float = 0.0,
) -> tuple[Any, bool]:
    """``UnifiedNeymanAllocator.allocate`` for a count target, without the heap.

    The heap pops ``(-priority, order)``; each stratum's priority sequence is
    non-increasing in its count, so the greedy result is the ``K`` smallest
    keys ``(-priority, order, position)`` of the union of the per-stratum
    sequences.  Candidates are generated per stratum up to a bound that is
    doubled for any stratum whose every generated item was selected, so the
    selected set equals the full union's.  Every receipt field is computed by
    the same scalar formulas as the heap path, so the result object is equal.

    The second return field is retained for the pending walk caller update
    and is always False; no comparison or replay is performed.
    """

    from mechanistic_systems.measurement.neyman_allocation import (
        InsufficientNeymanBudget,
        NeymanAllocationResult,
        UnifiedNeymanAllocator,
    )

    if (
        request.cumulative_count_target is None
        or request.exact_target_counts is not None
    ):
        raise ValueError("vectorized Neyman allocation covers the count-target mode")
    strata = request.strata
    existing = {
        item.stratum_id: int(request.existing_counts.get(item.stratum_id, 0))
        for item in strata
    }
    required = {
        item.stratum_id: max(existing[item.stratum_id], item.minimum_count)
        for item in strata
    }
    minimum_additions = {
        item.stratum_id: required[item.stratum_id] - existing[item.stratum_id]
        for item in strata
    }
    requested_count = int(request.cumulative_count_target)
    if requested_count < sum(existing.values()):
        raise ValueError("cumulative Neyman target cannot shrink")
    reachable = min(requested_count, sum(item.capacity for item in strata))
    if reachable < sum(required.values()):
        raise InsufficientNeymanBudget(
            "cumulative Neyman target cannot supply frozen support minima"
        )
    trace: list[Mapping[str, Any]] = []
    counts = dict(required)
    for item in strata:
        added = minimum_additions[item.stratum_id]
        if added:
            trace.append(
                {
                    "action": "minimum_stratum_support",
                    "stratum": item.stratum_id,
                    "add_count": int(added),
                    "target_count": int(counts[item.stratum_id]),
                    "realized_cost": float(added * item.unit_cost),
                }
            )
    floor = float(request.allocation_variance_floor)
    S = len(strata)
    start = np.asarray([counts[item.stratum_id] for item in strata], dtype=np.int64)
    capacity = np.asarray([item.capacity for item in strata], dtype=np.int64)
    coefficient = np.asarray(
        [
            max(
                float(item.variance_coefficient),
                float(item.estimand_weight) ** 2 * floor,
            )
            for item in strata
        ],
        dtype=np.float64,
    )
    unit_cost = np.asarray([float(item.unit_cost) for item in strata], dtype=np.float64)
    available = np.maximum(0, capacity - start)
    K = int(max(0, reachable - int(start.sum())))
    K = min(K, int(available.sum()))
    optimized = np.zeros(S, dtype=np.int64)
    last_priority = np.zeros(S, dtype=np.float64)
    if K > 0:
        # Per-stratum generation bound, doubled until no truncated stratum
        # had every generated item selected.
        limit = np.minimum(available, max(8, int(math.ceil(2.0 * K / max(1, S))) + 2))
        while True:
            per = np.minimum(available, limit)
            if int(per.sum()) < K:
                limit = np.where(per < available, 2 * np.maximum(limit, 1), limit)
                continue
            owner = np.repeat(np.arange(S, dtype=np.int64), per)
            first = np.concatenate(([0], np.cumsum(per)[:-1])).astype(np.int64)
            offset = np.arange(owner.size, dtype=np.int64) - np.repeat(first, per)
            n = start[owner] + offset
            n_f = n.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                priority = coefficient[owner] / (unit_cost[owner] * n_f * (n_f + 1.0))
            priority = np.where(coefficient[owner] <= 0.0, 0.0, priority)
            priority = np.where(
                (coefficient[owner] > 0.0) & (n <= 0), math.inf, priority
            )
            # heap key (-priority, order, position); lexsort's LAST key is
            # primary.
            key_order = np.lexsort((offset, owner, -priority))
            if K > key_order.size:
                raise RuntimeError("Neyman candidate generation undercounted")
            chosen = key_order[:K]
            chosen_per = np.bincount(owner[chosen], minlength=S)
            saturated = (chosen_per == per) & (per < available)
            if not saturated.any():
                break
            limit = np.where(saturated, np.minimum(available, 2 * np.maximum(limit, 1)), limit)
        optimized = chosen_per.astype(np.int64)
        for h in np.flatnonzero(optimized > 0).tolist():
            n_last = int(start[h] + optimized[h] - 1)
            last_priority[h] = UnifiedNeymanAllocator._priority(
                strata[h], n_last, floor
            )
    for index, item in enumerate(strata):
        counts[item.stratum_id] = int(start[index] + optimized[index])
    for index, item in enumerate(strata):
        added = int(optimized[index])
        if added:
            value = float(last_priority[index])
            trace.append(
                {
                    "action": "cost_aware_neyman_add",
                    "stratum": item.stratum_id,
                    "add_count": added,
                    "target_count": int(counts[item.stratum_id]),
                    "last_predicted_variance_reduction_per_cost": (
                        value
                        if math.isfinite(value)
                        else "unobserved_stratum_infinite_priority"
                    ),
                    "continuous_neyman_weight": float(item.continuous_neyman_weight),
                }
            )
    additions = {
        item.stratum_id: counts[item.stratum_id] - existing[item.stratum_id]
        for item in strata
    }
    incremental_cost = sum(additions[item.stratum_id] * item.unit_cost for item in strata)
    delivered = sum(counts.values())
    population = sum(item.capacity for item in strata)
    result = NeymanAllocationResult(
        target_counts=dict(counts),
        additions=additions,
        expected_variance=UnifiedNeymanAllocator._expected_variance(
            strata,
            counts,
            finite_population_correction=request.finite_population_correction,
        ),
        incremental_cost=float(incremental_cost),
        unused_cost_budget=0.0,
        requested_cumulative_count=requested_count,
        delivered_cumulative_count=int(delivered),
        population_count=int(population),
        full_census=all(counts[item.stratum_id] == item.capacity for item in strata),
        finite_population_correction=bool(request.finite_population_correction),
        trace=tuple(trace),
        role=request.role,
        policy_id=request.policy_id,
    )
    return result, False


# ---------------------------------------------------------------------------
# Exact incremental identities
# ---------------------------------------------------------------------------
class IncrementalContextsDigest:
    """``content_hash({"state_hash": s, "contexts": ids})`` in O(new ids).

    ``canonical_json`` of that dict is ``{"contexts":[...],"state_hash":...}``
    (sorted keys, compact separators, ASCII escapes); the SHA-256 state after
    the list's current end is kept and extended with each look's suffix, so
    every digest equals the full recomputation byte for byte.
    """

    __slots__ = ("_state", "_count")

    _HEAD = b'{"contexts":['

    def __init__(self) -> None:
        self._state = hashlib.sha256(self._HEAD)
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def extend(self, context_ids: Sequence[str]) -> None:
        values = tuple(context_ids)
        if not values:
            return
        if set(map(type, values)) - {str}:
            raise TypeError("incremental context digest requires exact str ids")
        text = ",".join(map(_json_str, values))
        if self._count:
            text = "," + text
        self._state.update(text.encode("ascii"))
        self._count += len(values)

    def digest(self, state_hash: str) -> str:
        final = self._state.copy()
        final.update(b'],"state_hash":' + _json_str(str(state_hash)).encode("ascii") + b"}")
        return final.hexdigest()


V6_PANEL_CODES_ATTRIBUTE = "_v6_row_stratum_codes"
V6_PANEL_UNIQUE_ATTRIBUTE = "_v6_unique_contexts_verified"


def mark_panel_v6(
    panel: Any,
    codes: np.ndarray,
    names: Sequence[str],
    *,
    unique_verified: bool = False,
) -> None:
    """Attach the row codes (and a uniqueness certificate) to a v6 panel.

    Stored in the instance ``__dict__`` (not dataclass fields), like the
    memoised ``panel_hash``: equality, ``asdict`` and ``replace`` ignore it.
    """

    object.__setattr__(
        panel,
        V6_PANEL_CODES_ATTRIBUTE,
        (tuple(names), np.array(codes, dtype=np.int64, copy=True)),
    )
    if unique_verified:
        object.__setattr__(panel, V6_PANEL_UNIQUE_ATTRIBUTE, True)


_FOLD_TABLE: dict[str, int] = {}
_FOLD_TABLE_LIMIT = 4_000_000


def calibration_fold_index(source_context_id: str, blocks: int) -> int:
    """``_calibration_fold_index`` through a per-process table (exact).

    The value is ``int(sha256(canonical_json({schema, id})), 16) % blocks``;
    the canonical JSON is formed by the same template (sorted keys, compact,
    ASCII escapes), computed once per source id and then read from the table.
    """

    key = str(source_context_id)
    cached = _FOLD_TABLE.get(key)
    if cached is None or cached >> 8 != int(blocks):
        text = (
            '{"schema":"irg_calibration_fold_assignment_v1","source_context_id":'
            + _json_str(key)
            + "}"
        )
        value = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % int(blocks)
        if len(_FOLD_TABLE) < _FOLD_TABLE_LIMIT and 0 < int(blocks) < 256:
            _FOLD_TABLE[key] = (int(blocks) << 8) | value
        return value
    return cached & 0xFF


def calibration_fold_table(source_context_ids: Sequence[str], blocks: int) -> np.ndarray:
    """int8 fold of every source id of one axis (a precomputed table)."""

    return np.fromiter(
        (calibration_fold_index(value, blocks) for value in source_context_ids),
        dtype=np.int8,
        count=len(source_context_ids),
    )


def names_for_codes(names: Sequence[str], codes: np.ndarray) -> tuple[str, ...]:
    table = tuple(names)
    return tuple(map(table.__getitem__, np.asarray(codes, dtype=np.int64).tolist()))


class GrowingIntArray:
    """An append-only int64 buffer with amortized O(1) appends."""

    __slots__ = ("_data", "_size")

    def __init__(self, capacity: int = 1024) -> None:
        self._data = np.empty(max(16, int(capacity)), dtype=np.int64)
        self._size = 0

    def extend(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.int64).reshape(-1)
        need = self._size + values.size
        if need > self._data.size:
            grown = np.empty(max(need, 2 * self._data.size), dtype=np.int64)
            grown[: self._size] = self._data[: self._size]
            self._data = grown
        self._data[self._size : need] = values
        self._size = need

    def reset(self, values: np.ndarray) -> None:
        self._size = 0
        self.extend(values)

    @property
    def view(self) -> np.ndarray:
        return self._data[: self._size]

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Exported evidence-plane interfaces
# ---------------------------------------------------------------------------


__all__ = [
    "EVIDENCE_PLANE_SCHEMA",
    "FIXED_PAIR_SELECTOR_ID",
    "HOST_SYNCS",
    "HostSyncCounter",
    "IncrementalContextsDigest",
    "GrowingIntArray",
    "REGISTERED_ADJUDICATION_ALIGNMENT_QUANTUM",
    "REGISTERED_CALIBRATION_ALIGNMENT_QUANTUM",
    "SegmentedMomentPlan",
    "V6MomentHistory",
    "V6MomentPacket",
    "V6StratumMoments",
    "V6_ENDPOINT_RTOL",
    "V6_MOMENT_PACKET_METADATA_KEY",
    "V6_MOMENT_RTOL",
    "V6_ROW_CODES_METADATA_KEY",
    "V6_UNHASHED_OUTCOME_PREFIX",
    "merge_moment_arrays",
    "names_for_codes",
    "neyman_count_target_allocation",
    "segmented_moments",
    "segmented_moments_device",
]
