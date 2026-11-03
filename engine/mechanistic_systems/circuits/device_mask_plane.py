"""Device side of the v6 committed masks and candidate views.

Within one route group a candidate is its parent minus one slice
``order[k_new:k_old]`` of a fixed route order.  The orders stay resident on
the walk's torch device; a candidate's removed indices are a device sort of
its slice, and its big-endian packed mask is the parent's resident packed
mask with those bits cleared on the device.  Accepting a candidate makes its
packed tensor the next parent's; nothing is copied.

The identities stay on the host.  ``HardMask.mask_hash`` (SHA-256 of the
packed bytes) and the removed-index hash key the oracle draws, so their
definitions cannot change.  The device produces exactly the bytes they hash,
and the host receives them, with a one-byte fault flag, in one transfer per
candidate.

All work runs on one side stream per CUDA device, so a candidate can be
formed while the main stream runs a forward.  On CPU tensors (tests, CPU
hosts) the same code runs synchronously.  Every operation is integer
arithmetic, so the device path is exact.
"""

from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np

_PLANES: dict[str, "MaskPlane"] = {}
_PLANES_LOCK = threading.Lock()
_INT32_LIMIT = 2**31 - 1

# Receipt-free counters (runtime telemetry only; never read by the walk).
V6_MASK_PLANE_TELEMETRY: dict[str, int] = {
    "order_uploads": 0,
    "order_upload_bytes": 0,
    "packed_uploads": 0,
    "candidates": 0,
    "fetches": 0,
    "fetch_bytes": 0,
}


def _torch():
    import torch

    return torch


def mask_plane_device() -> str:
    """The torch device of the walk's mask plane.

    CUDA (the current device) when present, otherwise CPU tensors, so CPU
    hosts and tests run the same code.  ``IRG_V6_SA_SORT_DEVICE`` (the
    existing v6 control-plane device setting) may name a torch device.
    """

    raw = os.environ.get("IRG_V6_SA_SORT_DEVICE", "auto").strip().lower() or "auto"
    if raw not in {"auto", "numpy"}:
        return raw
    torch = _torch()
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def plane_for_walk() -> "MaskPlane":
    return MaskPlane.for_device(mask_plane_device())


class MaskPlane:
    """Resident route orders and packed-mask arithmetic on one torch device."""

    def __init__(self, device: Any) -> None:
        torch = _torch()
        self.device = torch.device(device)
        self.stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self._big = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.uint8, device=self.device)
        self._little = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=self.device)
        table = np.array(
            [int(f"{value:08b}"[::-1], 2) for value in range(256)], dtype=np.uint8
        )
        self._reverse = torch.from_numpy(table).to(self.device)

    @classmethod
    def for_device(cls, device: Any) -> "MaskPlane":
        key = str(_torch().device(device))
        with _PLANES_LOCK:
            plane = _PLANES.get(key)
            if plane is None:
                plane = _PLANES[key] = cls(key)
            return plane

    def scope(self):
        torch = _torch()
        return torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

    # -- uploads -------------------------------------------------------------
    def upload_order(self, values: np.ndarray) -> Any:
        """One immutable route order (int32 when its values fit)."""

        torch = _torch()
        array = np.asarray(values).reshape(-1)
        dtype = np.int32 if (array.size == 0 or int(array.max()) <= _INT32_LIMIT) else np.int64
        host = np.array(array, dtype=dtype, copy=True)
        with self.scope():
            tensor = torch.from_numpy(host).to(self.device)
        self._settle()
        V6_MASK_PLANE_TELEMETRY["order_uploads"] += 1
        V6_MASK_PLANE_TELEMETRY["order_upload_bytes"] += int(host.nbytes)
        return tensor

    def gathered_order(self, eligible: np.ndarray, order: np.ndarray) -> Any:
        """``eligible[order]`` formed on the device (no host gather)."""

        torch = _torch()
        values, positions = self.upload_order(eligible), self.upload_order(order)
        with self.scope():
            tensor = values[positions.to(torch.int64)]
        del values, positions
        self._settle()
        return tensor

    def upload_packed(self, packed: bytes) -> Any:
        torch = _torch()
        host = np.frombuffer(packed, dtype=np.uint8).copy()
        with self.scope():
            tensor = torch.from_numpy(host).to(self.device)
        self._settle()
        V6_MASK_PLANE_TELEMETRY["packed_uploads"] += 1
        return tensor

    def upload_indices(self, values: np.ndarray) -> Any:
        torch = _torch()
        host = np.ascontiguousarray(values, dtype=np.int64)
        with self.scope():
            tensor = torch.from_numpy(host.copy()).to(self.device)
        self._settle()
        return tensor

    # -- candidate arithmetic --------------------------------------------------
    def sorted_slice(self, order: Any, start: int, stop: int) -> Any:
        """``sort(order[start:stop])``: one slice's removed indices, ascending."""

        torch = _torch()
        with self.scope():
            return torch.sort(order[int(start) : int(stop)]).values

    def segment(self, values: Any, start: int, stop: int, offset: int) -> Any:
        """``values[start:stop] - offset`` (one layer of a pooled slice)."""

        with self.scope():
            part = values[int(start) : int(stop)]
            return part - int(offset) if int(offset) else part

    def positions(self, parts: Sequence[tuple[Any, int]], *, wide: bool) -> Any:
        """Global bit positions: ``offset + values`` for each ascending part.

        The parts are given in axis order and occupy disjoint ascending
        ranges, so the concatenation is ascending.  ``wide`` selects int64.
        """

        torch = _torch()
        dtype = torch.int64 if wide else torch.int32
        with self.scope():
            pieces = [values.to(dtype) + int(offset) for values, offset in parts]
            return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def clear(self, packed: Any, positions: Any, low: int, high: int) -> tuple[Any, Any]:
        """The parent's big-endian packed bytes with ``positions`` cleared.

        ``positions`` are ascending bit positions inside bytes ``[low, high)``.
        Returns the candidate's packed tensor and a one-byte fault flag:
        nonzero when a position repeats or names a bit the parent does not
        retain (the invariant a candidate view relies on).  Distinct bits of
        one byte sum to their OR, so an int32 scatter-add builds the clear mask.
        """

        torch = _torch()
        with self.scope():
            wide = positions.to(torch.int64)
            byte = (wide >> 3) - int(low)
            bit = torch.bitwise_right_shift(
                torch.full_like(wide, 0x80), wide & 7
            ).to(torch.int32)
            mask = torch.zeros(int(high) - int(low), dtype=torch.int32, device=self.device)
            mask.index_add_(0, byte, bit)
            clear = mask.to(torch.uint8)
            segment = packed[int(low) : int(high)]
            fault = (segment & clear).ne(clear).any()
            if wide.numel() > 1:
                fault = fault | wide[1:].le(wide[:-1]).any()
            candidate = packed.clone()
            candidate[int(low) : int(high)] = segment & ~clear
            flag = fault.reshape(1).to(torch.uint8)
        V6_MASK_PLANE_TELEMETRY["candidates"] += 1
        return candidate, flag

    def little_bits(self, packed: Any, start: int, stop: int) -> Any:
        """Little-endian packed bytes of bits ``[start, stop)`` of a big-endian mask.

        These are the per-layer bytes ``sampled_path_mask_hash`` hexlifies
        (``np.packbits(layer, bitorder="little")``).  A byte-aligned layer is a
        bit-reversal table lookup; an unaligned one is unpacked and repacked.
        """

        torch = _torch()
        start, stop = int(start), int(stop)
        first = start >> 3
        last = (stop + 7) >> 3
        with self.scope():
            segment = packed[first:last]
            if start % 8 == 0:
                little = self._reverse[segment.to(torch.int64)]
                tail = stop % 8
                if tail:
                    little[-1] &= (1 << tail) - 1
                return little
            bits = ((segment.reshape(-1, 1) >> self._big) & 1).reshape(-1)
            bits = bits[start - 8 * first : stop - 8 * first]
            pad = (-int(bits.numel())) % 8
            if pad:
                bits = torch.cat((bits, torch.zeros(pad, dtype=torch.uint8, device=self.device)))
            return (bits.reshape(-1, 8) << self._little).sum(dim=1, dtype=torch.uint8)

    # -- transfers -------------------------------------------------------------
    def fetch(self, *tensors: Any) -> tuple[np.ndarray, ...]:
        """Host NumPy arrays of ``tensors`` with one synchronization."""

        torch = _torch()
        if self.stream is None:
            out = tuple(np.array(t.detach().reshape(t.shape).numpy(), copy=True) for t in tensors)
        else:
            staged = []
            with self.scope():
                for tensor in tensors:
                    flat = tensor.detach().reshape(-1)
                    buffer = torch.empty(flat.numel(), dtype=flat.dtype, pin_memory=True)
                    buffer.copy_(flat, non_blocking=True)
                    staged.append((buffer, tuple(tensor.shape)))
            self.stream.synchronize()
            out = tuple(buffer.numpy().reshape(shape) for buffer, shape in staged)
        V6_MASK_PLANE_TELEMETRY["fetches"] += 1
        V6_MASK_PLANE_TELEMETRY["fetch_bytes"] += int(sum(array.nbytes for array in out))
        return out

    def _settle(self) -> None:
        if self.stream is not None:
            self.stream.synchronize()


class DeviceRemoved:
    """One slice's sorted removed local indices, on the device until fetched.

    Copies of a ``_LayerSlice`` share this holder, so one fetch serves all
    of them.  After the host values are set the device tensor is released.
    """

    __slots__ = ("plane", "device", "count", "_host")
    fetched_host = False

    def __init__(self, plane: MaskPlane, device_values: Any, count: int) -> None:
        self.plane = plane
        self.device = device_values
        self.count = int(count)
        self._host: np.ndarray | None = None

    @property
    def fetched(self) -> bool:
        return self._host is not None

    def set_host(self, values: np.ndarray) -> None:
        owned = np.ascontiguousarray(values, dtype=np.int64).reshape(-1)
        if owned.size != self.count:
            raise ValueError("device removed set changed size on transfer")
        self._host = np.frombuffer(owned.tobytes(order="C"), dtype=np.int64)
        self.device = None

    def host(self) -> np.ndarray:
        if self._host is None:
            (values,) = self.plane.fetch(self.device)
            self.set_host(values)
        return self._host  # type: ignore[return-value]

    def tensor(self) -> Any:
        """The device values (re-uploaded from the host copy once released)."""

        if self.device is None:
            return self.plane.upload_indices(self._host)  # type: ignore[arg-type]
        return self.device


def fetch_removed(holders) -> None:
    """Fetch every pending device holder, one transfer per plane."""

    pending: dict[int, tuple[MaskPlane, list[DeviceRemoved]]] = {}
    for holder in holders:
        if isinstance(holder, DeviceRemoved) and not holder.fetched:
            pending.setdefault(id(holder.plane), (holder.plane, []))[1].append(holder)
    for plane, items in pending.values():
        values = plane.fetch(*(item.device for item in items))
        for item, array in zip(items, values, strict=True):
            item.set_host(array)


class OrderCache:
    """The resident device orders of one lowerer, keyed by (layer, partition).

    Held by the lowerer, so the device memory lives as long as the group.
    """

    def __init__(self) -> None:
        self._orders: dict[tuple[Any, ...], Any] = {}

    def get(self, plane: MaskPlane, key: tuple[Any, ...], build) -> Any:
        """The resident order ``key``; ``build(plane)`` forms it on first use."""

        cache_key = (str(plane.device), *key)
        tensor = self._orders.get(cache_key)
        if tensor is None:
            tensor = self._orders[cache_key] = build(plane)
        return tensor


__all__ = [
    "DeviceRemoved",
    "MaskPlane",
    "OrderCache",
    "V6_MASK_PLANE_TELEMETRY",
    "fetch_removed",
    "mask_plane_device",
    "plane_for_walk",
]
