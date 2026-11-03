"""Read-only mapped-bank gathers and byte-bounded fixed-order look-ahead.

CUDA calls are confined to registration and DeviceImageGather. The row planner
has no CUDA dependency and never decides how many statistical draws to use.
"""

from __future__ import annotations

import atexit
import ctypes
import mmap
import os
import resource
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from .gpu_runtime import cuda_reusable_memory_bytes
from .run_products import effective_settings

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU tests use Torch operations on the same row metadata.
    triton = tl = None


if triton is not None:
    @triton.jit
    def _image_identity_kernel(Delivered, Expected, Labels, SourceLabels, Result,
                               ROWS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = row < ROWS
        expected = tl.load(Expected + row, active, other=0)
        delivered = tl.load(Delivered + row, active, other=0)
        labels = tl.load(Labels + row, active, other=0)
        source_labels = tl.load(SourceLabels + expected, active, other=0)
        wrong = active & ((delivered != expected) | (labels != source_labels))
        good = tl.sum(wrong.to(tl.int32), 0) == 0
        tl.atomic_min(Result, good.to(tl.int32))


DEVICE_GATHER_ENV = "IRG_DEVICE_GATHER"
DEVICE_GATHER_MAX_BYTES_ENV = "IRG_DEVICE_GATHER_MAX_BYTES"
CUDA_HOST_REGISTER_MAPPED = 0x02
CUDA_HOST_REGISTER_READ_ONLY = 0x08
CUDA_HOST_REGISTER_FLAGS = CUDA_HOST_REGISTER_MAPPED | CUDA_HOST_REGISTER_READ_ONLY


class DeviceGatherError(RuntimeError):
    """An unavailable mapped-bank delivery path, with a receiptable reason."""


def memlock_limit() -> dict[str, int | str]:
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return {
        "soft_bytes": "unlimited" if soft == resource.RLIM_INFINITY else int(soft),
        "hard_bytes": "unlimited" if hard == resource.RLIM_INFINITY else int(hard),
    }


def mapping_span(bank: np.ndarray) -> tuple[int, int, int]:
    """The original mmap span and the npy payload offset within that span."""

    if not isinstance(bank, np.memmap) or getattr(bank, "_mmap", None) is None:
        raise DeviceGatherError("bank_requires_file_mapping")
    if bank.dtype != np.uint8 or bank.ndim != 4 or not bank.flags.c_contiguous:
        raise DeviceGatherError("bank_requires_contiguous_nchw_uint8")
    if bank.flags.writeable or getattr(bank, "mode", None) != "r":
        raise DeviceGatherError("bank_mapping_requires_read_only_mode")
    mapping = bank._mmap
    address = int(np.frombuffer(mapping, dtype=np.uint8, count=1).ctypes.data)
    size = len(mapping)
    offset = int(bank.ctypes.data) - address
    if offset < 0 or offset + int(bank.nbytes) > size:
        raise DeviceGatherError("bank_payload_outside_mapping")
    return address, size, offset


class CudaHostRuntime:
    """Use Torch's CUDA runtime binding and the installed CUDA driver ABI.

    Torch exposes cudaHostRegister/Unregister, but some versions omit
    cudaHostGetDevicePointer. cuMemHostGetDevicePointer_v2 is the corresponding
    driver call; ctypes is part of Python, and libcuda is already required by
    Torch CUDA. No CUDA calls or library loading occur at module import.
    """

    def __init__(self, framework: Any) -> None:
        self.framework = framework
        self.runtime = framework.cuda.cudart()
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.get_pointer = self.driver.cuMemHostGetDevicePointer_v2
        self.get_pointer.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint]
        self.get_pointer.restype = ctypes.c_int

    @staticmethod
    def _code(result: Any) -> int:
        if isinstance(result, tuple):
            result = result[0]
        return int(result)

    def register(self, address: int, size: int, flags: int) -> None:
        result = self._code(self.runtime.cudaHostRegister(address, size, flags))
        if result:
            try:
                description = self.runtime.cudaGetErrorString(result)
            except Exception:
                description = "runtime did not provide an error string"
            raise DeviceGatherError(f"cudaHostRegister_error:{result}:{description}")

    def device_pointer(self, address: int) -> int:
        pointer = ctypes.c_uint64()
        result = int(self.get_pointer(ctypes.byref(pointer), ctypes.c_void_p(address), 0))
        if result or not pointer.value:
            raise DeviceGatherError(f"cuMemHostGetDevicePointer_error:{result}")
        return int(pointer.value)

    def unregister(self, address: int) -> None:
        result = self._code(self.runtime.cudaHostUnregister(address))
        if result:
            raise DeviceGatherError(f"cudaHostUnregister_error:{result}")


def tensor_from_device_pointer(framework: Any, pointer: int, bank: np.ndarray, device: Any) -> Any:
    """Non-owning CUDA storage. The process registry retains its mmap owner.

    This binding is supplied by Torch itself. It does not allocate, copy, or
    free the bank. Registration marks the host mapping read-only; callers only
    index_select from this view, never write into it.
    """

    construct = getattr(framework._C, "_construct_storage_from_data_pointer", None)
    if construct is None:
        raise DeviceGatherError("torch_device_pointer_storage_unavailable")
    storage = construct(int(pointer), framework.device(device), int(bank.nbytes))
    strides = tuple(int(value) // int(bank.itemsize) for value in bank.strides)
    view = framework.empty(0, dtype=framework.uint8, device=device).set_(
        storage, 0, tuple(int(value) for value in bank.shape), strides,
    )
    if int(view.data_ptr()) != int(pointer) or tuple(view.shape) != tuple(bank.shape):
        raise DeviceGatherError("mapped_tensor_view_check_failed")
    return view


@dataclass
class RegisteredBank:
    bank: np.memmap
    address: int
    size: int
    offset: int
    runtime: Any
    views: dict[str, Any]
    resident: dict[tuple[str, str], ResidentImageRows] = field(default_factory=dict)
    label_tables: dict[str, tuple[Any, Any]] = field(default_factory=dict)


@dataclass
class ResidentImageRows:
    """An immutable raw population retained until the item finishes."""

    images: Any
    labels: Any
    ordinals: Any
    slots: dict[int, int]
    ready: Any
    loaded_bytes: int


@dataclass
class RequestedImageRows:
    rows: Any
    ordinals: Any
    images: Any
    labels: Any
    ready: Any


_REGISTRATIONS: dict[tuple[Any, ...], RegisteredBank | str] = {}
_REGISTRATION_LOCK = threading.Lock()


def register_readonly_bank(bank: np.memmap, device: Any, framework: Any, *, runtime: Any = None) -> RegisteredBank:
    """Register a bank once per process; reuse its mapping across data planes.

    A second source mapping of the same immutable file uses the original
    mapping. File identity and payload layout distinguish different banks.
    Failed registrations are remembered rather than retried for every look.
    """

    address, size, offset = mapping_span(bank)
    stat = Path(bank.filename).stat()
    payload_file_offset = (int(bank.offset) // mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY + offset
    key = (os.getpid(), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
           payload_file_offset, tuple(bank.shape), tuple(bank.strides))
    resolved = framework.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = framework.device("cuda", framework.cuda.current_device())
    device_key = str(resolved)
    with _REGISTRATION_LOCK:
        result = _REGISTRATIONS.get(key)
        if isinstance(result, str):
            raise DeviceGatherError(result)
        if result is not None:
            if device_key not in result.views:
                with framework.cuda.device(resolved):
                    pointer = result.runtime.device_pointer(result.address) + result.offset
                    result.views[device_key] = tensor_from_device_pointer(framework, pointer, result.bank, resolved)
            return result
        registered = False
        try:
            runtime = CudaHostRuntime(framework) if runtime is None else runtime
            with framework.cuda.device(resolved):
                runtime.register(address, size, CUDA_HOST_REGISTER_FLAGS)
                registered = True
                pointer = runtime.device_pointer(address) + offset
                view = tensor_from_device_pointer(framework, pointer, bank, resolved)
            result = RegisteredBank(bank, address, size, offset, runtime, {device_key: view})
            _REGISTRATIONS[key] = result
            return result
        except Exception as error:
            reason = f"registration_failed:{type(error).__name__}:{error}"
            if registered:
                try:
                    runtime.unregister(address)
                except Exception as cleanup_error:
                    reason += f";unregister_failed:{cleanup_error}"
            _REGISTRATIONS[key] = reason
            raise DeviceGatherError(reason) from error


def _unregister_process_banks() -> None:
    # Registrations and views stay alive until all device users have stopped.
    # CUDA teardown may already have started at interpreter exit; errors here
    # cannot change a completed run, and the OS releases the registration.
    for key, result in tuple(_REGISTRATIONS.items()):
        if key[0] != os.getpid() or not isinstance(result, RegisteredBank):
            continue
        try:
            for device in result.views:
                result.runtime.framework.cuda.synchronize(device)
            result.runtime.unregister(result.address)
        except Exception:
            pass


atexit.register(_unregister_process_banks)


@dataclass
class _FixedOrder:
    length: int
    read: Callable[[int, int], Sequence[int]]
    cursor: int


class FixedRowLookahead:
    """Plan unique resident rows along fixed orders, independently of looks.

    Orders are read in bounded slices and visited round-robin across strata.
    Cursors belong to delivery only. Installing a known statistical start
    replaces that delivery cursor without modifying the caller's counts.
    Consumers determine their actual rows and order, including repetitions.
    """

    def __init__(self, bank_rows: int, capacity: int, chunk_rows: int) -> None:
        if min(bank_rows, capacity, chunk_rows) <= 0:
            raise DeviceGatherError("device_row_budget_must_be_positive")
        self.bank_rows, self.capacity, self.chunk_rows = int(bank_rows), int(capacity), int(chunk_rows)
        self.resident: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(self.capacity - 1, -1, -1))
        self.consumed: set[int] = set()
        self.orders: OrderedDict[str, _FixedOrder] = OrderedDict()
        self.hits = self.misses = self.prefetched = self.peak_rows = 0

    def validate(self, rows: Sequence[int]) -> tuple[int, ...]:
        result = tuple(int(value) for value in rows)
        if any(row < 0 or row >= self.bank_rows for row in result):
            raise DeviceGatherError("row_ordinal_out_of_range")
        return result

    def install_order(self, key: str, rows: Sequence[int], *, start: int = 0) -> None:
        self.install_reader(key, len(rows), lambda a, b: rows[a:b], start=start)

    def install_reader(self, key: str, length: int, read: Callable[[int, int], Sequence[int]], *, start: int = 0) -> None:
        if not key or not 0 <= int(start) <= int(length):
            raise DeviceGatherError("invalid_fixed_row_order")
        self.orders[str(key)] = _FixedOrder(int(length), read, int(start))

    def _assign(self, row: int, protected: set[int]) -> int:
        if not self.free:
            victim = next((value for value in self.resident if value not in protected), None)
            if victim is None:
                raise DeviceGatherError("device_budget_smaller_than_consumer_chunk")
            self.free.append(self.resident.pop(victim))
            self.consumed.discard(victim)
        slot = self.free.pop()
        self.resident[row] = slot
        self.peak_rows = max(self.peak_rows, len(self.resident))
        return slot

    def demand(self, rows: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        rows = self.validate(rows)
        needed = set(rows)
        if len(needed) > self.capacity:
            raise DeviceGatherError("device_budget_smaller_than_consumer_chunk")
        missing = tuple(dict.fromkeys(row for row in rows if row not in self.resident))
        missing_set = set(missing)
        self.hits += len(rows) - sum(row in missing_set for row in rows)
        self.misses += len(missing)
        fill_slots = tuple(self._assign(row, needed) for row in missing)
        output_slots = tuple(self.resident[row] for row in rows)
        for row in rows:
            self.resident.move_to_end(row)
            self.consumed.discard(row)
        return missing, fill_slots, output_slots

    def mark_consumed(self, rows: Sequence[int]) -> None:
        self.consumed.update(int(row) for row in rows if int(row) in self.resident)

    def ahead(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # Reuse slots only after the side-stream assembly has read them. The
        # caller queues refill on that same stream, so no host wait is needed.
        for row in tuple(self.resident):
            if row in self.consumed:
                self.free.append(self.resident.pop(row))
                self.consumed.discard(row)
        rows, slots = [], []
        while self.free and len(rows) < self.chunk_rows and self.orders:
            width = max(1, (self.chunk_rows - len(rows)) // len(self.orders))
            key, order = self.orders.popitem(last=False)
            stop = min(order.length, order.cursor + min(len(self.free), width))
            if stop > order.cursor:
                values = self.validate(order.read(order.cursor, stop))
                if len(values) != stop - order.cursor:
                    raise DeviceGatherError("fixed_row_order_length_changed")
                order.cursor = stop
                for row in values:
                    if row not in self.resident:
                        rows.append(row)
                        slots.append(self._assign(row, set()))
            if order.cursor < order.length:
                self.orders[key] = order
        self.prefetched += len(rows)
        return tuple(rows), tuple(slots)


def gather_budget_bytes(free: int, process_budget: int, allocated: int, env: Any = None,
                        *, reusable_bytes: int | None = None,
                        minimum_bytes: int = 1) -> int:
    """Admit against reusable capacity while preserving valid default sizing."""

    source = os.environ if env is None else env
    headroom = int(process_budget) - int(allocated)
    sizing_bytes = max(0, min(int(free), headroom))
    available = max(0, min(
        int(free) if reusable_bytes is None else int(reusable_bytes), headroom,
    ))
    raw = str(source.get(DEVICE_GATHER_MAX_BYTES_ENV, "")).strip()
    if raw:
        requested = int(raw)
    else:
        requested = min(2 << 30, max(sizing_bytes // 8, int(minimum_bytes)))
    if requested <= 0 or requested > available:
        raise DeviceGatherError("device_gather_budget_unavailable")
    return requested


class DeviceImageGather:
    """Gather raw rows on a side stream, with no host image producer."""

    def __init__(self, registration: RegisteredBank, labels: Any, framework: Any,
                 device: Any, stream: Any, budget: int, chunk_rows: int,
                 *, process_memory_budget_bytes: int | None = None) -> None:
        device = framework.device(device)
        if device.type == "cuda" and device.index is None:
            device = framework.device("cuda", framework.cuda.current_device())
        self.registration, self.framework, self.device, self.stream = registration, framework, device, stream
        self.bank = registration.views[str(device)]
        self.row_bytes = int(registration.bank[0].nbytes)
        label_bytes = int(np.asarray(labels, dtype=np.int64).nbytes)
        self.chunk_rows = int(chunk_rows)
        scratch_bytes = self.chunk_rows * (self.row_bytes + 16)
        capacity = (int(budget) - label_bytes - scratch_bytes) // (self.row_bytes + 16)
        capacity = min(int(self.bank.shape[0]), capacity)
        if capacity < self.chunk_rows:
            raise DeviceGatherError("device_budget_smaller_than_consumer_chunk")
        self.planner = FixedRowLookahead(int(self.bank.shape[0]), capacity, self.chunk_rows)
        self.budget = int(budget)
        self.process_memory_budget_bytes = process_memory_budget_bytes
        self.allocator_memory_budget_bytes: int | None = None
        if device.type == "cuda" and self.process_memory_budget_bytes is None:
            from .gpu_runtime import process_device_memory_budget_bytes
            _, total = framework.cuda.mem_get_info(device)
            self.process_memory_budget_bytes = process_device_memory_budget_bytes(total)
        self.delivered_rows = self.gathered_rows = 0
        self.failure: str | None = None
        self.delivery_identity_error: Exception | None = None
        self.failure_details: dict[str, Any] | None = None
        self.closed = False
        self.worker_residency = device.type == "cuda"
        self.lock = threading.RLock()
        self.requests = self.requested_rows = self.request_missing_rows = 0
        self.resident_loaded_bytes = self.resident_served_rows = 0
        self.identity_checked_rows = 0
        self._identity_settled = False
        self._tensor_requests: OrderedDict[tuple[int, int], RequestedImageRows] = OrderedDict()
        self._tensor_request_rows = 0
        with framework.cuda.stream(stream):
            self.pages = framework.empty((capacity, *self.bank.shape[1:]), dtype=framework.uint8, device=device)
            self.scratch = framework.empty((self.chunk_rows, *self.bank.shape[1:]), dtype=framework.uint8, device=device)
            with _REGISTRATION_LOCK:
                label_table = registration.label_tables.get(str(device))
                if label_table is None:
                    self.labels = framework.as_tensor(np.array(labels, dtype=np.int64, copy=True), device=device)
                    label_ready = framework.cuda.Event()
                    label_ready.record(stream)
                    registration.label_tables[str(device)] = self.labels, label_ready
                else:
                    self.labels, label_ready = label_table
                    stream.wait_event(label_ready)
            self.row_indexes = framework.empty(self.chunk_rows, dtype=framework.long, device=device)
            self.slot_indexes = framework.empty(self.chunk_rows, dtype=framework.long, device=device)
            self.output_indexes = framework.empty(capacity, dtype=framework.long, device=device)
            self.label_indexes = framework.empty(capacity, dtype=framework.long, device=device)
            # Reuse the former label-index buffer as page identities. The
            # output-index buffer can hold requested ordinals after the
            # page selections have been queued on this same stream.
            self.page_ordinals = self.label_indexes
            self.identity_ok = framework.ones((), dtype=framework.int32, device=device)
        self.reserved_bytes = (capacity + self.chunk_rows) * (self.row_bytes + 16) + label_bytes

    def _copy_indexes(self, target: Any, values: Sequence[int]) -> Any:
        # A tensor constructed directly on CUDA from a Python sequence uses a
        # blocking upload. Pinned metadata and a nonblocking copy let the host
        # enqueue the entire look-ahead without waiting for its gathers. Torch's
        # pinned allocator retains the source until the copy completes; device
        # index buffers are reused only on this one ordered side stream.
        options = {"pin_memory": True} if self.device.type == "cuda" else {}
        host = self.framework.tensor(values, dtype=self.framework.long, device="cpu", **options)
        view = target[:len(values)]
        view.copy_(host, non_blocking=True)
        return view

    def _fill(self, rows: Sequence[int], slots: Sequence[int]) -> None:
        framework = self.framework
        for start in range(0, len(rows), self.chunk_rows):
            block = rows[start:start + self.chunk_rows]
            indexes = self._copy_indexes(self.row_indexes, block)
            destinations = self._copy_indexes(self.slot_indexes, slots[start:start + self.chunk_rows])
            target = self.scratch[:len(block)]
            framework.index_select(self.bank, 0, indexes, out=target)
            self.pages.index_copy_(0, destinations, target)
            self.page_ordinals.index_copy_(0, destinations, indexes)
            self.gathered_rows += len(block)

    def _check_identity(self, delivered: Any, expected: Any, labels: Any) -> None:
        """Accumulate row/label identity on the delivery stream, without a read."""

        count = int(expected.numel())
        if count and delivered.is_cuda:
            if triton is None:
                raise DeviceGatherError("image_identity_requires_triton")
            _image_identity_kernel[(triton.cdiv(count, 256),)](
                delivered, expected, labels, self.labels, self.identity_ok,
                ROWS=count, BLOCK=256,
            )
        elif count:
            expected_labels = self.labels.index_select(0, expected)
            good = delivered.eq(expected).all() & labels.eq(expected_labels).all()
            self.framework.minimum(self.identity_ok, good.to(self.framework.int32), out=self.identity_ok)
        self._identity_settled = False
        self.identity_checked_rows += int(expected.numel())

    def request(self, rows: Sequence[int] | Any, *, ready_event: Any = None) -> Any:
        """Queue already-drawn walk rows ahead; return their CUDA ready event.

        The planner does not advance any statistical stream. Requested pages
        and later batch assembly share this ordered, low-priority stream.
        No host wait and no gather on the consumer's compute stream occur.
        """

        framework = self.framework
        if framework.is_tensor(rows):
            if rows.device.type == "cuda":
                return self._request_tensor(rows, ready_event=ready_event)
            rows = rows.tolist()
        with self.lock, framework.cuda.stream(self.stream):
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            if self.delivery_identity_error is not None:
                raise self.delivery_identity_error
            if self.failure is not None:
                raise DeviceGatherError(self.failure)
            rows = self.planner.validate(rows)
            requested_count = len(rows)
            self.planner.orders.pop("requested_look", None)
            if len(set(rows)) > self.planner.capacity:
                # Large looks use the existing bounded delivery horizon.
                # This reader contains actual requested rows, not forecasts
                # from a statistical stream. Batch-ready events cover refills.
                self.planner.install_order("requested_look", rows, start=self.planner.capacity)
                rows = rows[:self.planner.capacity]
            missing, slots, _ = self.planner.demand(rows)
            self._fill(missing, slots)
            self.requests += 1
            self.requested_rows += requested_count
            self.request_missing_rows += len(missing)
            ready = self.framework.cuda.Event()
            ready.record(self.stream)
            return ready

    def _request_tensor(self, rows: Any, *, ready_event: Any = None) -> Any:
        """Device-planned rows stay on the device throughout image delivery."""

        framework = self.framework
        if rows.device != self.device or rows.dtype != framework.long or rows.ndim != 1 or not rows.is_contiguous():
            raise DeviceGatherError("image_request_requires_contiguous_device_row_vector")
        requested_count = int(rows.numel())
        count = min(requested_count, self.planner.capacity)
        key = (int(rows.data_ptr()), count)
        with self.lock:
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            if self.delivery_identity_error is not None:
                raise self.delivery_identity_error
            if self.failure is not None:
                raise DeviceGatherError(self.failure)
            pending = self._tensor_requests.pop(key, None)
            if pending is not None:
                self._tensor_request_rows -= int(pending.images.shape[0])
            while self._tensor_requests and self._tensor_request_rows + count > self.planner.capacity:
                _, previous = self._tensor_requests.popitem(last=False)
                self._tensor_request_rows -= int(previous.images.shape[0])
            if ready_event is None:
                ready_event = framework.cuda.Event()
                ready_event.record(framework.cuda.current_stream(self.device))
            with framework.cuda.stream(self.stream):
                self.stream.wait_event(ready_event)
                ordinals = rows[:count].clone()
                images = self.bank.index_select(0, ordinals)
                labels = self.labels.index_select(0, ordinals)
                ready = framework.cuda.Event()
                ready.record(self.stream)
                rows.record_stream(self.stream)
            self._tensor_requests[key] = RequestedImageRows(rows, ordinals, images, labels, ready)
            self._tensor_request_rows += count
            self.requests += 1
            self.requested_rows += requested_count
            self.request_missing_rows += count
            self.gathered_rows += count
            return ready

    def _gather_tensor(self, rows: Any) -> tuple[Any, Any, int]:
        if rows.device != self.device or rows.dtype != self.framework.long or rows.ndim != 1 or not rows.is_contiguous():
            raise DeviceGatherError("image_request_requires_contiguous_device_row_vector")
        pointer, count = int(rows.data_ptr()), int(rows.numel())
        if count > self.planner.capacity:
            raise DeviceGatherError("device_budget_smaller_than_consumer_chunk")
        with self.lock:
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            if self.delivery_identity_error is not None:
                raise self.delivery_identity_error
            if self.failure is not None:
                raise DeviceGatherError(self.failure)
            found = next((
                (key, pending, (pointer - key[0]) // rows.element_size())
                for key, pending in self._tensor_requests.items()
                if key[0] <= pointer
                and pointer + count * rows.element_size() <= key[0] + key[1] * rows.element_size()
            ), None)
            missing = count if found is None else 0
            if found is None:
                self._request_tensor(rows)
                key = (pointer, count)
                found = (key, self._tensor_requests[key], 0)
            key, pending, start = found
            self._tensor_requests.move_to_end(key)
            images, labels = pending.images[start:start + count], pending.labels[start:start + count]
            with self.framework.cuda.stream(self.stream):
                self._check_identity(pending.ordinals[start:start + count], rows, labels)
            return images, labels, missing

    def _admit_resident_population(self, row_count: int) -> int:
        """Charge raw rows and metadata to this process before allocating.

        CUDA's current allocation includes all banks and all live workspace
        allocations in the process. The allocator's configured fraction and
        the device's free bytes plus reusable cached blocks also limit admission.
        """
        required = int(row_count) * (self.row_bytes + 16)
        self._admit_resident_bytes(required)
        return required

    def _admit_resident_bytes(self, required: int) -> None:
        budget = self.process_memory_budget_bytes
        if budget is not None:
            if self.device.type == "cuda":
                free, total = self.framework.cuda.mem_get_info(self.device)
                allocated = self.framework.cuda.memory_allocated(self.device)
                get_fraction = getattr(self.framework.cuda, "get_per_process_memory_fraction", None)
                self.allocator_memory_budget_bytes = (
                    int(int(total) * float(get_fraction(self.device)))
                    if get_fraction is not None else int(total)
                )
                effective_budget = min(int(budget), self.allocator_memory_budget_bytes)
                reusable = cuda_reusable_memory_bytes(
                    self.device, framework=self.framework, free_bytes=int(free),
                )
                available = max(0, min(reusable, effective_budget - int(allocated)))
            else:
                retained = sum(pop.images.numel() * pop.images.element_size()
                               + pop.labels.numel() * pop.labels.element_size()
                               + pop.ordinals.numel() * pop.ordinals.element_size()
                               for pop in self.registration.resident.values())
                available = max(0, int(budget) - self.reserved_bytes - retained)
            if required > available:
                raise DeviceGatherError(
                    f"resident_image_population_exceeds_process_budget:"
                    f"required={required};available={available};process_budget={budget};"
                    f"allocator_budget={self.allocator_memory_budget_bytes}"
                )
        return None

    def retain(self, name: str, rows: Sequence[int]) -> Any:
        """Load a fixed calibration/validation population once per item.

        A later plane can request any order or subset of that population.
        The registered mapping owns its tensors, so closing a plane releases
        walk buffers while these populations remain available to other planes
        in this item. The worker releases them after all item planes close.
        """

        rows = tuple(dict.fromkeys(self.planner.validate(rows)))
        if not rows:
            raise DeviceGatherError("resident_image_population_is_empty")
        key = (str(self.device), str(name))
        with self.lock, _REGISTRATION_LOCK, self.framework.cuda.stream(self.stream):
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            population = self.registration.resident.get(key)
            if population is not None:
                if any(row not in population.slots for row in rows):
                    raise DeviceGatherError("resident_image_population_changed")
                self.stream.wait_event(population.ready)
                return population.ready
            self._admit_resident_population(len(rows))
            framework = self.framework
            images = framework.empty((len(rows), *self.bank.shape[1:]), dtype=framework.uint8, device=self.device)
            ordinals = framework.empty(len(rows), dtype=framework.long, device=self.device)
            labels = framework.empty(len(rows), dtype=framework.long, device=self.device)
            # Directly fill the final HBM allocation in bounded chunks. No
            # host image gather, full-population scratch or normalized copy.
            for start in range(0, len(rows), self.chunk_rows):
                block = rows[start:start + self.chunk_rows]
                indexes = self._copy_indexes(ordinals[start:start + len(block)], block)
                framework.index_select(self.bank, 0, indexes, out=images[start:start + len(block)])
                framework.index_select(self.labels, 0, indexes, out=labels[start:start + len(block)])
            ready = framework.cuda.Event()
            ready.record(self.stream)
            loaded_bytes = len(rows) * self.row_bytes
            self.registration.resident[key] = ResidentImageRows(
                images, labels, ordinals, {row: slot for slot, row in enumerate(rows)}, ready, loaded_bytes,
            )
            self.resident_loaded_bytes += loaded_bytes
            return ready

    def gather_resident(self, name: str, rows: Sequence[int]) -> tuple[Any, Any, int]:
        rows = self.planner.validate(rows)
        with self.lock, self.framework.cuda.stream(self.stream):
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            if self.delivery_identity_error is not None:
                raise self.delivery_identity_error
            if self.failure is not None:
                raise DeviceGatherError(self.failure)
            population = self.registration.resident[(str(self.device), str(name))]
            self.stream.wait_event(population.ready)
            slots = tuple(population.slots[row] for row in rows)
            indexes = self._copy_indexes(self.output_indexes, slots)
            images = population.images.index_select(0, indexes)
            labels = population.labels.index_select(0, indexes)
            delivered = population.ordinals.index_select(0, indexes)
            expected = self._copy_indexes(self.output_indexes, rows)
            self._check_identity(delivered, expected, labels)
            self.resident_served_rows += len(rows)
            return images, labels, 0

    def settle_identity(self) -> None:
        """Read the accumulated identity result after the GPU finishes."""

        with self.lock:
            if self.identity_ok is None or self._identity_settled:
                return
            self.stream.synchronize()
            if not bool(self.identity_ok.item()):
                self.failure = "image_row_or_label_identity_mismatch"
                raise DeviceGatherError(self.failure)
            self._identity_settled = True

    def prefetch(self) -> None:
        with self.lock, self.framework.cuda.stream(self.stream):
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            while True:
                rows, slots = self.planner.ahead()
                if not rows:
                    break
                self._fill(rows, slots)

    def gather(self, rows: Sequence[int]) -> tuple[Any, Any, int]:
        if self.framework.is_tensor(rows):
            if rows.device.type == "cuda":
                return self._gather_tensor(rows)
            rows = rows.tolist()
        with self.lock, self.framework.cuda.stream(self.stream):
            if self.closed:
                raise DeviceGatherError("device_gather_closed")
            if self.delivery_identity_error is not None:
                raise self.delivery_identity_error
            if self.failure is not None:
                raise DeviceGatherError(self.failure)
            missing, slots, output_slots = self.planner.demand(rows)
            self._fill(missing, slots)
            indexes = self._copy_indexes(self.output_indexes, output_slots)
            images = self.pages.index_select(0, indexes)
            delivered = self.page_ordinals.index_select(0, indexes)
            ordinals = self._copy_indexes(self.output_indexes, rows)
            labels = self.labels.index_select(0, ordinals)
            self._check_identity(delivered, ordinals, labels)
            self.planner.mark_consumed(rows)
        return images, labels, len(missing)

    def close(self, *, wait: bool = True) -> None:
        with self.lock:
            if self.closed:
                return
            if wait:
                # retain() can enqueue population copies after identity was
                # already settled. Wait for those copies and every resident
                # reader even when settle_identity() has no new flag to read.
                self.stream.synchronize()
                self.settle_identity()
            self.closed = True
            self.pages = self.scratch = self.labels = None
            self.row_indexes = self.slot_indexes = self.output_indexes = self.label_indexes = None
            self.page_ordinals = self.identity_ok = None
            self._tensor_requests.clear()
            self._tensor_request_rows = 0
            self.planner.resident.clear()
            self.planner.consumed.clear()
            self.planner.orders.clear()
            self.planner.free.clear()

    def release_resident_populations(self) -> int:
        """Evict item populations after all readers of this bank have settled.

        The worker closes every item plane before calling this hook. Keep the
        registered read-only mapping and label table, which are independent of
        the item. Evict by device so another device's populations stay intact.
        """

        with self.lock, _REGISTRATION_LOCK:
            if not self.closed:
                raise RuntimeError("resident population release requires a closed gather")
            keys = [key for key in self.registration.resident if key[0] == str(self.device)]
            released = 0
            for key in keys:
                population = self.registration.resident.pop(key)
                released += sum(int(value.numel() * value.element_size()) for value in (
                    population.images, population.labels, population.ordinals,
                ))
            return released

    def receipt(self) -> dict[str, Any]:
        return {
            "selected_path": "device", "registered_mapping_bytes": self.registration.size,
            "effective_settings": effective_settings(resolved={
                DEVICE_GATHER_ENV: "1", "IRG_PINNED_POOL_MAX_BYTES": "0",
            }),
            "registration_flags": CUDA_HOST_REGISTER_FLAGS, "mapping_read_only": True,
            "registration_scope": "process_bank", "max_bytes": self.budget,
            "reserved_bytes": self.reserved_bytes, "lookahead_capacity_rows": self.planner.capacity,
            "current_reserved_bytes": 0 if self.closed else self.reserved_bytes,
            "closed": self.closed,
            "lookahead_resident_rows": len(self.planner.resident), "lookahead_peak_rows": self.planner.peak_rows,
            "lookahead_prefetched_rows": self.planner.prefetched,
            "lookahead_orders": len(self.planner.orders), "gathered_rows": self.gathered_rows,
            "mapped_bank_bytes_read": self.gathered_rows * self.row_bytes + self.resident_loaded_bytes,
            "delivered_rows": self.delivered_rows, "cache_hits": self.planner.hits,
            "failure_reason": (str(self.delivery_identity_error) if self.delivery_identity_error is not None else self.failure),
            "run_failed": self.failure is not None or self.delivery_identity_error is not None,
            "whole_run_retry_required": False,
            "process_memory_budget_bytes": self.process_memory_budget_bytes,
            "allocator_memory_budget_bytes": self.allocator_memory_budget_bytes,
            **({"failure_details": self.failure_details} if self.failure_details is not None else {}),
            "memlock_limit": memlock_limit(), "host_image_producer_threads": 0,
            "request_count": self.requests, "requested_rows": self.requested_rows,
            "request_missing_rows": self.request_missing_rows,
            "pending_device_request_rows": self._tensor_request_rows,
            "resident_loaded_bytes": self.resident_loaded_bytes,
            "resident_served_rows": self.resident_served_rows,
            "resident_scope": "item_bank_device",
            "resident_populations": {
                name: {"rows": len(population.slots), "image_bytes": population.loaded_bytes}
                for (device, name), population in self.registration.resident.items()
                if device == str(self.device)
            },
            "identity_checked_rows": self.identity_checked_rows,
            "identity_state_bytes": 4,
            "identity_settlement": "after_gpu_work",
        }
