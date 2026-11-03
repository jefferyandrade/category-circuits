"""Exact bounded identities for ephemeral device-resident hard-mask ladders.

A relaxed-KKT implementation may label each physical route with the first
nested hard endpoint that removes it.  Keeping those one-byte shell labels on
the device avoids a route-scale ordered-index transfer and avoids retaining an
``endpoint x route`` mask matrix.  This module provides the authority-inert
identity seam for that representation:

* an explicitly evictable, nonserializable shell-label handle;
* a bounded physical-axis scan that validates nesting and hard legality;
* exact historical ``RemovedRouteIndexSet`` and ``HardMask`` identities for
  every cumulative endpoint; and
* materialization of exactly one already-frozen endpoint.

The receipts here certify representation identity and registered hard
legality only.  They grant no F/Q/R, calibration, oracle, commit, production,
or Pareto authority and are not a substitute for those downstream gates.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from threading import RLock
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.contracts.hard_mask_actions import (
    HardMask,
    RemovedRouteIndexSet,
)

if TYPE_CHECKING:
    from .modern_v1_candidate_composition import V1PackedRemovalAction


DEVICE_SHELL_BLOCK_SCHEMA: Final = (
    "partition_free_v1_device_shell_compact_block_capacity_v1"
)
DEVICE_SHELL_ACTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_shell_frozen_action_receipt_v1"
)
DEVICE_SHELL_LADDER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_shell_frozen_ladder_receipt_v1"
)
DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN: Final = (
    "partition_free_v1_device_shell_label_commitment_v1"
)
DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN: Final = (
    "partition_free_v1_device_shell_eligibility_commitment_v1"
)
DEVICE_SHELL_MAXIMUM_ENDPOINTS: Final = 255
DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE: Final = 1 << 20

DeviceShellHandleKind = Literal["numpy_reference", "torch_cuda"]
DeviceShellBlockLegalityScope = Literal[
    "registered_compact_capacities", "explicit_unconstrained_reference"
]


class V1DeviceShellBorrowOperation(str, Enum):
    """Closed set of label-native operations allowed to borrow shell labels."""

    FQ_HARD_REPLAY = "label_native_fq_hard_replay"
    C_DIRECTIONAL_CERTIFICATION = "label_native_c_directional_certification"
    BLOCK_SUPPORT_PHYSICAL_ALTERNATE_DOMAIN = "block_support_physical_alternate_domain"


class V1DeviceShellRefusalCode(str, Enum):
    """Typed fail-closed outcomes for the shell identity seam."""

    ENDPOINT_LIMIT_EXCEEDED = "endpoint_limit_exceeded"
    INVALID_LABEL = "invalid_label"
    EMPTY_SHELL = "empty_shell"
    INELIGIBLE_ROUTE = "ineligible_route"
    PARENT_ROUTE_NOT_LIVE = "parent_route_not_live"
    BLOCK_CAPACITY_EXCEEDED = "block_capacity_exceeded"
    MISSING_BLOCK_CAPACITY = "missing_block_capacity"
    HANDLE_EVICTED = "handle_evicted"
    STALE_PARENT = "stale_parent"
    IDENTITY_MISMATCH = "identity_mismatch"
    UNSUPPORTED_HANDLE = "unsupported_handle"
    HANDLE_HAS_ACTIVE_LEASES = "handle_has_active_leases"
    LABEL_LEASE_CLOSED = "label_lease_closed"
    LABEL_LEASE_OPERATION_MISMATCH = "label_lease_operation_mismatch"
    LABEL_STORAGE_MUTATED = "label_storage_mutated"


class V1DeviceShellRefusal(RuntimeError):
    """A shell handle or receipt cannot cross the exact identity seam."""

    def __init__(self, code: V1DeviceShellRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1DeviceShellRefusalCode, message: str) -> NoReturn:
    raise V1DeviceShellRefusal(code, message)


def _nonempty(value: object, name: str) -> str:
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _new_stream_digest(*, domain: str, metadata: object) -> "hashlib._Hash":
    """Create a length-framed, domain-separated streaming SHA-256 state."""

    encoded_domain = domain.encode("ascii")
    encoded_metadata = canonical_json(metadata).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_domain).to_bytes(8, "big", signed=False))
    digest.update(encoded_domain)
    digest.update(len(encoded_metadata).to_bytes(8, "big", signed=False))
    digest.update(encoded_metadata)
    return digest


@dataclass(frozen=True, slots=True)
class V1DeviceShellCompactBlockCapacity:
    """One contiguous registered hard-removal capacity on the physical axis."""

    block_id: str
    start: int
    stop: int
    removal_capacity: int
    schema: str = DEVICE_SHELL_BLOCK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _nonempty(self.block_id, "block_id"))
        start = _nonnegative_int(self.start, "block start")
        stop = _positive_int(self.stop, "block stop")
        capacity = _nonnegative_int(self.removal_capacity, "removal_capacity")
        if stop <= start:
            raise ValueError("block stop must exceed block start")
        if capacity > stop - start:
            raise ValueError("block removal capacity cannot exceed its span")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "removal_capacity", capacity)
        if self.schema != DEVICE_SHELL_BLOCK_SCHEMA:
            raise ValueError("unsupported device-shell block schema")

    @classmethod
    def from_registered_block(
        cls, block: object
    ) -> "V1DeviceShellCompactBlockCapacity":
        """Copy the four compact fields from a registered KKT block object."""

        try:
            return cls(
                block_id=getattr(block, "block_id"),
                start=getattr(block, "start"),
                stop=getattr(block, "stop"),
                removal_capacity=getattr(block, "removal_capacity"),
            )
        except AttributeError as error:
            raise TypeError(
                "registered block must expose block_id/start/stop/removal_capacity"
            ) from error

    @property
    def block_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-device-shell-block-capacity-"
        )


class V1DeviceShellLabelHandle:
    """Ephemeral ownership handle for a contiguous uint8 shell-label vector.

    CPU construction makes an owned immutable reference copy.  CUDA
    construction requires an already-contiguous CUDA ``torch.uint8`` tensor;
    the producing runtime must transfer ownership and must not mutate that
    tensor.  CPU storage is bytes-backed and cannot be reopened for mutation;
    CUDA storage identity and the tensor mutation version are checked on every
    scientific access for the handle's complete lifetime.  Full shell
    materialization additionally replays the registered content commitment.
    Thus mutation between two leases cannot become a newly trusted baseline.
    """

    __slots__ = (
        "_active_lease_tokens",
        "_backend_kind",
        "_device_identity",
        "_evicted",
        "_labels",
        "_lease_lock",
        "_registered_storage_invariant",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        labels: object,
        backend_kind: DeviceShellHandleKind,
        device_identity: str,
    ) -> None:
        self._labels = labels
        self._backend_kind = backend_kind
        self._device_identity = device_identity
        self._evicted = False
        self._active_lease_tokens: set[object] = set()
        self._lease_lock = RLock()
        # The resident object is an ephemeral capability, but its storage and
        # mutation version are frozen for the handle's complete lifetime.  A
        # per-lease witness alone is insufficient: mutation between two leases
        # would otherwise become the next lease's accepted baseline.
        self._registered_storage_invariant = self._storage_invariant_locked(labels)

    @classmethod
    def from_numpy_reference(cls, labels: np.ndarray) -> "V1DeviceShellLabelHandle":
        source = np.asarray(labels)
        if source.ndim != 1:
            raise ValueError("device-shell labels must be one-dimensional")
        if source.dtype != np.dtype(np.uint8):
            raise TypeError("device-shell labels must have exact uint8 dtype")
        if source.size <= 0:
            raise ValueError("device-shell label axis must be nonempty")
        # Back the reference vector by immutable ``bytes``.  An owning NumPy
        # array can be made writeable again with ``setflags(write=True)``, which
        # would defeat a zero-copy lifetime invariant between registered
        # operations.  A bytes-backed view cannot be reopened for mutation.
        owned = np.frombuffer(source.tobytes(order="C"), dtype=np.uint8)
        owned.setflags(write=False)
        return cls(
            labels=owned,
            backend_kind="numpy_reference",
            device_identity="cpu",
        )

    @classmethod
    def from_torch_cuda(cls, labels: object) -> "V1DeviceShellLabelHandle":
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent
            _refuse(
                V1DeviceShellRefusalCode.UNSUPPORTED_HANDLE,
                f"torch is unavailable for a CUDA shell handle: {error}",
            )
        if not isinstance(labels, torch.Tensor):
            raise TypeError("CUDA device-shell labels require a torch.Tensor")
        if labels.device.type != "cuda":
            raise ValueError("production device-shell labels must reside on CUDA")
        if labels.dtype is not torch.uint8:
            raise TypeError("device-shell labels must have exact torch.uint8 dtype")
        if labels.ndim != 1 or int(labels.numel()) <= 0:
            raise ValueError("device-shell labels must be a nonempty vector")
        if not labels.is_contiguous():
            raise ValueError("device-shell labels must be contiguous")
        return cls(
            labels=labels,
            backend_kind="torch_cuda",
            device_identity=str(labels.device),
        )

    def _require_live(self) -> object:
        with self._lease_lock:
            return self._require_live_locked()

    def _require_live_locked(self) -> object:
        labels = self._require_resident_locked()
        if self._storage_invariant_locked(labels) != self._registered_storage_invariant:
            _refuse(
                V1DeviceShellRefusalCode.LABEL_STORAGE_MUTATED,
                "resident label storage changed after handle registration",
            )
        return labels

    def _require_resident_locked(self) -> object:
        """Return resident storage without blessing a poisoned invariant.

        This narrower lifecycle primitive is used only to release a lease or
        evict storage after a mutation refusal.  Scientific consumers always
        go through ``_require_live_locked``.
        """

        if self._evicted or self._labels is None:
            _refuse(
                V1DeviceShellRefusalCode.HANDLE_EVICTED,
                "device-shell labels were evicted",
            )
        return self._labels

    def _route_count_locked(self, labels: object) -> int:
        if self._backend_kind == "numpy_reference":
            return int(np.asarray(labels).size)
        return int(labels.numel())  # type: ignore[union-attr]

    def _storage_invariant_locked(self, labels: object) -> tuple[object, ...]:
        """Return a zero-copy identity/version witness for resident storage."""

        if self._backend_kind == "numpy_reference":
            array = np.asarray(labels)
            return (
                "numpy_reference",
                int(array.__array_interface__["data"][0]),
                array.shape,
                array.strides,
                array.dtype.str,
                bool(array.flags.c_contiguous),
                bool(array.flags.writeable),
            )
        tensor = labels
        return (
            "torch_cuda",
            int(tensor.data_ptr()),  # type: ignore[union-attr]
            int(tensor.untyped_storage().data_ptr()),  # type: ignore[union-attr]
            int(tensor.storage_offset()),  # type: ignore[union-attr]
            tuple(int(value) for value in tensor.shape),  # type: ignore[union-attr]
            tuple(int(value) for value in tensor.stride()),  # type: ignore[union-attr]
            str(tensor.dtype),  # type: ignore[union-attr]
            str(tensor.device),  # type: ignore[union-attr]
            int(tensor._version),  # type: ignore[union-attr]
        )

    @property
    def backend_kind(self) -> DeviceShellHandleKind:
        self._require_live()
        return self._backend_kind

    @property
    def device_identity(self) -> str:
        self._require_live()
        return self._device_identity

    @property
    def route_count(self) -> int:
        labels = self._require_live()
        return self._route_count_locked(labels)

    @property
    def active_lease_count(self) -> int:
        """Number of live, authority-inert borrows preventing eviction."""

        with self._lease_lock:
            self._require_resident_locked()
            return len(self._active_lease_tokens)

    @property
    def is_evicted(self) -> bool:
        return bool(self._evicted)

    def copy_host_chunk(self, start: int, stop: int) -> np.ndarray:
        """Copy one bounded physical-axis slab to immutable host uint8 storage."""

        begin = _nonnegative_int(start, "chunk start")
        end = _positive_int(stop, "chunk stop")
        # Hold the lifecycle lock through the bounded copy/transfer.  Eviction
        # may occur between registered chunks, in which case the next chunk
        # refuses, but it cannot race the resident storage used by one chunk.
        with self._lease_lock:
            labels = self._require_live_locked()
            if end <= begin or end > self._route_count_locked(labels):
                raise ValueError("device-shell chunk lies outside the physical axis")
            if self._backend_kind == "numpy_reference":
                result = np.array(
                    np.asarray(labels)[begin:end],
                    dtype=np.uint8,
                    order="C",
                    copy=True,
                )
            else:  # CUDA, transferred in the registered bounded slab only.
                tensor = (
                    labels[begin:end]
                    .detach()
                    .to(device="cpu", non_blocking=False)  # type: ignore[index]
                )
                result = np.array(tensor.contiguous().numpy(), copy=True)
        result.setflags(write=False)
        return result

    def borrow_for_registered_operation(
        self, *, operation: V1DeviceShellBorrowOperation
    ) -> "V1DeviceShellLabelLease":
        """Borrow the resident label object without copying or transferring it.

        The lease is an ephemeral lifetime guard, not a receipt or an authority
        grant.  Callers must use it as a context manager and may pass the
        resident object only to the operation registered on that lease.
        """

        if type(operation) is not V1DeviceShellBorrowOperation:
            raise TypeError("operation must be an exact V1DeviceShellBorrowOperation")
        with self._lease_lock:
            labels = self._require_live_locked()
            token = object()
            self._active_lease_tokens.add(token)
            try:
                return V1DeviceShellLabelLease(
                    _construction_key=_DEVICE_SHELL_LEASE_CONSTRUCTION_KEY,
                    handle=self,
                    token=token,
                    operation=operation,
                    backend_kind=self._backend_kind,
                    device_identity=self._device_identity,
                    route_count=self._route_count_locked(labels),
                    storage_invariant=self._storage_invariant_locked(labels),
                )
            except BaseException:
                self._active_lease_tokens.remove(token)
                raise

    def _labels_for_live_lease(
        self,
        *,
        token: object,
        expected_storage_invariant: tuple[object, ...],
    ) -> object:
        with self._lease_lock:
            labels = self._require_live_locked()
            if token not in self._active_lease_tokens:
                _refuse(
                    V1DeviceShellRefusalCode.LABEL_LEASE_CLOSED,
                    "device-shell label lease is no longer registered",
                )
            if self._storage_invariant_locked(labels) != expected_storage_invariant:
                _refuse(
                    V1DeviceShellRefusalCode.LABEL_STORAGE_MUTATED,
                    "resident label storage or tensor version changed during a lease",
                )
            return labels

    def _close_live_lease(
        self,
        *,
        token: object,
        expected_storage_invariant: tuple[object, ...],
    ) -> None:
        with self._lease_lock:
            labels = self._require_resident_locked()
            if token not in self._active_lease_tokens:
                _refuse(
                    V1DeviceShellRefusalCode.LABEL_LEASE_CLOSED,
                    "device-shell label lease is no longer registered",
                )
            self._active_lease_tokens.remove(token)
            observed = self._storage_invariant_locked(labels)
            if (
                observed != self._registered_storage_invariant
                or observed != expected_storage_invariant
            ):
                _refuse(
                    V1DeviceShellRefusalCode.LABEL_STORAGE_MUTATED,
                    "resident label storage or tensor version changed during a lease",
                )

    def evict(self) -> None:
        """Irreversibly release the label vector owned by this handle."""

        with self._lease_lock:
            self._require_resident_locked()
            if self._active_lease_tokens:
                _refuse(
                    V1DeviceShellRefusalCode.HANDLE_HAS_ACTIVE_LEASES,
                    "device-shell labels cannot be evicted while a lease is live",
                )
            self._labels = None
            self._evicted = True

    @classmethod
    def evict_all_atomically(
        cls,
        handles: Sequence["V1DeviceShellLabelHandle"],
    ) -> None:
        """Evict an exact handle set without a partial-release failure.

        All lifecycle locks are acquired in stable object-identity order before
        any resident vector is released.  If one handle is stale or borrowed,
        every handle remains live.
        """

        normalized = tuple(handles)
        if not normalized or any(type(handle) is not cls for handle in normalized):
            raise TypeError(
                "atomic device-shell eviction requires exact nonempty handles"
            )
        if len({id(handle) for handle in normalized}) != len(normalized):
            raise ValueError("atomic device-shell eviction handles must be unique")
        ordered = tuple(sorted(normalized, key=id))
        with ExitStack() as stack:
            for handle in ordered:
                stack.enter_context(handle._lease_lock)
            for handle in normalized:
                handle._require_resident_locked()
                if handle._active_lease_tokens:
                    _refuse(
                        V1DeviceShellRefusalCode.HANDLE_HAS_ACTIVE_LEASES,
                        "no device-shell handle was evicted because one remains borrowed",
                    )
            for handle in normalized:
                handle._labels = None
                handle._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("device-shell handles are ephemeral and nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("device-shell handles are ephemeral and nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("device-shell handles are ephemeral and nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device-shell handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("device-shell handles cannot be copied")


_DEVICE_SHELL_LEASE_CONSTRUCTION_KEY: Final = object()


class V1DeviceShellLabelLease:
    """Ephemeral exact-type lifetime guard for one registered label operation.

    This object has no durable representation and grants no workflow
    authority.  Its private accessor returns the handle's existing resident
    object; it never copies NumPy storage and never transfers a CUDA tensor to
    the host.  A consumer must not retain the returned object beyond the
    context-manager lifetime.  Any illicit mutation remains detectable by the
    existing full commitment replay before a frozen action can be used.
    """

    __slots__ = (
        "_backend_kind",
        "_closed",
        "_device_identity",
        "_handle",
        "_operation",
        "_route_count",
        "_storage_invariant",
        "_token",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        handle: V1DeviceShellLabelHandle,
        token: object,
        operation: V1DeviceShellBorrowOperation,
        backend_kind: DeviceShellHandleKind,
        device_identity: str,
        route_count: int,
        storage_invariant: tuple[object, ...],
    ) -> None:
        if _construction_key is not _DEVICE_SHELL_LEASE_CONSTRUCTION_KEY:
            raise TypeError("device-shell label leases must be issued by a handle")
        if type(handle) is not V1DeviceShellLabelHandle:
            raise TypeError("device-shell label lease requires an exact handle")
        if type(operation) is not V1DeviceShellBorrowOperation:
            raise TypeError("operation must be an exact V1DeviceShellBorrowOperation")
        self._handle: V1DeviceShellLabelHandle | None = handle
        self._token: object | None = token
        self._operation = operation
        self._backend_kind = backend_kind
        self._device_identity = device_identity
        self._route_count = _positive_int(route_count, "lease route count")
        self._storage_invariant = tuple(storage_invariant)
        if not self._storage_invariant:
            raise ValueError("lease storage invariant must be nonempty")
        self._closed = False

    def _require_live(self) -> tuple[V1DeviceShellLabelHandle, object, object]:
        handle = self._handle
        token = self._token
        if self._closed or handle is None or token is None:
            _refuse(
                V1DeviceShellRefusalCode.LABEL_LEASE_CLOSED,
                "device-shell label lease is closed",
            )
        labels = handle._labels_for_live_lease(
            token=token,
            expected_storage_invariant=self._storage_invariant,
        )
        return handle, token, labels

    @property
    def operation(self) -> V1DeviceShellBorrowOperation:
        self._require_live()
        return self._operation

    @property
    def backend_kind(self) -> DeviceShellHandleKind:
        self._require_live()
        return self._backend_kind

    @property
    def device_identity(self) -> str:
        self._require_live()
        return self._device_identity

    @property
    def route_count(self) -> int:
        self._require_live()
        return self._route_count

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    def _labels_for_registered_operation(
        self, *, operation: V1DeviceShellBorrowOperation
    ) -> object:
        """Return resident labels only to the exact registered operation."""

        if type(operation) is not V1DeviceShellBorrowOperation:
            raise TypeError("operation must be an exact V1DeviceShellBorrowOperation")
        _, _, labels = self._require_live()
        if operation is not self._operation:
            _refuse(
                V1DeviceShellRefusalCode.LABEL_LEASE_OPERATION_MISMATCH,
                "label access differs from the operation registered on the lease",
            )
        return labels

    def close(self) -> None:
        """Close this lease once; repeated closure is harmless."""

        if self._closed:
            return
        handle = self._handle
        token = self._token
        if handle is None or token is None:  # pragma: no cover - defensive state.
            _refuse(
                V1DeviceShellRefusalCode.LABEL_LEASE_CLOSED,
                "device-shell label lease lost its owner",
            )
        try:
            handle._close_live_lease(
                token=token,
                expected_storage_invariant=self._storage_invariant,
            )
        finally:
            # A mutation refusal must still release the borrow so cleanup and
            # cache eviction cannot deadlock behind a poisoned consumer.
            self._handle = None
            self._token = None
            self._closed = True

    def __enter__(self) -> "V1DeviceShellLabelLease":
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("device-shell label leases are ephemeral and nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("device-shell label leases are ephemeral and nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("device-shell label leases are ephemeral and nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("device-shell label leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("device-shell label leases cannot be copied")


@dataclass(frozen=True, slots=True)
class V1DeviceShellFrozenActionReceipt:
    """Exact cumulative identity of one endpoint in a frozen shell ladder."""

    endpoint_label: int
    shell_count: int
    cumulative_removed_count: int
    removed_route_index_hash: str
    resulting_mask_packed_sha256: str
    resulting_mask_hash: str
    packed_action_hash: str
    schema: str = DEVICE_SHELL_ACTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_label, "endpoint label")
        shell = _positive_int(self.shell_count, "shell count")
        cumulative = _positive_int(
            self.cumulative_removed_count, "cumulative removed count"
        )
        if endpoint > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
            _refuse(
                V1DeviceShellRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
                "an action receipt cannot exceed 255 shell endpoints",
            )
        if shell > cumulative:
            raise ValueError("shell count cannot exceed cumulative removed count")
        object.__setattr__(self, "endpoint_label", endpoint)
        object.__setattr__(self, "shell_count", shell)
        object.__setattr__(self, "cumulative_removed_count", cumulative)
        object.__setattr__(
            self,
            "removed_route_index_hash",
            _nonempty(self.removed_route_index_hash, "removed route index hash"),
        )
        packed_digest = _nonempty(
            self.resulting_mask_packed_sha256,
            "resulting mask packed SHA-256",
        )
        if len(packed_digest) != 64 or packed_digest != packed_digest.lower():
            raise ValueError("resulting mask packed SHA-256 must be a lowercase digest")
        try:
            int(packed_digest, 16)
        except ValueError as error:
            raise ValueError(
                "resulting mask packed SHA-256 must be a lowercase digest"
            ) from error
        object.__setattr__(
            self,
            "resulting_mask_packed_sha256",
            packed_digest,
        )
        object.__setattr__(
            self,
            "resulting_mask_hash",
            _nonempty(self.resulting_mask_hash, "resulting mask hash"),
        )
        object.__setattr__(
            self,
            "packed_action_hash",
            _nonempty(self.packed_action_hash, "packed action hash"),
        )
        if self.schema != DEVICE_SHELL_ACTION_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell action receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-device-shell-action-receipt-"
        )


@dataclass(frozen=True, slots=True)
class V1DeviceShellFrozenLadderReceipt:
    """Array-free host receipt for one exact, nested shell-label ladder."""

    source_ladder_hash: str
    route_axis_hash: str
    parent_mask_hash: str
    route_count: int
    endpoint_count: int
    handle_kind: DeviceShellHandleKind
    device_identity: str
    label_commitment: str
    eligibility_commitment: str
    block_capacity_hash: str
    block_legality_scope: DeviceShellBlockLegalityScope
    block_capacities: tuple[V1DeviceShellCompactBlockCapacity, ...]
    shell_counts: tuple[int, ...]
    cumulative_counts: tuple[int, ...]
    block_prefix_counts: tuple[tuple[int, ...], ...]
    actions: tuple[V1DeviceShellFrozenActionReceipt, ...]
    authority_scope: str = "identity_and_registered_hard_legality_only_v1"
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_SHELL_LADDER_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        from .modern_v1_candidate_composition import V1PackedRemovalAction

        for name in (
            "source_ladder_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "device_identity",
            "label_commitment",
            "eligibility_commitment",
            "block_capacity_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        route_count = _positive_int(self.route_count, "route count")
        endpoint_count = _positive_int(self.endpoint_count, "endpoint count")
        if endpoint_count > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
            _refuse(
                V1DeviceShellRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
                "device-shell ladders support at most 255 endpoints",
            )
        if self.handle_kind not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported device-shell handle kind")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "endpoint_count", endpoint_count)
        blocks = _normalize_blocks(self.block_capacities, route_count=route_count)
        if self.block_legality_scope not in {
            "registered_compact_capacities",
            "explicit_unconstrained_reference",
        }:
            raise ValueError("unsupported device-shell block-legality scope")
        if bool(blocks) is not (
            self.block_legality_scope == "registered_compact_capacities"
        ):
            raise ValueError(
                "device-shell block-legality scope differs from its capacities"
            )
        shell_counts = tuple(
            _positive_int(value, "shell count") for value in self.shell_counts
        )
        cumulative = tuple(
            _positive_int(value, "cumulative count") for value in self.cumulative_counts
        )
        if len(shell_counts) != endpoint_count or len(cumulative) != endpoint_count:
            raise ValueError("shell and cumulative counts must cover every endpoint")
        expected_cumulative = tuple(int(value) for value in np.cumsum(shell_counts))
        if cumulative != expected_cumulative or cumulative[-1] > route_count:
            raise ValueError("device-shell cumulative histogram is inconsistent")
        block_prefix = tuple(
            tuple(_nonnegative_int(value, "block prefix count") for value in row)
            for row in self.block_prefix_counts
        )
        if len(block_prefix) != len(blocks) or any(
            len(row) != endpoint_count for row in block_prefix
        ):
            raise ValueError("block prefix counts do not align with the ladder")
        for block, row in zip(blocks, block_prefix, strict=True):
            if tuple(sorted(row)) != row:
                raise ValueError("block prefix counts must be nondecreasing")
            if row[-1] > block.removal_capacity:
                raise ValueError("block prefix counts exceed registered capacity")
        if (
            blocks
            and tuple(
                sum(row[endpoint] for row in block_prefix)
                for endpoint in range(endpoint_count)
            )
            != cumulative
        ):
            raise ValueError("block prefix counts do not partition cumulative removals")
        expected_block_hash = content_hash(
            blocks, prefix="partition-free-v1-device-shell-block-capacities-"
        )
        if self.block_capacity_hash != expected_block_hash:
            raise ValueError("device-shell block-capacity hash does not replay")
        actions = tuple(self.actions)
        if len(actions) != endpoint_count or any(
            type(action) is not V1DeviceShellFrozenActionReceipt for action in actions
        ):
            raise TypeError("every device-shell endpoint requires an action receipt")
        for endpoint, action in enumerate(actions, start=1):
            if (
                action.endpoint_label != endpoint
                or action.shell_count != shell_counts[endpoint - 1]
                or action.cumulative_removed_count != cumulative[endpoint - 1]
            ):
                raise ValueError(
                    "device-shell action receipt differs from its histogram"
                )
            expected_mask_hash = HardMask.mask_hash_from_packed_digest(
                route_axis_hash=self.route_axis_hash,
                route_count=route_count,
                packed_sha256=action.resulting_mask_packed_sha256,
            )
            expected_action_hash = (
                V1PackedRemovalAction.action_hash_from_streamed_identity(
                    route_axis_hash=self.route_axis_hash,
                    route_count=route_count,
                    resulting_mask_packed_sha256=(action.resulting_mask_packed_sha256),
                    removed_count=action.cumulative_removed_count,
                    removed_route_index_hash=action.removed_route_index_hash,
                )
            )
            if (
                action.resulting_mask_hash != expected_mask_hash
                or action.packed_action_hash != expected_action_hash
            ):
                raise ValueError(
                    "device-shell action streamed identities do not replay"
                )
        authority = (
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in authority):
            raise TypeError("device-shell authority fields must be exact booleans")
        if any(authority):
            raise PermissionError(
                "a device-shell identity receipt has no workflow authority"
            )
        object.__setattr__(self, "block_capacities", blocks)
        object.__setattr__(self, "shell_counts", shell_counts)
        object.__setattr__(self, "cumulative_counts", cumulative)
        object.__setattr__(self, "block_prefix_counts", block_prefix)
        object.__setattr__(self, "actions", actions)
        if self.schema != DEVICE_SHELL_LADDER_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell ladder receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self, prefix="partition-free-v1-device-shell-ladder-receipt-"
        )


@dataclass(frozen=True, slots=True)
class _DeviceShellScanResult:
    receipt: V1DeviceShellFrozenLadderReceipt
    selected_packed_bits: bytes | None


def _normalize_blocks(
    blocks: Sequence[V1DeviceShellCompactBlockCapacity], *, route_count: int
) -> tuple[V1DeviceShellCompactBlockCapacity, ...]:
    normalized = tuple(blocks)
    if any(
        type(block) is not V1DeviceShellCompactBlockCapacity for block in normalized
    ):
        raise TypeError("block capacities require exact device-shell block objects")
    if not normalized:
        return ()
    previous_stop = 0
    ids: set[str] = set()
    for block in normalized:
        if block.start != previous_stop:
            raise ValueError("compact block capacities must be contiguous and ordered")
        if block.stop > route_count:
            raise ValueError("compact block capacity exceeds the physical route axis")
        if block.block_id in ids:
            raise ValueError("compact block IDs must be unique")
        ids.add(block.block_id)
        previous_stop = block.stop
    if previous_stop != route_count:
        raise ValueError("compact block capacities must cover the physical route axis")
    return normalized


def _normalize_chunk_size(value: int) -> int:
    requested = _positive_int(value, "physical chunk size")
    # Packed child bytes must be streamed at byte-aligned physical boundaries.
    return max(8, (requested // 8) * 8)


def _normalize_eligibility(
    eligible_mask: Sequence[bool] | np.ndarray, *, route_count: int
) -> np.ndarray:
    if isinstance(eligible_mask, (str, bytes)):
        raise TypeError("eligible_mask must be a one-dimensional binary axis")
    eligible = np.asarray(eligible_mask)
    if eligible.ndim != 1 or int(eligible.size) != route_count:
        raise ValueError("eligible_mask must cover the physical route axis")
    if eligible.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("eligible_mask must have a binary integer or bool dtype")
    return eligible


def _scan_v1_device_shell_ladder(
    *,
    handle: V1DeviceShellLabelHandle,
    parent_mask: HardMask,
    eligible_mask: Sequence[bool] | np.ndarray,
    block_capacities: Sequence[V1DeviceShellCompactBlockCapacity],
    endpoint_count: int,
    source_ladder_hash: str,
    physical_chunk_size: int,
    selected_endpoint_label: int | None,
    allow_unconstrained_blocks: bool,
) -> _DeviceShellScanResult:
    from .modern_v1_candidate_composition import V1PackedRemovalAction

    if type(handle) is not V1DeviceShellLabelHandle:
        raise TypeError("device-shell scan requires an exact label handle")
    if type(parent_mask) is not HardMask:
        raise TypeError("device-shell scan requires an exact parent HardMask")
    endpoints = _positive_int(endpoint_count, "endpoint count")
    if endpoints > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
        _refuse(
            V1DeviceShellRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
            "device-shell ladders support at most 255 endpoints",
        )
    selected_endpoint: int | None = None
    if selected_endpoint_label is not None:
        selected_endpoint = _positive_int(
            selected_endpoint_label, "selected endpoint label"
        )
        if selected_endpoint > endpoints:
            raise ValueError("selected endpoint is outside the frozen ladder")
    route_count = parent_mask.route_count
    if handle.route_count != route_count:
        raise ValueError("device-shell labels differ from the parent route axis")
    source_hash = _nonempty(source_ladder_hash, "source ladder hash")
    eligible = _normalize_eligibility(eligible_mask, route_count=route_count)
    blocks = _normalize_blocks(block_capacities, route_count=route_count)
    if type(allow_unconstrained_blocks) is not bool:
        raise TypeError("allow_unconstrained_blocks must be an exact bool")
    if not blocks and not allow_unconstrained_blocks:
        _refuse(
            V1DeviceShellRefusalCode.MISSING_BLOCK_CAPACITY,
            "device-shell hard legality requires registered compact capacities; "
            "unconstrained reference use must be explicit",
        )
    block_legality_scope: DeviceShellBlockLegalityScope = (
        "registered_compact_capacities"
        if blocks
        else "explicit_unconstrained_reference"
    )
    chunk_size = _normalize_chunk_size(physical_chunk_size)
    handle_kind = handle.backend_kind
    device_identity = handle.device_identity

    label_digest = _new_stream_digest(
        domain=DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": parent_mask.route_axis_hash,
            "route_count": route_count,
            "endpoint_count": endpoints,
        },
    )
    eligibility_digest = _new_stream_digest(
        domain=DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": parent_mask.route_axis_hash,
            "route_count": route_count,
        },
    )
    index_digests = tuple(hashlib.sha256() for _ in range(endpoints))
    mask_digests = tuple(hashlib.sha256() for _ in range(endpoints))
    shell_counts = np.zeros(endpoints, dtype=np.int64)
    block_shell_counts = np.zeros((len(blocks), endpoints), dtype=np.int64)
    parent_bytes = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
    selected_packed = (
        bytearray(len(parent_mask.packed_bits))
        if selected_endpoint is not None
        else None
    )
    encoded_dtype_code: Literal["u32", "u64"] = (
        "u32" if route_count <= np.iinfo(np.uint32).max else "u64"
    )
    encoded_dtype = np.dtype("<u4" if encoded_dtype_code == "u32" else "<u8")

    for start in range(0, route_count, chunk_size):
        stop = min(route_count, start + chunk_size)
        labels = handle.copy_host_chunk(start, stop)
        if labels.shape != (stop - start,) or labels.dtype != np.dtype(np.uint8):
            _refuse(
                V1DeviceShellRefusalCode.UNSUPPORTED_HANDLE,
                "device-shell handle returned a malformed host slab",
            )
        if labels.size and int(np.max(labels)) > endpoints:
            _refuse(
                V1DeviceShellRefusalCode.INVALID_LABEL,
                "a shell label lies outside 0..endpoint_count",
            )
        label_digest.update(memoryview(labels).cast("B"))

        eligible_source = eligible[start:stop]
        if eligible_source.dtype.kind == "b":
            eligible_slab = np.ascontiguousarray(eligible_source, dtype=bool)
        else:
            if np.any((eligible_source != 0) & (eligible_source != 1)):
                _refuse(
                    V1DeviceShellRefusalCode.INELIGIBLE_ROUTE,
                    "eligible_mask contains a nonbinary value",
                )
            eligible_slab = np.ascontiguousarray(eligible_source != 0)
        eligibility_packed = np.packbits(eligible_slab, bitorder="big")
        eligibility_digest.update(memoryview(eligibility_packed).cast("B"))

        byte_start = start // 8
        byte_stop = byte_start + (labels.size + 7) // 8
        parent_slab = parent_bytes[byte_start:byte_stop]
        parent_live = np.unpackbits(
            parent_slab, bitorder="big", count=int(labels.size)
        ).astype(bool, copy=False)
        selected_any = labels != 0
        if np.any(selected_any & np.logical_not(parent_live)):
            _refuse(
                V1DeviceShellRefusalCode.PARENT_ROUTE_NOT_LIVE,
                "a shell attempts to remove a route absent from its parent",
            )
        if np.any(selected_any & np.logical_not(eligible_slab)):
            _refuse(
                V1DeviceShellRefusalCode.INELIGIBLE_ROUTE,
                "a shell attempts to remove an ineligible route",
            )
        histogram = np.bincount(labels, minlength=endpoints + 1)
        shell_counts += histogram[1 : endpoints + 1]

        for block_index, block in enumerate(blocks):
            overlap_start = max(start, block.start)
            overlap_stop = min(stop, block.stop)
            if overlap_start >= overlap_stop:
                continue
            local_start = overlap_start - start
            local_stop = overlap_stop - start
            block_histogram = np.bincount(
                labels[local_start:local_stop], minlength=endpoints + 1
            )
            block_shell_counts[block_index] += block_histogram[1 : endpoints + 1]

        for endpoint_offset in range(endpoints):
            endpoint_label = endpoint_offset + 1
            included = selected_any & (labels <= endpoint_label)
            relative_indices = np.flatnonzero(included)
            if relative_indices.size:
                physical_indices = np.asarray(
                    relative_indices + start, dtype=encoded_dtype
                )
                index_digests[endpoint_offset].update(
                    memoryview(physical_indices).cast("B")
                )
            clear_bytes = np.packbits(included, bitorder="big")
            child_bytes = np.bitwise_and(parent_slab, np.bitwise_not(clear_bytes))
            mask_digests[endpoint_offset].update(memoryview(child_bytes).cast("B"))
            if selected_endpoint == endpoint_label:
                if selected_packed is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("selected packed-mask buffer was not allocated")
                selected_packed[byte_start:byte_stop] = child_bytes.tobytes()

    shell_counts_tuple = tuple(int(value) for value in shell_counts)
    if any(value <= 0 for value in shell_counts_tuple):
        _refuse(
            V1DeviceShellRefusalCode.EMPTY_SHELL,
            "every registered endpoint must add a nonempty first-inclusion shell",
        )
    cumulative_counts = tuple(int(value) for value in np.cumsum(shell_counts))
    block_prefix_counts = tuple(
        tuple(int(value) for value in np.cumsum(row)) for row in block_shell_counts
    )
    for block, prefix in zip(blocks, block_prefix_counts, strict=True):
        if prefix[-1] > block.removal_capacity:
            _refuse(
                V1DeviceShellRefusalCode.BLOCK_CAPACITY_EXCEEDED,
                f"shell ladder exceeds removal capacity for block {block.block_id!r}",
            )

    actions: list[V1DeviceShellFrozenActionReceipt] = []
    for endpoint_offset, (shell_count, cumulative_count) in enumerate(
        zip(shell_counts_tuple, cumulative_counts, strict=True)
    ):
        packed_sha256 = mask_digests[endpoint_offset].hexdigest()
        index_hash = RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=route_count,
            count=cumulative_count,
            dtype_code=encoded_dtype_code,
            encoded_sha256=index_digests[endpoint_offset].hexdigest(),
        )
        mask_hash = HardMask.mask_hash_from_packed_digest(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=route_count,
            packed_sha256=packed_sha256,
        )
        action_hash = V1PackedRemovalAction.action_hash_from_streamed_identity(
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=route_count,
            resulting_mask_packed_sha256=packed_sha256,
            removed_count=cumulative_count,
            removed_route_index_hash=index_hash,
        )
        actions.append(
            V1DeviceShellFrozenActionReceipt(
                endpoint_label=endpoint_offset + 1,
                shell_count=shell_count,
                cumulative_removed_count=cumulative_count,
                removed_route_index_hash=index_hash,
                resulting_mask_packed_sha256=packed_sha256,
                resulting_mask_hash=mask_hash,
                packed_action_hash=action_hash,
            )
        )

    label_commitment = (
        "partition-free-v1-device-shell-labels-" + label_digest.hexdigest()
    )
    eligibility_commitment = (
        "partition-free-v1-device-shell-eligibility-" + eligibility_digest.hexdigest()
    )
    block_capacity_hash = content_hash(
        blocks, prefix="partition-free-v1-device-shell-block-capacities-"
    )
    receipt = V1DeviceShellFrozenLadderReceipt(
        source_ladder_hash=source_hash,
        route_axis_hash=parent_mask.route_axis_hash,
        parent_mask_hash=parent_mask.mask_hash,
        route_count=route_count,
        endpoint_count=endpoints,
        handle_kind=handle_kind,
        device_identity=device_identity,
        label_commitment=label_commitment,
        eligibility_commitment=eligibility_commitment,
        block_capacity_hash=block_capacity_hash,
        block_legality_scope=block_legality_scope,
        block_capacities=blocks,
        shell_counts=shell_counts_tuple,
        cumulative_counts=cumulative_counts,
        block_prefix_counts=block_prefix_counts,
        actions=tuple(actions),
    )
    return _DeviceShellScanResult(
        receipt=receipt,
        selected_packed_bits=(
            bytes(selected_packed) if selected_packed is not None else None
        ),
    )


def freeze_v1_device_shell_ladder(
    *,
    handle: V1DeviceShellLabelHandle,
    parent_mask: HardMask,
    eligible_mask: Sequence[bool] | np.ndarray,
    block_capacities: Sequence[V1DeviceShellCompactBlockCapacity] = (),
    endpoint_count: int,
    source_ladder_hash: str,
    physical_chunk_size: int = DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE,
    allow_unconstrained_blocks: bool = False,
) -> V1DeviceShellFrozenLadderReceipt:
    """Validate and freeze exact endpoint identities without materializing masks."""

    return _scan_v1_device_shell_ladder(
        handle=handle,
        parent_mask=parent_mask,
        eligible_mask=eligible_mask,
        block_capacities=block_capacities,
        endpoint_count=endpoint_count,
        source_ladder_hash=source_ladder_hash,
        physical_chunk_size=physical_chunk_size,
        selected_endpoint_label=None,
        allow_unconstrained_blocks=allow_unconstrained_blocks,
    ).receipt


def materialize_v1_device_shell_endpoint(
    *,
    handle: V1DeviceShellLabelHandle,
    receipt: V1DeviceShellFrozenLadderReceipt,
    parent_mask: HardMask,
    eligible_mask: Sequence[bool] | np.ndarray,
    endpoint_label: int,
    physical_chunk_size: int = DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE,
) -> V1PackedRemovalAction:
    """Materialize one frozen endpoint and replay every precommitted identity.

    This is an identity bridge only.  Returning the ordinary packed action
    does not grant any downstream certification or commit authority.
    """

    from .modern_v1_candidate_composition import V1PackedRemovalAction

    if type(receipt) is not V1DeviceShellFrozenLadderReceipt:
        raise TypeError("endpoint materialization requires an exact ladder receipt")
    if type(parent_mask) is not HardMask:
        raise TypeError("endpoint materialization requires an exact parent HardMask")
    if (
        parent_mask.mask_hash != receipt.parent_mask_hash
        or parent_mask.route_axis_hash != receipt.route_axis_hash
        or parent_mask.route_count != receipt.route_count
    ):
        _refuse(
            V1DeviceShellRefusalCode.STALE_PARENT,
            "the parent mask differs from the frozen shell receipt",
        )
    endpoint = _positive_int(endpoint_label, "endpoint label")
    if endpoint > receipt.endpoint_count:
        raise ValueError("selected endpoint is outside the frozen ladder")
    replay = _scan_v1_device_shell_ladder(
        handle=handle,
        parent_mask=parent_mask,
        eligible_mask=eligible_mask,
        block_capacities=receipt.block_capacities,
        endpoint_count=receipt.endpoint_count,
        source_ladder_hash=receipt.source_ladder_hash,
        physical_chunk_size=physical_chunk_size,
        selected_endpoint_label=endpoint,
        allow_unconstrained_blocks=(
            receipt.block_legality_scope == "explicit_unconstrained_reference"
        ),
    )
    if replay.receipt != receipt or replay.receipt.receipt_hash != receipt.receipt_hash:
        _refuse(
            V1DeviceShellRefusalCode.IDENTITY_MISMATCH,
            "shell labels, eligibility, legality, or endpoint identities changed",
        )
    if replay.selected_packed_bits is None:  # pragma: no cover - defensive invariant
        raise RuntimeError("selected endpoint did not materialize packed child bytes")
    child = HardMask(
        route_axis_hash=receipt.route_axis_hash,
        route_count=receipt.route_count,
        packed_bits=replay.selected_packed_bits,
    )
    expected = receipt.actions[endpoint - 1]
    if child.mask_hash != expected.resulting_mask_hash:
        _refuse(
            V1DeviceShellRefusalCode.IDENTITY_MISMATCH,
            "materialized child mask differs from its frozen identity",
        )
    action = V1PackedRemovalAction.from_resulting_mask(
        parent_mask=parent_mask,
        resulting_mask=child,
    )
    if (
        action.removed_count != expected.cumulative_removed_count
        or action.removed_route_index_hash != expected.removed_route_index_hash
        or action.resulting_mask.mask_hash != expected.resulting_mask_hash
        or action.action_hash != expected.packed_action_hash
    ):
        _refuse(
            V1DeviceShellRefusalCode.IDENTITY_MISMATCH,
            "materialized packed action differs from its frozen endpoint receipt",
        )
    return action


__all__ = [
    "DEVICE_SHELL_ACTION_RECEIPT_SCHEMA",
    "DEVICE_SHELL_BLOCK_SCHEMA",
    "DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE",
    "DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN",
    "DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN",
    "DEVICE_SHELL_LADDER_RECEIPT_SCHEMA",
    "DEVICE_SHELL_MAXIMUM_ENDPOINTS",
    "DeviceShellBlockLegalityScope",
    "DeviceShellHandleKind",
    "V1DeviceShellBorrowOperation",
    "V1DeviceShellCompactBlockCapacity",
    "V1DeviceShellFrozenActionReceipt",
    "V1DeviceShellFrozenLadderReceipt",
    "V1DeviceShellLabelHandle",
    "V1DeviceShellLabelLease",
    "V1DeviceShellRefusal",
    "V1DeviceShellRefusalCode",
    "freeze_v1_device_shell_ladder",
    "materialize_v1_device_shell_endpoint",
]
