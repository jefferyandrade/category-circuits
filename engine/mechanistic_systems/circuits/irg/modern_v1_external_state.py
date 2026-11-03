"""Content-addressed immutable storage for route-scale modern-v1 arrays.

The runtime checkpoint owns only :class:`V1ExternalStateDescriptor`.  Array
bytes live below an explicitly supplied durable root, but neither that root nor
any absolute path participates in the logical identity.  A store can therefore
be relocated byte-for-byte without changing a checkpoint's scientific state.

Publication is fail-closed and write-once.  Arrays are streamed into canonical
C-order, little-endian raw files and every file is fsynced.  A completed state
uses an atomic no-replace directory rename where the filesystem supports one.
The portable path claims the content-addressed directory with ``mkdir``, links
each immutable array without replacement under the per-manifest lock, and
links the canonical manifest last as the atomic commit marker.  Opening first
verifies that strict manifest and every file through no-follow descriptors,
then creates read-only mappings from those already verified descriptors.  No
pickle-like codec is used anywhere in this module.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Iterable, Mapping, NoReturn, Sequence, cast

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash


_DESCRIPTOR_SCHEMA: Final = "modern_v1_external_state_descriptor_v1"
_MANIFEST_SCHEMA: Final = "modern_v1_external_state_manifest_v1"
_LOGICAL_IDENTITY_SCHEMA: Final = "modern_v1_external_state_logical_identity_v1"
_ARRAY_IDENTITY_SCHEMA: Final = "modern_v1_external_array_identity_v1"
_INSPECTION_SCHEMA: Final = "modern_v1_external_state_inspection_v1"
_MANIFEST_FILE: Final = "manifest.json"
_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024
_ARRAY_STREAM_BUFFER_ELEMENTS: Final = 1024 * 1024
_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_SAFE_LOGICAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")
_SAFE_TOKEN = re.compile(r"[^\x00]{1,1024}\Z")
_ARRAY_KINDS: Final = frozenset("biufc")


class V1ExternalStateRefusalCode(str, Enum):
    """Stable refusal categories for external-state publication and opening."""

    INVALID_SCHEMA = "invalid_schema"
    INVALID_ARRAY = "invalid_array"
    NONFINITE_ARRAY = "nonfinite_array"
    DUPLICATE_LOGICAL_NAME = "duplicate_logical_name"
    PATH_TRAVERSAL = "path_traversal"
    ROOT_INVALID = "root_invalid"
    SYMLINK_REFUSED = "symlink_refused"
    STATE_MISSING = "state_missing"
    STATE_PARTIAL = "state_partial"
    MANIFEST_INVALID = "manifest_invalid"
    FILE_INVALID = "file_invalid"
    SIZE_MISMATCH = "size_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    ARTIFACT_CHANGED_DURING_INSPECTION = "artifact_changed_during_inspection"
    LIVE_ATTACHMENT_INVALID = "live_attachment_invalid"
    PUBLICATION_FAILED = "publication_failed"


class V1ExternalStateRefusal(ValueError):
    """Typed fail-closed external-state error."""

    def __init__(
        self,
        code: V1ExternalStateRefusalCode,
        message: str,
        *,
        artifact: str | os.PathLike[str] | None = None,
    ) -> None:
        self.code = code
        self.artifact = None if artifact is None else str(artifact)
        suffix = "" if self.artifact is None else f" [{self.artifact}]"
        super().__init__(f"{code.value}: {message}{suffix}")


def _refuse(
    code: V1ExternalStateRefusalCode,
    message: str,
    *,
    artifact: str | os.PathLike[str] | None = None,
) -> NoReturn:
    raise V1ExternalStateRefusal(code, message, artifact=artifact)


def _exact_token(value: object, label: str) -> str:
    if type(value) is not str or _SAFE_TOKEN.fullmatch(cast(str, value)) is None:
        _refuse(
            V1ExternalStateRefusalCode.INVALID_SCHEMA,
            f"{label} must be a nonempty string without NUL bytes",
        )
    return cast(str, value)


def _logical_name(value: object) -> str:
    if type(value) is not str or _SAFE_LOGICAL_NAME.fullmatch(cast(str, value)) is None:
        _refuse(
            V1ExternalStateRefusalCode.PATH_TRAVERSAL,
            "logical array names must use only ASCII letters, digits, '.', '-', and '_'",
        )
    name = cast(str, value)
    if name in {".", ".."}:
        _refuse(V1ExternalStateRefusalCode.PATH_TRAVERSAL, "path-like logical name")
    return name


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(cast(str, value)) != 64
        or any(char not in _LOWER_SHA256 for char in cast(str, value))
    ):
        _refuse(
            V1ExternalStateRefusalCode.INVALID_SCHEMA,
            f"{label} must be a lowercase SHA-256 digest",
        )
    return cast(str, value)


def _canonical_dtype(value: object) -> np.dtype:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        _refuse(V1ExternalStateRefusalCode.INVALID_ARRAY, f"invalid dtype: {exc}")
    if dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in _ARRAY_KINDS:
        _refuse(
            V1ExternalStateRefusalCode.INVALID_ARRAY,
            "only unstructured boolean, integer, floating, and complex arrays are supported",
        )
    if dtype.hasobject:
        _refuse(V1ExternalStateRefusalCode.INVALID_ARRAY, "object arrays are forbidden")
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    return dtype


def _validated_canonical_dtype(value: object) -> np.dtype:
    dtype = _canonical_dtype(value)
    if type(value) is str and cast(str, value) != dtype.str:
        _refuse(
            V1ExternalStateRefusalCode.INVALID_SCHEMA,
            f"dtype must already be canonical little-endian form {dtype.str!r}",
        )
    return dtype


def _shape(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "shape must be an exact tuple")
    shape = cast(tuple[object, ...], value)
    if any(type(item) is not int or cast(int, item) < 0 for item in shape):
        _refuse(
            V1ExternalStateRefusalCode.INVALID_SCHEMA,
            "shape dimensions must be exact nonnegative integers",
        )
    return cast(tuple[int, ...], shape)


def _element_count(shape: tuple[int, ...]) -> int:
    return math.prod(shape, start=1)


def _array_identity_payload(identity: "V1ExternalArrayIdentity") -> dict[str, object]:
    return {
        "schema": _ARRAY_IDENTITY_SCHEMA,
        "logical_name": identity.logical_name,
        "dtype": identity.dtype,
        "shape": list(identity.shape),
        "byte_count": identity.byte_count,
        "sha256": identity.sha256,
        "allow_nonfinite": identity.allow_nonfinite,
    }


@dataclass(frozen=True, slots=True)
class V1ExternalArrayPolicy:
    """Caller-declared schema restrictions for one logical array."""

    logical_name: str
    expected_dtype: str | None = None
    expected_shape: tuple[int, ...] | None = None
    allow_nonfinite: bool = False

    def __post_init__(self) -> None:
        _logical_name(self.logical_name)
        if self.expected_dtype is not None:
            object.__setattr__(self, "expected_dtype", _canonical_dtype(self.expected_dtype).str)
        if self.expected_shape is not None:
            _shape(self.expected_shape)
        if type(self.allow_nonfinite) is not bool:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "allow_nonfinite must be an exact boolean",
            )


@dataclass(frozen=True, slots=True)
class V1ExternalArrayIdentity:
    """Physical-location-independent identity of one canonical array."""

    logical_name: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    sha256: str
    allow_nonfinite: bool = False

    def __post_init__(self) -> None:
        _logical_name(self.logical_name)
        dtype = _validated_canonical_dtype(self.dtype)
        shape = _shape(self.shape)
        if type(self.byte_count) is not int or self.byte_count < 0:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "byte_count must be an exact nonnegative integer",
            )
        expected = _element_count(shape) * dtype.itemsize
        if self.byte_count != expected:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                f"byte_count {self.byte_count} does not match dtype/shape size {expected}",
            )
        _sha256(self.sha256, "array sha256")
        if type(self.allow_nonfinite) is not bool:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "allow_nonfinite must be an exact boolean",
            )

    @property
    def identity_hash(self) -> str:
        return content_hash(_array_identity_payload(self))

    def to_json_dict(self) -> dict[str, object]:
        return _array_identity_payload(self)

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ExternalArrayIdentity":
        if type(value) is not dict:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "array identity must be an object")
        payload = cast(dict[object, object], value)
        expected = {
            "schema",
            "logical_name",
            "dtype",
            "shape",
            "byte_count",
            "sha256",
            "allow_nonfinite",
        }
        if set(payload) != expected or payload.get("schema") != _ARRAY_IDENTITY_SCHEMA:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "array identity fields or schema are not exact",
            )
        raw_shape = payload["shape"]
        if type(raw_shape) is not list:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "JSON shape must be a list")
        return cls(
            logical_name=cast(str, payload["logical_name"]),
            dtype=cast(str, payload["dtype"]),
            shape=tuple(cast(list[object], raw_shape)),
            byte_count=cast(int, payload["byte_count"]),
            sha256=cast(str, payload["sha256"]),
            allow_nonfinite=cast(bool, payload["allow_nonfinite"]),
        )


@dataclass(frozen=True, slots=True)
class V1ExternalArrayManifestEntry:
    """Strict manifest entry binding a logical identity to a safe file name."""

    identity: V1ExternalArrayIdentity
    file_name: str

    def __post_init__(self) -> None:
        if type(self.identity) is not V1ExternalArrayIdentity:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "invalid array identity type")
        # The ordinal prefix is checked by the enclosing manifest, where the
        # canonical ordering is available.
        suffix = f"-{self.identity.sha256}.bin"
        if (
            type(self.file_name) is not str
            or not self.file_name.startswith("array-")
            or not self.file_name.endswith(suffix)
            or "/" in self.file_name
            or "\\" in self.file_name
            or self.file_name in {".", ".."}
        ):
            _refuse(V1ExternalStateRefusalCode.PATH_TRAVERSAL, "unsafe array file name")

    def to_json_dict(self) -> dict[str, object]:
        result = self.identity.to_json_dict()
        result["file_name"] = self.file_name
        return result

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ExternalArrayManifestEntry":
        if type(value) is not dict:
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "array entry must be an object")
        payload = cast(dict[object, object], value)
        expected = {
            "schema",
            "logical_name",
            "dtype",
            "shape",
            "byte_count",
            "sha256",
            "allow_nonfinite",
            "file_name",
        }
        if set(payload) != expected:
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "array manifest entry fields are not exact",
            )
        identity_payload = {key: payload[key] for key in expected - {"file_name"}}
        return cls(
            identity=V1ExternalArrayIdentity.from_json_dict(identity_payload),
            file_name=cast(str, payload["file_name"]),
        )


def _logical_identity_payload(
    namespace: str,
    state_schema: str,
    arrays: Sequence[V1ExternalArrayIdentity],
) -> dict[str, object]:
    return {
        "schema": _LOGICAL_IDENTITY_SCHEMA,
        "namespace": namespace,
        "state_schema": state_schema,
        "arrays": [identity.to_json_dict() for identity in arrays],
    }


def _array_file_name(identity: V1ExternalArrayIdentity, index: int) -> str:
    return f"array-{index:08d}-{identity.sha256}.bin"


def _manifest_payload(
    namespace: str,
    state_schema: str,
    logical_identity_hash: str,
    entries: Sequence[V1ExternalArrayManifestEntry],
) -> dict[str, object]:
    return {
        "schema": _MANIFEST_SCHEMA,
        "namespace": namespace,
        "state_schema": state_schema,
        "logical_identity_hash": logical_identity_hash,
        "arrays": [entry.to_json_dict() for entry in entries],
    }


@dataclass(frozen=True, slots=True)
class V1ExternalStateManifest:
    """Strict, self-verifying manifest stored beside immutable array files."""

    namespace: str
    state_schema: str
    logical_identity_hash: str
    arrays: tuple[V1ExternalArrayManifestEntry, ...]
    manifest_hash: str
    schema: str = _MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _MANIFEST_SCHEMA:
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "wrong manifest schema")
        _exact_token(self.namespace, "namespace")
        _exact_token(self.state_schema, "state_schema")
        _sha256(self.logical_identity_hash, "logical_identity_hash")
        _sha256(self.manifest_hash, "manifest_hash")
        if type(self.arrays) is not tuple or any(
            type(entry) is not V1ExternalArrayManifestEntry for entry in self.arrays
        ):
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "arrays must be an exact tuple")
        names = tuple(entry.identity.logical_name for entry in self.arrays)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            _refuse(
                V1ExternalStateRefusalCode.DUPLICATE_LOGICAL_NAME,
                "manifest arrays must have unique names in canonical order",
            )
        for index, entry in enumerate(self.arrays):
            if entry.file_name != _array_file_name(entry.identity, index):
                _refuse(
                    V1ExternalStateRefusalCode.MANIFEST_INVALID,
                    "array file name does not match canonical identity and ordinal",
                )
        identities = tuple(entry.identity for entry in self.arrays)
        expected_logical = content_hash(
            _logical_identity_payload(self.namespace, self.state_schema, identities)
        )
        if self.logical_identity_hash != expected_logical:
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "logical identity hash does not replay",
            )
        expected_manifest = content_hash(
            _manifest_payload(
                self.namespace,
                self.state_schema,
                self.logical_identity_hash,
                self.arrays,
            )
        )
        if self.manifest_hash != expected_manifest:
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "manifest hash does not replay")

    def to_json_dict(self) -> dict[str, object]:
        result = _manifest_payload(
            self.namespace,
            self.state_schema,
            self.logical_identity_hash,
            self.arrays,
        )
        result["manifest_hash"] = self.manifest_hash
        return result

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ExternalStateManifest":
        if type(value) is not dict:
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "manifest must be an object")
        payload = cast(dict[object, object], value)
        expected = {
            "schema",
            "namespace",
            "state_schema",
            "logical_identity_hash",
            "arrays",
            "manifest_hash",
        }
        if set(payload) != expected or payload.get("schema") != _MANIFEST_SCHEMA:
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "manifest fields or schema are not exact",
            )
        raw_arrays = payload["arrays"]
        if type(raw_arrays) is not list:
            _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, "manifest arrays must be a list")
        return cls(
            namespace=cast(str, payload["namespace"]),
            state_schema=cast(str, payload["state_schema"]),
            logical_identity_hash=cast(str, payload["logical_identity_hash"]),
            arrays=tuple(
                V1ExternalArrayManifestEntry.from_json_dict(item)
                for item in cast(list[object], raw_arrays)
            ),
            manifest_hash=cast(str, payload["manifest_hash"]),
            schema=cast(str, payload["schema"]),
        )


@dataclass(frozen=True, slots=True)
class V1ExternalStateDescriptor:
    """Checkpoint-safe descriptor with no physical root or path."""

    namespace: str
    state_schema: str
    arrays: tuple[V1ExternalArrayIdentity, ...]
    logical_identity_hash: str
    manifest_hash: str
    schema: str = _DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _DESCRIPTOR_SCHEMA:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "wrong descriptor schema")
        _exact_token(self.namespace, "namespace")
        _exact_token(self.state_schema, "state_schema")
        if type(self.arrays) is not tuple or any(
            type(identity) is not V1ExternalArrayIdentity for identity in self.arrays
        ):
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "arrays must be an exact tuple")
        names = tuple(identity.logical_name for identity in self.arrays)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            _refuse(
                V1ExternalStateRefusalCode.DUPLICATE_LOGICAL_NAME,
                "descriptor arrays must have unique names in canonical order",
            )
        _sha256(self.logical_identity_hash, "logical_identity_hash")
        _sha256(self.manifest_hash, "manifest_hash")
        expected_logical = content_hash(
            _logical_identity_payload(self.namespace, self.state_schema, self.arrays)
        )
        if self.logical_identity_hash != expected_logical:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "descriptor logical identity hash does not replay",
            )
        expected_manifest = _manifest_from_descriptor_fields(
            self.namespace,
            self.state_schema,
            self.arrays,
            self.logical_identity_hash,
        )
        if self.manifest_hash != expected_manifest.manifest_hash:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "descriptor manifest hash does not replay",
            )

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self.to_json_dict())

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "state_schema": self.state_schema,
            "arrays": [identity.to_json_dict() for identity in self.arrays],
            "logical_identity_hash": self.logical_identity_hash,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> "V1ExternalStateDescriptor":
        if type(value) is not dict:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "descriptor must be an object")
        payload = cast(dict[object, object], value)
        expected = {
            "schema",
            "namespace",
            "state_schema",
            "arrays",
            "logical_identity_hash",
            "manifest_hash",
        }
        if set(payload) != expected or payload.get("schema") != _DESCRIPTOR_SCHEMA:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_SCHEMA,
                "descriptor fields or schema are not exact",
            )
        raw_arrays = payload["arrays"]
        if type(raw_arrays) is not list:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "descriptor arrays must be a list")
        return cls(
            namespace=cast(str, payload["namespace"]),
            state_schema=cast(str, payload["state_schema"]),
            arrays=tuple(
                V1ExternalArrayIdentity.from_json_dict(item)
                for item in cast(list[object], raw_arrays)
            ),
            logical_identity_hash=cast(str, payload["logical_identity_hash"]),
            manifest_hash=cast(str, payload["manifest_hash"]),
            schema=cast(str, payload["schema"]),
        )


def _manifest_from_descriptor_fields(
    namespace: str,
    state_schema: str,
    identities: Sequence[V1ExternalArrayIdentity],
    logical_identity_hash: str,
) -> V1ExternalStateManifest:
    entries = tuple(
        V1ExternalArrayManifestEntry(
            identity=identity,
            file_name=_array_file_name(identity, index),
        )
        for index, identity in enumerate(identities)
    )
    manifest_hash = content_hash(
        _manifest_payload(namespace, state_schema, logical_identity_hash, entries)
    )
    return V1ExternalStateManifest(
        namespace=namespace,
        state_schema=state_schema,
        logical_identity_hash=logical_identity_hash,
        arrays=entries,
        manifest_hash=manifest_hash,
    )


@dataclass(frozen=True, slots=True)
class V1ExternalStateInspectionReceipt:
    """Content-addressed receipt for a completed all-file inspection barrier."""

    descriptor_hash: str
    manifest_hash: str
    array_identity_hashes: tuple[str, ...]
    all_files_verified: bool = True
    mmap_read_only: bool = True
    schema: str = _INSPECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _INSPECTION_SCHEMA:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "wrong inspection schema")
        _sha256(self.descriptor_hash, "descriptor_hash")
        _sha256(self.manifest_hash, "manifest_hash")
        if type(self.array_identity_hashes) is not tuple:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "identity hashes must be a tuple")
        for value in self.array_identity_hashes:
            _sha256(value, "array identity hash")
        if self.all_files_verified is not True or self.mmap_read_only is not True:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "inspection receipt is not closed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class V1ExternalStateView:
    """Verified read-only mappings and the barrier receipt that admitted them."""

    descriptor: V1ExternalStateDescriptor
    arrays: Mapping[str, np.ndarray]
    inspection_receipt: V1ExternalStateInspectionReceipt

    def close(self) -> None:
        for value in self.arrays.values():
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def __enter__(self) -> "V1ExternalStateView":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _V1ExternalStateMemberWitness:
    """One retained no-follow member descriptor and its immutable identity."""

    name: str
    fd: int
    signature: tuple[int, int, int, int, int]


_LIVE_ATTACHMENT_CONSTRUCTION_TOKEN: Final = object()


class V1ExternalStateLiveAttachment:
    """PID-bound live proof that one exact immutable publication is still present.

    The attachment is deliberately absent from every checkpoint and JSON
    surface.  It retains no-follow descriptors for the durable root, state
    directory, manifest, and every array.  A hot verification performs only
    O(file-count) directory/member/stat comparisons; it never substitutes for
    the exact digest-and-schema barrier required after a restart or cache miss.
    """

    __slots__ = (
        "_root",
        "_state_name",
        "_descriptor_hash",
        "_pid",
        "_root_fd",
        "_directory_fd",
        "_directory_signature",
        "_members",
        "_expected_names",
        "_closed",
        "_lock",
    )

    def __init__(
        self,
        *,
        root: Path,
        state_name: str,
        descriptor_hash: str,
        root_fd: int,
        directory_fd: int,
        directory_signature: tuple[int, int, int, int, int],
        members: tuple[_V1ExternalStateMemberWitness, ...],
        token: object,
    ) -> None:
        if token is not _LIVE_ATTACHMENT_CONSTRUCTION_TOKEN:
            raise TypeError("live external-state attachments are runtime-issued only")
        self._root = root
        self._state_name = state_name
        self._descriptor_hash = descriptor_hash
        self._pid = os.getpid()
        self._root_fd = root_fd
        self._directory_fd = directory_fd
        self._directory_signature = directory_signature
        self._members = members
        self._expected_names = frozenset(member.name for member in members)
        self._closed = False
        self._lock = threading.RLock()

    @property
    def descriptor_hash(self) -> str:
        return self._descriptor_hash

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def file_count(self) -> int:
        return len(self._members)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _refuse_invalid(self, message: str, *, artifact: Path | None = None) -> NoReturn:
        _refuse(
            V1ExternalStateRefusalCode.LIVE_ATTACHMENT_INVALID,
            message,
            artifact=self._root if artifact is None else artifact,
        )

    def verify(self, descriptor: V1ExternalStateDescriptor) -> None:
        """Recheck retained/path identities and exact immutable modes, without I/O hashing."""

        # A retained descriptor is a process-local capability.  Serializing
        # verification against close prevents a descriptor-number reuse race:
        # no thread may close these FDs while another thread is proving their
        # inode/mode/stat witnesses.
        with self._lock:
            self._verify_locked(descriptor)

    def _verify_locked(self, descriptor: V1ExternalStateDescriptor) -> None:
        """Verify while ``_lock`` excludes descriptor teardown."""

        if self._closed:
            self._refuse_invalid("live external-state attachment is closed")
        if os.getpid() != self._pid:
            self._refuse_invalid("live external-state attachment belongs to another PID")
        if type(descriptor) is not V1ExternalStateDescriptor:
            self._refuse_invalid("live attachment requires its exact descriptor type")
        strict_descriptor = V1ExternalStateDescriptor.from_json_dict(
            descriptor.to_json_dict()
        )
        if strict_descriptor.descriptor_hash != self._descriptor_hash:
            self._refuse_invalid("live attachment names another external descriptor")

        try:
            root_path_stat = os.lstat(self._root)
            root_fd_stat = os.fstat(self._root_fd)
        except (FileNotFoundError, OSError) as exc:
            self._refuse_invalid(f"durable root is unavailable: {exc}")
        if stat.S_ISLNK(root_path_stat.st_mode) or not stat.S_ISDIR(
            root_path_stat.st_mode
        ):
            self._refuse_invalid("durable root is no longer a real directory")
        if not stat.S_ISDIR(root_fd_stat.st_mode) or (
            root_path_stat.st_dev,
            root_path_stat.st_ino,
        ) != (root_fd_stat.st_dev, root_fd_stat.st_ino):
            self._refuse_invalid("durable root path no longer names the retained inode")

        state_path = self._root / self._state_name
        try:
            directory_path_stat = os.stat(
                self._state_name,
                dir_fd=self._root_fd,
                follow_symlinks=False,
            )
            directory_fd_stat = os.fstat(self._directory_fd)
        except (FileNotFoundError, OSError) as exc:
            self._refuse_invalid(
                f"durable publication directory is unavailable: {exc}",
                artifact=state_path,
            )
        if stat.S_ISLNK(directory_path_stat.st_mode) or not stat.S_ISDIR(
            directory_path_stat.st_mode
        ):
            self._refuse_invalid(
                "durable publication is no longer a real directory",
                artifact=state_path,
            )
        if (
            not stat.S_ISDIR(directory_fd_stat.st_mode)
            or stat.S_IMODE(directory_path_stat.st_mode) != 0o555
            or stat.S_IMODE(directory_fd_stat.st_mode) != 0o555
            or _stat_signature(directory_path_stat) != self._directory_signature
            or _stat_signature(directory_fd_stat) != self._directory_signature
        ):
            self._refuse_invalid(
                "durable publication directory identity or immutable mode changed",
                artifact=state_path,
            )
        try:
            observed_names = frozenset(os.listdir(self._directory_fd))
        except OSError as exc:
            self._refuse_invalid(
                f"cannot enumerate retained publication: {exc}",
                artifact=state_path,
            )
        if observed_names != self._expected_names:
            self._refuse_invalid(
                "durable publication members changed",
                artifact=state_path,
            )

        for member in self._members:
            member_path = state_path / member.name
            try:
                path_stat = os.stat(
                    member.name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
                fd_stat = os.fstat(member.fd)
            except (FileNotFoundError, OSError) as exc:
                self._refuse_invalid(
                    f"durable publication member is unavailable: {exc}",
                    artifact=member_path,
                )
            if (
                stat.S_ISLNK(path_stat.st_mode)
                or not stat.S_ISREG(path_stat.st_mode)
                or not stat.S_ISREG(fd_stat.st_mode)
                or stat.S_IMODE(path_stat.st_mode) != 0o444
                or stat.S_IMODE(fd_stat.st_mode) != 0o444
                or _stat_signature(path_stat) != member.signature
                or _stat_signature(fd_stat) != member.signature
            ):
                self._refuse_invalid(
                    "durable publication member identity or immutable mode changed",
                    artifact=member_path,
                )

    def close(self) -> None:
        """Release retained descriptors; repeated closes are harmless."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            for member in self._members:
                try:
                    os.close(member.fd)
                except OSError:
                    pass
            for fd in (self._directory_fd, self._root_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def __enter__(self) -> "V1ExternalStateLiveAttachment":
        with self._lock:
            if self._closed:
                self._refuse_invalid("live external-state attachment is closed")
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("live external-state attachments cannot be serialized")

    def __copy__(self) -> NoReturn:
        raise TypeError("live external-state attachments cannot be copied")

    def __deepcopy__(self, _memo: object) -> NoReturn:
        raise TypeError("live external-state attachments cannot be copied")

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup.
        try:
            self.close()
        except BaseException:
            pass


def _state_directory_name(manifest_hash: str) -> str:
    return f"state-{_sha256(manifest_hash, 'manifest_hash')}"


def _lstat_no_symlink(path: Path, *, require_directory: bool) -> os.stat_result:
    try:
        result = os.lstat(path)
    except FileNotFoundError:
        _refuse(V1ExternalStateRefusalCode.STATE_MISSING, "artifact is missing", artifact=path)
    if stat.S_ISLNK(result.st_mode):
        _refuse(V1ExternalStateRefusalCode.SYMLINK_REFUSED, "symlink refused", artifact=path)
    if require_directory and not stat.S_ISDIR(result.st_mode):
        _refuse(V1ExternalStateRefusalCode.ROOT_INVALID, "expected directory", artifact=path)
    if not require_directory and not stat.S_ISREG(result.st_mode):
        _refuse(V1ExternalStateRefusalCode.FILE_INVALID, "expected regular file", artifact=path)
    return result


def _prepare_root(root: str | os.PathLike[str], *, create: bool) -> Path:
    path = Path(root)
    if path.name in {"", ".", ".."} and not path.is_absolute():
        _refuse(V1ExternalStateRefusalCode.ROOT_INVALID, "root is not specific", artifact=path)
    if path.is_absolute() and path == Path(path.anchor):
        _refuse(
            V1ExternalStateRefusalCode.ROOT_INVALID,
            "the filesystem root is too broad for an external-state store",
            artifact=path,
        )
    if create:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _refuse(
                V1ExternalStateRefusalCode.ROOT_INVALID,
                f"cannot create durable root: {exc}",
                artifact=path,
            )
    try:
        result = os.lstat(path)
    except FileNotFoundError:
        _refuse(V1ExternalStateRefusalCode.ROOT_INVALID, "durable root is missing", artifact=path)
    if stat.S_ISLNK(result.st_mode):
        _refuse(V1ExternalStateRefusalCode.SYMLINK_REFUSED, "root symlink refused", artifact=path)
    if not stat.S_ISDIR(result.st_mode):
        _refuse(V1ExternalStateRefusalCode.ROOT_INVALID, "root is not a directory", artifact=path)
    return path


def _write_all(fd: int, data: memoryview | bytes) -> None:
    view = memoryview(data).cast("B")
    offset = 0
    while offset < len(view):
        written = os.write(fd, view[offset:])
        if written <= 0:
            raise OSError("short write while publishing external state")
        offset += written


def _open_new(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _stream_array_to_file(
    path: Path,
    value: np.ndarray,
    *,
    allow_nonfinite: bool,
) -> tuple[str, str, tuple[int, ...], int]:
    if not isinstance(value, np.ndarray):
        _refuse(V1ExternalStateRefusalCode.INVALID_ARRAY, "values must be NumPy arrays")
    if value.dtype.hasobject:
        _refuse(V1ExternalStateRefusalCode.INVALID_ARRAY, "object arrays are forbidden")
    dtype = _canonical_dtype(value.dtype)
    shape = tuple(int(item) for item in value.shape)
    expected_bytes = _element_count(shape) * dtype.itemsize
    digest = hashlib.sha256()
    fd = _open_new(path)
    written_bytes = 0
    try:
        iterator = np.nditer(
            value,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            op_dtypes=[dtype],
            casting="equiv",
            order="C",
            buffersize=_ARRAY_STREAM_BUFFER_ELEMENTS,
        )
        for raw_chunk in iterator:
            if dtype.kind == "b":
                # NumPy can expose nonzero bytes other than 0x01 through a
                # bool view.  Normalize the logical truth values, rather than
                # preserving those noncanonical storage bytes.
                chunk = np.ascontiguousarray(np.not_equal(raw_chunk, False), dtype=dtype)
            else:
                chunk = np.ascontiguousarray(raw_chunk, dtype=dtype)
            if not allow_nonfinite and dtype.kind in "fc" and not bool(np.isfinite(chunk).all()):
                _refuse(
                    V1ExternalStateRefusalCode.NONFINITE_ARRAY,
                    "nonfinite values require an explicit allow_nonfinite schema policy",
                    artifact=path,
                )
            bytes_view = memoryview(chunk).cast("B")
            digest.update(bytes_view)
            _write_all(fd, bytes_view)
            written_bytes += len(bytes_view)
        if written_bytes != expected_bytes:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                f"canonical stream produced {written_bytes} bytes, expected {expected_bytes}",
                artifact=path,
            )
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    return dtype.str, shape, expected_bytes, digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")


def _write_fsynced_file(path: Path, payload: bytes) -> None:
    fd = _open_new(path)
    try:
        _write_all(fd, payload)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _cleanup_stage(path: Path) -> None:
    try:
        os.chmod(path, 0o700)
    except FileNotFoundError:
        return
    shutil.rmtree(path)


class _PublicationLock:
    def __init__(self, root: Path, manifest_hash: str) -> None:
        lock_root = root / ".publication-locks"
        try:
            lock_root.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                f"cannot create publication lock directory: {exc}",
                artifact=lock_root,
            )
        result = os.lstat(lock_root)
        if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "publication lock directory is not a real directory",
                artifact=lock_root,
            )
        self._path = lock_root / f"{manifest_hash}.lock"
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(self._path, flags, 0o600)
        except OSError as exc:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                f"cannot open publication lock: {exc}",
                artifact=self._path,
            )
        if not stat.S_ISREG(os.fstat(self._fd).st_mode):
            os.close(self._fd)
            _refuse(
                V1ExternalStateRefusalCode.FILE_INVALID,
                "publication lock is not a regular file",
                artifact=self._path,
            )

    def __enter__(self) -> "_PublicationLock":
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


def _rename_directory_noreplace(source: Path, destination: Path) -> bool:
    """Try to atomically publish ``source`` without replacing ``destination``.

    ``True`` means the directory was renamed.  ``False`` is a capability
    result: the platform or backing filesystem does not implement the required
    no-replace operation, so the caller must use the locked manifest-last
    protocol.  Every other failure remains fail-closed.
    """

    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        # Darwin rename.h: RENAME_EXCL refuses an existing destination.
        if renamex_np(encoded_source, encoded_destination, 0x00000004) == 0:
            return True
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), destination)
        if error not in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                f"atomic no-replace rename failed: {os.strerror(error)}",
                artifact=destination,
            )
        return False
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, encoded_source, -100, encoded_destination, 1) == 0:
            return True
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), destination)
        if error not in {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                f"atomic no-replace rename failed: {os.strerror(error)}",
                artifact=destination,
            )
        return False
    return False


def _path_exists_no_follow(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def _open_directory_no_follow(path: Path) -> int:
    """Open one stable real directory and reject a path-swap or symlink."""

    try:
        before = os.lstat(path)
    except FileNotFoundError:
        _refuse(
            V1ExternalStateRefusalCode.STATE_MISSING,
            "publication directory is missing",
            artifact=path,
        )
    if stat.S_ISLNK(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.SYMLINK_REFUSED,
            "publication directory symlink refused",
            artifact=path,
        )
    if not stat.S_ISDIR(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "publication target is not a directory",
            artifact=path,
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "publication directory symlink refused",
                artifact=path,
            )
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            f"cannot open publication directory: {exc}",
            artifact=path,
        )
    after = os.fstat(fd)
    if not stat.S_ISDIR(after.st_mode):
        os.close(fd)
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "opened publication target is not a directory",
            artifact=path,
        )
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(fd)
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "publication directory changed while being opened",
            artifact=path,
        )
    return fd


def _claim_directory_noreplace(path: Path) -> bool:
    """Atomically claim ``path`` as a private partial publication directory."""

    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        return False
    except OSError as exc:
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            f"cannot claim portable publication directory: {exc}",
            artifact=path,
        )
    return True


def _member_stat_no_follow(directory_fd: int, name: str, *, artifact: Path) -> os.stat_result:
    try:
        result = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "publication member disappeared during inspection",
            artifact=artifact,
        )
    if stat.S_ISLNK(result.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.SYMLINK_REFUSED,
            "publication member symlink refused",
            artifact=artifact,
        )
    if not stat.S_ISREG(result.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "publication member is not a regular file",
            artifact=artifact,
        )
    return result


def _publication_member_names(directory_fd: int, *, artifact: Path) -> set[str]:
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            f"cannot enumerate publication directory: {exc}",
            artifact=artifact,
        )
    if any(type(name) is not str or "/" in name or "\\" in name for name in names):
        _refuse(
            V1ExternalStateRefusalCode.PATH_TRAVERSAL,
            "publication directory contains an unsafe member name",
            artifact=artifact,
        )
    return set(names)


def _verify_partial_array(
    directory_fd: int,
    entry: V1ExternalArrayManifestEntry,
    *,
    target: Path,
) -> None:
    """Verify a pre-existing resumable member without changing or replacing it."""

    path = target / entry.file_name
    result = _member_stat_no_follow(directory_fd, entry.file_name, artifact=path)
    if result.st_mode & 0o222:
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "partial publication array must already be read-only",
            artifact=path,
        )
    fd = _open_regular_at(directory_fd, entry.file_name, artifact=path)
    try:
        _verify_file(fd, entry.identity, path)
    finally:
        os.close(fd)


def _link_regular_noreplace(
    source_fd: int,
    destination_fd: int,
    name: str,
    *,
    source_path: Path,
    destination_path: Path,
) -> bool:
    """Hard-link one immutable regular file, atomically and without replace."""

    source_stat = _member_stat_no_follow(source_fd, name, artifact=source_path / name)
    if source_stat.st_mode & 0o222:
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "staged publication member is not read-only",
            artifact=source_path / name,
        )
    if source_stat.st_dev != os.fstat(destination_fd).st_dev:
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            "portable publication crosses filesystems",
            artifact=destination_path,
        )
    try:
        os.link(
            name,
            name,
            src_dir_fd=source_fd,
            dst_dir_fd=destination_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            detail = "portable publication crosses filesystems"
        else:
            detail = f"safe no-replace hard-link publication is unsupported: {exc}"
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            detail,
            artifact=destination_path / name,
        )
    destination_stat = _member_stat_no_follow(
        destination_fd,
        name,
        artifact=destination_path / name,
    )
    if (source_stat.st_dev, source_stat.st_ino) != (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "published member is not the staged immutable inode",
            artifact=destination_path / name,
        )
    return True


def _fsync_directory_fd(fd: int) -> None:
    os.fsync(fd)


def _make_partial_directory_writable(fd: int) -> None:
    os.fchmod(fd, 0o700)


def _seal_directory_read_only(fd: int) -> None:
    os.fchmod(fd, 0o555)


def _verify_and_reuse_committed_target(
    root: Path,
    target: Path,
    descriptor: V1ExternalStateDescriptor,
) -> None:
    """Verify a commit and close only the manifest-before-seal crash window.

    Array and manifest names, bytes, inodes, and modes are never repaired.  The
    sole admitted recovery mutation is the protocol-owned target directory's
    exact ``0700 -> 0555`` transition, after a complete content verification.
    That transition receives target/root durability barriers and a second full
    verification before reuse.
    """

    verify_external_state(root, descriptor)
    directory_fd = _open_directory_no_follow(target)
    try:
        before = os.fstat(directory_fd)
        expected_names = {_MANIFEST_FILE} | {
            _array_file_name(identity, index)
            for index, identity in enumerate(descriptor.arrays)
        }
        observed_names = _publication_member_names(directory_fd, artifact=target)
        if observed_names != expected_names:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "committed publication changed after verification",
                artifact=target,
            )
        for name in observed_names:
            member = _member_stat_no_follow(
                directory_fd,
                name,
                artifact=target / name,
            )
            if stat.S_IMODE(member.st_mode) != 0o444:
                _refuse(
                    V1ExternalStateRefusalCode.FILE_INVALID,
                    "committed publication members must be exactly read-only",
                    artifact=target / name,
                )

        directory_mode = stat.S_IMODE(before.st_mode)
        if directory_mode not in {0o700, 0o555}:
            _refuse(
                V1ExternalStateRefusalCode.FILE_INVALID,
                "committed publication directory has an inadmissible mode",
                artifact=target,
            )
        if directory_mode == 0o700:
            _seal_directory_read_only(directory_fd)
        _fsync_directory_fd(directory_fd)
        _fsync_directory(root)
        # The pre-open verification protects the decision to seal.  This
        # unconditional post-open verification binds reuse to the stable inode
        # opened above, including when no mode transition was needed: a 0555
        # target swapped in after the first verification cannot bypass content
        # verification merely because it was already sealed.
        verify_external_state(root, descriptor)

        after_fd = os.fstat(directory_fd)
        try:
            after_path = os.lstat(target)
        except FileNotFoundError:
            _refuse(
                V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                "committed publication disappeared during seal recovery",
                artifact=target,
            )
        if stat.S_ISLNK(after_path.st_mode):
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "committed publication became a symlink during seal recovery",
                artifact=target,
            )
        if (before.st_dev, before.st_ino) != (after_fd.st_dev, after_fd.st_ino) or (
            after_fd.st_dev,
            after_fd.st_ino,
        ) != (after_path.st_dev, after_path.st_ino):
            _refuse(
                V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                "committed publication changed during seal recovery",
                artifact=target,
            )
    finally:
        os.close(directory_fd)


def _portable_publish_directory(
    *,
    root: Path,
    stage: Path,
    target: Path,
    manifest: V1ExternalStateManifest,
    descriptor: V1ExternalStateDescriptor,
) -> None:
    """Publish or recover one state using a manifest-last write-once protocol."""

    claimed = _claim_directory_noreplace(target)
    if claimed:
        _fsync_directory(root)

    stage_fd = _open_directory_no_follow(stage)
    target_fd = _open_directory_no_follow(target)
    try:
        stage_stat = os.fstat(stage_fd)
        target_stat = os.fstat(target_fd)
        root_stat = os.lstat(root)
        if stage_stat.st_dev != target_stat.st_dev or target_stat.st_dev != root_stat.st_dev:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                "portable publication requires one filesystem",
                artifact=target,
            )
        if target_stat.st_uid != stage_stat.st_uid:
            _refuse(
                V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                "partial publication target has a foreign owner",
                artifact=target,
            )

        expected_arrays = {entry.file_name for entry in manifest.arrays}
        expected_stage = expected_arrays | {_MANIFEST_FILE}
        stage_names = _publication_member_names(stage_fd, artifact=stage)
        if stage_names != expected_stage:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "private stage has missing or unexpected members",
                artifact=stage,
            )
        for name in stage_names:
            _member_stat_no_follow(stage_fd, name, artifact=stage / name)

        observed = _publication_member_names(target_fd, artifact=target)
        for name in observed:
            _member_stat_no_follow(target_fd, name, artifact=target / name)
        if _MANIFEST_FILE in observed:
            # The manifest is the commit marker.  A target bearing it is never
            # edited, completed, or replaced before exact verification.
            os.close(target_fd)
            target_fd = -1
            os.close(stage_fd)
            stage_fd = -1
            _verify_and_reuse_committed_target(root, target, descriptor)
            return
        unexpected = observed - expected_arrays
        if unexpected:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "partial publication has unexpected members",
                artifact=target,
            )

        # A prior crash may have left any exact prefix/subset of immutable
        # arrays.  Validate those members before filling only the missing names.
        by_name = {entry.file_name: entry for entry in manifest.arrays}
        for name in observed:
            _verify_partial_array(target_fd, by_name[name], target=target)
        _make_partial_directory_writable(target_fd)
        for entry in manifest.arrays:
            if entry.file_name in observed:
                continue
            linked = _link_regular_noreplace(
                stage_fd,
                target_fd,
                entry.file_name,
                source_path=stage,
                destination_path=target,
            )
            if not linked:
                _verify_partial_array(target_fd, entry, target=target)
            observed.add(entry.file_name)

        if observed != expected_arrays:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "portable publication did not materialize every array",
                artifact=target,
            )
        if _publication_member_names(target_fd, artifact=target) != expected_arrays:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "publication target changed before commit",
                artifact=target,
            )
        _fsync_directory_fd(target_fd)

        # The canonical, already-fsynced manifest is a single atomic hard-link
        # and therefore the only commit transition readers recognize.
        manifest_linked = _link_regular_noreplace(
            stage_fd,
            target_fd,
            _MANIFEST_FILE,
            source_path=stage,
            destination_path=target,
        )
        if not manifest_linked:
            os.close(target_fd)
            target_fd = -1
            os.close(stage_fd)
            stage_fd = -1
            _verify_and_reuse_committed_target(root, target, descriptor)
            return
        _fsync_directory_fd(target_fd)
        _seal_directory_read_only(target_fd)
        _fsync_directory_fd(target_fd)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if stage_fd >= 0:
            os.close(stage_fd)
    _fsync_directory(root)


def _normalize_array_items(
    arrays: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
) -> tuple[tuple[str, np.ndarray], ...]:
    raw_items = list(arrays.items()) if isinstance(arrays, Mapping) else list(arrays)
    normalized: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for item in raw_items:
        if type(item) not in {tuple, list} or len(item) != 2:
            _refuse(
                V1ExternalStateRefusalCode.INVALID_ARRAY,
                "array entries must be name/value pairs",
            )
        name = _logical_name(item[0])
        if name in seen:
            _refuse(
                V1ExternalStateRefusalCode.DUPLICATE_LOGICAL_NAME,
                f"duplicate logical array name {name!r}",
            )
        seen.add(name)
        value = item[1]
        if not isinstance(value, np.ndarray):
            _refuse(V1ExternalStateRefusalCode.INVALID_ARRAY, "values must be NumPy arrays")
        normalized.append((name, value))
    return tuple(sorted(normalized, key=lambda pair: pair[0]))


def _normalize_policies(
    policies: Mapping[str, V1ExternalArrayPolicy] | Iterable[V1ExternalArrayPolicy],
) -> dict[str, V1ExternalArrayPolicy]:
    raw = list(policies.values()) if isinstance(policies, Mapping) else list(policies)
    result: dict[str, V1ExternalArrayPolicy] = {}
    for policy in raw:
        if type(policy) is not V1ExternalArrayPolicy:
            _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "invalid array policy")
        if policy.logical_name in result:
            _refuse(
                V1ExternalStateRefusalCode.DUPLICATE_LOGICAL_NAME,
                f"duplicate policy for {policy.logical_name!r}",
            )
        result[policy.logical_name] = policy
    return result


def _publish_external_state(
    root: str | os.PathLike[str],
    *,
    namespace: str,
    state_schema: str,
    arrays: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
    policies: Mapping[str, V1ExternalArrayPolicy] | Iterable[V1ExternalArrayPolicy] = (),
    issue_live_attachment: bool,
) -> tuple[V1ExternalStateDescriptor, V1ExternalStateLiveAttachment | None]:
    """Publish canonical immutable arrays and return a root-free descriptor.

    Identical concurrent calls are idempotent: exactly one directory wins the
    no-replace publication and every loser verifies that winner before using
    its descriptor.  A policy is optional for finite arrays; admitting NaN or
    infinity always requires an explicit ``allow_nonfinite=True`` policy.
    """

    namespace = _exact_token(namespace, "namespace")
    state_schema = _exact_token(state_schema, "state_schema")
    root_path = _prepare_root(root, create=True)
    items = _normalize_array_items(arrays)
    policy_by_name = _normalize_policies(policies)
    names = {name for name, _ in items}
    if set(policy_by_name) - names:
        _refuse(
            V1ExternalStateRefusalCode.INVALID_SCHEMA,
            "array policy names must be present in the publication",
        )

    stage = root_path / f".open-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        os.mkdir(stage, 0o700)
    except OSError as exc:
        _refuse(
            V1ExternalStateRefusalCode.PUBLICATION_FAILED,
            f"cannot create open-new staging directory: {exc}",
            artifact=stage,
        )

    identities: list[V1ExternalArrayIdentity] = []
    try:
        for index, (name, value) in enumerate(items):
            policy = policy_by_name.get(name, V1ExternalArrayPolicy(name))
            canonical_dtype = _canonical_dtype(value.dtype).str
            shape = tuple(int(item) for item in value.shape)
            if policy.expected_dtype is not None and policy.expected_dtype != canonical_dtype:
                _refuse(
                    V1ExternalStateRefusalCode.INVALID_SCHEMA,
                    f"{name!r} dtype {canonical_dtype} violates policy {policy.expected_dtype}",
                )
            if policy.expected_shape is not None and policy.expected_shape != shape:
                _refuse(
                    V1ExternalStateRefusalCode.INVALID_SCHEMA,
                    f"{name!r} shape {shape} violates policy {policy.expected_shape}",
                )
            temporary = stage / f".array-{index:08d}-{uuid.uuid4().hex}.open"
            dtype, shape, byte_count, digest = _stream_array_to_file(
                temporary,
                value,
                allow_nonfinite=policy.allow_nonfinite,
            )
            identity = V1ExternalArrayIdentity(
                logical_name=name,
                dtype=dtype,
                shape=shape,
                byte_count=byte_count,
                sha256=digest,
                allow_nonfinite=policy.allow_nonfinite,
            )
            final_name = _array_file_name(identity, index)
            final_path = stage / final_name
            if final_path.exists() or final_path.is_symlink():
                _refuse(
                    V1ExternalStateRefusalCode.PUBLICATION_FAILED,
                    "staging destination unexpectedly exists",
                    artifact=final_path,
                )
            os.rename(temporary, final_path)
            identities.append(identity)

        identity_tuple = tuple(identities)
        logical_identity_hash = content_hash(
            _logical_identity_payload(namespace, state_schema, identity_tuple)
        )
        manifest = _manifest_from_descriptor_fields(
            namespace,
            state_schema,
            identity_tuple,
            logical_identity_hash,
        )
        descriptor = V1ExternalStateDescriptor(
            namespace=namespace,
            state_schema=state_schema,
            arrays=identity_tuple,
            logical_identity_hash=logical_identity_hash,
            manifest_hash=manifest.manifest_hash,
        )
        manifest_temporary = stage / f".{_MANIFEST_FILE}.{uuid.uuid4().hex}.open"
        _write_fsynced_file(
            manifest_temporary,
            _canonical_json_bytes(manifest.to_json_dict()),
        )
        os.rename(manifest_temporary, stage / _MANIFEST_FILE)
        _fsync_directory(stage)
        os.chmod(stage, 0o555)
        _fsync_directory(stage)

        target = root_path / _state_directory_name(manifest.manifest_hash)
        with _PublicationLock(root_path, manifest.manifest_hash):
            renamed = False
            if not _path_exists_no_follow(target):
                try:
                    renamed = _rename_directory_noreplace(stage, target)
                except FileExistsError:
                    # An uncooperative publisher may have raced the lstat.  It
                    # receives exactly the same committed/partial inspection as
                    # a recovered target; no existing member is replaced.
                    renamed = False
            if renamed:
                _fsync_directory(root_path)
                attachment = (
                    _issue_live_attachment(root_path, descriptor)
                    if issue_live_attachment
                    else None
                )
                return descriptor, attachment
            _portable_publish_directory(
                root=root_path,
                stage=stage,
                target=target,
                manifest=manifest,
                descriptor=descriptor,
            )
            _cleanup_stage(stage)
            attachment = (
                _issue_live_attachment(root_path, descriptor)
                if issue_live_attachment
                else None
            )
        return descriptor, attachment
    except BaseException:
        _cleanup_stage(stage)
        raise


def publish_external_state(
    root: str | os.PathLike[str],
    *,
    namespace: str,
    state_schema: str,
    arrays: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
    policies: Mapping[str, V1ExternalArrayPolicy] | Iterable[V1ExternalArrayPolicy] = (),
) -> V1ExternalStateDescriptor:
    """Publish canonical immutable arrays and return a root-free descriptor."""

    descriptor, attachment = _publish_external_state(
        root,
        namespace=namespace,
        state_schema=state_schema,
        arrays=arrays,
        policies=policies,
        issue_live_attachment=False,
    )
    assert attachment is None
    return descriptor


def publish_external_state_with_attachment(
    root: str | os.PathLike[str],
    *,
    namespace: str,
    state_schema: str,
    arrays: Mapping[str, np.ndarray] | Iterable[tuple[str, np.ndarray]],
    policies: Mapping[str, V1ExternalArrayPolicy] | Iterable[V1ExternalArrayPolicy] = (),
) -> tuple[V1ExternalStateDescriptor, V1ExternalStateLiveAttachment]:
    """Durably publish arrays and issue a PID-bound live publication attachment."""

    descriptor, attachment = _publish_external_state(
        root,
        namespace=namespace,
        state_schema=state_schema,
        arrays=arrays,
        policies=policies,
        issue_live_attachment=True,
    )
    assert attachment is not None
    return descriptor, attachment


def _strict_json(raw: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _refuse(
                    V1ExternalStateRefusalCode.MANIFEST_INVALID,
                    f"duplicate JSON field {key!r}",
                )
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("ascii"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _refuse(V1ExternalStateRefusalCode.MANIFEST_INVALID, f"invalid manifest JSON: {exc}")


def _open_regular_at(directory_fd: int, name: str, *, artifact: Path) -> int:
    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        _refuse(
            V1ExternalStateRefusalCode.PATH_TRAVERSAL,
            "unsafe child path",
            artifact=artifact,
        )
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            V1ExternalStateRefusalCode.STATE_PARTIAL,
            "required file is missing",
            artifact=artifact,
        )
    if stat.S_ISLNK(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.SYMLINK_REFUSED,
            "symlink refused",
            artifact=artifact,
        )
    if not stat.S_ISREG(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "expected a regular file",
            artifact=artifact,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        _refuse(
            V1ExternalStateRefusalCode.STATE_PARTIAL,
            "required file is missing",
            artifact=artifact,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "symlink refused",
                artifact=artifact,
            )
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            f"cannot open file: {exc}",
            artifact=artifact,
        )
    after = os.fstat(fd)
    if not stat.S_ISREG(after.st_mode):
        os.close(fd)
        _refuse(V1ExternalStateRefusalCode.FILE_INVALID, "not a regular file", artifact=artifact)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(fd)
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "file was replaced while being opened",
            artifact=artifact,
        )
    return fd


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_manifest(
    directory_fd: int,
    path: Path,
) -> tuple[V1ExternalStateManifest, tuple[int, int, int, int, int]]:
    fd = _open_regular_at(directory_fd, _MANIFEST_FILE, artifact=path / _MANIFEST_FILE)
    try:
        before = os.fstat(fd)
        if before.st_size <= 0 or before.st_size > _MAX_MANIFEST_BYTES:
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "manifest size is outside the admitted range",
                artifact=path / _MANIFEST_FILE,
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                _refuse(
                    V1ExternalStateRefusalCode.STATE_PARTIAL,
                    "manifest ended before its declared file size",
                    artifact=path / _MANIFEST_FILE,
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if _stat_signature(before) != _stat_signature(after):
            _refuse(
                V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                "manifest changed during inspection",
                artifact=path / _MANIFEST_FILE,
            )
        raw = b"".join(chunks)
        value = _strict_json(raw)
        manifest = V1ExternalStateManifest.from_json_dict(value)
        if raw != _canonical_json_bytes(manifest.to_json_dict()):
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "manifest bytes are not canonical JSON",
                artifact=path / _MANIFEST_FILE,
            )
        return manifest, _stat_signature(after)
    finally:
        os.close(fd)


def _open_directory_at_no_follow(
    parent_fd: int,
    name: str,
    *,
    artifact: Path,
) -> int:
    """Open one real child directory and bind its path identity to the FD."""

    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            V1ExternalStateRefusalCode.STATE_MISSING,
            "state directory is missing",
            artifact=artifact,
        )
    if stat.S_ISLNK(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.SYMLINK_REFUSED,
            "state directory symlink refused",
            artifact=artifact,
        )
    if not stat.S_ISDIR(before.st_mode):
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            "state target is not a directory",
            artifact=artifact,
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "state directory symlink refused",
                artifact=artifact,
            )
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            f"cannot open state directory: {exc}",
            artifact=artifact,
        )
    after = os.fstat(fd)
    if not stat.S_ISDIR(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        os.close(fd)
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "state directory changed while being opened",
            artifact=artifact,
        )
    return fd


def _issue_live_attachment(
    root: Path,
    descriptor: V1ExternalStateDescriptor,
    *,
    verified_directory_signature: tuple[int, int, int, int, int] | None = None,
    verified_member_signatures: Mapping[
        str, tuple[int, int, int, int, int]
    ] | None = None,
) -> V1ExternalStateLiveAttachment:
    """Retain immutable no-follow witnesses after publication/exact opening."""

    root_fd = _open_directory_no_follow(root)
    state_name = _state_directory_name(descriptor.manifest_hash)
    state_path = root / state_name
    directory_fd = -1
    opened_members: list[_V1ExternalStateMemberWitness] = []
    try:
        directory_fd = _open_directory_at_no_follow(
            root_fd,
            state_name,
            artifact=state_path,
        )
        directory_stat = os.fstat(directory_fd)
        directory_signature = _stat_signature(directory_stat)
        if stat.S_IMODE(directory_stat.st_mode) != 0o555:
            _refuse(
                V1ExternalStateRefusalCode.LIVE_ATTACHMENT_INVALID,
                "live attachment requires an exactly 0555 publication directory",
                artifact=state_path,
            )
        if (
            verified_directory_signature is not None
            and directory_signature != verified_directory_signature
        ):
            _refuse(
                V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                "publication directory changed after its exact barrier",
                artifact=state_path,
            )
        manifest = _manifest_from_descriptor_fields(
            descriptor.namespace,
            descriptor.state_schema,
            descriptor.arrays,
            descriptor.logical_identity_hash,
        )
        expected_names = {_MANIFEST_FILE} | {
            entry.file_name for entry in manifest.arrays
        }
        if set(os.listdir(directory_fd)) != expected_names:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "publication members changed before live attachment issuance",
                artifact=state_path,
            )
        for name in sorted(expected_names):
            path = state_path / name
            fd = _open_regular_at(directory_fd, name, artifact=path)
            keep = False
            try:
                member_stat = os.fstat(fd)
                signature = _stat_signature(member_stat)
                if stat.S_IMODE(member_stat.st_mode) != 0o444:
                    _refuse(
                        V1ExternalStateRefusalCode.LIVE_ATTACHMENT_INVALID,
                        "live attachment requires exactly 0444 publication members",
                        artifact=path,
                    )
                if verified_member_signatures is not None:
                    expected_signature = verified_member_signatures.get(name)
                    if expected_signature is None or signature != expected_signature:
                        _refuse(
                            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                            "publication member changed after its exact barrier",
                            artifact=path,
                        )
                opened_members.append(
                    _V1ExternalStateMemberWitness(
                        name=name,
                        fd=fd,
                        signature=signature,
                    )
                )
                keep = True
            finally:
                if not keep:
                    os.close(fd)
        attachment = V1ExternalStateLiveAttachment(
            root=root,
            state_name=state_name,
            descriptor_hash=descriptor.descriptor_hash,
            root_fd=root_fd,
            directory_fd=directory_fd,
            directory_signature=directory_signature,
            members=tuple(opened_members),
            token=_LIVE_ATTACHMENT_CONSTRUCTION_TOKEN,
        )
        attachment.verify(descriptor)
        root_fd = -1
        directory_fd = -1
        opened_members = []
        return attachment
    finally:
        for member in opened_members:
            try:
                os.close(member.fd)
            except OSError:
                pass
        if directory_fd >= 0:
            os.close(directory_fd)
        if root_fd >= 0:
            os.close(root_fd)


def _verify_file(
    fd: int,
    identity: V1ExternalArrayIdentity,
    path: Path,
) -> tuple[int, int, int, int, int]:
    before = os.fstat(fd)
    if before.st_size != identity.byte_count:
        _refuse(
            V1ExternalStateRefusalCode.SIZE_MISMATCH,
            f"array has {before.st_size} bytes, expected {identity.byte_count}",
            artifact=path,
        )
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    remaining = identity.byte_count
    dtype = np.dtype(identity.dtype)
    check_finite = not identity.allow_nonfinite and dtype.kind in "fc"
    finite_remainder = b""
    finite_values = True
    while remaining:
        chunk = os.read(fd, min(remaining, _HASH_CHUNK_BYTES))
        if not chunk:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "array ended before its manifest byte count",
                artifact=path,
            )
        digest.update(chunk)
        if check_finite:
            finite_chunk = finite_remainder + chunk if finite_remainder else chunk
            usable = len(finite_chunk) - (len(finite_chunk) % dtype.itemsize)
            if usable:
                finite_values = finite_values and bool(
                    np.isfinite(
                        np.frombuffer(
                            finite_chunk,
                            dtype=dtype,
                            count=usable // dtype.itemsize,
                        )
                    ).all()
                )
            finite_remainder = finite_chunk[usable:]
        remaining -= len(chunk)
    if finite_remainder:
        _refuse(
            V1ExternalStateRefusalCode.SIZE_MISMATCH,
            "array byte count is not aligned to its dtype",
            artifact=path,
        )
    after = os.fstat(fd)
    if _stat_signature(before) != _stat_signature(after):
        _refuse(
            V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
            "array changed during inspection",
            artifact=path,
        )
    if digest.hexdigest() != identity.sha256:
        _refuse(
            V1ExternalStateRefusalCode.DIGEST_MISMATCH,
            "array SHA-256 does not match its manifest",
            artifact=path,
        )
    if not finite_values:
        _refuse(
            V1ExternalStateRefusalCode.NONFINITE_ARRAY,
            "array violates its finite-value schema policy",
            artifact=path,
        )
    return _stat_signature(after)


def _map_verified_fd(fd: int, identity: V1ExternalArrayIdentity) -> np.ndarray:
    dtype = np.dtype(identity.dtype)
    if identity.byte_count == 0:
        # POSIX cannot mmap a zero-byte file.  A view over immutable ``bytes``
        # preserves the same no-write contract without inventing payload bytes.
        return np.frombuffer(b"", dtype=dtype, count=0).reshape(identity.shape)
    duplicate = os.dup(fd)
    try:
        with os.fdopen(duplicate, "rb", closefd=True) as handle:
            result = np.memmap(handle, dtype=dtype, mode="r", shape=identity.shape, order="C")
        duplicate = -1
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    result.setflags(write=False)
    return result


def _open_external_state(
    root: str | os.PathLike[str],
    descriptor: V1ExternalStateDescriptor,
    *,
    issue_live_attachment: bool,
) -> tuple[V1ExternalStateView, V1ExternalStateLiveAttachment | None]:
    """Verify the complete state, then return read-only memory mappings."""

    if type(descriptor) is not V1ExternalStateDescriptor:
        _refuse(V1ExternalStateRefusalCode.INVALID_SCHEMA, "invalid descriptor type")
    # Reconstructing through strict JSON replays every descriptor invariant even
    # if a caller bypassed a frozen dataclass with object.__setattr__.
    descriptor = V1ExternalStateDescriptor.from_json_dict(descriptor.to_json_dict())
    root_path = _prepare_root(root, create=False)
    state_path = root_path / _state_directory_name(descriptor.manifest_hash)
    _lstat_no_symlink(state_path, require_directory=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(state_path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _refuse(
                V1ExternalStateRefusalCode.SYMLINK_REFUSED,
                "state symlink refused",
                artifact=state_path,
            )
        _refuse(
            V1ExternalStateRefusalCode.FILE_INVALID,
            f"cannot open state: {exc}",
            artifact=state_path,
        )

    opened: list[
        tuple[V1ExternalArrayIdentity, int, tuple[int, int, int, int, int]]
    ] = []
    mapped: dict[str, np.ndarray] = {}
    attachment: V1ExternalStateLiveAttachment | None = None
    try:
        manifest, manifest_signature = _read_manifest(directory_fd, state_path)
        expected_manifest = _manifest_from_descriptor_fields(
            descriptor.namespace,
            descriptor.state_schema,
            descriptor.arrays,
            descriptor.logical_identity_hash,
        )
        if manifest != expected_manifest or manifest.manifest_hash != descriptor.manifest_hash:
            _refuse(
                V1ExternalStateRefusalCode.MANIFEST_INVALID,
                "stored manifest does not exactly match the checkpoint descriptor",
                artifact=state_path / _MANIFEST_FILE,
            )
        expected_names = {_MANIFEST_FILE} | {entry.file_name for entry in manifest.arrays}
        observed_names = set(os.listdir(directory_fd))
        if observed_names != expected_names:
            _refuse(
                V1ExternalStateRefusalCode.STATE_PARTIAL,
                "state directory has missing or unexpected files",
                artifact=state_path,
            )
        for entry in manifest.arrays:
            path = state_path / entry.file_name
            fd = _open_regular_at(directory_fd, entry.file_name, artifact=path)
            try:
                signature = _verify_file(fd, entry.identity, path)
            except BaseException:
                os.close(fd)
                raise
            opened.append((entry.identity, fd, signature))

        # No mapping is exposed until every required file has passed the
        # manifest, type/shape/size, digest, and stable-inode barrier.
        for identity, fd, _ in opened:
            value = _map_verified_fd(fd, identity)
            mapped[identity.logical_name] = value
        for _, fd, signature in opened:
            if _stat_signature(os.fstat(fd)) != signature:
                _refuse(
                    V1ExternalStateRefusalCode.ARTIFACT_CHANGED_DURING_INSPECTION,
                    "array changed between verification and mapping",
                    artifact=state_path,
                )
        receipt = V1ExternalStateInspectionReceipt(
            descriptor_hash=descriptor.descriptor_hash,
            manifest_hash=descriptor.manifest_hash,
            array_identity_hashes=tuple(identity.identity_hash for identity in descriptor.arrays),
        )
        if issue_live_attachment:
            signatures = {_MANIFEST_FILE: manifest_signature}
            signatures.update(
                (
                    _array_file_name(identity, index),
                    signature,
                )
                for index, (identity, _fd, signature) in enumerate(opened)
            )
            attachment = _issue_live_attachment(
                root_path,
                descriptor,
                verified_directory_signature=_stat_signature(os.fstat(directory_fd)),
                verified_member_signatures=signatures,
            )
        return (
            V1ExternalStateView(
                descriptor=descriptor,
                arrays=MappingProxyType(mapped),
                inspection_receipt=receipt,
            ),
            attachment,
        )
    except BaseException:
        if attachment is not None:
            attachment.close()
        for value in mapped.values():
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
        raise
    finally:
        for _, fd, _ in opened:
            os.close(fd)
        os.close(directory_fd)


def open_external_state(
    root: str | os.PathLike[str],
    descriptor: V1ExternalStateDescriptor,
) -> V1ExternalStateView:
    """Run the exact cold barrier and return read-only memory mappings."""

    view, attachment = _open_external_state(
        root,
        descriptor,
        issue_live_attachment=False,
    )
    assert attachment is None
    return view


def open_external_state_with_attachment(
    root: str | os.PathLike[str],
    descriptor: V1ExternalStateDescriptor,
) -> tuple[V1ExternalStateView, V1ExternalStateLiveAttachment]:
    """Run the exact cold barrier and issue a process-local live attachment."""

    view, attachment = _open_external_state(
        root,
        descriptor,
        issue_live_attachment=True,
    )
    assert attachment is not None
    return view, attachment


def verify_external_state(
    root: str | os.PathLike[str],
    descriptor: V1ExternalStateDescriptor,
) -> V1ExternalStateInspectionReceipt:
    """Run the same all-file barrier as opening, without retaining mappings."""

    view = open_external_state(root, descriptor)
    try:
        return view.inspection_receipt
    finally:
        view.close()


__all__ = [
    "V1ExternalArrayIdentity",
    "V1ExternalArrayManifestEntry",
    "V1ExternalArrayPolicy",
    "V1ExternalStateDescriptor",
    "V1ExternalStateInspectionReceipt",
    "V1ExternalStateLiveAttachment",
    "V1ExternalStateManifest",
    "V1ExternalStateRefusal",
    "V1ExternalStateRefusalCode",
    "V1ExternalStateView",
    "open_external_state",
    "open_external_state_with_attachment",
    "publish_external_state",
    "publish_external_state_with_attachment",
    "verify_external_state",
]
