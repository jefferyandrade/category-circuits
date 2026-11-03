"""Durable exact replay for the modern-v1 AlexNet analytic envelope.

The analytic envelope is expensive enough that a resumed owner must reopen the
exact result that was admitted before outcomes, rather than recompute an
apparently equivalent replacement.  This module publishes that result as two
plain, independently verifiable files:

* ``envelope.f64le`` is the certificate's exact C-order little-endian float64
  array; and
* ``manifest.json`` is a canonical, versioned JSON rendering of every
  certificate coordinate, derivation-receipt coordinate, and layer receipt.

The manifest is linked last and is the atomic commit marker.  Publication uses
only open-new files and no-replace hard links below a content-addressed object
directory.  Opening rejects symlinks, unsafe paths, unexpected members, byte
or manifest changes, and any failure to reconstruct the original public
objects and pass their complete ``assert_integrity`` chain.  The checkpoint
descriptor contains no physical path, so a byte-for-byte store may be moved to
another durable root without changing scientific identity.

No pickle-like decoder is used.  Recalculation is deliberately absent from the
API: a missing or invalid object is a replay refusal, never authority to issue
a replacement envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import errno
import fcntl
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Final, Mapping, NoReturn, cast
import uuid

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_alexnet_envelope import (
    ALEXNET_ENVELOPE_RESULT_SCHEMA,
    V1AlexNetAttributionEnvelopeDerivationReceipt,
    V1AlexNetCorrectedAttributionEnvelopeResult,
    V1AlexNetEnvelopeLayerReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    V1CorrectedAttributionEnvelopeCertificate,
)
from mechanistic_systems.contracts.content_address import content_hash


ALEXNET_ENVELOPE_STORE_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_alexnet_envelope_store_descriptor_v1"
)
ALEXNET_ENVELOPE_STORE_MANIFEST_SCHEMA: Final = (
    "partition_free_v1_alexnet_envelope_store_manifest_v1"
)
ALEXNET_ENVELOPE_STORE_FORMAT_VERSION: Final = 1
ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE: Final = (
    "pre_outcome_analytic_envelope_exact_replay_only_"
    "no_recompute_outcome_or_mask_authority_v1"
)
ALEXNET_ENVELOPE_STORE_ARRAY_CONTENT_SCHEMA: Final = (
    "partition_free_v1_alexnet_envelope_store_array_content_v1"
)

_MANIFEST_CONTENT_PREFIX: Final = (
    "partition-free-v1-alexnet-envelope-store-manifest-content-"
)
_ARRAY_CONTENT_PREFIX: Final = (
    "partition-free-v1-alexnet-envelope-store-array-content-"
)
_DESCRIPTOR_PREFIX: Final = (
    "partition-free-v1-alexnet-envelope-store-descriptor-"
)
_OBJECT_PREFIX: Final = "object-"
_MANIFEST_FILE: Final = "manifest.json"
_ARRAY_FILE: Final = "envelope.f64le"
_MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
_HASH_CHUNK_BYTES: Final = 8 * 1024 * 1024
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_SAFE_OBJECT_NAME: Final = re.compile(r"object-[0-9a-f]{64}\Z")
_LE_F64: Final = np.dtype("<f8")
_STAGE_RMDIR_RETRY_DELAYS_SECONDS: Final = (
    0.0,
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
)


class V1AlexNetEnvelopeStoreRefusalCode(str, Enum):
    """Stable fail-closed categories for envelope replay storage."""

    INVALID_DESCRIPTOR = "invalid_descriptor"
    INVALID_RESULT = "invalid_result"
    ROOT_INVALID = "root_invalid"
    PATH_TRAVERSAL = "path_traversal"
    SYMLINK_REFUSED = "symlink_refused"
    PUBLICATION_CONFLICT = "publication_conflict"
    PUBLICATION_FAILED = "publication_failed"
    OBJECT_MISSING = "object_missing"
    OBJECT_PARTIAL = "object_partial"
    MANIFEST_INVALID = "manifest_invalid"
    FILE_INVALID = "file_invalid"
    SIZE_MISMATCH = "size_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    ARTIFACT_CHANGED = "artifact_changed"


class V1AlexNetEnvelopeStoreRefusal(ValueError):
    """Typed replay refusal with an optional physical artifact coordinate."""

    def __init__(
        self,
        code: V1AlexNetEnvelopeStoreRefusalCode,
        message: str,
        *,
        artifact: str | os.PathLike[str] | None = None,
    ) -> None:
        self.code = code
        self.artifact = None if artifact is None else str(artifact)
        suffix = "" if self.artifact is None else f" [{self.artifact}]"
        super().__init__(f"{code.value}: {message}{suffix}")


def _refuse(
    code: V1AlexNetEnvelopeStoreRefusalCode,
    message: str,
    *,
    artifact: str | os.PathLike[str] | None = None,
) -> NoReturn:
    raise V1AlexNetEnvelopeStoreRefusal(code, message, artifact=artifact)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not cast(str, value) or "\x00" in cast(str, value):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
            f"{name} must be one nonempty string without NUL bytes",
        )
    return cast(str, value)


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(cast(str, value)) != 64
        or any(character not in _LOWER_SHA256 for character in cast(str, value))
    ):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
            f"{name} must be one lowercase SHA-256 digest",
        )
    return cast(str, value)


def _exact_int(value: object, *, name: str, positive: bool = False) -> int:
    if type(value) is not int or cast(int, value) < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one exact {qualifier} integer",
        )
    return cast(int, value)


def _exact_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one exact boolean",
        )
    return cast(bool, value)


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one finite number",
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one finite number",
        )
    if not math.isfinite(result):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one finite number",
        )
    return result


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} fields are not exact",
        )
    return cast(dict[str, object], value)


def _string_pair(value: object, *, name: str) -> tuple[str, str]:
    if type(value) is not list or len(cast(list[object], value)) != 2:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one JSON pair",
        )
    values = cast(list[object], value)
    return (
        _identity(values[0], name=f"{name} item"),
        _identity(values[1], name=f"{name} item"),
    )


def _positive_int_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not list or len(cast(list[object], value)) != 2:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one JSON pair",
        )
    values = cast(list[object], value)
    return (
        _exact_int(values[0], name=f"{name} item", positive=True),
        _exact_int(values[1], name=f"{name} item", positive=True),
    )


def _positive_shape(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not list or not cast(list[object], value):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"{name} must be one nonempty JSON shape",
        )
    return tuple(
        _exact_int(item, name=f"{name} dimension", positive=True)
        for item in cast(list[object], value)
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_RESULT,
            f"envelope manifest is not canonical JSON: {error}",
        )
    return (rendered + "\n").encode("ascii")


def _strict_json(raw: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
                    "manifest contains a duplicate JSON key",
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> NoReturn:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"manifest contains nonfinite JSON constant {value!r}",
        )

    try:
        return json.loads(
            raw.decode("ascii"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except UnicodeDecodeError as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"manifest is not canonical ASCII JSON: {error}",
        )
    except json.JSONDecodeError as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            f"manifest JSON is invalid: {error}",
        )


@dataclass(frozen=True, slots=True)
class V1AlexNetEnvelopeStoreDescriptor:
    """Root-free, JSON/checkpoint-safe identity of one exact stored result."""

    manifest_sha256: str
    manifest_content_hash: str
    proof_receipt_hash: str
    certificate_hash: str
    envelope_array_sha256: str
    envelope_array_content_hash: str
    authority_scope: str = ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE
    format_version: int = ALEXNET_ENVELOPE_STORE_FORMAT_VERSION
    schema: str = ALEXNET_ENVELOPE_STORE_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema) is not str
            or self.schema != ALEXNET_ENVELOPE_STORE_DESCRIPTOR_SCHEMA
        ):
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
                "unsupported AlexNet envelope-store descriptor schema",
            )
        if (
            type(self.format_version) is not int
            or self.format_version != ALEXNET_ENVELOPE_STORE_FORMAT_VERSION
        ):
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
                "unsupported AlexNet envelope-store format version",
            )
        _sha256(self.manifest_sha256, name="manifest_sha256")
        _sha256(self.envelope_array_sha256, name="envelope_array_sha256")
        for name in (
            "manifest_content_hash",
            "proof_receipt_hash",
            "certificate_hash",
            "envelope_array_content_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.authority_scope) is not str
            or self.authority_scope != ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE
        ):
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
                "envelope-store descriptor claims an unsupported authority scope",
            )

    @property
    def object_name(self) -> str:
        name = f"{_OBJECT_PREFIX}{self.manifest_sha256}"
        if _SAFE_OBJECT_NAME.fullmatch(name) is None:  # defensive by construction
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
                "descriptor produced an unsafe object name",
            )
        return name

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self.to_json_dict(), prefix=_DESCRIPTOR_PREFIX)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "format_version": self.format_version,
            "manifest_sha256": self.manifest_sha256,
            "manifest_content_hash": self.manifest_content_hash,
            "proof_receipt_hash": self.proof_receipt_hash,
            "certificate_hash": self.certificate_hash,
            "envelope_array_sha256": self.envelope_array_sha256,
            "envelope_array_content_hash": self.envelope_array_content_hash,
            "authority_scope": self.authority_scope,
        }

    @classmethod
    def from_json_dict(cls, value: object) -> "V1AlexNetEnvelopeStoreDescriptor":
        keys = {
            "schema",
            "format_version",
            "manifest_sha256",
            "manifest_content_hash",
            "proof_receipt_hash",
            "certificate_hash",
            "envelope_array_sha256",
            "envelope_array_content_hash",
            "authority_scope",
        }
        if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
                "AlexNet envelope-store descriptor fields are not exact",
            )
        payload = cast(dict[str, object], value)
        return cls(
            schema=cast(str, payload["schema"]),
            format_version=cast(int, payload["format_version"]),
            manifest_sha256=cast(str, payload["manifest_sha256"]),
            manifest_content_hash=cast(str, payload["manifest_content_hash"]),
            proof_receipt_hash=cast(str, payload["proof_receipt_hash"]),
            certificate_hash=cast(str, payload["certificate_hash"]),
            envelope_array_sha256=cast(str, payload["envelope_array_sha256"]),
            envelope_array_content_hash=cast(
                str, payload["envelope_array_content_hash"]
            ),
            authority_scope=cast(str, payload["authority_scope"]),
        )


def _layer_to_json(layer: V1AlexNetEnvelopeLayerReceipt) -> dict[str, object]:
    if type(layer) is not V1AlexNetEnvelopeLayerReceipt:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_RESULT,
            "derivation contains a noncanonical layer receipt",
        )
    return {
        field.name: (
            list(cast(tuple[int, ...], getattr(layer, field.name)))
            if field.name in {"source_shape", "destination_shape"}
            else getattr(layer, field.name)
        )
        for field in fields(V1AlexNetEnvelopeLayerReceipt)
    }


def _receipt_to_json(
    receipt: V1AlexNetAttributionEnvelopeDerivationReceipt,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields(V1AlexNetAttributionEnvelopeDerivationReceipt):
        if field.name.startswith("_"):
            continue
        value = getattr(receipt, field.name)
        if field.name == "layer_receipts":
            result[field.name] = [_layer_to_json(layer) for layer in value]
        elif isinstance(value, tuple):
            result[field.name] = list(value)
        else:
            result[field.name] = value
    result["receipt_hash"] = receipt.assert_integrity()
    return result


def _certificate_to_json(
    certificate: V1CorrectedAttributionEnvelopeCertificate,
) -> dict[str, object]:
    excluded = {"absolute_upper_envelope_by_stratum", "_certificate_hash"}
    result: dict[str, object] = {}
    for field in fields(V1CorrectedAttributionEnvelopeCertificate):
        if field.name in excluded:
            continue
        value = getattr(certificate, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    result["certificate_hash"] = certificate.assert_integrity()
    return result


def _array_content_hash(array: np.ndarray) -> str:
    return content_hash(
        {
            "schema": ALEXNET_ENVELOPE_STORE_ARRAY_CONTENT_SCHEMA,
            "dtype": _LE_F64.str,
            "shape": tuple(int(value) for value in array.shape),
            "values": array,
        },
        prefix=_ARRAY_CONTENT_PREFIX,
    )


def _canonical_envelope_array(result: V1AlexNetCorrectedAttributionEnvelopeResult) -> np.ndarray:
    if type(result) is not V1AlexNetCorrectedAttributionEnvelopeResult:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_RESULT,
            "publication requires the exact AlexNet envelope-result type",
        )
    try:
        result.assert_integrity()
    except (TypeError, ValueError, PermissionError, RuntimeError) as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_RESULT,
            f"AlexNet envelope result failed public integrity: {error}",
        )
    array = np.asarray(result.certificate.absolute_upper_envelope_by_stratum)
    if (
        array.dtype != _LE_F64
        or array.shape != (2, result.certificate.route_count)
        or not array.flags.c_contiguous
        or not bool(np.isfinite(array).all())
        or bool(np.any(array < 0.0))
    ):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_RESULT,
            "certificate envelope is not canonical finite nonnegative <f8 C-order",
        )
    return array


def _manifest_and_descriptor(
    result: V1AlexNetCorrectedAttributionEnvelopeResult,
) -> tuple[bytes, V1AlexNetEnvelopeStoreDescriptor, memoryview]:
    array = _canonical_envelope_array(result)
    raw = memoryview(array).cast("B")
    array_sha256 = hashlib.sha256(raw).hexdigest()
    array_content_hash = _array_content_hash(array)
    receipt_hash, certificate_hash = result.assert_integrity()
    core: dict[str, object] = {
        "schema": ALEXNET_ENVELOPE_STORE_MANIFEST_SCHEMA,
        "format_version": ALEXNET_ENVELOPE_STORE_FORMAT_VERSION,
        "authority_scope": ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE,
        "result_schema": result.schema,
        "array": {
            "file_name": _ARRAY_FILE,
            "dtype": _LE_F64.str,
            "order": "C",
            "shape": [2, result.certificate.route_count],
            "byte_count": len(raw),
            "sha256": array_sha256,
            "content_hash": array_content_hash,
        },
        "certificate": _certificate_to_json(result.certificate),
        "derivation_receipt": _receipt_to_json(result.derivation_receipt),
    }
    manifest_content_hash = content_hash(core, prefix=_MANIFEST_CONTENT_PREFIX)
    manifest = dict(core)
    manifest["manifest_content_hash"] = manifest_content_hash
    manifest_bytes = _canonical_json_bytes(manifest)
    descriptor = V1AlexNetEnvelopeStoreDescriptor(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_content_hash=manifest_content_hash,
        proof_receipt_hash=receipt_hash,
        certificate_hash=certificate_hash,
        envelope_array_sha256=array_sha256,
        envelope_array_content_hash=array_content_hash,
    )
    return manifest_bytes, descriptor, raw


def _root_components(root: str | os.PathLike[str]) -> tuple[Path, tuple[str, ...]]:
    raw = os.fspath(root)
    if type(raw) is bytes:
        try:
            raw = raw.decode()
        except UnicodeDecodeError:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
                "durable root is not a filesystem string",
            )
    if not raw or "\x00" in raw:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
            "durable root must be one specific path",
        )
    lexical = Path(raw)
    if ".." in lexical.parts:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
            "durable root must not contain parent traversal",
            artifact=lexical,
        )
    absolute = lexical if lexical.is_absolute() else Path.cwd() / lexical
    if absolute == Path(absolute.anchor):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
            "filesystem root is too broad for an envelope store",
            artifact=absolute,
        )
    parts = tuple(part for part in absolute.parts if part not in {absolute.anchor, ""})
    return absolute, parts


def _open_root(
    root: str | os.PathLike[str],
    *,
    create: bool,
) -> tuple[Path, int]:
    """Traverse every root component through no-follow directory descriptors."""

    path, components = _root_components(root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    traversed = Path(path.anchor)
    try:
        for component in components:
            traversed /= component
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as error:
                    _refuse(
                        V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
                        f"cannot create durable root component: {error}",
                        artifact=traversed,
                    )
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
                    "durable root is missing",
                    artifact=traversed,
                )
            if stat.S_ISLNK(before.st_mode):
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED,
                    "durable root component is symlinked",
                    artifact=traversed,
                )
            if not stat.S_ISDIR(before.st_mode):
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID,
                    "durable root component is not a directory",
                    artifact=traversed,
                )
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                code = (
                    V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED
                    if error.errno == errno.ELOOP
                    else V1AlexNetEnvelopeStoreRefusalCode.ROOT_INVALID
                )
                _refuse(
                    code,
                    f"cannot safely open durable root component: {error}",
                    artifact=traversed,
                )
            after = os.fstat(child)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(child)
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
                    "durable root changed during no-follow traversal",
                    artifact=traversed,
                )
            os.close(descriptor)
            descriptor = child
        return path, descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_member_names(directory_fd: int, *, artifact: Path) -> set[str]:
    try:
        names = os.listdir(directory_fd)
    except OSError as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID,
            f"cannot enumerate envelope-store object: {error}",
            artifact=artifact,
        )
    if any(
        type(name) is not str
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or "\x00" in name
        for name in names
    ):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
            "envelope-store object contains an unsafe member name",
            artifact=artifact,
        )
    return set(names)


def _open_directory_at(parent_fd: int, name: str, *, artifact: Path) -> int:
    if _SAFE_OBJECT_NAME.fullmatch(name) is None and not name.startswith(".open-"):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
            "unsafe envelope-store directory name",
            artifact=artifact,
        )
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.OBJECT_MISSING,
            "envelope-store object is missing",
            artifact=artifact,
        )
    if stat.S_ISLNK(before.st_mode):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED,
            "envelope-store object directory is symlinked",
            artifact=artifact,
        )
    if not stat.S_ISDIR(before.st_mode):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID,
            "envelope-store object is not a directory",
            artifact=artifact,
        )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        code = (
            V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED
            if error.errno == errno.ELOOP
            else V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID
        )
        _refuse(code, f"cannot safely open object: {error}", artifact=artifact)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
            "envelope-store object changed while opening",
            artifact=artifact,
        )
    return descriptor


def _open_regular_at(directory_fd: int, name: str, *, artifact: Path) -> tuple[int, os.stat_result]:
    if name not in {_MANIFEST_FILE, _ARRAY_FILE}:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
            "unsafe envelope-store member name",
            artifact=artifact,
        )
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.OBJECT_PARTIAL,
            "envelope-store member is missing",
            artifact=artifact,
        )
    if stat.S_ISLNK(before.st_mode):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED,
            "envelope-store member is symlinked",
            artifact=artifact,
        )
    if not stat.S_ISREG(before.st_mode):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID,
            "envelope-store member is not a regular file",
            artifact=artifact,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        code = (
            V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED
            if error.errno == errno.ELOOP
            else V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID
        )
        _refuse(code, f"cannot safely open member: {error}", artifact=artifact)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
            "envelope-store member changed while opening",
            artifact=artifact,
        )
    return descriptor, after


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    """Stable identity/content coordinates, deliberately excluding atime."""

    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _write_all(descriptor: int, payload: memoryview | bytes) -> None:
    view = memoryview(payload).cast("B")
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("short write while publishing AlexNet envelope")
        offset += written


def _write_new_at(directory_fd: int, name: str, payload: memoryview | bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_small_file(descriptor: int, size: int, *, artifact: Path) -> bytes:
    if size < 1 or size > _MAX_MANIFEST_BYTES:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.SIZE_MISMATCH,
            "manifest size lies outside the admitted bound",
            artifact=artifact,
        )
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
                "manifest ended during inspection",
                artifact=artifact,
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
            "manifest grew during inspection",
            artifact=artifact,
        )
    return b"".join(chunks)


def _hash_fd(descriptor: int, *, expected_size: int, artifact: Path) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    observed = 0
    while observed < expected_size:
        chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, expected_size - observed))
        if not chunk:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
                "array ended during inspection",
                artifact=artifact,
            )
        digest.update(chunk)
        observed += len(chunk)
    if os.read(descriptor, 1):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
            "array grew during inspection",
            artifact=artifact,
        )
    return digest.hexdigest()


def _unlink_stage(root_fd: int, stage_fd: int, stage_name: str) -> bool:
    """Consume one stage FD and remove its non-authoritative namespace.

    The caller transfers exclusive ownership of ``stage_fd`` before entry;
    this helper closes it exactly once.  NFS-v3 may transiently report
    ``ENOTEMPTY`` after both names have been unlinked even when a descriptor
    enumeration is already empty.  That namespace lag is retried for a small,
    bounded interval and, if it persists, leaves only an empty inert ``.open``
    directory.  Every other cleanup error remains visible.

    ``True`` means the directory is absent.  ``False`` means the committed
    object is unaffected but an inert stage directory remains for later
    housekeeping.
    """

    try:
        for name in (_MANIFEST_FILE, _ARRAY_FILE):
            try:
                os.unlink(name, dir_fd=stage_fd)
            except FileNotFoundError:
                pass

        last_observed_members: tuple[str, ...] = ()
        for delay_seconds in _STAGE_RMDIR_RETRY_DELAYS_SECONDS:
            if delay_seconds:
                time.sleep(delay_seconds)
            try:
                os.rmdir(stage_name, dir_fd=root_fd)
                return True
            except FileNotFoundError:
                return True
            except OSError as error:
                if error.errno != errno.ENOTEMPTY:
                    raise
                # Continue through transient NFS silly-rename or cache lag,
                # but remember any visible member so persistent injection or
                # residue cannot be silently admitted as an empty stage.
                last_observed_members = tuple(sorted(os.listdir(stage_fd)))
        if last_observed_members:
            raise OSError(
                errno.ENOTEMPTY,
                "AlexNet envelope stage retains unexpected members after "
                f"bounded cleanup: {last_observed_members!r}",
            )
        return False
    finally:
        os.close(stage_fd)


def _publication_lock(root_fd: int, manifest_sha256: str) -> int:
    name = f".lock-{manifest_sha256}"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=root_fd)
    except OSError as error:
        code = (
            V1AlexNetEnvelopeStoreRefusalCode.SYMLINK_REFUSED
            if error.errno == errno.ELOOP
            else V1AlexNetEnvelopeStoreRefusalCode.PUBLICATION_FAILED
        )
        _refuse(code, f"cannot open publication lock: {error}")
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID,
            "publication lock is not a regular file",
        )
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def _manifest_core_and_hash(value: object) -> tuple[dict[str, object], str]:
    keys = {
        "schema",
        "format_version",
        "authority_scope",
        "result_schema",
        "array",
        "certificate",
        "derivation_receipt",
        "manifest_content_hash",
    }
    manifest = _exact_mapping(value, keys=keys, name="envelope manifest")
    declared = _identity(
        manifest["manifest_content_hash"], name="manifest_content_hash"
    )
    core = {key: value for key, value in manifest.items() if key != "manifest_content_hash"}
    observed = content_hash(core, prefix=_MANIFEST_CONTENT_PREFIX)
    if observed != declared:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
            "manifest content hash does not replay",
        )
    return core, declared


_LAYER_FIELDS: Final = {field.name for field in fields(V1AlexNetEnvelopeLayerReceipt)}
_RECEIPT_FIELDS: Final = {
    field.name
    for field in fields(V1AlexNetAttributionEnvelopeDerivationReceipt)
    if not field.name.startswith("_")
} | {"receipt_hash"}
_CERTIFICATE_FIELDS: Final = {
    field.name
    for field in fields(V1CorrectedAttributionEnvelopeCertificate)
    if field.name not in {"absolute_upper_envelope_by_stratum", "_certificate_hash"}
} | {"certificate_hash"}


def _layer_from_json(value: object) -> V1AlexNetEnvelopeLayerReceipt:
    payload = _exact_mapping(value, keys=_LAYER_FIELDS, name="layer receipt")
    return V1AlexNetEnvelopeLayerReceipt(
        layer_id=_identity(payload["layer_id"], name="layer_id"),
        module_path=_identity(payload["module_path"], name="module_path"),
        route_kind=_identity(payload["route_kind"], name="route_kind"),
        source_shape=_positive_shape(payload["source_shape"], name="source_shape"),
        destination_shape=_positive_shape(
            payload["destination_shape"], name="destination_shape"
        ),
        route_count=_exact_int(payload["route_count"], name="route_count", positive=True),
        live_route_count=_exact_int(payload["live_route_count"], name="live_route_count"),
        source_absolute_max=_finite(
            payload["source_absolute_max"], name="source_absolute_max"
        ),
        destination_cotangent_absolute_max=_finite(
            payload["destination_cotangent_absolute_max"],
            name="destination_cotangent_absolute_max",
        ),
        route_envelope_max=_finite(
            payload["route_envelope_max"], name="route_envelope_max"
        ),
        forward_rounding_factor=_finite(
            payload["forward_rounding_factor"], name="forward_rounding_factor"
        ),
        backward_rounding_factor=_finite(
            payload["backward_rounding_factor"], name="backward_rounding_factor"
        ),
        route_rounding_factor=_finite(
            payload["route_rounding_factor"], name="route_rounding_factor"
        ),
    )


def _receipt_from_json(
    value: object,
    *,
    expected_hash: str,
) -> V1AlexNetAttributionEnvelopeDerivationReceipt:
    payload = _exact_mapping(value, keys=_RECEIPT_FIELDS, name="derivation receipt")
    layers = payload["layer_receipts"]
    if type(layers) is not list:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            "layer_receipts must be one JSON list",
        )
    receipt = V1AlexNetAttributionEnvelopeDerivationReceipt(
        campaign_contract_hash=_identity(
            payload["campaign_contract_hash"], name="campaign_contract_hash"
        ),
        model_identity_hash=_identity(payload["model_identity_hash"], name="model_identity_hash"),
        model_state_hash=_identity(payload["model_state_hash"], name="model_state_hash"),
        query_contract_hash=_identity(payload["query_contract_hash"], name="query_contract_hash"),
        preprocessing_hash=_identity(payload["preprocessing_hash"], name="preprocessing_hash"),
        current_mask_hash=_identity(payload["current_mask_hash"], name="current_mask_hash"),
        raw_logit_stream_source_hash=_identity(
            payload["raw_logit_stream_source_hash"], name="raw_logit_stream_source_hash"
        ),
        sampling_law_hash=_identity(payload["sampling_law_hash"], name="sampling_law_hash"),
        proposal_generation=_exact_int(payload["proposal_generation"], name="proposal_generation"),
        route_count=_exact_int(payload["route_count"], name="route_count", positive=True),
        route_axis_hash=_identity(payload["route_axis_hash"], name="route_axis_hash"),
        route_axis_content_hash=_identity(
            payload["route_axis_content_hash"], name="route_axis_content_hash"
        ),
        stratum_names=_string_pair(payload["stratum_names"], name="stratum_names"),
        population_stratum_counts=_positive_int_pair(
            payload["population_stratum_counts"], name="population_stratum_counts"
        ),
        sealed_population_order_hashes=_string_pair(
            payload["sealed_population_order_hashes"], name="sealed_population_order_hashes"
        ),
        target_class_id=_exact_int(payload["target_class_id"], name="target_class_id"),
        architecture_id=_identity(payload["architecture_id"], name="architecture_id"),
        topology_hash=_identity(payload["topology_hash"], name="topology_hash"),
        graph_semantics_hash=_identity(
            payload["graph_semantics_hash"], name="graph_semantics_hash"
        ),
        input_box_hash=_identity(payload["input_box_hash"], name="input_box_hash"),
        arithmetic_profile_hash=_identity(
            payload["arithmetic_profile_hash"], name="arithmetic_profile_hash"
        ),
        absolute_envelope_vector_hash=_identity(
            payload["absolute_envelope_vector_hash"], name="absolute_envelope_vector_hash"
        ),
        verifier_source_hash=_identity(
            payload["verifier_source_hash"], name="verifier_source_hash"
        ),
        route_chunk_size=_exact_int(payload["route_chunk_size"], name="route_chunk_size", positive=True),
        layer_receipts=tuple(_layer_from_json(layer) for layer in cast(list[object], layers)),
        maximum_absolute_envelope=_finite(
            payload["maximum_absolute_envelope"], name="maximum_absolute_envelope"
        ),
        algorithm_id=_identity(payload["algorithm_id"], name="algorithm_id"),
        phi_bound_id=_identity(payload["phi_bound_id"], name="phi_bound_id"),
        observed_data_consumed=_exact_bool(
            payload["observed_data_consumed"], name="observed_data_consumed"
        ),
        terminal_outcomes_opened=_exact_bool(
            payload["terminal_outcomes_opened"], name="terminal_outcomes_opened"
        ),
        stratum_invariant_registered_support=_exact_bool(
            payload["stratum_invariant_registered_support"],
            name="stratum_invariant_registered_support",
        ),
        schema=_identity(payload["schema"], name="receipt schema"),
    )
    declared = _identity(payload["receipt_hash"], name="receipt_hash")
    if declared != expected_hash:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
            "descriptor proof receipt hash differs from the manifest",
        )
    try:
        receipt.assert_integrity(expected_hash)
    except ValueError as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
            f"derivation receipt does not replay: {error}",
        )
    return receipt


def _certificate_from_json(
    value: object,
    *,
    array: np.ndarray,
    expected_hash: str,
) -> V1CorrectedAttributionEnvelopeCertificate:
    payload = _exact_mapping(value, keys=_CERTIFICATE_FIELDS, name="envelope certificate")
    certificate = V1CorrectedAttributionEnvelopeCertificate(
        campaign_contract_hash=_identity(
            payload["campaign_contract_hash"], name="campaign_contract_hash"
        ),
        model_identity_hash=_identity(payload["model_identity_hash"], name="model_identity_hash"),
        query_contract_hash=_identity(payload["query_contract_hash"], name="query_contract_hash"),
        preprocessing_hash=_identity(payload["preprocessing_hash"], name="preprocessing_hash"),
        current_mask_hash=_identity(payload["current_mask_hash"], name="current_mask_hash"),
        raw_logit_stream_source_hash=_identity(
            payload["raw_logit_stream_source_hash"], name="raw_logit_stream_source_hash"
        ),
        sampling_law_hash=_identity(payload["sampling_law_hash"], name="sampling_law_hash"),
        proposal_generation=_exact_int(payload["proposal_generation"], name="proposal_generation"),
        route_count=_exact_int(payload["route_count"], name="route_count", positive=True),
        route_axis_hash=_identity(payload["route_axis_hash"], name="route_axis_hash"),
        stratum_names=_string_pair(payload["stratum_names"], name="stratum_names"),
        population_stratum_counts=_positive_int_pair(
            payload["population_stratum_counts"], name="population_stratum_counts"
        ),
        sealed_population_order_hashes=_string_pair(
            payload["sealed_population_order_hashes"], name="sealed_population_order_hashes"
        ),
        absolute_upper_envelope_by_stratum=array,
        verifier_source_hash=_identity(
            payload["verifier_source_hash"], name="verifier_source_hash"
        ),
        proof_artifact_hash=_identity(payload["proof_artifact_hash"], name="proof_artifact_hash"),
        derivation_kind=_identity(payload["derivation_kind"], name="derivation_kind"),
        issued_before_outcomes=_exact_bool(
            payload["issued_before_outcomes"], name="issued_before_outcomes"
        ),
        schema=_identity(payload["schema"], name="certificate schema"),
    )
    declared = _identity(payload["certificate_hash"], name="certificate_hash")
    if declared != expected_hash:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
            "descriptor certificate hash differs from the manifest",
        )
    try:
        certificate.assert_integrity(expected_hash)
    except ValueError as error:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
            f"envelope certificate does not replay: {error}",
        )
    return certificate


def _validate_manifest_header(
    core: dict[str, object],
    *,
    descriptor: V1AlexNetEnvelopeStoreDescriptor,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if (
        type(core.get("schema")) is not str
        or core.get("schema") != ALEXNET_ENVELOPE_STORE_MANIFEST_SCHEMA
        or type(core.get("format_version")) is not int
        or core.get("format_version") != ALEXNET_ENVELOPE_STORE_FORMAT_VERSION
        or type(core.get("authority_scope")) is not str
        or core.get("authority_scope") != ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE
        or type(core.get("result_schema")) is not str
        or core.get("result_schema") != ALEXNET_ENVELOPE_RESULT_SCHEMA
    ):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            "unsupported AlexNet envelope-store manifest identity",
        )
    array = _exact_mapping(
        core["array"],
        keys={
            "file_name",
            "dtype",
            "order",
            "shape",
            "byte_count",
            "sha256",
            "content_hash",
        },
        name="array manifest",
    )
    if (
        array["file_name"] != _ARRAY_FILE
        or array["dtype"] != _LE_F64.str
        or array["order"] != "C"
    ):
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.PATH_TRAVERSAL,
            "array manifest does not name the fixed canonical payload",
        )
    shape = _positive_shape(array["shape"], name="array shape")
    if len(shape) != 2 or shape[0] != 2:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
            "envelope array shape must be exactly [2, route_count]",
        )
    expected_bytes = math.prod(shape) * _LE_F64.itemsize
    if _exact_int(array["byte_count"], name="array byte_count", positive=True) != expected_bytes:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.SIZE_MISMATCH,
            "array byte_count differs from dtype and shape",
        )
    if _sha256(array["sha256"], name="array sha256") != descriptor.envelope_array_sha256:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
            "descriptor array SHA differs from the manifest",
        )
    if _identity(array["content_hash"], name="array content hash") != descriptor.envelope_array_content_hash:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
            "descriptor array content hash differs from the manifest",
        )
    certificate = _exact_mapping(
        core["certificate"], keys=_CERTIFICATE_FIELDS, name="envelope certificate"
    )
    receipt = _exact_mapping(
        core["derivation_receipt"], keys=_RECEIPT_FIELDS, name="derivation receipt"
    )
    if certificate["route_count"] != shape[1] or receipt["route_count"] != shape[1]:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
            "certificate or receipt route count differs from the stored array",
        )
    return array, certificate, receipt


def _inspect_and_open(
    root: str | os.PathLike[str],
    descriptor: V1AlexNetEnvelopeStoreDescriptor,
) -> V1AlexNetCorrectedAttributionEnvelopeResult:
    if type(descriptor) is not V1AlexNetEnvelopeStoreDescriptor:
        _refuse(
            V1AlexNetEnvelopeStoreRefusalCode.INVALID_DESCRIPTOR,
            "open requires the exact envelope-store descriptor type",
        )
    root_path, root_fd = _open_root(root, create=False)
    object_path = root_path / descriptor.object_name
    object_fd = -1
    manifest_fd = -1
    array_fd = -1
    mapping: mmap.mmap | None = None
    array_view: np.ndarray | None = None
    try:
        object_fd = _open_directory_at(
            root_fd, descriptor.object_name, artifact=object_path
        )
        names = _safe_member_names(object_fd, artifact=object_path)
        if names != {_MANIFEST_FILE, _ARRAY_FILE}:
            code = (
                V1AlexNetEnvelopeStoreRefusalCode.OBJECT_PARTIAL
                if _MANIFEST_FILE not in names or _ARRAY_FILE not in names
                else V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID
            )
            _refuse(
                code,
                "envelope-store object members are not exact",
                artifact=object_path,
            )
        manifest_path = object_path / _MANIFEST_FILE
        manifest_fd, manifest_stat = _open_regular_at(
            object_fd, _MANIFEST_FILE, artifact=manifest_path
        )
        raw_manifest = _read_small_file(
            manifest_fd, manifest_stat.st_size, artifact=manifest_path
        )
        if hashlib.sha256(raw_manifest).hexdigest() != descriptor.manifest_sha256:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
                "manifest SHA-256 differs from the descriptor",
                artifact=manifest_path,
            )
        decoded = _strict_json(raw_manifest)
        if _canonical_json_bytes(cast(Mapping[str, object], decoded)) != raw_manifest:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.MANIFEST_INVALID,
                "manifest bytes are not the unique canonical JSON rendering",
                artifact=manifest_path,
            )
        core, manifest_content_hash = _manifest_core_and_hash(decoded)
        if manifest_content_hash != descriptor.manifest_content_hash:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
                "descriptor manifest content hash differs",
                artifact=manifest_path,
            )
        array_meta, certificate_meta, receipt_meta = _validate_manifest_header(
            core, descriptor=descriptor
        )
        array_path = object_path / _ARRAY_FILE
        array_fd, array_stat = _open_regular_at(
            object_fd, _ARRAY_FILE, artifact=array_path
        )
        expected_bytes = cast(int, array_meta["byte_count"])
        if array_stat.st_size != expected_bytes:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.SIZE_MISMATCH,
                "stored array size differs from the manifest",
                artifact=array_path,
            )
        observed_sha = _hash_fd(
            array_fd, expected_size=expected_bytes, artifact=array_path
        )
        if observed_sha != descriptor.envelope_array_sha256:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
                "stored array SHA-256 differs from the descriptor",
                artifact=array_path,
            )
        shape = tuple(cast(list[int], array_meta["shape"]))
        mapping = mmap.mmap(array_fd, expected_bytes, access=mmap.ACCESS_READ)
        array_view = np.frombuffer(mapping, dtype=_LE_F64).reshape(shape)
        if not bool(np.isfinite(array_view).all()) or bool(np.any(array_view < 0.0)):
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.FILE_INVALID,
                "stored envelope array is nonfinite or negative",
                artifact=array_path,
            )
        if _array_content_hash(array_view) != descriptor.envelope_array_content_hash:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.DIGEST_MISMATCH,
                "stored array content hash differs from the descriptor",
                artifact=array_path,
            )
        try:
            receipt = _receipt_from_json(
                receipt_meta, expected_hash=descriptor.proof_receipt_hash
            )
            certificate = _certificate_from_json(
                certificate_meta,
                array=array_view,
                expected_hash=descriptor.certificate_hash,
            )
            result = V1AlexNetCorrectedAttributionEnvelopeResult(
                certificate=certificate,
                derivation_receipt=receipt,
                schema=cast(str, core["result_schema"]),
            )
            result.assert_integrity()
        except V1AlexNetEnvelopeStoreRefusal:
            raise
        except (TypeError, ValueError, PermissionError, RuntimeError) as error:
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.COORDINATE_MISMATCH,
                f"stored AlexNet envelope result failed deep integrity: {error}",
                artifact=object_path,
            )
        if (
            _stat_signature(os.fstat(manifest_fd)) != _stat_signature(manifest_stat)
            or _stat_signature(os.fstat(array_fd)) != _stat_signature(array_stat)
        ):
            _refuse(
                V1AlexNetEnvelopeStoreRefusalCode.ARTIFACT_CHANGED,
                "stored envelope files changed during reconstruction",
                artifact=object_path,
            )
        return result
    finally:
        # The public certificate constructor owns immutable bytes, so no live
        # mapping or filesystem descriptor escapes the cold verification gate.
        if array_view is not None:
            del array_view
        if mapping is not None:
            try:
                mapping.close()
            except BufferError:
                # A constructor-refusal traceback can transiently retain its
                # private input view until the active exception is released.
                # Nothing escapes this function; the read-only mapping closes
                # when that traceback unwinds instead of masking the refusal.
                pass
        for file_descriptor in (array_fd, manifest_fd, object_fd, root_fd):
            if file_descriptor >= 0:
                os.close(file_descriptor)


def publish_v1_alexnet_envelope_result(
    root: str | os.PathLike[str],
    result: V1AlexNetCorrectedAttributionEnvelopeResult,
) -> V1AlexNetEnvelopeStoreDescriptor:
    """Publish one exact envelope result with manifest-last no-replace semantics."""

    manifest_bytes, descriptor, array_bytes = _manifest_and_descriptor(result)
    root_path, root_fd = _open_root(root, create=True)
    stage_name = f".open-{os.getpid()}-{uuid.uuid4().hex}"
    stage_fd = -1
    target_fd = -1
    lock_fd = -1
    target_claimed = False
    try:
        os.mkdir(stage_name, 0o700, dir_fd=root_fd)
        stage_fd = _open_directory_at(
            root_fd, stage_name, artifact=root_path / stage_name
        )
        _write_new_at(stage_fd, _ARRAY_FILE, array_bytes)
        _write_new_at(stage_fd, _MANIFEST_FILE, manifest_bytes)
        os.fsync(stage_fd)
        lock_fd = _publication_lock(root_fd, descriptor.manifest_sha256)
        try:
            os.mkdir(descriptor.object_name, 0o700, dir_fd=root_fd)
            target_claimed = True
            os.fsync(root_fd)
        except FileExistsError:
            target_claimed = False
        target_fd = _open_directory_at(
            root_fd,
            descriptor.object_name,
            artifact=root_path / descriptor.object_name,
        )
        if target_claimed:
            try:
                os.link(
                    _ARRAY_FILE,
                    _ARRAY_FILE,
                    src_dir_fd=stage_fd,
                    dst_dir_fd=target_fd,
                    follow_symlinks=False,
                )
                os.fsync(target_fd)
                # The manifest is the sole commit transition and is linked
                # only after the exact array is durable.
                os.link(
                    _MANIFEST_FILE,
                    _MANIFEST_FILE,
                    src_dir_fd=stage_fd,
                    dst_dir_fd=target_fd,
                    follow_symlinks=False,
                )
                os.fsync(target_fd)
                os.fsync(root_fd)
            except FileExistsError:
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.PUBLICATION_CONFLICT,
                    "no-replace publication target was populated unexpectedly",
                    artifact=root_path / descriptor.object_name,
                )
            except OSError as error:
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.PUBLICATION_FAILED,
                    f"manifest-last hard-link publication failed: {error}",
                    artifact=root_path / descriptor.object_name,
                )
        else:
            # Existing targets are never completed, repaired, or replaced.
            # Idempotence is admitted only after the full cold integrity gate.
            try:
                _inspect_and_open(root, descriptor)
            except V1AlexNetEnvelopeStoreRefusal as error:
                _refuse(
                    V1AlexNetEnvelopeStoreRefusalCode.PUBLICATION_CONFLICT,
                    f"existing no-replace object is not the exact result: {error}",
                    artifact=root_path / descriptor.object_name,
                )
        os.close(target_fd)
        target_fd = -1
        cleanup_fd = stage_fd
        stage_fd = -1
        _unlink_stage(root_fd, cleanup_fd, stage_name)
        os.fsync(root_fd)
        # Verify the winner after the stage links are removed.  This binds the
        # returned descriptor to the final one-link durable object state, not
        # to write intent or to the transient two-link publication state.
        _inspect_and_open(root, descriptor)
        return descriptor
    finally:
        active_error = sys.exc_info()[1]
        if lock_fd >= 0:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
        if target_fd >= 0:
            os.close(target_fd)
        if stage_fd >= 0:
            cleanup_fd = stage_fd
            stage_fd = -1
            try:
                _unlink_stage(root_fd, cleanup_fd, stage_name)
            except OSError as cleanup_error:
                # Cleanup is not allowed to replace a publication/integrity
                # failure.  Preserve the original exception and attach the
                # namespace-cleanup evidence for diagnostics.
                if active_error is None:
                    raise
                active_error.add_note(
                    "AlexNet envelope stage cleanup also failed: "
                    f"{cleanup_error!r}"
                )
        if root_fd >= 0:
            os.close(root_fd)


def open_v1_alexnet_envelope_result(
    root: str | os.PathLike[str],
    descriptor: V1AlexNetEnvelopeStoreDescriptor,
) -> V1AlexNetCorrectedAttributionEnvelopeResult:
    """Cold-open, reconstruct, and deeply verify one exact stored result."""

    return _inspect_and_open(root, descriptor)


def assert_v1_alexnet_envelope_store_integrity(
    root: str | os.PathLike[str],
    descriptor: V1AlexNetEnvelopeStoreDescriptor,
) -> tuple[str, str]:
    """Run the full cold barrier and return proof/certificate identities."""

    result = _inspect_and_open(root, descriptor)
    return result.assert_integrity()


__all__ = [
    "ALEXNET_ENVELOPE_STORE_AUTHORITY_SCOPE",
    "ALEXNET_ENVELOPE_STORE_DESCRIPTOR_SCHEMA",
    "ALEXNET_ENVELOPE_STORE_FORMAT_VERSION",
    "ALEXNET_ENVELOPE_STORE_MANIFEST_SCHEMA",
    "V1AlexNetEnvelopeStoreDescriptor",
    "V1AlexNetEnvelopeStoreRefusal",
    "V1AlexNetEnvelopeStoreRefusalCode",
    "assert_v1_alexnet_envelope_store_integrity",
    "open_v1_alexnet_envelope_result",
    "publish_v1_alexnet_envelope_result",
]
